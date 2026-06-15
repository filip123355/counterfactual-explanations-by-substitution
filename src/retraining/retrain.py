from __future__ import annotations

from pathlib import Path
import os
import json
import tempfile

import torch
import torch.nn as nn
import mlflow
from mlflow.entities import Run
from PIL.Image import Image
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms
from tqdm import tqdm
from pydantic import BaseModel

from src.constants import BATCH_SIZE, I2SB_IMAGE_SIZE, TRACKING_URI
from src.data.loader import CelebADataset, CompositeFeature, Feature
from src.inpainter.guidance.classifier import get_classifier, DenseNetClassifier
from src.mlflow import get_runs_by_names
from src.utils import log_config_params, parse_args, load_config
from src.substitution import ColorFillSubstitution


def _prepare_image(image: Image) -> torch.Tensor:
    transform = transforms.Compose(
        [
            transforms.Resize((I2SB_IMAGE_SIZE, I2SB_IMAGE_SIZE)),
            transforms.ToTensor(),
        ]
    )
    return transform(image)


def _get_target_index(run: Run) -> int:
    raw_target_index = run.data.params.get("TARGET_INDEX")
    if raw_target_index is None:
        raise ValueError(
            f"Run {run.info.run_id} is missing required MLflow param TARGET_INDEX."
        )
    return int(raw_target_index)


def _parse_feature(feature_name: str) -> Feature | CompositeFeature:
    if feature_name in CompositeFeature._value2member_map_:
        return CompositeFeature(feature_name)
    if feature_name in Feature._value2member_map_:
        return Feature(feature_name)
    raise ValueError(f"Invalid feature type: {feature_name}")


class RetrainResult(BaseModel):
    train_loss: list[float]
    test_loss: list[float]
    test_accuracy: list[float]
    test_mean_confidence: list[float]
    n_train: int
    n_test: int
    n_total: int


def log_retraining_result(result: RetrainResult) -> None:
    for epoch, (train_loss, test_loss, test_accuracy, test_mean_confidence) in enumerate(
        zip(result.train_loss, result.test_loss, result.test_accuracy, result.test_mean_confidence),
        start=1,
    ):
        mlflow.log_metric("train_loss", float(train_loss), step=epoch)
        mlflow.log_metric("test_loss", float(test_loss), step=epoch)
        mlflow.log_metric("test_accuracy", float(test_accuracy), step=epoch)
        mlflow.log_metric("test_mean_confidence", float(test_mean_confidence), step=epoch)

    with tempfile.TemporaryDirectory() as tmpdir:
        results_path = Path(tmpdir) / "retrain_results.json"
        results_path.write_text(
            result.model_dump_json(indent=2),
            encoding="utf-8",
        )
        mlflow.log_artifact(str(results_path), artifact_path="retrain")


class Retrainer:
    def __init__(
        self,
        model: DenseNetClassifier,
        dataset: CelebADataset,
        device: torch.device | None = None,
    ):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.dataset = dataset
        self.substitution = ColorFillSubstitution(dataset=dataset)

    def get_coalition_images(
        self,
        top_k: int, 
        run_names: list[str],
        experiment_name: str,
        shapley_artifact_subdir: str,
    ) -> tuple[list[Image], list[int]]:
        runs = get_runs_by_names(run_names=run_names, experiment_name=experiment_name)
        
        coalition_images: list[Image] = []
        labels: list[int] = []
        
        for run in runs:
            
            target_image_idx = _get_target_index(run)
            target_image = self.dataset.get(target_image_idx)["full_image"]

            shapley_values_dir = mlflow.artifacts.download_artifacts(
                run_id=run.info.run_id,
                artifact_path=shapley_artifact_subdir.replace("XXXX", str(target_image_idx)),
            )
            with open(shapley_values_dir) as f:
                shapley_values: dict[str, float] = json.load(f)

            sorted_shapley_values = list(
                sorted(shapley_values.items(), key=lambda item: item[1], reverse=True)
            )

            masked_image = target_image.copy()
            for feature, _ in list(sorted_shapley_values)[:top_k]:
                feature_no_bra = feature[1:-1]
                masked_image = self.substitution.substitute(
                    dest_idx=target_image_idx,
                    feature=_parse_feature(feature_no_bra),
                    image=masked_image,
                    skip_missing=True,
                )

            coalition_images.append(masked_image)
            labels.append(self.dataset.get(target_image_idx)["label_value"])

        return coalition_images, labels

    def _make_dataloader(
        self,
        images: list[Image],
        labels: list[int],
        *,
        batch_size: int,
        shuffle: bool,
    ) -> DataLoader[tuple[torch.Tensor, torch.Tensor]]:
        image_tensors = torch.stack([_prepare_image(img) for img in images])
        label_tensor = torch.tensor(labels, dtype=torch.float32)

        dataset = TensorDataset(image_tensors, label_tensor)
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle) # ty: ignore

    def _evaluate(
        self, 
        loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
    ) -> tuple[float, float, float]:
        self.model.eval()
        total_loss = 0.0
        total_correct = 0
        total_count = 0
        total_confidence = 0.0
        criterion = nn.BCEWithLogitsLoss()

        with torch.no_grad():
            for X, y in loader:
                X = X.to(self.device)
                y = y.to(self.device)

                logits = self.model(X)[:, 0]
                loss = criterion(logits, y)

                probs = torch.sigmoid(logits)
                preds = (probs >= 0.5).long()

                total_loss += float(loss.item()) * X.size(0)
                total_correct += int((preds == y.long()).sum().item())
                total_count += int(X.size(0))
                total_confidence += float(probs.sum().item())

        mean_loss = total_loss / total_count if total_count else 0.0
        accuracy = total_correct / total_count if total_count else 0.0
        mean_confidence = total_confidence / total_count if total_count else 0.0
        return mean_loss, accuracy, mean_confidence

    def retrain(
        self,
        run_names: list[str],
        experiment_name: str,
        top_k: int,
        shapley_artifact_subdir: str,
        test_size: float = 0.2,
        num_epochs: int = 20,
        lr: float = 1e-3,
        batch_size: int = BATCH_SIZE,
        seed: int = 42,
        model_save_path: str | None = None,
    ) -> RetrainResult:
        coalition_images, labels = self.get_coalition_images(
            run_names=run_names,
            experiment_name=experiment_name,
            shapley_artifact_subdir=shapley_artifact_subdir,
            top_k=top_k,
        )
        if not coalition_images:
            raise ValueError("No coalition images were loaded from MLflow artifacts.")

        n_total = len(coalition_images)
        n_test = max(1, int(round(n_total * test_size)))
        n_train = n_total - n_test
        if n_train <= 0:
            raise ValueError(
                "test_size is too large for the number of loaded coalition images."
            )

        generator = torch.Generator().manual_seed(seed)
        permutation = torch.randperm(n_total, generator=generator).tolist()
        train_indices = permutation[:n_train]
        test_indices = permutation[n_train:]

        train_images = [coalition_images[i] for i in train_indices]
        train_labels = [labels[i] for i in train_indices]
        test_images = [coalition_images[i] for i in test_indices]
        test_labels = [labels[i] for i in test_indices]

        train_loader = self._make_dataloader(
            train_images, train_labels, batch_size=batch_size, shuffle=True
        )
        test_loader = self._make_dataloader(
            test_images, test_labels, batch_size=batch_size, shuffle=False
        )

        for param in self.model.feat_extract.parameters():
            param.requires_grad = False
        for param in self.model.classifier.parameters():
            param.requires_grad = True

        for module in self.model.classifier.modules():
            if isinstance(module, nn.Linear):
                module.reset_parameters()

        optimizer = torch.optim.Adam(self.model.classifier.parameters(), lr=lr)
        criterion = nn.BCEWithLogitsLoss()

        train_loss_history: list[float] = []
        test_loss_history: list[float] = []
        test_accuracy_history: list[float] = []
        test_mean_confidence_history: list[float] = []

        self.model.to(self.device)

        for epoch in tqdm(range(num_epochs), desc="Retraining"):
            self.model.train()
            self.model.feat_extract.eval()
            self.model.classifier.train()
            running_loss = 0.0
            running_count = 0

            for X, y in train_loader:
                X = X.to(self.device)
                y = y.to(self.device)

                optimizer.zero_grad(set_to_none=True)
                logits = self.model(X)[:, 0]
                loss = criterion(logits, y)
                loss.backward()
                optimizer.step()

                running_loss += float(loss.item()) * X.size(0)
                running_count += int(X.size(0))

            train_loss = running_loss / running_count if running_count else 0.0
            test_loss, test_accuracy, test_mean_confidence = self._evaluate(test_loader)

            train_loss_history.append(train_loss)
            test_loss_history.append(test_loss)
            test_accuracy_history.append(test_accuracy)
            test_mean_confidence_history.append(test_mean_confidence)

            tqdm.write(
                f"Epoch {epoch + 1}/{num_epochs} "
                f"train_loss={train_loss:.4f} "
                f"test_loss={test_loss:.4f} "
                f"test_accuracy={test_accuracy:.4f} "
                f"test_mean_confidence={test_mean_confidence:.4f}"
            )

        if model_save_path is not None:
            os.makedirs(model_save_path, exist_ok=True)
            torch.save(self.model.state_dict(), f"{model_save_path}/retrained_classifier.pth")

        return RetrainResult(
            train_loss=train_loss_history,
            test_loss=test_loss_history,
            test_accuracy=test_accuracy_history,
            test_mean_confidence=test_mean_confidence_history,
            n_train=n_train,
            n_test=n_test,
            n_total=n_total,
        )


def main(indices: list[int]) -> None:
    args = parse_args()
    config = load_config(args.config)

    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(config["TRAINING_EXPERIMENT_NAME"])

    run_name = config.get("TRAINING_RUN_NAME")

    with mlflow.start_run(run_name=run_name):
        log_config_params(config)
        mlflow.log_param("device", str(torch.device("cuda" if torch.cuda.is_available() else "cpu")))

        dataset = CelebADataset(split=config.get("DATASET_SPLIT", "test"))
        retrainer = Retrainer(
            model=get_classifier(),
            dataset=dataset,
        )

        run_names = [
            f"dataset_sub_target_{idx}"
            # f"dataset_target_{idx}_tau_1.0"
            for idx in indices
        ]

        result = retrainer.retrain(
            run_names=run_names,
            experiment_name=config["MLFLOW_EXPERIMENT_NAME"],
            test_size=config.get("TEST_SIZE", 0.2),
            num_epochs=config.get("NUM_EPOCHS", 20),
            lr=config.get("LR", 1e-3),
            batch_size=config.get("BATCH_SIZE", BATCH_SIZE),
            seed=config.get("SEED", 42),
            model_save_path=config.get("MODEL_SAVE_PATH"),
            top_k=config.get("TOP_K", 3),
            shapley_artifact_subdir=str(config.get("SHAPLEY_ARTIFACT_SUBDIR")),
        )

        log_retraining_result(result)

        model_save_path = config.get("MODEL_SAVE_PATH")
        if model_save_path is not None:
            mlflow.log_artifact(
                os.path.join(model_save_path, "retrained_classifier.pth"),
                artifact_path="retrain/model",
            )


if __name__ == "__main__":

    indices = [
        2471, 1586, 1275, 2646, 2712, 1050, 933, 1242, 497, 2606, 
        1855, 429, 942, 2813, 1865, 1745, 173, 1552, 2356, 2683, 
        2692, 622, 2217, 1258, 2189, 137, 988, 1622, 2781, 447, 
        1909, 575, 1982, 792, 2451, 2155, 1185, 386, 804, 2696, 
        1718, 228, 2049, 2021, 779, 2768, 1127, 674, 2257, 2060, 
        280, 664, 1777, 580, 503, 797, 2147, 502, 1215, 1688, 392, 
        2258, 1888, 456, 1954, 477, 1498, 419, 1310, 955, 1036, 
        1312, 227, 1136, 1466, 2290, 2812, 433, 1955, 2345, 2044,
        1311, 1349, 2385, 2316, 1424, 1648, 2809, 1582, 417, 1097,
        134, 2493, 1885, 434, 351, 2724, 237, 1935, 530
        ]

    main(indices=indices)
