from __future__ import annotations

import re
from pathlib import Path
import os
import json
import tempfile

import torch
import torch.nn as nn
import mlflow
from mlflow.entities import Run
from PIL import Image as PILImage
from PIL.Image import Image
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms
from tqdm import tqdm
from pydantic import BaseModel

from src.constants import BATCH_SIZE, I2SB_IMAGE_SIZE, TRACKING_URI
from src.data.loader import CelebADataset
from src.inpainter.guidance.classifier import get_classifier
from src.mlflow import get_runs_by_names, client
from src.utils import log_config_params, parse_args, load_config


def _prepare_image(image: Image) -> torch.Tensor:
    transform = transforms.Compose(
        [
            transforms.Resize((I2SB_IMAGE_SIZE, I2SB_IMAGE_SIZE)),
            transforms.ToTensor(),
        ]
    )
    return transform(image)


def _parse_sample_index(path: Path, coalition_label: str) -> int:
    pattern = re.compile(rf"^coalition_{re.escape(coalition_label)}_(\d+)\.png$")
    match = pattern.match(path.name)
    if match is None:
        raise ValueError(
            f"Artifact {path.name} does not match coalition naming for {coalition_label}."
        )
    return int(match.group(1))


def _get_target_index(run: Run) -> int:
    raw_target_index = run.data.params.get("TARGET_INDEX")
    if raw_target_index is None:
        raise ValueError(
            f"Run {run.info.run_id} is missing required MLflow param TARGET_INDEX."
        )
    return int(raw_target_index)


def _get_artifact_dir(run: Run, artifact_path: str) -> Path:
    artifact_root = Path(run.info.artifact_uri)
    local_artifact_dir = artifact_root / artifact_path
    if local_artifact_dir.exists():
        return local_artifact_dir
    return Path(client.download_artifacts(run.info.run_id, artifact_path))


def _parse_coalition_features(coalition_label: str) -> tuple[str, ...]:
    if coalition_label == "()":
        return ()

    inner = coalition_label.strip()[1:-1].strip()
    if not inner:
        return ()

    return tuple(part.strip() for part in inner.split(","))


def _coalition_sort_key(coalition_label: str) -> tuple[int, tuple[str, ...]]:
    features = _parse_coalition_features(coalition_label)
    return len(features), features


class RetrainResult(BaseModel):
    train_loss: list[float]
    test_loss: list[float]
    test_accuracy: list[float]
    n_train: int
    n_test: int
    n_total: int


def log_retraining_result(result: RetrainResult) -> None:
    for epoch, (train_loss, test_loss, test_accuracy) in enumerate(
        zip(result.train_loss, result.test_loss, result.test_accuracy),
        start=1,
    ):
        mlflow.log_metric("train_loss", float(train_loss), step=epoch)
        mlflow.log_metric("test_loss", float(test_loss), step=epoch)
        mlflow.log_metric("test_accuracy", float(test_accuracy), step=epoch)

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
        model: nn.Module,
        dataset: CelebADataset,
        device: torch.device | None = None,
    ):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.dataset = dataset

    def load_coalition_images(
        self,
        run_names: list[str],
        experiment_name: str,
    ) -> tuple[list[Image], list[int]]:
        runs = get_runs_by_names(run_names, experiment_name=experiment_name)

        coalition_images: list[Image] = []
        labels: list[int] = []
        artifact_path = "shapley/coalitions"

        for run in runs:
            target_index = _get_target_index(run)
            label = int(self.dataset.get(target_index)["label_value"])

            local_coalitions_dir = _get_artifact_dir(run, artifact_path)
            available_coalitions = sorted(
                {
                    path.stem.rsplit("_", 1)[0].removeprefix("coalition_")
                    for path in local_coalitions_dir.glob("coalition_*.png")
                },
                key=_coalition_sort_key,
            )

            if not available_coalitions:
                raise ValueError(
                    f"Run {run.info.run_id} does not contain any coalition artifacts under {artifact_path}."
                )

            for coalition_label in available_coalitions:
                matching_paths = sorted(
                    local_coalitions_dir.glob(f"coalition_{coalition_label}_*.png"),
                    key=lambda path: _parse_sample_index(path, coalition_label),
                )

                if not matching_paths:
                    raise ValueError(
                        f"Run {run.info.run_id} does not contain artifacts for coalition {coalition_label!r} "
                        f"under {artifact_path}."
                    )

                for image_path in matching_paths:
                    with PILImage.open(image_path) as image:
                        coalition_images.append(image.convert("RGB").copy())
                    labels.append(label)

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
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    def _evaluate(self, loader: DataLoader[tuple[torch.Tensor, torch.Tensor]]) -> tuple[float, float]:
        self.model.eval()
        total_loss = 0.0
        total_correct = 0
        total_count = 0
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

        mean_loss = total_loss / total_count if total_count else 0.0
        accuracy = total_correct / total_count if total_count else 0.0
        return mean_loss, accuracy

    def retrain(
        self,
        run_names: list[str],
        experiment_name: str,
        test_size: float = 0.2,
        num_epochs: int = 20,
        lr: float = 1e-3,
        batch_size: int = BATCH_SIZE,
        seed: int = 42,
        model_save_path: str | None = None,
    ) -> RetrainResult:
        coalition_images, labels = self.load_coalition_images(
            run_names=run_names,
            experiment_name=experiment_name,
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

        optimizer = torch.optim.Adam(self.model.classifier.parameters(), lr=lr)
        criterion = nn.BCEWithLogitsLoss()

        train_loss_history: list[float] = []
        test_loss_history: list[float] = []
        test_accuracy_history: list[float] = []

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
            test_loss, test_accuracy = self._evaluate(test_loader)

            train_loss_history.append(train_loss)
            test_loss_history.append(test_loss)
            test_accuracy_history.append(test_accuracy)

            tqdm.write(
                f"Epoch {epoch + 1}/{num_epochs} "
                f"train_loss={train_loss:.4f} "
                f"test_loss={test_loss:.4f} "
                f"test_accuracy={test_accuracy:.4f}"
            )

        if model_save_path is not None:
            os.makedirs(model_save_path, exist_ok=True)
            torch.save(self.model.state_dict(), f"{model_save_path}/retrained_classifier.pth")

        return RetrainResult(
            train_loss=train_loss_history,
            test_loss=test_loss_history,
            test_accuracy=test_accuracy_history,
            n_train=n_train,
            n_test=n_test,
            n_total=n_total,
        )


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(config["TRAINING_EXPERIMENT_NAME"])

    run_name = config.get("TRAINING_RUN_NAME", "retraining")

    with mlflow.start_run(run_name=run_name):
        log_config_params(config)
        mlflow.log_param("device", str(torch.device("cuda" if torch.cuda.is_available() else "cpu")))
        mlflow.log_param("source_run_names", json.dumps(config["MLFLOW_RUN_NAMES"]))

        dataset = CelebADataset(split=config.get("DATASET_SPLIT", "test"))
        retrainer = Retrainer(
            model=get_classifier(),
            dataset=dataset,
        )
        result = retrainer.retrain(
            run_names=config["MLFLOW_RUN_NAMES"],
            experiment_name=config["MLFLOW_EXPERIMENT_NAME"],
            test_size=config.get("TEST_SIZE", 0.2),
            num_epochs=config.get("NUM_EPOCHS", 20),
            lr=config.get("LR", 1e-3),
            batch_size=config.get("BATCH_SIZE", BATCH_SIZE),
            seed=config.get("SEED", 42),
            model_save_path=config.get("MODEL_SAVE_PATH"),
        )

        log_retraining_result(result)

        model_save_path = config.get("MODEL_SAVE_PATH")
        if model_save_path is not None:
            mlflow.log_artifact(
                os.path.join(model_save_path, "retrained_classifier.pth"),
                artifact_path="retrain/model",
            )


if __name__ == "__main__":
    main()
