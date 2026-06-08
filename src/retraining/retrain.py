from __future__ import annotations

import torch
import torch.nn as nn
from PIL.Image import Image
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms
from tqdm import tqdm
from pydantic import BaseModel

from src.constants import BATCH_SIZE, I2SB_IMAGE_SIZE
from src.data.loader import CelebADataset, CompositeFeature, Feature, FeatureType
from src.shapley.calculator import NShapleyValueCalculator
from src.substitution.core import ColorFillSubstitution


def _parse_feature(feature: str | FeatureType) -> FeatureType:
    if isinstance(feature, (Feature, CompositeFeature)):
        return feature

    try:
        return CompositeFeature(feature)
    except ValueError:
        return Feature(feature)


def _prepare_image(image: Image) -> torch.Tensor:
    transform = transforms.Compose(
        [
            transforms.Resize((I2SB_IMAGE_SIZE, I2SB_IMAGE_SIZE)),
            transforms.ToTensor(),
        ]
    )
    return transform(image)


def _shapley_sorted_features(shapley_values: dict[str, float]) -> list[FeatureType]:
    return [_parse_feature(feature) for feature, _ in sorted(shapley_values.items(), key=lambda item: item[1], reverse=True)]


class RetrainResult(BaseModel):
    train_loss: list[float]
    test_loss: list[float]
    test_accuracy: list[float]


class Retrainer:
    def __init__(
        self,
        model: nn.Module,
        dataset: CelebADataset,
        device: torch.device | None = None,
        substitution: ColorFillSubstitution | None = None,
    ):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.dataset = dataset
        self.substitution = substitution

    def substitute_features(
        self,
        ref_indices: list[int],
        shapley_values: list[dict[str, float]],
    ) -> tuple[list[Image], list[int]]:
        if len(ref_indices) != len(shapley_values):
            raise ValueError(
                "ref_indices and shapley_values must have the same length."
            )

        if self.substitution is None:
            raise ValueError("substitution is required for feature substitution.")

        substituted_images: list[Image] = []
        labels: list[int] = []

        for ref_idx, svs in zip(ref_indices, shapley_values):
            item = self.dataset.get(ref_idx)
            image = item["full_image"]
            label = int(item["label_value"])

            for feature in _shapley_sorted_features(svs):
                image = self.substitution.substitute(
                    dest_idx=ref_idx,
                    feature=feature,
                    image=image,
                    skip_missing=True,
                )

            substituted_images.append(image)
            labels.append(label)

        return substituted_images, labels

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
        ref_indices: list[int],
        shapley_values: list[dict[str, float]],
        test_size: float = 0.2,
        num_epochs: int = 20,
        lr: float = 1e-4,
        batch_size: int = BATCH_SIZE,
        seed: int = 42,
    ) -> RetrainResult:
        substituted_images, labels = self.substitute_features(ref_indices, shapley_values)
        if not substituted_images:
            raise ValueError("No substituted images were produced.")

        n_total = len(substituted_images)
        n_test = max(1, int(round(n_total * test_size)))
        n_train = n_total - n_test
        if n_train <= 0:
            raise ValueError(
                "test_size is too large for the number of substituted images."
            )

        generator = torch.Generator().manual_seed(seed)
        permutation = torch.randperm(n_total, generator=generator).tolist()
        train_indices = permutation[:n_train]
        test_indices = permutation[n_train:]

        train_images = [substituted_images[i] for i in train_indices]
        train_labels = [labels[i] for i in train_indices]
        test_images = [substituted_images[i] for i in test_indices]
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

        return RetrainResult(
            train_loss=train_loss_history,
            test_loss=test_loss_history,
            test_accuracy=test_accuracy_history,
        )
