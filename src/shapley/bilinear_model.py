import json
import numpy as np
import os
import itertools
import torch
import mlflow
from PIL import Image
from pathlib import Path

from src.data import CelebADataset, CompositeFeature, Feature, FeatureType
from src.inpainter.guidance.classifier import DenseNetClassifier, get_classifier
from src.substitution import ImageSubstitution, FaceKeypointDetector, MediapipeFaceKeypointDetector
from .calculator import NShapleyValueCalculator
from src.inpainter.i2sb import I2SB, SampleType
from src.inpainter.guidance import CLIPGuidance
from src.interface.clip import load_clip
from src.mlflow import client, get_run_by_name

class BilinearModel:
    first_order_coefficients: np.ndarray
    second_order_coefficients: np.ndarray

    def __init__(
        self,
        features: list[str | FeatureType],
        target_idx: int,
        first_order_experiment_name: str,
        second_order_experiment_name: str,
        run_name_temp: str = "target_XXX_male_Nnnn_tau_0.5_nfe_10",
        interaction_level: int = 2,
        dataset: CelebADataset | None = None,
    ) -> None:
        self.target_idx = target_idx
        self.features = features
        self.dataset = dataset
        self.interaction_level = interaction_level
        self.first_order_experiment_name = first_order_experiment_name
        self.second_order_experiment_name = second_order_experiment_name
        self.run_name_temp = run_name_temp

        self.load_coefficients(first_order_experiment_name, second_order_experiment_name)

    @staticmethod
    def _parse_feature(feature: str | FeatureType) -> FeatureType:
        if isinstance(feature, (Feature, CompositeFeature)):
            return feature

        for enum_cls in (Feature, CompositeFeature):
            try:
                return enum_cls(feature)
            except ValueError:
                pass

        raise ValueError(f"Unknown feature: {feature}")

    @staticmethod
    def _coalition_tokens(coalition_name: str) -> tuple[str, ...]:
        inner = coalition_name.strip()[1:-1].strip()
        if not inner:
            return ()
        return tuple(sorted(part.strip() for part in inner.split(",") if part.strip()))

    @classmethod
    def _matches_coalition_name(cls, image_path: Path, coalition_name: str) -> bool:
        stem = image_path.stem
        if not stem.startswith("coalition_"):
            return False

        try:
            coalition_part = stem[len("coalition_"):]
            coalition_part = coalition_part[:coalition_part.rindex("_")]
        except ValueError:
            return False

        return cls._coalition_tokens(coalition_part) == cls._coalition_tokens(coalition_name)

    def load_coefficients(
        self,
        first_order_experiment_name: str,
        second_order_experiment_name: str,
        shapley_subdir: str = "shapley/values",
        shapley_filename_template: str = "target_XXX_NNN_shapley_values_001.json",
    ) -> None:

        first_run = get_run_by_name(
            run_name=self.run_name_temp
                .replace("XXX", str(self.target_idx))
                .replace("nnn", "1"),
            experiment_name=first_order_experiment_name,
        )[0]

        second_run = get_run_by_name(
            run_name=self.run_name_temp
                .replace("XXX", str(self.target_idx))
                .replace("nnn", "2"),
            experiment_name=second_order_experiment_name,
        )[0]

        first_local_path = mlflow.artifacts.download_artifacts( 
            run_id=first_run.info.run_id,
            artifact_path=shapley_subdir,
        )

        second_local_path = mlflow.artifacts.download_artifacts(
            run_id=second_run.info.run_id,
            artifact_path=shapley_subdir,
        )

        first_values_path = Path(first_local_path) / shapley_filename_template.replace("XXX", str(self.target_idx)).replace("NNN", "1")
        second_values_path = Path(second_local_path) / shapley_filename_template.replace("XXX", str(self.target_idx)).replace("NNN", "2")

        with open(first_values_path, "r", encoding="utf-8") as f:
            first_order_values = json.load(f)

        with open(second_values_path, "r", encoding="utf-8") as f:
            second_order_values = json.load(f)

        self.first_order_coefficients = np.zeros(len(self.features), dtype=float)
        for i, feature in enumerate(self.features):
            key = f"({str(feature)})"
            self.first_order_coefficients[i] = first_order_values[key]

        rows, cols = np.triu_indices(len(self.features), k=1)
        self.second_order_coefficients = np.zeros(len(rows), dtype=float)
        
        for idx, (r, c) in enumerate(zip(rows, cols)):
            f1, f2 = str(self.features[r]), str(self.features[c])
            key = f"({', '.join(sorted([f1, f2]))})"
            self.second_order_coefficients[idx] = second_order_values[key]
            
    def predict_bmodel(self, feature_values: np.ndarray) -> float:

        if not np.all(np.isin(feature_values, [0, 1])):
            raise ValueError("Feature values must be binary (0 or 1).")
        
        linear_term = np.dot(self.first_order_coefficients, feature_values)
        Q = np.zeros((len(self.features), len(self.features)))
        rows, cols = np.triu_indices(len(self.features), k=1)
        Q[rows, cols] = self.second_order_coefficients
        quadratic_term = np.sum(
            Q * np.outer(feature_values, feature_values)
        )

        return linear_term + quadratic_term
    
    def predict_true_model(
            self, 
            feature_values: np.ndarray,
            model: DenseNetClassifier,
            device: torch.device,
            predict_prob: bool = False,
            coalitions_subdir: str = "shapley/coalitions",
    ) -> float:
        assert self.dataset is not None

        target_hq_idx = self.dataset.data.iloc[self.target_idx]["idx"]
        target_image_path = os.path.join(self.dataset.img_dir, f"{target_hq_idx}.jpg")
        target_item = Image.open(target_image_path).convert("RGB")

        reg = []
        for feature_name, feature_value in zip(self.features, feature_values):
            if feature_value == 0:
                reg.append(str(feature_name))

        coalition_name = ", ".join(reg) if len(reg) > 0 else ""
        coalition_name = "(" + coalition_name + ")"

        run = get_run_by_name(
            run_name=self.run_name_temp
                .replace("XXX", str(self.target_idx))
                .replace("nnn", "1"),
            experiment_name=self.first_order_experiment_name,
        )[0]

        local_path = mlflow.artifacts.download_artifacts(
            run_id=run.info.run_id,
            artifact_path=coalitions_subdir,
        )

        artifact_path = Path(local_path)

        images = [
            img for img in artifact_path.rglob("coalition_*")
            if img.suffix.lower() in {".png", ".jpg", ".jpeg"}
            and self._matches_coalition_name(img, coalition_name)
        ]

        if len(images) == 0:
            raise FileNotFoundError(
                f"No coalition images found for coalition_name={coalition_name!r} "
                f"in downloaded artifact path: {artifact_path}"
            )

        images_ref_tensor = torch.stack(
            [NShapleyValueCalculator.prepare_image(Image.open(img).convert("RGB")) for img in images]
        ).to(device)

        if predict_prob:
            outputs = model.pred_prob(images_ref_tensor)
        else:
            outputs = model(images_ref_tensor)

        return outputs[:, 0].mean().item()

    def calculate_r_squared(
            self,
            model: DenseNetClassifier,
            device: torch.device,
            predict_prob: bool = False,
    ) -> float:
        true_values = []
        predicted_values = []
        for i in range(1, self.interaction_level + 1):
            for subset in itertools.combinations(range(len(self.features)), i):
                feature_values = np.zeros(len(self.features))
                feature_values[list(subset)] = 1
                pred_bmodel = self.predict_bmodel(feature_values)
                true_model = self.predict_true_model(
                    feature_values,
                    model=model,
                    device=device,
                    predict_prob=predict_prob,
                )
                true_values.append(true_model)
                predicted_values.append(pred_bmodel)

        true_values = np.array(true_values)
        predicted_values = np.array(predicted_values)
        ss_res = np.sum((true_values - predicted_values) ** 2)
        ss_tot = np.sum((true_values - np.mean(true_values)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
        return r_squared


if __name__ == "__main__":
    TARGET_INDEX = 1586

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = CelebADataset(split="test")
    model = get_classifier().to(device)
    target_hq_idx = dataset.data.iloc[TARGET_INDEX]["idx"]
    target_image_path = os.path.join(dataset.img_dir, f"{target_hq_idx}.jpg")
    features: list[FeatureType | str] = [CompositeFeature.eyes, Feature.nose, CompositeFeature.mouth]

    bilinear_model = BilinearModel(
        features=features,
        target_idx=TARGET_INDEX,
        first_order_experiment_name="shapley",
        second_order_experiment_name="shapley",
        dataset=dataset,
        run_name_temp="target_XXX_male_Nnnn_sub_fixed3"
    )
    r_squared = bilinear_model.calculate_r_squared(model=model, device=device)
    print(f"R-squared: {r_squared:.4f}")
