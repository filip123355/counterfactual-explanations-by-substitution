import json
import numpy as np
import os
import itertools
import torch
from PIL import Image

from src.constants import PROJECT_ROOT, CLASSIFIER_LABEL
from src.data_loading import CelebADataset, CompositeFeature, Feature, FeatureType
from src.inpainter.guidance.classifier import DenseNetClassifier, get_classifier
from src.keypoints import FaceKeypointDetector, MediapipeFaceKeypointDetector
from src.substitution import Substitution
from src.shapley import NShapleyValueCalculator

class BilinearModel:

    first_order_coefficients: np.ndarray
    second_order_coefficients: np.ndarray

    def __init__(
        self,
        features: list[str],
        target_idx: int,
        interaction_level: int = 2,
        dataset: CelebADataset | None = None,
        face_keypoint_detector: FaceKeypointDetector | None = None,
    ) -> None:
        self.target_idx = target_idx
        self.features = features
        self.dataset = dataset
        self.face_keypoint_detector = face_keypoint_detector
        self.interaction_level = interaction_level

        self.load_coefficients(PROJECT_ROOT)

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

    def load_coefficients(self, path: str) -> None:
        path = path / "results" / str(self.target_idx) / CLASSIFIER_LABEL / "shapley_values"
        with open(path / f"target_{self.target_idx}_1_shapley_values.json", "r") as f:
            first_order_values = json.load(f)
        with open(path / f"target_{self.target_idx}_2_shapley_values.json", "r") as f:
            second_order_values = json.load(f)
        self.first_order_coefficients = np.array(list(first_order_values.values()))
        self.second_order_coefficients = np.array(list(second_order_values.values()))

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
            ref_indices: list[int],
            model: DenseNetClassifier,
            device: torch.device,
            predict_prob: bool = False,
    ) -> float:

        target_hq_idx = self.dataset.data.iloc[self.target_idx]["idx"]
        target_image_path = os.path.join(self.dataset.img_dir, f"{target_hq_idx}.jpg")
        target_item = Image.open(target_image_path).convert("RGB")
        
        substitution = Substitution(self.dataset, self.face_keypoint_detector)

        ref_images = []
        for ref_idx in ref_indices:
            substituted_image = target_item.copy()
            for feature_value, feature_name in zip(feature_values, self.features):
                if feature_value == 0:
                    feature = self._parse_feature(feature_name)
                    substituted_image = substitution.substitute(
                        src_idx=ref_idx,
                        dest_idx=self.target_idx,
                        feature=feature,
                        image=substituted_image,
                    )
            ref_images.append(substituted_image)

        images_tensor = torch.stack([NShapleyValueCalculator.prepare_image(img) for img in ref_images]).to(device)

        if predict_prob:
            outputs = model.pred_prob(images_tensor)
        else:
            outputs = model(images_tensor)

        return outputs[:, 0].mean().item()
    
    def calculate_r_squared(
            self,
            ref_indices: list[int],
            model: DenseNetClassifier,
            device: torch.device,
            predict_prob: bool = True,
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
                    ref_indices,
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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = CelebADataset(split="test")
    face_keypoint_detector = MediapipeFaceKeypointDetector()
    model = get_classifier().to(device)
    features = [CompositeFeature.eyes, Feature.nose, CompositeFeature.mouth]
    bilinear_model = BilinearModel(
        features=features,
        target_idx=9,
        dataset=dataset,
        face_keypoint_detector=face_keypoint_detector,
    )
    r_squared = bilinear_model.calculate_r_squared(ref_indices=[0, 1, 2, 3, 4], model=model, device=device)
    print(f"R-squared: {r_squared:.4f}")
