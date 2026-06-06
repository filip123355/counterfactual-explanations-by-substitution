import json
import numpy as np
import os
import itertools
import torch
from PIL import Image

from src.data import CelebADataset, CompositeFeature, Feature, FeatureType
from src.inpainter.guidance.classifier import DenseNetClassifier, get_classifier
from src.substitution import Substitution, FaceKeypointDetector, MediapipeFaceKeypointDetector
from .calculator import NShapleyValueCalculator

class BilinearModel:
    first_order_coefficients: np.ndarray
    second_order_coefficients: np.ndarray

    def __init__(
        self,
        features: list[str | FeatureType],
        target_idx: int,
        first_order_values_path: str,
        second_order_values_path: str,
        interaction_level: int = 2,
        dataset: CelebADataset | None = None,
        inpainter: I2SB | None = None,
        face_keypoint_detector: FaceKeypointDetector | None = None,
    ) -> None:
        self.target_idx = target_idx
        self.features = features
        self.dataset = dataset
        self.face_keypoint_detector = face_keypoint_detector
        self.interaction_level = interaction_level
        self.inpainter = inpainter

        self.load_coefficients(first_order_values_path, second_order_values_path)

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

    def load_coefficients(
        self,
        first_order_values_path: str,
        second_order_values_path: str,
    ) -> None:
        with open(first_order_values_path, "r", encoding="utf-8") as f:
            first_order_values = json.load(f)
        with open(second_order_values_path, "r", encoding="utf-8") as f:
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
            tau: float = 0.5,
            nfe: int = 100,
    ) -> float:
        assert self.dataset is not None

        target_hq_idx = self.dataset.data.iloc[self.target_idx]["idx"]
        target_image_path = os.path.join(self.dataset.img_dir, f"{target_hq_idx}.jpg")
        target_item = Image.open(target_image_path).convert("RGB")
        
        assert self.face_keypoint_detector is not None
        substitution = Substitution(self.dataset, self.face_keypoint_detector)

        inpainted_ref_images = []
        for ref_idx in ref_indices:
            substituted_image = target_item.copy()
            mask_list = []
            for feature_value, feature_name in zip(feature_values, self.features):
                feature = self._parse_feature(feature_name)
                if feature_value == 0:
                    substituted_image = substitution.substitute(
                        src_idx=ref_idx,
                        dest_idx=self.target_idx,
                        feature=feature,
                        image=substituted_image,
                    )
                    mask_dict = self.dataset.get(
                        self.target_idx, feature=feature, inflate_mask=10
                    )
                    if mask_dict and mask_dict.get("mask") is not None:
                        mask_list.append(mask_dict["mask"])

            if mask_list:
                combined_mask = NShapleyValueCalculator._combine_masks(mask_list)
                inpainted_img = self.inpainter.inpaint(
                    image=substituted_image,
                    mask=combined_mask,
                    tau=tau,
                    sampler_type=SampleType.DDPM,
                    nfe=nfe,
                )
                inpainted_ref_images.append(inpainted_img)
            else:
                inpainted_ref_images.append(substituted_image)

        images_ref_tensor = torch.stack([NShapleyValueCalculator.prepare_image(img) for img in inpainted_ref_images]).to(device)

        if predict_prob:
            outputs = model.pred_prob(images_ref_tensor)
        else:
            outputs = model(images_ref_tensor)

        return outputs[:, 0].mean().item()
    
    def calculate_r_squared(
            self,
            ref_indices: list[int],
            model: DenseNetClassifier,
            device: torch.device,
            predict_prob: bool = True,
            tau: float = 0.5,
            nfe: int = 100,
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
                    tau=tau,
                    nfe=nfe,
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
    TARGET_INDEX = 9

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = CelebADataset(split="test")
    face_keypoint_detector = MediapipeFaceKeypointDetector()
    model = get_classifier().to(device)
<<<<<<< HEAD:src/bilinear_model.py
    features = [CompositeFeature.eyes, Feature.nose, CompositeFeature.mouth]
    guidance = CLIPGuidance(load_clip(device=device))
    target_hq_idx = dataset.data.iloc[TARGET_INDEX]["idx"]
    target_image_path = os.path.join(dataset.img_dir, f"{target_hq_idx}.jpg")
    guidance.set_target(target_img=Image.open(target_image_path).convert("RGB"))
    inpainter = I2SB(device=device, guidance=guidance)
=======
    features: list[FeatureType | str] = [CompositeFeature.eyes, Feature.nose, CompositeFeature.mouth]

>>>>>>> fa8f9ccf342137294c63f60bdc665a734b5b779c:src/shapley/bilinear_model.py
    bilinear_model = BilinearModel(
        features=features,
        target_idx=9,
        first_order_values_path="results/9/male/shapley_values/target_9_1_shapley_values.json",
        second_order_values_path="results/9/male/shapley_values/target_9_2_shapley_values.json",
        dataset=dataset,
        face_keypoint_detector=face_keypoint_detector,
        inpainter=inpainter,
    )
    r_squared = bilinear_model.calculate_r_squared(ref_indices=[0, 1, 2, 3, 4], model=model, device=device)
    print(f"R-squared: {r_squared:.4f}")
