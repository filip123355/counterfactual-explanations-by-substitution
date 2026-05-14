import itertools
import math
import json
import os
from typing import List

import numpy as np
import torch
from loguru import logger
from PIL import Image


from src.data_loading import CelebADataset, CompositeFeature, Feature, FeatureType
from src.inpainter.i2sb import I2SB, SampleType
from src.inpainter.guidance.classifier import DenseNetClassifier, get_classifier
from src.substitution import Substitution
from src.keypoints import MediapipeFaceKeypointDetector
from src.inpainter.guidance import CLIPGuidance
from src.clip_inferance import load_clip
from src.visualize import show_shapley_values
from src.constants import PROJECT_ROOT


def _shapley_key_to_str(key: object) -> str:
    return str(key.value if hasattr(key, "value") else key)

class ShapleyValueCalculator:
    def __init__(
        self,
        dataset: CelebADataset,
        substitution: Substitution,
        inpainter: I2SB,
        background_as_feature: bool = False,
    ):

        self.dataset = dataset
        self.substitution = substitution
        self.inpainter = inpainter
        self.background_as_feature = background_as_feature

    @staticmethod
    def _combine_masks(mask_list: List[np.ndarray]) -> np.ndarray:
        if not mask_list:
            return None
        combined = mask_list[0].copy()
        for m in mask_list[1:]:
            combined = np.logical_or(combined, m).astype(np.uint8) * 255
        return combined
        
    def prepare_coalitions_inpainting(
        self,
        target_idx: int,
        ref_indices: List[int],
        features: List[FeatureType],
    ) -> dict:
        N = set(features)
        coalition_images = {}

        target_hq_idx = self.dataset.data.iloc[target_idx]["idx"]
        target_image_path = os.path.join(self.dataset.img_dir, f"{target_hq_idx}.jpg")
        target_item = Image.open(target_image_path).convert("RGB")
        
        all_subsets = []
        for r in range(len(N) + 1):
            all_subsets.extend(itertools.combinations(N, r))
            
        for S in all_subsets:
            S_set = set(S)
            not_in_S = list(N - S_set) 
            
            inpainted_for_S = []
            
            for ref_idx in ref_indices:
                current_img = target_item.copy()
                mask_list = []

                for feat in not_in_S:
                    current_img = self.substitution.substitute(
                        src_idx=ref_idx, 
                        dest_idx=target_idx, 
                        feature=feat, 
                        image=current_img
                    )
                    
                    mask_dict = self.dataset.get(target_idx, feature=feat, inflate_mask=10)
                    if mask_dict and mask_dict.get("mask") is not None:
                        mask_list.append(mask_dict["mask"])
                
                if mask_list:
                    combined_mask = self._combine_masks(mask_list)
                    inpainted_img = self.inpainter.inpaint(
                        image=current_img,
                        mask=combined_mask,
                        tau=0.5,
                        sampler_type=SampleType.DDPM,
                        nfe=100,
                    )
                    inpainted_for_S.append(inpainted_img)
                else:
                    inpainted_for_S.append(current_img)
                    
            coalition_images[tuple(sorted(S))] = inpainted_for_S
            
        return coalition_images
    
    def prepare_image(self, image: Image.Image) -> torch.Tensor:
        tensor = torch.from_numpy(np.array(image)) / 255.0
        return tensor.permute(2, 0, 1).float() 

    def compute_shapley_values(
        self,
        model: DenseNetClassifier,
        coalition_images: dict,
        features: List[FeatureType],
        device: torch.device,
    ) -> dict:
        
        N = sorted(features)
        num_features = len(N)
        
        v = {}
        for S, images_list in coalition_images.items():
            images = torch.stack([self.prepare_image(img) for img in images_list], dim=0).to(device)
            probs = model.pred_prob(images)
            mean_prob = probs[:, 0].mean().item()
            v[S] = mean_prob

        shapley_values = {feat: 0.0 for feat in N}
        
        for feat in N:
            subsets_without_i = [S for S in v.keys() if feat not in S]
            
            for S in subsets_without_i:
                S_union_i = tuple(sorted(list(S) + [feat]))
                
                S_size = len(S)
                weight = (
                    math.factorial(S_size) * 
                    math.factorial(num_features - S_size - 1) / 
                    math.factorial(num_features)
                )
                marginal_contribution = v[S_union_i] - v[S]
                
                shapley_values[feat] += weight * marginal_contribution
                
        return shapley_values
    

if __name__ == "__main__":
    face_keypoint_detector = None
    TARGET_INDEX = 9
    try:
        dataset = CelebADataset(split="test")
        face_keypoint_detector = MediapipeFaceKeypointDetector()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        guidance = CLIPGuidance(load_clip(device=device))

        target_hq_idx = dataset.data.iloc[TARGET_INDEX]["idx"]
        target_image_path = os.path.join(dataset.img_dir, f"{target_hq_idx}.jpg")
        guidance.set_target(target_img=Image.open(target_image_path).convert("RGB"))

        inpainter = I2SB(device=device, guidance=guidance)
        shap_calculator = ShapleyValueCalculator(
            dataset=dataset,
            substitution=Substitution(dataset, face_keypoint_detector),
            inpainter=inpainter,
            background_as_feature=False,
        )

        coalition_images = shap_calculator.prepare_coalitions_inpainting(
            target_idx=TARGET_INDEX,
            ref_indices=[0, 1, 2, 3, 4],
            features=[CompositeFeature.eyes, Feature.nose, CompositeFeature.mouth]
        )

        model = get_classifier().to(device)

        shapely_values = shap_calculator.compute_shapley_values(
            model=model,
            coalition_images=coalition_images,
            features=[CompositeFeature.eyes, Feature.nose, CompositeFeature.mouth],
            device=device,
        )
        print("Shapley Values:", shapely_values)

        values_dir = os.path.join(PROJECT_ROOT, "results", "shapley_values")
        plots_dir = os.path.join(PROJECT_ROOT, "results", "shapley_plots")
        os.makedirs(values_dir, exist_ok=True)
        os.makedirs(plots_dir, exist_ok=True)

        value_file = os.path.join(values_dir, f"target_{TARGET_INDEX}_shapley_values.json")
        plot_file = os.path.join(plots_dir, f"target_{TARGET_INDEX}_shapley_values.png")

        with open(value_file, "w", encoding="utf-8") as handle:
            json.dump(
                {_shapley_key_to_str(key): value for key, value in shapely_values.items()},
                handle,
                indent=2,
                ensure_ascii=False,
            )

        show_shapley_values(
            shapely_values,
            save_path=plot_file,
            title="Shapley Values for Facial Features",
        )
        print(f"Saved Shapley values to {value_file}")
        print(f"Saved Shapley plot to {plot_file}")

    finally:
        if face_keypoint_detector is not None:
            try:
                face_keypoint_detector.close()
            except Exception:
                pass
            finally:
                face_keypoint_detector = None

