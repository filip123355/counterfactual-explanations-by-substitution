import random
from pathlib import Path

import numpy as np
import torch
from loguru import logger
from tqdm import tqdm

from src.constants import I2SB_IMAGE_SIZE, I2SB_MASK_INFLATION
from src.data import CelebADataset, CompositeFeature, Feature
from src.inpainter.guidance import ClassifierGuidance
from src.inpainter.i2sb import I2SB, SampleType
from src.substitution import Substitution, MediapipeFaceKeypointDetector, ColorFillSubstitution


class FIDGenerator:
    dataset: CelebADataset
    substitution: Substitution
    inpainter: I2SB | None
    output_path: Path

    def __init__(
        self,
        dataset: CelebADataset,
        substitution: Substitution,
        inpainter: I2SB | None,
        output_path: str | Path,
    ):
        self.dataset = dataset
        self.substitution = substitution
        self.inpainter = inpainter
        self.output_path = Path(output_path)
        self.output_path.mkdir(parents=True, exist_ok=True)

        self.features = [
            CompositeFeature.eyes,
            CompositeFeature.mouth,
            Feature.nose,
        ]

    def generate(self, n_samples: int, tau: float = 1.0, nfe: int | None = None):
        generated_count = 0
        pbar = tqdm(total=n_samples, desc="Generating images")

        while generated_count < n_samples:
            src_idx, dest_idx = random.sample(range(len(self.dataset)), 2)

            num_features = random.randint(1, len(self.features))
            selected_features = random.sample(self.features, num_features)

            pbar.set_postfix(
                {
                    "src_idx": src_idx,
                    "dest_idx": dest_idx,
                    "features": [f.value for f in selected_features],
                }
            )

            subst_image = None
            combined_mask = None
            dest_image = None
            label_value = None

            try:
                for feature in selected_features:
                    subst_image = self.substitution.substitute(
                        src_idx=src_idx,
                        dest_idx=dest_idx,
                        feature=feature,
                        image=subst_image,
                    )

                    dest_item = self.dataset.get(
                        dest_idx, feature=feature, inflate_mask=I2SB_MASK_INFLATION
                    )
                    dest_mask = dest_item["mask"]

                    if dest_mask is None:
                        raise ValueError(f"Missing mask for {feature}")

                    if combined_mask is None:
                        combined_mask = dest_mask
                        dest_image = dest_item["full_image"]
                        label_value = dest_item["label_value"]
                    else:
                        combined_mask = np.maximum(combined_mask, dest_mask)

                assert (
                    combined_mask is not None
                    and subst_image is not None
                    and dest_image is not None
                )

                if self.inpainter is not None:
                    if self.inpainter.guidance is not None:
                        assert label_value is not None
                        self.inpainter.guidance.set_target(
                            target_img=dest_image,
                            label_value=label_value,
                        )

                    inpainted_image = self.inpainter.inpaint(
                        image=subst_image,
                        mask=combined_mask,
                        tau=tau,
                        nfe=nfe,
                        sampler_type=SampleType.DDPM,
                    )
                else:
                    inpainted_image = subst_image

                out_file = self.output_path / f"{generated_count:05d}.png"

                resized_image = inpainted_image.resize(
                    (I2SB_IMAGE_SIZE, I2SB_IMAGE_SIZE)
                )
                resized_image.save(out_file)
                # dest_image.save(self.output_path / f"{generated_count:05d}_dest.png")

                generated_count += 1
                pbar.update(1)

            except Exception as e:
                logger.debug(f"Skipping pair {src_idx}->{dest_idx}: {e}")
                continue

        pbar.close()


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    random.seed(42)
    tau = 0.3

    dataset = CelebADataset(split="test")
    face_keypoint_detector = MediapipeFaceKeypointDetector()
    # substitution = ImageSubstitution(dataset, face_keypoint_detector)
    substitution = ColorFillSubstitution(dataset)

    guidance = ClassifierGuidance(device=device, tau=tau)
    inpainter = None #I2SB(device=device, guidance=guidance)

    guidance_str = guidance.__class__.__name__ if guidance else "no_guidance"
    inpainter_str = inpainter.__class__.__name__ if inpainter else "no_inpainter"

    real_images_path = dataset.img_dir
    generated_images_path = f"generated/fid_samples_color_fill_{inpainter_str}_{guidance_str}"

    generator = FIDGenerator(
        dataset=dataset,
        substitution=substitution,
        inpainter=inpainter,
        output_path=generated_images_path,
    )

    generator.generate(n_samples=1000, tau=tau)
