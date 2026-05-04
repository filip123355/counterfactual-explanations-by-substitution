import os
from enum import StrEnum
from typing import TypedDict

import cv2
import numpy as np
import pandas as pd
import torch
from loguru import logger
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from src.constants import BATCH_SIZE, DATASET, IMAGENET_MEAN, IMAGENET_STD

DEFAULT_TRANSFORMS = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ]
)

CELEB_HQ_SIZE = (1024, 1024)


class CelebAItem(TypedDict):
    hq_idx: int
    full_image: Image.Image
    cropped_image: Image.Image | torch.Tensor | None
    bbox: tuple[int, int, int, int] | None
    mask: np.ndarray | None


class Feature(StrEnum):
    skin = "skin"
    l_brow = "l_brow"
    r_brow = "r_brow"
    l_eye = "l_eye"
    r_eye = "r_eye"
    eye_g = "eye_g"
    l_ear = "l_ear"
    r_ear = "r_ear"
    ear_r = "ear_r"
    nose = "nose"
    mouth = "mouth"
    u_lip = "u_lip"
    l_lip = "l_lip"
    neck = "neck"
    neck_l = "neck_l"
    cloth = "cloth"
    hair = "hair"
    hat = "hat"


class CompositeFeature(StrEnum):
    eyes = "eyes"
    eyebrows = "eyebrows"
    mouth = "mouth"
    ears = "ears"
    accessories = "accessories"
    face_full = "face_full"


FeatureType = Feature | CompositeFeature


def get_base_features(feature: FeatureType) -> list[Feature]:
    if isinstance(feature, Feature):
        return [feature]
    elif isinstance(feature, CompositeFeature):
        return CelebADataset.FEATURE_MAP[feature]
    else:
        raise ValueError(f"Invalid feature type: {feature}")


class CelebADataset:
    root_dir: str
    mask_dir: str
    img_dir: str
    data: pd.DataFrame

    # fmt: off
    FEATURE_MAP: dict[CompositeFeature, list[Feature]] = {
        CompositeFeature.eyes: [Feature.l_eye, Feature.r_eye],
        CompositeFeature.eyebrows: [Feature.l_brow, Feature.r_brow],
        CompositeFeature.mouth: [Feature.u_lip, Feature.l_lip, Feature.mouth],
        CompositeFeature.ears: [Feature.l_ear, Feature.r_ear],
        CompositeFeature.accessories: [Feature.eye_g, Feature.ear_r, Feature.neck_l, Feature.hat],
        CompositeFeature.face_full: [Feature.skin, Feature.l_eye, Feature.r_eye, Feature.l_brow,
                                     Feature.r_brow, Feature.nose, Feature.u_lip, Feature.l_lip, Feature.mouth]
    }
    # fmt: on

    def __init__(
        self,
        root_dir=DATASET,
        partition_file=os.path.join(DATASET, "list_eval_partition.txt"),
        mapping_file=os.path.join(DATASET, "CelebA-HQ-to-CelebA-mapping.txt"),
        split: str = "train",
    ):

        self.root_dir = root_dir
        self.mask_dir = os.path.join(root_dir, "CelebAMask-HQ-mask-anno")
        self.img_dir = os.path.join(root_dir, "CelebA-HQ-img")

        mapping_df = pd.read_csv(mapping_file, sep="\\s+", header=0)
        partition_df = pd.read_csv(
            partition_file, sep="\\s+", header=None, names=["orig_file", "split"]
        )
        merged = pd.merge(mapping_df, partition_df, on="orig_file")

        split_map = {"train": 0, "val": 1, "test": 2}
        self.data = merged[merged["split"] == split_map[split]].copy()

    def _get_bbox_and_mask(
        self,
        hq_idx: int,
        feature: FeatureType,
    ) -> tuple[tuple[int, int, int, int], np.ndarray] | None:
        """Generates bounding box coordinates based on the fature-specific mask."""

        combined_mask = None

        # Specyficzny podział katalogów masek
        folder_idx = hq_idx // 2000
        curr_mask_path = os.path.join(self.mask_dir, str(folder_idx))

        for part in get_base_features(feature):
            mask_file = os.path.join(
                curr_mask_path, f"{hq_idx:05d}_{part}.png"
            )  # Masks are on .png format

            if os.path.exists(mask_file):
                mask = np.array(Image.open(mask_file).convert("L"))
                if combined_mask is None:
                    combined_mask = mask
                else:
                    combined_mask = np.maximum(combined_mask, mask)

        if combined_mask is None or np.max(combined_mask) == 0:
            return None

        combined_mask_bgr = cv2.cvtColor(combined_mask, cv2.COLOR_GRAY2BGR)
        combined_mask_bgr = cv2.resize(
            combined_mask_bgr, CELEB_HQ_SIZE, interpolation=cv2.INTER_NEAREST
        )
        combined_mask = cv2.cvtColor(combined_mask_bgr, cv2.COLOR_BGR2GRAY)

        pos = np.where(combined_mask > 0)
        ymin, ymax = np.min(pos[0]).item(), np.max(pos[0]).item()
        xmin, xmax = np.min(pos[1]).item(), np.max(pos[1]).item()

        # Producing bounding box
        return (ymin, ymax, xmin, xmax), combined_mask

    def __len__(self):
        return len(self.data)

    def get(
        self,
        index: int,
        feature: FeatureType,
        padding: int = 20,
    ) -> CelebAItem:
        hq_idx = self.data.iloc[index]["idx"]
        img_path = os.path.join(self.img_dir, f"{hq_idx}.jpg")

        full_image = Image.open(img_path).convert("RGB")
        bbox_and_mask = self._get_bbox_and_mask(hq_idx, feature)

        if bbox_and_mask is None:
            cropped_image = None
            bbox = None
            mask = None
        else:
            bbox, mask = bbox_and_mask

            ymin, ymax, xmin, xmax = bbox
            w, h = full_image.size
            assert w == CELEB_HQ_SIZE[0] and h == CELEB_HQ_SIZE[1]

            ymin = max(0, ymin - padding)
            ymax = min(h, ymax + padding)
            xmin = max(0, xmin - padding)
            xmax = min(w, xmax + padding)

            cropped_image = full_image.crop((xmin, ymin, xmax, ymax))

        return {
            "hq_idx": hq_idx,
            "full_image": full_image,
            "cropped_image": cropped_image,
            "bbox": bbox,
            "mask": mask,
        }


class CelebAFeatureDataset(Dataset[CelebAItem]):
    dataset: CelebADataset
    feature: Feature | CompositeFeature
    transform: transforms.Compose | None
    padding: int

    def __init__(
        self,
        dataset: CelebADataset,
        feature: FeatureType,
        *,
        transform=None,
        padding: int = 20,
    ):
        self.dataset = dataset
        self.transform = transform
        self.padding = padding
        self.feature = feature

    def __len__(self):
        return len(self.dataset)

    def __getitem__(
        self,
        index: int,
    ) -> CelebAItem:
        item = self.dataset.get(index, feature=self.feature, padding=self.padding)
        cropped_image = item["cropped_image"]

        if self.transform:
            cropped_image = self.transform(cropped_image)

        return {
            **item,
            "cropped_image": cropped_image,
        }


def get_feature_loader(
    feature: FeatureType,
    split: str = "train",
    batch_size: int = BATCH_SIZE,
) -> DataLoader:
    """Fast loader producer."""

    def collate_fn(batch: list[CelebAItem]) -> dict[str, np.ndarray]:
        return {
            "hq_idx": np.array([item["hq_idx"] for item in batch]),
            "full_image": np.array([item["full_image"] for item in batch]),
            "cropped_image": np.array([item["cropped_image"] for item in batch]),
            "bbox": np.array([item["bbox"] for item in batch]),
            "mask": np.array([item["mask"] for item in batch]),
        }

    dataset = CelebAFeatureDataset(
        dataset=CelebADataset(split=split),
        feature=feature,
        transform=DEFAULT_TRANSFORMS,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == "train"),
        collate_fn=collate_fn,
    )


if __name__ == "__main__":
    # Smoke test
    eye_loader = get_feature_loader(split="test", feature=CompositeFeature.eyes)

    batch = next(iter(eye_loader))
    logger.info(f"ID images: {batch['hq_idx'][:67]}")
    logger.info(f"Full images shape: {batch['full_image'].shape}")
    logger.info(f"Cropped images shape: {batch['cropped_image'].shape}")
    logger.info(f"BBoxes shape: {batch['bbox'].shape}")
    logger.info(f"Masks shape: {batch['mask'].shape}")
