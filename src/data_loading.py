import os
from enum import StrEnum
from typing import TypedDict

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


class CelebAItem(TypedDict):
    hq_idx: int
    full_image: Image.Image
    cropped_image: Image.Image | torch.Tensor | None
    bbox: tuple[int, int, int, int] | None
    mask: np.ndarray | None


class Features(StrEnum):
    eyes = "eyes"
    eyebrows = "eyebrows"
    mouth = "mouth"
    nose = "nose"
    ears = "ears"
    hair = "hair"
    accessories = "accessories"
    face_full = "face_full"


class CelebAFeatureDataset(Dataset[CelebAItem]):
    # fmt: off
    ALL_FEATURES: list[str] = [
        'skin', 'l_brow', 'r_brow', 'l_eye', 'r_eye', 'eye_g', 'l_ear', 'r_ear', 
        'ear_r', 'nose', 'mouth', 'u_lip', 'l_lip', 'neck', 'neck_l', 'cloth', 
        'hair', 'hat'
    ]

    FEATURE_MAP: dict[Features, list[str]] = {
        Features.eyes: ['l_eye', 'r_eye'],
        Features.eyebrows: ['l_brow', 'r_brow'],
        Features.mouth: ['u_lip', 'l_lip', 'mouth'],
        Features.nose: ['nose'],
        Features.ears: ['l_ear', 'r_ear'],
        Features.hair: ['hair'],
        Features.accessories: ['eye_g', 'ear_r', 'neck_l', 'hat'],
        Features.face_full: ['skin', 'l_eye', 'r_eye', 'l_brow', 'r_brow', 'nose', 'u_lip', 'l_lip', 'mouth']
    }
    # fmt: on

    def __init__(
        self,
        root_dir: str,
        partition_file: str,
        mapping_file: str,
        split: str = "train",
        feature_name: Features = Features.eyes,
        transform=None,
        padding: int = 20,
    ):

        self.root_dir = root_dir
        self.mask_dir = os.path.join(root_dir, "CelebAMask-HQ-mask-anno")
        self.img_dir = os.path.join(root_dir, "CelebA-HQ-img")
        self.transform = transform
        self.feature_parts = self.FEATURE_MAP.get(feature_name, [feature_name])
        self.padding = padding

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
    ) -> tuple[tuple[int, int, int, int], np.ndarray] | None:
        """Generates bounding box coordinates based on the fature-specific mask."""

        combined_mask = None

        # Specyficzny podział katalogów masek
        folder_idx = hq_idx // 2000
        curr_mask_path = os.path.join(self.mask_dir, str(folder_idx))

        for part in self.feature_parts:
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

        pos = np.where(combined_mask > 0)
        ymin, ymax = np.min(pos[0]).item(), np.max(pos[0]).item()
        xmin, xmax = np.min(pos[1]).item(), np.max(pos[1]).item()

        # Producing bounding box
        return (ymin, ymax, xmin, xmax), combined_mask

    def __len__(self):
        return len(self.data)

    def __getitem__(
        self,
        index: int,
    ) -> CelebAItem:
        hq_idx = self.data.iloc[index]["idx"]
        img_path = os.path.join(self.img_dir, f"{hq_idx}.jpg")

        full_image = Image.open(img_path).convert("RGB")
        bbox_and_mask = self._get_bbox_and_mask(hq_idx)

        if bbox_and_mask is None:
            cropped_image = None
            bbox = None
            mask = None
        else:
            bbox, mask = bbox_and_mask

            ymin, ymax, xmin, xmax = bbox
            w, h = full_image.size
            ymin = max(0, ymin - self.padding)
            ymax = min(h, ymax + self.padding)
            xmin = max(0, xmin - self.padding)
            xmax = min(w, xmax + self.padding)

            cropped_image = full_image.crop((xmin, ymin, xmax, ymax))

            if self.transform:
                cropped_image = self.transform(cropped_image)

        return {
            "hq_idx": hq_idx,
            "full_image": full_image,
            "cropped_image": cropped_image,
            "bbox": bbox,
            "mask": mask,
        }


def get_feature_dataset(
    split: str = "train",
    feature: Features = Features.eyes,
    transforms=DEFAULT_TRANSFORMS,
) -> CelebAFeatureDataset:
    return CelebAFeatureDataset(
        root_dir=DATASET,
        partition_file=os.path.join(DATASET, "list_eval_partition.txt"),
        mapping_file=os.path.join(DATASET, "CelebA-HQ-to-CelebA-mapping.txt"),
        split=split,
        feature_name=feature,
        transform=transforms,
    )


class CelebAFeatureDatasetFactory:
    cache: dict[tuple[str, Features], CelebAFeatureDataset] = {}
    transforms: transforms.Compose

    def __init__(self, transforms=DEFAULT_TRANSFORMS):
        self.transforms = transforms

    def get_dataset(self, split: str, feature: Features) -> CelebAFeatureDataset:
        key = (split, feature)
        if key not in self.cache:
            self.cache[key] = get_feature_dataset(
                split=split, feature=feature, transforms=self.transforms
            )
        return self.cache[key]


def get_feature_loader(
    split: str = "train",
    feature: Features = Features.eyes,
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

    dataset = get_feature_dataset(split=split, feature=feature)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == "train"),
        collate_fn=collate_fn,
    )


if __name__ == "__main__":
    # Smoke test
    eye_loader = get_feature_loader(split="test", feature=Features.eyes)

    batch = next(iter(eye_loader))
    logger.info(f"ID images: {batch['hq_idx'][:67]}")
    logger.info(f"Full images shape: {batch['full_image'].shape}")
    logger.info(f"Cropped images shape: {batch['cropped_image'].shape}")
    logger.info(f"BBoxes shape: {batch['bbox'].shape}")
    logger.info(f"Masks shape: {batch['mask'].shape}")
