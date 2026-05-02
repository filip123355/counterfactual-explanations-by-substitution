import os
import torch
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from src.constants import DATASET, IMAGENET_MEAN, IMAGENET_STD, BATCH_SIZE

class CelebAFeatureDataset(Dataset):

    ALL_FEATURES: list[str] = [
        'skin', 'l_brow', 'r_brow', 'l_eye', 'r_eye', 'eye_g', 'l_ear', 'r_ear', 
        'ear_r', 'nose', 'mouth', 'u_lip', 'l_lip', 'neck', 'neck_l', 'cloth', 
        'hair', 'hat'
    ]
    
    FEATURE_MAP: dict[str, list[str]] = {
        'eyes': ['l_eye', 'r_eye'],
        'eyebrows': ['l_brow', 'r_brow'],
        'mouth': ['u_lip', 'l_lip', 'mouth'],
        'nose': ['nose'],
        'ears': ['l_ear', 'r_ear'],
        'accessories': ['eye_g', 'ear_r', 'neck_l', 'hat'],
        'face_full': ['skin', 'l_eye', 'r_eye', 'l_brow', 'r_brow', 'nose', 'u_lip', 'l_lip', 'mouth']
    }

    def __init__(self, root_dir: str, 
                 partition_file: str, 
                 mapping_file: str, 
                 split: str='train', 
                 feature_name: str='eyes', 
                 transform=None, 
                 padding: int=20):
        
        self.root_dir = root_dir
        self.mask_dir = os.path.join(root_dir, 'CelebAMask-HQ-mask-anno')
        self.img_dir = os.path.join(root_dir, 'CelebA-HQ-img')
        self.transform = transform
        self.feature_parts = self.FEATURE_MAP.get(feature_name, [feature_name])
        self.padding = padding

        mapping_df = pd.read_csv(mapping_file, sep='\\s+', header=0)
        partition_df = pd.read_csv(partition_file, sep='\\s+', header=None, names=['orig_file', 'split'])
        merged = pd.merge(mapping_df, partition_df, on='orig_file')
        
        split_map = {'train': 0, 'val': 1, 'test': 2}
        self.data = merged[merged['split'] == split_map[split]].copy()

    def _get_bbox_from_mask(self, 
                            hq_idx: int,
    ) -> tuple[int, int, int, int] | None:
        """Generates bounding box coordinates based on the fature-specific mask.
        """

        combined_mask = None
        
        # Specyficzny podział katalogów masek
        folder_idx = hq_idx // 2000
        curr_mask_path = os.path.join(self.mask_dir, str(folder_idx))

        for part in self.feature_parts:
            mask_file = os.path.join(curr_mask_path, f"{hq_idx:05d}_{part}.png") # Masks are on .png format
            
            if os.path.exists(mask_file):
                mask = np.array(Image.open(mask_file).convert('L'))
                if combined_mask is None:
                    combined_mask = mask
                else:
                    combined_mask = np.maximum(combined_mask, mask)

        if combined_mask is None or np.max(combined_mask) == 0:
            return None 
        
        pos = np.where(combined_mask > 0)
        ymin, ymax = np.min(pos[0]), np.max(pos[0])
        xmin, xmax = np.min(pos[1]), np.max(pos[1])
        
        # Producing bounding box 
        return (ymin, ymax, xmin, xmax)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, 
                    idx: int,
    ) -> tuple[torch.Tensor, int]:
        hq_idx = self.data.iloc[idx]['idx']
        img_path = os.path.join(self.img_dir, f"{hq_idx}.jpg")
        
        image = Image.open(img_path).convert('RGB')
        bbox = self._get_bbox_from_mask(hq_idx)

        if bbox:
            ymin, ymax, xmin, xmax = bbox
            w, h = image.size
            ymin = max(0, ymin - self.padding)
            ymax = min(h, ymax + self.padding)
            xmin = max(0, xmin - self.padding)
            xmax = min(w, xmax + self.padding)
            
            image = image.crop((xmin, ymin, xmax, ymax))

        if self.transform:
            image = self.transform(image)
            
        return image, hq_idx

def get_feature_loader(split: str='train', 
                       feature: str='eyes', 
                       batch_size: int=BATCH_SIZE,
) -> DataLoader:
    """Fast loader producer.
    """
    
    data_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])

    dataset = CelebAFeatureDataset(
        root_dir=DATASET,
        partition_file=os.path.join(DATASET, "list_eval_partition.txt"),
        mapping_file=os.path.join(DATASET, "CelebA-HQ-to-CelebA-mapping.txt"),
        split=split,
        feature_name=feature,
        transform=data_transforms
    )

    return DataLoader(dataset, batch_size=batch_size, shuffle=(split == 'train'))

if __name__ == "__main__":
    # Smoke test
    eye_loader = get_feature_loader(split='test', feature='eyes')
    
    images, ids = next(iter(eye_loader))
    print(f"Batch shape: {images.shape}")
    print(f"ID images: {ids[:67]}")