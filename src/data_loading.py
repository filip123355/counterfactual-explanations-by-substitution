import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import pandas as pd
from collections import defaultdict

from src.constants import CUB_DATASET, IMAGENET_MEAN, IMAGENET_STD, BATCH_SIZE


def load_cub_images(class_idx: int|None=None,
                    num_images: int|None=None,
) -> tuple[DataLoader, DataLoader]:
    """
    Loads the CUB dataset images and applies train/test split.
    """

    images_path = f"{CUB_DATASET}/CUB_200_2011/CUB_200_2011/images"
    tts_path = f"{CUB_DATASET}/CUB_200_2011/CUB_200_2011/train_test_split.txt"

    # Load images
    data_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])

    dataset = datasets.ImageFolder(
        root=images_path, 
        transform=data_transforms
    )

    if num_images is not None and num_images <= 0:
        raise ValueError("num_images must be a positive integer or None")

    if class_idx is not None and (class_idx < 0 or class_idx >= len(dataset.classes)):
        raise ValueError(
            f"class_idx={class_idx} is out of range. Valid range: 0..{len(dataset.classes)-1}"
        )
    
    selected_indices: list[int] = []
    class_counts: dict[int, int] = defaultdict(int)

    for idx, (_, label) in enumerate(dataset.samples):
        if class_idx is not None and label != class_idx:
            continue

        if num_images is not None and class_counts[label] >= num_images:
            continue

        selected_indices.append(idx)
        class_counts[label] += 1
    
    # Load TTS
    tts_df = pd.read_csv(tts_path, sep=' ', header=None, names=['image_id', 'is_train'])

    train_indices = tts_df[tts_df['is_train'] == 1].index.tolist()
    test_indices = tts_df[tts_df['is_train'] == 0].index.tolist()

    selected_index_set = set(selected_indices)
    train_indices = [i for i in train_indices if i in selected_index_set]
    test_indices = [i for i in test_indices if i in selected_index_set]

    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    test_dataset = torch.utils.data.Subset(dataset, test_indices)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    return train_loader, test_loader


def load_cub_preview_images(
    class_idx: int | None = None,
    max_images: int | None = 24,
    image_size: tuple[int, int] = (224, 224),
) -> tuple[list[torch.Tensor], list[str]]:
    """
    Load a small set of CUB images for visualization.
    """
    if max_images is not None and max_images <= 0:
        raise ValueError("max_images must be a positive integer or None")

    images_path = f"{CUB_DATASET}/CUB_200_2011/CUB_200_2011/images"

    preview_transforms = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
    ])

    dataset = datasets.ImageFolder(root=images_path, transform=preview_transforms)

    if class_idx is not None and (class_idx < 0 or class_idx >= len(dataset.classes)):
        raise ValueError(
            f"class_idx={class_idx} is out of range. Valid range: 0..{len(dataset.classes)-1}"
        )

    images: list[torch.Tensor] = []
    labels: list[str] = []

    for image, label in dataset:
        if class_idx is not None and label != class_idx:
            continue

        images.append(image)
        labels.append(dataset.classes[label])

        if max_images is not None and len(images) >= max_images:
            break

    return images, labels

    
if __name__ == "__main__":
    train_loader, test_loader = load_cub_images()
    print(f"Loaded {len(train_loader.dataset)} training images and {len(test_loader.dataset)} test images.")  
    X, y = next(iter(train_loader))
    print(f"Sample batch shape: {X.shape}, Labels shape: {y.shape}")  