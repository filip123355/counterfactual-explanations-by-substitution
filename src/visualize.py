import matplotlib.pyplot as plt
from PIL import Image

from src.data_loading import CelebAItem, CompositeFeature, CelebAFeatureDataset, CelebADataset


def show_bbox_and_mask(item: CelebAItem):
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    axes[0].imshow(item["full_image"])
    axes[0].set_title("Full Image")
    if item["bbox"] is not None:
        ymin, ymax, xmin, xmax = item["bbox"]
        rect = plt.Rectangle(
            (xmin, ymin),
            xmax - xmin,
            ymax - ymin,
            edgecolor="red",
            facecolor="none",
            linewidth=2,
        )
        axes[0].add_patch(rect)

    if item["mask"] is not None:
        masked_image = item["full_image"].copy()
        masked_image.putalpha(Image.fromarray(item["mask"]))
        axes[1].imshow(masked_image)
        axes[1].set_title("Masked Image")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    dataset = CelebAFeatureDataset(
        dataset=CelebADataset(split="test"),
        feature=CompositeFeature.eyes,
        transform=None,
    )
    item = dataset[0]
    show_bbox_and_mask(item)
