import matplotlib.pyplot as plt
from PIL import Image
from collections.abc import Mapping

from src.data import (
    CelebADataset,
    CelebAFeatureDataset,
    CelebAItem,
    CompositeFeature, FeatureType,
)


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


def show_substitution(
    src_image: Image.Image,
    dest_image: Image.Image,
    substituted_image: Image.Image,
    save_path: str | None = None,
):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(src_image)
    axes[0].set_title("Source Image")
    axes[1].imshow(dest_image)
    axes[1].set_title("Destination Image")
    axes[2].imshow(substituted_image)
    axes[2].set_title("Substituted Image")
    plt.tight_layout()

    if save_path is None:
        plt.show()
    else:
        plt.savefig(save_path)


def show_inpanting(
    original_image: Image.Image,
    subst_image: Image.Image,
    inp_image: Image.Image,
    save_path: str | None = None,
):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    subst_image = subst_image.resize(inp_image.size)

    axes[0].imshow(original_image)
    axes[0].set_title("Original Image")
    axes[1].imshow(subst_image)
    axes[1].set_title("Substituted Image")
    axes[2].imshow(inp_image)
    axes[2].set_title("Inpainted Image")
    plt.tight_layout()

    if save_path is None:
        plt.show()
    else:
        plt.savefig(save_path)


def show_top_k_similar(
    query_image: Image.Image,
    top_k_images: list[Image.Image],
):
    k = len(top_k_images)
    fig, axes = plt.subplots(1, k + 1, figsize=(5 * (k + 1), 5))
    axes[0].imshow(query_image)
    axes[0].set_title("Query Image")
    for i, img in enumerate(top_k_images):
        axes[i + 1].imshow(img)
        axes[i + 1].set_title(f"Top-{i + 1} Similar")
    plt.tight_layout()
    plt.show()


def show_shapley_values(
    shapley_values: Mapping[tuple[FeatureType, ...], float],
    save_path: str | None = None,
    title: str = "Shapley Values",
):
    items = sorted(shapley_values.items(), key=lambda item: item[1], reverse=True)
    labels = [str(key.value if hasattr(key, "value") else key) for key, _ in items]
    values = [value for _, value in items]

    fig, ax = plt.subplots(figsize=(max(8, 1.4 * len(items)), 5))
    colors = ["#2E86DE" if value >= 0 else "#D63625" for value in values]
    ax.bar(labels, values, color=colors)
    ax.axhline(0.0, color="black", linewidth=1)
    ax.set_title(title)
    ax.set_ylabel("Contribution")
    ax.set_xlabel("Feature")
    ax.tick_params(axis="x", rotation=30)
    plt.tight_layout()

    if save_path is None:
        plt.show()
    else:
        plt.savefig(save_path, bbox_inches="tight")


if __name__ == "__main__":
    dataset = CelebAFeatureDataset(
        dataset=CelebADataset(split="test"),
        feature=CompositeFeature.eyes,
        transform=None,
    )
    item = dataset[0]
    show_bbox_and_mask(item)
