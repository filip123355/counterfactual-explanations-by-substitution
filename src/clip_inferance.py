from typing import List, TypedDict, Union

import numpy as np
import torch
from loguru import logger
from PIL import Image
from torch.utils.data import DataLoader
from transformers import CLIPModel, CLIPProcessor

from src.constants import BATCH_SIZE, CLIP_MODEL_NAME, DATASET, USE_FP16
from src.data_loading import (
    CelebADataset,
    CelebAFeatureDataset,
    CelebAItem,
    CompositeFeature,
)


class SimilarityResult(TypedDict):
    indices: np.ndarray
    scores: np.ndarray | None


class CLIPInference:
    """CLIP-based image dot product similarity computation."""

    def __init__(
        self, model_name: str = CLIP_MODEL_NAME, device: torch.device | None = None
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Loading CLIP from {model_name} on {self.device}...")

        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model = CLIPModel.from_pretrained(model_name).to(self.device).eval()  # ty: ignore

        if USE_FP16:
            logger.info("Converting CLIP model to FP16...")
            self.model = self.model.half()

    def _load_images(
        self, images: Union[str, Image.Image, List[Union[str, Image.Image]]]
    ) -> List[Image.Image]:

        if isinstance(images, str):
            # Single file path
            return [Image.open(images).convert("RGB")]
        elif isinstance(images, Image.Image):
            # Single PIL Image
            return [images.convert("RGB")]
        elif isinstance(images, list):
            # List of PIL Images
            result = []
            for img in images:
                if isinstance(img, str):
                    result.append(Image.open(img).convert("RGB"))
                elif isinstance(img, Image.Image):
                    result.append(img.convert("RGB"))
                else:
                    raise TypeError(f"Unsupported image type in list: {type(img)}")
            return result
        else:
            raise TypeError(f"Unsupported input type: {type(images)}")

    # This assumes tensor is in the format right for CLIP
    def compute_image_embedding_from_tensor(
        self, image_tensor: torch.Tensor, normalize: bool = True
    ) -> torch.Tensor:
        if image_tensor.ndim == 3:
            image_tensor = image_tensor.unsqueeze(0)

        image_tensor = image_tensor.to(self.device)
        return self._compute_image_embeddings_from_preprocessed(
            {"pixel_values": image_tensor}, norm=normalize
        )

    @torch.no_grad()
    def compute_image_embeddings(
        self,
        images: str | Image.Image | List[str | Image.Image],
        normalize: bool = True,
    ) -> torch.Tensor:
        pil_images = self._load_images(images)
        inputs = self.processor(images=pil_images, return_tensors="pt").to(self.device)
        return self._compute_image_embeddings_from_preprocessed(inputs, norm=normalize)

    def _compute_image_embeddings_from_preprocessed(
        self, inputs, norm: bool = True
    ) -> torch.Tensor:
        output = self.model.get_image_features(**inputs)
        embeddings = None

        if isinstance(output, torch.Tensor):
            embeddings = output
        elif hasattr(output, "image_embeds"):
            embeddings = output.image_embeds
        elif hasattr(output, "pooler_output"):
            embeddings = output.pooler_output
        else:
            raise TypeError(f"Unsupported CLIP output type: {type(output)}")

        if norm:
            embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)

        return embeddings

    def _compute_embeddings_for_input(
        self,
        images: Union[
            str, Image.Image, List[Union[str, Image.Image]], CelebAFeatureDataset
        ],
    ) -> np.ndarray:
        if isinstance(images, CelebAFeatureDataset):
            return self.compute_dataset_embeddings(images)[0]

        return self.compute_image_embeddings(images, normalize=True).cpu().numpy()

    def compute_similarity(
        self,
        query_images: Union[
            str, Image.Image, List[Union[str, Image.Image]], CelebAFeatureDataset
        ],
        reference_images: Union[
            str, Image.Image, List[Union[str, Image.Image]], CelebAFeatureDataset
        ],
    ) -> np.ndarray:
        query_embs = self._compute_embeddings_for_input(query_images)
        ref_embs = self._compute_embeddings_for_input(reference_images)

        similarities = query_embs @ ref_embs.T

        return similarities

    def find_top_k_similar(
        self,
        query_images: Union[
            str, Image.Image, List[Union[str, Image.Image]], CelebAFeatureDataset
        ],
        reference_images: Union[
            str, Image.Image, List[Union[str, Image.Image]], CelebAFeatureDataset
        ],
        k: int = 5,
        return_scores: bool = True,
    ) -> SimilarityResult:
        similarities = self.compute_similarity(query_images, reference_images)
        top_k_scores, top_k_indices = torch.topk(torch.tensor(similarities), k=k, dim=1)

        result: SimilarityResult = {"indices": top_k_indices.numpy(), "scores": None}
        if return_scores:
            result["scores"] = top_k_scores.numpy()

        return result

    @torch.no_grad()
    def compute_dataset_embeddings(
        self, dataset: CelebAFeatureDataset, batch_size: int = BATCH_SIZE
    ) -> tuple[np.ndarray, np.ndarray]:
        def replace_none_with_empty(
            cropped_image: Image.Image | torch.Tensor | None,
        ) -> Image.Image:
            if isinstance(cropped_image, torch.Tensor):
                raise TypeError(
                    "Expected cropped_image to be a PIL Image or None, but got a torch.Tensor."
                )
            if cropped_image is None:
                return Image.new("RGB", (224, 224), color="black")
            return cropped_image

        def collate_fn(batch: list[CelebAItem]):
            images = [replace_none_with_empty(item["cropped_image"]) for item in batch]
            ids = [item["hq_idx"] for item in batch]
            return images, ids

        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn
        )

        embeddings_list = []
        ids_list = []

        for batch_images, batch_ids in loader:
            embs = self.compute_image_embeddings(batch_images, normalize=True)
            embeddings_list.append(embs.cpu().numpy())
            ids_list.extend(batch_ids)

        embeddings = np.vstack(embeddings_list).astype("float32")
        ids = np.array(ids_list)

        return embeddings, ids


def load_clip(
    model_name: str = CLIP_MODEL_NAME, device: torch.device | None = None
) -> CLIPInference:
    return CLIPInference(model_name=model_name, device=device)


if __name__ == "__main__":
    # Smoke test
    clip = load_clip()

    # Test 1
    # (requires actual image files)
    # sim = clip.compute_similarity(f"{DATASET}/CelebA-HQ-img/0.jpg",
    #                               f"{DATASET}/CelebA-HQ-img/1.jpg"
    #                               )
    # print(f"Similarity: {sim[0, 0]}")

    # Test 2
    # results = clip.find_top_k_similar(
    #     query_images=f"{DATASET}/CelebA-HQ-img/0.jpg",
    #     reference_images=[f"{DATASET}/CelebA-HQ-img/{i}.jpg" for i in range(1, 10)],
    #     k=5
    # )
    # print(f"Top-5 indices: {results['indices']}")
    # print(f"Top-5 scores: {results['scores']}")

    # Test 3
    # from src.data_loading import CelebAFeatureDataset

    # dataset = CelebAFeatureDataset(
    #     root_dir=DATASET,
    #     partition_file=f"{DATASET}/list_eval_partition.txt",
    #     mapping_file=f"{DATASET}/CelebA-HQ-to-CelebA-mapping.txt",
    #     split='test',
    #     feature_name='nose',
    #     transform=None  # no transforms for CLIP!!!
    # )
    # embeddings, ids = clip.compute_dataset_embeddings(dataset, batch_size=32)
    # print(f"Embeddings shape: {embeddings.shape}, IDs shape: {ids.shape}")

    # Test 4
    from src.data_loading import CelebAFeatureDataset

    dataset = CelebAFeatureDataset(
        dataset=CelebADataset(split="train"),
        feature=CompositeFeature.mouth,
        transform=None,  # no transforms for CLIP!!!
    )

    find_results = clip.find_top_k_similar(
        query_images=f"{DATASET}/CelebA-HQ-img/0.jpg", reference_images=dataset, k=10000
    )
    logger.info(f"Top-5 indices: {find_results['indices']}")
    logger.info(f"Top-5 scores: {find_results['scores']}")

    top_k_images = [dataset[idx]["cropped_image"] for idx in find_results["indices"][0]]

    for idx, img in enumerate(top_k_images):
        assert isinstance(img, Image.Image), (
            f"Expected a PIL Image, but got {type(img)}"
        )
        img.save(f"results/{idx + 1}.jpg")
