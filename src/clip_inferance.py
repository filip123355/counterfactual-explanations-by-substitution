import torch
import torch.nn.functional as F
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Union, List

from src.constants import CLIP_MODEL_NAME, BATCH_SIZE, DATASET

class CLIPInference:
    """CLIP-based image dot product similarity computation.
    """
    
    def __init__(self, 
                 model_name: str = CLIP_MODEL_NAME, 
                 device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading CLIP from {model_name} on {self.device}...")

        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model = CLIPModel.from_pretrained(model_name).to(self.device).eval()
    
    def _load_images(self, 
                     images: Union[str, Image.Image, List[Union[str, Image.Image]]],
    ) -> List[Image.Image]:

        if isinstance(images, str):
            # Single file path
            return [Image.open(images).convert('RGB')]
        elif isinstance(images, Image.Image):
            # Single PIL Image
            return [images.convert('RGB')]
        elif isinstance(images, list):
            # List of PIL Images
            result = []
            for img in images:
                if isinstance(img, str):
                    result.append(Image.open(img).convert('RGB'))
                elif isinstance(img, Image.Image):
                    result.append(img.convert('RGB'))
                else:
                    raise TypeError(f"Unsupported image type in list: {type(img)}")
            return result
        else:
            raise TypeError(f"Unsupported input type: {type(images)}")
    
    @torch.no_grad()
    def compute_image_embeddings(self, 
                                 images: Union[str, Image.Image, List[Union[str, Image.Image]]],
                                 normalize: bool = True,
    ) -> torch.Tensor:
        pil_images = self._load_images(images)
        inputs = self.processor(images=pil_images, return_tensors="pt").to(self.device)
        output = self.model.get_image_features(**inputs)
        if isinstance(output, torch.Tensor):
            embeddings = output
        elif hasattr(output, "image_embeds"):
            embeddings = output.image_embeds
        elif hasattr(output, "pooler_output"):
            embeddings = output.pooler_output
        else:
            raise TypeError(f"Unsupported CLIP output type: {type(output)}")
        
        if normalize:
            embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)
        
        return embeddings

    def _compute_embeddings_for_input(
        self,
        images: Union[str, Image.Image, List[Union[str, Image.Image]], Dataset],
    ) -> np.ndarray:
        if isinstance(images, Dataset):
            return self.compute_dataset_embeddings(images)[0]

        return self.compute_image_embeddings(images, normalize=True).cpu().numpy()
    
    def compute_similarity(self, 
                          query_images: Union[str, Image.Image, List[Union[str, Image.Image]], Dataset],
                          reference_images: Union[str, Image.Image, List[Union[str, Image.Image]], Dataset],
    ) -> np.ndarray:
        query_embs = self._compute_embeddings_for_input(query_images)
        ref_embs = self._compute_embeddings_for_input(reference_images)

        similarities = query_embs @ ref_embs.T  
    
        return similarities
    
    def find_top_k_similar(self, 
                          query_images: Union[str, Image.Image, List[Union[str, Image.Image]], Dataset],
                          reference_images: Union[str, Image.Image, List[Union[str, Image.Image]], Dataset],
                          k: int = 5,
                          return_scores: bool = True,
    ) -> dict:
        similarities = self.compute_similarity(query_images, reference_images)
        top_k_scores, top_k_indices = torch.topk(torch.tensor(similarities), k=k, dim=1)
        
        result = {'indices': top_k_indices.numpy()}
        if return_scores:
            result['scores'] = top_k_scores.numpy()
        
        return result
    
    @torch.no_grad()
    def compute_dataset_embeddings(self, 
                                   dataset: Dataset,
                                   batch_size: int = BATCH_SIZE,
    ) -> tuple[np.ndarray, np.ndarray]:
        def collate_fn(batch):
            images = [item[0] for item in batch]
            ids = [item[1] for item in batch]
            return images, ids
        
        loader = DataLoader(dataset, 
                            batch_size=batch_size, 
                            shuffle=False, 
                            collate_fn=collate_fn,
                            )
        
        embeddings_list = []
        ids_list = []
        
        for batch_images, batch_ids in loader:
            embs = self.compute_image_embeddings(batch_images, normalize=True)
            embeddings_list.append(embs.cpu().numpy())
            ids_list.extend(batch_ids)
        
        embeddings = np.vstack(embeddings_list).astype('float32')
        ids = np.array(ids_list)
        
        return embeddings, ids


def load_clip(model_name: str = CLIP_MODEL_NAME, 
              device: str = None,
) -> CLIPInference:
    return CLIPInference(
        model_name=model_name, 
        device=device
    )


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
        root_dir=DATASET,
        partition_file=f"{DATASET}/list_eval_partition.txt",
        mapping_file=f"{DATASET}/CelebA-HQ-to-CelebA-mapping.txt",
        split='test',
        feature_name='nose',
        transform=None # no transforms for CLIP!!!
    )
    
    find_results = clip.find_top_k_similar(
        query_images=f"{DATASET}/CelebA-HQ-img/0.jpg",
        reference_images=dataset,
        k=5
    )
    print(f"Top-5 indices: {find_results['indices']}")
    print(f"Top-5 scores: {find_results['scores']}")
