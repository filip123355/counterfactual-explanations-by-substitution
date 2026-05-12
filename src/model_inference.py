import torch
from transformers import ViTForImageClassification, ViTImageProcessor
from typing import List, Union
from PIL import Image
import numpy as np
from loguru import logger

from src.constants import PROJECT_ROOT


class ViTClassifier:

    def __init__(self, model_path: str = None, device: str = None):
        if model_path is None:
            model_path = str(PROJECT_ROOT / "models" / "vit-base-patch16-224")
            
        self.device = torch.device(
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        logger.info(f"Model loaded on device: {self.device}")

        self.processor = ViTImageProcessor.from_pretrained(model_path)
        self.model = ViTForImageClassification.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def predict_batch(
        self, 
        images: List[Union[Image.Image, np.ndarray]], 
        return_probs: bool = True,
    ) -> torch.Tensor:
        
        inputs = self.processor(images=images, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        outputs = self.model(**inputs)
        logits = outputs.logits
        
        if return_probs:
            return torch.softmax(logits, dim=-1)
        return logits


if __name__ == "__main__":
    # Test
    classifier = ViTClassifier()
    dummy_images = [Image.new('RGB', (224, 224), color = (i*20, 20, 20)) for i in range(5)]
    predictions = classifier.predict_batch(dummy_images)
    print(predictions.shape)
    print(torch.topk(predictions[0], k=5))
