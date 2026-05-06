from typing import Protocol

import torch
from loguru import logger
from PIL import Image
from torchvision import transforms
from transformers.image_utils import SizeDict

from src.clip_inferance import CLIPInference
from src.constants import USE_FP16
from src.data_loading import I2SB_TO_NORMAL, PIL_TO_I2SB


class GuidanceFn(Protocol):
    def __call__(
        self, *, xt: torch.Tensor, pred_x0: torch.Tensor, t: torch.Tensor | int
    ) -> torch.Tensor: ...


class Guidance:
    transform: transforms.Compose
    reverse_transform_to_tensor: transforms.Compose

    def __init__(
        self,
        *,
        transform: transforms.Compose = PIL_TO_I2SB,
        reverse_transform_to_tensor: transforms.Compose = I2SB_TO_NORMAL,
    ):
        self.transform = transform
        self.reverse_transform_to_tensor = reverse_transform_to_tensor

    def __call__(
        self, *, xt: torch.Tensor, pred_x0: torch.Tensor, t: torch.Tensor | int
    ) -> torch.Tensor:
        raise NotImplementedError("Guidance is not implemented yet.")


class CLIPGuidance(Guidance):
    target_embedding: torch.Tensor | None

    def __init__(self, clip: CLIPInference, **kwargs):
        super().__init__(**kwargs)

        self.clip = clip
        self.target_embedding = None

        clip_image_processor = self.clip.processor.image_processor
        clip_mean = clip_image_processor.image_mean
        clip_std = clip_image_processor.image_std

        size_info = clip_image_processor.size
        if isinstance(size_info, dict) or isinstance(size_info, SizeDict):
            clip_size = size_info.get("shortest_edge", size_info.get("height", 224))
        else:
            clip_size = size_info

        logger.info(
            f"CLIP image processor clip_size: {clip_size}, clip_mean: {clip_mean}, clip_std: {clip_std}"
        )

        self.transform_i2sb_to_clip = transforms.Compose(
            [
                self.reverse_transform_to_tensor,
                transforms.Resize((clip_size, clip_size)),
                transforms.Normalize(mean=clip_mean, std=clip_std),
            ]
        )

    def set_target(self, target: Image.Image | str) -> None:
        target_emb = self.clip.compute_image_embeddings(target, normalize=True)
        self.target_embedding = target_emb.detach().requires_grad_(False)

    def __call__(
        self, *, xt: torch.Tensor, pred_x0: torch.Tensor, t: torch.Tensor | int
    ) -> torch.Tensor:
        if self.target_embedding is None:
            raise ValueError("Target embedding not set.")

        with torch.autocast(
            dtype=torch.float16, enabled=USE_FP16, device_type=xt.device.type
        ):
            pred_emb = self.clip.compute_image_embedding_from_tensor(
                self.transform_i2sb_to_clip(pred_x0), normalize=True
            )
        pred_emb = pred_emb.float()

        target = self.target_embedding.expand(pred_emb.shape[0], -1)

        loss = -torch.cosine_similarity(pred_emb, target, dim=-1).mean()
        grad = torch.autograd.grad(outputs=loss, inputs=xt)[0]

        return -grad
