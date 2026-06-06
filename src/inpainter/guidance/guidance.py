from abc import abstractmethod
from typing import Protocol

import torch
import torch.nn.functional as F
from attr import dataclass
from loguru import logger
from PIL import Image
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from transformers.image_utils import SizeDict

from src.interface import CLIPInference
from src.constants import CLASSIFIER_SCALE, INTERVAL, USE_FP16
from src.inpainter.guidance.classifier import get_classifier
from src.inpainter.guidance.utils import ADAMGradientStabilization, AdaptiveNormalizer
from src.inpainter.transforms import I2SB_TO_NORMAL, PIL_TO_I2SB


@dataclass
class GuidanceResult:
    grad_pred_x0: torch.Tensor | None
    grad_xt: torch.Tensor | None


class GuidanceFn(Protocol):
    def __call__(
        self, *, xt: torch.Tensor, pred_x0: torch.Tensor, t: torch.Tensor | int
    ) -> GuidanceResult: ...


class Guidance:
    pil_to_i2sb: transforms.Compose
    i2sb_to_normal: transforms.Compose

    def __init__(
        self,
        *,
        pil_to_i2sb: transforms.Compose = PIL_TO_I2SB,
        i2sb_to_normal: transforms.Compose = I2SB_TO_NORMAL,
    ):
        self.pil_to_i2sb = pil_to_i2sb
        self.i2sb_to_normal = i2sb_to_normal

    def __call__(
        self, *, xt: torch.Tensor, pred_x0: torch.Tensor, t: torch.Tensor | int
    ) -> GuidanceResult:
        raise NotImplementedError("Guidance is not implemented yet.")

    @abstractmethod
    def set_target(
        self, *, target_img: Image.Image, label_value: int | None = None
    ) -> None:
        pass

    @abstractmethod
    def get_guidance_scale(self) -> float:
        pass


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
                self.i2sb_to_normal,
                transforms.Resize(
                    (clip_size, clip_size),
                    interpolation=InterpolationMode.BILINEAR,
                    antialias=True,
                ),
                transforms.Normalize(mean=clip_mean, std=clip_std),
            ]
        )

    def set_target(
        self, *, target_img: Image.Image, label_value: int | None = None
    ) -> None:
        target_emb = self.clip.compute_image_embeddings(target_img, normalize=True)
        self.target_embedding = target_emb.detach().requires_grad_(False)

    def get_guidance_scale(self) -> float:
        return 2.0

    def __call__(
        self, *, xt: torch.Tensor, pred_x0: torch.Tensor, t: torch.Tensor | int
    ) -> GuidanceResult:
        if self.target_embedding is None:
            raise ValueError("Target embedding not set.")

        pred_x0_in = pred_x0.detach().requires_grad_()

        with torch.autocast(
            dtype=torch.float16, enabled=USE_FP16, device_type=xt.device.type
        ):
            pred_emb = self.clip.compute_image_embedding_from_tensor(
                self.transform_i2sb_to_clip(pred_x0_in), normalize=True
            )

        pred_emb = pred_emb.float()

        target = self.target_embedding.expand(pred_emb.shape[0], -1)
        loss = -torch.cosine_similarity(pred_emb, target, dim=-1).mean()

        grad_x0 = torch.autograd.grad(outputs=loss, inputs=pred_x0_in)[0]

        return GuidanceResult(grad_pred_x0=-grad_x0, grad_xt=None)


class ClassifierGuidance(Guidance):
    target_image: torch.Tensor | None
    target_label: torch.Tensor | None

    def __init__(
        self,
        *,
        clf_scale: float = CLASSIFIER_SCALE,
        device: torch.device,
        nfe: int | None = None,
        tau: float | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        logger.info("Loading classifier model...")
        self.clf = get_classifier().to(device)
        self.clf_scale = clf_scale

        logger.info(
            f"Loaded classifier model with {sum(p.numel() for p in self.clf.parameters()):,} parameters."
        )

        if nfe is None:
            assert tau is not None, "Either nfe or tau must be provided."
            start_step = int(tau * (INTERVAL - 1))
            nfe = start_step - 1

        self.stabilization = ADAMGradientStabilization(
            beta_1=0.9, beta_2=0.999, eps=1e-8, reset_step=nfe
        )

        self.adaptive_normalizer = AdaptiveNormalizer(target_scale=1.0)

        self.target_image = None
        self.target_label = None

    def set_target(
        self, *, target_img: Image.Image, label_value: int | None = None
    ) -> None:
        assert label_value is not None, (
            "Label value must be provided for ClassifierGuidance."
        )

        if isinstance(target_img, Image.Image):
            target_tensor = self.pil_to_i2sb(target_img).unsqueeze(0)
        else:
            target_tensor = target_img

        self.target_image = target_tensor.to(torch.float32).detach()
        self.target_label = torch.tensor([label_value], dtype=torch.long)

    def get_guidance_scale(self) -> float:
        return 0.02

    def __call__(
        self, *, xt: torch.Tensor, pred_x0: torch.Tensor, t: torch.Tensor | int
    ) -> GuidanceResult:
        if self.target_label is None or self.target_image is None:
            raise ValueError("Target image or label not set.")

        x_in = pred_x0
        grad_x = xt

        y = self.target_label.to(x_in.device)
        if y.shape[0] != x_in.shape[0]:
            y = y.expand(x_in.shape[0])

        with torch.enable_grad():
            clf_logits = self.clf((x_in + 1.0) / 2.0)

            clf_log_probs = F.logsigmoid(clf_logits)
            clf_log_probs = clf_log_probs.gather(dim=1, index=y[:, None]).flatten()

            comps = self.clf_scale * clf_log_probs.sum()

            grad = torch.autograd.grad(outputs=comps, inputs=grad_x, create_graph=True)[
                0
            ]

        with torch.no_grad():
            grad = self.stabilization(grad)
            assert isinstance(t, int)
            grad = self.adaptive_normalizer(grad, t)

        return GuidanceResult(grad_pred_x0=None, grad_xt=grad)
