import cv2
import numpy as np
import torch
from loguru import logger
from PIL import Image
from torch_ema import ExponentialMovingAverage
from torchvision import transforms

import src.utils as util
from src.constants import (
    BETA_MAX,
    CLIP_DENOISE,
    EMA_DECAY,
    I2SB_MODEL_PATH,
    INTERVAL,
    MODEL_KWARGS,
    OT_ODE,
    T0,
    USE_FP16,
    T,
)
from src.data_loading import (
    DEFAULT_REVERSE_TRANSFORM,
    DEFAULT_TRANSFORMS,
    CelebADataset,
    CompositeFeature,
)
from src.inpainter.diffusion import Diffusion
from src.inpainter.network import Image256Net
from src.keypoints import MediapipeFaceKeypointDetector
from src.substitution import Substitution
from src.visualize import show_inpanting


def make_beta_schedule(n_timestep: int, linear_start: float, linear_end: float):
    betas = (
        torch.linspace(
            linear_start**0.5, linear_end**0.5, n_timestep, dtype=torch.float64
        )
        ** 2
    )
    return betas.numpy()


class I2SB:
    diffusion: Diffusion
    net: Image256Net
    ema: ExponentialMovingAverage
    transforms: transforms.Compose
    reverse_transforms: transforms.Compose
    device: torch.device

    def __init__(
        self,
        *,
        ckpt_path: str = I2SB_MODEL_PATH,
        device: torch.device,
        transforms=DEFAULT_TRANSFORMS,
        reverse_transforms=DEFAULT_REVERSE_TRANSFORM,
    ):
        self.device = device
        self.transforms = transforms
        self.reverse_transforms = reverse_transforms

        betas = make_beta_schedule(
            n_timestep=INTERVAL, linear_start=T0, linear_end=BETA_MAX / INTERVAL
        )
        betas = np.concatenate(
            [betas[: INTERVAL // 2], np.flip(betas[: INTERVAL // 2])]
        )
        self.diffusion = Diffusion(betas).to(device)

        noise_levels = torch.linspace(T0, T, INTERVAL, device=device) * INTERVAL
        noise_levels = noise_levels.to(device)

        self.net = Image256Net(noise_levels=noise_levels, model_kwargs=MODEL_KWARGS)
        self.ema = ExponentialMovingAverage(self.net.parameters(), decay=EMA_DECAY)

        checkpoint = torch.load(ckpt_path, map_location="cpu")
        self.net.load_state_dict(checkpoint["net"])
        self.ema.load_state_dict(checkpoint["ema"])

        self.ema.copy_to()
        self.net.float()
        self.net.to(device)
        del self.ema

        torch.cuda.empty_cache()

        # if USE_FP16:
        #     self.net.diffusion_model.convert_to_fp16()

        self.net.eval()

    def _compute_pred_x0(
        self,
        step: int | torch.Tensor,
        xt: torch.Tensor,
        net_out: torch.Tensor,
        clip_denoise=False,
    ):
        """Given network output, recover x0. This should be the inverse of Eq 12"""
        std_fwd = self.diffusion.get_std_fwd(step, xdim=xt.shape[1:])
        pred_x0 = xt - std_fwd * net_out
        if clip_denoise:
            pred_x0.clamp_(-1.0, 1.0)
        return pred_x0

    @torch.no_grad()
    def _ddpm_sampling(
        self,
        x0: torch.Tensor,
        x1: torch.Tensor,
        mask: torch.Tensor | None = None,
        nfe: int | None = None,
        tau: float = 1.0,
    ) -> torch.Tensor:
        nfe = nfe or INTERVAL - 1
        assert 0 < nfe < INTERVAL == len(self.diffusion.betas)
        assert 0.0 < tau <= 1.0

        x0 = x0.to(self.device)
        x1 = x1.to(self.device)

        start_step = int(tau * (INTERVAL - 1))

        all_steps = util.space_indices(INTERVAL, nfe + 1)
        steps = [s for s in all_steps if s <= start_step]

        if steps[-1] != start_step:
            steps.append(start_step)

        if tau < 1.0:
            step_tensor = torch.full(
                (x0.shape[0],), start_step, device=self.device, dtype=torch.long
            )
            xt = self.diffusion.q_sample(step_tensor, x0, x1, ot_ode=OT_ODE)
        else:
            xt = x1.clone()

        logger.info(f"[DDPM Sampling] interval={INTERVAL}, {nfe=}, {steps=}, {tau=}")

        def pred_x0_fn(xt: torch.Tensor, step: int) -> torch.Tensor:
            step: torch.Tensor = torch.full(
                (xt.shape[0],), step, device=self.device, dtype=torch.long
            )

            xt = xt.float()

            with torch.autocast(
                device_type=self.device.type, dtype=torch.float16, enabled=USE_FP16
            ):
                out = self.net(xt, step)

            out = out.float()

            return self._compute_pred_x0(step, xt, out, clip_denoise=CLIP_DENOISE)

        xs = self.diffusion.ddpm_sampling(
            steps=steps,
            pred_x0_fn=pred_x0_fn,
            xt=xt,
            x1=x1,
            mask=mask,
            ot_ode=OT_ODE,
        )

        assert xs.shape == x1.shape

        return xs

    def inpaint(
        self,
        image: Image.Image,
        mask: np.ndarray,
        tau: float = 1.0,
        nfe: int | None = None,
    ) -> Image.Image:
        x0 = self.transforms(image).unsqueeze(0).to(self.device)

        mask = cv2.resize(
            mask, (x0.shape[-1], x0.shape[-2]), interpolation=cv2.INTER_NEAREST
        )
        mask: torch.Tensor = torch.from_numpy(mask).to(torch.float32).to(self.device)
        mask = mask.to(self.device)
        mask = mask.unsqueeze(0).unsqueeze(0)
        mask = mask / 255.0

        x1 = (1.0 - mask) * x0 + mask * torch.randn_like(x0)
        xs = self._ddpm_sampling(x0=x0, x1=x1, mask=mask, nfe=nfe, tau=tau)

        return self.reverse_transforms(xs.squeeze(0).cpu())


if __name__ == "__main__":
    src_idx = 0
    dest_idx = 13
    feature = CompositeFeature.eyes

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    inpainter = I2SB(
        device=device,
    )

    dataset = CelebADataset(split="test")

    face_keypoint_detector = MediapipeFaceKeypointDetector()
    substitution = Substitution(dataset, face_keypoint_detector)

    subst_image = substitution.substitute(src_idx, dest_idx, feature)

    mask = dataset.get(dest_idx, feature=feature, inflate_mask=10)["mask"]
    assert mask is not None

    inp_image = inpainter.inpaint(subst_image, mask, tau=0.4)

    show_inpanting(
        dataset.get(dest_idx, feature=feature)["full_image"], subst_image, inp_image
    )
