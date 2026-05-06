# ---------------------------------------------------------------
# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# for I2SB. To view a copy of this license, see the LICENSE file.
# ---------------------------------------------------------------

from typing import Callable, Literal, TypedDict, Unpack

import numpy as np
import torch
from tqdm import tqdm

from src.constants import GUIDANCE_SCALE
from src.inpainter.guidance import GuidanceFn
from src.utils import unsqueeze_xdim


class CondFnParams(TypedDict):
    x_t: torch.Tensor
    pred_x0: torch.Tensor
    t: torch.Tensor | int


def compute_gaussian_product_coef(sigma1, sigma2):
    """Given p1 = N(x_t|x_0, sigma_1**2) and p2 = N(x_t|x_1, sigma_2**2)
    return p1 * p2 = N(x_t| coef1 * x0 + coef2 * x1, var)"""

    denom = sigma1**2 + sigma2**2
    coef1 = sigma2**2 / denom
    coef2 = sigma1**2 / denom
    var = (sigma1**2 * sigma2**2) / denom

    return coef1, coef2, var


class Diffusion(torch.nn.Module):
    betas: torch.Tensor
    std_fwd: torch.Tensor
    std_bwd: torch.Tensor
    std_sb: torch.Tensor
    mu_x0: torch.Tensor
    mu_x1: torch.Tensor

    def __init__(self, betas):
        super().__init__()

        # compute analytic std: eq 11
        std_fwd = torch.from_numpy(np.sqrt(np.cumsum(betas)))
        std_bwd = torch.from_numpy(np.sqrt(np.flip(np.cumsum(np.flip(betas)))))

        mu_x0, mu_x1, var = compute_gaussian_product_coef(std_fwd, std_bwd)
        std_sb = var.sqrt()

        self.betas = torch.from_numpy(betas).float()
        self.register_buffer("std_fwd", std_fwd)
        self.register_buffer("std_bwd", std_bwd)
        self.register_buffer("std_sb", std_sb)
        self.register_buffer("mu_x0", mu_x0)
        self.register_buffer("mu_x1", mu_x1)

    def get_std_fwd(self, step: int | torch.Tensor, xdim=None):
        std_fwd = self.std_fwd[step]
        return std_fwd if xdim is None else unsqueeze_xdim(std_fwd, xdim)

    def q_sample(
        self, step: int | torch.Tensor, x0: torch.Tensor, x1: torch.Tensor, ot_ode=False
    ):
        """Sample q(x_t | x_0, x_1), i.e. eq 11"""

        assert x0.shape == x1.shape
        batch, *xdim = x0.shape

        mu_x0 = unsqueeze_xdim(self.mu_x0[step], xdim)
        mu_x1 = unsqueeze_xdim(self.mu_x1[step], xdim)
        std_sb = unsqueeze_xdim(self.std_sb[step], xdim)

        xt = mu_x0 * x0 + mu_x1 * x1
        if not ot_ode:
            xt = xt + std_sb * torch.randn_like(xt)
        return xt.detach()

    def p_posterior(
        self,
        nprev: int,
        n: int,
        x_n: torch.Tensor,
        x0: torch.Tensor,
        ot_ode=False,
        verbose=False,
    ):
        """Sample p(x_{nprev} | x_n, x_0), i.e. eq 4"""

        assert nprev < n
        std_n = self.std_fwd[n]
        std_nprev = self.std_fwd[nprev]
        std_delta = (std_n**2 - std_nprev**2).sqrt()

        mu_x0, mu_xn, var = compute_gaussian_product_coef(std_nprev, std_delta)

        xt_prev = mu_x0 * x0 + mu_xn * x_n
        if not ot_ode and nprev > 0:
            xt_prev = xt_prev + var.sqrt() * torch.randn_like(xt_prev)

        if verbose:
            return xt_prev, mu_x0
        else:
            return xt_prev

    def ddpm_sampling(
        self,
        *,
        steps: list[int],
        pred_x0_fn: Callable[[Unpack[CondFnParams]], torch.Tensor],
        xt: torch.Tensor,
        x1: torch.Tensor,
        mask: torch.Tensor,
        cond_fn: GuidanceFn | None = None,
        ot_ode=False,
    ) -> torch.Tensor:
        xs = xt.detach()

        steps = steps[::-1]

        pair_steps = zip(steps[1:], steps[:-1])
        pair_steps = tqdm(pair_steps, desc="DDPM sampling", total=len(steps) - 1)

        for prev_step, step in pair_steps:
            assert prev_step < step, f"{prev_step=}, {step=}"

            if cond_fn is not None:
                xs = xs.detach().requires_grad_()

            pred_x0 = pred_x0_fn(xs, step)

            if cond_fn is not None:
                cond_grad = cond_fn(xt=xs, t=step, pred_x0=pred_x0)
                xs = xs + (GUIDANCE_SCALE * cond_grad)

                xs = xs.detach()
                pred_x0 = pred_x0.detach()
                del cond_grad

            xs = self.p_posterior(prev_step, step, xs, pred_x0, ot_ode=ot_ode)

            xt_true = x1.clone()
            if not ot_ode:
                _prev_step = torch.full((xs.shape[0],), prev_step, dtype=torch.long)
                std_sb = unsqueeze_xdim(self.std_sb[_prev_step], xdim=xs.shape[1:])
                xt_true = xt_true + std_sb * torch.randn_like(xt_true)

            xs = (1.0 - mask) * xt_true + mask * xs

        return xs

    def _guided_sampling_core(
        self,
        mode: Literal["shallow", "deep"],
        steps: list[int],
        pred_x0_fn: Callable[[torch.Tensor, int], torch.Tensor],
        xt: torch.Tensor,
        x1: torch.Tensor,
        x1_forw: torch.Tensor,
        mask: torch.Tensor,
        step_size: float = 1.0,
        ot_ode: bool = False,
        desc: str = "Guided Sampling",
    ) -> torch.Tensor:
        xs = xt.detach()

        steps = steps[::-1]
        pair_steps = zip(steps[1:], steps[:-1])
        pair_steps = tqdm(pair_steps, desc=desc, total=len(steps) - 1)

        for prev_step, step in pair_steps:
            assert prev_step < step, f"{prev_step=}, {step=}"

            xs = xs.detach().requires_grad_()
            pred_x0 = pred_x0_fn(xs, step)

            corrupt_x0_forw = (1.0 - mask) * pred_x0

            residual = corrupt_x0_forw - x1_forw
            residual_norm = torch.linalg.norm(residual) ** 2

            std_n = self.std_fwd[step]
            std_nprev = self.std_fwd[prev_step]
            std_delta = (std_n**2 - std_nprev**2).sqrt()
            mu_x0, _, _ = compute_gaussian_product_coef(std_nprev, std_delta)

            if mode == "shallow":
                norm_grad = torch.autograd.grad(outputs=residual_norm, inputs=pred_x0)[
                    0
                ]
                xs_next = self.p_posterior(prev_step, step, xs, pred_x0, ot_ode=ot_ode)
                xs = xs_next - (mu_x0 * step_size * norm_grad)

            elif mode == "deep":
                norm_grad = torch.autograd.grad(outputs=residual_norm, inputs=xs)[0]
                xs = xs - (mu_x0 * step_size * norm_grad)
                xs = self.p_posterior(prev_step, step, xs, pred_x0, ot_ode=ot_ode)
                del norm_grad

            xs.detach_()
            pred_x0.detach_()

            xt_true = x1.clone()
            if not ot_ode:
                _prev_step = torch.full(
                    (xs.shape[0],), prev_step, dtype=torch.long, device=xs.device
                )
                std_sb = unsqueeze_xdim(self.std_sb[_prev_step], xdim=xs.shape[1:])
                xt_true = xt_true + std_sb * torch.randn_like(xt_true)

            xs = (1.0 - mask) * xt_true + mask * xs

        return xs

    def dds_sampling(
        self,
        *,
        steps: list[int],
        pred_x0_fn: Callable[[torch.Tensor, int], torch.Tensor],
        xt: torch.Tensor,
        x1: torch.Tensor,
        x1_forw: torch.Tensor,
        mask: torch.Tensor,
        step_size: float = 1.0,
        ot_ode=False,
    ) -> torch.Tensor:
        return self._guided_sampling_core(
            mode="shallow",
            steps=steps,
            pred_x0_fn=pred_x0_fn,
            xt=xt,
            x1=x1,
            x1_forw=x1_forw,
            mask=mask,
            step_size=step_size,
            ot_ode=ot_ode,
            desc="CDDB (Shallow) sampling",
        )

    def ddpm_dps_sampling(
        self,
        *,
        steps: list[int],
        pred_x0_fn: Callable[[torch.Tensor, int], torch.Tensor],
        xt: torch.Tensor,
        x1: torch.Tensor,
        x1_forw: torch.Tensor,
        mask: torch.Tensor,
        step_size: float = 1.0,
        ot_ode=False,
    ) -> torch.Tensor:
        return self._guided_sampling_core(
            mode="deep",
            steps=steps,
            pred_x0_fn=pred_x0_fn,
            xt=xt,
            x1=x1,
            x1_forw=x1_forw,
            mask=mask,
            step_size=step_size,
            ot_ode=ot_ode,
            desc="CDDB-Deep (DPS) sampling",
        )
