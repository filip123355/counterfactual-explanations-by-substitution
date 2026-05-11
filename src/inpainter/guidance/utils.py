import torch
from loguru import logger


class ADAMGradientStabilization(torch.nn.Module):
    def __init__(
        self,
        reset_step: int,
        beta_1: float = 0.9,
        beta_2: float = 0.999,
        eps: float = 1e-8,
    ):
        """
        Applies gradient stabilization from ADAM.
        """
        super().__init__()
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.eps = eps
        self.m = None
        self.v = None
        self.step = 1
        self.reset_step = reset_step

    def __call__(self, classifier_gradient: torch.Tensor) -> torch.Tensor:

        if self.m is None or self.step == self.reset_step + 1:
            self.m = torch.zeros_like(classifier_gradient)
            self.v = torch.zeros_like(classifier_gradient)
            self.step = 1
            logger.info("Resetting stabilization")

        m = self.beta_1 * self.m + (1 - self.beta_1) * classifier_gradient
        self.m = m
        v = self.beta_2 * self.v + (1 - self.beta_2) * torch.square(classifier_gradient)  # ty: ignore
        self.v = v
        m_hat = m / (1 - (self.beta_1**self.step))
        v_hat = v / (1 - (self.beta_2**self.step))
        self.step += 1

        return m_hat / (torch.sqrt(v_hat) + self.eps)


class AdaptiveNormalizer(torch.nn.Module):
    def __init__(self, target_scale: float = 1.0):
        super().__init__()
        self.prev_t: int = 0
        self.target_scale = target_scale

    def __call__(self, grad: torch.Tensor, t: int) -> torch.Tensor:
        if self.prev_t < t:  # Update after new inpainting starts
            bs, _, _, _ = grad.shape
            max_grad_norm_tensor = grad.view(bs, -1).norm(p=2, dim=1).detach().float()
            self.register_buffer("grad_norm", max_grad_norm_tensor.view(bs, 1, 1, 1))
            logger.info("Resetting adaptive normalization")

        assert "grad_norm" in self._buffers
        self.prev_t = t
        return grad / self.grad_norm * self.target_scale  # ty: ignore
