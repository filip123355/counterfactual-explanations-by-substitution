import torch


def assert_not_none[T](val: T | None) -> T:
    if val is None:
        raise ValueError("Expected value to be not None")
    return val


def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def unsqueeze_xdim(z: torch.Tensor, xdim) -> torch.Tensor:
    bc_dim = (...,) + (None,) * len(xdim)
    return z[bc_dim]


def space_indices(num_steps: int, count: int) -> list[int]:
    assert count <= num_steps

    if count <= 1:
        frac_stride = 1
    else:
        frac_stride = (num_steps - 1) / (count - 1)

    cur_idx = 0.0
    taken_steps = []
    for _ in range(count):
        taken_steps.append(round(cur_idx))
        cur_idx += frac_stride

    return taken_steps
