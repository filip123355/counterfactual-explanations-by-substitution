import torch
import argparse
import yaml
import mlflow
import json


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


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        required=True,
        help="Path to the YAML file with configuration",
    )
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def log_config_params(config: dict) -> None:
    for key, value in config.items():
        if isinstance(value, (dict, list, tuple)):
            mlflow.log_param(key, json.dumps(value, ensure_ascii=False))
        else:
            mlflow.log_param(key, value)
