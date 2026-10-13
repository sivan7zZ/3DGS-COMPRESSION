"""Standard 3DGS checkpoint validation and atomic output helpers."""
import json
import math
import os
from pathlib import Path

import torch


def validate_model_state(state):
    if not isinstance(state, (tuple, list)) or len(state) != 12:
        raise ValueError("Expected a standard 12-item 3DGS model state")
    if not all(torch.is_tensor(t) for t in state[1:10]):
        raise ValueError("Checkpoint model fields must be tensors")
    n = state[1].shape[0]
    rest = state[3]
    if n == 0 or rest.ndim != 3 or rest.shape[2] != 3:
        raise ValueError("Empty model or invalid SH tensor")
    degree = int(math.sqrt(rest.shape[1] + 1)) - 1
    if degree < 0 or (degree + 1) ** 2 != rest.shape[1] + 1:
        raise ValueError("SH coefficients do not form a complete degree")
    if not 0 <= int(state[0]) <= degree:
        raise ValueError("Active SH degree exceeds stored coefficients")
    shapes = ((n, 3), (n, 1, 3), (n, rest.shape[1], 3),
              (n, 3), (n, 4), (n, 1), (n,), (n, 1), (n, 1))
    for value, shape in zip(state[1:10], shapes):
        if tuple(value.shape) != shape or not torch.isfinite(value).all():
            raise ValueError("Checkpoint contains invalid tensor shape or non-finite values")
    if not isinstance(state[10], dict) or "param_groups" not in state[10]:
        raise ValueError("Checkpoint is missing optimizer group metadata")
    if not math.isfinite(float(state[11])) or float(state[11]) <= 0:
        raise ValueError("Invalid scene spatial learning-rate scale")
    return degree


def load_checkpoint(path):
    # torch.load uses pickle in supported PyTorch 1.12: input must be trusted.
    payload = torch.load(str(path), map_location="cpu")
    if not isinstance(payload, (tuple, list)) or len(payload) != 2:
        raise ValueError("Expected (model_state, iteration) checkpoint")
    state, iteration = payload
    validate_model_state(state)
    if not isinstance(iteration, int) or iteration < 0:
        raise ValueError("Checkpoint iteration must be a nonnegative integer")
    return state, iteration


def atomic_torch_save(value, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    torch.save(value, str(temp))
    os.replace(str(temp), str(path))


def atomic_json_save(value, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    with temp.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False)
        stream.write("\n")
    os.replace(str(temp), str(path))
