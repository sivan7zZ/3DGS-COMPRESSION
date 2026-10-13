"""Checkpoint, reproducibility and view helpers for SH distillation.

View perturbation is adapted from VITA-Group/LightGaussian utils/pose_utils.py.
The Gaussian-Splatting license in LICENSE.md applies to derived project code.
"""

import copy
import hashlib
import json
import os
import random
import tempfile

import numpy as np
import torch


def load_torch_file(path):
    """Load trusted local training checkpoints across old/new PyTorch versions."""
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:  # PyTorch 1.12 has no weights_only argument.
        return torch.load(path, map_location="cpu")


def read_standard_checkpoint(path):
    value = load_torch_file(path)
    if not isinstance(value, (tuple, list)) or len(value) != 2:
        raise ValueError("Expected a (model_tuple, iteration) checkpoint: " + path)
    model, iteration = value
    if not isinstance(model, (tuple, list)) or len(model) != 12:
        raise ValueError("SH distillation requires a standard 12-field checkpoint: " + path)
    if not isinstance(iteration, int) or isinstance(iteration, bool) or iteration < 0:
        raise ValueError("Checkpoint iteration must be a nonnegative integer")
    return model, iteration


def atomic_write(path, writer):
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=".distill-", dir=directory)
    os.close(descriptor)
    try:
        writer(temporary)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def atomic_json_save(value, path):
    def write(temporary):
        with open(temporary, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
            stream.write("\n")
    atomic_write(path, write)


def atomic_torch_save(value, path):
    atomic_write(path, lambda temporary: torch.save(value, temporary))


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def checkpoint_provenance(path, stored_iteration):
    return {"path": os.path.realpath(path), "stored_iteration": stored_iteration,
            "sha256": sha256_file(path)}


def capture_rng_state():
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng_state(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state["cuda"]:
        if not torch.cuda.is_available():
            raise RuntimeError("This resume state requires CUDA")
        if len(state["cuda"]) != torch.cuda.device_count():
            raise RuntimeError("CUDA device count changed since the checkpoint")
        torch.cuda.set_rng_state_all(state["cuda"])


def model_fingerprint(model):
    """Exact identity check without keeping another teacher copy in GPU memory."""
    digest = hashlib.sha256()
    for name in ("_xyz", "_features_dc", "_features_rest", "_scaling", "_rotation", "_opacity"):
        tensor = getattr(model, name).detach().contiguous().cpu()
        digest.update(name.encode("ascii"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(tensor.numpy().tobytes())
    digest.update(str((model.active_sh_degree, model.max_sh_degree)).encode("ascii"))
    return digest.hexdigest()


def distillation_schedule(start, steps, iterations=None):
    if start < 0 or steps <= 0:
        raise ValueError("distill_start_iteration must be >= 0 and distill_steps > 0")
    end = start + steps
    if iterations is not None and iterations != end:
        raise ValueError("--iterations must equal --distill_start_iteration + --distill_steps")
    return end


def resume_configuration(saved, requested, explicit, allowed_overrides=()):
    """Restore saved settings; reject explicitly conflicting training settings."""
    merged = dict(saved)
    allowed = set(allowed_overrides)
    for key in explicit:
        if key in allowed:
            merged[key] = requested[key]
        elif key in saved and requested.get(key) != saved[key]:
            raise ValueError("Resume setting differs from its checkpoint: --" + key)
    return merged


def verify_resume_pair(checkpoint_path, state):
    if state.get("format") != "pwsr-sh-distill-v1":
        raise ValueError("Unsupported SH distillation resume sidecar")
    if sha256_file(checkpoint_path) != state.get("checkpoint_sha256"):
        raise ValueError("Checkpoint/sidecar mismatch (possibly an interrupted save)")


def augmented_camera(camera, translation_std=0.05):
    """LightGaussian virtual view: translate camera center with Gaussian noise.

    Rotation perturbation is zero in its standard distillation invocation. A
    shallow copy keeps immutable image data shared while replacing transforms.
    """
    from utils.graphics_utils import getWorld2View2
    result = copy.copy(camera)
    translation = np.random.normal(0.0, translation_std, 3)
    matrix = getWorld2View2(camera.R, camera.T, translation)
    result.world_view_transform = torch.as_tensor(
        matrix, dtype=camera.world_view_transform.dtype,
        device=camera.world_view_transform.device).transpose(0, 1)
    result.full_proj_transform = result.world_view_transform.unsqueeze(0).bmm(
        result.projection_matrix.unsqueeze(0)).squeeze(0)
    result.camera_center = result.world_view_transform.inverse()[3, :3]
    return result
