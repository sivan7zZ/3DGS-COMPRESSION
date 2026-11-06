"""PUP scores and fixed-budget ranking (independent of the training schedule).

Fisher CUDA: j-alex-hanson/rasterization_and_pup_fisher @ eaef2faba21f076867c1db84c3f998e5f673762f.
Score: gaussian-splatting-pup/prune_finetune.py, sum(log(svdvals(F))).
"""
import math
from pathlib import Path
import sys

import torch
from tqdm import tqdm


def fixed_budget_mask(scores, remove_count):
    """Remove precisely the budget, breaking equal scores by current point index."""
    scores = scores.detach().reshape(-1)
    if torch.isnan(scores).any():
        raise FloatingPointError("NaN pruning scores")
    if not 0 <= remove_count < scores.numel():
        raise ValueError("Pruning budget must leave at least one Gaussian")
    _, order = torch.sort(scores, stable=True)
    mask = torch.zeros(scores.numel(), dtype=torch.bool, device=scores.device)
    mask[order[:remove_count]] = True
    return mask


def unpack_fisher(packed):
    """PUP's 21 upper-triangular coefficients -> symmetric 6x6 Fisher blocks."""
    if packed.ndim != 2 or packed.shape[1] != 21:
        raise ValueError("Expected N x 21 packed Fisher coefficients")
    rows, cols = torch.triu_indices(6, 6, device=packed.device)
    blocks = packed.new_zeros((packed.shape[0], 6, 6))
    blocks[:, rows, cols] = packed
    blocks[:, cols, rows] = packed
    return blocks


def fisher_logdet(packed, chunk_size=65536):
    """Match upstream SVD log-determinant; zero singular values remain -inf.

    No diagonal damping, trace approximation, frequency weighting or sign flip.
    Chunking bounds SVD workspace without changing per-Gaussian blocks.
    """
    if not torch.isfinite(packed).all():
        raise FloatingPointError("Non-finite Fisher coefficients")
    values = []
    for part in packed.split(chunk_size):
        singular_values = torch.linalg.svdvals(unpack_fisher(part))
        values.append(singular_values.log().sum(dim=1))
    return torch.cat(values)


def _backend():
    root = Path(__file__).resolve().parents[1] / "submodules" / "pup-fisher"
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    try:
        import rasterization_and_pup_fisher as backend
    except ImportError as error:
        raise ImportError("Build local PUP Fisher: cd '{}' && python setup.py build_ext --inplace".format(root)) from error
    if root not in Path(backend.__file__).resolve().parents:
        raise ImportError("Expected the vendored local PUP Fisher backend")
    return backend


def pup_pruning_scores(cameras, model, pipe, background, resolution=1):
    """Exact upstream CUDA Fisher accumulation over all training cameras.

    Only request the dummy Fisher gradient: model .grad buffers are untouched.
    resolution=1 uses all pixels; larger values are upstream's image downsample.
    """
    if resolution < 1:
        raise ValueError("PUP Fisher resolution must be a positive integer")
    if pipe.antialiasing or pipe.compute_cov3D_python or pipe.convert_SHs_python:
        raise ValueError("PUP CUDA requires standard scale/rotation and SH inputs, without antialiasing")
    backend = _backend()
    packed = model.get_xyz.new_zeros((model.get_xyz.shape[0], 21))
    # These parameter values are fixed during scoring; only Fisher receives gradients.
    xyz = model.get_xyz.detach()
    shs = model.get_features.detach()
    opacity = model.get_opacity.detach()
    scales = model.get_scaling.detach()
    rotations = model.get_rotation.detach()
    with torch.enable_grad():
        for camera in tqdm(cameras, desc="PUP Fisher (all training views)"):
            settings = backend.GaussianRasterizationSettings(
                image_height=math.ceil(camera.image_height / resolution),
                image_width=math.ceil(camera.image_width / resolution),
                tanfovx=math.tan(camera.FoVx / 2), tanfovy=math.tan(camera.FoVy / 2),
                bg=background, scale_modifier=1., viewmatrix=camera.world_view_transform,
                projmatrix=camera.full_proj_transform, sh_degree=model.active_sh_degree,
                campos=camera.camera_center, prefiltered=False, debug=pipe.debug)
            dummy = torch.zeros_like(packed, requires_grad=True)
            image, _ = backend.GaussianRasterizer(settings)(
                means3D=xyz, means2D=torch.zeros_like(xyz), shs=shs,
                opacities=opacity, scales=scales, rotations=rotations, fishers=dummy)
            contribution = torch.autograd.grad(image.sum(), dummy)[0]
            packed.add_(contribution)
            del image, dummy, contribution
    return fisher_logdet(packed)
