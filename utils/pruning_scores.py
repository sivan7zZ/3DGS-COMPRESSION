"""PUP scores and fixed-budget ranking (independent of the training schedule).

Fisher CUDA: j-alex-hanson/rasterization_and_pup_fisher @ eaef2faba21f076867c1db84c3f998e5f673762f.
Score: gaussian-splatting-pup/prune_finetune.py, sum(log(svdvals(F))).
"""
import math
from pathlib import Path
import sys

import torch
from tqdm import tqdm


CONTINUOUS_BOUNDARY_FIELDS = ('sam_boundary_strength', 'depth_boundary_strength', 'rgb_boundary_strength')


def continuous_boundary_scores(scores, sam_strength, depth_strength, rgb_strength, config):
    """PUP logdet plus a continuous, bounded multimodal importance weight.

    No boundary threshold or legacy boolean mask participates in this score.
    ``c = w_s*SAM + w_d*depth + w_r*RGB`` and ``W = 1 + gain*c``;
    the score is ``raw + log1p(gain*c)``. In determinant space this multiplies
    importance by W, so negative logdet values cannot reverse the protection.
    A singular (-inf) Fisher score stays singular under finite multiplication.
    """
    scores = torch.as_tensor(scores).detach()
    if scores.ndim != 1 or not scores.is_floating_point():
        raise ValueError('Expected one floating-point PUP score per Gaussian')
    if torch.isnan(scores).any() or torch.isposinf(scores).any():
        raise FloatingPointError('PUP scores contain NaN or positive infinity')
    if config.get('fusion') != 'log1p':
        raise ValueError('Continuous boundary fusion must explicitly be log1p')
    weights = [float(config[key]) for key in ('sam_weight', 'depth_weight', 'rgb_edge_weight')]
    gain = float(config['gain'])
    if (not all(math.isfinite(w) and w >= 0 for w in weights)
            or not math.isclose(sum(weights), 1., rel_tol=0., abs_tol=1e-8)
            or not math.isfinite(gain) or gain < 0):
        raise ValueError('Boundary weights must be nonnegative and sum to one; gain must be finite/nonnegative')
    components = []
    for name, value in zip(CONTINUOUS_BOUNDARY_FIELDS, (sam_strength, depth_strength, rgb_strength)):
        value = torch.as_tensor(value, device=scores.device).detach()
        if value.shape != scores.shape or not value.is_floating_point():
            raise ValueError('Continuous boundary component must be aligned floating-point values: ' + name)
        if not torch.isfinite(value).all() or ((value < 0) | (value > 1)).any():
            raise ValueError('Continuous boundary component must be finite in [0, 1]: ' + name)
        components.append(value.to(scores.dtype))
    combined = sum(weight * component for weight, component in zip(weights, components))
    scaled = gain * combined
    multiplier = 1 + scaled
    boost = torch.log1p(scaled)
    weighted = scores.clone()
    finite = torch.isfinite(scores)
    weighted[finite] += boost[finite]
    if not torch.isfinite(multiplier).all() or torch.isposinf(weighted).any():
        raise FloatingPointError('Continuous boundary weighting overflowed the score dtype')
    return dict(scores=weighted, combined_boundary_strength=combined,
                boundary_multiplier=multiplier, boundary_log_boost=boost)


def boundary_weighted_scores(scores, boundary_strength, log_weight):
    """Increase PUP importance in log space without reversing negative scores.

    ``s = log(det(F)) + log_weight * b`` ranks ``det(F)*exp(log_weight*b)``.
    Boundary strengths must be aligned values in [0, 1]; the current experiment
    uses the existing binary boundary mask. Singular Fisher scores remain -inf:
    multiplying a zero determinant cannot invent visual evidence for a point.
    The original input is never modified and non-boundary values stay exact.
    """
    if not math.isfinite(float(log_weight)) or float(log_weight) < 0:
        raise ValueError("Boundary log weight must be finite and nonnegative")
    scores = torch.as_tensor(scores).detach()
    if scores.ndim != 1 or not scores.is_floating_point():
        raise ValueError("Expected one floating-point PUP score per Gaussian")
    if torch.isnan(scores).any() or torch.isposinf(scores).any():
        raise FloatingPointError("PUP scores contain NaN or positive infinity")
    strength = torch.as_tensor(boundary_strength, device=scores.device)
    if strength.shape != scores.shape:
        raise ValueError("Boundary strengths and PUP scores are not aligned")
    if not torch.isfinite(strength).all() or ((strength < 0) | (strength > 1)).any():
        raise ValueError("Boundary strengths must be finite and in [0, 1]")
    weighted = scores.clone()
    affected = (strength > 0) & torch.isfinite(scores)
    weighted[affected] += float(log_weight) * strength[affected].to(scores.dtype)
    if torch.isposinf(weighted).any():
        raise FloatingPointError("Boundary weighting overflowed the PUP score dtype")
    return weighted


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
