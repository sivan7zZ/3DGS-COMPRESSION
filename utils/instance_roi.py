"""Training-view ROI supervision for class-agnostic instance distillation.

The optional per-view ``<image_name>.npz`` contains ``weights`` (nonnegative
float H x W, zero outside the selected instances), and optionally ``small_mask``
(bool H x W), ``edge_weights`` (nonnegative float H x W), and
``sampling_weight`` (positive scalar).  These are SAM-derived training priors;
neither semantic labels nor held-out-image labels are consumed here.
"""

import os

import numpy as np
import torch
import torch.nn.functional as F

from utils.distill_utils import sha256_file


def weighted_region_l1(image, target, weights):
    """Mean RGB error within an ROI, independent of its share of the image.

    An empty ROI has a differentiable zero loss.  A tiny object is consequently
    not diluted by dividing by all background pixels in the training image.
    """
    if image.shape != target.shape or image.ndim != 3:
        raise ValueError("ROI images must have matching C x H x W shapes")
    if weights.shape != image.shape[-2:]:
        raise ValueError("ROI weights must match the rendered image size")
    values = weights.to(device=image.device, dtype=image.dtype)
    denominator = values.sum() * image.shape[0]
    return ((image - target).abs() * values.unsqueeze(0)).sum() / denominator.clamp_min(1e-8)


def weighted_boundary_l1(image, target, edge_weights):
    """Match finite-difference RGB gradients along selected object boundaries.

    Each horizontal/vertical edge is counted once, normalized by its total
    weight and channels.  The edge uses the maximum of its two endpoint weights
    so the immediate inside/outside contrast at a silhouette is supervised.
    """
    if image.shape != target.shape or image.ndim != 3:
        raise ValueError("Boundary images must have matching C x H x W shapes")
    if edge_weights.shape != image.shape[-2:]:
        raise ValueError("Boundary weights must match the rendered image size")
    values = edge_weights.to(device=image.device, dtype=image.dtype)
    weight_x = torch.maximum(values[:, 1:], values[:, :-1])
    weight_y = torch.maximum(values[1:, :], values[:-1, :])
    error_x = ((image[:, :, 1:] - image[:, :, :-1])
               - (target[:, :, 1:] - target[:, :, :-1])).abs()
    error_y = ((image[:, 1:, :] - image[:, :-1, :])
               - (target[:, 1:, :] - target[:, :-1, :])).abs()
    numerator = ((error_x * weight_x.unsqueeze(0)).sum()
                 + (error_y * weight_y.unsqueeze(0)).sum())
    denominator = (weight_x.sum() + weight_y.sum()) * image.shape[0]
    return numerator / denominator.clamp_min(1e-8)


def boundary_band(mask):
    """One-pixel inner/outer band, without treating image borders as objects."""
    if mask.ndim != 2:
        raise ValueError("Instance mask must be H x W")
    value = mask.to(dtype=torch.float32)[None, None]
    dilated = F.max_pool2d(value, 3, stride=1, padding=1)
    eroded = -F.max_pool2d(-value, 3, stride=1, padding=1)
    return (dilated - eroded)[0, 0]


def _nonnegative_map(value, name, shape=None):
    result = np.asarray(value, dtype=np.float32)
    if result.ndim != 2 or (shape is not None and result.shape != shape):
        raise ValueError(name + " must be an H x W array matching weights")
    if not np.isfinite(result).all() or (result < 0).any():
        raise ValueError(name + " must contain finite nonnegative values")
    return result


class TrainingInstanceROIs:
    """Load only explicitly supplied training cameras and fingerprint inputs."""

    def __init__(self, directory, cameras):
        self.directory = os.path.realpath(directory)
        self.records = {}
        self.provenance = {}
        for camera in cameras:
            name = camera.image_name
            if name in self.records:
                raise ValueError("Duplicate training camera name: " + name)
            # COLMAP names may include subdirectories.  Resolve them below the
            # prior directory and never permit an absolute/path-traversal name.
            path = os.path.realpath(os.path.join(self.directory, name + ".npz"))
            if os.path.commonpath([self.directory, path]) != self.directory:
                raise ValueError("ROI camera path escapes its directory: " + name)
            # This COLMAP reader retains '.png' in image_name, while SAM and
            # the profile exporter key their caches by image stem.
            if not os.path.isfile(path) and os.path.splitext(name)[1].lower() in {".png", ".jpg", ".jpeg", ".tif", ".tiff"}:
                path = os.path.realpath(os.path.join(self.directory, os.path.splitext(name)[0] + ".npz"))
                if os.path.commonpath([self.directory, path]) != self.directory:
                    raise ValueError("ROI camera path escapes its directory: " + name)
            if not os.path.isfile(path):
                raise FileNotFoundError("Missing training-view instance ROI: " + path)
            with np.load(path, allow_pickle=False) as data:
                if "weights" not in data:
                    raise ValueError("ROI archive has no weights: " + path)
                weights = _nonnegative_map(data["weights"], "weights")
                small = _nonnegative_map(data["small_mask"], "small_mask", weights.shape) > 0 if "small_mask" in data else weights > 0
                edge = _nonnegative_map(data["edge_weights"], "edge_weights", weights.shape) if "edge_weights" in data else None
                sampling = float(data["sampling_weight"]) if "sampling_weight" in data else 1.0 + min(3.0, 16.0 * float(small.mean()))
            if not np.isfinite(sampling) or sampling <= 0:
                raise ValueError("ROI sampling_weight must be finite and positive: " + path)
            size = (int(camera.image_height), int(camera.image_width))
            values = torch.from_numpy(weights.copy())
            small_tensor = torch.from_numpy(small.astype(np.float32))
            if tuple(values.shape) != size:
                values = F.interpolate(values[None, None], size=size, mode="nearest")[0, 0]
                small_tensor = F.interpolate(small_tensor[None, None], size=size, mode="nearest")[0, 0]
            if edge is None:
                edge_values = boundary_band(small_tensor)
            else:
                edge_values = torch.from_numpy(edge.copy())
                if tuple(edge_values.shape) != size:
                    edge_values = F.interpolate(edge_values[None, None], size=size, mode="nearest")[0, 0]
            self.records[name] = {"weights": values, "edge_weights": edge_values,
                                  "sampling_weight": sampling}
            self.provenance[name] = {"sha256": sha256_file(path), "image_size": list(size),
                                     "sampling_weight": sampling}

    def sampling_weights(self, view_names):
        return [self.records[name]["sampling_weight"] for name in view_names]

    def losses(self, name, image, target):
        record = self.records[name]
        return (weighted_region_l1(image, target, record["weights"]),
                weighted_boundary_l1(image, target, record["edge_weights"]))
