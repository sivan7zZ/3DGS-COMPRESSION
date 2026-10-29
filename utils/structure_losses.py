"""Differentiable visible-structure supervision on training cameras only."""
import math
import os

import numpy as np
import torch
import torch.nn.functional as F

from utils.instance_roi import TrainingInstanceROIs, weighted_region_l1


def _zero(image):
    return image.sum() * 0.0


def boundary_survival_loss(image, target, weights, margin=0.7):
    """Penalize missing, weakened or reversed reference boundary contrast."""
    if image.shape != target.shape or weights.shape != image.shape[-2:]:
        raise ValueError("Boundary supervision must match the image")
    w = weights.to(device=image.device, dtype=image.dtype)
    numerator, denominator = _zero(image), image.new_tensor(0.)
    for axis in (1, 2):
        if axis == 1:
            prediction = image[:, 1:, :] - image[:, :-1, :]
            reference = target[:, 1:, :] - target[:, :-1, :]
            active = torch.maximum(w[1:, :], w[:-1, :])
        else:
            prediction = image[:, :, 1:] - image[:, :, :-1]
            reference = target[:, :, 1:] - target[:, :, :-1]
            active = torch.maximum(w[:, 1:], w[:, :-1])
        energy = reference.square().sum(0)
        magnitude = energy.sqrt()
        active = active * (energy >= 3. * (2. / 255.) ** 2).to(active.dtype) * magnitude / (magnitude + .02)
        projection = (prediction * reference).sum(0) / energy.clamp_min(1e-8)
        numerator = numerator + (F.relu(margin - projection) * active).sum()
        denominator = denominator + active.sum()
    return numerator / denominator.clamp_min(1e-8)


def thin_survival_loss(image, target, weights, cell_size=16, margin=0.65):
    """Preserve thin contrast and penalize weak links within local line cells.

    The signed high-pass projection rejects reversed halos. A soft minimum
    over each occupied cell gives broken line sections a useful gradient even
    when most of a long structure remains present. This is a training surrogate,
    not a claim of guaranteed topological equivalence.
    """
    if image.shape != target.shape or weights.shape != image.shape[-2:]:
        raise ValueError("Thin supervision must match the image")
    if int(cell_size) != cell_size or cell_size < 1:
        raise ValueError("cell_size must be positive")
    w = weights.to(device=image.device, dtype=image.dtype)
    luma = image.new_tensor([.299, .587, .114]).view(3, 1, 1)
    gray = (image[:3] * luma).sum(0)[None, None]
    truth = (target[:3] * luma).sum(0)[None, None]
    hp = (gray - F.avg_pool2d(F.pad(gray, (1, 1, 1, 1), mode="replicate"), 3, stride=1))[0, 0]
    ref = (truth - F.avg_pool2d(F.pad(truth, (1, 1, 1, 1), mode="replicate"), 3, stride=1))[0, 0]
    strength = ref.abs()
    w = w * (strength >= 2. / 255.).to(w.dtype) * strength / (strength + .02)
    ratio = hp * torch.sign(ref) / strength.clamp_min(1e-8)
    pixel_loss = (F.relu(margin - ratio) * w).sum() / w.sum().clamp_min(1e-8)
    height, width = w.shape
    ph, pw = (-height) % cell_size, (-width) % cell_size
    padded_w = F.pad(w, (0, pw, 0, ph))
    padded_r = F.pad(ratio, (0, pw, 0, ph))
    nh, nw = padded_w.shape[0] // cell_size, padded_w.shape[1] // cell_size
    def cells(value):
        return value.reshape(nh, cell_size, nw, cell_size).permute(0, 2, 1, 3).reshape(-1, cell_size ** 2)
    cw, cr = cells(padded_w), cells(padded_r)
    mass = cw.sum(1)
    log_values = torch.where(cw > 0, torch.log(cw.clamp_min(1e-12)) - 8. * cr,
                             cw.new_full(cw.shape, -10000.))
    minimum = -(torch.logsumexp(log_values, dim=1) - torch.log(mass.clamp_min(1e-12))) / 8.
    valid = (mass > 0).to(image.dtype)
    continuity = (F.relu(margin - minimum) * valid).sum() / valid.sum().clamp_min(1.)
    return pixel_loss + .5 * continuity + weighted_region_l1(image, target, w)


class TrainingStructureROIs(TrainingInstanceROIs):
    def __init__(self, directory, cameras):
        super().__init__(directory, cameras)
        for camera in cameras:
            name = camera.image_name
            path = os.path.realpath(os.path.join(self.directory, name + ".npz"))
            if not os.path.isfile(path):
                path = os.path.realpath(os.path.join(self.directory, os.path.splitext(name)[0] + ".npz"))
            if os.path.commonpath([self.directory, path]) != self.directory:
                raise ValueError("Structure ROI path escapes its directory")
            with np.load(path, allow_pickle=False) as data:
                shape = data["weights"].shape
                for field in ("thin_weights", "structure_weights"):
                    value = np.asarray(data[field], dtype=np.float32)
                    if value.shape != shape or not np.isfinite(value).all() or (value < 0).any():
                        raise ValueError("Invalid " + field)
                    tensor = torch.from_numpy(value.copy())
                    size = tuple(self.records[name]["weights"].shape)
                    if tuple(tensor.shape) != size:
                        tensor = F.interpolate(tensor[None, None], size=size, mode="nearest")[0, 0]
                    self.records[name][field] = tensor

    def structure_losses(self, name, image, target):
        row = self.records[name]
        return (boundary_survival_loss(image, target, row["edge_weights"]),
                thin_survival_loss(image, target, row["thin_weights"]))


class AnchorTrustRegion:
    """Keep selected visible Gaussian atoms from fading or drifting away."""
    def __init__(self, structure):
        self.structure = structure
        self._cache = None

    def loss(self, model, original_ids):
        source = self.structure
        if len(source.anchors) == 0:
            return _zero(model.get_xyz)
        ids = np.asarray(original_ids, dtype=np.int64)
        key = (len(ids), str(model.get_xyz.device))
        if self._cache is None or self._cache[0] != key:
            if np.any(np.diff(ids) <= 0):
                raise ValueError("Anchor lineage must remain in original row order")
            index = np.searchsorted(ids, source.anchors)
            if np.any(index >= len(ids)) or not np.array_equal(ids[index], source.anchors):
                raise ValueError("A required visible structure anchor was removed")
            device = model.get_xyz.device
            def tensor(value):
                return torch.as_tensor(np.array(value, copy=True), device=device, dtype=model.get_xyz.dtype)
            weight = tensor(source.anchor_strength).clamp_min(.05).sqrt()
            weight = (weight / weight.mean().clamp_min(1e-8)).clamp(.25, 4.)
            self._cache = (key, torch.as_tensor(index, device=device, dtype=torch.long),
                           tensor(source.anchor_xyz), tensor(source.anchor_log_scaling),
                           tensor(source.anchor_opacity).reshape(-1), tensor(source.anchor_features_dc).reshape(-1, 3), weight)
        _, index, xyz, scaling, opacity, dc, weight = self._cache
        active = model.get_opacity[index].reshape(-1)
        required = .6 * opacity
        alpha_loss = (F.relu(required - active) * weight).mean()
        sigma = torch.exp(scaling).max(1)[0].clamp_min(1e-6)
        displacement = ((model.get_xyz[index] - xyz) / sigma[:, None]).norm(dim=1)
        position_loss = (F.relu(displacement - 1.).square() * weight).mean()
        scale_loss = (F.relu(scaling - math.log(2.) - model._scaling[index]).mean(1) * weight).mean()
        color_loss = ((model._features_dc[index].reshape(-1, 3) - dc).abs().mean(1) * weight).mean()
        return alpha_loss + .05 * position_loss + .1 * scale_loss + .1 * color_loss

    def freeze_gradients_(self, model, original_ids, geometry_only=False):
        """Preserve selected source rendering atoms during pruning recovery.

        Zeroing Adam moments as well as gradients prevents earlier momentum
        from moving an anchor. SH appearance can be released in the later
        distillation stage; this function is used by SH3 pruning only.
        """
        key = (len(original_ids), str(model.get_xyz.device))
        if self._cache is None or self._cache[0] != key:
            self.loss(model, original_ids)
        if self._cache is None:
            return
        index = self._cache[1]
        names = ("_xyz", "_scaling", "_rotation", "_opacity")
        if not geometry_only:
            names += ("_features_dc", "_features_rest")
        for name in names:
            parameter = getattr(model, name)
            if parameter.grad is not None:
                parameter.grad.index_fill_(0, index, 0.)
            for value in model.optimizer.state.get(parameter, {}).values():
                if torch.is_tensor(value) and value.shape == parameter.shape:
                    value.index_fill_(0, index, 0.)
