"""Instance-constrained SPS selection, with an exact global Gaussian budget.

This module never infers semantic classes. Input IDs are class-free pseudo-labels;
only sufficiently supported regions receive a hard retention floor. Original
indices and original instance statistics survive every pruning event.
"""
import hashlib
import json
import math
import os
import tempfile

import numpy as np


DEFAULT_CONFIG = {
    "min_original_count": 32,
    "min_direct_count": 16,
    "min_multiview_direct_count": 8,
    "min_direct_views": 2,
    "min_mean_direct_confidence": 0.45,
    "base_retention": 0.02,
    "small_retention_boost": 0.18,
    "min_keep": 16,
    "max_keep_floor": 2048,
    "small_count_reference": 1024,
    "small_area_fraction": 0.005,
    "within_instance_rank_weight": 0.20,
    "small_score_boost": 0.10,
    "propagation_weight": 0.25,
    "spatial_cells": 27,
}


def _numpy(value):
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def global_sps_keep(scores, keep_count):
    """Exact global SPS top-k, with row order breaking ties on old Torch builds."""
    values = _numpy(scores).reshape(-1)
    if not np.isfinite(values).all():
        raise ValueError("SPS scores must be finite")
    if int(keep_count) != keep_count or not 0 <= keep_count <= len(values):
        raise ValueError("Invalid global keep budget")
    keep = np.zeros(len(values), dtype=bool)
    keep[np.argsort(-values, kind="mergesort")[:int(keep_count)]] = True
    if hasattr(scores, "detach"):
        import torch
        return torch.as_tensor(keep, dtype=torch.bool, device=scores.device)
    return keep


def xyz_sha256(xyz):
    xyz = np.ascontiguousarray(_numpy(xyz), dtype="<f4")
    if xyz.ndim != 2 or xyz.shape[1] != 3 or not np.isfinite(xyz).all():
        raise ValueError("xyz must be a finite N by 3 array")
    return hashlib.sha256(xyz.tobytes()).hexdigest()


def _config(config):
    result = dict(DEFAULT_CONFIG)
    if config:
        unknown = set(config) - set(result)
        if unknown:
            raise ValueError("Unknown instance pruning settings: %s" % sorted(unknown))
        result.update(config)
    for key in ("base_retention", "small_retention_boost", "min_mean_direct_confidence",
                "within_instance_rank_weight", "small_score_boost", "propagation_weight"):
        if not 0 <= result[key] <= 1:
            raise ValueError("%s must be in [0, 1]" % key)
    for key in ("min_original_count", "min_direct_count", "min_multiview_direct_count",
                "min_direct_views", "min_keep", "max_keep_floor", "small_count_reference", "spatial_cells"):
        if int(result[key]) != result[key] or result[key] < 1:
            raise ValueError("%s must be a positive integer" % key)
    if not 0 < result["small_area_fraction"] <= 1:
        raise ValueError("small_area_fraction must be in (0, 1]")
    if result["base_retention"] + result["small_retention_boost"] > 1:
        raise ValueError("combined retention rate cannot exceed 1")
    if result["min_keep"] > result["max_keep_floor"]:
        raise ValueError("max_keep_floor cannot be smaller than min_keep")
    return result


def _ranks(values):
    """Average percentile rank: equal SPS scores remain equal."""
    if len(values) <= 1:
        return np.full(len(values), 0.5, dtype=np.float64)
    unique, inverse, counts = np.unique(values, return_inverse=True, return_counts=True)
    del unique
    ranks = (np.cumsum(counts) - 0.5 * (counts + 1)) / float(len(values) - 1)
    return ranks[inverse]


def _best(indices, priority, original_ids, count):
    """Stable top-k with immutable source index as the final tie breaker."""
    if count <= 0:
        return np.empty(0, dtype=np.int64)
    order = np.lexsort((original_ids[indices], -priority[indices]))
    return indices[order[:count]]


def _groups(ids):
    order = np.argsort(ids, kind="mergesort")
    boundaries = np.r_[0, np.flatnonzero(np.diff(ids[order])) + 1, len(ids)]
    for left, right in zip(boundaries[:-1], boundaries[1:]):
        if right > left:
            yield int(ids[order[left]]), order[left:right]


def _atomic_npz(path, arrays):
    parent = os.path.dirname(os.path.abspath(str(path)))
    os.makedirs(parent, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".instances-", suffix=".npz", dir=parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            np.savez_compressed(stream, **arrays)
        os.replace(temporary, str(path))
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


class InstancePruningState:
    """Per-Gaussian metadata plus immutable, original-model retention policy."""

    @classmethod
    def load(cls, priors_path, source_xyz, config=None, projected_area=None):
        """Load original label-free priors and verify exact row alignment.

        projected_area is an optional {instance_id: median_train_area_fraction}
        mapping. Missing measurements use an explicitly recorded count proxy.
        A value of zero means no valid area measurement and uses the proxy.
        """
        self = cls()
        self.config = _config(config)
        xyz = np.asarray(_numpy(source_xyz), dtype=np.float32)
        self.source_xyz_sha256 = xyz_sha256(xyz)
        with np.load(str(priors_path), allow_pickle=False) as prior:
            required = ("instance_ids", "instance_confidence", "instance_views", "instance_source", "source_xyz_sha256")
            if not all(key in prior.files for key in required):
                raise ValueError("Instance priors lack required label-free metadata")
            expected = str(prior["source_xyz_sha256"].item())
            if expected != self.source_xyz_sha256:
                raise ValueError("Instance prior source_xyz_sha256 mismatch: Gaussian rows are not aligned")
            self.instance_ids = np.array(prior["instance_ids"], dtype=np.int32)
            # Existing label-free City priors encode unassigned as -1; all
            # compressed sidecars consistently encode unknown as 0.
            self.instance_ids[self.instance_ids == -1] = 0
            self.instance_confidence = np.array(prior["instance_confidence"], dtype=np.float32)
            self.instance_views = np.array(prior["instance_views"], dtype=np.uint16)
            self.instance_source = np.array(prior["instance_source"], dtype=np.uint8)
        self.original_count = len(xyz)
        self.original_ids = np.arange(self.original_count, dtype=np.int64)
        self._validate_rows(len(xyz))
        projected_area = projected_area or {}
        self.policy = []
        for instance_id, indices in _groups(self.instance_ids):
            if instance_id == 0:
                continue
            direct = self.instance_source[indices] == 1
            direct_indices = indices[direct]
            multi_count = int(np.count_nonzero(self.instance_views[direct_indices] >= self.config["min_direct_views"]))
            confidence = float(self.instance_confidence[direct_indices].mean()) if len(direct_indices) else 0.0
            count = len(indices)
            reliable = (count >= self.config["min_original_count"] and len(direct_indices) >= self.config["min_direct_count"]
                        and multi_count >= self.config["min_multiview_direct_count"]
                        and confidence >= self.config["min_mean_direct_confidence"])
            area = projected_area.get(instance_id, projected_area.get(str(instance_id)))
            if area is not None and (not math.isfinite(float(area)) or not 0 <= float(area) <= 1):
                raise ValueError("Projected area must be a finite image-area fraction")
            if area is not None and area > 0:
                smallness = min(1.0, math.sqrt(self.config["small_area_fraction"] / float(area)))
                small = float(area) <= self.config["small_area_fraction"]
                smallness_source = "train_projected_area"
            else:
                smallness = min(1.0, math.sqrt(self.config["small_count_reference"] / float(count)))
                small = count <= self.config["small_count_reference"]
                smallness_source = "original_count_proxy"
            fraction = self.config["base_retention"] + self.config["small_retention_boost"] * smallness
            floor = min(count, self.config["max_keep_floor"], max(self.config["min_keep"], int(math.ceil(count * fraction)))) if reliable else 0
            self.policy.append({"instance_id": instance_id, "original_count": count, "direct_count": len(direct_indices),
                                "multiview_direct_count": multi_count, "mean_direct_confidence": confidence,
                                "reliable": bool(reliable), "protected": bool(reliable), "small": bool(small),
                                "smallness": float(smallness), "smallness_source": smallness_source,
                                "projected_area_fraction": None if area is None else float(area), "min_keep": int(floor),
                                "bbox_min": np.quantile(xyz[indices], 0.02, axis=0).tolist(),
                                "bbox_max": np.quantile(xyz[indices], 0.98, axis=0).tolist()})
        self._policy_by_id = {row["instance_id"]: row for row in self.policy}
        return self

    @classmethod
    def load_resume(cls, path, current_xyz, config=None):
        """Restore metadata using the CURRENT model fingerprint, never source xyz."""
        self = cls()
        with np.load(str(path), allow_pickle=False) as saved:
            if str(saved["format"].item()) != "pwsr-instance-v1":
                raise ValueError("Unsupported instance sidecar format")
            if str(saved["current_xyz_sha256"].item()) != xyz_sha256(current_xyz):
                raise ValueError("Instance sidecar current_xyz_sha256 mismatch")
            self.source_xyz_sha256 = str(saved["source_xyz_sha256"].item())
            self.original_count = int(saved["original_count"].item())
            self.config = _config(json.loads(str(saved["config_json"].item())))
            if config is not None and _config(config) != self.config:
                raise ValueError("Instance pruning configuration changed on resume")
            self.policy = json.loads(str(saved["policy_json"].item()))
            self.original_ids = np.array(saved["original_ids"], dtype=np.int64)
            self.instance_ids = np.array(saved["instance_ids"], dtype=np.int32)
            self.instance_confidence = np.array(saved["instance_confidence"], dtype=np.float32)
            self.instance_views = np.array(saved["instance_views"], dtype=np.uint16)
            self.instance_source = np.array(saved["instance_source"], dtype=np.uint8)
        self._validate_rows(len(_numpy(current_xyz)))
        self._policy_by_id = {int(row["instance_id"]): row for row in self.policy}
        if len(self._policy_by_id) != len(self.policy) or 0 in self._policy_by_id:
            raise ValueError("Invalid policy instance IDs")
        if sum(int(row["original_count"]) for row in self.policy) > self.original_count:
            raise ValueError("Invalid original instance statistics")
        for row in self.policy:
            if not 0 <= row["min_keep"] <= row["original_count"]:
                raise ValueError("Invalid original instance floor")
        return self

    def _validate_rows(self, count):
        for key in ("original_ids", "instance_ids", "instance_confidence", "instance_views", "instance_source"):
            value = getattr(self, key)
            if value.shape != (count,):
                raise ValueError("%s row count does not match model" % key)
        if np.any(self.instance_ids < 0) or np.any(~np.isin(self.instance_source, (0, 1, 2, 3))):
            raise ValueError("Invalid instance IDs or source codes")
        if not np.isfinite(self.instance_confidence).all() or np.any((self.instance_confidence < 0) | (self.instance_confidence > 1)):
            raise ValueError("Invalid instance confidence")
        if np.any(self.original_ids < 0) or np.any(self.original_ids >= self.original_count) or len(np.unique(self.original_ids)) != count:
            raise ValueError("Invalid or duplicated original Gaussian indices")

    def policy_dict(self):
        return {"format": "pwsr-instance-policy-v1", "source_xyz_sha256": self.source_xyz_sha256,
                "unknown_id": 0, "input_unknown_normalization": "-1 to 0",
                "original_count": self.original_count, "config": dict(self.config),
                "floor_total": sum(row["min_keep"] for row in self.policy), "instances": self.policy}

    def export_policy(self, path):
        parent = os.path.dirname(os.path.abspath(str(path)))
        os.makedirs(parent, exist_ok=True)
        fd, temporary = tempfile.mkstemp(prefix=".policy-", suffix=".json", dir=parent)
        try:
            with os.fdopen(fd, "w") as stream:
                json.dump(self.policy_dict(), stream, indent=2, allow_nan=False)
                stream.write("\n")
            os.replace(temporary, str(path))
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)

    def save(self, path, current_xyz):
        self._validate_rows(len(_numpy(current_xyz)))
        arrays = {key: getattr(self, key) for key in ("original_ids", "instance_ids", "instance_confidence", "instance_views", "instance_source")}
        arrays.update(format=np.asarray("pwsr-instance-v1"), original_count=np.asarray(self.original_count, dtype=np.int64),
                      source_xyz_sha256=np.asarray(self.source_xyz_sha256), current_xyz_sha256=np.asarray(xyz_sha256(current_xyz)),
                      config_json=np.asarray(json.dumps(self.config, sort_keys=True)),
                      policy_json=np.asarray(json.dumps(self.policy, sort_keys=True, allow_nan=False)))
        _atomic_npz(path, arrays)

    def subset_(self, keep):
        mask = _numpy(keep)
        if mask.dtype != np.bool_ or mask.shape != self.instance_ids.shape:
            raise ValueError("Metadata subset requires a model-aligned boolean keep mask")
        for key in ("original_ids", "instance_ids", "instance_confidence", "instance_views", "instance_source"):
            setattr(self, key, getattr(self, key)[mask].copy())
        return self

    def _spatial_representatives(self, indices, xyz, priority, row, quota):
        if quota <= 0 or not len(indices):
            return np.empty(0, dtype=np.int64)
        resolution = max(1, int(math.ceil(min(quota, self.config["spatial_cells"]) ** (1.0 / 3.0))))
        lower, upper = np.asarray(row["bbox_min"]), np.asarray(row["bbox_max"])
        normalized = np.clip((xyz[indices] - lower) / np.maximum(upper - lower, 1e-8), 0, 1 - 1e-8)
        cells = np.floor(normalized * resolution).astype(np.int64)
        keys = cells[:, 0] + resolution * cells[:, 1] + resolution ** 2 * cells[:, 2]
        representatives, locations = [], []
        for cell, local in _groups(keys):
            del cell
            candidate = _best(indices[local], priority, self.original_ids, 1)[0]
            representatives.append(candidate)
            locations.append(cells[local[0]].astype(float))
        representatives = np.asarray(representatives, dtype=np.int64)
        if len(representatives) <= quota:
            return representatives
        # Farthest-cell selection spreads small floors over the original extent.
        # Every tie uses SPS priority, then immutable original Gaussian index.
        locations = np.asarray(locations)
        selected = []
        distance = np.full(len(representatives), np.inf)
        for _ in range(quota):
            available = np.ones(len(representatives), dtype=bool)
            available[selected] = False
            candidates = np.flatnonzero(available)
            ordering = np.lexsort((self.original_ids[representatives[candidates]], -priority[representatives[candidates]], -distance[candidates]))
            chosen = candidates[ordering[0]]
            selected.append(chosen)
            distance = np.minimum(distance, ((locations - locations[chosen]) ** 2).sum(axis=1))
        return representatives[selected]

    def select_keep(self, scores, xyz, keep_count):
        """Return exact-K keep mask and audit; never silently relax hard floors.

        Scores may be NumPy or torch tensors; torch input returns torch.bool on
        the same device. Selection itself is deterministic CPU/NumPy work.
        """
        xyz = np.asarray(_numpy(xyz), dtype=np.float32)
        importance = np.asarray(_numpy(scores), dtype=np.float64).reshape(-1)
        count = len(self.instance_ids)
        if xyz.shape != (count, 3) or importance.shape != (count,) or not np.isfinite(importance).all() or not np.isfinite(xyz).all():
            raise ValueError("Scores/xyz must be finite and aligned to instance metadata")
        if int(keep_count) != keep_count or not 0 <= keep_count <= count:
            raise ValueError("keep_count must be an integer within the current Gaussian count")
        keep_count = int(keep_count)
        floor_total = sum(row["min_keep"] for row in self.policy)
        if floor_total > keep_count:
            raise ValueError("Instance floors require %d Gaussians but global keep budget is %d; increase budget or explicitly change protection policy" % (floor_total, keep_count))
        grouped = dict(_groups(self.instance_ids))
        for row in self.policy:
            available = len(grouped.get(row["instance_id"], ()))
            if available < row["min_keep"]:
                raise ValueError("Instance %d already has %d points below immutable floor %d" % (row["instance_id"], available, row["min_keep"]))
        priority = _ranks(importance)
        keep = np.zeros(count, dtype=bool)
        views_factor = np.minimum(self.instance_views.astype(np.float64) / 3.0, 1.0)
        source_weight = np.where(self.instance_source == 1, 1.0, np.where(self.instance_source == 2, self.config["propagation_weight"], 0.0))
        evidence = self.instance_confidence * views_factor * source_weight
        for instance_id, indices in grouped.items():
            row = self._policy_by_id.get(instance_id)
            if row is None or not row["reliable"]:
                continue
            priority[indices] += self.config["within_instance_rank_weight"] * evidence[indices] * (_ranks(importance[indices]) - 0.5)
            priority[indices] += self.config["small_score_boost"] * row["smallness"] * evidence[indices]
            quota = int(row["min_keep"])
            representatives = self._spatial_representatives(indices, xyz, priority, row, quota)
            keep[representatives] = True
            remainder = indices[~keep[indices]]
            keep[_best(remainder, priority, self.original_ids, quota - len(representatives))] = True
        remaining = np.flatnonzero(~keep)
        keep[_best(remaining, priority, self.original_ids, keep_count - int(keep.sum()))] = True
        instances = []
        for row in self.policy:
            indices = grouped.get(row["instance_id"], np.empty(0, dtype=np.int64))
            after = int(keep[indices].sum())
            record = {key: row[key] for key in ("instance_id", "original_count", "min_keep", "protected", "small", "smallness", "smallness_source", "reliable", "direct_count", "multiview_direct_count", "mean_direct_confidence", "projected_area_fraction")}
            record.update(before=len(indices), after=after, lost=after == 0, floor_violation=after < row["min_keep"])
            instances.append(record)
        audit = {"format": "pwsr-instance-prune-audit-v1", "before": count, "requested": keep_count,
                 "kept": int(keep.sum()), "floor_total": floor_total, "instances": instances,
                 "protected_instances": sum(row["protected"] for row in self.policy),
                 "protected_lost": sum(row["protected"] and row["lost"] for row in instances),
                 "unknown_before": int(np.count_nonzero(self.instance_ids == 0)),
                 "unknown_after": int(np.count_nonzero(keep & (self.instance_ids == 0)))}
        if int(keep.sum()) != keep_count or audit["protected_lost"] or any(row["floor_violation"] for row in instances):
            raise AssertionError("Exact-budget instance protection invariant failed")
        if hasattr(scores, "detach"):
            import torch
            keep = torch.as_tensor(keep, dtype=torch.bool, device=scores.device)
        return keep, audit
