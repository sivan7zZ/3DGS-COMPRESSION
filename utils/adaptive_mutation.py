"""Native Gaussian mutations for adaptive compression, with CPU auxiliary state.

No custom rendering attributes are introduced. ``original_ids`` is a teacher-row
index only: a moment-matched merge receives -1 and a fresh ``row_ids`` lineage ID.
Restored primitives keep their original row ID; merged parents cannot be restored
independently. Adam moments survive deletion for untouched rows and start at zero
for restored or merged rows (the optimizer's global step is retained).
"""
import os
import tempfile

import numpy as np
import torch

from utils.pruning_scores import CONTINUOUS_BOUNDARY_FIELDS


PARAMETERS = {"xyz": "_xyz", "f_dc": "_features_dc", "f_rest": "_features_rest",
              "opacity": "_opacity", "scaling": "_scaling", "rotation": "_rotation"}


def _array(value):
    if torch.is_tensor(value):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _rotation_matrix(quaternion):
    q = quaternion / torch.linalg.vector_norm(quaternion, dim=-1, keepdim=True).clamp_min(1e-12)
    w, x, y, z = q.unbind(-1)
    return torch.stack((1 - 2 * (y*y + z*z), 2 * (x*y - w*z), 2 * (x*z + w*y),
                        2 * (x*y + w*z), 1 - 2 * (x*x + z*z), 2 * (y*z - w*x),
                        2 * (x*z - w*y), 2 * (y*z + w*x), 1 - 2 * (x*x + y*y)),
                       dim=-1).reshape(-1, 3, 3)


def _matrix_quaternion(rotation):
    """Stable matrix-to-wxyz conversion, including rotations close to pi."""
    r = rotation
    qabs = torch.sqrt(torch.clamp(torch.stack((
        1 + r[:, 0, 0] + r[:, 1, 1] + r[:, 2, 2],
        1 + r[:, 0, 0] - r[:, 1, 1] - r[:, 2, 2],
        1 - r[:, 0, 0] + r[:, 1, 1] - r[:, 2, 2],
        1 - r[:, 0, 0] - r[:, 1, 1] + r[:, 2, 2]), dim=-1), min=0))
    candidates = torch.stack((
        torch.stack((qabs[:, 0]**2, r[:, 2, 1]-r[:, 1, 2], r[:, 0, 2]-r[:, 2, 0], r[:, 1, 0]-r[:, 0, 1]), -1),
        torch.stack((r[:, 2, 1]-r[:, 1, 2], qabs[:, 1]**2, r[:, 1, 0]+r[:, 0, 1], r[:, 0, 2]+r[:, 2, 0]), -1),
        torch.stack((r[:, 0, 2]-r[:, 2, 0], r[:, 1, 0]+r[:, 0, 1], qabs[:, 2]**2, r[:, 2, 1]+r[:, 1, 2]), -1),
        torch.stack((r[:, 1, 0]-r[:, 0, 1], r[:, 0, 2]+r[:, 2, 0], r[:, 2, 1]+r[:, 1, 2], qabs[:, 3]**2), -1)), dim=1)
    candidates = candidates / (2 * qabs.clamp_min(1e-8)[..., None])
    result = candidates[torch.arange(len(r), device=r.device), qabs.argmax(-1)]
    result = result / torch.linalg.vector_norm(result, dim=-1, keepdim=True).clamp_min(1e-12)
    return torch.where(result[:, :1] < 0, -result, result)


def covariance_from_parameters(log_scales, quaternion):
    rotation = _rotation_matrix(quaternion)
    return (rotation * torch.exp(2 * log_scales)[:, None, :]) @ rotation.transpose(1, 2)


@torch.no_grad()
def moment_match_pairs(params, pairs, visibility=None):
    """Return standard raw 3DGS parameters and normalized pair weights.

    Weights are sigmoid(opacity) * visibility, never the pruning score. The
    covariance includes both within-Gaussian covariance and center displacement.
    """
    device = params["xyz"].device
    pairs = torch.as_tensor(pairs, dtype=torch.long, device=device).reshape(-1, 2)
    if not len(pairs):
        return {key: value[:0].clone() for key, value in params.items()}, np.empty((0, 2))
    i, j = pairs.unbind(1)
    alpha = torch.sigmoid(params["opacity"]).reshape(-1)
    vis = torch.ones_like(alpha) if visibility is None else torch.as_tensor(visibility, dtype=alpha.dtype, device=device)
    vis = torch.nan_to_num(vis.reshape(-1), nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0)
    raw_weights = torch.stack((alpha[i]*vis[i], alpha[j]*vis[j]), dim=1)
    fallback = torch.stack((alpha[i], alpha[j]), dim=1).clamp_min(1e-12)
    raw_weights = torch.where((raw_weights.sum(1) <= 1e-12)[:, None], fallback, raw_weights)
    weights = raw_weights / raw_weights.sum(1, keepdim=True).clamp_min(1e-12)
    wi, wj = weights[:, 0], weights[:, 1]
    center = wi[:, None]*params["xyz"][i] + wj[:, None]*params["xyz"][j]
    ci = covariance_from_parameters(params["scaling"][i], params["rotation"][i])
    cj = covariance_from_parameters(params["scaling"][j], params["rotation"][j])
    di, dj = params["xyz"][i]-center, params["xyz"][j]-center
    covariance = wi[:, None, None]*(ci+di[:, :, None]*di[:, None, :]) + wj[:, None, None]*(cj+dj[:, :, None]*dj[:, None, :])
    values, vectors = torch.linalg.eigh((covariance+covariance.transpose(1, 2))*0.5)
    # Eigenvectors are orthonormal but may be a reflection; make a proper rotation.
    vectors[:, :, -1] *= torch.where(torch.linalg.det(vectors) < 0, -1.0, 1.0)[:, None]
    result = {"xyz": center, "scaling": 0.5*torch.log(values.clamp_min(1e-12)),
              "rotation": _matrix_quaternion(vectors)}
    for key in ("f_dc", "f_rest"):
        shape = (-1,) + (1,)*(params[key].ndim-1)
        result[key] = wi.reshape(shape)*params[key][i]+wj.reshape(shape)*params[key][j]
    union_alpha = (1-(1-alpha[i])*(1-alpha[j])).clamp(1e-7, 1-1e-7)
    result["opacity"] = torch.log(union_alpha/(1-union_alpha))[:, None]
    if not all(torch.isfinite(value).all().item() for value in result.values()):
        raise ValueError("Non-finite moment-matched Gaussian")
    return result, weights.cpu().numpy()


@torch.no_grad()
def find_safe_merge_pairs(model, metadata, max_pairs, knn=8,
                          confidence_threshold=0.5, distance_factor=0.5,
                          scale_ratio=1.5, sh_threshold=0.05,
                          instance_min_counts=None, query_batch=32768,
                          preserve_boundary=True):
    """Bounded CPU FLANN KNN search, then deterministic disjoint-pair selection.

    FLANN is approximate (8 kd-trees / 64 checks). Geometry, all-SH RMS and scale
    thresholds are verified exactly after lookup. There is never an NxN matrix.
    Optional instance quotas account for every accepted pair's one-row reduction.
    The boundary-weighted PUP variant alone disables the boundary veto; reliable
    instance, topology, geometry and appearance checks still apply unchanged.
    """
    n = len(metadata["instance_ids"])
    max_pairs = int(max_pairs)
    if max_pairs <= 0 or n < 2:
        return np.empty((0, 2), dtype=np.int64)
    ids = np.asarray(metadata["instance_ids"])
    confidence = np.asarray(metadata["confidence"])
    safe = (ids >= 0) & (confidence >= confidence_threshold)
    if preserve_boundary:
        safe &= ~np.asarray(metadata["boundary_mask"], dtype=bool)
    safe &= ~np.asarray(metadata["topology_critical"], dtype=bool)
    selected = np.flatnonzero(safe)
    if len(selected) < 2:
        return np.empty((0, 2), dtype=np.int64)
    xyz = model._xyz.detach().cpu().numpy()
    logs = model._scaling.detach().cpu().numpy()
    sh = torch.cat((model._features_dc, model._features_rest), dim=1).detach().cpu().numpy().reshape(n, -1)
    xyz_safe = np.ascontiguousarray(xyz[selected], dtype=np.float32)
    if not (np.isfinite(xyz_safe).all() and np.isfinite(logs[selected]).all() and np.isfinite(sh[selected]).all()):
        raise ValueError("Non-finite Gaussian merge search input")
    import cv2
    cv2.setRNGSeed(0)
    # A single exact tree also handles tiny/degenerate sets where OpenCV's
    # randomized multi-tree approximate search can fail to fill k neighbors.
    exact_search = len(selected) <= 4096
    index = cv2.flann_Index(xyz_safe, dict(algorithm=1, trees=1 if exact_search else 8))
    pair_parts, distance_parts = [], []
    k = min(int(knn)+1, len(selected))
    log_limit = np.log(float(scale_ratio))
    mean_scale = np.exp(logs).mean(axis=1)
    for start in range(0, len(selected), int(query_batch)):
        query = xyz_safe[start:start+int(query_batch)]
        neighbors, squared_distance = index.knnSearch(query, k, params=dict(checks=-1 if exact_search else 64))
        a = np.broadcast_to(selected[start:start+len(query), None], neighbors.shape).reshape(-1)
        b = selected[neighbors.reshape(-1)]
        d = np.sqrt(np.maximum(squared_distance.reshape(-1), 0))
        valid = (a < b) & (ids[a] == ids[b])
        valid &= d <= distance_factor*np.minimum(mean_scale[a], mean_scale[b])
        a, b, d = a[valid], b[valid], d[valid]
        if not len(a):
            continue
        ds = np.abs(logs[a]-logs[b]).max(axis=1)
        dh = np.sqrt(np.square(sh[a]-sh[b]).mean(axis=1))
        valid = (ds <= log_limit) & (dh <= sh_threshold)
        pair_parts.append(np.stack((a[valid], b[valid]), axis=1))
        distance_parts.append(d[valid]/np.maximum(np.minimum(mean_scale[a[valid]], mean_scale[b[valid]]), 1e-12)
                              +ds[valid]+dh[valid])
    del index
    if not pair_parts:
        return np.empty((0, 2), dtype=np.int64)
    pairs = np.concatenate(pair_parts)
    distance = np.concatenate(distance_parts)
    order = np.lexsort((pairs[:, 1], pairs[:, 0], distance))
    used = np.zeros(n, dtype=bool)
    counts = dict(zip(*np.unique(ids[ids >= 0], return_counts=True)))
    floors = {} if instance_min_counts is None else instance_min_counts
    chosen = []
    for p in order:
        a, b = pairs[p]
        instance = int(ids[a])
        if used[a] or used[b] or counts[instance]-1 < floors.get(instance, 0):
            continue
        chosen.append((int(a), int(b)))
        used[a] = used[b] = True
        counts[instance] -= 1
        if len(chosen) >= max_pairs:
            break
    return np.asarray(chosen, dtype=np.int64).reshape(-1, 2)


class CompressionState:
    """Central owner of aligned model mutations and recent CPU removed pool."""
    def __init__(self, model, metadata, pool_rounds=2):
        self.model = model
        n = len(model._xyz)
        self.metadata = {key: _array(value).copy() for key, value in metadata.items()}
        defaults = {"instance_ids": np.full(n, -1, dtype=np.int64),
                    "confidence": np.zeros(n, dtype=np.float32),
                    "original_ids": np.arange(n, dtype=np.int64),
                    "row_ids": np.arange(n, dtype=np.int64),
                    "lineage_count": np.ones(n, dtype=np.int64),
                    "boundary_mask": np.zeros(n, dtype=bool),
                    "topology_critical": np.zeros(n, dtype=bool),
                    "visibility": np.ones(n, dtype=np.float32),
                    "structure_risk": np.zeros(n, dtype=np.float32)}
        for key, value in defaults.items():
            self.metadata.setdefault(key, value)
        for key in ("instance_ids", "original_ids", "row_ids", "lineage_count"):
            self.metadata[key] = self.metadata[key].astype(np.int64)
        self.pool_rounds = max(0, int(pool_rounds))
        self.removed_pool = []
        self.next_row_id = int(self.metadata["row_ids"].max())+1 if n else 0
        self.lineage = []
        self.events = []
        self.audit()

    @property
    def count(self):
        return len(self.model._xyz)

    def _params(self):
        return {key: getattr(self.model, attr) for key, attr in PARAMETERS.items()}

    def expire_pool(self, round_id):
        self.removed_pool = [chunk for chunk in self.removed_pool
                             if chunk["round_id"] >= int(round_id)-self.pool_rounds+1 and chunk["active"].any()]

    @torch.no_grad()
    def remove(self, prune_mask, round_id, scores=None, pool=True):
        mask = _array(prune_mask)
        if mask.dtype != np.bool_ or mask.shape != (self.count,):
            raise ValueError("remove expects a boolean prune mask of current length")
        self.expire_pool(round_id)
        n_remove = int(mask.sum())
        if not n_remove:
            return 0
        if scores is not None and _array(scores).reshape(-1).shape != (self.count,):
            raise ValueError("Pruning score length mismatch")
        device_mask = torch.as_tensor(mask, dtype=torch.bool, device=self.model._xyz.device)
        if pool and self.pool_rounds:
            self.removed_pool.append({
                "params": {key: value[device_mask].detach().cpu().clone() for key, value in self._params().items()},
                "metadata": {key: value[mask].copy() for key, value in self.metadata.items()},
                "original_score": np.zeros(n_remove, dtype=np.float32) if scores is None else _array(scores).reshape(-1)[mask].copy(),
                "round_id": int(round_id), "active": np.ones(n_remove, dtype=bool)})
        self.model.prune_points(device_mask)
        self.metadata = {key: value[~mask] for key, value in self.metadata.items()}
        self.events.append({"type": "remove", "round_id": int(round_id), "count": n_remove, "pooled": bool(pool)})
        self.audit()
        return n_remove

    @torch.no_grad()
    def append(self, params, metadata, reason="restore"):
        if set(params) != set(PARAMETERS):
            raise ValueError("append requires all six native parameter tensors")
        if set(metadata) != set(self.metadata):
            raise ValueError("append metadata fields must match active state")
        n = len(params["xyz"])
        converted = {}
        for key, current in self._params().items():
            value = torch.as_tensor(params[key], dtype=current.dtype, device=current.device)
            if value.shape != (n,)+current.shape[1:] or not torch.isfinite(value).all().item():
                raise ValueError("Invalid append parameter: " + key)
            converted[key] = value.detach()
        meta = {key: _array(value).astype(self.metadata[key].dtype, copy=True) for key, value in metadata.items()}
        for key, value in meta.items():
            if value.shape != (n,)+self.metadata[key].shape[1:]:
                raise ValueError("Invalid append metadata shape: " + key)
        if len(np.unique(meta["row_ids"])) != n or np.intersect1d(self.metadata["row_ids"], meta["row_ids"]).size:
            raise ValueError("append would duplicate an active lineage row")
        if n == 0:
            return 0
        tensors = self.model.cat_tensors_to_optimizer(converted)
        tensors["features_dc"] = tensors.pop("f_dc")
        tensors["features_rest"] = tensors.pop("f_rest")
        self.model._set_optimizable_tensors(tensors)
        for key in ("xyz_gradient_accum", "denom", "max_radii2D", "tmp_radii"):
            current = getattr(self.model, key, None)
            if current is not None:
                setattr(self.model, key, torch.cat((current, current.new_zeros((n,)+current.shape[1:])), dim=0))
        self.metadata = {key: np.concatenate((value, meta[key]), axis=0) for key, value in self.metadata.items()}
        self.next_row_id = max(self.next_row_id, int(meta["row_ids"].max())+1)
        self.events.append({"type": "append", "reason": str(reason), "count": n})
        self.audit()
        return n

    def pool_arrays(self, round_id=None):
        """Snapshot of live pool rows. Indices refer to this concatenation order.

        Snapshot keys: params (CPU torch), metadata (NumPy), original_score and
        round_id (NumPy). Call restore before another removal/expiry changes it.
        Rows with original_ids=-1 lack teacher sparse support and cannot restore.
        """
        if round_id is not None:
            self.expire_pool(round_id)
        chunks = [chunk for chunk in self.removed_pool if chunk["active"].any()]
        if not chunks:
            return {"params": {key: value[:0].detach().cpu() for key, value in self._params().items()},
                    "metadata": {key: value[:0].copy() for key, value in self.metadata.items()},
                    "original_score": np.empty(0, dtype=np.float32), "round_id": np.empty(0, dtype=np.int64)}
        return {"params": {key: torch.cat([chunk["params"][key][torch.from_numpy(chunk["active"])] for chunk in chunks]) for key in PARAMETERS},
                "metadata": {key: np.concatenate([chunk["metadata"][key][chunk["active"]] for chunk in chunks]) for key in self.metadata},
                "original_score": np.concatenate([chunk["original_score"][chunk["active"]] for chunk in chunks]),
                "round_id": np.concatenate([np.full(int(chunk["active"].sum()), chunk["round_id"], dtype=np.int64) for chunk in chunks])}

    def restore(self, pool_indices, round_id, target_count):
        """Restore ranked live-pool indices up to available target headroom.

        New/merged rows (original_ids == -1) are skipped: teacher-based sparse
        overlap cannot score their changed support. No original-score fallback.
        """
        self.expire_pool(round_id)
        snapshot = self.pool_arrays()
        indices = np.asarray(pool_indices, dtype=np.int64).reshape(-1)
        n_pool = len(snapshot["original_score"])
        if np.any(indices < 0) or np.any(indices >= n_pool) or len(np.unique(indices)) != len(indices):
            raise ValueError("Invalid or duplicate restoration pool indices")
        indices = indices[snapshot["metadata"]["original_ids"][indices] >= 0]
        indices = indices[:max(0, int(target_count)-self.count)]
        if not len(indices):
            return 0
        restored = self.append({key: value[torch.from_numpy(indices)] for key, value in snapshot["params"].items()},
                               {key: value[indices] for key, value in snapshot["metadata"].items()}, reason="hard_region_restore")
        chosen = np.zeros(n_pool, dtype=bool)
        chosen[indices] = True
        offset = 0
        for chunk in self.removed_pool:
            live = np.flatnonzero(chunk["active"])
            chunk["active"][live[chosen[offset:offset+len(live)]]] = False
            offset += len(live)
        self.events.append({"type": "restore", "round_id": int(round_id), "count": restored})
        return restored

    @torch.no_grad()
    def merge(self, pairs, round_id, confidence_threshold=0.5, preserve_boundary=True):
        pairs = np.asarray(pairs, dtype=np.int64).reshape(-1, 2)
        if not len(pairs):
            return 0
        if np.any(pairs < 0) or np.any(pairs >= self.count) or np.unique(pairs).size != pairs.size:
            raise ValueError("Merge pairs must be disjoint valid current indices")
        a, b = pairs.T
        m = self.metadata
        if (np.any(m["instance_ids"][a] < 0) or np.any(m["instance_ids"][a] != m["instance_ids"][b])
                or np.any(m["confidence"][pairs] < confidence_threshold)
                or (preserve_boundary and np.any(m["boundary_mask"][pairs]))
                or np.any(m["topology_critical"][pairs])):
            raise ValueError("Unsafe cross-instance, ambiguous or protected merge")
        params, weights = moment_match_pairs(self._params(), pairs, m["visibility"])
        merged = {}
        for key, value in m.items():
            if key in CONTINUOUS_BOUNDARY_FIELDS:
                # Use the same opacity*visibility weights, with a double
                # precision weighted mean. Float32 normalized weights can sum
                # to 1+1ULP and otherwise move two saturated strengths above 1.
                w = weights.astype(np.float64)
                merged[key] = ((w[:, 0]*value[a].astype(np.float64)+w[:, 1]*value[b].astype(np.float64))
                               / w.sum(axis=1)).astype(value.dtype)
            elif value.dtype == np.bool_:
                merged[key] = value[a] | value[b]
            elif np.issubdtype(value.dtype, np.floating):
                wshape = (-1,)+(1,)*(value.ndim-1)
                merged[key] = weights[:, 0].reshape(wshape)*value[a]+weights[:, 1].reshape(wshape)*value[b]
            else:
                merged[key] = value[a].copy()
        merged["confidence"] = np.minimum(m["confidence"][a], m["confidence"][b])
        merged["structure_risk"] = np.maximum(m["structure_risk"][a], m["structure_risk"][b])
        merged["row_ids"] = np.arange(self.next_row_id, self.next_row_id+len(pairs), dtype=np.int64)
        merged["original_ids"] = np.full(len(pairs), -1, dtype=np.int64)
        merged["lineage_count"] = m["lineage_count"][a]+m["lineage_count"][b]
        lineage = [{"row_id": int(merged["row_ids"][k]), "parent_row_ids": [int(m["row_ids"][i]), int(m["row_ids"][j])],
                    "parent_original_ids": [int(m["original_ids"][i]), int(m["original_ids"][j])], "round_id": int(round_id)}
                   for k, (i, j) in enumerate(pairs)]
        if all(key in m for key in CONTINUOUS_BOUNDARY_FIELDS):
            # Record the same opacity*visibility mass weights used above so the
            # three continuous channels can be independently replayed exactly.
            for record, weight in zip(lineage, weights):
                record['merge_weights'] = weight.tolist()
        mask = np.zeros(self.count, dtype=bool)
        mask[pairs.reshape(-1)] = True
        self.remove(mask, round_id, pool=False)
        self.append(params, merged, reason="moment_matched_merge")
        self.lineage.extend(lineage)
        self.events.append({"type": "merge", "round_id": int(round_id), "pairs": len(pairs)})
        return len(pairs)

    def audit(self):
        n = self.count
        for key, value in self.metadata.items():
            if value.ndim == 0 or len(value) != n:
                raise ValueError("Metadata/model alignment failure: " + key)
        continuous = [key in self.metadata for key in CONTINUOUS_BOUNDARY_FIELDS]
        if any(continuous):
            if not all(continuous):
                raise ValueError('Continuous boundary metadata must contain all three modalities')
            for key in CONTINUOUS_BOUNDARY_FIELDS:
                value = self.metadata[key]
                if (value.shape != (n,) or not np.issubdtype(value.dtype, np.floating)
                        or not np.isfinite(value).all() or np.any(value < 0) or np.any(value > 1)):
                    raise ValueError('Invalid aligned continuous boundary metadata: ' + key)
        if len(np.unique(self.metadata["row_ids"])) != n:
            raise ValueError("Duplicate active lineage row IDs")
        for key, value in self._params().items():
            if len(value) != n:
                raise ValueError("Native parameter alignment failure: " + key)
        for key in ("xyz_gradient_accum", "denom", "max_radii2D", "tmp_radii"):
            value = getattr(self.model, key, None)
            if value is not None and len(value) != n:
                raise ValueError("Native accumulator alignment failure: " + key)
        optimizer = getattr(self.model, "optimizer", None)
        if optimizer is not None:
            for group in optimizer.param_groups:
                name = group.get("name")
                if name not in PARAMETERS:
                    continue
                param = group["params"][0]
                if param is not getattr(self.model, PARAMETERS[name]):
                    raise ValueError("Optimizer parameter alias failure: " + name)
                for key in ("exp_avg", "exp_avg_sq"):
                    value = optimizer.state.get(param, {}).get(key)
                    if value is not None and value.shape != param.shape:
                        raise ValueError("Optimizer momentum alignment failure: " + name)
        return {"count": n, "teacher_lineage_rows": int((self.metadata["original_ids"] >= 0).sum()),
                "merged_rows": int((self.metadata["original_ids"] < 0).sum()),
                "pool_rows": sum(int(chunk["active"].sum()) for chunk in self.removed_pool),
                "next_row_id": self.next_row_id}

    def save(self, path, extra=None):
        """Auxiliary compression state; native model checkpoint saved separately."""
        payload = {"format_version": 1, "metadata": self.metadata, "removed_pool": self.removed_pool,
                   "pool_rounds": self.pool_rounds, "next_row_id": self.next_row_id,
                   "lineage": self.lineage, "events": self.events, "audit": self.audit(),
                   "extra": {} if extra is None else extra}
        parent = os.path.dirname(os.path.abspath(path))
        os.makedirs(parent, exist_ok=True)
        descriptor, temporary = tempfile.mkstemp(prefix=".compression_state_", suffix=".tmp", dir=parent)
        os.close(descriptor)
        try:
            torch.save(payload, temporary)
            os.replace(temporary, path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)

    @classmethod
    def load(cls, model, path):
        payload = torch.load(path, map_location="cpu")
        if payload.get("format_version") != 1:
            raise ValueError("Unsupported compression state version")
        state = cls(model, payload["metadata"], payload["pool_rounds"])
        state.removed_pool = payload["removed_pool"]
        state.next_row_id = int(payload["next_row_id"])
        state.lineage = payload["lineage"]
        state.events = payload["events"]
        state.audit()
        return state, payload.get("extra", {})
