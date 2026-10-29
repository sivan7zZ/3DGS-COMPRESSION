"""Immutable, training-visible image coverage certificates for exact-K SPS.

An anchor is a source Gaussian with measured alpha*T image contribution.  The
certificate is about source image support, not a claim that an optimized atom
remains visible.  Recovery losses and post-training rendered checks must verify
the latter separately.
"""
import hashlib
import json

import numpy as np

from utils.instance_pruning import xyz_sha256, _ranks


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def greedy_coverage(candidates, offsets, strength, budget=None):
    """Deterministic weighted set cover with an explicit infeasibility result.

    candidates is a flattened source-ID vector; each CSR row is one critical
    screen cell. strength gives the visible alpha*T mass for every candidate.
    The returned certificate covers every row. No groups are silently dropped.
    """
    import heapq
    candidates = np.asarray(candidates, dtype=np.int64)
    offsets = np.asarray(offsets, dtype=np.int64)
    strength = np.asarray(strength, dtype=np.float64)
    if offsets.ndim != 1 or len(offsets) < 1 or offsets[0] != 0 or offsets[-1] != len(candidates):
        raise ValueError("Invalid coverage CSR offsets")
    if np.any(np.diff(offsets) <= 0) or strength.shape != candidates.shape:
        raise ValueError("Every critical screen cell needs at least one candidate")
    if np.any(candidates < 0) or not np.isfinite(strength).all() or np.any(strength <= 0):
        raise ValueError("Invalid visible candidate evidence")
    groups = len(offsets) - 1
    reverse = {}
    utility = {}
    for group in range(groups):
        left, right = offsets[group:group+2]
        normalization = max(float(strength[left:right].max()), 1e-12)
        for source_id, value in zip(candidates[left:right], strength[left:right]):
            source_id = int(source_id)
            reverse.setdefault(source_id, []).append(group)
            utility[source_id] = utility.get(source_id, 0.0) + .25 + .75 * float(value) / normalization
    covered = np.zeros(groups, dtype=bool)
    heap = [(-value, source_id) for source_id, value in utility.items()]
    heapq.heapify(heap)
    selected = []
    while not covered.all():
        if not heap:
            raise AssertionError("Coverage graph is disconnected from its candidates")
        _, source_id = heapq.heappop(heap)
        adjacent = np.asarray(reverse[source_id], dtype=np.int64)
        live = adjacent[~covered[adjacent]]
        if not len(live):
            continue
        value = 0.0
        for group in live:
            left, right = offsets[group:group+2]
            edge = np.flatnonzero(candidates[left:right] == source_id)[0] + left
            value += .25 + .75 * strength[edge] / strength[left:right].max()
        # Lazy heap bounds are valid because covering cells only decreases gain.
        if heap and value < -heap[0][0] - 1e-12:
            heapq.heappush(heap, (-value, source_id))
            continue
        selected.append(source_id)
        covered[adjacent] = True
        if budget is not None and len(selected) > int(budget):
            raise ValueError("Critical screen coverage needs more than the explicit anchor budget %d; choose a different declared cell policy or a larger final model" % int(budget))
    return np.sort(np.asarray(selected, dtype=np.int64))


class StructurePruningState:
    @classmethod
    def load(cls, path, source_xyz=None):
        self = cls()
        with np.load(path, allow_pickle=False) as data:
            if str(data["format"].item()) != "pwsr-structure-v1":
                raise ValueError("Unsupported structure coverage certificate")
            self.source_xyz_sha256 = str(data["source_xyz_sha256"].item())
            for name in ("anchors", "anchor_xyz", "anchor_log_scaling", "anchor_opacity",
                         "anchor_features_dc", "anchor_strength", "candidates", "offsets", "group_kind"):
                setattr(self, name, np.array(data[name]))
            self.group_strength = np.array(data["group_strength"]) if "group_strength" in data else None
            self.group_total_strength = np.array(data["group_total_strength"]) if "group_total_strength" in data else None
            self.manifest = json.loads(str(data["manifest_json"].item()))
        if source_xyz is not None and xyz_sha256(source_xyz) != self.source_xyz_sha256:
            raise ValueError("Structure priors source XYZ does not match the unpruned model")
        if len(np.unique(self.anchors)) != len(self.anchors) or np.any(self.anchors < 0):
            raise ValueError("Invalid duplicate/negative structure anchors")
        if source_xyz is not None:
            original = source_xyz.detach().cpu().numpy() if hasattr(source_xyz, "detach") else np.asarray(source_xyz)
            if np.any(self.anchors >= len(original)) or not np.array_equal(self.anchor_xyz, original[self.anchors].astype(np.float32)):
                raise ValueError("Stored structure anchor XYZ differs from the source model")
        size = len(self.anchors)
        for name, shape in (("anchor_xyz", (size, 3)), ("anchor_log_scaling", (size, 3)),
                            ("anchor_features_dc", (size, 3)), ("anchor_opacity", (size,)),
                            ("anchor_strength", (size,))):
            if getattr(self, name).shape != shape or not np.isfinite(getattr(self, name)).all():
                raise ValueError("Invalid " + name)
        if np.any((self.anchor_opacity < 0) | (self.anchor_opacity > 1)) or np.any((self.anchor_strength <= 0) | (self.anchor_strength > 1)):
            raise ValueError("Anchor opacity/strength is outside its probability range")
        self.provenance = dict(format="pwsr-structure-v1", sha256=file_sha256(path),
                               source_xyz_sha256=self.source_xyz_sha256,
                               anchors=size, groups=len(self.group_kind))
        self.audit(self.anchors, require_anchors=True)
        return self

    def audit(self, selected_original_ids, require_anchors=True):
        selected = np.asarray(selected_original_ids, dtype=np.int64)
        present = np.isin(self.candidates, selected)
        if len(self.offsets) != len(self.group_kind) + 1 or self.offsets[0] != 0 or self.offsets[-1] != len(present) or np.any(np.diff(self.offsets) <= 0):
            raise ValueError("Malformed critical coverage groups")
        covered = np.logical_or.reduceat(present, self.offsets[:-1])
        missing_anchors = self.anchors[~np.isin(self.anchors, selected)]
        result = dict(anchors=len(self.anchors), missing_anchors=int(len(missing_anchors)),
                      groups=len(covered), uncovered_groups=int((~covered).sum()), by_kind={})
        retained_mass = total_mass = None
        if self.group_strength is not None:
            retained_mass = np.add.reduceat(np.where(present, self.group_strength, 0), self.offsets[:-1])
            total_mass = self.group_total_strength if self.group_total_strength is not None else np.add.reduceat(self.group_strength, self.offsets[:-1])
            result["source_dominant_mass_coverage"] = float(retained_mass.sum() / max(float(total_mass.sum()), 1e-8))
            result["mass_coverage_scope"] = ("lower bound on retained source max(alpha*T) mass in critical cells" if self.group_total_strength is not None else "retained fraction within saved top-candidate source max(alpha*T) mass") + "; not optimized rendered visibility"
        for kind, label in ((1, "small_region_cells"), (2, "boundary_segments"), (3, "thin_segments")):
            take = self.group_kind == kind
            result["by_kind"][label] = dict(groups=int(take.sum()), uncovered=int(np.count_nonzero(take & ~covered)))
            if retained_mass is not None:
                result["by_kind"][label]["source_dominant_mass_coverage"] = float(retained_mass[take].sum() / max(float(total_mass[take].sum()), 1e-8))
        if require_anchors and (len(missing_anchors) or not covered.all()):
            raise ValueError("Required training-visible structure coverage was lost: " + json.dumps(result))
        return result

    def required_cost(self, instance_state):
        if instance_state.source_xyz_sha256 != self.source_xyz_sha256:
            raise ValueError("Structure and instance source fingerprints differ")
        # This also checks that an earlier prune did not lose a hard anchor.
        if not np.isin(self.anchors, instance_state.original_ids).all():
            raise ValueError("Required structure anchor missing from current instance lineage")
        ids = instance_state.instance_ids[np.isin(instance_state.original_ids, self.anchors)]
        unique, counts = np.unique(ids, return_counts=True)
        anchor_counts = dict(zip(unique.tolist(), counts.tolist()))
        return int(len(self.anchors) + sum(max(0, int(row["min_keep"]) - anchor_counts.get(row["instance_id"], 0)) for row in instance_state.policy))

    def audit_coverage(self, selected_original_ids, require_anchors=True):
        """Validate all cells even when replacement of old anchors is allowed."""
        result = self.audit(selected_original_ids, require_anchors=require_anchors)
        if result["uncovered_groups"]:
            raise ValueError("Required source-visible screen cells lost coverage: " + json.dumps(result))
        return result

    def select_keep(self, scores, original_ids, keep_count, instance_state=None, xyz=None):
        if instance_state is None or xyz is None:
            raise ValueError("Structure pruning requires aligned instance lineage and current XYZ")
        if not np.array_equal(np.asarray(original_ids), instance_state.original_ids):
            raise ValueError("Structure selection row order differs from instance lineage")
        keep, audit = instance_state.select_keep(scores, xyz, keep_count,
                                                 required_original_ids=self.anchors)
        mask = keep.detach().cpu().numpy() if hasattr(keep, "detach") else keep
        audit["selection_mode"] = "visible_structure_coverage_constrained_SPS"
        audit["structure_coverage"] = self.audit(instance_state.original_ids[mask])
        return keep, audit

    def select_keep_adaptive_final(self, scores, original_ids, keep_count, instance_state=None, xyz=None):
        """Keep ordinary instance-SPS whenever its screen support is sufficient.

        Intermediate rounds retain the fixed certificate so the final round
        has a known feasible fallback. At the declared final K, repair only
        uncovered cells, then discard the lowest-SPS removable atoms. An atom
        cannot be removed if it is a cell's last source-visible representative
        or its instance is already at its immutable floor.

        This is a deterministic greedy repair, not a globally optimal solution
        to the constrained ranking problem. The fixed certificate is used only
        if this local repair cannot attain exact K; that fallback is audited.
        """
        import heapq
        if instance_state is None or xyz is None:
            raise ValueError("Adaptive coverage requires instance lineage and current XYZ")
        ids = np.asarray(original_ids, dtype=np.int64)
        if not np.array_equal(ids, instance_state.original_ids):
            raise ValueError("Adaptive coverage Gaussian rows are misaligned")
        if self.source_xyz_sha256 != instance_state.source_xyz_sha256:
            raise ValueError("Adaptive coverage and instance source models differ")
        final_budget = int(self.manifest["final_budget"])
        if int(keep_count) != keep_count or keep_count < final_budget:
            raise ValueError("Adaptive pruning cannot go below the declared final coverage budget")
        if keep_count > final_budget:
            keep, audit = self.select_keep(scores, ids, keep_count, instance_state, xyz)
            audit["adaptive_final"] = dict(applied=False, reason="retain feasible certificate until final K")
            return keep, audit
        values = scores.detach().cpu().numpy() if hasattr(scores, "detach") else np.asarray(scores)
        values = np.asarray(values, dtype=np.float64).reshape(-1)
        if values.shape != ids.shape or not np.isfinite(values).all():
            raise ValueError("Adaptive SPS scores must be finite and model-aligned")
        initial, audit = instance_state.select_keep(scores, xyz, keep_count)
        initial = initial.detach().cpu().numpy() if hasattr(initial, "detach") else np.asarray(initial)
        keep = initial.copy()
        lookup = np.full(instance_state.original_count, -1, dtype=np.int64)
        lookup[ids] = np.arange(len(ids), dtype=np.int64)
        rows = lookup[self.candidates]
        available = rows >= 0
        lengths = np.diff(self.offsets)
        group_index = np.repeat(np.arange(len(lengths), dtype=np.int64), lengths)
        present = np.zeros(len(rows), dtype=bool)
        present[available] = keep[rows[available]]
        cover_count = np.add.reduceat(present.astype(np.int32), self.offsets[:-1])
        uncovered = cover_count == 0
        initial_uncovered = int(uncovered.sum())
        # Reverse CSR: current Gaussian row -> all critical cells it can cover.
        edges = np.flatnonzero(available)
        order = np.argsort(rows[edges], kind="mergesort")
        row_edges = edges[order]
        row_offsets = np.r_[0, np.cumsum(np.bincount(rows[edges], minlength=len(ids)))]
        adjacency = group_index[row_edges]
        initial_uncovered_edges = available & uncovered[group_index]
        benefit = np.bincount(rows[initial_uncovered_edges], minlength=len(ids)).astype(np.float64)
        quality = 1. + .25 * _ranks(values)
        active = np.flatnonzero((benefit > 0) & ~keep)
        heap = [(-benefit[row] * quality[row], -values[row], int(ids[row]), int(row)) for row in active]
        heapq.heapify(heap)
        repaired = 0
        uncovered_count = initial_uncovered
        failure = None
        while uncovered_count:
            if not heap:
                failure = "available source candidates cannot cover all critical cells"
                break
            _, _, _, row = heapq.heappop(heap)
            if keep[row]:
                continue
            groups = adjacency[row_offsets[row]:row_offsets[row + 1]]
            live = groups[uncovered[groups]]
            if not len(live):
                continue
            current = len(live) * quality[row]
            if heap and current < -heap[0][0] - 1e-12:
                heapq.heappush(heap, (-current, -values[row], int(ids[row]), int(row)))
                continue
            keep[row] = True
            cover_count[groups] += 1
            uncovered[live] = False
            uncovered_count -= len(live)
            repaired += 1
        policy_maximum = max([int(row["instance_id"]) for row in instance_state.policy] + [0])
        minimum = np.zeros(max(int(instance_state.instance_ids.max()), policy_maximum) + 1, dtype=np.int64)
        for policy in instance_state.policy:
            minimum[int(policy["instance_id"])] = int(policy["min_keep"])
        counts = np.bincount(instance_state.instance_ids[keep], minlength=len(minimum))
        if failure is None:
            excess = int(keep.sum()) - int(keep_count)
            selected = np.flatnonzero(keep)
            removal_order = selected[np.lexsort((ids[selected], values[selected]))]
            for row in removal_order:
                if not excess:
                    break
                instance_id = int(instance_state.instance_ids[row])
                if counts[instance_id] <= minimum[instance_id]:
                    continue
                groups = adjacency[row_offsets[row]:row_offsets[row + 1]]
                if len(groups) and np.any(cover_count[groups] <= 1):
                    continue
                keep[row] = False
                cover_count[groups] -= 1
                counts[instance_id] -= 1
                excess -= 1
            if excess:
                failure = "local greedy repair could not remove %d excess atoms without violating coverage/floors" % excess
        fallback = failure is not None
        if fallback:
            fixed, audit = self.select_keep(scores, ids, keep_count, instance_state, xyz)
            keep = fixed.detach().cpu().numpy() if hasattr(fixed, "detach") else np.asarray(fixed)
        coverage = self.audit(ids[keep], require_anchors=False)
        if int(keep.sum()) != int(keep_count) or coverage["uncovered_groups"]:
            raise AssertionError("Adaptive exact-K image-cell coverage invariant failed")
        for policy_record in audit["instances"]:
            instance_id = int(policy_record["instance_id"])
            after = int(np.count_nonzero(keep & (instance_state.instance_ids == instance_id)))
            policy_record.update(after=after, lost=after == 0,
                                 floor_violation=after < int(policy_record["min_keep"]))
            if policy_record["floor_violation"]:
                raise AssertionError("Adaptive selection violated an immutable instance floor")
        audit["kept"] = int(keep.sum())
        audit["unknown_after"] = int(np.count_nonzero(keep & (instance_state.instance_ids == 0)))
        audit["selection_mode"] = "adaptive_final_visible_cell_repair_SPS"
        audit["structure_coverage"] = coverage
        audit["adaptive_final"] = dict(applied=True, initial_uncovered_groups=initial_uncovered,
                                       greedy_rescue_additions=repaired,
                                       added_to_instance_SPS=int(np.count_nonzero(keep & ~initial)),
                                       removed_from_instance_SPS=int(np.count_nonzero(initial & ~keep)),
                                       fixed_certificate_fallback=fallback, fallback_reason=failure,
                                       original_fixed_anchors_retained=int(np.isin(self.anchors, ids[keep]).sum()),
                                       coverage_scope="source-visible candidate-cell support, not a guarantee of optimized RGB visibility")
        if hasattr(scores, "detach"):
            import torch
            keep = torch.as_tensor(keep, dtype=torch.bool, device=scores.device)
        return keep, audit
