"""Exact lineage-based retention measurements; NumPy only, no renderer needed.

An instance that retains points can still disappear visually. These measurements
are structural diagnostics, not evidence of perceptual quality on their own.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


def load_metadata(path, *, require_original_ids=False):
    """Read the two required arrays, accepting old prior field spellings.

    A source prior is ordered like its original PLY, so its implicit IDs may be
    arange(N). A compressed model MUST store explicit IDs; XYZ matching would be
    unreliable after position optimization and is deliberately unsupported here.
    """
    with np.load(path, allow_pickle=False) as z:
        data = {key: np.asarray(z[key]) for key in z.files}
    instance_key = next((k for k in ("instance_ids", "instance_id") if k in data), None)
    if instance_key is None:
        raise ValueError(f"{path}: no instance_ids array")
    ids = data[instance_key]
    if ids.ndim != 1 or not np.issubdtype(ids.dtype, np.integer):
        raise ValueError("instance IDs must be a one-dimensional integer array")
    data["instance_ids"] = np.maximum(ids.astype(np.int64), 0)
    original_key = next((k for k in ("original_ids", "original_id") if k in data), None)
    if original_key is None:
        if require_original_ids:
            raise ValueError("compressed metadata requires stable original_ids; geometry matching is not exact")
        data["original_ids"] = np.arange(len(ids), dtype=np.int64)
        data["original_ids_implicit"] = True
    else:
        original_ids = data[original_key]
        if original_ids.shape != ids.shape or not np.issubdtype(original_ids.dtype, np.integer):
            raise ValueError("original_ids must have the same shape and integer dtype as instance_ids")
        if np.any(original_ids < 0):
            raise ValueError("original_ids must be nonnegative")
        data["original_ids"] = original_ids.astype(np.int64)
        data["original_ids_implicit"] = False
    if len(np.unique(data["original_ids"])) != len(ids):
        raise ValueError("original_ids must be unique; a pruning-only lineage cannot duplicate Gaussians")
    return data


def _records(value):
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, dict) and isinstance(value.get("instances"), list):
        return value["instances"]
    raise ValueError("catalog/policy must be a list or a JSON object containing instances: [...] ")


def projected_area_statistics(mask_tracks, train_frames, image_shapes=None):
    """Summarize SAM pixel areas using TRAINING frames only.

    Nodes from one track can contain multiple regions in one frame; sum their
    disjoint region-map areas before taking the median across views. Never use
    held-out frames to choose training protection quotas.
    """
    train_frames = set(train_frames)
    if not train_frames:
        raise ValueError("an explicit nonempty training-frame list is required")
    nodes = mask_tracks.get("nodes", [])
    grouped = {}
    excluded_frames = set()
    for node in nodes:
        frame = str(node.get("frame", ""))
        if frame not in train_frames:
            excluded_frames.add(frame)
            continue
        iid = int(node.get("instance_id", 0))
        if iid <= 0:
            continue
        area = float(node["area"])
        if not np.isfinite(area) or area < 0:
            raise ValueError("SAM region areas must be finite and nonnegative")
        frame_values = grouped.setdefault(iid, {})
        frame_values[frame] = frame_values.get(frame, 0.) + area
    result = {}
    for iid, per_view in grouped.items():
        areas = list(per_view.values())
        fractions = []
        if image_shapes is not None:
            for frame, area in per_view.items():
                height, width = image_shapes[frame]
                if height <= 0 or width <= 0:
                    raise ValueError("image dimensions must be positive")
                if area > height * width + 1:
                    raise ValueError("summed region area exceeds the image area")
                fractions.append(area / (height * width))
        result[iid] = {
            "training_view_count": len(per_view),
            "median_projected_pixels": float(np.median(areas)),
            "max_projected_pixels": float(max(areas)),
            "median_projected_area_fraction": float(np.median(fractions)) if fractions else None,
        }
    return result, sorted(excluded_frames)


def retention_report(original, compressed, *, catalog=None, policy=None, projected_areas=None):
    """Check stable subset identity and report point/instance retention exactly."""
    original_ids = original["original_ids"]
    final_ids = compressed["original_ids"]
    source_instances = original["instance_ids"]
    final_instances = compressed["instance_ids"]
    if len(np.unique(original_ids)) != len(original_ids) or len(np.unique(final_ids)) != len(final_ids):
        raise ValueError("duplicate original_ids in metadata")
    order = np.argsort(original_ids)
    sorted_ids = original_ids[order]
    positions = np.searchsorted(sorted_ids, final_ids)
    valid = positions < len(sorted_ids)
    valid[valid] &= sorted_ids[positions[valid]] == final_ids[valid]
    if not np.all(valid):
        raise ValueError(f"{int((~valid).sum())} compressed original_ids are absent from source metadata")
    expected = source_instances[order[positions]]
    if not np.array_equal(expected, final_instances):
        raise ValueError(f"instance identity changed for {int((expected != final_instances).sum())} surviving Gaussians")
    source_keys, source_counts = np.unique(source_instances, return_counts=True)
    final_keys, final_counts = np.unique(final_instances, return_counts=True)
    before = dict(zip(map(int, source_keys), map(int, source_counts)))
    after = dict(zip(map(int, final_keys), map(int, final_counts)))
    catalog_by_id = {int(r["instance_id"]): r for r in _records(catalog)}
    policy_by_id = {int(r["instance_id"]): r for r in _records(policy)}
    for iid, p in policy_by_id.items():
        if iid > 0 and iid not in before:
            raise ValueError(f"policy instance {iid} not found in original metadata")
        floor = p.get("min_keep", 0)
        if isinstance(floor, bool) or int(floor) != floor or floor < 0 or floor > before.get(iid, 0):
            raise ValueError(f"invalid min_keep for instance {iid}")
    rows = []
    for iid in sorted(k for k in before if k > 0):
        p = policy_by_id.get(iid, {})
        c = catalog_by_id.get(iid, {})
        floor = int(p.get("min_keep", 0))
        row = dict(instance_id=iid, original_count=before[iid], final_count=after.get(iid, 0),
                   retention_fraction=after.get(iid, 0) / before[iid],
                   extinct=after.get(iid, 0) == 0, min_keep=floor,
                   floor_violation=after.get(iid, 0) < floor,
                   protected=bool(p.get("protected", floor > 0)),
                   small=bool(p.get("small", c.get("small", False))),
                   reliable=bool(p.get("reliable", c.get("reliable", False))))
        for key in ("direct_count", "multiview_direct_count", "mean_direct_confidence", "smallness"):
            if key in p:
                row[key] = p[key]
            elif key in c:
                row[key] = c[key]
        if projected_areas and iid in projected_areas:
            row.update(projected_areas[iid])
        rows.append(row)

    def summarize(selected):
        counts = [r["retention_fraction"] for r in selected]
        return dict(instance_count=len(selected),
                    extinct_count=sum(r["extinct"] for r in selected),
                    extinction_fraction=sum(r["extinct"] for r in selected) / len(selected) if selected else None,
                    floor_violation_count=sum(r["floor_violation"] for r in selected),
                    original_gaussians=sum(r["original_count"] for r in selected),
                    final_gaussians=sum(r["final_count"] for r in selected),
                    median_retention_fraction=float(np.median(counts)) if counts else None,
                    min_retention_fraction=float(min(counts)) if counts else None)

    return dict(schema_version=1,
                measurement="exact_stable_original_id_subset",
                checks={"unique_original_ids": True, "all_survivors_in_original": True,
                        "instance_identity_preserved": True},
                gaussian_counts={"original": len(original_ids), "final": len(final_ids),
                                 "unresolved_original": before.get(0, 0), "unresolved_final": after.get(0, 0),
                                 "retention_fraction": len(final_ids) / len(original_ids) if len(original_ids) else None},
                groups={"all_instances": summarize(rows),
                        "protected": summarize([r for r in rows if r["protected"]]),
                        "policy_small": summarize([r for r in rows if r["small"]]),
                        "protected_small": summarize([r for r in rows if r["small"] and r["protected"]])},
                instances=rows,
                limitations=["Retaining Gaussians does not prove rendered object visibility, boundary quality, or appearance.",
                             "Instance IDs are automatic class-agnostic region pseudo-labels, not manually verified physical objects.",
                             "No XYZ nearest-neighbor fallback is used; optimized centers cannot establish exact historical identity."])


def inspect_standard_ply(path, expected_sh_degree=None):
    """Read only the header; identify SH format and forbidden DoG extensions."""
    properties = []
    vertex_count = None
    in_vertex = False
    with open(path, "rb") as f:
        for _ in range(10000):
            line = f.readline()
            if not line:
                raise ValueError("incomplete PLY header")
            text = line.decode("ascii").strip()
            if text.startswith("element "):
                parts = text.split()
                in_vertex = parts[1] == "vertex"
                if in_vertex:
                    vertex_count = int(parts[2])
            elif text.startswith("property ") and in_vertex:
                properties.append(text.split()[-1])
            if text == "end_header":
                break
        else:
            raise ValueError("PLY header exceeds 10000 lines")
    rest = sorted([name for name in properties if name.startswith("f_rest_")],
                  key=lambda name: int(name[7:]))
    sh_coefficients = len(rest) / 3 + 1
    degree = int(np.sqrt(sh_coefficients) - 1)
    valid_sh = ((degree + 1) ** 2 == sh_coefficients and
                rest == [f"f_rest_{i}" for i in range(len(rest))])
    forbidden = [name for name in properties if "negative" in name.lower() or "dog" in name.lower()]
    standard = (["x", "y", "z", "nx", "ny", "nz"] + [f"f_dc_{i}" for i in range(3)] + rest +
                ["opacity"] + [f"scale_{i}" for i in range(3)] + [f"rot_{i}" for i in range(4)])
    extensions = [p for p in properties if p not in standard]
    return dict(path=str(Path(path).resolve()), vertex_count=vertex_count,
                property_count=len(properties), sh_rest_property_count=len(rest),
                inferred_sh_degree=degree if valid_sh else None,
                valid_sh_layout=valid_sh,
                expected_sh_degree=expected_sh_degree,
                expected_sh_matches=expected_sh_degree is None or (valid_sh and degree == expected_sh_degree),
                missing_standard_properties=[p for p in standard if p not in properties],
                extra_properties=extensions, forbidden_dog_properties=forbidden,
                no_dog_properties=not forbidden)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())
