#!/usr/bin/env python3
"""Audit per-instance opacity-threshold point counts using exact row lineage.

Opacity alone is a weak visibility proxy. A high-opacity Gaussian can be fully
occluded, outside a view, or have little rendered contribution. This script does
not reinterpret raw-count protection floors as active-opacity guarantees.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from utils.instance_metrics import load_metadata, read_json, retention_report


PLY_TYPES = {"char": "i1", "int8": "i1", "uchar": "u1", "uint8": "u1",
             "short": "i2", "int16": "i2", "ushort": "u2", "uint16": "u2",
             "int": "i4", "int32": "i4", "uint": "u4", "uint32": "u4",
             "float": "f4", "float32": "f4", "double": "f8", "float64": "f8"}


def read_ply_vertices(path):
    """Memory-map scalar vertex properties of standard binary 3DGS PLY."""
    fields = []
    vertex_count = None
    current_element = None
    first_element = None
    byteorder = None
    with open(path, "rb") as stream:
        if stream.readline().strip() != b"ply":
            raise ValueError("not a PLY file")
        for _ in range(10000):
            line = stream.readline()
            if not line:
                raise ValueError("incomplete PLY header")
            parts = line.decode("ascii").strip().split()
            if not parts:
                continue
            if parts[0] == "format":
                byteorder = {"binary_little_endian": "<", "binary_big_endian": ">"}.get(parts[1])
                if byteorder is None:
                    raise ValueError("standard binary PLY required; ASCII is unsupported")
            elif parts[0] == "element":
                current_element = parts[1]
                first_element = first_element or current_element
                if current_element == "vertex":
                    vertex_count = int(parts[2])
            elif parts[0] == "property" and current_element == "vertex":
                if byteorder is None:
                    raise ValueError("PLY format must be declared before vertex properties")
                if len(parts) != 3 or parts[1] not in PLY_TYPES:
                    raise ValueError("only scalar standard vertex properties are supported")
                fields.append((parts[2], byteorder + PLY_TYPES[parts[1]]))
            elif parts[0] == "end_header":
                offset = stream.tell()
                break
        else:
            raise ValueError("PLY header exceeds 10000 lines")
    if first_element != "vertex" or vertex_count is None or vertex_count < 0 or byteorder is None:
        raise ValueError("standard PLY must begin with a vertex element")
    dtype = np.dtype(fields)
    if Path(path).stat().st_size < offset + vertex_count * dtype.itemsize:
        raise ValueError("PLY vertex data is truncated")
    if vertex_count == 0:
        return np.empty(0, dtype=dtype)
    return np.memmap(path, mode="r", dtype=dtype, offset=offset, shape=(vertex_count,))


def xyz_hash(vertices):
    xyz = np.ascontiguousarray(np.column_stack([vertices[k] for k in ("x", "y", "z")]), dtype="<f4")
    if not np.isfinite(xyz).all():
        raise ValueError("PLY coordinates must be finite")
    return hashlib.sha256(xyz.tobytes()).hexdigest()


def active_counts(instance_ids, opacity, threshold, opacity_space="logit"):
    if not 0 < threshold < 1:
        raise ValueError("opacity threshold must be in (0,1)")
    if len(instance_ids) != len(opacity):
        raise ValueError("metadata and opacity rows differ")
    if opacity_space == "logit":
        cutoff = np.log(threshold) - np.log1p(-threshold)
    elif opacity_space == "alpha":
        cutoff = threshold
        if np.any(opacity < 0) or np.any(opacity > 1):
            raise ValueError("alpha-space opacity must be in [0,1]")
    else:
        raise ValueError("unknown opacity parameterization")
    unique, inverse = np.unique(instance_ids, return_inverse=True)
    values = np.bincount(inverse, weights=np.asarray(opacity > cutoff, dtype=np.int64), minlength=len(unique))
    return {int(iid): int(count) for iid, count in zip(unique, values)}


def opacity_audit(original, compressed, original_opacity, compressed_opacity, *, policy=None,
                  thresholds=(.01, .05), opacity_space="logit"):
    base = retention_report(original, compressed, policy=policy)
    rows = base["instances"]
    threshold_groups = {}
    if np.isnan(original_opacity).any() or np.isnan(compressed_opacity).any():
        raise ValueError("NaN opacity values cannot establish active-point counts")
    for threshold in thresholds:
        key = format(float(threshold), ".6g")
        before = active_counts(original["instance_ids"], original_opacity, threshold, opacity_space)
        after = active_counts(compressed["instance_ids"], compressed_opacity, threshold, opacity_space)
        for row in rows:
            iid = row["instance_id"]
            row.setdefault("opacity_counts", {})[key] = dict(
                original_active=before.get(iid, 0), compressed_active=after.get(iid, 0),
                active_retention_fraction=after.get(iid, 0) / before[iid] if before.get(iid, 0) else None,
                active_extinct=after.get(iid, 0) == 0,
                active_below_raw_count_floor=after.get(iid, 0) < row["min_keep"])

        def group(selected):
            return dict(instance_count=len(selected),
                        original_active=sum(before.get(r["instance_id"], 0) for r in selected),
                        compressed_active=sum(after.get(r["instance_id"], 0) for r in selected),
                        active_extinct_count=sum(after.get(r["instance_id"], 0) == 0 for r in selected),
                        active_below_raw_count_floor_count=sum(after.get(r["instance_id"], 0) < r["min_keep"] for r in selected))
        threshold_groups[key] = dict(
            all_instances=group(rows),
            protected=group([r for r in rows if r["protected"]]),
            protected_small=group([r for r in rows if r["protected"] and r["small"]]),
            unresolved_original_active=before.get(0, 0), unresolved_compressed_active=after.get(0, 0))
    base["opacity_thresholds"] = list(map(float, thresholds))
    base["opacity_space"] = opacity_space
    base["opacity_groups"] = threshold_groups
    base["limitations"].extend([
        "Opacity-threshold counts are a weak visibility proxy, not measured per-view rendered contribution.",
        "Hard minimum retention floors constrain raw point counts only; active counts may fall below those floors.",
        "An active-opacity retention ratio can exceed one because optimization changes opacity."])
    return base


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--original_ply", type=Path, required=True)
    p.add_argument("--compressed_ply", type=Path, required=True)
    p.add_argument("--original_metadata", type=Path, required=True)
    p.add_argument("--compressed_metadata", type=Path, required=True)
    p.add_argument("--floors_json", type=Path)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--thresholds", type=float, nargs="+", default=[.01, .05])
    p.add_argument("--opacity_space", choices=["logit", "alpha"], default="logit")
    args = p.parse_args(argv)
    original = load_metadata(args.original_metadata)
    compressed = load_metadata(args.compressed_metadata, require_original_ids=True)
    before_ply = read_ply_vertices(args.original_ply)
    after_ply = read_ply_vertices(args.compressed_ply)
    if len(before_ply) != len(original["instance_ids"]) or len(after_ply) != len(compressed["instance_ids"]):
        raise ValueError("PLY and corresponding metadata row counts differ")
    if "opacity" not in before_ply.dtype.names or "opacity" not in after_ply.dtype.names:
        raise ValueError("PLY must contain standard opacity property")
    before_hash, after_hash = xyz_hash(before_ply), xyz_hash(after_ply)
    expected_source = original.get("source_xyz_sha256")
    if expected_source is not None and str(expected_source.item()) != before_hash:
        raise ValueError("original PLY rows are not aligned with source metadata XYZ hash")
    expected_current = compressed.get("current_xyz_sha256")
    if expected_current is not None and str(expected_current.item()) != after_hash:
        raise ValueError("compressed PLY rows are not aligned with metadata XYZ hash")
    compressed_source = compressed.get("source_xyz_sha256")
    if compressed_source is not None and str(compressed_source.item()) != before_hash:
        raise ValueError("compressed lineage refers to a different original XYZ source")
    policy = read_json(args.floors_json) if args.floors_json else None
    if policy is None and "policy_json" in compressed:
        policy = json.loads(str(compressed["policy_json"].item()))
    report = opacity_audit(original, compressed, before_ply["opacity"], after_ply["opacity"],
                           policy=policy, thresholds=args.thresholds, opacity_space=args.opacity_space)
    report["ply_alignment_checks"] = dict(original_row_count=len(before_ply), compressed_row_count=len(after_ply),
                                          original_xyz_sha256=before_hash, compressed_xyz_sha256=after_hash,
                                          original_hash_checked=expected_source is not None,
                                          compressed_current_hash_checked=expected_current is not None,
                                          source_hash_checked=compressed_source is not None)
    report["inputs"] = {k: str(v.resolve()) for k, v in vars(args).items() if isinstance(v, Path) and k != "output"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    print(json.dumps(dict(output=str(args.output.resolve()), opacity_groups=report["opacity_groups"]), indent=2))


if __name__ == "__main__":
    main()
