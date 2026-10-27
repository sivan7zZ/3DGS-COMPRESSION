#!/usr/bin/env python3
"""Prepare TRAIN-ONLY small-region profiles and balanced image-loss weights.

No class labels or held-out RGB/SAM observations are read. Input instance priors
must themselves have been fused from exactly the declared training cameras.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from utils.instance_metrics import (load_metadata, projected_area_statistics,
                                     read_json, sha256_file)


def read_frames(path):
    text = Path(path).read_text().strip()
    if text.startswith("["):
        frames = json.loads(text)
    elif text.startswith("{"):
        record = json.loads(text)
        frames = record.get("train_frames", record.get("frames", []))
    else:
        frames = [line.strip() for line in text.splitlines() if line.strip()]
    frames = [Path(f).stem for f in frames]
    if not frames or len(set(frames)) != len(frames):
        raise ValueError("training frames must be nonempty and unique")
    return frames


def make_profiles(metadata, catalog, tracks, frames, shapes, *, small_area_fraction=.005,
                  min_views=3, min_direct_count=100, min_confidence=.45,
                  max_propagated_to_direct=1.):
    """Use only SAM projected size and original training evidence for selection."""
    areas, excluded = projected_area_statistics(tracks, frames, shapes)
    if excluded:
        raise ValueError("instance tracks contain nontraining frames: " + ", ".join(excluded))
    ids = metadata["instance_ids"]
    source = np.asarray(metadata.get("instance_source", np.zeros(len(ids), dtype=np.uint8)))
    confidence = np.asarray(metadata.get("instance_confidence", np.zeros(len(ids), dtype=np.float32)))
    views = np.asarray(metadata.get("instance_views", np.zeros(len(ids), dtype=np.uint8)))
    if source.shape != ids.shape or confidence.shape != ids.shape or views.shape != ids.shape:
        raise ValueError("metadata confidence/source/views arrays must align with Gaussian IDs")
    if not np.all(np.isfinite(confidence)) or np.any(confidence < 0) or np.any(confidence > 1):
        raise ValueError("confidence must be finite and between zero and one")
    catalog_records = catalog.get("instances", []) if isinstance(catalog, dict) else catalog
    lookup = {int(r["instance_id"]): r for r in catalog_records}
    all_ids, counts = np.unique(ids, return_counts=True)
    profiles = []
    for iid, count in zip(all_ids, counts):
        iid = int(iid)
        if iid <= 0:
            continue
        take = ids == iid
        direct = take & (source == 1)
        propagated_count = int(np.count_nonzero(take & (source == 2)))
        direct_count = int(direct.sum())
        mean_confidence = float(confidence[take].mean())
        mean_direct_confidence = float(confidence[direct].mean()) if direct_count else 0.
        stats = areas.get(iid, {})
        fraction = stats.get("median_projected_area_fraction")
        view_count = stats.get("training_view_count", 0)
        small = fraction is not None and fraction <= small_area_fraction
        reliable = (view_count >= min_views and direct_count >= min_direct_count and
                    mean_confidence >= min_confidence and
                    propagated_count <= direct_count * max_propagated_to_direct)
        c = lookup.get(iid, {})
        if c.get("gaussian_count", int(count)) != int(count):
            raise ValueError("catalog and original prior Gaussian counts differ for instance " + str(iid))
        for frame in c.get("views", []):
            if Path(frame).stem not in frames:
                raise ValueError("catalog includes held-out frame " + frame)
        profiles.append(dict(instance_id=iid, original_count=int(count), gaussian_count=int(count),
                             direct_count=direct_count, propagated_count=propagated_count,
                             mean_confidence=mean_confidence, mean_direct_confidence=mean_direct_confidence,
                             multiview_direct_count=int(np.count_nonzero(direct & (views >= 2))),
                             train_view_count=int(view_count), projected_area_fraction=fraction,
                             small=bool(small), reliable=bool(reliable),
                             selected_for_roi=bool(small and reliable), **stats))
    return profiles


def roi_weights(global_ids):
    """Every visible selected instance gets equal total ROI loss mass.

    weights have image mean one when nonempty; the loss helper should divide by
    weights.sum() times RGB channels. A zero image skips the extra ROI loss.
    edge_weights mark one pixel on either side of every selected instance-ID
    change, including adjacent selected objects, without image-boundary wrap.
    """
    if global_ids.ndim != 2:
        raise ValueError("global ROI labels must be H x W")
    weights = np.zeros(global_ids.shape, dtype=np.float32)
    ids, counts = np.unique(global_ids, return_counts=True)
    for iid, count in zip(ids, counts):
        if iid > 0:
            weights[global_ids == iid] = 1. / count
    mass = float(weights.sum(dtype=np.float64))
    if mass:
        weights *= global_ids.size / mass
    edges = np.zeros(global_ids.shape, dtype=bool)
    vertical = (global_ids[1:] != global_ids[:-1]) & ((global_ids[1:] > 0) | (global_ids[:-1] > 0))
    horizontal = (global_ids[:, 1:] != global_ids[:, :-1]) & ((global_ids[:, 1:] > 0) | (global_ids[:, :-1] > 0))
    edges[1:] |= vertical
    edges[:-1] |= vertical
    edges[:, 1:] |= horizontal
    edges[:, :-1] |= horizontal
    edge_weights = edges.astype(np.float32)
    if edges.any():
        edge_weights *= global_ids.size / int(edges.sum())
    return weights, edge_weights


def write_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--metadata", type=Path, required=True)
    p.add_argument("--catalog", type=Path, required=True)
    p.add_argument("--mask_tracks", type=Path, required=True)
    p.add_argument("--sam_masks", type=Path, required=True)
    p.add_argument("--train_frames", type=Path, required=True)
    p.add_argument("--output_dir", type=Path, required=True)
    p.add_argument("--small_area_fraction", type=float, default=.005)
    p.add_argument("--min_views", type=int, default=3)
    p.add_argument("--min_direct_count", type=int, default=100)
    p.add_argument("--min_confidence", type=float, default=.45)
    p.add_argument("--max_propagated_to_direct", type=float, default=1.)
    p.add_argument("--sampling_weight_cap", type=float, default=4.)
    args = p.parse_args(argv)
    if not (0 < args.small_area_fraction <= 1 and 0 <= args.min_confidence <= 1 and
            args.min_views >= 1 and args.min_direct_count >= 1 and
            args.max_propagated_to_direct >= 0 and args.sampling_weight_cap >= 1):
        p.error("invalid selection or sampling thresholds")
    frames = read_frames(args.train_frames)
    tracks = read_json(args.mask_tracks)
    nodes = tracks.get("nodes", [])
    nontraining = sorted({n.get("frame", "") for n in nodes if n.get("frame", "") not in frames})
    if nontraining:
        raise ValueError("training initialization tracks contain held-out frames: " + ", ".join(nontraining))
    shapes = {}
    node_maps = {frame: {} for frame in frames}
    for node in nodes:
        frame = node["frame"]
        local = int(node["local_region"])
        iid = max(int(node.get("instance_id", 0)), 0)
        if local in node_maps[frame] and node_maps[frame][local] != iid:
            raise ValueError("a local SAM region maps to conflicting global instances")
        node_maps[frame][local] = iid
    for frame in frames:
        with np.load(args.sam_masks / (frame + ".npz"), allow_pickle=False) as z:
            shapes[frame] = z["labels"].shape
    metadata = load_metadata(args.metadata)
    profiles = make_profiles(metadata, read_json(args.catalog), tracks, frames, shapes,
                             small_area_fraction=args.small_area_fraction, min_views=args.min_views,
                             min_direct_count=args.min_direct_count, min_confidence=args.min_confidence,
                             max_propagated_to_direct=args.max_propagated_to_direct)
    selected = {r["instance_id"]: r for r in profiles if r["selected_for_roi"]}
    all_ids = {r["instance_id"] for r in profiles}
    mapped_ids = {iid for mapping in node_maps.values() for iid in mapping.values() if iid > 0}
    if not mapped_ids <= all_ids:
        raise ValueError("mask_tracks references instance IDs absent from source priors")
    roi_dir = args.output_dir / "roi"
    roi_dir.mkdir(parents=True, exist_ok=True)
    per_frame = {}
    for frame in frames:
        with np.load(args.sam_masks / (frame + ".npz"), allow_pickle=False) as z:
            labels = z["labels"]
        mapping = node_maps[frame]
        lut = np.zeros(int(labels.max()) + 1, dtype=np.int32)
        for local, iid in mapping.items():
            if local >= len(lut):
                raise ValueError("mask_tracks local region absent from SAM label map")
            if iid in selected:
                lut[local] = iid
        global_ids = lut[labels]
        weights, edge_weights = roi_weights(global_ids)
        visible = np.unique(global_ids)
        visible = visible[visible > 0]
        if len(visible):
            inverse_size = [np.sqrt(args.small_area_fraction /
                max(selected[int(iid)]["projected_area_fraction"], 1. / labels.size)) for iid in visible]
            sampling_weight = min(args.sampling_weight_cap, 1. + float(np.mean(inverse_size)))
        else:
            sampling_weight = 1.
        path = roi_dir / (frame + ".npz")
        np.savez_compressed(path, weights=weights, edge_weights=edge_weights,
                            small_mask=(global_ids > 0), instance_ids=global_ids,
                            sampling_weight=np.asarray(sampling_weight, dtype=np.float32),
                            frame=np.asarray(frame))
        per_frame[frame] = dict(roi_file=str(path.resolve()), small_instances=visible.tolist(),
                               small_pixels=int(np.count_nonzero(global_ids)), sampling_weight=sampling_weight)
    record = dict(schema_version=1, train_frames=frames,
                  selection_protocol="training-view SAM projected size and direct Gaussian support only",
                  thresholds={k: getattr(args, k) for k in ("small_area_fraction", "min_views", "min_direct_count",
                              "min_confidence", "max_propagated_to_direct", "sampling_weight_cap")},
                  selected_small_reliable_count=len(selected), small_count=sum(r["small"] for r in profiles),
                  roi_weighting="equal total loss mass per visible selected instance; image mean one; zero outside",
                  edge_weighting="one pixel on either side of 4-neighbor instance-ID boundaries; image mean one",
                  metadata_sha256=sha256_file(args.metadata), mask_tracks_sha256=sha256_file(args.mask_tracks),
                  instances=profiles, per_frame=per_frame)
    if "source_xyz_sha256" in metadata:
        record["source_xyz_sha256"] = str(metadata["source_xyz_sha256"])
    write_json(args.output_dir / "instance_profiles.json", record)
    write_json(args.output_dir / "projected_area.json",
               {str(r["instance_id"]): r["projected_area_fraction"] for r in profiles
                if r["projected_area_fraction"] is not None})
    write_json(args.output_dir / "roi_manifest.json", dict(train_frames=frames, views=per_frame,
                                                          selected_small_reliable_count=len(selected)))
    print(json.dumps({"output_dir": str(args.output_dir.resolve()), "training_views": len(frames),
                      "instances": len(profiles), "small": record["small_count"],
                      "selected_small_reliable": len(selected)}, indent=2))


if __name__ == "__main__":
    main()
