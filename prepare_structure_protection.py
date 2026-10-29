#!/usr/bin/env python3
"""Build train-only visible coverage constraints for small and thin structures.

RGB and SAM masks are training-view inputs. Source Gaussian alpha compositing
supplies occlusion-aware contributor IDs; a projected center never certifies
visibility. No semantic annotations, test images, or test masks are consumed.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image

from utils.instance_pruning import xyz_sha256
from utils.structure_pruning import file_sha256, greedy_coverage
from utils.structure_scene_io import read_ply, read_colmap
from utils.structure_visibility import prepare_from_ply, render_visibility


def read_frames(path):
    text = Path(path).read_text().strip()
    value = json.loads(text) if text.startswith(("[", "{")) else text.splitlines()
    if isinstance(value, dict):
        value = value.get("train_frames", value.get("frames", []))
    frames = [Path(name.strip()).stem for name in value if name.strip()]
    if not frames or len(set(frames)) != len(frames):
        raise ValueError("Training frame list must be nonempty and unique")
    return sorted(frames)


def normalize_map(value):
    value = np.asarray(value, dtype=np.float32)
    mass = float(value.sum(dtype=np.float64))
    return value * (value.size / mass) if mass else value


def zhang_suen(mask):
    """Binary one-pixel skeleton, preserving diagonal connectivity."""
    image = np.pad(np.asarray(mask, dtype=np.uint8), 1)
    while True:
        changed = False
        for phase in (0, 1):
            p = [image[:-2, 1:-1], image[:-2, 2:], image[1:-1, 2:], image[2:, 2:],
                 image[2:, 1:-1], image[2:, :-2], image[1:-1, :-2], image[:-2, :-2]]
            neighbours = sum(p)
            transitions = sum((p[index] == 0) & (p[(index + 1) % 8] != 0) for index in range(8))
            if phase == 0:
                safe = (p[0] * p[2] * p[4] == 0) & (p[2] * p[4] * p[6] == 0)
            else:
                safe = (p[0] * p[2] * p[6] == 0) & (p[0] * p[4] * p[6] == 0)
            remove = (image[1:-1, 1:-1] != 0) & (neighbours >= 2) & (neighbours <= 6) & (transitions == 1) & safe
            if remove.any():
                image[1:-1, 1:-1][remove] = 0
                changed = True
        if not changed:
            return image[1:-1, 1:-1].astype(bool)


def keep_long_components(mask, minimum=12):
    import cv2
    count, labels, stats, _ = cv2.connectedComponentsWithStats(np.asarray(mask, dtype=np.uint8), 8)
    eligible = np.zeros(count, dtype=bool)
    if count > 1:
        eligible[1:] = stats[1:, cv2.CC_STAT_AREA] >= minimum
    return eligible[labels]


def image_structure_masks(rgb, teacher, labels, region_boxes, quality, thin_width=4, min_length=12):
    """Return photo-supported SAM boundaries and narrow SAM/RGB skeletons."""
    import cv2
    gray = cv2.cvtColor(rgb.astype(np.float32), cv2.COLOR_RGB2GRAY)
    teacher_gray = cv2.cvtColor(teacher.astype(np.float32), cv2.COLOR_RGB2GRAY)
    gx, gy = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3) / 8, cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3) / 8
    tx, ty = cv2.Sobel(teacher_gray, cv2.CV_32F, 1, 0, ksize=3) / 8, cv2.Sobel(teacher_gray, cv2.CV_32F, 0, 1, ksize=3) / 8
    gradient, teacher_gradient = np.hypot(gx, gy), np.hypot(tx, ty)
    alignment = (gx * tx + gy * ty) / np.maximum(gradient * teacher_gradient, 1e-8)
    threshold = max(.018, float(np.quantile(gradient, .70)))
    supported = (gradient >= threshold) & (teacher_gradient >= .5 * threshold) & (alignment >= .5)
    boundary = np.zeros(labels.shape, dtype=bool)
    change_y = (labels[1:] != labels[:-1]) & ((labels[1:] > 0) | (labels[:-1] > 0))
    change_x = (labels[:, 1:] != labels[:, :-1]) & ((labels[:, 1:] > 0) | (labels[:, :-1] > 0))
    boundary[1:] |= change_y
    boundary[:-1] |= change_y
    boundary[:, 1:] |= change_x
    boundary[:, :-1] |= change_x
    supported_band = cv2.dilate(supported.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    boundary &= supported_band
    thin = np.zeros(labels.shape, dtype=bool)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * thin_width + 1, 2 * thin_width + 1))
    for local, (box, confidence) in enumerate(zip(region_boxes, quality), 1):
        if confidence < .85:
            continue
        x, y, width, height = [int(v) for v in box]
        if width < 2 or height < 2:
            continue
        mask = np.pad((labels[y:y+height, x:x+width] == local).astype(np.uint8), thin_width + 1)
        # A narrow branch disappears under opening, unlike a broad roof/road.
        opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        narrow = (mask > 0) & (opened == 0)
        skeleton = keep_long_components(zhang_suen(narrow), min_length)
        pad = thin_width + 1
        thin[y:y+height, x:x+width] |= skeleton[pad:-pad, pad:-pad]
    # Curvilinear RGB ridges cover wires/poles that SAM can miss. Both bright
    # and dark ridges are accepted; teacher/photograph support rejects ghosts.
    ridge = np.zeros(labels.shape, dtype=np.float32)
    teacher_ridge = np.zeros_like(ridge)
    for sigma in (1., 2.):
        for value, destination in ((gray, ridge), (teacher_gray, teacher_ridge)):
            smooth = cv2.GaussianBlur(value, (0, 0), sigma)
            xx = cv2.Sobel(smooth, cv2.CV_32F, 2, 0, ksize=3) / 4
            yy = cv2.Sobel(smooth, cv2.CV_32F, 0, 2, ksize=3) / 4
            xy = cv2.Sobel(smooth, cv2.CV_32F, 1, 1, ksize=3) / 4
            root = np.sqrt((xx - yy) ** 2 + 4 * xy ** 2)
            first, second = .5 * (xx + yy + root), .5 * (xx + yy - root)
            maximum = np.maximum(abs(first), abs(second))
            minimum = np.minimum(abs(first), abs(second))
            response = maximum * np.maximum(0, 1 - minimum / np.maximum(maximum, 1e-8)) * sigma ** 2
            np.maximum(destination, response, out=destination)
    ridge_threshold = max(.012, float(np.quantile(ridge, .95)))
    ridges = (ridge >= ridge_threshold) & (teacher_ridge >= .5 * ridge_threshold)
    ridges = keep_long_components(zhang_suen(ridges), min_length)
    thin |= ridges
    # Thin centers may have zero first derivative; require nearby rather than
    # exact-center agreement. Keep the raw centerline for topology evaluation.
    thin &= cv2.dilate(supported.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    thin = keep_long_components(thin, min_length)
    return boundary, thin, gradient


def screen_cell_groups(mask, instance_ids, id_map, contribution, kind, cell_size,
                       max_candidates=8, minimum_pixels=2, group_first_index=0):
    """Critical pixels -> cells -> genuinely visible source-Gaussian choices."""
    yy, xx = np.nonzero(mask)
    if not len(xx):
        return [], [], [], [], np.zeros(mask.shape, dtype=np.float32)
    cells_x = (mask.shape[1] + cell_size - 1) // cell_size
    cells_y = (mask.shape[0] + cell_size - 1) // cell_size
    # Distinct foreground objects sharing one cell retain separate groups.
    region = np.maximum(instance_ids[yy, xx], 0) if kind != 3 else np.zeros(len(xx), dtype=np.int32)
    keys = (region.astype(np.int64) * cells_y + yy // cell_size) * cells_x + xx // cell_size
    order = np.argsort(keys, kind="mergesort")
    boundaries = np.r_[0, np.flatnonzero(np.diff(keys[order])) + 1, len(order)]
    candidates, strengths, group_records, offsets = [], [], [], [0]
    balanced = np.zeros(mask.shape, dtype=np.float32)
    for left, right in zip(boundaries[:-1], boundaries[1:]):
        selected = order[left:right]
        if len(selected) < minimum_pixels:
            continue
        sy, sx = yy[selected], xx[selected]
        gaussian, inverse = np.unique(id_map[sy, sx], return_inverse=True)
        mass = np.bincount(inverse, weights=contribution[sy, sx], minlength=len(gaussian))
        ranking = np.lexsort((gaussian, -mass))[:max_candidates]
        choices, values = gaussian[ranking], mass[ranking]
        usable = (choices >= 0) & (values > 0)
        choices, values = choices[usable], values[usable]
        if not len(choices):
            continue
        candidates.extend(choices.tolist())
        strengths.extend(values.tolist())
        offsets.append(len(candidates))
        group_records.append(dict(kind=int(kind), x=int(sx.min()), y=int(sy.min()),
                                  width=int(sx.max()-sx.min()+1), height=int(sy.max()-sy.min()+1),
                                  pixels=int(len(sx)), instance_id=int(region[selected[0]]),
                                  source_total_dominant_mass=float(contribution[sy, sx].sum())))
        balanced[sy, sx] += 1. / len(selected)
    return candidates, strengths, offsets, group_records, normalize_map(balanced)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source_ply", "colmap", "images", "sam_masks", "train_frames", "instance_profiles", "mask_tracks", "instance_metadata", "teacher_renders", "output_dir"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--visibility_cache", type=Path)
    parser.add_argument("--final_budget", type=int, required=True)
    parser.add_argument("--anchor_budget_fraction", type=float, default=.35)
    parser.add_argument("--small_cell", type=int, default=24)
    parser.add_argument("--boundary_cell", type=int, default=24)
    parser.add_argument("--thin_cell", type=int, default=16)
    parser.add_argument("--max_candidates", type=int, default=8)
    parser.add_argument("--min_contribution", type=float, default=.035)
    parser.add_argument("--min_alpha", type=float, default=.5)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args(argv)
    if args.final_budget < 1 or not 0 < args.anchor_budget_fraction <= 1 or min(args.small_cell, args.boundary_cell, args.thin_cell, args.max_candidates) < 1:
        parser.error("Invalid cell policy or explicit final budget")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    roi_dir = args.output_dir / "roi"
    roi_dir.mkdir(exist_ok=True)
    cache_dir = args.output_dir / "visibility_cache"
    cache_dir.mkdir(exist_ok=True)
    build_dir = args.output_dir / "cpu_visibility_build"
    frames = read_frames(args.train_frames)
    cameras = read_colmap(args.colmap)
    if not set(frames) <= set(cameras):
        raise ValueError("Training frames missing from COLMAP")
    profiles = json.loads(args.instance_profiles.read_text())
    if sorted(profiles["train_frames"]) != frames:
        raise ValueError("Instance profiles and declared training views differ")
    small_ids = {int(row["instance_id"]) for row in profiles["instances"] if row["small"] and
                 row["train_view_count"] >= 2 and row["direct_count"] >= 16 and
                 row["multiview_direct_count"] >= 8 and row["mean_direct_confidence"] >= .45}
    tracks = json.loads(args.mask_tracks.read_text())
    maps = {frame: {} for frame in frames}
    for row in tracks["nodes"]:
        if row["frame"] not in maps:
            raise ValueError("Instance graph contains a held-out frame: " + row["frame"])
        maps[row["frame"]][int(row["local_region"])] = max(0, int(row["instance_id"]))
    ply, _ = read_ply(args.source_ply)
    xyz = np.column_stack([ply[name] for name in ("x", "y", "z")]).astype(np.float32)
    source_xyz_hash = xyz_sha256(xyz)
    source_file_hash = file_sha256(args.source_ply)
    with np.load(args.instance_metadata, allow_pickle=False) as data:
        if str(data["source_xyz_sha256"].item()) != source_xyz_hash:
            raise ValueError("Source instance metadata Gaussian alignment mismatch")
        source_instances = np.maximum(np.asarray(data["instance_ids"], dtype=np.int32), 0)
    gaussians = None
    all_candidates, all_strengths, all_offsets, all_groups, per_frame = [], [], [0], [], {}
    for frame_index, frame in enumerate(frames):
        cache = cache_dir / (frame + "_visibility.npz")
        cache_json = cache_dir / (frame + "_visibility.json")
        camera_hash = hashlib.sha256((json.dumps(cameras[frame]["camera"], sort_keys=True) + cameras[frame]["R"].tobytes().hex() + cameras[frame]["t"].tobytes().hex()).encode()).hexdigest()
        fingerprint = dict(source_xyz_sha256=source_xyz_hash, camera_sha256=camera_hash,
                           implementation="CPU anisotropic native-resolution max(alpha*T), no radius cap")
        usable = cache.is_file() and cache_json.is_file() and json.loads(cache_json.read_text()) == fingerprint
        old_cache = args.visibility_cache / (frame + "_visibility.npz") if args.visibility_cache else None
        old_evidence = args.visibility_cache / (frame + "_evidence.npz") if args.visibility_cache else None
        if not usable and old_cache is not None and old_cache.is_file() and old_evidence.is_file():
            with np.load(old_evidence, allow_pickle=False) as data:
                old_fingerprint = json.loads(str(data["fingerprint"].item()))
            if old_fingerprint.get("source_sha256") == source_file_hash:
                # Both old artifacts belong to the same original CPU association
                # run; bind the reused native maps to this explicit fingerprint.
                import shutil
                shutil.copy2(old_cache, cache)
                cache_json.write_text(json.dumps(fingerprint, sort_keys=True))
                usable = True
        if not usable:
            if gaussians is None:
                gaussians = prepare_from_ply(ply)
            result = render_visibility(gaussians, cameras[frame], threads=args.threads, build_dir=build_dir)
            np.savez_compressed(cache, **{name: result[name] for name in ("id_map", "contribution", "depth", "alpha")})
            cache_json.write_text(json.dumps(fingerprint, sort_keys=True))
        with np.load(cache, allow_pickle=False) as data:
            id_map, contribution, alpha = (np.array(data[name]) for name in ("id_map", "contribution", "alpha"))
        with np.load(args.sam_masks / (frame + ".npz"), allow_pickle=False) as data:
            labels = np.array(data["labels"])
            boxes, quality = np.array(data["region_bbox"]), np.array(data["region_quality"])
        if labels.shape != id_map.shape or np.any(id_map >= len(xyz)):
            raise ValueError("Native visibility and SAM arrays do not match")
        camera_name = cameras[frame]["name"]
        rgb = np.asarray(Image.open(args.images / camera_name).convert("RGB"), dtype=np.float32) / 255.
        teacher_path = args.teacher_renders / ("%05d.png" % frame_index)
        teacher = np.asarray(Image.open(teacher_path).convert("RGB"), dtype=np.float32) / 255.
        if rgb.shape != teacher.shape or rgb.shape[:2] != labels.shape:
            raise ValueError("Teacher/photo/native-SAM size mismatch")
        lut = np.zeros(int(labels.max()) + 1, dtype=np.int32)
        for local, instance_id in maps[frame].items():
            lut[local] = instance_id
        global_ids = lut[labels]
        boundary, skeleton, gradient = image_structure_masks(rgb, teacher, labels, boxes, quality)
        valid = (id_map >= 0) & (contribution >= args.min_contribution) & (alpha >= args.min_alpha)
        small = np.isin(global_ids, np.asarray(sorted(small_ids)))
        matching = np.zeros(labels.shape, dtype=bool)
        matching[valid] = source_instances[id_map[valid]] == global_ids[valid]
        masks = ((1, small & valid & matching, args.small_cell),
                 (2, boundary & valid, args.boundary_cell), (3, skeleton & valid, args.thin_cell))
        weight_maps, frame_groups = {}, []
        for kind, mask, cell in masks:
            candidates, strengths, offsets, records, weights = screen_cell_groups(mask, global_ids, id_map, contribution, kind, cell, args.max_candidates)
            if len(records):
                base = len(all_candidates)
                all_candidates.extend(candidates)
                all_strengths.extend(strengths)
                all_offsets.extend((base + np.asarray(offsets[1:], dtype=np.int64)).tolist())
                for record in records:
                    record["frame"] = frame
                all_groups.extend(records)
                frame_groups.extend(records)
            weight_maps[kind] = weights
        import cv2
        edge_band = cv2.dilate((weight_maps[2] > 0).astype(np.uint8), np.ones((3, 3), np.uint8)).astype(np.float32)
        thin_band = cv2.dilate((weight_maps[3] > 0).astype(np.uint8), np.ones((3, 3), np.uint8)).astype(np.float32)
        structure_weights = normalize_map(weight_maps[1] + weight_maps[2] + 2 * weight_maps[3])
        np.savez_compressed(roi_dir / (frame + ".npz"), weights=weight_maps[1], small_mask=small,
                            edge_weights=normalize_map(edge_band), thin_weights=normalize_map(thin_band),
                            structure_weights=structure_weights, thin_skeleton=skeleton,
                            boundary_mask=boundary, sampling_weight=np.asarray(1., dtype=np.float32),
                            frame=np.asarray(frame))
        per_frame[frame] = dict(groups=len(frame_groups), small_pixels=int((small & valid & matching).sum()),
                                boundary_pixels=int((boundary & valid).sum()), thin_pixels=int((skeleton & valid).sum()),
                                teacher_sha256=file_sha256(teacher_path), photo_sha256=file_sha256(args.images / camera_name))
        print("[%d/%d] %s: groups=%d small=%d boundary=%d thin=%d" % (frame_index+1, len(frames), frame, len(frame_groups), per_frame[frame]["small_pixels"], per_frame[frame]["boundary_pixels"], per_frame[frame]["thin_pixels"]), flush=True)
    candidates, offsets, strength = np.asarray(all_candidates, dtype=np.int64), np.asarray(all_offsets, dtype=np.int64), np.asarray(all_strengths, dtype=np.float32)
    if not all_groups:
        raise ValueError("No source-visible critical training groups were found")
    anchor_budget = int(args.final_budget * args.anchor_budget_fraction)
    anchors = greedy_coverage(candidates, offsets, strength, budget=anchor_budget)
    source_strength = np.zeros(len(xyz), dtype=np.float32)
    np.maximum.at(source_strength, candidates, strength)
    anchor_strength = source_strength[anchors]
    anchor_strength = np.clip(anchor_strength / max(float(np.quantile(anchor_strength, .90)), 1e-8), .1, 1.)
    kind = np.asarray([row["kind"] for row in all_groups], dtype=np.uint8)
    manifest = dict(format="pwsr-structure-v1", train_frames=frames, source_xyz_sha256=source_xyz_hash,
                    source_ply_sha256=source_file_hash, final_budget=args.final_budget,
                    anchor_budget=anchor_budget, anchors=len(anchors), groups=len(all_groups),
                    group_policy="Every declared critical screen cell has at least one immutable source-visible alpha*T anchor; all groups covered without dropping cells",
                    limits=dict(small_cell=args.small_cell, boundary_cell=args.boundary_cell, thin_cell=args.thin_cell,
                                max_candidates=args.max_candidates, min_contribution=args.min_contribution, min_alpha=args.min_alpha),
                    protected_small_regions=len(small_ids), per_frame=per_frame,
                    by_kind={name: int((kind == code).sum()) for code, name in ((1, "small_region_cells"), (2, "boundary_segments"), (3, "thin_segments"))},
                    limitation="Source coverage is not a rendered visibility guarantee after optimization; render comparisons and anti-fading/trust-region recovery are required")
    np.savez_compressed(args.output_dir / "structure_priors.npz", format=np.asarray("pwsr-structure-v1"),
                        source_xyz_sha256=np.asarray(source_xyz_hash), anchors=anchors,
                        anchor_xyz=xyz[anchors],
                        anchor_log_scaling=np.column_stack([ply["scale_%d" % index][anchors] for index in range(3)]),
                        anchor_opacity=1 / (1 + np.exp(-np.clip(ply["opacity"][anchors], -80, 80))),
                        anchor_features_dc=np.column_stack([ply["f_dc_%d" % index][anchors] for index in range(3)]),
                        anchor_strength=anchor_strength, candidates=candidates, offsets=offsets,
                        group_strength=strength, group_total_strength=np.asarray([row["source_total_dominant_mass"] for row in all_groups], dtype=np.float32),
                        group_kind=kind, manifest_json=np.asarray(json.dumps(manifest, sort_keys=True)))
    (args.output_dir / "structure_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    (args.output_dir / "critical_screen_cells.json").write_text(json.dumps(all_groups, separators=(",", ":")) + "\n")
    print(json.dumps({key: manifest[key] for key in ("anchors", "groups", "by_kind", "anchor_budget", "final_budget")}, indent=2), flush=True)


if __name__ == "__main__":
    main()
