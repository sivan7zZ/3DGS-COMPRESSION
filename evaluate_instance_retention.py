#!/usr/bin/env python3
"""Report structural small-region survival using stable original Gaussian IDs.

For old baselines without lineage, run an unprotected ablation through the same
code. Approximate XYZ matching is intentionally not presented as exact retention.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from utils.instance_metrics import (inspect_standard_ply, load_metadata,
    projected_area_statistics, read_json, retention_report, sha256_file)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--original_metadata", required=True, type=Path)
    p.add_argument("--compressed_metadata", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--catalog", type=Path)
    p.add_argument("--floors_json", type=Path, help="Final pruning audit with explicit per-instance min_keep")
    p.add_argument("--mask_tracks", type=Path)
    p.add_argument("--train_frames", type=Path, help="JSON list or newline-separated training frame stems")
    p.add_argument("--image_width", type=int, default=1592)
    p.add_argument("--image_height", type=int, default=894)
    p.add_argument("--compressed_ply", type=Path)
    p.add_argument("--expected_sh_degree", type=int, choices=[0, 1, 2, 3])
    p.add_argument("--teacher_checkpoint", type=Path, help="Hash original SH3 teacher for provenance")
    args = p.parse_args(argv)
    if bool(args.mask_tracks) != bool(args.train_frames):
        p.error("--mask_tracks and --train_frames must be provided together")
    original = load_metadata(args.original_metadata)
    compressed = load_metadata(args.compressed_metadata, require_original_ids=True)
    catalog = read_json(args.catalog) if args.catalog else None
    policy = read_json(args.floors_json) if args.floors_json else None
    areas = None
    train_frames = []
    excluded = []
    if args.mask_tracks:
        frames_text = args.train_frames.read_text().strip()
        if frames_text.startswith("["):
            train_frames = json.loads(frames_text)
        elif frames_text.startswith("{"):
            frame_record = json.loads(frames_text)
            train_frames = frame_record.get("train_frames", frame_record.get("frames", []))
        else:
            train_frames = [line.strip() for line in frames_text.splitlines() if line.strip()]
        train_frames = [Path(frame).stem for frame in train_frames]
        shapes = {f: (args.image_height, args.image_width) for f in train_frames}
        areas, excluded = projected_area_statistics(read_json(args.mask_tracks), train_frames, shapes)
    report = retention_report(original, compressed, catalog=catalog, policy=policy, projected_areas=areas)
    report["inputs"] = {key: str(value.resolve()) for key, value in vars(args).items()
                        if isinstance(value, Path) and key != "output"}
    if args.mask_tracks:
        report["projected_area_protocol"] = {
            "training_frames": train_frames, "excluded_nontraining_frames": excluded,
            "width": args.image_width, "height": args.image_height,
            "area_statistic": "median over training views of summed SAM local-region pixels / image pixels",
            "used_for": "descriptive audit; protected/small groups come from explicit pruning policy",
        }
    if args.compressed_ply:
        report["ply"] = inspect_standard_ply(args.compressed_ply, args.expected_sh_degree)
        if report["ply"]["vertex_count"] != len(compressed["instance_ids"]):
            raise ValueError("compressed PLY and lineage NPZ have different Gaussian counts")
    if args.teacher_checkpoint:
        report["original_teacher_sha256"] = sha256_file(args.teacher_checkpoint)
    passed = report["groups"]["all_instances"]["floor_violation_count"] == 0
    if "ply" in report:
        ply = report["ply"]
        passed &= (ply["valid_sh_layout"] and ply["expected_sh_matches"] and
                   ply["no_dog_properties"] and not ply["missing_standard_properties"])
    report["structural_checks_passed"] = bool(passed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    print(json.dumps({"report": str(args.output.resolve()), "structural_checks_passed": bool(passed),
                      "gaussian_counts": report["gaussian_counts"], "groups": report["groups"]}, indent=2))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
