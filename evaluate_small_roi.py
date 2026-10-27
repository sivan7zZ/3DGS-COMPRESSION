#!/usr/bin/env python3
"""Compare rendered quality on independent held-out SAM small-region ROIs.

This evaluates image evidence in addition to exact Gaussian survival. These are
automatic regions, so the scores cannot by themselves establish true instance
detection recall. Held-out masks are evaluation-only and must not set quotas.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from prepare_instance_protection import read_frames
from utils.instance_metrics import sha256_file


def image(path):
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.


def gradients(rgb):
    gx = np.zeros_like(rgb)
    gy = np.zeros_like(rgb)
    gx[:, 1:] = rgb[:, 1:] - rgb[:, :-1]
    gy[1:] = rgb[1:] - rgb[:-1]
    return gx, gy


def error_maps(reference, predicted):
    if reference.shape != predicted.shape:
        raise ValueError("reference/predicted resolutions must agree exactly")
    difference = predicted - reference
    ref_x, ref_y = gradients(reference)
    pred_x, pred_y = gradients(predicted)
    return dict(mse=np.mean(np.square(difference), axis=2, dtype=np.float64),
                l1=np.mean(np.abs(difference), axis=2, dtype=np.float64),
                boundary_gradient_l1=(np.mean(np.abs(ref_x - pred_x), axis=2, dtype=np.float64) +
                                      np.mean(np.abs(ref_y - pred_y), axis=2, dtype=np.float64)) / 2.)


def metrics_from_maps(errors, mask, edge_mask=None):
    if mask.shape != errors["mse"].shape:
        raise ValueError("error maps and ROI mask resolutions must agree exactly")
    if not np.any(mask):
        raise ValueError("cannot measure an empty ROI")
    mse = float(errors["mse"][mask].mean())
    mae = float(errors["l1"][mask].mean())
    metrics = dict(mse=mse, l1=mae, psnr=float(-10 * np.log10(max(mse, 1e-12))))
    if edge_mask is not None and np.any(edge_mask):
        metrics["boundary_gradient_l1"] = float(errors["boundary_gradient_l1"][edge_mask].mean())
    else:
        metrics["boundary_gradient_l1"] = None
    return metrics


def masked_metrics(reference, predicted, mask, edge_mask=None):
    return metrics_from_maps(error_maps(reference, predicted), mask, edge_mask)


def boundary_mask(mask):
    edges = np.zeros(mask.shape, dtype=bool)
    vertical = mask[1:] != mask[:-1]
    horizontal = mask[:, 1:] != mask[:, :-1]
    edges[1:] |= vertical
    edges[:-1] |= vertical
    edges[:, 1:] |= horizontal
    edges[:, :-1] |= horizontal
    return edges


def summarize(rows, model_names):
    result = {}
    for name in model_names:
        metrics = [r["models"][name] for r in rows]
        result[name] = {}
        for metric in ("mse", "l1", "psnr", "boundary_gradient_l1"):
            values = [m[metric] for m in metrics if m[metric] is not None]
            result[name]["mean_" + metric] = float(np.mean(values)) if values else None
            result[name]["median_" + metric] = float(np.median(values)) if values else None
        result[name]["region_count"] = len(metrics)
        result[name]["psnr_from_equal_region_mean_mse"] = (
            float(-10 * np.log10(max(result[name]["mean_mse"], 1e-12))) if metrics else None)
    return result


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sam_masks", type=Path, required=True)
    p.add_argument("--test_frames", type=Path, required=True, help="Ordered like render.py test cameras")
    p.add_argument("--train_frames", type=Path, required=True)
    p.add_argument("--ground_truth_dir", type=Path, required=True, help="Indexed render.py gt PNGs")
    p.add_argument("--model", action="append", required=True, help="LABEL=/absolute/renders/directory")
    p.add_argument("--baseline_label", help="Default: first --model")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--source_images", type=Path, help="Validate indexed GT against each named source RGB")
    p.add_argument("--filename_pattern", default="{index:05d}.png")
    p.add_argument("--small_area_fraction", type=float, default=.005)
    p.add_argument("--min_region_pixels", type=int, default=16)
    p.add_argument("--min_mask_quality", type=float, default=.85)
    args = p.parse_args(argv)
    if not (0 < args.small_area_fraction <= 1 and args.min_region_pixels >= 1 and
            0 <= args.min_mask_quality <= 1):
        p.error("invalid ROI size or SAM quality thresholds")
    train = read_frames(args.train_frames)
    test = read_frames(args.test_frames)
    if set(train) & set(test):
        raise ValueError("test-frame and training-frame lists overlap")
    models = {}
    for spec in args.model:
        label, separator, directory = spec.partition("=")
        if not separator or not label or label in models:
            raise ValueError("each --model must have a unique LABEL=RENDERSDIR")
        models[label] = Path(directory)
    baseline = args.baseline_label or next(iter(models))
    if baseline not in models:
        raise ValueError("baseline_label is absent from --model inputs")
    rows = []
    full_views = []
    source_checks = []
    mask_inputs = []
    for index, frame in enumerate(test):
        filename = args.filename_pattern.format(index=index, frame=frame)
        reference = image(args.ground_truth_dir / filename)
        if args.source_images:
            matches = [p for p in args.source_images.iterdir() if p.stem == frame]
            if len(matches) != 1:
                raise ValueError("expected exactly one named source image for " + frame)
            source = image(matches[0])
            if source.shape != reference.shape or float(np.abs(source - reference).max()) > 2. / 255.:
                raise ValueError("GT index/name mapping or RGB resolution is inconsistent for " + frame)
            source_checks.append(dict(frame=frame, source=str(matches[0]), max_abs_rgb_error=float(np.abs(source - reference).max())))
        predictions = {name: image(directory / filename) for name, directory in models.items()}
        for name, predicted in predictions.items():
            if predicted.shape != reference.shape:
                raise ValueError(name + " rendered RGB shape differs from ground truth")
        errors = {name: error_maps(reference, predicted) for name, predicted in predictions.items()}
        sam_path = args.sam_masks / (frame + ".npz")
        with np.load(sam_path, allow_pickle=False) as z:
            labels = z["labels"]
            quality = z["region_quality"]
        if labels.shape != reference.shape[:2]:
            raise ValueError("SAM masks must match native rendered image shape; implicit resizing is forbidden")
        if len(quality) != int(labels.max()) or np.any(labels < 0):
            raise ValueError("SAM labels and per-region quality arrays disagree")
        mask_inputs.append(dict(frame=frame, path=str(sam_path.resolve()), sha256=sha256_file(sam_path)))
        counts = np.bincount(labels.ravel())
        full_mask = np.ones(labels.shape, dtype=bool)
        full_views.append(dict(frame=frame, render_filename=filename,
                               models={name: metrics_from_maps(err, full_mask) for name, err in errors.items()}))
        for local in range(1, len(counts)):
            area = int(counts[local])
            fraction = area / labels.size
            if area < args.min_region_pixels or fraction > args.small_area_fraction or quality[local - 1] < args.min_mask_quality:
                continue
            mask = labels == local
            edges = boundary_mask(mask)
            rows.append(dict(frame=frame, render_filename=filename, local_region=local, pixels=area,
                             area_fraction=fraction, mask_quality=float(quality[local - 1]),
                             models={name: metrics_from_maps(err, mask, edges) for name, err in errors.items()}))
    comparisons = {}
    for name in models:
        if name == baseline:
            continue
        differences = [r["models"][name]["psnr"] - r["models"][baseline]["psnr"] for r in rows]
        l1_differences = [r["models"][name]["l1"] - r["models"][baseline]["l1"] for r in rows]
        comparisons[name + "_vs_" + baseline] = dict(
            mean_roi_psnr_delta=float(np.mean(differences)) if rows else None,
            median_roi_psnr_delta=float(np.median(differences)) if rows else None,
            fraction_regions_with_higher_psnr=float(np.mean(np.array(differences) > 0)) if rows else None,
            mean_roi_l1_delta=float(np.mean(l1_differences)) if rows else None)
    report = dict(schema_version=1,
                  protocol="held-out local SAM regions; evaluation-only masks; equal weight per ROI",
                  train_frames=train, test_frames=test,
                  thresholds=dict(small_area_fraction=args.small_area_fraction, min_region_pixels=args.min_region_pixels,
                                  min_mask_quality=args.min_mask_quality),
                  psnr_perfect_cap_db=120, roi_count=len(rows),
                  small_roi_metrics=summarize(rows, models),
                  full_image_metrics=summarize(full_views, models), comparisons=comparisons,
                  inputs=dict(ground_truth_dir=str(args.ground_truth_dir.resolve()),
                              models={k: str(v.resolve()) for k, v in models.items()}, masks=mask_inputs),
                  source_gt_mapping_checks=source_checks,
                  regions=rows, views=full_views,
                  limitations=["SAM small regions may be object parts or background, and are not semantic object ground truth.",
                               "ROI image quality does not measure physical object detection recall.",
                               "Test masks are used solely for reporting and may not be fed to training protection quotas.",
                               "PSNR uses 8-bit saved PNG images; do not mix with float-render metrics."])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    print(json.dumps({"output": str(args.output.resolve()), "roi_count": len(rows),
                      "small_roi_metrics": report["small_roi_metrics"], "comparisons": comparisons}, indent=2))


if __name__ == "__main__":
    main()
