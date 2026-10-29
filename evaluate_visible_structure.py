#!/usr/bin/env python3
"""Evaluate pre-SH rendered structure survival, with native-pixel evidence crops.

Automatic metrics rank review candidates. They do not establish physical object
recall: SAM can split objects, and an edge is not necessarily an object. Only
GT evidence also reproduced by the uncompressed teacher enters denominators.
Requires NumPy, OpenCV and Pillow (no SciPy/skimage).
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def read_frames(path):
    rows = [Path(s.strip()).stem for s in Path(path).read_text().splitlines() if s.strip()]
    if not rows or len(set(rows)) != len(rows):
        raise ValueError("frame list must be nonempty and unique")
    return rows


def sha256_file(path):
    h = hashlib.sha256()
    with open(str(path), "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def rgb(path):
    return np.asarray(Image.open(str(path)).convert("RGB"), dtype=np.float32) / 255.


def edge_features(image, low=40., high=100.):
    gray = cv2.cvtColor(np.asarray(np.clip(image * 255., 0, 255), np.uint8), cv2.COLOR_RGB2GRAY)
    # Shared thresholds, scale and blur for every model; never tune per method.
    gray = cv2.GaussianBlur(gray, (3, 3), .7)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    angle = np.mod(np.arctan2(gy, gx), np.pi)
    bins = np.mod(np.floor(angle * 8. / np.pi + .5).astype(np.int8), 8)
    edges = cv2.Canny(gray, low, high, L2gradient=True) != 0
    return edges, bins, np.hypot(gx, gy)


def dilate(mask, radius):
    if radius == 0:
        return np.asarray(mask, bool).copy()
    return cv2.dilate(np.asarray(mask, np.uint8), np.ones((2 * radius + 1, 2 * radius + 1), np.uint8)) != 0


def oriented_matches(reference_edges, reference_bins, predicted_edges, predicted_bins, tolerance=2):
    """Match nearby edge locations and approximately consistent normal angle.

    Undirected normals use eight bins. Permitting the neighboring bin is a fixed
    discretized angular tolerance (up to 45 degrees), shared by all methods.
    """
    matched = np.zeros(reference_edges.shape, bool)
    for b in range(8):
        compatible = predicted_edges & ((predicted_bins == b) | (predicted_bins == (b - 1) % 8) |
                                         (predicted_bins == (b + 1) % 8))
        matched |= reference_edges & (reference_bins == b) & dilate(compatible, tolerance)
    return matched


def relative_gradient_matches(reference_edges, reference_bins, reference_strength,
                              predicted_bins, predicted_strength, tolerance=2, ratio=.35):
    """Match signed-independent orientation and relative gradient amplitude.

    Model images are not subjected to Canny hysteresis. Otherwise an imperceptible
    threshold crossing can label a visibly present long edge as fully missing.
    Only reference GT uses Canny to identify meaningful edge locations.
    """
    amplitude = np.zeros(reference_edges.shape, np.float32)
    for b in range(8):
        compatible = ((predicted_bins == b) | (predicted_bins == (b - 1) % 8) |
                      (predicted_bins == (b + 1) % 8))
        strength = np.where(compatible, predicted_strength, 0).astype(np.float32)
        if tolerance:
            strength = cv2.dilate(strength, np.ones((2 * tolerance + 1, 2 * tolerance + 1), np.uint8))
        indices = reference_edges & (reference_bins == b)
        amplitude[indices] = strength[indices] / np.maximum(reference_strength[indices], 1e-6)
    return reference_edges & (amplitude >= ratio), np.clip(amplitude, 0, 1)


def boundary(mask):
    eroded = cv2.erode(np.asarray(mask, np.uint8), np.ones((3, 3), np.uint8)) != 0
    return mask & ~eroded


def skeletonize(mask):
    """Zhang-Suen one-pixel centerline, computed within a padded region crop."""
    z = np.pad(np.asarray(mask, np.uint8), 1)
    for _ in range(max(z.shape) + 1):
        removed = 0
        for phase in (0, 1):
            p2, p3, p4 = z[:-2, 1:-1], z[:-2, 2:], z[1:-1, 2:]
            p5, p6, p7 = z[2:, 2:], z[2:, 1:-1], z[2:, :-2]
            p8, p9 = z[1:-1, :-2], z[:-2, :-2]
            neighbors = (p2, p3, p4, p5, p6, p7, p8, p9)
            count = sum(neighbors)
            transitions = sum(((neighbors[k] == 0) & (neighbors[(k + 1) % 8] != 0)).astype(np.uint8)
                              for k in range(8))
            if phase == 0:
                extra = ((p2 * p4 * p6) == 0) & ((p4 * p6 * p8) == 0)
            else:
                extra = ((p2 * p4 * p8) == 0) & ((p2 * p6 * p8) == 0)
            delete = (z[1:-1, 1:-1] != 0) & (count >= 2) & (count <= 6) & (transitions == 1) & extra
            removed += int(delete.sum())
            z[1:-1, 1:-1][delete] = 0
        if not removed:
            return z[1:-1, 1:-1] != 0
    raise RuntimeError("centerline thinning did not converge")


def longest_component(mask):
    if not np.any(mask):
        return 0
    _, _, stats, _ = cv2.connectedComponentsWithStats(np.asarray(mask, np.uint8), 8)
    return int(stats[1:, cv2.CC_STAT_AREA].max())


def local_contrast(image, mask, sigma=3.):
    """Signed foreground minus nearby outside-mask RGB, with local support."""
    outside = (~mask).astype(np.float32)
    denom = cv2.GaussianBlur(outside, (0, 0), sigma)
    backdrop = cv2.GaussianBlur(image * outside[:, :, None], (0, 0), sigma)
    backdrop /= np.maximum(denom[:, :, None], 1e-6)
    return image - backdrop, mask & (denom >= .04)


def contrast_evidence(reference, teacher, mask, min_contrast=.035, ratio=.4):
    ref, valid = local_contrast(reference, mask)
    teach, _ = local_contrast(teacher, mask)
    energy = np.sum(ref * ref, axis=2)
    agreement = np.sum(ref * teach, axis=2) / np.maximum(energy, 1e-8)
    teacher_cos = np.sum(ref * teach, axis=2) / np.maximum(
        np.sqrt(energy * np.sum(teach * teach, axis=2)), 1e-8)
    supported = valid & (np.sqrt(energy / 3.) >= min_contrast) & (agreement >= ratio) & (teacher_cos >= .5)
    return ref, supported


def contrast_retained(reference_contrast, predicted, mask, ratio=.35):
    pred, _ = local_contrast(predicted, mask)
    energy = np.sum(reference_contrast * reference_contrast, axis=2)
    dot = np.sum(reference_contrast * pred, axis=2)
    amplitude = dot / np.maximum(energy, 1e-8)
    cosine = dot / np.maximum(np.sqrt(energy * np.sum(pred * pred, axis=2)), 1e-8)
    return (amplitude >= ratio) & (cosine >= .5), np.clip(amplitude, 0, 1)


def bbox_for_mask(mask, padding=0):
    ys, xs = np.nonzero(mask)
    if not len(xs):
        raise ValueError("cannot crop an empty mask")
    return (max(0, int(xs.min()) - padding), max(0, int(ys.min()) - padding),
            min(mask.shape[1], int(xs.max()) + padding + 1), min(mask.shape[0], int(ys.max()) + padding + 1))


def crop_array(arr, box):
    x0, y0, x1, y1 = box
    return arr[y0:y1, x0:x1]


def region_metrics(reference_edges, matches, predicted_edges, valid_reference_edges, eval_mask,
                   reference_contrast, contrast_supported, retained, amplitude, skeleton, edge_amplitude=None):
    target_edges = reference_edges & eval_mask
    missing = target_edges & ~matches
    denominator = int(target_edges.sum())
    area = int(eval_mask.sum())
    extra = predicted_edges & eval_mask & ~dilate(valid_reference_edges, 2)
    contrast_count = int(contrast_supported.sum())
    thin_target = skeleton & contrast_supported
    thin_count = int(thin_target.sum())
    return dict(reference_edge_pixels=denominator,
                edge_recall=float((target_edges & matches).sum() / denominator) if denominator else None,
                mean_relative_edge_strength=float(edge_amplitude[target_edges].mean()) if denominator and edge_amplitude is not None else None,
                largest_missing_edge_fragment_px=longest_component(missing),
                unsupported_edge_pixels=int(extra.sum()),
                unsupported_edge_fraction=float(extra.sum() / max(1, int((predicted_edges & eval_mask).sum()))),
                contrast_reference_pixels=contrast_count,
                contrast_recall=float((retained & contrast_supported).sum() / contrast_count) if contrast_count else None,
                contrast_strength=float(amplitude[contrast_supported].mean()) if contrast_count else None,
                thin_reference_pixels=thin_count,
                thin_centerline_recall=float((retained & thin_target).sum() / thin_count) if thin_count else None,
                largest_missing_centerline_fragment_px=longest_component(thin_target & ~retained),
                evaluated_pixels=area)


def summarize(rows, labels):
    output = {}
    for name in labels:
        output[name] = {}
        for category in ("small_region", "boundary", "thin_region", "line_segment", "edge_tile", "reviewed_example"):
            selected = [r for r in rows if category in r["categories"]]
            stats = dict(candidates=len(selected))
            for key in ("edge_recall", "mean_relative_edge_strength", "largest_missing_edge_fragment_px", "unsupported_edge_fraction",
                        "contrast_recall", "thin_centerline_recall", "largest_missing_centerline_fragment_px"):
                values = [r["models"][name][key] for r in selected if r["models"][name][key] is not None]
                stats["mean_" + key] = float(np.mean(values)) if values else None
                stats["median_" + key] = float(np.median(values)) if values else None
            stats["low_contrast_survival_candidates"] = sum(
                r["models"][name]["contrast_reference_pixels"] >= 8 and
                r["models"][name]["contrast_recall"] < .25 for r in selected)
            output[name][category] = stats
    return output


def candidate_score(row, model, category):
    m = row["models"][model]
    if category == "small_region":
        return 1. - m["contrast_recall"] if m["contrast_reference_pixels"] >= 8 else None
    if category in ("thin_region", "line_segment"):
        return 1. - m["thin_centerline_recall"] if m["thin_reference_pixels"] >= 8 else None
    return 1. - m["edge_recall"] if m["reference_edge_pixels"] >= 12 else None


def choose_cases(rows, baseline, proposed, per_category=3):
    chosen, seen = [], set()
    for category in ("small_region", "boundary", "thin_region", "line_segment"):
        candidates = []
        for row in rows:
            if category not in row["categories"]:
                continue
            b, p = candidate_score(row, baseline, category), candidate_score(row, proposed, category)
            if b is not None and p is not None:
                candidates.append((b - p, b, row))
        for bucket, ordered, count in (
                ("proposed regression", sorted([x for x in candidates if x[0] < 0], key=lambda x: (x[0], x[2]["id"])), 2),
                ("baseline structural loss", sorted(candidates, key=lambda x: (-x[1], -x[0], x[2]["id"])), per_category),
                ("largest proposed improvement", sorted([x for x in candidates if x[0] > 0], key=lambda x: (-x[0], x[2]["id"])), per_category)):
            added = 0
            for delta, failure, row in ordered:
                key = (category, row["id"])
                if key in seen:
                    continue
                seen.add(key)
                chosen.append(dict(category=category, bucket=bucket, baseline_failure_score=failure,
                                   survival_improvement=delta, candidate_id=row["id"]))
                added += 1
                if added >= count:
                    break
    return chosen


def font(size):
    try:
        return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", size)
    except OSError:
        return ImageFont.load_default()


def write_sheets(report, selected, masks, output_dir, cell_width=270, cell_height=205):
    output_dir.mkdir(parents=True, exist_ok=True)
    labels = ["GT", "teacher"] + list(report["inputs"]["models"])
    directories = {"GT": Path(report["inputs"]["ground_truth_dir"]),
                   "teacher": Path(report["inputs"]["teacher_renders"])}
    directories.update({k: Path(v) for k, v in report["inputs"]["models"].items()})
    rows_by_id = {r["id"]: r for r in report["candidates"]}
    manifests = []
    nearest = getattr(Image, "Resampling", Image).NEAREST
    for category in ("small_region", "boundary", "thin_region", "line_segment", "reviewed_example"):
        choices = [c for c in selected if c["category"] == category]
        if not choices:
            continue
        row_h = cell_height + 68
        sheet = Image.new("RGB", (cell_width * len(labels), 78 + row_h * len(choices)), "white")
        draw = ImageDraw.Draw(sheet)
        draw.text((8, 5), category + ": GT+teacher-supported evidence; automatically ranked, physical identity unconfirmed", fill="black", font=font(15))
        draw.text((8, 27), "Includes baseline failures, improvements and regressions. Model crops are native and unannotated.", fill="black", font=font(12))
        for col, name in enumerate(labels):
            draw.text((col * cell_width + 8, 52), name, fill="black", font=font(17))
        native_root = output_dir / (category + "_native_crops")
        native_root.mkdir(parents=True, exist_ok=True)
        for row_idx, choice in enumerate(choices):
            row = rows_by_id[choice["candidate_id"]]
            box = tuple(row["crop_bbox_xyxy"])
            y = 78 + row_idx * row_h
            draw.rectangle((0, y, sheet.width - 1, y + row_h - 1), outline="#cccccc")
            desc = "{} | {} | {} | delta {:+.3f} | border margin {} px".format(choice["bucket"], row["frame"], row["id"], choice["survival_improvement"], row["min_image_margin_px"])
            draw.text((8, y + 3), desc, fill="black", font=font(12))
            native_paths = {}
            for col, name in enumerate(labels):
                im = Image.open(str(directories[name] / row["render_filename"])).convert("RGB")
                crop = im.crop(box)
                native_path = native_root / ("{:02d}_{}_{}.png".format(row_idx, row["id"], name))
                crop.save(str(native_path))
                native_paths[name] = str(native_path.resolve())
                show = crop.copy()
                if name == "GT":
                    region = masks[row["id"]]["mask"]
                    a = np.array(show)
                    a[boundary(region)] = (255, 225, 0)
                    show = Image.fromarray(a)
                scale = min((cell_width - 12) / show.width, (cell_height - 12) / show.height)
                show = show.resize((max(1, int(round(show.width * scale))), max(1, int(round(show.height * scale)))), nearest)
                sheet.paste(show, (col * cell_width + (cell_width - show.width) // 2,
                                  y + 62 + (cell_height - show.height) // 2))
                if name in row["models"]:
                    m = row["models"][name]
                    recall = m["thin_centerline_recall"] if category in ("thin_region", "line_segment") else (
                        m["contrast_recall"] if category == "small_region" else m["edge_recall"])
                    value = "n/a" if recall is None else "{:.3f}".format(recall)
                    caption = "survival {} | edge-gap {} px | extra {:.2f}".format(
                        value, m["largest_missing_edge_fragment_px"], m["unsupported_edge_fraction"])
                else:
                    caption = "native {}x{}".format(crop.width, crop.height)
                draw.text((col * cell_width + 7, y + 27), caption, fill="black", font=font(11))
            manifests.append(dict(choice, native_crops=native_paths, crop_bbox_xyxy=list(box),
                                  review_status=row["review_status"], description=row.get("description", "")))
        sheet_path = output_dir / (category + "_comparison.png")
        sheet.save(str(sheet_path))
    (output_dir / "selection.json").write_text(json.dumps(manifests, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sam_masks", type=Path, required=True)
    p.add_argument("--test_frames", type=Path, required=True)
    p.add_argument("--train_frames", type=Path, required=True)
    p.add_argument("--ground_truth_dir", type=Path, required=True)
    p.add_argument("--source_images", type=Path, required=True)
    p.add_argument("--teacher_renders", type=Path, required=True)
    p.add_argument("--model", action="append", required=True, help="LABEL=RENDERSDIR, e.g. LG10=/.../renders")
    p.add_argument("--baseline", required=True)
    p.add_argument("--proposed", required=True)
    p.add_argument("--output_dir", type=Path, required=True)
    p.add_argument("--reviewed_examples", type=Path, help="Optional JSON list of human-identified bboxes; never training input")
    p.add_argument("--filename_pattern", default="{index:05d}.png")
    p.add_argument("--min_mask_quality", type=float, default=.85)
    p.add_argument("--small_area_fraction", type=float, default=.005)
    p.add_argument("--edge_tolerance", type=int, default=2)
    p.add_argument("--canny_low", type=float, default=40.)
    p.add_argument("--canny_high", type=float, default=100.)
    p.add_argument("--tile_size", type=int, default=96)
    p.add_argument("--max_line_segments", type=int, default=120)
    p.add_argument("--crop_padding", type=int, default=20)
    p.add_argument("--cases_per_category", type=int, default=3)
    args = p.parse_args(argv)
    test, train = read_frames(args.test_frames), read_frames(args.train_frames)
    if set(test) & set(train):
        raise ValueError("train/test overlap")
    models = {}
    for spec in args.model:
        name, sep, directory = spec.partition("=")
        if not sep or not name or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.-" for c in name) or name in models or name in ("GT", "teacher"):
            raise ValueError("model names must be unique LABEL=RENDERSDIR and not GT/teacher")
        models[name] = Path(directory)
    if args.teacher_renders.resolve() == args.ground_truth_dir.resolve():
        raise ValueError("teacher must be an uncompressed render directory, not the GT directory")
    if args.baseline not in models or args.proposed not in models:
        raise ValueError("baseline/proposed must occur among model labels")
    if args.edge_tolerance < 0 or args.canny_low <= 0 or args.canny_high <= args.canny_low or args.tile_size < 16 or args.max_line_segments < 0:
        raise ValueError("invalid fixed edge parameters")
    reviewed = json.loads(args.reviewed_examples.read_text()) if args.reviewed_examples else []
    rows, masks, mapping, mask_inputs = [], {}, [], []
    for index, frame in enumerate(test):
        filename = args.filename_pattern.format(index=index, frame=frame)
        gt = rgb(args.ground_truth_dir / filename)
        source_matches = [x for x in args.source_images.iterdir() if x.stem == frame]
        if len(source_matches) != 1:
            raise ValueError("source frame is absent or ambiguous: " + frame)
        source = rgb(source_matches[0])
        if source.shape != gt.shape or float(np.max(np.abs(source - gt))) > 2. / 255.:
            raise ValueError("GT/source pixel mapping mismatch: " + frame)
        teacher = rgb(args.teacher_renders / filename)
        predictions = {name: rgb(directory / filename) for name, directory in models.items()}
        if teacher.shape != gt.shape or any(im.shape != gt.shape for im in predictions.values()):
            raise ValueError("all RGB dimensions must match native GT exactly; implicit resizing forbidden")
        mapping.append(dict(frame=frame, render_filename=filename, source=str(source_matches[0]),
                            gt_source_max_error=float(np.max(np.abs(source - gt))),
                            gt_sha256=sha256_file(args.ground_truth_dir / filename),
                            teacher_sha256=sha256_file(args.teacher_renders / filename),
                            model_sha256={name: sha256_file(directory / filename) for name, directory in models.items()}))
        gt_edges, gt_bins, gt_strength = edge_features(gt, args.canny_low, args.canny_high)
        teacher_edges, teacher_bins, teacher_strength = edge_features(teacher, args.canny_low, args.canny_high)
        stable_edges, _ = relative_gradient_matches(gt_edges, gt_bins, gt_strength, teacher_bins, teacher_strength, args.edge_tolerance, .4)
        model_edges, model_matches, model_edge_amplitude = {}, {}, {}
        for name, im in predictions.items():
            model_edges[name], bins, strength = edge_features(im, args.canny_low, args.canny_high)
            model_matches[name], model_edge_amplitude[name] = relative_gradient_matches(
                stable_edges, gt_bins, gt_strength, bins, strength, args.edge_tolerance, .35)
        sam_file = args.sam_masks / (frame + ".npz")
        with np.load(str(sam_file), allow_pickle=False) as data:
            labels, quality = data["labels"], data["region_quality"]
        if labels.shape != gt.shape[:2] or len(quality) != int(labels.max()) or np.any(labels < 0):
            raise ValueError("native SAM shape or label table mismatch")
        mask_inputs.append(dict(frame=frame, path=str(sam_file.resolve()), sha256=sha256_file(sam_file)))
        counts = np.bincount(labels.ravel())
        candidates = []
        for local in range(1, len(counts)):
            area = int(counts[local])
            if area < 16 or quality[local - 1] < args.min_mask_quality:
                continue
            mask = labels == local
            box = bbox_for_mask(mask, 2)
            region = crop_array(mask, box)
            perimeter = int(boundary(region).sum())
            plausibly_thin = area / max(1, perimeter) <= 6. and max(region.shape) >= 25
            skel = skeletonize(region) if plausibly_thin else np.zeros(region.shape, bool)
            distance = cv2.distanceTransform(np.asarray(region, np.uint8), cv2.DIST_L2, 3)
            median_halfwidth = float(np.median(distance[skel])) if skel.any() else 0.
            thin = skel.sum() >= 20 and median_halfwidth <= 4. and max(region.shape) >= 25
            categories = []
            if area / labels.size <= args.small_area_fraction:
                categories.append("small_region")
            if area >= 64:
                categories.append("boundary")
            if thin:
                categories.append("thin_region")
            if categories:
                candidates.append(dict(id=frame + "_r" + str(local), mask_box=box, mask_crop=region, categories=categories,
                                       local_region=local, mask_quality=float(quality[local - 1]),
                                       median_halfwidth_px=median_halfwidth, skeleton=skel, skeleton_bbox=box,
                                       review_status="unreviewed automatic region"))
        # Mask-free edge tiles cover real wires/railings that SAM may omit entirely.
        for y in range(0, labels.shape[0], args.tile_size):
            for x in range(0, labels.shape[1], args.tile_size):
                tile = np.zeros(labels.shape, bool)
                tile[y:y + args.tile_size, x:x + args.tile_size] = True
                if int((stable_edges & tile).sum()) >= 24:
                    candidates.append(dict(id="{}_tile_{}_{}".format(frame, x, y),
                                           mask_box=(x, y, min(x + args.tile_size, labels.shape[1]), min(y + args.tile_size, labels.shape[0])),
                                           mask_crop=np.ones((min(args.tile_size, labels.shape[0] - y), min(args.tile_size, labels.shape[1] - x)), bool),
                                           categories=["edge_tile"], review_status="unreviewed edge tile"))
        # Explicitly named line segments cover elongated geometry outside SAM.
        # They are geometric review candidates, not certified physical thin objects.
        lines = cv2.HoughLinesP(np.asarray(stable_edges, np.uint8) * 255, 1., np.pi / 180.,
                               threshold=20, minLineLength=24., maxLineGap=2.)
        line_candidates = []
        if lines is not None:
            supported_band = dilate(stable_edges, 1)
            for raw in np.asarray(lines).reshape(-1, 4):
                x0, y0, x1, y1 = [int(q) for q in raw]
                length = float(np.hypot(x1 - x0, y1 - y0))
                n = int(max(abs(x1 - x0), abs(y1 - y0))) + 1
                xs = np.rint(np.linspace(x0, x1, n)).astype(np.int32)
                ys = np.rint(np.linspace(y0, y1, n)).astype(np.int32)
                if supported_band[ys, xs].mean() >= .85:
                    line_candidates.append((length, x0, y0, x1, y1))
            line_candidates.sort(key=lambda item: (-item[0], item[1:]))
            for line_idx, (length, x0, y0, x1, y1) in enumerate(line_candidates[:args.max_line_segments]):
                bx0, by0 = max(0, min(x0, x1) - 2), max(0, min(y0, y1) - 2)
                bx1, by1 = min(labels.shape[1], max(x0, x1) + 3), min(labels.shape[0], max(y0, y1) + 3)
                region = np.zeros((by1 - by0, bx1 - bx0), np.uint8)
                cv2.line(region, (x0 - bx0, y0 - by0), (x1 - bx0, y1 - by0), 1, 1, cv2.LINE_8)
                candidates.append(dict(id="{}_line{}".format(frame, line_idx), mask_box=(bx0, by0, bx1, by1),
                    mask_crop=region != 0, skeleton=region != 0, skeleton_bbox=(bx0, by0, bx1, by1),
                    categories=["line_segment"], segment_xyxy=[x0,y0,x1,y1], segment_length_px=length,
                    review_status="unreviewed GT+teacher-supported line; physical identity unconfirmed"))
        for reviewed_idx, record in enumerate(reviewed):
            if Path(record["frame"]).stem != frame:
                continue
            x0, y0, x1, y1 = [int(k) for k in record["bbox_xyxy"]]
            if not (0 <= x0 < x1 <= labels.shape[1] and 0 <= y0 < y1 <= labels.shape[0]):
                raise ValueError("reviewed native bbox is invalid")
            mask = np.zeros(labels.shape, bool)
            mask[y0:y1, x0:x1] = True
            candidates.append(dict(id="{}_reviewed{}".format(frame, reviewed_idx), mask_box=(x0,y0,x1,y1), mask_crop=mask[y0:y1,x0:x1],
                                   categories=["reviewed_example"], review_category=record["category"],
                                   description=record.get("description", ""),
                                   review_status=record.get("review_status", "human-identified geometry, survival not confirmed")))
        for candidate in candidates:
            mask = np.zeros(labels.shape, bool)
            mx0, my0, mx1, my1 = candidate.pop("mask_box")
            mask[my0:my1, mx0:mx1] = candidate.pop("mask_crop")
            box = bbox_for_mask(mask, args.crop_padding)
            roi_mask = crop_array(mask, box)
            gt_roi, teacher_roi = crop_array(gt, box), crop_array(teacher, box)
            ref_contrast, contrast_support = contrast_evidence(gt_roi, teacher_roi, roi_mask)
            skeleton = np.zeros(roi_mask.shape, bool)
            if "skeleton" in candidate:
                full_skeleton = np.zeros(mask.shape, bool)
                sx0, sy0, sx1, sy1 = candidate.pop("skeleton_bbox")
                full_skeleton[sy0:sy1, sx0:sx1] = candidate.pop("skeleton")
                skeleton = crop_array(full_skeleton, box)
            eval_mask = roi_mask | dilate(boundary(roi_mask), 2)
            measures = {}
            for name, im in predictions.items():
                retained, amplitude = contrast_retained(ref_contrast, crop_array(im, box), roi_mask)
                measures[name] = region_metrics(crop_array(stable_edges, box), crop_array(model_matches[name], box),
                    crop_array(model_edges[name], box), crop_array(gt_edges | teacher_edges, box), eval_mask,
                    ref_contrast, contrast_support, retained, amplitude, skeleton, crop_array(model_edge_amplitude[name], box))
            if "line_segment" in candidate["categories"]:
                supported_centerline = skeleton & dilate(crop_array(stable_edges, box), 1)
                count = int(supported_centerline.sum())
                for name in predictions:
                    kept = supported_centerline & dilate(crop_array(model_matches[name], box), 1)
                    measures[name]["thin_reference_pixels"] = count
                    measures[name]["thin_centerline_recall"] = float(kept.sum() / count) if count else None
                    measures[name]["largest_missing_centerline_fragment_px"] = longest_component(supported_centerline & ~kept)
            # Objects and outlines have different denominators: boundary measurement
            # uses a fixed band around the SAM outline, not interior roof textures.
            boundary_measures = {}
            boundary_band = dilate(boundary(roi_mask), 3)
            for name in predictions:
                chosen = crop_array(stable_edges, box) & boundary_band
                denominator = int(chosen.sum())
                missing = chosen & ~crop_array(model_matches[name], box)
                boundary_measures[name] = dict(reference_edge_pixels=denominator,
                    edge_recall=float((chosen & crop_array(model_matches[name], box)).sum() / denominator) if denominator else None,
                    mean_relative_edge_strength=float(crop_array(model_edge_amplitude[name], box)[chosen].mean()) if denominator else None,
                    largest_missing_edge_fragment_px=longest_component(missing))
            tight = bbox_for_mask(mask, 0)
            margin = min(tight[0], tight[1], labels.shape[1] - tight[2], labels.shape[0] - tight[3])
            candidate.update(frame=frame, render_filename=filename, pixels=int(mask.sum()),
                             min_image_margin_px=int(margin), touches_image_boundary=margin == 0,
                             crop_bbox_xyxy=list(box), models=measures, boundary_models=boundary_measures)
            # For boundary-category ranking always use the explicitly named
            # boundary_models field; general edge scores remain available.
            rows.append(candidate)
            masks[candidate["id"]] = dict(mask=roi_mask.copy(), box=box)
        print("evaluated {}: {} candidates".format(frame, len(candidates)), flush=True)
    # Named boundary metrics replace general ROI edge metrics only in boundary
    # aggregates/ranking through explicit clones, preserving the all-region record.
    boundary_rows = []
    for row in rows:
        if "boundary" in row["categories"]:
            cloned = dict(row)
            cloned["id"] = row["id"] + "_outline"
            cloned["categories"] = ["boundary"]
            cloned["models"] = {name: dict(row["models"][name], **row["boundary_models"][name]) for name in models}
            boundary_rows.append(cloned)
            masks[cloned["id"]] = masks[row["id"]]
            row["categories"] = [c for c in row["categories"] if c != "boundary"]
    rows.extend(boundary_rows)
    report = dict(schema_version=1, protocol="held-out native RGB; GT AND uncompressed teacher supported structure; pre-SH3 comparison",
        train_frames=train, test_frames=test, source_gt_mapping=mapping,
        parameters=dict(edge_tolerance_px=args.edge_tolerance, canny_low=args.canny_low, canny_high=args.canny_high,
                        min_mask_quality=args.min_mask_quality, small_area_fraction=args.small_area_fraction,
                        contrast_min_rms=.035, contrast_survival_ratio=.35, edge_survival_relative_gradient_ratio=.35,
                        teacher_edge_relative_gradient_ratio=.4, model_edge_detector="relative oriented gradient; no Canny hysteresis", thin_max_median_halfwidth_px=4., max_line_segments_per_view=args.max_line_segments,
                        image_scale="native, no resizing"),
        inputs=dict(ground_truth_dir=str(args.ground_truth_dir.resolve()), teacher_renders=str(args.teacher_renders.resolve()),
                    models={k: str(v.resolve()) for k, v in models.items()}, masks=mask_inputs),
        comparison_reference=dict(baseline=args.baseline, proposed=args.proposed),
        summary=summarize(rows, models),
        interior_summary=summarize([r for r in rows if r["min_image_margin_px"] >= 24], models), candidates=rows,
        limitations=["Automatic regions/edge fragments/line segments are not verified physical objects; count loss only after visual confirmation.",
                     "Edge matching permits a fixed 2-pixel displacement and discretized orientation tolerance; small residual misalignment can affect gaps.",
                     "Largest missing fragment is an 8-connected reference-pixel component size, not physical gap length.",
                     "Contrast survival measures foreground evidence reproduced by the teacher; texture and shadows can be candidates.",
                     "Extra edges outside BOTH GT and teacher support are reported to reveal noise/sharpening false positives.",
                     "All test masks and reviewed test bboxes are evaluation-only; never use these records as training targets."])
    rankings = {}
    for category in ("small_region", "boundary", "thin_region", "line_segment", "edge_tile"):
        ranked = []
        for row in rows:
            if category not in row["categories"]:
                continue
            b, n = candidate_score(row, args.baseline, category), candidate_score(row, args.proposed, category)
            if b is not None and n is not None:
                ranked.append(dict(candidate_id=row["id"], baseline_failure_score=b,
                                   proposed_failure_score=n, survival_improvement=b - n,
                                   review_status=row["review_status"]))
        rankings[category] = sorted(ranked, key=lambda x: (-x["baseline_failure_score"], -x["survival_improvement"], x["candidate_id"]))
    report["rankings"] = rankings
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "visible_structure.json").write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    with open(str(args.output_dir / "all_candidates.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "frame", "categories", "model", "review_status", "edge_recall", "largest_missing_edge_fragment_px",
                         "contrast_recall", "thin_centerline_recall", "unsupported_edge_fraction"])
        for row in rows:
            for name, m in row["models"].items():
                writer.writerow([row["id"], row["frame"], "|".join(row["categories"]), name, row["review_status"],
                                 m["edge_recall"], m["largest_missing_edge_fragment_px"], m["contrast_recall"],
                                 m["thin_centerline_recall"], m["unsupported_edge_fraction"]])
    selected = choose_cases(rows, args.baseline, args.proposed, args.cases_per_category)
    for row in rows:
        if "reviewed_example" in row["categories"]:
            b, n = candidate_score(row, args.baseline, "reviewed_example"), candidate_score(row, args.proposed, "reviewed_example")
            selected.append(dict(category="reviewed_example", bucket="explicit visual review",
                baseline_failure_score=b, survival_improvement=(b-n) if b is not None and n is not None else 0.,
                candidate_id=row["id"]))
    write_sheets(report, selected, masks, args.output_dir / "evidence")
    print(json.dumps(dict(output_dir=str(args.output_dir.resolve()), summary=report["summary"], selected=len(selected)), indent=2))


if __name__ == "__main__":
    main()
