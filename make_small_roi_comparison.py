#!/usr/bin/env python3
"""Make transparent qualitative comparisons from held-out small-ROI metrics.

Includes improvement extremes, representatives near the median PSNR change, and
at least two degradation cases when available. Native-resolution crops and a
machine-readable selection manifest accompany the contact sheet.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def select_cases(regions, control_label, protected_label, improved=2, median=2, degraded=2):
    """Deterministic selection; do not replace negative examples with successes."""
    if min(improved, median, degraded) < 0:
        raise ValueError("selection counts must be nonnegative")
    if degraded < 2:
        raise ValueError("at least two degradation slots are required")
    scored = []
    for r in regions:
        delta = float(r["models"][protected_label]["psnr"] - r["models"][control_label]["psnr"])
        if not np.isfinite(delta):
            raise ValueError("PSNR changes must be finite")
        scored.append((delta, str(r["frame"]), int(r["local_region"]), r))
    if not scored:
        return [], None
    center = float(np.median([r[0] for r in scored]))
    chosen = []
    seen = set()

    def add(candidates, limit, bucket):
        count = 0
        for delta, frame, local, row in candidates:
            key = (frame, local)
            if key in seen:
                continue
            seen.add(key)
            chosen.append(dict(group=bucket, psnr_delta=delta, region=row))
            count += 1
            if count == limit:
                break

    if improved:
        add(sorted([r for r in scored if r[0] > 0], key=lambda r: (-r[0], r[1], r[2])), improved, "largest improvement")
    # Reserve negative cases before choosing median representatives, so a median
    # near a negative extreme cannot consume one of the required worst examples.
    negative = sorted([r for r in scored if r[0] < 0], key=lambda r: (r[0], r[1], r[2]))
    add(negative, degraded, "degradation")
    if median:
        add(sorted(scored, key=lambda r: (abs(r[0] - center), r[1], r[2])), median, "near overall median")
    order = {"largest improvement": 0, "near overall median": 1, "degradation": 2}
    chosen.sort(key=lambda r: order[r["group"]])
    return chosen, center


def region_bbox(labels, local_region, padding):
    ys, xs = np.nonzero(labels == local_region)
    if not len(xs):
        raise ValueError("requested local region is absent from SAM labels")
    return (max(0, int(xs.min()) - padding), max(0, int(ys.min()) - padding),
            min(labels.shape[1], int(xs.max()) + padding + 1),
            min(labels.shape[0], int(ys.max()) + padding + 1))


def font(size):
    try:
        return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", size)
    except OSError:
        return ImageFont.load_default()


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--report", type=Path, required=True, help="evaluate_small_roi.py JSON")
    p.add_argument("--sam_masks", type=Path, required=True)
    p.add_argument("--ground_truth_dir", type=Path)
    p.add_argument("--control_renders", type=Path)
    p.add_argument("--protected_renders", type=Path)
    p.add_argument("--control_label", default="control")
    p.add_argument("--protected_label", default="protected")
    p.add_argument("--output", type=Path, required=True, help="PNG contact sheet; native crops saved alongside")
    p.add_argument("--improved", type=int, default=2)
    p.add_argument("--median", type=int, default=2)
    p.add_argument("--degraded", type=int, default=2)
    p.add_argument("--padding", type=int, default=24)
    p.add_argument("--cell_width", type=int, default=320)
    p.add_argument("--cell_height", type=int, default=230)
    args = p.parse_args(argv)
    if args.padding < 0 or args.cell_width < 120 or args.cell_height < 100:
        p.error("invalid crop padding or contact-sheet cell size")
    report = json.loads(args.report.read_text())
    control_dir = args.control_renders or Path(report["inputs"]["models"][args.control_label])
    protected_dir = args.protected_renders or Path(report["inputs"]["models"][args.protected_label])
    gt_dir = args.ground_truth_dir or Path(report["inputs"]["ground_truth_dir"])
    chosen, center = select_cases(report["regions"], args.control_label, args.protected_label,
                                 args.improved, args.median, args.degraded)
    if not chosen:
        raise ValueError("no small ROIs available in report")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    crop_root = args.output.parent / (args.output.stem + "_native_crops")
    crop_root.mkdir(parents=True, exist_ok=True)
    header_height, row_label_height, footer_height = 90, 53, 54
    row_height = args.cell_height + row_label_height
    sheet = Image.new("RGB", (3 * args.cell_width, header_height + row_height * len(chosen) + footer_height), "white")
    draw = ImageDraw.Draw(sheet)
    draw.text((12, 8), "Held-out automatic small SAM regions (not physical-instance ground truth)", fill="black", font=font(17))
    draw.text((12, 32), "Selection: largest improvement, near overall median, and worst degradation", fill="black", font=font(14))
    for column, title in enumerate(("Ground truth RGB", "Global SPS control", "Instance-protected")):
        draw.text((column * args.cell_width + 12, 60), title, fill="black", font=font(17))
    cache = {}
    records = []
    for row_index, choice in enumerate(chosen):
        region = choice["region"]
        frame, local = str(region["frame"]), int(region["local_region"])
        if frame not in cache:
            with np.load(args.sam_masks / (frame + ".npz"), allow_pickle=False) as z:
                cache[frame] = z["labels"]
        labels = cache[frame]
        bbox = region_bbox(labels, local, args.padding)
        y = header_height + row_height * row_index
        desc = "{} | {} / region {} | delta PSNR {:+.3f} dB".format(choice["group"], frame, local, choice["psnr_delta"])
        draw.rectangle((0, y, sheet.width, y + row_height - 1), outline="#cccccc")
        draw.text((12, y + 4), desc, fill="black", font=font(14))
        native_files = {}
        for column, (title, directory, model_label) in enumerate((
                ("gt", gt_dir, None), ("control", control_dir, args.control_label),
                ("protected", protected_dir, args.protected_label))):
            source = Image.open(directory / region["render_filename"]).convert("RGB")
            if source.size != (labels.shape[1], labels.shape[0]):
                raise ValueError("SAM labels and rendered image shapes differ")
            crop = source.crop(bbox)
            native_path = crop_root / ("{:02d}_{}_r{}_{}.png".format(row_index, frame, local, title))
            crop.save(native_path)
            native_files[title] = str(native_path.resolve())
            max_width, max_height = args.cell_width - 16, args.cell_height - 16
            scale = min(max_width / crop.width, max_height / crop.height)
            nearest = getattr(Image, "Resampling", Image).NEAREST
            display = crop.resize((max(1, round(crop.width * scale)), max(1, round(crop.height * scale))), nearest)
            x = column * args.cell_width + (args.cell_width - display.width) // 2
            image_y = y + row_label_height + (args.cell_height - display.height) // 2
            sheet.paste(display, (x, image_y))
            label = "native crop {} x {}".format(crop.width, crop.height)
            if model_label:
                label += " | PSNR {:.3f} dB".format(region["models"][model_label]["psnr"])
            draw.text((column * args.cell_width + 10, y + 27), label, fill="black", font=font(12))
        records.append(dict(group=choice["group"], frame=frame, local_region=local,
                            pixels=region["pixels"], area_fraction=region["area_fraction"],
                            bbox_xyxy=list(bbox), psnr_delta=choice["psnr_delta"],
                            model_metrics=region["models"], native_crops=native_files))
    footer_y = sheet.height - footer_height
    draw.text((12, footer_y + 5), "Crops use the same native-pixel bbox + padding; contact sheet enlarged with nearest-neighbor.", fill="black", font=font(12))
    draw.text((12, footer_y + 25), "All-region numeric report is authoritative; selected crops do not estimate detection recall.", fill="black", font=font(12))
    sheet.save(args.output)
    manifest = dict(source_report=str(args.report.resolve()), image=str(args.output.resolve()),
                    median_psnr_delta=center, selected=records,
                    total_report_regions=len(report["regions"]),
                    available_degradation_cases=sum(r["models"][args.protected_label]["psnr"] <
                                                    r["models"][args.control_label]["psnr"] for r in report["regions"]),
                    native_crops_resize=False, contact_sheet_resampling="nearest",
                    disclaimer="Automatic small SAM regions are not verified physical-instance ground truth.")
    manifest_path = args.output.with_suffix(".selection.json")
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    print(json.dumps(dict(image=str(args.output.resolve()), manifest=str(manifest_path.resolve()), cases=len(chosen)), indent=2))


if __name__ == "__main__":
    main()
