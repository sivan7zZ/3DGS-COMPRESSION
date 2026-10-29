#!/usr/bin/env python3
"""Audit retained SOURCE alpha*T support; this is not optimized visibility.

The RGB render of the final optimized model must be assessed independently.
This diagnostic makes explicit how much of the original dominant-contributor
mass is represented by fixed anchors or compressed-model source lineage.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from utils.structure_pruning import StructurePruningState
from utils.structure_scene_io import read_colmap


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("structure_priors", "structure_dir", "images", "colmap", "output_dir"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--selected_metadata", type=Path)
    args = p.parse_args(argv)
    state = StructurePruningState.load(args.structure_priors)
    selected = state.anchors
    if args.selected_metadata:
        with np.load(args.selected_metadata, allow_pickle=False) as data:
            selected = np.asarray(data["original_ids"], dtype=np.int64)
            if str(data["source_xyz_sha256"].item()) != state.source_xyz_sha256:
                raise ValueError("Selected metadata belongs to a different source model")
    frames = state.manifest["train_frames"]
    cameras = read_colmap(args.colmap)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    chosen = {frames[index] for index in (0, len(frames)//2, len(frames)-1)}
    details, totals, panels = {}, {}, []
    min_contribution = state.manifest["limits"]["min_contribution"]
    min_alpha = state.manifest["limits"]["min_alpha"]
    for frame in frames:
        with np.load(args.structure_dir / "visibility_cache" / (frame + "_visibility.npz"), allow_pickle=False) as data:
            ids, mass, alpha = np.array(data["id_map"]), np.array(data["contribution"]), np.array(data["alpha"])
        with np.load(args.structure_dir / "roi" / (frame + ".npz"), allow_pickle=False) as data:
            masks = dict(small_regions=np.asarray(data["weights"]) > 0,
                         boundary_segments=np.asarray(data["boundary_mask"], dtype=bool),
                         thin_segments=np.asarray(data["thin_skeleton"], dtype=bool))
        valid = (ids >= 0) & (mass >= min_contribution) & (alpha >= min_alpha)
        retained = np.isin(ids, selected) & valid
        details[frame] = {}
        for name, mask in masks.items():
            mask = mask & valid
            total_mass = float(mass[mask].sum(dtype=np.float64))
            kept_mass = float(mass[mask & retained].sum(dtype=np.float64))
            record = dict(source_dominant_mass=total_mass, retained_source_dominant_mass=kept_mass,
                          source_mass_retention=kept_mass / max(total_mass, 1e-12),
                          critical_pixels=int(mask.sum()), retained_dominant_pixels=int((mask & retained).sum()))
            details[frame][name] = record
            aggregate = totals.setdefault(name, dict(source_dominant_mass=0., retained_source_dominant_mass=0., critical_pixels=0, retained_dominant_pixels=0))
            for key in aggregate:
                aggregate[key] += record[key]
        if frame in chosen:
            rgb = np.asarray(Image.open(args.images / cameras[frame]["name"]).convert("RGB"))
            critical = np.logical_or.reduce(list(masks.values())) & valid
            colors = np.asarray(rgb, dtype=np.float32) * .35
            colors[masks["small_regions"] & valid] = [.1 * 255, .8 * 255, .8 * 255]
            colors[masks["boundary_segments"] & valid] = [.2 * 255, .5 * 255, 255]
            colors[masks["thin_segments"] & valid] = [255, .6 * 255, .1 * 255]
            coverage = np.asarray(rgb, dtype=np.float32) * .35
            coverage[critical & ~retained] = [240, 70, 60]
            coverage[critical & retained] = [70, 240, 90]
            height, width = ids.shape
            canvas = Image.new("RGB", (width * 3, height + 40), "white")
            canvas.paste(Image.fromarray(rgb), (0, 40))
            canvas.paste(Image.fromarray(colors.astype(np.uint8)), (width, 40))
            canvas.paste(Image.fromarray(coverage.astype(np.uint8)), (2 * width, 40))
            draw = ImageDraw.Draw(canvas)
            font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
            font = ImageFont.truetype(font_path, 22) if Path(font_path).is_file() else ImageFont.load_default()
            for x, text in ((10, frame + " / training RGB"), (width + 10, "critical: cyan=small blue=boundary orange=thin"),
                            (2 * width + 10, "SOURCE support: green=retained red=not selected")):
                draw.text((x, 8), text, font=font, fill="black")
            canvas.save(args.output_dir / (frame + "_source_support.png"))
            panels.append(canvas)
    for record in totals.values():
        record["source_mass_retention"] = record["retained_source_dominant_mass"] / max(record["source_dominant_mass"], 1e-12)
    report = dict(scope="Source max(alpha*T) support, not rendered visibility after optimization",
                  selected_mode="compressed original_ids" if args.selected_metadata else "immutable anchors only",
                  selected_source_atoms=len(selected), train_frames=frames,
                  cell_coverage=state.audit(selected, require_anchors=False), totals=totals, per_frame=details,
                  representative_training_frames=sorted(chosen),
                  warning="Red pixels may still be reconstructed by other non-dominant Gaussians; green source pixels may fade or move during optimization. Assess final renders separately.")
    (args.output_dir / "source_support.json").write_text(json.dumps(report, indent=2) + "\n")
    if panels:
        montage = Image.new("RGB", (panels[0].width, sum(panel.height for panel in panels)), "white")
        top = 0
        for panel in panels:
            montage.paste(panel, (0, top))
            top += panel.height
        montage.save(args.output_dir / "source_support_overview.png")
    print(json.dumps({"selected_source_atoms": len(selected), "totals": totals}, indent=2), flush=True)


if __name__ == "__main__":
    main()
