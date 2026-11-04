"""City human-label registration and a fixed held-out semantic protocol.

This module has no torch dependency. Class zero is unannotated, never background.
"""
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
from PIL import Image


CLASS_NAMES = ["unannotated", "Building", "Tree", "Road", "Vehicle", "Grass", "Ground"]
INNER_TRAIN = ["frame_%03d" % i for i in [7, 14, 28, 35, 49, 63]]
INNER_VAL = ["frame_021", "frame_042"]
TEST = ["frame_000", "frame_056"]


def sha256_file(path):
    digest = hashlib.sha256()
    with open(str(path), "rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def array_sha256(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def combine_binary_masks(masks):
    """Accept six disjoint bool masks, return IDs 1..6; leave gaps at zero."""
    masks = [np.asarray(mask, dtype=bool) for mask in masks]
    if len(masks) != 6 or len({mask.shape for mask in masks}) != 1:
        raise ValueError("Expected six masks with identical shapes")
    occupied = np.zeros(masks[0].shape, dtype=bool)
    labels = np.zeros(masks[0].shape, dtype=np.uint8)
    for class_id, mask in enumerate(masks, 1):
        if np.any(occupied & mask):
            raise ValueError("City class masks overlap")
        labels[mask] = class_id
        occupied |= mask
    return labels


def make_registration_maps(distorted, undistorted):
    """COLMAP OPENCV distorted input -> PINHOLE render-pixel coordinates."""
    import cv2
    if distorted.model != "OPENCV" or undistorted.model != "PINHOLE":
        raise ValueError("Expected actual City's OPENCV -> PINHOLE camera pair")
    a, b = np.asarray(distorted.params), np.asarray(undistorted.params)
    old_k = np.array([[a[0], 0, a[2]], [0, a[1], a[3]], [0, 0, 1]], np.float64)
    new_k = np.array([[b[0], 0, b[2]], [0, b[1], b[3]], [0, 0, 1]], np.float64)
    return cv2.initUndistortRectifyMap(
        old_k, a[4:8], None, new_k,
        (int(undistorted.width), int(undistorted.height)), cv2.CV_32FC1)


def register_labels(labels, map_x, map_y):
    import cv2
    return cv2.remap(np.asarray(labels, dtype=np.uint8), map_x, map_y,
                     cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def _camera_record(camera):
    return {"model": camera.model, "width": int(camera.width),
            "height": int(camera.height), "params": list(map(float, camera.params))}


def prepare_dataset(output, city_source, segmentation_source, repo_root):
    import cv2
    output, city_source = Path(output).resolve(), Path(city_source).resolve()
    segmentation_source = Path(segmentation_source).resolve()
    output.mkdir(parents=True, exist_ok=True)
    (output / "labels").mkdir(exist_ok=True)
    loader = Path(repo_root) / "scene" / "colmap_loader.py"
    spec = importlib.util.spec_from_file_location("city_semantic_colmap", str(loader))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    old_path = city_source / "distorted/sparse/0/cameras.bin"
    new_path = city_source / "sparse/0/cameras.bin"
    old_cameras = module.read_intrinsics_binary(str(old_path))
    new_cameras = module.read_intrinsics_binary(str(new_path))
    if len(old_cameras) != 1 or len(new_cameras) != 1:
        raise ValueError("This protocol expects City's shared single camera")
    old, new = list(old_cameras.values())[0], list(new_cameras.values())[0]
    map_x, map_y = make_registration_maps(old, new)
    maps_path = output / "registration_maps.npz"
    np.savez_compressed(str(maps_path), map_x=map_x, map_y=map_y)
    available = sorted(path.name for path in segmentation_source.iterdir() if path.is_dir())
    expected = sorted(INNER_TRAIN + INNER_VAL + TEST)
    if available != expected:
        raise ValueError("Annotation frame inventory changed; review the split explicitly")
    records = []
    for name in available:
        folder = segmentation_source / name
        raw_path = city_source / "input" / (name + ".png")
        annotation_rgb = folder / "frame.png"
        rgb_path = city_source / "images" / (name + ".png")
        if sha256_file(raw_path) != sha256_file(annotation_rgb):
            raise ValueError("Annotation RGB differs from actual COLMAP input: " + name)
        raw = np.asarray(Image.open(str(raw_path)).convert("RGB"))
        rgb = np.asarray(Image.open(str(rgb_path)).convert("RGB"))
        if raw.shape[:2] != (old.height, old.width) or rgb.shape[:2] != (new.height, new.width):
            raise ValueError("RGB shape differs from registered COLMAP cameras")
        masks = []
        mask_hashes = {}
        for class_name in CLASS_NAMES[1:]:
            path = folder / (class_name + ".png")
            im = np.asarray(Image.open(str(path)).convert("RGB"))
            if im.shape[:2] != raw.shape[:2] or not np.all(im[..., 0] == im[..., 1]) or not np.all(im[..., 1] == im[..., 2]):
                raise ValueError("Expected grayscale binary masks in RGBA/RGB PNG")
            if not set(np.unique(im[..., 0]).tolist()).issubset({0, 255}):
                raise ValueError("Class mask is not binary")
            masks.append(im[..., 0] > 127)
            mask_hashes[class_name] = sha256_file(path)
        labels_raw = combine_binary_masks(masks)
        union = np.asarray(Image.open(str(folder / "mask.png")).convert("L")) > 127
        if not np.array_equal(labels_raw > 0, union):
            raise ValueError("mask.png must equal annotated class union")
        labels = register_labels(labels_raw, map_x, map_y)
        label_path = output / "labels" / (name + ".png")
        Image.fromarray(labels, mode="L").save(str(label_path))
        remapped_rgb = cv2.remap(raw, map_x, map_y, cv2.INTER_LINEAR)
        error = np.abs(remapped_rgb.astype(np.float32) - rgb.astype(np.float32))
        split = "test" if name in TEST else ("inner_val" if name in INNER_VAL else "inner_train")
        records.append({"frame": name, "split": split, "rgb_path": str(rgb_path),
                        "rgb_sha256": sha256_file(rgb_path), "raw_rgb_sha256": sha256_file(raw_path),
                        "annotation_json_sha256": sha256_file(folder / "frame.json"),
                        "source_mask_sha256": mask_hashes, "label_path": str(label_path),
                        "label_sha256": sha256_file(label_path), "shape": list(labels.shape),
                        "class_pixel_counts": np.bincount(labels.ravel(), minlength=7).tolist(),
                        "registration_rgb_mae_255": float(error.mean()),
                        "registration_rgb_max_error_255": float(error.max())})
    manifest = {"schema": "city_semantic_registered_v1", "classes": CLASS_NAMES,
                "ignore_label": 0, "num_predictive_classes": 6,
                "city_source": str(city_source), "segmentation_source": str(segmentation_source),
                "registration": {"method": "actual City COLMAP OPENCV to PINHOLE, nearest labels",
                                 "distorted_camera": _camera_record(old), "render_camera": _camera_record(new),
                                 "distorted_cameras_sha256": sha256_file(old_path),
                                 "render_cameras_sha256": sha256_file(new_path),
                                 "maps_path": str(maps_path), "maps_file_sha256": sha256_file(maps_path),
                                 "map_x_array_sha256": array_sha256(map_x), "map_y_array_sha256": array_sha256(map_y)},
                "splits": {"inner_train": INNER_TRAIN, "inner_val": INNER_VAL,
                           "refit_train": sorted(INNER_TRAIN + INNER_VAL), "test": TEST},
                "protocol": "Select step using six original RGB labeled train frames and two inner validation frames. Refit all eight non-test labeled original RGB frames from same ImageNet initialization/seed for selected steps. Test labels never choose weights/hyperparameters. Semantic test support is only two of the nine 3DGS held-out views.",
                "frames": records}
    path = output / "dataset_manifest.json"
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    return path
