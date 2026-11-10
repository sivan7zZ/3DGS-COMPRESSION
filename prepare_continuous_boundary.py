"""Prepare independent continuous boundary priors from 61 training views.

No GPU, manual semantic annotation, test image or compression output is read.
Per-frame input/code hashes allow verified resume; shared instance, topology,
binary boundary and visibility metadata remain unchanged.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import time

import numpy as np
from PIL import Image

from utils.continuous_boundary_prior import (
    CHANNELS, accumulate_continuous, accumulate_projected, continuous_maps,
    load_kernel, reference_projected,
)
from utils.structure_scene_io import read_colmap, read_ply, sha256_file
from utils.structure_visibility import prepare_from_ply
from utils.instance_pruning import xyz_sha256


ROOT = Path(__file__).resolve().parent
DEFAULT_OUT = ROOT / 'output/city2_continuous_boundary_20261109/continuous_priors'
SOURCE = Path('/home/zhangys/gaussian-splatting（原生）')
SHARED = ROOT / 'output/city2_adaptive_constraints_20261102/shared'
INPUTS = ROOT / 'output/city2_structure_guard_20261027/inputs'
SPLIT = ROOT / 'output/city2_instance_guard_20261026/inputs'


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix='.continuous-', dir=str(path.parent))
    try:
        with os.fdopen(descriptor, 'w') as stream:
            json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
            stream.write('\n')
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def save_npz(path, **arrays):
    path = Path(path)
    temporary = path.with_name('.' + path.name + '.partial.npz')
    np.savez_compressed(temporary, **arrays)
    temporary.replace(path)


def self_check(output, kernel):
    """Run a tiny anisotropic/occluded scene against a per-pixel reference."""
    height, width = 23, 29
    projected = np.array([[11.2, 10.1, 2., .35, .08, .2, .97, 13., 10., 10.],
                          [11.2, 10.1, 3., .35, .08, .2, .97, 13., 10., 10.],
                          [17.4, 8.8, 2.5, .2, -.04, .45, .55, 12., 9., 8.]], np.float32)
    ids = np.array([0, 2, 4], np.int32)
    order = np.array([0, 2, 1], np.int32)
    y, x = np.mgrid[:height, :width]
    maps = np.stack([x / (width - 1), y / (height - 1), .2 + .5 * ((x + y) % 2)]).astype(np.float32)
    reference = reference_projected(projected, ids, order, maps, 6)
    measured = accumulate_projected(projected, ids, order, maps, 6, kernel, 4)
    for key in ('total', 'mass'):
        np.testing.assert_allclose(measured[key], reference[key], rtol=3e-6, atol=3e-6)
    result = dict(status='passed', native_footprint=True, anisotropic=True,
        occlusion_aware=True, channels=CHANNELS, image_shape=[height, width],
        max_total_absolute_error=float(np.max(np.abs(measured['total'] - reference['total']))),
        max_mass_absolute_error=float(np.max(np.abs(measured['mass'] - reference['mass']))),
        reference='Independent non-tiled per-pixel depth-sorted float64 alpha*T',
        tolerance=dict(rtol=3e-6, atol=3e-6), gpu_used=False)
    write_json(output / 'kernel_reference_validation.json', result)
    return result


def compare_visibility(total, expected, rtol=2e-5, atol=2e-5):
    delta = np.abs(np.asarray(total, np.float64) - np.asarray(expected, np.float64))
    tolerance = atol + rtol * np.abs(expected)
    bad = delta > tolerance
    result = dict(passed=not bool(bad.any()), compared_rows=int(len(total)),
        violations=int(bad.sum()), rtol=rtol, atol=atol,
        max_absolute_error=float(delta.max(initial=0)),
        mean_absolute_error=float(delta.mean()),
        same_positive_support=bool(np.array_equal(np.asarray(total) > 0, np.asarray(expected) > 0)))
    if not result['passed'] or not result['same_positive_support']:
        raise ValueError('Continuous projection differs from existing native visibility: ' + json.dumps(result))
    return result


def prepare(args):
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    for name in ('frame_mass', 'maps', 'cpu_build'):
        (out / name).mkdir(exist_ok=True)
    shared_path = args.shared_metadata / 'adaptive_metadata.npz'
    shared_manifest = read_json(args.shared_metadata / 'metadata_manifest.json')
    split = read_json(args.split)
    frames = sorted(Path(name).stem for name in args.train_frames.read_text().splitlines() if name.strip())
    if frames != sorted(split['train']) or len(frames) != len(set(frames)) or set(frames) & set(split['test']):
        raise ValueError('Declared training/test split differs')
    if frames != shared_manifest['train_frames'] or split['test'] != shared_manifest['test_frames_excluded']:
        raise ValueError('Continuous priors must use exactly the shared metadata train-only split')
    source_sha = sha256_file(args.source_ply)
    if source_sha != shared_manifest['source_ply_sha256']:
        raise ValueError('Source PLY does not match unchanged shared metadata')
    ply, _ = read_ply(args.source_ply)
    n = len(ply)
    xyz = np.column_stack([ply[key] for key in ('x', 'y', 'z')]).astype(np.float32)
    xyz_hash = xyz_sha256(xyz)
    if n != shared_manifest['source_count'] or xyz_hash != shared_manifest['source_xyz_sha256']:
        raise ValueError('Source Gaussian rows changed')
    with np.load(shared_path) as original:
        visibility = np.array(original['visibility'])
        original_ids = np.array(original['original_ids'])
        if not np.array_equal(original_ids, np.arange(n, dtype=np.int64)):
            raise ValueError('Shared source-row IDs changed')
    shared_sha = sha256_file(shared_path)
    cameras = read_colmap(args.colmap)
    implementation = {str(path.relative_to(ROOT)): sha256_file(path) for path in (
        Path(__file__).resolve(), ROOT / 'utils/continuous_boundary_prior.py', ROOT / 'utils/adaptive_metadata.py',
        ROOT / 'utils/structure_visibility.py')}
    kernel = load_kernel(out / 'cpu_build')
    self_check(out, kernel)
    gaussians = prepare_from_ply(ply)
    all_frames = frames[:]
    if args.limit_views:
        frames = frames[:args.limit_views]
    total = np.zeros(n, np.float64)
    mass = np.zeros((n, 3), np.float64)
    records = {}
    start_all = time.time()
    for index, frame in enumerate(frames):
        start = time.time()
        camera = cameras[frame]
        camera_hash = hashlib.sha256((json.dumps(camera['camera'], sort_keys=True) +
            camera['R'].tobytes().hex() + camera['t'].tobytes().hex()).encode()).hexdigest()
        depth_record_path = args.visibility_cache / (frame + '_visibility.json')
        depth_record = read_json(depth_record_path)
        if depth_record['camera_sha256'] != camera_hash or depth_record['source_xyz_sha256'] != xyz_hash:
            raise ValueError('Cached original depth belongs to different Gaussian rows or camera')
        sam_path = args.sam_masks / (frame + '.npz')
        image_path = args.images / camera['name']
        depth_path = args.visibility_cache / (frame + '_visibility.npz')
        old_receipt = read_json(args.shared_metadata / 'mask_support' / (frame + '.json'))
        fingerprint = dict(source_ply_sha256=source_sha, source_xyz_sha256=xyz_hash,
            shared_metadata_sha256=shared_sha, camera_sha256=camera_hash,
            sam_sha256=sha256_file(sam_path), image_sha256=sha256_file(image_path),
            depth_sha256=sha256_file(depth_path), depth_record_sha256=sha256_file(depth_record_path),
            algorithm='continuous SAM Gaussian-distance sigma2 + normalized Sobel depth/RGB; one native anisotropic alpha*T traversal v1',
            sam_sigma_px=float(args.sam_sigma_px), implementation_sha256=implementation,
            runtime=dict(python_executable=sys.executable, numpy_version=np.__version__))
        for key in ('source_ply_sha256', 'camera_sha256', 'sam_sha256', 'image_sha256', 'depth_sha256'):
            if fingerprint[key] != old_receipt['fingerprint'][key]:
                raise ValueError('Input differs from reviewed shared metadata: ' + frame + '/' + key)
        mass_path, record_path = out / 'frame_mass' / (frame + '.npz'), out / 'frame_mass' / (frame + '.json')
        cached = mass_path.exists() and record_path.exists()
        if cached:
            record = read_json(record_path)
            if record['fingerprint'] != fingerprint or sha256_file(mass_path) != record['mass_sha256']:
                raise ValueError('Existing continuous frame cache has changed inputs/code/content: ' + frame)
            if sha256_file(out / 'maps' / (frame + '.npz')) != record['maps_sha256']:
                raise ValueError('Continuous per-frame maps changed')
            with np.load(mass_path) as data:
                frame_total, frame_mass = np.array(data['total']), np.array(data['mass'])
        else:
            with np.load(sam_path) as data:
                labels = np.array(data['labels'])
            with np.load(depth_path) as data:
                depth, alpha = np.array(data['depth']), np.array(data['alpha'])
            rgb = np.asarray(Image.open(image_path).convert('RGB'), np.float32) / 255.
            if labels.shape != (camera['camera']['height'], camera['camera']['width']):
                raise ValueError('SAM map/native camera shape mismatch')
            maps = continuous_maps(rgb, labels, depth, alpha, args.sam_sigma_px)
            result = accumulate_continuous(gaussians, camera, maps, kernel, args.threads)
            frame_total, frame_mass = result['total'], result['mass']
            with np.load(args.shared_metadata / 'mask_support' / (frame + '.npz')) as data:
                projection_audit = compare_visibility(frame_total, data['total'])
            maps_path = out / 'maps' / (frame + '.npz')
            save_npz(maps_path, **{name: maps[channel] for channel, name in enumerate(CHANNELS)})
            save_npz(mass_path, total=frame_total, mass=frame_mass)
            record = dict(fingerprint=fingerprint, seconds=time.time() - start,
                mass_sha256=sha256_file(mass_path), maps_sha256=sha256_file(maps_path),
                tile_references=result['tile_references'],
                visible_gaussians=int(np.count_nonzero(frame_total)),
                total_contribution=float(frame_total.sum(dtype=np.float64)),
                projection_visibility_audit=projection_audit,
                map_statistics={name: dict(min=float(maps[c].min()), max=float(maps[c].max()),
                    mean=float(maps[c].mean(dtype=np.float64)), fractional_pixels=int(np.count_nonzero((maps[c] > 0) & (maps[c] < 1))))
                    for c, name in enumerate(CHANNELS)})
            write_json(record_path, record)
        total += frame_total
        mass += frame_mass
        records[frame] = record
        status = dict(status='rasterizing', completed_views=index + 1, train_views=len(all_frames),
            last_frame=frame, seconds=time.time() - start_all,
            estimated_total_seconds=(time.time() - start_all) * len(all_frames) / (index + 1))
        write_json(out / 'preprocessing_status.json', status)
        print(json.dumps(dict(frame=frame, index=index + 1, views=len(frames), cache=cached,
            seconds=round(time.time() - start, 3), estimated_total_seconds=round(status['estimated_total_seconds'], 1))), flush=True)
    if args.limit_views:
        write_json(out / 'benchmark.json', dict(status='benchmark_only', views=len(frames),
            seconds=time.time() - start_all, source_count=n, full_train_views=len(all_frames)))
        return
    final_projection_audit = compare_visibility(total, visibility)
    strength = np.divide(mass, total[:, None], out=np.zeros_like(mass), where=total[:, None] > 0)
    if not np.isfinite(strength).all() or np.any(strength < -1e-6) or np.any(strength > 1 + 2e-5):
        raise ValueError('Projected continuous strengths are outside [0,1]')
    strength = np.clip(strength, 0, 1).astype(np.float32)
    sidecar = out / 'continuous_metadata.npz'
    save_npz(sidecar, original_ids=original_ids, visibility=visibility,
        source_ply_sha256=np.asarray(source_sha), source_xyz_sha256=np.asarray(xyz_hash),
        **{name: strength[:, channel] for channel, name in enumerate(CHANNELS)})
    manifest = dict(schema_version=1, schema='continuous_train_only_boundary_prior_v1', status='complete',
        source_ply=str(args.source_ply.resolve()), source_ply_sha256=source_sha,
        source_xyz_sha256=xyz_hash, source_count=n, shared_metadata=str(shared_path.resolve()),
        shared_metadata_sha256=shared_sha, sidecar=str(sidecar.resolve()), sidecar_sha256=sha256_file(sidecar),
        train_frames=all_frames, test_frames_excluded=split['test'],
        channels=list(CHANNELS), sam_sigma_px=float(args.sam_sigma_px),
        map_definitions=dict(sam='Four-neighbor SAM label discontinuities on both adjacent pixels; precise Euclidean distance d; exp(-d*d/(2*sigma*sigma)); zero for an edge-free label map; no dilation/hard threshold.',
            depth='Unchanged normalized_sobel: 3x3 Sobel/8 magnitude, eroded valid finite(depth)&alpha>.5; divide by valid q95, clamp [0,1].',
            rgb='Unchanged normalized_sobel on cv2 RGB-to-gray, q95 normalization, clamp [0,1].'),
        projection='Sum all training-view/pixel alpha*T * continuous map / sum alpha*T, using original native anisotropic source footprints, opacity>=1/255 and T>=1e-4; not dominant-only and not Gaussian-center sampling.',
        visibility_semantics='Bitwise copy of unchanged shared.visibility for source identity. Continuous strengths use newly accumulated alpha*T denominator; its agreement with shared denominator is audited separately.',
        projection_visibility_audit=final_projection_audit,
        binary_boundary_mask_modified=False, instance_metadata_modified=False, topology_metadata_modified=False,
        no_semantic_annotations=True, gpu_used=False, implementation_sha256=implementation,
        runtime=dict(python_executable=sys.executable, numpy_version=np.__version__),
        preprocessing_seconds=time.time() - start_all,
        component_statistics={name: dict(dtype=str(strength.dtype), min=float(strength[:, c].min()),
            max=float(strength[:, c].max()), mean=float(strength[:, c].mean(dtype=np.float64)),
            positive_rows=int(np.count_nonzero(strength[:, c])),
            fractional_rows=int(np.count_nonzero((strength[:, c] > 0) & (strength[:, c] < 1))))
            for c, name in enumerate(CHANNELS)}, frames=records)
    write_json(out / 'metadata_manifest.json', manifest)
    write_json(out / 'preprocessing_status.json', dict(status='complete', completed_views=len(all_frames),
        train_views=len(all_frames), sidecar_sha256=manifest['sidecar_sha256'], seconds=time.time() - start_all))
    print(json.dumps({key: manifest[key] for key in ('status', 'source_count', 'sidecar', 'sidecar_sha256', 'component_statistics', 'projection_visibility_audit', 'preprocessing_seconds')}, indent=2), flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source_ply', type=Path, default=SOURCE / 'output/city2/point_cloud/iteration_30000/point_cloud.ply')
    parser.add_argument('--images', type=Path, default=SOURCE / 'data/city/images')
    parser.add_argument('--colmap', type=Path, default=SOURCE / 'data/city/sparse/0')
    parser.add_argument('--sam_masks', type=Path, default=INPUTS / 'sam_masks')
    parser.add_argument('--visibility_cache', type=Path, default=INPUTS / 'structure/visibility_cache')
    parser.add_argument('--shared_metadata', type=Path, default=SHARED)
    parser.add_argument('--train_frames', type=Path, default=SPLIT / 'train_frames.txt')
    parser.add_argument('--split', type=Path, default=SPLIT / 'split.json')
    parser.add_argument('--output_dir', type=Path, default=DEFAULT_OUT)
    parser.add_argument('--sam_sigma_px', type=float, default=2.)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--limit_views', type=int, default=0, help='Benchmark first N views without emitting final sidecar; cached frames can be resumed by the full run.')
    prepare(parser.parse_args(argv))


if __name__ == '__main__':
    main()
