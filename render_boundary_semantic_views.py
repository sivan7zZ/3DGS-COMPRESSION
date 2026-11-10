"""Render exactly eight annotated training views with the common SH3 renderer.

This is a downstream auxiliary evaluation. It does not train or modify models.
The nine held-out-view visual metrics remain the independent main comparison.
Run this GPU phase only after the compression queue finishes.
"""
import argparse
import hashlib
from pathlib import Path

from PIL import Image

from evaluate_boundary_weighted_semantics import (
    DEFAULT_ROOT, MODEL_SOURCE, NON_TEST, ROOT, TEST, camera_fingerprints,
    read_json, replace_json, validated_models,
)
from utils.city_semantic_data import sha256_file


def pixels_sha256(path):
    with Image.open(path) as image:
        return hashlib.sha256(image.convert('RGB').tobytes()).hexdigest()


def select_auxiliary_cameras(cameras):
    by_name = {Path(camera.image_name).stem: camera for camera in cameras}
    if len(by_name) != len(cameras):
        raise ValueError('Duplicate COLMAP camera names')
    if not set(NON_TEST) <= set(by_name):
        raise ValueError('An annotated auxiliary camera is absent from the actual model')
    selected = [by_name[name] for name in NON_TEST]
    if any(camera.is_test for camera in selected):
        raise ValueError('Auxiliary frame unexpectedly belongs to independent test cameras')
    return selected


def cached_receipt_valid(receipt, fingerprint):
    if any(receipt.get(key) != value for key, value in fingerprint.items()):
        return False
    if sorted(receipt.get('frames', {})) != NON_TEST:
        return False
    anchor = receipt.get('native_test_anchor', {})
    if not anchor.get('pixel_identical'):
        return False
    if sha256_file(anchor['reference']) != anchor['reference_sha256']:
        raise ValueError('Native test anchor changed after auxiliary rendering')
    if pixels_sha256(anchor['anchor']) != anchor['pixel_sha256']:
        raise ValueError('Auxiliary native-test anchor changed')
    for row in receipt['frames'].values():
        if not Path(row['render_path']).is_file() or sha256_file(row['render_path']) != row['render_sha256']:
            raise ValueError('Cached auxiliary render content changed')
        if sha256_file(row['gt_path']) != row['gt_sha256']:
            raise ValueError('Auxiliary GT original RGB changed')
    return True


def run(output, resume=False):
    import numpy as np
    import torch
    import torchvision
    from gaussian_renderer import render
    from scene import GaussianModel
    from scene.colmap_loader import read_extrinsics_binary, read_intrinsics_binary
    from scene.dataset_readers import readColmapCameras
    from utils.camera_utils import cameraList_from_camInfos

    if not torch.cuda.is_available():
        raise RuntimeError('Common native rendering requires the GPU queue')
    comparison = Path(output) / 'comparison'
    registry_path = comparison / 'models.json'
    registry = read_json(registry_path)
    source = Path(registry.get('source', MODEL_SOURCE))
    if source.resolve() != MODEL_SOURCE.resolve():
        raise ValueError('This reviewed City protocol cannot silently change to another scene')
    folder = comparison / 'auxiliary_renders'
    folder.mkdir(parents=True, exist_ok=True)
    global_path = folder / 'manifest.json'
    if global_path.exists() and not resume:
        raise FileExistsError('Use --resume to verify/skip completed auxiliary renders')
    extrinsics = read_extrinsics_binary(str(source / 'sparse/0/images.bin'))
    intrinsics = read_intrinsics_binary(str(source / 'sparse/0/cameras.bin'))
    test_names = [name + '.png' for name in registry['test_frames']]
    infos = readColmapCameras(extrinsics, intrinsics, None, str(source / 'images'), '', test_names)
    selected = select_auxiliary_cameras(infos)
    by_name = {Path(camera.image_name).stem: camera for camera in infos}
    anchor_info = by_name[TEST[0]]
    if not anchor_info.is_test:
        raise ValueError('The native-renderer anchor must be an independent test camera')
    camera_args = argparse.Namespace(resolution=-1, data_device='cpu', train_test_exp=False)
    cameras = cameraList_from_camInfos(selected + [anchor_info], 1., camera_args, False, False)
    auxiliary_cameras, anchor = cameras[:-1], cameras[-1]
    renderer_sources = {str(path.relative_to(ROOT)): sha256_file(path) for path in (
        ROOT / 'gaussian_renderer/__init__.py', ROOT / 'utils/camera_utils.py',
        ROOT / 'scene/cameras.py', ROOT / 'render.py', ROOT / 'render_boundary_semantic_views.py')}
    state = dict(status='running', models_json_sha256=sha256_file(registry_path),
        source=str(source), frames=NON_TEST, model_camera_fingerprints=camera_fingerprints(source),
        renderer_sources=renderer_sources, models={},
        qualification='Eight annotated 3DGS training views, auxiliary only; native SH3, no resizing, black background, no exposure, no antialiasing.')
    replace_json(global_path, state)
    pipe = argparse.Namespace(convert_SHs_python=False, compute_cov3D_python=False,
                              debug=False, antialiasing=False)
    background = torch.zeros(3, dtype=torch.float32, device='cuda')
    with torch.no_grad():
        for model in validated_models(registry):
            model_dir = folder / model['id']
            model_dir.mkdir(exist_ok=True)
            receipt_path = model_dir / 'render_receipt.json'
            ply = Path(model['ply'])
            ply_sha = sha256_file(ply)
            if model.get('ply_sha256') and model['ply_sha256'] != ply_sha:
                raise ValueError('Model PLY changed since comparison registry preparation')
            fingerprint = dict(model_id=model['id'], ply=str(ply.resolve()), ply_sha256=ply_sha,
                renderer_sources=renderer_sources,
                model_camera_fingerprints=state['model_camera_fingerprints'])
            if receipt_path.exists() and cached_receipt_valid(read_json(receipt_path), fingerprint):
                state['models'][model['id']] = read_json(receipt_path)
                replace_json(global_path, state)
                print('Auxiliary render verified: ' + model['id'], flush=True)
                continue
            gaussians = GaussianModel(3)
            gaussians.load_ply(str(ply), False)
            if gaussians.get_xyz.shape[0] != int(model['actual_count']):
                raise ValueError('Model Gaussian count differs from comparison registry')
            # Pixel equality against its existing common test image verifies
            # that camera conversion and the shared renderer match prior runs.
            anchor_render = render(anchor, gaussians, pipe, background, use_trained_exp=False)['render']
            anchor_path = model_dir / 'anchor_frame_000.png'
            torchvision.utils.save_image(anchor_render, str(anchor_path))
            reference_anchor = Path(model['common_render_dir']) / ('%05d.png' % registry['test_frames'].index(TEST[0]))
            if pixels_sha256(anchor_path) != pixels_sha256(reference_anchor):
                raise ValueError('Auxiliary renderer differs from existing common test render: ' + model['id'])
            receipt = dict(fingerprint, status='complete', frames={},
                native_test_anchor=dict(frame=TEST[0], reference=str(reference_anchor),
                    reference_sha256=sha256_file(reference_anchor), anchor=str(anchor_path),
                    pixel_sha256=pixels_sha256(anchor_path), pixel_identical=True))
            for name, camera in zip(NON_TEST, auxiliary_cameras):
                image = render(camera, gaussians, pipe, background, use_trained_exp=False)['render']
                destination = model_dir / (name + '.png')
                torchvision.utils.save_image(image, str(destination))
                gt_path = source / 'images' / (name + '.png')
                if Image.open(destination).size != Image.open(gt_path).size:
                    raise ValueError('Native render/GT shape mismatch')
                expected_gt = np.asarray(Image.open(gt_path).convert('RGB'))
                actual_gt = (camera.original_image[:3].permute(1, 2, 0).cpu().numpy() * 255).round().astype(np.uint8)
                if not np.array_equal(expected_gt, actual_gt):
                    raise ValueError('Camera native GT differs from registered input image')
                receipt['frames'][name] = dict(render_path=str(destination.resolve()),
                    render_sha256=sha256_file(destination), gt_path=str(gt_path.resolve()),
                    gt_sha256=sha256_file(gt_path), size_wh=list(Image.open(destination).size))
            replace_json(receipt_path, receipt)
            state['models'][model['id']] = receipt
            replace_json(global_path, state)
            del gaussians, image, anchor_render
            torch.cuda.empty_cache()
            print('Auxiliary render complete: ' + model['id'], flush=True)
    state['status'] = 'complete'
    replace_json(global_path, state)
    print(str(global_path), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=DEFAULT_ROOT)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    run(args.root, args.resume)


if __name__ == '__main__':
    main()
