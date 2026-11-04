"""Postprocess saved City PNGs with VGG LPIPS v0.1 using standard [-1,1] inputs.

Preserves every historical results.json/per_view.json and all compression models.
The historical repository evaluator passes [0,1] into a network expecting [-1,1];
these legacy values remain linked in each new comparison_lpips.json receipt.
"""
import argparse
from contextlib import contextmanager
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import time

import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parent
DEFAULT_ROOT = REPO / 'output/city2_comparison_20261104'
NAMES = ['{:05d}.png'.format(i) for i in range(9)]
FRAMES = ['frame_{:03d}'.format(i) for i in range(0, 70, 8)]
MAIN_IDS = {'original_3dgs', 'lightgaussian_10x', 'lightgaussian_20x', 'pup_10x', 'pup_20x',
            'speedy_10x', 'speedy_20x', 'pwsr_10x', 'pwsr_20x', 'ours_10x', 'ours_20x', 'v3_10x', 'v3_20x'}
ABLATIONS = {'no_instance_survival', 'no_boundary_quota', 'no_topology_guard',
             'no_adaptive_schedule', 'no_merging', 'no_budget_reallocation', 'no_structure_loss'}


def read(path):
    return json.loads(Path(path).read_text())


def sha256_file(path):
    h = hashlib.sha256()
    with open(str(path), 'rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def json_signature(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     ensure_ascii=True, allow_nan=False).encode('utf-8')).hexdigest()


def atomic_json(value, path):
    path = Path(path)
    tmp = path.with_name(path.name + '.tmp.{}'.format(os.getpid()))
    try:
        tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
        os.replace(str(tmp), str(path))
    finally:
        if tmp.exists():
            tmp.unlink()


@contextmanager
def output_lock(root):
    with open(str(root / '.lpips_standard.lock'), 'a') as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError('Another standard-range LPIPS postprocess is active')
        yield


def rgb_array(path):
    with Image.open(str(path)) as im:
        image = np.asarray(im.convert('RGB'))
    if image.shape != (894, 1592, 3):
        raise ValueError('Expected native 1592x894 RGB: ' + str(path))
    return image


def standard_input(image):
    """CHW float32: RGB uint8/255 mapped from [0,1] to [-1,1]."""
    image = np.asarray(image)
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError('Expected RGB uint8 image')
    return np.ascontiguousarray(image.transpose(2, 0, 1), dtype=np.float32) / np.float32(255) * np.float32(2) - np.float32(1)


def collect_jobs(root, main_only=False):
    root = Path(root)
    manifest = read(root / 'models.json')
    if manifest.get('test_frames') != FRAMES:
        raise ValueError('Unexpected City test view mapping')
    models = manifest['models']
    if {m['id'] for m in models} != MAIN_IDS or any(m.get('status') == 'missing' for m in models):
        raise ValueError('All 13 main models must be present')
    jobs = []
    for model in models:
        jobs.append({'id': model['id'], 'kind': 'main', 'model_dir': str(Path(model['comparison_dir'])),
                     'render_dir': model['common_render_dir'], 'gt_dir': model['common_gt_dir'],
                     'method_key': 'ours_{}'.format(model['iteration'])})
    if not main_only:
        plan = read(root / 'ablations20/ablation_plan.json')
        completed = read(root / 'ablations20/ablation_manifest.json')
        if completed.get('status') != 'complete' or {v['slug'] for v in plan['variants']} != ABLATIONS:
            raise ValueError('All seven ablations must be completed, or select --main_only')
        for variant in plan['variants']:
            slug = variant['slug']
            if completed['jobs'][slug]['status'] != 'complete':
                raise ValueError('Ablation is incomplete: ' + slug)
            folder = root / 'ablations20' / slug / 'stage3'
            run = read(folder / 'run_args.json')
            validation = read(folder / 'ablation_validation.json')
            if run.get('status') != 'complete' or validation.get('status') != 'complete':
                raise ValueError('Ablation validation is incomplete: ' + slug)
            method = 'ours_{}'.format(run['final_iteration'])
            jobs.append({'id': 'ablation_' + slug, 'kind': 'ablation', 'model_dir': str(folder),
                         'render_dir': str(folder / 'test' / method / 'renders'),
                         'gt_dir': str(folder / 'test' / method / 'gt'), 'method_key': method})
    return jobs


def prepare_inputs(job):
    folder, method = Path(job['model_dir']), job['method_key']
    legacy_path, legacy_views = folder / 'results.json', folder / 'per_view.json'
    legacy = read(legacy_path)[method]['LPIPS']
    legacy_per_view = read(legacy_views)[method]['LPIPS']
    if set(legacy_per_view) != set(NAMES) or not math.isclose(
            legacy, sum(legacy_per_view.values()) / 9, rel_tol=1e-10, abs_tol=1e-10):
        raise ValueError('Legacy source does not contain the exact nine test views')
    images = []
    for directory in (job['render_dir'], job['gt_dir']):
        if sorted(p.name for p in Path(directory).glob('*.png')) != NAMES:
            raise ValueError('Expected exactly nine native test PNGs: ' + str(directory))
    for name, frame in zip(NAMES, FRAMES):
        render, gt = Path(job['render_dir']) / name, Path(job['gt_dir']) / name
        a, b = rgb_array(render), rgb_array(gt)
        images.append({'filename': name, 'frame': frame, 'render_path': str(render),
                       'render_sha256': sha256_file(render), 'gt_path': str(gt),
                       'gt_sha256': sha256_file(gt), 'size': [1592, 894],
                       'gt_rgb_sha256': hashlib.sha256(b.tobytes()).hexdigest()})
    legacy_ref = {'results_path': str(legacy_path), 'results_sha256': sha256_file(legacy_path),
                  'per_view_path': str(legacy_views), 'per_view_sha256': sha256_file(legacy_views),
                  'method_key': method, 'LPIPS': float(legacy), 'per_view': legacy_per_view,
                  'input_range': [0, 1], 'qualification': 'historical local repository range convention'}
    return images, legacy_ref


def implementation_provenance(repo, root):
    import torch
    import torchvision
    paths = [Path(__file__).resolve(), repo / 'lpipsPyTorch/__init__.py',
             repo / 'lpipsPyTorch/modules/lpips.py', repo / 'lpipsPyTorch/modules/networks.py',
             repo / 'lpipsPyTorch/modules/utils.py']
    weights = [Path(torch.hub.get_dir()) / 'checkpoints/vgg16-397923af.pth',
               Path(torch.hub.get_dir()) / 'checkpoints/vgg.pth']
    for path in paths + weights:
        if not path.is_file():
            raise FileNotFoundError('Require existing local implementation/weights; no download: ' + str(path))
    official = root / 'inventory/lpips_author_reference.py'
    return {'net': 'vgg', 'version': '0.1', 'local_implementation': 'lpipsPyTorch.LPIPS',
            'torch_version': torch.__version__, 'torchvision_version': torchvision.__version__,
            'source_files_sha256': {str(p): sha256_file(p) for p in paths},
            'weight_files_sha256': {str(p): sha256_file(p) for p in weights},
            'weight_sources': {'VGG16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
                               'linear_v0.1': 'https://raw.githubusercontent.com/richzhang/PerceptualSimilarity/master/lpips/weights/v0.1/vgg.pth'},
            'official_range_reference': {'url': 'https://github.com/richzhang/PerceptualSimilarity/blob/master/lpips/lpips.py',
                                         'local_path': str(official) if official.exists() else None,
                                         'sha256': sha256_file(official) if official.exists() else None},
            'input_range': [-1, 1], 'preprocessing': 'RGB uint8 -> float32/255 -> 2*x-1',
            'spatial_processing': 'full native 1592x894, no resize/crop/tile',
            'local_scaling_mean': [-0.030, -0.088, -0.188],
            'local_scaling_std': [0.458, 0.448, 0.450]}


def state_dict_sha256(module):
    h = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        value = value.detach().cpu().contiguous()
        h.update(name.encode('utf-8'))
        h.update(str(tuple(value.shape)).encode('ascii'))
        h.update(str(value.dtype).encode('ascii'))
        h.update(value.numpy().tobytes())
    return h.hexdigest()


def cache_valid(result, signature):
    if result.get('status') != 'complete' or result.get('cache_signature') != signature:
        return False
    values = result.get('per_view', {})
    if set(values) != set(NAMES) or any(not math.isfinite(float(v)) for v in values.values()):
        return False
    if not math.isfinite(float(result.get('LPIPS', float('nan')))):
        return False
    if not math.isclose(result['LPIPS'], sum(values.values()) / 9, rel_tol=1e-12, abs_tol=1e-12):
        return False
    control = result.get('self_control', {})
    return (control.get('status') == 'passed' and math.isfinite(float(control.get('value', float('nan'))))
            and abs(control['value']) <= 1e-10)


def evaluate(root, main_only=False, device='cuda', force=False, dry_run=False):
    import torch
    from lpipsPyTorch import LPIPS
    root = Path(root).resolve()
    jobs = collect_jobs(root, main_only=main_only)
    prepared = [(job,) + prepare_inputs(job) for job in jobs]
    canonical = {x['filename']: x['gt_rgb_sha256'] for x in prepared[0][1]}
    if any({x['filename']: x['gt_rgb_sha256'] for x in images} != canonical for _, images, _ in prepared):
        raise ValueError('GT RGB differs across models/ablations')
    criterion_meta = implementation_provenance(REPO, root)
    if dry_run:
        print(json.dumps({'status': 'dry_run_validated', 'models': [j['id'] for j in jobs],
                          'n_views_each': 9, 'criterion': criterion_meta}, indent=2))
        return
    if device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; no silent CPU fallback')
    criterion, control, actual_state_sha = None, None, None
    manifest = {'schema': 'city_lpips_standard_v1', 'status': 'running',
                'scope': 'main_only' if main_only else 'main_and_seven_ablations',
                'input_range': [-1, 1], 'n_test_views': 9, 'test_frames': FRAMES,
                'criterion': dict(criterion_meta), 'models': {}}
    manifest_path = root / 'lpips_standard_manifest.json'
    with output_lock(root):
        previous = read(manifest_path) if manifest_path.exists() else {}
        previous_entries = previous.get('models', {})
        atomic_json(manifest, manifest_path)
        for job, images, legacy in prepared:
            cache_material = {'schema': manifest['schema'], 'model_id': job['id'],
                              'criterion': criterion_meta, 'images': images, 'legacy': legacy}
            signature = json_signature(cache_material)
            output = Path(job['model_dir']) / 'comparison_lpips.json'
            cached = read(output) if output.exists() else None
            previous_entry = previous_entries.get(job['id'], {})
            trusted_cache = (cached is not None and previous_entry.get('path') == str(output) and
                             previous_entry.get('cache_signature') == signature and
                             previous_entry.get('sha256') == sha256_file(output))
            if not force and trusted_cache and cache_valid(cached, signature):
                result = cached
                print(job['id'] + ' cached LPIPS=' + str(result['LPIPS']), flush=True)
            else:
                if criterion is None:
                    criterion = LPIPS('vgg', '0.1').to(device).eval()
                    criterion.requires_grad_(False)
                    actual_state_sha = state_dict_sha256(criterion)
                    gt_path = prepared[0][1][0]['gt_path']
                    x = torch.from_numpy(standard_input(rgb_array(gt_path))).unsqueeze(0).to(device)
                    with torch.no_grad():
                        value = float(criterion(x, x).mean().item())
                    if not math.isfinite(value) or abs(value) > 1e-10:
                        raise ValueError('Real GT self-comparison LPIPS control is not zero: ' + str(value))
                    control = {'status': 'passed', 'type': 'real native GT RGB self-comparison',
                               'image_path': gt_path, 'image_sha256': sha256_file(gt_path),
                               'value': value, 'tolerance': 1e-10, 'input_range': [-1, 1]}
                    del x
                values = {}
                started = time.time()
                for image in images:
                    x = torch.from_numpy(standard_input(rgb_array(image['render_path']))).unsqueeze(0).to(device)
                    y = torch.from_numpy(standard_input(rgb_array(image['gt_path']))).unsqueeze(0).to(device)
                    with torch.no_grad():
                        value = float(criterion(x, y).mean().item())
                    if not math.isfinite(value):
                        raise ValueError('Nonfinite LPIPS at ' + job['id'] + '/' + image['filename'])
                    values[image['filename']] = value
                    del x, y
                result = {'schema': manifest['schema'], 'status': 'complete', 'model_id': job['id'],
                          'LPIPS': float(sum(values.values()) / 9), 'per_view': values,
                          'input_range': [-1, 1], 'preprocessing': criterion_meta['preprocessing'],
                          'criterion': dict(criterion_meta, state_dict_sha256=actual_state_sha),
                          'legacy': legacy, 'images': images, 'cache_signature': signature,
                          'self_control': control, 'runtime_device': str(device),
                          'torch_version': torch.__version__, 'elapsed_seconds': time.time() - started,
                          'qualification': 'Standard input range VGG LPIPS v0.1, local implementation and exact weights recorded; original legacy results unchanged.'}
                # Guard against an input or legacy file changing while the GPU evaluates it.
                _, after_images, after_legacy = (job,) + prepare_inputs(job)
                if images != after_images or legacy != after_legacy:
                    raise ValueError('Evaluation inputs changed during postprocess')
                atomic_json(result, output)
                print(job['id'] + ' standard LPIPS=' + str(result['LPIPS']) +
                      ' legacy=' + str(legacy['LPIPS']), flush=True)
            manifest['models'][job['id']] = {'LPIPS': result['LPIPS'], 'path': str(output),
                                            'sha256': sha256_file(output), 'cache_signature': signature,
                                            'legacy_LPIPS': result['legacy']['LPIPS']}
            manifest['self_control'] = result['self_control']
            if ('state_dict_sha256' in manifest['criterion'] and
                    manifest['criterion']['state_dict_sha256'] != result['criterion']['state_dict_sha256']):
                raise ValueError('LPIPS criterion state differs across model receipts')
            manifest['criterion']['state_dict_sha256'] = result['criterion']['state_dict_sha256']
            atomic_json(manifest, manifest_path)
        expected = 13 if main_only else 20
        if len(manifest['models']) != expected:
            raise ValueError('Incomplete LPIPS postprocess')
        manifest['status'] = 'complete'
        atomic_json(manifest, manifest_path)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=DEFAULT_ROOT)
    parser.add_argument('--main_only', action='store_true')
    parser.add_argument('--device', choices=['cuda', 'cpu'], default='cuda')
    parser.add_argument('--force', action='store_true')
    parser.add_argument('--dry_run', action='store_true')
    args = parser.parse_args()
    evaluate(args.root, main_only=args.main_only, device=args.device, force=args.force, dry_run=args.dry_run)


if __name__ == '__main__':
    main()
