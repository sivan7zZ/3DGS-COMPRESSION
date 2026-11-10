"""Wait for compression, then serially render, cross-fit and evaluate all methods."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from utils.distill_utils import atomic_json_save, sha256_file

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'output/city2_boundary_weighted_20261106'
VGGT = Path('/home/zhangys/anaconda3/envs/VGGT/bin/python')
EVALUATION_FILES = ['evaluate_boundary_weighted_semantics.py', 'render_boundary_semantic_views.py',
    'run_boundary_weighted_evaluation.py', 'finalize_boundary_weighted_results.py', 'city_semantic.py',
    'utils/city_semantic_data.py', 'utils/semantic_comparison_metrics.py', 'scene/colmap_loader.py',
    'scene/dataset_readers.py', 'scene/cameras.py', 'utils/camera_utils.py', 'gaussian_renderer/__init__.py',
    'finalize_city_benchmark.py']


def worker_environment(python, parent=None):
    """Do not inject the rasterizer environment's cuDNN into a different Torch."""
    environment = dict(os.environ if parent is None else parent)
    if Path(python).resolve() == VGGT.resolve():
        environment.pop('LD_LIBRARY_PATH', None)
        environment.pop('LD_PRELOAD', None)
    return environment


def read(path):
    return json.loads(Path(path).read_text())


def snapshot():
    folder = OUT / 'evaluation_source_snapshot'
    manifest = folder / 'manifest.json'
    if manifest.exists():
        for name, digest in read(manifest)['files'].items():
            if sha256_file(ROOT / name) != digest or sha256_file(folder / name) != digest:
                raise ValueError('Frozen evaluation source changed: ' + name)
        return
    records = {}
    for name in EVALUATION_FILES:
        target = folder / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(str(ROOT / name), str(target))
        records[name] = sha256_file(ROOT / name)
    atomic_json_save(dict(status='frozen', files=records), manifest)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--wait', action='store_true')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    while True:
        path = OUT / 'experiment_queue.json'
        state = read(path) if path.exists() else {}
        if state.get('status') == 'complete':
            break
        if state.get('status') == 'failed':
            raise RuntimeError('Compression queue failed: ' + state.get('error', 'unknown'))
        if not args.wait:
            raise RuntimeError('Compression has not completed; use --wait to queue evaluation')
        time.sleep(20)
    snapshot()
    manifest = OUT / 'evaluation_queue.json'
    if manifest.exists() and not args.resume:
        raise FileExistsError('Use --resume for the existing evaluation queue')
    state = read(manifest) if manifest.exists() else dict(status='running', completed=[], tasks=[])
    state['status'] = 'running'
    atomic_json_save(state, manifest)
    tasks = [
        ('render_auxiliary', sys.executable, 'render_boundary_semantic_views.py', []),
        ('fit_semantic_folds', VGGT, 'evaluate_boundary_weighted_semantics.py', ['--phase', 'train']),
        ('infer_semantic_folds', VGGT, 'evaluate_boundary_weighted_semantics.py', ['--phase', 'infer']),
        ('evaluate_semantics', VGGT, 'evaluate_boundary_weighted_semantics.py', ['--phase', 'evaluate']),
        ('finalize_comparison', VGGT, 'finalize_boundary_weighted_results.py', []),
    ]
    try:
        for name, python, script, options in tasks:
            snapshot()
            if name in state['completed']:
                continue
            command = [str(python), str(ROOT / script)] + options
            if args.resume and name not in ('finalize_comparison', 'evaluate_semantics'):
                command.append('--resume')
            state['tasks'].append(dict(name=name, command=command))
            atomic_json_save(state, manifest)
            with (OUT / (name + '.log')).open('a') as stream:
                subprocess.check_call(command, cwd=str(ROOT), stdout=stream, stderr=subprocess.STDOUT,
                                      env=worker_environment(python))
            state['completed'].append(name)
            atomic_json_save(state, manifest)
        snapshot()
        state['status'] = 'complete'
    except BaseException as error:
        state.update(status='failed', error=str(error))
        atomic_json_save(state, manifest)
        raise
    atomic_json_save(state, manifest)


if __name__ == '__main__':
    main()
