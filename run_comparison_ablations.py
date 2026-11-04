#!/usr/bin/env python3
"""Prepare and run seven SH3-only, leave-one-module-out 20x ablations.

Preparation is CPU-only. GPU execution requires the explicit --run argument.
The caller serializes this process with all other GPU jobs on this computer.
"""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time


ROOT = Path('/home/zhangys/Prune-Wisely-Reconstruct-Sharply')
PACKAGE = Path(__file__).resolve().parent
INSTALLED = PACKAGE == ROOT
CONFIG_DIRECTORY = ROOT / 'configs/comparison_ablations20' if INSTALLED else PACKAGE / 'configs'
VERIFY_SCRIPT = ROOT / 'validate_comparison_ablation.py' if INSTALLED else PACKAGE / 'verify_ablation.py'
DEFAULT_OUTPUT = ROOT / 'output/city2_comparison_20261104/ablations20'
BASE_CONFIG = ROOT / 'configs/city_adaptive_constraints_20x.json'
FULL_RUN = ROOT / 'output/city2_adaptive_constraints_20261102/full_20x'
VARIANTS = [
    ('no_instance_survival', 'instance_protection'),
    ('no_boundary_quota', 'boundary_protection'),
    ('no_topology_guard', 'topology_protection'),
    ('no_adaptive_schedule', 'adaptive_compression'),
    ('no_merging', 'gaussian_merging'),
    ('no_budget_reallocation', 'budget_reallocation'),
    ('no_structure_loss', 'structure_loss'),
]
DESCRIPTIONS = {
    'no_instance_survival': 'Remove initial pseudo-instance minimum-count floors, including their merge-floor check. Reliable labels and same-instance topology/merge restrictions remain; all recovery losses remain.',
    'no_boundary_quota': 'Remove minimum boundary-Gaussian counts from feasibility pruning. Boundary risk/maps, topology boundary-edge exclusion, merge boundary exclusion, and recovery losses remain.',
    'no_topology_guard': 'Disable PCA/candidate graph and sequential connectivity checks. Existing implementation also returns an empty thin-candidate mask, so merges no longer exclude PCA thin candidates. This measures that implemented topology module jointly.',
    'no_adaptive_schedule': 'Keep relative rounds but fix pruning ratio at 0.20. Nonfinal rounds recover exactly 200 updates; final round recovers 1500. This changes optimization effort and is not compute matched.',
    'no_merging': 'Disable native Gaussian merging; unchanged feasibility-constrained global SPS supplies the exact same final count. No merge tranche is reserved.',
    'no_budget_reallocation': 'Disable 1% reserved capacity, recent-deletion pool, hard-region restoration, and all reinsertions; prune directly to each round count.',
    'no_structure_loss': 'Remove only the additional 0.05 risk-weighted L1. All common v3 instance ROI/edge/boundary/thin recovery terms stay enabled.',
}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def atomic_json(data, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')
    os.replace(str(tmp), str(path))


def absolute(value):
    p = Path(value)
    return p if p.is_absolute() else ROOT / p


def diff(a, b, prefix=''):
    out = []
    if isinstance(a, dict) and isinstance(b, dict):
        for key in sorted(set(a) | set(b)):
            out += diff(a.get(key), b.get(key), prefix + ('.' if prefix else '') + key)
    elif a != b:
        out.append(dict(key=prefix, before=a, after=b))
    return out


def frozen_sources():
    names = ['train_adaptive_compression.py', 'run_adaptive_pipeline.py', 'train_prune.py',
             'render.py', 'metrics.py', 'scripts/run.sh', 'validate_adaptive_results.py']
    for folder in ('utils', 'scene', 'gaussian_renderer', 'arguments', 'lpipsPyTorch'):
        names += [str(p.relative_to(ROOT)) for p in (ROOT / folder).rglob('*.py')]
    # Independent downstream semantic evaluation is not a compression dependency.
    excluded = {'utils/city_semantic_data.py', 'utils/semantic_comparison_metrics.py'}
    return {name: sha(ROOT / name) for name in sorted(set(names)) if name not in excluded}


def prepare(output, refresh=False):
    base = json.loads(BASE_CONFIG.read_text())
    full = json.loads((FULL_RUN / 'stage3/run_args.json').read_text())
    if full['status'] != 'complete' or full['final_gaussians'] != 89405:
        raise ValueError('Full-method SH3 20x baseline is not complete')
    if full['config'] != base or base['pruning_target'] != 89405:
        raise ValueError('Saved full method and source configuration differ')
    sources = frozen_sources()
    baseline_snapshot = json.loads((FULL_RUN.parent / 'source_snapshot/manifest.json').read_text())
    for name in ('train_adaptive_compression.py', 'run_adaptive_pipeline.py', 'train_prune.py',
                 'utils/adaptive_schedule.py', 'utils/adaptive_constraints.py',
                 'utils/adaptive_mutation.py', 'utils/adaptive_metadata.py'):
        if sources[name] != baseline_snapshot['files'][name]:
            raise ValueError('Training source differs from completed full method: ' + name)
    shared = absolute(base['shared_metadata'])
    shared_files = sorted(p for p in shared.rglob('*') if p.is_file())
    shared_hashes = {str(p.relative_to(shared)): sha(p) for p in shared_files}
    metadata = json.loads((shared / 'metadata_manifest.json').read_text())
    input_sha = sha(absolute(base['input_ply']))
    if input_sha != metadata['source_ply_sha256']:
        raise ValueError('Original input PLY changed')
    rows = []
    for slug, key in VARIANTS:
        config = copy.deepcopy(base)
        config[key]['enabled'] = False
        config['output_root'] = str(output / slug)
        differences = diff(base, config)
        expected = {key + '.enabled', 'output_root'}
        if {d['key'] for d in differences} != expected:
            raise ValueError('Unexpected configuration changes: ' + slug)
        path = CONFIG_DIRECTORY / (slug + '.json')
        atomic_json(config, path)
        rows.append(dict(slug=slug, module=key, config=str(path), config_sha256=sha(path),
                         output_root=config['output_root'], differences=differences,
                         intervention=DESCRIPTIONS[slug], status='planned'))
    fixed_count = metadata['source_count']
    fixed_rounds = 0
    while fixed_count > base['pruning_target']:
        import math
        fixed_count = max(base['pruning_target'], fixed_count - max(1, int(math.ceil(fixed_count * .2))))
        fixed_rounds += 1
    plan = dict(schema_version=1, status='prepared_not_started', output_root=str(output),
                method='20x SH3 leave-one-implemented-module-out',
                baseline_config=str(BASE_CONFIG), baseline_config_sha256=sha(BASE_CONFIG),
                baseline_run=str(FULL_RUN), baseline_checkpoint=full['last_checkpoint'],
                baseline_checkpoint_sha256=sha(full['last_checkpoint']),
                baseline_ply=full['last_ply'], baseline_ply_sha256=sha(full['last_ply']),
                baseline_rounds=len(full['events']), baseline_recovery_updates=full['completed_steps'],
                source_count=metadata['source_count'], target=base['pruning_target'],
                input_ply_sha256=input_sha, shared_metadata=str(shared), shared_files=shared_hashes,
                source_files=sources, test_frames=metadata['test_frames_excluded'],
                train_frames=metadata['train_frames'], variants=rows,
                disabled_adaptive_expected_rounds=fixed_rounds,
                disabled_adaptive_expected_updates=(fixed_rounds - 1) * 200 + 1500,
                common_v3_recovery_losses=base['recovery_losses'],
                distillation=False, semantic_labels_used_for_compression=False,
                interpretation='One seed, exact same final Gaussian budget and fixed nine test cameras. Adaptive rounds change realized update counts. Differences include interactions of the implemented module; leave-one-out is not additive causal attribution.',
                queue_command=[sys.executable, str(Path(__file__).resolve()), '--run', '--resume'])
    plan_path = output / 'ablation_plan.json' if INSTALLED else PACKAGE / 'plan.json'
    if plan_path.exists() and json.loads(plan_path.read_text()) != plan:
        if not refresh:
            raise ValueError('An existing frozen plan differs; use another output root or --refresh_preparation before any job starts')
        state_path = output / 'ablation_manifest.json'
        state = json.loads(state_path.read_text()) if state_path.is_file() else dict(jobs={})
        if any(row.get('status') != 'planned' for row in state['jobs'].values()) or any((output / slug / 'stage3').exists() for slug, _ in VARIANTS):
            raise ValueError('Cannot refresh preparation after an ablation job starts')
        old = json.loads(plan_path.read_text())
        if old['variants'] != plan['variants'] or old['input_ply_sha256'] != plan['input_ply_sha256'] or old['shared_files'] != plan['shared_files']:
            raise ValueError('Preparation refresh cannot change scientific interventions or input data')
        # Only independent semantic source exclusions may differ; no trained code changes.
        for name, expected in plan['source_files'].items():
            if old['source_files'].get(name) != expected:
                raise ValueError('Preparation refresh cannot change a compression source: ' + name)
        archive = output / 'preparation_archive' / ('prepared_%d' % int(time.time()))
        archive.mkdir(parents=True, exist_ok=False)
        shutil.copy2(str(plan_path), str(archive / plan_path.name))
        if state_path.is_file():
            shutil.copy2(str(state_path), str(archive / state_path.name))
            state_path.unlink()
    atomic_json(plan, plan_path)
    if INSTALLED and not (output / 'ablation_manifest.json').is_file():
        atomic_json(dict(status='prepared_not_started', plan_sha256=sha(plan_path),
                         baseline_reference=str(FULL_RUN), target=89405,
                         jobs={row['slug']: dict(status='planned', output_root=row['output_root']) for row in rows}),
                    output / 'ablation_manifest.json')
    print(json.dumps({k: plan[k] for k in ('status', 'target', 'baseline_recovery_updates',
          'disabled_adaptive_expected_rounds', 'disabled_adaptive_expected_updates')}, indent=2))
    return plan


def check_frozen(plan):
    for name, expected in plan['source_files'].items():
        if sha(ROOT / name) != expected:
            raise ValueError('Frozen training/evaluation source changed: ' + name)
    if sha(BASE_CONFIG) != plan['baseline_config_sha256']:
        raise ValueError('Full baseline configuration changed')
    if sha(plan['baseline_ply']) != plan['baseline_ply_sha256']:
        raise ValueError('Full baseline PLY changed')
    if sha(plan['baseline_checkpoint']) != plan['baseline_checkpoint_sha256']:
        raise ValueError('Full baseline checkpoint changed')
    if sha(absolute(json.loads(BASE_CONFIG.read_text())['input_ply'])) != plan['input_ply_sha256']:
        raise ValueError('Original source model changed')
    for name, expected in plan['shared_files'].items():
        if sha(Path(plan['shared_metadata']) / name) != expected:
            raise ValueError('Shared training-only prior changed: ' + name)


def launch(plan, only, resume, plan_path):
    check_frozen(plan)
    output = Path(plan['output_root'])
    output.mkdir(parents=True, exist_ok=True)
    queue_path = output / 'ablation_manifest.json'
    state = json.loads(queue_path.read_text()) if queue_path.is_file() else dict(
        status='running', plan_sha256=sha(plan_path), baseline_reference=str(FULL_RUN), target=89405, jobs={})
    if state['plan_sha256'] != sha(plan_path):
        raise ValueError('Queue plan differs from the previous run')
    snapshot = output / 'source_snapshot'
    for name, expected in plan['source_files'].items():
        dest = snapshot / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.is_file() and sha(dest) != expected:
            raise ValueError('Saved source snapshot differs: ' + name)
        if not dest.is_file():
            shutil.copy2(str(ROOT / name), str(dest))
    atomic_json(plan, output / 'plan.json')
    atomic_json(dict(files=plan['source_files']), snapshot / 'manifest.json')
    shutil.copy2(str(Path(__file__).resolve()), str(output / Path(__file__).name))
    shutil.copy2(str(VERIFY_SCRIPT), str(output / VERIFY_SCRIPT.name))
    lock = output / '.gpu_queue.lock'
    # Advisory lock serializes accidental duplicate invocations of this queue.
    import fcntl
    with lock.open('w') as lock_file:
        fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for item in plan['variants']:
            slug = item['slug']
            if only and slug not in only:
                continue
            if sha(item['config']) != item['config_sha256']:
                raise ValueError('Ablation config changed: ' + slug)
            config_dest = output / 'configs' / (slug + '.json')
            config_dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(item['config'], str(config_dest))
            model_root = Path(item['output_root'])
            cmd = ['bash', str(ROOT / 'scripts/run.sh'), 'run_adaptive_pipeline.py',
                   '--config', str(config_dest), '--skip_distill']
            if (model_root / 'pipeline.json').is_file():
                if not resume:
                    raise ValueError('Existing experiment requires --resume: ' + slug)
                cmd.append('--resume')
            if (model_root / 'stage3').exists() and not (model_root / 'pipeline.json').is_file():
                raise ValueError('Partial output without pipeline manifest: ' + slug)
            previous = state['jobs'].get(slug, {})
            row = dict(previous, status='running', command=cmd, started_unix=time.time())
            state['jobs'][slug] = row
            state['status'] = 'running'
            atomic_json(state, queue_path)
            print('ABLATION START ' + slug, flush=True)
            try:
                model_root.mkdir(parents=True, exist_ok=True)
                with (model_root / 'queue_pipeline.log').open('a') as log:
                    process = subprocess.Popen(cmd, cwd=str(ROOT), stdout=subprocess.PIPE,
                                               stderr=subprocess.STDOUT, universal_newlines=True, bufsize=1)
                    for line in process.stdout:
                        print(line, end='', flush=True)
                        log.write(line)
                        log.flush()
                    result = process.wait()
                    if result:
                        raise subprocess.CalledProcessError(result, cmd)
                env = dict(os.environ, CUDA_VISIBLE_DEVICES='')
                verify = ['bash', str(ROOT / 'scripts/run.sh'), str(VERIFY_SCRIPT),
                          '--stage', str(model_root / 'stage3'), '--config', str(config_dest),
                          '--test-frames', ','.join(plan['test_frames'])]
                subprocess.run(verify, cwd=str(ROOT), env=env, check=True)
                check_frozen(plan)
                validation = json.loads((model_root / 'stage3/ablation_validation.json').read_text())
                row.update(status='complete', validation=validation, finished_unix=time.time())
            except BaseException as exc:
                row.update(status='failed', error=str(exc), finished_unix=time.time())
                state['status'] = 'failed'
                atomic_json(state, queue_path)
                raise
            atomic_json(state, queue_path)
            print('ABLATION COMPLETE ' + slug, flush=True)
    state['status'] = 'complete' if all(state['jobs'].get(x['slug'], {}).get('status') == 'complete'
                                      for x in plan['variants']) else 'partially_complete'
    atomic_json(state, queue_path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--refresh_preparation', action='store_true', help='Archive/refresh only an unstarted, scientifically unchanged preparation')
    parser.add_argument('--run', action='store_true')
    parser.add_argument('--resume', action='store_true', help='Resume saved compression and completed pipeline steps')
    parser.add_argument('--only', nargs='+', choices=[x[0] for x in VARIANTS])
    parser.add_argument('--output-root', type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if args.prepare:
        prepare(args.output_root.resolve(), args.refresh_preparation)
    if args.run:
        plan_path = args.output_root.resolve() / 'ablation_plan.json' if INSTALLED else PACKAGE / 'plan.json'
        launch(json.loads(plan_path.read_text()), args.only, args.resume, plan_path)
    if not args.prepare and not args.run:
        parser.error('Choose --prepare (CPU only) or --run (GPU queue)')


if __name__ == '__main__':
    main()
