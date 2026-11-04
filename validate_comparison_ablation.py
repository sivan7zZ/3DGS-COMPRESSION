#!/usr/bin/env python3
"""Read-only CPU verification of a single completed SH3 ablation."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path('/home/zhangys/Prune-Wisely-Reconstruct-Sharply')
sys.path.insert(0, str(ROOT))
from validate_adaptive_results import verify_stage
from utils.adaptive_schedule import audit_quotas
from utils.checkpoint_utils import atomic_json_save


def assert_test_frames(actual, expected):
    """Compare ordered camera names independent of image-file extensions."""
    actual = [Path(str(value)).stem for value in actual]
    expected = [Path(str(value)).stem for value in expected]
    if actual != expected:
        raise ValueError('Test-camera split mismatch')
    return actual


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--stage', required=True, type=Path)
    p.add_argument('--config', required=True, type=Path)
    p.add_argument('--test-frames', required=True)
    a = p.parse_args()
    config = json.loads(a.config.read_text())
    pipeline = json.loads((a.stage.parent / 'pipeline.json').read_text())
    stage = json.loads((a.stage / 'run_args.json').read_text())
    if pipeline['config'] != config or stage['config'] != config:
        raise ValueError('Output config mismatch')
    if pipeline['status'] != 'complete' or stage['status'] != 'complete':
        raise ValueError('Pipeline or SH3 stage incomplete')
    if set(pipeline['completed']) != {'compress', 'render_stage3', 'metrics'}:
        raise ValueError('SH3-only pipeline steps differ')
    if (a.stage.parent / 'sh2').exists():
        raise ValueError('Unexpected SH2 output in SH3-only ablation')
    test_frames = assert_test_frames(stage['test_views'], a.test_frames.split(','))
    audit, meta = verify_stage(a.stage, Path(stage['last_checkpoint']),
                               Path(stage['last_ply']), int(config['pruning_target']))
    policy = json.loads((a.stage / 'protection_policy.json').read_text())
    amin = {int(k): v for k, v in policy['instance_min'].items()}
    bmin = {int(k): v for k, v in policy['boundary_min'].items()}
    quota = audit_quotas(meta, amin, bmin)
    for event in stage['events']:
        if event['after_reallocation'] != event['target']:
            raise ValueError('Historical round target mismatch')
        if not event['selection']['target_reached']:
            raise ValueError('Historical infeasible prune')
        if event['quota_audit']['instance_violations'] or event['quota_audit']['boundary_violations']:
            raise ValueError('Historical quota violation')
    folder = a.stage / 'test' / ('ours_%d' % stage['final_iteration'])
    if len(list((folder / 'renders').glob('*.png'))) != 9 or len(list((folder / 'gt').glob('*.png'))) != 9:
        raise ValueError('Expected nine native test renders and GT')
    results = json.loads((a.stage / 'results.json').read_text())
    key = 'ours_%d' % stage['final_iteration']
    if key not in results or any(name not in results[key] for name in ('PSNR', 'SSIM', 'LPIPS')):
        raise ValueError('Final PNG metrics missing')
    audit.update(status='complete', rounds=len(stage['events']),
                 recovery_updates=stage['completed_steps'], png_metrics=results[key],
                 quota_violations=0, instance_floors=len(amin),
                 merged_total=sum(e['merged'] for e in stage['events']),
                 restored_total=sum(e['restored'] for e in stage['events']),
                 test_frames=test_frames, source_test_view_names=stage['test_views'], sh3_only=True)
    atomic_json_save(audit, a.stage / 'ablation_validation.json')
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
