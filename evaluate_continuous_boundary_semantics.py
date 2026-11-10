"""Evaluate continuous-boundary models with the five unchanged frozen evaluators.

Reuse is keyed by checkpoint AND decoded input-file SHA, never by model name.
No evaluator fitting or checkpoint selection occurs in this experiment.
"""
import argparse
import copy
from pathlib import Path
import shutil

import evaluate_boundary_weighted_semantics as common
from utils.city_semantic_data import sha256_file

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'output/city2_continuous_boundary_20261109'
PREVIOUS = ROOT / 'output/city2_boundary_weighted_20261106'
ORIGINAL_VERIFY = common.verify_metadata


def verify_metadata(metadata, fold, protocol):
    expected = fold.get('expected_data_manifest_sha256')
    if not expected or metadata['data_manifest_sha256'] != expected:
        raise ValueError('Frozen evaluator registered-data identity differs')
    # Historical main and auxiliary checkpoints have distinct manifest hashes.
    # Keep every other original train/predict/selection validation unchanged.
    adjusted = dict(protocol, historical_data_manifest_sha256=expected)
    ORIGINAL_VERIFY(metadata, fold, adjusted)


def prediction_cache(checkpoint_sha256, output=OUT):
    protocol = common.read_json(Path(output) / 'comparison/semantic/protocol.json')
    if sha256_file(protocol['reused_metrics']) != protocol['reused_metrics_sha256']:
        raise ValueError('Original frozen semantic metrics receipt changed')
    cache = {}
    for fold in protocol['folds']:
        if fold['checkpoint_sha256'] != checkpoint_sha256:
            continue
        path = Path(fold['original_predictions_manifest'])
        if sha256_file(path) != fold['original_predictions_manifest_sha256']:
            raise ValueError('Original prediction manifest differs from frozen receipt')
        value = common.read_json(path)
        if value['checkpoint_sha256'] != checkpoint_sha256:
            continue
        if sha256_file(value['checkpoint']) != checkpoint_sha256:
            raise ValueError('Cached evaluator checkpoint content changed')
        for row in value['predictions']:
            if sha256_file(row['image_path']) != row['image_sha256'] or sha256_file(row['prediction_path']) != row['prediction_sha256']:
                raise ValueError('Cached semantic RGB or prediction content changed')
            old = cache.get(row['image_sha256'])
            if old and old['prediction_sha256'] != row['prediction_sha256']:
                raise ValueError('Conflicting predictions under identical input and checkpoint')
            cache[row['image_sha256']] = row
    return cache


def install_frozen_protocol(output=OUT):
    common.verify_metadata = verify_metadata
    common.historical_prediction_cache = lambda checkpoint_sha: prediction_cache(checkpoint_sha, output)


def copy_exact(source, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if sha256_file(target) != sha256_file(source):
            raise ValueError('Existing copied scientific input differs: ' + str(target))
    else:
        shutil.copy2(str(source), str(target))


def prepare(output=OUT):
    semantic_old, old, manifest = common.load_context(PREVIOUS)
    old_metrics_path = semantic_old / 'metrics.json'
    old_metrics = common.read_json(old_metrics_path)
    if old_metrics['status'] != 'complete':
        raise ValueError('Previous semantic evaluation is incomplete')
    output = Path(output)
    semantic = output / 'comparison/semantic'
    manifest_path = semantic / 'dataset_manifest.json'
    # Preserve the byte-exact manifest used to fit the four auxiliary models.
    copy_exact(Path(old['data_manifest']), manifest_path)
    for name in ('label_audit.json', 'README_标签核对.md'):
        copy_exact(PREVIOUS / name, output / name)
    protocol = copy.deepcopy(old)
    protocol.update(schema='city_continuous_boundary_frozen_protocol_v1',
        status='frozen_evaluator_reuse', data_manifest=str(manifest_path),
        data_manifest_sha256=sha256_file(manifest_path),
        reused_protocol=str(semantic_old / 'protocol.json'),
        reused_protocol_sha256=sha256_file(semantic_old / 'protocol.json'),
        reused_metrics=str(old_metrics_path), reused_metrics_sha256=sha256_file(old_metrics_path),
        fitting='No evaluator fitting; reuse the five hash-verified 20261106 frozen checkpoints')
    for fold in protocol['folds']:
        original = next(row for row in old['folds'] if row['id'] == fold['id'])
        checkpoint = common.fold_checkpoint(semantic_old, original)
        receipt_path = (Path(old['historical_training_protocol']) if original['checkpoint']
                        else checkpoint.parent / 'training_protocol.json')
        receipt = common.read_json(receipt_path)
        if sha256_file(checkpoint) != receipt['checkpoint_sha256']:
            raise ValueError('Frozen checkpoint differs from its original training receipt')
        if not original['checkpoint'] and receipt['protocol_sha256'] != sha256_file(semantic_old / 'protocol.json'):
            raise ValueError('Auxiliary checkpoint training protocol differs')
        fold.update(checkpoint=str(checkpoint), checkpoint_sha256=sha256_file(checkpoint),
            expected_data_manifest_sha256=receipt['metadata']['data_manifest_sha256'],
            original_training_receipt=str(receipt_path),
            original_training_receipt_sha256=sha256_file(receipt_path),
            original_predictions_manifest=str(semantic_old / 'folds' / fold['id'] / 'predictions_manifest.json'),
            original_predictions_manifest_sha256=old_metrics['folds'][fold['id']]['predictions_manifest_sha256'],
            fitting='reuse_verified_20261106_frozen_evaluator')
        if sha256_file(fold['original_predictions_manifest']) != fold['original_predictions_manifest_sha256']:
            raise ValueError('Original prediction manifest differs from completed Nov6 metrics')
        verify_metadata(receipt['metadata'], fold, protocol)
    common.validate_folds(protocol['folds'], protocol['mode'])
    common.write_json(semantic / 'protocol.json', protocol)
    return protocol


def verify_checkpoints(output=OUT):
    import torch
    semantic, protocol, _ = common.load_context(output)
    if sha256_file(protocol['reused_protocol']) != protocol['reused_protocol_sha256']:
        raise ValueError('Original frozen protocol changed')
    records = []
    for fold in protocol['folds']:
        if sha256_file(fold['original_training_receipt']) != fold['original_training_receipt_sha256']:
            raise ValueError('Frozen checkpoint training receipt changed')
        checkpoint = common.fold_checkpoint(semantic, fold)
        if sha256_file(checkpoint) != fold['checkpoint_sha256']:
            raise ValueError('Frozen checkpoint changed')
        saved = torch.load(str(checkpoint), map_location='cpu')
        verify_metadata(saved['metadata'], fold, protocol)
        records.append(dict(id=fold['id'], checkpoint=str(checkpoint),
            checkpoint_sha256=fold['checkpoint_sha256'], train_frames=fold['train_frames'],
            predict_frames=fold['predict_frames'], no_training=True))
        del saved
    common.replace_json(semantic / 'frozen_evaluators_verified.json', dict(status='complete',
        protocol_sha256=sha256_file(semantic / 'protocol.json'), folds=records,
        no_training=True, no_model_selection=True))


def seed_auxiliary_render_cache(output=OUT):
    from render_boundary_semantic_views import cached_receipt_valid
    old = common.read_json(PREVIOUS / 'comparison/auxiliary_renders/manifest.json')
    registry = common.read_json(Path(output) / 'comparison/models.json')
    models = {row['id']: row for row in registry['models']}
    for group, receipt in old['models'].items():
        model = models[group]
        if model['ply'] != receipt['ply'] or sha256_file(model['ply']) != receipt['ply_sha256']:
            raise ValueError('Reused auxiliary PLY differs from registry')
        for relative, digest in receipt['renderer_sources'].items():
            if sha256_file(ROOT / relative) != digest:
                raise ValueError('Auxiliary renderer source changed: ' + relative)
        if not cached_receipt_valid(receipt, {key: receipt[key] for key in (
                'model_id', 'ply', 'ply_sha256', 'renderer_sources', 'model_camera_fingerprints')}):
            raise ValueError('Previous native render receipt is incomplete')
        common.write_json(Path(output) / 'comparison/auxiliary_renders' / group / 'render_receipt.json', receipt)
    common.replace_json(Path(output) / 'comparison/auxiliary_render_reuse.json', dict(
        status='complete', original_manifest=str(PREVIOUS / 'comparison/auxiliary_renders/manifest.json'),
        original_manifest_sha256=sha256_file(PREVIOUS / 'comparison/auxiliary_renders/manifest.json'),
        models=sorted(old['models']), content_and_renderer_verified=True))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=OUT)
    parser.add_argument('--phase', choices=['prepare', 'verify', 'infer', 'evaluate'], required=True)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    install_frozen_protocol(args.root)
    if args.phase == 'prepare':
        prepare(args.root)
        seed_auxiliary_render_cache(args.root)
    elif args.phase == 'verify':
        verify_checkpoints(args.root)
    elif args.phase == 'infer':
        verify_checkpoints(args.root)
        common.infer_folds(args.root, args)
    else:
        common.evaluate_folds(args.root)


if __name__ == '__main__':
    main()
