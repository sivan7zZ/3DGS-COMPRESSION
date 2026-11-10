"""Audit annotation identity and prepare independent City semantic evaluation.

The audit/prepare phases use CPU file checks only. Country annotations must not
be assigned to City Gaussians just because both datasets use frame_NNN names.
All labels are registered with the actual model's COLMAP cameras.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import tempfile

from PIL import Image

from utils.city_semantic_data import (
    CLASS_NAMES, INNER_TRAIN, INNER_VAL, TEST, prepare_dataset, sha256_file,
)


ROOT = Path(__file__).resolve().parent
DEFAULT_ROOT = ROOT / 'output/city2_boundary_weighted_20261106'
MODEL_SOURCE = Path('/home/zhangys/gaussian-splatting（原生）/data/city')
LABEL_ROOT = Path('/home/zhangys/semantic_data/3D-AS_dataset')
HISTORICAL_ROOT = ROOT / 'output/city2_comparison_20261104/semantic'
NON_TEST = sorted(INNER_TRAIN + INNER_VAL)
AUXILIARY_FOLDS = [NON_TEST[first:first + 2] for first in range(0, 8, 2)]


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n'
    if path.exists() and path.read_text() != text:
        raise FileExistsError('Refusing to overwrite a different audit/protocol: ' + str(path))
    path.write_text(text)


def replace_json(path, value):
    """Write progress atomically; scientific input/protocol files use write_json."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix='.semantic-', dir=str(path.parent))
    try:
        with os.fdopen(descriptor, 'w') as stream:
            json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
            stream.write('\n')
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def inspect_annotation(folder, model_inputs):
    """Identity means file AND decoded RGB pixels, never a shared basename."""
    folder = Path(folder)
    rgb_path = folder / 'frame.png'
    with Image.open(rgb_path) as image:
        rgb = image.convert('RGB')
        size = list(rgb.size)
        pixel_sha = hashlib.sha256(rgb.tobytes()).hexdigest()
    raw_sha = sha256_file(rgb_path)
    matched = sorted(name for name, row in model_inputs.items()
                     if row['rgb_pixel_sha256'] == pixel_sha and row['size_wh'] == size)
    same_name = model_inputs.get(folder.name)
    annotation = read_json(folder / 'frame.json')
    class_names = sorted({row['label'] for row in annotation.get('shapes', [])})
    mask_files = {p.stem: {'path': str(p.resolve()), 'sha256': sha256_file(p)}
                  for p in sorted(folder.glob('*.png')) if p.stem not in ('frame', 'mask')}
    return dict(frame=folder.name, annotation_rgb=str(rgb_path.resolve()),
                annotation_rgb_sha256=raw_sha, annotation_rgb_pixel_sha256=pixel_sha,
                size_wh=size, matching_model_input_frames=matched,
                same_name_input=same_name,
                same_name_file_identical=bool(same_name and same_name['sha256'] == raw_sha),
                same_name_pixels_identical=bool(same_name and same_name['rgb_pixel_sha256'] == pixel_sha),
                annotation_json_sha256=sha256_file(folder / 'frame.json'),
                annotation_json_size_wh=[annotation.get('imageWidth'), annotation.get('imageHeight')],
                shape_classes=class_names, mask_files=mask_files,
                union_mask_sha256=sha256_file(folder / 'mask.png'))


def camera_fingerprints(source):
    loader = ROOT / 'scene/colmap_loader.py'
    spec = importlib.util.spec_from_file_location('boundary_label_audit_colmap', str(loader))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    result = {}
    for relative in ('sparse/0/cameras.bin', 'sparse/0/images.bin',
                     'distorted/sparse/0/cameras.bin', 'distorted/sparse/0/images.bin'):
        path = Path(source) / relative
        if not path.is_file():
            continue
        row = {'path': str(path.resolve()), 'sha256': sha256_file(path)}
        if path.name == 'cameras.bin':
            row['cameras'] = [dict(id=int(cam.id), model=cam.model, width=int(cam.width),
                                   height=int(cam.height), params=list(map(float, cam.params)))
                              for cam in module.read_intrinsics_binary(str(path)).values()]
        result[relative] = row
    return result


def audit_labels(output, source=MODEL_SOURCE, label_root=LABEL_ROOT):
    output, source, label_root = Path(output), Path(source), Path(label_root)
    inputs = {}
    for path in sorted((source / 'input').glob('*.png')):
        with Image.open(path) as image:
            rgb = image.convert('RGB')
            inputs[path.stem] = dict(path=str(path.resolve()), sha256=sha256_file(path),
                rgb_pixel_sha256=hashlib.sha256(rgb.tobytes()).hexdigest(), size_wh=list(rgb.size))
    scenes = {}
    for scene in ('Country', 'City'):
        scene_root = label_root / scene / 'scene0'
        segmentation = scene_root / 'segmentation'
        frames = [inspect_annotation(folder, inputs) for folder in sorted(segmentation.iterdir())
                  if folder.is_dir()]
        matched = [row['frame'] for row in frames if row['matching_model_input_frames']]
        scenes[scene] = dict(source=str(segmentation.resolve()), n_annotation_frames=len(frames),
            classes=sorted({name for row in frames for name in row['shape_classes']}),
            n_frames_matching_model_input=len(matched), matching_frames=matched,
            camera_fingerprints=camera_fingerprints(scene_root), frames=frames)
    annotated = [row['frame'] for row in scenes['City']['frames']]
    model_test = ['frame_%03d' % i for i in range(0, len(inputs), 8)]
    audit = dict(schema='annotation_model_identity_audit_v1', status='complete',
        model_source=str(source.resolve()), model_input_count=len(inputs),
        model_camera_fingerprints=camera_fingerprints(source),
        requested_segmentation_source=scenes['Country']['source'],
        requested_source_matches_current_model=(scenes['Country']['n_frames_matching_model_input'] == len(annotated)),
        identity_rule='Decoded RGB pixel SHA256 and dimensions; also retain original PNG SHA256. A shared frame basename is insufficient.',
        scenes=scenes, model_test_frames=model_test,
        matching_city_annotation_frames=annotated,
        matching_city_annotated_test_frames=sorted(set(annotated) & set(model_test)),
        matching_city_annotated_training_frames=sorted(set(annotated) - set(model_test)),
        existing_evaluator=dict(checkpoint=str(HISTORICAL_ROOT / 'model/model.pt'),
            fitted_on=NON_TEST, held_out=TEST,
            qualification='Ten matching City annotations exist. Only two intersect the nine held-out 3DGS cameras; the eight other annotations fitted the historical evaluator.'),
        conclusion='Country is a separate agricultural scene with an additional Agriculture class. Its labels cannot evaluate the current City model. All ten City annotation RGBs match actual model input pixels. City labels require registration using actual model cameras, not semantic_data/City cameras.')
    write_json(output / 'label_audit.json', audit)
    lines = ['# 标签与当前 City 模型核对', '',
        'Country/scene0/segmentation 与 City/scene0/segmentation 均有10张标注。逐张比较PNG与解码RGB像素、遍历当前模型全部输入影像后，Country的10张均不匹配；City的10张全部与模型输入逐字节匹配。两个目录虽然同为frame_000等文件名，实际属于不同场景。', '',
        '| 来源 | 标签帧数 | 与当前模型匹配 | 原标注尺寸 | 语义类别 |',
        '|---|---:|---:|---|---|']
    for name, row in scenes.items():
        sizes = sorted({tuple(frame['size_wh']) for frame in row['frames']})
        lines.append('| %s | %d | %d | %s | %s |' % (name, row['n_annotation_frames'],
            row['n_frames_matching_model_input'], ', '.join('%d×%d' % size for size in sizes), ', '.join(row['classes'])))
    lines += ['', '当前模型的去畸变渲染相机是1592×894；Country是1574×888。City数据集副本的相机也是另一套重建（1592×895），因此匹配的City原始标签仍须用当前模型data/city的真实COLMAP相机注册，不能直接使用数据集副本的去畸变标签。', '',
        '此前“2张”指9个独立测试视角中的标签交集frame_000、frame_056，并非只有2张标签。其余8张标签处在3DGS训练视角，且用于历史分割评估器拟合。', '',
        '全10帧方案将2张独立测试帧、8张训练视角辅助帧和10帧汇总分别报告。独立测试继续使用历史冻结评估器；额外8帧采用4折留2帧预测，每折只用其他6个非测试原RGB标签拟合，固定1500步、不按该折评测标签选步。所有评估器均不使用frame_000、frame_056标签拟合。压缩训练不读取这些人工语义标签。', '',
        '审计与准备阶段不启动训练或GPU。原始影像、标注、COLMAP与历史结果保持原样。逐帧PNG/像素/标签SHA256、尺寸、类别与相机指纹详见 [label_audit.json](label_audit.json)。', '']
    (output / 'README_标签核对.md').write_text('\n'.join(lines))
    return audit


def build_protocol(manifest, historical_protocol, mode='city_oof'):
    if mode not in ('city_oof', 'city_heldout'):
        raise ValueError('Explicitly select city_oof or city_heldout after reviewing the label audit')
    available = sorted(row['frame'] for row in manifest['frames'])
    if available != sorted(NON_TEST + TEST):
        raise ValueError('Expected exactly the ten reviewed City annotation frames')
    metadata = historical_protocol['metadata']
    if sorted(metadata['train_frames']) != NON_TEST or metadata.get('test_frames_used_for_training_or_selection'):
        raise ValueError('Historical evaluator training provenance differs from reviewed protocol')
    if set(metadata['train_frames']) & set(TEST):
        raise ValueError('Independent test annotation leaked into historical fitting')
    folds = [dict(id='main_heldout_2', role='independent_3dgs_test',
        predict_frames=TEST, train_frames=NON_TEST, validation_frames=[],
        fitting='reuse_verified_historical_frozen_evaluator',
        checkpoint=historical_protocol['checkpoint'],
        checkpoint_sha256=historical_protocol['checkpoint_sha256'],
        steps=int(historical_protocol['selected_step']), seed=20261104)]
    if mode == 'city_oof':
        for index, held_out in enumerate(AUXILIARY_FOLDS, 1):
            folds.append(dict(id='aux_fold_%d' % index, role='3dgs_training_view_auxiliary',
                predict_frames=held_out,
                train_frames=sorted(set(NON_TEST) - set(held_out)), validation_frames=[],
                fitting='fixed_steps_no_evaluation_frame_selection', steps=1500, seed=20261104,
                checkpoint=None, checkpoint_sha256=None))
    validate_folds(folds, mode)
    return dict(schema='city_boundary_semantic_protocol_v1', status='prepared_cpu_only', mode=mode,
        classes=CLASS_NAMES, ignore_label=0, folds=folds,
        independent_test_frames=TEST, auxiliary_training_view_frames=NON_TEST if mode == 'city_oof' else [],
        aggregate_groups={'independent_test_2': TEST,
            **({'auxiliary_training_views_8': NON_TEST, 'all_annotated_10': sorted(NON_TEST + TEST)}
               if mode == 'city_oof' else {})},
        fitting_rules=['Only original RGB and annotation labels fit the scene evaluator.',
            'No compressed RGB, independent test label, or current fold prediction label fits/selects its evaluator.',
            'Each fold uses the same frozen weights for GT RGB and every compared model.',
            'All ten frames are not ten independent 3DGS test views: eight are training-view auxiliary checks.'],
        recall_definition='Same-class 8-connected semantic RegionRecall; minimum GT area 16px, one-to-one matching at IoU>=0.5. Not physical-instance Recall.',
        semantic_metrics=['region_recall', 'b_miou', 'miou', 'macc'],
        boundary_iou_rule='Existing common protocol: inner band at 2% image diagonal; exclude GT void boundary bands.',
        visual_metrics='PSNR, SSIM, standard VGG LPIPS[-1,1] use the original nine independent test frames; auxiliary view metrics are reported separately.')


def validate_folds(folds, mode):
    seen = []
    for fold in folds:
        train, validation, predict = map(set, (fold['train_frames'], fold['validation_frames'], fold['predict_frames']))
        if train & predict or validation & predict or (train | validation) & set(TEST):
            raise ValueError('Evaluation label leaked into fold fitting or selection: ' + fold['id'])
        if not train <= set(NON_TEST) or validation:
            raise ValueError('Unexpected training or validation labels')
        if fold['role'] == '3dgs_training_view_auxiliary' and (len(train) != 6 or len(predict) != 2):
            raise ValueError('Each auxiliary fold must fit six and predict two non-test frames')
        seen.extend(fold['predict_frames'])
    expected = TEST if mode == 'city_heldout' else sorted(NON_TEST + TEST)
    if sorted(seen) != sorted(expected) or len(seen) != len(set(seen)):
        raise ValueError('Every selected annotation frame must be predicted exactly once')


def prepare(output, source=MODEL_SOURCE, label_root=LABEL_ROOT, mode='city_oof'):
    output, source, label_root = Path(output), Path(source), Path(label_root)
    audit = audit_labels(output, source, label_root)
    if audit['scenes']['City']['n_frames_matching_model_input'] != 10:
        raise ValueError('City annotations no longer match the actual model inputs')
    semantic = output / 'comparison/semantic'
    manifest_path = prepare_dataset(semantic, source, label_root / 'City/scene0/segmentation', ROOT)
    manifest = read_json(manifest_path)
    original_manifest = read_json(HISTORICAL_ROOT / 'dataset_manifest.json')
    old_frames = {row['frame']: row for row in original_manifest['frames']}
    stable_keys = ('rgb_sha256', 'raw_rgb_sha256', 'label_sha256', 'source_mask_sha256', 'shape')
    for row in manifest['frames']:
        for key in stable_keys:
            if row[key] != old_frames[row['frame']][key]:
                raise ValueError('Registered City frame differs from historical protocol: ' + row['frame'] + '/' + key)
    for key in ('distorted_cameras_sha256', 'render_cameras_sha256', 'map_x_array_sha256', 'map_y_array_sha256'):
        if manifest['registration'][key] != original_manifest['registration'][key]:
            raise ValueError('Actual model camera registration changed: ' + key)
    historical_path = HISTORICAL_ROOT / 'model/training_protocol.json'
    historical = read_json(historical_path)
    if sha256_file(historical['checkpoint']) != historical['checkpoint_sha256']:
        raise ValueError('Historical frozen segmenter changed')
    protocol = build_protocol(manifest, historical, mode)
    protocol.update(data_manifest=str(manifest_path), data_manifest_sha256=sha256_file(manifest_path),
        historical_data_manifest=str(HISTORICAL_ROOT / 'dataset_manifest.json'),
        historical_data_manifest_sha256=sha256_file(HISTORICAL_ROOT / 'dataset_manifest.json'),
        historical_training_protocol=str(historical_path), historical_training_protocol_sha256=sha256_file(historical_path),
        registration_equal_to_historical=True, label_audit_sha256=sha256_file(output / 'label_audit.json'))
    write_json(semantic / 'protocol.json', protocol)
    return semantic / 'protocol.json'


def load_context(output):
    semantic = Path(output) / 'comparison/semantic'
    protocol_path = semantic / 'protocol.json'
    protocol = read_json(protocol_path)
    validate_folds(protocol['folds'], protocol['mode'])
    if sha256_file(protocol['data_manifest']) != protocol['data_manifest_sha256']:
        raise ValueError('Registered annotation manifest changed')
    if sha256_file(protocol['historical_data_manifest']) != protocol['historical_data_manifest_sha256']:
        raise ValueError('Historical dataset manifest changed')
    if sha256_file(protocol['historical_training_protocol']) != protocol['historical_training_protocol_sha256']:
        raise ValueError('Historical evaluator training receipt changed')
    manifest = read_json(protocol['data_manifest'])
    for row in manifest['frames']:
        if sha256_file(row['rgb_path']) != row['rgb_sha256'] or sha256_file(row['label_path']) != row['label_sha256']:
            raise ValueError('RGB/GT label content changed: ' + row['frame'])
    return semantic, protocol, manifest


def fold_checkpoint(semantic, fold):
    return Path(fold['checkpoint']) if fold['checkpoint'] else semantic / 'folds' / fold['id'] / 'model.pt'


def verify_metadata(metadata, fold, protocol):
    expected_hash = (protocol['historical_data_manifest_sha256'] if fold['checkpoint']
                     else protocol['data_manifest_sha256'])
    if metadata['data_manifest_sha256'] != expected_hash:
        raise ValueError('Evaluator belongs to a different registered dataset')
    if sorted(metadata['train_frames']) != sorted(fold['train_frames']):
        raise ValueError('Evaluator training frames differ from the predeclared fold')
    if metadata.get('test_frames_used_for_training_or_selection'):
        raise ValueError('Independent test labels used by evaluator')
    if set(metadata['train_frames']) & (set(TEST) | set(fold['predict_frames'])):
        raise ValueError('Fold evaluation label used to fit evaluator')
    if metadata.get('validation_frames'):
        raise ValueError('Final frozen evaluator must not have evaluation-based model selection')
    if int(metadata['selected_step']) != fold['steps']:
        raise ValueError('Unexpected evaluator training duration')


def fit_folds(output, args):
    """Run only after the compression queue releases the GPU.

    Completed folds are hash-verified and skipped with --resume. If interrupted
    inside a fold, that fold restarts at the same deterministic initialization;
    no partial optimizer state is presented as a resumed complete training run.
    """
    import torch
    from city_semantic import train_run

    semantic, protocol, manifest = load_context(output)
    if args.device.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('GPU unavailable')
    state_path = semantic / 'fold_training.json'
    state = read_json(state_path) if state_path.exists() else dict(status='running', completed=[])
    if state_path.exists() and not args.resume:
        raise FileExistsError('Use --resume to verify/skip completed semantic folds')
    state.update(status='running', protocol_sha256=sha256_file(semantic / 'protocol.json'),
                 current_fold=None, interruption_policy='Restart an incomplete fold from its fixed seed; skip only hash-verified completed folds.')
    for fold in protocol['folds']:
        checkpoint = fold_checkpoint(semantic, fold)
        state['current_fold'] = fold['id']
        replace_json(state_path, state)
        if checkpoint.exists():
            if fold['checkpoint']:
                if sha256_file(checkpoint) != fold['checkpoint_sha256']:
                    raise ValueError('Historical evaluator checkpoint changed')
            else:
                receipt = read_json(checkpoint.parent / 'training_protocol.json')
                if receipt['checkpoint_sha256'] != sha256_file(checkpoint):
                    raise ValueError('Completed auxiliary evaluator checkpoint changed')
                if receipt['protocol_sha256'] != sha256_file(semantic / 'protocol.json'):
                    raise ValueError('Auxiliary evaluator belongs to a different fold protocol')
            saved = torch.load(str(checkpoint), map_location='cpu')
            verify_metadata(saved['metadata'], fold, protocol)
            del saved
        else:
            folder = checkpoint.parent
            folder.mkdir(parents=True, exist_ok=True)
            train_args = argparse.Namespace(device=args.device, seed=fold['seed'],
                pretrained=args.pretrained, data_manifest=protocol['data_manifest'],
                lr=2e-4, batch_size=4, crop=384, val_interval=100)
            saved, selected = train_run(manifest, fold['train_frames'], [], folder,
                train_args, fold['steps'], 'auxiliary_fixed_1500')
            saved['metadata'].update(selection_source='Fixed 1500 steps declared before rendering; no evaluation-label selection',
                fold_id=fold['id'], excluded_independent_test_frames=TEST,
                test_frames_used_for_training_or_selection=[])
            verify_metadata(saved['metadata'], fold, protocol)
            temporary = checkpoint.with_suffix('.partial.pt')
            torch.save(saved, str(temporary))
            temporary.replace(checkpoint)
            replace_json(folder / 'training_protocol.json', dict(status='complete',
                checkpoint=str(checkpoint), checkpoint_sha256=sha256_file(checkpoint),
                selected_step=selected, metadata=saved['metadata'],
                protocol_sha256=sha256_file(semantic / 'protocol.json'),
                qualification='Auxiliary 3DGS training-view evaluator; held-out fold labels and both independent test labels excluded from fitting and selection.'))
            del saved
        if fold['id'] not in state['completed']:
            state['completed'].append(fold['id'])
        replace_json(state_path, state)
        print('Semantic fold ready: ' + fold['id'], flush=True)
    state.update(status='complete', current_fold=None)
    replace_json(state_path, state)


def validated_models(registry):
    result, seen = [], set()
    for row in registry['models']:
        if row.get('status') == 'missing':
            continue
        name = row['id']
        if not re.fullmatch(r'[A-Za-z0-9_.-]+', name) or name in ('.', '..') or name in seen or name == 'GT':
            raise ValueError('Unsafe or duplicate model ID: ' + name)
        seen.add(name)
        result.append(row)
    return result


def prepare_fold_inputs(output, semantic, protocol, manifest):
    comparison = Path(output) / 'comparison'
    registry_path = comparison / 'models.json'
    registry = read_json(registry_path)
    models = validated_models(registry)
    frame_index = {name: index for index, name in enumerate(registry['test_frames'])}
    frames = {row['frame']: row for row in manifest['frames']}
    auxiliary = None
    if protocol['mode'] == 'city_oof':
        auxiliary = read_json(comparison / 'auxiliary_renders/manifest.json')
        if auxiliary['status'] != 'complete' or auxiliary['models_json_sha256'] != sha256_file(registry_path):
            raise ValueError('Auxiliary renders do not belong to the completed comparison model registry')
    for fold in protocol['folds']:
        images, groups = [], {}
        for group in ['GT'] + [row['id'] for row in models]:
            ids = []
            for name in fold['predict_frames']:
                if group == 'GT':
                    image = Path(frames[name]['rgb_path'])
                elif name in TEST:
                    model = next(row for row in models if row['id'] == group)
                    image = Path(model['common_render_dir']) / ('%05d.png' % frame_index[name])
                else:
                    row = auxiliary['models'][group]['frames'][name]
                    image = Path(row['render_path'])
                    if sha256_file(image) != row['render_sha256']:
                        raise ValueError('Auxiliary model render changed')
                with Image.open(image) as value:
                    if list(reversed(value.size)) != frames[name]['shape']:
                        raise ValueError('Render and registered label dimensions differ')
                ident = group + '/' + name
                images.append(dict(id=ident, frame=name, path=str(image.resolve()), image_sha256=sha256_file(image)))
                ids.append(ident)
            groups[group] = ids
        path = semantic / 'folds' / fold['id'] / 'inputs.json'
        write_json(path, dict(images=images, groups=groups, predict_frames=fold['predict_frames'],
            models={row['id']: row for row in models}, models_json_sha256=sha256_file(registry_path),
            protocol_sha256=sha256_file(semantic / 'protocol.json'),
            data_manifest_sha256=protocol['data_manifest_sha256']))


def historical_prediction_cache(checkpoint_sha256):
    """Reuse only identical RGB content under the identical frozen checkpoint."""
    paths = [ROOT / 'output/city2_comparison_20261104/semantic/predictions_main/predictions_manifest.json',
             ROOT / 'output/city2_pup_rir_dpra_20261105/comparison/semantic/predictions_main/predictions_manifest.json']
    cache = {}
    for path in paths:
        if not path.is_file():
            continue
        value = read_json(path)
        if value['checkpoint_sha256'] != checkpoint_sha256:
            continue
        for row in value['predictions']:
            if sha256_file(row['image_path']) != row['image_sha256'] or sha256_file(row['prediction_path']) != row['prediction_sha256']:
                raise ValueError('Historical semantic prediction source changed')
            cache[row['image_sha256']] = row
    return cache


def infer_folds(output, args):
    import torch
    from city_semantic import infer

    semantic, protocol, manifest = load_context(output)
    prepare_fold_inputs(output, semantic, protocol, manifest)
    for fold in protocol['folds']:
        folder = semantic / 'folds' / fold['id']
        checkpoint = fold_checkpoint(semantic, fold)
        saved = torch.load(str(checkpoint), map_location='cpu')
        verify_metadata(saved['metadata'], fold, protocol)
        metadata = saved['metadata']
        del saved
        checkpoint_sha = sha256_file(checkpoint)
        inputs_path = folder / 'inputs.json'
        inputs = read_json(inputs_path)
        destination = folder / 'predictions_manifest.json'
        if destination.exists():
            if not args.resume:
                raise FileExistsError('Use --resume to verify/skip completed fold predictions')
            old = read_json(destination)
            if old['inputs_json_sha256'] != sha256_file(inputs_path):
                raise ValueError('Completed predictions belong to different fold inputs')
            verify_predictions(old, inputs, checkpoint_sha)
            print('Semantic predictions verified: ' + fold['id'], flush=True)
            continue
        cache = historical_prediction_cache(checkpoint_sha) if fold['checkpoint'] else {}
        rows, missing = [], []
        for item in inputs['images']:
            cached = cache.get(item['image_sha256'])
            if cached is None:
                missing.append(item)
            else:
                rows.append(dict(cached, id=item['id'], image_path=item['path'],
                    reuse_provenance='Identical rendered/GT RGB SHA256 under identical historical frozen checkpoint'))
        if missing:
            missing_path = folder / 'inputs_missing.json'
            write_json(missing_path, dict(images=missing))
            infer(argparse.Namespace(checkpoint=str(checkpoint), inputs_json=str(missing_path),
                output=str(folder / 'predictions_new'), device=args.device, tile=512, stride=384, batch_size=4))
            rows.extend(read_json(folder / 'predictions_new/predictions_manifest.json')['predictions'])
        by_id = {row['id']: row for row in rows}
        result = dict(schema='city_boundary_fold_predictions_v1', checkpoint=str(checkpoint),
            checkpoint_sha256=checkpoint_sha, metadata=metadata,
            inputs_json_sha256=sha256_file(inputs_path), predictions=[by_id[item['id']] for item in inputs['images']],
            new_prediction_count=len(missing), reused_prediction_count=len(rows) - len(missing),
            inference_settings=dict(tile=512, stride=384, batch_size=4, native_resolution=True,
                aggregation='mean softmax probabilities'))
        verify_predictions(result, inputs, checkpoint_sha)
        write_json(destination, result)
        print('Semantic predictions ready: ' + fold['id'], flush=True)


def verify_predictions(predictions, inputs, checkpoint_sha):
    if predictions['checkpoint_sha256'] != checkpoint_sha:
        raise ValueError('Fold predictions belong to a different checkpoint')
    by_id = {row['id']: row for row in predictions['predictions']}
    if len(by_id) != len(predictions['predictions']) or set(by_id) != {row['id'] for row in inputs['images']}:
        raise ValueError('Fold predictions missing, duplicated or unrelated to input IDs')
    for item in inputs['images']:
        row = by_id[item['id']]
        if row['image_sha256'] != item['image_sha256'] or sha256_file(item['path']) != item['image_sha256']:
            raise ValueError('Fold source RGB changed')
        if sha256_file(row['prediction_path']) != row['prediction_sha256']:
            raise ValueError('Fold prediction changed')


def evaluate_folds(output):
    import csv
    import numpy as np
    from utils.semantic_comparison_metrics import evaluate_semantics

    semantic, protocol, manifest = load_context(output)
    frames = {row['frame']: row for row in manifest['frames']}
    all_predictions, all_metadata, model_order = {}, {}, None
    for fold in protocol['folds']:
        folder = semantic / 'folds' / fold['id']
        inputs = read_json(folder / 'inputs.json')
        if inputs['models_json_sha256'] != sha256_file(Path(output) / 'comparison/models.json'):
            raise ValueError('Model registry changed after fold input preparation')
        predictions = read_json(folder / 'predictions_manifest.json')
        verify_metadata(predictions['metadata'], fold, protocol)
        if sha256_file(predictions['checkpoint']) != predictions['checkpoint_sha256']:
            raise ValueError('Frozen evaluator weights changed')
        if predictions['inputs_json_sha256'] != sha256_file(folder / 'inputs.json'):
            raise ValueError('Prediction input manifest changed')
        verify_predictions(predictions, inputs, predictions['checkpoint_sha256'])
        if model_order is None:
            model_order = list(inputs['groups'])
        elif list(inputs['groups']) != model_order:
            raise ValueError('Model registry differs across folds')
        for row in predictions['predictions']:
            if row['id'] in all_predictions:
                raise ValueError('Duplicate frame predicted by multiple folds')
            all_predictions[row['id']] = row
        all_metadata[fold['id']] = dict(checkpoint=predictions['checkpoint'],
            checkpoint_sha256=predictions['checkpoint_sha256'],
            predictions_manifest_sha256=sha256_file(folder / 'predictions_manifest.json'),
            train_frames=fold['train_frames'], predict_frames=fold['predict_frames'])
    scopes = {}
    csv_rows = []
    for scope, names in protocol['aggregate_groups'].items():
        targets = [np.asarray(Image.open(frames[name]['label_path']).convert('L')) for name in names]
        results = {}
        for group in model_order:
            predicted = [np.asarray(Image.open(all_predictions[group + '/' + name]['prediction_path']).convert('L')) for name in names]
            metrics = evaluate_semantics(predicted, targets)
            results[group] = dict(metrics=metrics, semantic_frames=names, n_semantic_views=len(names))
            csv_rows.append(dict(scope=scope, model_id=group, n_views=len(names),
                region_recall=metrics['region_recall']['recall'], region_tp=metrics['region_recall']['tp'],
                region_gt=metrics['region_recall']['gt'], small_region_recall=metrics['region_recall']['small_recall'],
                b_miou=metrics['b_miou'], miou=metrics['miou'], macc=metrics['macc']))
        scopes[scope] = dict(frames=names, results=results)
    result = dict(status='complete', schema='city_boundary_semantic_results_v1',
        protocol_sha256=sha256_file(semantic / 'protocol.json'),
        data_manifest_sha256=protocol['data_manifest_sha256'], folds=all_metadata, scopes=scopes,
        qualification='Two independent 3DGS test views are the main semantic result. Eight 3DGS training views use additional leave-two-out evaluators, and are auxiliary. All ten-frame summary is mixed-view support, not an independent ten-test-view claim.')
    replace_json(semantic / 'metrics.json', result)
    with (semantic / 'metrics.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(csv_rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(csv_rows)
    lines = ['# 原生SH3压缩模型：统一语义评测', '',
        '所有方法共享对应折的固定分割器。Recall表示同类别连通语义区域的一对一IoU≥0.5召回率，不是人工实例召回。0类忽略；mIoU、mAcc及B-mIoU沿用原统一定义。', '',
        '主表的2帧均未被3DGS训练或分割评估器拟合。辅助8帧虽由各折语义评估器留出，但属于3DGS训练视角。全10帧混合汇总另列，不能称10张独立测试结果。', '']
    for scope, data in scopes.items():
        lines += ['## ' + scope, '', '帧：' + '、'.join(data['frames']), '',
            '| 模型 | Recall↑ (%) | B-mIoU↑ (%) | mIoU↑ (%) | mAcc↑ (%) |',
            '|---|---:|---:|---:|---:|']
        for group, row in data['results'].items():
            metrics = row['metrics']
            values = [metrics['region_recall']['recall'], metrics['b_miou'], metrics['miou'], metrics['macc']]
            lines.append('| %s | %s |' % (group, ' | '.join('—' if value is None else '%.3f' % (100 * value) for value in values)))
        lines.append('')
    (semantic / 'README_统一语义指标.md').write_text('\n'.join(lines))
    print(str(semantic / 'metrics.json'), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=DEFAULT_ROOT)
    parser.add_argument('--source', type=Path, default=MODEL_SOURCE)
    parser.add_argument('--label_root', type=Path, default=LABEL_ROOT)
    parser.add_argument('--phase', choices=['audit', 'prepare', 'train', 'infer', 'evaluate'], default='audit')
    parser.add_argument('--protocol', choices=['city_oof', 'city_heldout'])
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--pretrained', default='/home/zhangys/.cache/torch/hub/checkpoints/resnet34-333f7ec4.pth')
    args = parser.parse_args()
    args.root.mkdir(parents=True, exist_ok=True)
    if args.phase == 'audit':
        audit_labels(args.root, args.source, args.label_root)
        print(str(args.root / 'label_audit.json'))
    elif args.phase == 'prepare':
        if args.protocol is None:
            parser.error('--phase prepare requires an explicit --protocol after label review')
        print(str(prepare(args.root, args.source, args.label_root, args.protocol)))
    elif args.phase == 'train':
        fit_folds(args.root, args)
    elif args.phase == 'infer':
        infer_folds(args.root, args)
    else:
        evaluate_folds(args.root)


if __name__ == '__main__':
    main()
