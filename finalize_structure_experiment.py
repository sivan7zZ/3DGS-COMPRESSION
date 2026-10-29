#!/usr/bin/env python3
"""Finalize this City's native SH3 structure comparison and optional SH2 exports.

This CPU-only postprocessor never trains a model or chooses physical examples
from a PSNR ranking.  The six review boxes are frozen before the new model is
examined.  Automatic structure scores cannot establish object disappearance;
the report keeps an explicit, separate assistant image-review record.
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import subprocess
import sys

import numpy as np


DEFAULT_REPO = Path('/home/zhangys/Prune-Wisely-Reconstruct-Sharply')
DEFAULT_ROOT = DEFAULT_REPO / 'output/city2_structure_guard_20261027'
DEFAULT_WORKSPACE = Path('/home/zhangys/Documents/Codex/2026-10-23/gt-semantic-3dgs-ply-ply-ply')
RAW_OUTPUT = Path('/home/zhangys/gaussian-splatting（原生）/output/city2')
RAW_DATA = Path('/home/zhangys/gaussian-splatting（原生）/data/city')
MODEL_ORDER = ('LG10', 'PWSR10', 'V2', 'LG20', 'PWSR20', 'Structure20')
MODEL_LABELS = {
    'LG10': 'LightGaussian 10×（外部方法）',
    'PWSR10': '原 PWSR 10×（全局 SPS）',
    'V2': 'v2 实例配额版 10×',
    'LG20': 'LightGaussian 20×（外部方法）',
    'PWSR20': '原 PWSR 20×（同预算对照）',
    'Structure20': '本次结构保护版 20×',
}


def file_sha256(path):
    digest = hashlib.sha256()
    with open(str(path), 'rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path):
    with open(str(path), encoding='utf-8') as stream:
        return json.load(stream)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    with open(str(temporary), 'w', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    os.replace(str(temporary), str(path))


def require_file(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(str(path))
    return path


def ply_header(path):
    """Read only the ASCII header, leaving large point data memory mapped later."""
    properties, count, format_name = [], None, None
    in_vertex = False
    with open(str(path), 'rb') as stream:
        if stream.readline().strip() != b'ply':
            raise ValueError('Not a PLY: ' + str(path))
        for _ in range(1024):
            line = stream.readline()
            if not line:
                raise ValueError('Truncated PLY header: ' + str(path))
            fields = line.decode('ascii').strip().split()
            if not fields:
                continue
            if fields[0] == 'format':
                format_name = fields[1]
            elif fields[0] == 'element':
                in_vertex = fields[1] == 'vertex'
                if in_vertex:
                    count = int(fields[2])
            elif fields[0] == 'property' and in_vertex:
                if len(fields) != 3 or fields[1] == 'list':
                    raise ValueError('Only scalar Gaussian PLY properties are supported')
                properties.append(fields[2])
            elif fields[0] == 'end_header':
                break
        else:
            raise ValueError('PLY header exceeds safe length')
    if count is None:
        raise ValueError('PLY has no vertex element')
    rest = sum(name.startswith('f_rest_') for name in properties)
    degree = int(round(math.sqrt(rest / 3. + 1.) - 1.))
    if rest != 3 * ((degree + 1) ** 2 - 1):
        raise ValueError('Invalid standard SH coefficient count')
    return dict(gaussians=count, sh_degree=degree, properties=properties,
                property_count=len(properties), format=format_name, bytes=Path(path).stat().st_size)


def model_specs(repo, root, proposed_root=None):
    previous = repo / 'output/city2_instance_guard_20261026'
    proposed_root = proposed_root or root / 'protected20'
    return {
        'Native': (root / 'reference/original', 30000),
        'LG10': (root / 'reference/lightgaussian10', 35000),
        'PWSR10': (previous / 'control/stage3', 25199),
        'V2': (previous / 'protected/stage3', 25199),
        'LG20': (root / 'lightgaussian20/stage3', 35000),
        'PWSR20': (root / 'control20/stage3', 25199),
        'Structure20': (proposed_root / 'stage3', 25199),
    }


def report_directory(args):
    return args.experiment_root / 'reports' / args.report_name


def legacy_fixed(args):
    return (args.selection_mode == 'fixed' and args.report_name == 'primary_sh3' and
            args.proposed_root == args.experiment_root / 'protected20')


def artifact_directory(args):
    """Preserve fixed-version locations; other runs have separate inventories."""
    return args.experiment_root if legacy_fixed(args) else args.proposed_root


def export_directory(args):
    return args.experiment_root / 'reports/exports' if legacy_fixed(args) else report_directory(args) / 'exports'


def selected_configuration(args):
    if args.config:
        path = require_file(args.config)
        return path, read_json(path)
    pipeline = args.proposed_root / 'pipeline_all.json'
    if pipeline.is_file() and isinstance(read_json(pipeline).get('config'), dict):
        return pipeline, read_json(pipeline)['config']
    path = require_file(args.repo / 'configs/city_structure20.json')
    return path, read_json(path)


def render_dir(spec):
    directory, iteration = spec
    return directory / ('test/ours_%d/renders' % iteration)


def standard_ply(spec):
    directory, iteration = spec
    return directory / ('point_cloud/iteration_%d/point_cloud.ply' % iteration)


def frame_names(path):
    return [Path(line.strip()).stem for line in path.read_text().splitlines() if line.strip()]


def validate_fixed_reviews(path, tests):
    reviews = read_json(path)
    if not isinstance(reviews, list) or len(reviews) != 6:
        raise ValueError('This experiment requires its six frozen physical review boxes')
    actual = [(Path(row['frame']).stem, tuple(row['bbox_xyxy'])) for row in reviews]
    expected = [
        ('frame_024', (1321, 30, 1380, 69)),
        ('frame_064', (1350, 821, 1373, 844)),
        ('frame_056', (1352, 70, 1382, 101)),
        ('frame_008', (1183, 77, 1232, 116)),
        ('frame_000', (157, 31, 247, 70)),
        ('frame_024', (83, 192, 111, 225)),
    ]
    if actual != expected or any(frame not in tests for frame, _ in actual):
        raise ValueError('Review coordinates/order changed; do not select new boxes after seeing results')
    if reviews[-1].get('baseline_visibility_loss_confirmed') is not False:
        raise ValueError('The sixth fixed case is a negative control that remains visible')
    return reviews


def validated_manual_review(path, reviews):
    if path is None:
        return dict(status='pending', reviewer=None, cases=[],
                    overall_conclusion='新模型尚未完成助手目视检查；以下自动指标不构成地物获救结论。')
    review = read_json(path)
    if review.get('status') not in ('pending', 'complete'):
        raise ValueError('manual review status must be pending or complete')
    expected = {(Path(row['frame']).stem, tuple(row['bbox_xyxy'])) for row in reviews}
    seen = set()
    for row in review.get('cases', []):
        key = (Path(row['frame']).stem, tuple(row['bbox_xyxy']))
        if key not in expected or key in seen:
            raise ValueError('Assistant review must refer once to each frozen case')
        seen.add(key)
        for name in ('new_model_observation', 'comparison_to_pwsr20', 'conclusion'):
            if not isinstance(row.get(name), str) or not row[name].strip():
                raise ValueError('Review requires nonempty ' + name)
    if review['status'] == 'complete' and (seen != expected or not review.get('reviewer')):
        raise ValueError('Complete assistant review requires all six fixed cases and reviewer attribution')
    review['source'] = str(Path(path).resolve())
    review['source_sha256'] = file_sha256(path)
    return review


def run_logged(command, log_path, cwd):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(str(log_path), 'w', encoding='utf-8') as stream:
        stream.write(json.dumps(command, ensure_ascii=False) + '\n')
        stream.flush()
        result = subprocess.run(command, cwd=str(cwd), stdout=stream, stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError('Postprocessor failed (%d), see %s' % (result.returncode, log_path))


def build_inventory(specs, final=False):
    inventory = {}
    for label, spec in specs.items():
        path = require_file(standard_ply(spec))
        record = ply_header(path)
        if record['sh_degree'] != 3 or record['property_count'] != 62:
            raise ValueError('Primary comparison requires unmodified standard SH3: ' + str(path))
        record.update(ply=str(path), model_dir=str(spec[0]), iteration=spec[1], renders=str(render_dir(spec)))
        inventory[label] = record
    source_bytes = inventory['Native']['bytes']
    source_n = inventory['Native']['gaussians']
    for record in inventory.values():
        record['size_compression_ratio'] = source_bytes / record['bytes']
        record['count_compression_ratio'] = source_n / record['gaussians']
    for label in ('LG20', 'PWSR20', 'Structure20'):
        if inventory[label]['gaussians'] != 89405:
            raise ValueError('20x comparison must use exactly 89,405 Gaussians: ' + label)
    if final:
        root = specs['Structure20'][0].parent
        spec = (root / 'sh2', 40000)
        path = require_file(standard_ply(spec))
        record = ply_header(path)
        if record['sh_degree'] != 2 or record['gaussians'] != 89405 or record['property_count'] != 41:
            raise ValueError('Final distilled model must be standard SH2 with the same exact count')
        record.update(ply=str(path), model_dir=str(spec[0]), iteration=40000, renders=str(render_dir(spec)),
                      size_compression_ratio=source_bytes / record['bytes'],
                      count_compression_ratio=source_n / record['gaussians'])
        inventory['Structure20_SH2'] = record
    return inventory


def evaluate_primary(args, specs, reviews_path, train_path, test_path):
    destination = report_directory(args)
    output = destination / 'visible_structure.json'
    native = specs['Native']
    gt = native[0] / ('test/ours_%d/gt' % native[1])
    tests = frame_names(test_path)
    for label, spec in specs.items():
        for index in range(len(tests)):
            require_file(render_dir(spec) / ('%05d.png' % index))
    if not args.skip_evaluation:
        command = [args.analysis_python, str(require_file(args.repo / 'evaluate_visible_structure.py')),
                   '--sam_masks', str(args.sam_masks), '--train_frames', str(train_path), '--test_frames', str(test_path),
                   '--source_images', str(args.source_images), '--ground_truth_dir', str(gt),
                   '--teacher_renders', str(render_dir(native)), '--baseline', 'PWSR20', '--proposed', 'Structure20',
                   '--reviewed_examples', str(reviews_path), '--output_dir', str(destination)]
        for label in MODEL_ORDER:
            command += ['--model', label + '=' + str(render_dir(specs[label]))]
        run_logged(command, destination / 'evaluation.log', args.repo)
    report = read_json(require_file(output))
    expected_models = {label: str(render_dir(specs[label]).resolve()) for label in MODEL_ORDER}
    if report['inputs']['models'] != expected_models or report['test_frames'] != tests:
        raise ValueError('Existing evaluation does not match the required model paths/test split')
    if report['comparison_reference'] != dict(baseline='PWSR20', proposed='Structure20'):
        raise ValueError('Primary comparison must use original PWSR20 as the proposed-method ablation')
    fixed_dir = destination / 'reviewed_six_cases'
    command = [args.analysis_python, str(require_file(args.repo / 'make_reviewed_structure_evidence.py')),
               '--reviews', str(reviews_path), '--test_frames', str(test_path), '--source_images', str(args.source_images),
               '--ground_truth_dir', str(gt), '--teacher_renders', str(render_dir(native)), '--output_dir', str(fixed_dir)]
    for label in MODEL_ORDER:
        command += ['--model', label + '=' + str(render_dir(specs[label]))]
    run_logged(command, fixed_dir / 'export.log', args.repo)
    return report, destination, read_json(require_file(fixed_dir / 'reviewed_native_manifest.json'))


def instance_export(repo, spec, reports):
    directory, iteration = spec
    ply = require_file(standard_ply(spec))
    metadata = require_file(directory / ('chkpnt%d.pth.instances.npz' % iteration))
    output = ply.with_name('point_cloud_instances.ply')
    report_path = reports / ('instance_export_SH%d.json' % ply_header(ply)['sh_degree'])
    if output.exists() or report_path.exists():
        report = read_json(require_file(report_path))
        for field, path in (('input_ply_sha256', ply), ('metadata_sha256', metadata), ('output_ply_sha256', require_file(output))):
            if report[field] != file_sha256(path):
                raise ValueError('Existing instance export changed: ' + str(path))
        return report
    spec_obj = importlib.util.spec_from_file_location('city_export_instance_ply', str(repo / 'export_instance_ply.py'))
    module = importlib.util.module_from_spec(spec_obj)
    spec_obj.loader.exec_module(module)
    return module.export_instance_ply(str(ply), str(metadata), str(output), str(report_path))


def audit_frozen_atoms(source_ply, compressed_ply, metadata_path, priors_path, require_exact=True,
                       require_all_anchors=True):
    """Use immutable source lineage; nearest-neighbor matching would be invalid."""
    from plyfile import PlyData
    source = PlyData.read(str(source_ply))['vertex'].data
    compressed = PlyData.read(str(compressed_ply))['vertex'].data
    with np.load(str(metadata_path), allow_pickle=False) as data:
        lineage = np.asarray(data['original_ids'], dtype=np.int64)
        metadata_source_hash = str(data['source_xyz_sha256'].item())
    with np.load(str(priors_path), allow_pickle=False) as data:
        anchors = np.asarray(data['anchors'], dtype=np.int64)
        source_hash = str(data['source_xyz_sha256'].item())
        manifest = json.loads(str(data['manifest_json'].item()))
    if metadata_source_hash != source_hash or len(lineage) != len(compressed) or len(np.unique(lineage)) != len(lineage):
        raise ValueError('Invalid source lineage for frozen atom audit')
    if manifest.get('source_ply_sha256') and file_sha256(source_ply) != manifest['source_ply_sha256']:
        raise ValueError('Source PLY appearance/geometry bytes changed after constructing structure priors')
    if np.any(anchors < 0) or np.any(anchors >= len(source)):
        raise ValueError('Anchor source index outside original model')
    source_xyz = np.column_stack([source[axis] for axis in ('x', 'y', 'z')]).astype(np.float32)
    actual_xyz_hash = hashlib.sha256(np.ascontiguousarray(source_xyz).tobytes()).hexdigest()
    if actual_xyz_hash != source_hash:
        raise ValueError('Raw source XYZ does not match structure certificate')
    order = np.argsort(lineage, kind='stable')
    positions = np.searchsorted(lineage[order], anchors)
    valid = positions < len(lineage)
    matches = np.zeros(len(anchors), dtype=bool)
    matches[valid] = lineage[order[positions[valid]]] == anchors[valid]
    if require_all_anchors and not matches.all():
        raise ValueError('Compressed model is missing %d required anchors' % int((~matches).sum()))
    retained_anchors = anchors[matches]
    selected_rows = order[positions[matches]]
    source_names = list(source.dtype.names)
    names = [name for name in source_names if name in compressed.dtype.names]
    if require_exact and names != source_names:
        raise ValueError('SH3 critical audit requires all original 62 PLY properties')
    details = {}
    for name in names:
        a, b = source[name][retained_anchors], compressed[name][selected_rows]
        unequal = a != b
        details[name] = dict(exactly_equal=bool(not unequal.any()), mismatching_atoms=int(unequal.sum()),
                             max_absolute_difference=float(np.max(np.abs(a.astype(np.float64) - b.astype(np.float64)))) if len(a) else 0.)
    exact = all(row['exactly_equal'] for row in details.values())
    if require_exact and not exact:
        raise ValueError('SH3 hard-freeze audit failed: ' + ', '.join(name for name, row in details.items() if not row['exactly_equal']))
    return dict(scope='Retained original anchor properties; neither source lineage nor field equality guarantees final rendered visibility',
                source_ply=str(source_ply), compressed_ply=str(compressed_ply), metadata=str(metadata_path),
                source_xyz_sha256=source_hash, anchors=len(anchors), missing_anchors=int((~matches).sum()),
                retained_old_anchors=len(retained_anchors), all_old_anchors_required=require_all_anchors,
                original_properties=len(source_names), compared_properties=len(names),
                exact_match_required=require_exact, all_compared_properties_exact=exact if len(retained_anchors) else None,
                missing_source_properties=[name for name in source_names if name not in names],
                certificate_groups=manifest['groups'], property_comparison=details,
                note=('SH3 critical anchors are fixed to original geometry, opacity, rotation and all SH coefficients.' if require_exact else
                      'This is descriptive comparison only. Optimization/SH reduction may change these source atoms; bit-exact preservation is not required or claimed.'))


def audit_adaptive_coverage(priors_path, metadata_path):
    from utils.structure_pruning import StructurePruningState
    state = StructurePruningState.load(str(priors_path))
    with np.load(str(metadata_path), allow_pickle=False) as data:
        lineage = np.asarray(data['original_ids'], dtype=np.int64)
        source_hash = str(data['source_xyz_sha256'].item())
    if source_hash != state.source_xyz_sha256 or len(np.unique(lineage)) != len(lineage):
        raise ValueError('Adaptive final lineage has a different source/duplicate source IDs')
    coverage = state.audit(lineage, require_anchors=False)
    if coverage['uncovered_groups'] != 0:
        raise ValueError('Adaptive selection lost %d critical screen cells' % coverage['uncovered_groups'])
    return dict(selection_mode='adaptive', original_anchor_identity_required=False,
                original_anchor_bit_exactness_required=False, retained_source_cell_coverage=coverage,
                all_declared_source_cells_covered=True,
                note='Source candidate membership covers all declared critical screen cells; optimized rendered visibility still needs independent image assessment.')


def audit_instance_floors(original_metadata, compressed_metadata):
    """Use the existing lineage audit, including policy stored with this model."""
    from utils.instance_metrics import load_metadata, retention_report
    original = load_metadata(str(original_metadata))
    compressed = load_metadata(str(compressed_metadata), require_original_ids=True)
    if 'policy_json' not in compressed:
        raise ValueError('Compressed model lacks its immutable instance-floor policy')
    policy = json.loads(str(compressed['policy_json'].item()))
    if not policy:
        raise ValueError('Empty instance-floor policy cannot certify adaptive retention')
    result = retention_report(original, compressed, policy=policy)
    if result['groups']['all_instances']['floor_violation_count']:
        raise ValueError('Compressed model violates its declared instance floors')
    result['all_instance_floors_passed'] = True
    return result


def fmt(value, digits=4):
    return '—' if value is None else ('%.*f' % (digits, value))


def markdown_link(label, path):
    return '[%s](<%s>)' % (label.replace('|', '\\|'), str(path))


def artifact_record(path, hashes=True):
    path = Path(path)
    record = dict(path=str(path.absolute()), resolved_path=str(path.resolve()), bytes=path.stat().st_size)
    if hashes:
        record['sha256'] = file_sha256(path)
    if path.is_symlink():
        record['symlink_target'] = os.readlink(str(path))
    return record


def write_provenance(args, inventory, reports_dir):
    """Record all real inputs/checkpoints, while keeping all generated files here."""
    manifest = read_json(require_file(args.experiment_root / 'code_changes.json'))
    changes = []
    for row in manifest:
        path = require_file(args.repo / row['path'])
        actual = file_sha256(path)
        changes.append(dict(row, actual_sha256=actual, matches_recorded_sha256=actual == row['new_sha256']))
    config_path, config = selected_configuration(args)
    code_extra = ['finalize_structure_experiment.py', 'evaluate_visible_structure.py', 'make_reviewed_structure_evidence.py',
                  'export_instance_ply.py', 'configs/city_structure20.json', 'configs/city_structure_policy.json',
                  'configs/city_structure_control20.json', 'STRUCTURE_PROTECTION.md', 'PROGRESS.md', 'WORKFLOW.md']
    extras = [artifact_record(args.repo / name) for name in code_extra if (args.repo / name).is_file()]
    extras += [artifact_record(path) for path in sorted((args.repo / 'configs').glob('city_structure*adaptive*.json'))]
    write_json(reports_dir / 'code_manifest.json', dict(changes=changes, additional_sources=extras,
               all_recorded_changes_match=all(row['matches_recorded_sha256'] for row in changes)))
    paths = set()
    def add(path):
        path = Path(path)
        if path.is_file():
            paths.add(path.absolute())
        elif path.is_dir():
            for item in path.rglob('*'):
                if item.is_file() and '__pycache__' not in item.parts:
                    paths.add(item.absolute())
    add(args.experiment_root / 'inputs')
    for name in ('experiment_plan.json', 'baseline_jobs.json', 'code_changes.json'):
        add(args.experiment_root / name)
    add(args.proposed_root / 'pipeline_all.json')
    add(config_path)
    add(report_directory(args) / 'manual_visual_review.json')
    if legacy_fixed(args):
        add(args.experiment_root / 'reports/manual_visual_review.json')
    if args.manual_review:
        add(args.manual_review)
    add(args.split_root / 'inputs/train_frames.txt')
    add(args.split_root / 'inputs/test_frames.txt')
    add(args.split_root / 'inputs/split.json')
    add(args.split_root / 'inputs/train_labels')
    add(args.sam_masks)
    add(args.source_images)
    add(args.colmap)
    for key in ('input_ply', 'input_checkpoint', 'teacher_checkpoint', 'instance_metadata', 'instance_policy_config', 'instance_projected_area'):
        if config.get(key):
            path = Path(config[key])
            add(path if path.is_absolute() else args.repo / path)
    for record in inventory.values():
        add(record['ply'])
        directory = Path(record['model_dir'])
        iteration = record['iteration']
        for name in ('cfg_args', 'run_args.json', 'pipeline_all.json', 'chkpnt%d.pth' % iteration,
                     'chkpnt%d.pth.state.pth' % iteration, 'chkpnt%d.pth.instances.npz' % iteration):
            add(directory / name)
    records = [artifact_record(path) for path in sorted(paths)]
    write_json(reports_dir / 'input_checkpoint_manifest.json', dict(scope='Source RGB/COLMAP, SAM evaluation caches, train-only structure/instance priors and evaluated final checkpoints',
               files=records, count=len(records), total_bytes=sum(row['bytes'] for row in records)))
    return dict(code_manifest=str(reports_dir / 'code_manifest.json'), input_checkpoint_manifest=str(reports_dir / 'input_checkpoint_manifest.json'))


def write_native_index(path, manifest, manual):
    lines = ['# 固定六例：原生像素结构对比', '',
             '六个框在新模型结果生成前固定；不按 PSNR 排序。PNG 裁剪不改像素，汇总图仅用最近邻放大。', '',
             '“近乎消失”允许存在模糊残影；不把边界缺失称为整栋建筑消失。PWSR20 是原全局 SPS 方法。', '',
             '助手目视检查状态：' + ('已完成' if manual['status'] == 'complete' else '待完成') + '。', '',
             '![固定六例总图](<%s>)' % (path.parent / 'reviewed_six_cases/reviewed_native_comparison.png'), '']
    for index, row in enumerate(manifest['cases']):
        lines += ['## %d. %s / %s' % (index + 1, row['frame'], row['category']), '',
                  '固定框：`%s`。' % row['bbox_xyxy'], '', row['description'], '',
                  '| 模型 | 原生裁剪 | 对应完整渲染 |', '| --- | --- | --- |']
        for label, files in row['files'].items():
            lines.append('| %s | ![native](<%s>) | %s |' % (label, files['native_crop'], markdown_link('完整 PNG', files['source'])))
        lines.append('')
    path.write_text('\n'.join(lines), encoding='utf-8')


def write_chinese_report(args, inventory, visible, report_dir, native_manifest, manual, audits, exports, provenance):
    source = inventory['Native']
    feasibility = read_json(args.experiment_root / 'inputs/structure/feasibility.json')
    config_path, config = selected_configuration(args)
    mode = args.selection_mode
    if mode == 'fixed':
        structure_items = [
            '- 用源高斯 alpha×T 可见贡献建立小区域、边界段和细杆屏幕单元；%s 个固定锚点覆盖 %s 个声明单元。' % (format(feasibility['anchors'], ','), format(feasibility['coverage']['groups'], ',')),
            '- 锚点与实例最低配额并集占 %s 个点，剩余 %s 个点由 SPS 分配；最终保持 89,405 个点。' % (format(feasibility['anchors_plus_floors_cost'], ','), format(feasibility['remaining_global_SPS'], ',')),
            '- SH3 阶段固定关键锚点的原始位置、尺度、旋转、不透明度及全部 SH 系数；其他高斯继续恢复。固定源原子并不保证被其他高斯遮挡后仍可见，最终仍需检查渲染。',
        ]
    else:
        structure_items = [
            '- adaptive 选择从每个关键屏幕单元已有的可见源候选中自适应保留代表，同时满足实例最低配额；要求全部 %s 个声明单元仍有候选，不要求原先 %s 个锚点全部保留。' % (format(feasibility['coverage']['groups'], ','), format(feasibility['anchors'], ',')),
            '- 固定版的 33,959 个“原锚点与实例配额并集”只是该固定方案的成本，不是 adaptive 的硬预算。adaptive 最终高斯总数同样严格为 89,405。',
            '- adaptive 不作原锚点 62 字段逐位不变承诺；源原子可替换、参数可恢复。来源单元覆盖不等于优化后仍可见，最终仍需检查原生渲染。',
        ]
        coverage = audits.get('adaptive_coverage', {}).get('SH3')
        if coverage:
            certificate = coverage['retained_source_cell_coverage']
            structure_items.append('- SH3 实际证书审计：%s 个单元未覆盖；原参考锚点保留 %s/%s，实例最低配额全部通过。' %
                                   (certificate['uncovered_groups'], format(certificate['anchors'] - certificate['missing_anchors'], ','), format(certificate['anchors'], ',')))
    loss_item = ('- 实际恢复项权重：区域 %s、边缘 %s、边界对比裕量 %s、细杆 %s、锚点 %s；SPS 结构项权重 %s。无 3D-DoG，无人工语义标签。' %
                 tuple(config.get(key, 0) for key in ('instance_roi_weight', 'instance_edge_weight', 'structure_boundary_weight',
                      'structure_thin_weight', 'structure_anchor_weight', 'structure_score_weight')))
    lines = ['# City 结构保护压缩结果（以 SH 蒸馏前 SH3 为主）', '',
             '本轮目标是保留原模型已有的可见小地物、边界段和细长结构；主比较全部使用 SH3 剪枝 PLY、同一渲染后端及原生 1592×894 测试视角。', '',
             '**新模型的助手目视检查状态：%s。** %s' % ('已完成' if manual['status'] == 'complete' else '待完成', manual.get('overall_conclusion', '')), '',
             '已有固定六例中，LG20 有车辆形状、微小白色物体、屋面轮廓和窗框细杆损失，但原 PWSR20 已经保留其中的结构。新版本相对 LG20 的改善不能直接归因于本次新增模块，也不能替代与 PWSR20 的同预算对照。', '',
             '## 输出与预算', '',
             '全部模型、checkpoint、渲染和报告保存在 `%s` 及其引用的既有实验目录。未覆盖旧结果。' % args.experiment_root, '',
             '本报告方案：`%s`；当前模型目录：`%s`；报告目录：`%s`。' % (mode, args.proposed_root, report_dir), '',
             '| 模型 | 高斯数 | SH | 标准 PLY MiB | PLY 体积压缩倍数 | 标准 PLY |', '| --- | ---: | ---: | ---: | ---: | --- |']
    for label in ('Native',) + MODEL_ORDER:
        row = inventory[label]
        lines.append('| %s | %s | %d | %.3f | %.4f× | %s |' %
                     ('原始 3DGS' if label == 'Native' else MODEL_LABELS[label], format(row['gaussians'], ','), row['sh_degree'], row['bytes'] / 1048576., row['size_compression_ratio'], markdown_link('PLY', row['ply'])))
    lines += ['', 'LG10 沿用原始 LG 阈值法，实际 178,810 个点；PWSR10/v2 为 178,809 个点。三种 20× 模型均严格使用 89,405 个点。压缩倍数按标准渲染 PLY 计算，实例标签附加字段另占空间。', '',
              'LG 为单次重要性剪枝与 5,000 次恢复循环；PWSR 对照与新版均为 10,199 次恢复/逐步剪枝循环。外部方法训练策略不同，同预算并不等于所有优化设置一致。', '',
              '## 本次代码改动', '',
              '- 仅从 61 个训练视角的 RGB、SAM 跨视图伪实例和原始教师渲染建立可见性证据；测试 SAM 和六个测试框只用于评估。'] + structure_items + [loss_item, '',
              '## 原生结构指标（辅助证据）', '',
              '边缘/中心线召回越高越好，缺失连通分量越小越好。以下为自动候选均值，不能直接当作消失的物体个数；缺失量为参考像素的 8 连通分量大小，不是物理长度。', '',
              '| 模型 | 小区域对比证据召回 ↑ | 边界召回 ↑ | 最大缺失边界分量均值 ↓ | 细区域中心线召回 ↑ | 线段中心线召回 ↑ |',
              '| --- | ---: | ---: | ---: | ---: | ---: |']
    for label in MODEL_ORDER:
        values = visible['summary'][label]
        lines.append('| %s | %s | %s | %s | %s | %s |' % (label,
                     fmt(values['small_region']['mean_contrast_recall']), fmt(values['boundary']['mean_edge_recall']),
                     fmt(values['boundary']['mean_largest_missing_edge_fragment_px'], 2), fmt(values['thin_region']['mean_thin_centerline_recall']),
                     fmt(values['line_segment']['mean_thin_centerline_recall'])))
    lines += ['', '候选数量：' + '；'.join('%s=%d' % (name, visible['summary']['Structure20'][name]['candidates']) for name in ('small_region', 'boundary', 'thin_region', 'line_segment')) + '。', '',
              markdown_link('全部候选 JSON（含退化案例）', report_dir / 'visible_structure.json') + ' · ' + markdown_link('全部候选 CSV', report_dir / 'all_candidates.csv'), '',
              '## 训练前固定的六个可视例子', '',
              markdown_link('完整原生裁剪索引（含各模型完整渲染）', report_dir / 'INDEX_原生结构对比.md'), '',
              '![固定六例](<%s>)' % (report_dir / 'reviewed_six_cases/reviewed_native_comparison.png'), '',
              '| 固定例 / 模型 | 边缘召回 ↑ | 对比证据召回 ↑ | 最大缺失边缘分量 ↓ |', '| --- | ---: | ---: | ---: |']
    reviewed_rows = {row['id']: row for row in visible['candidates'] if 'reviewed_example' in row['categories']}
    for index, record in enumerate(native_manifest['cases']):
        row = reviewed_rows['%s_reviewed%d' % (Path(record['frame']).stem, index)]
        for label in ('LG20', 'PWSR20', 'Structure20'):
            measure = row['models'][label]
            lines.append('| %d %s / %s | %s | %s | %d |' % (index + 1, row['frame'], label, fmt(measure['edge_recall']), fmt(measure['contrast_recall']), measure['largest_missing_edge_fragment_px']))
    lines += ['', '第六例红车是“仍然存在”的负对照，不能称为被压缩消失的目标。固定框附近还有背景像素，框内指标不是独立目标识别准确率。', '', '### 助手目视检查记录', '']
    if manual['status'] == 'complete':
        lines += ['复核者：%s。此记录是助手查看图像后的描述，不是用户人工标注的物体真值。' % manual['reviewer'], '']
        for index, row in enumerate(manual['cases']):
            lines += ['%d. `%s %s`：%s 与 PWSR20 对比：%s 结论：%s' % (index + 1, row['frame'], row['bbox_xyxy'], row['new_model_observation'], row['comparison_to_pwsr20'], row['conclusion']), '']
    else:
        lines += ['待助手逐例查看本次新模型、LG20 与 PWSR20 的同位置原生裁剪；完成之前不宣布“新方法挽救了原 PWSR 消失的目标”。', '']
    lines += ['## SH 蒸馏前后保留', '']
    if args.stage == 'final':
        for label, key in (('SH3（主分析）', 'SH3'), ('SH2（附加交付，未参与主结构结论）', 'SH2')):
            row = exports[key]
            lines += ['- %s：%s；%s；%s。' % (label,
                      markdown_link('标准 PLY', row['input_ply']), markdown_link('带实例标签 PLY', row['output_ply']),
                      markdown_link('checkpoint', Path(row['metadata']).with_name(Path(row['metadata']).name.replace('.instances.npz', ''))))]
        sh2 = inventory['Structure20_SH2']
        lines += ['', 'SH2 仍为 %s 个点，标准 PLY %.3f MiB。SH 降阶会改变外观表示，未宣称保留 SH3 关键原子的全部外观属性。' % (format(sh2['gaussians'], ','), sh2['bytes'] / 1048576.), '',
                  ('SH3 固定锚点审计：%s 个锚点全部存在，原始 %d 个字段逐点精确相等。此结论只覆盖被选中的源原子，不等于渲染中的所有真实对象均可见。' % (format(audits['SH3']['anchors'], ','), audits['SH3']['compared_properties']) if mode == 'fixed' else
                   'adaptive 最终审计要求：SH3/SH2 源候选覆盖均无未覆盖单元，实例最低配额均通过；旧锚点保留数量与字段差异仅作统计，不作逐位不变承诺。'), '',
                  markdown_link('SH3/SH2 来源覆盖与锚点统计', export_directory(args) / 'frozen_atom_audit.json'), '']
    else:
        lines += ['SH3 主分析已输出；SH2 训练/导出尚未完成本脚本的 final 核验。完成后运行 `--stage final --skip-evaluation` 保留主分析并追加交付信息。', '']
    lines += ['## 复现与边界', '',
              '- 原始输入：`%s`，原始 %s 个 SH3 高斯。' % (RAW_OUTPUT, format(source['gaussians'], ',')),
              '- 分割标签是自动区域 ID，不是语义类别，也不保证一个 ID 就是真实完整物体。',
              '- 对训练前原模型已经缺失、SAM 未发现或缺少训练视角证据的地物不作保留保证。',
              '- 边缘容差固定为 2 像素；模糊、轻微错位和纹理可影响指标。所有案例保留原生图像，不能靠锐化噪声宣称恢复结构。', '',
              markdown_link('运行状态与模型清单', artifact_directory(args) / 'finalization_progress.json') + ' · ' + markdown_link('本轮实际配置', config_path), '']
    if provenance:
        lines += [markdown_link('源码 SHA 清单', provenance['code_manifest']) + ' · ' + markdown_link('所有输入与 checkpoint SHA 清单', provenance['input_checkpoint_manifest']), '']
    (artifact_directory(args) / 'README_结构保护结果.md').write_text('\n'.join(lines), encoding='utf-8')


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=('primary', 'final'), required=True)
    parser.add_argument('--repo', type=Path, default=DEFAULT_REPO)
    parser.add_argument('--experiment-root', type=Path, default=DEFAULT_ROOT)
    parser.add_argument('--proposed-root', type=Path, help='Current model root; default EXPERIMENT_ROOT/protected20')
    parser.add_argument('--report-name', default='primary_sh3', help='Separate report subdirectory; adaptive must choose a new name')
    parser.add_argument('--selection-mode', choices=('fixed', 'adaptive'), default='fixed')
    parser.add_argument('--config', type=Path, help='Actual run config; otherwise read PROPOSED_ROOT/pipeline_all.json')
    parser.add_argument('--split-root', type=Path, default=DEFAULT_REPO / 'output/city2_instance_guard_20261026')
    parser.add_argument('--sam-masks', type=Path, default=DEFAULT_WORKSPACE / 'work/label_free/sam_masks')
    parser.add_argument('--source-images', type=Path, default=RAW_DATA / 'images')
    parser.add_argument('--colmap', type=Path, default=RAW_DATA / 'sparse/0')
    parser.add_argument('--reviewed-examples', type=Path)
    parser.add_argument('--manual-review', type=Path)
    parser.add_argument('--analysis-python', default=sys.executable)
    parser.add_argument('--skip-evaluation', action='store_true', help='Reuse validated primary JSON; fixed six crops are still refreshed')
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    args.repo = args.repo.resolve()
    args.experiment_root = args.experiment_root.resolve()
    if (args.repo / 'output').resolve() not in args.experiment_root.parents:
        raise ValueError('All generated deliverables must be inside the actual PWSR repository output folder')
    args.proposed_root = (args.proposed_root or args.experiment_root / 'protected20').resolve()
    if args.experiment_root not in args.proposed_root.parents:
        raise ValueError('Current proposed model must be inside this experiment root')
    if Path(args.report_name).name != args.report_name or args.report_name in ('', '.', '..'):
        raise ValueError('Report name must be one safe directory component')
    if args.selection_mode == 'adaptive' and (args.report_name == 'primary_sh3' or args.proposed_root == args.experiment_root / 'protected20'):
        raise ValueError('Adaptive requires a separate --report-name and --proposed-root to preserve the fixed results')
    args.experiment_root.mkdir(parents=True, exist_ok=True)
    artifact_directory(args).mkdir(parents=True, exist_ok=True)
    if str(args.repo) not in sys.path:
        sys.path.insert(0, str(args.repo))
    train_path = require_file(args.split_root / 'inputs/train_frames.txt')
    test_path = require_file(args.split_root / 'inputs/test_frames.txt')
    if set(frame_names(train_path)) & set(frame_names(test_path)):
        raise ValueError('Training/test view overlap')
    review_path = require_file(args.reviewed_examples or args.experiment_root / 'inputs/reviewed_examples_20x.json')
    reviews = validate_fixed_reviews(review_path, frame_names(test_path))
    manual_path = args.manual_review
    if manual_path is None and (report_directory(args) / 'manual_visual_review.json').is_file():
        manual_path = report_directory(args) / 'manual_visual_review.json'
    if manual_path is None and legacy_fixed(args) and (args.experiment_root / 'reports/manual_visual_review.json').is_file():
        manual_path = args.experiment_root / 'reports/manual_visual_review.json'
    manual = validated_manual_review(manual_path, reviews)
    specs = model_specs(args.repo, args.experiment_root, args.proposed_root)
    inventory = build_inventory(specs, final=args.stage == 'final')
    for label in ('Structure20', 'Structure20_SH2'):
        if label in inventory:
            inventory[label]['selection_mode'] = args.selection_mode
            inventory[label]['coverage_commitment'] = ('Original fixed source anchors retained with all 62 SH3 properties exact' if args.selection_mode == 'fixed' and label == 'Structure20' else
                                                       'All declared source screen cells covered and instance floors retained; old anchor identities/bit-exact properties not required' if args.selection_mode == 'adaptive' else
                                                       'Source anchors retained; SH2 appearance is optimized and is not bit-exact SH3')
    visible, report_dir, native_manifest = evaluate_primary(args, specs, review_path, train_path, test_path)
    write_json(report_dir / 'assistant_visual_review.json', manual)
    write_native_index(report_dir / 'INDEX_原生结构对比.md', native_manifest, manual)
    exports, audits, provenance = {}, {}, {}
    priors = require_file(args.experiment_root / 'inputs/structure/structure_priors.npz')
    if args.selection_mode == 'adaptive':
        _, config = selected_configuration(args)
        original_metadata = Path(config['instance_metadata'])
        if not original_metadata.is_absolute():
            original_metadata = args.repo / original_metadata
        metadata = require_file(specs['Structure20'][0] / 'chkpnt25199.pth.instances.npz')
        audits['adaptive_coverage'] = dict(SH3=audit_adaptive_coverage(priors, metadata))
        audits['instance_floors'] = dict(SH3=audit_instance_floors(original_metadata, metadata))
        write_json(report_dir / 'selection_audit.json', audits)
    if args.stage == 'final':
        export_dir = export_directory(args)
        export_dir.mkdir(parents=True, exist_ok=True)
        source = Path(inventory['Native']['ply'])
        for key, spec in (('SH3', specs['Structure20']), ('SH2', (args.proposed_root / 'sh2', 40000))):
            exports[key] = instance_export(args.repo, spec, export_dir)
            audits[key] = audit_frozen_atoms(source, standard_ply(spec), exports[key]['metadata'], priors,
                                             require_exact=args.selection_mode == 'fixed' and key == 'SH3',
                                             require_all_anchors=args.selection_mode == 'fixed')
            if args.selection_mode == 'adaptive':
                audits['adaptive_coverage'][key] = audit_adaptive_coverage(priors, exports[key]['metadata'])
                audits['instance_floors'][key] = audit_instance_floors(original_metadata, exports[key]['metadata'])
        write_json(export_dir / 'frozen_atom_audit.json', audits)
        repro_dir = args.experiment_root / 'reports/reproducibility' if legacy_fixed(args) else report_dir / 'reproducibility'
        provenance = write_provenance(args, inventory, repro_dir)
    write_json(artifact_directory(args) / 'model_inventory.json', inventory)
    progress = dict(stage=args.stage, completed_at_utc=datetime.datetime.utcnow().isoformat() + 'Z',
                    primary_comparison='pre-distillation SH3, 89405 equal 20x points; original PWSR20 ablation is explicit',
                    status='final_outputs_verified' if args.stage == 'final' else 'primary_sh3_ready',
                    assistant_visual_review_status=manual['status'], assistant_visual_review_reviewer=manual.get('reviewer'),
                    native_fixed_cases=6, source_reviews_sha256=file_sha256(review_path),
                    output_root=str(args.experiment_root), proposed_root=str(args.proposed_root),
                    selection_mode=args.selection_mode, report_name=args.report_name,
                    inventory=inventory, exports=exports, provenance=provenance,
                    warning='Automatic recall/coverage is not proof of physical-object disappearance or rescue; inspect fixed native render crops.')
    write_json(artifact_directory(args) / 'finalization_progress.json', progress)
    write_chinese_report(args, inventory, visible, report_dir, native_manifest, manual, audits, exports, provenance)
    print(json.dumps(dict(status=progress['status'], output_root=str(args.experiment_root),
                         primary_report=str(artifact_directory(args) / 'README_结构保护结果.md'),
                         fixed_sheet=str(report_dir / 'reviewed_six_cases/reviewed_native_comparison.png'),
                         assistant_review_status=manual['status']), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
