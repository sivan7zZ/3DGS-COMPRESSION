"""Archive completed continuous-boundary experiments without Torch or GPU.

This report-only entry point refuses incomplete queues, mismatched snapshots,
changed priors/models, missing independent semantic audits or missing crop
reviews. It never trains, renders, recalculates metrics or edits frozen inputs.
Only README_实验总览.md and final_artifact_manifest.json are written.
"""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import tempfile


REPO = Path(__file__).resolve().parent
DEFAULT_OUT = REPO / 'output/city2_continuous_boundary_20261109'
PARAMETERS = dict(sam_weight=.4, depth_weight=.4, rgb_edge_weight=.2, gain=5., fusion='log1p')
COMPONENTS = ('sam_boundary_strength', 'depth_boundary_strength', 'rgb_boundary_strength')
SCOPES = ('independent_test_2', 'auxiliary_training_views_8', 'all_annotated_10')
NUMERIC_COLUMNS = ('ratio', 'gaussians', 'visual_test_views', 'semantic_views',
                   'PSNR', 'SSIM', 'LPIPS_standard', 'Recall', 'region_tp', 'region_gt',
                   'B_mIoU', 'mIoU', 'mAcc')
EXPECTED_LPIPS_WEIGHT_SHA256 = 'b9ba014c09665fa4b7e91256673cf20178d7d8f0961fb308fe154ee074c81bc0'


def read(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError('Archive is not ready; required file missing: ' + str(path))
    return json.loads(path.read_text())


def complete(path):
    value = read(path)
    if value.get('status') != 'complete':
        raise ValueError('Archive requires status=complete: ' + str(path))
    return value


def atomic_text(path, content):
    path = Path(path)
    descriptor, temporary = tempfile.mkstemp(prefix='.archive-', dir=str(path.parent))
    try:
        with os.fdopen(descriptor, 'w') as stream:
            stream.write(content)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


class Artifacts:
    def __init__(self, repo):
        self.repo = Path(repo).resolve()
        self.rows = {}

    def resolve(self, path):
        path = Path(path)
        return (path if path.is_absolute() else self.repo / path).resolve()

    def add(self, path, kind, expected=None):
        path = self.resolve(path)
        if not path.is_file():
            raise FileNotFoundError('Archive is not ready; missing artifact: ' + str(path))
        key = str(path)
        if key not in self.rows:
            digest = hashlib.sha256()
            with path.open('rb') as stream:
                for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
                    digest.update(block)
            self.rows[key] = dict(path=key, sha256=digest.hexdigest(),
                                  size_bytes=path.stat().st_size, kinds=[])
        row = self.rows[key]
        if kind not in row['kinds']:
            row['kinds'].append(kind)
        if expected is not None and row['sha256'] != expected:
            raise ValueError('Artifact SHA256 differs from frozen receipt: ' + key)
        return row


def verify_snapshot(output, name, artifacts):
    folder = output / name
    manifest_path = folder / 'manifest.json'
    manifest = read(manifest_path)
    if manifest.get('status') != 'frozen' or not manifest.get('files'):
        raise ValueError('Missing frozen source inventory: ' + str(manifest_path))
    for relative, expected in manifest['files'].items():
        relative = Path(relative)
        if relative.is_absolute() or '..' in relative.parts:
            raise ValueError('Unsafe source snapshot path')
        artifacts.add(artifacts.repo / relative, name + '_live_source', expected)
        artifacts.add(folder / relative, name + '_frozen_copy', expected)
    record = artifacts.add(manifest_path, 'source_snapshot_manifest')
    return manifest, dict(files=len(manifest['files']), manifest_sha256=record['sha256'],
                          live_and_frozen_copies_exact=True)


def validate_metric_tables(output, registry, artifacts):
    metrics_path, csv_path = output / 'comparison_all_metrics.json', output / 'comparison_all_metrics.csv'
    metrics = complete(metrics_path)
    rows = metrics.get('rows', [])
    with csv_path.open(newline='') as stream:
        csv_rows = list(csv.DictReader(stream))
    if len(rows) != len(csv_rows) or not rows:
        raise ValueError('CSV/JSON metric row inventories differ')
    for expected, observed in zip(rows, csv_rows):
        if set(expected) != set(observed):
            raise ValueError('CSV/JSON metric columns differ')
        for key, value in expected.items():
            if value in ('', None):
                if observed[key] != '':
                    raise ValueError('CSV blank differs from JSON: ' + key)
            elif key in NUMERIC_COLUMNS:
                number = float(observed[key])
                if not math.isfinite(number) or not math.isclose(number, float(value), rel_tol=0, abs_tol=1e-12):
                    raise ValueError('CSV/JSON numerical metric differs: ' + key)
            elif observed[key] != str(value):
                raise ValueError('CSV/JSON string metric differs: ' + key)
    model_ids = [row['id'] for row in registry['models']]
    if len(model_ids) != 19 or len(set(model_ids)) != 19 or 'GT' in model_ids:
        raise ValueError('Expected 19 unique original/compressed models plus GT RGB')
    expected_groups = set(model_ids) | {'GT'}
    indexed = {}
    for row in rows:
        key = (row['semantic_scope'], row['model_id'])
        if key in indexed:
            raise ValueError('Duplicate model/scope metric row')
        indexed[key] = row
    for scope, count in zip(SCOPES, (2, 8, 10)):
        selected = [row for row in rows if row['semantic_scope'] == scope]
        if {row['model_id'] for row in selected} != expected_groups or len(selected) != 20:
            raise ValueError('Missing or additional metric group: ' + scope)
        for row in selected:
            if int(row['semantic_views']) != count:
                raise ValueError('Semantic support differs from declared scope')
            if row['model_id'] != 'GT' and int(row['visual_test_views']) != 9:
                raise ValueError('Visual metrics must use nine independent held-out views')
            for key in ('Recall', 'B_mIoU', 'mIoU', 'mAcc'):
                if not math.isfinite(float(row[key])) or not 0 <= float(row[key]) <= 1:
                    raise ValueError('Invalid fractional semantic metric: ' + key)
    if len(rows) != 60:
        raise ValueError('Expected exactly 20 groups x three declared scopes')
    artifacts.add(metrics_path, 'final_comparison_metrics')
    artifacts.add(csv_path, 'final_comparison_metrics')
    return metrics, indexed


def verify_crop_review(output, ratio, artifacts):
    folder = output / 'reports' / ('direct_continuous_%dx' % ratio)
    review_path, manifest_path = folder / 'visual_review.json', folder / 'manifest.json'
    review = complete(review_path)
    observations = review.get('observations')
    if not isinstance(observations, list) or not observations or any(not isinstance(item, str) or not item.strip() for item in observations):
        raise ValueError('Crop review requires nonempty observations:list[str]: ' + str(review_path))
    manifest = read(manifest_path)
    if manifest.get('ratio') != ratio or len(manifest.get('cases', [])) != 6:
        raise ValueError('Expected the six fixed native crop cases: ' + str(manifest_path))
    figure = artifacts.add(manifest['figure'], 'native_crop_figure', manifest['figure_sha256'])
    if review.get('figure_sha256') != figure['sha256'] or str(Path(review.get('figure', '')).resolve()) != figure['path']:
        raise ValueError('Visual review does not identify the exact reviewed crop figure')
    for case in manifest['cases']:
        for row in case['files'].values():
            if row.get('native_exact') is not True:
                raise ValueError('A crop lacks native-pixel verification')
            artifacts.add(row['crop'], 'native_crop', row['crop_sha256'])
    artifacts.add(review_path, 'root_visual_review')
    artifacts.add(manifest_path, 'native_crop_manifest')
    return dict(observations=observations, figure=figure['path'], review=str(review_path))


def verify_standard_lpips(registry, metrics, artifacts):
    records, weights = {}, set()
    for model in registry['models']:
        if model['id'].startswith(('continuous_', 'weighted_', 'v320_')):
            path = artifacts.resolve(model['model_dir']) / 'metrics_standard.json'
        else:
            path = artifacts.resolve(model['comparison_dir']) / 'comparison_lpips.json'
        value = complete(path)
        if value.get('input_range') != [-1, 1] or len(value.get('images', [])) != 9:
            raise ValueError('Standard LPIPS must use [-1,1] inputs and nine held-out images: ' + model['id'])
        digest = value.get('criterion', {}).get('state_dict_sha256')
        if not isinstance(digest, str) or not re.fullmatch(r'[0-9a-f]{64}', digest):
            raise ValueError('Standard LPIPS has no frozen weights identity: ' + model['id'])
        weights.add(digest)
        observed = metrics['visual'][model['id']]['LPIPS']
        if not math.isclose(float(observed), float(value['LPIPS']), rel_tol=0, abs_tol=1e-12):
            raise ValueError('Comparison LPIPS differs from its standard source: ' + model['id'])
        source = artifacts.add(path, 'standard_lpips_source')
        records[model['id']] = dict(path=source['path'], sha256=source['sha256'],
            input_range=[-1, 1], views=9, state_dict_sha256=digest)
    if weights != {EXPECTED_LPIPS_WEIGHT_SHA256}:
        raise ValueError('Compared standard LPIPS evaluators do not share the verified historical frozen weights')
    return dict(shared_weights_sha256=EXPECTED_LPIPS_WEIGHT_SHA256, models=len(records), sources=records)


def verify_semantic_artifacts(output, artifacts):
    folder = output / 'comparison/semantic'
    protocol = read(folder / 'protocol.json')
    metrics = complete(folder / 'metrics.json')
    verified = complete(folder / 'frozen_evaluators_verified.json')
    if len(protocol.get('folds', [])) != 5 or not verified.get('no_training') or not verified.get('no_model_selection'):
        raise ValueError('Expected five unchanged frozen evaluators without fitting or selection')
    protocol_sha = artifacts.add(folder / 'protocol.json', 'semantic_protocol')['sha256']
    if verified['protocol_sha256'] != protocol_sha or metrics['protocol_sha256'] != protocol_sha:
        raise ValueError('Semantic frozen protocol identity changed')
    for fold in protocol['folds']:
        train, predict = set(fold['train_frames']), set(fold['predict_frames'])
        if train & predict or train & {'frame_000', 'frame_056'}:
            raise ValueError('Evaluation label leaked into frozen evaluator fitting')
        artifacts.add(fold['checkpoint'], 'frozen_semantic_checkpoint', fold['checkpoint_sha256'])
        artifacts.add(fold['original_training_receipt'], 'frozen_semantic_training_receipt', fold['original_training_receipt_sha256'])
        artifacts.add(fold['original_predictions_manifest'], 'previous_prediction_manifest', fold['original_predictions_manifest_sha256'])
        fold_path = folder / 'folds' / fold['id']
        predicted_path = fold_path / 'predictions_manifest.json'
        predicted = read(predicted_path)
        if predicted['checkpoint_sha256'] != fold['checkpoint_sha256']:
            raise ValueError('Prediction used a different frozen evaluator')
        artifacts.add(predicted_path, 'current_prediction_manifest', metrics['folds'][fold['id']]['predictions_manifest_sha256'])
        artifacts.add(fold_path / 'inputs.json', 'semantic_prediction_inputs', predicted['inputs_json_sha256'])
    for name in ('dataset_manifest.json', 'metrics.json', 'metrics.csv',
                 'frozen_evaluators_verified.json', 'README_统一语义指标.md'):
        artifacts.add(folder / name, 'semantic_final_artifact')
    auxiliary = complete(output / 'comparison/auxiliary_renders/manifest.json')
    artifacts.add(output / 'comparison/auxiliary_renders/manifest.json', 'auxiliary_render_manifest')
    if len(auxiliary.get('models', {})) != 19:
        raise ValueError('Auxiliary renderer model count differs')
    return protocol


def unit_test_record(output, artifacts):
    path = output / 'unit_tests.log'
    if not path.exists():
        return dict(available=False, statement='本次总测试日志尚未归档；不推断测试数量。')
    text = path.read_text(errors='replace')
    runs = list(re.finditer(r'Ran\s+(\d+)\s+tests?\s+in\s+([\d.]+)s', text))
    if not runs or not re.search(r'^OK(?:\s*\([^\n]*\))?\s*$', text[runs[-1].end():], re.MULTILINE):
        raise ValueError('Provided unit_tests.log does not contain a successful final test run')
    tail = text[runs[-1].end():]
    skipped = re.search(r'skipped=(\d+)', tail)
    total, skipped_count = int(runs[-1].group(1)), int(skipped.group(1)) if skipped else 0
    artifacts.add(path, 'unit_test_log')
    return dict(available=True, run=total, skipped=skipped_count,
                seconds=float(runs[-1].group(2)),
                statement='日志记录运行%d项，结果OK，跳过%d项。' % (total, skipped_count))


def percent(value):
    return '%.3f' % (100 * float(value))


def table(rows, mixed=False):
    if mixed:
        lines = ['| 模型 | Recall↑ (%) | B-mIoU↑ (%) | mIoU↑ (%) | mAcc↑ (%) |',
                 '|---|---:|---:|---:|---:|']
    else:
        lines = ['| 模型 | 倍率 | 高斯数 | PSNR↑ | SSIM↑ | LPIPS↓ | Recall↑ (%) | B-mIoU↑ (%) | mIoU↑ (%) | mAcc↑ (%) |',
                 '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for row in rows:
        name = '%s (%s)' % (row['method'], row['model_id'])
        semantic = [percent(row[key]) for key in ('Recall', 'B_mIoU', 'mIoU', 'mAcc')]
        if mixed:
            values = [name] + semantic
        else:
            visual = ['—' if row[key] in ('', None) else '%.5f' % float(row[key])
                      for key in ('PSNR', 'SSIM', 'LPIPS_standard')]
            values = [name, str(row['ratio']) or '—', str(row['gaussians']) or '—'] + visual + semantic
        lines.append('| ' + ' | '.join(values) + ' |')
    return lines


def build_readme(output, metrics, indexed, validations, prior, reviews, tests):
    n = int(prior['source_count'])
    fractions = [100 * prior['component_statistics'][name]['fractional_rows'] / n for name in COMPONENTS]
    lines = ['# City：SAM、深度与RGB连续边界PUP压缩实验总览', '',
        '10×和20×均完成原生SH3压缩、统一视觉及语义比较与核验。19个原始/压缩模型另加GT RGB，共20组。结果与检查回执以本目录冻结记录为准。', '',
        '## 方法与预声明参数', '',
        '采用PUP六维xyz/log-scale Fisher logdet评分和原PWSR RIR/DPRA。SAM标签边缘到像素的精确欧氏距离d转为exp(−d²/(2×2²))；深度直接计算3×3 Sobel幅值，RGB先转灰度再计算Sobel，两者均以q95归一化并截断到[0,1]。深度有效域为finite(depth)且alpha>0.5。', '',
        '三分量通过61个训练视角中源高斯各向异性footprint的alpha×T可见贡献累积、归一化，保留遮挡关系。B=0.4×SAM+0.4×depth+0.2×RGB；新评分为PUP logdet+log(1+5B)，det重要性乘数为1+5B，理论范围[1,6]。奇异Fisher的−∞保留。权重和gain来自训练前冻结配置，未使用人工评测标签调参，未选择语义评估器。', '',
        '实例/拓扑保护、恢复损失沿用11月6日版本；边界不设置独立剪枝生存配额或合并否决。每轮合并上限为输入高斯的1%，回填预算为该轮目标数的2%；实际合并仍受安全相似候选限制。每组运行10199次更新，不运行SH蒸馏。', '',
        '源模型共%s个高斯。SAM/深度/RGB强度严格处于(0,1)的行比例分别为%.2f%%、%.2f%%、%.2f%%，因此它们是连续先验。该分布证明不等同于最终压缩改善。' % (format(n, ','), *fractions), '',
        '[三训练视角连续分量与融合图](continuous_priors/reports/continuous_boundary_training_views.png)；[全部源高斯直方图](continuous_priors/reports/continuous_gaussian_strength_histograms.png)；[精确统计](continuous_priors/reports/continuous_gaussian_strength_statistics.json)；[先验与输入指纹](continuous_priors/metadata_manifest.json)。', '',
        '## 独立测试主比较', '',
        'PSNR/SSIM/标准VGG0.1 LPIPS使用9张固定独立测试图、原生PNG；LPIPS输入范围为[-1,1]。语义主结果仅使用人工标签与独立测试集交集frame_000、frame_056，所有模型共享同一冻结分割器。GT RGB也经过该分割器，语义指标不设为100%。', '']
    lines += table([row for row in metrics['rows'] if row['semantic_scope'] == SCOPES[0]])
    lines += ['', '## 全部10张标签的混合汇总', '',
        '全部10张匹配City标签均参与评测：主2帧是3DGS独立测试视角，辅助8帧属于3DGS训练视角。辅助8帧沿用4折冻结评估器，每折仅用另外6张非测试原RGB标签拟合；frame_000、frame_056标签不用于任何拟合。本次复用同一组5个冻结权重，不重新拟合或选步。下表是2+8的计数聚合，不能称为10张独立测试。辅助8帧的独立范围结果保留在完整CSV/JSON中。', '']
    lines += table([row for row in metrics['rows'] if row['semantic_scope'] == SCOPES[2]], mixed=True)
    lines += ['', '## 与11月6日二值边界评分比较', '',
        '以下均为本次连续版减去同倍率二值版。LPIPS差为负时表示改善；语义差使用百分点。', '',
        '| 倍率 | ΔPSNR | ΔSSIM | ΔLPIPS | ΔRecall | ΔB-mIoU | ΔmIoU | ΔmAcc |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for ratio in (10, 20):
        new = indexed[(SCOPES[0], 'continuous_%dx' % ratio)]
        old = indexed[(SCOPES[0], 'weighted_%dx' % ratio)]
        values = ['%+0.5f' % (float(new[key]) - float(old[key])) for key in ('PSNR', 'SSIM', 'LPIPS_standard')]
        values += ['%+.3f' % (100 * (float(new[key]) - float(old[key]))) for key in ('Recall', 'B_mIoU', 'mIoU', 'mAcc')]
        lines.append('| %d× | %s |' % (ratio, ' | '.join(values)))
    if all(all(float(indexed[(SCOPES[0], 'continuous_%dx' % ratio)][key]) <
                   float(indexed[(SCOPES[0], 'weighted_%dx' % ratio)][key])
                   for key in ('B_mIoU', 'mIoU', 'mAcc')) for ratio in (10, 20)):
        lines += ['', '本次两倍率的独立主测试B-mIoU、mIoU和mAcc均低于二值边界版。连续评分已正确实现，但这组固定参数在现有评测中没有改善边界保护，不能把PSNR提高表述为整体地物保护改善。']
    lines += ['', '相对11月6日仅改变边界评分和连续先验，实例、拓扑、恢复损失及合并/回填参数保持。未进行消融，不能声称权重最优或把复杂局部表现归因于单一信号。', '',
              '## 固定原生裁剪的人工检查', '',
              '以下观察逐字取自本次固定六处裁剪的检查记录，不从均值指标或先验图预判目标恢复。', '']
    for ratio in (10, 20):
        lines += ['### %d×' % ratio, '']
        lines += ['- ' + observation for observation in reviews[ratio]['observations']]
        lines += ['', '[固定原生裁剪图](reports/direct_continuous_%dx/native_comparison.png)；[人工检查记录](reports/direct_continuous_%dx/visual_review.json)。' % (ratio, ratio), '']
    lines += ['## 模型、核验与范围', '',
        '| 倍率 | 高斯数 | SH3 PLY | checkpoint | 元数据/完整恢复状态 | 核验 |',
        '|---|---:|---|---|---|---|']
    for ratio in (10, 20):
        prefix = 'full_%dx' % ratio
        lines.append('| %d× | %d | [PLY](%s/stage3/point_cloud/iteration_25199/point_cloud.ply) | [checkpoint](%s/stage3/chkpnt25199.pth) | [元数据](%s/stage3/compression_metadata.npz) / [状态](%s/stage3/compression_state.pt) | [validation](%s/validation.json) |' %
                     (ratio, validations[ratio]['target'], prefix, prefix, prefix, prefix, prefix))
    lines += ['', '两组精确预算、PLY/checkpoint原生参数、逐行连续分量随剪枝/合并/回填的行对齐、评分公式复算、实例配额与原RIR/DPRA重放均有完成回执；归档另外校验源码快照副本/live SHA、冻结先验、最终模型与所有关键文件指纹。', '',
        tests['statement'], '',
        '[完整七指标CSV](comparison_all_metrics.csv)；[完整JSON](comparison_all_metrics.json)；[比较核验](comparison_validation.json)；[独立语义审计](independent_semantic_audit.json)；[训练源码快照](source_snapshot/manifest.json)；[评测源码快照](evaluation_source_snapshot/manifest.json)；[最终文件清单](final_artifact_manifest.json)。', '',
        'Recall为同类别8连通语义区域的一对一IoU≥0.5召回率，GT区域至少16像素，不是人工物理实例召回；B-mIoU为2%图像对角线宽的内边界带IoU宏平均。mIoU/mAcc按数据集计数聚合并忽略0类。独立语义审计的具体覆盖以其scope说明为准，不把未独立重算的指标表述为独立重算。', '',
        '现有标签没有电线或井盖的独立人工实例/细线标注，不能单独证明这些对象恢复或连续性改善。分割评估器有预测误差；LightGaussian10×比其他10×多1点，Speedy初始化不同。模型与渲染保留在本机，本文不表示已Git提交、推送或发布新标签。', '']
    return '\n'.join(lines)


def archive(output, repo=REPO, semantic_audit=None):
    output, repo = Path(output).resolve(), Path(repo).resolve()
    artifacts = Artifacts(repo)
    gates = {}
    mandatory = ['experiment_queue.json', 'evaluation_queue.json', 'full_10x/validation.json',
                 'full_20x/validation.json', 'comparison_validation.json', 'summary.json']
    for relative in mandatory:
        gates[relative] = complete(output / relative)
        artifacts.add(output / relative, 'completion_receipt')
    audit_path = Path(semantic_audit).resolve() if semantic_audit else output / 'independent_semantic_audit.json'
    audit = complete(audit_path)
    artifacts.add(audit_path, 'independent_semantic_audit')
    if audit.get('model_groups') != 20:
        raise ValueError('Independent semantic audit must cover 19 models plus GT')
    frozen, training_snapshot = verify_snapshot(output, 'source_snapshot', artifacts)
    _, evaluation_snapshot = verify_snapshot(output, 'evaluation_source_snapshot', artifacts)
    if frozen.get('distillation') is not False or gates['experiment_queue.json'].get('distillation') is not False:
        raise ValueError('This archive is restricted to the SH3-only experiment')
    for key in ('sidecar', 'manifest'):
        receipt = frozen['continuous_priors'][key]
        artifacts.add(receipt['path'], 'frozen_continuous_prior', receipt['sha256'])
    prior_path = artifacts.resolve(frozen['continuous_priors']['manifest']['path'])
    prior = complete(prior_path)
    if prior['sidecar_sha256'] != frozen['continuous_priors']['sidecar']['sha256'] or len(prior['train_frames']) != 61 or len(prior['test_frames_excluded']) != 9:
        raise ValueError('Continuous prior source/split differs from frozen experiment')
    artifacts.add(prior['shared_metadata'], 'unchanged_shared_metadata', prior['shared_metadata_sha256'])
    registry = read(output / 'comparison/models.json')
    artifacts.add(output / 'comparison/models.json', 'comparison_model_registry')
    metrics, indexed = validate_metric_tables(output, registry, artifacts)
    lpips = verify_standard_lpips(registry, metrics, artifacts)
    validations = {}
    for ratio, expected_count in ((10, 178809), (20, 89405)):
        run = output / ('full_%dx' % ratio)
        validation = gates['full_%dx/validation.json' % ratio]
        validations[ratio] = validation
        config = read(run / 'config.json')
        if config.get('boundary_score') != PARAMETERS or validation.get('boundary_score') != PARAMETERS:
            raise ValueError('Boundary weights/gain differ from predeclared parameters')
        if validation.get('target') != expected_count or validation.get('distillation_updates') != 0 or (run / 'sh2').exists():
            raise ValueError('Model budget or SH3-only scope differs')
        if validation['continuous_metadata_sha256'] != frozen['continuous_priors']['sidecar']['sha256'] or validation['continuous_manifest_sha256'] != frozen['continuous_priors']['manifest']['sha256']:
            raise ValueError('Completed model used another continuous prior')
        if validation['source_snapshot']['manifest_sha256'] != training_snapshot['manifest_sha256']:
            raise ValueError('Completed model source snapshot differs')
        native = validation['SH3']
        for key in ('checkpoint', 'ply'):
            artifacts.add(native[key], 'final_native_model', native[key + '_sha256'])
        for relative in ('config.json', 'pipeline.json', 'stage3/run_args.json', 'stage3/compression_metadata.npz',
                         'stage3/compression_state.pt', 'stage3/protection_policy.json',
                         'stage3/results.json', 'stage3/per_view.json', 'stage3/metrics_standard.json', 'stage3/metrics_float.json'):
            artifacts.add(run / relative, 'completed_compression_artifact')
        for event in validation['rounds']:
            for key in ('primary_scores', 'post_merge_scores'):
                receipt = event.get(key)
                if receipt:
                    artifacts.add(receipt['path'], 'per_event_score_snapshot', receipt['sha256'])
    verify_semantic_artifacts(output, artifacts)
    reviews = {ratio: verify_crop_review(output, ratio, artifacts) for ratio in (10, 20)}
    tests = unit_test_record(output, artifacts)
    for relative in ('README_压缩结果.md', 'README_完整结果对比.md', 'metrics_standard.json',
                     'city_label_source_identity.json', 'smoke/independent_smoke_validation.json',
                     'continuous_priors/preprocessing_status.json', 'continuous_priors/kernel_reference_validation.json',
                     'continuous_priors/reports/manifest.json', 'continuous_priors/reports/visual_review.json',
                     'continuous_priors/reports/README_连续先验可视化.md',
                     'continuous_priors/reports/continuous_boundary_training_views.png',
                     'continuous_priors/reports/continuous_gaussian_strength_histograms.png',
                     'continuous_priors/reports/continuous_gaussian_strength_statistics.json'):
        artifacts.add(output / relative, 'final_report_or_prior_visual')
    prior_visuals = read(output / 'continuous_priors/reports/manifest.json')
    for row in prior_visuals['outputs'].values():
        artifacts.add(row['path'], 'prior_visual_manifest_artifact', row['sha256'])
    script = artifacts.add(Path(__file__).resolve(), 'archive_source')
    # All readiness/identity checks above finish before either output is written.
    overview = output / 'README_实验总览.md'
    atomic_text(overview, build_readme(output, metrics, indexed, validations, prior, reviews, tests))
    artifacts.add(overview, 'final_master_report')
    result = dict(status='complete', schema='continuous_boundary_final_artifacts_v1',
        archived_utc=datetime.now(timezone.utc).isoformat(), output_root=str(output),
        archive_source=dict(path=script['path'], sha256=script['sha256']), gpu_used=False,
        training_restarted=False, source_changes_committed=False,
        training_snapshot=training_snapshot, evaluation_snapshot=evaluation_snapshot,
        frozen_continuous_prior_sha256=frozen['continuous_priors']['sidecar']['sha256'],
        standard_lpips_shared_weights_sha256=lpips['shared_weights_sha256'], standard_lpips=lpips,
        completion_receipts_verified=mandatory + [str(audit_path)],
        models=19, groups_including_gt=20, semantic_scopes=dict(zip(SCOPES, (2, 8, 10))),
        tests=tests, artifacts=sorted(artifacts.rows.values(), key=lambda row: row['path']),
        artifact_count=len(artifacts.rows),
        collection_scope='Explicit final models/checkpoints/metadata/resume states, score receipts, five evaluator receipts/checkpoints/prediction manifests, source live/copy identities, final reports/crops/prior visuals and validations. No recursive scan/hash of per-frame mass caches, raw images or all segmentation PNGs; their existing immutable manifests retain those fingerprints.',
        self_hash_excluded='final_artifact_manifest.json cannot contain its own SHA; archive script SHA and final master report SHA are included.')
    target = output / 'final_artifact_manifest.json'
    atomic_text(target, json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    print(str(overview))
    print(str(target))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=DEFAULT_OUT)
    parser.add_argument('--repo-root', type=Path, default=REPO)
    parser.add_argument('--semantic-audit', type=Path)
    args = parser.parse_args()
    archive(args.root, args.repo_root, args.semantic_audit)


if __name__ == '__main__':
    main()
