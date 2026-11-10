"""Join independent visual metrics and split-aware semantic comparisons."""
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'output/city2_continuous_boundary_20261109'
BINARY = ROOT / 'output/city2_boundary_weighted_20261106'
BASE = ROOT / 'output/city2_comparison_20261104'
PREVIOUS = ROOT / 'output/city2_pup_rir_dpra_20261105'


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def visual_metrics(model):
    if model['id'].startswith(('continuous_', 'weighted_', 'v320_')):
        folder = Path(model['model_dir'])
        value = read(folder / 'metrics_standard.json')
        if value.get('input_range') != [-1, 1] or len(value['images']) != 9:
            raise ValueError('Visual input protocol differs')
        source = folder / 'metrics_standard.json'
        result = {key: value[key] for key in ('PSNR', 'SSIM', 'LPIPS')}
    else:
        folder = Path(model['comparison_dir'])
        value = next(iter(read(folder / 'results.json').values()))
        standard = read(folder / 'comparison_lpips.json')
        source = folder / 'comparison_lpips.json'
        result = dict(PSNR=value['PSNR'], SSIM=value['SSIM'], LPIPS=standard['LPIPS'])
    return dict(result, provenance=dict(path=str(source), sha256=sha(source)), n_views=9)


def make_crops(ratio, registry):
    from finalize_city_benchmark import BOXES, font, NEAREST
    models = {row['id']: row for row in registry['models']}
    reference = models['original_3dgs']
    columns = [('GT', Path(reference['common_gt_dir']))]
    for model_id, title in [('weighted_%dx' % ratio, 'Binary boundary'), ('continuous_%dx' % ratio, 'Continuous boundary'),
                            ('pwsr_%dx' % ratio, 'PWSR'), ('lightgaussian_%dx' % ratio, 'LightGaussian')]:
        columns.append((title, Path(models[model_id]['common_render_dir'])))
    output = OUT / 'reports' / ('direct_continuous_%dx' % ratio)
    (output / 'native_crops').mkdir(parents=True, exist_ok=True)
    width, height = 330, 250
    sheet = Image.new('RGB', (width * len(columns), 70 + height * len(BOXES)), 'white')
    draw = ImageDraw.Draw(sheet)
    draw.text((8, 3), '%dx SH3 / fixed coordinates / exact native pixels / nearest-neighbour overview' % ratio, fill='black', font=font(16))
    draw.text((8, 25), 'No locations selected from the new outcome. Full seven-metric comparisons use the declared frame groups.', fill='black', font=font(12))
    for column, (title, _) in enumerate(columns):
        draw.text((column * width + 8, 47), title, fill='black', font=font(17))
    cases = []
    for index, (frame, box, category, description) in enumerate(BOXES):
        name = '%05d.png' % registry['test_frames'].index(frame)
        y = 70 + index * height
        draw.rectangle((0, y, sheet.width - 1, y + height - 1), outline='#cccccc')
        draw.text((8, y + 4), '%s | %s | %dx%d native pixels' %
                  (frame, description, box[2] - box[0], box[3] - box[1]), fill='black', font=font(13))
        gt = np.asarray(Image.open(columns[0][1] / name).convert('RGB'))
        records = {}
        for column, (title, folder) in enumerate(columns):
            path = folder / name
            image = Image.open(path).convert('RGB')
            if image.size != (1592, 894):
                raise ValueError('Native crop geometry differs')
            if column and not np.array_equal(np.asarray(Image.open(folder.parent / 'gt' / name).convert('RGB')), gt):
                raise ValueError('Fixed-crop GT mismatch')
            crop = image.crop(box)
            destination = output / 'native_crops' / ('%02d_%s_%d.png' % (index, frame, column))
            crop.save(destination)
            if not np.array_equal(np.asarray(crop), np.asarray(image)[box[1]:box[3], box[0]:box[2]]):
                raise ValueError('Crop pixels changed')
            factor = min((width - 12) / crop.width, 188. / crop.height)
            zoom = crop.resize((round(crop.width * factor), round(crop.height * factor)), NEAREST)
            sheet.paste(zoom, (column * width + (width - zoom.width) // 2, y + 42 + (200 - zoom.height) // 2))
            records[title] = dict(source=str(path), source_sha256=sha(path), crop=str(destination), crop_sha256=sha(destination), native_exact=True)
        cases.append(dict(frame=frame, bbox_xyxy=box, category=category, description=description, files=records))
    target = output / 'native_comparison.png'
    sheet.save(target)
    write(output / 'manifest.json', dict(ratio=ratio, columns=[row[0] for row in columns], cases=cases,
                                       figure=str(target), figure_sha256=sha(target)))
    return target


def validate_existing_main(semantic):
    """Every unchanged model/GT must reproduce all three Nov6 scopes exactly."""
    previous = read(BINARY / 'comparison/semantic/metrics.json')
    checked = []
    for scope, source in previous['scopes'].items():
        current = semantic['scopes'][scope]['results']
        for group, expected in source['results'].items():
            if current[group] != expected:
                raise ValueError('Unchanged semantic counts differ: ' + scope + '/' + group)
            checked.append(scope + '/' + group)
    return checked


def validate_aggregation(semantic):
    """Audit count-based means and the disjoint 2+8=10 aggregation."""
    scopes = semantic['scopes']
    checked = 0
    for scope in scopes.values():
        for row in scope['results'].values():
            metric = row['metrics']
            confusion = np.asarray(metric['confusion'], dtype=np.int64)
            gt, pred, intersection = confusion.sum(1), confusion.sum(0), np.diag(confusion)
            union = gt + pred - intersection
            classes = np.arange(1, 7)
            iou_ids, acc_ids = classes[union[classes] > 0], classes[gt[classes] > 0]
            expected_iou = float(np.mean(intersection[iou_ids] / union[iou_ids]))
            expected_acc = float(np.mean(intersection[acc_ids] / gt[acc_ids]))
            boundaries = [item['boundary_intersection'] / item['boundary_union']
                          for item in metric['per_class'].values() if item['boundary_union']]
            for key, expected in [('miou', expected_iou), ('macc', expected_acc),
                                  ('b_miou', float(np.mean(boundaries)))]:
                if not np.isclose(metric[key], expected, atol=1e-12, rtol=0):
                    raise ValueError('Count-based semantic aggregate differs: ' + key)
            tp = sum(item['regions']['tp'] for item in metric['per_class'].values())
            total = sum(item['regions']['gt'] for item in metric['per_class'].values())
            if metric['region_recall']['tp'] != tp or metric['region_recall']['gt'] != total or not np.isclose(metric['region_recall']['recall'], tp / total):
                raise ValueError('Region count aggregation differs')
            checked += 1
    if 'all_annotated_10' in scopes:
        for group, row in scopes['all_annotated_10']['results'].items():
            main = scopes['independent_test_2']['results'][group]['metrics']
            auxiliary = scopes['auxiliary_training_views_8']['results'][group]['metrics']
            total = row['metrics']
            if not np.array_equal(np.asarray(total['confusion']), np.asarray(main['confusion']) + np.asarray(auxiliary['confusion'])):
                raise ValueError('Ten-view confusion does not equal disjoint 2+8 counts')
            for class_id, item in total['per_class'].items():
                for key in ('intersection', 'union', 'boundary_intersection', 'boundary_union'):
                    if item[key] != main['per_class'][class_id][key] + auxiliary['per_class'][class_id][key]:
                        raise ValueError('Ten-view class counts do not equal disjoint 2+8 counts')
                for key in ('tp', 'gt', 'small_tp', 'small_gt'):
                    if item['regions'][key] != main['per_class'][class_id]['regions'][key] + auxiliary['per_class'][class_id]['regions'][key]:
                        raise ValueError('Ten-view region counts do not equal disjoint 2+8 counts')
    return checked


def finalize():
    registry = read(OUT / 'comparison/models.json')
    semantic = read(OUT / 'comparison/semantic/metrics.json')
    if semantic['status'] != 'complete':
        raise ValueError('Semantic comparison incomplete')
    visual = {model['id']: visual_metrics(model) for model in registry['models']}
    groups = {model['id']: model for model in registry['models']}
    verified = validate_existing_main(semantic)
    aggregates_checked = validate_aggregation(semantic)
    rows = []
    for scope, data in semantic['scopes'].items():
        for group, result in data['results'].items():
            values = result['metrics']
            model, metrics = groups.get(group, {}), visual.get(group, {})
            rows.append(dict(semantic_scope=scope, model_id=group, method=model.get('method', 'GT RGB'),
                ratio=model.get('target_ratio', ''), gaussians=model.get('actual_count', ''),
                visual_test_views=9 if metrics else '', semantic_views=result['n_semantic_views'],
                PSNR=metrics.get('PSNR', ''), SSIM=metrics.get('SSIM', ''), LPIPS_standard=metrics.get('LPIPS', ''),
                Recall=values['region_recall']['recall'], region_tp=values['region_recall']['tp'],
                region_gt=values['region_recall']['gt'], B_mIoU=values['b_miou'], mIoU=values['miou'], mAcc=values['macc']))
    with (OUT / 'comparison_all_metrics.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)
    write(OUT / 'comparison_all_metrics.json', dict(status='complete', visual=visual,
          semantic=semantic, rows=rows, historical_main_groups_reproduced=verified))
    lines = ['# City：无蒸馏、连续边界PUP完整比较', '',
        '本次仅输出SH3：PUP logdet + log(1+5B)，B=0.4×SAM轮廓+0.4×深度边缘+0.2×RGB边缘。三信号按高斯原生alpha*T可见贡献汇总为[0,1]连续值；det乘数[1,6]。边界不设独立生存配额或合并否决；实例/拓扑、恢复损失、合并1%和回填2%保留。', '',
        '视觉指标PSNR/SSIM/标准VGG0.1 LPIPS[-1,1]始终来自9个固定独立测试视角。语义使用全部10张匹配City标签，分主2帧、辅助8帧及10帧混合汇总；辅助折评估器只用其他6张非测试原RGB标签拟合。压缩不使用人工语义标签。', '',
        'Recall为同类别8连通语义区域的一对一IoU≥0.5召回（最小16像素）；B-mIoU为2%图像对角线宽的内边界带IoU宏平均；mIoU/mAcc按数据集类别统计，0类忽略。它们不是物理实例或电线/井盖专属指标。', '',
        '## 独立测试主比较', '',
        '| 方法 | 倍率 | PSNR↑ | SSIM↑ | LPIPS↓ | Recall↑ (%) | B-mIoU↑ (%) | mIoU↑ (%) | mAcc↑ (%) |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for row in rows:
        if row['semantic_scope'] != 'independent_test_2':
            continue
        metrics = [('—' if row[key] == '' else ('%.4f' % row[key])) for key in ('PSNR', 'SSIM', 'LPIPS_standard')]
        percentages = ['%.3f' % (100 * row[key]) for key in ('Recall', 'B_mIoU', 'mIoU', 'mAcc')]
        lines.append('| %s | %s | %s |' % (row['method'], row['ratio'], ' | '.join(metrics + percentages)))
    lines += ['', '## 全部标签的辅助比较', '',
        '下表的8帧均为3DGS训练视角，语义评估器对当前帧留出；10帧混合汇总不是10张独立测试。所有方法共享同一折权重，GT RGB也经过相同分割器，语义值不设为100%。', '']
    for scope in ('auxiliary_training_views_8', 'all_annotated_10'):
        if scope not in semantic['scopes']:
            continue
        lines += ['### ' + scope, '', '| 模型 | Recall↑ (%) | B-mIoU↑ (%) | mIoU↑ (%) | mAcc↑ (%) |', '|---|---:|---:|---:|---:|']
        for row in rows:
            if row['semantic_scope'] == scope:
                lines.append('| %s | %s |' % (row['model_id'], ' | '.join('%.3f' % (100 * row[key]) for key in ('Recall', 'B_mIoU', 'mIoU', 'mAcc'))))
        lines.append('')
    lines += ['## 相对上一轮二值边界版本', '']
    for ratio in (10, 20):
        new = next(row for row in rows if row['semantic_scope'] == 'independent_test_2' and row['model_id'] == 'continuous_%dx' % ratio)
        previous = next(row for row in rows if row['semantic_scope'] == 'independent_test_2' and row['model_id'] == 'weighted_%dx' % ratio)
        lines += ['- %d×：PSNR差%+.4f dB，SSIM差%+.5f，LPIPS差%+.5f；Recall/B-mIoU/mIoU/mAcc差分别为%s个百分点。' %
                  (ratio, new['PSNR'] - previous['PSNR'], new['SSIM'] - previous['SSIM'], new['LPIPS_standard'] - previous['LPIPS_standard'],
                   '/'.join('%+.3f' % (100 * (new[key] - previous[key])) for key in ('Recall', 'B_mIoU', 'mIoU', 'mAcc')))]
        figure = make_crops(ratio, registry)
        lines += ['', '[%d×固定六处原生裁剪](%s)' % (ratio, figure), '']
    lines += ['## 复核与限制', '',
        '两组均完整10199次更新，精确预算和原RIR/DPRA重放、原生PLY/checkpoint参数、实例配额、逐行加权评分核验见full_10x/validation.json及full_20x/validation.json。不存在SH2阶段。',
        '旧方法/GT的%d组分范围语义统计与上一轮逐值一致，所有比较沿用相同五折冻结分割器；未重新训练或选择分割器。原生渲染/预测只在输入和checkpoint的SHA同时一致时复用。' % len(verified),
        '相对上一轮二值版本只改变边界评分及其连续先验，权重预先固定，未用人工评测标签调参；未做消融，不能据此宣称权重最优。LightGaussian10×多1点，Speedy初始化不同。',
        '完整机器可读结果见comparison_all_metrics.csv和comparison_all_metrics.json；模型及checkpoint在full_{10x,20x}/stage3。当前源代码尚未另行提交Git或发布新标签。']
    (OUT / 'README_完整结果对比.md').write_text('\n'.join(lines) + '\n')
    write(OUT / 'comparison_validation.json', dict(status='complete', models=len(groups) + 1,
        scopes={key: len(value['frames']) for key, value in semantic['scopes'].items()},
        old_main_groups_reproduced=verified, identical_heldout_statistics=True,
        semantic_aggregates_checked=aggregates_checked, disjoint_two_plus_eight_counts_verified=True,
        visual_n_views=9, standard_lpips_input_range=[-1, 1], source_snapshot_sha256=sha(OUT / 'source_snapshot/manifest.json'),
        semantic_protocol_sha256=sha(OUT / 'comparison/semantic/protocol.json')))
    print(str(OUT / 'README_完整结果对比.md'), flush=True)


if __name__ == '__main__':
    finalize()
