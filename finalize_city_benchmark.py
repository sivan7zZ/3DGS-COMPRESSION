#!/usr/bin/env python3
"""CPU-only aggregation of canonical SH3 renders, frozen semantics and ablations.

This script never trains, renders with CUDA, or invents missing metric values.
Metric JSONs and exact native image pixels are the inputs to all report artifacts.
"""
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import shutil

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path('/home/zhangys/Prune-Wisely-Reconstruct-Sharply')
DEFAULT = ROOT / 'output/city2_comparison_20261104'
BOXES = [
    ('frame_024', (1321,30,1380,69), 'small_region', 'Black parked vehicle'),
    ('frame_064', (1350,821,1373,844), 'small_region', 'Small white garden object'),
    ('frame_056', (1352,70,1382,101), 'boundary', 'Small rooftop-unit outline'),
    ('frame_008', (1183,77,1232,116), 'thin_region', 'Narrow window-frame divisions'),
    ('frame_000', (157,31,247,70), 'boundary', 'Garden fence and panel divisions'),
    ('frame_024', (83,192,111,225), 'small_region', 'Red parked vehicle / control'),
]
CLASSES = ['unannotated','Building','Tree','Road','Vehicle','Grass','Ground']
PALETTE = [(0,0,0),(70,70,70),(0,100,0),(128,64,128),(0,0,142),(107,142,35),(152,251,152)]
VARIANTS = ['no_instance_survival','no_boundary_quota','no_topology_guard',
            'no_adaptive_schedule','no_merging','no_budget_reallocation','no_structure_loss']
VARIANT_LABELS = dict(zip(VARIANTS, ['关闭实例存活配额','关闭边界配额','关闭拓扑模块',
                      '关闭自适应调度','关闭高斯合并','关闭预算回填','关闭新增结构损失']))
METRICS = ['PSNR','SSIM','LPIPS','Recall','B-mIoU','mIoU','mAcc']
NEAREST = getattr(Image, 'Resampling', Image).NEAREST


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4*1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    os.replace(str(tmp), str(path))


def font(size):
    for name in ('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',
                 '/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf'):
        if Path(name).is_file():
            return ImageFont.truetype(name, size)
    return ImageFont.load_default()


def number(value, name, delta=False):
    if value is None:
        return '—'
    if not math.isfinite(float(value)):
        raise ValueError('Nonfinite report metric: ' + name)
    if name in ('Recall','B-mIoU','mIoU','mAcc'):
        value = float(value) * 100.
        return ('%+.2f' if delta else '%.2f') % value
    if name == 'PSNR':
        return ('%+.4f' if delta else '%.4f') % value
    return ('%+.5f' if delta else '%.5f') % value


def metric_values(visual, semantic):
    recall = semantic.get('region_recall', {}) if semantic else {}
    return dict(PSNR=visual.get('PSNR') if visual else None,
                SSIM=visual.get('SSIM') if visual else None,
                LPIPS=visual.get('LPIPS') if visual else None,
                Recall=recall.get('recall'),
                **{'B-mIoU': semantic.get('b_miou') if semantic else None,
                   'mIoU': semantic.get('miou') if semantic else None,
                   'mAcc': semantic.get('macc') if semantic else None})


def semantic_bundle(root, allow_pending):
    path = root / 'semantic/all_metrics.json'
    if not path.is_file():
        path = root / 'semantic/main_metrics.json'
    if not path.is_file():
        if not allow_pending:
            raise FileNotFoundError('Semantic metrics not ready')
        return dict(status='pending', results={}, predictions={}, path=None, provenance={})
    metrics = read(path)
    if metrics['status'] != 'complete':
        raise ValueError('Semantic metric file incomplete')
    prediction_folder = 'predictions_all' if path.name == 'all_metrics.json' else 'predictions_main'
    pred_path = root / 'semantic' / prediction_folder / 'predictions_manifest.json'
    if sha(pred_path) != metrics['predictions_manifest_sha256']:
        raise ValueError('Semantic metrics and prediction manifest differ')
    predictions = read(pred_path)
    if sha(metrics['checkpoint']) != metrics['checkpoint_sha256'] or predictions['checkpoint_sha256'] != metrics['checkpoint_sha256']:
        raise ValueError('Frozen segmentation checkpoint changed')
    if sha(root / 'semantic/dataset_manifest.json') != metrics['data_manifest_sha256']:
        raise ValueError('Semantic GT dataset manifest changed')
    by_id = {r['id']: r for r in predictions['predictions']}
    if len(by_id) != len(predictions['predictions']):
        raise ValueError('Duplicate semantic prediction ID')
    for row in by_id.values():
        if sha(row['prediction_path']) != row['prediction_sha256'] or sha(row['image_path']) != row['image_sha256']:
            raise ValueError('Semantic prediction or its RGB source changed')
    return dict(status='complete', results=metrics['results'], predictions=by_id, path=str(path),
                provenance=dict(metrics_path=str(path), metrics_sha256=sha(path),
                                predictions_path=str(pred_path), predictions_sha256=sha(pred_path),
                                checkpoint=metrics['checkpoint'], checkpoint_sha256=metrics['checkpoint_sha256'],
                                protocol=metrics['protocol']))


def lpips_bundle(root):
    """One experiment-wide LPIPS protocol, never a mixture of legacy and corrected."""
    path = root / 'lpips_standard_manifest.json'
    if not path.is_file():
        return dict(protocol='legacy_0_1', models={}, path=None,
                    qualification='Historical repository convention; pending standard [-1,1] postprocess')
    manifest = read(path)
    if manifest.get('status') != 'complete' or manifest.get('input_range') != [-1,1]:
        raise ValueError('Standard LPIPS postprocess is incomplete; keep previous report until it finishes')
    if manifest.get('n_test_views') != 9:
        raise ValueError('Standard LPIPS test support differs')
    return dict(protocol='standard_minus1_plus1',models=manifest['models'],path=str(path),
                sha256=sha(path),criterion=manifest['criterion'],self_control=manifest['self_control'])


def apply_lpips(item, bundle):
    if item.get('visual_status') != 'complete':
        return item
    legacy = item['visual_metrics']['LPIPS']
    item['LPIPS_legacy_0_1'] = legacy
    item['lpips_protocol'] = bundle['protocol']
    if bundle['protocol'] == 'legacy_0_1':
        return item
    entry = bundle['models'].get(item['id'])
    if entry is None:
        raise ValueError('Missing uniform standard LPIPS: '+item['id'])
    path = Path(entry['path'])
    if sha(path) != entry['sha256']:
        raise ValueError('Standard LPIPS receipt changed: '+item['id'])
    value = read(path)
    if value.get('status') != 'complete' or value.get('model_id') != item['id'] or value.get('input_range') != [-1,1]:
        raise ValueError('Invalid standard LPIPS receipt: '+item['id'])
    if value.get('cache_signature') != entry['cache_signature'] or value['criterion'] != bundle['criterion']:
        raise ValueError('Standard LPIPS protocol or signature differs')
    if not math.isclose(value['legacy']['LPIPS'],legacy,rel_tol=1e-10,abs_tol=1e-10):
        raise ValueError('Legacy LPIPS lineage differs')
    expected = {'%05d.png'%i for i in range(9)}
    if set(value['per_view']) != expected or not math.isclose(value['LPIPS'],sum(value['per_view'].values())/9,rel_tol=1e-10,abs_tol=1e-10):
        raise ValueError('Standard LPIPS per-view support or mean differs')
    if not math.isfinite(float(value['LPIPS'])) or value['LPIPS'] != entry['LPIPS']:
        raise ValueError('Invalid standard LPIPS mean')
    for image in value['images']:
        name=image['filename']
        source=item['visual_metric_provenance']
        if image['render_sha256'] != source['render_sha256'][name] or image['gt_sha256'] != source['gt_sha256'][name]:
            raise ValueError('Standard LPIPS used different PNGs')
    item['visual_metrics']['LPIPS']=float(value['LPIPS'])
    item['standard_lpips_provenance']=dict(path=str(path),sha256=entry['sha256'],input_range=[-1,1])
    return item


def visual_record(entry, tests, allow_pending, lpips=None):
    item = dict(entry)
    item.update(visual_metrics=None, visual_status='pending', visual_metric_provenance=None)
    if entry.get('status') == 'missing':
        item['visual_status'] = 'missing'
        if not allow_pending:
            raise FileNotFoundError('Missing registered model ' + entry['id'])
        return item
    ply = Path(entry['ply'])
    if not ply.is_file():
        if not allow_pending:
            raise FileNotFoundError(str(ply))
        return item
    actual_ply_sha=sha(ply)
    if entry.get('ply_sha256') and actual_ply_sha != entry['ply_sha256']:
        raise ValueError('Registered PLY content changed: ' + entry['id'])
    item['ply_sha256']=actual_ply_sha
    item['ply_bytes'] = ply.stat().st_size
    result_path = Path(entry['comparison_dir']) / 'results.json'
    key = 'ours_%d' % int(entry['iteration'])
    renders, gt = Path(entry['common_render_dir']), Path(entry['common_gt_dir'])
    expected = ['%05d.png' % i for i in range(len(tests))]
    if result_path.is_file() and key in read(result_path) and all((renders/f).is_file() and (gt/f).is_file() for f in expected):
        values = read(result_path)[key]
        if any(not math.isfinite(float(values[k])) for k in ('PSNR','SSIM','LPIPS')):
            raise ValueError('Nonfinite PNG metrics')
        item['visual_metrics'] = {k: float(values[k]) for k in ('PSNR','SSIM','LPIPS')}
        item['visual_status'] = 'complete'
        item['visual_metric_provenance'] = dict(path=str(result_path), sha256=sha(result_path),
                                                key=key, test_views=tests, n_views=9,
                                                render_sha256={f: sha(renders/f) for f in expected},
                                                gt_sha256={f: sha(gt/f) for f in expected})
    elif not allow_pending:
        raise ValueError('Canonical PNG metrics/renders incomplete: ' + entry['id'])
    return apply_lpips(item,lpips) if lpips else item


def attach_semantics(item, bundle, allow_pending):
    row = bundle['results'].get(item['id'])
    item['semantic_metrics'] = row['metrics'] if row else None
    item['semantic_status'] = 'complete' if row else 'pending'
    item['semantic_frames'] = row['semantic_frames'] if row else None
    item['metrics'] = metric_values(item.get('visual_metrics'), item['semantic_metrics'])
    if not row and not allow_pending:
        raise ValueError('Semantic evaluation missing: ' + item['id'])
    return item


def native_crops(root, ratio, records, tests, allow_pending):
    ids = ['original_3dgs','lightgaussian_%dx'%ratio,'pup_%dx'%ratio,
           'speedy_%dx'%ratio,'pwsr_%dx'%ratio,'ours_%dx'%ratio]
    if any(records.get(x, {}).get('visual_status') != 'complete' for x in ids):
        if allow_pending:
            return None
        raise ValueError('Cannot build incomplete native comparisons')
    destination = root / 'reports' / ('%dx_sh3' % ratio)
    native = destination / 'native_crops'
    native.mkdir(parents=True, exist_ok=True)
    gt_dir = Path(records['original_3dgs']['common_gt_dir'])
    columns = [('GT', gt_dir)] + [(name, Path(records[ident]['common_render_dir'])) for name,ident in zip(
        ['3DGS','LightGaussian','PUP-3DGS','Speedy-Splat','PWSR','Ours'], ids)]
    width, rowheight = 290, 250
    sheet = Image.new('RGB', (width*len(columns),70+rowheight*len(BOXES)), 'white')
    draw = ImageDraw.Draw(sheet)
    draw.text((8,4), 'Native SH3 / %dx Gaussian count / six predefined locations' % ratio, fill='black', font=font(17))
    draw.text((8,28), 'Exact native crops. Nearest-neighbour zoom only. Object disappearance/rescue is not assumed.', fill='black', font=font(12))
    for col,(label,_) in enumerate(columns):
        draw.text((col*width+8,49), label, fill='black', font=font(17))
    cases=[]
    for row,(frame,box,category,description) in enumerate(BOXES):
        filename='%05d.png'%tests.index(frame)
        reference=np.asarray(Image.open(gt_dir/filename).convert('RGB'))
        y=70+row*rowheight
        draw.rectangle((0,y,sheet.width-1,y+rowheight-1), outline='#cccccc')
        draw.text((8,y+4), '%s | %s | %s | %dx%d native px' % (frame,category,description,box[2]-box[0],box[3]-box[1]), fill='black', font=font(13))
        case=dict(frame=frame,bbox_xyxy=list(box),category=category,description=description,files={})
        for col,(label,folder) in enumerate(columns):
            path=folder/filename
            image=Image.open(path).convert('RGB')
            if image.size!=(1592,894):
                raise ValueError('Non-native image geometry')
            if label!='GT' and not np.array_equal(np.asarray(Image.open(folder.parent/'gt'/filename).convert('RGB')), reference):
                raise ValueError('Crop cameras or GT differ: '+label)
            crop=image.crop(box)
            target=native/('%02d_%s_%s.png'%(row,frame,label))
            crop.save(str(target))
            if not np.array_equal(np.asarray(Image.open(target)),np.asarray(image)[box[1]:box[3],box[0]:box[2]]):
                raise ValueError('Native crop pixels changed')
            scale=min((width-12)/float(crop.width),188./float(crop.height))
            zoom=crop.resize((int(round(crop.width*scale)),int(round(crop.height*scale))),NEAREST)
            sheet.paste(zoom,(col*width+(width-zoom.width)//2,y+42+(200-zoom.height)//2))
            case['files'][label]=dict(source=str(path),source_sha256=sha(path),crop=str(target),
                                     crop_sha256=sha(target),native_pixels_exact=True)
        cases.append(case)
    target=destination/'native_sh3_comparison.png'
    sheet.save(str(target))
    write(destination/'crop_manifest.json',dict(ratio=ratio,columns=[x[0] for x in columns],cases=cases,
          protocol='six fixed coordinates inherited before this comparison; exact native pixels; nearest-neighbour overview',
          visual_conclusion='pending manual review; no automatic disappearance/rescue claim'))
    return str(target)


def palette_image(array):
    value=np.asarray(array)
    if value.ndim!=2 or np.any(value<0) or np.any(value>6):
        raise ValueError('Invalid semantic labels')
    image=Image.fromarray(value.astype(np.uint8),'P')
    palette=[x for rgb in PALETTE for x in rgb]
    image.putpalette(palette+[0]*(768-len(palette)))
    return image


def semantic_sheets(root, ratio, bundle, data, allow_pending):
    if bundle['status']!='complete':
        return None
    test=data['splits']['test']
    columns=[('Label GT',None),('Seg(GT RGB)','GT'),('Seg(3DGS)','original_3dgs')]
    columns += [(name,ident+'_%dx'%ratio) for name,ident in [('LightGaussian','lightgaussian'),
                ('PUP-3DGS','pup'),('Speedy-Splat','speedy'),('PWSR','pwsr'),('Ours','ours')]]
    if any(group not in bundle['results'] for _,group in columns if group is not None):
        if allow_pending:
            return None
        raise ValueError('Semantic map comparison is incomplete')
    destination=root/'reports'/('%dx_semantic'%ratio)
    native=destination/'native_maps'
    native.mkdir(parents=True,exist_ok=True)
    width, thumbnail_height,rowheight=360,202,245
    sheet=Image.new('RGB',(len(columns)*width,100+len(test)*rowheight),'white')
    draw=ImageDraw.Draw(sheet)
    draw.text((8,4),'Frozen common semantic evaluator / %dx SH3 / two annotated held-out views'%ratio,fill='black',font=font(18))
    draw.text((8,29),'Label GT is manual annotation. Seg(GT RGB) is measured, not perfect. Colors show exact label IDs; nearest-neighbour overview.',fill='black',font=font(13))
    for i,(label,_) in enumerate(columns):
        draw.text((i*width+7,76),label,fill='black',font=font(17))
    for cls,color in enumerate(PALETTE):
        x=8+cls*190
        draw.rectangle((x,53,x+16,67),fill=color)
        draw.text((x+22,52),'%d %s'%(cls,CLASSES[cls]),fill='black',font=font(12))
    frames={r['frame']:r for r in data['frames']}
    maps=[]
    for row,frame in enumerate(test):
        y=100+row*rowheight
        draw.text((8,y+3),frame,fill='black',font=font(15))
        for col,(label,group) in enumerate(columns):
            if group is None:
                source=Path(frames[frame]['label_path'])
            else:
                ids=bundle['results'][group]['prediction_ids']
                ident=ids[test.index(frame)]
                if ident.split('/')[-1]!=frame:
                    raise ValueError('Semantic prediction frame order differs')
                source=Path(bundle['predictions'][ident]['prediction_path'])
            array=np.asarray(Image.open(source).convert('L'))
            if array.shape!=(894,1592):
                raise ValueError('Semantic native geometry differs')
            colored=palette_image(array)
            stem=frame+'_'+('manual_GT' if group is None else group)
            target=native/(stem+'_palette.png')
            colored.save(str(target))
            labels=native/(stem+'_labels.png')
            Image.fromarray(array.astype(np.uint8),'L').save(str(labels))
            if not np.array_equal(np.asarray(Image.open(target)),array) or not np.array_equal(np.asarray(Image.open(labels)),array):
                raise ValueError('Semantic palette/label IDs changed')
            thumb=colored.convert('RGB').resize((width-8,thumbnail_height),NEAREST)
            sheet.paste(thumb,(col*width+4,y+29))
            maps.append(dict(frame=frame,label=label,group=group,source=str(source),source_sha256=sha(source),
                             palette_png=str(target),label_png=str(labels),native_label_ids_exact=True))
    target=destination/'semantic_comparison.png'
    sheet.save(str(target))
    write(destination/'map_manifest.json',dict(ratio=ratio,frames=test,columns=[x[0] for x in columns],
          palette={str(i):dict(name=CLASSES[i],rgb=list(PALETTE[i])) for i in range(7)},
          maps=maps,protocol='two annotated held-out frames; native exact IDs; nearest-neighbour contact sheet'))
    return str(target)


def ablation_records(root, tests, bundle, full, allow_pending, lpips=None):
    rows=[]
    for slug in VARIANTS:
        directory=root/'ablations20'/slug
        run_path=directory/'stage3/run_args.json'
        entry=dict(id='ablation_'+slug,label=VARIANT_LABELS[slug],slug=slug,target_ratio=20,
                   status='pending',actual_count=None,visual_metrics=None,visual_status='pending',
                   semantic_metrics=None,semantic_status='pending',rounds=None,recovery_updates=None,
                   merged_total=None,restored_total=None,metrics=metric_values(None,None))
        if run_path.is_file():
            run=read(run_path)
            entry['training_status']=run['status']
            if run.get('status')=='complete' and 'final_iteration' in run:
                iteration=run['final_iteration']
                model=dict(id=entry['id'],status='saved',ply=run['last_ply'],actual_count=run['final_gaussians'],
                           comparison_dir=str(directory/'stage3'),iteration=iteration,
                           common_render_dir=str(directory/'stage3/test'/('ours_%d'%iteration)/'renders'),
                           common_gt_dir=str(directory/'stage3/test'/('ours_%d'%iteration)/'gt'))
                entry.update(visual_record(model,tests,allow_pending,lpips))
                entry.update(rounds=len(run['events']),recovery_updates=run['completed_steps'],
                             merged_total=sum(e['merged'] for e in run['events']),
                             restored_total=sum(e['restored'] for e in run['events']))
                validation=directory/'stage3/ablation_validation.json'
                if validation.is_file():
                    entry['validation']=read(validation)
                elif not allow_pending:
                    raise ValueError('Ablation validation missing: '+slug)
                if run['final_gaussians']!=89405:
                    raise ValueError('Ablation final budget differs')
        attach_semantics(entry,bundle,allow_pending)
        entry['status']='complete' if entry['visual_status']=='complete' and entry['semantic_status']=='complete' else 'pending'
        entry['delta_vs_full']={k:(entry['metrics'][k]-full['metrics'][k] if entry['metrics'][k] is not None and full['metrics'][k] is not None else None) for k in METRICS}
        if entry['status']!='complete' and not allow_pending:
            raise ValueError('Ablation incomplete: '+slug)
        rows.append(entry)
    return rows


def csv_write(path, keys, rows):
    with Path(path).open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=keys)
        writer.writeheader()
        writer.writerows({k:row.get(k) for k in keys} for row in rows)


def markdown_table(rows, with_count=True):
    headers=['方法']+(['高斯数'] if with_count else [])+['PSNR↑','SSIM↑','LPIPS↓','Recall↑ (%)','B-mIoU↑ (%)','mIoU↑ (%)','mAcc↑ (%)']
    lines=['| '+' | '.join(headers)+' |','|'+'|'.join(['---']+['---:']*(len(headers)-1))+'|']
    for row in rows:
        cells=[row['label']]+([str(row.get('actual_count')) if row.get('actual_count') is not None else '—'] if with_count else [])
        cells += [number(row['metrics'][k],k) for k in METRICS]
        lines.append('| '+' | '.join(cells)+' |')
    return lines


def region_details(row):
    semantic=row.get('semantic_metrics') or {}
    counts=semantic.get('region_recall',{})
    return dict(overall_tp=counts.get('tp'),overall_gt=counts.get('gt'),overall_recall=counts.get('recall'),
                small_tp=counts.get('small_tp'),small_gt=counts.get('small_gt'),small_recall=counts.get('small_recall'),
                vehicle_tp=counts.get('vehicle_tp'),vehicle_gt=counts.get('vehicle_gt'),vehicle_recall=counts.get('vehicle_recall'))


def region_table(rows):
    lines=['| 方法 | 全部区域 TP/GT | Recall (%) | 小区域 TP/GT | Small Recall (%) | Vehicle区域 TP/GT | Vehicle Recall (%) |',
           '|---|---:|---:|---:|---:|---:|---:|']
    for row in rows:
        detail=region_details(row)
        cells=[row['label']]
        for prefix in ('overall','small','vehicle'):
            tp,gt=detail[prefix+'_tp'],detail[prefix+'_gt']
            cells += [('%d/%d'%(tp,gt)) if tp is not None and gt is not None else '—',
                      number(detail[prefix+'_recall'],'Recall')]
        lines.append('| '+' | '.join(cells)+' |')
    return lines


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=DEFAULT)
    p.add_argument('--allow_pending',action='store_true')
    args=p.parse_args()
    root=args.root.resolve()
    reports=root/'reports'
    reports.mkdir(parents=True,exist_ok=True)
    registry=read(root/'models.json')
    tests=registry['test_frames']
    if tests!=['frame_%03d'%i for i in range(0,70,8)]:
        raise ValueError('Canonical nine-view camera list differs')
    data=read(root/'semantic/dataset_manifest.json')
    if data['splits']['test']!=['frame_000','frame_056'] or data['ignore_label']!=0:
        raise ValueError('Semantic held-out split or ignore label differs')
    bundle=semantic_bundle(root,args.allow_pending)
    lpips=lpips_bundle(root)
    records={entry['id']:attach_semantics(visual_record(entry,tests,args.allow_pending,lpips),bundle,args.allow_pending)
             for entry in registry['models']}
    gt_sem=bundle['results'].get('GT',{}).get('metrics')
    gt=dict(id='GT',label='GT RGB（分割器参考）',actual_count=None,visual_metrics=None,semantic_metrics=gt_sem,
            metrics=metric_values(None,gt_sem),visual_status='reference',
            semantic_status='complete' if gt_sem else 'pending')
    summary=dict(schema_version=1,status='pending',experiment_root=str(root),models=records,
                 visual_test_frames=tests,n_visual_views=9,semantic_test_frames=data['splits']['test'],n_semantic_views=2,
                 metric_units='raw values are fractions; report semantic/recall values are percentages; delta semantic values are percentage points',
                 GT=gt,ratios={},ablations20={},semantic_provenance=bundle['provenance'],lpips_provenance=lpips,
                 provenance=dict(models_json=str(root/'models.json'),models_sha256=sha(root/'models.json'),
                                 dataset_manifest_sha256=sha(root/'semantic/dataset_manifest.json'),
                                 report_script_sha256=sha(Path(__file__))),
                 manual_visual_review=dict(status='pending',conclusion='固定位置对比图供人工审阅；不从全局分数或语义指标自动推断地物消失或挽救。'))
    review_path=reports/'manual_visual_review.json'
    if review_path.is_file():
        summary['manual_visual_review']=read(review_path)
    flat=[]
    per_class=[]
    regional=[]
    lines=['# City 统一对比与模块消融','',
           '主对比使用蒸馏前标准 SH3 模型：GT、原始3DGS、LightGaussian、PUP-3DGS、Speedy-Splat、PWSR及本次自适应约束方法。所有模型重新采用同一原生渲染器、1592×894分辨率及相同9个测试相机计算PNG视觉指标。原实验的SH2模型继续保留，未混入主表。','',
           '**语义评估只有2张带标注的测试视图：frame_000、frame_056。** 使用同一个冻结的场景分割器对各模型RGB渲染图分割，再与配准后的人工City标签比较；分割器只用8张非测试标注影像拟合，选步数时使用其中6张训练、2张内部验证。压缩过程不使用这些语义真值。GT RGB行也要经过分割器，实测分数不设为100%。','',
           'Recall为同类别8连通语义区域的召回：GT区域至少16像素、IoU≥0.5、一对一最大匹配。它不等于物理目标实例召回。B-mIoU为按类别计算的Boundary IoU均值，内边界宽度为图像对角线的2%（本分辨率37像素），GT标签0及其附近无效域不计入。mIoU/mAcc在两张图汇总像素计数后按有效类别宏平均；mAcc为各类像素召回的均值。语义与召回在下表显示百分数。','']
    method_keys=[('3DGS','original_3dgs'),('LightGaussian','lightgaussian'),('PUP-3DGS','pup'),
                 ('Speedy-Splat*','speedy'),('PWSR','pwsr'),('Ours','ours')]
    for ratio in (10,20):
        rows=[dict(gt)]
        for label,prefix in method_keys:
            ident=prefix if prefix=='original_3dgs' else prefix+'_%dx'%ratio
            if ident not in records:
                if not args.allow_pending:
                    raise ValueError('Required registered baseline missing: '+ident)
                row=dict(id=ident,label=label,actual_count=None,metrics=metric_values(None,None))
            else:
                row=dict(records[ident],label=label)
            rows.append(row)
        crop=native_crops(root,ratio,records,tests,args.allow_pending)
        semantic_sheet=semantic_sheets(root,ratio,bundle,data,args.allow_pending)
        summary['ratios'][str(ratio)]=dict(rows=rows,native_comparison=crop,semantic_comparison=semantic_sheet)
        lines += ['## %d×主对比'%ratio,'']+markdown_table(rows)+['']
        if crop:
            lines += ['[固定六处原生地物对比图](%s)'%crop,'']
        if semantic_sheet:
            lines += ['[人工标签与各方法语义分割图](%s)'%semantic_sheet,'']
        lines += ['区域召回明细：小区域定义为16像素至整图面积1%的GT语义连通区域；Vehicle也按语义连通区域计数，同类相连车辆可能属于同一区域。','']
        lines += region_table(rows)+['']
        for row in rows:
            regional.append(dict(group='main',ratio=ratio,model_id=row['id'],method=row['label'],**region_details(row)))
        summary['ratios'][str(ratio)]['region_recall_breakdown']=[dict(model_id=row['id'],**region_details(row)) for row in rows]
        lines += ['各类别IoU（%）；其他逐类指标见CSV。','',
                  '| 方法 | Building | Tree | Road | Vehicle | Grass | Ground |',
                  '|---|---:|---:|---:|---:|---:|---:|']
        for row in rows:
            sem=row.get('semantic_metrics')
            cells=[]
            for cls in range(1,7):
                value=sem.get('per_class',{}).get(str(cls),{}) if sem else {}
                cells.append(number(value.get('iou'),'mIoU'))
                if value:
                    per_class.append(dict(ratio=ratio,model_id=row['id'],method=row['label'],class_id=cls,class_name=CLASSES[cls],
                        iou=value['iou'],pixel_recall=value['pixel_recall'],boundary_iou=value['boundary_iou'],
                        region_recall=value['regions']['recall'],small_region_recall=value['regions']['small_recall'],
                        gt_pixels=value['gt_pixels'],gt_regions=value['regions']['gt'],tp_regions=value['regions']['tp']))
            lines.append('| '+row['label']+' | '+' | '.join(cells)+' |')
            flat.append(dict(ratio=ratio,model_id=row['id'],method=row['label'],gaussians=row.get('actual_count'),
                             **row['metrics']))
        lines += ['']
    full=dict(records['ours_20x'],label='完整方法')
    full_run=read(Path(full['model_dir'])/'run_args.json')
    full.update(rounds=len(full_run['events']),recovery_updates=full_run['completed_steps'],
                merged_total=sum(e['merged'] for e in full_run['events']),
                restored_total=sum(e['restored'] for e in full_run['events']))
    ablations=ablation_records(root,tests,bundle,full,args.allow_pending,lpips)
    summary['ablations20']=dict(reference=full,variants=ablations,
            protocol='20x SH3 leave-one-implemented-module-out, same input/config except one enabled flag and output path; one seed; not compute matched')
    lines += ['## 20×逐模块消融','',
              '每项从原始未压缩模型重新开始，最终均为89,405点；完整方法复用已有结果。共同保留v3恢复损失。关闭新增结构损失只移除风险加权L1；关闭拓扑模块同时取消其细线候选对合并的排除。关闭自适应会改变总轮数与恢复步数，因此本组固定模型预算，未固定计算量。','']
    lines += markdown_table([full]+ablations)+['',
              '| 变体 | 轮数 | 恢复步数 | 合并总数 | 回填总数 | 状态 |',
              '|---|---:|---:|---:|---:|---|']
    for row in [full]+ablations:
        lines.append('| %s | %s | %s | %s | %s | %s |'%(row['label'],
                     row.get('rounds') if row.get('rounds') is not None else '—',
                     row.get('recovery_updates') if row.get('recovery_updates') is not None else '—',
                     row.get('merged_total') if row.get('merged_total') is not None else '—',
                     row.get('restored_total') if row.get('restored_total') is not None else '—',
                     row.get('status','complete')))
    lines += ['','区域召回明细（20×消融，分母与完整方法相同；总体Recall提高不代表Vehicle或小区域均提高）。','']
    lines += region_table([full]+ablations)
    for row in [full]+ablations:
        regional.append(dict(group='ablation20',ratio=20,model_id=row['id'],method=row['label'],**region_details(row)))
    summary['ablations20']['region_recall_breakdown']=[dict(model_id=row['id'],**region_details(row)) for row in [full]+ablations]
    lines += ['','Δ=关闭该模块后的指标−完整方法；语义/召回Δ单位为百分点。PSNR/SSIM/Recall/B-mIoU/mIoU/mAcc正值较好，LPIPS负值较好。单种子的逐项关闭含有模块交互，差值不可直接相加。','',
              '| 关闭模块 | ΔPSNR | ΔSSIM | ΔLPIPS | ΔRecall | ΔB-mIoU | ΔmIoU | ΔmAcc |',
              '|---|---:|---:|---:|---:|---:|---:|---:|']
    for row in ablations:
        lines.append('| '+row['label']+' | '+' | '.join(number(row['delta_vs_full'][k],k,delta=True) for k in METRICS)+' |')
    lines += ['','## 既有v3结果（补充）','']
    v3rows=[]
    for ratio in (10,20):
        ident='v3_%dx'%ratio
        if ident in records:
            v3rows.append(dict(records[ident],label='v3 %d×'%ratio))
    lines += markdown_table(v3rows)+['','## LPIPS输入范围与历史值','']
    if lpips['protocol']=='standard_minus1_plus1':
        lines += ['主表及所有消融的LPIPS统一使用VGG v0.1、原生整图、RGB归一化后转换到[-1,1]，符合[作者实现的输入范围](https://github.com/richzhang/PerceptualSimilarity/blob/master/lpips/lpips.py)。旧脚本直接输入[0,1]，旧results.json/per_view.json保持原样；PSNR、SSIM和语义指标不变。下表仅供追溯，不能把两种范围的LPIPS混合比较。','']
    else:
        lines += ['当前LPIPS为旧脚本[0,1]输入约定，标准[-1,1]后处理尚未完成。全部模型当前使用同一旧协议。','']
    lines += ['| 模型 | LPIPS主表 | LPIPS历史[0,1] |','|---|---:|---:|']
    legacy_rows=[]
    for row in list(records.values())+ablations:
        value=row.get('visual_metrics') or {}
        legacy_rows.append(dict(model_id=row['id'],LPIPS=value.get('LPIPS'),
                                LPIPS_legacy_0_1=row.get('LPIPS_legacy_0_1'),protocol=lpips['protocol']))
        lines += ['| %s | %s | %s |'%(row['id'],number(value.get('LPIPS'),'LPIPS'),number(row.get('LPIPS_legacy_0_1'),'LPIPS'))]
    csv_write(reports/'lpips_protocol_comparison.csv',['model_id','LPIPS','LPIPS_legacy_0_1','protocol'],legacy_rows)
    lines += ['','[LPIPS协议追溯CSV](%s)。'%(reports/'lpips_protocol_comparison.csv'),'','## 比较范围与数据来源','',
              '- Speedy-Splat的预算与City2相同，但其模型由COLMAP稀疏点训练，并非直接剪枝同一City2模型；统一原生渲染消除了渲染器差异，不能消除初始化和训练过程差异。','- PWSR一栏采用本项目不含3D-DoG的RIR/DPRA/SPS全局剪枝恢复对照。','- LightGaussian 10×保留178,810点，其他10×为178,809点；20×目标均为89,405点。','- 本评估测量统一2D分割器在渲染影像上的下游表现，并非直接训练3D语义高斯。只有两张语义测试图，结果含分割器误差，不能替代多场景或物理实例真值评估。','- B-mIoU评价语义类别轮廓；同类别内部窗框、栅栏分隔或细线不一定产生类别边界，不能由该指标直接判定其连续性。','- 固定六处裁剪沿用实验前位置，没有按本次优势重选。语义区域Recall、图约束满足或全局图像指标都不能单独证明某个具体地物被挽救。','',
              summary['manual_visual_review'].get('conclusion',summary['manual_visual_review'].get('overall_conclusion','目视结论待人工审查。')),'',
              '[模型来源与哈希](%s)；[汇总JSON](%s)；[主表CSV](%s)；[逐类CSV](%s)；[消融CSV](%s)；[区域召回分母CSV](%s)。'%(root/'models.json',reports/'summary.json',reports/'tables.csv',reports/'per_class.csv',reports/'ablations20.csv',reports/'region_recall_details.csv'),'']
    if (ROOT/'COMPARISON_PROTOCOL.md').is_file():
        lines += ['[完整比较协议](%s)。'%(ROOT/'COMPARISON_PROTOCOL.md'),'']
    all_main=all(r.get('visual_status')=='complete' and r.get('semantic_status')=='complete' for r in records.values())
    summary['status']='complete' if all_main and all(r['status']=='complete' for r in ablations) and bundle['status']=='complete' else 'pending'
    if summary['status']!='complete' and not args.allow_pending:
        raise ValueError('Final report has pending metrics')
    csv_write(reports/'tables.csv',['ratio','model_id','method','gaussians']+METRICS,flat)
    csv_write(reports/'region_recall_details.csv',['group','ratio','model_id','method','overall_tp','overall_gt','overall_recall',
              'small_tp','small_gt','small_recall','vehicle_tp','vehicle_gt','vehicle_recall'],regional)
    for row in v3rows+ablations:
        sem=row.get('semantic_metrics')
        if not sem:
            continue
        for cls in range(1,7):
            value=sem['per_class'][str(cls)]
            per_class.append(dict(ratio=row['target_ratio'],model_id=row['id'],method=row['label'],class_id=cls,class_name=CLASSES[cls],
                iou=value['iou'],pixel_recall=value['pixel_recall'],boundary_iou=value['boundary_iou'],
                region_recall=value['regions']['recall'],small_region_recall=value['regions']['small_recall'],
                gt_pixels=value['gt_pixels'],gt_regions=value['regions']['gt'],tp_regions=value['regions']['tp']))
    csv_write(reports/'per_class.csv',['ratio','model_id','method','class_id','class_name','iou','pixel_recall','boundary_iou',
              'region_recall','small_region_recall','gt_pixels','gt_regions','tp_regions'],per_class)
    csv_write(reports/'ablations20.csv',['id','label','actual_count','rounds','recovery_updates','merged_total','restored_total']+
              METRICS+['delta_'+k for k in METRICS],
              [dict({k:r.get(k) for k in ('id','label','actual_count','rounds','recovery_updates','merged_total','restored_total')},
                    **r['metrics'],**{'delta_'+k:r.get('delta_vs_full',{}).get(k) for k in METRICS}) for r in [full]+ablations])
    write(reports/'summary.json',summary)
    (root/'README_统一对比与消融.md').write_text('\n'.join(lines))
    print(json.dumps(dict(status=summary['status'],report=str(root/'README_统一对比与消融.md'),
                         ablations_complete=sum(r['status']=='complete' for r in ablations),
                         semantic_metrics=bundle['path']),ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
