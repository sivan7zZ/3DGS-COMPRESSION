"""Preservation benchmark on existing City SH3 renders; no training or ablations.

CPU prepare: VGGT Python (existing scikit-image); GPU lpips: scripts/run.sh.
The two phases share immutable native ROI coordinates and input SHA receipts.
"""
import argparse
import csv
import hashlib
import json
import platform
from pathlib import Path
import sys

import numpy as np
from PIL import Image,ImageDraw,ImageFont

from evaluate_comparison_lpips import read,sha256_file,json_signature,atomic_json,standard_input
from utils.preservation_metrics import (evaluate_masks,hard_skeleton,mean_present,
                                       roi_box,roi_psnr,small_regions)

ROOT=Path(__file__).resolve().parent
DEFAULT=ROOT/'output/city2_comparison_20261104'
NAMES=['Building','Tree','Road','Vehicle','Grass','Ground']
PHASE_PROTOCOL=dict(semantic_frames=['frame_000','frame_056'],min_region_area=16,
    max_small_fraction=.01,iou_threshold=.5,connectivity=8,boundary_tolerance_px=2,
    boundary_contour='one-pixel inner contour; outside image foreground; no artificial frame edge',
    boundary_valid_domain='GT nonvoid eroded by tolerance+1, outside image valid; Euclidean matching disk',
    skeleton='scikit-image hard skeletonize(method=zhang), zero-pad one pixel; same valid GT domain',
    cldice='dataset aggregate per-class skeleton overlaps then six-class macro mean; one-pixel void margin',
    roi_context_padding_px=8,roi_min_native_size=32,roi_resize=False,
    roi_aggregation='equal ROI mean; PSNR per crop then arithmetic mean; LPIPS VGG0.1 [-1,1]',
    perfect_psnr_cap_db=120,additional_annotations=False,ablations_evaluated=False)


def rgb(path):
    with Image.open(path) as im:value=np.asarray(im.convert('RGB'))
    if value.shape!=(894,1592,3):raise ValueError('Non-native image: '+str(path))
    return value


def array_sha(value):return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def prepare(root,out):
    import cv2
    import skimage
    data=read(root/'semantic/dataset_manifest.json')
    inputs=read(root/'semantic/inputs_main.json')
    predictions_path=root/'semantic/predictions_main/predictions_manifest.json'
    predictions=read(predictions_path)
    if data['splits']['test']!=PHASE_PROTOCOL['semantic_frames']:raise ValueError('Different labeled test views')
    if set(data['splits']['refit_train'])&set(data['splits']['test']):raise ValueError('Test label leakage')
    if predictions['checkpoint_sha256']!=sha256_file(predictions['checkpoint']):raise ValueError('Frozen weights changed')
    if predictions['inputs_json_sha256']!=sha256_file(root/'semantic/inputs_main.json'):raise ValueError('Different prediction inputs')
    if predictions['metadata']['data_manifest_sha256']!=sha256_file(root/'semantic/dataset_manifest.json'):raise ValueError('Different label registration')
    if (set(predictions['metadata']['train_frames'])!=set(data['splits']['refit_train']) or
        predictions['metadata']['test_frames_used_for_training_or_selection']):raise ValueError('Invalid frozen evaluator split')
    byid={x['id']:x for x in predictions['predictions']}
    if len(byid)!=len(predictions['predictions']):raise ValueError('Duplicate predicted IDs')
    frames=data['splits']['test'];gtrows={x['frame']:x for x in data['frames']}
    targets=[];gtskeletons=[];maps={};rois=[];refs={}
    for frame in frames:
        source=gtrows[frame]
        if sha256_file(source['label_path'])!=source['label_sha256']:raise ValueError('GT labels changed')
        label=np.asarray(Image.open(source['label_path']).convert('L'))
        targets.append(label)
        gtskeletons.append({c:hard_skeleton(label==c) for c in range(1,7)})
        regions,mapping=small_regions(label)
        maps[frame]=mapping
        refs[frame]=dict(path=source['rgb_path'],sha256=source['rgb_sha256'],pixel_sha256=array_sha(rgb(source['rgb_path'])))
        if sha256_file(source['rgb_path'])!=source['rgb_sha256']:raise ValueError('GT RGB changed')
        for roi in regions:
            rois.append(dict(roi,id=frame+'/small/%03d'%roi['region_id'],frame=frame,group='semantic_small'))
    if len(rois)!=248:raise ValueError('Expected the same 248 GT small regions')
    # Reuse six coordinates fixed before this experiment; never rank locations by improvement.
    crop_manifest=read(root/'reports/20x_sh3/crop_manifest.json')
    registry=read(root/'models.json');tests=registry['test_frames']
    original=next(x for x in registry['models'] if x['id']=='original_3dgs')
    for k,case in enumerate(crop_manifest['cases']):
        frame=case['frame'];filename='%05d.png'%tests.index(frame)
        path=Path(original['common_gt_dir'])/filename
        image=rgb(path)
        if frame in refs and refs[frame]['pixel_sha256']!=array_sha(image):raise ValueError('RGB frame mapping differs')
        refs.setdefault(frame,dict(path=str(path),sha256=sha256_file(path),pixel_sha256=array_sha(image)))
        rois.append(dict(id='fixed/%02d'%k,frame=frame,group='fixed_structure',
                         bbox_xyxy=case['bbox_xyxy'],crop_xyxy=roi_box(case['bbox_xyxy'],1592,894),
                         category=case['category'],description=case['description']))
    np.savez_compressed(str(out/'roi_region_ids.npz'),**maps)
    models={};result={}
    registry_byid={x['id']:x for x in registry['models']}
    reference_images={f:rgb(v['path']) for f,v in refs.items()}
    for group,ids in inputs['groups'].items():
        if group.startswith('ablation_'):continue
        if [x.split('/')[-1] for x in ids]!=frames:raise ValueError('Prediction order differs')
        predicted=[];images={};pred_receipts=[]
        for ident in ids:
            row=byid[ident];frame=ident.split('/')[-1]
            if sha256_file(row['prediction_path'])!=row['prediction_sha256'] or sha256_file(row['image_path'])!=row['image_sha256']:
                raise ValueError('Prediction/input image changed')
            predicted.append(np.asarray(Image.open(row['prediction_path']).convert('L')))
            images[frame]=dict(path=row['image_path'],sha256=row['image_sha256'])
            pred_receipts.append(dict(id=ident,prediction_path=row['prediction_path'],prediction_sha256=row['prediction_sha256']))
        for frame in refs:
            if frame in images:continue
            if group=='GT':images[frame]=dict(path=refs[frame]['path'],sha256=refs[frame]['sha256'])
            else:
                model=registry_byid[group];filename='%05d.png'%tests.index(frame)
                path=Path(model['common_render_dir'])/filename
                gt=Path(model['common_gt_dir'])/filename
                if array_sha(rgb(gt))!=refs[frame]['pixel_sha256']:raise ValueError('Canonical GT differs')
                receipt=read(Path(model['comparison_dir'])/'render_receipt.json')
                digest=sha256_file(path)
                if receipt['render_sha256'][filename]!=digest:raise ValueError('Render receipt differs')
                images[frame]=dict(path=str(path),sha256=digest)
        mask_metrics=evaluate_masks(predicted,targets,gt_skeletons=gtskeletons)
        if mask_metrics['small_gt']!=248 or mask_metrics['vehicle_small_gt']!=59:raise ValueError('Changed small region denominator')
        image_cache={f:rgb(v['path']) for f,v in images.items()}
        quality=[]
        for roi in rois:
            frame=roi['frame'];gt,pred=reference_images[frame],image_cache[frame]
            q=roi_psnr(gt,pred,roi['crop_xyxy'])
            if roi['group']=='semantic_small':
                q['masked_object_psnr']=roi_psnr(gt,pred,roi['crop_xyxy'],maps[frame]==roi['region_id'])['psnr']
            quality.append(dict(id=roi['id'],group=roi['group'],**q))
        pergroup={g:dict(count=sum(q['group']==g for q in quality),
                        roi_psnr=mean_present(q['psnr'] for q in quality if q['group']==g),
                        masked_object_psnr=mean_present(q.get('masked_object_psnr') for q in quality if q['group']==g))
                  for g in ('semantic_small','fixed_structure')}
        models[group]=dict(images=images,predictions=pred_receipts)
        result[group]=dict(mask_metrics=mask_metrics,roi_quality=pergroup,per_roi=quality)
        print(group,'SmallRecall',mask_metrics['small_object_recall'],'BF1',mask_metrics['boundary_f1'],'clDice',mask_metrics['cldice'],flush=True)
    if len(models)!=14 or 'GT' not in models:raise ValueError('Expected 13 models and GT; no ablations')
    reference_control=evaluate_masks(targets,targets,gt_skeletons=gtskeletons)
    for key in ('small_object_recall','boundary_f1','cldice'):
        if reference_control[key]!=1:raise ValueError('Perfect-label control failed: '+key)
    provenance=dict(dataset_manifest_sha256=sha256_file(root/'semantic/dataset_manifest.json'),
                    predictions_manifest_sha256=sha256_file(predictions_path),checkpoint_sha256=predictions['checkpoint_sha256'],
                    prediction_manifest=str(predictions_path),python=sys.version,numpy=np.__version__,
                    opencv=cv2.__version__,skimage=skimage.__version__,
                    source_sha256={str(p):sha256_file(p) for p in [Path(__file__),ROOT/'utils/preservation_metrics.py',ROOT/'utils/semantic_comparison_metrics.py']})
    protocol=dict(PHASE_PROTOCOL)
    payload=dict(status='complete',protocol=protocol,provenance=provenance,references=refs,models=models,rois=rois,
                 roi_region_ids_path=str(out/'roi_region_ids.npz'),roi_region_ids_sha256=sha256_file(out/'roi_region_ids.npz'))
    payload['input_signature']=json_signature(payload)
    atomic_json(payload,out/'inputs.json')
    atomic_json(dict(status='complete',protocol=protocol,provenance=provenance,input_signature=payload['input_signature'],
                     perfect_label_control={k:reference_control[k] for k in ('small_object_recall','boundary_f1','cldice')},
                     results=result),out/'mask_and_psnr.json')
    make_roi_overview(out,payload)


def make_roi_overview(out,inputs):
    palette=[(70,70,70),(0,180,0),(200,70,200),(0,100,255),(150,170,0),(230,130,0)]
    sheet=Image.new('RGB',(1592,2*(894+28)), 'white');draw=ImageDraw.Draw(sheet)
    for i,frame in enumerate(inputs['protocol']['semantic_frames']):
        y=i*(894+28)
        image=Image.open(inputs['references'][frame]['path']).convert('RGB')
        d=ImageDraw.Draw(image)
        for r in inputs['rois']:
            if r['group']=='semantic_small' and r['frame']==frame:
                d.rectangle(tuple(r['bbox_xyxy']),outline=palette[r['class_id']-1],width=1)
        sheet.paste(image,(0,y+28));draw.text((8,y+7),frame+' / all GT semantic small regions / fixed across methods',fill='black')
    sheet.save(out/'roi_overview.png')


def validate_inputs(inputs):
    signature=inputs['input_signature'];material=dict(inputs);del material['input_signature']
    if signature!=json_signature(material):raise ValueError('ROI input manifest changed')
    if sha256_file(inputs['roi_region_ids_path'])!=inputs['roi_region_ids_sha256']:raise ValueError('ROI regions changed')
    for ref in inputs['references'].values():
        if sha256_file(ref['path'])!=ref['sha256']:raise ValueError('GT image changed')
    for model in inputs['models'].values():
        for image in model['images'].values():
            if sha256_file(image['path'])!=image['sha256']:raise ValueError('Rendered image changed')


def lpips(root,out,device):
    import torch
    from lpipsPyTorch import LPIPS
    from evaluate_comparison_lpips import implementation_provenance,state_dict_sha256
    inputs=read(out/'inputs.json');validate_inputs(inputs)
    criterion_meta=implementation_provenance(ROOT,root)
    criterion_meta['spatial_processing']='fixed native ROI context crop; >=32 pixels; no resize or masking'
    criterion_meta['source_files_sha256'][str(Path(__file__))]=sha256_file(__file__)
    criterion_meta['source_files_sha256'][str(ROOT/'utils/preservation_metrics.py')]=sha256_file(ROOT/'utils/preservation_metrics.py')
    if device=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA unavailable')
    net=LPIPS('vgg','0.1').to(device).eval();net.requires_grad_(False)
    criterion_meta['state_dict_sha256']=state_dict_sha256(net)
    references={f:rgb(x['path']) for f,x in inputs['references'].items()}
    def evaluate_crop(a,b,box):
        x0,y0,x1,y1=box
        if min(x1-x0,y1-y0)<32:raise ValueError('Native ROI smaller than fixed minimum')
        x=torch.from_numpy(standard_input(a[y0:y1,x0:x1])).unsqueeze(0).to(device)
        y=torch.from_numpy(standard_input(b[y0:y1,x0:x1])).unsqueeze(0).to(device)
        with torch.no_grad():value=float(net(x,y).mean().item())
        if not np.isfinite(value):raise ValueError('Nonfinite ROI LPIPS')
        return value
    first=inputs['rois'][0]
    control=evaluate_crop(references[first['frame']],references[first['frame']],first['crop_xyxy'])
    if abs(control)>1e-10:raise ValueError('GT crop self-control failed')
    results={};(out/'lpips_models').mkdir(exist_ok=True)
    atomic_json(dict(status='running',input_signature=inputs['input_signature']),out/'roi_lpips.json')
    for ident,model in inputs['models'].items():
        cache_path=out/'lpips_models'/(ident+'.json')
        signature=json_signature(dict(input_signature=inputs['input_signature'],criterion=criterion_meta,model_id=ident))
        cache=read(cache_path) if cache_path.is_file() else None
        if cache and cache.get('cache_signature')==signature and cache.get('status')=='complete':
            results[ident]=cache;continue
        images={f:rgb(x['path']) for f,x in model['images'].items()}
        rows=[]
        for roi in inputs['rois']:
            value=evaluate_crop(images[roi['frame']],references[roi['frame']],roi['crop_xyxy'])
            rows.append(dict(id=roi['id'],group=roi['group'],lpips=value))
        if ident=='GT' and any(abs(x['lpips'])>1e-10 for x in rows):raise ValueError('GT crop pixels differ')
        value=dict(status='complete',model_id=ident,cache_signature=signature,input_signature=inputs['input_signature'],
                   criterion=criterion_meta,self_control=control,per_roi=rows,
                   groups={g:dict(count=sum(x['group']==g for x in rows),
                       roi_lpips=mean_present(x['lpips'] for x in rows if x['group']==g)) for g in ('semantic_small','fixed_structure')})
        atomic_json(value,cache_path);results[ident]=value
        print(ident,{k:v['roi_lpips'] for k,v in value['groups'].items()},flush=True)
    validate_inputs(inputs)
    atomic_json(dict(status='complete',input_signature=inputs['input_signature'],criterion=criterion_meta,
                     self_control=control,results=results),out/'roi_lpips.json')


def report(root,out):
    inputs=read(out/'inputs.json');validate_inputs(inputs)
    mask=read(out/'mask_and_psnr.json');perceptual=read(out/'roi_lpips.json')
    if any(x['status']!='complete' or x['input_signature']!=inputs['input_signature'] for x in (mask,perceptual)):
        raise ValueError('Incomplete or mismatched preservation measurements')
    results={}
    roi_lookup={x['id']:x for x in inputs['rois']}
    roirows=[];classrows=[]
    for ident,value in mask['results'].items():
        m=value['mask_metrics'];q=value['roi_quality'];l=perceptual['results'][ident]
        if set(x['id'] for x in l['per_roi'])!=set(roi_lookup):raise ValueError('ROI LPIPS support differs')
        results[ident]=dict(small_object_recall=m['small_object_recall'],small_tp=m['small_tp'],small_gt=m['small_gt'],
              vehicle_small_recall=m['vehicle_small_recall'],vehicle_small_tp=m['vehicle_small_tp'],vehicle_small_gt=m['vehicle_small_gt'],
              boundary_f1=m['boundary_f1'],cldice=m['cldice'],roi_psnr=q['semantic_small']['roi_psnr'],
              roi_lpips=l['groups']['semantic_small']['roi_lpips'],masked_object_psnr=q['semantic_small']['masked_object_psnr'],
              fixed_roi_psnr=q['fixed_structure']['roi_psnr'],fixed_roi_lpips=l['groups']['fixed_structure']['roi_lpips'])
        if ident=='GT':
            for k in ('roi_psnr','roi_lpips','masked_object_psnr','fixed_roi_psnr','fixed_roi_lpips'):results[ident][k]=None
        lp={x['id']:x['lpips'] for x in l['per_roi']}
        for row in value['per_roi']:
            roi=roi_lookup[row['id']]
            roirows.append(dict(model_id=ident,roi_id=row['id'],frame=roi['frame'],group=roi['group'],class_id=roi.get('class_id'),
                                area=roi.get('area'),crop_xyxy=json.dumps(roi['crop_xyxy']),psnr=row['psnr'],
                                masked_object_psnr=row.get('masked_object_psnr'),lpips=lp[row['id']]))
        for c,v in m['per_class'].items():
            r=v['regions'];s=v['skeleton'];b=v['boundary']
            classrows.append(dict(model_id=ident,class_id=c,class_name=NAMES[int(c)-1],boundary_f1=b['f1'],
                     boundary_precision=b['precision'],boundary_recall=b['recall'],cldice=s['cldice'],
                     topology_precision=s['topology_precision'],topology_recall=s['topology_recall'],
                     small_tp=r['small_tp'],small_gt=r['small_gt'],small_recall=r['small_tp']/r['small_gt'] if r['small_gt'] else None))
    def csvfile(name,rows):
        with (out/name).open('w',newline='') as f:
            w=csv.DictWriter(f,list(rows[0]));w.writeheader();w.writerows(rows)
    csvfile('metrics.csv',[dict(model_id=k,**v) for k,v in results.items()])
    csvfile('per_roi.csv',roirows);csvfile('per_class.csv',classrows)
    text=['# City 地物保护指标（现有标签）','',
        '本次仅评估现有蒸馏前SH3结果，不启动训练或消融。全部使用相同原生渲染、相同ROI和同一冻结分割器。','',
        '**标签覆盖限制：City仅有六类语义标签，没有电线、井盖的独立实例/细线标注。下述Small-object Recall是小语义区域代理，clDice是语义掩膜骨架指标，不能直接解释为电线/井盖召回或连通性。** 只有frame_000、frame_056两张测试图有标签。','',
        '- Small-object Recall：同类8连通GT区域，面积16像素至整图1%，IoU≥0.5一对一最大匹配；248个区域，另列59个Vehicle小区域。','- Boundary F1：1像素内轮廓，2像素欧氏容差，双向精确率/召回率调和平均；两帧累计各类匹配计数后六类宏平均。GT void附近3像素不计入，不计人工图像外框边。','- clDice：硬Zhang骨架与另一方同类别掩膜交叠，计算拓扑精确率/召回率调和平均；两帧先累计各类计数，再宏平均。void附近1像素不计入。不是单独的连通分量数或物理电线断点指标。','- ROI-PSNR / ROI-LPIPS：248个GT小区域的包围框，固定加8像素上下文、边长至少32像素、边缘平移裁剪，全程不缩放、不将背景涂黑。逐ROI计算后等权平均；LPIPS为VGG0.1标准[-1,1]输入。另存只在目标像素计算的masked-object PSNR。','- GT RGB行经过同一分割器，语义指标为实测参考，非100%上界；视觉自比较用“—”，原始自比较检查保存在JSON。','']
    def fmt(v,percentage=False):return '—' if v is None else ('%.2f'%(100*v) if percentage else '%.5f'%v)
    for ratio in (10,20):
        text+=['## %d×'%ratio,'','| 方法 | Small Recall↑ (%) | Vehicle Small Recall↑ (%) | Boundary F1↑ (%) | clDice↑ (%) | ROI-PSNR↑ (dB) | ROI-LPIPS↓ |','|---|---:|---:|---:|---:|---:|---:|']
        for label,ident in [('GT RGB','GT'),('原始3DGS','original_3dgs'),('LightGaussian','lightgaussian_%dx'%ratio),
                ('PUP-3DGS','pup_%dx'%ratio),('Speedy-Splat*','speedy_%dx'%ratio),('PWSR','pwsr_%dx'%ratio),('本次方法','ours_%dx'%ratio)]:
            r=results[ident]
            text+=['| '+label+' | '+' | '.join(fmt(r[k],k in ('small_object_recall','vehicle_small_recall','boundary_f1','cldice'))
                         for k in ('small_object_recall','vehicle_small_recall','boundary_f1','cldice','roi_psnr','roi_lpips'))+' |']
        text+=['','Small Recall分母248，Vehicle Small Recall分母59；精确TP/GT见CSV。','',
               '固定六处地物ROI（此前坐标，未按本次优势重选；含车辆、屋顶轮廓、窗框和围栏；仅RGB局部质量，不当作电线/井盖标注）：','',
               '| 方法 | 固定ROI-PSNR↑ | 固定ROI-LPIPS↓ |','|---|---:|---:|']
        for label,prefix in [('原始3DGS','original_3dgs'),('LightGaussian','lightgaussian'),('PUP-3DGS','pup'),('Speedy-Splat*','speedy'),('PWSR','pwsr'),('本次方法','ours')]:
            ident=prefix if prefix=='original_3dgs' else prefix+'_%dx'%ratio;r=results[ident]
            text+=['| %s | %s | %s |'%(label,fmt(r['fixed_roi_psnr']),fmt(r['fixed_roi_lpips']))]
        text+=['']
    text+=['## 范围与复现','',
        'Speedy-Splat由COLMAP稀疏点训练，其初始化不同于city2直接压缩；LG10实际178810点，其余10×178809点、20×89405点。单场景、两张标注测试图和分割器误差限制结论，不做统计显著性或“已解决所有小目标消失”结论。','',
        'Boundary F1参考[DAVIS评测](https://github.com/davisvideochallenge/davis2017-evaluation/blob/master/davis2017/metrics.py)的双向容差匹配，轮廓、void及类别聚合适配如上；clDice参考[作者实现](https://github.com/jocpae/clDice/blob/master/cldice_metric/cldice.py)，本次采用其硬骨架定义。','',
        'CPU阶段使用已有VGGT环境的scikit-image；LPIPS使用项目原环境。没有重新训练分割器或压缩模型。','',
        '```bash','/home/zhangys/anaconda3/envs/VGGT/bin/python evaluate_preservation.py --phase prepare',
        'bash scripts/run.sh evaluate_preservation.py --phase lpips',
        'bash scripts/run.sh evaluate_preservation.py --phase report','```','',
        '[总表CSV](%s)；[逐ROI明细](%s)；[逐类明细](%s)；[汇总JSON](%s)；[固定ROI与哈希](%s)。'%(out/'metrics.csv',out/'per_roi.csv',out/'per_class.csv',out/'summary.json',out/'inputs.json'),'']
    summary=dict(status='complete',protocol=inputs['protocol'],input_signature=inputs['input_signature'],results=results,
                 source=dict(mask_and_psnr_sha256=sha256_file(out/'mask_and_psnr.json'),roi_lpips_sha256=sha256_file(out/'roi_lpips.json')),
                 limitations=['semantic region proxies, not physical instances','semantic skeletons do not annotate electric wires or manholes',
                              'only two labeled held-out views','fixed common segmenter has prediction error','no ablations in this evaluation'])
    atomic_json(summary,out/'summary.json');(out/'README_地物保护指标.md').write_text('\n'.join(text))
    print(json.dumps(dict(status='complete',report=str(out/'README_地物保护指标.md'),models=len(results),small_regions=248),ensure_ascii=False))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=DEFAULT);parser.add_argument('--output',type=Path)
    parser.add_argument('--phase',choices=['prepare','lpips','report'],required=True)
    parser.add_argument('--device',choices=['cuda','cpu'],default='cuda')
    args=parser.parse_args();root=args.root.resolve();out=(args.output or root/'preservation_metrics').resolve();out.mkdir(parents=True,exist_ok=True)
    if args.phase=='prepare':prepare(root,out)
    elif args.phase=='lpips':lpips(root,out,args.device)
    else:report(root,out)


if __name__=='__main__':main()
