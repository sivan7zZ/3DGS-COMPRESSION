"""Serial City 10x/20x PUP + RIR/DPRA experiment, validation and native metrics."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
from PIL import Image

from utils.distill_utils import atomic_json_save,sha256_file

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'output/city2_pup_rir_dpra_20261105'
TRAINING_FILES=['train_pup_rir_compression.py','run_pup_rir_pipeline.py','train_adaptive_compression.py','train_prune.py',
    'utils/pruning_scores.py','utils/rir_dpra_schedule.py','utils/schedule_utils.py','utils/adaptive_schedule.py',
    'utils/adaptive_constraints.py','utils/adaptive_mutation.py','utils/adaptive_metadata.py','utils/structure_losses.py',
    'utils/structure_visibility.py','utils/checkpoint_utils.py','utils/distill_utils.py','utils/instance_roi.py',
    'gaussian_renderer/__init__.py','scene/gaussian_model.py','scene/__init__.py','arguments/__init__.py',
    'distill_train.py','render.py','metrics.py','configs/city_pup_rir_dpra_10x.json','configs/city_pup_rir_dpra_20x.json']


def read(path):return json.loads(Path(path).read_text())


def snapshot():
    folder=OUT/'source_snapshot';manifest=folder/'manifest.json'
    if manifest.is_file():
        record=read(manifest)
        for name,digest in record['files'].items():
            if sha256_file(ROOT/name)!=digest:raise ValueError('Frozen experiment source changed: '+name)
        return
    files=list(TRAINING_FILES)
    for p in (ROOT/'submodules/pup-fisher').rglob('*'):
        if p.is_file() and not any(x in ('build','__pycache__') for x in p.parts) and p.suffix in ('.cu','.h','.cpp','.py','.so'):
            files.append(str(p.relative_to(ROOT)))
    records={}
    for name in files:
        p=ROOT/name;q=folder/name;q.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,q);records[name]=sha256_file(p)
    atomic_json_save(dict(status='frozen',files=records,git_base='v3.1.0',
                         upstream_pup_fisher='eaef2faba21f076867c1db84c3f998e5f673762f'),manifest)


def validate(ratio):
    from validate_adaptive_results import verify_stage
    from utils.adaptive_schedule import survival_quotas,audit_quotas
    from utils.schedule_utils import TrainingScheduler
    from types import SimpleNamespace
    run=OUT/('full_%dx'%ratio);config=read(run/'config.json');stage=read(run/'stage3/run_args.json')
    distill=read(run/'sh2/run_args.json');expected=int(config['pruning_target'])
    if read(run/'pipeline.json')['status']!='complete' or stage['status']!='complete':raise ValueError('Pipeline incomplete')
    if stage['method']!='PUP_RIR_DPRA_v31_protected' or stage['completed_steps']!=10199 or stage['final_iteration']!=25199:
        raise ValueError('Unexpected score/schedule/update count')
    if distill['status']!='complete' or distill['completed_steps']!=5000 or not distill['teacher_unchanged']:
        raise ValueError('Distillation/teacher validation failed')
    sh3,m3=verify_stage(run/'stage3',Path(stage['last_checkpoint']),Path(stage['last_ply']),expected)
    sh2,m2=verify_stage(run/'sh2',run/'sh2/chkpnt40000.pth',run/'sh2/point_cloud/iteration_40000/point_cloud.ply',expected)
    if any(not np.array_equal(m3[k],m2[k]) for k in m3):raise ValueError('SH distillation changed metadata rows')
    shared=ROOT/config['shared_metadata']
    with np.load(str(shared/'adaptive_metadata.npz')) as z:
        initial={k:np.array(z[k]) for k in ('instance_ids','instance_fragility','boundary_mask')}
    amin,bmin,_=survival_quotas(initial,config);audit_quotas(m3,amin,bmin)
    if len(stage['events'])!=5:raise ValueError('Original DPRA must reach budget in five nonzero events')
    scheduler=TrainingScheduler(SimpleNamespace(pruning_stop_iter=25200));current=int(stage['source_count'])
    scheduler.calculate_pruning_target(current,expected);prune_count=2;rate=scheduler.get_pruning_rate(15200,current,1)
    stamp=1.;waiting=0;by_iteration={x['iteration']:x for x in stage['events']};previous=15200
    for check in stage['rir_checks']:
        it=check['iteration'];loss=check['current_loss']
        if it!=previous+500:raise ValueError('RIR check interval differs');
        previous=it;forced=waiting+500>=1500;trigger=loss<.99*stamp or forced
        if check['trigger']!=trigger or check['forced']!=forced or check['pruning_count']!=prune_count:
            raise ValueError('RIR decision differs from original loop')
        goal=max(expected,int(round(current*(1-rate))))
        if check['round_target']!=goal or not np.isclose(check['loss_stamp'],stamp):raise ValueError('DPRA target/loss stamp differs')
        if trigger:
            if current>expected:
                event=by_iteration[it]
                if event['before']!=current or event['after_reallocation']!=goal:raise ValueError('Compression budget mismatch')
            current=goal;stamp=loss;waiting=0
            rate=scheduler.get_pruning_rate(it,current,prune_count);prune_count+=1
        else:waiting+=500
    if current!=expected:raise ValueError('Final schedule budget mismatch')
    for event in stage['events']:
        if not event['selection']['target_reached'] or event['selection']['score_order']!='unchanged_global_ascending_stable':
            raise ValueError('Feasibility order differs')
        if event['quota_audit']['instance_violations'] or event['quota_audit']['boundary_violations']:raise ValueError('Quota violation')
        receipt=event['pup_scores']
        if sha256_file(receipt['path'])!=receipt['sha256']:raise ValueError('Raw PUP score audit changed')
        with np.load(receipt['path']) as z:
            scores=z['scores'];ids=z['row_ids']
            if len(scores)!=event['before'] or len(np.unique(ids))!=len(scores) or np.isnan(scores).any() or np.isposinf(scores).any():
                raise ValueError('Invalid raw PUP ranking')
            if int(np.isneginf(scores).sum())!=receipt['negative_infinity_count']:raise ValueError('Singular score audit differs')
    result=dict(status='complete',ratio=ratio,target=expected,SH3=sh3,SH2=sh2,rir_dpra_matches_original=True,
        quota_violations=0,protected_instances=len(amin),metadata_rows_exact=True,teacher_unchanged=True,
        compression_events=len(stage['events']),event_iterations=[e['iteration'] for e in stage['events']],
        recovery_updates=stage['completed_steps'],distillation_updates=distill['completed_steps'],
        merged_total=sum(e['merged'] for e in stage['events']),restored_total=sum(e['restored'] for e in stage['events']))
    atomic_json_save(result,run/'validation.json');return result


def standard_metrics():
    import torch
    from lpipsPyTorch import LPIPS
    from evaluate_comparison_lpips import standard_input,implementation_provenance,state_dict_sha256
    criterion=LPIPS('vgg','0.1').cuda().eval();criterion.requires_grad_(False)
    meta=implementation_provenance(ROOT,ROOT/'output/city2_comparison_20261104')
    meta['state_dict_sha256']=state_dict_sha256(criterion)
    meta['source_files_sha256'][str(Path(__file__))]=sha256_file(__file__)
    original=next(x for x in read(ROOT/'output/city2_comparison_20261104/models.json')['models'] if x['id']=='original_3dgs')
    names=['%05d.png'%i for i in range(9)];records={}
    for ratio in (10,20):
        for label,folder,iteration in [('SH3',OUT/('full_%dx'%ratio)/'stage3',25199),('SH2',OUT/('full_%dx'%ratio)/'sh2',40000)]:
            method='ours_%d'%iteration;test=folder/'test'/method
            old=read(folder/'results.json')[method];per={};images=[]
            if sorted(p.name for p in (test/'renders').glob('*.png'))!=names:raise ValueError('Expected exactly nine test images')
            for name in names:
                render=np.asarray(Image.open(test/'renders'/name).convert('RGB'));gt=np.asarray(Image.open(test/'gt'/name).convert('RGB'))
                canonical=np.asarray(Image.open(Path(original['common_gt_dir'])/name).convert('RGB'))
                if render.shape!=(894,1592,3) or not np.array_equal(gt,canonical):raise ValueError('Native camera/GT mismatch')
                a=torch.from_numpy(standard_input(render))[None].cuda();b=torch.from_numpy(standard_input(gt))[None].cuda()
                with torch.no_grad():value=float(criterion(a,b).mean())
                if not np.isfinite(value):raise ValueError('Nonfinite LPIPS')
                per[name]=value;images.append(dict(filename=name,render_sha256=sha256_file(test/'renders'/name),gt_sha256=sha256_file(test/'gt'/name)))
            result=dict(status='complete',PSNR=old['PSNR'],SSIM=old['SSIM'],LPIPS=float(np.mean(list(per.values()))),
                LPIPS_legacy_0_1=old['LPIPS'],per_view_standard_lpips=per,criterion=meta,input_range=[-1,1],images=images)
            atomic_json_save(result,folder/'metrics_standard.json');records['%dx_%s'%(ratio,label)]=result
            print('STANDARD PNG',ratio,label,{k:result[k] for k in ('PSNR','SSIM','LPIPS')},flush=True)
    atomic_json_save(dict(status='complete',results=records),OUT/'metrics_standard.json')


def report():
    metrics=read(OUT/'metrics_standard.json')['results'];rows=[]
    for ratio in (10,20):
        val=read(OUT/('full_%dx'%ratio)/'validation.json')
        for stage in ('SH3','SH2'):
            m=metrics['%dx_%s'%(ratio,stage)];native=val[stage]
            rows.append(dict(ratio=ratio,stage=stage,gaussians=val['target'],**{k:m[k] for k in ('PSNR','SSIM','LPIPS')},
                             ply=native['ply'],checkpoint=native['checkpoint'],ply_sha256=native['ply_sha256']))
    lines=['# City：PUP评分 + RIR/DPRA + v3.1保护','',
        '保留实例/边界/拓扑保护、矩匹配合并、预算回填和恢复损失，仅把SPS替换为PUP Fisher评分，把相对自适应轮次替换回原RIR/DPRA。','',
        'PUP为位置与log-scale的6×6 Fisher矩阵，对全部61训练视图、原生分辨率累积，评分为奇异值对数之和；无阻尼、trace近似、频域或区域加分，奇异矩阵的−inf保留并稳定打破并列。RIR每500次检查全训练视图L1+原ROI项，beta=0.99，最长1500次强制剪枝；DPRA沿用原TrainingScheduler。','',
        '两组均从同一1,788,093点原始SH3 PLY初始化，虚拟15001–25199共10199次更新，5次非零剪枝；SH2另用原始未剪枝教师蒸馏5000次。保护信息仅来自训练视图，测试图不进入决策。','',
        '| 倍率 | 阶段 | 高斯数 | PSNR↑ | SSIM↑ | 标准LPIPS↓ |','|---|---|---:|---:|---:|---:|']
    for row in rows:lines.append('| %d× | %s | %d | %.4f | %.5f | %.5f |'%(row['ratio'],row['stage'],row['gaussians'],row['PSNR'],row['SSIM'],row['LPIPS']))
    lines+=['','LPIPS采用VGG0.1标准[-1,1]输入；旧脚本[0,1]值独立留存。所有视觉指标为相同9张原生PNG的测试视图平均值。','',
             '## 蒸馏前与v3.1.0比较','',
             '| 倍率 | 版本 | PSNR↑ | SSIM↑ | 标准LPIPS↓ |','|---|---|---:|---:|---:|']
    old_root=ROOT/'output/city2_comparison_20261104'
    for ratio in (10,20):
        folder=old_root/'renders'/('ours_%dx'%ratio)
        old=next(iter(read(folder/'results.json').values()))
        old_lpips=read(folder/'comparison_lpips.json')['LPIPS']
        new=metrics['%dx_SH3'%ratio]
        lines+=['| %d× | v3.1.0 | %.4f | %.5f | %.5f |'%(ratio,old['PSNR'],old['SSIM'],old_lpips),
                '| %d× | PUP + RIR/DPRA保护 | %.4f | %.5f | %.5f |'%(ratio,new['PSNR'],new['SSIM'],new['LPIPS'])]
    from finalize_city_benchmark import native_crops
    registry=read(old_root/'models.json');records={x['id']:dict(x,visual_status='complete') for x in registry['models']}
    for ratio in (10,20):
        stage=OUT/('full_%dx'%ratio)/'stage3';records['ours_%dx'%ratio].update(
            common_render_dir=str(stage/'test/ours_25199/renders'),common_gt_dir=str(stage/'test/ours_25199/gt'))
        image=native_crops(OUT,ratio,records,registry['test_frames'],False)
        lines+=['','[%d×固定六处：GT、3DGS、四种基线与本次版本](%s)'%(ratio,image)]
    lines+=['','## 模型与核验','']
    for row in rows:lines+=['- %d× %s：[PLY](%s)；[checkpoint](%s)。'%(row['ratio'],row['stage'],row['ply'],row['checkpoint'])]
    for ratio in (10,20):
        val=read(OUT/('full_%dx'%ratio)/'validation.json')
        lines+=['','%d×剪枝发生于%s；累计合并%d次，回填%d次；实例和边界配额零违规。PLY/checkpoint原生参数逐值一致，SH3→SH2元数据行序一致，教师未变。'%(ratio,','.join(map(str,val['event_iterations'])),val['merged_total'],val['restored_total'])]
    lines+=['','评分和调度同时改变，且相较v3.1.0恢复更新数不同；不能将差值单独归因于PUP评分。实例配额及图连通约束满足不等于已经证明所有地物在渲染中被挽救。','',
        'PUP来源：[作者训练实现](https://github.com/j-alex-hanson/gaussian-splatting-pup)、[Fisher CUDA固定版本](https://github.com/j-alex-hanson/rasterization_and_pup_fisher/tree/eaef2faba21f076867c1db84c3f998e5f673762f)。','',
        '复现：`bash scripts/run.sh run_pup_rir_experiments.py --resume`。输出根目录包含配置、冻结源码、每次原始PUP评分、RIR判断、元数据、恢复状态和验证回执。']
    (OUT/'README_PUP_RIR_DPRA结果.md').write_text('\n'.join(lines)+'\n')
    atomic_json_save(dict(status='complete',rows=rows,method='PUP_RIR_DPRA_v31_protected'),OUT/'summary.json')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--resume',action='store_true');args=p.parse_args()
    OUT.mkdir(parents=True,exist_ok=True);snapshot();manifest=OUT/'experiment_queue.json'
    if manifest.is_file():
        if not args.resume:raise FileExistsError(str(manifest))
        state=read(manifest);state['status']='running'
    else:state=dict(status='running',completed=[],tasks=[])
    atomic_json_save(state,manifest)
    def completed(name):state['completed'].append(name);atomic_json_save(state,manifest)
    try:
        for ratio in (10,20):
            snapshot();name='pipeline_%dx'%ratio
            if name not in state['completed']:
                command=[sys.executable,str(ROOT/'run_pup_rir_pipeline.py'),'--config',str(ROOT/('configs/city_pup_rir_dpra_%dx.json'%ratio))]
                if args.resume:command+=['--resume']
                state['tasks'].append(dict(name=name,command=command));atomic_json_save(state,manifest)
                with (OUT/(name+'.log')).open('a') as stream:
                    subprocess.check_call(command,cwd=str(ROOT),stdout=stream,stderr=subprocess.STDOUT)
                completed(name)
            if 'validate_%dx'%ratio not in state['completed']:validate(ratio);completed('validate_%dx'%ratio)
        if 'standard_metrics' not in state['completed']:standard_metrics();completed('standard_metrics')
        report();state['status']='complete';snapshot()
    except BaseException as error:
        state.update(status='failed',error=str(error));atomic_json_save(state,manifest);raise
    atomic_json_save(state,manifest)
    print('PUP RIR/DPRA EXPERIMENTS COMPLETE',flush=True)


if __name__=='__main__':main()
