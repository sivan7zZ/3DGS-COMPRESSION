"""Evaluate new SH3 results with the exact frozen City segmenter and ROI protocol.

Unchanged baseline predictions are reused with verified hashes; only the four
new 10x/20x annotated renders need inference. No segmentation training occurs.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np
from PIL import Image

from prepare_semantic_comparison_inputs import prepare
from utils.distill_utils import atomic_json_save,sha256_file

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'output/city2_pup_rir_dpra_20261105'
OLD=ROOT/'output/city2_comparison_20261104'
COMP=OUT/'comparison'
CPU='/home/zhangys/anaconda3/envs/VGGT/bin/python'


def read(path):return json.loads(Path(path).read_text())


def prepare_context():
    registry=copy.deepcopy(read(OLD/'models.json'))
    for ratio in (10,20):
        stage=OUT/('full_%dx'%ratio)/'stage3';run=read(stage/'run_args.json')
        model=next(x for x in registry['models'] if x['id']=='ours_%dx'%ratio)
        model.update(method='PUP + RIR/DPRA (v3.1 protected)',model_dir=str(stage),iteration=25199,
            ply=run['last_ply'],ply_sha256=sha256_file(run['last_ply']),actual_count=run['final_gaussians'],
            comparison_dir=str(stage),common_render_dir=str(stage/'test/ours_25199/renders'),
            common_gt_dir=str(stage/'test/ours_25199/gt'),status='prepared',
            provenance_notes=['PUP Fisher score + original RIR/DPRA; v3.1 protection retained; SH3 before distillation'])
        names=['%05d.png'%i for i in range(9)];gt={};renders={}
        for name in names:
            image=np.asarray(Image.open(Path(model['common_gt_dir'])/name).convert('RGB'))
            gt[name]=hashlib.sha256(image.tobytes()).hexdigest()
            renders[name]=sha256_file(Path(model['common_render_dir'])/name)
        atomic_json_save(dict(ply_sha256=model['ply_sha256'],gt_rgb_sha256=gt,render_sha256=renders,
                             renderer_sha256=sha256_file(ROOT/'render.py'),test_frames=registry['test_frames']),stage/'render_receipt.json')
    COMP.mkdir(parents=True,exist_ok=True);atomic_json_save(registry,COMP/'models.json')
    (COMP/'semantic/model').mkdir(parents=True,exist_ok=True)
    shutil.copy2(OLD/'semantic/dataset_manifest.json',COMP/'semantic/dataset_manifest.json')
    shutil.copy2(OLD/'semantic/model/training_protocol.json',COMP/'semantic/model/training_protocol.json')
    (COMP/'reports/20x_sh3').mkdir(parents=True,exist_ok=True)
    shutil.copy2(OLD/'reports/20x_sh3/crop_manifest.json',COMP/'reports/20x_sh3/crop_manifest.json')
    inputs_path=prepare(COMP)
    inputs=read(inputs_path)
    new=[x for x in inputs['images'] if x['id'].split('/')[0] in ('ours_10x','ours_20x')]
    if len(new)!=4:raise ValueError('Expected exactly four new annotated renders')
    atomic_json_save(dict(images=new),COMP/'semantic/inputs_new.json')


def merge_predictions():
    oldpath=OLD/'semantic/predictions_main/predictions_manifest.json'
    newpath=COMP/'semantic/predictions_new/predictions_manifest.json'
    old,new=read(oldpath),read(newpath)
    if old['checkpoint_sha256']!=new['checkpoint_sha256'] or old['metadata']!=new['metadata']:
        raise ValueError('Frozen segmentation protocol changed')
    replacement={x['id']:x for x in new['predictions']}
    if set(replacement)!={'%s/%s'%(m,f) for m in ('ours_10x','ours_20x') for f in ('frame_000','frame_056')}:
        raise ValueError('Unexpected prediction replacement')
    rows=[]
    for row in old['predictions']:
        actual=replacement.get(row['id'],row)
        if sha256_file(actual['image_path'])!=actual['image_sha256'] or sha256_file(actual['prediction_path'])!=actual['prediction_sha256']:
            raise ValueError('Reused/new segmentation input changed')
        rows.append(actual)
    merged=copy.deepcopy(old);merged.update(predictions=rows,
        inputs_json_sha256=sha256_file(COMP/'semantic/inputs_main.json'),
        merge_provenance=dict(unchanged_baseline_manifest=str(oldpath),unchanged_baseline_manifest_sha256=sha256_file(oldpath),
                              new_manifest=str(newpath),new_manifest_sha256=sha256_file(newpath),new_prediction_count=4,reused_prediction_count=24))
    folder=COMP/'semantic/predictions_main';folder.mkdir(exist_ok=True)
    atomic_json_save(merged,folder/'predictions_manifest.json')


def finish_report():
    old=read(OLD/'preservation_metrics/summary.json')['results']
    new=read(COMP/'preservation_metrics/summary.json')['results']
    lines=['# PUP + RIR/DPRA：地物保护与v3.1.0比较','',
        '相同冻结分割器、相同两张人工标注测试图、相同248个小语义区域与原生ROI。只有4张新渲染图进行分割推理，其余24张既有GT/基线预测经SHA核对后复用，未重新训练评估器。','',
        '| 倍率 | 方法 | Small Recall↑ (%) | Boundary F1↑ (%) | 语义clDice↑ (%) | ROI-PSNR↑ | ROI-LPIPS↓ |',
        '|---|---|---:|---:|---:|---:|---:|']
    records={}
    for ratio in (10,20):
        ident='ours_%dx'%ratio
        for label,values in [('v3.1.0',old),('PUP + RIR/DPRA保护',new)]:
            v=values[ident]
            lines.append('| %d× | %s | %.2f | %.2f | %.2f | %.4f | %.5f |'%(ratio,label,
                100*v['small_object_recall'],100*v['boundary_f1'],100*v['cldice'],v['roi_psnr'],v['roi_lpips']))
        records[str(ratio)]=dict(previous_v310=old[ident],pup_rir_dpra=new[ident])
    lines+=['','City没有电线或井盖独立实例/细线标签；Recall与clDice分别是语义区域和语义骨架代理，不能当作物理电线/井盖测量。只有两张标注测试图，分割器存在预测误差。评分、调度与恢复更新数同时变化，不单独归因于某一模块。','',
        '[与四种基线、原始3DGS、GT的完整新指标表](%s)；[新指标CSV](%s)；[独立核验](%s)。'%(COMP/'preservation_metrics/README_地物保护指标.md',COMP/'preservation_metrics/metrics.csv',COMP/'preservation_metrics/validation.json')]
    (OUT/'README_地物保护指标对比.md').write_text('\n'.join(lines)+'\n')
    atomic_json_save(dict(status='complete',results=records,new_metrics=str(COMP/'preservation_metrics/summary.json'),
        new_metrics_sha256=sha256_file(COMP/'preservation_metrics/summary.json'),frozen_evaluator_sha256='35a50751d2e145f489b4f17016fe7397272f72e373e40913b73fb0ff88643519'),OUT/'preservation_comparison.json')


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--wait',action='store_true');parser.add_argument('--resume',action='store_true');args=parser.parse_args()
    while True:
        state=read(OUT/'experiment_queue.json')
        if state['status']=='complete':break
        if state['status']=='failed':raise RuntimeError('Main compression queue failed: '+str(state.get('error')))
        if not args.wait:raise RuntimeError('Finish the serial compression/metrics queue first')
        time.sleep(20)
    manifest=OUT/'preservation_queue.json'
    if manifest.is_file():
        if not args.resume:raise FileExistsError(str(manifest))
        state=read(manifest);state['status']='running'
    else:state=dict(status='running',completed=[])
    def done(name):state['completed'].append(name);atomic_json_save(state,manifest)
    def run(name,command):
        if name in state['completed']:return
        with (OUT/(name+'.log')).open('a') as log:subprocess.check_call(command,cwd=str(ROOT),stdout=log,stderr=subprocess.STDOUT)
        done(name)
    try:
        if 'prepare_context' not in state['completed']:prepare_context();done('prepare_context')
        run('infer_new',[sys.executable,str(ROOT/'city_semantic.py'),'infer','--checkpoint',str(OLD/'semantic/model/model.pt'),
                         '--inputs_json',str(COMP/'semantic/inputs_new.json'),'--output',str(COMP/'semantic/predictions_new')])
        if 'merge_predictions' not in state['completed']:merge_predictions();done('merge_predictions')
        run('evaluate_semantics',[sys.executable,str(ROOT/'evaluate_semantic_comparison.py'),'--root',str(COMP),
              '--inputs',str(COMP/'semantic/inputs_main.json'),'--predictions',str(COMP/'semantic/predictions_main/predictions_manifest.json'),
              '--output',str(COMP/'semantic/main_metrics.json')])
        run('preservation_prepare',[CPU,str(ROOT/'evaluate_preservation.py'),'--root',str(COMP),'--phase','prepare'])
        run('preservation_lpips',[sys.executable,str(ROOT/'evaluate_preservation.py'),'--root',str(COMP),'--phase','lpips'])
        run('preservation_report',[sys.executable,str(ROOT/'evaluate_preservation.py'),'--root',str(COMP),'--phase','report'])
        run('preservation_validate',[CPU,str(ROOT/'validate_preservation.py'),'--root',str(COMP)])
        finish_report();state['status']='complete'
    except BaseException as error:
        state.update(status='failed',error=str(error));atomic_json_save(state,manifest);raise
    atomic_json_save(state,manifest);print('PUP RIR PRESERVATION EVALUATION COMPLETE',flush=True)


if __name__=='__main__':main()
