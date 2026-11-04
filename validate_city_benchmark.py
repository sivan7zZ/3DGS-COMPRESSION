"""CPU audit of experiment provenance, view alignment and reported metric values."""
import argparse
import json
import math
from pathlib import Path
import hashlib
import numpy as np
from PIL import Image
from utils.distill_utils import sha256_file,atomic_json_save

ROOT=Path(__file__).resolve().parent


def read(p):return json.loads(Path(p).read_text())
def rgb_hash(p):
    with Image.open(p) as im:
        value=np.asarray(im.convert('RGB'))
    if value.shape!=(894,1592,3):raise ValueError('Non-native RGB dimensions')
    return hashlib.sha256(value.tobytes()).hexdigest()
def close(a,b):
    if a is None or b is None:
        if a is not b:raise ValueError('Unexpected missing value')
    elif not math.isfinite(float(a)) or not math.isclose(float(a),float(b),rel_tol=1e-10,abs_tol=1e-10):
        raise ValueError('Reported metric differs from source: %s %s'%(a,b))


def check_visual(row,folder,key,standard=None):
    """Audit the untouched legacy files and, separately, the primary LPIPS receipt."""
    folder=Path(folder)
    metrics=read(folder/'results.json')[key]
    per_view=read(folder/'per_view.json')[key]
    names={'%05d.png'%i for i in range(9)}
    for k in ('PSNR','SSIM','LPIPS'):
        if set(per_view[k])!=names:raise ValueError('Visual metric view support differs')
        close(metrics[k],sum(per_view[k].values())/9)
        if k!='LPIPS' or standard is None:close(metrics[k],row['metrics'][k])
    if standard is None:return
    entry=standard['models'][row['id']]
    path=Path(entry['path'])
    if sha256_file(path)!=entry['sha256']:raise ValueError('Corrected LPIPS receipt changed')
    receipt=read(path)
    if (receipt['status']!='complete' or receipt['model_id']!=row['id'] or
        receipt['input_range']!=[-1,1] or receipt['criterion']!=standard['criterion'] or
        receipt['cache_signature']!=entry['cache_signature']):raise ValueError('Corrected LPIPS protocol differs')
    if set(receipt['per_view'])!=names:raise ValueError('Corrected LPIPS test support differs')
    close(receipt['LPIPS'],sum(receipt['per_view'].values())/9)
    close(receipt['LPIPS'],entry['LPIPS']);close(receipt['LPIPS'],row['metrics']['LPIPS'])
    close(metrics['LPIPS'],row['LPIPS_legacy_0_1']);close(metrics['LPIPS'],receipt['legacy']['LPIPS'])
    legacy=receipt['legacy']
    if (legacy['method_key']!=key or legacy['input_range']!=[0,1] or
        sha256_file(folder/'results.json')!=legacy['results_sha256'] or
        sha256_file(folder/'per_view.json')!=legacy['per_view_sha256']):raise ValueError('Legacy LPIPS provenance differs')
    control=receipt['self_control']
    if control['status']!='passed' or abs(control['value'])>1e-10:raise ValueError('GT LPIPS control failed')
    if len(receipt['images'])!=9 or {x['filename'] for x in receipt['images']}!=names:raise ValueError('LPIPS PNG support differs')
    for image in receipt['images']:
        for part in ('render','gt'):
            expected=Path(row['common_render_dir'] if part=='render' else row['common_gt_dir'])/image['filename']
            if (Path(image[part+'_path'])!=expected or sha256_file(expected)!=image[part+'_sha256']):
                raise ValueError('Corrected LPIPS image changed')


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',type=Path,default=ROOT/'output/city2_comparison_20261104');parser.add_argument('--main_only',action='store_true');args=parser.parse_args();root=args.root
    summary=read(root/'reports/summary.json');models=read(root/'models.json');render_status=read(root/'render_pipeline.json')
    standard_path=root/'lpips_standard_manifest.json'
    standard=read(standard_path) if standard_path.is_file() else None
    if standard is not None:
        if standard['status']!='complete' or standard['input_range']!=[-1,1] or standard['n_test_views']!=9:
            raise ValueError('Standard LPIPS evaluation incomplete')
        if summary['lpips_provenance']['sha256']!=sha256_file(standard_path):raise ValueError('Report uses a different LPIPS manifest')
        for group in ('source_files_sha256','weight_files_sha256'):
            for path,digest in standard['criterion'][group].items():
                if sha256_file(path)!=digest:raise ValueError('LPIPS implementation/weights changed')
    elif not args.main_only:raise ValueError('Final comparison requires uniform standard-range LPIPS')
    if render_status['status']!='complete' or not render_status['visual_metrics_complete']:raise ValueError('Common renders incomplete')
    if summary['visual_test_frames']!=models['test_frames'] or summary['n_visual_views']!=9 or summary['n_semantic_views']!=2:raise ValueError('Evaluation support mismatch')
    data=read(root/'semantic/dataset_manifest.json');protocol=read(root/'semantic/model/training_protocol.json');checkpoint=Path(protocol['checkpoint'])
    if protocol['status']!='complete' or protocol['checkpoint_sha256']!=sha256_file(checkpoint):raise ValueError('Semantic model changed')
    if not protocol['refit_all8'] or set(protocol['metadata']['train_frames'])!=set(data['splits']['refit_train']):raise ValueError('Unexpected semantic fitting protocol')
    if set(data['splits']['test'])&set(data['splits']['refit_train']):raise ValueError('Semantic test label leakage')
    if protocol['metadata']['test_frames_used_for_training_or_selection']:raise ValueError('Semantic test labels fitted')
    if data['splits']['test']!=['frame_000','frame_056']:raise ValueError('Annotated held-out support differs')
    for frame in data['frames']:
        if sha256_file(frame['label_path'])!=frame['label_sha256'] or sha256_file(frame['rgb_path'])!=frame['rgb_sha256']:raise ValueError('Labels/RGB changed')
    sem_file=root/('semantic/main_metrics.json' if args.main_only else 'semantic/all_metrics.json');sem=read(sem_file)
    if sem['status']!='complete' or sem['checkpoint_sha256']!=protocol['checkpoint_sha256']:raise ValueError('Different semantic evaluator used')
    pred_path=root/'semantic'/('predictions_main' if args.main_only else 'predictions_all')/'predictions_manifest.json';pred=read(pred_path)
    if pred['checkpoint_sha256']!=protocol['checkpoint_sha256']:raise ValueError('Prediction weight mismatch')
    for item in pred['predictions']:
        if sha256_file(item['image_path'])!=item['image_sha256'] or sha256_file(item['prediction_path'])!=item['prediction_sha256']:raise ValueError('Prediction/source file changed')
        with Image.open(item['prediction_path']) as im:value=np.asarray(im)
        if value.shape!=(894,1592) or value.min()<1 or value.max()>6:raise ValueError('Invalid semantic prediction IDs')
    names=['%05d.png'%i for i in range(9)]
    original=next(x for x in models['models'] if x['id']=='original_3dgs')
    canonical={f:rgb_hash(Path(original['common_gt_dir'])/f) for f in names}
    def check_sem(ident,row):
        actual=sem['results'][ident]['metrics']
        for label,key in [('mIoU','miou'),('mAcc','macc'),('B-mIoU','b_miou')]:close(row['metrics'][label],actual[key])
        close(row['metrics']['Recall'],actual['region_recall']['recall'])
        rr=actual['region_recall'];close(rr['recall'],rr['tp']/rr['gt'])
        if rr['gt']!=283 or rr['small_gt']!=248 or rr['vehicle_gt']!=59:raise ValueError('Region recall denominator changed')
    audited=[]
    for model in models['models']:
        if model['status']=='missing':continue
        folder=Path(model['comparison_dir']);receipt=read(folder/'render_receipt.json')
        if sha256_file(model['ply'])!=model['ply_sha256'] or receipt['ply_sha256']!=model['ply_sha256']:raise ValueError('Model source changed')
        if receipt['gt_rgb_sha256']!=canonical:raise ValueError('GT pixel alignment mismatch')
        for file in names:
            if sha256_file(Path(model['common_render_dir'])/file)!=receipt['render_sha256'][file]:raise ValueError('Render changed')
        row=summary['models'][model['id']]
        check_visual(row,folder,'ours_%d'%model['iteration'],standard)
        check_sem(model['id'],row);audited.append(model['id'])
    check_sem('GT',summary['GT'])
    variants=[]
    if not args.main_only:
        manifest=read(root/'ablations20/ablation_manifest.json');plan=read(root/'ablations20/ablation_plan.json')
        if manifest['status']!='complete':raise ValueError('Ablations incomplete')
        for v in summary['ablations20']['variants']:
            if v['status']!='complete':raise ValueError('Report has pending ablation')
            stage=root/'ablations20'/v['slug']/'stage3';validation=read(stage/'ablation_validation.json');run=read(stage/'run_args.json')
            if manifest['jobs'][v['slug']]['status']!='complete' or validation['status']!='complete':raise ValueError('Incomplete ablation verification')
            if validation['gaussians']!=89405 or not validation['native_fields_exact'] or validation['quota_violations']:raise ValueError('Ablation native/count validation failed')
            if sha256_file(validation['ply'])!=validation['ply_sha256'] or sha256_file(validation['checkpoint'])!=validation['checkpoint_sha256']:raise ValueError('Ablation model changed')
            test=stage/'test'/('ours_%d'%run['final_iteration']);gt={f:rgb_hash(test/'gt'/f) for f in names}
            if gt!=canonical:raise ValueError('Ablation GT differs from common evaluation')
            check_visual(v,stage,'ours_%d'%run['final_iteration'],standard)
            check_sem('ablation_'+v['slug'],v)
            ref=summary['ablations20']['reference']['metrics']
            for metric,val in v['metrics'].items():
                if val is None:raise ValueError('Missing ablation metric')
                expected=val-ref[metric]
                # JSON deltas retain fraction units; report presentation may convert to pp.
                close(v['delta_vs_full'][metric],expected)
            variants.append(v['slug'])
        if len(variants)!=7:raise ValueError('Seven leave-one-out variants required')
        if len(standard['models'])!=20:raise ValueError('LPIPS must cover all 13 models and seven ablations')
        if summary['status']!='complete':raise ValueError('Report is not complete')
    result=dict(status='main_complete' if args.main_only else 'complete',native_comparison_models=len(audited),model_ids=audited,
        visual_test_views=9,semantic_test_views=2,semantic_frames=data['splits']['test'],semantic_checkpoint_sha256=protocol['checkpoint_sha256'],
        semantic_test_label_leakage=False,recall_denominators=dict(all=283,small=248,vehicle=59),ablation_variants=variants,
        source_gt_alignment_exact=True,reported_metrics_match_source=True,prediction_count=len(pred['predictions']),
        lpips_protocol='standard_minus1_plus1' if standard else 'legacy_0_1',legacy_lpips_files_preserved=True,
        report_sha256=sha256_file(root/'README_统一对比与消融.md'),summary_sha256=sha256_file(root/'reports/summary.json'))
    atomic_json_save(result,root/('main_validation.json' if args.main_only else 'final_validation.json'))
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
