"""Independent CPU audit: distance-based boundary matches and hard clDice formula."""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from skimage.morphology import skeletonize

from evaluate_comparison_lpips import read,sha256_file,atomic_json
from evaluate_preservation import validate_inputs

ROOT=Path(__file__).resolve().parent


def close(a,b):
    if a is None or b is None:
        if a is not b:raise ValueError('Unexpected missing metric')
    elif not np.isclose(a,b,rtol=1e-11,atol=1e-11):raise ValueError('Metric differs: %s %s'%(a,b))


def erosion(mask,n):
    return cv2.erode(mask.astype(np.uint8),np.ones((3,3),np.uint8),iterations=n,
                     borderType=cv2.BORDER_CONSTANT,borderValue=1).astype(bool)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=ROOT/'output/city2_comparison_20261104')
    p.add_argument('--output',type=Path);args=p.parse_args();root=args.root;out=args.output or root/'preservation_metrics'
    inputs=read(out/'inputs.json');validate_inputs(inputs)
    mask=read(out/'mask_and_psnr.json');lp=read(out/'roi_lpips.json');summary=read(out/'summary.json')
    if any(x['status']!='complete' or x['input_signature']!=inputs['input_signature'] for x in (mask,lp,summary)):
        raise ValueError('Missing/mismatched final results')
    for path,digest in mask['provenance']['source_sha256'].items():
        if sha256_file(path)!=digest:raise ValueError('CPU metric source changed')
    for group in ('source_files_sha256','weight_files_sha256'):
        for path,digest in lp['criterion'][group].items():
            if sha256_file(path)!=digest:raise ValueError('LPIPS source/weights changed')
    data=read(root/'semantic/dataset_manifest.json');frozen=read(root/'semantic/model/training_protocol.json')
    if frozen['checkpoint_sha256']!=mask['provenance']['checkpoint_sha256']:raise ValueError('Evaluator weight differs')
    if sha256_file(frozen['checkpoint'])!=frozen['checkpoint_sha256']:raise ValueError('Evaluator changed')
    byframe={x['frame']:x for x in data['frames']};frames=inputs['protocol']['semantic_frames']
    gts={f:np.asarray(Image.open(byframe[f]['label_path'])) for f in frames}
    for f in frames:
        if sha256_file(byframe[f]['label_path'])!=byframe[f]['label_sha256']:raise ValueError('GT labels changed')
    small=[x for x in inputs['rois'] if x['group']=='semantic_small'];fixed=[x for x in inputs['rois'] if x['group']=='fixed_structure']
    if len(small)!=248 or len(fixed)!=6 or len(inputs['models'])!=14:raise ValueError('Support differs')
    maps=np.load(inputs['roi_region_ids_path'],allow_pickle=False)
    for r in small:
        support=maps[r['frame']]==r['region_id'];label=gts[r['frame']]
        if support.sum()!=r['area'] or not np.all(label[support]==r['class_id']):raise ValueError('ROI mask differs from GT')
        if not 16<=r['area']<=.01*label.size:raise ValueError('Ineligible small region')
    reference={f:np.asarray(Image.open(x['path']).convert('RGB')) for f,x in inputs['references'].items()}
    old=read(root/'semantic/main_metrics.json')['results']
    count=0
    for ident,model in inputs['models'].items():
        result=mask['results'][ident];preds={x['id'].split('/')[-1]:np.asarray(Image.open(x['prediction_path'])) for x in model['predictions']}
        values=[];skvalues=[]
        for c in range(1,7):
            b=dict(predicted=0,target=0,matched_predicted=0,matched_target=0)
            sk=dict(predicted_skeleton=0,target_skeleton=0,predicted_in_target=0,target_in_predicted=0)
            for f in frames:
                gt=gts[f];valid=gt!=0;gm=gt==c;pm=preds[f]==c
                safe=erosion(valid,3);gb=(gm & ~erosion(gm,1)) & safe;pb=(pm & ~erosion(pm,1)) & safe
                b['predicted']+=int(pb.sum());b['target']+=int(gb.sum())
                if gb.any():
                    distances=cv2.distanceTransform((~gb).astype(np.uint8),cv2.DIST_L2,cv2.DIST_MASK_PRECISE)
                    b['matched_predicted']+=int(np.count_nonzero(pb & (distances<=2+1e-6)))
                if pb.any():
                    distances=cv2.distanceTransform((~pb).astype(np.uint8),cv2.DIST_L2,cv2.DIST_MASK_PRECISE)
                    b['matched_target']+=int(np.count_nonzero(gb & (distances<=2+1e-6)))
                pm=pm & valid
                ps=skeletonize(np.pad(pm,1),method='zhang')[1:-1,1:-1] & erosion(valid,1)
                gs=skeletonize(np.pad(gm,1),method='zhang')[1:-1,1:-1] & erosion(valid,1)
                sk['predicted_skeleton']+=int(ps.sum());sk['target_skeleton']+=int(gs.sum())
                sk['predicted_in_target']+=int((ps & gm).sum());sk['target_in_predicted']+=int((gs & pm).sum())
            stored=result['mask_metrics']['per_class'][str(c)]
            for k,v in b.items():close(v,stored['boundary'][k])
            if b['predicted'] or b['target']:
                pr=b['matched_predicted']/b['predicted'] if b['predicted'] else 1.
                rc=b['matched_target']/b['target'] if b['target'] else 1.
                values.append(2*pr*rc/(pr+rc) if pr+rc else 0.)
            for k,v in sk.items():close(v,stored['skeleton'][k])
            if sk['predicted_skeleton'] or sk['target_skeleton']:
                pr=sk['predicted_in_target']/sk['predicted_skeleton'] if sk['predicted_skeleton'] else 0.
                rc=sk['target_in_predicted']/sk['target_skeleton'] if sk['target_skeleton'] else 0.
                skvalues.append(2*pr*rc/(pr+rc) if pr+rc else 0.)
        close(float(np.mean(values)),summary['results'][ident]['boundary_f1'])
        close(float(np.mean(skvalues)),summary['results'][ident]['cldice'])
        rr=old[ident]['metrics']['region_recall']
        close(summary['results'][ident]['small_object_recall'],rr['small_recall'])
        if summary['results'][ident]['small_gt']!=248:raise ValueError('Recall support differs')
        images={f:np.asarray(Image.open(x['path']).convert('RGB')) for f,x in model['images'].items()}
        qbyid={x['id']:x for x in result['per_roi']}
        lbyid={x['id']:x for x in lp['results'][ident]['per_roi']}
        if set(qbyid)!=set(lbyid) or len(qbyid)!=254:raise ValueError('Local quality ROI IDs differ')
        for roi in inputs['rois']:
            f=roi['frame'];x0,y0,x1,y1=roi['crop_xyxy']
            error=(reference[f][y0:y1,x0:x1].astype(np.float64)-images[f][y0:y1,x0:x1].astype(np.float64))/255
            mse=float(np.square(error).mean());q=qbyid[roi['id']]
            close(q['mse'],mse);close(q['psnr'],-10*np.log10(max(mse,1e-12)))
            if roi['group']=='semantic_small':
                support=maps[f][y0:y1,x0:x1]==roi['region_id']
                close(q['masked_object_psnr'],-10*np.log10(max(float(np.square(error[support]).mean()),1e-12)))
            if not np.isfinite(lbyid[roi['id']]['lpips']):raise ValueError('Nonfinite ROI LPIPS')
            count+=1
        for group in ('semantic_small','fixed_structure'):
            qs=[x['psnr'] for x in qbyid.values() if x['group']==group]
            ls=[x['lpips'] for x in lbyid.values() if x['group']==group]
            close(float(np.mean(qs)),result['roi_quality'][group]['roi_psnr'])
            close(float(np.mean(ls)),lp['results'][ident]['groups'][group]['roi_lpips'])
            if ident!='GT':
                close(np.mean(qs),summary['results'][ident]['roi_psnr' if group=='semantic_small' else 'fixed_roi_psnr'])
                close(np.mean(ls),summary['results'][ident]['roi_lpips' if group=='semantic_small' else 'fixed_roi_lpips'])
        if lp['results'][ident]['criterion']!=lp['criterion'] or abs(lp['results'][ident]['self_control'])>1e-10:
            raise ValueError('LPIPS protocol differs')
    controls=mask['perfect_label_control']
    if any(v!=1. for v in controls.values()) or lp['self_control']!=0.:raise ValueError('Perfect reference controls failed')
    result=dict(status='complete',models=14,annotated_test_frames=frames,gt_small_regions=248,gt_vehicle_small_regions=59,
                fixed_structure_rois=6,roi_measurements_audited=count,
                boundary_matches_independently_verified_with_euclidean_distance=True,
                hard_cldice_independently_recomputed=True,roi_psnr_independently_recomputed=True,
                roi_lpips_aggregation_and_source_hashes_verified=True,recall_matches_prior_independent_evaluation=True,
                no_ablations_or_training_executed=True,
                summary_sha256=sha256_file(out/'summary.json'),report_sha256=sha256_file(out/'README_地物保护指标.md'))
    atomic_json(result,out/'validation.json');print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
