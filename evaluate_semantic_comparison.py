"""Evaluate frozen semantic predictions against registered City held-out labels."""
import argparse
import json
from pathlib import Path
import numpy as np
from PIL import Image
from utils.distill_utils import atomic_json_save,sha256_file
from utils.semantic_comparison_metrics import evaluate_semantics

ROOT=Path(__file__).resolve().parent


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=ROOT/'output/city2_comparison_20261104');p.add_argument('--inputs',type=Path,required=True);p.add_argument('--predictions',type=Path,required=True);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    data=json.loads((args.root/'semantic/dataset_manifest.json').read_text());inputs=json.loads(args.inputs.read_text());preds=json.loads(args.predictions.read_text())
    if inputs['semantic_test_frames']!=data['splits']['test']:raise ValueError('Semantic test frames changed')
    if set(data['splits']['test'])&set(data['splits']['refit_train']):raise ValueError('Test data leaked into segmentation fitting')
    if preds['inputs_json_sha256']!=sha256_file(args.inputs):raise ValueError('Predictions belong to different input images')
    if preds['checkpoint_sha256']!=sha256_file(preds['checkpoint']):raise ValueError('Frozen segmenter changed')
    meta=preds['metadata']
    manifest_sha=sha256_file(args.root/'semantic/dataset_manifest.json')
    if inputs['data_manifest_sha256']!=manifest_sha or meta['data_manifest_sha256']!=manifest_sha:
        raise ValueError('Dataset registration or label manifest changed after fitting')
    if set(meta['train_frames'])!=set(data['splits']['refit_train']):
        raise ValueError('Expected frozen model refitted on exactly eight non-test labeled views')
    if set(meta['train_frames'])&set(data['splits']['test']) or meta['test_frames_used_for_training_or_selection']:raise ValueError('Test leakage in semantic model')
    by_id={r['id']:r for r in preds['predictions']}
    if len(by_id)!=len(preds['predictions']):raise ValueError('Duplicate predicted IDs')
    labels={r['frame']:r for r in data['frames']}
    targets=[]
    for name in data['splits']['test']:
        row=labels[name]
        if sha256_file(row['label_path'])!=row['label_sha256']:raise ValueError('GT label changed')
        targets.append(np.asarray(Image.open(row['label_path']).convert('L')))
    results={}
    for group,ids in inputs['groups'].items():
        if len(ids)!=len(targets):raise ValueError('Missing annotated test view in '+group)
        predicted=[]
        for ident in ids:
            r=by_id[ident]
            if sha256_file(r['image_path'])!=r['image_sha256'] or sha256_file(r['prediction_path'])!=r['prediction_sha256']:raise ValueError('Prediction/source image content changed')
            value=np.asarray(Image.open(r['prediction_path']).convert('L'))
            if value.shape!=targets[len(predicted)].shape:raise ValueError('Prediction/GT geometry mismatch')
            predicted.append(value)
        metrics=evaluate_semantics(predicted,targets)
        results[group]=dict(metrics=metrics,prediction_ids=ids,model=inputs['models'].get(group),
                            semantic_frames=data['splits']['test'],n_semantic_views=2)
    atomic_json_save(dict(status='complete',checkpoint=preds['checkpoint'],checkpoint_sha256=preds['checkpoint_sha256'],
                    data_manifest_sha256=sha256_file(args.root/'semantic/dataset_manifest.json'),predictions_manifest_sha256=sha256_file(args.predictions),
                    protocol='One frozen scene-specific 2D segmenter for every RGB render and GT RGB; six semantic classes, ignore GT zero; exactly two annotated held-out views',
                    results=results),args.output)
    print(json.dumps({k:{m:round(v['metrics'][m],5) for m in ('miou','macc','b_miou')} for k,v in results.items()},indent=2))

if __name__=='__main__':main()
