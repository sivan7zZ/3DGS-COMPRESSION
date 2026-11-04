"""Read-only numerical verification of the new completed adaptive runs."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from plyfile import PlyData
from utils.checkpoint_utils import load_checkpoint,atomic_json_save
from utils.distill_utils import sha256_file
from utils.adaptive_schedule import survival_quotas,audit_quotas

ROOT=Path(__file__).resolve().parent

def verify_stage(stage,checkpoint,ply,expected):
    state,iteration=load_checkpoint(checkpoint)
    state=tuple(x.detach() if torch.is_tensor(x) else x for x in state)
    vertices=PlyData.read(str(ply))['vertex'].data
    n=len(vertices)
    if n!=expected or state[1].shape[0]!=expected:raise ValueError('Model count mismatch')
    fields={}
    for j,k in enumerate(('x','y','z')):fields[k]=state[1][:,j].numpy()
    dc=state[2].transpose(1,2).flatten(1).numpy();rest=state[3].transpose(1,2).flatten(1).numpy()
    for j in range(dc.shape[1]):fields['f_dc_%d'%j]=dc[:,j]
    for j in range(rest.shape[1]):fields['f_rest_%d'%j]=rest[:,j]
    fields['opacity']=state[6][:,0].numpy()
    for j in range(3):fields['scale_%d'%j]=state[4][:,j].numpy()
    for j in range(4):fields['rot_%d'%j]=state[5][:,j].numpy()
    for k,v in fields.items():
        if not np.array_equal(vertices[k],v):raise ValueError('PLY/checkpoint parameter mismatch: '+k)
    if set(vertices.dtype.names)!=set(fields)|{'nx','ny','nz'}:raise ValueError('Nonstandard PLY fields')
    for k in ('nx','ny','nz'):
        if np.any(vertices[k]!=0):raise ValueError('Unexpected normal')
    with np.load(stage/'compression_metadata.npz') as z:
        meta={k:np.array(z[k],copy=True) for k in z.files}
    if any(len(v)!=n for v in meta.values()):raise ValueError('Metadata length mismatch')
    if len(np.unique(meta['row_ids']))!=n:raise ValueError('Duplicate active row IDs')
    source=meta['original_ids'][meta['original_ids']>=0]
    if len(np.unique(source))!=len(source):raise ValueError('Duplicate restored source IDs')
    return dict(gaussians=n,checkpoint=str(checkpoint),checkpoint_sha256=sha256_file(checkpoint),
        ply=str(ply),ply_sha256=sha256_file(ply),native_fields_exact=True,standard_property_count=len(vertices.dtype.names),
        metadata_rows=n,source_lineage_rows=len(source),merged_rows=n-len(source),unique_row_ids=True),meta


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',default='output/city2_adaptive_constraints_20261102');args=p.parse_args()
    root=Path(args.root);root=root if root.is_absolute() else ROOT/root
    snap=json.loads((root/'source_snapshot/manifest.json').read_text())
    # Training sources are frozen; report-only updates are recorded separately.
    training=[f for f in snap['files'] if f not in ('finalize_adaptive_results.py',) and not f.endswith('.md')]
    for f in training:
        if sha256_file(ROOT/f)!=snap['files'][f]:raise ValueError('Training source changed: '+f)
    with np.load(root/'shared/adaptive_metadata.npz') as z:initial={k:np.array(z[k]) for k in ('instance_ids','instance_fragility','boundary_mask')}
    results=dict(status='complete',training_sources_match_snapshot=True,ratios={})
    for ratio in (10,20):
        run=root/('full_%dx'%ratio);config=json.loads((run/'config.json').read_text());man=json.loads((run/'pipeline.json').read_text())
        if man['status']!='complete':raise ValueError('Incomplete pipeline')
        stage=json.loads((run/'stage3/run_args.json').read_text());distill=json.loads((run/'sh2/run_args.json').read_text())
        if stage['status']!='complete' or distill['status']!='complete' or distill['completed_steps']!=5000 or distill['teacher_unchanged'] is not True:raise ValueError('Incomplete stages/teacher integrity')
        expected=int(config['pruning_target']);amin,bmin,_=survival_quotas(initial,config)
        sh3,m3=verify_stage(run/'stage3',Path(stage['last_checkpoint']),Path(stage['last_ply']),expected)
        sh2,m2=verify_stage(run/'sh2',run/'sh2/chkpnt40000.pth',run/'sh2/point_cloud/iteration_40000/point_cloud.ply',expected)
        if any(not np.array_equal(m3[k],m2[k]) for k in m3):raise ValueError('Distillation changed metadata rows')
        quotas=audit_quotas(m3,amin,bmin)
        for e in stage['events']:
            if e['after_reallocation']!=e['target'] or e['temporary_count']>e['target']:raise ValueError('Round budget error')
            if not e['selection']['target_reached'] or e['selection']['score_order']!='unchanged_global_ascending_stable':raise ValueError('Selection differs from specification')
            if e['quota_audit']['instance_violations'] or e['quota_audit']['boundary_violations']:raise ValueError('Historical quota violation')
        results['ratios'][str(ratio)]=dict(SH3=sh3,SH2=sh2,quota_violations=0,protected_instances=len(amin),
                round_count=len(stage['events']),recovery_updates=stage['completed_steps'],distillation_updates=5000,
                merged_total=sum(e['merged'] for e in stage['events']),restored_total=sum(e['restored'] for e in stage['events']),
                teacher_unchanged=True,sh2_metadata_exact=True)
    atomic_json_save(results,root/'final_validation.json')
    print(json.dumps(results,indent=2))

if __name__=='__main__':main()
