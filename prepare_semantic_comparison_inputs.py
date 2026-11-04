"""Build immutable two-held-out-view inputs for the same frozen semantic model."""
import argparse
import json
from pathlib import Path
from utils.distill_utils import atomic_json_save,sha256_file

ROOT=Path(__file__).resolve().parent


def prepare(root,include_ablations=False):
    data=json.loads((root/'semantic/dataset_manifest.json').read_text())
    test=data['splits']['test'];all_test=json.loads((root/'models.json').read_text())['test_frames']
    frame_index={name:i for i,name in enumerate(all_test)}
    models=json.loads((root/'models.json').read_text())['models']
    images=[];groups={}
    frames={r['frame']:r for r in data['frames']}
    for name in test:images.append(dict(id='GT/'+name,path=frames[name]['rgb_path']))
    groups['GT']=[r['id'] for r in images]
    def add(model_id,render_dir,entry):
        ids=[]
        for name in test:
            image=Path(render_dir)/('%05d.png'%frame_index[name])
            if not image.is_file():raise FileNotFoundError(str(image))
            ident=model_id+'/'+name;images.append(dict(id=ident,path=str(image.resolve()),image_sha256=sha256_file(image)));ids.append(ident)
        groups[model_id]=ids
    entries={}
    for m in models:
        if m['status']=='missing':continue
        add(m['id'],m['common_render_dir'],m);entries[m['id']]=m
    if include_ablations:
        plan=json.loads((root/'ablations20/ablation_plan.json').read_text())
        for p in sorted((root/'ablations20').iterdir()):
            cfg=p/'stage3/run_args.json'
            if not cfg.is_file():continue
            run=json.loads(cfg.read_text())
            if run['status']!='complete':raise ValueError('Incomplete ablation '+p.name)
            ident='ablation_'+p.name
            add(ident,p/'stage3/test'/('ours_%d'%run['final_iteration'])/'renders',run)
            entries[ident]=dict(id=ident,method=p.name,target_ratio=20,ablation=True,model_dir=str(p/'stage3'),
                                ply=run['last_ply'],iteration=run['final_iteration'],actual_count=run['final_gaussians'],
                                common_render_dir=str(p/'stage3/test'/('ours_%d'%run['final_iteration'])/'renders'))
    dest=root/'semantic'/('inputs_all.json' if include_ablations else 'inputs_main.json')
    atomic_json_save(dict(images=images,groups=groups,semantic_test_frames=test,visual_test_frames=all_test,models=entries,
               data_manifest_sha256=sha256_file(root/'semantic/dataset_manifest.json'),protocol='Two annotated held-out frames only; no test data fitted'),dest)
    print(str(dest))
    return dest


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=ROOT/'output/city2_comparison_20261104');p.add_argument('--include_ablations',action='store_true');args=p.parse_args();prepare(args.root,args.include_ablations)

if __name__=='__main__':main()
