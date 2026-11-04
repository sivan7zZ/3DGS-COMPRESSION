"""Rerender all located SH3 baselines with one native renderer and camera split."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from PIL import Image
import numpy as np
from plyfile import PlyData
from utils.distill_utils import sha256_file,atomic_json_save

ROOT=Path(__file__).resolve().parent
SOURCE=Path('/home/zhangys/gaussian-splatting（原生）/data/city')
DEFAULT=ROOT/'output/city2_comparison_20261104'


def read_json(p):return json.loads(Path(p).read_text())

def execute(command,log):
    log.parent.mkdir(parents=True,exist_ok=True)
    with log.open('a') as stream:
        process=subprocess.Popen(list(map(str,command)),cwd=str(ROOT),stdout=subprocess.PIPE,stderr=subprocess.STDOUT,universal_newlines=True,bufsize=1)
        for line in process.stdout:stream.write(line);stream.flush();print(line,end='',flush=True)
        status=process.wait()
    if status:raise subprocess.CalledProcessError(status,command)


def prepare(root,inventory):
    models=read_json(inventory)
    records=[]
    for entry in models:
        record=dict(entry);ply=Path(entry['ply']) if entry.get('ply') else None
        if ply is None or not ply.is_file():
            record.update(status='missing',comparison_dir=None);records.append(record);continue
        folder=root/'renders'/entry['id'];iteration=int(entry['iteration'])
        target=folder/'point_cloud'/('iteration_%d'%iteration)/'point_cloud.ply'
        target.parent.mkdir(parents=True,exist_ok=True)
        if target.is_symlink():
            if target.resolve()!=ply.resolve():raise ValueError('Render input symlink changed')
        elif target.exists():raise FileExistsError(str(target))
        else:target.symlink_to(ply.resolve())
        cfg=argparse.Namespace(sh_degree=entry['sh_degree'],source_path=str(SOURCE),model_path=str(folder),
             images='images',depths='',resolution=-1,white_background=False,train_test_exp=False,data_device='cpu',eval=True)
        (folder/'cfg_args').write_text(str(cfg))
        record.update(status='prepared',ply_sha256=sha256_file(ply),comparison_dir=str(folder),
            common_render_dir=str(folder/'test'/('ours_%d'%iteration)/'renders'),
            common_gt_dir=str(folder/'test'/('ours_%d'%iteration)/'gt'),
            actual_ratio=1788093./int(entry['actual_count']))
        records.append(record)
    atomic_json_save(dict(source=str(SOURCE),representation='native SH3; pre-SH-distillation',
          test_frames=['frame_%03d'%i for i in range(0,70,8)],models=records),root/'models.json')
    return records


def run(root,models):
    state_path=root/'render_pipeline.json'
    state=read_json(state_path) if state_path.is_file() else dict(status='running',completed=[])
    state['status']='running';atomic_json_save(state,state_path)
    renderer_sha=sha256_file(ROOT/'render.py')
    canonical=None
    for item in models:
        if item['status']=='missing':continue
        folder=Path(item['comparison_dir']);receipt=folder/'render_receipt.json'
        renders=Path(item['common_render_dir']);gt=Path(item['common_gt_dir'])
        expected=['%05d.png'%i for i in range(9)]
        valid=False
        if receipt.is_file():
            old=read_json(receipt)
            valid=old['ply_sha256']==item['ply_sha256'] and old['renderer_sha256']==renderer_sha and all((renders/f).is_file() and (gt/f).is_file() for f in expected)
            if valid:
                for file in expected:
                    if sha256_file(renders/file)!=old['render_sha256'][file]:raise ValueError('Cached render content changed')
        if not valid:
            execute([sys.executable,ROOT/'render.py','-s',SOURCE,'-m',folder,'--iteration',item['iteration'],
                     '--skip_train','--eval','--resolution',-1,'--data_device','cpu'],root/'logs'/('render_'+item['id']+'.log'))
        if sorted(p.name for p in renders.glob('*.png'))!=expected:raise ValueError('Test view count/order mismatch')
        gt_hash={};render_hash={}
        for file in expected:
            with Image.open(gt/file) as im:
                im=im.convert('RGB')
                if im.size!=(1592,894):raise ValueError('GT dimensions differ')
                import hashlib
                gt_hash[file]=hashlib.sha256(np.asarray(im).tobytes()).hexdigest()
            with Image.open(renders/file) as im:
                if im.size!=(1592,894):raise ValueError('Render dimensions differ')
            render_hash[file]=sha256_file(renders/file)
        if canonical is None:canonical=gt_hash
        if gt_hash!=canonical:raise ValueError('Canonical GT pixels differ: '+item['id'])
        atomic_json_save(dict(model_id=item['id'],ply_sha256=item['ply_sha256'],renderer_sha256=renderer_sha,
               gt_rgb_sha256=gt_hash,render_sha256=render_hash,frame_mapping={f:'frame_%03d'%(i*8) for i,f in enumerate(expected)},
               image_size=[1592,894],renderer='PWSR native standard, antialiasing=False, SH3',status='complete'),receipt)
        if item['id'] not in state['completed']:state['completed'].append(item['id'])
        atomic_json_save(state,state_path)
    state['status']='complete';atomic_json_save(state,state_path)
    folders=[Path(e['comparison_dir']) for e in models if e['status']!='missing']
    execute([sys.executable,ROOT/'metrics.py','-m']+folders+['--device','cuda'],root/'logs/visual_metrics.log')
    state['visual_metrics_complete']=True;atomic_json_save(state,state_path)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=DEFAULT);p.add_argument('--inventory',type=Path,required=True);p.add_argument('--prepare_only',action='store_true');args=p.parse_args()
    args.root.mkdir(parents=True,exist_ok=True)
    models=prepare(args.root,args.inventory)
    if not args.prepare_only:run(args.root,models)

if __name__=='__main__':main()
