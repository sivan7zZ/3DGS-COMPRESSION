"""Run the new relative adaptive compression, optional SH2, and native evaluation."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from utils.checkpoint_utils import atomic_json_save
from utils.distill_utils import sha256_file

ROOT=Path(__file__).resolve().parent

def resolve(value):
    p=Path(value).expanduser()
    return p if p.is_absolute() else ROOT/p

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',required=True)
    p.add_argument('--resume',action='store_true')
    p.add_argument('--skip_distill',action='store_true')
    p.add_argument('--smoke_rounds',type=int,default=0)
    args=p.parse_args()
    config=json.loads(resolve(args.config).read_text())
    output=resolve(config['output_root']).resolve()
    if (ROOT/'output').resolve() not in output.parents:raise ValueError('Output must be under repository output/')
    output.mkdir(parents=True,exist_ok=True)
    if args.resume and (output/'pipeline.json').is_file():
        previous=json.loads((output/'pipeline.json').read_text())
        if previous['config']!=config:raise ValueError('Resume config differs from saved pipeline')
    atomic_json_save(config,output/'config.json')
    manifest=dict(status='running',config=config,completed=[],commands=[],resume=args.resume)
    if args.resume and (output/'pipeline.json').is_file():
        manifest=json.loads((output/'pipeline.json').read_text());manifest['status']='running';manifest['resume']=True
    def run(name,argv):
        if name in manifest['completed']:return
        command=[sys.executable,str(ROOT/argv[0])]+list(map(str,argv[1:]))
        manifest['commands'].append(dict(name=name,command=command))
        atomic_json_save(manifest,output/'pipeline.json')
        with (output/(name+'.log')).open('a') as log:
            process=subprocess.Popen(command,cwd=str(ROOT),stdout=subprocess.PIPE,stderr=subprocess.STDOUT,universal_newlines=True,bufsize=1)
            for line in process.stdout:
                print(line,end='',flush=True);log.write(line);log.flush()
            result=process.wait()
        if result:raise subprocess.CalledProcessError(result,command)
        manifest['completed'].append(name);atomic_json_save(manifest,output/'pipeline.json')
    try:
        initial=['train_adaptive_compression.py','--config',resolve(args.config)]
        if args.resume and (output/'stage3'/'compression_state.pt').is_file():initial+=['--resume']
        if args.smoke_rounds:initial+=['--smoke_rounds',args.smoke_rounds]
        run('compress',initial)
        stage=json.loads((output/'stage3'/'run_args.json').read_text())
        if stage['status']!='complete' and not args.smoke_rounds:raise ValueError('Compression target not reached')
        common=['-s',resolve(config['source_path']),'--eval','--resolution',config.get('resolution',-1)]
        run('render_stage3',['render.py']+common+['-m',output/'stage3','--iteration',stage['final_iteration'],'--skip_train'])
        models=[output/'stage3']
        if not args.skip_distill and not args.smoke_rounds:
            d=['distill_train.py']+common+['-m',output/'sh2','--start_checkpoint',stage['last_checkpoint'],
                '--teacher_model',resolve(config['teacher_checkpoint']),'--new_max_sh',2,
                '--distill_start_iteration',35000,'--distill_steps',config.get('distill_steps',5000),'--enable_covariance']
            if args.resume and 'distill' not in manifest['completed'] and (output/'sh2').is_dir():
                saved=[p for p in (output/'sh2').glob('chkpnt*.pth') if Path(str(p)+'.state.pth').is_file()]
                if saved:
                    latest=max(saved,key=lambda p:int(p.stem.replace('chkpnt','')))
                    d=['distill_train.py','--resume_checkpoint',latest]
            run('distill',d)
            iteration=35000+config.get('distill_steps',5000)
            run('render_sh2',['render.py']+common+['-m',output/'sh2','--iteration',iteration,'--skip_train'])
            models.append(output/'sh2')
            shutil.copy2(str(output/'stage3'/'compression_metadata.npz'),str(output/'sh2'/'compression_metadata.npz'))
            atomic_json_save(dict(source=str(output/'stage3'/'compression_metadata.npz'),sha256=sha256_file(output/'sh2'/'compression_metadata.npz'),
                semantics='row-aligned compression-time labels and risk; SH distillation preserves Gaussian count and row order; geometric priors remain source-stage metadata'),output/'sh2'/'compression_metadata_provenance.json')
        run('metrics',['metrics.py','-m']+models+['--device','cuda'])
        manifest['status']='smoke_complete' if args.smoke_rounds else 'complete'
    except BaseException as e:
        manifest.update(status='failed',error=str(e));atomic_json_save(manifest,output/'pipeline.json');raise
    atomic_json_save(manifest,output/'pipeline.json')

if __name__=='__main__':main()
