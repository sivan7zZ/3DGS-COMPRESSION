"""Benchmark native render-only throughput, excluding loading and PNG encoding."""
import argparse
import ast
import gc
import json
from pathlib import Path
import numpy as np
import torch
from arguments import ModelParams, PipelineParams
from gaussian_renderer import render
from scene import GaussianModel, Scene
from utils.checkpoint_utils import atomic_json_save
from utils.distill_utils import sha256_file


def read_cfg(file):
    call=ast.parse(Path(file).read_text(),mode='eval').body
    if not isinstance(call,ast.Call) or not isinstance(call.func,ast.Name) or call.func.id!='Namespace' or call.args:
        raise ValueError('Expected plain Namespace config')
    return {kw.arg:ast.literal_eval(kw.value) for kw in call.keywords if kw.arg is not None}


@torch.no_grad()
def benchmark(directory,warmup=18,frames=90):
    directory=Path(directory).resolve()
    cfg=read_cfg(directory/'cfg_args')
    iterations=sorted(int(p.name.split('_')[-1]) for p in (directory/'point_cloud').glob('iteration_*') if (p/'point_cloud.ply').is_file())
    if not iterations:raise ValueError('No native model')
    iteration=iterations[-1]
    p=argparse.ArgumentParser();lp=ModelParams(p);pp=PipelineParams(p)
    args=p.parse_args([])
    for k,v in cfg.items():
        if hasattr(args,k):setattr(args,k,v)
    args.model_path=str(directory);args.data_device='cpu';args.eval=True
    dataset,pipeline=lp.extract(args),pp.extract(args)
    model=GaussianModel(dataset.sh_degree)
    scene=Scene(dataset,model,load_iteration=iteration,shuffle=False)
    cameras=scene.getTestCameras()
    if not cameras:raise ValueError('No fixed test cameras')
    background=torch.tensor([1.,1.,1.] if dataset.white_background else [0.,0.,0.],device='cuda')
    for i in range(warmup):render(cameras[i%len(cameras)],model,pipeline,background)
    torch.cuda.synchronize()
    timings=[]
    for i in range(frames):
        start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
        start.record();result=render(cameras[i%len(cameras)],model,pipeline,background);end.record()
        end.synchronize();timings.append(start.elapsed_time(end));del result
    total_ms=float(sum(timings))
    ply=directory/'point_cloud'/('iteration_%d'%iteration)/'point_cloud.ply'
    report=dict(method='CUDA events around native render; frame-synchronized; excludes model/image loading and PNG writing',
                gpu=torch.cuda.get_device_name(0),frames=frames,warmup_frames=warmup,test_views=[c.image_name for c in cameras],
                image_sizes=sorted(set((int(c.image_width),int(c.image_height)) for c in cameras)),
                gaussian_count=int(model.get_xyz.shape[0]),sh_degree=int(model.active_sh_degree),
                iteration=iteration,ply=str(ply),ply_sha256=sha256_file(ply),fps=frames*1000./total_ms,
                mean_ms=total_ms/frames,median_ms=float(np.median(timings)),p95_ms=float(np.percentile(timings,95)))
    atomic_json_save(report,directory/'render_benchmark.json')
    print(json.dumps(report),flush=True)
    del scene,model,cameras;gc.collect();torch.cuda.empty_cache()
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--models',nargs='+',required=True);p.add_argument('--warmup',type=int,default=18);p.add_argument('--frames',type=int,default=90);args=p.parse_args()
    if args.frames<1 or args.warmup<1:p.error('Positive warmup/frames required')
    for model in args.models:benchmark(model,args.warmup,args.frames)

if __name__=='__main__':main()
