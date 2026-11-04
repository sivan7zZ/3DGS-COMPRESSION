"""Constraint-guided global SPS with adaptive recovery, merging and reallocation.

All decisions use TRAIN cameras. Native SH3 rendering attributes stay unchanged
in format; compression metadata is saved separately. No semantic supervision.
"""
import argparse
import json
import math
import os
from pathlib import Path
import random
import time
import numpy as np
import torch
import torch.nn.functional as F
from arguments import ModelParams, OptimizationParams, PipelineParams
from gaussian_renderer import render
from scene import GaussianModel, Scene
from train_prune import prune, evaluate
from utils.loss_utils import l1_loss, ssim
from utils.image_utils import psnr
from utils.general_utils import safe_state
from utils.distill_utils import atomic_json_save, atomic_torch_save, atomic_write, sha256_file, capture_rng_state, restore_rng_state
from utils.checkpoint_utils import load_checkpoint
from utils.adaptive_schedule import AdaptiveSchedule, survival_quotas, audit_quotas
from utils.adaptive_constraints import build_topology, select_feasible_prune
from utils.adaptive_mutation import CompressionState, find_safe_merge_pairs
from utils.structure_losses import TrainingStructureROIs

ROOT = Path(__file__).resolve().parent

def path(value):
    p=Path(value).expanduser()
    return p if p.is_absolute() else ROOT/p


class TrainProbes:
    """Fixed TRAIN views for recoverability and reference-supported restoration."""
    def __init__(self, cameras, shared, model, pipeline, background, risk_maps):
        names = [c.image_name for c in cameras]
        chosen = np.linspace(0,len(names)-1,8,dtype=int)
        self.cameras = [cameras[int(i)] for i in chosen]
        self.risks = risk_maps
        self.shared = shared
        self.reference = {}
        self.pipeline,self.background = pipeline,background
        with torch.no_grad():
            for c in self.cameras:
                self.reference[c.image_name] = render(c,model,pipeline,background)['render'].detach().cpu()
        self.criterion = None

    @torch.no_grad()
    def measure(self, model, hard=False, full_metrics=False, hard_config=None):
        values={'loss':0.,'PSNR':0.,'SSIM':0.}
        if full_metrics:
            if self.criterion is None:
                from lpipsPyTorch import LPIPS
                saved_rng=capture_rng_state()
                self.criterion=LPIPS('vgg','0.1').cuda().eval()
                restore_rng_state(saved_rng)
            values['LPIPS']=0.
        hard_maps={}
        for c in self.cameras:
            image=render(c,model,self.pipeline,self.background)['render']
            gt=c.original_image.to(image.device)
            values['loss']+=float(l1_loss(image,gt))/len(self.cameras)
            if full_metrics:
                values['PSNR']+=float(psnr(image,gt).mean())/len(self.cameras)
                values['SSIM']+=float(ssim(image,gt).mean())/len(self.cameras)
                values['LPIPS']+=float(self.criterion(image[None],gt[None]).mean())/len(self.cameras)
            if hard:
                ref=self.reference[c.image_name].to(image.device)
                error=(image-ref).abs().mean(0)
                risk=self.risks[c.image_name].to(image.device)
                hc=hard_config or {}
                h=error*(hc.get('rgb_weight',1.)+hc.get('structure_weight',1.)*risk)
                # Absolute RGB-difference units; no per-instance score transform.
                h=torch.where(h>=hc.get('hard_region_threshold',.01),h,torch.zeros_like(h))
                hard_maps[c.image_name]=h.cpu().numpy()
        if not full_metrics:
            values.pop('PSNR');values.pop('SSIM')
        return values,hard_maps

    def restoration_scores(self, hard_maps, source_count):
        q=np.zeros(source_count,np.float64)
        for name,h in hard_maps.items():
            with np.load(str(self.shared/'contribution_cells'/(Path(name).stem+'.npz')),allow_pickle=False) as z:
                if tuple(z['image_shape'])!=h.shape:
                    raise ValueError('Source support/risk dimensions differ: '+name)
                cell=int(z['cell_size']);gh,gw=map(int,z['grid_shape'])
                padded=np.zeros((gh*cell,gw*cell),np.float32)
                padded[:h.shape[0],:h.shape[1]]=h
                valid=np.zeros_like(padded);valid[:h.shape[0],:h.shape[1]]=1
                summed=padded.reshape(gh,cell,gw,cell).sum((1,3))
                counts=valid.reshape(gh,cell,gw,cell).sum((1,3))
                mean=(summed/np.maximum(counts,1)).ravel()
                q+=np.bincount(z['gaussian_ids'],weights=z['mass']*mean[z['cell_ids']],minlength=source_count)
        return q


def load_risk_maps(cameras, directory):
    result={}
    for c in cameras:
        with np.load(str(directory/(Path(c.image_name).stem+'.npz')),allow_pickle=False) as z:
            value=torch.from_numpy(np.array(z['risk'],np.float32,copy=True))
        shape=(int(c.image_height),int(c.image_width))
        if tuple(value.shape)!=shape:
            value=F.interpolate(value[None,None],size=shape,mode='bilinear',align_corners=False)[0,0]
        result[c.image_name]=value
    return result


def make_model(config,output,resume=None):
    parser=argparse.ArgumentParser()
    lp,op,pp=ModelParams(parser),OptimizationParams(parser),PipelineParams(parser)
    args=parser.parse_args(['-s',str(path(config['source_path'])),'-m',str(output),'--eval','--resolution',str(config.get('resolution',-1))])
    args.data_device='cpu'
    ds,opt,pipe=lp.extract(args),op.extract(args),pp.extract(args)
    model=GaussianModel(ds.sh_degree)
    scene=Scene(ds,model,shuffle=False,initialize_gaussians=False)
    if resume:
        saved,iteration=load_checkpoint(resume)
        model.load_checkpoint_model(saved,opt,restore_optimizer=True)
    else:
        model.load_ply(str(path(config['input_ply'])))
        model.spatial_lr_scale=scene.cameras_extent
        model.training_setup(opt)
    with (output/'cfg_args').open('w') as stream:
        stream.write(str(argparse.Namespace(**vars(ds))))
    return ds,opt,pipe,model,scene


def filter_merge_quotas(pairs,metadata,instance_min):
    counts=dict(zip(*np.unique(metadata['instance_ids'],return_counts=True)))
    selected=[]
    for i,j in pairs:
        k=int(metadata['instance_ids'][int(i)])
        if counts[k]-1>=instance_min.get(k,0):
            selected.append((int(i),int(j)));counts[k]-=1
    return np.asarray(selected,dtype=np.int64).reshape(-1,2)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True)
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--smoke_rounds',type=int,default=0)
    args=parser.parse_args()
    config=json.loads(path(args.config).read_text())
    output=path(config['output_root'])/'stage3'
    if not args.resume and output.exists() and any(output.iterdir()):
        raise FileExistsError('Use a new output directory or --resume: '+str(output))
    output.mkdir(parents=True,exist_ok=True)
    safe_state(False)
    torch.set_num_threads(int(config.get('cpu_threads',8)))
    shared=path(config['shared_metadata'])
    with np.load(str(shared/'adaptive_metadata.npz'),allow_pickle=False) as z:
        fields=('instance_ids','confidence','original_ids','instance_fragility','boundary_mask','structure_risk','visibility')
        metadata={k:np.array(z[k],copy=True) for k in fields}
    metadata['topology_critical']=np.zeros(len(metadata['instance_ids']),bool)
    source_count=len(metadata['instance_ids'])
    if not np.array_equal(metadata['original_ids'],np.arange(source_count)):
        raise ValueError('Initial metadata is not in source PLY row order')
    meta_manifest=json.loads((shared/'metadata_manifest.json').read_text())
    if sha256_file(path(config['input_ply']))!=meta_manifest['source_ply_sha256']:
        raise ValueError('Metadata belongs to a different source PLY')
    for key,actual in [('association_threshold',meta_manifest['association_threshold']),('instance_confidence_threshold',meta_manifest['instance_threshold'])]:
        if config['instance_protection'][key]!=actual:raise ValueError('Metadata/config mismatch: '+key)
    if config['boundary_protection']['boundary_threshold']!=meta_manifest['boundary_threshold']:
        raise ValueError('Boundary threshold mismatch')
    target=int(config['pruning_target'])
    if not 0<target<source_count:raise ValueError('Invalid target')
    amin,bmin,policies=survival_quotas(metadata,config)
    conservative=sum(max(amin.get(k,0),bmin.get(k,0)) for k in set(amin)|set(bmin))
    if conservative>target:raise ValueError('Declared instance/boundary floors exceed target: '+str(conservative))
    atomic_json_save(dict(instance_min=amin,boundary_min=bmin,rows=policies,lower_bound=conservative,target=target),output/'protection_policy.json')
    ds,opt,pipe,model,scene=make_model(config,output,output/'latest.pth' if args.resume else None)
    schedule=AdaptiveSchedule(config['adaptive_compression'])
    total_steps,remaining,events=0,[],[]
    if args.resume:
        state,extra=CompressionState.load(model,str(output/'compression_state.pt'))
        if extra['config']!=config or extra['checkpoint_sha256']!=sha256_file(output/'latest.pth'):
            raise ValueError('Resume configuration/checkpoint mismatch')
        if extra['input_hash']!=sha256_file(path(config['input_ply'])) or extra['metadata_hash']!=sha256_file(shared/'adaptive_metadata.npz') or extra['reference_hash']!=sha256_file(output/'probe_reference.pt'):
            raise ValueError('Resume source/metadata/reference content changed')
        total_steps=extra['total_steps'];remaining=extra['remaining'];events=extra['events']
        schedule.ratio=extra['schedule_ratio'];schedule.round=extra['schedule_round'];schedule.history=extra['schedule_history']
        restore_rng_state(extra['rng'])
    else:
        state=CompressionState(model,metadata,pool_rounds=config['budget_reallocation']['removed_pool_rounds'])
    cameras=scene.getTrainCameras()
    if [Path(c.image_name).stem for c in cameras]!=meta_manifest['train_frames']:
        raise ValueError('Training view order differs from metadata')
    if [Path(c.image_name).stem for c in scene.getTestCameras()]!=meta_manifest['test_frames_excluded']:
        raise ValueError('Test split differs from metadata')
    risks=load_risk_maps(cameras,shared/'structure_maps')
    rois=TrainingStructureROIs(str(path(config['legacy_structure_roi_dir'])),cameras) if config.get('preserve_v3_losses',True) else None
    background=torch.tensor([0.,0.,0.],device='cuda')
    # Initial full-resolution reference must remain the original source for resume.
    ref_path=output/'probe_reference.pt'
    if args.resume:
        reference=torch.load(str(ref_path),map_location='cpu')
        probes=TrainProbes(cameras,shared,model,pipe,background,risks)
        probes.reference=reference
    else:
        probes=TrainProbes(cameras,shared,model,pipe,background,risks)
        atomic_torch_save(probes.reference,ref_path)
    manifest=dict(status='running',method='global_SPS_feasibility_adaptive_merge_reallocate',config=config,
                  source_count=source_count,target=target,events=events,train_views=[c.image_name for c in cameras],
                  test_views=[c.image_name for c in scene.getTestCameras()],probe_views=[c.image_name for c in probes.cameras],
                  score='original SPS image.sum + 0.10 frequency detail; no instance/structure score additions',
                  restoration='reference alpha*T support pooled in 16px cells; hard error from current compressed TRAIN renders; merged rows ineligible',
                  input_ply_sha256=sha256_file(path(config['input_ply'])),metadata_sha256=sha256_file(shared/'adaptive_metadata.npz'))
    atomic_json_save(manifest,output/'run_args.json')
    started=time.monotonic()
    max_rounds=int(config['adaptive_compression'].get('max_rounds',60))
    while state.count>target:
        if schedule.round>=max_rounds:raise RuntimeError('Max rounds reached before target')
        round_id=schedule.round+1;before=state.count
        round_target=schedule.round_target(before,target)
        print('ROUND %d START N=%d target=%d ratio=%.5f'%(round_id,before,round_target,schedule.ratio),flush=True)
        measure_before,_=probes.measure(model)
        topo=build_topology(model.get_xyz.detach().cpu().numpy(),state.metadata['instance_ids'],
                            state.metadata['confidence'],state.metadata['boundary_mask'],config['topology_protection'])
        state.metadata['topology_critical']=np.asarray(topo['critical_mask'],bool)
        state.metadata['linearity']=np.asarray(topo['linearity'],np.float32)
        state.metadata['principal_direction']=np.asarray(topo['direction'],np.float32)
        state.metadata['thin_candidate_mask']=np.asarray(topo['thin_candidate_mask'],bool)
        scores,_=prune(scene,model,pipe,background,0.,ds,protect_instances=False,
                       score_roi_weight=0.,structure_score_weight=0.,return_scores=True)
        scores=scores.reshape(-1).cpu().numpy()
        reserve=int(round_target*config['budget_reallocation']['reserve_ratio']) if config['budget_reallocation']['enabled'] else 0
        temporary_target=max(1,round_target-reserve)
        # Permit a small safe merge tranche after feasibility pruning.
        merge_budget=min(int(before*config['gaussian_merging'].get('max_fraction',.005)),before-round_target) if config['gaussian_merging']['enabled'] else 0
        prune_goal=before-temporary_target-merge_budget
        mask,audit=select_feasible_prune(scores,state.metadata['instance_ids'],amin,prune_goal,
                    boundary_mask=state.metadata['boundary_mask'] if config['boundary_protection']['enabled'] else None,
                    boundary_min_counts=bmin,boundary_mode=config['boundary_protection']['mode'],topology=topo)
        if not audit.get('target_reached',int(mask.sum())==prune_goal):
            raise RuntimeError('Infeasible pruning round; constraints were not relaxed: '+str(audit))
        state.remove(mask,round_id,scores=scores,pool=config['budget_reallocation']['enabled'])
        after_prune=state.count
        merged=0
        if merge_budget:
            safe_meta=dict(state.metadata)
            safe_meta['topology_critical']=state.metadata['topology_critical']|state.metadata['thin_candidate_mask']
            mc=dict(config['gaussian_merging']);mc.pop('enabled',None);mc.pop('max_fraction',None)
            pairs=find_safe_merge_pairs(model,safe_meta,max_pairs=merge_budget,**mc)
            pairs=filter_merge_quotas(pairs,state.metadata,amin)
            if len(pairs):merged=state.merge(pairs,round_id)
        after_merge=state.count
        # A lack of redundant merge candidates consumes ordinary global SPS budget.
        if state.count>temporary_target:
            topo=build_topology(model.get_xyz.detach().cpu().numpy(),state.metadata['instance_ids'],
                                state.metadata['confidence'],state.metadata['boundary_mask'],config['topology_protection'])
            state.metadata['topology_critical']=np.asarray(topo['critical_mask'],bool)
            # Merged rows receive a freshly computed BASE score, never an invented rank.
            refreshed,_=prune(scene,model,pipe,background,0.,ds,protect_instances=False,return_scores=True)
            refreshed=refreshed.reshape(-1).cpu().numpy()
            extra_mask,extra_audit=select_feasible_prune(refreshed,state.metadata['instance_ids'],amin,state.count-temporary_target,
                boundary_mask=state.metadata['boundary_mask'] if config['boundary_protection']['enabled'] else None,
                boundary_min_counts=bmin,boundary_mode=config['boundary_protection']['mode'],topology=topo)
            if int(extra_mask.sum())!=state.count-temporary_target:raise RuntimeError('Infeasible post-merge fill pruning')
            state.remove(extra_mask,round_id,scores=refreshed,pool=config['budget_reallocation']['enabled'])
            audit['post_merge_pruning']=extra_audit
        after_compression=state.count
        audit_quotas(state.metadata,amin,bmin)
        damaged,hard_maps=probes.measure(model,hard=config['budget_reallocation']['enabled'],hard_config=config['budget_reallocation'])
        restored=0;restore_stats={}
        if reserve:
            q=probes.restoration_scores(hard_maps,source_count)
            pool=state.pool_arrays(round_id)
            ids=pool['metadata']['original_ids'];eligible=np.flatnonzero(ids>=0)
            values=q[ids[eligible]]
            ordered=eligible[np.argsort(-values,kind='stable')]
            positive=ordered[q[ids[ordered]]>0]
            # Fill residual reserved slots by the separately defined restoration Q
            # (zero-score ties are deterministic), not by per-instance SPS ranking.
            chosen=ordered[:min(reserve,round_target-state.count)]
            restored=state.restore(chosen,round_id,round_target)
            restore_stats=dict(positive_candidates=int(len(positive)),selected_positive=int(np.sum(q[ids[chosen]]>0)),
                               pool_size=int(len(ids)),q_max=float(values.max()) if len(values) else 0.)
            del pool, q, ids, values, ordered, positive, eligible, chosen
        if state.count!=round_target:raise RuntimeError('Round target mismatch %d != %d'%(state.count,round_target))
        audit_quotas(state.metadata,amin,bmin)
        history=[];reason=''
        while True:
            done,reason=schedule.recovery_done(history,final=state.count==target)
            if done:break
            if not remaining:remaining=list(range(len(cameras)))
            c=cameras[remaining.pop(random.randrange(len(remaining)))]
            # Relative update counter controls LR; no fixed iteration triggers compression.
            model.update_learning_rate(15000+min(total_steps,15000))
            image=render(c,model,pipe,background)['render'];gt=c.original_image.to(image.device)
            loss=(1-opt.lambda_dssim)*l1_loss(image,gt)+opt.lambda_dssim*(1-ssim(image,gt))
            if rois is not None:
                region,edge=rois.losses(c.image_name,image,gt)
                boundary,thin=rois.structure_losses(c.image_name,image,gt)
                weights=config['recovery_losses']
                loss=loss+weights['instance_roi']*region+weights['instance_edge']*edge+weights['boundary']*boundary+weights['thin']*thin
            if config['structure_loss']['enabled']:
                risk=risks[c.image_name].to(image.device)
                loss=loss+config['structure_loss']['weight']*(risk*(image-gt).abs().mean(0)).mean()
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite recovery loss')
            loss.backward();model.optimizer.step();model.optimizer.zero_grad(set_to_none=True)
            history.append(float(loss.detach()));total_steps+=1
            if len(history)%100==0:print('ROUND %d RECOVERY %d loss=%.7f N=%d'%(round_id,len(history),history[-1],state.count),flush=True)
        recovered,_=probes.measure(model,full_metrics=True)
        change=schedule.update(measure_before['loss'],damaged['loss'],recovered['loss'])
        event=dict(round_id=round_id,before=before,after_pruning=after_prune,after_merging=after_merge,
                   temporary_count=after_compression,after_reallocation=state.count,target=round_target,
                   pruned=before-after_prune+after_merge-after_compression,merged=int(merged),restored=int(restored),
                   recovery_steps=len(history),total_steps=total_steps,recovery_trigger=reason,recovery_loss_history=history,
                   adaptation=change,train_probe_metrics=recovered,selection=audit,topology=topo.get('audit',{}),
                   restoration=restore_stats,quota_audit=audit_quotas(state.metadata,amin,bmin))
        events.append(event)
        atomic_json_save(event,output/'rounds'/('round_%03d.json'%round_id))
        checkpoint=output/'latest.pth'
        atomic_torch_save((model.capture(),total_steps),checkpoint)
        extra=dict(config=config,checkpoint_sha256=sha256_file(checkpoint),input_hash=manifest['input_ply_sha256'],metadata_hash=manifest['metadata_sha256'],reference_hash=sha256_file(ref_path),total_steps=total_steps,remaining=remaining,
                   events=events,schedule_ratio=schedule.ratio,schedule_round=schedule.round,schedule_history=schedule.history,rng=capture_rng_state())
        state.save(str(output/'compression_state.pt.tmp'),extra=extra)
        os.replace(str(output/'compression_state.pt.tmp'),str(output/'compression_state.pt'))
        manifest.update(events=events,completed_steps=total_steps,current_gaussians=state.count,last_checkpoint=str(checkpoint))
        atomic_json_save(manifest,output/'run_args.json')
        print('ROUND %d COMPLETE N=%d recovered=%.7f ratio_next=%.5f'%(round_id,state.count,recovered['loss'],schedule.ratio),flush=True)
        if args.smoke_rounds and schedule.round>=args.smoke_rounds:break
    complete=state.count==target
    checkpoint=output/('chkpnt%d.pth'%total_steps)
    atomic_torch_save((model.capture(),total_steps),checkpoint)
    ply=output/'point_cloud'/('iteration_%d'%total_steps)/'point_cloud.ply'
    atomic_write(str(ply),model.save_ply)
    with (output/'compression_metadata.npz').open('wb') as stream:
        np.savez_compressed(stream,**state.metadata)
    final_metrics=evaluate(scene,model,pipe,background,ds,total_steps)
    atomic_json_save(final_metrics,output/'metrics_float.json')
    manifest.update(status='complete' if complete else 'smoke_complete',completed_steps=total_steps,
                    final_gaussians=state.count,last_checkpoint=str(checkpoint),last_ply=str(ply),final_iteration=total_steps,
                    elapsed_seconds=time.monotonic()-started,quota_audit=audit_quotas(state.metadata,amin,bmin))
    atomic_json_save(manifest,output/'run_args.json')
    print('ADAPTIVE COMPRESSION '+manifest['status']+' '+json.dumps(final_metrics),flush=True)

if __name__=='__main__':main()
