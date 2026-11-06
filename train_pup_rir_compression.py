"""PUP Fisher scoring + original RIR/DPRA, retaining v3.1 protection/mutation.

All compression and recovery decisions use training cameras. No semantic GT.
"""
import argparse,json,os,random,time
from pathlib import Path
import numpy as np
import torch
from gaussian_renderer import render
from train_prune import reconstruction_detection,evaluate
from train_adaptive_compression import (path,make_model,TrainProbes,load_risk_maps,filter_merge_quotas)
from utils.pruning_scores import pup_pruning_scores
from utils.rir_dpra_schedule import RIRDPRA
from utils.adaptive_schedule import survival_quotas,audit_quotas
from utils.adaptive_constraints import build_topology,select_feasible_prune
from utils.adaptive_mutation import CompressionState,find_safe_merge_pairs
from utils.structure_losses import TrainingStructureROIs
from utils.loss_utils import l1_loss,ssim
from utils.general_utils import safe_state
from utils.distill_utils import (atomic_json_save,atomic_torch_save,atomic_write,sha256_file,
                                 capture_rng_state,restore_rng_state)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True);parser.add_argument('--resume',action='store_true')
    parser.add_argument('--smoke_rounds',type=int,default=0);args=parser.parse_args()
    config=json.loads(path(args.config).read_text())
    if config.get('pruning_score')!='pup' or config.get('compression_schedule')!='rir_dpra':
        raise ValueError('This version requires PUP scoring and RIR/DPRA scheduling')
    if config['adaptive_compression'].get('enabled',True):
        raise ValueError('Relative adaptive schedule must be disabled explicitly')
    output=path(config['output_root'])/'stage3'
    if not args.resume and output.exists() and any(output.iterdir()):raise FileExistsError(str(output))
    output.mkdir(parents=True,exist_ok=True);safe_state(False)
    torch.set_num_threads(int(config.get('cpu_threads',8)))
    shared=path(config['shared_metadata']);meta_manifest=json.loads((shared/'metadata_manifest.json').read_text())
    if sha256_file(path(config['input_ply']))!=meta_manifest['source_ply_sha256']:raise ValueError('Different source PLY')
    with np.load(str(shared/'adaptive_metadata.npz'),allow_pickle=False) as z:
        fields=('instance_ids','confidence','original_ids','instance_fragility','boundary_mask','structure_risk','visibility')
        metadata={k:np.array(z[k],copy=True) for k in fields}
    source_count=len(metadata['instance_ids']);metadata['topology_critical']=np.zeros(source_count,bool)
    if not np.array_equal(metadata['original_ids'],np.arange(source_count)):raise ValueError('Metadata source row order differs')
    for key,actual in [('association_threshold',meta_manifest['association_threshold']),('instance_confidence_threshold',meta_manifest['instance_threshold'])]:
        if config['instance_protection'][key]!=actual:raise ValueError('Metadata/config mismatch: '+key)
    if config['boundary_protection']['boundary_threshold']!=meta_manifest['boundary_threshold']:raise ValueError('Boundary threshold differs')
    target=int(config['pruning_target']);amin,bmin,policies=survival_quotas(metadata,config)
    conservative=sum(max(amin.get(k,0),bmin.get(k,0)) for k in set(amin)|set(bmin))
    if not 0<target<source_count or conservative>target:raise ValueError('Invalid/infeasible final budget')
    atomic_json_save(dict(instance_min=amin,boundary_min=bmin,rows=policies,lower_bound=conservative,target=target),output/'protection_policy.json')
    ds,opt,pipe,model,scene=make_model(config,output,output/'latest.pth' if args.resume else None)
    schedule=RIRDPRA(config['rir_dpra'],target);remaining=[];events=[];checks=[];first_iteration=schedule.start
    if args.resume:
        state,extra=CompressionState.load(model,str(output/'compression_state.pt'))
        if extra['config']!=config or extra['checkpoint_sha256']!=sha256_file(output/'latest.pth'):raise ValueError('Resume settings/checkpoint differ')
        if extra['metadata_hash']!=sha256_file(shared/'adaptive_metadata.npz') or extra['input_hash']!=sha256_file(path(config['input_ply'])):raise ValueError('Resume source changed')
        if extra['reference_hash']!=sha256_file(output/'probe_reference.pt'):raise ValueError('Original teacher probes changed')
        first_iteration=extra['virtual_iteration'];remaining=extra['remaining'];events=extra['events'];checks=extra['checks']
        schedule.load_state_dict(extra['schedule']);restore_rng_state(extra['rng'])
    else:state=CompressionState(model,metadata,pool_rounds=config['budget_reallocation']['removed_pool_rounds'])
    cameras=scene.getTrainCameras()
    if [Path(c.image_name).stem for c in cameras]!=meta_manifest['train_frames']:raise ValueError('Training split/order differs')
    if [Path(c.image_name).stem for c in scene.getTestCameras()]!=meta_manifest['test_frames_excluded']:raise ValueError('Test split differs')
    risks=load_risk_maps(cameras,shared/'structure_maps')
    rois=TrainingStructureROIs(str(path(config['legacy_structure_roi_dir'])),cameras) if config.get('preserve_v3_losses',True) else None
    background=torch.tensor([0.,0.,0.],device='cuda')
    probes=TrainProbes(cameras,shared,model,pipe,background,risks)
    if args.resume:probes.reference=torch.load(str(output/'probe_reference.pt'),map_location='cpu')
    else:atomic_torch_save(probes.reference,output/'probe_reference.pt')
    manifest=dict(status='running',method='PUP_RIR_DPRA_v31_protected',config=config,source_count=source_count,target=target,
        events=events,rir_checks=checks,train_views=[c.image_name for c in cameras],test_views=[c.image_name for c in scene.getTestCameras()],
        virtual_start=schedule.start,virtual_end=schedule.end,score='PUP sum(log(svdvals(F_xyz_logscale_6x6))); all61 train views; no damping or frequency/ROI additions',
        input_ply_sha256=sha256_file(path(config['input_ply'])),metadata_sha256=sha256_file(shared/'adaptive_metadata.npz'))
    atomic_json_save(manifest,output/'run_args.json');started=time.monotonic();last_prune_iteration=schedule.start
    if events:last_prune_iteration=events[-1]['iteration']
    for iteration in range(first_iteration+1,schedule.end+1):
        model.update_learning_rate(iteration)
        if not remaining:remaining=list(range(len(cameras)))
        c=cameras[remaining.pop(random.randrange(len(remaining)))]
        image=render(c,model,pipe,background)['render'];gt=c.original_image.to(image.device)
        loss=(1-opt.lambda_dssim)*l1_loss(image,gt)+opt.lambda_dssim*(1-ssim(image,gt))
        if rois is not None:
            region,edge=rois.losses(c.image_name,image,gt);boundary,thin=rois.structure_losses(c.image_name,image,gt)
            weights=config['recovery_losses']
            loss=loss+weights['instance_roi']*region+weights['instance_edge']*edge+weights['boundary']*boundary+weights['thin']*thin
        if config['structure_loss']['enabled']:
            risk=risks[c.image_name].to(image.device)
            loss=loss+config['structure_loss']['weight']*(risk*(image-gt).abs().mean(0)).mean()
        if not torch.isfinite(loss):raise FloatingPointError('Nonfinite training loss')
        loss.backward();model.optimizer.step();model.optimizer.zero_grad(set_to_none=True)
        if iteration==schedule.pruning_start:schedule.initialize(iteration,state.count)
        if schedule.is_check(iteration):
            current_loss=reconstruction_detection(scene,model,pipe,background,ds,rois,config['recovery_losses']['instance_roi'] if rois else 0.)
            decision=schedule.decide(iteration,state.count,current_loss);checks.append(decision)
            print('RIR CHECK '+json.dumps(decision),flush=True)
            if decision['trigger'] and state.count>target:
                round_id=len(events)+1;before=state.count;round_target=decision['round_target']
                topo=build_topology(model.get_xyz.detach().cpu().numpy(),state.metadata['instance_ids'],
                                    state.metadata['confidence'],state.metadata['boundary_mask'],config['topology_protection'])
                state.metadata['topology_critical']=np.asarray(topo['critical_mask'],bool)
                state.metadata['linearity']=np.asarray(topo['linearity'],np.float32)
                state.metadata['principal_direction']=np.asarray(topo['direction'],np.float32)
                state.metadata['thin_candidate_mask']=np.asarray(topo['thin_candidate_mask'],bool)
                scores=pup_pruning_scores(cameras,model,pipe,background,resolution=config['pup_fisher_resolution']).cpu().numpy()
                score_path=output/'rounds'/('round_%03d_pup_scores.npz'%round_id)
                score_path.parent.mkdir(parents=True,exist_ok=True)
                np.savez_compressed(str(score_path),scores=scores,row_ids=state.metadata['row_ids'])
                reserve=int(round_target*config['budget_reallocation']['reserve_ratio']) if config['budget_reallocation']['enabled'] else 0
                temporary_target=max(1,round_target-reserve)
                # Permit a small safe merge tranche after feasibility pruning.
                merge_budget=min(int(before*config['gaussian_merging'].get('max_fraction',.005)),before-round_target) if config['gaussian_merging']['enabled'] else 0
                prune_goal=before-temporary_target-merge_budget
                mask,audit=select_feasible_prune(scores,state.metadata['instance_ids'],amin,prune_goal,
                            boundary_mask=state.metadata['boundary_mask'] if config['boundary_protection']['enabled'] else None,
                            boundary_min_counts=bmin,boundary_mode=config['boundary_protection']['mode'],topology=topo,allow_negative_infinity=True)
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
                # A lack of redundant merge candidates consumes ordinary global PUP budget.
                if state.count>temporary_target:
                    topo=build_topology(model.get_xyz.detach().cpu().numpy(),state.metadata['instance_ids'],
                                        state.metadata['confidence'],state.metadata['boundary_mask'],config['topology_protection'])
                    state.metadata['topology_critical']=np.asarray(topo['critical_mask'],bool)
                    # Merged rows receive a freshly computed PUP Fisher score, never an invented rank.
                    refreshed=pup_pruning_scores(cameras,model,pipe,background,resolution=config['pup_fisher_resolution']).cpu().numpy()
                    extra_mask,extra_audit=select_feasible_prune(refreshed,state.metadata['instance_ids'],amin,state.count-temporary_target,
                        boundary_mask=state.metadata['boundary_mask'] if config['boundary_protection']['enabled'] else None,
                        boundary_min_counts=bmin,boundary_mode=config['boundary_protection']['mode'],topology=topo,allow_negative_infinity=True)
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
                    # (zero-score ties are deterministic), not by per-instance PUP ranking.
                    chosen=ordered[:min(reserve,round_target-state.count)]
                    restored=state.restore(chosen,round_id,round_target)
                    restore_stats=dict(positive_candidates=int(len(positive)),selected_positive=int(np.sum(q[ids[chosen]]>0)),
                                       pool_size=int(len(ids)),q_max=float(values.max()) if len(values) else 0.)
                    del pool, q, ids, values, ordered, positive, eligible, chosen
                if state.count!=round_target:raise RuntimeError('Round target mismatch %d != %d'%(state.count,round_target))
                audit_quotas(state.metadata,amin,bmin)
                measured,_=probes.measure(model)
                event=dict(round_id=round_id,iteration=iteration,before=before,after_pruning=after_prune,after_merging=after_merge,
                    temporary_count=after_compression,after_reallocation=state.count,target=round_target,
                    pruned=before-after_prune+after_merge-after_compression,merged=int(merged),restored=int(restored),
                    recovery_steps=iteration-last_prune_iteration,total_steps=iteration-schedule.start,
                    rir=decision,selection=audit,topology=topo.get('audit',{}),restoration=restore_stats,
                    train_probe_after_compression=measured,quota_audit=audit_quotas(state.metadata,amin,bmin),
                    pup_scores=dict(path=str(score_path),sha256=sha256_file(score_path),negative_infinity_count=int(np.isneginf(scores).sum()),count=len(scores)))
                events.append(event);last_prune_iteration=iteration
                atomic_json_save(event,output/'rounds'/('round_%03d.json'%round_id))
                schedule.complete(iteration,state.count,current_loss)
                print('PUP RIR/DPRA PRUNED '+json.dumps({k:event[k] for k in ('iteration','before','after_reallocation','merged','restored')}),flush=True)
            elif decision['trigger']:
                # Original loop can schedule a zero-rate extra event after reaching target.
                # No mutation or new score computation is necessary at a zero budget.
                schedule.complete(iteration,state.count,current_loss)
        save=(iteration-schedule.start)%500==0 or iteration==schedule.end or (events and events[-1]['iteration']==iteration)
        if save:
            checkpoint=output/'latest.pth';atomic_torch_save((model.capture(),iteration),checkpoint)
            extra=dict(config=config,checkpoint_sha256=sha256_file(checkpoint),input_hash=manifest['input_ply_sha256'],
                metadata_hash=manifest['metadata_sha256'],reference_hash=sha256_file(output/'probe_reference.pt'),
                virtual_iteration=iteration,remaining=remaining,events=events,checks=checks,schedule=schedule.state_dict(),rng=capture_rng_state())
            state.save(str(output/'compression_state.pt'),extra=extra)
            manifest.update(events=events,rir_checks=checks,completed_steps=iteration-schedule.start,current_gaussians=state.count,last_checkpoint=str(checkpoint))
            atomic_json_save(manifest,output/'run_args.json')
        if iteration%100==0:print('ITER %d N=%d loss=%.7f'%(iteration,state.count,float(loss.detach())),flush=True)
        if args.smoke_rounds and len(events)>=args.smoke_rounds:break
    complete=state.count==target and iteration==schedule.end
    if not complete and not args.smoke_rounds:raise RuntimeError('RIR/DPRA did not reach target; no quota was relaxed')
    checkpoint=output/('chkpnt%d.pth'%iteration);atomic_torch_save((model.capture(),iteration),checkpoint)
    ply=output/'point_cloud'/('iteration_%d'%iteration)/'point_cloud.ply';atomic_write(str(ply),model.save_ply)
    np.savez_compressed(str(output/'compression_metadata.npz'),**state.metadata)
    final_metrics=evaluate(scene,model,pipe,background,ds,iteration);atomic_json_save(final_metrics,output/'metrics_float.json')
    manifest.update(status='complete' if complete else 'smoke_complete',completed_steps=iteration-schedule.start,
        final_gaussians=state.count,last_checkpoint=str(checkpoint),last_ply=str(ply),final_iteration=iteration,
        elapsed_seconds=time.monotonic()-started,quota_audit=audit_quotas(state.metadata,amin,bmin),
        final_schedule=schedule.state_dict())
    atomic_json_save(manifest,output/'run_args.json')
    print('PUP RIR/DPRA '+manifest['status']+' '+json.dumps(final_metrics),flush=True)


if __name__=='__main__':main()
