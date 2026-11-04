#!/usr/bin/env python3
"""Build sparse contribution-weighted SAM instances and structure priors.

Only declared training frames are read. Native-resolution alpha*T accumulation
uses every contributing Gaussian, not only a pixel's dominant contributor.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from PIL import Image
from utils.adaptive_metadata import (load_kernel,accumulate_support,structure_risk_map,weighted_jaccard_pairs,UnionFind,aggregate_assignments,instance_statistics)
from utils.structure_scene_io import read_ply,read_colmap,sha256_file
from utils.structure_visibility import prepare_from_ply
from utils.instance_pruning import xyz_sha256


def write_json(path,value):
    Path(path).write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n')


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('source_ply','images','colmap','sam_masks','train_frames','split','visibility_cache','visibility_manifest','output_dir'):
        p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--association_threshold',type=float,default=.15)
    p.add_argument('--instance_threshold',type=float,default=.55)
    p.add_argument('--boundary_threshold',type=float,default=.55)
    p.add_argument('--association_neighbors',type=int,default=12)
    p.add_argument('--threads',type=int,default=4)
    p.add_argument('--limit_views',type=int,default=0,help='Benchmark only; produces no final metadata when limited.')
    args=p.parse_args(argv)
    out=args.output_dir;out.mkdir(parents=True,exist_ok=True)
    for sub in ('mask_support','structure_maps','contribution_cells','cpu_build'):(out/sub).mkdir(exist_ok=True)
    frames=sorted(Path(name).stem for name in args.train_frames.read_text().splitlines() if name.strip())
    split=json.loads(args.split.read_text())
    if len(frames)!=len(set(frames)) or frames!=sorted(split['train']) or set(frames)&set(split['test']):raise ValueError('Train/test declaration mismatch')
    all_frames=frames[:];probe=[frames[i] for i in np.linspace(0,len(frames)-1,8,dtype=int)]
    if args.limit_views:frames=frames[:args.limit_views]
    cameras=read_colmap(args.colmap);ply,_=read_ply(args.source_ply)
    xyz=np.column_stack([ply[k] for k in ('x','y','z')]).astype(np.float32)
    source_hash=sha256_file(args.source_ply);xyz_hash=xyz_sha256(xyz)
    cache_manifest=json.loads(args.visibility_manifest.read_text())
    if cache_manifest['source_ply_sha256']!=source_hash or cache_manifest['source_xyz_sha256']!=xyz_hash:raise ValueError('Cached rendered depth belongs to another source')
    kernel=load_kernel(out/'cpu_build');gaussians=None;n=len(xyz)
    total=np.zeros(n,np.float64);risk_mass=np.zeros(n,np.float64)
    supports={};areas={};frame_stats={}
    for frame_index,frame in enumerate(frames):
        start=time.time();support_file=out/'mask_support'/(frame+'.npz');record_file=out/'mask_support'/(frame+'.json')
        with np.load(args.sam_masks/(frame+'.npz')) as d:labels=np.asarray(d['labels'],np.int32)
        areas[frame]=np.bincount(labels.ravel())/labels.size
        camera=cameras[frame]
        camera_hash=hashlib.sha256((json.dumps(camera['camera'],sort_keys=True)+camera['R'].tobytes().hex()+camera['t'].tobytes().hex()).encode()).hexdigest()
        depth_record=json.loads((args.visibility_cache/(frame+'_visibility.json')).read_text())
        if depth_record['source_xyz_sha256']!=xyz_hash or depth_record['camera_sha256']!=camera_hash:raise ValueError('Depth camera/source misalignment')
        image=args.images/camera['name']
        fingerprint=dict(source_ply_sha256=source_hash,camera_sha256=camera_hash,sam_sha256=sha256_file(args.sam_masks/(frame+'.npz')),image_sha256=sha256_file(image),depth_sha256=sha256_file(args.visibility_cache/(frame+'_visibility.npz')),algorithm='native anisotropic full sparse alpha*T v1',implementation_sha256=sha256_file(Path(__file__).parent/'utils'/'adaptive_metadata.py'),probe_cell_size=16,probe=frame in probe)
        usable=support_file.exists() and record_file.exists() and json.loads(record_file.read_text()).get('fingerprint')==fingerprint
        if usable:
            with np.load(support_file) as d:
                supports[frame]={k:np.array(d[k]) for k in ('gaussian_ids','labels','mass')}
                total+=d['total'];risk_mass+=d['risk_mass']
            frame_stats[frame]=json.loads(record_file.read_text())
        else:
            with np.load(args.visibility_cache/(frame+'_visibility.npz')) as d:depth=np.array(d['depth']);alpha=np.array(d['alpha'])
            rgb=np.asarray(Image.open(image).convert('RGB'),np.float32)/255
            maps=structure_risk_map(rgb,labels,depth,alpha)
            np.savez_compressed(out/'structure_maps'/(frame+'.npz'),frame=np.asarray(frame),risk=maps['structure_risk'],**maps)
            if gaussians is None:gaussians=prepare_from_ply(ply)
            result=accumulate_support(gaussians,camera,labels,maps['structure_risk'],kernel,args.threads,16,frame in probe)
            supports[frame]={k:result['support_'+k] for k in ('gaussian_ids','labels','mass')}
            total+=result['total'];risk_mass+=result['risk_mass']
            np.savez_compressed(support_file,total=result['total'],risk_mass=result['risk_mass'],**supports[frame])
            if frame in probe:
                np.savez_compressed(out/'contribution_cells'/(frame+'.npz'),gaussian_ids=result['cell_gaussian_ids'],cell_ids=result['cell_labels'],mass=result['cell_mass'],cell_size=np.asarray(16,np.int32),image_shape=np.asarray(labels.shape,np.int32),grid_shape=np.asarray(((labels.shape[0]+15)//16,(labels.shape[1]+15)//16),np.int32),frame=np.asarray(frame))
            frame_stats[frame]=dict(fingerprint=fingerprint,seconds=time.time()-start,mask_pairs=len(result['support_mass']),cell_pairs=len(result.get('cell_mass',[])),visible_gaussians=int(np.count_nonzero(result['total'])),total_contribution=float(result['total'].sum(dtype=np.float64)))
            write_json(record_file,frame_stats[frame])
        print(json.dumps(dict(frame=frame,index=frame_index+1,views=len(frames),seconds=round(time.time()-start,2),pairs=len(supports[frame]['mass']),cache=usable)),flush=True)
        write_json(out/'preprocessing_status.json',dict(status='rasterizing',complete_views=frame_index+1,train_views=len(all_frames),last_frame=frame))
    if args.limit_views:
        print('Benchmark only; final metadata intentionally not emitted.',flush=True);return
    # Nearby cameras bound the association cost. Shared Gaussian support alone
    # determines every edge weight; camera proximity only generates candidates.
    centers=np.stack([-cameras[f]['R'].T@cameras[f]['t'] for f in frames])
    distance=np.sum((centers[:,None]-centers[None,:])**2,axis=2);np.fill_diagonal(distance,np.inf)
    pairs=set()
    for i in range(len(frames)):
        for j in np.argsort(distance[i])[:args.association_neighbors]:pairs.add(tuple(sorted((i,int(j)))))
    offsets={};node_frames=[];nodes=[];offset=0
    for i,f in enumerate(frames):
        offsets[f]=offset
        for label in range(1,len(areas[f])):nodes.append((f,label));node_frames.append(i)
        offset+=len(areas[f])-1
    edges=[]
    for pair_index,(i,j) in enumerate(sorted(pairs)):
        a,b=frames[i],frames[j];x,y,score=weighted_jaccard_pairs(supports[a],supports[b],kernel)
        for lx,ly,s in zip(x,y,score):
            if s>args.association_threshold and lx>0 and ly>0:edges.append((float(s),offsets[a]+int(lx)-1,offsets[b]+int(ly)-1))
        if pair_index%30==0:print(json.dumps(dict(association_pair=pair_index+1,pairs=len(pairs),edges=len(edges))),flush=True)
    uf=UnionFind(len(nodes));views=[{v} for v in node_frames];conflicts=0;accepted=0
    for score,a,b in sorted(edges,reverse=True):
        ra,rb=uf.find(a),uf.find(b)
        if ra==rb:continue
        if views[ra]&views[rb]:conflicts+=1;continue
        merged=views[ra]|views[rb];uf.union(ra,rb);views[uf.find(ra)]=merged;accepted+=1
    roots=sorted({uf.find(i) for i in range(len(nodes))});root_id={r:i for i,r in enumerate(roots)}
    mapping={f:np.full(len(areas[f]),-1,np.int32) for f in frames}
    node_records=[]
    for i,(f,l) in enumerate(nodes):
        iid=root_id[uf.find(i)];mapping[f][l]=iid
        node_records.append(dict(frame=f,local_region=l,instance_id=iid))
    print('Aggregating Gaussian instance probabilities...',flush=True)
    assigned,confidence,view_count,mask_visibility=aggregate_assignments(supports,mapping,n,args.instance_threshold,2)
    rows,fragility=instance_statistics(assigned,xyz,mapping,areas,view_count,total)
    risk=np.divide(risk_mass,total,out=np.zeros(n,np.float64),where=total>0).astype(np.float32)
    boundary=(risk>args.boundary_threshold)&(total>0)
    manifest=dict(schema_version=1,method='full sparse native alpha*T mask support + weighted Jaccard UnionFind',source_ply=str(args.source_ply.resolve()),source_ply_sha256=source_hash,source_xyz_sha256=xyz_hash,source_count=n,train_frames=frames,test_frames_excluded=split['test'],probe_frames=probe,association_threshold=args.association_threshold,association_neighbors=args.association_neighbors,association_candidate_pairs=len(pairs),association_edges_above_threshold=len(edges),accepted_unions=accepted,same_view_component_conflicts_rejected=conflicts,instance_threshold=args.instance_threshold,min_visible_views=2,instance_count=len(rows),unknown_gaussians=int(np.count_nonzero(assigned<0)),boundary_threshold=args.boundary_threshold,boundary_gaussians=int(boundary.sum()),structure_weights=dict(sam=.5,rendered_depth=.25,rgb=.25),depth_kind='source-rendered dominant alpha*T Gaussian camera depth, native resolution',contribution_cells=dict(size=16,views=probe,meaning='full native alpha*T mass summed per Gaussian per 16px cell; downstream hard-map cell mean is a spatial approximation'),limitations=['SAM regions are class-free pseudo instances; these are not manual semantic labels.','Association candidates are 12 nearest camera centers per train view; same-view region merges are rejected to avoid transitive collapse.','Structure depth is the rendered dominant-contributor depth, not monocular estimated or expected depth.','Confidence and survival constraints do not guarantee semantic identity or final rendered object visibility.'])
    np.savez_compressed(out/'adaptive_metadata.npz',original_ids=np.arange(n,dtype=np.int64),instance_ids=assigned,confidence=confidence,structure_risk=risk,boundary_mask=boundary,visibility=total.astype(np.float32),view_count=view_count,instance_fragility=fragility,source_xyz_sha256=np.asarray(xyz_hash),source_ply_sha256=np.asarray(source_hash),manifest_json=np.asarray(json.dumps(manifest)),instance_stat_ids=np.asarray([r['instance_id'] for r in rows],np.int32),instance_initial_count=np.asarray([r['initial_count'] for r in rows],np.int64),instance_extent3d=np.asarray([r['extent3d'] for r in rows],np.float32),instance_projected_area=np.asarray([r['projected_area'] for r in rows],np.float32),instance_visible_views=np.asarray([r['visible_views'] for r in rows],np.int32),instance_stat_fragility=np.asarray([r['fragility'] for r in rows],np.float32))
    write_json(out/'instance_stats.json',dict(instances=rows))
    write_json(out/'mask_associations.json',dict(nodes=node_records))
    write_json(out/'metadata_manifest.json',manifest)
    write_json(out/'preprocessing_status.json',dict(status='complete',complete_views=len(frames),train_views=len(frames),metadata_sha256=sha256_file(out/'adaptive_metadata.npz')))
    print(json.dumps(manifest,indent=2),flush=True)


if __name__=='__main__':main()
