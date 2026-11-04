"""Sparse train-only alpha*T metadata for adaptive Gaussian compression.

No tensor of shape [Gaussians,H,W] is allocated. Native anisotropic footprints
and front-to-back transmittance follow utils.structure_visibility. SAM local
IDs are associated by weighted support overlap, never by their numeric value.
"""
import ctypes
import hashlib
import json
import os
from pathlib import Path
import subprocess
import numpy as np

from utils.structure_visibility import _quadratic

_CPP = r'''
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>
#include <utility>
#include <omp.h>
struct Entry { uint64_t key; float value; };
static std::vector<Entry> supports, cells;
static void reduce(std::vector<Entry>& v) {
 std::sort(v.begin(),v.end(),[](const Entry&a,const Entry&b){return a.key<b.key;});
 size_t j=0;
 for(size_t i=0;i<v.size();++i) {
   if(j && v[j-1].key==v[i].key) v[j-1].value+=v[i].value;
   else v[j++]=v[i];
 }
 v.resize(j);
}
extern "C" long long raster(const float* p,const int32_t* ids,const int32_t* order,
 int n,int W,int H,int threads,const int32_t* labels,int L,const float* risk,
 float* total,float* riskmass,int cellsize,int savecells) {
 const int TS=16,nx=(W+TS-1)/TS,ny=(H+TS-1)/TS;
 const int ncx=(W+cellsize-1)/cellsize,ncy=(H+cellsize-1)/cellsize;
 std::vector<std::vector<int32_t>> bins(nx*ny);
 long long refs=0;
 for(int k=0;k<n;++k) {
  int g=order[k]; const float*q=p+10*g;
  int x0=std::min(nx,std::max(0,(int)((q[0]-q[7])/TS)));
  int x1=std::min(nx,std::max(0,(int)((q[0]+q[7]+TS-1)/TS)));
  int y0=std::min(ny,std::max(0,(int)((q[1]-q[7])/TS)));
  int y1=std::min(ny,std::max(0,(int)((q[1]+q[7]+TS-1)/TS)));
  for(int y=y0;y<y1;++y)for(int x=x0;x<x1;++x){bins[y*nx+x].push_back(g);++refs;}
 }
 threads=std::max(1,std::min(4,threads)); omp_set_num_threads(threads);
 std::vector<std::vector<Entry>> ls(threads),lc(threads);
 #pragma omp parallel for schedule(dynamic,1)
 for(int tile=0;tile<nx*ny;++tile){
  int tid=omp_get_thread_num(),x0=(tile%nx)*TS,y0=(tile/nx)*TS;
  int x1=std::min(W,x0+TS),y1=std::min(H,y0+TS);
  float T[256];bool done[256]; for(int i=0;i<256;++i){T[i]=1;done[i]=false;}
  for(int g:bins[tile]){
   const float*q=p+10*g; int gid=ids[g]; float sum=0,rsum=0;
   std::vector<std::pair<int,float>> lm,cm;
   int xmin=std::max(x0,(int)std::ceil(q[0]-q[8]));
   int xmax=std::min(x1-1,(int)std::floor(q[0]+q[8]));
   int ymin=std::max(y0,(int)std::ceil(q[1]-q[9]));
   int ymax=std::min(y1-1,(int)std::floor(q[1]+q[9]));
   for(int y=ymin;y<=ymax;++y)for(int x=xmin;x<=xmax;++x){
    int local=(y-y0)*TS+x-x0;if(done[local])continue;
    float dx=q[0]-x,dy=q[1]-y;
    float power=-.5f*(q[3]*dx*dx+q[5]*dy*dy)-q[4]*dx*dy;
    if(power>0)continue;float a=std::min(.99f,q[6]*std::exp(power));
    if(a<1.f/255.f)continue;float next=T[local]*(1-a);
    if(next<.0001f){done[local]=true;continue;}
    int pixel=y*W+x;float w=T[local]*a;sum+=w;rsum+=w*risk[pixel];
    int label=labels[pixel];
    if(label>0){bool found=false;for(auto&v:lm)if(v.first==label){v.second+=w;found=true;break;}
     if(!found)lm.emplace_back(label,w);}
    if(savecells){int cell=(y/cellsize)*ncx+x/cellsize;bool found=false;
     for(auto&v:cm)if(v.first==cell){v.second+=w;found=true;break;}
     if(!found)cm.emplace_back(cell,w);}
    T[local]=next;
   }
   if(sum>0){
    #pragma omp atomic update
    total[gid]+=sum;
    #pragma omp atomic update
    riskmass[gid]+=rsum;
    for(auto&v:lm)ls[tid].push_back({uint64_t(gid)*L+v.first,v.second});
    for(auto&v:cm)lc[tid].push_back({uint64_t(gid)*(ncx*ncy)+v.first,v.second});
   }
  }
 }
 supports.clear();cells.clear();
 for(int t=0;t<threads;++t){supports.insert(supports.end(),ls[t].begin(),ls[t].end());cells.insert(cells.end(),lc[t].begin(),lc[t].end());}
 reduce(supports);reduce(cells);return refs;
}
extern "C" long long sparse_count(int mode){return mode?cells.size():supports.size();}
extern "C" void copy_sparse(int mode,int stride,int32_t* gid,int32_t* label,float* mass){
 const auto&v=mode?cells:supports;
 for(size_t i=0;i<v.size();++i){gid[i]=v[i].key/stride;label[i]=v[i].key%stride;mass[i]=v[i].value;}
}
extern "C" void pair_intersection(const int32_t* ga,const int32_t* la,const float* wa,int na,
 const int32_t* gb,const int32_t* lb,const float* wb,int nb,int Lb,float* out){
 int a=0,b=0;
 while(a<na && b<nb){
  if(ga[a]<gb[b]){++a;continue;}if(gb[b]<ga[a]){++b;continue;}
  int ae=a+1,be=b+1;while(ae<na && ga[ae]==ga[a])++ae;while(be<nb && gb[be]==gb[b])++be;
  for(int i=a;i<ae;++i)for(int j=b;j<be;++j)out[la[i]*Lb+lb[j]]+=std::min(wa[i],wb[j]);
  a=ae;b=be;
 }
}
'''


def load_kernel(build_dir):
    build_dir = Path(build_dir)
    build_dir.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256(_CPP.encode()).hexdigest()[:16]
    source = build_dir / ('adaptive_' + digest + '.cpp')
    library = build_dir / ('adaptive_' + digest + '.so')
    if not library.exists():
        source.write_text(_CPP)
        subprocess.run([os.environ.get('CXX', '/usr/bin/g++'), '-O3', '-std=c++17', '-shared', '-fPIC', '-fopenmp', str(source), '-o', str(library)], check=True, capture_output=True)
    lib = ctypes.CDLL(str(library))
    fp = np.ctypeslib.ndpointer(dtype=np.float32, flags='C_CONTIGUOUS')
    ip = np.ctypeslib.ndpointer(dtype=np.int32, flags='C_CONTIGUOUS')
    ci = ctypes.c_int
    lib.raster.argtypes = [fp,ip,ip,ci,ci,ci,ci,ip,ci,fp,fp,fp,ci,ci]
    lib.raster.restype = ctypes.c_longlong
    lib.sparse_count.argtypes = [ci]
    lib.sparse_count.restype = ctypes.c_longlong
    lib.copy_sparse.argtypes = [ci,ci,ip,ip,fp]
    lib.pair_intersection.argtypes = [ip,ip,fp,ci,ip,ip,fp,ci,ci,fp]
    return lib


def project_native(gaussians, camera, chunk_size=250000):
    """Native projection shared mathematically with structure_visibility.py."""
    c = camera['camera']; width, height = c['width'], c['height']
    fx, fy, cx, cy = c['fx'], c['fy'], c['cx']-.5, c['cy']-.5
    R, t = np.asarray(camera['R'], np.float32), np.asarray(camera['t'], np.float32)
    xyz, cov, opacity = [gaussians[k] for k in ('xyz','covariance','opacity')]
    projected, original = [], []
    for start in range(0,len(xyz),chunk_size):
        stop=min(len(xyz),start+chunk_size); cp=xyz[start:stop]@R.T+t
        z=cp[:,2]; safe=np.where(z>.2,z,1); xx,yy=cp[:,0]/safe,cp[:,1]/safe
        u,v=fx*xx+cx,fy*yy+cy
        jx=(fx/safe)[:,None]*(R[0]-np.clip(xx,-1.3*width/(2*fx),1.3*width/(2*fx))[:,None]*R[2])
        jy=(fy/safe)[:,None]*(R[1]-np.clip(yy,-1.3*height/(2*fy),1.3*height/(2*fy))[:,None]*R[2])
        cc=cov[start:stop]; a=_quadratic(jx,jx,cc)+.3;b=_quadratic(jx,jy,cc);d=_quadratic(jy,jy,cc)+.3
        det=np.maximum(a*d-b*b,1e-20);mid=.5*(a+d)
        radius=np.ceil(3*np.sqrt(np.maximum(mid+np.sqrt(np.maximum(.1,mid*mid-det)),.01)))
        eligible=(z>.2)&(opacity[start:stop]>=1/255)&np.isfinite(radius)
        nx,ny=(width+15)//16,(height+15)//16
        keep=eligible&(np.clip(np.trunc((u+radius+15)/16),0,nx)>np.clip(np.trunc((u-radius)/16),0,nx))&(np.clip(np.trunc((v+radius+15)/16),0,ny)>np.clip(np.trunc((v-radius)/16),0,ny))
        idx=np.flatnonzero(keep); limit=np.maximum(0,2*np.log(np.maximum(255*opacity[start:stop][idx],1)))
        projected.append(np.column_stack((u[idx],v[idx],z[idx],d[idx]/det[idx],-b[idx]/det[idx],a[idx]/det[idx],opacity[start:stop][idx],radius[idx],np.sqrt(limit*a[idx])+1e-4,np.sqrt(limit*d[idx])+1e-4)))
        original.append((idx+start).astype(np.int32))
    p=np.ascontiguousarray(np.concatenate(projected),np.float32)
    return p,np.ascontiguousarray(np.concatenate(original),np.int32),np.ascontiguousarray(np.argsort(p[:,2],kind='stable'),np.int32)


def accumulate_support(gaussians,camera,labels,risk,kernel,threads=4,cell_size=16,save_cells=False):
    p,ids,order=project_native(gaussians,camera)
    labels=np.ascontiguousarray(labels,np.int32);risk=np.ascontiguousarray(risk,np.float32)
    if labels.shape!=risk.shape or labels.min()<0:raise ValueError('Invalid label/risk maps')
    h,w=labels.shape; L=int(labels.max())+1;n=len(gaussians['xyz'])
    total=np.zeros(n,np.float32);riskmass=np.zeros(n,np.float32)
    refs=kernel.raster(p,ids,order,len(ids),w,h,threads,labels,L,risk,total,riskmass,cell_size,int(save_cells))
    result={'total':total,'risk_mass':riskmass,'tile_references':int(refs)}
    for mode,prefix,stride in ((0,'support',L),(1,'cell',((w+cell_size-1)//cell_size)*((h+cell_size-1)//cell_size))):
        if mode and not save_cells:continue
        count=kernel.sparse_count(mode);g=np.empty(count,np.int32);l=np.empty(count,np.int32);a=np.empty(count,np.float32)
        kernel.copy_sparse(mode,stride,g,l,a)
        result[prefix+'_gaussian_ids']=g;result[prefix+'_labels']=l;result[prefix+'_mass']=a
    return result


def normalized_sobel(value, valid=None):
    import cv2
    value=np.asarray(value,np.float32)
    if valid is None:valid=np.isfinite(value)
    value=np.where(valid,value,0)
    gx=cv2.Sobel(value,cv2.CV_32F,1,0,ksize=3)/8
    gy=cv2.Sobel(value,cv2.CV_32F,0,1,ksize=3)/8
    edge=np.hypot(gx,gy)
    inside=cv2.erode(valid.astype(np.uint8),np.ones((3,3),np.uint8))>0
    edge[~inside]=0
    scale=float(np.quantile(edge[inside],.95)) if inside.any() else 0
    return np.clip(edge/max(scale,1e-6),0,1).astype(np.float32)


def structure_risk_map(rgb,labels,depth,alpha,weights=(.5,.25,.25)):
    import cv2
    labels=np.asarray(labels);boundary=np.zeros(labels.shape,np.uint8)
    by=labels[1:]!=labels[:-1];bx=labels[:,1:]!=labels[:,:-1]
    boundary[1:]|=by;boundary[:-1]|=by;boundary[:,1:]|=bx;boundary[:,:-1]|=bx
    boundary=cv2.dilate(boundary,np.ones((3,3),np.uint8))
    gray=cv2.cvtColor(np.asarray(rgb,np.float32),cv2.COLOR_RGB2GRAY)
    rgb_edge=normalized_sobel(gray)
    depth_edge=normalized_sobel(depth,np.isfinite(depth)&(alpha>.5))
    risk=weights[0]*boundary+weights[1]*depth_edge+weights[2]*rgb_edge
    return dict(structure_risk=np.asarray(risk,np.float32),sam_boundary=boundary.astype(bool),depth_edge=depth_edge,rgb_edge=rgb_edge)


class UnionFind:
    def __init__(self,n):self.parent=np.arange(n,dtype=np.int32)
    def find(self,x):
        while self.parent[x]!=x:self.parent[x]=self.parent[self.parent[x]];x=int(self.parent[x])
        return x
    def union(self,a,b):
        a,b=self.find(a),self.find(b)
        if a!=b:self.parent[max(a,b)]=min(a,b)


def weighted_jaccard_pairs(a,b,kernel=None):
    """All positive mask overlaps in a pair of views; sparse source-row joins."""
    La=int(a['labels'].max())+1 if len(a['labels']) else 1
    Lb=int(b['labels'].max())+1 if len(b['labels']) else 1
    intersection=np.zeros((La,Lb),np.float32)
    if kernel is not None:
        kernel.pair_intersection(a['gaussian_ids'],a['labels'],a['mass'],len(a['mass']),b['gaussian_ids'],b['labels'],b['mass'],len(b['mass']),Lb,intersection)
    else:
        # Small deterministic reference used by the unit tests.
        for g in np.intersect1d(a['gaussian_ids'],b['gaussian_ids']):
            ai=np.flatnonzero(a['gaussian_ids']==g);bi=np.flatnonzero(b['gaussian_ids']==g)
            for i in ai:
                for j in bi:intersection[a['labels'][i],b['labels'][j]]+=min(a['mass'][i],b['mass'][j])
    ma=np.bincount(a['labels'],weights=a['mass'],minlength=La)
    mb=np.bincount(b['labels'],weights=b['mass'],minlength=Lb)
    x,y=np.nonzero(intersection)
    sim=intersection[x,y]/np.maximum(ma[x]+mb[y]-intersection[x,y],1e-12)
    return x,y,sim


def aggregate_assignments(supports,mask_instance,n,threshold=.55,min_views=2):
    """Sparse Gaussian/global-ID sums. Unknown remains -1, including no support."""
    gids=[];iids=[];weights=[];views=np.zeros(n,np.uint16);visibility=np.zeros(n,np.float32)
    max_instance=max((int(v.max()) for v in mask_instance.values()),default=-1)+1
    for frame,support in supports.items():
        g=support['gaussian_ids'];k=mask_instance[frame][support['labels']];a=support['mass']
        valid=k>=0;gids.append(g[valid]);iids.append(k[valid]);weights.append(a[valid])
        views[np.unique(g[a>1e-4])]+=1
        visibility+=np.bincount(g,weights=a,minlength=n).astype(np.float32)
    g=np.concatenate(gids);k=np.concatenate(iids);w=np.concatenate(weights)
    key=g.astype(np.int64)*max(max_instance,1)+k
    order=np.argsort(key,kind='stable');key=key[order];w=w[order]
    starts=np.r_[0,np.flatnonzero(np.diff(key))+1] if len(key) else np.array([],np.int64)
    keys=key[starts];mass=np.add.reduceat(w,starts) if len(starts) else w
    gg=keys//max(max_instance,1);kk=keys%max(max_instance,1)
    total=np.bincount(gg,weights=mass,minlength=n)
    best=np.zeros(n,np.float32);assigned=np.full(n,-1,np.int32)
    # Stable sorting ascending makes the largest mass the last entry per row.
    ranked=np.argsort(mass,kind='stable');assigned[gg[ranked]]=kk[ranked];best[gg[ranked]]=mass[ranked]
    confidence=np.divide(best,total,out=np.zeros(n,np.float64),where=total>0).astype(np.float32)
    assigned[(confidence<threshold)|(views<min_views)]=-1
    return assigned,confidence,views,visibility


def instance_statistics(instance_ids,xyz,mask_instance,mask_areas,view_count,visibility):
    ids=np.unique(instance_ids[instance_ids>=0]);rows=[]
    for iid in ids:
        selected=instance_ids==iid;points=xyz[selected];center=points.mean(0)
        extent=float(np.sqrt(np.mean(np.sum((points-center)**2,axis=1))))
        areas=[];visible=0
        for frame,lut in mask_instance.items():
            local=np.flatnonzero(lut==iid)
            if len(local):areas.append(float(np.asarray(mask_areas[frame])[local].sum()));visible+=1
        rows.append(dict(instance_id=int(iid),initial_count=int(selected.sum()),extent3d=extent,projected_area=float(np.median(areas)) if areas else 0,visible_views=visible,mean_visibility=float(visibility[selected].mean()),mean_gaussian_views=float(view_count[selected].mean())))
    if not rows:return rows,np.zeros(len(xyz),np.float32)
    # Dataset-scale taus are derived solely from training-source statistics.
    taus={key:max(float(np.median([r[key] for r in rows])),1e-8) for key in ('initial_count','extent3d','projected_area','visible_views')}
    fragility=np.zeros(len(xyz),np.float32)
    for row in rows:
        row['fragility']=float(sum(np.exp(-row[key]/taus[key]) for key in taus)/len(taus))
        fragility[instance_ids==row['instance_id']]=row['fragility']
    return rows,fragility
