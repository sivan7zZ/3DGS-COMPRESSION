#!/usr/bin/env python3
"""CPU EWA Gaussian visibility for label-free instance association.

This is an independent, no-PyTorch CPU implementation. It uses the existing
3DGS parameters and never accesses semantic labels. The reference equations
and thresholds are in the local original project's
submodules/diff-gaussian-rasterization/cuda_rasterizer/forward.cu:
computeCov2D/computeCov3D, preprocessCUDA (+0.3 covariance diagonal), renderCUDA
(alpha <= .99, alpha >= 1/255, stop transmittance < .0001).

Approximation: the requested (optionally resized) camera is rendered with an
optional radius cap. Native 3-sigma tile bounds are evaluated, cutting each
pixel loop only where alpha is mathematically below 1/255. This is not a
bitwise reproduction of CUDA: floating-point order differs. Unlike center-only z buffering, this module
does evaluate anisotropic screen ellipses and depth-ordered alpha*T weights.
No image RGB, labels, instance IDs, or prior output labels enter visibility.

The only optional build dependency is an existing C++ compiler with OpenMP;
the caller supplies an experiment-local build directory under output/.
"""
from __future__ import annotations

import ctypes
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

import numpy as np


_CPP = r'''
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>
#include <omp.h>

// p = (u,v,z, inverseCovXX, inverseCovXY, inverseCovYY, opacity, radius,
//      alphaSupportX, alphaSupportY)
extern "C" long long raster_visibility(const float* p, const int32_t* ids,
    const int32_t* order, int n, int W, int H, int threads,
    int32_t* dominant, float* maxweight, float* depth, float* front,
    float* expected, float* alpha_out, int use_labels, const int32_t* labels,
    const int32_t* target, float* total_weight, float* matched_weight) {
  constexpr int TS=16;
  const int nx=(W+TS-1)/TS, ny=(H+TS-1)/TS;
  std::vector<std::vector<int32_t>> bins(nx*ny);
  long long references=0;
  // This global depth order also sorts every per-tile bin front to back.
  for(int k=0;k<n;++k) {
    int g=order[k]; const float* q=p+10*g;
    // Match auxiliary.h:getRect's integer truncation and exclusive tile max.
    int tx0=std::min(nx,std::max(0,(int)((q[0]-q[7])/TS)));
    int tx1=std::min(nx,std::max(0,(int)((q[0]+q[7]+TS-1)/TS)));
    int ty0=std::min(ny,std::max(0,(int)((q[1]-q[7])/TS)));
    int ty1=std::min(ny,std::max(0,(int)((q[1]+q[7]+TS-1)/TS)));
    for(int ty=ty0;ty<ty1;++ty)
      for(int tx=tx0;tx<tx1;++tx) {
        bins[ty*nx+tx].push_back(g); ++references;
      }
  }
  omp_set_num_threads(std::max(1,std::min(4,threads)));
  #pragma omp parallel for schedule(dynamic,1)
  for(int tile=0;tile<nx*ny;++tile) {
    const int x0=(tile%nx)*TS, y0=(tile/nx)*TS;
    const int x1=std::min(W,x0+TS), y1=std::min(H,y0+TS);
    float T[TS*TS]; bool done[TS*TS];
    for(int i=0;i<TS*TS;++i) { T[i]=1.f; done[i]=false; }
    for(int g: bins[tile]) {
      const float* q=p+10*g;
      const int original_id=ids[g];
      const int target_label=use_labels ? target[original_id] : 0;
      float local_total=0.f, local_matched=0.f;
      int xmin=std::max(x0,(int)std::ceil(q[0]-q[8]));
      int xmax=std::min(x1-1,(int)std::floor(q[0]+q[8]));
      int ymin=std::max(y0,(int)std::ceil(q[1]-q[9]));
      int ymax=std::min(y1-1,(int)std::floor(q[1]+q[9]));
      for(int y=ymin;y<=ymax;++y) for(int x=xmin;x<=xmax;++x) {
        int local=(y-y0)*TS+x-x0; if(done[local]) continue;
        float dx=q[0]-x, dy=q[1]-y;
        float power=-.5f*(q[3]*dx*dx+q[5]*dy*dy)-q[4]*dx*dy;
        if(power>0.f) continue;
        float a=std::min(.99f,q[6]*std::exp(power));
        if(a<1.f/255.f) continue;
        float next=T[local]*(1.f-a);
        // Match the native rasterizer's early-exit convention.
        if(next<.0001f) { done[local]=true; continue; }
        int pixel=y*W+x; float w=T[local]*a;
        if(!std::isfinite(front[pixel])) front[pixel]=q[2];
        expected[pixel]+=w*q[2];
        if(w>maxweight[pixel]) {
          maxweight[pixel]=w; dominant[pixel]=original_id; depth[pixel]=q[2];
        }
        if(use_labels) {
          local_total+=w;
          if(target_label>0 && labels[pixel]==target_label) local_matched+=w;
        }
        T[local]=next;
      }
      // One atomic per Gaussian/tile, not per contributing pixel.
      if(use_labels && local_total>0.f) {
        #pragma omp atomic update
        total_weight[original_id]+=local_total;
        if(local_matched>0.f) {
          #pragma omp atomic update
          matched_weight[original_id]+=local_matched;
        }
      }
    }
    for(int y=y0;y<y1;++y) for(int x=x0;x<x1;++x) {
      int pixel=y*W+x; float a=1.f-T[(y-y0)*TS+x-x0];
      alpha_out[pixel]=a;
      expected[pixel]=a>0.f ? expected[pixel]/a : INFINITY;
    }
  }
  return references;
}
'''


_KERNEL = None


def _load_kernel(build_dir=None):
    global _KERNEL
    if _KERNEL is not None:
        return _KERNEL
    if build_dir is None:
        raise ValueError('An explicit experiment-local build_dir is required')
    build_dir = Path(build_dir)
    build_dir.mkdir(parents=True, exist_ok=True)
    key = hashlib.sha256(_CPP.encode()).hexdigest()[:16]
    source = build_dir / f'visibility_{key}.cpp'
    library = build_dir / f'visibility_{key}.so'
    if not library.exists():
        source.write_text(_CPP)
        compiler = os.environ.get('CXX', '/usr/bin/g++')
        subprocess.run([compiler, '-O3', '-std=c++17', '-shared', '-fPIC',
                        '-fopenmp', str(source), '-o', str(library)], check=True,
                       capture_output=True, text=True)
    lib = ctypes.CDLL(str(library))
    f = lib.raster_visibility
    fp = np.ctypeslib.ndpointer(dtype=np.float32, flags='C_CONTIGUOUS')
    ip = np.ctypeslib.ndpointer(dtype=np.int32, flags='C_CONTIGUOUS')
    f.argtypes = [fp, ip, ip, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                  ctypes.c_int, ip, fp, fp, fp, fp, fp,ctypes.c_int,ip,ip,fp,fp]
    f.restype = ctypes.c_longlong
    _KERNEL = f
    return f


def prepare_gaussians(xyz, scales, quaternions, opacity, chunk_size=250000):
    """Prepare once. scales=exp(raw scale), opacity=sigmoid(raw opacity).

    Quaternion convention is w,x,y,z and is normalized here. Returns xyz,
    packed world covariance (xx,xy,xz,yy,yz,zz), and opacity, all float32.
    """
    xyz = np.ascontiguousarray(xyz, dtype=np.float32)
    n = len(xyz)
    scales = np.asarray(scales)
    quaternions = np.asarray(quaternions)
    opacity = np.ascontiguousarray(opacity, dtype=np.float32).reshape(-1)
    if xyz.shape != (n, 3) or scales.shape != (n, 3) or quaternions.shape != (n, 4):
        raise ValueError('Expected xyz/scales (N,3), quaternions (N,4)')
    if len(opacity) != n or not np.isfinite(xyz).all():
        raise ValueError('Invalid Gaussian arrays')
    cov = np.empty((n, 6), dtype=np.float32)
    for start in range(0, n, chunk_size):
        stop = min(n, start+chunk_size)
        q = np.asarray(quaternions[start:stop], dtype=np.float32)
        q = q / np.maximum(np.linalg.norm(q, axis=1, keepdims=True), 1e-20)
        w, x, y, z = q.T
        r0 = np.stack((1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y)), axis=1)
        r1 = np.stack((2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x)), axis=1)
        r2 = np.stack((2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)), axis=1)
        s2 = np.square(np.asarray(scales[start:stop], dtype=np.float32))
        for col, (a,b) in enumerate(((r0,r0),(r0,r1),(r0,r2),
                                     (r1,r1),(r1,r2),(r2,r2))):
            cov[start:stop, col] = np.sum(a*b*s2, axis=1)
    if not np.isfinite(cov).all() or not np.isfinite(opacity).all():
        raise ValueError('Nonfinite covariance/opacity')
    return dict(xyz=xyz, covariance=cov, opacity=opacity)


def prepare_from_ply(ply):
    """Convenience helper accepting an existing structured PLY vertex array."""
    xyz = np.column_stack([ply[name] for name in ('x','y','z')])
    scales = np.exp(np.column_stack([ply[f'scale_{i}'] for i in range(3)]))
    rotations = np.column_stack([ply[f'rot_{i}'] for i in range(4)])
    raw_opacity = np.clip(np.asarray(ply['opacity']), -80, 80)
    opacity = 1 / (1+np.exp(-raw_opacity))
    return prepare_gaussians(xyz, scales, rotations, opacity)


def _quadratic(a, b, cov):
    return (a[:,0]*b[:,0]*cov[:,0] +
            (a[:,0]*b[:,1]+a[:,1]*b[:,0])*cov[:,1] +
            (a[:,0]*b[:,2]+a[:,2]*b[:,0])*cov[:,2] +
            a[:,1]*b[:,1]*cov[:,3] +
            (a[:,1]*b[:,2]+a[:,2]*b[:,1])*cov[:,4] +
            a[:,2]*b[:,2]*cov[:,5])


def render_visibility(gaussians, camera, width=None, height=None, *,
                      max_radius=None, opacity_floor=1/255, threads=4,
                      near_plane=.2, chunk_size=250000, build_dir=None,
                      pixel_center_offset=.5, label_map=None):
    """Render a max(alpha*T)-contributor ID map on CPU, using <=4 threads.

    camera is {'R':(3,3), 't':(3,), 'camera': {'fx','fy','cx','cy',
    'width','height'}}; R,t transform world to camera. Requested width/height
    resize intrinsics. COLMAP pixel centers have corner-origin coordinates
    (.5,.5); returned/raster coordinates use integer array-index centers, so
    pixel_center_offset=.5 is subtracted AFTER resizing. For centered City
    intrinsics this is native ndc2Pix: cx_native=(W-1)/2, cy_native=(H-1)/2.

    Returned maps: id_map (int32, -1 empty), contribution (float32), depth
    (dominant Gaussian camera z), front_depth (nearest contributing Gaussian
    z), expected_depth (alpha*T weighted mean z), alpha. Full-length arrays
    u,v,z,in_frame describe Gaussian CENTER projections for optional sampling.
    All maps are HxW. stats documents radius clipping and timing. Set
    max_radius=None to disable the optional screen-radius cap.

    Optional label_map is a nonnegative int32 (H,W) image, with 0 unknown.
    It adds full-length mask_target, mask_total_weight, mask_matched_weight:
    target is the center's nearest mask label, total sums all alpha*T pixel
    contributions of this Gaussian, matched sums contributions inside target.
    matched/total is footprint label purity; use total to reject occluded dots.
    These weights include ALL contributing Gaussians, not only max contributors.
    """
    t0 = time.perf_counter()
    c = camera.get('camera', camera)
    width = int(width or c['width'])
    height = int(height or c['height'])
    sx, sy = width/c['width'], height/c['height']
    fx, fy = c['fx']*sx, c['fy']*sy
    cx, cy = c['cx']*sx-pixel_center_offset, c['cy']*sy-pixel_center_offset
    R = np.asarray(camera['R'], dtype=np.float32)
    t = np.asarray(camera['t'], dtype=np.float32)
    xyz, cov, opacity = (gaussians[k] for k in ('xyz','covariance','opacity'))
    n = len(xyz)
    u = np.empty(n, dtype=np.float32)
    v = np.empty(n, dtype=np.float32)
    z = np.empty(n, dtype=np.float32)
    projected, original_ids = [], []
    n_clipped, max_uncapped = 0, 0.
    for start in range(0, n, chunk_size):
        stop = min(n, start+chunk_size)
        cp = xyz[start:stop] @ R.T + t
        zz = cp[:,2]
        zz_safe = np.where(zz > near_plane, zz, 1.)
        xx, yy = cp[:,0]/zz_safe, cp[:,1]/zz_safe
        uu, vv = fx*xx+cx, fy*yy+cy
        u[start:stop], v[start:stop], z[start:stop] = uu,vv,zz
        # Match native covariance projection's 1.3*tan(fov) clamp.
        jx = (fx/zz_safe)[:,None]*(R[0]-np.clip(xx,-1.3*width/(2*fx),1.3*width/(2*fx))[:,None]*R[2])
        jy = (fy/zz_safe)[:,None]*(R[1]-np.clip(yy,-1.3*height/(2*fy),1.3*height/(2*fy))[:,None]*R[2])
        cc = cov[start:stop]
        a = _quadratic(jx,jx,cc)+.3
        b = _quadratic(jx,jy,cc)
        d = _quadratic(jy,jy,cc)+.3
        det = np.maximum(a*d-b*b, 1e-20)
        mid = .5*(a+d)
        radius = np.ceil(3*np.sqrt(np.maximum(mid+np.sqrt(np.maximum(.1,mid*mid-det)), .01)))
        eligible = (zz > near_plane)&(opacity[start:stop]>=opacity_floor)&np.isfinite(radius)
        relevant = eligible & (uu+radius>=0)&(uu-radius<width)&(vv+radius>=0)&(vv-radius<height)
        if relevant.any():
            max_uncapped = max(max_uncapped, float(radius[relevant].max()))
        if max_radius is not None:
            n_clipped += int(np.count_nonzero(relevant & (radius>max_radius)))
            radius = np.minimum(radius, max_radius)
        # Native getRect may touch the padded last tile beyond W/H.
        nx, ny = (width+15)//16, (height+15)//16
        tx0 = np.clip(np.trunc((uu-radius)/16),0,nx)
        tx1 = np.clip(np.trunc((uu+radius+15)/16),0,nx)
        ty0 = np.clip(np.trunc((vv-radius)/16),0,ny)
        ty1 = np.clip(np.trunc((vv+radius+15)/16),0,ny)
        keep = eligible & (tx1>tx0)&(ty1>ty0)
        idx = np.flatnonzero(keep)
        alpha_limit = np.maximum(0, 2*np.log(np.maximum(255*opacity[start:stop][idx],1)))
        supportx = np.sqrt(alpha_limit*a[idx])+1e-4
        supporty = np.sqrt(alpha_limit*d[idx])+1e-4
        projected.append(np.column_stack((uu[idx],vv[idx],zz[idx],d[idx]/det[idx],
                           -b[idx]/det[idx],a[idx]/det[idx],opacity[start:stop][idx],radius[idx],supportx,supporty)))
        original_ids.append((idx+start).astype(np.int32))
    p = np.ascontiguousarray(np.concatenate(projected), dtype=np.float32)
    ids = np.ascontiguousarray(np.concatenate(original_ids), dtype=np.int32)
    order = np.ascontiguousarray(np.argsort(p[:,2],kind='stable'),dtype=np.int32)
    projection_seconds = time.perf_counter()-t0
    shape = (height,width)
    dominant = np.full(shape,-1,dtype=np.int32)
    maxweight = np.zeros(shape,dtype=np.float32)
    depth = np.full(shape,np.inf,dtype=np.float32)
    front = np.full(shape,np.inf,dtype=np.float32)
    expected = np.zeros(shape,dtype=np.float32)
    alpha = np.zeros(shape,dtype=np.float32)
    if label_map is not None:
        label_map=np.ascontiguousarray(label_map,dtype=np.int32)
        if label_map.shape != shape or np.any(label_map<0):
            raise ValueError('label_map must be a nonnegative (H,W) integer map')
        mask_target=np.zeros(n,dtype=np.int32)
        # Integer array-index centers; np.rint chooses the nearest pixel.
        px,py=np.rint(u).astype(np.int32),np.rint(v).astype(np.int32)
        valid_target=(z>near_plane)&(px>=0)&(px<width)&(py>=0)&(py<height)
        mask_target[valid_target]=label_map[py[valid_target],px[valid_target]]
        mask_total=np.zeros(n,dtype=np.float32)
        mask_matched=np.zeros(n,dtype=np.float32)
    else:
        label_map=mask_target=np.zeros(1,dtype=np.int32)
        mask_total=mask_matched=np.zeros(1,dtype=np.float32)
        valid_target=None
    kernel = _load_kernel(build_dir)
    raster_t0 = time.perf_counter()
    references = kernel(p,ids,order,len(ids),width,height,int(threads),
                        dominant,maxweight,depth,front,expected,alpha,
                        int(valid_target is not None),label_map,mask_target,mask_total,mask_matched)
    raster_seconds = time.perf_counter()-raster_t0
    in_frame = (z>near_plane)&(u>=0)&(u<width)&(v>=0)&(v<height)
    stats = dict(gaussians=n,projected_gaussians=len(ids),width=width,height=height,
        radius_cap=max_radius,radius_capped_gaussians=n_clipped,
        max_uncapped_radius=max_uncapped,tile_references=int(references),
        covered_pixels=int(np.count_nonzero(dominant>=0)),
        dominant_gaussians=int(np.unique(dominant[dominant>=0]).size),
        projection_seconds=projection_seconds,raster_seconds=raster_seconds,
        total_seconds=time.perf_counter()-t0,threads=min(4,max(1,int(threads))),
        method='CPU anisotropic EWA depth-ordered alpha compositing, max(alpha*T)',
        pixel_center_offset=pixel_center_offset,
        approximation='requested camera resolution; native 3sigma tile bounds; optional radius cap')
    result=dict(id_map=dominant,contribution=maxweight,depth=depth,front_depth=front,
                expected_depth=expected,alpha=alpha,u=u,v=v,z=z,in_frame=in_frame,stats=stats)
    if valid_target is not None:
        result.update(mask_target=mask_target,mask_total_weight=mask_total,
                      mask_matched_weight=mask_matched)
        stats.update(mask_target_gaussians=int(np.count_nonzero(mask_target)),
                     contributing_gaussians=int(np.count_nonzero(mask_total)),
                     matched_gaussians=int(np.count_nonzero(mask_matched)))
    return result
