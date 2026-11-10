"""Continuous SAM/depth/RGB fields and one-pass native alpha*T projection.

This CPU-only module never reads semantic labels or test views. The projection
uses the unchanged source-model EWA equations in adaptive_metadata.py, while
the new rasterizer accumulates three dense per-Gaussian channel masses in one
footprint traversal. It emits neither sparse SAM support nor N x H x W arrays.
"""
import ctypes
import hashlib
import os
from pathlib import Path
import subprocess

import numpy as np

from utils.adaptive_metadata import normalized_sobel, project_native


CHANNELS = ('sam_boundary_strength', 'depth_boundary_strength', 'rgb_boundary_strength')

_CPP = r'''
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>
#include <omp.h>
extern "C" long long continuous_raster(const float* p,const int32_t* ids,
 const int32_t* order,int n,int W,int H,int threads,const float* maps,
 float* total,float* mass) {
 const int TS=16,nx=(W+TS-1)/TS,ny=(H+TS-1)/TS;
 std::vector<std::vector<int32_t>> bins(nx*ny);
 long long references=0;
 for(int k=0;k<n;++k) {
  int g=order[k]; const float* q=p+10*g;
  int x0=std::min(nx,std::max(0,(int)((q[0]-q[7])/TS)));
  int x1=std::min(nx,std::max(0,(int)((q[0]+q[7]+TS-1)/TS)));
  int y0=std::min(ny,std::max(0,(int)((q[1]-q[7])/TS)));
  int y1=std::min(ny,std::max(0,(int)((q[1]+q[7]+TS-1)/TS)));
  for(int y=y0;y<y1;++y)for(int x=x0;x<x1;++x){bins[y*nx+x].push_back(g);++references;}
 }
 omp_set_num_threads(std::max(1,std::min(4,threads)));
 #pragma omp parallel for schedule(dynamic,1)
 for(int tile=0;tile<nx*ny;++tile) {
  int x0=(tile%nx)*TS,y0=(tile/nx)*TS;
  int x1=std::min(W,x0+TS),y1=std::min(H,y0+TS);
  float T[256]; bool done[256];
  for(int i=0;i<256;++i){T[i]=1.f;done[i]=false;}
  for(int g:bins[tile]) {
   const float* q=p+10*g; int gid=ids[g];
   float sum=0.f, channel[3]={0.f,0.f,0.f};
   int xmin=std::max(x0,(int)std::ceil(q[0]-q[8]));
   int xmax=std::min(x1-1,(int)std::floor(q[0]+q[8]));
   int ymin=std::max(y0,(int)std::ceil(q[1]-q[9]));
   int ymax=std::min(y1-1,(int)std::floor(q[1]+q[9]));
   for(int y=ymin;y<=ymax;++y)for(int x=xmin;x<=xmax;++x) {
    int local=(y-y0)*TS+x-x0;if(done[local])continue;
    float dx=q[0]-x,dy=q[1]-y;
    float power=-.5f*(q[3]*dx*dx+q[5]*dy*dy)-q[4]*dx*dy;
    if(power>0.f)continue;
    float alpha=std::min(.99f,q[6]*std::exp(power));
    if(alpha<1.f/255.f)continue;
    float next=T[local]*(1.f-alpha);
    if(next<.0001f){done[local]=true;continue;}
    int pixel=y*W+x;float weight=T[local]*alpha;
    sum+=weight;
    for(int c=0;c<3;++c)channel[c]+=weight*maps[c*W*H+pixel];
    T[local]=next;
   }
   if(sum>0.f) {
    #pragma omp atomic update
    total[gid]+=sum;
    for(int c=0;c<3;++c) {
     #pragma omp atomic update
     mass[3*gid+c]+=channel[c];
    }
   }
  }
 }
 return references;
}
'''


def load_kernel(build_dir):
    build_dir = Path(build_dir)
    build_dir.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256(_CPP.encode()).hexdigest()[:16]
    source = build_dir / ('continuous_' + digest + '.cpp')
    library = build_dir / ('continuous_' + digest + '.so')
    if not library.exists():
        source.write_text(_CPP)
        subprocess.run([os.environ.get('CXX', '/usr/bin/g++'), '-O3', '-std=c++17',
                        '-shared', '-fPIC', '-fopenmp', str(source), '-o', str(library)],
                       check=True, capture_output=True)
    lib = ctypes.CDLL(str(library))
    fp = np.ctypeslib.ndpointer(dtype=np.float32, flags='C_CONTIGUOUS')
    ip = np.ctypeslib.ndpointer(dtype=np.int32, flags='C_CONTIGUOUS')
    ci = ctypes.c_int
    lib.continuous_raster.argtypes = [fp, ip, ip, ci, ci, ci, ci, fp, fp, fp]
    lib.continuous_raster.restype = ctypes.c_longlong
    return lib


def continuous_maps(rgb, labels, depth, alpha, sam_sigma_px=2.):
    import cv2
    sigma = float(sam_sigma_px)
    if not np.isfinite(sigma) or sigma <= 0:
        raise ValueError('SAM distance sigma must be finite and positive')
    rgb, labels, depth, alpha = map(np.asarray, (rgb, labels, depth, alpha))
    shape = labels.shape
    if labels.ndim != 2 or rgb.shape != shape + (3,) or depth.shape != shape or alpha.shape != shape:
        raise ValueError('SAM, RGB, source depth and alpha must share native pixel coordinates')
    if not np.issubdtype(labels.dtype, np.integer) or np.any(labels < 0):
        raise ValueError('SAM region IDs must be nonnegative integers')
    if not np.isfinite(rgb).all() or not np.isfinite(alpha).all():
        raise ValueError('Nonfinite RGB/alpha input')
    if np.any(rgb < 0) or np.any(rgb > 1) or np.any(alpha < 0) or np.any(alpha > 1):
        raise ValueError('RGB/alpha input must be in [0,1]')
    edge = np.zeros(shape, np.uint8)
    by, bx = labels[1:] != labels[:-1], labels[:, 1:] != labels[:, :-1]
    edge[1:] |= by
    edge[:-1] |= by
    edge[:, 1:] |= bx
    edge[:, :-1] |= bx
    if edge.any():
        distance = cv2.distanceTransform(1 - edge, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
        sam = np.exp(-np.square(distance) / (2 * sigma * sigma)).astype(np.float32)
    else:
        sam = np.zeros(shape, np.float32)
    gray = cv2.cvtColor(np.asarray(rgb, np.float32), cv2.COLOR_RGB2GRAY)
    depth_edge = normalized_sobel(depth, np.isfinite(depth) & (alpha > .5))
    rgb_edge = normalized_sobel(gray)
    result = np.ascontiguousarray(np.stack((sam, depth_edge, rgb_edge)), np.float32)
    if not np.isfinite(result).all() or np.any(result < 0) or np.any(result > 1):
        raise ValueError('Invalid continuous boundary maps')
    return result


def accumulate_projected(projected, ids, order, maps, source_count, kernel, threads=4):
    maps = np.ascontiguousarray(maps, np.float32)
    if maps.ndim != 3 or maps.shape[0] != 3 or not np.isfinite(maps).all() or np.any(maps < 0) or np.any(maps > 1):
        raise ValueError('Expected three finite native continuous maps in [0,1]')
    projected = np.ascontiguousarray(projected, np.float32)
    ids, order = np.ascontiguousarray(ids, np.int32), np.ascontiguousarray(order, np.int32)
    if projected.shape != (len(ids), 10) or len(order) != len(ids) or np.any(ids < 0) or np.any(ids >= source_count):
        raise ValueError('Projected row/source ID mismatch')
    if not np.array_equal(np.sort(order), np.arange(len(ids), dtype=np.int32)):
        raise ValueError('Depth ordering must be a permutation of projected rows')
    total = np.zeros(source_count, np.float32)
    mass = np.zeros((source_count, 3), np.float32)
    _, height, width = maps.shape
    references = kernel.continuous_raster(projected, ids, order, len(ids), width, height,
                                         int(threads), maps, total, mass)
    return dict(total=total, mass=mass, tile_references=int(references))


def accumulate_continuous(gaussians, camera, maps, kernel, threads=4):
    projected, ids, order = project_native(gaussians, camera)
    return accumulate_projected(projected, ids, order, maps, len(gaussians['xyz']), kernel, threads)


def reference_projected(projected, ids, order, maps, source_count):
    """Independent tiny-image per-pixel reference; deliberately not tiled."""
    projected, maps = np.asarray(projected, np.float64), np.asarray(maps, np.float64)
    _, height, width = maps.shape
    total, mass = np.zeros(source_count), np.zeros((source_count, 3))
    nx, ny = (width + 15) // 16, (height + 15) // 16
    for y in range(height):
        for x in range(width):
            transmittance = 1.
            for index in order:
                q = projected[index]
                tx0 = min(nx, max(0, int((q[0] - q[7]) / 16)))
                tx1 = min(nx, max(0, int((q[0] + q[7] + 15) / 16)))
                ty0 = min(ny, max(0, int((q[1] - q[7]) / 16)))
                ty1 = min(ny, max(0, int((q[1] + q[7] + 15) / 16)))
                if not (tx0 <= x // 16 < tx1 and ty0 <= y // 16 < ty1):
                    continue
                dx, dy = q[0] - x, q[1] - y
                if abs(dx) > q[8] or abs(dy) > q[9]:
                    continue
                power = -.5 * (q[3] * dx * dx + q[5] * dy * dy) - q[4] * dx * dy
                if power > 0:
                    continue
                alpha = min(.99, q[6] * np.exp(power))
                if alpha < 1. / 255:
                    continue
                following = transmittance * (1 - alpha)
                if following < .0001:
                    break
                weight = transmittance * alpha
                total[ids[index]] += weight
                mass[ids[index]] += weight * maps[:, y, x]
                transmittance = following
    return dict(total=total, mass=mass)
