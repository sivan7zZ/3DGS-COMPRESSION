"""Generic PLY and COLMAP I/O. No label or annotation inputs."""

from pathlib import Path
import hashlib
import struct
import numpy as np

PLY_TYPES = {'float': '<f4', 'float32': '<f4', 'double': '<f8',
             'uchar': 'u1', 'uint8': 'u1', 'char': 'i1',
             'short': '<i2', 'ushort': '<u2', 'int': '<i4', 'uint': '<u4'}

def sha256_file(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()

def read_ply(path):
    fields, lines, count = [], [], None
    with open(path, 'rb') as f:
        while True:
            line = f.readline()
            if not line:
                raise ValueError('Missing end_header')
            lines.append(line)
            parts = line.decode('ascii').strip().split()
            if parts[:1] == ['format'] and parts[1] != 'binary_little_endian':
                raise ValueError('Only binary_little_endian PLY supported')
            if parts[:1] == ['element']:
                if parts[1] != 'vertex' or count is not None:
                    raise ValueError('Expected a vertex-only PLY')
                count = int(parts[2])
            if parts[:1] == ['property']:
                if len(parts) != 3 or parts[1] not in PLY_TYPES:
                    raise ValueError(f'Unsupported property: {parts}')
                fields.append((parts[2], PLY_TYPES[parts[1]]))
            if parts == ['end_header']:
                offset = f.tell()
                break
    dtype = np.dtype(fields)
    if Path(path).stat().st_size != offset + count * dtype.itemsize:
        raise ValueError('PLY size does not match header')
    return np.memmap(path, dtype=dtype, mode='r', offset=offset, shape=(count,)), b''.join(lines)

def read_colmap(folder):
    # COLMAP binary uses little endian; poses transform WORLD to CAMERA.
    def unpack(f, fmt):
        return struct.unpack('<' + fmt, f.read(struct.calcsize('<' + fmt)))
    cameras = {}
    with open(Path(folder) / 'cameras.bin', 'rb') as f:
        for _ in range(unpack(f, 'Q')[0]):
            cid, model, w, h = unpack(f, 'iiQQ')
            if model not in (0, 1):
                raise ValueError('Use undistorted SIMPLE_PINHOLE/PINHOLE cameras')
            par = unpack(f, 'ddd' if model == 0 else 'dddd')
            fx, fy, cx, cy = (par[0], par[0], par[1], par[2]) if model == 0 else par
            cameras[cid] = dict(width=w, height=h, fx=fx, fy=fy, cx=cx, cy=cy)
    images = {}
    with open(Path(folder) / 'images.bin', 'rb') as f:
        for _ in range(unpack(f, 'Q')[0]):
            vals = unpack(f, 'idddddddi')
            name = bytearray()
            while True:
                c = f.read(1)
                if c == b'\0':
                    break
                if not c:
                    raise ValueError('Truncated image record')
                name.extend(c)
            n = unpack(f, 'Q')[0]
            f.seek(24 * n, 1)
            w, x, y, z = vals[1:5]
            r = np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                          [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                          [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])
            images[Path(name.decode('utf-8')).stem] = dict(
                R=r, t=np.array(vals[5:8]), camera=cameras[vals[8]],
                name=name.decode('utf-8'))
    return images

def project(xyz, camera):
    p = xyz @ camera['R'].T + camera['t']
    c = camera['camera']
    z = p[:, 2]
    denom = np.maximum(z, 1e-8)
    u = c['fx'] * p[:, 0] / denom + c['cx']
    v = c['fy'] * p[:, 1] / denom + c['cy']
    valid = (z > 0) & (u >= 0) & (v >= 0) & (u < c['width']) & (v < c['height'])
    return u, v, z, valid

def colors_for_ids(ids):
    # Stable deterministic pseudo-random hue; 0 is gray.
    import colorsys
    palette = np.array([colorsys.hsv_to_rgb((i*.618033988749895) % 1, .68, .94)
                        for i in range(int(ids.max())+1)], dtype=np.float32)
    palette[0] = [.35, .35, .35]
    return np.round(255*palette[ids]).astype(np.uint8)
