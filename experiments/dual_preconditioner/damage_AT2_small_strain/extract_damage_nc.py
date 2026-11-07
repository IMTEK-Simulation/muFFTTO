"""
Damage-only copy of an AT2 output .nc (all fields per load step, 8 GB at 128^3) for viewing in muEye: keeps the variable
'damage' (every EVERY-th frame plus the last one), its dimensions and all global attributes (field layouts, run
parameters). With a _checkpoint.npz instead of a .nc: one frame with the damage of the checkpoint (an unfinished run,
e.g. inside the break step), with the layout attributes taken from the run's .nc (TEMPLATE, default <checkpoint>.nc).
usage:  python extract_damage_nc.py <run .nc | run _checkpoint.npz> <out.nc> [EVERY=1]
"""
import os
import sys

import netCDF4
import numpy as np

src, out = sys.argv[1], sys.argv[2]
every = int(os.environ.get('EVERY', 1))
template = os.environ.get('TEMPLATE', src.replace('_checkpoint.npz', '.nc'))
with netCDF4.Dataset(template) as d:
    v = d['damage']
    dims = [(n, None if d.dimensions[n].isunlimited() else len(d.dimensions[n])) for n in v.dimensions]
    attrs = {k: d.getncattr(k) for k in d.ncattrs()}
    if src.endswith('.npz'):
        c = np.load(src)
        frames = [np.asarray(c['damage']).reshape(v.shape[1:])]
        attrs['description'] = f'damage of the checkpoint: load step {int(c["step"])}, staggered iteration ' \
                               f'{int(c["staggered_iteration"])}'
    else:
        n = v.shape[0]
        idx = sorted(set(range(0, n, every)) | {n - 1})
        frames = [np.asarray(v[i]) for i in idx]
        attrs['description'] = f'damage of load steps (output frames) {idx}'
with netCDF4.Dataset(out, 'w', format='NETCDF3_64BIT_DATA') as o:
    for n, size in dims:
        o.createDimension(n, size)
    w = o.createVariable('damage', 'f4', [n for n, _ in dims])
    for i, f in enumerate(frames):
        w[i] = f.astype(np.float32)
    o.setncatts(attrs)
print(out, len(frames), 'frame(s)', frames[0].shape, f'max d {max(f.max() for f in frames):.3f}')
