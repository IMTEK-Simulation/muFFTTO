"""Plot the 3D hollow-bar runs (pressurized_bar_3D.py): deformed mid-sections on their exact voxels (colour = voxel
det F) at four pressures, pressure vs cavity volume, Hessian products per Newton solve for G_wall and G.
Usage: python plot_bar3d.py [nnn]   (latest runs/bar3d_n<nnn>_*_{G_wall,G}_* ; deformation from G_wall)"""
import glob
import os
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.collections import PolyCollection  # noqa: E402

plt.rcParams.update({'font.size': 13})
HERE = os.path.dirname(os.path.realpath(__file__))
nnn = int(sys.argv[1]) if len(sys.argv) > 1 else 32


def latest(P):
    runs = sorted(glob.glob(os.path.join(HERE, 'runs', f'bar3d_n{nnn}_*_{P}_[cg]pu_*')),   # _G_ alone would match _G_wall_
                  key=os.path.getmtime)
    return np.load(os.path.join(runs[-1], 'state.npz')) if runs else None


d = latest('G_wall')
N, h = tuple(int(v) for v in d['N']), float(d['h'])
solid, cav, clamp = d['solid_px'], d['cav_px'], d['clamp_node']
U = d['u'].astype(float)                                                # [inc, 3, x, y, z] nodal
p = d['p']
# DISPLAY ONLY: the pinned cavity-interior nodes get the harmonic extension of the cavity-surface displacement
void_node = np.all([np.roll(~solid, (a, b, c), (0, 1, 2)) for a in (0, 1) for b in (0, 1) for c in (0, 1)], axis=0)
cav_node = void_node & ~clamp & np.any([np.roll(cav, (a, b, c), (0, 1, 2)) for a in (0, 1) for b in (0, 1)
                                        for c in (0, 1)], axis=0)
frames = np.unique(np.linspace(0, len(p) - 1, 4).round().astype(int))
for k in frames:
    for _ in range(3000):
        avg = sum(np.roll(U[k], s, ax) for s in (1, -1) for ax in (1, 2, 3)) / 6.0
        U[k] = np.where(cav_node, avg, U[k])


def section_polys(u, axis, mask):
    """quads of the voxels of the mid layer normal to `axis`, in the two in-plane coordinates (deformed)"""
    a, b = [ax for ax in range(3) if ax != axis]
    layer = N[axis] // 2
    sl = [slice(None)] * 3
    sl[axis] = layer
    m2 = mask[tuple(sl)]
    ii, jj = np.nonzero(m2)
    polys = []
    for di, dj in ((0, 0), (1, 0), (1, 1), (0, 1)):
        pts = []
        for comp in (a, b):
            val = 0.0
            for dk in (0, 1):                                            # mid-plane of the voxel layer
                idx = [None] * 3
                idx[a], idx[b], idx[axis] = (ii + di) % N[a], (jj + dj) % N[b], np.full_like(ii, (layer + dk) % N[axis])
                val = val + 0.5 * u[comp][tuple(idx)]
            base = (ii + di) * h if comp == a else (jj + dj) * h
            pts.append(base + val)
        polys.append(np.stack(pts, axis=-1))
    polys = np.stack(polys, axis=1)
    x, y = polys[..., 0], polys[..., 1]
    area = 0.5 * np.sum(x * np.roll(y, -1, 1) - np.roll(x, -1, 1) * y, axis=1) / h ** 2
    return polys, area


AX = {2: ('cross-section z = Lz/2', 'x', 'y'), 1: ('section y = Ly/2', 'x', 'z'), 0: ('section x = Lx/2', 'y', 'z')}
fig = plt.figure(figsize=(4.6 * len(frames), 4.3 * 3 + 4.6))
gs = fig.add_gridspec(4, len(frames), height_ratios=[1, 1, 1, 1.05], hspace=0.38)
norm = matplotlib.colors.Normalize(vmin=0.5, vmax=1.5)
for row, axis in enumerate((2, 1, 0)):
    lims = []
    for k in frames:
        pp, _ = section_polys(U[k], axis, solid | cav)
        lims.append((pp[..., 0].min(), pp[..., 0].max(), pp[..., 1].min(), pp[..., 1].max()))
    lo0, hi0 = min(l[0] for l in lims), max(l[1] for l in lims)
    lo1, hi1 = min(l[2] for l in lims), max(l[3] for l in lims)
    for col, k in enumerate(frames):
        ax = fig.add_subplot(gs[row, col])
        pc_gas, _ = section_polys(U[k], axis, cav)
        ax.add_collection(PolyCollection(pc_gas, facecolor='#e06c5a', edgecolor='none'))
        ps, J2 = section_polys(U[k], axis, solid)
        pc = PolyCollection(ps, array=J2, cmap='viridis', norm=norm, edgecolor='k', linewidth=0.1)
        ax.add_collection(pc)
        if axis != 0:                                                    # clamp planes are x = const
            for xc in np.nonzero(clamp[:, 0, 0])[0]:
                ax.axvline(xc * h, color='k', lw=3)
        ax.set_xlim(lo0 - 0.02, hi0 + 0.02)
        ax.set_ylim(lo1 - 0.02, hi1 + 0.02)
        ax.set_aspect('equal')
        ax.set_title(f'{AX[axis][0]}, p = {p[k]:.1f}', fontsize=12)
        ax.set_xlabel(AX[axis][1])
        ax.set_ylabel(AX[axis][2])
cb = fig.colorbar(pc, ax=fig.axes[:3 * len(frames)], shrink=0.6, pad=0.01)
cb.set_label('in-plane area ratio of the section voxel (~ det F)')

STY = {'G_wall': dict(color='#238b45', ls='-.', marker='^', mfc='#238b45', label='corrected Green, wall FFT'),
       'G': dict(color='#74c476', ls='--', marker='s', mfc='white', label='projected Green')}
ax1 = fig.add_subplot(gs[3, :len(frames) // 2])
ax1.plot(d['volume'] / d['volume'][0], p, 'o-', color='k')
ax1.set_xlabel('cavity volume / initial volume')
ax1.set_ylabel('pressure p   (E = 100)')
ax1.set_xlim(0, None)
ax1.set_ylim(0, None)
ax1.grid(alpha=0.3)
ax2 = fig.add_subplot(gs[3, len(frames) // 2:])
for P, st in STY.items():
    r = latest(P)
    if r is None:
        continue
    ax2.plot(r['p'][1:], r['hessp'][1:] / r['nit'][1:], ms=5, lw=1.6, **st,)
    ax2.text(0.02, 0.95 - 0.08 * list(STY).index(P),
             f'{st["label"]}: mean {r["hessp"].sum() / r["nit"].sum():.1f} per solve, total {r["hessp"].sum()}, '
             f'{r["nit"].sum()} solves, {r["time"].sum():.0f} s', transform=ax2.transAxes, fontsize=11,
             color=st['color'], va='top')
ax2.set_xlabel('pressure p')
ax2.set_ylabel('Hessian products per Newton solve')
ax2.set_xlim(0, 1.03 * float(d['P_MAX']))
ax2.set_ylim(0, 1.45 * ax2.get_ylim()[1])                    # room for the summary text
ax2.grid(alpha=0.3)
ax2.legend(loc='lower right')
fig.suptitle(f'3D hollow bar, x faces clamped, h = 1/{nnn}, cell {N[0]} x {N[1]} x {N[2]} voxels (GPU)')
out = os.path.join(HERE, f'bar3d_n{nnn}.png')
fig.savefig(out, dpi=130, bbox_inches='tight')
print(out)
