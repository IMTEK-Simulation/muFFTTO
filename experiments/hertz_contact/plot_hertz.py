"""
Hertz comparison of the hertz_contact.py runs (regular and adapted grid).

    left    force per unit length P vs the cap-top displacement
    middle  contact half-width a vs P (a: half of the width where p > 5 % of its peak, interpolated), Hertz
            a = sqrt(4 P R / (pi E*))
    right   surface pressure p(x) at a few loads, each with the Hertz profile for its measured P

usage:  python plot_hertz.py [run folder ...]      (default: the newest regular and adapted run)
        -> hertz_comparison_N<n>.png
"""
import glob
import os
import re
import shutil
import sys
import tempfile

import numpy as np
import netCDF4
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.realpath(__file__))
E_STAR = 100.0 / (2 * (1 - 0.3 ** 2))
STYLE = {'regular': dict(color='#86b6ef', ls=(0, (4, 2)), marker='s', mfc='white'),
         'adapted': dict(color='#c4441a', ls='-', marker='o', mfc='#c4441a'),
         'adapted, periodic sides': dict(color='#f4a582', ls=(0, (1, 1.5)), marker='D', mfc='white'),
         'regular, periodic sides': dict(color='#184f95', ls=(0, (1, 1.5)), marker='D', mfc='white'),
         'adapted, graded 0.6': dict(color='#7f1d0a', ls=(0, (4, 2)), marker='^', mfc='white'),
         'adapted, graded 0.75': dict(color='#e7298a', ls=(0, (1, 1.5)), marker='v', mfc='#e7298a'),
         # the same grids after the fix (smoothing term only in the gap medium): green family
         'adapted, smoothing in gap only': dict(color='#1b7837', ls='-', marker='o', mfc='#1b7837'),
         'adapted, graded 0.6, smoothing in gap only': dict(color='#5aae61', ls=(0, (4, 2)), marker='^', mfc='white'),
         'adapted, graded 0.75, smoothing in gap only': dict(color='#a6dba0', ls=(0, (1, 1.5)), marker='v',
                                                             mfc='#a6dba0')}


def newest(kind):
    runs = sorted(glob.glob(os.path.join(HERE, 'runs', f'hertz_*_{kind}_N*')), key=os.path.getmtime)
    return runs[-1] if runs else None


def load(folder):
    tmp = os.path.join(tempfile.mkdtemp(), 'c.nc')       # a copy: the run may still be writing
    shutil.copy(os.path.join(folder, 'contact_fracture.nc'), tmp)
    with netCDF4.Dataset(tmp) as ds:
        r = {k: np.asarray(ds[k][:]) for k in ('crown_u', 'crown_force', 'p_surface')}
        r['R'] = float(ds.getncattr('R'))
        r['n'] = r['p_surface'].shape[-1]
        r['grade'] = float(ds.getncattr('grade')) if 'grade' in ds.ncattrs() else 0.0
        r['x'] = surface_x(np.asarray(ds['phase'][0]), np.asarray(ds['grid_displacement'][0]), r['n'])
    return r


def surface_x(phase, d_g, n):
    """physical x of the pixel centres of the block's top pixel row (where p is measured), relative to the cap axis:
    the mean of the four corner nodes (graded / adapted grids are not uniform)"""
    X = np.array(np.meshgrid(np.arange(n) / n, np.arange(n) / n, indexing='ij'))
    x = np.pad(X + d_g.reshape(2, n, n), ((0, 0), (0, 1), (0, 1)), mode='wrap')
    x[0, -1, :] += 1.0
    x[1, :, -1] += 1.0
    c = 0.25 * (x[:, :-1, :-1] + x[:, 1:, :-1] + x[:, :-1, 1:] + x[:, 1:, 1:])
    block = (phase.reshape(n, n) > 0) & (c[1] < 0.51)
    j = np.max(np.nonzero(block.any(axis=0))[0])
    return c[0, :, j] - 0.5


def half_width(x, p):
    """contact half-width from the half-maximum: walk outward from the peak to where p first drops below p_max / 2
    (interpolated), a = x_1/2 / sqrt(3/4) for a Hertz (elliptic) profile; insensitive to the third-medium tail"""
    if p.max() <= 0:
        return 0.0
    k0, half = int(np.argmax(p)), 0.5 * p.max()
    ends = []
    for step in (1, -1):
        k = k0
        while 0 <= k + step < len(p) and p[k + step] >= half:
            k += step
        k1 = k + step
        ends.append(x[k] if not 0 <= k1 < len(p) else np.interp(half, [p[k1], p[k]], [x[k1], x[k]]))
    return 0.5 * abs(ends[0] - ends[1]) / np.sqrt(0.75)


folders = sys.argv[1:] or [f for f in (newest('regular'), newest('adapted')) if f]
fig, axes = plt.subplots(1, 3, figsize=(19, 6), layout='constrained')
n = None
for folder in folders:
    kind = ('adapted' if '_adapted_' in folder else 'regular') + (', periodic sides' if '_periodic_' in folder else '')
    grade = re.search(r'_grade([0-9.]+)_', folder)
    kind += f', graded {grade.group(1)}' if grade else ''
    kind += ', smoothing in gap only' if '_reggap_' in folder else ''
    try:
        r = load(folder)
    except (OSError, AttributeError, KeyError, IndexError):  # just started: nothing written yet
        print(f'skipping {folder}: no data yet')
        continue
    n = r['n']
    x = r['x']                                             # pixel centres, relative to the cap axis
    st = STYLE[kind]
    P = r['crown_force']
    axes[0].plot(r['crown_u'], P, ls=st['ls'], color=st['color'], marker=st['marker'], mfc=st['mfc'], ms=4,
                 label=kind.replace('adapted', 'adapted grid', 1).replace('regular', 'regular grid', 1))
    contact = P > 1e-3                                      # the filler alone carries ~1e-4
    a = np.array([half_width(x, p) for p in r['p_surface']])
    axes[1].plot(P[contact], a[contact], ls=st['ls'], color=st['color'], marker=st['marker'], mfc=st['mfc'], ms=5,
                 label=kind.replace('adapted', 'adapted grid', 1).replace('regular', 'regular grid', 1))
    idx = np.nonzero(contact)[0]
    for k in idx[np.linspace(0, len(idx) - 1, min(3, len(idx))).astype(int)] if len(idx) else []:
        axes[2].plot(x, r['p_surface'][k], ls=st['ls'], color=st['color'], lw=1.5, marker=st['marker'], ms=3,
                     mfc=st['mfc'], label=f'{kind}, P = {P[k]:.3g}')
        a_h = np.sqrt(4 * P[k] * r['R'] / (np.pi * E_STAR))
        xs = np.linspace(-a_h, a_h, 200)
        axes[2].plot(xs, 2 * P[k] / (np.pi * a_h ** 2) * np.sqrt(a_h ** 2 - xs ** 2), color='k', lw=1, alpha=0.6)
P_line = np.linspace(0, max(ax_P.get_xlim()[1] for ax_P in [axes[1]]), 200)
axes[1].plot(P_line, np.sqrt(4 * P_line * r['R'] / (np.pi * E_STAR)), color='k', lw=1.5, label='Hertz')
if n:
    axes[1].axhline(1.0 / n, color='0.6', lw=1, ls=':')
    axes[1].text(axes[1].get_xlim()[0], 1.0 / n, ' one pixel', va='bottom', fontsize=9, color='0.4')
axes[0].set(xlabel='cap-top displacement (gap 0.02)', ylabel='force per unit length P',
            title='load-displacement (contact from ~0.02)')
axes[1].set(xlabel='force per unit length P', ylabel='contact half-width a', title='contact half-width (from the half-maximum width) vs Hertz')
axes[2].set(xlabel='x (relative to the cap axis)', ylabel='surface pressure p = $-P_{yy}$',
            title='pressure profiles (thin black: Hertz at the same P)', xlim=(-0.2, 0.2))
for ax in axes:
    ax.grid(True, color='0.9')
    ax.set_ylim(bottom=0)
    ax.legend(frameon=False, fontsize=9)
fig.suptitle(f'2D Hertz test of third-medium contact ({n}$^2$): cylinder cap R = {r["R"]:g} on a block, E* = {E_STAR:.1f}',
             fontsize=13)
out = os.path.join(HERE, f'hertz_comparison_N{n}' + (f'_R{r["R"]:g}' if r['R'] != 1.0 else '') + '.png')
fig.savefig(out, dpi=120)
print(f'wrote {out}')
