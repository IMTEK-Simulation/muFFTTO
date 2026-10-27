"""
Patch test with interface laws in the gap medium.

Stick (E_upper / E_lower != 1, so frictionless blocks slide sideways at the interface):
    left    sigma_xx along the vertical line x = 0.25 (between the axis and the free side), last increment, for each run
            (frictionless: 0; full stick: the bonded reference, the same blocks without a gap)
    middle  relative slip at the interface (u_x of the upper minus the lower face) vs x, last increment
Adhesion (push-pull run):
    right   force vs top displacement: tensile plateau -sigma_0 W while the faces separate (Dugdale), compared with the
            exact plateau and the work of adhesion

usage:  python plot_interface_laws.py --stick <run> ... --adhesion <run>      -> interface_laws.png
"""
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
args = sys.argv[1:]
stick_runs = args[args.index('--stick') + 1:args.index('--adhesion')] if '--stick' in args else []
adhesion_runs = args[args.index('--adhesion') + 1:] if '--adhesion' in args else []
STYLES = [dict(color='#c6dbef', ls=':', marker='s'), dict(color='#86b6ef', ls='--', marker='D'),
          dict(color='#3d7fd0', ls='-.', marker='^'), dict(color='#08306b', ls='-', marker='o'),
          dict(color='#7b3294', ls='-', marker='v')]
BONDED = dict(color='k', ls='-', marker=None)


def load(folder):
    tmp = os.path.join(tempfile.mkdtemp(), 'c.nc')
    shutil.copy(os.path.join(folder, 'contact_fracture.nc'), tmp)
    return netCDF4.Dataset(tmp)


def label(folder):
    if '_gap0px_' in folder:
        return 'bonded (no gap): full-stick reference'
    m = re.search(r'_stick([0-9.e+-]+)_', folder)
    return f'stick c = {m.group(1)}' if m else 'frictionless (no stick law)'


fig, axes = plt.subplots(1, 3, figsize=(20, 6), layout='constrained')
for folder, st in zip(stick_runs, STYLES):
    st = BONDED if '_gap0px_' in folder else st
    ds = load(folder)
    n = ds['phase'].shape[-1]
    h = 1.0 / n
    P = float(ds['crown_force'][-1])
    W = 1.0 - 2 * h
    p = P / W
    sxx = np.asarray(ds['sxx'][-1]).reshape(n, n)
    phase = np.asarray(ds['phase'][0]).reshape(n, n)
    i = n // 4                                               # x = 0.25
    y = (np.arange(n) + 0.5) * h
    solid = phase[i] > 0
    axes[0].plot(sxx[i][solid] / p, y[solid], color=st['color'], ls=st['ls'], marker=st['marker'], ms=3,
                 label=label(folder))
    u = np.asarray(ds['u_fluc'][-1]).reshape(2, n, n)
    js = np.nonzero(phase[n // 2] > 0)[0]
    lower_top = js[js < n // 2].max() + 1                   # node row of the lower block's top face
    upper_bottom = js[js >= n // 2].min() if (js >= n // 2).any() else lower_top
    x = np.arange(n) * h
    inside = (x > h) & (x < 1 - h)
    slip = u[0][:, upper_bottom] - u[0][:, lower_top]
    axes[1].plot(x[inside], slip[inside] * 1e3, color=st['color'], ls=st['ls'], marker=st['marker'], ms=3,
                 label=label(folder))
axes[0].axvline(0, color='0.5', lw=0.8)
axes[0].set(xlabel='$\\sigma_{xx}$ / p at x = 0.25', ylabel='y', title='lateral stress across both blocks '
            '(frictionless exact: 0)')
axes[0].legend(frameon=False, fontsize=9)
axes[0].grid(True, color='0.9')
axes[1].axhline(0, color='0.5', lw=0.8)
axes[1].set(xlabel='x', ylabel='slip $u_x^{upper} - u_x^{lower}$ (x 1e-3)',
            title='relative sliding of the faces (upper block softer: expands more)')
axes[1].legend(frameon=False, fontsize=9)
axes[1].grid(True, color='0.9')
for folder in adhesion_runs:
    ds = load(folder)
    n = ds['phase'].shape[-1]
    h = 1.0 / n
    u = np.asarray(ds['crown_u'][:])
    P = np.asarray(ds['crown_force'][:])
    sigma0 = float(re.search(r'_adh([0-9.e+-]+)jc', folder).group(1))
    jc = float(re.search(r'jc([0-9.e+-]+)', folder).group(1))
    W = 1.0 - 2 * h
    turn = np.argmax(u)
    axes[2].plot(u[:turn + 1], P[:turn + 1], 'o-', color='#3d7fd0', ms=3, label='push')
    axes[2].plot(u[turn:], P[turn:], 's-', color='#c4441a', ms=3, mfc='white', label='pull back')
    axes[2].axhline(-sigma0 * W, color='k', ls='--', lw=1, label=f'exact plateau $-\\sigma_0 W$ = {-sigma0 * W:.3f}')
    axes[2].axvline(h, color='0.6', ls=':', lw=1)
    axes[2].axvline(h * (1 - jc), color='0.6', ls=':', lw=1)
    axes[2].text(h * (1 - jc), 0.6 * P.max(), f' faces apart beyond\n gap (1 - J_c) = {h * (1 - jc):.4f}', fontsize=9)
    pull = slice(turn, None)
    work = -np.trapezoid(np.minimum(P[pull], 0), u[pull]) / W
    axes[2].set(xlabel='top displacement u (gap = 1 pixel = {:.4f})'.format(h), ylabel='force P',
                title=f'adhesion (Dugdale, $\\sigma_0$ = {sigma0:g}, J_c = {jc:g}): min P = {P.min():.3f}, '
                      f'work of separation {work:.2e} per width\n(estimate $\\sigma_0$ (J_c - w/2) gap = '
                      f'{sigma0 * (jc - 0.025) * h:.2e})')
    axes[2].set_ylim(1.4 * min(P.min(), -sigma0 * W), min(P.max(), 3 * sigma0 * W))
    axes[2].legend(frameon=False, fontsize=9)
    axes[2].grid(True, color='0.9')
fig.suptitle('Patch test with interface laws in the third medium (prototypes): elastic stick and Dugdale adhesion',
             fontsize=13)
out = os.path.join(HERE, 'interface_laws.png')
fig.savefig(out, dpi=115)
print(f'wrote {out}')
