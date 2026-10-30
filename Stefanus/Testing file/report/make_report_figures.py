"""
Figures of report_true_holes.tex (Composite A, compression in X, true holes, 1 x 2 cell, Q1-SRI, TR Newton + Green):

  fig_curves.png     (a) the simplified material at 128 px against the paper's material and paper Fig. 4,
                     (b) the simplified material at 256 px against the paper's material at 256 px and paper Fig. 4
  fig_snapshots.png  the deformed 256 px cell (simplified material) at eps = 0.010, 0.025, 0.100, 0.200

Reads exp_data/frames_TR_X_*/inc*.npz and exp_data/paper_fig4_compositeA.csv.

    python report/make_report_figures.py
"""
import glob
import os

import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, TwoSlopeNorm

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, '..', 'exp_data')
OUT = os.path.join(HERE, 'figures')
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({'font.size': 10})

fig4 = np.genfromtxt(os.path.join(DATA, 'paper_fig4_compositeA.csv'), delimiter=',', skip_header=2, names=True,
                     dtype=None, encoding=None)


def frames(tag):
    return [dict(np.load(f)) for f in sorted(glob.glob(os.path.join(DATA, f'frames_TR_X_{tag}', 'inc*.npz')))]


def curve(tag):
    fs = frames(tag)
    return np.r_[0.0, [float(d['eps']) for d in fs]], np.r_[0.0, [float(d['stress_MPa']) for d in fs]]


def paper(ax):
    for c, col, ls, lab in (('periodic', 'tab:red', '-.', 'Li & Rudykh, simulation'),
                            ('experiment', 'k', ':', 'Li & Rudykh, experiment')):
        m = (fig4['direction'] == 'X') & (fig4['curve'] == c)
        ax.plot(fig4['eps'][m], fig4['stress'][m], color=col, ls=ls, lw=2, label=lab)


# --- stress-strain curves --------------------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout='constrained')
ax = axes[0]
paper(ax)
for tag, col, ls, mk, lab in (('128x222_cells1x2_sri_gpu', '0.35', '-', 's', 'paper material (1000, 1000)'),
                              ('128x222_cells1x2_sri_c10_L100', 'tab:green', '-', 'o', 'contrast 10, $\\Lambda/\\mu$ = 100'),
                              ('128x222_cells1x2_sri_c10_L10', 'tab:orange', '--', '^', 'contrast 10, $\\Lambda/\\mu$ = 10')):
    e, s = curve(tag)
    ax.plot(e, s, color=col, ls=ls, marker=mk, ms=4, lw=1.6, label=f'128 px: {lab}')
ax.set(xlim=(0, 0.052), ylim=(0, 0.0075), title='(a) material simplification, 128 px')
ax = axes[1]
paper(ax)
for tag, col, ls, mk, lab in (('256x444_cells1x2_sri_gpu', '0.35', '-', 's', 'paper material (1000, 1000)'),
                              ('256x444_cells1x2_sri_c10_L100', 'tab:green', '-', 'o', 'contrast 10, $\\Lambda/\\mu$ = 100')):
    e, s = curve(tag)
    ax.plot(e, s, color=col, ls=ls, marker=mk, ms=3, lw=1.6, label=f'256 px: {lab}')
ax.set(xlim=(0, 0.205), ylim=(0, 0.011), title='(b) simplified material, 256 px, to $\\varepsilon$ = 0.2')
for ax in axes:
    ax.set(xlabel='nominal strain $\\varepsilon_X$', ylabel='nominal stress [MPa]')
    ax.grid(alpha=0.3)
    ax.legend(frameon=False, fontsize=8, loc='lower right')
fig.savefig(os.path.join(OUT, 'fig_curves.png'), dpi=200)

# --- deformation snapshots -------------------------------------------------------------------------------------
data = frames('256x444_cells1x2_sri_c10_L100')
eps_all = np.array([float(d['eps']) for d in data])
shown = [int(np.argmin(np.abs(eps_all - e))) for e in (0.010, 0.025, 0.100, 0.200)]
d0 = data[0]
Lx, Ly = d0['domain_size']
mm, phase = float(d0['R_MM']), d0['phase']
nx, ny = phase.shape
X, Y = np.meshgrid(np.arange(nx + 1) * Lx / nx, np.arange(ny + 1) * Ly / ny, indexing='ij')
vmax = np.percentile(np.abs(np.concatenate([data[i]['P_kk'][phase == 0].ravel() for i in shown])), 99)
norm = TwoSlopeNorm(vcenter=0.0, vmin=-vmax, vmax=vmax)
fig, axes = plt.subplots(1, 4, figsize=(12, 5.2), layout='constrained')
for ax, i in zip(axes, shown):
    d = data[i]
    Fm = d['F_macro']
    ux = np.pad(d['u'][0], ((0, 1), (0, 1)), mode='wrap')
    uy = np.pad(d['u'][1], ((0, 1), (0, 1)), mode='wrap')
    for a in range(2):                      # the computed 1 x 2 cell, repeated 2 x 1 (two cells wide)
        Xs = X + a * Lx
        xd = (Fm[0, 0] * Xs + Fm[0, 1] * Y + ux) * mm
        yd = (Fm[1, 0] * Xs + Fm[1, 1] * Y + uy) * mm
        pcm = ax.pcolormesh(xd, yd, np.ma.masked_where(phase != 0, d['P_kk']), cmap='coolwarm', norm=norm,
                            shading='flat')
        ax.pcolormesh(xd, yd, np.ma.masked_where(phase != 2, phase), cmap=ListedColormap(['#4d4d4d']), shading='flat')
    box = Fm @ (np.array([[0, 1, 1, 0, 0], [0, 0, 1, 1, 0]]) * np.array([[Lx], [Ly]]))
    ax.plot(box[0] * mm, box[1] * mm, 'k-', lw=1.5)
    ax.set(aspect='equal', title=f'$\\varepsilon$ = {float(d["eps"]):.3f},  {float(d["stress_MPa"]):.4f} MPa',
           xlabel='x [mm]')
    ax.set_xlim(-0.5, 2 * Lx * mm + 0.5)
axes[0].set_ylabel('y [mm]')
fig.colorbar(pcm, ax=axes, shrink=0.7, extend='both', label='$P_{xx}$ in the matrix [MPa]')
fig.savefig(os.path.join(OUT, 'fig_snapshots.png'), dpi=200)
print('saved', OUT)
