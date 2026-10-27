"""
Figure of run_box_kff.sh: inner PCG on S = P G0 P preconditioned by K_FF = P K_ref P (the fixed-dof block of the
reference stiffness), against no preconditioner and the family FFT preconditioner (2D square, all four faces clamped).

    (a) sorted eigenvalues of C^-1 S at N_SPEC     (b) cond(C^-1 S) vs N     (c) inner steps per outer iteration vs N

usage:  python box_kff_plot.py      -> exp_data/box_kff.png (copied to report/figs)
"""
import os
import shutil

import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker

HERE = os.path.dirname(os.path.realpath(__file__))
RUNS = os.path.join(HERE, 'exp_data', 'runs_box')
NS = (32, 64, 128, 256)
N_SPEC = int(os.environ.get('N_SPEC', 128))
plt.rcParams.update({'font.size': 12})
CURVES = [('none', 'cg1e-06', 'no preconditioner', 'tab:blue', ':'),
          ('kff', 'kff1e-06', '$\\mathbf{K}_{FF} = \\mathbf{P}\\mathbf{K}_{ref}\\mathbf{P}$', 'tab:red', '-'),
          ('family', 'family1e-06', 'family FFT', 'tab:orange', '-')]


def load(n, tag):
    f = os.path.join(RUNS, f'N{n}_box_{tag}.npz')
    return {k: v[()] if v.ndim == 0 else v for k, v in np.load(f).items()} if os.path.exists(f) else None


def label_end(ax, x, y, text, colour, dy=-4):
    ax.annotate(text, (x, y), textcoords='offset points', xytext=(8, dy), fontsize=10.5, color=colour)


fig, axes = plt.subplots(1, 3, figsize=(20, 6), layout='constrained')
spec = load(N_SPEC, 'dense')
for key, _, label, colour, ls in CURVES:
    ev = spec.get(f'eig_{key}') if spec else None
    if ev is None:
        continue
    ev = ev / ev.max()  # scaling does not matter for CG; normalized so that lambda_max = 1
    axes[0].plot(np.linspace(0, 1, len(ev)), ev, ls=ls, color=colour, lw=2.2,
                 label=f'{label}: cond {1 / ev[0]:.1f}')
axes[0].set(yscale='log', xlabel='eigenvalue index (fraction of $m$)',
            ylabel='eigenvalue of $\\mathbf{C}^{-1}\\mathbf{S}$ (scaled to $\\lambda_{\\max} = 1$)',
            title=f'(a) spectrum, N = {N_SPEC}')
axes[0].legend(frameon=False, fontsize=10.5, loc='lower right')
for key, run, label, colour, ls in CURVES:
    pts = [(n, r[f'eig_{key}']) for n in NS if (r := load(n, 'dense')) is not None and f'eig_{key}' in r]
    if pts:
        conds = [ev[-1] / ev[0] for _, ev in pts]
        axes[1].plot([n for n, _ in pts], conds, ls=ls, marker='o', color=colour, lw=2, ms=6)
        label_end(axes[1], pts[-1][0], conds[-1], f'{label}: {conds[-1]:.0f}', colour)
    pts = [(n, r) for n in NS if (r := load(n, run)) is not None]
    if pts:
        steps = [r['inner_its'].mean() for _, r in pts]
        axes[2].plot([n for n, _ in pts], steps, ls=ls, marker='o', color=colour, lw=2, ms=6)
        label_end(axes[2], pts[-1][0], steps[-1], f'{label}: {steps[-1]:.1f}', colour)
axes[1].set(yscale='log', ylabel='cond($\\mathbf{C}^{-1}\\mathbf{S}$)', title='(b) condition number vs mesh')
axes[2].set(ylim=(0, None), ylabel='inner steps per outer iteration (tolerance 1e-6)',
            title='(c) inner PCG steps per outer CG iteration')
for ax in axes[1:]:
    ax.set(xscale='log', xlabel='N (pixels over the square)')
    ax.set_xticks(NS, [str(n) for n in NS])
    ax.set_xlim(26, 900)
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
for ax in axes:
    ax.grid(True, color='0.92', which='both')
    ax.spines[['top', 'right']].set_visible(False)
fig.suptitle('Inner solve of $\\mathbf{S} = \\mathbf{P}\\mathbf{G}_0\\mathbf{P}$ (2D square, all four faces clamped): '
             'is the sub-block $\\mathbf{K}_{FF}$ of the reference stiffness its inverse?', fontsize=13)
out = os.path.join(HERE, 'exp_data', 'box_kff.png')
fig.savefig(out, dpi=110)
shutil.copy(out, os.path.join(HERE, 'report', 'figs', 'box_kff.png'))
print('wrote', out)
