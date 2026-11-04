"""
2D: phase contrast x crack length x grid, for Green, Green-Jacobi and dual -- all runs of the smoothness study
(run_AT2_smoothness_2D*.sh: N = 32, 64, 128; l = 1/8 ... 1/64 with l/h >= 1). Work per elastic solve = CG iterations x
K applications per iteration (dual 2), mean over the run.

Material settings (rows), by stiffness contrast:
    pores = k = 1e-3   pores of stiffness 1e-3 without damage, g(d) = (1 - d)^2 + 1e-3     contrast 1e3 throughout
    pores 1e-3 + d     pores 1e-3 g(d) with damage solved in them, k = 1e-4                1e3, a few voxels 1e-7 cracked
    pores = k = 1e-4   pores of stiffness 1e-4 without damage, k = 1e-4                    1e4 throughout
    inclusions 10x     stiff tough inclusions, k = 1e-4                                    10 before, 1e5 cracked
    inclusions 1000x   the same, 1000x stiffer (mirror of the 1e-3 pores)                  1e3 before, 1e7 cracked

figures/AT2_contrast_map_2D.png       one panel per (setting, preconditioner): work over grid (x) and l (y); a row of
                                      cells = fixed l (smooth crack), a diagonal = fixed l/h (sharp crack)
figures/AT2_contrast_scaling_2D.png   grid exponent b (work ~ N^b, fit over 32^2 - 128^2) per preconditioner against the
                                      setting, for a sharp (l/h = 2) and a smooth (l = 1/16) crack

PHASE (env): average over the whole run (all, default) or one phase of it, from the stress-strain curve of the .nc
output: pre (load steps up to the peak stress), cracking (after the peak up to and including the load step in which
sigma_xx drops below 0.2 peak), post (the later steps, cell cut through); output files get the suffix _<PHASE>.

usage:  python plot_AT2_contrast_map_2D.py        PHASE=cracking python plot_AT2_contrast_map_2D.py
"""
import os

import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.realpath(__file__))
GRIDS = (32, 64, 128)
LENGTHS = (0.125, 0.0625, 0.03125, 0.015625)
PRECS = [('Green', 'Green', 1, 'C2', 'o'), ('Green_Jacobi', 'Green-Jacobi', 1, 'C0', 's'),
         ('dual', 'dual (2 per iteration)', 2, 'C3', '^')]
SETTINGS = [('_holes0.001', '_k0.001', 'pores = k = 1e-3', 'contrast 1e3'),
            ('', '', 'pores 1e-3 + damage, k = 1e-4', 'contrast 1e3 (1e7 in a few cracked pore voxels)'),
            ('_holes0.0001', '', 'pores = k = 1e-4', 'contrast 1e4'),
            ('_incl10', '', 'stiff inclusions 10x, k = 1e-4', 'contrast 10 before, 1e5 after cracking'),
            ('_incl1000', '', 'stiff inclusions 1000x, k = 1e-4', 'contrast 1e3 before, 1e7 after cracking')]


PHASE = os.environ.get('PHASE', 'all')
SUFFIX = '' if PHASE == 'all' else f'_{PHASE}'
PHASE_TITLE = {'all': 'whole run', 'pre': 'pre-crack (up to the peak stress)',
               'cracking': 'cracking (after the peak, incl. the breaking step)',
               'post': 'post-crack (after the breaking step)'}[PHASE]


def phase_work(base, p):
    """work per elastic solve of run <base> (exp_data/<base>_iterations.csv, .nc) in PHASE; nan if missing or empty"""
    f, nc = os.path.join(HERE, 'exp_data', base + '_iterations.csv'), os.path.join(HERE, 'exp_data', base + '.nc')
    if not os.path.exists(f):
        return np.nan
    d = np.genfromtxt(f, delimiter=',', names=True, dtype=None, encoding=None)
    keep = np.ones(len(d), bool)
    if PHASE != 'all':
        if not os.path.exists(nc):
            return np.nan
        import netCDF4
        with netCDF4.Dataset(nc) as ds:
            steps, sxx = np.asarray(ds['step'][:]).astype(int), np.asarray(ds['macro_stress'][:, 0, 0])
        peak_step = steps[sxx.argmax()]
        broken = steps[(steps > peak_step) & (sxx < 0.2 * sxx.max())]
        first_broken = broken.min() if len(broken) else np.inf
        # the step that breaks the cell (the first one below 0.2 peak) belongs to cracking
        keep = {'pre': d['step'] <= peak_step, 'cracking': (d['step'] > peak_step) & (d['step'] <= first_broken),
                'post': d['step'] > first_broken}[PHASE]
    if not keep.any():
        return np.nan
    return d['cg_u'][keep].mean() * dict((q[0], q[2]) for q in PRECS)[p]


def work(n, p, pre, post, ell):
    base = f'AT2_N{n}_{p}_polydisperse_pores{pre}_l{ell:g}{post}_smooth'
    log = os.path.join(HERE, 'logs', base + '.log')
    if not os.path.exists(log) or 'Elapsed time' not in open(log).read():
        return np.nan
    return phase_work(base, p)


W = np.array([[[[work(n, p, pre, post, ell) if n * ell >= 1 else np.nan for n in GRIDS] for ell in LENGTHS]
               for p, *_ in PRECS] for pre, post, *_ in SETTINGS])          # [setting, prec, l, N]

# ---------------------------------------------------------------------------------------------------------------------
fig, axes = plt.subplots(len(SETTINGS), 3, figsize=(15, 4.1 * len(SETTINGS)), layout='constrained', dpi=150)
for s, (_, _, name, contrast) in enumerate(SETTINGS):
    vmax = np.nanmax(W[s]) if np.isfinite(W[s]).any() else 1.0
    for k, (_, label, *_) in enumerate(PRECS):
        ax = axes[s, k]
        im = ax.imshow(W[s, k], cmap='viridis', vmin=0, vmax=vmax, aspect='auto')
        for i, ell in enumerate(LENGTHS):
            for j, n in enumerate(GRIDS):
                v = W[s, k, i, j]
                if n * ell < 1:
                    continue
                txt = '...' if np.isnan(v) else f'{v:.0f}'
                ax.text(j, i - 0.1, txt, ha='center', va='center', fontsize=14, fontweight='bold',
                        color='w' if np.isnan(v) or v < 0.6 * vmax else 'k')
                ax.text(j, i + 0.28, f'l/h = {n * ell:g}', ha='center', va='center', fontsize=9,
                        color='w' if np.isnan(v) or v < 0.6 * vmax else 'k')
        ax.set_xticks(range(len(GRIDS))), ax.set_xticklabels([f'{n}$^2$' for n in GRIDS], fontsize=12)
        ax.set_yticks(range(len(LENGTHS))), ax.set_yticklabels([f'l = 1/{1 / ell:g}' for ell in LENGTHS], fontsize=12)
        ax.set_xlabel('grid', fontsize=12)
        ax.set_title(f'{label}\n{name}' + ('' if k else f'\n({contrast})'), fontsize=12)
    fig.colorbar(im, ax=axes[s], shrink=0.85, label='work per elastic solve')
out = os.path.join(HERE, 'figures', f'AT2_contrast_map_2D{SUFFIX}.png')
fig.savefig(out)
print(f'wrote {out}')

# ---------------------------------------------------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(15, 6), layout='constrained', dpi=150, sharey=True)
x = np.arange(len(SETTINGS))
SHORT = ['pores = k\n= 1e-3\n1e3 | 1e3', 'pores 1e-3\n+ damage\n1e3 | (1e7)', 'pores = k\n= 1e-4\n1e4 | 1e4',
         'inclusions\n10x\n10 | 1e5', 'inclusions\n1000x\n1e3 | 1e7']   # contrast before | after cracking
print('grid exponents b, settings in the order: ' + ' | '.join(q[2] for q in SETTINGS))
for ax, (kind, title) in zip(axes, (('sharp', 'sharp crack: fixed l/h = 2 (l = 2/N)'),
                                    ('smooth', 'smooth crack: fixed l = 1/16 (l/h = 2, 4, 8)'))):
    for k, (p, label, _, c, m) in enumerate(PRECS):
        b = []
        for s in range(len(SETTINGS)):
            if kind == 'sharp':
                y = [W[s, k, LENGTHS.index(2 / n), j] for j, n in enumerate(GRIDS)]
            else:
                y = list(W[s, k, LENGTHS.index(0.0625)])
            ok = np.isfinite(y)
            b.append(np.polyfit(np.log(np.array(GRIDS)[ok]), np.log(np.array(y)[ok]), 1)[0] if ok.sum() >= 2
                     else np.nan)
        ax.bar(x + (k - 1) * 0.26, b, width=0.26, color=c, label=label)
        for xi, bi in zip(x + (k - 1) * 0.26, b):
            if np.isfinite(bi):
                ax.text(xi, bi + (0.02 if bi >= 0 else -0.06), f'{bi:.2f}', ha='center', fontsize=10)
        print(f'{kind:6s} {p:13s} b = ' + ' '.join(f'{bi:5.2f}' for bi in b))
    ax.axhline(0, color='k', lw=0.8)
    ax.set_xticks(x), ax.set_xticklabels(SHORT, fontsize=11)
    ax.set_title(f'{title}\nwork ~ N$^b$ over 32$^2$ - 128$^2$ (b = 0: grid independent; '
                 f'b = 1: doubles per grid doubling)', fontsize=12)
    ax.set_ylabel('grid exponent b', fontsize=12), ax.tick_params(axis='y', labelsize=12, labelleft=True)
    ax.set_xlabel('setting (stiffness contrast before | after cracking)', fontsize=12)
    ax.grid(True, axis='y', color='0.9')
axes[0].legend(fontsize=12)
out = os.path.join(HERE, 'figures', f'AT2_contrast_scaling_2D{SUFFIX}.png')
fig.savefig(out)
print(f'\nwrote {out}')
