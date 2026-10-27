"""
What sets the length of the CG plateau of the cantilever? Exact CG energy error vs iteration for one mesh, varying
one thing at a time (results stored by cantilever_mesh_study.py in exp_data/runs/, run them with run_plateau.sh):

    A. beam length L = 0.5, 1, 2, 4               (same clamped nodes, longer distance load -> clamp)
    B. clamped node columns 1, 2, 4, 8            (more clamped nodes, nearly the same beam)
    C. clamped part of the left face 1, 1/2, 1/4  (fewer clamped nodes)
    D. beam L = 4 loaded at x = d = 0.5, 1, 2, 4   (distance load -> clamp at a fixed beam length)

usage:  python plateau_study.py [--n_H 32]
writes exp_data/plateau_study_n_H<n>.png
"""
import argparse
import os

import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--n_H', type=int, default=32)
parser.add_argument('--cg_tol', type=float, default=1e-6)
args = parser.parse_args()

RUNS = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'exp_data', 'runs')
studies = [('A. beam length L', [('L', v) for v in (0.5, 1.0, 2.0, 4.0)], 'L'),
           ('B. clamped node columns', [('cc', v) for v in (1, 2, 4, 8)], 'columns'),
           ('C. clamped part of the left face', [('span', v) for v in (1.0, 0.5, 0.25)], 'span'),
           ('D. beam L = 4, load moved to x = d', [('load', v) for v in (0.5, 1.0, 2.0, 4.0)], 'd')]
defaults = {'L': 1.0, 'cc': 1, 'span': 1.0}


def load(key, value):
    if key == 'load':  # beam L = 4, loaded at x = value (the right face for value 4)
        tag = '_L4' + (f'_load{value:g}' if value != 4.0 else '')
    else:
        tag = f'_{key}{value:g}' if value != defaults[key] else ''
    path = os.path.join(RUNS, f'n_H{args.n_H}_cgtol{args.cg_tol:g}{tag}.npz')
    return np.load(path) if os.path.exists(path) else None


def plateau_end(e):
    """first iteration with the error below 0.3 (the end of the stagnation; a load near the clamp drops to ~0.6 at once)"""
    below = np.flatnonzero(e < 0.3)
    return int(below[0]) if len(below) else len(e)


fig, axes = plt.subplots(1, 4, figsize=(24, 6.2), layout='constrained', sharey=True)
summary = []
for ax, (title, variants, name) in zip(axes, studies):
    colours = plt.cm.viridis(np.linspace(0, 0.85, len(variants)))
    for (key, value), c in zip(variants, colours):
        r = load(key, value)
        if r is None:
            continue
        Pi = r['energy_functional']
        e = np.sqrt(np.maximum(Pi[:-1] - Pi[-1], 0) / -Pi[-1])
        e[e < 1e-7] = np.nan  # round-off of the energy difference
        k_plateau = plateau_end(e)
        nb_clamped = int(r['nb_clamped_nodes']) if 'nb_clamped_nodes' in r.files else args.n_H + 1  # older runs: face
        n_L = int(round(float(r['pixels'][0]))) - (int(r['pixels'][1]) - args.n_H)  # beam pixels along x
        ax.semilogy(e, color=c, lw=1.6, label=f'{name} = {value:g}:  {nb_clamped} clamped nodes, '
                                              f'plateau {k_plateau}, {int(r["cg_its"])} its')
        summary.append((title[:2], f'{name}={value:g}', n_L, nb_clamped, k_plateau, int(r['cg_its'])))
    ax.axhline(args.cg_tol, color='C3', ls='-.', lw=1)
    ax.set_title(title, fontsize=11)
    ax.set_xlabel('CG iteration')
    ax.grid(True, color='0.9', lw=0.5)
    ax.legend(frameon=False, fontsize=9, loc='upper center', bbox_to_anchor=(0.5, -0.13))
axes[0].set_ylabel(r'relative energy error  $\|u - u_k\|_K / \|u\|_K$  (exact)')
fig.suptitle(f'cantilever with void, $n_H$ = {args.n_H} pixels over the height: what sets the CG plateau?',
             fontsize=12)
out = os.path.join(os.path.dirname(RUNS), f'plateau_study_n_H{args.n_H}.png')
fig.savefig(out, dpi=150)
print(f'{"study":6s} {"variant":12s} {"beam px":>8s} {"clamped":>8s} {"plateau":>8s} {"CG its":>7s}')
for row in summary:
    print(f'{row[0]:6s} {row[1]:12s} {row[2]:8d} {row[3]:8d} {row[4]:8d} {row[5]:7d}')
print(f'wrote {out}')
