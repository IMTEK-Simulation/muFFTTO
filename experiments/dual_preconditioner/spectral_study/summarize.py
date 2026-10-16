"""Tables and figures of study.py:  python summarize.py 16 32  -> results/summary.md, results/*.png"""
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
NAMES = ['green', 'green_jacobi', 'dual', 'dual_solid', 'schur_neumann_1', 'schur_neumann_3', 'schur_A']
LABEL = {'green': 'Green', 'green_jacobi': 'Green-Jacobi', 'dual': 'dual (C_ref = I)',
         'dual_solid': 'dual (C_ref = solid)', 'schur_neumann_1': 'Schur, Neumann 1', 'schur_neumann_3': 'Schur, Neumann 3',
         'schur_A': 'Schur, S~ = Γ_S A Γ_S'}
COLOR = {'green': '#2a78d6', 'green_jacobi': '#eb6834', 'dual': '#1baf7a', 'dual_solid': '#008300',
         'schur_neumann_1': '#e87ba4', 'schur_neumann_3': '#a3456b', 'schur_A': '#52514e'}
MARKER = {'green': 'o', 'green_jacobi': 's', 'dual': '^', 'dual_solid': 'v', 'schur_neumann_1': 'D',
          'schur_neumann_3': 'P', 'schur_A': 'x'}

sizes = list(map(int, sys.argv[1:] or ['16']))
data = {N: json.load(open(HERE / 'results' / f'study_N{N}.json')) for N in sizes
        if (HERE / 'results' / f'study_N{N}.json').exists()}

lines = ['# Dual preconditioner study', '',
         f'PCG iterations to ||e_k||_K <= 1e-6 ||e_0||_K (true energy error); "-" = P indefinite. '
         'Cost per application in Green (FFT-pair) applications: Green, Green-Jacobi 1; dual 2 (+1 K); '
         'Schur Neumann k: k+3; Schur A: 4.', '']
for N, cases in data.items():
    lines += [f'## {N}²', '', '| physics | geometry | k_v | ' + ' | '.join(LABEL[n] for n in NAMES) + ' |',
              '|---|---|---|' + '---|' * len(NAMES)]
    for c in cases:
        it = [c['results'].get(n, {}).get('iterations') for n in NAMES]
        lines.append(f'| {c["physics"]} | {c["geometry"]} | {c["k_v"]:g} | '
                     + ' | '.join('-' if v is None else str(v) for v in it) + ' |')
    lines += ['', f'Spectral interval [lambda_min, lambda_max] of P K, {N}²:', '',
              '| physics | geometry | k_v | ' + ' | '.join(LABEL[n] for n in NAMES[:-1]) + ' |',
              '|---|---|---|' + '---|' * (len(NAMES) - 1)]
    for c in cases:
        cell = []
        for n in NAMES[:-1]:
            r = c['results'].get(n, {})
            cell.append('-' if r.get('lam_min') is None else f'[{r["lam_min"]:.2g}, {r["lam_max"]:.2g}]')
        lines.append(f'| {c["physics"]} | {c["geometry"]} | {c["k_v"]:g} | ' + ' | '.join(cell) + ' |')
    lines.append('')
(HERE / 'results' / 'summary.md').write_text('\n'.join(lines) + '\n')
print('\n'.join(lines))

# iterations against the contrast: rows physics, columns geometry; solid lines N = largest, dashed N = smallest
for N, cases in data.items():
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    for row, physics in enumerate(['scalar', 'elastic']):
        for col, geom in enumerate(['bubble_array', 'square_inclusion']):
            ax = axes[row, col]
            sel = sorted([c for c in cases if c['physics'] == physics and c['geometry'] == geom], key=lambda c: c['k_v'])
            for n in NAMES[:-1]:
                kv = [c['k_v'] for c in sel if c['results'].get(n, {}).get('iterations') is not None]
                it = [c['results'][n]['iterations'] for c in sel if c['results'].get(n, {}).get('iterations') is not None]
                ax.loglog(kv, it, color=COLOR[n], marker=MARKER[n], lw=2, ms=7, label=LABEL[n])
            ax.set_title(f'{physics}, {geom}, {N}²')
            ax.invert_xaxis()
            ax.set_ylim(1, 300)
            ax.grid(alpha=0.3, which='both')
            if row == 1:
                ax.set_xlabel('void / inclusion stiffness $k_v$ (contrast →)')
            if col == 0:
                ax.set_ylabel('PCG iterations (energy error 1e-6)')
    axes[0, 0].legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(HERE / 'results' / f'iterations_N{N}.png', dpi=150)
    plt.close(fig)
