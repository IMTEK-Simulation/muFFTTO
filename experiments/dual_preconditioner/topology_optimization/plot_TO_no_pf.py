"""Phase-field TO with and without the phase-field term (PF_WEIGHT = 0 vs 0.2), 64^2 and 128^2, sine start,
Green-Jacobi, contrast 1e4, eta = 4h (irrelevant for w = 0), up to 1000 L-BFGS iterations (logs/TO_no_pf/).
Final designs on their exact pixels; printed: objective, grey fraction (0.05 < rho < 0.95), iterations.
    python plot_TO_no_pf.py  ->  figures/TO_no_pf.png
"""
import glob
import os
import re

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import netCDF4
import numpy as np

HERE = os.path.dirname(os.path.realpath(__file__))
D = os.path.join(HERE, 'data', 'example_2D_elasticity_TO')
plt.rcParams.update({'text.usetex': True, 'font.family': 'serif', 'font.size': 13})
fig, axs = plt.subplots(2, 2, figsize=(9, 9.6), layout='constrained')
for i, w in enumerate(('0', '0.2')):
    for j, N in enumerate((64, 128)):
        fs = glob.glob(os.path.join(D, f'N{N}_soft0.0001_energy-warm_Green_Jacobi_eta_*_pfw{w}_l_bfgs_nopf_test_final.nc'))
        running = not fs                                   # unfinished run: last stored iterate of the history
        f = fs[0] if fs else glob.glob(os.path.join(
            D, f'N{N}_soft0.0001_energy-warm_Green_Jacobi_eta_*_pfw{w}_l_bfgs_nopf_test_history.nc'))[0]
        with netCDF4.Dataset(f) as d:
            v = d['phase_field_solution'] if 'phase_field_solution' in d.variables else d['phase_field_iterate']
            r = np.asarray(v[-1, 0])
        log = open(os.path.join(HERE, 'logs', 'TO_no_pf', f'N{N}_w{w}.log')).read()
        its = re.findall(r'^\s*(\d+)\s+([0-9.]+e[+-]\d+)\s+[0-9.]+e[+-]\d+', log, re.M)
        nit, fobj = int(its[-1][0]), float(its[-1][1])
        grey = float(((r > 0.05) & (r < 0.95)).mean())
        ax = axs[i, j]
        ax.imshow(r.T, origin='lower', cmap='Greys', vmin=0, vmax=1, interpolation='nearest', extent=(0, 1, 0, 1))
        ax.set_xticks([]), ax.set_yticks([])
        ax.set_title(rf'${N}^2$, ' + (r'\textbf{no phase-field term} ($w = 0$)' if w == '0' else r'phase field $w = 0.2$')
                     + f'\n$f = {fobj:.3g}$, grey {100 * grey:.0f}\\,\\%, {nit} it' + (' (running)' if running else ''),
                     fontsize=13)
        cgs = [int(c) for c in re.findall(r'CG mech\s*=\s*(\d+)', log)]
        print(f'N {N} w {w}: f {fobj:.4g}, grey fraction {grey:.3f}, iterations {nit}, mean CG per solve '
              f'{np.mean(cgs):.0f} (last 30 solves {np.mean(cgs[-30:]):.0f})' + (' RUNNING' if running else ''))
out = os.path.join(HERE, 'figures', 'TO_no_pf.png')
fig.savefig(out, dpi=130)
print('wrote', out)
