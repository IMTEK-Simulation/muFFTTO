"""
3D counterpart of plot_TO_CH_starts.py: TO 64^3 (eta = 4h = 1/16, contrast 1e4, w = 0.2, sw5, Green-Jacobi) from the
standard sine start and from the 3D Cahn-Hilliard starts bi_s0 / bi_s1 (make_TO3D_CH_starts.py; NEMO, several
segments). Row 1: start design (soft phase isosurface rho = 0.5); row 2: final design; row 3 (one panel): objective
over the L-BFGS iterations of all segments. Printed: final objective and homogenised properties.
    python plot_TO3D_CH_starts_results.py  ->  figures/TO3D_CH_starts_results.png
"""
import glob
import os
import re

import matplotlib.pyplot as plt
import netCDF4
import numpy as np

from elastic_properties import properties
from plot_TO3D_results import surface, voigt_from_stresses

HERE = os.path.dirname(os.path.realpath(__file__))
DATA = os.path.join(HERE, 'data', 'example_3D_elasticity_TO')
STEM = 'N64_soft0.0001_energy-warm_Green_Jacobi_eta_0.0625_pfw0.2_sw5_l_bfgs_start_{}'
LABELS = ['sine', 'bi_s0', 'bi_s1']
plt.rcParams.update({'font.size': 13})


def segs(lab):
    fs = glob.glob(os.path.join(DATA, STEM.format(lab) + '*_history.nc'))
    key = lambda f: int(re.search(r'_seg(\d+)_history', f).group(1)) if '_seg' in f else 1
    return sorted([f for f in fs if re.search(rf'start_{lab}(_seg\d+)?_history\.nc$', f)], key=key)


fig = plt.figure(figsize=(16, 11))
gs = fig.add_gridspec(3, 3, height_ratios=[1, 1, 0.8], hspace=0.25)
axo = fig.add_subplot(gs[2, :])
for j, lab in enumerate(LABELS):
    hs = segs(lab)
    first = netCDF4.Dataset(hs[0])
    start = np.array(first['phase_field_iterate'][0, 0])
    obj, it0 = [], 0
    for h in hs:
        with netCDF4.Dataset(h) as d:
            o = np.array(d['objective'][:])
            obj.append((it0 + 10 * np.arange(len(o)), o))
            it0 += 10 * (len(o) - 1) + 1
    fin_files = [h.replace('_history.nc', '_final.nc') for h in hs if os.path.exists(h.replace('_history.nc', '_final.nc'))]
    with netCDF4.Dataset(fin_files[-1]) as d:
        final = np.array(d['phase_field_solution'][0, 0])
        pr = properties(voigt_from_stresses(np.array(d.getncattr('homogenized_stresses')).reshape(6, 3, 3)))
    f_end = obj[-1][1][-1]
    surface(fig.add_subplot(gs[0, j], projection='3d'), start, f'{lab}: start')
    surface(fig.add_subplot(gs[1, j], projection='3d'), final, f'{lab}: final, f = {f_end:.3f}, {it0} it')
    x = np.concatenate([a for a, _ in obj]); y = np.concatenate([b for _, b in obj])
    axo.plot(x, y, ':' if lab == 'sine' else '-', lw=2, label=lab)
    print(lab, 'segments', len(hs), 'iterations ~', it0, 'final f', round(float(f_end), 4), 'properties', pr)
axo.set_yscale('log'), axo.set(xlabel='L-BFGS iteration (all segments)', ylabel='objective f')
axo.grid(alpha=0.3, which='both'), axo.legend()
fig.suptitle('3D TO $64^3$ ($\\eta$ = 4h, contrast 1e4, w = 0.2) from different starts: soft phase shown', fontsize=15)
out = os.path.join(HERE, 'figures', 'TO3D_CH_starts_results.png')
fig.savefig(out, dpi=110)
print('wrote', out)
