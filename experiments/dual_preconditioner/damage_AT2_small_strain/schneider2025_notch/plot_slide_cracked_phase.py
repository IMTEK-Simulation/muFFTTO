"""
Slide figure 'the dual pays off in the cracked phase': CH two-phase AT2, 128^2, l/h = 4, k_res 1e-4
(run_AT2_cahn_hilliard_kres.sh logs), against the macroscopic strain:
  top     macroscopic stress (one curve, identical for all preconditioners); shaded: cracked phase (max d > 0.95)
  middle  mean CG per elasticity solve in each load step: Green / Green-Jacobi / dual
  bottom  cumulative elasticity work (CG iterations, dual x 2) -> the gap opens only once the crack is there
    python plot_slide_cracked_phase.py  ->  figures/slide_cracked_phase.{png,pdf} (+ talk figures/)
"""
import os
import re
import shutil

import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.realpath(__file__))
TALK = os.path.expanduser('~/Projects/Talks/Slides-Green-Jacobi-topology-optimization-NANTES/figures')
PRE = [('Green', 'Green', 'tab:green', 1), ('Green_Jacobi', 'Green-Jacobi', 'tab:blue', 1), ('dual', 'dual', 'tab:red', 2)]
pat = re.compile(r'^step\s+(\d+)\s+E_xx = ([\d.e+-]+)\s+sigma_xx = ([\d.e+-]+)\s+max d = ([\d.]+).*?step CG \(u, d\) = '
                 r'\((\d+), \d+\)\s+mean CG u = ([\d.]+)')
runs = {}
for P, *_ in PRE:
    rows = [pat.match(l) for l in open(os.path.join(HERE, 'logs', f'AT2_N128_{P}_incl0.1_l0.03125_k0.0001_ch.log'))]
    a = np.array([[float(m.group(i)) for i in (1, 3, 4, 5, 6)] for m in rows if m])
    a[:, 0] = (a[:, 0] + 1) * 2e-3 / 200                # E_yy from the step index (log prints only 4 decimals)
    runs[P] = a                                         # strain, stress, max d, step CG, mean CG per solve
plt.rcParams.update({'text.usetex': True, 'font.family': 'serif', 'font.size': 13})
fig, (a1, a2, a3) = plt.subplots(3, 1, figsize=(4.6, 6.2), sharex=True, layout='constrained',
                                 gridspec_kw=dict(height_ratios=[0.7, 1, 1]))
ref = runs['dual']
e = ref[:, 0] * 1e3
crack = e[np.argmax(ref[:, 2] > 0.95)]
for ax in (a1, a2, a3):
    ax.axvspan(crack, e.max(), color='0.9', zorder=0)
    ax.grid(alpha=0.3)
a1.plot(e, ref[:, 1] * 1e4, 'k-', lw=2)
a1.set(ylabel=r'$\bar\sigma\ [10^{-4}]$')
a1.set_ylim(0, None)
a1.text(crack + 0.02, a1.get_ylim()[1] * 0.12, 'cracked', fontsize=11)
for P, lab, col, c in PRE:
    r = runs[P]
    a2.plot(r[:, 0] * 1e3, c * r[:, 4], '-', color=col, lw=1.8, label=lab + (r' ($\times 2$)' if c == 2 else ''))
    a3.plot(r[:, 0] * 1e3, c * np.cumsum(r[:, 3]) / 1e3, '-', color=col, lw=2.2, label=lab + (r' ($\times 2$)' if c == 2 else ''))
    print(P, 'total work', int(c * r[:, 3].sum()), 'before crack', int(c * r[r[:, 0] * 1e3 < crack, 3].sum()))
a2.set(ylabel=r'CG per solve (dual $\times 2$)'), a2.set_ylim(0, None)
a2.legend(loc='upper left', fontsize=10, framealpha=0.95)
a3.set(ylabel=r'work [$10^3$ CG]', xlabel=r'macroscopic strain $\bar\varepsilon\ [10^{-3}]$')
a3.set_ylim(0, None), a3.set_xlim(0, e.max())
a3.legend(loc='upper left', fontsize=10, framealpha=0.95)
out = os.path.join(HERE, 'figures', 'slide_cracked_phase')
for ext in ('png', 'pdf'):
    fig.savefig(f'{out}.{ext}', dpi=150)
    shutil.copy(f'{out}.{ext}', TALK)
print('wrote', out, 'crack at strain', crack)
