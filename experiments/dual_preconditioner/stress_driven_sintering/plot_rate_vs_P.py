"""
Stage 1 test: shrinkage rate -d(tip distance)/dt versus pressure, compared at equal neck width (same geometry), the
two-particle analogue of the fixed-density comparison of Seiz et al. 2024, Fig. 2b.  Uses analyse_two_particles.py output.
Local rate = slope of a line fitted to tip(t) on snapshots with neck within +-NECK_BAND of the target.
Writes figures/rb_rate_vs_P.png
"""
import os
import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

here = os.path.dirname(os.path.abspath(__file__))
base = 'SINT_N128_Green_W4_P0u_k0.0001'
pressures = [0, 50, 100, 200]
targets = [42.0, 46.0, 48.0]
NECK_BAND = 3.0


def load(tag):
    return np.load(os.path.join(here, 'exp_data', f'{base}_{tag}_two_particle.npz'))


def rate_at_neck(m, neck):
    w = np.abs(m['neck'] - neck) <= NECK_BAND
    w &= m['t'] > 5e-5                                     # skip the initial neck-formation transient
    return -np.polyfit(m['t'][w] * 1e3, m['tip'][w], 1)[0]


print('tip shrinkage rate [nm/ms] at equal neck width')
print('neck   ' + ''.join(f'{p:>9d}' for p in pressures) + '    no-RB')
table = {}
for n in targets:
    table[n] = [rate_at_neck(load(f'rb_P{p}_tau2e-6'), n) for p in pressures]
    print(f'{n:5.1f}  ' + ''.join(f'{v:9.2f}' for v in table[n]) + f'{rate_at_neck(load("two_P0"), n):9.2f}')

plt.rcParams.update({'font.size': 13})
fig, ax = plt.subplots(figsize=(6.4, 4.6), constrained_layout=True)
for n, c, mk in zip(targets, ('#444444', '#1f77b4', '#d95f02'), ('o', 's', '^')):
    ax.plot(pressures, table[n], color=c, marker=mk, ms=8, lw=2, label=f'neck width {n:g} nm')
ax.set_xlabel('pressure [MPa]')
ax.set_ylabel('tip shrinkage rate [nm/ms]')
ax.set_xlim(left=0)
ax.set_ylim(bottom=0)
ax.grid(alpha=0.3)
ax.legend()
fig.savefig(os.path.join(here, 'figures', 'rb_rate_vs_P.png'), dpi=150)
