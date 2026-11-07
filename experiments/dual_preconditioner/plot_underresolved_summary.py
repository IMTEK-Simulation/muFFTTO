"""
Does an under-resolved crack favour Green?  Cross-study summary (2026-11-07) of the mean CG iterations per elasticity
solve DURING cracking, Green vs dual, against the resolution l/h (l = phase-field length, b for PF-CZM, l_c for the
Fourel nonlocal model).
  (a) cost ratio  Green / (c x dual),  c = 2 in 2D, 1.5 in 3D (cost of a dual iteration in Green iterations)
      > 1: dual cheaper, < 1: Green cheaper
  (b) the same cost ratio against the grid size N (l/h and N are coupled in all studies but CH small l)
Numbers are the study summaries (each study's own plot script / README):
  CH small l    schneider2025_notch/plot_AT2_ch_small_l_cost.py (128^2, whole run to cut-through)
  CH 3D         schneider2025_notch/plot_AT2_ch3D_crack_staggered.py (crack step, l = 1/16, 32^3 ... 256^3)
  fatigue       phase_field_fatigue/plot_FAT_mesh_study.py (NEMO, during growth)
  PF-CZM        localization_softening_bands/plot_pfczm_mesh.py (tension, cracking)
  battery       battery_particle_fracture/plot_battery_mesh.py (Tu parameters, l = 0.23 um, 6 um cell, during)
  ASR (old)     asr_gel_expansion/plot_asr_mesh_study.py (Falsafi, AT2 hybrid, during)
  ASR Fourel    asr_gel_expansion/plot_fourel_mesh.py (their nonlocal model, Newton + PCG, during cracking)
  ASR PF A / B  asr_gel_expansion/plot_asrpf_mesh.py (AT1 Fourel-matched / PF-CZM real Gf; A 256^2 only to 0.5 %)
    python plot_underresolved_summary.py   ->  figures/underresolved_green_vs_dual.png
"""
import os

import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.realpath(__file__))
plt.rcParams.update({'font.size': 13})

# name: (dim, [(l/h, Green CG/solve, dual CG/solve), ...], colour, marker)
S = {
    'CH (AT2), 2D, l shrunk at 128$^2$': (2, [(0.25, 123, 102), (0.5, 122, 95), (1, 153, 75), (2, 178, 63),
                                                (4, 276, 62)], 'k', 'o'),
    'CH (AT2), 3D, crack step': (3, [(2, 70, 36), (4, 320, 61), (8, 249, 42), (16, 324, 40)], 'k', 's'),
    'fatigue (Carrara), 2D': (2, [(1, 104, 93), (2, 167, 84), (4, 283, 70), (8, 500, 58)], 'tab:purple', 'D'),
    'fatigue (Carrara), 3D': (3, [(0.5, 76, 240), (1, 109, 191)], 'tab:purple', 'd'),
    'PF-CZM concrete (Wu), 2D': (2, [(0.48, 162, 71), (0.96, 316, 101), (1.92, 510, 110), (3.84, 662, 108),
                                     (7.7, 706, 100)], 'tab:orange', 'v'),
    'battery (Tu), 2D': (2, [(1.23, 109, 56), (2.45, 233, 53), (4.9, 270, 51), (9.8, 322, 52), (19.6, 336, 52),
                             (39, 357, 58)], 'tab:brown', '^'),
    'battery (Tu), 3D': (3, [(1.23, 117, 54), (2.45, 170, 49), (4.9, 209, 46)], 'tab:brown', '<'),
    'ASR old (AT2 hybrid), 2D': (2, [(0.5, 79, 77), (1, 134, 66), (2, 156, 59), (4, 191, 69), (8, 255, 76),
                                     (16, 265, 85)], 'tab:cyan', 'p'),
    'ASR old (AT2 hybrid), 3D': (3, [(0.5, 40, 38), (1, 93, 76), (2, 80, 52)], 'tab:cyan', 'h'),
    'ASR Fourel nonlocal, 2D': (2, [(0.46, 260, 585), (0.91, 610, 955), (1.83, 520, 1050)], 'tab:pink', 'X'),
    'ASR PF A (AT1 Fourel-matched), 2D': (2, [(0.91, 104, 143), (1.8, 168, 156), (3.7, 31, 37.5)], 'tab:olive', 'P'),
    'ASR PF B (PF-CZM real Gf), 2D': (2, [(1.8, 27, 18.4), (3.7, 24, 16.2)], 'tab:gray', '*'),
}

# grid size N (pixels per side) of each point, same order as the points above
NGRID = {
    'CH (AT2), 2D, l shrunk at 128$^2$': [128] * 5,
    'CH (AT2), 3D, crack step': [32, 64, 128, 256],
    'fatigue (Carrara), 2D': [64, 128, 256, 512],
    'fatigue (Carrara), 3D': [32, 64],
    'PF-CZM concrete (Wu), 2D': [32, 64, 128, 256, 512],
    'battery (Tu), 2D': [32, 64, 128, 256, 512, 1024],
    'battery (Tu), 3D': [32, 64, 128],
    'ASR old (AT2 hybrid), 2D': [32, 64, 128, 256, 512, 1024],
    'ASR old (AT2 hybrid), 3D': [32, 64, 128],
    'ASR Fourel nonlocal, 2D': [64, 128, 256],
    'ASR PF A (AT1 Fourel-matched), 2D': [64, 128, 256],
    'ASR PF B (PF-CZM real Gf), 2D': [128, 256],
}
fig, axs = plt.subplots(1, 2, figsize=(18, 7.5), layout='constrained')
for name, (dim, pts, col, mk) in S.items():
    lh, g, d = (np.array(v, dtype=float) for v in zip(*pts))
    c = 2.0 if dim == 2 else 1.5
    ls = '-' if dim == 2 else '--'
    axs[0].plot(lh, g / (c * d), mk + ls, color=col, lw=2, ms=9, label=name)
    axs[1].plot(NGRID[name], g / (c * d), mk + ls, color=col, lw=2, ms=9, label=name)
    for x, y, n in zip(lh, g / (c * d), NGRID[name]):
        axs[0].annotate(f'{n}', (x, y), textcoords='offset points', xytext=(5, 4), fontsize=8.5, color=col)

for a, yl, t in [(axs[0], 'Green / (c x dual)   [cost ratio, c = 2 (2D), 1.5 (3D)]',
                  '(a) cost ratio vs l/h (labels: grid N)'),
                 (axs[1], 'Green / (c x dual)   [cost ratio]', '(b) the same against the grid size N')]:
    a.set_xscale('log'), a.set_yscale('log')
    a.axhline(1, color='k', lw=1.5)
    yt = [0.1, 0.2, 0.5, 1, 2, 5, 10]
    a.set_yticks(yt), a.set_yticklabels([f'{v:g}' for v in yt])
    a.minorticks_off()
    a.set_ylim(0.1, 10)
    a.set(ylabel=yl, title=t)
    a.grid(alpha=0.3)
a = axs[0]
a.axvspan(0.2, 2, color='0.9', zorder=0)
a.text(0.24, 7, 'l/h < 2\nunder-resolved', fontsize=12, va='top')
xt = [0.25, 0.5, 1, 2, 4, 8, 16, 32]
a.set_xticks(xt), a.set_xticklabels([f'{v:g}' for v in xt]), a.minorticks_off()
a.set_xlim(0.2, 45), a.set_xlabel('l / h  (phase-field length per pixel)')
a = axs[1]
xt = [32, 64, 128, 256, 512, 1024]
a.set_xticks(xt), a.set_xticklabels([str(v) for v in xt]), a.minorticks_off()
a.set_xlim(25, 1300), a.set_xlabel('grid N (N$^2$ in 2D, N$^3$ in 3D)')
axs[1].legend(fontsize=10.5, loc='lower right', ncol=1, framealpha=0.95, title='solid 2D, dashed 3D',
              title_fontsize=11)
fig.suptitle('During cracking: Green vs dual against the resolution of the crack (all studies)', fontsize=16)
out = os.path.join(HERE, 'figures', 'underresolved_green_vs_dual.png')
os.makedirs(os.path.dirname(out), exist_ok=True)
fig.savefig(out, dpi=120)
print('wrote', out)
