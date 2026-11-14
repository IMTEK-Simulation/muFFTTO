"""Geometry panels for the appendix slides "fracture models" of the Freiburg talk: one periodic unit cell per study of
frac_summary_total (plot_slide_fracture_whole_run.py), built from the geometry files of the runs; matrix light, second
phase grey, pores white, notch / pre-crack red, macroscopic load arrows.
    python plot_appendix_fracture_geometries.py  ->  figures/appendix_geom_<name>.pdf (+ copy to the talk figures)"""
import os
import shutil
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

HERE = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, os.path.join(HERE, '../..'))
sys.path.insert(0, os.path.join(HERE, '../../.claude/worktrees/polycrystal-crack/experiments/dual_preconditioner/ernesti2020_fibers'))
TALK = os.path.expanduser('~/Projects/Talks/Slides-Green-Jacobi-topology-optimization-NANTES/figures')
plt.rcParams.update({'text.usetex': True, 'font.family': 'serif', 'font.size': 11})
MATRIX, PHASE2, VOID, CRACK, TEXT = '#e9e8e4', '#8f8d87', '#ffffff', '#d6402a', '#0b0b0b'


def panel(name, img, colours, load=None, crack=None, label='', extra=None):
    """img[x, y] integer phase image on the unit cell; colours per phase value; load 'xx' / 'yy' / None;
    crack ((x0, x1), y) red segment; label = cell size text under the cell"""
    fig, ax = plt.subplots(figsize=(2.6, 2.6))
    ax.imshow(img.T, origin='lower', extent=(0, 1, 0, 1), interpolation='nearest',
              cmap=ListedColormap(colours), vmin=-0.5, vmax=len(colours) - 0.5)
    ax.plot([0, 1, 1, 0, 0], [0, 0, 1, 1, 0], color=TEXT, lw=0.8)
    if crack is not None:
        (x0, x1), y = crack
        ax.plot([x0, x1], [y, y], color=CRACK, lw=2.2, solid_capstyle='butt')
    kw = dict(arrowstyle='-|>', mutation_scale=11, lw=1.4, color=TEXT)
    if load == 'xx':
        for y in (0.3, 0.5, 0.7):
            ax.annotate('', (1.17, y), (1.03, y), arrowprops=kw)
            ax.annotate('', (-0.17, y), (-0.03, y), arrowprops=kw)
        ax.text(1.1, 0.83, r'$\bar\varepsilon_{xx}$', ha='center')
    elif load == 'yy':
        for x in (0.3, 0.5, 0.7):
            ax.annotate('', (x, 1.17), (x, 1.03), arrowprops=kw)
            ax.annotate('', (x, -0.17), (x, -0.03), arrowprops=kw)
        ax.text(0.86, 1.08, r'$\bar\varepsilon_{yy}$', ha='left')
    if extra:
        extra(ax)
    ax.text(0.5, -0.27 if load == 'yy' else -0.1, label, ha='center', va='top', fontsize=10)
    ax.set_xlim(-0.22, 1.22), ax.set_ylim(-0.32, 1.22), ax.set_aspect('equal'), ax.axis('off')
    out = os.path.join(HERE, 'figures', f'appendix_geom_{name}.pdf')
    fig.savefig(out, bbox_inches='tight', pad_inches=0.02)
    shutil.copy(out, TALK)
    print(out)


# porous: B2 nine polydisperse pores (AT2 and AT1 runs), pores stiffness 1e-3, uniaxial strain xx
from muFFTTO import microstructure_library
n = 1024
x = np.arange(n) / n
pores = np.asarray(microstructure_library.get_geometry(nb_voxels=(n, n), microstructure_name='polydisperse_pores',
                                                       coordinates=np.array(np.meshgrid(x, x, indexing='ij')))) == 0
panel('porous', pores.astype(int), [MATRIX, VOID], 'xx', label=r'$L = 1$, 9 pores')

# Cahn-Hilliard two-phase image (Schneider & Kaestner 2025 type), 0 = phase 2 (stiffness 0.1, tough), E_yy
ch = np.load(os.path.join(HERE, 'damage_AT2_small_strain/schneider2025_notch/geometry/cahn_hilliard_512.npy'))
panel('cahn_hilliard', (ch == 0).astype(int), [MATRIX, PHASE2], 'yy', label=r'$L = 1$ m, phase 2: 42\,\%')

# fatigue: glass beads (0 = bead) 20 %, cell 64 l, phase-field pre-crack x = 0.375 ... 0.625 at y = 1/2, cyclic E_yy
beads = np.load(os.path.join(HERE, 'phase_field_fatigue/geometry/beads2D_2048.npy'))
panel('fatigue', (beads == 0).astype(int), [MATRIX, PHASE2], 'yy', crack=((0.375, 0.625), 0.5),
      label=r'$L = 64\,l = 12.8$ mm')

# PF-CZM concrete: aggregates (1) 45 %, cell 100 mm, E_xx
agg = np.load(os.path.join(HERE, 'localization_softening_bands/exp_data_nemo/PFCZM2D_N1024_dual_xx_b1.5mm_k1e-05_phase.npy'))
panel('concrete', (agg == 1).astype(int), [MATRIX, PHASE2], 'xx', label=r'$L = 100$ mm, aggregates 45\,\%')

# battery: NMC811 core R = 2 um, NMC532 shell 0.2 R, soft matrix, 6 um cell; surface crack 0.3 h from the surface along +x
c = (np.arange(n) + 0.5) / n * 6.0 - 3.0
r = np.hypot(*np.meshgrid(c, c, indexing='ij'))
bat = np.where(r < 2.0, 2, np.where(r < 2.4, 1, 0))
panel('battery', bat, [VOID, '#c9c6bf', PHASE2], None, crack=((0.5 + 2.28 / 6, 0.5 + 2.4 / 6), 0.5),
      label=r'$L = 6\,\mu$m, $\bar\varepsilon = 0$, Li swelling',
      extra=lambda ax: (ax.text(0.5, 0.5, 'core', ha='center', va='center', fontsize=9, color='w'),
                        ax.text(0.5, 0.5 + 2.2 / 6, 'shell', ha='center', va='center', fontsize=7.5)))

# fibres: 16 glass fibres 45 %, 512 um cell, seed 0, E_xx
import fiber_geometry
fib = np.asarray(fiber_geometry.phase(n, 0))
panel('fibres', (fib == 0).astype(int), [MATRIX, PHASE2], 'xx', label=r'$L = 512\,\mu$m, 16 fibres')
