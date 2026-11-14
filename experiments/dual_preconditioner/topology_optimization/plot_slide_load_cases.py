"""Slide 'Stress difference minimization' (new version): the three macroscopic load cases in one row -- unit cell
undeformed (grey) and deformed by the macro strain eps_i (exaggerated, blue outline): eps_1 uniaxial xx, eps_2
uniaxial yy, eps_3 shear xy; below each 'eps_i -> sigma_i^target'. Replaces marco_loading_noC.pdf (Ipe) on the slide.
    python plot_slide_load_cases.py             ->  figures/slide_load_cases.{pdf,png} and the talk figures/
    LAYOUT=column python plot_slide_load_cases.py  ->  slide_load_cases_col: the three cells in a column, label right"""
import os
import shutil
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon, FancyArrowPatch

HERE = os.path.dirname(os.path.realpath(__file__))
TALK = os.path.expanduser('~/Projects/Talks/Slides-Green-Jacobi-topology-optimization-NANTES/figures')
plt.rcParams.update({'text.usetex': True, 'font.family': 'serif', 'font.size': 14,
                     'text.latex.preamble': r'\usepackage{amsmath}'})
A = 0.28                                                 # exaggeration of the strain in the drawing
CASES = [np.array([[A, 0.0], [0.0, 0.0]]), np.array([[0.0, 0.0], [0.0, A]]), np.array([[0.0, A / 2], [A / 2, 0.0]])]
SQ = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], float)
COLUMN = os.environ.get('LAYOUT') == 'column'
if COLUMN:
    fig, axs = plt.subplots(3, 1, figsize=(2.9, 5.6))
    fig.subplots_adjust(left=0.0, right=0.62, top=0.99, bottom=0.01, hspace=0.08)
else:
    fig, axs = plt.subplots(1, 3, figsize=(6.0, 2.15))
    fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.17, wspace=0.12)
for k, (ax, e) in enumerate(zip(axs, CASES)):
    c = SQ.mean(0)
    deformed = c + (SQ - c) @ (np.eye(2) + e).T          # deformation about the cell centre
    ax.add_patch(Polygon(SQ, closed=True, fc='0.88', ec='0.45', lw=1.2))
    ax.add_patch(Polygon(deformed, closed=True, fc='none', ec='tab:blue', lw=2.0, ls=(0, (4, 2))))
    # one arrow per loaded edge: from the undeformed to the deformed edge midpoint
    for m in ((0, 0.5), (1, 0.5), (0.5, 0), (0.5, 1)):
        m = np.array(m, float)
        q = c + (np.eye(2) + e) @ (m - c)
        if np.linalg.norm(q - m) > 1e-9:
            ax.add_patch(FancyArrowPatch(m, q, arrowstyle='-|>', mutation_scale=13, color='tab:blue', lw=1.8,
                                         shrinkA=0, shrinkB=0))
    ax.text(*((1.45, 0.5) if COLUMN else (0.5, -0.42)), rf'$\bar{{\boldsymbol{{\varepsilon}}}}_{k + 1} \;\longrightarrow\; '
                        rf'\boldsymbol{{\sigma}}_{k + 1}^{{\mathrm{{target}}}}$', ha='left' if COLUMN else 'center', va='center', fontsize=23 if COLUMN else 15, clip_on=False)
    ax.set_xlim(-0.35, 1.35), ax.set_ylim(-0.25, 1.25), ax.set_aspect('equal'), ax.axis('off')
out = os.path.join(HERE, 'figures', 'slide_load_cases' + ('_col' if COLUMN else ''))
for ext in ('pdf', 'png'):
    fig.savefig(f'{out}.{ext}', dpi=200, **(dict(bbox_inches='tight', pad_inches=0.03) if COLUMN else {}))
    shutil.copy(f'{out}.{ext}', TALK)
print(out)
