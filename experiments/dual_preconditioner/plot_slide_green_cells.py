"""Slide 'Green's operator preconditioning' (new version): the periodic cell on an 8 x 8 pixel grid (P1 triangles, as
the old Ipe drawings domain_1_data_discrete_Prec*_y.pdf) -- left the heterogeneous data C(x): three inclusions
rasterised onto the pixels (pixel centre inside -> pixel takes the phase; the smooth shapes as thin outlines), right
the homogeneous reference C^ref on the same grid.
    python plot_slide_green_cells.py  ->  talk figures/slide_green_cell_{hetero,ref}.pdf"""
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse, Polygon

TALK = os.path.expanduser('~/Projects/Talks/Slides-Green-Jacobi-topology-optimization-NANTES/figures')
plt.rcParams.update({'text.usetex': True, 'font.family': 'serif', 'font.size': 14,
                     'text.latex.preamble': r'\usepackage{amsmath}'})
N = 8
# inclusions: (centre, width, height, angle [deg], colour) -- ellipses, as the three blobs of the old drawing
INCL = [((0.37, 0.72), 0.50, 0.17, -25, '#c97a7a'),     # red, upper
        ((0.68, 0.43), 0.17, 0.27, 30, '#7a7ac9'),      # blue, right
        ((0.27, 0.30), 0.20, 0.42, 10, '#8c8c8c')]      # grey, lower left


def inside(e, x, y):
    (cx, cy), w, hh, ang, _ = e
    t = np.deg2rad(ang)
    u = (x - cx) * np.cos(t) + (y - cy) * np.sin(t)
    v = -(x - cx) * np.sin(t) + (y - cy) * np.cos(t)
    return (u / (w / 2)) ** 2 + (v / (hh / 2)) ** 2 <= 1


def cell(hetero, name):
    fig, ax = plt.subplots(figsize=(2.4, 2.4))
    ax.set_position([0.02, 0.02, 0.96, 0.96])
    h = 1.0 / N
    for i in range(N):
        for j in range(N):
            xc, yc = (i + 0.5) * h, (j + 0.5) * h
            col = '#eef3ee' if not hetero else '#ffffff'
            if hetero:
                for e in INCL:
                    if inside(e, xc, yc):
                        col = e[4]
            ax.add_patch(Polygon([(i * h, j * h), ((i + 1) * h, j * h), ((i + 1) * h, (j + 1) * h), (i * h, (j + 1) * h)],
                                 closed=True, fc=col, ec='none', alpha=0.75 if hetero else 1.0))
    for k in range(N + 1):                                  # grid lines
        ax.plot([k * h, k * h], [0, 1], color='0.35', lw=0.5)
        ax.plot([0, 1], [k * h, k * h], color='0.35', lw=0.5)
    for k in range(-N, N):                                  # diagonals of the P1 triangles
        x0, x1 = max(0, -k * h), min(1, 1 - k * h)
        ax.plot([x0, x1], [x0 + k * h, x1 + k * h], color='0.45', lw=0.35)
    if hetero:
        for (c, w, hh, ang, colr) in INCL:
            ax.add_patch(Ellipse(c, w, hh, angle=ang, fill=False, ec=colr, lw=2.2, ls=(0, (3, 2))))
    ax.plot([0, 1, 1, 0, 0], [0, 0, 1, 1, 0], color='k', lw=1.6)
    ax.text(0.03, 0.03, r'$\mathcal{Y}$', ha='left', va='bottom', fontsize=24,
            bbox=dict(fc='white', ec='none', pad=0.5))
    lab = r'$\mathbf{C}(\boldsymbol{x})$' if hetero else r'$\mathbf{C}^{\mathrm{ref}}$'
    ax.text(0.97, 0.97, lab, ha='right', va='top', fontsize=26, bbox=dict(fc='white', ec='none', pad=1.0, alpha=0.9))
    ax.set_xlim(-0.01, 1.01), ax.set_ylim(-0.01, 1.01), ax.set_aspect('equal'), ax.axis('off')
    out = os.path.join(TALK, f'slide_green_cell_{name}.pdf')
    fig.savefig(out), fig.savefig(out.replace('.pdf', '.png'), dpi=200)
    plt.close(fig)
    print(out)


cell(True, 'hetero')
cell(False, 'ref')
