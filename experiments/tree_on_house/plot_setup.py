"""Setup sketch of the pole-on-house simulation at N^2 pixels (exact pixels).  usage: N=64 python plot_setup.py"""
import os

import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch, FancyArrowPatch

import tree_house_geometry as g

N = int(os.environ.get('N', 64))
HERE = os.path.dirname(os.path.realpath(__file__))
centres = np.stack(np.meshgrid((np.arange(N) + 0.5) / N, (np.arange(N) + 0.5) / N, indexing='ij'))
lab = g.labels(centres)
crown = g.crown(centres)
COLOURS = ['#f4f7fb', '#7a6a55', '#b8b8bd', '#b2182b', '#3d7a2f']
NAMES = ['third medium (air, contact)', 'ground (bottom row clamped)', 'walls and ceiling', 'roof', 'pole']
fig, ax = plt.subplots(figsize=(8.5, 8.5), layout='constrained')
edges = np.linspace(0, 1, N + 1)
ax.pcolormesh(edges, edges, lab.T, cmap=ListedColormap(COLOURS), vmin=-0.5, vmax=4.5, edgecolors='0.85', linewidth=0.2)
ax.pcolormesh(edges, edges, np.ma.masked_where(~crown.T, np.ones_like(lab.T)), cmap=ListedColormap(['#9fd18a']), edgecolors='none')
# clamp, load, notch, fall
ax.plot([g.FRAME, 1 - g.FRAME], [g.GROUND[0] - 0.006] * 2, color='k', lw=4)
s_top = np.array(g.POLE_BASE) + g.POLE_L * np.array([np.sin(g.TILT), np.cos(g.TILT)])
ax.add_patch(FancyArrowPatch(s_top + (0.03, 0.0), s_top + (0.16, 0.0), arrowstyle='-|>', mutation_scale=22, lw=2.5,
                             color='#c4441a'))
ax.text(*(s_top + (0.17, 0.0)), 'prescribed horizontal\ndisplacement of the crown', va='center', fontsize=10,
        color='#c4441a')
notch = np.array(g.POLE_BASE) + g.NOTCH[0] * np.array([np.sin(g.TILT), np.cos(g.TILT)])
ax.annotate('notch (tension side):\nthe pole breaks here', notch - (0.03, 0.0), xytext=(0.04, 0.22), fontsize=10,
            arrowprops=dict(arrowstyle='->', lw=1.2))
theta = np.linspace(g.TILT, np.deg2rad(52), 40)
r = g.POLE_L * 0.92
ax.plot(notch[0] + r * np.sin(theta), notch[1] + r * np.cos(theta), ls=(0, (4, 3)), color='#3d7a2f', lw=1.5)
ax.text(0.55, 0.78, 'it bends, cracks at the notch,\nrotates onto the roof (contact through\n'
        'the third medium) and pushes it:\nroof and walls crack (AT2)', fontsize=10, color='#3d7a2f')
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_aspect('equal')
ax.set_xticks([])
ax.set_yticks([])
ax.legend(handles=[Patch(color=c, label=n) for c, n in zip(COLOURS, NAMES)] +
          [Patch(color='#9fd18a', label='pole crown (displacement prescribed)')],
          loc='upper left', fontsize=9, frameon=True)
ax.set_title(f'pole falling on a house: 2D setup, {N}$^2$ pixels, finite strain, third-medium contact, AT2 fracture',
             fontsize=11)
out = os.path.join(HERE, f'setup_tree_house_N{N}.png')
fig.savefig(out, dpi=130)
print(f'wrote {out}')
