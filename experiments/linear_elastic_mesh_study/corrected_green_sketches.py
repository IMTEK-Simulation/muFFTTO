"""
Schematic figures for report/corrected_green.tex (section on solving the capacitance system S = E^T G E):

    report/figs/fixed_sets.pdf     2D grid sketches: clamp on the solid face only / the full grid line through the void /
                                   two parallel lines / four lines (box) with the two families and the crossing nodes
    report/figs/wall_symbol.pdf    the wall FFT on real data (2D, full clamp line): the kernel of S along the line, and
                                   the eigenvalues of its 2x2 symbol per wave number (cond(S) ~ 1/h)
    report/figs/planes_3D.pdf      3D sketches: the cantilever with its clamp plane through the cell; the cube with the
                                   six face planes in three families of parallel planes

usage:  python corrected_green_sketches.py      (also writes .png copies to look at)
"""
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

HERE = os.path.dirname(os.path.realpath(__file__))
FIGS = os.path.join(HERE, 'report', 'figs')
os.makedirs(FIGS, exist_ok=True)
plt.rcParams.update({'font.size': 11})
SOLID, VOID_EDGE = '#c9c9c9', '#e8e8e8'
FAM_V, FAM_H = 'tab:blue', 'tab:orange'


def save(fig, name):
    fig.savefig(os.path.join(FIGS, name + '.pdf'), bbox_inches='tight')
    fig.savefig(os.path.join(FIGS, name + '.png'), bbox_inches='tight', dpi=130)
    plt.close(fig)


# ---------------------------------------------------------------------------------------------------------------------
# 1. fixed sets on a periodic 2D grid
nx, ny = 14, 10  # pixels of the periodic cell; nodes 0..nx-1 (node nx = node 0)
sx, sy = (2, 12), (2, 8)  # solid pixels sx[0] <= i < sx[1]; its faces are the node lines i = sx[0], sx[1]


def cell(ax, title, caption):
    for i in range(nx):
        for j in range(ny):
            solid = sx[0] <= i < sx[1] and sy[0] <= j < sy[1]
            ax.add_patch(Rectangle((i, j), 1, 1, facecolor=SOLID if solid else 'white',
                                   edgecolor='white' if solid else VOID_EDGE, lw=0.8))
    ax.add_patch(Rectangle((0, 0), nx, ny, fill=False, ls='--', lw=1.2, edgecolor='0.35'))
    ii, jj = np.meshgrid(np.arange(nx), np.arange(ny), indexing='ij')
    ax.plot(ii.ravel(), jj.ravel(), '.', color='0.6', ms=2.5)
    for x in (nx / 2,):  # the periodicity: top and bottom (and left and right) are the same nodes
        ax.add_patch(FancyArrowPatch((nx + 0.6, 0.3), (nx + 0.6, ny - 0.3), arrowstyle='<->', mutation_scale=9,
                                     color='0.45', lw=0.8))
    ax.text(nx + 0.95, ny / 2, 'periodic', rotation=90, va='center', fontsize=8, color='0.4')
    ax.set(xlim=(-0.6, nx + 1.4), ylim=(-0.6, ny + 0.4), aspect='equal', xticks=[], yticks=[])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title(title, fontsize=11.5, loc='left')
    ax.text(0, -0.9, caption, va='top', fontsize=9.5, linespacing=1.35)


def nodes(ax, pts, colour, solid_only_fill=True):
    """fixed nodes: filled on (or at the boundary of) the solid, hollow in the void (fixed, but changes nothing)"""
    for i, j in pts:
        on_solid = sx[0] <= i <= sx[1] and sy[0] <= j <= sy[1]
        ax.plot(i, j, 's', ms=6.5, mec=colour, mfc=colour if on_solid else 'white', mew=1.4, zorder=5)
        for image in ([(i, ny)] if j == 0 else []) + ([(nx, j)] if i == 0 else []):  # periodic image of the node
            ax.plot(*image, 's', ms=6.5, mec=colour, mfc='white', mew=1.2, alpha=0.4, zorder=5)


fig, axes = plt.subplots(1, 4, figsize=(17, 4.6))
cell(axes[0], '(a) clamp on the solid face',
     '$\\mathbf{S}$ not circulant: the fixed nodes do not\nform a full periodic line.\n'
     '$\\to$ dense $\\mathbf{S}$ ($m$ Green applications, $m^3$)\n   or inner CG / PCG')
nodes(axes[0], [(sx[0], j) for j in range(sy[0], sy[1] + 1)], 'k')
cell(axes[1], '(b) full grid line through the void',
     'void nodes fixed too: zero stiffness,\nsame problem. $\\mathbf{S}$ block-circulant.\n'
     '$\\to$ 1D FFT along the line, $2\\times2$ block\n   per wave number: exact, no assembly')
nodes(axes[1], [(sx[0], j) for j in range(ny)], 'k')
cell(axes[2], '(c) two parallel full lines',
     'still diagonal in the wave number\nalong the lines.\n'
     '$\\to$ 1D FFT, $4\\times4$ block per wave number:\n   exact')
nodes(axes[2], [(x, j) for x in sx for j in range(ny)], 'k')
cell(axes[3], '(d) four lines (all faces): two families',
     'crossings break the invariance: no exact FFT.\n'
     '$\\to$ inner PCG, block-Jacobi over the families,\n   each family exact by its FFT; crossing\n'
     '   nodes in $\\mathbf{both}$ families (overlap)')
nodes(axes[3], [(x, j) for x in sx for j in range(ny) if j not in sy], FAM_V)
nodes(axes[3], [(i, y) for y in sy for i in range(nx) if i not in sx], FAM_H)
for x in sx:  # crossing nodes: in both families
    for y in sy:
        axes[3].plot(x, y, 's', ms=9, mfc=FAM_V, mec=FAM_H, mew=2.4, zorder=6)
axes[3].plot([], [], 's', mfc=FAM_V, mec=FAM_V, label='vertical family')
axes[3].plot([], [], 's', mfc=FAM_H, mec=FAM_H, label='horizontal family')
axes[3].plot([], [], 's', mfc=FAM_V, mec=FAM_H, mew=2, label='crossing: both')
axes[3].legend(loc='upper left', bbox_to_anchor=(-0.02, -0.33), ncol=3, fontsize=8.5, frameon=False,
               handletextpad=0.2, columnspacing=0.8)
h = [plt.Line2D([], [], marker='s', ls='', mfc='k', mec='k', ms=6.5, label='fixed node on the solid'),
     plt.Line2D([], [], marker='s', ls='', mfc='white', mec='k', mew=1.4, ms=6.5, label='fixed node in the void'),
     Rectangle((0, 0), 1, 1, facecolor=SOLID, label='solid (reference material)'),
     Rectangle((0, 0), 1, 1, facecolor='white', edgecolor=VOID_EDGE, label='void (zero stiffness)'),
     plt.Line2D([], [], marker='s', ls='', mfc='white', mec='k', mew=1.2, alpha=0.4, ms=6.5,
                label='periodic image (same node)')]
fig.legend(handles=h, loc='lower center', ncol=5, frameon=False, bbox_to_anchor=(0.5, -0.02), fontsize=10)
fig.subplots_adjust(wspace=0.12, bottom=0.36)
save(fig, 'fixed_sets')

# ---------------------------------------------------------------------------------------------------------------------
# 2. the wall FFT on real data: a full clamp line in a 2D periodic cell, reference = isotropic solid
from muFFTTO import domain, material_models

lam, mu = material_models.get_lame_parameters(E=1.0, poisson=0.3)
fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), layout='constrained')
for n_line, ls in ((32, ':'), (64, '--'), (128, '-')):
    cellp = domain.PeriodicUnitCell(domain_size=[1.0, 1.0], problem_type='elasticity')
    disc = domain.Discretization(cell=cellp, nb_of_pixels_global=(n_line, n_line),
                                 discretization_type='finite_element', element_type='bilinear_rectangle')
    green = disc.get_preconditioner_Green_mugrid(
        reference_material_data_ijkl=material_models.get_elastic_tensor_from_lame(dim=2, lam=lam, mu=mu),
        name=f'green_{n_line}')
    f, g = disc.get_unknown_size_field(name=f'f{n_line}'), disc.get_unknown_size_field(name=f'g{n_line}')
    kernel = np.zeros((2, 2, n_line))  # line x = 0, unit force at node (0, 0)
    for e in range(2):
        f.s[...] = 0
        f.s[e, 0, 0, 0] = 1.0
        disc.fft.communicate_ghosts(f)
        disc.apply_preconditioner_mugrid(preconditioner_Fourier_fnfnqks=green, input_nodal_field_fnxyz=f,
                                         output_nodal_field_fnxyz=g)
        kernel[:, e] = g.s[:, 0, 0, :]
    offsets = np.fft.fftshift(np.fft.fftfreq(n_line, 1 / n_line))
    for (d, e), colour, lab in (((0, 0), 'tab:blue', '$S_{xx}$'), ((1, 1), 'tab:red', '$S_{yy}$'),
                                ((0, 1), 'tab:green', '$S_{xy}$')):
        axes[0].plot(offsets / n_line, np.fft.fftshift(kernel[d, e]), ls=ls, color=colour,
                     label=f'{lab}' if n_line == 128 else None)
    symbol = np.fft.fft(kernel, axis=-1).transpose(2, 0, 1)
    ev = np.linalg.eigvalsh(0.5 * (symbol + np.conj(symbol.transpose(0, 2, 1))))
    xi = np.fft.fftfreq(n_line, 1 / n_line)
    order = np.argsort(xi)
    for k, colour in ((0, 'tab:purple'), (1, 'tab:brown')):
        axes[1].plot(xi[order][xi[order] > 0], ev[order, k][xi[order] > 0], ls=ls, color=colour,
                     label=(f'eigenvalue {k + 1} of $\\hat{{\\mathbf{{S}}}}(\\xi)$' if n_line == 128 else None))
    axes[1].plot([], [], ls=ls, color='0.3', label=f'{n_line} nodes on the line')
axes[0].set(xlabel='offset along the line (cell lengths)', ylabel='kernel of $\\mathbf{S}$',
            title='response along the line to a unit force at one line node\n(one Green application per component)')
axes[0].legend(frameon=False)
axes[1].set(xscale='log', yscale='log', xlabel='wave number $\\xi$ along the line',
            ylabel='eigenvalues of the $2\\times2$ symbol',
            title='its FFT: a $2\\times2$ block per wave number, $\\hat{\\mathbf{S}}(\\xi) \\sim 1/\\xi$\n'
                  '(so cond $\\mathbf{S} \\sim 1/h$; inverted block by block: exact)')
axes[1].legend(frameon=False, fontsize=9)
for ax in axes:
    ax.grid(True, color='0.92', which='both')
    ax.spines[['top', 'right']].set_visible(False)
save(fig, 'wall_symbol')


# ---------------------------------------------------------------------------------------------------------------------
# 3. 3D sketches
def box_faces(x0, x1, y0, y1, z0, z1):
    c = np.array([[x, y, z] for x in (x0, x1) for y in (y0, y1) for z in (z0, z1)])
    idx = [[0, 1, 3, 2], [4, 5, 7, 6], [0, 1, 5, 4], [2, 3, 7, 6], [0, 2, 6, 4], [1, 3, 7, 5]]
    return [c[i] for i in idx]


def plane(axis, p, lo, hi):
    pts = {0: [(p, lo, lo), (p, hi, lo), (p, hi, hi), (p, lo, hi)],
           1: [(lo, p, lo), (hi, p, lo), (hi, p, hi), (lo, p, hi)],
           2: [(lo, lo, p), (hi, lo, p), (hi, hi, p), (lo, hi, p)]}[axis]
    return [np.array(pts)]


def cell_wire(ax, size):
    for f in box_faces(0, size[0], 0, size[1], 0, size[2]):
        ax.add_collection3d(Poly3DCollection([f], facecolor=(0, 0, 0, 0), edgecolor='0.55', lw=0.7, linestyle='--'))


fig = plt.figure(figsize=(14, 6))
ax = fig.add_subplot(1, 2, 1, projection='3d')
L, H, W, t = 1.0, 0.2, 0.2, 0.08
cell_wire(ax, (L + 2 * t, H + 2 * t, W + 2 * t))
ax.add_collection3d(Poly3DCollection(box_faces(t, t + L, t, t + H, t, t + W), facecolor=SOLID, edgecolor='0.4',
                                     lw=0.6, alpha=0.95))
pl = np.array([(t, 0, 0), (t, H + 2 * t, 0), (t, H + 2 * t, W + 2 * t), (t, 0, W + 2 * t)])
ax.add_collection3d(Poly3DCollection([pl], facecolor=(0.12, 0.47, 0.71, 0.35), edgecolor=FAM_V, lw=1.5))
ax.add_collection3d(Poly3DCollection([np.array([(t, t, t), (t, t + H, t), (t, t + H, t + W), (t, t, t + W)])],
                                     facecolor='k', edgecolor='k', alpha=0.85))
ax.quiver(t + L, t + H / 2, t + W / 2, 0, -0.12, 0, color='tab:red', lw=2, arrow_length_ratio=0.3)
ax.text(t + L + 0.02, t + H / 2 - 0.15, t + W / 2, 'load', color='tab:red', fontsize=10)
ax.set_box_aspect((L + 2 * t, H + 2 * t, W + 2 * t))
ax.set_title('cantilever: the clamp plane through the whole cell\n(blue: the solid face and its continuation '
             'through the void)\n$\\to$ 2D FFT over the plane, $3\\times3$ block per $(\\xi_y, \\xi_z)$: exact', fontsize=10.5)
ax.view_init(elev=22, azim=-58)
ax.set_axis_off()

ax = fig.add_subplot(1, 2, 2, projection='3d')
s, t = 1.0, 0.12
cell_wire(ax, (s + 2 * t,) * 3)
ax.add_collection3d(Poly3DCollection(box_faces(t, t + s, t, t + s, t, t + s), facecolor=SOLID, edgecolor='0.4',
                                     lw=0.6, alpha=0.25))
for axis, colour in ((0, FAM_V), (1, FAM_H), (2, 'tab:green')):
    rgba = matplotlib.colors.to_rgba(colour, 0.22)
    for p in (t, t + s):
        ax.add_collection3d(Poly3DCollection(plane(axis, p, 0, s + 2 * t), facecolor=rgba, edgecolor=colour, lw=1.2))
ax.set_box_aspect((1, 1, 1))
ax.set_title('cube clamped on all faces: three families of two parallel planes\n(normal x: blue, y: orange, z: green; '
             'edges and corners in every family)\n$\\to$ inner PCG, each family exact by a 2D FFT, $6\\times6$ block',
             fontsize=10.5)
ax.view_init(elev=24, azim=-55)
ax.set_axis_off()
fig.subplots_adjust(wspace=0.0)
save(fig, 'planes_3D')
print('wrote report/figs/fixed_sets, wall_symbol, planes_3D (.pdf, .png)')
