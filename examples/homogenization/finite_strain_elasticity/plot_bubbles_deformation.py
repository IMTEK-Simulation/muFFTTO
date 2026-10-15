"""Deformation and statistics of the finite-strain bubble-array homogenization runs.

Reads exp_data/example_2D_homogenization_finite_strain_bubbles_zero_stiffness/Nx=<n>Ny=<n>_<preconditioner>[_void<r>]/
(info_log_final.npz and fields.npz, written with --save_fields).

Figure 1 (<out>_deformation.png): deformed cell at three load levels, one row per void stiffness. Every pixel is drawn
on its deformed corner nodes (exact pixels, no smoothing), coloured by det F in the solid; bubbles are white.
Figure 2 (<out>_statistics.png): macroscopic stress, CG iterations per Newton step, det F distribution in the solid,
CG iterations per load increment.

usage: python plot_bubbles_deformation.py OUT_PREFIX [--n 64] [--ratios 0 1e-05]
"""
import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

# reference palette, light surface; colour follows the preconditioner, line style the void stiffness
SURFACE, TEXT_1, TEXT_2, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1", "#a3a29c"
COLORS = {"Green": "#2a78d6", "Green_Jacobi": "#eb6834"}
LABELS = {"Green": "Green", "Green_Jacobi": "Green-Jacobi"}
# diverging det F: blue = compressed, neutral grey = unchanged volume, orange = expanded
DET_F_CMAP = LinearSegmentedColormap.from_list("det_F", ["#0d366b", "#3987e5", "#d9d8d4", "#f0a070", "#a8401a"])

parser = argparse.ArgumentParser()
parser.add_argument('out')
parser.add_argument('--n', type=int, default=64)
parser.add_argument('--ratios', type=float, nargs='+', default=[0.0, 1e-5])
args = parser.parse_args()

root = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'exp_data',
                    'example_2D_homogenization_finite_strain_bubbles_zero_stiffness')


def load(preconditioner, ratio):
    folder = os.path.join(root, f'Nx={args.n}Ny={args.n}_{preconditioner}' + (f'_void{ratio:g}' if ratio > 0 else ''))
    return (dict(np.load(os.path.join(folder, 'info_log_final.npz'), allow_pickle=True)),
            dict(np.load(os.path.join(folder, 'fields.npz'))))


def void_label(ratio):
    return 'void stiffness 0' if ratio == 0 else f'void stiffness {ratio:g} x solid'


def deformed_corners(displacement, macro_gradient):
    """Deformed positions of the pixel corners, (2, n+1, n+1): x = X + H X + u(X), u periodic."""
    n = displacement.shape[-1]
    X = np.stack(np.meshgrid(np.arange(n + 1) / n, np.arange(n + 1) / n, indexing='ij'))
    u = np.pad(displacement, ((0, 0), (0, 1), (0, 1)), mode='wrap')
    return X + np.einsum('ij,jxy->ixy', macro_gradient, X) + u


def style(ax, title=None, xlabel=None, ylabel=None):
    ax.set_facecolor(SURFACE)
    if title:
        ax.set_title(title, loc='left', fontsize=11.5, color=TEXT_1, pad=6)
    if xlabel:
        ax.set_xlabel(xlabel, color=TEXT_2, fontsize=10)
    if ylabel:
        ax.set_ylabel(ylabel, color=TEXT_2, fontsize=10)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    for side in ('left', 'bottom'):
        ax.spines[side].set_color(AXIS)
    ax.tick_params(colors=TEXT_2, labelsize=9.5)
    ax.grid(axis='y', color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


runs = {(p, r): load(p, r) for p in ('Green', 'Green_Jacobi') for r in args.ratios}
ninc = runs[('Green', args.ratios[0])][1]['det_F'].shape[0]
shown = sorted({max(int(round(f * ninc)) - 1, 0) for f in (0.3, 0.6, 1.0)})

# ---- figure 1: deformed cell ----------------------------------------------------------------------------------------
solid_det_F = np.concatenate([runs[('Green', r)][1]['det_F'][k][runs[('Green', r)][1]['solid']]
                              for r in args.ratios for k in shown])
spread = max(abs(solid_det_F.max() - 1), abs(1 - solid_det_F.min()))
norm = TwoSlopeNorm(vmin=1 - spread, vcenter=1.0, vmax=1 + spread)
fig, axes = plt.subplots(len(args.ratios), len(shown), figsize=(4.4 * len(shown), 4.4 * len(args.ratios) + 0.9),
                         squeeze=False, facecolor=SURFACE)
fig.subplots_adjust(left=0.03, right=0.88, top=0.86, bottom=0.03, wspace=0.08, hspace=0.18)
fig.text(0.03, 0.955, f'Bubble array under finite strain: deformed cell, {args.n}²', fontsize=15, fontweight='bold',
         color=TEXT_1)
total = runs[('Green', args.ratios[0])][1]['macro_gradient'][-1]
fig.text(0.03, 0.92, f'neo-Hookean solid, porosity 0.5, F - I up to [[{total[0, 0]:g}, {total[0, 1]:g}], '
                     f'[{total[1, 0]:g}, {total[1, 1]:g}]]; colour = det F in the solid, bubbles white; exact pixels',
         fontsize=10.5, color=TEXT_2)
for row, ratio in enumerate(args.ratios):
    _, fields = runs[('Green', ratio)]
    for column, k in enumerate(shown):
        ax = axes[row, column]
        corners = deformed_corners(fields['displacement'][k], fields['macro_gradient'][k])
        det_F = np.ma.masked_where(~fields['solid'], fields['det_F'][k])
        ax.pcolormesh(corners[0], corners[1], det_F, shading='flat', cmap=DET_F_CMAP, norm=norm, rasterized=True)
        outline = deformed_corners(np.zeros_like(fields['displacement'][k]), fields['macro_gradient'][k])
        for edge in (outline[:, 0, :], outline[:, -1, :], outline[:, :, 0], outline[:, :, -1]):
            ax.plot(edge[0], edge[1], color=AXIS, linewidth=0.8)
        ax.set_aspect('equal')
        ax.axis('off')
        ax.set_title(f'{void_label(ratio)} - load {100 * (k + 1) / ninc:.0f} %', loc='left', fontsize=10.5,
                     color=TEXT_1)
colorbar_axis = fig.add_axes([0.9, 0.12, 0.015, 0.6])
colorbar = fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=DET_F_CMAP), cax=colorbar_axis)
colorbar.set_label('det F (solid)', color=TEXT_2)
colorbar.outline.set_visible(False)
colorbar.ax.tick_params(colors=TEXT_2, labelsize=9)
fig.savefig(f'{args.out}_deformation.png', dpi=200, facecolor=SURFACE)

# ---- figure 2: statistics ---------------------------------------------------------------------------------------------
dashes = {ratio: style_ for ratio, style_ in zip(args.ratios, ('-', (0, (4, 2)), (0, (1, 1.5))))}
fig, axes = plt.subplots(2, 2, figsize=(14, 9), facecolor=SURFACE)
fig.subplots_adjust(left=0.07, right=0.98, top=0.88, bottom=0.08, hspace=0.38, wspace=0.22)
fig.text(0.07, 0.955, f'Bubble array under finite strain: response and solver statistics, {args.n}²', fontsize=15,
         fontweight='bold', color=TEXT_1)
fig.text(0.07, 0.92, 'Newton-CG with full steps, 10 load increments; CG stops on the energy-error estimate 1e-6 '
                     '(delay 10)', fontsize=10.5, color=TEXT_2)

ax = axes[0, 0]
style(ax, '(a) macroscopic first Piola-Kirchhoff stress', 'load fraction', 'mean P / E_solid')
for ratio in args.ratios:
    info, _ = runs[('Green', ratio)]
    load_fraction = np.arange(1, info['mean_P'].shape[0] + 1) / info['mean_P'].shape[0]
    for (i, j), marker in (((0, 1), 'o'), ((1, 1), 's')):
        ax.plot(load_fraction, info['mean_P'][:, i, j], color=TEXT_1 if (i, j) == (1, 1) else '#52514e',
                linestyle=dashes[ratio], marker=marker, markersize=4, linewidth=1.8,
                label=f'P{i}{j}, {void_label(ratio)}')
difference = max(np.abs(runs[('Green_Jacobi', r)][0]['mean_P'] - runs[('Green', r)][0]['mean_P']).max()
                 for r in args.ratios)
ax.legend(frameon=False, fontsize=9, loc='upper left', title_fontsize=9,
          title=f'Green and Green-Jacobi agree to |dP| <= {difference:.1e}')

ax = axes[0, 1]
style(ax, '(b) CG iterations per Newton step', 'Newton step (all increments)', 'CG iterations')
for (p, ratio), (info, _) in runs.items():
    y = info['cg_per_newton_flat']
    ax.plot(np.arange(1, len(y) + 1), y, color=COLORS[p], linestyle=dashes[ratio], linewidth=1.6, marker='o',
            markersize=3, label=f'{LABELS[p]}, {void_label(ratio)}')
ax.set_ylim(bottom=0)
ax.legend(frameon=False, fontsize=9, loc='upper left')

ax = axes[1, 0]
style(ax, '(c) det F in the solid at the final load', 'det F', 'fraction of solid pixels')
for ratio in args.ratios:
    _, fields = runs[('Green', ratio)]
    values = fields['det_F'][-1][fields['solid']]
    ax.hist(values, bins=60, histtype='step', linewidth=1.8, color=TEXT_1, linestyle=dashes[ratio],
            weights=np.full(values.size, 1.0 / values.size),
            label=f'{void_label(ratio)}: mean {values.mean():.3f}, min {values.min():.3f}, max {values.max():.3f}')
ax.axvline(1.0, color=AXIS, linewidth=1, linestyle=(0, (2, 2)))
ax.legend(frameon=False, fontsize=9, loc='upper right')

ax = axes[1, 1]
style(ax, '(d) CG iterations per load increment (totals in the legend)', 'load increment', 'CG iterations')
width = 0.8 / len(runs)
for index, ((p, ratio), (info, _)) in enumerate(runs.items()):
    y = info['cg_per_increment']
    ax.bar(np.arange(1, len(y) + 1) + (index - (len(runs) - 1) / 2) * width, y, width=width * 0.9,
           color=COLORS[p], alpha=1.0 if ratio == args.ratios[0] else 0.45,
           label=f'{LABELS[p]}, {void_label(ratio)}: {int(info["sum_CG_its"])}')
ax.legend(frameon=False, fontsize=9, loc='upper left', ncol=1)
fig.savefig(f'{args.out}_statistics.png', dpi=200, facecolor=SURFACE)

for (p, ratio), (info, fields) in runs.items():
    print(f'{LABELS[p]:13s} {void_label(ratio):32s} CG {int(info["sum_CG_its"]):5d}  Newton {int(info["sum_Newton_its"])}'
          f'  time {float(info["elapsed_time"]):5.1f} s  final P01 {info["mean_P"][-1, 0, 1]:.5f}'
          f'  P11 {info["mean_P"][-1, 1, 1]:.5f}  det F solid [{fields["det_F"][-1][fields["solid"]].min():.3f}, '
          f'{fields["det_F"][-1][fields["solid"]].max():.3f}]')
