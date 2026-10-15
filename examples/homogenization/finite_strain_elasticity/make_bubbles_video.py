"""Video of a finite-strain bubble-array homogenization run: deformation, stress-strain curve, CG per increment.

Reads exp_data/example_2D_homogenization_finite_strain_bubbles_zero_stiffness/Nx=<n>Ny=<n>_<preconditioner>
[_void<r>][_load...][_inc...]/ (info_log_final.npz and fields.npz, written with --save_fields), for Green and
Green-Jacobi and the given void stiffnesses.

One frame per load increment:
    left          deformed cell of the first void stiffness (Green run), exact pixels coloured by det F, bubbles white
    right, top    macroscopic stress of the loaded components over the load fraction (all void stiffnesses)
    right, bottom CG iterations per load increment, Green and Green-Jacobi

usage: python make_bubbles_video.py OUT_PREFIX --suffix _load0_0_0_-0.4_inc100 [--n 64] [--ratios 0 1e-05]
writes OUT_PREFIX.mp4 and OUT_PREFIX_final.png
"""
import argparse
import os

import imageio_ffmpeg
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

plt.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()

# reference palette, light surface; colour follows the preconditioner, line style the void stiffness
SURFACE, TEXT_1, TEXT_2, GRID, AXIS, FUTURE = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1", "#a3a29c", "#d4d3cd"
COLORS = {"Green": "#2a78d6", "Green_Jacobi": "#eb6834"}
LABELS = {"Green": "Green", "Green_Jacobi": "Green-Jacobi"}
STRESS_COLORS = ["#0b0b0b", "#8f8e89"]
DET_F_CMAP = LinearSegmentedColormap.from_list("det_F", ["#0d366b", "#3987e5", "#d9d8d4", "#f0a070", "#a8401a"])

parser = argparse.ArgumentParser()
parser.add_argument('out')
parser.add_argument('--suffix', default='', help="folder suffix of the run, e.g. '_load0_0_0_-0.4_inc100'")
parser.add_argument('--n', type=int, default=64)
parser.add_argument('--ratios', type=float, nargs='+', default=[0.0, 1e-5])
parser.add_argument('--fps', type=int, default=8)
args = parser.parse_args()

root = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'exp_data',
                    'example_2D_homogenization_finite_strain_bubbles_zero_stiffness')


def folder(preconditioner, ratio):
    return os.path.join(root, f'Nx={args.n}Ny={args.n}_{preconditioner}'
                        + (f'_void{ratio:g}' if ratio > 0 else '') + args.suffix)


def void_label(ratio):
    return 'void 0' if ratio == 0 else f'void {ratio:g}'


runs = {}
for p in ('Green', 'Green_Jacobi'):
    for r in args.ratios:
        info = dict(np.load(os.path.join(folder(p, r), 'info_log_final.npz'), allow_pickle=True))
        fields = dict(np.load(os.path.join(folder(p, r), 'fields.npz')))
        runs[(p, r)] = (info, fields)

shown_info, shown = runs[('Green', args.ratios[0])]
nb_increments = int(shown_info['ninc'])
nb_frames = shown['det_F'].shape[0]
total_load = np.asarray(shown_info['macro_gradient_total'])
loaded = [tuple(index) for index in np.argwhere(np.abs(total_load) > 0)]
mu_solid = float(shown_info['mu_solid'])


def deformed_corners(displacement, macro_gradient):
    """Deformed positions of the pixel corners, (2, n+1, n+1): x = X + H X + u(X), u periodic."""
    n = displacement.shape[-1]
    X = np.stack(np.meshgrid(np.arange(n + 1) / n, np.arange(n + 1) / n, indexing='ij'))
    u = np.pad(displacement, ((0, 0), (0, 1), (0, 1)), mode='wrap')
    return X + np.einsum('ij,jxy->ixy', macro_gradient, X) + u


def style(ax, title, xlabel, ylabel):
    ax.set_facecolor(SURFACE)
    ax.set_title(title, loc='left', fontsize=12, color=TEXT_1, pad=6)
    ax.set_xlabel(xlabel, color=TEXT_2, fontsize=10.5)
    ax.set_ylabel(ylabel, color=TEXT_2, fontsize=10.5)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    for side in ('left', 'bottom'):
        ax.spines[side].set_color(AXIS)
    ax.tick_params(colors=TEXT_2, labelsize=10)
    ax.grid(axis='y', color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


# fixed view and colour range over all frames
all_corners = [deformed_corners(shown['displacement'][k], shown['macro_gradient'][k]) for k in range(nb_frames)]
x_min = min(c[0].min() for c in all_corners)
x_max = max(c[0].max() for c in all_corners)
y_min = min(c[1].min() for c in all_corners)
y_max = max(c[1].max() for c in all_corners)
solid_det_F = shown['det_F'][:, shown['solid']]
spread = max(abs(solid_det_F.max() - 1), abs(1 - solid_det_F.min()), 1e-3)
norm = TwoSlopeNorm(vmin=1 - spread, vcenter=1.0, vmax=1 + spread)

fig = plt.figure(figsize=(16, 9), dpi=100, facecolor=SURFACE)
grid = fig.add_gridspec(2, 2, width_ratios=[1.15, 1.0], left=0.03, right=0.97, top=0.86, bottom=0.08,
                        wspace=0.18, hspace=0.42)
ax_cell = fig.add_subplot(grid[:, 0])
ax_stress = fig.add_subplot(grid[0, 1])
ax_cg = fig.add_subplot(grid[1, 1])
load_text = ', '.join(f'F{i}{j} - I{i}{j} = {total_load[i, j]:+g}' for i, j in loaded)
fig.text(0.03, 0.955, f'Bubble array, finite strain, {args.n}² pixels: {load_text} in {nb_increments} increments',
         fontsize=16, fontweight='bold', color=TEXT_1)
header = fig.text(0.03, 0.915, '', fontsize=12, color=TEXT_2)

ax_cell.set_aspect('equal')
ax_cell.set_xlim(x_min - 0.02, x_max + 0.02)
ax_cell.set_ylim(y_min - 0.02, y_max + 0.02)
ax_cell.axis('off')
colorbar = fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=DET_F_CMAP), ax=ax_cell, orientation='horizontal',
                        fraction=0.04, pad=0.02, aspect=40)
colorbar.set_label(f'det F in the solid ({void_label(args.ratios[0])}, Green); bubbles white; exact pixels',
                   color=TEXT_2, fontsize=10.5)
colorbar.outline.set_visible(False)
colorbar.ax.tick_params(colors=TEXT_2, labelsize=9.5, length=0)

style(ax_stress, 'Macroscopic stress (Green)', 'load fraction', 'mean P / mu_solid')
style(ax_cg, 'CG iterations per load increment', 'load increment', 'CG iterations')
dashes = {ratio: d for ratio, d in zip(args.ratios, ('-', (0, (4, 2)), (0, (1, 1.5))))}
load_fraction = np.arange(1, nb_increments + 1) / nb_increments
stress_lines = []
for ratio in args.ratios:
    info, _ = runs[('Green', ratio)]
    for (i, j), color in zip(loaded, STRESS_COLORS):
        y = info['mean_P'][:, i, j] / mu_solid
        x = load_fraction[:len(y)]
        ax_stress.plot(x, y, color=FUTURE, linestyle=dashes[ratio], linewidth=2)
        done, = ax_stress.plot([], [], color=color, linestyle=dashes[ratio], linewidth=2,
                               label=f'P{i}{j}, {void_label(ratio)}')
        dot, = ax_stress.plot([], [], 'o', color=color, markersize=7, markeredgecolor=SURFACE, markeredgewidth=1.5)
        stress_lines.append((x, y, done, dot))
ax_stress.legend(frameon=False, fontsize=9.5, loc='upper left' if np.sum(total_load) > 0 else 'lower left')

cg_lines = []
for (p, ratio), (info, _) in runs.items():
    y = np.asarray(info['cg_per_increment'])
    x = np.arange(1, len(y) + 1)
    ax_cg.plot(x, y, color=FUTURE, linestyle=dashes[ratio], linewidth=1.8)
    done, = ax_cg.plot([], [], color=COLORS[p], linestyle=dashes[ratio], linewidth=2,
                       label=f'{LABELS[p]}, {void_label(ratio)} (total {int(info["sum_CG_its"])})')
    cg_lines.append((x, y, done))
ax_cg.set_xlim(0, nb_increments + 1)
cg_max = max(y.max() for _, y, _ in cg_lines)
cg_min = max(min(y.min() for _, y, _ in cg_lines), 1)
if cg_max / cg_min > 20:                       # widely different costs: log scale, room above for the legend
    ax_cg.set_yscale('log')
    ax_cg.set_ylim(0.7 * cg_min, cg_max * (cg_max / cg_min) ** 0.45)
else:
    ax_cg.set_ylim(0, 1.45 * cg_max)
ax_cg.legend(frameon=False, fontsize=9.5, loc='upper center', ncol=2)
for ax in (ax_stress,):
    ax.set_xlim(0, 1.02)


def draw(k):
    for mesh in list(ax_cell.collections):
        mesh.remove()
    corners = all_corners[k]
    det_F = np.ma.masked_where(~shown['solid'], shown['det_F'][k])
    ax_cell.pcolormesh(corners[0], corners[1], det_F, shading='flat', cmap=DET_F_CMAP, norm=norm, rasterized=True)
    for x, y, done, dot in stress_lines:
        m = min(k + 1, len(y))
        done.set_data(x[:m], y[:m])
        dot.set_data(x[m - 1:m], y[m - 1:m])
    for x, y, done in cg_lines:
        m = min(k + 1, len(y))
        done.set_data(x[:m], y[:m])
    det_F_solid = shown['det_F'][k][shown['solid']]
    header.set_text(f'increment {k + 1:3d} / {nb_increments}   load {100 * (k + 1) / nb_increments:5.1f} %   '
                    f'det F in the solid {det_F_solid.min():.3f} ... {det_F_solid.max():.3f}   '
                    f'Green CG this increment {int(shown_info["cg_per_increment"][k])}')


frames = list(range(nb_frames)) + [nb_frames - 1] * (2 * args.fps)      # hold the final state for 2 s
writer = animation.FFMpegWriter(fps=args.fps, bitrate=6000, codec='libx264', extra_args=['-pix_fmt', 'yuv420p'])
with writer.saving(fig, args.out + '.mp4', dpi=100):
    for k in frames:
        draw(k)
        writer.grab_frame(facecolor=SURFACE)
draw(nb_frames - 1)
fig.savefig(args.out + '_final.png', dpi=150, facecolor=SURFACE)
print(f'wrote {args.out}.mp4 ({len(frames)} frames) and {args.out}_final.png')
