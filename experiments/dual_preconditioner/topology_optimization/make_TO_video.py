"""Video of one topology optimization run (example_2D_elasticity_TO.py, CG_MODE=energy, warm start), in the layout of
the slide videos (as damage_AT2_small_strain/make_AT2_video.py). One frame per optimizer iteration, from
data/example_2D_elasticity_TO/N<N>_soft<s>_energy-warm_<prec>_..._history.nc and the log:
    left          phase field on the exact pixels (white void -> dark solid), 2 x 2 periodic cells up to 256^2, one cell on finer
                  grids (2 x 2 would need more video pixels than the frame has)
    right, top    objective over the iterations: this run in colour (future part light grey)
    right, bottom CG iterations per objective evaluation (forward + adjoint, 3 load cases): this run in colour,
                  the other preconditioner of the same grid and soft phase in grey

    python make_TO_video.py [N] [SOFT_EXP] [PRECONDITIONER]     (default 256 3 dual)
        -> figures/TO_N<N>_soft1e-<e>_<prec>.mp4 and _final.png
"""
import glob
import os
import re
import sys

import imageio_ffmpeg
import matplotlib
import matplotlib.ticker

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import netCDF4
import numpy as np
from matplotlib import animation

plt.rcParams['animation.ffmpeg_path'] = imageio_ffmpeg.get_ffmpeg_exe()
HERE = os.path.dirname(os.path.realpath(__file__))
N = int(sys.argv[1]) if len(sys.argv) > 1 else 256
SOFT_EXP = int(sys.argv[2]) if len(sys.argv) > 2 else 3
PREC = sys.argv[3] if len(sys.argv) > 3 else 'dual'
FPS = 10

SURFACE, TEXT_1, TEXT_2, GRID, AXIS, FUTURE = '#fcfcfb', '#0b0b0b', '#52514e', '#e6e5e1', '#a3a29c', '#d4d3cd'
RUN_COLOR, REFERENCE_COLOR = '#2a78d6', '#b5b4ae'
LABELS = {'Green': 'Green', 'Green_Jacobi': 'Green-Jacobi', 'dual': 'dual'}
SOFT = {3: '0.001', 5: '1e-05'}[SOFT_EXP]


def history(p):
    f = glob.glob(os.path.join(HERE, 'data', 'example_2D_elasticity_TO',
                               f'N{N}_soft{SOFT}_energy-warm_{p}_eta_0.03125_*_history.nc'))
    return netCDF4.Dataset(f[0]) if f else None


def cg_per_evaluation(p):
    """forward + adjoint CG iterations of the 3 load cases, per objective evaluation"""
    log = os.path.join(HERE, 'logs', f'TO_N{N}_soft1e-{SOFT_EXP}_eta1-32_warm_{p}.log')
    if not os.path.exists(log):
        return None
    text = open(log).read()
    m = [int(x) for x in re.findall(r'nb_ steps CG mech +=(\d+)', text)]
    a = [int(x) for x in re.findall(r'nb_steps CG adjoint +=(\d+)', text)]
    n = min(len(m), len(a)) // 3 * 3
    return (np.array(m[:n]) + np.array(a[:n])).reshape(-1, 3).sum(1)


d = history(PREC)
phase = d['phase_field_iterate']
objective = np.asarray(d['objective'][:])
evaluations = np.asarray(d['nb_evaluations'][:]).astype(int)
nb_frames = len(objective)
cg = cg_per_evaluation(PREC)


def style(ax, title, xlabel, ylabel):
    ax.set_facecolor(SURFACE)
    ax.set_title(title, loc='left', fontsize=12.5, color=TEXT_1, pad=6)
    ax.set_xlabel(xlabel, color=TEXT_2, fontsize=11)
    ax.set_ylabel(ylabel, color=TEXT_2, fontsize=11)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    for side in ('left', 'bottom'):
        ax.spines[side].set_color(AXIS)
    ax.tick_params(colors=TEXT_2, labelsize=10)
    ax.grid(axis='y', color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


fig = plt.figure(figsize=(16, 9), dpi=100, facecolor=SURFACE)
gs = fig.add_gridspec(2, 2, width_ratios=[1.1, 1.0], left=0.03, right=0.97, top=0.86, bottom=0.08, wspace=0.16,
                      hspace=0.42)
ax_cell, ax_obj, ax_cg = fig.add_subplot(gs[:, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, 1])
fig.text(0.03, 0.955, f'Topology optimization of an auxetic cell, {N}$^2$, soft phase $10^{{-{SOFT_EXP}}}$, '
                      f'{LABELS[PREC]} preconditioner', fontsize=18, fontweight='bold', color=TEXT_1)
header = fig.text(0.03, 0.915, '', fontsize=12.5, color=TEXT_2)
# exact pixels: nearest neighbour; CELLS x CELLS periodic copies of the unit cell
CELLS = 2 if N <= 256 else 1
image = ax_cell.imshow(np.tile(np.asarray(phase[0, 0]).T, (CELLS, CELLS)), origin='lower', cmap='Greys', vmin=0,
                       vmax=1, interpolation='nearest', extent=(0, CELLS, 0, CELLS))
if CELLS == 2:
    ax_cell.plot([1, 1], [0, 2], color=RUN_COLOR, lw=0.8, alpha=0.5)
    ax_cell.plot([0, 2], [1, 1], color=RUN_COLOR, lw=0.8, alpha=0.5)
ax_cell.axis('off')
colorbar = fig.colorbar(image, ax=ax_cell, orientation='horizontal', fraction=0.04, pad=0.02, aspect=40, shrink=0.85)
colorbar.set_label(r'phase field $\rho$ (0 soft phase, 1 solid' + (r'; $2\times2$ periodic cells)' if CELLS == 2 else ', one cell)'), color=TEXT_2,
                   fontsize=10.5)
colorbar.outline.set_visible(False)
colorbar.ax.tick_params(colors=TEXT_2, labelsize=9.5, length=0)

style(ax_obj, 'Objective', 'optimizer iteration', 'objective (log)')
style(ax_cg, 'CG iterations per objective evaluation (forward + adjoint, 3 load cases)', 'objective evaluation',
      'CG iterations')
ax_obj.semilogy(np.arange(1, nb_frames + 1), objective, color=FUTURE, lw=2)
obj_line, = ax_obj.semilogy([], [], color=RUN_COLOR, lw=2.2)
obj_dot, = ax_obj.semilogy([], [], 'o', color=RUN_COLOR, markersize=8, markeredgecolor=SURFACE, markeredgewidth=1.5)
ax_obj.set_xlim(0, nb_frames + 1)
ax_obj.yaxis.set_major_locator(matplotlib.ticker.LogLocator(base=10, subs=(1, 2, 5)))
ax_obj.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%g'))
ax_obj.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
for p in [q for q in ('Green', 'Green_Jacobi', 'dual') if q != PREC]:
    c = cg_per_evaluation(p)
    if c is not None and len(c):
        ax_cg.plot(np.arange(1, len(c) + 1), c, color=REFERENCE_COLOR, lw=1.3, ls=(0, (5, 2)), label=LABELS[p])
ax_cg.plot(np.arange(1, len(cg) + 1), cg, color=FUTURE, lw=1.6)
cg_line, = ax_cg.plot([], [], color=RUN_COLOR, lw=1.8, label=f'{LABELS[PREC]} (this run)')
ax_cg.set_xlim(0, max(len(cg), evaluations[-1]) + 1)
ax_cg.set_ylim(0, 1.1 * max(np.percentile(cg, 99), *[np.percentile(l.get_ydata(), 99) for l in ax_cg.lines[:-2]]))
ax_cg.legend(frameon=False, fontsize=10, loc='upper right')


def draw(k):
    image.set_data(np.tile(np.asarray(phase[k, 0]).T, (CELLS, CELLS)))
    obj_line.set_data(np.arange(1, k + 2), objective[:k + 1])
    obj_dot.set_data([k + 1], objective[k:k + 1])
    n_eval = min(evaluations[k], len(cg))
    cg_line.set_data(np.arange(1, n_eval + 1), cg[:n_eval])
    header.set_text(f'iteration {k + 1} / {nb_frames}    objective {objective[k]:.4f}    '
                    f'volume fraction {float(d["volume_fraction"][k]):.3f}    evaluations {evaluations[k]}    '
                    f'mean CG per evaluation {cg[:n_eval].mean():.0f}')


os.makedirs(os.path.join(HERE, 'figures'), exist_ok=True)
out = os.path.join(HERE, 'figures', f'TO_N{N}_soft1e-{SOFT_EXP}_{PREC}')
frames = list(range(nb_frames)) + [nb_frames - 1] * (2 * FPS)
writer = animation.FFMpegWriter(fps=FPS, bitrate=6000, codec='libx264', extra_args=['-pix_fmt', 'yuv420p'])
with writer.saving(fig, out + '.mp4', dpi=100):
    for k in frames:
        draw(k)
        writer.grab_frame(facecolor=SURFACE)
draw(nb_frames - 1)
fig.savefig(out + '_final.png', dpi=130, facecolor=SURFACE)
print(out + '.mp4')
