"""Video of one small-strain AT2 run (example_2D_phase_field_fracture_AT2.py), in the layout of the slide videos
(examples/phase_field_fracture_adapted_grid/make_slide_video.py). One frame per load step:
    left          deformed cell at true scale, exact pixels: pores white, the solid coloured by the damage (light
                  intact -> dark cracked, fading to transparent from d = 0.6 to d = 0.95, so the opened crack is a clean gap)
    right, top    sigma_xx over E_xx: this run in colour (future part light grey), the other grids in grey
    right, bottom elastic CG iterations per elastic solve, mean over the staggered iterations of the load step
                  (from exp_data/..._iterations.csv; older runs without it: the last staggered iteration of the step):
                  this run in colour, the other preconditioners of the same grid in grey

    python make_AT2_video.py [N] [PRECONDITIONER] [TAG]    (default 256 dual)  -> figures/AT2_N<N>_<prec><TAG>.mp4, _final.png
    STRESS_ONLY=1 python make_AT2_video.py ...      slide version: without the CG panel (stress panel alone on the
                                                    right) and without the title and header line
    STRESS_ONLY=1 GRID_STUDY=1 python make_AT2_video.py ...   the same with the iteration count study below the stress
                                                    panel: mean elastic CG iterations per solve over the grid size,
                                                    all preconditioners (static) -> ..._study.mp4, _final.png
                                                    -> figures/AT2_N<N>_<prec><TAG>_stress.mp4, _final.png
    TAG e.g. _l0.03125: the series with a fixed physical length scale (32^2-512^2)
"""
import os
import re
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

import imageio_ffmpeg
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import netCDF4
import numpy as np
from matplotlib import animation
from matplotlib.colors import LinearSegmentedColormap, ListedColormap

from muFFTTO import microstructure_library

plt.rcParams['animation.ffmpeg_path'] = imageio_ffmpeg.get_ffmpeg_exe()
HERE = os.path.dirname(os.path.realpath(__file__))
N = int(sys.argv[1]) if len(sys.argv) > 1 else 256
PREC = sys.argv[2] if len(sys.argv) > 2 else 'dual'
TAG = sys.argv[3] if len(sys.argv) > 3 else ''
# reference grids: the finished runs of the series
GRIDS = [n for n in ((32, 64, 128, 256, 512) if TAG else (64, 128, 256))
         if os.path.exists(os.path.join(HERE, 'logs', f'AT2_N{n}_{PREC}{TAG}.log'))
         and 'Elapsed time' in open(os.path.join(HERE, 'logs', f'AT2_N{n}_{PREC}{TAG}.log')).read()]
DASHES = [(0, (5, 2)), (0, (1, 1.5)), (0, (3, 1, 1, 1)), (0, (8, 2, 1, 2)), (0, (12, 3))]
FPS = 10
STRESS_ONLY = os.environ.get('STRESS_ONLY', '0') == '1'
GRID_STUDY = STRESS_ONLY and os.environ.get('GRID_STUDY', '0') == '1'

SURFACE, TEXT_1, TEXT_2, GRID, AXIS, FUTURE = '#ffffff', '#0b0b0b', '#52514e', '#e6e5e1', '#a3a29c', '#d4d3cd'
REFERENCE_COLOR = '#b5b4ae'
DAMAGE_CMAP = LinearSegmentedColormap.from_list('damage_only', ['#e9e8e4', '#f4b183', '#d9622b', '#8c2a0c', '#2b0a02'])
FADE_START, FADE_END = 0.6, 0.95     # fully transparent from FADE_END: no streaks of stretched, almost cracked pixels
LABELS = {'Green': 'Green', 'Green_Jacobi': 'Green-Jacobi', 'dual': 'dual'}
COLORS = {'Green': '#1baf7a', 'Green_Jacobi': '#2a78d6', 'dual': '#d6402a'}   # as in the figures
RUN_COLOR = COLORS[PREC]


def damage_colours(damage):
    rgba = DAMAGE_CMAP(np.clip(damage, 0.0, 1.0))
    rgba[..., 3] = np.clip((FADE_END - damage) / (FADE_END - FADE_START), 0.0, 1.0)
    return rgba


class JoinedRun:
    """The frames of a run followed by those of its continuation (restart, <run>_restart.nc): d[name][k, ...]
    and d[name][:, ...] as for one netCDF4 dataset."""

    def __init__(self, first, second):
        self.parts = [first, second]
        # frames of the first part: the load steps before the first one of the continuation
        self.counts = [int(np.sum(np.asarray(first['step'][:]) < np.asarray(second['step'][:]).min())),
                       second['step'].shape[0]]
        self.microstructure = getattr(first, 'microstructure', 'circle_inclusion')

    def __getitem__(self, name):
        return _JoinedVariable([part[name] for part in self.parts], self.counts)


class _JoinedVariable:
    def __init__(self, variables, counts):
        self.variables, self.counts = variables, counts

    def __getitem__(self, key):
        frame, rest = (key[0], key[1:]) if isinstance(key, tuple) else (key, ())
        if isinstance(frame, slice):                    # all frames
            return np.concatenate([np.asarray(v[(slice(0, n),) + rest])
                                   for v, n in zip(self.variables, self.counts)])[frame]
        for v, n in zip(self.variables, self.counts):   # one frame
            if frame < n:
                return v[(frame,) + rest]
            frame -= n
        raise IndexError(key)


def read(n, p):
    """dataset, load, sigma_xx, elastic CG per solve (step mean) and staggered iterations per load step; a run
    continued after a restart (<run>_restart.*) is joined to its first part"""
    name = os.path.join(HERE, '{}', f'AT2_N{n}_{p}{TAG}')
    d = netCDF4.Dataset(name.format('exp_data') + '.nc')
    log = open(name.format('logs') + '.log').read()
    csv = name.format('exp_data') + '_iterations.csv'
    if os.path.exists(name.format('exp_data') + '_restart.nc'):
        d = JoinedRun(d, netCDF4.Dataset(name.format('exp_data') + '_restart.nc'))
        restart_log = open(name.format('logs') + '_restart.log').read()
        first_restart_step = int(re.search(r'^step +(\d+)', restart_log, re.M)[1])
        log = '\n'.join(m[0] for m in re.finditer(r'^step +(\d+) .*$', log, re.M) if int(m[1]) < first_restart_step) \
            + '\n' + restart_log
    nb_steps = len(re.findall(r'^step ', log, re.M))                  # finished load steps
    if os.path.exists(csv):
        it = np.genfromtxt(csv, delimiter=',', names=True, ndmin=1)
        if isinstance(d, JoinedRun):        # the continued load steps from the restart (the first part stopped in one)
            restart = np.genfromtxt(name.format('exp_data') + '_restart_iterations.csv', delimiter=',', names=True,
                                    ndmin=1)
            it = np.concatenate([it[it['step'] < restart['step'].min()], restart])
        cg = [it['cg_u'][it['step'] == k].mean() for k in range(nb_steps)]
        its = [np.sum(it['step'] == k) for k in range(nb_steps)]
    else:                                                             # older runs: the last staggered iteration
        cg = [int(m[1]) for m in re.finditer(r'CG its \(u, d\) = \((\d+), \d+\)', log)]
        its = [int(m[1]) for m in re.finditer(r'staggered its = +(\d+)', log)]
    return (d, np.asarray(d['load'][:]), np.asarray(d['macro_stress'][:, 0, 0]), np.array(cg, dtype=float),
            np.array(its))


def cg_per_solve(n, p):
    """mean elastic CG iterations per elastic solve of a finished run (joined with its restart), None if missing"""
    name = os.path.join(HERE, 'exp_data', f'AT2_N{n}_{p}{TAG}')
    log = os.path.join(HERE, 'logs', f'AT2_N{n}_{p}{TAG}')
    if not os.path.exists(name + '_iterations.csv'):
        return None
    restarted = os.path.exists(name + '_restart_iterations.csv')
    if 'Elapsed time' not in open(log + ('_restart' if restarted else '') + '.log').read():
        return None
    it = np.genfromtxt(name + '_iterations.csv', delimiter=',', names=True, ndmin=1)
    if restarted:
        r = np.genfromtxt(name + '_restart_iterations.csv', delimiter=',', names=True, ndmin=1)
        it = np.concatenate([it[it['step'] < r['step'].min()], r])
    return it['cg_u'].mean()


PER_SOLVE = os.path.exists(os.path.join(HERE, 'exp_data', f'AT2_N{N}_{PREC}{TAG}_iterations.csv'))
d, load, stress, cg, its = read(N, PREC)
nb_frames = len(load)
X = np.stack(np.meshgrid(np.arange(N + 1) / N, np.arange(N + 1) / N, indexing='ij'))
pore = microstructure_library.get_geometry(nb_voxels=(N, N), microstructure_name=getattr(d, 'microstructure', 'circle_inclusion'),
                                           coordinates=X[:, :-1, :-1]) == 0    # same call as the simulation


def corners(k):
    """deformed pixel corners at true scale: x = X + E X + u(X), u periodic"""
    u = np.pad(np.asarray(d['u_fluctuation'][k, :, 0]), ((0, 0), (0, 1), (0, 1)), mode='wrap')
    E = np.array([[load[k], 0.0], [0.0, 0.0]])
    return X + np.einsum('ij,jxy->ixy', E, X) + u


def pixel_damage(k):
    nodal = np.asarray(d['damage'][k, 0])
    return 0.25 * (nodal + np.roll(nodal, -1, 0) + np.roll(nodal, -1, 1) + np.roll(np.roll(nodal, -1, 0), -1, 1))


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
if STRESS_ONLY:        # the stress panel alone, centred on the right
    # the cell (square, colour bar on its left) as high as the frame allows: a wider left column
    gs = fig.add_gridspec(3, 2, width_ratios=[1.6, 1.0], height_ratios=[0.02, 1.0, 0.02], left=0.035, right=0.985,
                          top=0.98, bottom=0.02, wspace=0.1, hspace=0.0)
    ax_cell, ax_stress, ax_cg = fig.add_subplot(gs[:, 0]), fig.add_subplot(gs[1, 1]), None
else:
    gs = fig.add_gridspec(2, 2, width_ratios=[1.1, 1.0], left=0.03, right=0.97, top=0.86, bottom=0.08, wspace=0.16,
                          hspace=0.42)
    ax_cell, ax_stress, ax_cg = fig.add_subplot(gs[:, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, 1])
if not STRESS_ONLY:
    fig.text(0.03, 0.955, f'Small-strain AT2 phase-field fracture, {N}$^2$, elastic problem with the {LABELS[PREC]} '
                      'preconditioner' + (f', $\\ell$ = {re.search(r"_l([0-9.]+)", TAG)[1]}' if re.search(r"_l([0-9.]+)", TAG) else ''), fontsize=18, fontweight='bold', color=TEXT_1)
header = fig.text(0.03, 0.915, '', fontsize=12.5, color=TEXT_2, visible=not STRESS_ONLY)
all_corners = [corners(k) for k in range(nb_frames)]
ax_cell.set_aspect('equal')
ax_cell.set_xlim(min(c[0].min() for c in all_corners) - 0.02, max(c[0].max() for c in all_corners) + 0.02)
ax_cell.set_ylim(min(c[1].min() for c in all_corners) - 0.02, max(c[1].max() for c in all_corners) + 0.02)
ax_cell.axis('off')
levels = np.linspace(0.0, 1.0, 256)
faded = damage_colours(levels)
faded = faded[:, :3] * faded[:, 3:] + np.array(matplotlib.colors.to_rgb(SURFACE)) * (1.0 - faded[:, 3:])
colorbar = fig.colorbar(matplotlib.cm.ScalarMappable(norm=plt.Normalize(0, 1), cmap=ListedColormap(faded)),
                        ax=ax_cell, **(dict(location='left', fraction=0.03, pad=0.01, aspect=25, shrink=0.6)
                                       if STRESS_ONLY else dict(orientation='horizontal', fraction=0.04, pad=0.02,
                                                                aspect=40)))
colorbar.set_label('damage $d$' if STRESS_ONLY else 'damage $d$ (pore white, cracked pixels transparent; true scale)',
                   color=TEXT_2, fontsize=10.5)
colorbar.outline.set_visible(False)
colorbar.ax.tick_params(colors=TEXT_2, labelsize=9.5, length=0)
if STRESS_ONLY:        # vertical bar left of the cell: the quantity on top, as in the other slide videos
    colorbar.set_label('')
    colorbar.ax.set_title('damage $d$', fontsize=13, color=TEXT_2, pad=20)
    colorbar.ax.tick_params(labelsize=11)

style(ax_stress, 'Macroscopic stress', r'macroscopic strain $\bar\varepsilon_{xx}$', r'$\bar\sigma_{xx}$')
if not STRESS_ONLY:
    style(ax_cg, 'Elastic CG iterations per solve (mean over the load step)' if PER_SOLVE
          else 'Elastic CG iterations per load step (last staggered iteration)',
          r'macroscopic strain $\bar\varepsilon_{xx}$', 'CG iterations')
reference_grids = [g for g in GRIDS if g != N]
assert len(reference_grids) <= len(DASHES), 'more reference grids than dash patterns'
# reference grids light (coarse) to dark (fine)
for n, dash, grey in zip(reference_grids, DASHES, np.linspace(0.78, 0.45, max(len(reference_grids), 2))):
    _, l_ref, s_ref, _, _ = read(n, PREC)
    ax_stress.plot(np.r_[0, l_ref], np.r_[0, s_ref], color=str(grey), lw=1.8, ls=dash, label=f'{n}$^2$')
for p, dash in zip([] if STRESS_ONLY else [q for q in LABELS if q != PREC                     # the other preconditioners that ran on this grid
                    and os.path.exists(os.path.join(HERE, 'exp_data', f'AT2_N{N}_{q}{TAG}.nc'))],
                   [(0, (5, 2)), (0, (1, 1.5))]):
    _, l_ref, _, cg_ref, _ = read(N, p)
    ax_cg.plot(l_ref[:len(cg_ref)], cg_ref, color=REFERENCE_COLOR, lw=1.3, ls=dash,
               label=LABELS[p])
ax_stress.plot(np.r_[0, load], np.r_[0, stress], color=FUTURE, lw=2)
stress_line, = ax_stress.plot([], [], color=RUN_COLOR, lw=2.2, label=f'{N}$^2$, this run')
stress_dot, = ax_stress.plot([], [], 'o', color=RUN_COLOR, markersize=8, markeredgecolor=SURFACE, markeredgewidth=1.5)
if not STRESS_ONLY:
    ax_cg.plot(load[:len(cg)], cg, color=FUTURE, lw=1.6)
    cg_line, = ax_cg.plot([], [], color=RUN_COLOR, lw=1.8, label=f'{LABELS[PREC]} (this run)')
    ax_cg.set_ylim(0, 1.3 * max(line.get_ydata().max() for line in ax_cg.get_lines() if len(line.get_ydata())))   # room for the legend
ax_stress.set_ylim(0, 1.08 * max(stress.max(), *[read(n, PREC)[2].max() for n in GRIDS]))
for ax in (ax_stress,) if STRESS_ONLY else (ax_stress, ax_cg):
    ax.set_xlim(0, load[-1])
ax_stress.legend(frameon=False, fontsize=10, loc='upper left')
if STRESS_ONLY:        # the stress panel as high as the drawn cell
    fig.canvas.draw()
    # figure coordinates of the cell bottom and top (y = 0, 1), not of its padded axes box
    y0, y1 = fig.transFigure.inverted().transform(ax_cell.transData.transform([[0.0, 0.0], [0.0, 1.0]]))[:, 1]
    box = ax_stress.get_position()
    if GRID_STUDY:     # stress above, iteration count study below, together as high as the cell
        gap = 0.12         # x labels of the stress panel and the title of the study panel
        height = (y1 - y0 - gap) / 2
        ax_stress.set_position([box.x0, y1 - height, box.width, height])
        ax_study = fig.add_axes([box.x0, y0, box.width, height])
        style(ax_study, 'Elastic CG iterations per solve', r'grid $N$ ($N^2$ pixels)', 'CG iterations')
        for p in LABELS:
            points = [(n, cg_per_solve(n, p)) for n in (32, 64, 128, 256, 512, 1024)]
            points = [(n, v) for n, v in points if v is not None]
            if not points:
                continue
            x, y = np.array(points).T
            ax_study.plot(x, y, color=COLORS[p], lw=2.2, marker='o', ms=7, label=LABELS[p])
            ax_study.annotate(f'{y[-1]:.0f}', (x[-1], y[-1]), textcoords='offset points',   # Green above, GJ below
                              xytext=(9, {'Green': 6, 'Green_Jacobi': -6}.get(p, 0)),
                              va='center', fontsize=10, color=TEXT_2)
        ax_study.set_xscale('log', base=2)
        ax_study.set_xticks([32, 64, 128, 256, 512, 1024], [f'${n}^2$' for n in (32, 64, 128, 256, 512, 1024)])
        ax_study.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
        ax_study.set_xlim(32 / 1.15, 1024 * 1.3)
        ax_study.set_ylim(0, 1.15 * ax_study.get_ylim()[1])
        ax_study.legend(frameon=False, fontsize=10, loc='upper left')
else:
    ax_cg.legend(frameon=False, fontsize=10, loc='upper left', ncol=2)


def draw(k):
    for mesh in list(ax_cell.collections):
        mesh.remove()
    x, y = all_corners[k]
    damage = pixel_damage(k)
    colours = damage_colours(damage)
    colours[pore, 3] = 0.0
    mesh = ax_cell.pcolormesh(x, y, damage, shading='flat', rasterized=True, edgecolors='none', linewidth=0,
                              antialiased=False)
    mesh.set_array(None)
    mesh.set_facecolor(colours.reshape(-1, 4))
    stress_line.set_data(np.r_[0, load[:k + 1]], np.r_[0, stress[:k + 1]])
    stress_dot.set_data(load[k:k + 1], stress[k:k + 1])
    if not STRESS_ONLY:
        cg_line.set_data(load[:min(k + 1, len(cg))], cg[:k + 1])
    header.set_text(rf'$\bar\varepsilon_{{xx}}$ = {load[k]:.3f}    load step {k + 1} / {nb_frames}    '
                    rf'$\bar\sigma_{{xx}}$ = {stress[k]:.2e}    max damage {damage[~pore].max():.2f}    '
                    f'staggered its {its[k] if k < len(its) else 0}'
                    + ('' if STRESS_ONLY else f'    elastic CG per solve {cg[k] if k < len(cg) else 0:.0f}'))


os.makedirs(os.path.join(HERE, 'figures'), exist_ok=True)
out = os.path.join(HERE, 'figures', f'AT2_N{N}_{PREC}{TAG}' + ('_study' if GRID_STUDY else '_stress' if STRESS_ONLY else ''))
frames = list(range(nb_frames)) + [nb_frames - 1] * (2 * FPS)
writer = animation.FFMpegWriter(fps=FPS, bitrate=6000, codec='libx264', extra_args=['-pix_fmt', 'yuv420p'])
with writer.saving(fig, out + '.mp4', dpi=100):
    for k in frames:
        draw(k)
        writer.grab_frame(facecolor=SURFACE)
draw(nb_frames - 1)
fig.savefig(out + '_final.png', dpi=130, facecolor=SURFACE)
print(out + '.mp4')
