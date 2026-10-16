"""Video of one small-strain AT2 run (example_2D_phase_field_fracture_AT2.py), in the layout of the slide videos
(examples/phase_field_fracture_adapted_grid/make_slide_video.py). One frame per load step:
    left          deformed cell at true scale, exact pixels: pores white, the solid coloured by the damage (light
                  intact -> dark cracked, fading to transparent from d = 0.6 to d = 1, so the opened crack is a gap)
    right, top    sigma_xx over E_xx: this run in colour (future part light grey), the other grids in grey
    right, bottom elastic CG iterations per load step (last staggered iteration): this run in colour, the other
                  preconditioners of the same grid in grey

    python make_AT2_video.py [N] [PRECONDITIONER]          (default 256 dual)  -> figures/AT2_N<N>_<prec>.mp4, _final.png
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
FPS = 10

SURFACE, TEXT_1, TEXT_2, GRID, AXIS, FUTURE = '#fcfcfb', '#0b0b0b', '#52514e', '#e6e5e1', '#a3a29c', '#d4d3cd'
RUN_COLOR, REFERENCE_COLOR = '#2a78d6', '#b5b4ae'
DAMAGE_CMAP = LinearSegmentedColormap.from_list('damage_only', ['#e9e8e4', '#f4b183', '#d9622b', '#8c2a0c', '#2b0a02'])
FADE_START = 0.6
LABELS = {'Green': 'Green', 'Green_Jacobi': 'Green-Jacobi', 'dual': 'dual'}


def damage_colours(damage):
    rgba = DAMAGE_CMAP(np.clip(damage, 0.0, 1.0))
    rgba[..., 3] = np.clip((1.0 - damage) / (1.0 - FADE_START), 0.0, 1.0)
    return rgba


def read(n, p):
    d = netCDF4.Dataset(os.path.join(HERE, 'exp_data', f'AT2_N{n}_{p}.nc'))
    cg = [int(m[1]) for m in re.finditer(r'CG its \(u, d\) = \((\d+), \d+\)',
                                         open(os.path.join(HERE, 'logs', f'AT2_N{n}_{p}.log')).read())]
    return d, np.asarray(d['load'][:]), np.asarray(d['macro_stress'][:, 0, 0]), np.array(cg, dtype=float)


d, load, stress, cg = read(N, PREC)
nb_frames = len(load)
X = np.stack(np.meshgrid(np.arange(N + 1) / N, np.arange(N + 1) / N, indexing='ij'))
pore = microstructure_library.get_geometry(nb_voxels=(N, N), microstructure_name='circle_inclusion',
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
gs = fig.add_gridspec(2, 2, width_ratios=[1.1, 1.0], left=0.03, right=0.97, top=0.86, bottom=0.08, wspace=0.16,
                      hspace=0.42)
ax_cell, ax_stress, ax_cg = fig.add_subplot(gs[:, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, 1])
fig.text(0.03, 0.955, f'Small-strain AT2 phase-field fracture, {N}$^2$, elastic problem with the {LABELS[PREC]} '
                      'preconditioner', fontsize=18, fontweight='bold', color=TEXT_1)
header = fig.text(0.03, 0.915, '', fontsize=12.5, color=TEXT_2)
all_corners = [corners(k) for k in range(nb_frames)]
ax_cell.set_aspect('equal')
ax_cell.set_xlim(min(c[0].min() for c in all_corners) - 0.02, max(c[0].max() for c in all_corners) + 0.02)
ax_cell.set_ylim(min(c[1].min() for c in all_corners) - 0.02, max(c[1].max() for c in all_corners) + 0.02)
ax_cell.axis('off')
levels = np.linspace(0.0, 1.0, 256)
faded = damage_colours(levels)
faded = faded[:, :3] * faded[:, 3:] + np.array(matplotlib.colors.to_rgb(SURFACE)) * (1.0 - faded[:, 3:])
colorbar = fig.colorbar(matplotlib.cm.ScalarMappable(norm=plt.Normalize(0, 1), cmap=ListedColormap(faded)),
                        ax=ax_cell, orientation='horizontal', fraction=0.04, pad=0.02, aspect=40)
colorbar.set_label(rf'damage $d$ (pore white, cracked pixels transparent; true scale)', color=TEXT_2, fontsize=10.5)
colorbar.outline.set_visible(False)
colorbar.ax.tick_params(colors=TEXT_2, labelsize=9.5, length=0)

style(ax_stress, 'Macroscopic stress', r'macroscopic strain $\bar\varepsilon_{xx}$', r'$\bar\sigma_{xx}$')
style(ax_cg, 'Elastic CG iterations per load step (last staggered iteration)', r'macroscopic strain $\bar\varepsilon_{xx}$', 'CG iterations (log)')
for n, dash in zip([g for g in (64, 128, 256) if g != N], [(0, (5, 2)), (0, (1, 1.5))]):
    _, l_ref, s_ref, _ = read(n, PREC)
    ax_stress.plot(np.r_[0, l_ref], np.r_[0, s_ref], color=REFERENCE_COLOR, lw=1.8, ls=dash, label=f'{n}$^2$')
for p, dash in zip([q for q in LABELS if q != PREC], [(0, (5, 2)), (0, (1, 1.5))]):
    _, l_ref, _, cg_ref = read(N, p)
    ax_cg.plot(l_ref[:len(cg_ref)], cg_ref, color=REFERENCE_COLOR, lw=1.3, ls=dash,
               label=LABELS[p])
ax_stress.plot(np.r_[0, load], np.r_[0, stress], color=FUTURE, lw=2)
stress_line, = ax_stress.plot([], [], color=RUN_COLOR, lw=2.2, label=f'{N}$^2$, this run')
stress_dot, = ax_stress.plot([], [], 'o', color=RUN_COLOR, markersize=8, markeredgecolor=SURFACE, markeredgewidth=1.5)
ax_cg.plot(load[:len(cg)], cg, color=FUTURE, lw=1.6)
cg_line, = ax_cg.plot([], [], color=RUN_COLOR, lw=1.8, label=f'{LABELS[PREC]} (this run)')
ax_cg.set_yscale('log')
ax_stress.set_ylim(0, 1.08 * max(stress.max(), *[read(n, PREC)[2].max() for n in (64, 128, 256)]))
for ax in (ax_stress, ax_cg):
    ax.set_xlim(0, load[-1])
ax_stress.legend(frameon=False, fontsize=10, loc='upper left')
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
    cg_line.set_data(load[:min(k + 1, len(cg))], cg[:k + 1])
    header.set_text(rf'$\bar\varepsilon_{{xx}}$ = {load[k]:.3f}    load step {k + 1} / {nb_frames}    '
                    rf'$\bar\sigma_{{xx}}$ = {stress[k]:.2e}    max damage {damage[~pore].max():.2f}    '
                    f'elastic CG this step {int(cg[k]) if k < len(cg) else 0}')


os.makedirs(os.path.join(HERE, 'figures'), exist_ok=True)
out = os.path.join(HERE, 'figures', f'AT2_N{N}_{PREC}')
frames = list(range(nb_frames)) + [nb_frames - 1] * (2 * FPS)
writer = animation.FFMpegWriter(fps=FPS, bitrate=6000, codec='libx264', extra_args=['-pix_fmt', 'yuv420p'])
with writer.saving(fig, out + '.mp4', dpi=100):
    for k in frames:
        draw(k)
        writer.grab_frame(facecolor=SURFACE)
draw(nb_frames - 1)
fig.savefig(out + '_final.png', dpi=130, facecolor=SURFACE)
print(out + '.mp4')
