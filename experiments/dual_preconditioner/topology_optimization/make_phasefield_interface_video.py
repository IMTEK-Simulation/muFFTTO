"""Slide video (slide-13 style) of phasefield_interface_study.py: the 1024^2 TO design with a phase-field interface of
growing width eta. One frame per eta that has results in exp_data/phasefield_interface_N<N>.csv (rerun when more exist).
    left    density s on the exact pixels (grey, log scale, 1e-4 ... 1), colour bar on its left
    middle  s along the row through the middle of the widest void (sharp geometry grey, current red), linear scale
    right   CG iterations vs eta/h: Green, Green-Jacobi, dual (x2, cost) up to the current eta
    python make_phasefield_interface_video.py  ->  figures/phasefield_interface_N1024.mp4, _initial.png, _final.png
"""
import csv
import os

import imageio_ffmpeg
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation

import phasefield_interface_study as study      # geometry (signed distance, phi_of); its solver runs only as __main__

plt.rcParams['animation.ffmpeg_path'] = imageio_ffmpeg.get_ffmpeg_exe()
HERE = os.path.dirname(os.path.realpath(__file__))
N, k, p = study.N, study.k, study.p
rows = list(csv.DictReader(open(os.path.join(HERE, 'exp_data', f'phasefield_interface_N{N}.csv'))))
PRECS = [('Green', 'Green', '#1a8a1a', 1), ('Green_Jacobi', 'Green-Jacobi', '#2a78d6', 1),
         ('dual', r'dual ($\times 2$)', '#d62a2a', 2)]
its = {P: {} for P, *_ in PRECS}
for r in rows:
    its[r['precond']][float(r['eta_over_h'])] = int(r['cg'])
etas = sorted(e for e in its['Green'] if all(e in its[P] for P, *_ in PRECS))
print('eta/h with all preconditioners:', etas)


def density(eta):
    """s on the pixels (phi at the nodes, as the solver's quadrature interpolation of a nodal field ~ nodal value)"""
    phi = study.phi_of(eta)
    return np.where(phi > 0.5, 1.0, k) if eta == 0 else phi ** p * (1 - k) + k


ROW = int(os.environ.get('ROW', np.argmin(study.signed.min(axis=0))))   # row (y index); default: through the deepest void point
x = np.arange(N)
_edges = np.flatnonzero(np.diff(density(0)[:, ROW]) != 0)            # void edges along the row
_slope = np.abs(np.gradient(study.signed[:, ROW]))                      # |d signed / dx| = 1: the row cuts the edge head-on
WIN = 48 if N >= 1024 else 24                   # profile window half-width [pixels]
_sharp = density(0)[:, ROW] > 0.5


def _room(e):
    """pixels of the same phase on both sides of edge e (periodic), capped at WIN"""
    left = right = 0
    while left < WIN and _sharp[(e - left) % N] == _sharp[e]:
        left += 1
    while right < WIN and _sharp[(e + 1 + right) % N] == _sharp[(e + 1) % N]:
        right += 1
    return min(left, right)


edge = int(max(_edges, key=lambda e: (round(min(_slope[e - 3:e + 5]), 2), _room(e), -abs(e - N // 2))))   # head-on edge
print('profile edge at pixel', edge, 'slope', round(float(min(_slope[edge - 3:edge + 5])), 3))

SURFACE, TEXT_1, TEXT_2, GRID, AXIS = '#fcfcfb', '#0b0b0b', '#3a3936', '#e6e5e1', '#a3a29c'
FONT, TICK, FPS, DPI = 20, 16, 2, 260            # DPI: the 1024^2 image gets >= ~1 screen pixel per pixel
fig = plt.figure(figsize=(16, 5.4), dpi=100, facecolor=SURFACE)
gs = fig.add_gridspec(1, 3, left=0.075, right=0.99, top=0.9, bottom=0.17, wspace=0.18)
ax_im, ax_line, ax_its = (fig.add_subplot(gs[0, j]) for j in range(3))
for ax in (ax_im, ax_line, ax_its):
    ax.set_box_aspect(1)


def style(ax, title, xlabel):
    ax.set_facecolor(SURFACE)
    ax.set_title(title, loc='center', fontsize=FONT, color=TEXT_1, pad=10)
    ax.set_xlabel(xlabel, color=TEXT_2, fontsize=FONT)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    for side in ('left', 'bottom'):
        ax.spines[side].set_color(AXIS)
    ax.tick_params(colors=TEXT_2, labelsize=TICK)
    ax.grid(axis='y', color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


LIN = os.environ.get('IMAGE_SCALE') == 'linear'     # IMAGE_SCALE=linear: grey linear in rho (0 ... 1), output tag _lin
shade = (lambda d: d) if LIN else np.log10
image = ax_im.imshow(shade(density(0)).T, origin='lower', cmap='Greys', vmin=k if LIN else -4, vmax=1 if LIN else 0, interpolation='nearest',
                     resample=False)
ax_im.set_xticks([]), ax_im.set_yticks([])
for side in ax_im.spines.values():
    side.set_color(AXIS)
ax_im.axhline(ROW, color='#d62a2a', lw=1.5, ls=(0, (5, 3)))
cax = ax_im.inset_axes([-0.14, 0.0, 0.05, 1.0])
bar = fig.colorbar(image, cax=cax)
bar.set_ticks([1, 0.5, k], labels=['$1$', '$0.5$', '$10^{-4}$']) if LIN else bar.set_ticks([0, -2, -4], labels=['$1$', '$10^{-2}$', '$10^{-4}$'])
cax.yaxis.set_ticks_position('left')
bar.outline.set_visible(False)
cax.tick_params(colors=TEXT_2, labelsize=TICK, length=0)
ax_im.set_title(r'density $\rho$', loc='center', fontsize=FONT, color=TEXT_1, pad=10)

style(ax_line, 'profile', 'pixel')
ax_line.axvspan(edge + 0.5, edge + 0.5, color=AXIS)
ax_line.plot(x, density(0)[:, ROW], color=AXIS, lw=2.0, drawstyle='steps-mid')
profile, = ax_line.plot(x, density(0)[:, ROW], color='#d62a2a', lw=2.0, drawstyle='steps-mid', zorder=5)
ax_line.set_xlim(*([float(v) for v in os.environ['PROFILE_XLIM'].split(',')] if 'PROFILE_XLIM' in os.environ
                   else (edge - WIN, edge + WIN))), ax_line.set_ylim(k, 1.02)
ax_im.plot(list(ax_line.get_xlim()), [ROW, ROW], color='#d62a2a', lw=4, solid_capstyle='butt')   # zoomed part of the row
ax_line.set_yticks([k, 0.5, 1], ['$10^{-4}$', '$0.5$', '$1$'])

style(ax_its, 'Conjugate gradients', r'phase-field width $l/h$')
ax_its.set_ylabel('number of iterations', color=TEXT_2, fontsize=FONT)
lines = {}
SHOWN = os.environ.get('SHOWN', 'Green,Green_Jacobi,dual').split(',')   # curves drawn; y range from all three
for P, label, c, f in PRECS:
    if P not in SHOWN:
        continue
    lines[P], = ax_its.plot([], [], color=c, lw=3, marker='o', ms=6, label=label)
ax_its.set_xlim(-0.1, 4.1)
ax_its.set_ylim(0, 1.45 * max(f * v for P, _, _, f in PRECS for v in its[P].values()))
fig.canvas.draw()                                  # long title: shrink until it fits the panel width
_t = ax_its.title
while _t.get_window_extent().width > ax_its.get_window_extent().width * 1.08 and _t.get_fontsize() > 12:
    _t.set_fontsize(_t.get_fontsize() - 0.5)
    fig.canvas.draw()
print('right title font', _t.get_fontsize())
ax_its.legend(frameon=False, fontsize=TICK, loc='upper right', borderaxespad=0.1)


def draw(j):
    eta = etas[j]
    s = density(eta)
    image.set_data(shade(s).T)
    profile.set_data(x, s[:, ROW])
    ax_line.set_title(rf'profile, $l = {eta:g}\,h$', loc='center', fontsize=FONT, color=TEXT_1, pad=10)
    for P, _, _, f in PRECS:
        if P not in lines:
            continue
        lines[P].set_data(etas[:j + 1], [f * its[P][e] for e in etas[:j + 1]])


out = os.path.join(HERE, 'figures', f'phasefield_interface_N{N}' + ('_lin' if LIN else '')
                   + ('' if len(SHOWN) == 3 else '_' + '-'.join({'Green': 'G', 'Green_Jacobi': 'GJ', 'dual': 'D'}[q] for q in SHOWN)))
if os.environ.get('FRAMES') == '1':             # FRAMES=1: one PNG per eta (as the video frames) for beamer overlays
    for j in range(len(etas)):
        draw(j)
        fig.savefig(out + f'_frame{j:02d}.png', dpi=DPI, facecolor=SURFACE)
    print(out + '_frameXX.png', len(etas), 'frames')
    raise SystemExit
frames = list(range(len(etas))) + [len(etas) - 1] * (3 * FPS)
writer = animation.FFMpegWriter(fps=FPS, bitrate=12000, codec='libx264', extra_args=['-pix_fmt', 'yuv420p'])
with writer.saving(fig, out + '.mp4', dpi=DPI):
    for j in frames:
        draw(j)
        writer.grab_frame(facecolor=SURFACE)
fig.savefig(out + '_final.png', dpi=150, facecolor=SURFACE)
# static right panel (last frame) for the Green-Jacobi / dual slides
fig.canvas.draw()
_bb = ax_its.get_tightbbox(fig.canvas.get_renderer()).transformed(fig.dpi_scale_trans.inverted()).padded(0.08)
fig.savefig(out + '_iterations_final.png', dpi=200, facecolor=SURFACE, bbox_inches=_bb)
draw(0)
fig.savefig(out + '_initial.png', dpi=150, facecolor=SURFACE)
print(out + '.mp4', len(etas), 'eta values')
