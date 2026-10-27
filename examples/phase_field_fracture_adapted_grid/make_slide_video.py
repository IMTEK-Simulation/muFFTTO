"""Slide video of one run of example_2D_phase_field_fracture_contact_adapted_grid.py (regular grid), with the same
layout for every slide, so that a sequence of runs (Newton-CG, trust region, contact, fracture) can be compared.

One frame per accepted load increment:
    left          deformed cell, exact pixels: the solid coloured by det F, bubbles / third medium white; damaged
                  solid pixels black -> red, fully cracked pixels (d > 0.95) not drawn
    right, top    |P11| / mu over F11, the run in colour, reference runs (other slides) in grey
    right, bottom mechanics CG iterations per load increment (log scale), reference runs in grey

usage: python make_slide_video.py RUN.nc OUT_PREFIX --title "..." [--reference label=OTHER.nc ...]
                                  [--stress-max 0.3] [--cg-range 10 1e5] [--fps 12] [--no-cg]
--no-cg: only the cell and the stress curve (no CG panel, no CG numbers in the header line)
writes OUT_PREFIX.mp4 and OUT_PREFIX_final.png
"""
import argparse

import imageio_ffmpeg
import matplotlib
import matplotlib.ticker

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import netCDF4
import numpy as np
from matplotlib import animation
from matplotlib.colors import LinearSegmentedColormap, ListedColormap, TwoSlopeNorm

plt.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()

SURFACE, TEXT_1, TEXT_2, GRID, AXIS, FUTURE = "#ffffff", "#0b0b0b", "#52514e", "#e6e5e1", "#a3a29c", "#d4d3cd"
RUN_COLOR = "#2a78d6"
REFERENCE_COLOR = "#b5b4ae"
DET_F_CMAP = LinearSegmentedColormap.from_list("det_F", ["#0d366b", "#3987e5", "#d9d8d4", "#f0a070", "#a8401a"])
DAMAGE_CMAP = LinearSegmentedColormap.from_list("damage", ["#000000", "#ff0000"])
# damage alone: one hue, light (intact) -> dark (cracked)
DAMAGE_ONLY_CMAP = LinearSegmentedColormap.from_list("damage_only", ["#e9e8e4", "#f4b183", "#d9622b", "#8c2a0c",
                                                                     "#2b0a02"])
CRACKED = 0.95                       # damage above this: the pixel is not drawn
FADE_START = 0.6                     # colour = damage: pixels fade from opaque (d <= FADE_START) to transparent (d = 1)
DAMAGED = 0.3                        # damage above this: the pixel is drawn with the damage colours (AT2 damage
                                     # spreads small values over the whole solid; those keep their det F colour)



WHITE_ABOVE = None                   # --white-above: damage above this is not drawn at all (white), no grey fade


def damage_opacity(damage):
    """1 up to FADE_START, then linearly to 0 at d = 1: fully damaged pixels are transparent. With WHITE_ABOVE: fully
    opaque below it and transparent (white) above it."""
    if WHITE_ABOVE is not None:
        return (np.asarray(damage) < WHITE_ABOVE).astype(float)
    return np.clip((1.0 - damage) / (1.0 - FADE_START), 0.0, 1.0)


def damage_colours(damage):
    """RGBA of the damage colour map with the damage opacity."""
    rgba = DAMAGE_ONLY_CMAP(np.clip(damage, 0.0, 1.0))
    rgba[..., 3] = damage_opacity(damage)
    return rgba


parser = argparse.ArgumentParser()
parser.add_argument('run')
parser.add_argument('out')
parser.add_argument('--title', default='')
parser.add_argument('--label', default='this run', help='legend label of the run')
parser.add_argument('--reference', action='append', default=[], help='label=OTHER.nc, drawn in grey (split at the last =)')
parser.add_argument('--compare', action='append', default=[],
                    help='label=OTHER.nc, a comparison run drawn in orange (e.g. the regular grid next to the adapted grid)')
parser.add_argument('--stress-max', type=float, default=None,
                    help='upper limit of |P11| / mu (default: the largest value shown, plus a margin)')
parser.add_argument('--cg-range', type=float, nargs=2, default=None, help='limits of the CG axis (same for all slides)')
parser.add_argument('--det-f-spread', type=float, default=None, help='det F colour range 1 +- spread')
parser.add_argument('--fps', type=int, default=12)
parser.add_argument('--white-above', type=float, default=None,
                    help='damage colouring: pixels with d above this white (not drawn), no grey fade')
parser.add_argument('--cells', type=int, default=1, help='draw CELLS x CELLS periodic copies of the cell')
parser.add_argument('--no-header', action='store_true', help='no title and no header line (the text goes on the slide)')
parser.add_argument('--grid-study', default=None,
                    help='JSON with grids, G, GJ (mean CG per solve): bottom-right panel = CG over the grid size, two lines')
parser.add_argument('--scientific', action='store_true',
                    help='standard colour maps (det F: coolwarm, damage: magma_r), framed colour bar, short label on top')
parser.add_argument('--colorbar-left', action='store_true', help='colour bar vertical, left of the cell')
parser.add_argument('--no-cell-outline', action='store_true',
                    help='with --cells > 1: no red dashed outline of the (deformed) unit cell in the middle')
parser.add_argument('--no-cg', action='store_true', help='no CG panel: the cell and the stress curve only')
parser.add_argument('--colour', choices=('det_F', 'damage'), default='det_F',
                    help="colour of the solid: det F (damaged pixels black -> red) or the damage alone")
parser.add_argument('--initial-only', action='store_true',
                    help='only write OUT_initial.png: the first frame without the curves ahead (poster), no video')
parser.add_argument('--still', type=int, nargs='+', default=None,
                    help='only write these frames as OUT_PREFIX_frame<k>.png, no video')
args = parser.parse_args()
if args.scientific:        # perceptually uniform / standard diverging maps instead of the pastel slide palette
    DET_F_CMAP = plt.get_cmap('coolwarm')         # neutral (det F = 1) light grey: the undeformed solid stays visible
    DAMAGE_ONLY_CMAP = plt.get_cmap('magma_r')
WHITE_ABOVE = args.white_above


def read(path):
    d = netCDF4.Dataset(path)
    H = np.asarray(d.H_macro).reshape(2, 2)
    mu = float(d.E_solid) / (2 * (1 + float(d.nu_solid)))
    lam = np.asarray(d['lam'][:])
    return {'F11': 1 + lam * H[1, 1], 'P11': np.abs(np.asarray(d['mean_P'][:, 1, 1])) / mu,
            'cg': np.asarray(d['nb_hessp'][:]).astype(float), 'dataset': d, 'H': H, 'lam_end': float(d.lam_end),
            'n': int(np.atleast_1d(d.nb_grid_pts)[0])}


run = read(args.run)
references = [(label, read(path)) for label, path in (r.rsplit('=', 1) for r in args.reference)]   # labels may hold '='
comparisons = [(label, read(path)) for label, path in (r.rsplit('=', 1) for r in args.compare)]
COMPARE_COLOR = '#eb6834'
d = run['dataset']
n = run['n']
phase = np.asarray(d['phase'][0, 0])
solid = phase > 0
nb_frames = len(run['F11'])


# reference positions of the pixel corners: the regular grid moved by the grid displacement d_g of an adapted grid
# (zero on the regular grid), (2, n+1, n+1), periodic
regular = np.stack(np.meshgrid(np.arange(n + 1) / n, np.arange(n + 1) / n, indexing='ij'))
reference = regular + np.pad(np.asarray(d['grid_displacement'][0, :, 0]), ((0, 0), (0, 1), (0, 1)), mode='wrap')


def deformed_corners(k):
    """Deformed positions of the pixel corners, (2, n+1, n+1): x = X + H X + u(X), u periodic."""
    u = np.pad(np.asarray(d['u_fluc'][k, :, 0]), ((0, 0), (0, 1), (0, 1)), mode='wrap')
    H = np.asarray(d['lam'][k]) * run['H']
    return reference + np.einsum('ij,jxy->ixy', H, reference) + u


def pixel_damage(k):
    """Damage per pixel: the mean of its four corner nodes."""
    nodal = np.asarray(d['damage'][k, 0])
    return 0.25 * (nodal + np.roll(nodal, -1, 0) + np.roll(nodal, -1, 1) + np.roll(np.roll(nodal, -1, 0), -1, 1))


corners = [deformed_corners(k) for k in range(nb_frames)]
# rigid shift of every frame so that the centre of the drawn cell stays where it is in the first frame: the cell
# does not drift through the picture, and the fixed axes only have to hold its largest shape (true scale kept)
centre_0 = np.array([(corners[0][i].min() + corners[0][i].max()) / 2 for i in range(2)])
corners = [c - (np.array([(c[i].min() + c[i].max()) / 2 for i in range(2)]) - centre_0)[:, None, None]
           for c in corners]


def cell_copies(k):
    """Pixel corners of the CELLS x CELLS periodic copies: copy (a, b) shifted by the deformed lattice vectors
    (I + H) (a, b) (unit cell)."""
    F = np.eye(2) + np.asarray(d['lam'][k]) * run['H']
    return [corners[k] + (F @ np.array([a, b], dtype=float))[:, None, None]
            for a in range(args.cells) for b in range(args.cells)]


x_min = min(c[0].min() for k in range(nb_frames) for c in cell_copies(k))
x_max = max(c[0].max() for k in range(nb_frames) for c in cell_copies(k))
y_min = min(c[1].min() for k in range(nb_frames) for c in cell_copies(k))
y_max = max(c[1].max() for k in range(nb_frames) for c in cell_copies(k))
# colour range from the intact solid only: nearly cracked pixels have meaningless deformations
intact_det_F = np.concatenate([np.asarray(d['detF'][k, 0])[solid & (pixel_damage(k) <= DAMAGED)]
                               for k in range(nb_frames)])
spread = args.det_f_spread or max(abs(intact_det_F.max() - 1), abs(1 - intact_det_F.min()), 1e-3)
det_F_norm = TwoSlopeNorm(vmin=1 - spread, vcenter=1.0, vmax=1 + spread)


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


fig = plt.figure(figsize=(16, 7) if args.no_header else (16, 9), dpi=100, facecolor=SURFACE)   # no header: no empty band on top
# left column as wide as the cell needs to fill the height (the deformed cell is wider than tall), within limits so
# that the plots on the right stay readable
fig_w, fig_h = fig.get_size_inches()
cell_aspect = (x_max - x_min + 0.04) / (y_max - y_min + 0.04)
cell_height = (0.96 - 0.06) * fig_h if args.no_header else (0.86 - 0.08) * fig_h
cell_column = cell_aspect * cell_height / (0.88 if args.colorbar_left else 1.0)    # the colour bar takes ~12 %
LEFT_RATIO = float(np.clip(cell_column / (0.97 * fig_w - cell_column - 0.1 * fig_w), 1.1, 1.9))
if args.no_cg:      # the stress curve alone in the right column, vertically centred
    grid = fig.add_gridspec(3, 2, width_ratios=[LEFT_RATIO, 1.0], height_ratios=[0.06, 1.0, 0.06] if args.no_header else [0.22, 1.0, 0.22], left=0.035, right=0.985,
                            top=0.97 if args.no_header else 0.86, bottom=0.1 if args.no_header else 0.08, wspace=0.07, hspace=0.0)
    ax_cell = fig.add_subplot(grid[:, 0])
    ax_stress = fig.add_subplot(grid[1, 1])
    ax_cg = fig.add_axes([0, 0, 0.01, 0.01])        # kept for the shared code below, not shown
    ax_cg.set_visible(False)
else:
    grid = fig.add_gridspec(2, 2, width_ratios=[LEFT_RATIO, 1.0], left=0.035, right=0.985, top=0.94 if args.no_header else 0.86,
                            bottom=0.08, wspace=0.07, hspace=0.42)
    ax_cell = fig.add_subplot(grid[:, 0])
    ax_stress = fig.add_subplot(grid[0, 1])
    ax_cg = fig.add_subplot(grid[1, 1])
fig.text(0.03, 0.955, '' if args.no_header else args.title, fontsize=18, fontweight='bold', color=TEXT_1)
header = fig.text(0.03, 0.915, '', fontsize=12.5, color=TEXT_2)
if args.no_header:
    header.set_visible(False)

ax_cell.set_aspect('equal')
# colour bar below the cell (default) or vertical on its left (--colorbar-left)
CB_KW = (dict(location='left', fraction=0.03, pad=0.01, aspect=25, shrink=0.6) if args.colorbar_left else
         dict(orientation='horizontal', fraction=0.04, pad=0.02, aspect=40))
ax_cell.set_xlim(x_min - 0.02, x_max + 0.02)
ax_cell.set_ylim(y_min - 0.02, y_max + 0.02)
ax_cell.set_anchor('C')
ax_cell.set_position(ax_cell.get_position().expanded(1.0, (0.96 - 0.04) / ax_cell.get_position().height)
                     if args.no_header else ax_cell.get_position())     # the cell column: the full height
ax_cell.axis('off')
if args.colour == 'det_F':
    colorbar = fig.colorbar(matplotlib.cm.ScalarMappable(norm=det_F_norm, cmap=DET_F_CMAP), ax=ax_cell,
                            extend='both', **CB_KW)
    has_damage = any(np.asarray(d['damage'][k]).max() > DAMAGED for k in range(nb_frames))
    colorbar.set_label(r'$\det \mathbf{F}$ in the solid (bubbles white; '
                       + (rf'damage ${DAMAGED:g} \leq d \leq {CRACKED:g}$ black $\rightarrow$ red, cracked not drawn; '
                          if has_damage else '')
                       + 'exact pixels)', color=TEXT_2, fontsize=10.5)
else:
    # the colour bar shows the colours as drawn: the damage colour faded into the surface
    top = WHITE_ABOVE if WHITE_ABOVE is not None else 1.0      # with --white-above the bar ends where white starts
    levels = np.linspace(0.0, top, 256, endpoint=WHITE_ABOVE is None)
    surface = np.array(matplotlib.colors.to_rgb(SURFACE))
    faded = damage_colours(levels)
    faded = faded[:, :3] * faded[:, 3:] + surface * (1.0 - faded[:, 3:])
    colorbar = fig.colorbar(matplotlib.cm.ScalarMappable(norm=plt.Normalize(0, top), cmap=ListedColormap(faded)),
                            ax=ax_cell, **CB_KW)
    colorbar.set_label((rf'damage $d$ in the solid (bubbles and cracked pixels $d > {WHITE_ABOVE:g}$ white; exact pixels)'
                        if WHITE_ABOVE is not None else
                        rf'damage $d$ in the solid (bubbles white; fading to transparent from $d = {FADE_START:g}$ to '
                        rf'$d = 1$; exact pixels)'), color=TEXT_2, fontsize=10.5)
if args.scientific:        # framed bar, outward ticks, the quantity as a short label on top
    colorbar.set_label('')
    colorbar.outline.set_visible(True)
    colorbar.outline.set_edgecolor('black')
    colorbar.outline.set_linewidth(0.8)
    colorbar.ax.tick_params(colors='black', labelsize=13, length=4, width=0.8, direction='out')
    colorbar.ax.set_title(r'$\det\mathbf{F}$' if args.colour == 'det_F' else r'$d$', fontsize=16, pad=10)
else:
    colorbar.outline.set_visible(False)
    colorbar.ax.tick_params(colors=TEXT_2, labelsize=9.5, length=0)

F11_end = 1 + run['lam_end'] * run['H'][1, 1]
F11_LABEL = r'$F_{11}$ (compression $\rightarrow$)'
style(ax_stress, 'Macroscopic stress', F11_LABEL, r'$|P_{11}|\,/\,\mu$')
style(ax_cg, 'Mechanics CG iterations per load increment', F11_LABEL, 'CG iterations (log)')
for ax in (ax_stress, ax_cg):
    ax.set_xlim(1.0, F11_end - 0.01)                      # compression to the right
reference_styles = [(0, (5, 2)), (0, (1, 1.5)), (0, (6, 2, 1, 2)), (0, (12, 3)), (0, (3, 1, 1, 1, 1, 1))]
assert len(references) <= len(reference_styles), 'more references than line styles'
for (label, reference), dash in zip(references, reference_styles):
    ax_stress.plot(reference['F11'], reference['P11'], color=REFERENCE_COLOR, linewidth=1.8, linestyle=dash,
                   label=label)
    ax_cg.plot(reference['F11'], reference['cg'], color=REFERENCE_COLOR, linewidth=1.3, linestyle=dash,
               label=f'{label} (total {int(reference["cg"].sum()):,})')
for (label, other), dash in zip(comparisons, ['-', (0, (5, 2)), (0, (1, 1.5))]):
    ax_stress.plot(other['F11'], other['P11'], color=COMPARE_COLOR, linewidth=1.6, linestyle=dash, alpha=0.85,
                   label=label)
ax_stress.plot(run['F11'], run['P11'], color=FUTURE, linewidth=2)
stress_line, = ax_stress.plot([], [], color=RUN_COLOR, linewidth=2.2, label=args.label)
stress_dot, = ax_stress.plot([], [], 'o', color=RUN_COLOR, markersize=8, markeredgecolor=SURFACE, markeredgewidth=1.5)
ax_cg.plot(run['F11'], run['cg'], color=FUTURE, linewidth=1.6)
cg_line, = ax_cg.plot([], [], color=RUN_COLOR, linewidth=1.8, label=f'{args.label} (total {int(run["cg"].sum()):,})')
ax_cg.set_yscale('log')
# stress axis: fixed for the whole video, the largest value of all curves shown plus a margin (or --stress-max)
stress_top = max([run['P11'].max()] + [reference['P11'].max() for _, reference in references + comparisons])
ax_stress.set_ylim(0, args.stress_max or 1.08 * stress_top)
if args.cg_range:
    ax_cg.set_ylim(*args.cg_range)
ax_stress.legend(frameon=False, fontsize=10, loc='upper left')
ax_cg.legend(frameon=False, fontsize=10, loc='upper left', ncol=2)
if args.grid_study:        # bottom-right panel: grid-size study instead of CG per increment (static)
    import json
    study = json.load(open(args.grid_study))
    ax_cg.cla()
    style(ax_cg, 'Mean CG iterations per solve', 'grid size', 'CG iterations')
    series = study.get('series') or [dict(label=l, grids=study['grids'], values=study[k], color=c, marker=m)
                                      for k, l, c, m in (('G', 'Green', '#1a8a1a', 'o'), ('GJ', 'Green-Jacobi', '#d62a2a', 's'))]
    for ser in series:     # solid markers: run reached F11 = 0.6; hollow: stopped earlier (mean over the part it ran)
        x, y = np.asarray(ser['grids']), np.asarray(ser['values'], dtype=float)
        partial = np.asarray(ser.get('partial', [False] * len(x)))
        ax_cg.plot(x, y, color=ser['color'], lw=2.4, ls=ser.get('ls', '-'), label=ser['label'])
        ax_cg.plot(x[~partial], y[~partial], ls='none', marker=ser['marker'], ms=8, color=ser['color'])
        ax_cg.plot(x[partial], y[partial], ls='none', marker=ser['marker'], ms=8, mfc='white', mec=ser['color'], mew=2)
    study['grids'] = sorted({g for ser in series for g in ser['grids']})
    ax_cg.set_xscale('log', base=2)
    ax_cg.set_xticks(study['grids'], [f'${n}^2$' for n in study['grids']])
    ax_cg.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax_cg.set_xlim(study['grids'][0] / 1.08, study['grids'][-1] * 1.08)
    ax_cg.set_ylim(0, study.get('ymax', 1000))
    ax_cg.legend(frameon=False, fontsize=11, loc=study.get('legend', 'center right'))
if not np.isclose(run['F11'][-1], F11_end):               # the run stopped early: mark where
    for ax in (ax_stress, ax_cg):
        ax.axvline(run['F11'][-1], color=TEXT_2, linewidth=1, linestyle=(0, (4, 3)))
    ax_stress.text(run['F11'][-1], 0.96, '  stopped', transform=ax_stress.get_xaxis_transform(), color=TEXT_2,
                   fontsize=10, va='top')


# the deformed boundary of one unit cell (the middle copy), red dashed, drawn above the pixels
cell_outline, = ax_cell.plot([], [], color='red', linestyle=(0, (5, 3)), linewidth=1.6, zorder=10)
show_outline = args.cells > 1 and not args.no_cell_outline


def outline(k):
    """The unit cell of the periodic structure as the deformed lattice parallelogram: the corner of the middle copy
    plus (I + H) times the unit square (straight edges; the material boundary would zigzag across the cracks)."""
    c = cell_copies(k)[(args.cells // 2) * args.cells + args.cells // 2]
    F = np.eye(2) + np.asarray(d['lam'][k]) * run['H']
    square = np.array([[0, 1, 1, 0, 0], [0, 0, 1, 1, 0]], dtype=float)
    vertices = c[:, 0, 0][:, None] + F @ square
    return vertices[0], vertices[1]


def draw(k):
    for mesh in list(ax_cell.collections):
        mesh.remove()
    if show_outline:
        cell_outline.set_data(*outline(k))
    damage = pixel_damage(k)
    det_F = np.asarray(d['detF'][k, 0])
    for x, y in cell_copies(k):
        if args.colour == 'det_F':
            intact = solid & (damage <= DAMAGED)
            damaged = solid & (damage > DAMAGED) & (damage <= CRACKED)
            ax_cell.pcolormesh(x, y, np.ma.masked_where(~intact, det_F), shading='flat', cmap=DET_F_CMAP,
                               norm=det_F_norm, rasterized=True)
            if damaged.any():
                ax_cell.pcolormesh(x, y, np.ma.masked_where(~damaged, damage), shading='flat', cmap=DAMAGE_CMAP,
                                   vmin=DAMAGED, vmax=CRACKED, rasterized=True)
        else:
            colours = damage_colours(damage)
            colours[~solid, 3] = 0.0
            mesh = ax_cell.pcolormesh(x, y, damage, shading='flat', rasterized=True, edgecolors='none', linewidth=0,
                                      antialiased=False)            # no outlines around the transparent pixels
            mesh.set_array(None)                                       # the face colours below, not a colour map
            mesh.set_facecolor(colours.reshape(-1, 4))
    stress_line.set_data(run['F11'][:k + 1], run['P11'][:k + 1])
    stress_dot.set_data(run['F11'][k:k + 1], run['P11'][k:k + 1])
    cg_line.set_data(run['F11'][:k + 1], run['cg'][:k + 1])
    header.set_text(f'$F_{{11}}$ = {run["F11"][k]:.3f}    increment {k + 1} / {nb_frames}'
                    + ('' if args.no_cg else
                       f'    CG this increment {int(run["cg"][k])}    total {int(run["cg"][:k + 1].sum()):,}')
                    + (f'    max damage {damage[solid].max():.2f}' if np.asarray(d['damage'][k]).max() > 0 else ''))


if args.initial_only:      # poster: the setting only -- first frame, no future curve, references, 'stopped' mark
    draw(0)
    for line in ax_stress.lines:
        if line not in (stress_line, stress_dot):
            line.set_visible(False)
    for text in ax_stress.texts:
        text.set_visible(False)
    ax_stress.legend(handles=[stress_line], frameon=False, fontsize=10, loc='upper left')
    fig.savefig(f'{args.out}_initial.png', dpi=150, facecolor=SURFACE)
    print(f'wrote {args.out}_initial.png')
    raise SystemExit
if args.still:
    for k in args.still:
        draw(k)
        fig.savefig(f'{args.out}_frame{k}.png', dpi=150, facecolor=SURFACE)
    raise SystemExit
frames = list(range(nb_frames)) + [nb_frames - 1] * (2 * args.fps)       # hold the final state for 2 s
writer = animation.FFMpegWriter(fps=args.fps, bitrate=6000, codec='libx264', extra_args=['-pix_fmt', 'yuv420p'])
with writer.saving(fig, args.out + '.mp4', dpi=100):
    for k in frames:
        draw(k)
        writer.grab_frame(facecolor=SURFACE)
draw(nb_frames - 1)
fig.savefig(args.out + '_final.png', dpi=150, facecolor=SURFACE)
print(f'wrote {args.out}.mp4 ({len(frames)} frames), final F11 = {run["F11"][-1]:.3f}, '
      f'total CG {int(run["cg"].sum()):,}')
