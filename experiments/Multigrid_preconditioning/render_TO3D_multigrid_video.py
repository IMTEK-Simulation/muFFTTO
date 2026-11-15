"""
Video of a 3D topology optimisation that runs through a SEQUENCE OF GRIDS (e.g. 128^3 -> 256^3 -> 512^3): the solid
(rho >= 0.5, closed at the cell faces) of every stored design, in the order of the files, camera turning slowly; at the
end one full turn of the last design. Right: objective against the cumulative L-BFGS iteration, one colour-neutral
curve per grid, the current design marked. Every shown design is a stored iterate (no blending in time).
Adapted from experiments/dual_preconditioner/topology_optimization/render_TO3D_video.py (single grid).

    python render_TO3D_multigrid_video.py <history or checkpoint .nc> [<next .nc> ...]
Files: `_history.nc` of to3d_large.py (all frames; one every HISTORY_EVERY iterations) or `_checkpoint.nc` / `_final.nc`
(one design; its objective is taken from OBJECTIVES). A history whose first frame is the interpolated design of the
previous grid shows that frame too (it is the start of the stage).
Environment: OUT (figures/TO3D_multigrid_video), TEST=1 (three stills instead of the video), TITLE, HISTORY_EVERY (50),
             OBJECTIVES="<file index>:<objective>,..." for single-design files, HOLD (video frames per design, 45).
Runs headless on NEMO (VTK EGL) with ~/software/render-venv (pyvista, vtk, scikit-image, netCDF4, imageio-ffmpeg):
sbatch_render_TO3D_multigrid.sh.
"""
import os
import sys

import imageio_ffmpeg
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import netCDF4
import numpy as np
import pyvista as pv
from skimage import measure

HERE = os.path.dirname(os.path.abspath(__file__))
FILES = sys.argv[1:]
OUT = os.environ.get('OUT', os.path.join(HERE, 'figures', 'TO3D_multigrid_video'))
TEST = os.environ.get('TEST') == '1'
EVERY = int(os.environ.get('HISTORY_EVERY', 50))
HOLD = int(os.environ.get('HOLD', 45))
GIVEN = {int(k): float(v) for k, v in (item.split(':') for item in os.environ.get('OBJECTIVES', '').split(',') if item)}
W, H = 3600, 2024
FPS, TURN = 30, 360                                   # frames of the final full turn
ENC = ['-profile:v', 'high', '-level', '5.1', '-crf', '17', '-g', '30', '-keyint_min', '30', '-preset', 'slow',
       '-maxrate', '10M', '-bufsize', '20M', '-movflags', '+faststart']
SOLID = '#6f8fb8'
SHADES = ['#9aa0a6', '#5f6b7a', '#1f2a3a', '#000000']                    # one grey per grid, coarse -> fine
plt.rcParams.update({'font.size': 30, 'axes.spines.top': False, 'axes.spines.right': False})

# ---- data: (file, frame index, grid, iteration within the grid, cumulative iteration, objective)
designs, offset, previous_grid, grid_iteration = [], 0, None, 0
for index, name in enumerate(FILES):
    with netCDF4.Dataset(name) as data:
        field = data['phase_field_iterate'] if 'phase_field_iterate' in data.variables else data['phase_field_solution']
        n = field.shape[-1]
        if n != previous_grid:
            offset, grid_iteration, previous_grid = offset + grid_iteration, 0, n
        if 'objective' in data.variables:                               # history: frame k is iteration k * EVERY
            objectives = np.array(data['objective'][:])
            for k in range(field.shape[0]):
                grid_iteration = k * EVERY
                designs.append((name, k, n, grid_iteration, offset + grid_iteration, float(objectives[k])))
        else:                                                           # one design, iteration from the attributes
            grid_iteration = int(data.getncattr('iteration')) if 'iteration' in data.ncattrs() else grid_iteration + EVERY
            designs.append((name, field.shape[0] - 1, n, grid_iteration, offset + grid_iteration, GIVEN.get(index, np.nan)))
grids = sorted({d[2] for d in designs})
print(f'{len(designs)} designs on the grids {grids}', flush=True)
title = os.environ.get('TITLE', '3D phase-field topology optimization through the grids '
                       + ' → '.join(f'{n}³' for n in grids))


def load(design):
    name, k, n = design[:3]
    with netCDF4.Dataset(name) as data:
        field = data['phase_field_iterate'] if 'phase_field_iterate' in data.variables else data['phase_field_solution']
        return np.asarray(field[k]).reshape(n, n, n).astype(np.float32)


def surface(rho):
    """rho = 0.5 surface of the solid, closed at the cell faces, in cell units [0, 1]"""
    n = rho.shape[0]
    vertices, faces, normals, _ = measure.marching_cubes(np.pad(rho, 1, constant_values=0.0), 0.5)
    mesh = pv.PolyData((vertices - 0.5) / n, np.hstack([np.full((len(faces), 1), 3), faces]).astype(np.int64))
    mesh.point_data['Normals'] = -normals
    return mesh


plotter = pv.Plotter(off_screen=True, window_size=(W, H), lighting='three lights')
plotter.set_background('white')
plotter.add_mesh(pv.Box((0, 1, 0, 1, 0, 1)).extract_all_edges(), color='#8c8c8c', line_width=2)
plotter.enable_anti_aliasing('ssaa')
actor = None


def set_design(design):
    global actor
    if actor is not None:
        plotter.remove_actor(actor)
    actor = plotter.add_mesh(surface(load(design)), color=SOLID, smooth_shading=True, specular=0.35, specular_power=18)


def set_view(azimuth, elevation=22.0, distance=3.5):
    a, e = np.radians(azimuth), np.radians(elevation)
    centre = np.array([0.5, 0.5, 0.5])
    plotter.camera.focal_point = tuple(centre)
    plotter.camera.position = tuple(centre + distance * np.array([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)]))
    plotter.camera.up = (0, 0, 1)
    plotter.camera.view_angle = 30.0


# ---- overlay: title, grid / iteration text, objective against the cumulative iteration (3D view on the left 62 %)
fig = plt.figure(figsize=(W / 150, H / 150), dpi=150)
fig.patch.set_alpha(0.0)
fig.text(0.5, 0.955, title, ha='center', va='center', fontsize=34)
fig.text(0.31, 0.885, 'solid shown (periodic unit cell)', ha='center', va='center', fontsize=28)
text = fig.text(0.31, 0.06, '', ha='center', va='center', fontsize=32)
ax = fig.add_axes([0.70, 0.22, 0.27, 0.52])
ax.patch.set_alpha(1.0)
cumulative = np.array([d[4] for d in designs], dtype=float)
objective = np.array([d[5] for d in designs])
finite = objective[np.isfinite(objective)]
top = 1.15 * np.sort(finite)[-2] if len(finite) > 2 else 1.15 * finite.max()      # the very first value is far off scale
for shade, n in zip(SHADES, grids):
    members = [i for i, d in enumerate(designs) if d[2] == n and np.isfinite(d[5])]
    ax.plot(cumulative[members], objective[members], color=shade, linewidth=3.5, marker='o', markersize=9)
    ax.annotate(f'{n}³', (cumulative[members[-1]], objective[members[-1]]), xytext=(0, 18), textcoords='offset points',
                ha='center', fontsize=26, color=shade)
dot, = ax.plot([], [], 'o', color='#d1242f', markersize=20, zorder=5)
ax.set(xlabel='L-BFGS iteration (all grids)', ylabel='objective', xlim=(0, cumulative.max() * 1.03), ylim=(0, top))
ax.grid(axis='y', alpha=0.3)


def overlay(i):
    name, k, n, iteration, total, f = designs[i]
    dot.set_data([total], [min(f, top)] if np.isfinite(f) else [np.nan])
    text.set_text(f'grid {n}³,  iteration {iteration}' + (f',  objective {f:.4g}' if np.isfinite(f) else ''))
    fig.canvas.draw()
    return np.asarray(fig.canvas.buffer_rgba()).astype(np.float32) / 255.0


def compose(i):
    plotter.render()
    rgb = np.asarray(plotter.screenshot(return_img=True))[:, :, :3].astype(np.float32) / 255.0
    rgb = np.roll(rgb, -int(0.19 * W), axis=1)                          # 3D view to the left part
    rgb[:, -int(0.19 * W):] = 1.0
    layer = overlay(i)
    alpha = layer[:, :, 3:4]
    return (np.clip(rgb * (1 - alpha) + layer[:, :, :3] * alpha, 0, 1) * 255).astype(np.uint8)


AZIMUTH0, SPIN = -60.0, 0.25                          # degrees per video frame during the optimisation
os.makedirs(os.path.dirname(OUT), exist_ok=True)
if TEST:
    for label, i in (('first', 0), ('middle', len(designs) // 2), ('last', len(designs) - 1)):
        set_design(designs[i]), set_view(AZIMUTH0)
        plt.imsave(f'{OUT}_test_{label}.png', compose(i))
        print('saved', f'{OUT}_test_{label}.png', flush=True)
    sys.exit()

writer = imageio_ffmpeg.write_frames(OUT + '.mp4', (W, H), fps=FPS, quality=None, codec='libx264', pix_fmt_out='yuv420p',
                                     macro_block_size=8, output_params=ENC)
writer.send(None)
azimuth = AZIMUTH0
for i, design in enumerate(designs):
    set_design(design)
    for _ in range(HOLD):
        set_view(azimuth)
        image = compose(i)
        writer.send(image)
        azimuth += SPIN
    print('design', i, design[2], design[3], flush=True)
for j in range(TURN):                                                   # one full turn of the last design
    set_view(azimuth + 360.0 * j / TURN)
    image = compose(len(designs) - 1)
    writer.send(image)
writer.close()
plt.imsave(OUT + '_final.png', image)
print('saved', OUT + '.mp4')
