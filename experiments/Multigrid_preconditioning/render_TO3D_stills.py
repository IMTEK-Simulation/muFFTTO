"""
3D stills of single designs (start / final .nc of to3d_large.py or make_CH512_start.py): the solid (rho >= 0.5, closed at
the cell faces, same surface and colour as render_TO3D_multigrid_video.py), whole periodic cell and a zoom on one corner
block (the first ZOOM fraction of the cell in x, y, z), then one sheet with all of them.

    python render_TO3D_stills.py <a.nc> [<b.nc> ...]     LABELS="label a|label b|..."  OUT (figures/TO3D_stills/stills)
        -> <OUT>_<i>_cell.png, <OUT>_<i>_zoom.png per file, <OUT>_sheet.png (one row per file: cell | zoom)
Env: LABELS ('|'-separated, one per file), ZOOM (0.25), SIZE (still edge in pixels, 1400), PER_ROW (files per sheet
     row, 1), OUT.
Runs headless on NEMO (VTK EGL) with ~/software/render-venv: sbatch_render_TO3D_stills.sh.
"""
import os
import sys

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import netCDF4
import numpy as np
import pyvista as pv
from skimage import measure

HERE = os.path.dirname(os.path.abspath(__file__))
FILES = sys.argv[1:]
LABELS = os.environ.get('LABELS', '|'.join(os.path.basename(f) for f in FILES)).split('|')
OUT = os.environ.get('OUT', os.path.join(HERE, 'figures', 'TO3D_stills', 'stills'))
ZOOM, SIZE = float(os.environ.get('ZOOM', 0.25)), int(os.environ.get('SIZE', 1400))
SOLID = '#6f8fb8'
os.makedirs(os.path.dirname(OUT), exist_ok=True)


def load(name):
    with netCDF4.Dataset(name) as data:
        field = data['phase_field_iterate'] if 'phase_field_iterate' in data.variables else data['phase_field_solution']
        n = field.shape[-1]
        return np.asarray(field[field.shape[0] - 1]).reshape(n, n, n).astype(np.float32)


def surface(rho, extent):
    """rho = 0.5 surface of the solid, closed at the block faces, in cell units (block of size extent)"""
    n = rho.shape[0]
    vertices, faces, normals, _ = measure.marching_cubes(np.pad(rho, 1, constant_values=0.0), 0.5)
    mesh = pv.PolyData((vertices - 0.5) * extent / n, np.hstack([np.full((len(faces), 1), 3), faces]).astype(np.int64))
    mesh.point_data['Normals'] = -normals
    return mesh


def still(rho, extent, path):
    plotter = pv.Plotter(off_screen=True, window_size=(SIZE, SIZE), lighting='three lights')
    plotter.set_background('white')
    plotter.add_mesh(pv.Box((0, extent, 0, extent, 0, extent)).extract_all_edges(), color='#8c8c8c', line_width=2)
    plotter.enable_anti_aliasing('ssaa')
    plotter.add_mesh(surface(rho, extent), color=SOLID, smooth_shading=True, specular=0.35, specular_power=18)
    a, e = np.radians(-60.0), np.radians(22.0)
    centre = np.full(3, extent / 2)
    plotter.camera.focal_point = tuple(centre)
    plotter.camera.position = tuple(centre + 3.5 * extent * np.array([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a),
                                                                       np.sin(e)]))
    plotter.camera.up = (0, 0, 1)
    plotter.camera.view_angle = 30.0
    plotter.screenshot(path)
    plotter.close()
    print('saved', path, flush=True)


rows = []
for i, name in enumerate(FILES):
    rho = load(name)
    n = rho.shape[0]
    m = max(int(round(ZOOM * n)), 2)
    solid, grey = float(np.mean(rho >= 0.5)), float(np.mean((rho > 0.1) & (rho < 0.9)))
    print(f'{LABELS[i]}: {n}^3, solid fraction {solid:.3f}, grey fraction {grey:.3f}', flush=True)
    cell, zoom = f'{OUT}_{i}_cell.png', f'{OUT}_{i}_zoom.png'
    still(rho, 1.0, cell)
    still(np.ascontiguousarray(rho[:m, :m, :m]), ZOOM, zoom)
    rows.append((LABELS[i], n, solid, grey, cell, zoom))
    del rho

per_row = int(os.environ.get('PER_ROW', 1))                           # files per sheet row (start | final: 2)
nrows = -(-len(rows) // per_row)
fig, axes = plt.subplots(nrows, 2 * per_row, figsize=(7 * 2 * per_row, 7 * nrows), squeeze=False)
for ax in axes.flat:
    ax.set_axis_off()
for k, (label, n, solid, grey, cell, zoom) in enumerate(rows):
    left, right = axes[k // per_row, 2 * (k % per_row)], axes[k // per_row, 2 * (k % per_row) + 1]
    left.imshow(plt.imread(cell)), right.imshow(plt.imread(zoom))
    left.set_title(f'{label}\n{n}³, solid {solid:.2f}, grey {grey:.3f}: whole cell', fontsize=16)
    right.set_title(f'zoom: corner block {ZOOM:g} of the cell', fontsize=16)
fig.tight_layout()
fig.savefig(f'{OUT}_sheet.png', dpi=SIZE / 7)
print('saved', f'{OUT}_sheet.png', flush=True)
