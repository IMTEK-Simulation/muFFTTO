"""Animation of the deformation mechanism of a 3D TO design: uniaxial compression along z with free lateral faces
(macro stress sigma = -e_z e_z), linear elastic, magnified. Solid rho >= 0.5 in blue, 2 x 2 x 2 periodic cells,
compressed and released periodically. Left: oblique view; right: side view (parallel projection, looking along y).
Grey box: undeformed outline; red box: outline moved by the macroscopic strain (inward = auxetic).

Input: <design>_deformation.nc of make_TO3D_deformation.py (6 unit load cases, displacement in grid-point units).
Macro strain E = C_eff^-1 sigma (Voigt, engineering shear = load-case weights); x = X + s t (E X + sum_k E_k u~_k),
t = (1 - cos) / 2 in [0, 1], s chosen so the axial strain is shown as STRAIN_SHOWN (same factor for every frame).
    <pyvista python> render_auxetic_mechanism.py <deformation.nc> <out.mp4> [title]
"""
import os
import sys

import imageio_ffmpeg
import matplotlib

matplotlib.use('Agg')
import netCDF4
import numpy as np
import pyvista as pv
from scipy import ndimage
from skimage import measure

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from elastic_properties import properties, voigt_from_stresses

fn, out = sys.argv[1], sys.argv[2]
title = sys.argv[3] if len(sys.argv) > 3 else ''
STRAIN_SHOWN, TILE = 0.15, 2
W, H, FPS, PERIOD, CYCLES = 3200, 1600, 30, 4.0, 3
ENC = ['-profile:v', 'high', '-level', '5.1', '-crf', '17', '-g', '30', '-keyint_min', '30', '-preset', 'slow',
       '-maxrate', '10M', '-bufsize', '20M', '-movflags', '+faststart']
VOIGT = [(0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1)]

with netCDF4.Dataset(fn) as d:
    rho = np.array(d['phase_field'][0, 0])
    U = np.array(d['displacement'][:, :, 0])                     # (6, 3, N, N, N), grid-point units
    hs = np.array(d['homogenized_stress'])
N = rho.shape[0]
C = voigt_from_stresses(hs)
p = properties(C)
sig = np.zeros(6)
sig[2] = -1.0                                                    # compression along z
Ev = np.linalg.solve(C, sig)
E = np.zeros((3, 3))
for k, (i, j) in enumerate(VOIGT):
    E[i, j] = E[j, i] = Ev[k] if i == j else 0.5 * Ev[k]
u = np.tensordot(Ev, U, axes=(0, 0)) / N                         # fluctuation in cell units, nodal (3, N, N, N)
s = STRAIN_SHOWN / abs(E[2, 2])
print(f'nu: {p}  macro strain diag {np.diag(E)}  lateral/axial {E[0, 0] / E[2, 2]:+.3f}, {E[1, 1] / E[2, 2]:+.3f}')

# solid surface of the 2 x 2 x 2 tiling, closed at the outer faces; nodal values at X = i / N
rt = np.tile(rho, (TILE,) * 3)
v, f, nrm, _ = measure.marching_cubes(np.pad(rt, 1, constant_values=0.0), 0.5)
idx = v - 1.0                                                    # node index in the tiled grid
X = idx / N
ut = np.tile(u, (1,) + (TILE,) * 3)
disp_fluct = np.stack([ndimage.map_coordinates(ut[c], idx.T, order=1, mode='grid-wrap') for c in range(3)], axis=1)
disp = X @ E.T + disp_fluct                                      # per vertex, for t = 1, unmagnified
mesh = pv.PolyData(X.copy(), np.hstack([np.full((len(f), 1), 3), f]).astype(np.int64))
L = float(TILE)
corners = np.array([[a, b, c] for a in (0, L) for b in (0, L) for c in (0, L)])
edges = [(i, j) for i in range(8) for j in range(i + 1, 8) if np.sum(np.abs(corners[i] - corners[j]) > 0) == 1]


def box(pts):
    lines = np.hstack([[2, i, j] for i, j in edges])
    return pv.PolyData(pts, lines=lines)


pl = pv.Plotter(off_screen=True, window_size=(W, H), shape=(1, 2), border=False, lighting='three lights')
actors = []
for col in range(2):
    pl.subplot(0, col)
    pl.set_background('white')
    m = mesh.copy()
    pl.add_mesh(m, color='#6f8fb8', smooth_shading=True, specular=0.3, specular_power=15)
    pl.add_mesh(box(corners), color='#9a9a9a', line_width=3)
    b = box(corners.astype(float))
    pl.add_mesh(b, color='#d62728', line_width=5)
    actors.append((m, b))
    c0 = np.array([L / 2] * 3)
    if col == 0:
        a, e, dist = np.radians(-55), np.radians(20), 4.2 * L
        pl.camera.position = tuple(c0 + dist * np.array([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)]))
        pl.add_text('oblique view', position='upper_left', font_size=14, color='black')
    else:
        pl.enable_parallel_projection()
        pl.camera.position = tuple(c0 + np.array([0.0, -4.0 * L, 0.0]))
        pl.camera.parallel_scale = 0.75 * L
        pl.add_text('side view (along y): sides move inward = auxetic', position='upper_left', font_size=14,
                    color='black')
    pl.camera.focal_point = tuple(c0)
    pl.camera.up = (0, 0, 1)
pl.subplot(0, 0)
pl.add_text(f'{title}\ncompression along z, free sides; nu_31 = {E[0, 0] / E[2, 2] * -1:+.2f}, '
            f'nu_32 = {E[1, 1] / E[2, 2] * -1:+.2f}  (deformation magnified)', position='lower_left', font_size=12,
            color='black')
pl.enable_anti_aliasing('ssaa')

w = imageio_ffmpeg.write_frames(out, (W, H), fps=FPS, quality=None, codec='libx264', pix_fmt_out='yuv420p',
                                macro_block_size=8, output_params=ENC)
w.send(None)
n = int(PERIOD * FPS)
for k in range(n * CYCLES + FPS):
    t = 0.5 * (1 - np.cos(2 * np.pi * min(k, n * CYCLES) / n))
    for m, b in actors:
        m.points = X + s * t * disp
        b.points = corners + s * t * corners @ E.T
    pl.render()
    img = np.asarray(pl.screenshot(return_img=True))[:, :, :3]
    w.send(np.ascontiguousarray(img))
    if k == n // 2:                                              # fully compressed: still image
        import matplotlib.pyplot as plt
        plt.imsave(out.replace('.mp4', '_compressed.png'), img)
w.close()
print('saved', out)
