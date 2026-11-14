"""Talk video of a 3D phase-field topology optimisation run: the design rho = 0.5 surface (solid side, closed at the cell
faces) of every stored history frame (one every 10 L-BFGS iterations), camera turning slowly; at the end one full turn
of the final design. Right: objective vs L-BFGS iteration up to the current frame. Frames are held (no blending in
time: every shown design is a stored iterate).

Runs headless on NEMO (VTK EGL) with ~/software/render-venv (pyvista, vtk, scikit-image, netCDF4, imageio-ffmpeg).
No LaTeX there: matplotlib mathtext with Computer Modern (cm) for the text.
    python render_TO3D_video.py <history.nc> [<history_seg2.nc> ...]     env: TEST=1 (one frame -> PNG), OUT, TITLE
    CURVE_CSV=<_objsplit.csv of eval_TO3D_objective_split.py>: plot the stress mismatch S instead of the objective,
    REF_S (dashed reference level), REF_LABEL
    SHOW=solid: the solid rho >= 0.5 in blue (default: the soft phase in orange); TILE=<n>: n x n x n periodic cells;
    NOTE: extra text line under the iteration counter (e.g. the measured Poisson ratio)
-> <OUT>.mp4 (+ _final.png), default OUT = figures/TO3D_video
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
OUT = os.environ.get('OUT', os.path.join(HERE, 'figures', 'TO3D_video'))
TEST = os.environ.get('TEST') == '1'
W, H = 3600, 2024
FPS, TURN = 30, 360                             # frames of the final full turn
HOLD_EARLY, N_EARLY, HOLD_LATE = 20, 10, 1      # video frames per design frame: the first N_EARLY frames (where the
                                                # design forms) slowly, the converged tail quickly
ENC = ['-profile:v', 'high', '-level', '5.1', '-crf', '17', '-g', '30', '-keyint_min', '30', '-preset', 'slow',
       '-maxrate', '10M', '-bufsize', '20M', '-movflags', '+faststart']
SHOW = os.environ.get('SHOW', 'soft')
TILE = int(os.environ.get('TILE', 1))
SOLID = '#6f8fb8' if SHOW == 'solid' else '#d9822b'                # shown phase colour
plt.rcParams.update({'mathtext.fontset': 'cm', 'font.family': 'serif', 'font.serif': ['cmr10', 'DejaVu Serif'],
                     'axes.formatter.use_mathtext': True, 'font.size': 30})

# ---- data: all segments, one design frame every 10 iterations
frames, obj = [], []
for f in FILES:
    with netCDF4.Dataset(f) as d:
        pf = d['phase_field_iterate']
        o = np.array(d['objective'][:])
        start = 1 if frames else 0                        # a continuation segment repeats its start design
        for k in range(start, pf.shape[0]):
            frames.append(np.asarray(pf[k, 0], dtype=np.float32))
            obj.append(float(o[k]))
        if f is FILES[0]:
            N = pf.shape[-1]
            prec = str(d.getncattr('preconditioner_type'))
            soft = float(d.getncattr('soft_phase'))
obj = np.array(obj)
CURVE_CSV = os.environ.get('CURVE_CSV', '')
if CURVE_CSV:                                    # stress mismatch S per frame (single-segment histories)
    obj = np.genfromtxt(CURVE_CSV, delimiter=',', names=True)['stress_mismatch_S'][:len(frames)]
    frames = frames[:len(obj)]
its = 10 * np.arange(len(frames))
print(f'{len(frames)} design frames, N = {N}, preconditioner {prec}, soft {soft:g}, final objective {obj[-1]:.4g}',
      flush=True)
title = os.environ.get('TITLE', rf'3D phase-field topology optimization, ${N}^3$ voxels, contrast $10^{{{int(round(np.log10(soft)))}}}$')


def surface(rho):
    """rho = 0.5 surface of the SOFT phase (as plot_TO3D_*: the solid is the majority), closed at the cell faces,
    in cell units [0, 1]."""
    field = rho if SHOW == 'solid' else 1.0 - rho
    a = np.pad(np.tile(field, (TILE,) * 3), 1, constant_values=0.0)
    v, f, n, _ = measure.marching_cubes(a, 0.5)
    v = (v - 0.5) / (N * TILE)                         # node values at voxel centres; whole picture in [0, 1]
    m = pv.PolyData(v, np.hstack([np.full((len(f), 1), 3), f]).astype(np.int64))
    m.point_data['Normals'] = -n
    return m


p = pv.Plotter(off_screen=True, window_size=(W, H), lighting='three lights')
p.set_background('white')
p.add_mesh(pv.Box((0, 1, 0, 1, 0, 1)).extract_all_edges(), color='#8c8c8c', line_width=2)
actor = None
p.enable_anti_aliasing('ssaa')


def set_design(rho):
    global actor
    if actor is not None:
        p.remove_actor(actor)
    actor = p.add_mesh(surface(rho), color=SOLID, smooth_shading=True, specular=0.35, specular_power=18)


def set_view(azim, elev=22.0, dist=3.5):
    a, e = np.radians(azim), np.radians(elev)
    c = np.array([0.5, 0.5, 0.5])
    p.camera.focal_point = tuple(c)
    p.camera.position = tuple(c + dist * np.array([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)]))
    p.camera.up = (0, 0, 1)
    p.camera.view_angle = 30.0


# ---- overlay: title, iteration text, objective curve (3D view on the left 62 %)
fig = plt.figure(figsize=(W / 150, H / 150), dpi=150)
fig.patch.set_alpha(0.0)
fig.text(0.5, 0.955, title, ha='center', va='center', fontsize=34)
fig.text(0.31, 0.885, (r'solid $\rho \geq 0.5$ shown' if SHOW == 'solid' else r'soft phase $\rho < 0.5$ shown')
         + (rf', ${TILE} \times {TILE} \times {TILE}$ periodic cells' if TILE > 1 else ' (periodic unit cell)'),
         ha='center', va='center', fontsize=28)
if os.environ.get('NOTE'):
    fig.text(0.31, 0.02, os.environ['NOTE'], ha='center', va='center', fontsize=30)
it_txt = fig.text(0.31, 0.06, '', ha='center', va='center', fontsize=30)
ax = fig.add_axes([0.70, 0.22, 0.27, 0.52])
ax.patch.set_alpha(1.0)
ax.plot(its, obj, color='0.85', lw=2.5)
line, = ax.plot([], [], color='tab:red' if prec == 'dual' else 'k', lw=3.5)
dot, = ax.plot([], [], 'o', color=line.get_color(), ms=12)
if CURVE_CSV:
    ax.set_ylim(0, min(1.05 * obj.max(), 3.0))
    if os.environ.get('REF_S'):
        ax.axhline(float(os.environ['REF_S']), color='0.4', ls='--', lw=2)
        ax.text(its[-1], float(os.environ['REF_S']), os.environ.get('REF_LABEL', 'reference'), ha='right', va='bottom',
                fontsize=22, color='0.3')
    ax.set(xlabel='L-BFGS iteration', ylabel=r'stress mismatch $S$', xlim=(0, its[-1]))
else:
    ax.set_ylim(0, 1.05 * obj.max())
    ax.set(xlabel='L-BFGS iteration', ylabel='objective $f$', xlim=(0, its[-1]))
ax.grid(alpha=0.3, which='both')


def overlay(k):
    line.set_data(its[:k + 1], obj[:k + 1])
    dot.set_data([its[k]], [obj[k]])
    it_txt.set_text(rf'iteration ${its[k]}$,  ' + (rf'$S = {obj[k]:.3g}$' if CURVE_CSV else rf'$f = {obj[k]:.3g}$'))
    fig.canvas.draw()
    return np.asarray(fig.canvas.buffer_rgba()).astype(np.float32) / 255.0


def compose(k):
    p.render()
    rgb = np.asarray(p.screenshot(return_img=True))[:, :, :3].astype(np.float32) / 255.0
    rgb = np.roll(rgb, -int(0.19 * W), axis=1)                          # 3D view to the left part
    rgb[:, -int(0.19 * W):] = 1.0
    ov = overlay(k)
    a = ov[:, :, 3:4]
    return (np.clip(rgb * (1 - a) + ov[:, :, :3] * a, 0, 1) * 255).astype(np.uint8)


AZ0, SPIN = -60.0, 0.25                         # degrees per video frame during the optimisation
if TEST:
    k = len(frames) // 3
    set_design(frames[k]), set_view(AZ0)
    plt.imsave(OUT + '_test.png', compose(k))
    set_design(frames[-1]), set_view(AZ0)
    plt.imsave(OUT + '_test_final.png', compose(len(frames) - 1))
    print('saved', OUT + '_test.png')
    sys.exit()

w = imageio_ffmpeg.write_frames(OUT + '.mp4', (W, H), fps=FPS, quality=None, codec='libx264', pix_fmt_out='yuv420p',
                                macro_block_size=8, output_params=ENC)
w.send(None)
az = AZ0
for k, rho in enumerate(frames):
    set_design(rho)
    for _ in range(HOLD_EARLY if k < N_EARLY else HOLD_LATE):
        set_view(az)
        img = compose(k)
        w.send(img)
        az += SPIN
    if k % 10 == 0:
        print('design frame', k, flush=True)
for j in range(TURN):                                                 # one full turn of the final design
    set_view(az + 360.0 * j / TURN)
    img = compose(len(frames) - 1)
    w.send(img)
w.close()
plt.imsave(OUT + '_final.png', img)
print('saved', OUT + '.mp4')
