"""Talk video (high resolution, 3600 x 1440 px, 30 fps, H.264): 2D core-shell battery particle, NEMO mesh study.

Left: Li filling c / c_max (particle only) and damage d of the dual run, on their exact pixels (1 data pixel = 1 video
pixel, imshow interpolation='nearest', central crop of the cell). The fields are stored every 10 time steps; the frame
shows the latest stored snapshot (no interpolation in time).
Right: mean CG iterations per elastic solve in each time step, Green (green) / Green-Jacobi (blue) / dual (red),
drawn up to the current time (identical trajectories, so the curves compare like with like).

    python make_battery_slide_video.py [N=1024] [DIM=2]
    DIM=3: the 3D run's mid-plane cut z = L/2 (the only per-step 3D field output), Green and dual only (no Green-Jacobi
    on damage); a curve of a run still in progress stops at its last step (marked in the legend)
    needs exp_data/nemo_fields/battery_N{N}_dual_ms0.01_tu_snapshots.npz (rsync from NEMO) and exp_data/nemo/*.csv
    -> figures/slides/movie_battery_2D_N{N}.mp4 and _final.png
"""
import os
import sys

import imageio_ffmpeg
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Rectangle

HERE = os.path.dirname(os.path.abspath(__file__))
N = int(sys.argv[1]) if len(sys.argv) > 1 else 1024
DIM = int(sys.argv[2]) if len(sys.argv) > 2 else 2
TAG = '_3D' if DIM == 3 else ''
FPS, PER_STEP, HOLD_END = 30, 3, 3.0
# player-safe H.264 (slides froze with the defaults): High@5.1, CRF 17, keyframe every second, faststart
ENC = ['-profile:v', 'high', '-level', '5.1', '-crf', '17', '-g', '30', '-keyint_min', '30', '-preset', 'slow', '-maxrate', '10M', '-bufsize', '20M', '-movflags', '+faststart']
PRECS = ['Green', 'Green_Jacobi', 'dual'] if DIM == 2 else ['Green', 'dual']
COLOR = {'Green': '#2ca02c', 'Green_Jacobi': '#1f77b4', 'dual': '#d62728'}
LABEL = {'Green': 'Green', 'Green_Jacobi': 'Green-Jacobi', 'dual': r'dual ($\times 2$)'}
plt.rcParams.update({'text.usetex': True, 'font.family': 'serif', 'font.size': 22, 'axes.labelsize': 24,
                     'axes.titlesize': 25, 'legend.fontsize': 22, 'xtick.labelsize': 21, 'ytick.labelsize': 21})

z = np.load(os.path.join(HERE, 'exp_data', 'nemo_fields', f'battery_N{N}_dual_ms0.01{TAG}_tu_snapshots.npz'))
snap_t, conc, dmg, particle = z['time'], z['concentration'], z['damage'], z['particle']
T_SCALE, L_CELL = float(z['t_scale']), float(z['L_cell']) * 1e6       # s, um
steps = {P: np.genfromtxt(os.path.join(HERE, 'exp_data', 'nemo', f'battery_N{N}_{P}_ms0.01{TAG}_tu_steps.csv'),
                          delimiter=',', names=True, dtype=None, encoding=None) for P in PRECS}
ref = steps['dual']
ref['cg_u_mean'] *= 2                                                # dual iterations counted x2 (cost)
n_steps = len(ref)
running = {P: len(steps[P]) < n_steps for P in PRECS}
if DIM == 2:
    assert not any(running.values()), {P: len(steps[P]) for P in PRECS}

# central crop, exactly CROP x CROP data pixels shown on CROP x CROP video pixels
CROP = min(N, 900 * N // 1024)
i0 = (N - CROP) // 2
sl = slice(i0, i0 + CROP)
um = L_CELL / N                                                     # pixel size in um
ext = (0, CROP * um, 0, CROP * um)
R_CORE, R_OUT = 2.0, 2.4
ctr = (N / 2 - i0) * um

W, H, DPI = 3600, 1440, 225
fig = plt.figure(figsize=(W / DPI, H / DPI), dpi=DPI)
fig.patch.set_facecolor('white')


def px_axes(x, y, w, h):
    return fig.add_axes([x / W, y / H, w / W, h / H])


P_SIZE = CROP * 1024 // N if N >= 1024 else 900                   # panel size in video pixels (1:1 at N=1024)
y0 = 185
ax_c = px_axes(70, y0, P_SIZE, P_SIZE)
cb_c = px_axes(70 + P_SIZE + 25, y0, 34, P_SIZE)
ax_d = px_axes(70 + P_SIZE + 250, y0, P_SIZE, P_SIZE)
cb_d = px_axes(70 + 2 * P_SIZE + 275, y0, 34, P_SIZE)
gx = 70 + 2 * P_SIZE + 275 + 34 + 360
ax_g = px_axes(gx, y0, W - gx - 60, P_SIZE)

cmask = np.where(particle[sl, sl] > 0, 1.0, np.nan)
im_c = ax_c.imshow((conc[0][sl, sl] * cmask).T, origin='lower', interpolation='nearest', extent=ext, vmin=0.1,
                   vmax=1.0, cmap='viridis')
im_d = ax_d.imshow(dmg[0][sl, sl].T, origin='lower', interpolation='nearest', extent=ext, vmin=0, vmax=1,
                   cmap='inferno')
ax_c.set_facecolor('#cfcfcf')
for ax in (ax_c, ax_d):
    for R in (R_CORE, R_OUT):
        ax.add_patch(Circle((ctr, ctr), R, fill=False, ec='white', lw=1.0, ls=(0, (4, 4)), alpha=0.8))
    ax.set_xticks([]), ax.set_yticks([])
    ax.add_patch(Rectangle((0.25, 0.25), 1.0, 0.07, color='white' if ax is ax_d else 'black'))
    ax.text(0.75, 0.40, r'1\,$\mu$m', color='white' if ax is ax_d else 'black', ha='center', va='bottom',
            fontsize=20)
ax_c.set_title(r'Li filling $c/c_{\max}$', pad=14)
ax_d.set_title(r'damage $d$', pad=14)
fig.colorbar(im_c, cax=cb_c, ticks=[0.2, 0.4, 0.6, 0.8, 1.0], format="%.1f")
fig.colorbar(im_d, cax=cb_d, ticks=[0, 0.5, 1.0], format="%g")

t_all = ref['time'] * T_SCALE
ymax = max(float(np.max(steps[P]['cg_u_mean'])) for P in PRECS)
lines, dots = {}, {}
for P in [q for q in ['Green_Jacobi', 'Green', 'dual'] if q in PRECS]:     # dual on top
    lab = LABEL[P] + (r' (run in progress)' if running[P] else '')
    lines[P], = ax_g.plot([], [], color=COLOR[P], lw=3.0, label=lab, zorder=3 + (P == 'dual'))
    dots[P], = ax_g.plot([], [], 'o', color=COLOR[P], ms=11, zorder=5, clip_on=False)
ax_g.set_xlim(0, 5000)
ax_g.set_ylim(0, 1.04 * ymax)
ax_g.set_xticks([0, 1000, 2000, 3000, 4000, 5000])
ax_g.set_xlabel(r'time [s]')
ax_g.set_ylabel(r'CG per elastic solve (dual $\times 2$)')
ax_g.set_title(rf'elasticity, ${N}\times{N}$ grid' if DIM == 2 else rf'elasticity, ${N}^3$ grid', pad=14)
ax_g.grid(alpha=0.3)
h_, l_ = ax_g.get_legend_handles_labels()
order = sorted(range(len(l_)), key=lambda i: ['Green', 'Green-Jacobi', 'dual'].index(l_[i].split(' (')[0]))
ax_g.legend([h_[i] for i in order], [l_[i] for i in order], loc='upper left', frameon=True, framealpha=0.95)
title = fig.text(0.5, 1 - 70 / H, '', ha='center', va='center', fontsize=27)
MESH_PANEL = os.environ.get('MESH_PANEL')                      # MESH_PANEL=<png>: static CG-vs-grid figure instead of CG over time
if MESH_PANEL:
    ax_g.set_visible(False)
    img = plt.imread(MESH_PANEL)
    pw = 1250                                                  # from right of the damage colour bar labels to the frame edge
    ph = int(pw * img.shape[0] / img.shape[1])
    ax_m = px_axes(W - 40 - pw, y0 + (P_SIZE - ph) // 2, pw, ph)
    ax_m.imshow(img, interpolation='lanczos')
    ax_m.set_axis_off()


def draw(k):
    """State after time step k (fields: latest snapshot at or before that time)."""
    t = ref['time'][k]
    j = max(0, int(np.searchsorted(snap_t, t + 1e-9)) - 1)
    im_c.set_data((conc[j][sl, sl] * cmask).T)
    im_d.set_data(dmg[j][sl, sl].T)
    for P in PRECS:
        kk = min(k, len(steps[P]) - 1)
        lines[P].set_data(t_all[:kk + 1], steps[P]['cg_u_mean'][:kk + 1])
        dots[P].set_data([t_all[kk]], [steps[P]['cg_u_mean'][kk]])
    head = r'core-shell NMC particle' if DIM == 2 else r'3D particle, mid-plane cut $z = L/2$'
    title.set_text(head + rf', 1C lithiation:\quad $t = {t * T_SCALE:4.0f}$\,s, \quad '
                   rf'SOL $= {ref["sol"][k]:.2f}$, \quad broken fraction $= {ref["a_c"][k]:.3f}$')


def frame():
    fig.canvas.draw()
    return np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()


os.makedirs(os.path.join(HERE, 'figures', 'slides'), exist_ok=True)
if os.environ.get('TEST_FRAME'):                               # debug: one frame only
    draw(int(os.environ['TEST_FRAME']))
    fig.savefig(os.environ.get('TEST_OUT', 'test_frame.png'), dpi=DPI)
    sys.exit()
out = os.path.join(HERE, 'figures', 'slides', f'movie_battery_{DIM}D_N{N}' + ('_cut' if DIM == 3 else '') + ('_mesh' if MESH_PANEL else '') + '.mp4')
writer = imageio_ffmpeg.write_frames(out, (W, H), fps=FPS, quality=None, codec='libx264', pix_fmt_out='yuv420p',
                                     macro_block_size=8, output_params=ENC)
writer.send(None)
for k in range(n_steps):
    draw(k)
    img = frame()
    for _ in range(PER_STEP):
        writer.send(img)
    if k % 25 == 0:
        print(f'step {k}/{n_steps}', flush=True)
for _ in range(int(HOLD_END * FPS)):
    writer.send(img)
writer.close()
fig.savefig(out.replace('.mp4', '_final.png'), dpi=DPI)
print('saved', out)
