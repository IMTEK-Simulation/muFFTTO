"""Slide video of the 2D 1024^2 NEMO fatigue run (Carrara 2020 AT2 fatigue, glass beads, pre-crack), see README.md.

    python make_FAT_slide_video.py [N]          (default N = 1024)
    -> figures/FAT_slide_video_N<N>.mp4 and _final.png (poster = last frame)

left : damage d of the dual run (identical solution for all preconditioners) on its exact pixels (each pixel = 2 x 2
       video pixels at 1024^2), glass beads tinted; snapshots are stored every 5th cycle (NC_EVERY=5) + the final
       state, the field shown is the latest stored snapshot <= the current cycle.
right: top    CG iterations per elastic solve vs cycle (mean over the staggered solves of the cycle),
       bottom cumulative wall time (elastic + damage solves) [h], Green / Green-Jacobi / dual, up to the current cycle.
"""
import os
import sys

import imageio_ffmpeg
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import netCDF4
import numpy as np

HERE = os.path.dirname(os.path.realpath(__file__))
N = int(sys.argv[1]) if len(sys.argv) > 1 else 1024
TAG = '_incl11.6667_l0.015625_k0.0001_nemo2D_precrack_asymptotic_amp0.002_cyc1500'
E_MPA = 6000.0
PRECS = [('Green', 'Green', 'tab:green'), ('Green_Jacobi', 'Green-Jacobi', 'tab:blue'), ('dual', r'dual ($\times 2$)', 'tab:red')]
W_PX, H_PX, DPI = 4232 if os.environ.get('MESH_PANEL') else 3600, 2160, 100     # MESH_PANEL: damage half / graphs half
FPS = 10
HOLD = 25                     # extra frames on the last cycle
plt.rcParams.update({'text.usetex': True, 'font.family': 'serif', 'font.size': 40, 'axes.labelsize': 42,
                     'xtick.labelsize': 36, 'ytick.labelsize': 36, 'legend.fontsize': 38, 'axes.linewidth': 1.6,
                     'xtick.major.width': 1.6, 'ytick.major.width': 1.6, 'xtick.major.size': 8,
                     'ytick.major.size': 8})
BIG = os.environ.get('MESH_PANEL') is not None         # slide layout: label sizes for >= 8 pt on the slide
if BIG:
    plt.rcParams.update({'font.size': 62, 'axes.labelsize': 66, 'xtick.labelsize': 62, 'ytick.labelsize': 62})

base = os.path.join(HERE, 'exp_data', f'FAT_N{N}_dual{TAG}')
nc = netCDF4.Dataset(base + '.nc')
snap_steps = list(np.asarray(nc['step'][:]).astype(int))
snaps = [np.asarray(nc['damage'][k, 0]) for k in range(len(snap_steps))]
fin = np.load(base + '_damage.npz')
if int(fin['steps'][-1]) > snap_steps[-1]:
    snap_steps.append(int(fin['steps'][-1]))
    snaps.append(np.asarray(fin['damage'][-1]))
bead = fin['phase'] < 0.5                        # phase = 1 matrix, 0 glass bead
snap_steps = np.array(snap_steps)

runs = {}
for P, label, c in PRECS:
    d = np.genfromtxt(os.path.join(HERE, 'exp_data', f'FAT_N{N}_{P}{TAG}_cycles.csv'), delimiter=',', names=True)
    runs[P] = dict(cyc=d['step'] + 1, cg=d['cg_u_step'] / d['staggered_its'],
                   t=np.cumsum(d['time_u_step'] + d['time_d_step']) / 3600, sigma=d['sigma'] * E_MPA)
ncyc = len(runs['dual']['cyc'])

cmap = plt.get_cmap('inferno')
bead_rgb = np.array([0.55, 0.70, 0.88])


def rgb(dmg):
    img = cmap(np.clip(dmg, 0, 1))[..., :3]
    img[bead] = 0.45 * img[bead] + 0.55 * bead_rgb
    return np.transpose(img, (1, 0, 2))         # x horizontal, y vertical (origin lower)


fig = plt.figure(figsize=(W_PX / DPI, H_PX / DPI), dpi=DPI)
fig.patch.set_facecolor('white')
px = lambda x0, y0, w, h: [x0 / W_PX, y0 / H_PX, w / W_PX, h / H_PX]
S = 2048 // N * N if N <= 1024 else N                 # integer magnification: exact pixels
# MESH_PANEL (env, PNG of fatigue_cg_vs_grid_2D): layout damage left, mesh-study graph upper right, wall time lower right
MESH_PANEL = os.environ.get('MESH_PANEL')
X_D = 330 if MESH_PANEL else 40
ax_d = fig.add_axes(px(X_D, 56, S, S))
im = ax_d.imshow(rgb(snaps[0]), origin='lower', interpolation='nearest', resample=False, extent=(0, 1, 0, 1))
ax_d.set_xticks([]), ax_d.set_yticks([])
ax_cb = fig.add_axes(px(270 if MESH_PANEL else X_D + S + 30, 56 + S * 0.15, 40, S * 0.7))
cb = fig.colorbar(plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 1)), cax=ax_cb)
cb.set_ticks([0, 0.5, 1]), cb.set_ticklabels(['0', '0.5', '1'])
cb.set_label('damage $d$')
if MESH_PANEL:
    ax_cb.yaxis.set_ticks_position('left'), ax_cb.yaxis.set_label_position('left')
ax_cg = fig.add_axes(px(2470, 1220, 1080, 760))
ax_t = fig.add_axes(px(2470, 200, 1080, 760))
head = fig.text(2470 / W_PX, (1035 if MESH_PANEL else 2085) / H_PX, '', ha='left', va='center', fontsize=44)
if MESH_PANEL:
    ax_cg.set_visible(False)
    ax_t.set_position(px(2820, 260, 1200, 650))
    ax_m = fig.add_axes(px(2420, 1090, 1800, 1040))
    ax_m.imshow(plt.imread(MESH_PANEL), interpolation='antialiased')
    ax_m.set_axis_off()
    head.remove()
    head = ax_t.text(0.04, 0.94, '', transform=ax_t.transAxes, ha='left', va='top', fontsize=66)   # cycle counter inside the graph
ax_d.text(0.015, 0.985, f'${N}^2$, $\\ell/h = {N // 64}$', transform=ax_d.transAxes, ha='left', va='top',
          color='white', fontsize=66 if BIG else 44)
lines = {}
for P, label, c in PRECS:
    lines[P] = (ax_cg.plot([], [], '-', color=c, lw=3.5, label=label)[0],
                ax_cg.plot([], [], 'o', color=c, ms=14)[0],
                ax_t.plot([], [], '-', color=c, lw=3.5)[0],
                ax_t.plot([], [], 'o', color=c, ms=14)[0],
                ax_t.text(0, 0, '', color=c, fontsize=62 if BIG else 36, ha='left', va='center'))
runs['dual']['cg'] = 2 * runs['dual']['cg']                     # dual iterations counted x2 (cost); wall time stays measured
ymax_cg = max(r['cg'].max() for r in runs.values())
tmax = max(r['t'][-1] for r in runs.values())
for ax in (ax_cg, ax_t):
    ax.set_xlim(0, ncyc)
    ax.grid(alpha=0.3)
ax_cg.set_ylim(0, 1.05 * ymax_cg)
ax_t.set_ylim(0, 1.12 * tmax)
ax_t.set_xlim(0, ncyc * (1.36 if BIG else 1.18))
ax_cg.set_xlim(0, ncyc * 1.18)
ax_cg.set_ylabel(r'CG per elastic solve (dual $\times 2$)')
ax_t.set_ylabel('wall time [h]')
ax_t.set_xlabel('load cycle')
ax_cg.set_xticklabels([])
for ax in (ax_cg, ax_t):
    ax.set_xticks([0, 50, 100, 150])
ax_cg.legend(loc='upper left', frameon=False, handlelength=1.4)
if BIG:
    ax_t.set_xlim(0, 250), ax_t.set_xticks([0, 50, 100, 150, 200, 250])
    ax_t.set_ylim(0, 15), ax_t.set_yticks([0, 5, 10, 15])


def frame(n):
    """state after cycle n (1-based)"""
    k = int(np.searchsorted(snap_steps, n - 1, side='right') - 1)
    im.set_data(rgb(snaps[k]))
    s = runs['dual']['sigma'][n - 1]
    head.set_text(f'cycle {n:3d}' if MESH_PANEL else f'cycle {n:3d}, \\quad peak stress {s:.1f} MPa')
    for P, label, c in PRECS:
        r = runs[P]
        l1, m1, l2, m2, txt = lines[P]
        l1.set_data(r['cyc'][:n], r['cg'][:n])
        m1.set_data([r['cyc'][n - 1]], [r['cg'][n - 1]])
        l2.set_data(r['cyc'][:n], r['t'][:n])
        m2.set_data([r['cyc'][n - 1]], [r['t'][n - 1]])
        txt.set_text(f'{r["t"][n - 1]:.1f} h')
    ys = sorted(((runs[P]['t'][n - 1], P) for P, _, _ in PRECS))   # label positions, min. separation
    gap, prev = (0.17 if BIG else 0.085) * 1.12 * tmax, -1e9
    for y, P in ys:
        y = max(y, prev + gap)
        lines[P][4].set_position((runs[P]['cyc'][n - 1] + 4, y))
        prev = y


def grab():
    fig.canvas.draw()
    return np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()


out = os.path.join(HERE, 'figures', f'FAT_slide_video_N{N}{"_meshleft" if MESH_PANEL else ""}.mp4')
frames_list = list(range(1, ncyc + 1)) + [ncyc] * HOLD
if os.environ.get('TEST_FRAME'):
    frame(int(os.environ['TEST_FRAME']))
    fig.savefig(out.replace('.mp4', '_test.png'), dpi=DPI)
    print('test frame written')
    sys.exit()
w = imageio_ffmpeg.write_frames(out, (W_PX, H_PX), fps=FPS, quality=9, codec='libx264', pix_fmt_out='yuv420p',
                                macro_block_size=8)
w.send(None)
for i, n in enumerate(frames_list):
    if i == 0 or n != frames_list[i - 1]:
        frame(n)
        buf = grab()
    w.send(buf)
w.close()
fig.savefig(out.replace('.mp4', '_final.png'), dpi=DPI)
print('saved', out, len(frames_list), 'frames')
