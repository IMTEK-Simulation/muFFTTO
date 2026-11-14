"""Video of the converged 2-grain chain series (t_r = 1e-7 s, dt = 1e-8 s, N = 96, W = 4 nm), P = 0 / 25 / 50 / 100 MPa.
Top: density in PHYSICAL coordinates (y = s(t) Y, cell contracts about its centre), grains coloured, exact pixels.
Bottom: chain period L(t) for all pressures with a time marker.
Writes videos/chain_pressure.mp4 and figures/chain_video_last_frame.png"""
import glob, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
import imageio_ffmpeg
matplotlib.rcParams['animation.ffmpeg_path'] = imageio_ffmpeg.get_ffmpeg_exe()   # no system ffmpeg

here = os.path.dirname(os.path.abspath(__file__))
L0, N = 90.24, 96
ps = [0, 25, 50, 100]
line_col = ['#111111', '#1f77b4', '#d95f02', '#7b3294']
line_sty = [':', '--', '-.', '-']
g_col = np.array([[0.85, 0.37, 0.01], [0.46, 0.44, 0.70]])   # grain 0 / grain 1


def files(p):
    return sorted(glob.glob(os.path.join(here, f'exp_data/SINT_N96_Green_W4_P{p}u_k0.0001_chain_P{p}_tau1e-7_dt1e-8_snaps/step_*.npz')),
                  key=lambda f: int(os.path.basename(f)[5:-4]))


def rgb(d):
    r = np.clip(d['rho'].astype(float), 0, 1)
    e = np.clip(d['etas'].astype(float), 0, 1)
    w = e.sum(0) + 1e-12
    c = (e[0, ..., None] * g_col[0] + e[1, ..., None] * g_col[1]) / w[..., None]
    img = r[..., None] * c + (1 - r[..., None]) * 1.0
    return np.transpose(img, (1, 0, 2))                              # [y, x, rgb]


runs = {p: files(p) for p in ps}
n_frames = min(len(v) for v in runs.values())
curves = {}
for p in ps:
    T, L = [], []
    for f in runs[p]:
        d = np.load(f); T.append(float(d['t']) * 1e6); L.append(float(d['s']) * L0)
    curves[p] = (np.array(T), np.array(L))

plt.rcParams.update({'font.size': 13})
fig = plt.figure(figsize=(15, 8.2), constrained_layout=True)
gs = fig.add_gridspec(2, 4, height_ratios=[1.35, 1])
ax_im = [fig.add_subplot(gs[0, k]) for k in range(4)]
ax_L = fig.add_subplot(gs[1, :])
ims, titles = [], []
for k, p in enumerate(ps):
    a = ax_im[k]
    ims.append(a.imshow(np.ones((N, N, 3)), origin='lower', interpolation='nearest', extent=[0, L0, 0, L0]))
    a.set_xlim(0, L0); a.set_ylim(0, L0); a.set_aspect('equal')
    a.set_xlabel('x [nm]'); a.set_ylabel('y [nm]')
    titles.append(a.set_title(''))
for p, c, l in zip(ps, line_col, line_sty):
    ax_L.plot(*curves[p], color=c, ls=l, lw=2, label=f'{p} MPa')
marker = ax_L.axvline(0, color='0.4', lw=1.5)
ax_L.set_xlabel(r'time [$\mu$s]'); ax_L.set_ylabel('chain period L [nm]')
ax_L.set_xlim(0, curves[0][0][-1]); ax_L.set_ylim(40, 95); ax_L.grid(alpha=.3)
ax_L.ticklabel_format(style='plain', useOffset=False)
ax_L.legend(loc='lower left', ncol=4)
fig.suptitle('Pressure-assisted sintering, 2-grain periodic chain (load along y, grains coloured, white = pore)')


def draw(i):
    for k, p in enumerate(ps):
        d = np.load(runs[p][i]); s = float(d['s'])
        lo, hi = L0 / 2 * (1 - s), L0 / 2 * (1 + s)                 # cell contracts about its centre
        ims[k].set_data(rgb(d)); ims[k].set_extent([0, L0, lo, hi])
        titles[k].set_text(f'{p} MPa, L = {s * L0:.1f} nm')
    t = curves[0][0][i]
    marker.set_xdata([t, t])
    ax_L.set_title(f't = {t:.0f} $\\mu$s')


os.makedirs(os.path.join(here, 'videos'), exist_ok=True)
wr = FFMpegWriter(fps=6, bitrate=4000)
with wr.saving(fig, os.path.join(here, 'videos', 'chain_pressure.mp4'), dpi=110):
    for i in range(n_frames):
        draw(i); wr.grab_frame()
    for _ in range(12):                                              # hold the last frame 2 s
        wr.grab_frame()
fig.savefig(os.path.join(here, 'figures', 'chain_video_last_frame.png'), dpi=150)
print('frames', n_frames)
