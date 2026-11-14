"""Video of a Fracture-Bench Dataset 1 run (fb_dataset1_AT2.py, 128^2): damage on its exact pixels (5x magnified, hole
and notches white) | macro stress | CG per elastic solve (Green, dual x2) up to the step.
    python make_fb_dataset1_video.py             -> figures/fb_dataset1_local.mp4  (laptop run, their 44 steps)
    SRC=nemo python make_fb_dataset1_video.py    -> figures/fb_dataset1_nemo_N128.mp4 (exp_data_nemo/, run to the break)"""
import os
import imageio_ffmpeg
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import netCDF4
import numpy as np

HERE = os.path.dirname(os.path.realpath(__file__))
TAG = '_fb_hole_holes0.001_l0.0068182_k0.001_biax'
SRC = os.environ.get('SRC', 'local')
R = {}
for P in ('Green', 'dual'):
    with netCDF4.Dataset(os.path.join(HERE, 'exp_data_nemo' if SRC == 'nemo' else 'exp_data', f'AT2_N128_{P}{TAG}.nc')) as ds:
        R[P] = {k: np.asarray(ds[k][:]) for k in ('damage', 'load', 'macro_stress', 'staggered_iteration', 'cg_u_step')}
N, M = 128, 5                                         # exact pixels: each pixel = M x M video pixels
h = 1.0 / N
c = np.stack(np.meshgrid(np.arange(N) * h, np.arange(N) * h, indexing='ij'))
x, y = c[0] - 0.5, c[1] - 0.5
hole = x ** 2 + y ** 2 < (0.5 / 2.2) ** 2
for k in range(8):
    th = k * np.pi / 4
    s, t = x * np.cos(th) + y * np.sin(th), -x * np.sin(th) + y * np.cos(th)
    hole |= (s >= 0.5 / 2.2 - 2 * h) & (s <= 0.5 / 2.2 + 0.05 / 2.2) & (np.abs(t) < 0.75 * h)
cmap = plt.get_cmap('inferno_r')
W, H, DPI = 1920, 800, 100
fig = plt.figure(figsize=(W / DPI, H / DPI), dpi=DPI)
px = lambda x0, y0, w, hh: [x0 / W, y0 / H, w / W, hh / H]
S = N * M
a0 = fig.add_axes(px(70, (H - S) // 2 - 20, S, S))
a0.set_xticks([]), a0.set_yticks([])
im = a0.imshow(np.zeros((N, N, 3)), origin='lower', interpolation='nearest')
ttl = a0.set_title('', fontsize=15)
a1 = fig.add_axes(px(820, 110, 470, 600))
a2 = fig.add_axes(px(1400, 110, 470, 600))
nst = len(R['dual']['load'])
L = {}
for P, col, f in (('Green', 'tab:green', 1), ('dual', 'tab:red', 2)):
    L[P, 1], = a1.plot([], [], '-o' if P == 'Green' else '--', color=col, ms=3, label=P)
    L[P, 2], = a2.plot([], [], '-o', color=col, ms=3, label=P + (r' ($\times 2$)' if f == 2 else ''))
a1.set(xlim=(0, R['dual']['load'][-1] * 1e3 * 1.03), ylim=(0, R['dual']['macro_stress'][:, 0, 0].max() * 1e3 * 1.08),
       xlabel=r'biaxial strain $\bar\varepsilon$ [$10^{-3}$]', ylabel=r'$\bar\sigma_{xx}/\mu$ [$10^{-3}$]', title='macro stress')
a2.set(xlim=(0, nst + 1), ylim=(0, 1.08 * max(f * (R[P]['cg_u_step'] / R[P]['staggered_iteration']).max() for P, f in (('Green', 1), ('dual', 2)))), xlabel='load step', ylabel='CG per elastic solve', title='elastic CG (dual x2)')
for a in (a1, a2):
    a.grid(alpha=0.3), a.legend(loc='upper left')
fig.suptitle('Fracture-Bench Dataset 1, periodic analogue: 128$^2$, $\\ell/h = 0.87$, ' + ('NEMO run to the break (load steps of $10^{-4}$)' if SRC == 'nemo' else 'local run, their 44 load steps'),
             fontsize=15)


def frame(j):
    d = R['dual']['damage'][j, 0]
    img = cmap(np.clip(d, 0, 1))[..., :3]
    img[hole] = 1.0
    im.set_data(np.transpose(img, (1, 0, 2)))
    ttl.set_text(f'damage $d$, step {j + 1}, max $d$ = {d.max():.2f}')
    for P, f in (('Green', 1), ('dual', 2)):
        r = R[P]
        L[P, 1].set_data(r['load'][:j + 1] * 1e3, r['macro_stress'][:j + 1, 0, 0] * 1e3)
        L[P, 2].set_data(np.arange(1, j + 2), f * r['cg_u_step'][:j + 1] / r['staggered_iteration'][:j + 1])
    fig.canvas.draw()
    return np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()


out = os.path.join(HERE, 'figures', 'fb_dataset1_nemo_N128.mp4' if SRC == 'nemo' else 'fb_dataset1_local.mp4')
w = imageio_ffmpeg.write_frames(out, (W, H), fps=6, quality=9, codec='libx264', pix_fmt_out='yuv420p', macro_block_size=8)
w.send(None)
for j in list(range(nst)) + [nst - 1] * 12:
    w.send(frame(j))
w.close()
print(out)
