"""Video of the Fracture-Bench Dataset 2 periodic analogue on NEMO (fb_dataset2_pfczm.py), 512 x 1024, dual run:
damage on the exact pixels (1:1, inferno; hard inclusion tinted blue, soft one green) | macro stress vs load step |
CG iterations per elastic solve per load step, Green vs dual x2 (Green run stopped at step 42, its time budget).
Frames: the stored damage every 2nd load step (exp_data_nemo/<run>_frames/d_XXXX.npy).
    python make_fb_dataset2_video.py  ->  figures/fb_dataset2_N512_dual.mp4 (+ _final.png)"""
import glob
import os

import imageio_ffmpeg
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.realpath(__file__))
E = os.path.join(HERE, 'exp_data_nemo')
RUN = 'FB2_512x1024_{}_hard-soft_tip1.5-4.2_linear_spectral_ft154_k0.0001'
fac = np.load(os.path.join(E, RUN.format('dual') + '_phase.npy'))          # 10 hard, 0.1 soft, 1 matrix
frames = sorted(glob.glob(os.path.join(E, RUN.format('dual') + '_frames', 'd_*.npy')))
fsteps = [int(os.path.basename(f)[2:6]) for f in frames]
rd = lambda f: np.genfromtxt(f, delimiter=',', names=True, ndmin=1)
st = rd(os.path.join(E, RUN.format('dual') + '_steps.csv'))
cg = {}
for P, f in (('Green', 1), ('dual', 2)):
    it = rd(os.path.join(E, RUN.format(P) + '_iterations.csv'))
    s = np.unique(it['step'])
    cg[P] = (s, np.array([f * it['cg_u'][it['step'] == k].mean() for k in s]))
cmap = plt.get_cmap('inferno')
tint = {10.0: np.array([0.35, 0.55, 0.95]), 0.1: np.array([0.30, 0.80, 0.45])}


def rgb(d):
    img = cmap(np.clip(d, 0, 1))[..., :3]
    for v, c in tint.items():
        m = np.isclose(fac, v)
        img[m] = 0.55 * img[m] + 0.45 * c
    return np.transpose(img, (1, 0, 2))                              # x horizontal, y vertical


NX, NY = fac.shape
W, H, DPI = 1920, 1200, 100
plt.rcParams.update({'font.family': 'serif', 'font.size': 15, 'mathtext.fontset': 'cm'})
fig = plt.figure(figsize=(W / DPI, H / DPI), dpi=DPI, facecolor='white')
px = lambda x0, y0, w, h: [x0 / W, y0 / H, w / W, h / H]
a0 = fig.add_axes(px(60, (H - NY) // 2 - 20, NX, NY))
a0.set_xticks([]), a0.set_yticks([])
im = a0.imshow(rgb(np.load(frames[0]).astype(float)), origin='lower', interpolation='nearest')
info = a0.text(0.03, 0.985, '', transform=a0.transAxes, ha='left', va='top', color='white', fontsize=16)
fig.text(60 / W, (H - 30) / H, 'Fracture-Bench Dataset 2 analogue: PF-CZM, $512 \\times 1024$, $b/h = 5.1$', fontsize=19, va='top')
fig.text(60 / W, 22 / H, 'blue: hard inclusion ($10\\times E$), green: soft ($0.1\\times E$); pre-crack from the left edge',
         fontsize=13, va='bottom')
a1 = fig.add_axes(px(720, 700, 1120, 400))
a2 = fig.add_axes(px(720, 150, 1120, 430))
ls, = a1.plot([], [], 'k-', lw=2)
dot, = a1.plot([], [], 'o', color='tab:red', ms=8)
a1.set(xlim=(0, len(st['step']) + 1), ylim=(0, 1.08 * st['stress'].max() * 377e3), ylabel='macro stress [MPa]')
lc = {P: a2.plot([], [], '-', color=c, lw=2, label=lab)[0] for P, c, lab in
      (('Green', 'tab:green', 'Green'), ('dual', 'tab:red', r'dual ($\times 2$)'))}
a2.set(xlim=(0, len(st['step']) + 1), ylim=(0, 1.05 * max(v.max() for _, v in cg.values())), xlabel='load step',
       ylabel='CG iterations per solve')
a2.legend(loc='upper right', fontsize=13)
a2.text(43, 0.93 * a2.get_ylim()[1], 'Green run stopped at step 42\n(time budget)', fontsize=11, color='tab:green', va='top')
for a in (a1, a2):
    a.grid(alpha=0.3)


def frame(j):
    k = fsteps[j]
    im.set_data(rgb(np.load(frames[j]).astype(float)))
    info.set_text(f'load step {k + 1}')
    m = st['step'] <= k
    ls.set_data(st['step'][m] + 1, st['stress'][m] * 377e3)
    dot.set_data([k + 1], [st['stress'][m][-1] * 377e3])
    for P in cg:
        s, v = cg[P]
        lc[P].set_data(s[s <= k] + 1, v[s <= k])
    fig.canvas.draw()
    return np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()


out = os.path.join(HERE, 'figures', 'fb_dataset2_N512_dual')
w = imageio_ffmpeg.write_frames(out + '.mp4', (W, H), fps=6, quality=9, codec='libx264', pix_fmt_out='yuv420p',
                                macro_block_size=8)
w.send(None)
for j in list(range(len(frames))) + [len(frames) - 1] * 12:
    img = frame(j)
    w.send(img)
w.close()
plt.imsave(out + '_final.png', img)
print('wrote', out + '.mp4', len(frames), 'frames')
