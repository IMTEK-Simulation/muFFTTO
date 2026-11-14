"""
Sintering video with what the CG sees: density rho | stiffness C(rho)/C_Cu on a log scale (the coefficient field of the
elastic solve, contrast 1 / k_res) | CG iterations per elastic solve of Green, Green-Jacobi and dual over time
(same physics on all three; the snapshots are those of the dual run).
    python make_sintering_video_cg.py [N (128)] [RUN_TAG (W4_P50u_k0.0001)]
-> figures/sintering_N<N>_cg_all.mp4 (+ _final.png)
"""
import glob
import os
import sys

import imageio_ffmpeg
import matplotlib
import numpy as np

matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
N = int(sys.argv[1]) if len(sys.argv) > 1 else 128
TAG = sys.argv[2] if len(sys.argv) > 2 else 'W4_P50u_k0.0001'
PRE = [('Green', 'Green', 'tab:green'), ('Green_Jacobi', 'Green-Jacobi', 'tab:blue'), ('dual', 'dual', 'tab:red')]
plt.rcParams.update({'font.size': 14})

# SNAP_PREC (env, default dual): whose snapshots to show; OUT_TAG (env): added to the file name. Preconditioners with no
# steps file, or whose run has reached less than 50 us, are left out of the CG panel.
SNAP_PREC = os.environ.get('SNAP_PREC', 'dual')
OUT_TAG = os.environ.get('OUT_TAG', '')
steps = {}
for P, _, _ in list(PRE):
    f = os.path.join(HERE, 'exp_data', f'SINT_N{N}_{P}_{TAG}_steps.npz')
    if os.path.exists(f) and float(np.load(f, allow_pickle=True)['t'][-1]) >= 50e-6:
        steps[P] = np.load(f, allow_pickle=True)
PRE = [p for p in PRE if p[0] in steps]
k_res = float(steps[SNAP_PREC]['k_res'])
width, h = float(steps[SNAP_PREC]['width']), float(steps[SNAP_PREC]['h'])
snaps = sorted(glob.glob(os.path.join(HERE, 'exp_data', f'SINT_N{N}_{SNAP_PREC}_{TAG}_snaps', 'step_*.npz')))
cell = N * h


def hfun(r):
    r = np.clip(r, 0, 1)
    return r ** 3 * (10 - 15 * r + 6 * r ** 2)


fig = plt.figure(figsize=(19.2, 6.4), dpi=100)
gs = fig.add_gridspec(1, 3, width_ratios=[1, 1.0, 1.35], left=0.035, right=0.985, top=0.83, bottom=0.12, wspace=0.38)
a, b, c = (fig.add_subplot(gs[0, j]) for j in range(3))
ext = (0, cell, 0, cell)
z0 = np.load(snaps[0])
im_r = a.imshow(z0['rho'].T, cmap='gray_r', vmin=0, vmax=1, origin='lower', interpolation='nearest', extent=ext)
a.set(title='density $\\rho$ (black = Cu)', xlabel='x [nm]', ylabel='y [nm]')
logC = lambda r: np.log10(hfun(r) * (1 - k_res) + k_res)
im_c = b.imshow(logC(z0['rho']).T, cmap='viridis', vmin=np.log10(k_res), vmax=0, origin='lower',
                interpolation='nearest', extent=ext)
b.set(title=f'what CG sees: $\\mathbb{{C}}(\\rho)/\\mathbb{{C}}_{{Cu}}$, contrast $10^{{{int(round(-np.log10(k_res)))}}}$', xlabel='x [nm]')
cb = fig.colorbar(im_c, ax=b, fraction=0.046, pad=0.03)
ticks = list(range(int(np.log10(k_res)), 1))
cb.set_ticks(ticks), cb.set_ticklabels([f'$10^{{{t}}}$' if t else '1' for t in ticks])
lines, dots = {}, {}
tmax = max(float(steps[P]['t'][-1]) for P, _, _ in PRE) * 1e6
ymax = 0
for P, lab, col in PRE:
    t, cg = steps[P]['t'] * 1e6, steps[P]['cg_mean']
    ymax = max(ymax, np.nanmax(cg))
    c.plot(t, cg, color=col, lw=1, alpha=0.25)
    lines[P], = c.plot([], [], color=col, lw=2.5, label=f'{lab} (mean {np.nanmean(cg):.0f})')
    dots[P], = c.plot([], [], 'o', color=col, ms=8)
c.set_xlim(0, tmax), c.set_ylim(0, 1.1 * ymax)
c.set(xlabel='time [$\\mu$s]', ylabel='CG iterations per elastic solve', title='CG per elastic solve (dual costs 2x)')
c.grid(alpha=0.3)
c.legend(loc='upper right', fontsize=13)
sup = fig.suptitle('', fontsize=17)

exe = imageio_ffmpeg.get_ffmpeg_exe()
out = os.path.join(HERE, 'figures', f'sintering_N{N}{OUT_TAG}_cg_all.mp4')
W, H = fig.canvas.get_width_height()
writer = imageio_ffmpeg.write_frames(out, (W, H), fps=10, codec='libx264', pix_fmt_out='yuv420p', quality=8,
                                     macro_block_size=8)
writer.send(None)
for f in snaps:
    z = np.load(f)
    tt = float(z['t']) * 1e6
    im_r.set_data(z['rho'].T)
    im_c.set_data(logC(z['rho']).T)
    for P, _, _ in PRE:
        t, cg = steps[P]['t'] * 1e6, steps[P]['cg_mean']
        m = t <= tt + 1e-9
        lines[P].set_data(t[m], cg[m])
        if m.any():
            dots[P].set_data([t[m][-1]], [cg[m][-1]])
    sup.set_text(f'Cu powder sintering, {N}$^2$ (h = {h:g} nm, surface width {width:g} nm = {width / h:g} h), '
                 f'50 MPa, t = {tt:6.1f} $\\mu$s')
    fig.canvas.draw()
    writer.send(np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy())
writer.close()
fig.savefig(out.replace('.mp4', '_final.png'), dpi=100)
print('wrote', out)
