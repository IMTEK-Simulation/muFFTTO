"""
Video of density rho and grain map (rho > 0.5) from the snapshots of one run, exact pixels (nearest), ffmpeg mp4.
    python make_sintering_video.py [N (128)] [PRECOND (dual)] [RUN_TAG (W4_P50u_k0.0001)]
-> figures/sintering_N<N>_<precond>.mp4
"""
import os
import sys
import glob
import shutil
import subprocess
import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

here = os.path.dirname(os.path.abspath(__file__))
N = int(sys.argv[1]) if len(sys.argv) > 1 else 128
prec = sys.argv[2] if len(sys.argv) > 2 else 'dual'
tag = sys.argv[3] if len(sys.argv) > 3 else 'W4_P50u_k0.0001'
run = f'SINT_N{N}_{prec}_{tag}'
snaps = sorted(glob.glob(os.path.join(here, 'exp_data', run + '_snaps', 'step_*.npz')))
steps = np.load(os.path.join(here, 'exp_data', run + '_steps.npz'), allow_pickle=True)
frame_dir = os.path.join(here, 'figures', run + '_frames')
shutil.rmtree(frame_dir, ignore_errors=True)
os.makedirs(frame_dir)
plt.rcParams.update({'font.size': 13})
cmap = ListedColormap(np.vstack([[1, 1, 1, 1], plt.cm.tab10(np.arange(10))]))
t_all = np.array(steps['t']) * 1e6
for k, f in enumerate(snaps):
    s = np.load(f)
    t = float(s['t']) * 1e6
    fig = plt.figure(figsize=(15, 5.8))
    ax = [fig.add_axes([0.03, 0.10, 0.27, 0.72]), fig.add_axes([0.33, 0.10, 0.27, 0.72]),
          fig.add_axes([0.69, 0.12, 0.29, 0.68])]
    ext = [0, 128, 0, 128]
    ax[0].imshow(s['rho'].T, origin='lower', cmap='gray_r', vmin=0, vmax=1, interpolation='nearest', extent=ext)
    ax[0].set_title('density rho (black = Cu)')
    ax[1].imshow(np.where(s['rho'] > 0.5, s['grains'] + 1, 0).T, origin='lower', cmap=cmap, vmin=-0.5, vmax=10.5,
                 interpolation='nearest', extent=ext)
    ax[1].set_title('grains (rho > 0.5)')
    for a in ax[:2]:
        a.set_xlabel('x [nm]')
        a.set_ylabel('y [nm]')
    m = t_all <= t + 1e-9
    ax[2].plot(t_all, steps['cg_mean'], color='0.8', lw=1)
    ax[2].plot(t_all[m], np.array(steps['cg_mean'])[m], color='tab:red', lw=2)
    ax[2].set(xlabel='time [$\\mu$s]', ylabel='CG iterations per elastic solve', xlim=(0, t_all[-1]),
              ylim=(0, 1.1 * np.max(steps['cg_mean'])), title=f'{prec} preconditioner')
    ax[2].grid(alpha=0.3)
    fig.suptitle(y=0.97, t=f'Cu powder sintering, {N}$^2$ (h = {128 / N:g} nm), t = {t:6.1f} $\\mu$s')
    fig.savefig(os.path.join(frame_dir, f'frame_{k:04d}.png'), dpi=100)
    plt.close(fig)
out = os.path.join(here, 'figures', f'sintering_N{N}_{prec}.mp4')
import imageio_ffmpeg
subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), '-y', '-loglevel', 'error', '-framerate', '10', '-i', os.path.join(frame_dir, 'frame_%04d.png'),
                '-vf', 'pad=ceil(iw/2)*2:ceil(ih/2)*2', '-c:v', 'libx264', '-pix_fmt', 'yuv420p', out], check=True)
shutil.rmtree(frame_dir)
print('written', out)
