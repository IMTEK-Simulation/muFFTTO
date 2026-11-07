"""
mp4 of the damage evolution of one fourel_nonlocal_damage.py run (exact pixels, no smoothing), next to the
macroscopic expansion tr E and the cumulative solver cost.

    python make_fourel_video.py exp_data/FOUREL_N128_Green_free_k0.0001      # -> figures/FOUREL_N128_Green_free_k0.0001.mp4
env: FPS (12)
"""
import os
import sys
import shutil
import subprocess
import imageio_ffmpeg
import tempfile
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

here = os.path.dirname(os.path.abspath(__file__))
base = sys.argv[1]
base = base if os.path.isabs(base) else os.path.join(here, base)
name = os.path.basename(base)
fps = int(os.environ.get('FPS', 12))
plt.rcParams.update({'font.size': 13})

st = np.load(base + '_steps.npz', allow_pickle=True)
sv = np.atleast_1d(np.genfromtxt(base + '_solves.csv', delimiter=',', names=True))
lab = st['label']
N = int(st['N'])
eps = 100 * st['eps_asr']
trE = 100 * (st['E'][:, 0] + st['E'][:, 1])
cum = np.cumsum([sv['linear_its'][sv['step'] == s].sum() + sv['unit_its'][sv['step'] == s].sum()
                 for s in range(len(eps))]) * (2.0 if str(st['solver']) == 'dual' else 1.0)
cmap_lab = ListedColormap(['#2b2b2b', '#9a9a9a', '#e8c547'])
tmp = tempfile.mkdtemp()
for k in range(len(eps)):
    phi = np.load(base + f'_damage/step_{k:03d}.npy')
    fig = plt.figure(figsize=(13, 6.2))
    a0 = fig.add_axes([0.02, 0.06, 0.46, 0.86])
    a0.imshow(lab.T, origin='lower', cmap=cmap_lab, interpolation='nearest', vmin=0, vmax=2, alpha=0.4)
    a0.imshow(np.ma.masked_less(phi.T, 0.05), origin='lower', cmap='Reds', interpolation='nearest', vmin=0, vmax=1)
    a0.set_xticks([])
    a0.set_yticks([])
    a0.set_title(f'damage, {N}$^2$, $\\varepsilon_{{ASR}}$ = {eps[k]:.3f} %')
    a1 = fig.add_axes([0.58, 0.58, 0.39, 0.34])
    a1.plot(eps, trE, color='0.75')
    a1.plot(eps[:k + 1], trE[:k + 1], color='tab:green' if 'Green' in name else 'black')
    a1.set_xlim(0, eps[-1])
    a1.set_ylabel(r'tr $\bar\varepsilon$ [%]')
    a1.grid(alpha=0.3)
    a2 = fig.add_axes([0.58, 0.1, 0.39, 0.34])
    a2.plot(eps, cum / 1e3, color='0.75')
    a2.plot(eps[:k + 1], cum[:k + 1] / 1e3, color='tab:green' if 'Green' in name else 'black')
    a2.set_xlim(0, eps[-1])
    a2.set_ylim(bottom=0)
    a2.set_xlabel(r'gel eigenstrain $\varepsilon_{ASR}$ [%]')
    a2.set_ylabel('cumulative cost [10$^3$ its]')
    a2.grid(alpha=0.3)
    fig.savefig(os.path.join(tmp, f'f{k:04d}.png'), dpi=110)
    plt.close(fig)
out = os.path.join(here, 'figures', name + '.mp4')
subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), '-y', '-loglevel', 'error', '-framerate', str(fps), '-i', os.path.join(tmp, 'f%04d.png'),
                '-vf', 'scale=trunc(iw/2)*2:trunc(ih/2)*2', '-pix_fmt', 'yuv420p', '-c:v', 'libx264', out], check=True)
shutil.rmtree(tmp)
print('->', out)
