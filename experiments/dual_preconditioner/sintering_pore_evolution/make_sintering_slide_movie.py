"""
Slide movie of the sintering run (density rho, black = Cu, white = pore), high resolution, LaTeX title:
256^2, W = 10 nm (Seiz et al. 2024), dual run's snapshots (same physics for all preconditioners).
    python make_sintering_slide_movie.py [N (256)] [TAG]  ->  talk movies/movie_sintering_N<N>.mp4 (+ _final.png)
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
N = int(sys.argv[1]) if len(sys.argv) > 1 else 256
TAG = sys.argv[2] if len(sys.argv) > 2 else 'W10_P50u_k0.0001_W10'
OUT = os.path.expanduser(f'~/Projects/Talks/Slides-Green-Jacobi-topology-optimization-NANTES/movies/movie_sintering_N{N}')
# GRAINS=1: each grain its own colour, faded to white (pore) with rho -> _grains; GRAINS=muted: dark tones, only tinted
# per grain (between the black-and-white density and the full colours) -> _grains_muted
GRAINS = os.environ.get('GRAINS') in ('1', 'muted')
MUTED = os.environ.get('GRAINS') == 'muted'
if GRAINS:
    OUT += '_grains_muted' if MUTED else '_grains'
snaps = sorted(glob.glob(os.path.join(HERE, 'exp_data', f'SINT_N{N}_dual_{TAG}_snaps', 'step_*.npz')))
plt.rcParams.update({'text.usetex': True, 'font.family': 'serif', 'font.size': 64})
S = 1600
fig = plt.figure(figsize=(S / 100, (S + 220) / 100), dpi=100, facecolor='white')
ax = fig.add_axes([0, 0, 1, S / (S + 220)])
ax.set_axis_off()


def image(z):
    if not GRAINS:
        return z['rho'].T
    r = np.clip(z['rho'].T, 0, 1)[..., None]
    c = plt.cm.tab10(z['grains'].T % 10)[..., :3]
    return r * (0.4 * c + 0.6 * 0.22 if MUTED else c) + 1 - r


z = np.load(snaps[0])
im = ax.imshow(image(z), cmap='gray_r', vmin=0, vmax=1, origin='lower', interpolation='nearest')
title = fig.text(0.5, 1 - 110 / (S + 220), '', ha='center', va='center')
W, H = fig.canvas.get_width_height()
wr = imageio_ffmpeg.write_frames(OUT + '.mp4', (W, H), fps=12, codec='libx264', pix_fmt_out='yuv420p', quality=9,
                                 macro_block_size=8)
wr.send(None)
for f in snaps:
    z = np.load(f)
    im.set_data(image(z))
    title.set_text(f'Cu powder sintering, $t = {float(z["t"]) * 1e6:5.0f}\\,\\mu$s')
    fig.canvas.draw()
    wr.send(np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy())
wr.close()
fig.savefig(OUT + '_final.png', dpi=100)
print('wrote', OUT + '.mp4', W, H)
