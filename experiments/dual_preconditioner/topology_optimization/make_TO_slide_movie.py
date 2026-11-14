"""
Slide movie of a TO run in the style of the talk movie movies/movie_w_0.3_short.mp4 (black solid / white void,
periodic copies, serif title 'Iteration: N', 980 x 620 aspect, 7 s at 30 fps), here for a sharp design:
1024^2, contrast 1e4, eta = 4h = 1/256, dual (NEMO eta / contrast study; history every 10 L-BFGS iterations, the
frames in between are linear blends of consecutive saved iterates).
    python make_TO_slide_movie.py [history.nc] [out stem]
    -> <out>.mp4, <out>_initial.png, <out>_final.png  (default: talk folder movies/movie_TO_1024_eta4h_dual)
env: W, H (frame size, default 3648 x 2312), FS (title font size, 74), TWOLINE=1 (title on two lines, for narrow
     column videos), TOP (height fraction of the image area, 0.885), TITLE=iteration (one line 'Optimizer step: N',
     as the slide-6 movie movie_TO_no_pf_N64_random.mp4)
"""
import os
import sys

import imageio_ffmpeg
import matplotlib
import netCDF4
import numpy as np

matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.realpath(__file__))
HIST = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
    HERE, 'data', 'example_2D_elasticity_TO', 'nemo',
    'N1024_soft0.0001_energy-warm_dual_eta_0.00390625_w_5.0_l_bfgs_history.nc')
OUT = sys.argv[2] if len(sys.argv) > 2 else os.path.expanduser(
    '~/Projects/Talks/Slides-Green-Jacobi-topology-optimization-NANTES/movies/movie_TO_1024_eta4h_dual')
W, H, FPS, SECONDS = 3648, 2312, 30, 7          # each periodic cell ~1024 px = native 1024^2 (1:1); same aspect as the reference 980 x 620
W, H = int(os.environ.get('W', W)), int(os.environ.get('H', H))
FS, TWOLINE, TOP = float(os.environ.get('FS', 74)), os.environ.get('TWOLINE') == '1', float(os.environ.get('TOP', 0.885))
EVERY = 10                                      # HISTORY_EVERY of the run

d = netCDF4.Dataset(HIST)
rho = np.asarray(d['phase_field_iterate'][:, 0])      # frame, nx, ny
nf, n = rho.shape[0], rho.shape[1]
iters = 1 + EVERY * np.arange(nf)
print(f'{nf} saved iterates, grid {n}^2, last iteration {iters[-1]}')

# tiled periodic copies: 2 cells high, enough cells wide for the aspect; cell = 1024 pixels
tile_y = 2
tile_x = int(np.ceil(tile_y * W / H)) + 1
dpi = 100
fig = plt.figure(figsize=(W / dpi, H / dpi), dpi=dpi, facecolor='white')
ax = fig.add_axes([0.0, 0.0, 1.0, TOP])
title = fig.text(0.5, (1 + TOP) / 2, '', ha='center', va='center', fontsize=FS, family='serif', linespacing=1.3)
ax.set_axis_off()
img = ax.imshow(np.zeros((tile_y * n, tile_x * n)), cmap='gray_r', vmin=0, vmax=1, interpolation='nearest',
                origin='lower', aspect='equal')
ax.set_xlim(0, tile_y * n * W / (H * TOP))
ax.set_ylim(0, tile_y * n)


def frame(r, it):
    img.set_data(np.tile(r.T, (tile_y, tile_x)))      # x horizontal, y vertical
    if os.environ.get('TITLE') == 'iteration':
        title.set_text(f'Optimizer step: {it}')   # 'step' (outer L-BFGS), not CG iteration
    else:
        title.set_text(f'Phase-field TO, 1024$^2$, contrast $10^4$' + ('\n' if TWOLINE else ', ') + f'Iteration: {it}')
    fig.canvas.draw()
    return np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()


total = FPS * SECONDS
pos = np.linspace(0, nf - 1, total)
writer = imageio_ffmpeg.write_frames(OUT + '.mp4', (W, H), fps=FPS, codec='libx264', pix_fmt_out='yuv420p',
                                     quality=9, macro_block_size=8)
writer.send(None)
for k, p in enumerate(pos):
    i = int(np.floor(p))
    j = min(i + 1, nf - 1)
    a = p - i
    r = (1 - a) * rho[i] + a * rho[j]
    it = int(round((1 - a) * iters[i] + a * iters[j]))
    writer.send(frame(r, it))
writer.close()
for tag, i in (('initial', 0), ('final', nf - 1)):
    plt.imsave(OUT + f'_{tag}.png', frame(rho[i], iters[i]))
print('wrote', OUT + '.mp4', OUT + '_initial.png', OUT + '_final.png')
