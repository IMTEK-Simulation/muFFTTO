"""Slide video: topology optimization WITHOUT the phase-field term (w = 0), one panel, exact pixels (solid black, void
white), sine start (logs/TO_no_pf; history every 10 L-BFGS iterations, each stored iterate held, no blending).
    python make_TO_no_pf_single_video.py [N=128]  ->  talk movies/movie_TO_no_pf_N<N>.mp4 (+ _final.png)
env: SUFFIX (run suffix, default _nopf_test), EVERY (HISTORY_EVERY of the run, 10), HOLD (video frames per stored
     iterate, 8), TAG (appended to the output name), NOTITLE=1 (square video, design only), LABEL=1 ('optimizer
     iteration k' in a white box inside the frame, top left), STRIDE (use every STRIDE-th stored iterate, 1),
     TALKTITLE=1 (title as the talk's other TO videos, make_TO_slide_movie.py: serif text above the design, two lines
     'TO without phase-field term, N^2, contrast 1e4' / 'Iteration: k', frame aspect 1.2 like the column video;
     now one line 'Optimizer step: k'), INIT_HOLD (frames of the start design, 2 s)
slide-6 video: SUFFIX=_nopf_random TAG=_random EVERY=1 STRIDE=2 HOLD=1 KMAX=555 TALKTITLE=1 INIT_HOLD=6 python ... 64
"""
import glob
import os
import sys

import imageio_ffmpeg
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import netCDF4
import numpy as np

HERE = os.path.dirname(os.path.realpath(__file__))
D = os.path.join(HERE, 'data', 'example_2D_elasticity_TO')
N = int(sys.argv[1]) if len(sys.argv) > 1 else 128
SUFFIX, EVERY = os.environ.get('SUFFIX', '_nopf_test'), int(os.environ.get('EVERY', 10))
OUT = os.path.expanduser(f'~/Projects/Talks/Slides-Green-Jacobi-topology-optimization-NANTES/movies/movie_TO_no_pf_N{N}'
                         + os.environ.get('TAG', ''))
NOTITLE = os.environ.get('NOTITLE') == '1'
TALKTITLE = os.environ.get('TALKTITLE') == '1'
STRIDE = int(os.environ.get('STRIDE', 1))
W, H, FPS, HOLD = 1600, 1600 if NOTITLE else (1920 if TALKTITLE else 1760), 30, int(os.environ.get('HOLD', 8))
INIT_HOLD = int(os.environ.get('INIT_HOLD', 2 * FPS))   # frames showing the start design (slide 6 video: 6 = 0.2 s)
ENC = ['-profile:v', 'high', '-level', '5.1', '-crf', '17', '-g', '30', '-keyint_min', '30', '-preset', 'slow',
       '-movflags', '+faststart']
f = glob.glob(os.path.join(D, f'N{N}_soft0.0001_energy-warm_Green_Jacobi_eta_*_pfw0_l_bfgs{SUFFIX}_history.nc'))[0]
with netCDF4.Dataset(f) as d:
    R = np.asarray(d['phase_field_iterate'][:, 0])
idx = list(range(0, len(R), STRIDE)) + ([len(R) - 1] if (len(R) - 1) % STRIDE else [])
KMAX = os.environ.get('KMAX')                                     # KMAX (env): stop the video at this iteration
if KMAX:
    kmax = int(KMAX) // EVERY
    idx = [k for k in idx if k < kmax] + [kmax]
print(f'{len(R)} stored iterates (last iteration {EVERY * (len(R) - 1)})')
plt.rcParams.update({'text.usetex': not TALKTITLE, 'font.family': 'serif', 'font.size': 40})
fig = plt.figure(figsize=(W / 100, H / 100), dpi=100)
ax = fig.add_axes([0, 0, 1, W / H])
ax.set_axis_off()
im = ax.imshow(R[0].T, origin='lower', cmap='Greys', vmin=0, vmax=1, interpolation='nearest')
title = fig.text(0.5, (1 + W / H) / 2, '', ha='center', va='center', fontsize=58 if TALKTITLE else 44,
                 linespacing=1.3)
LABEL = os.environ.get('LABEL') == '1'
label = fig.text(0.03, 0.97 * W / H, '', ha='left', va='top', fontsize=50,
                 bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='0.3', alpha=0.92)) if LABEL else None


def draw(k):
    im.set_data(R[k].T)
    if TALKTITLE:
        title.set_text(f'Optimizer step: {EVERY * k}')   # 'step' (outer L-BFGS), not CG iteration
    elif not NOTITLE:
        title.set_text(rf'no phase-field term, ${N}^2$: iteration {EVERY * k}')
    if label is not None:
        label.set_text(rf'optimizer iteration $k = {EVERY * k}$')
    fig.canvas.draw()
    return np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()


w = imageio_ffmpeg.write_frames(OUT + '.mp4', (W, H), fps=FPS, quality=None, codec='libx264', pix_fmt_out='yuv420p',
                                macro_block_size=8, output_params=ENC)
w.send(None)
for k in idx:
    img = draw(k)
    for _ in range(HOLD if k else INIT_HOLD):
        w.send(img)
for _ in range(3 * FPS):
    w.send(img)
w.close()
plt.imsave(OUT + '_final.png', img)
print('wrote', OUT + '.mp4')
