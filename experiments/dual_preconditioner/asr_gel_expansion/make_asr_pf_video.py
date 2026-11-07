"""
MP4 of an asr_pf_fourel.py run: damage on the exact pixels (aggregate outlines) per load step, next to the macro
volume change tr E and the stiffness ratio with a moving marker.
    python make_asr_pf_video.py [VARIANT] [N] [PREC] [SUFFIX]     (fourel 128 dual '')
-> figures/ASRpf_video_<VARIANT>_N<N>_<PREC><SUFFIX>.mp4
"""
import os
import sys
import glob
import imageio_ffmpeg
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
from matplotlib.collections import LineCollection

plt.rcParams['animation.ffmpeg_path'] = imageio_ffmpeg.get_ffmpeg_exe()
plt.rcParams.update({'font.size': 13})
here = os.path.dirname(os.path.abspath(__file__))
V = sys.argv[1] if len(sys.argv) > 1 else 'fourel'
N = int(sys.argv[2]) if len(sys.argv) > 2 else 128
P = sys.argv[3] if len(sys.argv) > 3 else 'dual'
S = sys.argv[4] if len(sys.argv) > 4 else ''
base = os.path.join(here, 'exp_data', f'ASRpf_{V}_N{N}_{P}{S}')
st = np.load(base + '_steps.npz', allow_pickle=True)
eps, E, stiff, label = 100 * st['eps_asr'], st['E'], st['stiffness_ratio'], st['label']
trE = 100 * (E[:, 0] + E[:, 1])
key = 'loss' if glob.glob(base + '_damage/loss_*.npy') else 'step'      # PF-CZM: 1 - omega(d)
damage = [np.load(f) for f in sorted(glob.glob(base + f'_damage/{key}_*.npy'))][:len(eps)]

agg = label >= 1
segs = [[(i + 1, j), (i + 1, j + 1)] for i in range(N - 1) for j in range(N) if agg[i, j] != agg[i + 1, j]]
segs += [[(i, j + 1), (i + 1, j + 1)] for i in range(N) for j in range(N - 1) if agg[i, j] != agg[i, j + 1]]

fig = plt.figure(figsize=(13, 6.2))
ax_d = fig.add_axes([0.03, 0.08, 0.45, 0.82])
ax_e = fig.add_axes([0.60, 0.56, 0.37, 0.36])
ax_s = fig.add_axes([0.60, 0.10, 0.37, 0.36])
im = ax_d.imshow(damage[0].T, origin='lower', interpolation='nearest', cmap='magma_r', vmin=0, vmax=1,
                 extent=(0, N, 0, N))
ax_d.add_collection(LineCollection(segs, colors='tab:cyan', linewidths=0.4))
ax_d.set_xticks([])
ax_d.set_yticks([])
fig.colorbar(im, ax=ax_d, fraction=0.046, pad=0.02, label='1 - omega(d)' if key == 'loss' else 'damage d')
title = ax_d.set_title('')
ax_e.plot(eps, trE, color='k')
ax_s.plot(eps, stiff, color='k')
m_e, = ax_e.plot([], [], 'o', color='tab:red', ms=8)
m_s, = ax_s.plot([], [], 'o', color='tab:red', ms=8)
ax_e.set_ylabel('macro tr E [%]')
ax_s.set_ylabel('|C_eff| / |C_eff,0|')
for ax in (ax_e, ax_s):
    ax.set_xlabel('gel eigenstrain eps_ASR [%]')
    ax.set_xlim(0, eps[-1])
    ax.grid(alpha=0.3)
ax_s.set_ylim(0, 1.05)


def frame(k):
    im.set_data(damage[k].T)
    title.set_text(f'{V}, {N}^2: eps_ASR = {eps[k]:.3f} %')
    m_e.set_data([eps[k]], [trE[k]])
    m_s.set_data([eps[k]], [stiff[k]])
    return im, m_e, m_s, title


anim = animation.FuncAnimation(fig, frame, frames=len(damage), blit=False)
out = os.path.join(here, 'figures', f'ASRpf_video_{V}_N{N}_{P}{S}.mp4')
anim.save(out, writer=animation.FFMpegWriter(fps=6, bitrate=3000), dpi=110)
print('->', out)
