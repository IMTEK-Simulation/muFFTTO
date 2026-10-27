"""
Rotating view of the house geometry and the wind load (house_wind_3D.py), rendered in parallel.

usage:  H=0.125 python house_geometry_video.py     writes exp_data/house_geometry_h<h>.mp4
"""
import os
import shutil
import subprocess
from multiprocessing import Pool

import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import colors

import house_wind_3D as house

HERE = os.path.dirname(os.path.realpath(__file__))
FRAMES = os.path.join(HERE, 'exp_data', 'house', 'frames_geometry')
NB_FRAMES, NB_WORKERS = 120, 12
ARROWS = {'windward': 'C3', 'leeward': 'C0', 'side': 'C4', 'roof': 'C2'}


def render(frames):
    phase, base = house.geometry()
    _, faces = house.wind_forces(phase)
    phase = phase[:, base:, :]
    nx, ny, nz = phase.shape
    fc = np.zeros(phase.shape + (4,))
    fc[phase == 1] = colors.to_rgba('#d9c7a3')
    fc[phase == 2] = colors.to_rgba('#5b9bd5', 0.9)
    fig = plt.figure(figsize=(10, 7.5))
    ax = fig.add_axes((0, 0.06, 1, 0.88), projection='3d')
    ax.voxels(phase.transpose(0, 2, 1) > 0, facecolors=fc.transpose(0, 2, 1, 3), edgecolor=(0, 0, 0, 0.12),
              linewidth=0.2)
    for kind, axis, sign, idx in faces:
        step = 6
        idx = idx[np.all(idx[:, [a for a in range(3) if a != axis]] % step == step // 2, axis=1)]
        c = (idx + 0.5).astype(float)
        c[:, 1] -= base
        c[:, axis] += sign * 0.5
        vec = np.zeros(3)
        vec[axis] = -house.CP[kind] * sign * 2.5
        start = c - vec if house.CP[kind] > 0 else c
        ax.quiver(start[:, 0], start[:, 2], start[:, 1], vec[0], vec[2], vec[1], color=ARROWS[kind], lw=1.0,
                  arrow_length_ratio=0.3)
    ax.set_box_aspect((nx, nz, ny))
    ax.set_axis_off()
    fig.text(0.5, 0.96, f'house 6 × 8 m, ridge 5 m, masonry 0.25 m, voxel {house.h * 100:g} cm — wind from −x, '
             f'q = 1 kPa', ha='center', fontsize=12)
    fig.text(0.5, 0.02, 'red: windward pressure +0.8q   blue: leeward suction −0.5q   purple: side suction −0.7q   '
             'green: roof uplift −0.6q   blue panels: door and windows', ha='center', fontsize=9)
    for k in frames:
        ax.view_init(elev=22 + 8 * np.sin(2 * np.pi * k / NB_FRAMES), azim=-130 + 360 * k / NB_FRAMES)
        fig.savefig(os.path.join(FRAMES, f'f{k:04d}.png'), dpi=110)
    plt.close(fig)


if __name__ == '__main__':
    shutil.rmtree(FRAMES, ignore_errors=True)
    os.makedirs(FRAMES)
    with Pool(NB_WORKERS) as pool:
        pool.map(render, [list(range(w, NB_FRAMES, NB_WORKERS)) for w in range(NB_WORKERS)])
    out = os.path.join(HERE, 'exp_data', f'house_geometry_h{house.h:g}.mp4')
    import imageio_ffmpeg
    subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), '-y', '-loglevel', 'error', '-framerate', '24', '-i', os.path.join(FRAMES, 'f%04d.png'),
                    '-vf', 'pad=ceil(iw/2)*2:ceil(ih/2)*2', '-pix_fmt', 'yuv420p', out], check=True)
    print(f'wrote {out}')
