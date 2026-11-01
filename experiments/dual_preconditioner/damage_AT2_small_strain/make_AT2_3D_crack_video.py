"""
Video of the 3D crack at 64^3, l/h = 2 (the runs of plot_AT2_3D_crack.py), side by side: stiff inclusions (10x, dual
run) and pores (Green-Jacobi run; all preconditioners give the same fields).

    part 1  the last load steps up to the cut-through: the voxels with d >= 0.5, coloured by d (exact voxels, a voxel
            takes the damage of its lower-left node), balls translucent; each panel shows its own load
    part 2  the final crack, one full turn of the camera

Frames are rendered in parallel (WORKERS env, default 8) into figures/AT2_3D_crack_frames/, then encoded with ffmpeg.

usage:  python make_AT2_3D_crack_video.py      writes figures/AT2_3D_crack.mp4
"""
import os
import subprocess
import sys
from multiprocessing import Pool

import numpy as np
import netCDF4
import imageio_ffmpeg
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, '../../..')))
from muFFTTO import geometry  # noqa: E402

CASES = [('stiff inclusions (10x)', 'AT2_N64_dual_polydisperse_pores_incl10_l0.03125_3D.nc', '0.55'),
         ('pores', 'AT2_N64_Green_Jacobi_polydisperse_pores_l0.03125_3D.nc', 'C0')]
THRESHOLD = 0.5
NB_GROWTH = 20                 # last load steps shown
HOLD_GROWTH = 4                # video frames per load step
NB_TURN = 120                  # frames of the full turn
FPS = 15
ELEV, AZIM = 22, -55
FRAMES = os.path.join(HERE, 'figures', 'AT2_3D_crack_frames')
centres, radii = geometry.polydisperse_disk_centres_and_radii(3, 0.25, (0.05, 0.14), 0.04, 0)
u, v = np.meshgrid(np.linspace(0, 2 * np.pi, 24), np.linspace(0, np.pi, 12))
SPHERE = np.array([np.cos(u) * np.sin(v), np.sin(u) * np.sin(v), np.cos(v)])
CMAP = plt.cm.inferno_r


def frame_indices(name):
    with netCDF4.Dataset(os.path.join(HERE, 'exp_data', name)) as ds:
        nb = ds['damage'].shape[0]
    return list(range(nb - NB_GROWTH, nb))


def render(job):
    k, frames, azim, part = job
    fig = plt.figure(figsize=(16, 8.4), dpi=100)
    for col, ((title, name, ball_color), i) in enumerate(zip(CASES, frames)):
        with netCDF4.Dataset(os.path.join(HERE, 'exp_data', name)) as ds:
            d = np.asarray(ds['damage'][i, 0])
            load = float(ds['load'][i])
            stress = float(ds['macro_stress'][i, 0, 0])
        n = d.shape[0]
        x = np.arange(n + 1) / n
        X, Y, Z = np.meshgrid(x, x, x, indexing='ij')
        ax = fig.add_axes([0.01 + 0.5 * col, 0.11, 0.47, 0.74], projection='3d')
        filled = d >= THRESHOLD
        ax.voxels(X, Y, Z, filled, facecolors=CMAP(0.15 + 0.85 * np.clip((d - THRESHOLD) / (1 - THRESHOLD), 0, 1)),
                  edgecolor='none', shade=True)
        for c, r in zip(centres, radii):
            for shift in np.ndindex(3, 3, 3):
                cc = c + np.array(shift) - 1
                if np.any(cc + r < 0) or np.any(cc - r > 1):
                    continue
                s = cc[:, None, None] + r * SPHERE
                s = np.where((s >= 0) & (s <= 1), s, np.nan)
                ax.plot_surface(*s, color=ball_color, alpha=0.15, linewidth=0, shade=True)
        ax.set_xlim(0, 1), ax.set_ylim(0, 1), ax.set_zlim(0, 1), ax.set_box_aspect((1, 1, 1))
        ax.set_xlabel('x (load)', fontsize=12), ax.set_ylabel('y', fontsize=12), ax.set_zlabel('z', fontsize=12)
        ax.view_init(elev=ELEV, azim=azim)
        ax.set_title(f'3D {title}, {n}$^3$, l/h = 2\n$\\bar\\varepsilon_{{xx}}$ = {load:.3f}, '
                     f'$\\bar\\sigma_{{xx}}$ = {stress:.2e}, max d = {d.max():.2f}', fontsize=14)
    sm = plt.cm.ScalarMappable(cmap=CMAP, norm=matplotlib.colors.Normalize(THRESHOLD - 0.15 / 0.85 * (1 - THRESHOLD), 1))
    cax = fig.add_axes([0.3, 0.09, 0.4, 0.02])
    cb = fig.colorbar(sm, cax=cax, orientation='horizontal')
    cb.set_ticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0]), cb.ax.tick_params(labelsize=11)
    cb.set_label(f'damage d (voxels with d $\\geq$ {THRESHOLD} shown)', fontsize=12)
    fig.text(0.5, 0.965, 'AT2 phase-field fracture, uniaxial macroscopic strain in x: '
             + ('the last load steps before the break' if part == 1 else 'final crack'),
             ha='center', fontsize=15)
    fig.savefig(os.path.join(FRAMES, f'frame_{k:04d}.png'))
    plt.close(fig)
    return k


if __name__ == '__main__':
    os.makedirs(FRAMES, exist_ok=True)
    growth = [frame_indices(name) for _, name, _ in CASES]
    jobs = [(k, (growth[0][k], growth[1][k]), AZIM, 1) for k in range(NB_GROWTH)]
    last = (growth[0][-1], growth[1][-1])
    jobs += [(NB_GROWTH + k, last, AZIM + 360.0 * k / NB_TURN, 2) for k in range(NB_TURN)]
    with Pool(int(os.environ.get('WORKERS', 8))) as pool:
        for k in pool.imap_unordered(render, jobs):
            print(f'frame {k}', flush=True)
    # concat list: each growth frame held HOLD_GROWTH video frames, then the turn
    listing = os.path.join(FRAMES, 'frames.txt')
    with open(listing, 'w') as f:
        for k in range(NB_GROWTH):
            f.write(f"file 'frame_{k:04d}.png'\nduration {HOLD_GROWTH / FPS}\n")
        for k in range(NB_GROWTH, NB_GROWTH + NB_TURN):
            f.write(f"file 'frame_{k:04d}.png'\nduration {1 / FPS}\n")
        f.write(f"file 'frame_{NB_GROWTH + NB_TURN - 1:04d}.png'\n")
    out = os.path.join(HERE, 'figures', 'AT2_3D_crack.mp4')
    subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), '-y', '-f', 'concat', '-safe', '0', '-i', listing,
                    '-vf', f'fps={FPS},pad=ceil(iw/2)*2:ceil(ih/2)*2', '-c:v', 'libx264', '-pix_fmt', 'yuv420p',
                    '-crf', '20', out], check=True, capture_output=True)
    print(out)
