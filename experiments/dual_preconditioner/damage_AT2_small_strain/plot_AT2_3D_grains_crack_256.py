"""
3D crack of the stiff tough grains case (grains3D: polydisperse balls 10x stiffer, l = 1/16, dual runs on NEMO) at
128^3 (l/h = 8) and 256^3 (l/h = 16), final load step (cell cut through).

The crack is drawn as the level set d = THRESHOLD of the nodal damage (marching cubes), red, with smooth lighting. The face-on map shows the crack position x. Balls: translucent spheres
(periodic images clipped to the cell).

Input: damage-only copies made with extract_damage_nc.py, in exp_data/nemo_view/.

    python plot_AT2_3D_grains_crack_256.py           figures/AT2_3D_grains_crack_128_256.png
    python plot_AT2_3D_grains_crack_256.py video     figures/AT2_3D_grains_crack_128_256.mp4 (one camera turn)
"""
import os
import subprocess
import sys
from multiprocessing import Pool

import imageio_ffmpeg
import matplotlib
import netCDF4
import numpy as np

matplotlib.use('Agg')
import matplotlib.pyplot as plt                                   # noqa: E402
from matplotlib.colors import LightSource                         # noqa: E402
from mpl_toolkits.mplot3d.art3d import Poly3DCollection          # noqa: E402

HERE = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, '../../..')))
from muFFTTO import geometry  # noqa: E402

VIEW = os.path.join(HERE, 'exp_data', 'nemo_view')
CASES = [(128, 'AT2_grains3D_128_dual_damage.nc'), (256, 'AT2_grains3D_256_dual_damage.nc')]
LOG = os.path.join(HERE, 'logs', 'nemo_3D', 'AT2_N{N}_dual_polydisperse_pores_incl10_l0.0625_3D.log')
THRESHOLD = 0.9
ELEV, AZIM = 22, -55
NB_TURN, FPS = 120, 15
FRAMES = os.path.join(HERE, 'figures', 'AT2_3D_grains_crack_frames')
LIGHT = np.array([0.35, -0.75, 0.55]) / np.linalg.norm([0.35, -0.75, 0.55])
CMAP = plt.cm.viridis                      # face-on map of the crack position
CRACK_RGB = (0.78, 0.10, 0.08)             # crack in 3D: red, shaded
centres, radii = geometry.polydisperse_disk_centres_and_radii(3, 0.25, (0.05, 0.14), 0.04, 0)
u, v = np.meshgrid(np.linspace(0, 2 * np.pi, 32), np.linspace(0, np.pi, 16))
SPHERE = np.array([np.cos(u) * np.sin(v), np.sin(u) * np.sin(v), np.cos(v)])


def final_damage(name):
    with netCDF4.Dataset(os.path.join(VIEW, name)) as ds:
        return np.asarray(ds['damage'][-1, 0])


def last_step(N):
    """(step, E_xx, max d) of the last load step in the run log."""
    line = [ln for ln in open(LOG.format(N=N)) if ln.startswith('step')][-1].split()
    return int(line[1]), float(line[4]), float(line[11])


def exposed_faces(mask):
    """Quads (k, 4, 3) of the faces of the voxels in mask that do not touch another voxel of mask, their outward
    normals (k, 3) and the voxel x index (k,). The cell boundary counts as exposed."""
    n = mask.shape[0]
    quads, normals, xs = [], [], []
    corner = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])
    for ax in range(3):
        a, b = [i for i in range(3) if i != ax]
        for side in (-1, 1):
            nb = np.zeros_like(mask)
            sl_dst = [slice(None)] * 3
            sl_src = [slice(None)] * 3
            if side == 1:
                sl_dst[ax], sl_src[ax] = slice(0, n - 1), slice(1, n)
            else:
                sl_dst[ax], sl_src[ax] = slice(1, n), slice(0, n - 1)
            nb[tuple(sl_dst)] = mask[tuple(sl_src)]
            idx = np.argwhere(mask & ~nb)                                   # (k, 3) voxel indices
            q = np.repeat(idx[:, None, :].astype(float), 4, axis=1)          # (k, 4, 3)
            q[:, :, ax] += 1.0 if side == 1 else 0.0
            q[:, :, a] += corner[:, 0]
            q[:, :, b] += corner[:, 1]
            nrm = np.zeros(3); nrm[ax] = side
            quads.append(q / n), normals.append(np.repeat(nrm[None], len(idx), 0)), xs.append(idx[:, 0])
    return np.concatenate(quads), np.concatenate(normals), np.concatenate(xs)


def crack_geometry(N, name):
    """Crack surface = level set d = THRESHOLD of the nodal damage (marching cubes, linear interpolation between
    nodes; no filtering of d), one colour, shaded by matplotlib's light source."""
    from skimage.measure import marching_cubes
    d = final_damage(name)
    verts, faces, _, _ = marching_cubes(d, THRESHOLD, spacing=(1 / N,) * 3)
    return dict(N=N, d=d, tris=verts[faces], nb_voxels=int((d >= THRESHOLD).sum()), nb_tris=len(faces))


def draw_3d(ax, g, azim):
    t = g['tris']
    nrm = np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0])
    nrm /= np.linalg.norm(nrm, axis=1, keepdims=True) + 1e-30
    a, e = np.radians(azim + 40), np.radians(35)                       # light from the camera side, turns with it
    light = np.array([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)])
    shade = 0.35 + 0.65 * np.abs(nrm @ light)                          # two-sided Lambert
    rgb = np.clip(np.array(CRACK_RGB)[None] * shade[:, None], 0, 1)
    ax.add_collection3d(Poly3DCollection(t, facecolors=rgb, edgecolors='none', linewidths=0))
    for c, r in zip(centres, radii):
        for shift in np.ndindex(3, 3, 3):
            cc = c + np.array(shift) - 1
            if np.any(cc + r < 0) or np.any(cc - r > 1):
                continue
            s = cc[:, None, None] + r * SPHERE
            s = np.where((s >= 0) & (s <= 1), s, np.nan)
            ax.plot_surface(*s, color='0.55', alpha=0.13, linewidth=0, shade=True)
    ax.set_xlim(0, 1), ax.set_ylim(0, 1), ax.set_zlim(0, 1), ax.set_box_aspect((1, 1, 1))
    ax.set_xlabel('x (load)', fontsize=12), ax.set_ylabel('y', fontsize=12), ax.set_zlabel('z', fontsize=12)
    ax.tick_params(labelsize=10)
    ax.view_init(elev=ELEV, azim=azim)
    step, exx, dmax = last_step(g['N'])
    ax.set_title(f'{g["N"]}$^3$ (l/h = {g["N"] // 16}), load step {step}, $\\bar\\varepsilon_{{xx}}$ = {exx:.3f}\n'
                 f'voxels with d $\\geq$ {THRESHOLD}: {g["nb_voxels"]:,}'.replace(',', ' '), fontsize=13)


def face_on(ax, fig, g):
    d, N = g['d'], g['N']
    x_crack = np.where(d.max(axis=0) >= THRESHOLD, d.argmax(axis=0) / N, np.nan)
    x_crack = np.where(x_crack > 0.6, x_crack - 1, x_crack)                           # periodic unwrap      # [y, z]
    im = ax.imshow(x_crack.T, origin='lower', extent=(0, 1, 0, 1), interpolation='nearest', cmap=CMAP,
                   vmin=np.nanpercentile(x_crack, 1), vmax=np.nanpercentile(x_crack, 99))
    cb = fig.colorbar(im, ax=ax, shrink=0.85)
    cb.set_label('crack position x', fontsize=12), cb.ax.tick_params(labelsize=11)
    x0 = np.nanmean(x_crack)
    t = np.linspace(0, 2 * np.pi, 200)
    for c, r in zip(centres, radii):
        for shift in np.ndindex(3, 3, 3):
            cc = c + np.array(shift) - 1
            if abs(cc[0] - x0) < r:
                rc = np.sqrt(r ** 2 - (cc[0] - x0) ** 2)
                ax.plot(cc[1] + rc * np.cos(t), cc[2] + rc * np.sin(t), '-', color='w', lw=1.2)
    ax.set_xlim(0, 1), ax.set_ylim(0, 1), ax.set_aspect('equal')
    ax.set_xlabel('y', fontsize=12), ax.set_ylabel('z', fontsize=12), ax.tick_params(labelsize=11)
    ax.set_title(f'{N}$^3$: crack face-on, position x over (y, z)\n'
                 f'(node of max d along x; white: balls cut at x = {x0:.2f})', fontsize=13)


GEOM = None


def init_worker():
    global GEOM
    GEOM = [crack_geometry(N, name) for N, name in CASES]


def render_turn(k):
    azim = AZIM + 360.0 * k / NB_TURN
    fig = plt.figure(figsize=(16, 8.4), dpi=100)
    for col, g in enumerate(GEOM):
        ax = fig.add_axes([0.01 + 0.5 * col, 0.04, 0.47, 0.82], projection='3d')
        draw_3d(ax, g, azim)
    fig.text(0.5, 0.955, 'AT2 fracture, stiff tough grains (10x), l = 1/16, dual preconditioner: final crack (surface d = 0.9)', ha='center', fontsize=15)
    fig.savefig(os.path.join(FRAMES, f'frame_{k:04d}.png'))
    plt.close(fig)
    return k


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == 'video':
        os.makedirs(FRAMES, exist_ok=True)
        with Pool(int(os.environ.get('WORKERS', 6)), initializer=init_worker) as pool:
            for k in pool.imap_unordered(render_turn, range(NB_TURN)):
                print(f'frame {k}', flush=True)
        out = os.path.join(HERE, 'figures', 'AT2_3D_grains_crack_128_256.mp4')
        subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), '-y', '-framerate', str(FPS), '-i',
                        os.path.join(FRAMES, 'frame_%04d.png'), '-vf', 'pad=ceil(iw/2)*2:ceil(ih/2)*2',
                        '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '20', out], check=True, capture_output=True)
        print(out)
    else:
        init_worker()
        fig = plt.figure(figsize=(16, 15), layout='constrained', dpi=130)
        for row, g in enumerate(GEOM):
            draw_3d(fig.add_subplot(2, 2, 2 * row + 1, projection='3d'), g, AZIM)
            face_on(fig.add_subplot(2, 2, 2 * row + 2), fig, g)
        fig.suptitle('AT2 fracture, stiff tough grains (10x), l = 1/16, dual preconditioner: final crack at 128$^3$ '
                     'and 256$^3$', fontsize=16)
        out = os.path.join(HERE, 'figures', 'AT2_3D_grains_crack_128_256.png')
        fig.savefig(out)
        print('wrote', out)
