"""3D talk visualisation of the core-shell battery particle (NEMO 3D run, dual): crack surfaces = damage isosurface
d = 0.9 (skimage marching cubes on the voxel field) inside the semi-transparent particle shell (outer radius 2.4 um,
core radius 2 um), plus a cut-away view: one octant removed, the three cut faces show the Li filling c / c_max on
their exact voxels, cracked voxels (d >= 0.9) left out so that the crack surface shows through.

The 3D runs store full 3D fields only in their checkpoint (written every 30 min of wall time); the snapshots are
mid-plane slices. The state shown is therefore the last checkpoint (256^3 dual: step 213 of 224, t = 4726 s,
a_c = 0.0686 = the final value). No crack-growth sequence in 3D, so the video rotates the final state.

Needs pyvista (not in muffto-gpu): run with a venv made by
    ~/micromamba/envs/muffto-gpu/bin/python -m venv --system-site-packages VENV && VENV/bin/pip install pyvista
    VENV/bin/python render_battery_3D.py [N=256] [video=1]
Inputs: exp_data/nemo_fields/N{N}_dual_3D_ckpt_{damage,conc}.npy, extracted from the checkpoint on NEMO with
    ssh nemo "cd .../exp_data && unzip -p battery_N256_dual_ms0.01_3D_tu_checkpoint.npz damage.npy" > ...
-> figures/slides/battery_3D_N{N}_{overview,cutaway}.png, movie_battery_3D_N{N}.mp4 (+ _final.png)
"""
import os
import sys

import imageio_ffmpeg
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv
from skimage import measure

HERE = os.path.dirname(os.path.abspath(__file__))
N = int(sys.argv[1]) if len(sys.argv) > 1 else 256
VIDEO = (sys.argv[2] != '0') if len(sys.argv) > 2 else True
FIELDS = os.path.join(HERE, 'exp_data', 'nemo_fields')
OUT = os.path.join(HERE, 'figures', 'slides')
os.makedirs(OUT, exist_ok=True)
L, R_CORE, R_OUT, D_ISO = 6.0, 2.0, 2.4, 0.9                       # um
STATE = {256: (213, 4726, 0.0686), 128: (219, 4859, 0.0769)}      # checkpoint step, t [s], broken fraction a_c
W, H = 3600, 2024
# player-safe H.264 (slides froze with the defaults): High@5.1, CRF 17, keyframe every second, faststart
ENC = ['-profile:v', 'high', '-level', '5.1', '-crf', '17', '-g', '30', '-keyint_min', '30', '-preset', 'slow', '-maxrate', '10M', '-bufsize', '20M', '-movflags', '+faststart']
SOL = {256: (0.787, 0.981), 128: (0.796, 0.980)}          # core / shell SOL at the checkpoint step
CRACK = '#f57d15'                                                  # inferno(0.7): crack colour
SHELL = '#9fb6cd'
h = L / N
c0 = L / 2

d = np.load(os.path.join(FIELDS, f'N{N}_dual_3D_ckpt_damage.npy'))[0, 0]
c = np.load(os.path.join(FIELDS, f'N{N}_dual_3D_ckpt_conc.npy'))[0, 0] if os.path.exists(
    os.path.join(FIELDS, f'N{N}_dual_3D_ckpt_conc.npy')) else None
xc = (np.arange(N) + 0.5) * h                                       # voxel centres
r = np.sqrt((xc[:, None, None] - c0) ** 2 + (xc[None, :, None] - c0) ** 2 + (xc[None, None, :] - c0) ** 2)
particle = r < R_OUT

# crack surface: marching cubes on the voxel-centre values (only inside the particle, the matrix never cracks)
v, f, nrm, _ = measure.marching_cubes(np.where(particle, d, 0.0), D_ISO)
v = v * h + 0.5 * h
crack = pv.PolyData(v, np.hstack([np.full((len(f), 1), 3), f]).astype(np.int64))
crack.point_data['Normals'] = -nrm
print(f'crack surface: {crack.n_cells} triangles, area {measure.mesh_surface_area(v, f):.1f} um^2 '
      f'(core sphere {4 * np.pi * R_CORE ** 2:.1f})')

shell = pv.Sphere(radius=R_OUT, center=(c0, c0, c0), theta_resolution=240, phi_resolution=240)
# removed octant {x > c0, y < c0, z > c0}, camera looks into it
box = (c0, L + 1, -1, c0, c0, L + 1)


def cut_faces(field):
    """Three cut planes of the removed octant, exact voxels, particle only, cracked voxels left out."""
    i_x, i_y, i_z = N // 2 - 1, N // 2, N // 2 - 1                # first kept voxel layer next to each cut plane
    meshes = []
    for axis, idx, plane in ((0, i_x, c0), (1, i_y, c0), (2, i_z, c0)):
        sl = [slice(None)] * 3
        sl[axis] = idx
        vals, dd, pp = field[tuple(sl)], d[tuple(sl)], particle[tuple(sl)]
        a, b = [k for k in range(3) if k != axis]
        # part of the plane that bounds the removed octant
        ra = xc > c0 if a == 0 else (xc < c0 if a == 1 else xc > c0)
        rb = xc > c0 if b == 0 else (xc < c0 if b == 1 else xc > c0)
        keep = pp & (dd < D_ISO) & ra[:, None] & rb[None, :]
        dims = [1, 1, 1]
        dims[a], dims[b] = N + 1, N + 1
        origin = [0.0, 0.0, 0.0]
        origin[axis] = plane
        g = pv.ImageData(dimensions=dims, spacing=(h, h, h), origin=origin)
        g.cell_data['val'] = vals.ravel(order='F') if a < b else vals.T.ravel(order='F')
        g.cell_data['keep'] = (keep.ravel(order='F') if a < b else keep.T.ravel(order='F')).astype(float)
        meshes.append(g.threshold(0.5, scalars='keep').extract_surface(algorithm="dataset_surface"))
    return meshes


def plotter(cutaway):
    p = pv.Plotter(off_screen=True, window_size=(W, H), lighting='three lights')
    p.set_background('white')
    if cutaway:
        p.add_mesh(crack.clip_box(box, invert=True), color=CRACK, smooth_shading=True, specular=0.4,
                   specular_power=20)
        p.add_mesh(shell.clip_box(box, invert=True), color=SHELL, opacity=0.22, smooth_shading=True)
        for m in cut_faces(c):
            p.add_mesh(m, scalars='val', cmap='viridis', clim=(0.1, 1.0), show_scalar_bar=False,
                       lighting=False)
    else:
        p.add_mesh(crack, color=CRACK, smooth_shading=True, specular=0.4, specular_power=20)
        p.add_mesh(shell, color=SHELL, opacity=0.22, smooth_shading=True)
    p.enable_depth_peeling(number_of_peels=8)
    p.enable_anti_aliasing('ssaa')
    p.camera.focal_point = (c0, c0, c0 - 0.15)
    p.camera.up = (0, 0, 1)
    return p


def set_view(p, azim, elev=24.0, dist=12.3):
    a, e = np.radians(azim), np.radians(elev)
    p.camera.position = (c0 + dist * np.cos(e) * np.cos(a), c0 + dist * np.cos(e) * np.sin(a),
                         c0 - 0.15 + dist * np.sin(e))
    p.camera.view_angle = 30.0


plt.rcParams.update({'text.usetex': True, 'font.family': 'serif', 'font.size': 30})


def overlay(cutaway):
    """Transparent W x H LaTeX overlay: title, legend, colour bar."""
    step, t, a_c = STATE[N]
    fig = plt.figure(figsize=(W / 150, H / 150), dpi=150)
    fig.patch.set_alpha(0.0)
    fig.text(0.5, 0.955, rf'core-shell NMC particle, ${N}^3$ voxels, dual run, $t = {t}$\,s (end of 1C lithiation)',
             ha='center', va='center', fontsize=34)
    lx, ly = 0.035, 0.80
    fig.patches.append(matplotlib.patches.Rectangle((lx, ly), 0.022, 0.035, color=CRACK, transform=fig.transFigure))
    fig.text(lx + 0.03, ly + 0.017, rf'crack surface $d = {D_ISO}$', va='center')
    fig.patches.append(matplotlib.patches.Rectangle((lx, ly - 0.06), 0.022, 0.035, color=SHELL, alpha=0.6,
                                                    transform=fig.transFigure))
    fig.text(lx + 0.03, ly - 0.043, r'particle surface (shell)', va='center')
    fig.text(lx, ly - 0.13, r'core debonded from the shell,' '\n' r'seed crack through the shell' '\n'
             rf'broken fraction $a_c = {a_c:.3f}$', va='top', fontsize=27, linespacing=1.4)
    if cutaway:
        cax = fig.add_axes([0.90, 0.25, 0.016, 0.42])
        cb = fig.colorbar(matplotlib.cm.ScalarMappable(matplotlib.colors.Normalize(0.1, 1.0), 'viridis'), cax=cax,
                          ticks=[0.2, 0.4, 0.6, 0.8, 1.0], format='%.1f')
        cb.ax.tick_params(labelsize=27)
        fig.text(0.908, 0.70, r'Li filling' '\n' r'$c/c_{\max}$ on' '\n' r'cut faces', ha='center', va='bottom',
                 fontsize=27, linespacing=1.3)
        fig.text(lx, ly - 0.30, r'the debond blocks Li transport:' '\n' rf'core SOL $= {SOL[N][0]:.2f}$, '
                 rf'shell SOL $= {SOL[N][1]:.2f}$', va='top', fontsize=27, linespacing=1.4)
    fig.canvas.draw()
    img = np.asarray(fig.canvas.buffer_rgba()).astype(float) / 255.0
    plt.close(fig)
    return img


def compose(rgb, ov):
    a = ov[:, :, 3:4]
    return np.clip(rgb / 255.0 * (1 - a) + ov[:, :, :3] * a, 0, 1) * 255.0


def shot(p):
    p.render()
    return np.asarray(p.screenshot(return_img=True))[:, :, :3].astype(float)


AZ0 = -50.0
for name, cut in (('overview', False), ('cutaway', True)):
    p = plotter(cut)
    set_view(p, AZ0)
    img = compose(shot(p), overlay(cut)).astype(np.uint8)
    out = os.path.join(OUT, f'battery_3D_N{N}_{name}.png')
    plt.imsave(out, img)
    print('saved', out)
    if cut and VIDEO:
        mov = os.path.join(OUT, f'movie_battery_3D_N{N}.mp4')
        ov = overlay(True)
        w = imageio_ffmpeg.write_frames(mov, (W, H), fps=30, quality=None, codec='libx264', pix_fmt_out='yuv420p',
                                        macro_block_size=8, output_params=ENC)
        w.send(None)
        n = 480                                                    # 16 s, one full turn, starting at the cut-away
        for k in range(n + 30):
            set_view(p, AZ0 + 360.0 * min(k, n) / n, elev=24.0 + 8.0 * np.sin(2 * np.pi * min(k, n) / n))
            w.send(compose(shot(p), ov).astype(np.uint8))
            if k % 60 == 0:
                print('frame', k, flush=True)
        w.close()
        plt.imsave(mov.replace('.mp4', '_final.png'), img)
        print('saved', mov)
    p.close()
