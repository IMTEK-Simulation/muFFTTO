"""Video of a 3D hollow-bar run (pressurized_bar_3D.py): the three deformed mid-sections (cross-section z = Lz/2,
longitudinal y = Ly/2 and x = Lx/2) on their exact voxels, one frame per load increment, colour = in-plane area ratio
of the section voxel. The pinned cavity-interior nodes are filled by a harmonic extension for display only.
Usage: python bar3d_video.py <run folder> [fps]   -> <run folder>/bar3d.mp4"""
import os
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.animation import FFMpegWriter  # noqa: E402
from matplotlib.collections import PolyCollection  # noqa: E402
from mpl_toolkits.mplot3d.art3d import Poly3DCollection  # noqa: E402

try:                                                 # no system ffmpeg: use the imageio-ffmpeg binary
    import imageio_ffmpeg
    plt.rcParams['animation.ffmpeg_path'] = imageio_ffmpeg.get_ffmpeg_exe()
except ImportError:
    pass

plt.rcParams.update({'font.size': 13})
run = sys.argv[1]
fps = float(sys.argv[2]) if len(sys.argv) > 2 else 3
d = dict(np.load(os.path.join(run, 'state.npz')))
if 'u_increments' in d:                              # OUTPUT_EVERY > 1: u only for the stored increments
    for _key in ('p', 'volume', 'phase', 'hessp', 'nit', 'min_J', 'time'):
        if _key in d:
            d[_key] = d[_key][d['u_increments']]
N, h = tuple(int(v) for v in d['N']), float(d['h'])
solid, cav, clamp = d['solid_px'], d['cav_px'], d['clamp_node']
p, vol = d['p'], d['volume']
void_node = np.all([np.roll(~solid, (a, b, c), (0, 1, 2)) for a in (0, 1) for b in (0, 1) for c in (0, 1)], axis=0)
cav_node = void_node & ~clamp & np.any([np.roll(cav, (a, b, c), (0, 1, 2)) for a in (0, 1) for b in (0, 1)
                                        for c in (0, 1)], axis=0)


def filled(k, start=None, iterations=3000):
    """harmonic fill of the pinned cavity nodes; warm start from the previous frame's fill (fewer iterations)"""
    u = d['u'][k].astype(float)
    if start is not None:
        u = np.where(cav_node, start, u)
    for _ in range(iterations):
        u = np.where(cav_node, sum(np.roll(u, s, ax) for s in (1, -1) for ax in (1, 2, 3)) / 6.0, u)
    return u


def section_polys(u, axis, mask):
    a, b = [ax for ax in range(3) if ax != axis]
    layer = N[axis] // 2
    sl = [slice(None)] * 3
    sl[axis] = layer
    ii, jj = np.nonzero(mask[tuple(sl)])
    polys = []
    for di, dj in ((0, 0), (1, 0), (1, 1), (0, 1)):
        pts = []
        for comp in (a, b):
            val = 0.0
            for dk in (0, 1):                                # mid-plane of the voxel layer
                idx = [None] * 3
                idx[a], idx[b], idx[axis] = (ii + di) % N[a], (jj + dj) % N[b], np.full_like(ii, (layer + dk) % N[axis])
                val = val + 0.5 * u[comp][tuple(idx)]
            pts.append(((ii + di) if comp == a else (jj + dj)) * h + val)
        polys.append(np.stack(pts, axis=-1))
    polys = np.stack(polys, axis=1)
    x, y = polys[..., 0], polys[..., 1]
    return polys, 0.5 * np.sum(x * np.roll(y, -1, 1) - np.roll(x, -1, 1) * y, axis=1) / h ** 2


from display_fill import HarmonicFill  # noqa: E402
_fill = HarmonicFill(cav_node)
periodic_z = bool(cav.any(axis=(0, 1)).all())              # the channel runs through the whole cell in z
U = [_fill(d['u'][0].astype(float))]
for k in range(1, len(p)):
    U.append(_fill(d['u'][k].astype(float), start=U[-1]))
AX = [(2, 'cross-section z = Lz/2', 'x', 'y'), (1, 'section y = Ly/2', 'x', 'z'), (0, 'section x = Lx/2', 'y', 'z')]
lims = {}
for axis, *_ in AX:
    P_all = [section_polys(U[k], axis, solid | cav)[0] for k in (0, len(p) - 1)]
    lims[axis] = (min(q[..., 0].min() for q in P_all) - 0.03, max(q[..., 0].max() for q in P_all) + 0.03,
                  min(q[..., 1].min() for q in P_all) - 0.03, max(q[..., 1].max() for q in P_all) + 0.03)
norm = matplotlib.colors.Normalize(vmin=0.5, vmax=1.5)

# 3D cut-away: the boundary faces of the solid voxels with x < Lx/2 (exact voxel faces, deformed), colour = voxel
# det F; the cell ends z = 0 and z = Lz get their faces too (periodic in z: drawn as the ends of the computed segment)
keep = solid.copy()
keep[N[0] // 2:, :, :] = False
_faces, _owner = [], []
for a in range(3):
    b, c = [o for o in range(3) if o != a]
    for side in (0, 1):
        nb = np.roll(keep, -1 if side else 1, a)                         # the neighbour across this face
        if a == 2:                                                       # close the segment at z = 0 and z = Lz
            nb[:, :, -1 if side else 0] = False
        vox = np.argwhere(keep & ~nb)                                    # [n, 3]
        corners = []
        for db, dc in ((0, 0), (1, 0), (1, 1), (0, 1)):
            nd = vox.copy()
            nd[:, a] += side
            nd[:, b] += db
            nd[:, c] += dc
            corners.append(nd)
        _faces.append(np.stack(corners, axis=1))                         # [n, 4, 3] unwrapped node indices
        _owner.append(vox)
faces3d = np.concatenate(_faces)
owner3d = np.concatenate(_owner)
_fw = faces3d % np.array(N)                                              # wrapped node indices (periodic)


def voxel_detF(u):
    G = np.zeros((3, 3) + N)
    for ax in range(3):
        acc = 0.0
        others = [o for o in range(3) if o != ax]
        for a_ in (0, 1):
            for b_ in (0, 1):
                shift = [0, 0, 0]
                shift[others[0]], shift[others[1]] = -a_, -b_
                base = np.roll(u, shift, (1, 2, 3))
                acc = acc + (np.roll(base, -1, ax + 1) - base) / h
        G[:, ax] = 0.25 * acc
    F = G + np.eye(3).reshape(3, 3, 1, 1, 1)
    return (F[0, 0] * (F[1, 1] * F[2, 2] - F[1, 2] * F[2, 1]) - F[0, 1] * (F[1, 0] * F[2, 2] - F[1, 2] * F[2, 0])
            + F[0, 2] * (F[1, 0] * F[2, 1] - F[1, 1] * F[2, 0]))


def faces_deformed(u):
    pos = faces3d * h + np.stack([u[comp][_fw[..., 0], _fw[..., 1], _fw[..., 2]] for comp in range(3)], axis=-1)
    J = voxel_detF(u)[owner3d[:, 0], owner3d[:, 1], owner3d[:, 2]]
    return pos, J


_last = faces_deformed(U[-1])[0]
lim3d = [(min(0.0, _last[..., k].min()) - 0.02, max(N[k] * h, _last[..., k].max()) + 0.02) for k in range(3)]
VIDEO_3D = os.environ.get('VIDEO_3D', '0') == '1'      # 1: extra 3D cut-away row (off by default)
if VIDEO_3D:
    fig = plt.figure(figsize=(21, 12))
    gs = fig.add_gridspec(2, 4, width_ratios=[1, 1, 1, 1.15], height_ratios=[1, 1.05], hspace=0.08)
    axes = np.array([fig.add_subplot(gs[0, k]) for k in range(4)])
    ax3d = fig.add_subplot(gs[1, :], projection='3d')
else:
    fig, axes = plt.subplots(1, 4, figsize=(21, 6.0), gridspec_kw=dict(width_ratios=[1, 1, 1, 1.15]))
    fig.subplots_adjust(top=0.80)
PLOT3 = (2, 0, 1)            # plot axes (horizontal, depth, up) = (z, x, y): the channel lies along the horizontal


def draw3d(k):
    ax3d.clear()
    pos, J = faces_deformed(U[k])
    ax3d.add_collection3d(Poly3DCollection(pos[..., list(PLOT3)], facecolors=plt.cm.viridis(norm(J)),
                                           edgecolors=(0, 0, 0, 0.12), linewidths=0.2))
    lims = [lim3d[a] for a in PLOT3]
    ax3d.set_xlim(*lims[0])
    ax3d.set_ylim(*lims[1])
    ax3d.set_zlim(*lims[2])
    ax3d.set_box_aspect([l[1] - l[0] for l in lims], zoom=1.25)
    ax3d.view_init(elev=22, azim=62)
    ax3d.set_xlabel('z (channel axis)', labelpad=12)
    ax3d.set_ylabel('x', labelpad=8)
    ax3d.set_zlabel('y')
    ax3d.text2D(0.5, 0.9, 'deformed solid, half x < Lx/2 (cut face towards the viewer), colour = voxel det F; '
                'periodic in z: one segment z = 0 .. Lz', transform=ax3d.transAxes, ha='center', fontsize=13)
out = os.path.join(run, 'bar3d.mp4')
writer = FFMpegWriter(fps=fps, bitrate=4000)
with writer.saving(fig, out, dpi=110):
    for k in range(len(p)):
        if VIDEO_3D:
            draw3d(k)
        for ax in axes:
            ax.clear()
        for ax, (axis, title, xl, yl) in zip(axes[:3], AX):
            ax.add_collection(PolyCollection(section_polys(U[k], axis, cav)[0], facecolor='#e06c5a', edgecolor='none'))
            ps, J2 = section_polys(U[k], axis, solid)
            pc = PolyCollection(ps, array=J2, cmap='viridis', norm=norm, edgecolor='k', linewidth=0.1)
            ax.add_collection(pc)
            if axis != 0:
                for xc in np.nonzero(clamp[:, 0, 0])[0]:
                    ax.axvline(xc * h, color='k', lw=3)
            ax.set_xlim(*lims[axis][:2])
            ax.set_ylim(*lims[axis][2:])
            if periodic_z and axis != 2:                         # z vertical: the computed segment z = 0 .. Lz
                ax.set_ylim(0.0, N[2] * h)
                for zz in (0.0, N[2] * h):
                    ax.axhline(zz, color='k', ls='--', lw=1.2)
            ax.set_aspect('equal')
            ax.set_title(title + ('\nz = 0 \u2261 z = Lz (periodic)' if periodic_z and axis != 2 else ''))
            ax.set_xlabel(xl)
            ax.set_ylabel(yl)
        ax = axes[3]
        if 'wave' in d and int(d['wave']):
            z = np.linspace(0, N[2] * h, 200)
            ph = float(d['phase'][k])
            kw = float(d['WAVES']) if 'WAVES' in d else 1.0
            if 'pulse' in d and int(d['pulse']):
                Lz = N[2] * h
                zeta = np.mod(z - ph * float(d['PULSE_TRAVEL']) * Lz, Lz)
                ell = float(d['PULSE_LEN']) * Lz
                prof = p[k] * np.where(zeta < ell, np.sin(2 * np.pi * zeta / ell), 0.0)
            else:
                prof = p[k] * (float(d['P_OFFSET']) + float(d['P_VAR']) * np.cos(2 * np.pi * (kw * z / (N[2] * h) - ph)))
            ax.axvline(0, color='0.6', lw=1)
            ax.fill_betweenx(z, 0, prof, where=prof >= 0, color='#e06c5a', alpha=0.6, label='pressure')
            ax.fill_betweenx(z, 0, prof, where=prof < 0, color='#6a8fd1', alpha=0.6, label='suction')
            ax.plot(prof, z, color='k')
            ax.set_xlim(-1.1 * p.max(), 1.1 * p.max())
            ax.set_ylim(0, N[2] * h)
            ax.set_xlabel('gas pressure p(z, t)   (E = 100)')
            ax.set_ylabel('z')
            ax.grid(alpha=0.3)
            ax.legend(loc='upper right', fontsize=11)
            ax.set_title(f'amplitude {p[k]:.1f}, t / T = {ph:.3f}')
            if k == 0:
                cb = fig.colorbar(pc, ax=axes[:3].tolist(), shrink=0.8, pad=0.01)
                cb.set_label('section voxel area ratio')
                fig.suptitle(f'pressure wave in a periodic 3D channel, x walls clamped, '
                             f'h = 1/{int(round(1 / h))} ({N[0]} x {N[1]} x {N[2]} voxels)')
            writer.grab_frame()
            continue
        ax.plot(vol / vol[0], p, '-', color='0.7')
        ax.plot(vol[:k + 1] / vol[0], p[:k + 1], 'o-', color='k')
        ax.set_xlim(0, 1.05 * vol.max() / vol[0])
        ax.set_ylim(0, 1.05 * p.max())
        ax.set_xlabel('cavity volume / initial volume')
        ax.set_ylabel('pressure p   (E = 100)')
        ax.grid(alpha=0.3)
        ax.set_title(f'p = {p[k]:.1f}, volume x{vol[k] / vol[0]:.2f}')
        if k == 0:
            cb = fig.colorbar(pc, ax=axes[:3].tolist(), shrink=0.8, pad=0.01)
            cb.set_label('section voxel area ratio')
            fig.suptitle(f'3D hollow bar, x faces clamped, pressurised cavity, h = 1/{int(round(1 / h))} '
                         f'({N[0]} x {N[1]} x {N[2]} voxels)')
        writer.grab_frame()
print(out)
