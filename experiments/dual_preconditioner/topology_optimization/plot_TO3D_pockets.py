"""3D check of the topology of converged 3D TO designs: solid rho >= 0.5 (translucent), closed void pockets (void
components not connected to the main void network across the periodic faces, orange) and detached solid specks
(solid components at rho >= 0.5 other than the main one, red, marked by a sphere since they are only a few voxels).
    <pyvista python> plot_TO3D_pockets.py  ->  figures/TO3D_pockets.png
Inputs: the 64^3 sw5 starts (sine, bi_s0, bi_s1) in data/example_3D_elasticity_TO and the 128^3 dual final design
(seg2 history of the NEMO run, copied to FIN128).
"""
import os

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import netCDF4
import numpy as np
import pyvista as pv
from scipy import ndimage
from skimage import measure

HERE = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(HERE, 'data', 'example_3D_elasticity_TO')
FIN128 = os.environ.get('FIN128', os.path.join(D, 'N128_soft0.0001_energy-warm_dual_eta_0.03125_pfw0.2_l_bfgs_seg2_history.nc'))
S64 = os.path.join(D, 'N64_soft0.0001_energy-warm_Green_Jacobi_eta_0.0625_pfw0.2_sw5_l_bfgs_start_{}_final.nc')
CASES = [(FIN128, r'$128^3$ dual, sine start, $a = 1$'),
         (S64.format('sine_seg3'), r'$64^3$, sine start, $a = 5$'),
         (S64.format('bi_s0_seg4'), r'$64^3$, Cahn--Hilliard start bi\_s0, $a = 5$'),
         (S64.format('bi_s1_seg8'), r'$64^3$, Cahn--Hilliard start bi\_s1, $a = 5$')]
# CASES (env): 'file|title;file|title;...' replaces the list (a _history.nc gives its last iterate); OUT: figure name
if os.environ.get('CASES'):
    CASES = [tuple(c.split('|', 1)) for c in os.environ['CASES'].split(';')]


def load(fn):
    with netCDF4.Dataset(fn) as d:
        v = d['phase_field_solution'] if 'phase_field_solution' in d.variables else d['phase_field_iterate']
        return np.asarray(v[-1, 0] if v.ndim == 5 else v[0, 0], dtype=float)


def periodic_labels(mask):
    """Connected components of mask, merged across the periodic faces: label array -> component id, sizes."""
    N = mask.shape[0]
    lab, n = ndimage.label(mask)
    par = list(range(n + 1))

    def f(a):
        while par[a] != a:
            par[a] = par[par[a]]
            a = par[a]
        return a
    for ax in range(3):
        a, b = np.take(lab, 0, axis=ax), np.take(lab, N - 1, axis=ax)
        m = (a > 0) & (b > 0)
        for x, y in zip(a[m], b[m]):
            par[f(x)] = f(y)
    root = np.array([f(k) for k in range(n + 1)])
    comp = root[lab]
    ids, cnt = np.unique(comp[comp > 0], return_counts=True)
    return comp, dict(zip(ids, cnt))


def iso(field, level, N):
    v, f, n, _ = measure.marching_cubes(np.pad(field, 1, constant_values=0.0), level)
    m = pv.PolyData((v - 0.5) / N, np.hstack([np.full((len(f), 1), 3), f]).astype(np.int64))
    m.point_data['Normals'] = -n
    return m


def render(rho):
    N = rho.shape[0]
    p = pv.Plotter(off_screen=True, window_size=(1400, 1400), lighting='three lights')
    p.set_background('white')
    p.add_mesh(pv.Box((0, 1, 0, 1, 0, 1)).extract_all_edges(), color='#8c8c8c', line_width=2)
    p.add_mesh(iso(rho, 0.5, N), color='#6f8fb8', opacity=0.18, smooth_shading=True)
    vcomp, vsz = periodic_labels(rho < 0.5)
    main_v = max(vsz, key=vsz.get)
    pockets = [k for k in vsz if k != main_v]
    for k in pockets:
        p.add_mesh(iso((vcomp == k).astype(float), 0.5, N), color='#d9822b', smooth_shading=True)
    scomp, ssz = periodic_labels(rho >= 0.5)
    main_s = max(ssz, key=ssz.get)
    specks = [k for k in ssz if k != main_s]
    for k in specks:
        c = (np.argwhere(scomp == k).mean(axis=0) + 0.5) / N
        p.add_mesh(pv.Sphere(radius=0.04, center=c), color='#d62728', opacity=0.5)
        p.add_mesh(iso((scomp == k).astype(float), 0.5, N), color='#d62728')
    p.enable_depth_peeling(number_of_peels=8)
    p.enable_anti_aliasing('ssaa')
    a, e, dist = np.radians(-60), np.radians(22), 3.4
    p.camera.focal_point = (0.5, 0.5, 0.5)
    p.camera.position = (0.5 + dist * np.cos(e) * np.cos(a), 0.5 + dist * np.cos(e) * np.sin(a), 0.5 + dist * np.sin(e))
    p.camera.up = (0, 0, 1)
    p.camera.view_angle = 30.0
    img = p.screenshot(return_img=True)
    p.close()
    info = (f'{len(pockets)} closed void pockets ' + (f'({", ".join(f"{100 * vsz[k] / rho.size:.1f}" for k in pockets)}\\,\\%)' if pockets else '')
            + f'\n{len(specks)} solid specks ' + (f'({", ".join(str(ssz[k]) for k in specks)} voxels)' if specks else ''))
    return img, info


plt.rcParams.update({'text.usetex': True, 'font.family': 'serif', 'font.size': 15})
fig, axs = plt.subplots(1, len(CASES), figsize=(max(5 * len(CASES), 8), 6.2), layout='constrained', squeeze=False)
axs = axs[0]
for ax, (fn, title) in zip(axs, CASES):
    img, info = render(load(fn))
    ax.imshow(img)
    ax.set_axis_off()
    ax.set_title(title, fontsize=16)
    ax.text(0.5, -0.02, info, transform=ax.transAxes, ha='center', va='top', fontsize=14)
    print(title, '|', info.replace('\n', ' | '))
fig.suptitle(os.environ.get('HEAD', 'converged 3D designs') + '\n' + r'solid $\rho \ge 0.5$ (translucent blue), closed void pockets (orange), '
             r'detached solid specks (red, in a sphere)', fontsize=17)
out = os.path.join(HERE, 'figures', os.environ.get('OUT', 'TO3D_pockets.png'))
fig.savefig(out, dpi=110)
print('wrote', out)
