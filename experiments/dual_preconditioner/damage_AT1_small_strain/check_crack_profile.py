"""Measured crack profile vs theory (AT1: d = (1 - s/2l)^2, support 2l per side; AT2: d = exp(-s/l)).
Final damage field, s = distance of a node from the crack centre {d >= 0.99} (periodic), pore pixels and pixels closer
to a pore than to the crack excluded; median d per s bin.
    python check_crack_profile.py   ->  figures/AT1_AT2_crack_profile.png + numbers in the terminal"""
import os, sys, numpy as np, netCDF4, matplotlib
from scipy.ndimage import distance_transform_edt
matplotlib.use('Agg'); import matplotlib.pyplot as plt
H = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(H, '../../..'))
from muFFTTO import microstructure_library as ml
A2 = os.path.join(H, '../damage_AT2_small_strain/exp_data')
L = 0.03125
CASES = [('AT1', 128, os.path.join(H, 'exp_data/AT1_N128_dual_polydisperse_pores_l0.03125.nc')),
         ('AT1', 256, os.path.join(H, 'exp_data/nemo/AT1_N256_dual_polydisperse_pores_l0.03125.nc')),
         ('AT2', 128, os.path.join(A2, 'AT2_N128_dual_polydisperse_pores_l0.03125.nc')),
         ('AT2', 256, os.path.join(A2, 'AT2_N256_dual_polydisperse_pores_l0.03125.nc'))]


def periodic_edt(mask, h):
    """distance (in units of the cell) to the nearest True node of a periodic mask"""
    n = mask.shape[0]
    big = np.tile(~mask, (3, 3))
    return distance_transform_edt(big)[n:2 * n, n:2 * n] * h


fig, axs = plt.subplots(1, 2, figsize=(11, 4.2))
print(f'l = {L} (1/32); theory: AT1 d = 0 for s >= 2l = {2 * L:.4f}, AT2 decay length l')
for model, N, f in CASES:
    with netCDF4.Dataset(f) as ds:
        d = np.asarray(ds['damage'][-1, 0])
    h = 1.0 / N
    x = np.arange(N) / N                                   # nodes at the lower-left pixel corners
    pore = np.asarray(ml.get_geometry(nb_voxels=(N, N), microstructure_name='polydisperse_pores',
                                      coordinates=np.array(np.meshgrid(x, x, indexing='ij')))) == 0
    s_crack = periodic_edt(d >= 0.99, h)
    s_pore = periodic_edt(pore, h)
    keep = (~pore) & (s_pore > s_crack) & (d < 0.99)
    s, dv = s_crack[keep] / L, d[keep]                     # s in units of l
    bins = np.arange(0, 6.01, 0.25 if N == 128 else 0.125)
    idx = np.digitize(s, bins) - 1
    sc = 0.5 * (bins[1:] + bins[:-1])
    med = np.array([np.median(dv[idx == i]) if (idx == i).sum() > 5 else np.nan for i in range(len(sc))])
    ax = axs[0 if model == 'AT1' else 1]
    ax.plot(sc, med, 'o-', ms=3, label=f'{model} {N}$^2$ (l/h = {L * N:g}), median')
    # measured widths
    if model == 'AT1':
        s_zero = sc[np.argmax(med < 0.01)]                 # first bin with median d < 0.01
        tail = np.nanmedian(dv[(s > 2.5) & (s < 4)])
        print(f'{model} {N}^2: median d < 0.01 from s = {s_zero:.2f} l (theory 2 l); '
              f'd at s = 1 l: {np.interp(1, sc, med):.3f} (theory 0.25); median d at 2.5-4 l (tail): {tail:.4f}')
    else:
        m = (med > 0.03) & (med < 0.6)
        slope = np.polyfit(sc[m], np.log(med[m]), 1)[0]
        print(f'{model} {N}^2: decay length from fit of log d (0.03 < d < 0.6): {-1 / slope:.2f} l (theory 1 l); '
              f'd at s = 1 l: {np.interp(1, sc, med):.3f} (theory 0.368)')
    print(f'      centre set d >= 0.99: {(d >= 0.99).sum()} nodes; pixels used {keep.sum()}')
ss = np.linspace(0, 6, 300)
axs[0].plot(ss, np.where(ss < 2, (1 - ss / 2) ** 2, 0), 'k--', label='theory $(1 - s/2l)^2$')
axs[1].plot(ss, np.exp(-ss), 'k--', label='theory $e^{-s/l}$')
for ax, t in zip(axs, ('AT1', 'AT2')):
    ax.set_xlabel('distance from crack centre  s / l'); ax.set_ylabel('damage d'); ax.set_title(f'{t}, l = 1/32')
    ax.set_xlim(0, 6); ax.set_ylim(0, 1.02); ax.grid(alpha=0.3); ax.legend(fontsize=8)
axs[1].set_yscale('log'); axs[1].set_ylim(1e-3, 1.1)
out = os.path.join(H, 'figures/AT1_AT2_crack_profile.png'); fig.tight_layout(); fig.savefig(out, dpi=150); print(out)
