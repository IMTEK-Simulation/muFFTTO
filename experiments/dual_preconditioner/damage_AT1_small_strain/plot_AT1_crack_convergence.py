"""AT1 B2 (nine pores, l = 1/32): crack pattern per grid (damage at peak stress and at the end, pores outlined)
+ stress-strain mesh convergence (AT2 B2 dashed).
    python plot_AT1_crack_convergence.py [N ...]   (default 32 64)  ->  figures/AT1_crack_convergence.png"""
import os, sys, numpy as np, netCDF4, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
H = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(H, '../../..'))
from muFFTTO import microstructure_library as ml
A2 = os.path.join(H, '../damage_AT2_small_strain/exp_data')
Ns = [int(n) for n in sys.argv[1:]] or [32, 64]
def load(f):
    with netCDF4.Dataset(f) as d:
        return np.asarray(d['load'][:]), np.asarray(d['macro_stress'][:, 0, 0]), np.asarray(d['damage'][:, 0])
x = (np.arange(1024) + 0.5) / 1024
pore = np.asarray(ml.get_geometry(nb_voxels=(1024, 1024), microstructure_name='polydisperse_pores',
                                  coordinates=np.array(np.meshgrid(x, x, indexing='ij'))))
fig = plt.figure(figsize=(12.5, 5.6 * len(Ns) / 2 + 1.0))
gs = fig.add_gridspec(len(Ns), 3, width_ratios=[1, 1, 1.25], wspace=0.08, hspace=0.18)
cols = ['#1f77b4', '#d62728', '#2ca02c', '#9467bd']
axc = fig.add_subplot(gs[:, 2])
for k, N in enumerate(Ns):
    L, s, D = load(os.path.join(H, f'exp_data/AT1_N{N}_dual_polydisperse_pores_l0.03125.nc'))
    ip = int(s.argmax())
    for j, (i, what) in enumerate(((ip, 'at peak'), (len(L) - 1, 'final'))):
        ax = fig.add_subplot(gs[k, j])
        im = ax.imshow(D[i].T, origin='lower', cmap='Reds', vmin=0, vmax=1, interpolation='nearest', extent=(0, 1, 0, 1))
        ax.contour(x, x, pore.T, levels=[0.5], colors='k', linewidths=0.8)
        ax.set_title(f'{N}$^2$ (l/h = {N / 32:g}), {what}: E$_{{xx}}$ = {L[i]:.3f}', fontsize=10)
        ax.set_xticks([]); ax.set_yticks([]); ax.set_aspect('equal')
    axc.plot(L, s * 1e2, color=cols[k], lw=2, label=f'AT1 {N}$^2$  (peak {s.max()*1e2:.3f})')
    axc.plot(L[ip], s[ip] * 1e2, 'o', color=cols[k])
    f2 = os.path.join(A2, f'AT2_N{N}_dual_polydisperse_pores_l0.03125.nc')
    if os.path.exists(f2):
        L2, s2, _ = load(f2); axc.plot(L2, s2 * 1e2, '--', color=cols[k], lw=1.2, label=f'AT2 {N}$^2$  (peak {s2.max()*1e2:.3f})')
cax = fig.add_axes([0.06, 0.12, 0.012, 0.76]); fig.colorbar(im, cax=cax, label='damage d'); cax.yaxis.set_label_position('left'); cax.yaxis.tick_left()
axc.set_xlabel('macroscopic strain E$_{xx}$'); axc.set_ylabel(r'macroscopic stress $\sigma_{xx}\ \times 10^{2}$')
axc.set_xlim(0, 0.085); axc.set_ylim(bottom=0); axc.grid(alpha=0.3); axc.legend(fontsize=9)
axc.set_title('mesh convergence, fixed l = 1/32 (dots: peak)', fontsize=10)
out = os.path.join(H, 'figures/AT1_crack_convergence.png'); fig.savefig(out, dpi=150, bbox_inches='tight'); print(out)
