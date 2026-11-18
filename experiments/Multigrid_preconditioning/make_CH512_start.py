"""
Initial geometry for the TO from the Cahn-Hilliard evolution at N^3, with the TO's interface parameter
(user, 2026-11-15: run the Cahn-Hilliard evolution equation with the same interface width to generate the start).

    dc/dt = Lap(c^3 - c - eps^2 Lap c), periodic, semi-implicit spectral steps with muFFTTO's FFT (discretization
    engine, GPU), discrete 7-point Laplacian symbol as in the Helmholtz H_0 of to3d_large.py.

Same procedure as the 128^3 geometry bi_s11 of make_new_CH_geometries.py (random start c = 0.05 N(0, 1) around the
mean 0, seed 11, time step 0.0625 eps^2, end time 250 eps^2), but on the N^3 grid and with eps from the TO:
eps = EPS_ETA x eta, eta = ETA_H / N. Default EPS_ETA = sqrt(2): the Cahn-Hilliard interface
c = tanh(x / (sqrt(2) eps)) is then exactly the TO's equilibrium profile rho = (1 + tanh(x / (2 eta))) / 2 of
eta |grad rho|^2 + rho^2 (1 - rho)^2 / eta (same interface width). The structure size scales with eps, so at 512^3 it is
~3.6x finer than bi_s11 (eps 0.02). Design rho = (1 - c) / 2 (rho = 1 stiff where c < 0, as bi_s11), no threshold.

    DEVICE=gpu NPIX=512 python make_CH512_start.py   -> data/CH3D/CH<N>_start_s11_eta<ETA_H>h.nc (or OUT), INIT_FROM
                                                        figures/CH<N>_start_s11_eta<ETA_H>h.png (mid slice + zoom)
Env: NPIX (512), ETA_H (2), EPS_ETA (sqrt 2), T_EPS2 (250: end time / eps^2), DT_EPS2 (0.0625), SEED (11),
     DEVICE (gpu), OUT.
"""
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent))
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

from muFFTTO import domain, io_utils
from muFFTTO.tensor_operations import to_host

N = int(os.environ.get('NPIX', 512))
ETA_H = float(os.environ.get('ETA_H', 2))
T_EPS2, DT_EPS2 = float(os.environ.get('T_EPS2', 250)), float(os.environ.get('DT_EPS2', 0.0625))
EPS_ETA, SEED = float(os.environ.get('EPS_ETA', np.sqrt(2))), int(os.environ.get('SEED', 11))
DEVICE = os.environ.get('DEVICE', 'gpu')
eta = ETA_H / N
eps = EPS_ETA * eta
dt, steps = DT_EPS2 * eps ** 2, int(round(T_EPS2 / DT_EPS2))
label = f'CH{N}_start_s{SEED}_eta{ETA_H:g}h'
out = Path(os.environ.get('OUT', HERE / 'data' / 'CH3D' / f'{label}.nc'))
start_time = time.time()

cell = domain.PeriodicUnitCell(domain_size=[1, 1, 1], problem_type='elasticity')
disc = domain.Discretization(cell=cell, nb_of_pixels_global=(N,) * 3, discretization_type='finite_element',
                             element_type='trilinear_hexahedron', device=DEVICE)
assert disc.communicator.size == 1, 'serial script (one GPU)'
xp = __import__('cupy') if DEVICE == 'gpu' else np
c = disc.get_scalar_field(name='ch_c')
nonlinear = disc.get_scalar_field(name='ch_nonlinear')
c_hat = disc.ffield_collection.complex_field(name='ch_c_hat', components=(1,))
n_hat = disc.ffield_collection.complex_field(name='ch_n_hat', components=(1,))
h = 1.0 / N
neg_laplace = xp.asarray(sum(4 / h ** 2 * np.sin(np.pi * np.asarray(to_host(disc.fft.ifftfreq[a]), dtype=float) / N)
                             ** 2 for a in range(3)))
denominator = 1.0 + dt * eps ** 2 * neg_laplace ** 2
normalisation = disc.fft.normalisation

# ---- start: random noise around the mean 0 (as make_new_CH_geometries.py)
c.s[0, 0] = xp.asarray(0.05 * np.random.default_rng(SEED).standard_normal((N,) * 3))
print(f'{label}: N {N}, eta {eta:.5g} (= {ETA_H:g} h), eps {eps:.5g}, dt {dt:.3g}, {steps} steps to T {T_EPS2:g} eps^2, '
      f'stiff fraction {float(xp.mean(c.s[0, 0] < 0)):.4f}', flush=True)

for step in range(1, steps + 1):
    cs = c.s[0, 0]
    nonlinear.s[0, 0] = cs * cs * cs - cs
    disc.fft.fft(c, c_hat)
    disc.fft.fft(nonlinear, n_hat)
    c_hat.s[0, 0] -= dt * neg_laplace * n_hat.s[0, 0]
    c_hat.s[0, 0] /= denominator
    disc.fft.ifft(c_hat, c)
    c.s[0, 0] *= normalisation
    if step % max(steps // 10, 1) == 0 or step == steps:
        cs = c.s[0, 0]
        print(f'step {step}: c in [{float(cs.min()):+.3f}, {float(cs.max()):+.3f}], mean {float(cs.mean()):+.5f}, '
              f'grey fraction (|c| < 0.8) {float(xp.mean(xp.abs(cs) < 0.8)):.4f}', flush=True)

# ---- design rho = (1 - c) / 2 in [0, 1]; the interface of the TO: check the grey band against the chain's start
solution = disc.get_scalar_field(name='phase_field_solution')
solution.s[0, 0] = xp.clip((1.0 - c.s[0, 0]) / 2.0, 0.0, 1.0)
rho = solution.s[0, 0]
grey = float(xp.mean((rho > 0.1) & (rho < 0.9)))
out.parent.mkdir(parents=True, exist_ok=True)
io_utils.save_fields(str(out), [solution], attributes={
    'description': f'Cahn-Hilliard evolution at {N}^3 from noise (seed {SEED}) with the TO interface (eps = {EPS_ETA:.4g} '
                   f'eta, eta = {ETA_H:g} h, T = {T_EPS2:g} eps^2), make_CH512_start.py', 'eta': eta, 'ch_eps': eps})
print(f'saved {out}: stiff fraction (rho > 0.5) {float(xp.mean(rho > 0.5)):.4f}, mean rho {float(xp.mean(rho)):.4f}, '
      f'grey fraction (0.1 < rho < 0.9) {grey:.4f}, {time.time() - start_time:.0f} s', flush=True)

# ---- mid slice z = 1/2 and a zoom (exact pixels)
new = to_host(rho[:, :, N // 2])
zoom = slice(0, N // 4)
fig, axes = plt.subplots(1, 2, figsize=(14, 7.4))
axes[0].imshow(new.T, origin='lower', cmap='gray_r', vmin=0, vmax=1, interpolation='nearest')
axes[0].set_title(f'Cahn-Hilliard at {N}³, eps = {EPS_ETA:.3g} eta, eta = {ETA_H:g}h, T = {T_EPS2:g} eps²\n'
                  f'z = 1/2, stiff {float(np.mean(new > 0.5)):.2f}, grey (0.1-0.9) {grey:.3f}', fontsize=13)
axes[1].imshow(new[zoom, zoom].T, origin='lower', cmap='gray_r', vmin=0, vmax=1, interpolation='nearest')
axes[1].set_title('zoom: first quarter of the cell', fontsize=13)
for ax in axes:
    ax.set_xticks([]), ax.set_yticks([])
fig.tight_layout()
figure = HERE / 'figures' / f'{label}.png'
fig.savefig(figure, dpi=max(N // 7, 100))
print('saved', figure, flush=True)
