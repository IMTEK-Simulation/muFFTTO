"""
N^3 Cahn-Hilliard start for the TO whose interface is the TO's own (user, 2026-11-15: "512 Cahn-Hilliard geometry
with the same interphase sharpness parameter", as the init guess of a 512^3 TO).

    dc/dt = Lap(c^3 - c - eps^2 Lap c), periodic, semi-implicit spectral steps with muFFTTO's FFT (discretization
    engine, GPU), discrete 7-point Laplacian symbol as in the Helmholtz H_0 of to3d_large.py.

Geometry: the bi_s11 structure of the chain's start (data/CH3D/ch3d_bi_s11.npy: 128^3 Cahn-Hilliard, eps 0.02,
black / white, 1 = stiff), nearest-neighbour upsampled to N^3, c = 1 - 2 rho. It is relaxed with
eps = sqrt(2) eta, eta = ETA_H / N, so that the Cahn-Hilliard interface c = tanh(x / (sqrt(2) eps)) is exactly the
TO's equilibrium profile rho = (1 + tanh(x / (2 eta))) / 2 of eta |grad rho|^2 + rho^2 (1 - rho)^2 / eta, for a time
T = T_EPS2 eps^2 (default 10): the interfaces reach their profile, the structure does not coarsen (coarsening needs
times ~ L^3 / eps, L ~ 0.2 the feature size). The chain's start (CH3D_start_bi_s11.nc) is the same structure blurred
by 1.5 voxels at 128^3, i.e. ~6 voxels at 512^3 instead of the TO's 2.

    DEVICE=gpu NPIX=512 python make_CH512_start.py   -> data/CH3D/CH<N>_start_bi_s11_eta<ETA_H>h.nc (INIT_FROM),
                                                        figures/CH<N>_start_bi_s11_eta<ETA_H>h.png (mid slices)
Env: NPIX (512), ETA_H (2), T_EPS2 (10), DT_EPS2 (0.03: dt = DT_EPS2 eps^2), DEVICE (gpu), OUT (output .nc path).
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
T_EPS2, DT_EPS2 = float(os.environ.get('T_EPS2', 10)), float(os.environ.get('DT_EPS2', 0.03))
DEVICE = os.environ.get('DEVICE', 'gpu')
eta = ETA_H / N
eps = np.sqrt(2) * eta
dt, steps = DT_EPS2 * eps ** 2, int(round(T_EPS2 / DT_EPS2))
label = f'CH{N}_start_bi_s11_eta{ETA_H:g}h'
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

# ---- start: the 128^3 black / white bi_s11 structure, nearest-neighbour upsampled
coarse = np.load(HERE / 'data' / 'CH3D' / 'ch3d_bi_s11.npy')                        # int8, 1 = stiff
factor = N // coarse.shape[0]
assert factor * coarse.shape[0] == N, (N, coarse.shape)
rho0 = xp.asarray(coarse, dtype=float)
for axis in range(3):
    rho0 = xp.repeat(rho0, factor, axis=axis)
c.s[0, 0] = 1.0 - 2.0 * rho0
del rho0
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
    'description': f'Cahn-Hilliard start bi_s11 at {N}^3 with the TO interface (eps = sqrt(2) eta, eta = {ETA_H:g} h, '
                   f'T = {T_EPS2:g} eps^2), make_CH512_start.py', 'eta': eta, 'ch_eps': eps})
print(f'saved {out}: stiff fraction (rho > 0.5) {float(xp.mean(rho > 0.5)):.4f}, mean rho {float(xp.mean(rho)):.4f}, '
      f'grey fraction (0.1 < rho < 0.9) {grey:.4f}, {time.time() - start_time:.0f} s', flush=True)

# ---- mid slices: the chain's start (128^3, blurred, bilinearly interpolated in the slice) vs the new start
old = np.asarray(io_utils.read_file(str(HERE / 'data' / 'CH3D' / 'CH3D_start_bi_s11.nc')).fields['phase_field_solution'][-1])
nc = int(round(old.size ** (1 / 3)))
old = old.reshape((nc,) * 3)[:, :, nc // 2]                 # the plane z = 1/2 (exactly a coarse plane)
for axis in range(2):
    position = np.arange(N) * (nc / N)
    base = np.floor(position).astype(int)
    fraction = (position - base).reshape([-1 if a == axis else 1 for a in range(2)])
    old = np.take(old, base % nc, axis=axis) * (1 - fraction) + np.take(old, (base + 1) % nc, axis=axis) * fraction
new = to_host(rho[:, :, N // 2])
zoom = slice(0, N // 4)
fig, axes = plt.subplots(2, 2, figsize=(14, 14))
for col, (image, name) in enumerate([(old, 'chain start: 128³ blurred, interpolated'),
                                     (new, f'new: Cahn-Hilliard at {N}³, interface eta = {ETA_H:g}h')]):
    axes[0, col].imshow(image.T, origin='lower', cmap='gray_r', vmin=0, vmax=1, interpolation='nearest')
    axes[0, col].set_title(f'{name}\nz = 1/2, grey (0.1-0.9) {np.mean((image > 0.1) & (image < 0.9)):.3f}', fontsize=13)
    axes[1, col].imshow(image[zoom, zoom].T, origin='lower', cmap='gray_r', vmin=0, vmax=1, interpolation='nearest')
    axes[1, col].set_title('zoom: first quarter of the cell', fontsize=13)
    for row in range(2):
        axes[row, col].set_xticks([]), axes[row, col].set_yticks([])
fig.tight_layout()
figure = HERE / 'figures' / f'{label}.png'
fig.savefig(figure, dpi=max(N // 7, 100))
print('saved', figure, flush=True)
