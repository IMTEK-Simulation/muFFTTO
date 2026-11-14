"""What 'snapping to the grid' means for an under-resolved phase-field interface.

Top left:  the 1D interface profile rho(s) = 1 / (1 + exp(-s / eta)) sampled on pixels, eta = 1h vs eta = 4h,
           for two positions of the interface inside a pixel (shift 0 and h/2).
(Checked and dropped: the discrete phase-field energy of the sampled profile hardly depends on the interface position
 inside a pixel, < 1e-3 % even at eta = 1h, so the snapping is not an energy pinning of the regularisation term; it is
 the representation: with 2-3 pixels across the interface, shapes can only change by whole pixels.)
Bottom:    final TO designs with the same physical eta = 1/128 (NEMO eta / contrast study, contrast 1e4, Green) on
           128^2 (eta = 1h), 256^2 (2h), 512^2 (4h); exact pixels, the same physical window; red: rho = 0.5 contour.
    python plot_grid_snapping.py  ->  figures/grid_snapping.png
"""
import os

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import netCDF4
import numpy as np

HERE = os.path.dirname(os.path.realpath(__file__))
D = os.path.join(HERE, 'data', 'example_2D_elasticity_TO', 'nemo_eta2h')
F = 'N{}_soft0.0001_energy-warm_Green_eta_0.0078125_w_5.0_l_bfgs_final.nc'


def profile(s, eta):
    return 1.0 / (1.0 + np.exp(-s / eta))


def energy(eta_h, delta):
    h, eta = 1.0, eta_h
    x = np.arange(-60, 61) * h
    r = profile(x - delta, eta)
    return np.sum(eta * np.diff(r) ** 2 / h) + np.sum(r ** 2 * (1 - r) ** 2 / eta * h)


plt.rcParams.update({'text.usetex': True, 'font.family': 'serif', 'font.size': 13})
fig = plt.figure(figsize=(15, 9.5), layout='constrained')
gs = fig.add_gridspec(2, 3, height_ratios=[0.8, 1])

# --- top: the same interface (length unit eta) on pixels of size h = eta (left) and h = eta/4 (right)
ax = fig.add_subplot(gs[0, :])
s_ = np.linspace(-6, 6, 800)
for eh, col, x0 in ((1, 'tab:red', -7.0), (4, 'tab:blue', 7.0)):
    for d, ls, mk in ((0.0, '-', 'o'), (0.5, '--', 's')):
        xs = np.arange(-6 * eh, 6 * eh + 1) / eh                  # pixel centres in units of eta
        yy = profile(xs - d / eh, 1.0)                            # shift by d pixels
        ax.plot(x0 + s_, profile(s_ - d / eh, 1.0), ls, color=col, lw=1.0, alpha=0.45)
        ax.step(x0 + xs, yy, where='mid', color=col, ls=ls, lw=2.0)
        ax.plot(x0 + xs, yy, mk, color=col, ms=4 if eh == 4 else 6, mfc='white' if d else col)
    ax.text(x0, 1.1, rf'$\eta = {eh}h$: {eh} pixel{"s" if eh > 1 else ""} per $\eta$, '
            rf'{round(4.39 * eh)} pixels across the band', color=col, ha='center', fontsize=15)
ax.set_xticks([x0 + t for x0 in (-7, 7) for t in (-4.39 / 2, 4.39 / 2)])
ax.set_xticklabels([r'$-2.2\eta$', r'$+2.2\eta$'] * 2)
ax.set_ylim(-0.08, 1.2), ax.set_ylabel(r'$\rho$')
ax.plot([], [], 'k-', lw=2.0, label='interface centred on a pixel')
ax.plot([], [], 'k--', lw=2.0, label=r'interface moved by half a pixel')
ax.legend(loc='center', fontsize=11)
ax.set_title(r'the same interface, moved by half a pixel (dashed): at $\eta = 1h$ every pixel value jumps, '
             r'at $\eta = 4h$ the move is resolved')

# --- bottom: designs at the same physical eta
win = (0.0, 0.5)                                       # physical window [x0, x0 + 0.5]^2
with netCDF4.Dataset(os.path.join(D, F.format(512))) as d:
    ref = np.array(d['phase_field_solution'][0, 0])
for j, (N, eh) in enumerate(((128, 1), (256, 2), (512, 4))):
    ax = fig.add_subplot(gs[1, j])
    with netCDF4.Dataset(os.path.join(D, F.format(N))) as d:
        r = np.array(d['phase_field_solution'][0, 0])
    # align to the 512^2 design (best periodic shift, nearest upsampling: the pixels stay exact)
    up = np.kron(r, np.ones((512 // N, 512 // N)))
    c = np.real(np.fft.ifft2(np.fft.fft2(ref - ref.mean()) * np.conj(np.fft.fft2(up - up.mean()))))
    sx, sy = np.unravel_index(np.argmax(c), c.shape)
    r = np.roll(r, (int(round(sx * N / 512)), int(round(sy * N / 512))), axis=(0, 1))
    n0, n1 = int(win[0] * N), int(win[1] * N)
    sub = r[n0:n1, n0:n1]
    ext = (win[0], win[1], win[0], win[1])
    ax.imshow(sub.T, origin='lower', cmap='Greys', vmin=0, vmax=1, interpolation='nearest', extent=ext)
    xc = (np.arange(n0, n1) + 0.5) / N
    ax.contour(xc, xc, sub.T, levels=[0.5], colors='tab:red', linewidths=1.2)
    ax.set_xticks([]), ax.set_yticks([])
    ax.set_title(rf'${N}^2$: $\eta = 1/128 = {eh}h$' + ('  (snapped)' if eh == 1 else ''), fontsize=14)
fig.suptitle(r'Snapping to the grid: an under-resolved interface ($\eta \leq 1h$) follows the pixel lattice, '
             r'not the physics', fontsize=16)
out = os.path.join(HERE, 'figures', 'grid_snapping.png')
fig.savefig(out, dpi=120)
print('wrote', out)
