"""Grain-boundary profile at 50 MPa: mu(x) - mu(edge) [MPa] from the phase field against the Coble parabola
V (a^2 - x^2) / (2 B), and the elastic -sigma_yy(x) of the welded solve. Writes figures/chain_gb_profile.png"""
import glob, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
L0 = 90.24; N = 96; h = L0 / N
B = 5.5e-11 * 2e-10 * 1.22e-29 / (1.380649e-23 * 700.0)
def cr(l):
    s = l - .5; i = np.where(s[:-1] * s[1:] < 0)[0]; return i + s[i] / (s[i] - s[i + 1])
fs = sorted(glob.glob('exp_data/SINT_N96_Green_W4_P50u_k0.0001_profile_snaps/step_*.npz'), key=lambda f: int(os.path.basename(f)[5:-4]))
plt.rcParams.update({'font.size': 13})
show = fs[1:]
fig, axes = plt.subplots(1, len(show), figsize=(5.2 * len(show), 4.6), constrained_layout=True, squeeze=False)
for ax, f in zip(axes[0], show):
    d = np.load(f); r = d['rho'].astype(float); k = cr(r[:, N // 2]) * h; x0 = .5 * (k.min() + k.max()); a = .5 * (k.max() - k.min())
    x = np.arange(N) * h - x0; row = d['mu'][:, N // 2] / 1e6; syy = -d['syy'][:, N // 2] / 1e6; V = float(d['V'])
    edge = np.interp(a, x, row); inside = np.abs(x) <= a
    par = V * 1e-9 * ((a * 1e-9) ** 2 - (x * 1e-9) ** 2) / (2 * B) / 1e6
    m_mean, p_mean = (row[inside] - edge).mean(), par[inside].mean()
    print(f't {float(d["t"]) * 1e6:.0f} us: a {a:.1f} nm V {V:.0f} nm/s, mean mu-edge {m_mean:.1f} MPa, parabola mean {p_mean:.1f}, centre mu-edge {np.interp(0, x, row) - edge:.1f}, parabola centre {np.interp(0, x, par):.1f}, sigma_gb {float(d["sigma_gb"]) / 1e6:.1f}, elastic -syy centre {np.interp(0, x, syy):.1f}, edge-region max {syy[inside].max():.1f}')
    ax.plot(x[inside], row[inside] - edge, color='#111111', lw=2, marker='o', ms=5, label=r'phase field $\mu-\mu_{\rm edge}$')
    ax.plot(x[inside], par[inside], color='#d95f02', ls='--', lw=2, label='Coble parabola')
    ax.plot(x[inside], syy[inside], color='#1f77b4', ls='-.', lw=2, label=r'elastic $-\sigma_{yy}$ (welded)')
    ax.set_xlabel('x along boundary [nm]'); ax.set_ylabel('MPa'); ax.set_title(f't = {float(d["t"]) * 1e6:g} $\\mu$s'); ax.grid(alpha=.3)
axes[0, 0].legend(fontsize=11)
fig.savefig('figures/chain_gb_profile.png', dpi=150)
