"""Convergence of the 50 MPa chain (t_r at dt = 1e-8, mesh N = 96 / 128) and test of V against the plate Coble rate
V = 3 B sigma / a^2, B = D_gb delta Omega / (kB T), a = neck / 2. Writes figures/chain_convergence.png, _coble.png"""
import glob, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
L0 = 90.24
B = 5.5e-11 * 2e-10 * 1.22e-29 / (1.380649e-23 * 700.0)         # m^6/(J s)
def cr(l):
    s = l - .5; i = np.where(s[:-1] * s[1:] < 0)[0]; return i + s[i] / (s[i] - s[i + 1])
def load(N, tag):
    fs = sorted(glob.glob(f'exp_data/SINT_N{N}_Green_W4_P50u_k0.0001_{tag}_snaps/step_*.npz'), key=lambda f: int(os.path.basename(f)[5:-4]))
    a = []
    for f in fs:
        d = np.load(f); r = d['rho'].astype(float); k = cr(r[:, N // 2])
        a.append((float(d['t']) * 1e6, float(d['s']) * L0, (k.max() - k.min()) * L0 / N, float(d['s']) * r.mean(), float(d['sigma_gb']) / 1e6, float(d['V']), r.max()))
    return np.array(a).T
def fit(T, L, m): return -np.polyfit(T[m] / 1e3, L[m], 1)[0] if m.sum() > 1 else float('nan')
runs = [('t_r 1e-6, dt 1e-8', 96, 'chain_P50_tau1e-6_dt1e-8', 1e-6, '#d95f02', '-'), ('t_r 5e-7, dt 1e-8', 96, 'chain_P50_tau5e-7_dt1e-8', 5e-7, '#7b3294', '--'),
        ('t_r 2e-7, dt 1e-8', 96, 'chain_P50_tau2e-7_dt1e-8', 2e-7, '#1f77b4', '-.'), ('t_r 1e-7, dt 1e-8', 96, 'chain_P50_tau1e-7_dt1e-8', 1e-7, '#111111', '-'),
        ('N=128, t_r 2e-7', 128, 'chain_P50_tau2e-7_dt1e-8', 2e-7, '#2a9d8f', ':'), ('t_r 1e-6, dt 1e-7', 96, 'chain_P50_tau1e-6', 1e-6, '#d95f02', ':')]
plt.rcParams.update({'font.size': 13})
fig, ax = plt.subplots(1, 2, figsize=(12, 4.7), constrained_layout=True)
fc, axc = plt.subplots(figsize=(6.8, 4.8), constrained_layout=True)
print(f'{"run":20s} L_end neck_end rate@42 rate@46-50 mass rho_max')
for lab, N, tag, tr, c, ls in runs:
    T, L, Nk, M, S, V, R = load(N, tag)
    print(f'{lab:20s} {L[-1]:.2f} {Nk[-1]:.1f} {fit(T, L, (abs(Nk - 42) <= 2) & (T > 50)):.2f} {fit(T, L, (Nk >= 46) & (Nk <= 50)):.2f} {M[-1] / M[0] - 1:.1e} {R.max():.3f}')
    ax[0].plot(T, L, color=c, ls=ls, lw=2, label=lab); ax[1].plot(T, Nk, color=c, ls=ls, lw=2)
    if tag == 'chain_P50_tau1e-7_dt1e-8':
        a = Nk * 1e-9 / 2; Vc = 3 * B * S * 1e6 / a ** 2 * 1e9
        w = T > 50
        axc.plot(T[w], V[w], color='#111111', lw=2, label='model V (t_r 1e-7)')
        axc.plot(T[w], Vc[w], color='#d95f02', ls='--', lw=2, label=r'Coble plate $3B\sigma/a^2$, stress only')
        print('Coble check t[us] V_model V_coble ratio:')
        for t in (100, 400, 800, 1200, 1600, 2000):
            i = np.argmin(abs(T - t)); print(f'  {T[i]:6.0f} neck {Nk[i]:.1f} sigma {S[i]:.1f} MPa  V {V[i]:.0f}  Vc {Vc[i]:.0f}  {V[i] / Vc[i]:.2f}')
for a_, y in zip(ax, ('chain period L [nm]', 'neck width [nm]')):
    a_.set_xlabel(r'time [$\mu$s]'); a_.set_ylabel(y); a_.set_xlim(left=0); a_.grid(alpha=.3)
ax[0].legend(fontsize=11)
axc.set_xlabel(r'time [$\mu$s]'); axc.set_ylabel('closing velocity V [nm/s]'); axc.set_xlim(left=0); axc.set_ylim(bottom=0); axc.grid(alpha=.3); axc.legend()
fig.savefig('figures/chain_convergence.png', dpi=150); fc.savefig('figures/chain_coble.png', dpi=150)
