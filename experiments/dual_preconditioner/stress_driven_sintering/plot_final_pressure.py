"""Final 2-grain chain series: t_r = 1e-7 s, dt = 1e-8 s, N = 96. V against the Coble plate rate for every P.
Writes figures/chain_final.png, chain_final_snapshots.png"""
import glob, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
L0 = 90.24; N = 96
B = 5.5e-11 * 2e-10 * 1.22e-29 / (1.380649e-23 * 700.0)
def cr(l):
    s = l - .5; i = np.where(s[:-1] * s[1:] < 0)[0]; return i + s[i] / (s[i] - s[i + 1])
def files(p):
    return sorted(glob.glob(f'exp_data/SINT_N96_Green_W4_P{p}u_k0.0001_chain_P{p}_tau1e-7_dt1e-8_snaps/step_*.npz'), key=lambda f: int(os.path.basename(f)[5:-4]))
def load(p):
    a = []
    for f in files(p):
        d = np.load(f); r = d['rho'].astype(float); k = cr(r[:, N // 2])
        a.append((float(d['t']) * 1e6, float(d['s']) * L0, (k.max() - k.min()) * L0 / N, float(d['s']) * r.mean(), float(d['sigma_gb']) / 1e6, float(d['V']), r.max(), min(np.sum(r * e) for e in d['etas'].astype(float))))
    return np.array(a).T
def fit(T, L, m): return -np.polyfit(T[m] / 1e3, L[m], 1)[0] if m.sum() > 1 else float('nan')
ps = [0, 25, 50, 100]; col = ['#111111', '#1f77b4', '#d95f02', '#7b3294']; ls = [':', '--', '-.', '-']
plt.rcParams.update({'font.size': 13})
fig, ax = plt.subplots(1, 3, figsize=(15, 4.6), constrained_layout=True)
res = {}
for p, c, l in zip(ps, col, ls):
    T, L, Nk, M, S, V, R, A = load(p); res[p] = (T, L, Nk)
    a = Nk * 1e-9 / 2; Vc = 3 * B * S * 1e6 / a ** 2 * 1e9
    w = T >= 100
    print(f'P {p:3d}: L_end {L[-1]:.2f} neck_end {Nk[-1]:.1f} mass {M[-1] / M[0] - 1:.1e} rho_max {R.max():.3f} min grain {A.min():.0f}/{A[0]:.0f} sigma/FB end {S[-1]:.1f}/{p * L0 / Nk[-1]:.1f}  V/Vc min-max ' + (f'{(V[w] / Vc[w]).min():.2f}-{(V[w] / Vc[w]).max():.2f}' if p else 'n/a') + f'  V_end {V[-1]:.0f}')
    lab = f'{p} MPa'
    ax[0].plot(T, L, color=c, ls=l, lw=2, label=lab); ax[1].plot(T, Nk, color=c, ls=l, lw=2)
    if p: ax[2].plot(T[w], V[w] / Vc[w], color=c, ls=l, lw=2)
print('rate -dL/dt [nm/ms] at equal neck, P=', ps)
for n in (38, 42, 46, 50):
    print(f'neck {n}: ' + '  '.join(f'{fit(res[p][0], res[p][1], (abs(res[p][2] - n) <= 2) & (res[p][0] > 50)):7.2f}' for p in ps))
for a_, y in zip(ax, ('chain period L [nm]', 'neck width [nm]', r'$V / V_{\rm Coble}$ (stress only)')):
    a_.set_xlabel(r'time [$\mu$s]'); a_.set_ylabel(y); a_.set_xlim(left=0); a_.grid(alpha=.3)
ax[2].set_ylim(0, 2)
h, lb = ax[0].get_legend_handles_labels(); fig.legend(h, lb, loc='outside upper center', ncol=4)
fig.savefig('figures/chain_final.png', dpi=150)
fig, axes = plt.subplots(2, 3, figsize=(11, 7.6), constrained_layout=True)
for r, p in enumerate((50, 100)):
    fs = files(p)
    for c, i in enumerate((0, len(fs) // 2, len(fs) - 1)):
        d = np.load(fs[i]); Ly = L0 * float(d['s'])
        im = axes[r, c].imshow(d['rho'].T, origin='lower', cmap='gray_r', vmin=0, vmax=1, interpolation='nearest', extent=(0, L0, 0, Ly))
        axes[r, c].set_ylim(0, L0); axes[r, c].set_title(f'{p} MPa, t = {float(d["t"]) * 1e6:g} $\\mu$s')
        axes[r, c].set_xlabel('x [nm]'); axes[r, c].set_ylabel('y [nm]')
fig.colorbar(im, ax=axes, label=r'$\rho$', shrink=0.8)
fig.savefig('figures/chain_final_snapshots.png', dpi=150)
