"""Crack profile of the straight notch crack (NOTCH=0,0.5,0.5, load yy, l = 1/32) vs theory.
Columns x in [0.6, 0.9] (matrix, crack grown from the notch tip), s = vertical distance from the column's crack nodes
{d >= 0.99}. Theory: AT2 exp(-s/l); threshold-AT1 exp(-s/(sqrt2 l)); exact AT1 (1 - s/2l)^2 for s < 2l.
    python check_crack_profile_notch.py  ->  figures/crack_profile_notch.png"""
import os, numpy as np, netCDF4, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
H = os.path.dirname(os.path.abspath(__file__)); L = 0.03125; Gc = 5e-4
fig, axs = plt.subplots(1, 2, figsize=(11, 4.3))
for k, model in enumerate(('AT1', 'AT2')):
    ax = axs[k]
    for N, mk in ((128, 'o'), (256, 's')):
        with netCDF4.Dataset(os.path.join(H, f'exp_data/{model}_N{N}_dual_l0.03125_notch.nc')) as ds:
            d = np.asarray(ds['damage'][-1, 0]); Hq = np.asarray(ds['history_qp'][-1]).reshape(-1)
        y = np.arange(N) / N
        cols = range(int(0.6 * N), int(0.9 * N))
        ss, dd, yc = [], [], []
        for i in cols:
            c = np.where(d[i] >= 0.99)[0]
            if len(c) == 0: continue
            yc.append(y[c].mean())
            dist = np.min(np.abs(((y[:, None] - y[c][None, :]) + 0.5) % 1.0 - 0.5), axis=1)
            m = d[i] < 0.99
            ss.append(dist[m] / L); dd.append(d[i][m])
        s, dv = np.concatenate(ss), np.concatenate(dd)
        bins = np.arange(0, 8.01, 4.0 / (L * N)); sc = 0.5 * (bins[1:] + bins[:-1])
        idx = np.digitize(s, bins) - 1
        med = np.array([np.median(dv[idx == j]) if (idx == j).sum() > 3 else np.nan for j in range(len(sc))])
        f = (med > 0.03) & (med < 0.6)
        lam = -1 / np.polyfit(sc[f], np.log(med[f]), 1)[0]
        psi_c = 3 * Gc / (16 * L)
        far = np.median(dv[s > 6]) if (s > 6).any() else np.nan
        print(f'{model} {N}^2 (l/h {L*N:g}): crack y = {np.min(yc):.3f}-{np.max(yc):.3f}; fitted decay length {lam:.2f} l; '
              f'd(s=1 l) {np.interp(1, sc, med):.3f}, d(s=2 l) {np.interp(2, sc, med):.3f}; far d (s>6 l) {far:.4f}; '
              f'max H / psi_c(AT1) in cell {np.percentile(Hq, 50) / psi_c:.2e} (median)')
        ax.plot(sc, med, mk + '-', ms=3, label=f'{N}$^2$ (l/h = {L*N:g}): decay {lam:.2f} l')
    t = np.linspace(0, 8, 400)
    if model == 'AT1':
        ax.plot(t, np.exp(-t / np.sqrt(2)), 'k--', label=r'threshold AT1: $e^{-s/(\sqrt{2}l)}$')
        ax.plot(t, np.where(t < 2, (1 - t / 2) ** 2, 0), 'k:', label=r'exact AT1: $(1-s/2l)^2$, width 4l')
    else:
        ax.plot(t, np.exp(-t), 'k--', label=r'AT2: $e^{-s/l}$')
    ax.set_yscale('log'); ax.set_ylim(1e-4, 1.1); ax.set_xlim(0, 8); ax.grid(alpha=0.3)
    ax.set_xlabel('distance from crack centre  s / l'); ax.set_ylabel('damage d'); ax.set_title(f'{model}, l = 1/32, straight crack')
    ax.legend(fontsize=8)
out = os.path.join(H, 'figures/crack_profile_notch.png'); fig.tight_layout(); fig.savefig(out, dpi=150); print(out)
