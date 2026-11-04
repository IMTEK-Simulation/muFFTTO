"""
2D: power-law extrapolation work ~ a N^b of the contrast study (plot_AT2_contrast_map_2D.py, fits over 32^2 - 128^2) to
256^2, 512^2 and 1024^2, per material setting and preconditioner, for a sharp (fixed l/h = 2) and a smooth (fixed
l = 1/16) crack. Work per elastic solve = CG iterations x K applications per iteration (dual 2), mean over the run.

Check: the full 256^2 runs of run_AT2_smoothness_2D_256.sh (every setting, both lines) as filled diamonds; for the
sharp crack also the real 256^2 / 512^2 runs of the earlier fixed-l/h study (run_AT2_incl_lh_256.sh,
run_overnight_2D_512_cpu.sh; pores 1e-3 + damage and inclusions 10x; no Green-Jacobi at 512^2) as open diamonds.

usage:  [PHASE=pre|cracking|post] python plot_AT2_contrast_extrapolation_2D.py
        writes figures/AT2_contrast_extrapolation_2D[_<PHASE>].png, prints the table (PHASE: plot_AT2_contrast_map_2D.py)
"""
import os

import numpy as np
import matplotlib
import matplotlib.ticker

matplotlib.use('Agg')
import matplotlib.pyplot as plt

from plot_AT2_contrast_map_2D import (GRIDS, LENGTHS, PRECS, SETTINGS, SHORT, W, HERE, work, phase_work,  # noqa: E402
                                      PHASE_TITLE, SUFFIX)

EXTRA = (256, 512, 1024)
CHECK = {1: '', 3: '_incl10'}          # settings with real 256^2 / 512^2 runs (fixed l/h = 2, suffix _lh)


def check_work(n, p, tag):
    v = phase_work(f'AT2_N{n}_{p}_polydisperse_pores{tag}_l{2 / n:g}_lh', p)
    return None if not np.isfinite(v) else v


fig, axes = plt.subplots(2, len(SETTINGS), figsize=(4.3 * len(SETTINGS), 9.5), layout='constrained', dpi=150,
                         sharex=True)
rows = []
for r, (kind, title) in enumerate((('sharp', 'sharp crack, l/h = 2'), ('smooth', 'smooth crack, l = 1/16'))):
    for s, short in enumerate(SHORT):
        ax = axes[r, s]
        for k, (p, label, _, c, m) in enumerate(PRECS):
            y = np.array([W[s, k, LENGTHS.index(2 / n), j] for j, n in enumerate(GRIDS)] if kind == 'sharp'
                         else W[s, k, LENGTHS.index(0.0625)])
            ok = np.isfinite(y)
            if ok.sum() < 2:
                continue
            b, log_a = np.polyfit(np.log(np.array(GRIDS)[ok]), np.log(y[ok]), 1)
            ext = np.exp(log_a) * np.array(EXTRA, float) ** b
            ax.plot(np.array(GRIDS)[ok], y[ok], linestyle='none', marker=m, ms=7, color=c, label=label)
            xs = np.array([GRIDS[0], *EXTRA], float)
            ax.plot(xs, np.exp(log_a) * xs ** b, ':', lw=2, color=c)
            ax.plot(EXTRA, ext, linestyle='none', marker=m, ms=6, color=c, mfc='white')
            real = {n: work(n, p, SETTINGS[s][0], SETTINGS[s][1], 2 / n if kind == 'sharp' else 0.0625)
                    for n in (256, 512)}
            real256 = real[256]
            pts = [(n, v) for n, v in real.items() if np.isfinite(v)]
            if pts:
                ax.plot(*np.array(pts).T, linestyle='none', marker='D', ms=10, color=c, alpha=0.6,
                        label='256$^2$ / 512$^2$ run (check)' if k == 0 else None)
            rows.append((kind, SETTINGS[s][2], p, b, *y, *ext, real256, real[512]))
            if kind == 'sharp' and s in CHECK:
                pts = [(n, check_work(n, p, CHECK[s])) for n in (256, 512)]
                pts = [q for q in pts if q[1] is not None]
                if pts:
                    ax.plot(*np.array(pts).T, linestyle='none', marker='D', ms=9, mfc='none', mew=2, color=c,
                            label='real run' if k == 0 else None)
        ax.set_xscale('log'), ax.set_yscale('log')
        ax.set_xticks([32, 64, 128, 256, 512, 1024])
        ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f'{v:g}$^2$'))
        ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
        ax.yaxis.set_major_locator(matplotlib.ticker.FixedLocator([20, 50, 100, 200, 500, 1000, 2000]))
        ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f'{v:g}'))
        ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax.set_ylim(15, 3000)
        ax.set_title(f'{short.replace(chr(10), " ", 1).replace(chr(10), " (", 1)})\n{title}', fontsize=11)
        ax.tick_params(labelsize=10), ax.grid(True, color='0.9')
        if s == 0:
            ax.set_ylabel('work per elastic solve', fontsize=12)
        if r == 1:
            ax.set_xlabel('grid', fontsize=12)
handles, labels = axes[0, 1].get_legend_handles_labels()
fig.legend(handles, labels, loc='outside lower center', ncol=4, fontsize=12)
fig.suptitle(f'{PHASE_TITLE.upper()}\nmarkers: runs 32$^2$ - 128$^2$; dotted + open markers: fit a N$^b$ extrapolated; filled diamonds: '
             '256$^2$ / 512$^2$ runs; open diamonds: earlier 256$^2$ / 512$^2$ runs (check)', fontsize=12)
out = os.path.join(HERE, 'figures', f'AT2_contrast_extrapolation_2D{SUFFIX}.png')
fig.savefig(out)

print(f'{"crack":6s} {"setting":32s} {"precond":13s} {"b":>5s} | {"32":>5s} {"64":>5s} {"128":>5s} | '
      f'{"256":>6s} {"512":>6s} {"1024":>6s} | {"real 256":>8s} {"real/fit":>8s}')
for kind, name, p, b, *v in rows:
    print(f'{kind:6s} {name:32s} {p:13s} {b:5.2f} | ' + ' '.join(f'{x:5.0f}' for x in v[:3]) + ' | '
          + ' '.join(f'{x:6.0f}' for x in v[3:6]) + f' | {v[6]:8.0f} {v[6] / v[3]:8.2f}'
          + (f' | real 512 {v[7]:5.0f} ({v[7] / v[4]:.2f})' if np.isfinite(v[7]) else ''))
for s, tag in CHECK.items():
    for p, *_ in PRECS:
        got = [check_work(n, p, tag) for n in (256, 512)]
        print(f'check  {SETTINGS[s][2]:32s} {p:13s} real 256: {got[0] or 0:5.0f}  512: {got[1] or 0:5.0f}')
print(f'wrote {out}')
