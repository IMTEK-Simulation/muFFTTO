"""Compare iteration counts of contact-fracture runs: python compare.py out.png run1.nc run2.nc ..."""
import sys
sys.path.insert(0, '/home/martin/Programming/microTopOpt/muFFTTO')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from muFFTTO import io_utils

out, files = sys.argv[1], sys.argv[2:]
fig, ax = plt.subplots(2, 3, figsize=(15, 8))
ax = ax.ravel()
for f in files:
    r = io_utils.read_file(f)
    h = r.frame_variables
    name = f.split('_prec-')[1].rsplit('_2026', 1)[0]
    lam = h['lam']
    F = 1 + lam * np.asarray(r.attributes['H_macro']).reshape(2, 2)[1, 1]
    st, hp, outer = h['staggered_iterations'], h['nb_hessp'], h['nb_outer']
    t = h['elapsed_time']
    dt = np.diff(np.r_[0, t])
    kw = dict(marker='.', ms=4, lw=1, label=name)
    ax[0].plot(F, h['mean_P'][:, 1, 1], **kw)
    ax[1].semilogy(F, hp, **kw)
    ax[2].plot(F, st, **kw)
    ax[3].semilogy(F, hp / np.maximum(outer, 1), **kw)
    ax[4].semilogy(F, h['cg_damage'] / np.maximum(st, 1), **kw)
    ax[5].plot(F, t / 3600, **kw)
    print(f"{name:32s} incr {r.nb_frames:3d}  lam {lam[-1]:.3f}  time {t[-1]/3600:.2f} h  hessp {hp.sum():.3g}  "
          f"TR outer {outer.sum():.0f}  stagger {st.sum():.0f}  CG/TR-iter {hp.sum()/outer.sum():.1f}  "
          f"damage CG {h['cg_damage'].sum():.3g}  ms/hessp {1e3*t[-1]/hp.sum():.2f}")
titles = [r'$\bar P_{11}$', 'Hessian products per increment', 'staggered iterations per increment',
          'Hessian products per trust-region iteration (CG length)', 'damage CG iterations per staggered iteration',
          'cumulative wall time [h]']
for a, title in zip(ax, titles):
    a.set_title(title, fontsize=10)
    a.set_xlabel(r'$\bar F_{11}$')
    a.invert_xaxis()
    a.grid(alpha=0.3)
ax[0].legend()
fig.suptitle('sinusoidal_ligaments, compression')
fig.tight_layout()
fig.savefig(out, dpi=90)
print('saved', out)
