"""
Do different initial designs give the same optimum? 2D TO at 128^2 (eta = 4h = 1/32, contrast 1e4, w = 0.2,
Green-Jacobi) from 6 Cahn-Hilliard starts (make_TO_CH_starts.py) and the standard sine start (run_TO_CH_starts.sh).

Figure figures/TO_CH_starts_results.png:
  row 1: start designs; row 2: final designs, 2 x 2 periodic copies (the cell is periodic, a design may sit shifted);
  row 3: objective against L-BFGS iteration.
Printed: final objective, stress-mismatch and perimeter terms, homogenised stiffness (Voigt C11, C22, C12, C33) vs the
target, and the distance between every pair of final designs after the best periodic shift and square symmetry
(rotations, mirrors, phase not flipped): 0 = the same design.
usage:  python plot_TO_CH_starts.py                      (128^2, eta 1/32)
        NPIX=256 ETA=0.015625 python plot_TO_CH_starts.py  (the runs continued from the 128^2 optima,
                                                           suffix _from128, logs logs/TO_CH_starts_256)
"""
import glob
import os
import re

import matplotlib.pyplot as plt
import netCDF4
import numpy as np

from elastic_properties import properties

HERE = os.path.dirname(os.path.realpath(__file__))
plt.rcParams.update({'font.size': 13})
LABELS = ['sine', 'bi_s0', 'bi_s1', 'bi_s2', 'bi_s3', 'drop_s0', 'drop_s1']
N = int(os.environ.get('NPIX', 128))
ETA = os.environ.get('ETA', '0.03125')
SFX = '' if N == 128 else '_from128'
LOGDIR = os.path.join(HERE, 'logs', 'TO_CH_starts' + ('' if N == 128 else f'_{N}'))


def final_file(label):
    hits = glob.glob(os.path.join(HERE, 'data', '**', f'N{N}_*Green_Jacobi_eta_{ETA}_pfw0.2_*_start_{label}{SFX}_final.nc'),
                     recursive=True)
    return hits[0] if hits else None


def start_design(label):
    if N != 128:                                              # continued runs: the 128^2 optimum is the start
        hits = glob.glob(os.path.join(HERE, 'data', '**', f'N128_*Green_Jacobi_eta_0.03125_pfw0.2_*_start_{label}_final.nc'),
                         recursive=True)
        with netCDF4.Dataset(hits[0]) as d:
            return np.array(d['phase_field_solution'][0, 0])
    if label == 'sine':
        x = (np.arange(N) / N)[:, None]; y = (np.arange(N) / N)[None, :]
        return np.clip((np.sin(4 * np.pi * x) + np.sin(4 * np.pi * y) + 2) / 4 + 0.25, 0, 1)   # + noise (mean 0.25)
    with netCDF4.Dataset(os.path.join(HERE, 'data', 'TO_CH_starts', f'CH_start_{label}.nc')) as d:
        r = np.array(d['phase_field_solution'][0, 0])
    return r[::r.shape[0] // N, ::r.shape[1] // N]


def objective_history(label):
    its, f = [], []
    for line in open(os.path.join(LOGDIR, f'{label}.log')):
        m = re.match(r'^(\d+)\s+([0-9.eE+-]+)\s+([0-9.eE+-]+)\s+(\S+)\s*$', line)
        if m:
            its.append(int(m.group(1))); f.append(float(m.group(2)))
    return np.array(its), np.array(f)


def distance(a, b):
    """min over periodic shifts and the 8 square symmetries of the RMS difference of the designs (rho in [0, 1])."""
    best = np.inf
    for k in range(4):
        for flip in (False, True):
            t = np.rot90(b, k)
            t = t.T if flip else t
            # sum (a - shift(t))^2 = |a|^2 + |t|^2 - 2 corr(a, t): best shift from the FFT cross-correlation
            corr = np.real(np.fft.ifft2(np.fft.fft2(a) * np.conj(np.fft.fft2(t))))
            best = min(best, (np.sum(a ** 2) + np.sum(t ** 2) - 2 * corr.max()) / a.size)
    return np.sqrt(max(best, 0.0))


results, finals = {}, {}
for label in LABELS:
    f = final_file(label)
    if f is None:
        print(label, 'no final file yet'); continue
    with netCDF4.Dataset(f) as d:
        rho = np.array(d['phase_field_solution'][0, 0])
        C = np.array(d.getncattr('homogenized_C_voigt')).reshape(3, 3)
        sig = np.atleast_1d(d.getncattr('norms_sigma')); pf = np.atleast_1d(d.getncattr('norms_pf'))
        its = int(d.getncattr('nb_iterations'))
        T = np.array(d.getncattr('target_stresses')).reshape(3, 2, 2)
    Ct = np.array([[T[0, 0, 0], T[1, 0, 0], 0], [T[0, 1, 1], T[1, 1, 1], 0], [0, 0, T[2, 0, 1]]])
    finals[label] = rho
    results[label] = dict(its=its, f=sig[-1] + pf[-1], f_sigma=sig[-1], P=pf[-1] / 0.2, C=C,
                          Cerr=np.linalg.norm(C - Ct) / np.linalg.norm(Ct), stiff=np.mean(rho > 0.5),
                          grey=np.mean((rho > 0.1) & (rho < 0.9)))

print(f'{"start":>8} {"its":>5} {"f":>8} {"f_sigma":>8} {"P":>6} {"stiff":>6} {"grey":>5}   C11    C22    C12    C33   err    nu12    mu   A_Z')
for label, r in results.items():
    C = r['C']
    print(f'{label:>8} {r["its"]:5d} {r["f"]:8.4f} {r["f_sigma"]:8.4f} {r["P"]:6.2f} {r["stiff"]:6.2f} {r["grey"]:5.2f}  '
          f'{C[0, 0]:.3f}  {C[1, 1]:.3f} {C[0, 1]:+.3f}  {C[2, 2]:.3f}  {r["Cerr"]:.3f}  {properties(C)["nu12"]:+.3f} '
          f'{properties(C)["mu"]:.3f} {properties(C)["A_Z"]:.2f}')
print('target: C11 = C22 = 0.289, C12 = -0.096, C33 = 0.193; nu12 = -1/3, mu = 0.193, A_Z = 1')
labels = list(finals)
print('\ndistance between final designs (RMS of rho after the best shift / rotation / mirror; 0 = same design):')
print(' ' * 9 + ''.join(f'{l:>9}' for l in labels))
for a in labels:
    print(f'{a:>9}' + ''.join(f'{distance(finals[a], finals[b]):9.3f}' for b in labels))

fig = plt.figure(figsize=(3.3 * len(LABELS), 12.5))
for c, label in enumerate(LABELS):
    ax = fig.add_subplot(3, len(LABELS), c + 1)
    ax.imshow(start_design(label).T, origin='lower', cmap='Greys', vmin=0, vmax=1, interpolation='nearest',
              extent=(0, 1, 0, 1))
    ax.set_title((f'start {label}' + (' (+ noise)' if label == 'sine' else '')) if N == 128 else f'{label}: 128$^2$ optimum (start)', fontsize=13)
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax = fig.add_subplot(3, len(LABELS), len(LABELS) + c + 1)
    if label in finals:
        r = results[label]
        ax.imshow(np.tile(finals[label], (2, 2)).T, origin='lower', cmap='Greys', vmin=0, vmax=1,
                  interpolation='nearest', extent=(0, 2, 0, 2))
        ax.set_title(f'f = {r["f"]:.3f}, {r["its"]} it\n$\\nu_{{12}}$ = {properties(r["C"])["nu12"]:+.3f}, err {r["Cerr"]:.2f}', fontsize=12)
    ax.set_xticks([0, 1, 2]); ax.set_yticks([0, 1, 2])
ax = fig.add_subplot(3, 1, 3)
for label in LABELS:
    p = os.path.join(LOGDIR, f'{label}.log')
    if os.path.exists(p):
        it, f = objective_history(label)
        ax.plot(it, f, lw=2, ls='-' if label.startswith('bi') else ('--' if label.startswith('drop') else ':'),
                label=label, color='black' if label == 'sine' else None)
ax.set(yscale='log', xlabel='L-BFGS iteration', ylabel='objective f', xlim=(0, None))
ax.set_yticks([0.3, 0.5, 1, 2, 3, 5]); ax.set_yticklabels(['0.3', '0.5', '1', '2', '3', '5'])
ax.minorticks_off(); ax.grid(alpha=0.3); ax.legend(ncol=7, fontsize=12)
fig.suptitle(f'2D TO {N}$^2$ ($\\eta$ = {ETA} = 4h, contrast 1e4, w = 0.2) from different initial designs: '
             'starts (top), optima as 2 x 2 periodic copies (middle), objective (bottom)', fontsize=14)
fig.tight_layout()
out = os.path.join(HERE, 'figures', f'TO_CH_starts_results_N{N}.png')
fig.savefig(out, dpi=100)
print('saved', out)
