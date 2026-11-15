"""
Prototype (2D, fast) of a stopping test of the design iteration that looks at the ENERGIES, not at the whole
objective: the objective is  f = (stiffness misfit: sum of squared relative misfits of the target energies)
                                + (phase-field interface penalty),
and the user's question is when the stiffness part is "done" (2026-11-15).

One long run of the 2D TO (example_2D_elasticity_TO.py, energy objective, MG, Helmholtz L-BFGS) records per
iteration: f, its two parts, the homogenised energies W_k (C1111, C1122, C2222, C1212), nu12 = C1122 / C2222, the
largest design change and the grey fraction. Afterwards candidate stopping tests are evaluated on the record:

    E_close(tol)    max_k |W_k - W*_k| / |W*_k| < tol                      "the energies are close to the target"
    E_settled(tol)  max_k |W_k(it) - W_k(it - 10)| / |W*_k| < tol          "the energies no longer change"
    F_settled(tol)  (f(it - 10) - f(it)) / f(it) < tol                     whole objective (FTOL of to3d_large.py)
    step            max |rho(it) - rho(it - 1)| < 1e-3                     the test used so far (xtol)

For each: the iteration at which it would have stopped, the share of the full run's evaluations spent until then, and
how far the design then is from the final one (nu12, energies, mean |rho - rho_final|).

    TARGET_SCALE=0.3 python stop_criterion_2D.py     -> logs/stop_criterion/N<N>_nu<nu>_s<scale>.{npz,txt}
Environment: NPIX (128), TARGET_NU (-0.5), TARGET_SCALE (0.3), MAXITER (600), ETA (2 / NPIX), TAG and OUT_NAME
(suffix of the output name, folder under logs/; used by run_mesh_independence_2D.sh).
"""
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
N = int(os.environ.get('NPIX', 128))
ETA = float(os.environ.get('ETA', 2 / N))                 # interface width; default 2 h
os.environ.update({'NPIX': str(N), 'ETA': str(ETA), 'SOFT_EXP': '3', 'PF_NORMALIZED': '1', 'PF_WEIGHT': '0.2',
                   'CG_MODE': 'energy', 'CG_DELAY': '0', 'SHOW_PLOTS': '0', 'JAX_PLATFORMS': 'cpu',
                   'OBJECTIVE': 'energy', 'TARGET_NU': os.environ.get('TARGET_NU', '-0.5'),
                   'TARGET_SCALE': os.environ.get('TARGET_SCALE', '0.3')})
sys.argv = [sys.argv[0], 'MG']
sys.path.insert(0, str(HERE))
import numpy as np

import design_optimizers
import example_2D_elasticity_TO as TO
from muFFTTO import io_utils

WINDOW = 10
max_iterations = int(os.environ.get('MAXITER', 600))
tag = f'N{N}_nu{os.environ["TARGET_NU"]}_s{os.environ["TARGET_SCALE"]}' + os.environ.get('TAG', '')
out = HERE / 'logs' / os.environ.get('OUT_NAME', 'stop_criterion')
out.mkdir(parents=True, exist_ok=True)

# ---- start design: the Cahn-Hilliard pattern of the 2D study, periodic bilinear interpolation (axis by axis)
start = np.asarray(io_utils.read_file(str(HERE / 'data/CH_starts_new/CH_start_bi_s11.nc')).fields['phase_field_solution'][-1])
start = start.reshape((int(round(start.size ** 0.5)),) * 2)
for axis in range(2):
    position = np.arange(N) * (start.shape[axis] / N)
    base = np.floor(position).astype(int)
    fraction = (position - base).reshape([-1 if a == axis else 1 for a in range(2)])
    start = (np.take(start, base % start.shape[axis], axis=axis) * (1 - fraction)
             + np.take(start, (base + 1) % start.shape[axis], axis=axis) * fraction)
x0 = np.clip(start, 0, 1).ravel()

# ---- Helmholtz initial inverse Hessian (as in the script's own main)
disc = TO.discretization
helm_in = disc.get_scalar_field(name='helmholtz_in')
helm_hat = disc.ffield_collection.complex_field(name='helmholtz_hat', components=(1,))
to_host = (lambda a: a.get()) if TO.device == 'gpu' else np.asarray        # DEVICE=gpu: field data are cupy arrays
symbol = TO.xp.asarray(1.0 / (1.0 + 4.0 * TO.eta ** 2 * sum(
    4 * N ** 2 * np.sin(np.pi * np.asarray(disc.fft.ifftfreq[a], dtype=float) / N) ** 2 for a in range(2))))


def helmholtz(v):
    helm_in.s[0, 0] = TO.xp.asarray(v.reshape(N, N))
    disc.fft.fft(helm_in, helm_hat)
    helm_hat.s[0, 0] *= symbol
    disc.fft.ifft(helm_hat, helm_in)
    return to_host(helm_in.s[0, 0] * disc.fft.normalisation).ravel()


# ---- the energies of the objective: (load case, left strain, target) -> name
terms = [(0, TO.macro_gradients[0], 'C1111'), (0, TO.macro_gradients[1], 'C1122'), (1, TO.macro_gradients[1], 'C2222'),
         (2, TO.macro_gradients[2], 'C1212')]
targets = np.array([float(np.einsum('ij,ijkl,kl->', E, TO.elastic_C_target, TO.macro_gradients[lc])) for lc, E, _ in terms])
record = {key: [] for key in ('f', 'misfit', 'interface', 'energies', 'step', 'grey', 'evaluations')}
state = {'evaluations': 0, 'previous': x0.copy(), 'designs': {}}


def objective(x):
    state['evaluations'] += 1
    return TO.objective_function_multiple_load_cases(x)


def callback(x):
    x = np.asarray(x).ravel()
    stresses = TO.last_evaluation['homogenized_stresses']           # of the accepted iterate (evaluated last)
    record['energies'].append([float(np.sum(E * stresses[lc])) for lc, E, _ in terms])
    record['misfit'].append(float(np.sum(TO.norms_sigma[-1])))
    record['interface'].append(float(np.sum(TO.norms_pf[-1])))
    record['f'].append(record['misfit'][-1] + record['interface'][-1])
    record['step'].append(float(np.abs(x - state['previous']).max()))
    record['grey'].append(float(np.mean((x > 0.1) & (x < 0.9))))
    record['evaluations'].append(state['evaluations'])
    state['previous'] = x.copy()
    state['designs'][len(record['f'])] = x.astype(np.float32)


result = design_optimizers.l_bfgs_bounded_precond(objective, x0, precond=helmholtz, maxcor=20, bounds_lo=0.0,
                                                  bounds_hi=1.0, gtol=0.0, xtol=0.0, maxiter=max_iterations, c1=1e-4,
                                                  max_halvings=40, callback=callback, disp=False)
data = {key: np.array(value) for key, value in record.items()}
W, final = data['energies'], np.asarray(result.x).ravel()
nu = W[:, 1] / W[:, 2]
close = (np.abs(W - targets) / np.abs(targets)).max(axis=1)
settled = np.full(len(W), np.inf)
settled[WINDOW:] = (np.abs(W[WINDOW:] - W[:-WINDOW]) / np.abs(targets)).max(axis=1)
f_settled = np.full(len(W), np.inf)
f_settled[WINDOW:] = (data['f'][:-WINDOW] - data['f'][WINDOW:]) / data['f'][WINDOW:]
np.savez(out / f'{tag}.npz', eta=ETA, final=final.reshape(N, N), targets=targets, nu=nu, close=close, settled=settled, f_settled=f_settled, **data)

lines = [f'2D TO {N}^2, energy objective, target nu* {os.environ["TARGET_NU"]} scale {os.environ["TARGET_SCALE"]}: '
         f'{len(W)} iterations, {state["evaluations"]} evaluations',
         'targets  ' + '  '.join(f'{name} {t:+.5f}' for (_, _, name), t in zip(terms, targets)),
         'final    ' + '  '.join(f'{name} {w:+.5f}' for (_, _, name), w in zip(terms, W[-1]))
         + f'   nu12 {nu[-1]:+.4f}  f {data["f"][-1]:.5f} (misfit {data["misfit"][-1]:.5f}, interface '
           f'{data["interface"][-1]:.5f})  max relative energy misfit {close[-1]:.3f}',
         '',
         f'{"stopping test":<22s} {"iteration":>9s} {"evaluations":>12s} {"share":>6s} {"nu12":>8s} {"max rel. misfit":>16s}'
         f' {"f":>9s} {"mean |rho - rho_final|":>23s}']


def report(label, satisfied):
    hits = np.nonzero(satisfied)[0]
    if len(hits) == 0:
        lines.append(f'{label:<22s} {"never":>9s}')
        return
    i = hits[0]
    difference = float(np.abs(state['designs'][i + 1].astype(float) - final).mean())
    lines.append(f'{label:<22s} {i + 1:9d} {data["evaluations"][i]:12d} {data["evaluations"][i] / state["evaluations"]:6.0%} '
                 f'{nu[i]:+8.4f} {close[i]:16.3f} {data["f"][i]:9.5f} {difference:23.4f}')


for tol in (0.2, 0.1, 0.05, 0.02):
    report(f'E_close({tol:g})', close < tol)
for tol in (1e-2, 3e-3, 1e-3, 3e-4):
    report(f'E_settled({tol:g})', settled < tol)
for tol in (1e-2, 1e-3, 1e-4):
    report(f'F_settled({tol:g})', f_settled < tol)
step_test = data['step'] < 1e-3
step_test[0] = False
report('step < 1e-3 (xtol)', step_test)
lines.append(f'{"full run":<22s} {len(W):9d} {state["evaluations"]:12d} {1:6.0%} {nu[-1]:+8.4f} {close[-1]:16.3f} '
             f'{data["f"][-1]:9.5f} {0:23.4f}')
(out / f'{tag}.txt').write_text('\n'.join(lines) + '\n')
print('\n'.join(lines))
