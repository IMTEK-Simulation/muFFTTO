"""
2D TO from a RANDOM NOISE start: does the L-BFGS first-step fix (LBFGS_FIRST_STEP of to3d_large.py) prevent the
collapse to the uniform grey "mean-field" design seen from noise on grids >= 256? (user, 2026-11-15)

Same problem as stop_criterion_2D.py (example_2D_elasticity_TO.py, energy objective nu* -0.5 scale 0.3, MG, eta = 2h,
local Helmholtz L-BFGS of design_optimizers.py), but the start is i.i.d. uniform noise in [0, 1].
FIRST_STEP > 0: the first step changes the design by at most FIRST_STEP (instead of the unscaled H_0 = Helmholtz,
whose first step shrinks with the grid because the gradient of the volume-averaged objective scales with 1/N^2).

    NPIX=512 FIRST_STEP=0.05 SEED=0 python noise_start_2D.py   -> logs/noise_start_2D/N<N>_noise<M>_first<x>_seed<s>.*
Environment: NPIX (256), FIRST_STEP (0), SEED (0), MAXITER (200), ETA (2 / NPIX), DEVICE (cpu | gpu),
NOISE_N (default NPIX): the noise is drawn on a NOISE_N^2 grid and interpolated (periodic bilinear) to NPIX^2, i.e. the
same random field with correlation length 1 / NOISE_N on every grid. White noise on the grid itself (NOISE_N = NPIX)
has an interface energy that grows linearly with NPIX (eta = 2h: 104 / 207 / 412 at 256 / 512 / 1024, misfit ~40).
The run stops when the design has become uniform (std < 1e-3: collapsed to the mean-field solution).
Per iteration (printed and stored): f, misfit, interface, std of the design (-> 0: uniform "mean field"), grey
fraction (0.1 < rho < 0.9), nu12, evaluations, largest step.
"""
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
N = int(os.environ.get('NPIX', 256))
ETA = float(os.environ.get('ETA', 2 / N))
FIRST_STEP = float(os.environ.get('FIRST_STEP', 0.0))
SEED = int(os.environ.get('SEED', 0))
NOISE_N = int(os.environ.get('NOISE_N', N))
os.environ.update({'NPIX': str(N), 'ETA': str(ETA), 'SOFT_EXP': '3', 'PF_NORMALIZED': '1', 'PF_WEIGHT': '0.2',
                   'CG_MODE': 'energy', 'CG_DELAY': '0', 'SHOW_PLOTS': '0', 'JAX_PLATFORMS': 'cpu',
                   'OBJECTIVE': 'energy', 'TARGET_NU': '-0.5', 'TARGET_SCALE': '0.3'})
sys.argv = [sys.argv[0], 'MG']
sys.path.insert(0, str(HERE))
import numpy as np

import design_optimizers
import example_2D_elasticity_TO as TO

max_iterations = int(os.environ.get('MAXITER', 200))
tag = f'N{N}_noise{NOISE_N}_first{FIRST_STEP:g}_seed{SEED}'
out = HERE / 'logs' / 'noise_start_2D'
out.mkdir(parents=True, exist_ok=True)
x0 = np.random.default_rng(SEED).uniform(0.0, 1.0, (NOISE_N, NOISE_N))
for axis in range(2):                                     # periodic bilinear interpolation NOISE_N -> N, axis by axis
    position = np.arange(N) * (NOISE_N / N)
    base = np.floor(position).astype(int)
    fraction = (position - base).reshape([-1 if a == axis else 1 for a in range(2)])
    x0 = np.take(x0, base % NOISE_N, axis=axis) * (1 - fraction) + np.take(x0, (base + 1) % NOISE_N, axis=axis) * fraction
x0 = x0.ravel()

disc = TO.discretization
helm_in = disc.get_scalar_field(name='helmholtz_in')
helm_hat = disc.ffield_collection.complex_field(name='helmholtz_hat', components=(1,))
to_host = (lambda a: a.get()) if TO.device == 'gpu' else np.asarray
symbol = TO.xp.asarray(1.0 / (1.0 + 4.0 * TO.eta ** 2 * sum(
    4 * N ** 2 * np.sin(np.pi * np.asarray(disc.fft.ifftfreq[a], dtype=float) / N) ** 2 for a in range(2))))


def helmholtz(v):
    helm_in.s[0, 0] = TO.xp.asarray(v.reshape(N, N))
    disc.fft.fft(helm_in, helm_hat)
    helm_hat.s[0, 0] *= symbol
    disc.fft.ifft(helm_hat, helm_in)
    return to_host(helm_in.s[0, 0] * disc.fft.normalisation).ravel()


terms = [(0, TO.macro_gradients[0]), (0, TO.macro_gradients[1]), (1, TO.macro_gradients[1])]
record = {key: [] for key in ('f', 'misfit', 'interface', 'std', 'grey', 'nu', 'evaluations', 'step', 'wall')}
state = {'evaluations': 0, 'previous': x0.copy(), 'designs': {0: x0.astype(np.float32)}, 'start': time.time()}
SNAPSHOTS = {1, 2, 5, 10, 20, 50, 100, 200, 400, 600}


def objective(x):
    state['evaluations'] += 1
    return TO.objective_function_multiple_load_cases(x)


def callback(x):
    x = np.asarray(x).ravel()
    stresses = TO.last_evaluation['homogenized_stresses']
    W = [float(np.sum(E * stresses[lc])) for lc, E in terms]
    it = len(record['f']) + 1
    record['misfit'].append(float(np.sum(TO.norms_sigma[-1])))
    record['interface'].append(float(np.sum(TO.norms_pf[-1])))
    record['f'].append(record['misfit'][-1] + record['interface'][-1])
    record['std'].append(float(np.std(x)))
    record['grey'].append(float(np.mean((x > 0.1) & (x < 0.9))))
    record['nu'].append(W[1] / W[2])
    record['evaluations'].append(state['evaluations'])
    record['step'].append(float(np.abs(x - state['previous']).max()))
    record['wall'].append(time.time() - state['start'])
    state['previous'] = x.copy()
    if it in SNAPSHOTS:
        state['designs'][it] = x.astype(np.float32)
    print(f'it {it:4d}  f {record["f"][-1]:.5e}  misfit {record["misfit"][-1]:.4e}  interface '
          f'{record["interface"][-1]:.4e}  std {record["std"][-1]:.4f}  grey {record["grey"][-1]:.3f}  '
          f'nu {record["nu"][-1]:+.3f}  evals {state["evaluations"]}  step {record["step"][-1]:.2e}  '
          f'wall {record["wall"][-1]:.0f} s', flush=True)
    if record['std'][-1] < 1e-3:
        state['designs'][it] = x.astype(np.float32)
        return True                                       # collapsed to a uniform design: stop


try:
    result = design_optimizers.l_bfgs_bounded_precond(objective, x0, precond=helmholtz, maxcor=20, bounds_lo=0.0,
                                                      bounds_hi=1.0, gtol=0.0, xtol=0.0, maxiter=max_iterations,
                                                      c1=1e-4, max_halvings=40, callback=callback, disp=False,
                                                      first_step=FIRST_STEP)
    final, message = np.asarray(result.x).ravel(), result.message
    message = 'collapsed to a uniform design (std < 1e-3)' if message == 'callback' else message
except RuntimeError as error:                             # e.g. the elastic CG on a uniform design
    final, message = state['previous'], f'stopped by error: {error}'
data = {key: np.array(value) for key, value in record.items()}
np.savez(out / f'{tag}.npz', eta=ETA, first_step=FIRST_STEP, seed=SEED, final=final.reshape(N, N).astype(np.float32),
         snapshot_iterations=np.array(sorted(state['designs'])),
         snapshots=np.array([state['designs'][k].reshape(N, N) for k in sorted(state['designs'])]), **data)
summary = (f'2D noise start {N}^2 (noise drawn on {NOISE_N}^2), FIRST_STEP {FIRST_STEP:g}, seed {SEED}: {len(data["f"])} iterations, '
           f'{state["evaluations"]} evaluations, {data["wall"][-1]:.0f} s; final f {data["f"][-1]:.5f} '
           f'(misfit {data["misfit"][-1]:.4f}, interface {data["interface"][-1]:.4f}), std {data["std"][-1]:.4f}, '
           f'grey {data["grey"][-1]:.3f}, nu12 {data["nu"][-1]:+.3f}; optimizer: {message}')
(out / f'{tag}.txt').write_text(summary + '\n')
print(summary, flush=True)
