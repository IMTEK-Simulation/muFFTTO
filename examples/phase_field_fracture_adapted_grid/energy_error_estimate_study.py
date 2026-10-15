"""How accurate is the energy-error estimate that stops the inner CG, for Green and Green-Jacobi?

The inner CG of the trust-region / Newton solvers stops on the delayed Hestenes-Stiefel estimate (Strakos & Tichy):
with Delta_j = alpha_j (r_j, z_j) = |s_{j+1} - s_j|_K^2 it stops at the first k >= d with
    sum_{j=k-d}^{k} Delta_j <= tol^2 sum_{j=0}^{k} Delta_j,
i.e. the estimated relative energy error eta_{k-d} of the iterate s_{k-d} is below tol (it returns s_k).

Here the same CG is run on real Newton systems K s = -g of a finished run: at an increment k the state of the first
Newton step of increment k+1 (predictor u_k + dlam_{k+1} / dlam_k (u_k - u_{k-1}) at lam_{k+1}, like the solver),
for both preconditioners, and compared with the true relative energy error
    eps_j = |s* - s_j|_K / |s*|_K
(evaluated with an explicit Hessian product: the recursively updated CG residual drifts from b - K s_j), with s* from
a tight Green-Jacobi solve; a tight Green solve checks s* (their energy distance is the floor of the measurement).
For comparison the figure also shows the relative residual |r_j| / |b| (Euclidean, the recursively updated CG
residual) and where a residual stop |r_j| <= tol |b| would end. The Hessian, the preconditioners and the stopping rule
are those of the simulation script: the generated run.py is executed up to its load loop.

usage: python energy_error_estimate_study.py RUN_PY RUN_NC OUT_PREFIX [--increments 10 50 80 100]
       python energy_error_estimate_study.py --replot OUT_PREFIX      (figure only, from OUT_PREFIX.npz)
writes OUT_PREFIX.npz and OUT_PREFIX.png
"""
import argparse
import os
import time

import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('paths', nargs='+', help='RUN_PY RUN_NC OUT_PREFIX, or OUT_PREFIX with --replot (RUN_PY: the '
                                             'run.py that produced RUN_NC, its setup is reused)')
parser.add_argument('--replot', action='store_true', help='only redraw the figure from OUT_PREFIX.npz')
parser.add_argument('--increments', type=int, nargs='+', default=None,
                    help='frames whose next Newton system is studied (default: 4 spread over the run)')
parser.add_argument('--tol', type=float, default=1e-6, help='tolerance of the stopping rule (as in the runs)')
parser.add_argument('--delay', type=int, default=10)
parser.add_argument('--reference-tol', type=float, default=1e-13, help='relative residual of the reference solve')
parser.add_argument('--maxiter', type=int, default=200000)
args = parser.parse_args()
if args.replot:
    (args.out,) = args.paths
else:
    args.run_py, args.run_nc, args.out = args.paths

PRECONDITIONERS = ('Green', 'Green_Jacobi')


def load_setup(preconditioner):
    """Namespace of run.py executed up to its load loop, with the given mechanics preconditioner, no output."""
    source = open(args.run_py).read()
    source = source[:source.index('start_time = time.time()')]
    replacements = {"mechanics_preconditioner = 'Green_Jacobi'": f"mechanics_preconditioner = '{preconditioner}'",
                    "make_figures = True": "make_figures = False"}
    for old, new in replacements.items():
        source = source.replace(old, new)
    source = source.replace('\noutput_root = ', '\noutput_root = None  # ', 1)
    namespace = {'__name__': 'study', '__file__': os.path.abspath(args.run_py)}
    exec(compile(source, args.run_py, 'exec'), namespace)
    assert namespace['mechanics_preconditioner'] == preconditioner
    return namespace


def newton_system(setup, u, damage, lam):
    """(rhs b = -g, K v, P v) of the first Newton step at the fluctuation u, the damage and the load lam."""
    xp = setup['xp']                     # numpy, or cupy for device = 'gpu'
    setup['damage'].s[...] = xp.asarray(damage.reshape(setup['damage'].s.shape))
    setup['update_degradation']()
    setup['lam_current'] = lam
    setup['u_fluc'].s[...] = xp.asarray(u.reshape(setup['u_shape']))
    if not setup['set_F'](lam, setup['u_fluc']) > 0.0:
        raise ValueError('predictor not admissible')
    if setup['mechanics_preconditioner'] == 'Green_Jacobi':
        setup['update_mechanics_jacobi'](lam, setup['u_fluc'])
    x = setup['field_to_optimizer'](setup['u_fluc'])
    _, gradient = setup['fun_grad'](x)
    return -gradient.ravel(), (lambda v: setup['hessp'](x, v).ravel()), (lambda v: setup['precond'](v).ravel())


def pcg(b, K, P, maxiter, reference=None, stop_relative_residual=None):
    """PCG from s_0 = 0. Records Delta_j, the relative residual and, with a reference s*, the true squared energy
    error (s* - s_j) . K (s* - s_j) of every iterate. Stops on negative curvature, at maxiter or at the relative residual."""
    s = np.zeros_like(b)
    r = b.copy()
    z = P(r)
    p = z.copy()
    rz = r @ z
    b_norm = np.linalg.norm(b)
    def energy_error2(s):
        return (reference - s) @ K(reference - s)

    record = {'Delta': [], 'residual': [1.0], 'error2': [] if reference is None else [energy_error2(s)]}
    for _ in range(maxiter):
        Kp = K(p)
        curvature = p @ Kp
        if curvature <= 0.0:
            record['negative_curvature'] = len(record['Delta'])
            break
        alpha = rz / curvature
        s += alpha * p
        r -= alpha * Kp
        record['Delta'].append(alpha * rz)
        record['residual'].append(np.linalg.norm(r) / b_norm)
        if reference is not None:
            record['error2'].append(energy_error2(s))
        if stop_relative_residual is not None and record['residual'][-1] <= stop_relative_residual:
            break
        z = P(r)
        rz_new = r @ z
        p = z + (rz_new / rz) * p
        rz = rz_new
    return s, {key: np.asarray(value) for key, value in record.items()}


def stopping_iteration(Delta, tol, delay):
    """First k at which the rule of the runs stops (it returns s_{k+1}, i.e. after k + 1 CG steps)."""
    cumulative = np.cumsum(Delta)
    window = window_sums(Delta, delay)
    for k in range(delay, len(Delta)):
        if window[k - delay] <= tol ** 2 * cumulative[k]:
            return k
    return None


def window_sums(Delta, delay):
    """sum_{j=k-d}^{k} Delta_j for k = d, d+1, ...; summed directly: as differences of cumulative sums they cancel
    once the relative error is below ~1e-8."""
    return np.convolve(Delta, np.ones(delay + 1), mode='valid')


def estimates(Delta, delay):
    """eta_{k-d} for every k >= d: the estimate of the relative energy error of s_{k-d} available at step k."""
    k = np.arange(delay, len(Delta))
    return k - delay, np.sqrt(window_sums(Delta, delay) / np.cumsum(Delta)[k])


def study():
    """Run the study for the chosen increments; returns ({frame: entry}, n)."""
    import netCDF4

    dataset = netCDF4.Dataset(args.run_nc)
    lam_path = np.asarray(dataset['lam'][:])
    u_path = np.asarray(dataset['u_fluc'][:])
    damage_path = np.asarray(dataset['damage'][:])
    F11_path = 1 + lam_path * np.asarray(dataset.H_macro).reshape(2, 2)[1, 1]
    nb_frames = len(lam_path)
    increments = args.increments or sorted({1, nb_frames // 3, (2 * nb_frames) // 3, nb_frames - 2})
    setups = {p: load_setup(p) for p in PRECONDITIONERS}
    n = int(np.atleast_1d(dataset.nb_grid_pts)[0])

    results = {}
    for k in increments:
        # the first Newton system of increment k + 1 (frames are 0-based: frame k is increment k + 1)
        lam_next = lam_path[k + 1]
        u_k, u_previous = u_path[k], u_path[k - 1] if k > 0 else np.zeros_like(u_path[k])
        dlam_k = lam_path[k] - (lam_path[k - 1] if k > 0 else 0.0)
        u_predictor = u_k + (lam_next - lam_path[k]) / dlam_k * (u_k - u_previous)
        systems = {}
        for p, setup in setups.items():
            try:
                systems[p] = newton_system(setup, u_predictor, damage_path[k], lam_next)
            except ValueError:
                systems[p] = newton_system(setup, u_k, damage_path[k], lam_next)
        b, K, _ = systems['Green_Jacobi']
        start = time.time()
        reference, ref_record = pcg(b, K, systems['Green_Jacobi'][2], args.maxiter,
                                    stop_relative_residual=args.reference_tol)
        if 'negative_curvature' in ref_record:
            print(f'frame {k}: negative curvature in the reference solve, skipped')
            continue
        reference_energy2 = reference @ K(reference)
        check, _ = pcg(b, K, systems['Green'][2], args.maxiter, stop_relative_residual=args.reference_tol)
        floor = np.sqrt((reference - check) @ K(reference - check) / reference_energy2)
        entry = {'F11': F11_path[k + 1], 'reference_iterations': len(ref_record['Delta']), 'floor': floor}
        for p in PRECONDITIONERS:
            b_p, K_p, P_p = systems[p]
            _, record = pcg(b_p, K_p, P_p, args.maxiter, reference=reference,
                            stop_relative_residual=args.reference_tol)
            true_error = np.sqrt(np.maximum(record['error2'], 0.0) / reference_energy2)
            stop = stopping_iteration(record['Delta'], args.tol, args.delay)
            index, eta = estimates(record['Delta'], args.delay)
            entry[p] = {'true_error': true_error, 'residual': record['residual'], 'eta_index': index, 'eta': eta,
                        'stop': stop}
            if stop is not None:
                # the runs return s_{stop+1}; the estimate certifies s_{stop-d}
                entry[p].update(error_returned=true_error[stop + 1], error_certified=true_error[stop - args.delay],
                                estimate_at_stop=eta[stop - args.delay])
            print(f'frame {k:3d} F11 {entry["F11"]:.3f} {p:12s}: stop after {"-" if stop is None else stop + 1:>6} CG, '
                  f'true error of the returned iterate {entry[p].get("error_returned", np.nan):.2e}, '
                  f'estimate {entry[p].get("estimate_at_stop", np.nan):.2e} vs true '
                  f'{entry[p].get("error_certified", np.nan):.2e} at s_(k-d)')
        print(f'  ({time.time() - start:.0f} s, reference {entry["reference_iterations"]} CG, Green and Green-Jacobi '
              f'references differ by {floor:.1e} in the relative energy norm)')
        results[k] = entry
    return results, n


if args.replot:
    saved = np.load(args.out + '.npz', allow_pickle=True)
    results, n = saved['results'].item(), int(saved['n'])
    args.tol, args.delay = float(saved['tol']), int(saved['delay'])
else:
    results, n = study()
    np.savez(args.out + '.npz', results=np.array(results, dtype=object), run_nc=args.run_nc, n=n, tol=args.tol,
             delay=args.delay)

# ---- figure: one column per studied increment ----------------------------------------------------------------------
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

SURFACE, TEXT_1, TEXT_2, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1", "#a3a29c"
COLORS = {'Green': "#2a78d6", 'Green_Jacobi': "#eb6834"}
LABELS = {'Green': 'Green', 'Green_Jacobi': 'Green-Jacobi'}


def style(ax, title, xlabel, ylabel):
    ax.set_facecolor(SURFACE)
    ax.set_title(title, loc='left', fontsize=11, color=TEXT_1)
    ax.set_xlabel(xlabel, color=TEXT_2, fontsize=10)
    ax.set_ylabel(ylabel, color=TEXT_2, fontsize=10)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    for side in ('left', 'bottom'):
        ax.spines[side].set_color(AXIS)
    ax.tick_params(colors=TEXT_2, labelsize=9)
    ax.grid(axis='y', color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


columns = list(results)
fig, axes = plt.subplots(2, len(columns), figsize=(4.6 * len(columns), 8.4), squeeze=False, facecolor=SURFACE)
fig.subplots_adjust(left=0.07, right=0.98, top=0.84, bottom=0.08, hspace=0.38, wspace=0.28)
fig.text(0.07, 0.95, f'Energy-error estimate of the inner CG vs the true energy error, {n}²', fontsize=15,
         fontweight='bold', color=TEXT_1)
fig.text(0.07, 0.885, f'first Newton system of an increment; solid: true |s* - s_k|_K / |s*|_K, dashed: delayed '
                      f'Hestenes-Stiefel estimate (d = {args.delay}) drawn at the iterate it certifies; '
                      f'dot: the stop (tol {args.tol:g});\ndotted: relative residual |r_k| / |b|, '
                      f'cross: where a residual stop |r_k| <= {args.tol:g} |b| would end, at its true energy error',
         fontsize=10, color=TEXT_2)
for column, k in enumerate(columns):
    entry = results[k]
    top, bottom = axes[0, column], axes[1, column]
    style(top, f'F11 = {entry["F11"]:.3f}', 'CG iteration', 'relative energy error')
    style(bottom, 'effectivity: estimate / true', 'CG iteration', 'eta / eps')
    for p in PRECONDITIONERS:
        e = entry[p]
        iterations = np.arange(len(e['true_error']))
        top.semilogy(iterations, e['true_error'], color=COLORS[p], linewidth=1.6, label=f'{LABELS[p]}, true')
        top.semilogy(e['eta_index'], e['eta'], color=COLORS[p], linewidth=1.3, linestyle=(0, (4, 2)),
                     label=f'{LABELS[p]}, estimate')
        top.semilogy(np.arange(len(e['residual'])), e['residual'], color=COLORS[p], linewidth=1.0,
                     linestyle=(0, (1, 1.5)), label=f'{LABELS[p]}, residual')
        residual_stop = np.flatnonzero(e['residual'] <= args.tol)
        if residual_stop.size:
            top.plot([residual_stop[0]], [e['true_error'][residual_stop[0]]], 'X', color=COLORS[p], markersize=8,
                     markeredgecolor=SURFACE, markeredgewidth=1.2)
        if e['stop'] is not None:
            top.plot([e['stop'] + 1], [e['error_returned']], 'o', color=COLORS[p], markersize=7,
                     markeredgecolor=SURFACE, markeredgewidth=1.5)
        valid = e['true_error'][e['eta_index']] > 1e-13
        bottom.plot(e['eta_index'][valid], e['eta'][valid] / e['true_error'][e['eta_index']][valid], color=COLORS[p],
                    linewidth=1.3, label=LABELS[p])
    top.axhline(args.tol, color=AXIS, linewidth=1, linestyle=(0, (2, 2)))
    top.axhspan(1e-12, entry['floor'], color=GRID, zorder=0)       # below the accuracy of s*: not measurable
    top.set_ylim(1e-12, 10)
    bottom.axhline(1.0, color=AXIS, linewidth=1, linestyle=(0, (2, 2)))
    bottom.set_ylim(0, 1.6)
    if column == 0:
        top.legend(frameon=False, fontsize=8, loc='lower left', ncol=2)
        bottom.legend(frameon=False, fontsize=8.5, loc='lower left')
fig.savefig(args.out + '.png', dpi=150, facecolor=SURFACE)
print(f'wrote {args.out}.npz and {args.out}.png')
