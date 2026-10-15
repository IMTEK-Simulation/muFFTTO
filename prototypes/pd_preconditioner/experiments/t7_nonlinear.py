"""T7 nonlinear elasticity (Ladecky et al. 2026, Sec. 4.5, in 2D): linear-elastic square inclusion (area 1/4) in a
power-law matrix, macroscopic shear eps6 = 0.025 in one load step, full Newton steps. Every Newton step solves
K_t du = -r with PCG (G, GJ, PD); the right-hand side is a general Newton residual, not the macroscopic load.

Reference materials: G -> inclusion stiffness, GJ -> identity (both as in the paper), PD -> inclusion stiffness and
identity ('PD_I'). Newton stops when ||B du|| <= 1e-5 ||E||; CG stops when ||r_k|| <= 1e-5 ||r_0||.
"""
import sys

import numpy as np

from common import COLORS, NAMES, RESULTS, SURFACE, TEXT_1, TEXT_2, plt, style, write_csv
from fem import B, BT, apply_K, weight
from nonlinear import linear_isotropic, stress_and_tangent
from pcg import pcg
from precond import make_green, make_green_jacobi, make_primal_dual_tangent, new_counter

MU0 = 1.0
C_INC = linear_isotropic(2.0 * MU0, MU0)
PARAMETERS = dict(K=20.0 * MU0, sigma0=5.0 * MU0, eps0=0.01)
E = np.array([0.0, 0.0, 0.025])
METHODS = ('G', 'GJ', 'PD', 'PD_I')
COLORS = dict(COLORS, PD_I='#eda100')
NAMES = dict(NAMES, PD_I='primal-dual Green, C_ref = I')


def solve_newton(n, omega, method, newton_tol=1e-5, max_newton=50):
    d = 2
    x = (np.arange(n) + 0.5) / n
    x1, x2 = np.meshgrid(x, x, indexing='ij')
    matrix = ~((np.abs(x1 - 0.5) < 0.25) & (np.abs(x2 - 0.5) < 0.25))
    w = weight(n)
    E_field = np.broadcast_to(E[None, :, None, None], (2, 3, n, n))
    u = np.zeros((d, n, n))
    cg, cg_energy, steps = [], [], []
    for it in range(max_newton):
        eps = E_field + B(u)
        sigma, C_t = stress_and_tangent(eps, matrix, omega=omega, C_inclusion=C_INC, **PARAMETERS)
        f = -BT(w * sigma)
        counter = new_counter()
        if method == 'G':
            M = make_green(C_INC, d, n, counter)
        elif method == 'GJ':
            M = make_green_jacobi(C_t, np.eye(3), d, n, counter)
        elif method == 'PD':
            M = make_primal_dual_tangent(C_t, C_INC, d, n, counter)
        else:
            M = make_primal_dual_tangent(C_t, np.eye(3), d, n, counter)
        result = pcg(C_t, f, M, counter, tol=1e-5, maxiter=5000)
        du = result['x']
        u += du
        cg.append(result['iterations'])
        cg_energy.append(result['it_energy'])
        step = np.linalg.norm(B(du)) / np.linalg.norm(E_field)
        steps.append(step)
        if step <= newton_tol:
            break
    sigma, _ = stress_and_tangent(E_field + B(u), matrix, omega=omega, C_inclusion=C_INC, **PARAMETERS)
    mean_sigma = sigma.mean(axis=(0, 2, 3))
    return dict(cg=cg, cg_energy=cg_energy, steps=steps, mean_sigma=mean_sigma)


if __name__ == '__main__':
    ns = [int(a) for a in sys.argv[1:]] or [16, 32, 64, 128, 256]
    rows, per_step = [], {}
    for omega in (3, 4, 5):
        for n in ns:
            for method in METHODS:
                out = solve_newton(n, omega, method)
                per_step[(omega, n, method)] = out['cg']
                rows.append(dict(omega=omega, n=n, method=method, newton_iterations=len(out['cg']),
                                 cg_total=sum(out['cg']), cg_per_newton=' '.join(str(c) for c in out['cg']),
                                 cg_energy_per_newton=' '.join(str(c) for c in out['cg_energy']),
                                 last_newton_step=f"{out['steps'][-1]:.2e}",
                                 mean_sigma6=f"{out['mean_sigma'][2]:.6e}"))
                print(rows[-1], flush=True)
    write_csv('t7_nonlinear_elastic.csv', rows)

    fig, axes = plt.subplots(1, 2, figsize=(15, 5.6), facecolor=SURFACE)
    fig.subplots_adjust(left=0.06, right=0.98, top=0.8, bottom=0.12, wspace=0.2)
    fig.text(0.06, 0.94, 'T7 nonlinear (power-law) elasticity: PCG iterations in Newton', fontsize=14,
             fontweight='bold', color=TEXT_1)
    fig.text(0.06, 0.89, 'linear square inclusion in a power-law matrix, shear 0.025 in one step; '
                         'line style: omega = 3 solid, 4 dashed, 5 dotted', fontsize=10, color=TEXT_2)
    dashes = {3: '-', 4: (0, (5, 2)), 5: (0, (1, 1.5))}
    n_max = max(ns)
    for omega in (3, 4, 5):
        for method in METHODS:
            y = per_step[(omega, n_max, method)]
            axes[0].plot(range(len(y)), y, color=COLORS[method], linestyle=dashes[omega], linewidth=2, marker='o',
                         markersize=4, label=NAMES[method] if omega == 3 else None)
            totals = [sum(per_step[(omega, n, method)]) for n in ns]
            axes[1].plot(ns, totals, color=COLORS[method], linestyle=dashes[omega], linewidth=2, marker='o',
                         markersize=4)
    axes[0].set_yscale('log')
    axes[1].set_xscale('log', base=2)
    axes[1].set_yscale('log')
    axes[1].set_xticks(ns, [str(n) for n in ns])
    for ax in axes:
        ax.grid(axis='y', color='#e6e5e1', linewidth=0.8)
    style(axes[0], f'(a) PCG iterations per Newton iteration, n = {n_max}', 'Newton iteration i', 'PCG iterations')
    style(axes[1], '(b) total PCG iterations vs grid', 'n (nodes per direction)', 'total PCG iterations')
    axes[0].legend(frameon=False, fontsize=9.5)
    fig.savefig(f'{RESULTS}/t7_nonlinear_elastic.png', dpi=150, facecolor=SURFACE)
