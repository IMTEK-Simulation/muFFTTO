"""Optimizer study on the phase-field formulation: the objective and adjoint gradient of example_2D_elasticity_TO.py
(imported unchanged: nodal phase field, SIMP p = 2 at the quadrature points, gradient + double-well energy with
ETA, 3 forward + 3 adjoint CG solves per evaluation, warm started, energy stop rtol 1e-6, delay 10), minimized by
    OPTIMIZER = lbfgs (scipy L-BFGS-B) | gcmma | mma   (mma.py, as in example_2D_elasticity_TO_simp_mma.py)
from the start design INIT_SEED (initial_design of the SIMP driver: the same 5 starts as the SIMP study), with the
stop rule of the SIMP study: relative decrease of the objective over the last 10 accepted iterations < STAGE_TOL
(1e-4), or MAXITER (500) iterations. One stage (no grid or eta continuation).

Final design: the phase field interpolated to the quadrature points and thresholded > 0.5 (exactly two-phase at the
quadrature points, as the material the optimizer sees, rounded), J of the stress match only.

    NPIX=64 SOFT_EXP=5 ETA=0.03125 OPTIMIZER=gcmma INIT_SEED=0 python example_2D_elasticity_TO_phase_field_optimizers.py Green_Jacobi
Output: data/example_2D_elasticity_TO_phase_field_optimizers/<name>.npz
"""
import os
import sys
import time

for key, value in (('NPIX', '64'), ('SOFT_EXP', '5'), ('ETA', '0.03125'), ('CG_MODE', 'energy'),
                   ('SHOW_PLOTS', '0')):
    os.environ.setdefault(key, value)
sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))

import numpy as np
import scipy.optimize

import example_2D_elasticity_TO as pf  # configuration from the environment and argv[1] (preconditioner)
from example_2D_elasticity_TO_simp_mma import initial_design
from mma import GCMMA, MMA

pf.print = lambda *args, **kwargs: None  # the base objective prints every CG solve
optimizer_type = os.environ.get('OPTIMIZER', 'lbfgs')
init_seed = int(os.environ.get('INIT_SEED', 0))
stage_tol = float(os.environ.get('STAGE_TOL', 1e-4))
max_iterations = int(os.environ.get('MAXITER', 500))
max_inner, move = 10, 0.2
n = pf.n_pix

evaluations = []  # per evaluation: objective, J (stress match), CG iterations (forward + adjoint, all load cases)


def evaluate(x):
    nb_mech, nb_adjoint = len(pf.info_mech['num_iteration_adjoint']), len(pf.info_adjoint['num_iteration_adjoint'])
    F, gradient = pf.objective_function_multiple_load_cases(np.ascontiguousarray(x).ravel())
    J = pf.w * float(pf.norms_sigma[-1][0])
    cg = sum(pf.info_mech['num_iteration_adjoint'][nb_mech:]) + sum(pf.info_adjoint['num_iteration_adjoint'][nb_adjoint:])
    evaluations.append((float(F), J, cg))
    return float(F), np.asarray(gradient).reshape(x.shape), J


if __name__ == '__main__':
    start = time.time()
    data_folder = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data',
                               'example_2D_elasticity_TO_phase_field_optimizers')
    os.makedirs(data_folder, exist_ok=True)
    name = (f'N{n}_soft{pf.soft_phase:g}_{pf.preconditioner_type}_{optimizer_type}_eta{pf.eta:g}'
            f'_dw{pf.double_well_depth_test:g}_seed{init_seed}' + os.environ.get('RUN_SUFFIX', ''))
    print(f'run {name}', flush=True)
    history = {k: [] for k in ('F', 'J', 'time', 'nb_evaluations')}

    def record(F, J, iteration, extra=0):
        for key, value in (('F', F), ('J', J), ('time', time.time() - start), ('nb_evaluations', len(evaluations))):
            history[key].append(value)
        print(f'it {iteration:4d}  objective = {F:.6e}  J (stress match) = {J:.6e}  CG/evaluation '
              f'{evaluations[-1][2]:5d}' + (f'  [{extra} extra evaluations]' if extra else ''), flush=True)

    def stage_done(F_stage, iteration):
        return iteration >= max_iterations or (len(F_stage) > 10 and
                                               F_stage[-11] - F_stage[-1] < stage_tol * abs(F_stage[-1]))

    x = initial_design(n, init_seed)
    if optimizer_type == 'lbfgs':
        evaluated, last = {}, {'iterations': 0}

        def fun(x_flat):
            F, gradient, J = evaluate(x_flat.reshape(x.shape))
            evaluated[x_flat.tobytes()] = J
            return F, gradient.ravel()

        F_stage = [fun(x.ravel())[0]]
        record(F_stage[0], evaluations[-1][1], 0)

        def callback(intermediate_result):  # record the accepted iterate, not the last line-search trial
            last['iterations'] += 1
            record(intermediate_result.fun, evaluated[intermediate_result.x.tobytes()], last['iterations'])
            evaluated.clear()
            F_stage.append(intermediate_result.fun)
            if stage_done(F_stage, 0):
                raise StopIteration

        result = scipy.optimize.minimize(fun, x.ravel(), jac=True, method='L-BFGS-B', bounds=[(0, 1)] * x.size,
                                         callback=callback, options=dict(maxiter=max_iterations, ftol=1e-12, gtol=0,
                                                                         maxcor=10))
        x = result.x.reshape(x.shape)
        print(f'L-BFGS-B: {result.message}')
    elif optimizer_type == 'gcmma':
        optimizer = GCMMA(x.size)
        F, gradient, J = evaluate(x)
        F_stage, iteration, nb_inner = [F], 0, 0
        while True:
            record(F, J, iteration, nb_inner)
            if stage_done(F_stage, iteration):
                break
            x_trial, nb_inner = optimizer.outer(x.ravel(), F, gradient.ravel()).reshape(x.shape), 0
            while True:
                F_trial, gradient_trial, J_trial = evaluate(x_trial)
                if optimizer.conservative(F_trial) or nb_inner == max_inner:
                    break
                nb_inner += 1
                x_trial = optimizer.inner(F_trial).reshape(x.shape)
            if not optimizer.conservative(F_trial) and F_trial >= F:
                print(f'no conservative point after {max_inner} inner iterations, converged')
                break
            x, F, gradient, J, iteration = x_trial, F_trial, gradient_trial, J_trial, iteration + 1
            F_stage.append(F)
    else:
        optimizer = MMA(x.size, move=move)
        F, gradient, J = evaluate(x)
        F_stage, iteration, rejected = [F], 0, 0
        while True:
            record(F, J, iteration, rejected)
            if stage_done(F_stage, iteration):
                break
            x_trial, rejected = optimizer.update(x.ravel(), gradient.ravel()).reshape(x.shape), 0
            while True:
                F_trial, gradient_trial, J_trial = evaluate(x_trial)
                if F_trial < F or rejected == 6:
                    break
                rejected += 1
                x_trial = optimizer.retry().reshape(x.shape)
            if F_trial >= F:
                print(f'no descent after {rejected} conservative retries, converged')
                break
            x, F, gradient, J, iteration = x_trial, F_trial, gradient_trial, J_trial, iteration + 1
            F_stage.append(F)

    # the final phase field: J of the continuous design, and of its threshold at the quadrature points
    F_final, _, J_final = evaluate(x)
    nb_search_evaluations = len(evaluations) - 1
    pf.phase_field_1nxyz.s[0, 0] = x
    pf.discretization.apply_N_operator_mugrid(pf.phase_field_1nxyz, pf.phase_field_at_quad_poits_1qxyz)
    sharp_quad = (np.asarray(pf.phase_field_at_quad_poits_1qxyz.s[0, 0]) > 0.5).astype(float)  # [q, x, y]
    pf.quad_point_density = lambda rho_q: (rho_q > 0.5).astype(float)
    _, _, J_sharp = evaluate(x)
    pf.quad_point_density = None
    print(f'final: objective = {F_final:.6e}, J = {J_final:.6e} (phase field), J = {J_sharp:.6e} (thresholded at '
          f'the quadrature points, two-phase), {nb_search_evaluations} evaluations, '
          f'{sum(e[2] for e in evaluations[:nb_search_evaluations]):.0f} CG iterations, {time.time() - start:.1f} s',
          flush=True)
    np.savez(os.path.join(data_folder, name + '.npz'), **{k: np.asarray(v) for k, v in history.items()},
             evaluations=np.asarray(evaluations[:nb_search_evaluations]), phase_field=x, sharp_quad=sharp_quad,
             J_final=J_final, F_final=F_final, J_sharp=J_sharp, elapsed_time=time.time() - start,
             soft_phase=pf.soft_phase, eta=pf.eta, p=pf.p, double_well_depth=pf.double_well_depth_test)
    print(f'data: {os.path.join(data_folder, name + ".npz")}')
