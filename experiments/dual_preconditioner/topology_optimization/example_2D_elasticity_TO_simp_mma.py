"""Inverse homogenization by density topology optimization: SIMP + periodic Helmholtz filter + Heaviside projection
with beta continuation, optimized by MMA (own implementation, mma.py), with grid continuation (coarse -> fine).
Same problem as example_2D_elasticity_TO.py: match the target stresses of the auxetic tangent under the 3 unit macro
strains, J = w/3 sum_a |sigma_t^a - <sigma>^a|^2 / |sigma_t^a|^2 (w = 5), linear triangles, void = 10^-SOFT_EXP C_0.

Design: one variable x per pixel in [0, 1]
    x_f   = F x                    periodic Helmholtz filter, FFT symbol 1 / (1 + r^2 |k|^2), r = FILTER_R / (2 sqrt 3)
    rho   = H_beta(x_f)            Heaviside projection, threshold 0.5
    C     = C_void + rho^p (C_0 - C_void)   at both quadrature points (triangles) of the pixel, SIMP p = 3

Sensitivities without adjoint solves: J depends only on the homogenized stresses, and the cell problem is
self-adjoint, so for any macro tensor T: T : d<sigma>^a / d rho_q = w_q / |Y| e^T : dC/drho_q : e^a, with e^a the
total strain of load case a and e^T = sum_b alpha_b e^b the total strain of the macro strain T (a combination of the 3
solved load cases). 3 CG solves per design (the base script: 3 forward + 3 adjoint).

    python example_2D_elasticity_TO_simp_mma.py [Green | Green_Jacobi | auto]
        auto: Green-Jacobi while beta < BETA_SWITCH (grey design), Green from then on (nearly two-phase design)
    env SCHEDULE   stages 'n:beta:maxiter,...' (default 32:1:80,32:2:40,32:4:40,64:4:30,64:8:40,64:16:40,64:32:60)
        FILTER_R   filter length in cell units (default 2/64, i.e. 2 pixels at 64^2)
        OPTIMIZER (mma | gcmma | lbfgs: scipy L-BFGS-B; all on the same stages and stage stop), INIT_SEED (0), BETA_SWITCH (8), SOFT_EXP (5), SIMP_P (3), MOVE (0.2, MMA move limit), STAGE_TOL (1e-4: a stage ends when the
        relative decrease of J over the last 5 iterations is below it), RUN_SUFFIX
CG: energy-error stop, rtol 1e-6, delay 10, warm start from the previous design's solution (prolongated on a grid
change). Output: data/example_2D_elasticity_TO_simp_mma/<name>.npz (history, designs per grid, final tangent).
"""
import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))

import numpy as np

from muFFTTO import domain, material_models, solvers, topology_optimization
from mma import GCMMA, MMA
import scipy.optimize

prec_mode = sys.argv[1] if len(sys.argv) > 1 else 'auto'
schedule = [tuple(float(v) for v in stage.split(':')) for stage in os.environ.get(
    'SCHEDULE', '32:1:80,32:2:40,32:4:40,64:4:30,64:8:40,64:16:40,64:32:60').split(',')]
filter_length = float(os.environ.get('FILTER_R', 2 / 64))
beta_switch = float(os.environ.get('BETA_SWITCH', 8))
soft_phase = 10.0 ** -int(os.environ.get('SOFT_EXP', 5))
simp_p = float(os.environ.get('SIMP_P', 3))
move = float(os.environ.get('MOVE', 0.2))
stage_tol = float(os.environ.get('STAGE_TOL', 1e-4))
optimizer_type = os.environ.get('OPTIMIZER', 'mma')  # 'mma', 'gcmma' or 'lbfgs' (scipy L-BFGS-B), same stages
init_seed = int(os.environ.get('INIT_SEED', 0))  # start design, see initial_design
max_inner = 10  # GCMMA: inner iterations per outer iteration (each one evaluation)
cg_rtol, cg_delay = 1e-6, 10
weight, nb_load_cases, dim = 5.0, 3, 2

# material and target (as example_2D_elasticity_TO.py)
K_0, G_0 = 1.0, 0.5
C_0 = material_models.get_elastic_material_tensor(dim=dim, K=K_0, mu=G_0, kind='linear')
C_void = soft_phase * C_0
E_0 = 9 * K_0 * G_0 / (3 * K_0 + G_0)
poisson_target, G_target = -0.5, (3 / 20) * E_0
K_target, G_target = material_models.get_bulk_and_shear_modulus(E=2 * G_target * (1 + poisson_target),
                                                                poisson=poisson_target)
C_target = material_models.get_elastic_material_tensor(dim=dim, K=K_target, mu=G_target, kind='linear')
macro_gradients = np.array([[[1.0, 0.0], [0.0, 0.0]], [[0.0, 0.0], [0.0, 1.0]], [[0.0, 0.5], [0.5, 0.0]]])
target_stresses = np.einsum('ijkl,akl->aij', C_target, macro_gradients)
cell = domain.PeriodicUnitCell(domain_size=[1, 1], problem_type='elasticity')


class Grid:
    """Discretization, material, preconditioners and solution fields of one n x n grid."""

    def __init__(self, n):
        self.n = n
        self.disc = domain.Discretization(cell=cell, nb_of_pixels_global=(n, n),
                                          discretization_type='finite_element', element_type='linear_triangles')
        d = self.disc
        self.weights = np.asarray(d.quadrature_weights)  # physical, per quadrature point
        self.scale = d.get_quad_field_scalar(name='simp_scale')
        self.material = domain.ScaledMaterial([(self.scale, C_0 - C_void), (1.0, C_void)])
        self.green = {'Green': d.get_preconditioner_Green_mugrid(reference_material_data_ijkl=C_0, name='green_C0'),
                      'Green_Jacobi': d.get_preconditioner_Green_mugrid(
                          reference_material_data_ijkl=2 * G_0 * np.einsum('ik,jl->ijkl', np.eye(dim), np.eye(dim)),
                          name='green_2muI')}
        self.u = [d.get_unknown_size_field(name=f'u_{a}') for a in range(nb_load_cases)]
        self.rhs = d.get_unknown_size_field(name='rhs')
        self.temp = d.get_unknown_size_field(name='gj_temp')
        self.strain = d.get_gradient_size_field(name='strain')
        self.macro = []
        for a in range(nb_load_cases):
            field = d.get_gradient_size_field(name=f'macro_{a}')
            d.get_macro_gradient_field_mugrid(macro_gradient_ij=macro_gradients[a], macro_gradient_field_ijqxyz=field)
            self.macro.append(field)
        k = 2 * np.pi * np.fft.fftfreq(n, d=1 / n)
        k2 = k[:, None] ** 2 + k[None, :] ** 2
        self.filter_symbol = 1 / (1 + (filter_length / (2 * np.sqrt(3))) ** 2 * k2)

    def filter(self, x):
        return np.real(np.fft.ifft2(np.fft.fft2(x) * self.filter_symbol))

    def preconditioner(self, name):
        d = self.disc
        if name == 'Green':
            def M(x, Px):
                d.fft.communicate_ghosts(x)
                d.apply_preconditioner_mugrid(preconditioner_Fourier_fnfnqks=self.green['Green'],
                                              input_nodal_field_fnxyz=x, output_nodal_field_fnxyz=Px)
            return M
        diagonal = d.get_preconditioner_Jacobi_mugrid(material_data_field_ijklqxyz=self.material)

        def M(x, Px):
            d.fft.communicate_ghosts(x)
            self.temp.s[...] = diagonal.s * x.s
            d.apply_preconditioner_mugrid(preconditioner_Fourier_fnfnqks=self.green['Green_Jacobi'],
                                          input_nodal_field_fnxyz=self.temp, output_nodal_field_fnxyz=Px)
            Px.s[...] = diagonal.s * Px.s
            d.fft.communicate_ghosts(Px)
        return M

    def solve(self, rho, prec):
        """3 cell problems for the pixel density rho: homogenized stresses, total strains [a, i, j, q, x, y] and the
        CG iterations of each solve."""
        d = self.disc
        self.scale.s[0, 0] = (rho ** simp_p)[None]
        M = self.preconditioner(prec)

        def K(x, Kx):
            d.apply_system_matrix_mugrid(material_data_field=self.material, input_field_inxyz=x, output_field_inxyz=Kx,
                                         formulation='small_strain')

        stresses, strains, iterations = np.zeros((nb_load_cases, dim, dim)), [], []
        for a in range(nb_load_cases):
            self.rhs.s.fill(0)
            d.get_rhs_mugrid(material_data_field_ijklqxyz=self.material, macro_gradient_field_ijqxyz=self.macro[a],
                             rhs_inxyz=self.rhs)
            count = []
            solvers.conjugate_gradients_mugrid_experimental(
                comm=d.communicator, fc=d.field_collection, hessp=K, b=self.rhs, x=self.u[a], P=M, maxiter=100000,
                callback=lambda *args: count.append(1), stop_crit_norm='energy_lower_estim', d_min=cg_delay,
                **topology_optimization.energy_stop_kwargs(d, K, self.u[a], cg_rtol))
            iterations.append(len(count))
            d.fft.communicate_ghosts(self.u[a])
            d.apply_gradient_operator_symmetrized_mugrid(u_inxyz=self.u[a], grad_u_ijqxyz=self.strain)
            strain = np.array(self.strain.s) + np.array(self.macro[a].s)
            strains.append(strain)
            stiffness = self.scale.s[0, 0] * (1 - soft_phase) + soft_phase  # [q, x, y]
            stresses[a] = np.einsum('ijkl,klqxy,qxy,q->ij', C_0, strain, stiffness, self.weights) / cell.domain_volume
        return stresses, np.array(strains), iterations

    def prolongate_solution(self, coarse):
        """Warm start on this (2x finer) grid: the coarse displacements at the even nodes, linear in between."""
        for a in range(nb_load_cases):
            u = np.array(coarse.u[a].s)  # [i, n, x, y]
            fine = np.zeros(u.shape[:2] + (self.n, self.n))
            fine[..., ::2, ::2] = u
            fine[..., 1::2, ::2] = (u + np.roll(u, -1, -2)) / 2
            fine[..., :, 1::2] = (fine[..., :, ::2] + np.roll(fine[..., :, ::2], -1, -1)) / 2
            self.u[a].s[...] = fine


def projection(x_f, beta):
    t = np.tanh(beta * 0.5)
    rho = (t + np.tanh(beta * (x_f - 0.5))) / (2 * t)
    return rho, beta * (1 - np.tanh(beta * (x_f - 0.5)) ** 2) / (2 * t)


evaluations = []  # every design evaluation (accepted or rejected): n, beta, mean CG iterations per solve


def evaluate(grid, x, beta, prec):
    """J and dJ/dx for the pixel design x on a grid."""
    x_f = grid.filter(x)
    rho, drho = projection(x_f, beta)
    stresses, strains, iterations = grid.solve(rho, prec)
    evaluations.append((grid.n, beta, np.mean(iterations)))
    misfit = target_stresses - stresses
    norms = np.sum(target_stresses ** 2, axis=(1, 2))
    J = weight / nb_load_cases * np.sum(np.sum(misfit ** 2, axis=(1, 2)) / norms)
    T = -2 * weight / nb_load_cases * misfit / norms[:, None, None]  # dJ/d<sigma>^a
    alpha = np.stack([T[:, 0, 0], T[:, 1, 1], T[:, 0, 1] + T[:, 1, 0]], axis=1)  # T^a = sum_b alpha_ab E^b
    strain_T = np.einsum('ab,bijqxy->aijqxy', alpha, strains)
    dJ_drho_q = np.einsum('aijqxy,ijkl,aklqxy,q->qxy', strain_T, C_0 - C_void, strains, grid.weights) / cell.domain_volume
    dJ_drho = simp_p * rho ** (simp_p - 1) * dJ_drho_q.sum(0)
    return J, grid.filter(dJ_drho * drho), dict(rho=rho, stresses=stresses, iterations=iterations)


def initial_design(n, seed):
    """Sin pattern + uniform noise, scaled to [0, 1]. Seed 0: the start of the phase-field runs (no phase shift);
    seed s > 0: random phase shifts of the two sines and its own noise (rng seed s)."""
    if seed == 0:
        np.random.seed(0)
        shift, noise = (0.0, 0.0), np.random.rand(n, n)
    else:
        rng = np.random.default_rng(seed)
        shift, noise = tuple(rng.uniform(0, 2 * np.pi, 2)), rng.random((n, n))
    x = (np.arange(n) + 0.5) / n
    X, Y = np.meshgrid(x, x, indexing='ij')
    rho = (np.sin(X * 4 * np.pi + shift[0]) + np.sin(Y * 4 * np.pi + shift[1]) + 2) / 4 + 0.5 * noise
    return (rho - rho.min()) / (rho.max() - rho.min())


def homogenized_voigt(stresses):
    return np.array([[s[0, 0], s[1, 1], s[0, 1]] for s in stresses]).T


if __name__ == '__main__':
    start = time.time()
    here = os.path.dirname(os.path.realpath(__file__))
    data_folder = os.path.join(here, 'data', 'example_2D_elasticity_TO_simp_mma')
    os.makedirs(data_folder, exist_ok=True)
    tag = '-'.join(f'{int(n)}b{beta:g}' for n, beta, _ in schedule)
    name = (f'soft{soft_phase:g}_{prec_mode}_{optimizer_type}_p{simp_p:g}_r{filter_length * 64:g}px64_{tag}_seed{init_seed}'
            + (f'_switch{beta_switch:g}' if prec_mode == 'auto' else '') + os.environ.get('RUN_SUFFIX', ''))
    print(f'run {name}')

    grids, history, designs = {}, {k: [] for k in ('n', 'beta', 'J', 'cg', 'time', 'grey', 'prec')}, {}
    x, grid, optimizer = None, None, None
    for n, beta, max_iterations in schedule:
        n, max_iterations = int(n), int(max_iterations)
        if n not in grids:
            previous = grid
            grids[n] = grid = Grid(n)
            if x is None:
                x = initial_design(n, init_seed)
            else:  # pixel design onto the finer grid (exact), displacements prolongated
                x = np.kron(x, np.ones((n // x.shape[0],) * 2))
                if previous is not None and n == 2 * previous.n:
                    grid.prolongate_solution(previous)
        grid = grids[n]
        prec = prec_mode if prec_mode != 'auto' else ('Green_Jacobi' if beta < beta_switch else 'Green')

        def record(J, info, iteration, rejected=0):
            rho = info['rho']
            grey = np.mean((rho > 0.01) & (rho < 0.99))
            for key, value in (('n', n), ('beta', beta), ('J', J), ('cg', np.mean(info['iterations'])),
                               ('time', time.time() - start), ('grey', grey), ('prec', prec)):
                history[key].append(value)
            designs.setdefault(n, []).append(rho)
            print(f'n {n:4d} beta {beta:4g} it {iteration:3d}  J = {J:.6e}  CG/solve '
                  f'{np.mean(info["iterations"]):6.1f} ({prec})  grey {100 * grey:4.1f}%'
                  + (f'  [{rejected} extra evaluations: rejected (MMA) / inner (GCMMA)]' if rejected else ''), flush=True)

        if optimizer_type == 'lbfgs':  # L-BFGS-B on the same stage, bounds and gradient (comparison with MMA)
            evaluated, last = {}, {'iterations': 0}  # evaluated: design bytes -> its evaluation (line-search trials too)

            def fun(x_flat):
                J, gradient, info = evaluate(grid, x_flat.reshape(x.shape), beta, prec)
                evaluated[x_flat.tobytes()] = info
                return J, gradient.ravel()

            J_stage = [evaluate(grid, x, beta, prec)[0]]

            def callback(intermediate_result):  # the stage stop of the MMA branch
                last['iterations'] += 1  # record the accepted iterate itself, not the last line-search trial
                record(intermediate_result.fun, evaluated[intermediate_result.x.tobytes()], last['iterations'])
                evaluated.clear()
                J_stage.append(intermediate_result.fun)
                if len(J_stage) > 10 and J_stage[-11] - J_stage[-1] < stage_tol * J_stage[-1]:
                    raise StopIteration

            result = scipy.optimize.minimize(fun, x.ravel(), jac=True, method='L-BFGS-B', bounds=[(0, 1)] * x.size,
                                             callback=callback, options=dict(maxiter=max_iterations, ftol=1e-12,
                                                                             gtol=0, maxcor=10))
            x = result.x.reshape(x.shape)
            J, _, info = evaluate(grid, x, beta, prec)
            record(J, info, last['iterations'])
            print(f'n {n:4d} beta {beta:4g}: L-BFGS-B {result.message}')
            continue
        if optimizer_type == 'gcmma':  # outer iterations: conservative inner loop, monotone descent
            if optimizer is None or optimizer.x_min.size != x.size:  # new asymptotes on a new grid only
                optimizer = GCMMA(x.size)
            J, gradient, info = evaluate(grid, x, beta, prec)
            J_stage, iteration, nb_inner = [J], 0, 0
            while True:
                record(J_stage[-1], info, iteration, nb_inner)
                if iteration >= max_iterations or (len(J_stage) > 10 and
                                                   J_stage[-11] - J_stage[-1] < stage_tol * J_stage[-1]):
                    break
                x_trial, nb_inner = optimizer.outer(x.ravel(), J, gradient.ravel()).reshape(x.shape), 0
                while True:
                    J_trial, gradient_trial, info = evaluate(grid, x_trial, beta, prec)
                    if optimizer.conservative(J_trial) or nb_inner == max_inner:
                        break
                    nb_inner += 1
                    x_trial = optimizer.inner(J_trial).reshape(x.shape)
                if not optimizer.conservative(J_trial) and J_trial >= J:
                    print(f'n {n:4d} beta {beta:4g}: no conservative point after {max_inner} inner iterations, '
                          f'stage converged')
                    break
                x, J, gradient, iteration = x_trial, J_trial, gradient_trial, iteration + 1
                J_stage.append(J)
            continue
        if optimizer is None or optimizer.x_min.size != x.size:  # new asymptotes on a new grid only
            optimizer = MMA(x.size, move=move)
        J, gradient, info = evaluate(grid, x, beta, prec)  # the stage's start (a new beta changes J)
        J_stage, iteration, rejected = [J], 0, 0
        while True:
            record(J_stage[-1], info, iteration, rejected)
            if iteration >= max_iterations or (len(J_stage) > 10 and J_stage[-11] - J_stage[-1] < stage_tol * J_stage[-1]):
                break
            # conservative MMA: a step that does not decrease J is redone with tighter asymptotes; 6 failures in a
            # row: no descent left at this beta, the stage is converged (x stays)
            x_trial, rejected = optimizer.update(x.ravel(), gradient.ravel()).reshape(x.shape), 0
            while True:
                J_trial, gradient_trial, info = evaluate(grid, x_trial, beta, prec)
                if J_trial < J_stage[-1] or rejected == 6:
                    break
                rejected += 1
                x_trial = optimizer.retry().reshape(x.shape)
            if J_trial >= J_stage[-1]:
                print(f'n {n:4d} beta {beta:4g}: no descent after {rejected} conservative retries, stage converged')
                break
            x, gradient, iteration = x_trial, gradient_trial, iteration + 1
            J_stage.append(J_trial)

    # the final design as an exactly two-phase material: threshold of the projected density, Green
    rho_final = designs[grid.n][-1]
    sharp = (rho_final > 0.5).astype(float)
    stresses_sharp, _, iterations_sharp = grid.solve(sharp, 'Green')
    misfit = target_stresses - stresses_sharp
    J_sharp = weight / nb_load_cases * np.sum(np.sum(misfit ** 2, axis=(1, 2)) / np.sum(target_stresses ** 2, axis=(1, 2)))
    C_final = homogenized_voigt(stresses_sharp)
    print(f'final: J = {history["J"][-1]:.6e} (projected), J = {J_sharp:.6e} (thresholded, two-phase), '
          f'{len(evaluations)} evaluations, {3 * sum(e[2] for e in evaluations):.0f} CG iterations, '
          f'{time.time() - start:.1f} s')
    print('homogenized C (Voigt) of the thresholded design:\n', np.array2string(C_final, precision=5))
    np.savez(os.path.join(data_folder, name + '.npz'),
             **{k: np.asarray(v) for k, v in history.items()}, evaluations=np.asarray(evaluations),
             **{f'design_N{n}': np.asarray(v) for n, v in designs.items()},
             sharp_design=sharp, J_sharp=J_sharp, C_sharp_voigt=C_final, C_target_voigt=homogenized_voigt(
                 np.einsum('ijkl,akl->aij', C_target, macro_gradients)),
             elapsed_time=time.time() - start, filter_length=filter_length, soft_phase=soft_phase)
    print(f'data: {os.path.join(data_folder, name + ".npz")}')
