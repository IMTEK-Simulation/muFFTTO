"""Which preconditioner for the damage problem of the AT2 runs?

Offline test on saved states of the runs with the dual elastic preconditioner (exp_data/AT2_N<N>_dual<TAG>.nc):
for selected load steps the converged history H of the step is read, the damage system

    [Gc l B^T W B + N^T W c N] d = N^T W 2H,        c = Gc/l + 2H  (reaction coefficient, varies in space)

is set up exactly as in example_2D_phase_field_fracture_AT2.py and solved with CG (energy-error stopping,
tol 1e-6, delay 10, as in the runs) with every preconditioner below, from two initial guesses:
    warm   the damage of the previous load step (the first staggered iteration of the step)
    cold   zero (preconditioner quality without the help of the warm start)

Preconditioners (G(c_ref) = FFT inverse of the operator with the constant reaction c_ref):
    Green           G(Gc/l)                       (c_ref = smallest c: H = 0)
    Green_Jacobi    D^-1/2 G(Gc/l) D^-1/2         D = diag of the current operator (used in the runs)
    Jacobi          D^-1
    Green_mean      G(mean c)                     reference updated with the current H
    Green_Jacobi_mean  D^-1/2 G(mean c) D^-1/2
    dual            G(Gc/l) K(c_ref^2 / c) G(Gc/l)   the dual idea on the reaction coefficient (diffusion kept)

Output: table on stdout and exp_data/damage_preconditioner_study<TAG>.csv
    python damage_preconditioner_study.py [TAG] [GRIDS...]      (default _polydisperse_pores_l0.03125, 32 64 128)
"""
import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

import netCDF4
import numpy as np

from muFFTTO import domain
from muFFTTO import solvers
from muFFTTO import topology_optimization   # energy_stop_kwargs

HERE = os.path.dirname(os.path.realpath(__file__))
TAG = sys.argv[1] if len(sys.argv) > 1 else '_polydisperse_pores_l0.03125'
GRIDS = [int(n) for n in sys.argv[2:]] or [32, 64, 128]
PRECONDITIONERS = ['Green', 'Green_Jacobi', 'Jacobi', 'Green_mean', 'Green_Jacobi_mean', 'dual']
cg_tol, cg_delay = 1e-6, 10


def representative_steps(sigma):
    """load steps: intact (mid-way, just before the first break), first break, partly cracked, last break, broken"""
    drop = np.flatnonzero(np.r_[False, sigma[1:] < 0.95 * sigma[:-1]])
    steps = {'intact mid': drop[0] // 2, 'before 1st break': drop[0] - 1, '1st break': drop[0],
             'partly cracked': (drop[0] + drop[-1]) // 2, 'last break': drop[-1],
             'broken': min(drop[-1] + 5, len(sigma) - 1)}
    return steps


rows = []
for N in GRIDS:
    data = netCDF4.Dataset(os.path.join(HERE, 'exp_data', f'AT2_N{N}_dual{TAG}.nc'))
    Gc, length_scale = float(data.Gc), float(data.length_scale)
    cell = domain.PeriodicUnitCell(domain_size=[1, 1], problem_type='conductivity')
    disc = domain.Discretization(cell=cell, nb_of_pixels_global=(N, N), discretization_type='finite_element',
                                 element_type='bilinear_rectangle')
    damage = disc.get_unknown_size_field(name='damage')
    rhs = disc.get_unknown_size_field(name='rhs')
    reaction = disc.get_quad_field_scalar(name='reaction')
    dual_reaction = disc.get_quad_field_scalar(name='dual_reaction')
    grad = disc.get_gradient_size_field(name='grad')
    temp_q = disc.get_quad_field_scalar(name='temp_q')
    temp_n = disc.get_unknown_size_field(name='temp_n')
    temp_1 = disc.get_unknown_size_field(name='temp_1')
    temp_2 = disc.get_unknown_size_field(name='temp_2')
    x_scaled = disc.get_unknown_size_field(name='x_scaled')

    def apply_operator(x, Ax, c):
        """[Gc l B^T W B + N^T W c N] x (as in example_2D_phase_field_fracture_AT2.py)"""
        disc.apply_gradient_operator_mugrid(u_inxyz=x, grad_u_ijqxyz=grad)
        grad.s[...] *= Gc * length_scale
        disc.apply_gradient_transposed_operator_mugrid(gradient_field_ijqxyz=grad, div_u_fnxyz=Ax, apply_weights=True)
        disc.apply_N_operator_mugrid(x, temp_q)
        temp_q.s[...] *= c
        disc.apply_N_transposed_operator_mugrid(quad_field_ijqxyz=temp_q, nodal_field_inxyz=temp_n, apply_weights=True)
        Ax.s[...] += temp_n.s

    def K(x, Ax):
        apply_operator(x, Ax, reaction.s)

    def K_as_operator(input_field_inxyz, output_field_inxyz):
        apply_operator(input_field_inxyz, output_field_inxyz, reaction.s)

    def green(c_ref, name):
        return disc.get_preconditioner_Green_mugrid(
            reference_material_data_ijkl=None, invert_zero_mode=True, name=name,     # distinct names: no overwrite
            operator=lambda input_field_inxyz, output_field_inxyz: apply_operator(input_field_inxyz,
                                                                                   output_field_inxyz, c_ref))

    green_min = green(Gc / length_scale, 'green_min')

    sigma = np.asarray(data['macro_stress'][:, 0, 0])
    for label, k in representative_steps(sigma).items():
        reaction.s[...] = Gc / length_scale + 2.0 * np.asarray(data['history_qp'][k]).reshape(reaction.s.shape)
        dual_reaction.s[...] = (Gc / length_scale) ** 2 / reaction.s
        temp_q.s[...] = 2.0 * (reaction.s - Gc / length_scale)
        disc.apply_N_transposed_operator_mugrid(quad_field_ijqxyz=temp_q, nodal_field_inxyz=rhs, apply_weights=True)
        previous = np.asarray(data['damage'][k - 1]).reshape(damage.s.shape) if k > 0 else 0.0
        c = reaction.s
        jacobi = disc.get_preconditioner_Jacobi_mugrid(operator=K_as_operator, name='jacobi')
        green_mean = green(float(c.mean()), 'green_mean')

        def M(p):
            def apply(x, Px):
                if p == 'Green':
                    disc.apply_preconditioner_mugrid(preconditioner_Fourier_fnfnqks=green_min, input_nodal_field_fnxyz=x,
                                                     output_nodal_field_fnxyz=Px)
                elif p in ('Green_Jacobi', 'Green_Jacobi_mean'):
                    x_scaled.s[...] = jacobi.s * x.s
                    disc.apply_preconditioner_mugrid(
                        preconditioner_Fourier_fnfnqks=green_min if p == 'Green_Jacobi' else green_mean,
                        input_nodal_field_fnxyz=x_scaled, output_nodal_field_fnxyz=Px)
                    Px.s[...] *= jacobi.s
                elif p == 'Jacobi':
                    Px.s[...] = jacobi.s ** 2 * x.s
                elif p == 'Green_mean':
                    disc.apply_preconditioner_mugrid(preconditioner_Fourier_fnfnqks=green_mean, input_nodal_field_fnxyz=x,
                                                     output_nodal_field_fnxyz=Px)
                elif p == 'dual':
                    disc.apply_preconditioner_mugrid(preconditioner_Fourier_fnfnqks=green_min, input_nodal_field_fnxyz=x,
                                                     output_nodal_field_fnxyz=temp_1)
                    apply_operator(temp_1, temp_2, dual_reaction.s)
                    disc.apply_preconditioner_mugrid(preconditioner_Fourier_fnfnqks=green_min,
                                                     input_nodal_field_fnxyz=temp_2, output_nodal_field_fnxyz=Px)
            return apply

        for p in PRECONDITIONERS:
            for start in ('warm', 'cold'):
                damage.s[...] = previous if start == 'warm' else 0.0
                t0 = time.time()
                norms = solvers.conjugate_gradients_mugrid_experimental(
                    comm=disc.communicator, fc=disc.field_collection, hessp=K, b=rhs, x=damage, P=M(p),
                    maxiter=20000, stop_crit_norm='energy_lower_estim', d_min=cg_delay,
                    **topology_optimization.energy_stop_kwargs(disc, K, damage, cg_tol))[1]
                its = int(norms['stop_iteration']['energy_lower_estim'])
                rows.append((N, label, k, p, start, its, time.time() - t0, float(c.max() / c.min())))
        line = '  '.join(f'{p}={next(r[5] for r in rows[::-1] if r[:5] == (N, label, k, p, "cold"))}'
                         f'/{next(r[5] for r in rows[::-1] if r[:5] == (N, label, k, p, "warm"))}'
                         for p in PRECONDITIONERS)
        print(f'N={N:4d}  {label:17s} step {k:2d}  c_max/c_min = {c.max() / c.min():9.3g}   cold/warm: {line}',
              flush=True)

with open(os.path.join(HERE, 'exp_data', f'damage_preconditioner_study{TAG}.csv'), 'w') as f:
    f.write('N,state,step,preconditioner,start,cg_iterations,time,contrast\n')
    for r in rows:
        f.write(','.join(str(v) for v in r) + '\n')
