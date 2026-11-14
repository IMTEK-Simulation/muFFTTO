"""When is the dual preconditioner (G K(C_ref C^-1 C_ref) G) better than Green (G = K_ref^-1)?
In 1D (and for a laminate, where every macro load gives a 1D fluctuation) the dual is the exact inverse for any C(x).
Test: crack-like soft bands, stiffness s(x) = (1 - d)^2 (1 - k) + k with the AT2 profile d = exp(-dist / l), contrast
1/k = 1e4, l = 4h, 256^2, linear triangles; CG from zero, energy-error stop 1e-6, delay d_min 10 (as in every study).
Geometries:  band      one straight crack through the cell (x = 1/2): a laminate
             band2ph   the same, two-phase (s = k where d > 1/2, else 1): sharp jump
             finite    a crack of length 1/2 with two tips (x = 1/2, 1/4 < y < 3/4)
             cross     two crossing straight cracks (x = 1/2 and y = 1/2)
             disc      a circular soft inclusion (radius 1/4), diffuse rim
Loads: E_xx (across the band x = 1/2), E_yy (along it), E_xy (shear).
    python crack_orientation_test.py [N=256]   ->  crack_orientation_results[_N<N>][_dmin<d>].csv
env: DMIN (CG energy-stop delay d_min, default 10; 1 shows how fast an exact preconditioner really converges),
     PRECS (default Green,dual; also Green_Jacobi: C_ref = 2 mu I with Jacobi scaling, as in the studies;
     dual+GJ: additive combination P_dual + P_GJ, cost about one dual + one Green-Jacobi application)
"""
import csv
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

import numpy as np

from muFFTTO import domain, material_models, solvers

HERE = os.path.dirname(os.path.realpath(__file__))
N = int(sys.argv[1]) if len(sys.argv) > 1 else 256
K_RES, L = 1e-4, 4.0 / N
dim = 2
C_0 = material_models.get_elastic_material_tensor(dim=dim, K=1.0, mu=0.5, kind='linear')
LOADS = {'E_xx (across)': np.array([[1.0, 0.0], [0.0, 0.0]]), 'E_yy (along)': np.array([[0.0, 0.0], [0.0, 1.0]]),
         'E_xy (shear)': np.array([[0.0, 0.5], [0.5, 0.0]])}

cell = domain.PeriodicUnitCell(domain_size=[1, 1], problem_type='elasticity')
disc = domain.Discretization(cell=cell, nb_of_pixels_global=(N, N), discretization_type='finite_element',
                             element_type='linear_triangles')
green_C0 = disc.get_preconditioner_Green_mugrid(reference_material_data_ijkl=C_0, name='green_C0')
green_2muI = disc.get_preconditioner_Green_mugrid(
    reference_material_data_ijkl=2 * 0.5 * np.einsum('ik,jl->ijkl', np.eye(dim), np.eye(dim)), name='green_2muI')
DMIN = int(os.environ.get('DMIN', 10))
PRECS = os.environ.get('PRECS', 'Green,dual').split(',')
state = {}
phase = disc.get_scalar_field(name='phase')
phase_q = disc.get_quad_field_scalar(name='phase_q')
A = disc.get_material_data_size_field_mugrid(name='A')
A_dual = disc.get_material_data_size_field_mugrid(name='A_dual')
macro = disc.get_gradient_size_field(name='macro')
rhs = disc.get_unknown_size_field(name='rhs')
u = disc.get_unknown_size_field(name='u')
t1 = disc.get_unknown_size_field(name='t1')
t2 = disc.get_unknown_size_field(name='t2')
t3 = disc.get_unknown_size_field(name='t3')


def K_fun(x, Ax):
    disc.apply_system_matrix_mugrid(material_data_field=A, input_field_inxyz=x, output_field_inxyz=Ax,
                                    formulation='small_strain')
    disc.fft.communicate_ghosts(Ax)


def green(x, Px):
    disc.fft.communicate_ghosts(x)
    disc.apply_preconditioner_mugrid(preconditioner_Fourier_fnfnqks=green_C0, input_nodal_field_fnxyz=x,
                                     output_nodal_field_fnxyz=Px)


def green_jacobi(x, Px):
    t1.s[...] = state['jacobi'].s * x.s
    disc.fft.communicate_ghosts(t1)
    disc.apply_preconditioner_mugrid(preconditioner_Fourier_fnfnqks=green_2muI, input_nodal_field_fnxyz=t1,
                                     output_nodal_field_fnxyz=Px)
    Px.s[...] *= state['jacobi'].s


def dual(x, Px):
    green(x, t1)
    disc.apply_system_matrix_mugrid(material_data_field=A_dual, input_field_inxyz=t1, output_field_inxyz=t2,
                                    formulation='small_strain')
    disc.fft.communicate_ghosts(t2)
    green(t2, Px)


def dual_plus_gj(x, Px):
    dual(x, t3)
    green_jacobi(x, Px)
    Px.s[...] += t3.s


def periodic_dist(a, b):
    d = np.abs(a - b)
    return np.minimum(d, 1 - d)


# nodal coordinates x_i = i / N (the phase field is nodal; the stiffness is interpolated to the quadrature points)
X, Y = np.meshgrid(np.arange(N) / N, np.arange(N) / N, indexing='ij')
dx = periodic_dist(X, 0.5)
dy = periodic_dist(Y, 0.5)
seg = np.where(np.abs(Y - 0.5) <= 0.25, dx, np.hypot(dx, np.abs(Y - 0.5) - 0.25))
GEOM = {'band': dx, 'band2ph': dx, 'finite': seg, 'cross': np.minimum(dx, dy),
        'disc': np.maximum(np.hypot(dx, dy) - 0.25, 0.0), 'smooth': 'smooth', 'wide_disc': 'wide_disc'}
if os.environ.get('GEOMS'):
    GEOM = {k: v for k, v in GEOM.items() if k in os.environ['GEOMS'].split(',')}

rows = []
for g, dist in GEOM.items():
    if isinstance(dist, str) and dist == 'smooth':              # rho field, SIMP p = 2 (as the TO)
        phase.s[0, 0] = 0.5 + 0.5 * np.cos(2 * np.pi * X) * np.cos(2 * np.pi * Y)
        disc.apply_N_operator_mugrid(phase, phase_q)
        s = phase_q.s[0, 0] ** 2 * (1 - K_RES) + K_RES
    else:
        width = 4 * L if (isinstance(dist, str) and dist == 'wide_disc') else L
        dd = np.maximum(np.hypot(dx, dy) - 0.25, 0.0) if isinstance(dist, str) else dist
        phase.s[0, 0] = np.exp(-dd / width)
        disc.apply_N_operator_mugrid(phase, phase_q)
        dq = phase_q.s[0, 0]
        s = np.where(dq > 0.5, K_RES, 1.0) if g == 'band2ph' else (1 - dq) ** 2 * (1 - K_RES) + K_RES
    A.s[...] = C_0[..., None, None, None] * s
    A_dual.s[...] = C_0[..., None, None, None] / s
    state['jacobi'] = disc.get_preconditioner_Jacobi_mugrid(material_data_field_ijklqxyz=A)
    for lname, E in LOADS.items():
        disc.get_macro_gradient_field_mugrid(macro_gradient_ij=E, macro_gradient_field_ijqxyz=macro)
        rhs.s.fill(0)
        disc.get_rhs_mugrid(material_data_field_ijklqxyz=A, macro_gradient_field_ijqxyz=macro, rhs_inxyz=rhs)
        res = {}
        for P, M in [(p, {'Green': green, 'dual': dual, 'Green_Jacobi': green_jacobi, 'dual+GJ': dual_plus_gj}[p]) for p in PRECS]:
            u.s.fill(0)
            norms = solvers.conjugate_gradients_mugrid_experimental(
                comm=disc.communicator, fc=disc.field_collection, hessp=K_fun, b=rhs, x=u, P=M, tol=1e-6,
                maxiter=int(os.environ.get('MAXIT', 2000)), stop_crit_norm='energy_lower_estim', energy_rtol=True,
                d_min=DMIN)[1]
            it = norms['stop_iteration']['energy_lower_estim']
            res[P] = int(it) if it is not None else -len(norms.get('residual_rr', norms.get('residual_rz', [])))
        rows.append([g, lname] + [res[p] for p in PRECS])
        print(f'{g:8s} {lname:15s} ' + '  '.join(f'{p} {res[p]:5d}' for p in PRECS), flush=True)
tag = ('' if N == 256 else f'_N{N}') + ('' if DMIN == 10 else f'_dmin{DMIN}') + ('' if PRECS == ['Green', 'dual'] else '_' + '_'.join(PRECS))
with open(os.path.join(HERE, f'crack_orientation_results{tag}.csv'), 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['geometry', 'load'] + [f'cg_{p}' for p in PRECS])
    w.writerows(rows)
