"""
Diffuse vs two-phase: the final TO designs at l = 2h (Green runs 32^2 ... 512^2, the dual run at 1024^2 where Green was
not run; NEMO eta / contrast study, fetched to data/example_2D_elasticity_TO/nemo_eta2h/), each solved from zero for
the 3 unit load cases twice:
    diffuse     s(x) = rho_q^p (1 - k) + k               (as in the optimisation, rho at the quadrature points)
    two-phase   s(x) = 1 if rho_q > 0.5 else k             (thresholded at the quadrature points: exactly two values)
with A = s C_0, CG energy-error stop (relative 1e-6, delay d_min 10) as in all runs. Preconditioners: PRECS env
(default 'Green'; also 'Green_Jacobi', 'dual'). Output: CG per solve (mean of the 3 load cases).
    PRECS=Green python threshold_designs_study.py [N ...]   ->  exp_data/threshold_designs_l2h.csv (appended)
"""
import csv
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

import netCDF4
import numpy as np

from muFFTTO import domain, material_models, solvers

HERE = os.path.dirname(os.path.realpath(__file__))
D = os.path.join(HERE, 'data', 'example_2D_elasticity_TO', 'nemo_eta2h')
GRIDS = [int(a) for a in sys.argv[1:]] or [32, 64, 128, 256, 512, 1024]
SOFTS = ['0.01', '0.001', '0.0001', '1e-05']
PRECS = os.environ.get('PRECS', 'Green').split(',')
OUT = os.path.join(HERE, 'exp_data', 'threshold_designs_l2h.csv')
os.makedirs(os.path.dirname(OUT), exist_ok=True)
K_0, G_0, p, dim = 1.0, 0.5, 2, 2
C_0 = material_models.get_elastic_material_tensor(dim=dim, K=K_0, mu=G_0, kind='linear')
macro_gradients = [np.array([[1.0, 0.0], [0.0, 0.0]]), np.array([[0.0, 0.0], [0.0, 1.0]]),
                   np.array([[0.0, 0.5], [0.5, 0.0]])]


def design_file(N, soft):
    for prec in ('Green', 'dual'):
        for eta in (f'{2 / N:g}', repr(2 / N)):
            for seg in ('_seg3', '_seg2', ''):
                f = os.path.join(D, f'N{N}_soft{soft}_energy-warm_{prec}_eta_{eta}_w_5.0_l_bfgs{seg}_final.nc')
                if os.path.exists(f):
                    return f, prec
    return None, None


new = not os.path.exists(OUT)
fout = open(OUT, 'a', newline='')
w = csv.writer(fout)
if new:
    w.writerow(['N', 'contrast', 'design_from', 'material', 'precond', 'cg_per_solve', 'cg_1', 'cg_2', 'cg_3'])
for N in GRIDS:
    cell = domain.PeriodicUnitCell(domain_size=[1, 1], problem_type='elasticity')
    disc = domain.Discretization(cell=cell, nb_of_pixels_global=(N, N), discretization_type='finite_element',
                                 element_type='linear_triangles')
    green_C0 = disc.get_preconditioner_Green_mugrid(reference_material_data_ijkl=C_0, name='green_C0')
    green_2muI = disc.get_preconditioner_Green_mugrid(
        reference_material_data_ijkl=2 * G_0 * np.einsum('ik,jl->ijkl', np.eye(dim), np.eye(dim)), name='green_2muI')
    phase = disc.get_scalar_field(name='phase')
    phase_q = disc.get_quad_field_scalar(name='phase_q')
    A = disc.get_material_data_size_field_mugrid(name='A')
    A_dual = disc.get_material_data_size_field_mugrid(name='A_dual')
    macro = disc.get_gradient_size_field(name='macro')
    rhs = disc.get_unknown_size_field(name='rhs')
    u = disc.get_unknown_size_field(name='u')
    t1 = disc.get_unknown_size_field(name='t1')
    t2 = disc.get_unknown_size_field(name='t2')
    state = {}

    def K_fun(x, Ax):
        disc.apply_system_matrix_mugrid(material_data_field=A, input_field_inxyz=x, output_field_inxyz=Ax,
                                        formulation='small_strain')
        disc.fft.communicate_ghosts(Ax)

    def green(G):
        def apply(x, Px):
            disc.fft.communicate_ghosts(x)
            disc.apply_preconditioner_mugrid(preconditioner_Fourier_fnfnqks=G, input_nodal_field_fnxyz=x,
                                             output_nodal_field_fnxyz=Px)
        return apply

    def green_jacobi(x, Px):
        t1.s[...] = state['jacobi'].s * x.s
        green(green_2muI)(t1, Px)
        Px.s[...] *= state['jacobi'].s

    def dual(x, Px):
        green(green_C0)(x, t1)
        disc.apply_system_matrix_mugrid(material_data_field=A_dual, input_field_inxyz=t1, output_field_inxyz=t2,
                                        formulation='small_strain')
        disc.fft.communicate_ghosts(t2)
        green(green_C0)(t2, Px)

    M = {'Green': green(green_C0), 'Green_Jacobi': green_jacobi, 'dual': dual}
    for soft in SOFTS:
        f, src = design_file(N, soft)
        if f is None:
            print(f'N {N} soft {soft}: no design', flush=True)
            continue
        k = float(soft)
        phase.s[0, 0] = np.asarray(netCDF4.Dataset(f)['phase_field_solution'][:]).squeeze()
        disc.apply_N_operator_mugrid(phase, phase_q)
        for mat in ('diffuse', 'two-phase'):
            rq = phase_q.s[0, 0]
            s = (np.power(rq, p) * (1 - k) + k) if mat == 'diffuse' else np.where(rq > 0.5, 1.0, k)
            A.s[...] = C_0[..., None, None, None] * s
            A_dual.s[...] = C_0[..., None, None, None] / s
            state['jacobi'] = disc.get_preconditioner_Jacobi_mugrid(material_data_field_ijklqxyz=A)
            for P in PRECS:
                its = []
                for E in macro_gradients:
                    disc.get_macro_gradient_field_mugrid(macro_gradient_ij=E, macro_gradient_field_ijqxyz=macro)
                    rhs.s.fill(0)
                    disc.get_rhs_mugrid(material_data_field_ijklqxyz=A, macro_gradient_field_ijqxyz=macro,
                                        rhs_inxyz=rhs)
                    u.s.fill(0)
                    norms = solvers.conjugate_gradients_mugrid_experimental(
                        comm=disc.communicator, fc=disc.field_collection, hessp=K_fun, b=rhs, x=u, P=M[P], tol=1e-6,
                        maxiter=50000, stop_crit_norm='energy_lower_estim', energy_rtol=True, d_min=10)[1]
                    its.append(int(norms['stop_iteration']['energy_lower_estim']))
                w.writerow([N, f'{1 / k:g}', src, mat, P, f'{np.mean(its):.1f}', *its])
                fout.flush()
                print(f'N {N:5d} contrast {1 / k:8g} ({src} design) {mat:9s} {P:12s} CG/solve {np.mean(its):7.1f} '
                      f'{its}', flush=True)
fout.close()
