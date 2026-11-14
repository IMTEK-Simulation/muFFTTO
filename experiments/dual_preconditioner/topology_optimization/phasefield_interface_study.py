"""
Phase-field interface on a fixed TO geometry: the final 1024^2 TO design (NEMO eta / contrast study, l = 2h, contrast
1e4, data/example_2D_elasticity_TO/nemo_eta2h/) is thresholded at 0.5 (sharp two-phase geometry) and then given the
1D equilibrium profile of the phase-field (double-well) energy with width eta:
    phi(d) = 1/2 [1 + tanh(d / (2 eta))],   d = periodic signed distance to the thresholded boundary [pixels]
    s(x)   = phi_q^p (1 - k) + k            (SIMP p = 2 as in the TO, phi at the quadrature points),   A = s C_0
eta = 0: the two-phase material s = 1 if phi_q > 0.5 else k (as in threshold_designs_study.py).
Solved from zero for the uniaxial macro strain E_xx = 1 (the slide-13 video load), CG energy-error stop (relative 1e-6,
delay d_min 10) as in all runs. Preconditioners: PRECS env (default Green,Green_Jacobi,dual), raw CG counts (dual not x2).
    python phasefield_interface_study.py [eta/h ...]   ->  exp_data/phasefield_interface_N1024.csv (appended)
Env: DEVICE (cpu | gpu: muGrid with CUDA, field data are cupy arrays), N (default 1024; other grids upsample the 128^2-family design is NOT done, the design file must exist), SIMP_P.
"""
import csv
import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

import netCDF4
import numpy as np
from scipy import ndimage

from muFFTTO import domain, material_models, solvers

device = os.environ.get('DEVICE', 'cpu')
if device == 'gpu':
    import cupy as xp
else:
    xp = np

HERE = os.path.dirname(os.path.realpath(__file__))
N = int(os.environ.get('N', 1024))
DESIGN = {1024: 'N1024_soft0.0001_energy-warm_dual_eta_0.001953125_w_5.0_l_bfgs_final.nc',
          256: 'N256_soft0.0001_energy-warm_Green_eta_0.0078125_w_5.0_l_bfgs_final.nc',
          128: 'N128_soft0.0001_energy-warm_Green_eta_0.015625_w_5.0_l_bfgs_final.nc'}[N]
DESIGN = os.path.join(HERE, 'data', 'example_2D_elasticity_TO', 'nemo_eta2h', DESIGN)
ETAS = [float(a) for a in sys.argv[1:]] or [0, 0.25, 0.5, 1, 1.5, 2, 3, 4]
PRECS = os.environ.get('PRECS', 'Green,Green_Jacobi,dual').split(',')
p = int(os.environ.get('SIMP_P', 2))
k = 1e-4
OUT = os.path.join(HERE, 'exp_data', os.environ.get('OUT_CSV', f'phasefield_interface_N{N}.csv'))
os.makedirs(os.path.dirname(OUT), exist_ok=True)
K_0, G_0, dim = 1.0, 0.5, 2
C_0 = material_models.get_elastic_material_tensor(dim=dim, K=K_0, mu=G_0, kind='linear')
E_xx = np.array([[1.0, 0.0], [0.0, 0.0]])

# sharp geometry and its periodic signed distance on the nodes (boundary half-way between nodes)
design = np.asarray(netCDF4.Dataset(DESIGN)['phase_field_solution'][:]).squeeze()
assert design.shape == (N, N), design.shape
solid = design > 0.5
big = np.tile(solid, (3, 3))
d_in = ndimage.distance_transform_edt(big)[N:2 * N, N:2 * N]
d_out = ndimage.distance_transform_edt(~big)[N:2 * N, N:2 * N]
signed = np.where(solid, d_in - 0.5, -(d_out - 0.5))


def phi_of(eta):
    return solid.astype(float) if eta == 0 else 0.5 * (1 + np.tanh(signed / (2 * eta)))


if __name__ == '__main__':
    cell = domain.PeriodicUnitCell(domain_size=[1, 1], problem_type='elasticity')
    disc = domain.Discretization(cell=cell, nb_of_pixels_global=(N, N), discretization_type='finite_element',
                                 element_type='linear_triangles', device=device)
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

    C_0_dev = xp.asarray(C_0)
    M = {'Green': green(green_C0), 'Green_Jacobi': green_jacobi, 'dual': dual}
    new = not os.path.exists(OUT)
    fout = open(OUT, 'a', newline='')
    w = csv.writer(fout)
    if new:
        w.writerow(['N', 'eta_over_h', 'simp_p', 'precond', 'cg', 'min_s', 'time_s'])
    disc.get_macro_gradient_field_mugrid(macro_gradient_ij=E_xx, macro_gradient_field_ijqxyz=macro)
    for eta in ETAS:
        phase.s[0, 0] = xp.asarray(phi_of(eta))
        disc.apply_N_operator_mugrid(phase, phase_q)
        rq = phase_q.s[0, 0]
        s = xp.where(rq > 0.5, 1.0, k) if eta == 0 else xp.power(rq, p) * (1 - k) + k
        smin = float(s.min())
        A.s[...] = C_0_dev[..., None, None, None] * s
        A_dual.s[...] = C_0_dev[..., None, None, None] / s
        state['jacobi'] = disc.get_preconditioner_Jacobi_mugrid(material_data_field_ijklqxyz=A)
        rhs.s.fill(0)
        disc.get_rhs_mugrid(material_data_field_ijklqxyz=A, macro_gradient_field_ijqxyz=macro, rhs_inxyz=rhs)
        for P in PRECS:
            u.s.fill(0)
            t0 = time.time()
            norms = solvers.conjugate_gradients_mugrid_experimental(
                comm=disc.communicator, fc=disc.field_collection, hessp=K_fun, b=rhs, x=u, P=M[P], tol=1e-6,
                maxiter=50000, stop_crit_norm='energy_lower_estim', energy_rtol=True, d_min=10)[1]
            it = int(norms['stop_iteration']['energy_lower_estim'])
            dt = time.time() - t0
            w.writerow([N, eta, p, P, it, f'{smin:.3e}', f'{dt:.1f}'])
            fout.flush()
            print(f'N {N} eta/h {eta:5.2f} p {p} {P:12s} CG {it:6d}  min s {smin:.2e}  {dt:7.1f} s', flush=True)
    fout.close()
