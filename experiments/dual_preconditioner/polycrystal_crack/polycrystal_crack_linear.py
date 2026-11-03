"""
Stage 1: linear elastic polycrystal with an edge crack (Nkoumbou Kaptchouang & Gelebart 2022, Sec. 5.3), one solve
of the periodic small-strain problem under macroscopic uniaxial strain E_xx, with the elastic preconditioners of the
AT2 study (damage_AT2_small_strain/.../example_2D_phase_field_fracture_AT2.py, same operators and CG settings):

    Green          Green operator of C_ref = C_iso                                   (one FFT pair per iteration)
    Green_Jacobi   D^-1/2 G(2 mu I_sym) D^-1/2, D = diag K                           (one FFT pair per iteration)
    dual           G K(C_ref C^-1 C_ref) G                                           (two FFT pairs + one K)

All materials are s(x) C_iso (isotropic, one Poisson ratio), so C is a domain.ScaledMaterial with the stiffness
factor s = E(x) / E_MIN; the crack and the void layer have s = VOID_STIFFNESS (the dual needs C^-1, so not 0).
CG: energy-error estimate <= 1e-6 x energy of the solution, cold start, adaptive delay d_min = 10.

usage:  NPIX=64 VOID_STIFFNESS=1e-4 python polycrystal_crack_linear.py dual
writes  exp_data/poly_N<N>_<precond>_k<k>.npz   (iterations, time, sigma_xx, energy-error history; fields at N <= 256)
"""
import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
import numpy as np
from muFFTTO import domain
from muFFTTO import material_models
from muFFTTO import solvers
from muFFTTO import topology_optimization   # energy_stop_kwargs

import polycrystal_geometry as geo

HERE = os.path.dirname(os.path.realpath(__file__))
preconditioner_type = sys.argv[1] if len(sys.argv) > 1 else 'dual'
assert preconditioner_type in ('Green', 'Green_Jacobi', 'dual')
N = int(os.environ.get('NPIX', 64))
void_stiffness = float(os.environ.get('VOID_STIFFNESS', 1e-4))
load = float(os.environ.get('LOAD', 1e-3))          # E_xx
cg_tol, cg_delay = 1e-6, 10

cell = domain.PeriodicUnitCell(domain_size=[1, 1], problem_type='elasticity')
disc = domain.Discretization(cell=cell, nb_of_pixels_global=(N, N), discretization_type='finite_element',
                             element_type='bilinear_rectangle')
assert disc.communicator.size == 1, 'serial only'

# material: C(x) = s(x) C_iso, C_iso of E_MIN, POISSON
K0, G0 = material_models.get_bulk_and_shear_modulus(E=geo.E_MIN, poisson=geo.POISSON)
lam, mu = material_models.get_lame_parameters_from_bulk_and_shear(K0, G0, dim=2)
I2 = np.eye(2)
C_iso = (lam * np.einsum('ij,kl->ijkl', I2, I2)
         + mu * (np.einsum('ik,jl->ijkl', I2, I2) + np.einsum('il,jk->ijkl', I2, I2)))
I_sym = 0.5 * (np.einsum('ik,jl->ijkl', I2, I2) + np.einsum('il,jk->ijkl', I2, I2))
scale = disc.get_quad_field_scalar(name='stiffness_scale')
s_pixel = geo.young_modulus(N, void_stiffness) / geo.E_MIN
scale.s[0, 0, ...] = s_pixel                    # broadcast over the quadrature points
C = domain.ScaledMaterial([(scale, C_iso)])

macro_strain = disc.get_gradient_size_field(name='macro_strain')
disc.get_macro_gradient_field_mugrid(macro_gradient_ij=np.array([[load, 0.0], [0.0, 0.0]]),
                                     macro_gradient_field_ijqxyz=macro_strain)
rhs = disc.get_unknown_size_field(name='rhs')
u = disc.get_unknown_size_field(name='u_fluctuation')


def K(x, Ax):
    disc.apply_system_matrix_mugrid(material_data_field=C, input_field_inxyz=x, output_field_inxyz=Ax,
                                    formulation='small_strain')


t_setup = time.time()
green = disc.get_preconditioner_Green_mugrid(
    reference_material_data_ijkl=2.0 * mu * I_sym if preconditioner_type == 'Green_Jacobi' else C_iso,
    formulation='small_strain')
if preconditioner_type == 'Green_Jacobi':
    jacobi = disc.get_preconditioner_Jacobi_mugrid(material_data_field_ijklqxyz=C, formulation='small_strain',
                                                   name='jacobi')
    x_scaled = disc.get_unknown_size_field(name='x_scaled')
if preconditioner_type == 'dual':
    C_dual = domain.ScaledMaterial([(lambda: 1.0 / scale.s[0, 0], C_iso)])     # C_iso (s C_iso)^-1 C_iso
    tmp1 = disc.get_unknown_size_field(name='dual_tmp1')
    tmp2 = disc.get_unknown_size_field(name='dual_tmp2')
t_setup = time.time() - t_setup


def M(x, Px):
    if preconditioner_type == 'dual':
        disc.fft.communicate_ghosts(x)
        disc.apply_preconditioner_mugrid(preconditioner_Fourier_fnfnqks=green, input_nodal_field_fnxyz=x,
                                         output_nodal_field_fnxyz=tmp1)
        disc.apply_system_matrix_mugrid(material_data_field=C_dual, input_field_inxyz=tmp1,
                                        output_field_inxyz=tmp2, formulation='small_strain')
        disc.fft.communicate_ghosts(tmp2)
        disc.apply_preconditioner_mugrid(preconditioner_Fourier_fnfnqks=green, input_nodal_field_fnxyz=tmp2,
                                         output_nodal_field_fnxyz=Px)
    elif preconditioner_type == 'Green_Jacobi':
        x_scaled.s[...] = jacobi.s * x.s
        disc.apply_preconditioner_mugrid(preconditioner_Fourier_fnfnqks=green, input_nodal_field_fnxyz=x_scaled,
                                         output_nodal_field_fnxyz=Px)
        Px.s[...] *= jacobi.s
    else:
        disc.apply_preconditioner_mugrid(preconditioner_Fourier_fnfnqks=green, input_nodal_field_fnxyz=x,
                                         output_nodal_field_fnxyz=Px)


disc.get_rhs_mugrid(material_data_field_ijklqxyz=C, macro_gradient_field_ijqxyz=macro_strain, rhs_inxyz=rhs)
u.s[...] = 0.0
t0 = time.time()
x_out, norms = solvers.conjugate_gradients_mugrid_experimental(
    comm=disc.communicator, fc=disc.field_collection, hessp=K, b=rhs, x=u, P=M, maxiter=100000,
    stop_crit_norm='energy_lower_estim', d_min=cg_delay,
    **topology_optimization.energy_stop_kwargs(disc, K, u, cg_tol))[:2]
t_cg = time.time() - t0
its = int(norms['stop_iteration']['energy_lower_estim'])

sigma = disc.get_homogenized_stress_mugrid(material_data_field_ijklqxyz=C, displacement_field_inxyz=u,
                                           macro_gradient_field_ijqxyz=macro_strain, formulation='small_strain')
out = dict(N=N, precond=preconditioner_type, void_stiffness=void_stiffness, load=load, cg_its=its,
           time_cg=t_cg, time_setup=t_setup, sigma=np.asarray(sigma),
           energy_estimate=np.asarray(norms.get('energy_lower_estim', [])))
if N <= 256:   # fields for plots: stiffness factor and the von Mises stress per pixel (mean over quad points)
    strain = disc.get_gradient_size_field(name='strain')
    disc.apply_gradient_operator_symmetrized_mugrid(u_inxyz=u, grad_u_ijqxyz=strain)
    eps = strain.s + macro_strain.s                                        # [i, j, q, x, y]
    tr = eps[0, 0] + eps[1, 1]
    s_q = scale.s[0, 0]
    sig = s_q * (lam * tr * I2.reshape(2, 2, 1, 1, 1) + 2 * mu * eps)      # plane, 2D Lame of C_iso
    vm = np.sqrt(sig[0, 0] ** 2 + sig[1, 1] ** 2 - sig[0, 0] * sig[1, 1] + 3 * sig[0, 1] ** 2)
    out.update(stiffness=s_pixel, von_mises=vm.mean(axis=0))
os.makedirs(os.path.join(HERE, 'exp_data'), exist_ok=True)
path = os.path.join(HERE, 'exp_data', f'poly_N{N}_{preconditioner_type}_k{void_stiffness:g}.npz')
np.savez(path, **out)
print(f'N={N:4d} {preconditioner_type:13s} k={void_stiffness:7.0e}  CG its {its:6d}  '
      f'CG {t_cg:7.2f} s  sigma_xx {float(sigma[0, 0]):.6e}  -> {os.path.basename(path)}')
