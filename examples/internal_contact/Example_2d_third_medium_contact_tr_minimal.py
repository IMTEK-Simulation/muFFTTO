"""
2D third-medium contact, neo-Hookean finite strain, solved as an ENERGY
MINIMIZATION with NuMPI's trust-region Newton-CG.

    Pi(u) = int_Omega W(F) dx
          + k_r/2 int_Omega ( Hu : Hu - (1/dim) Lu . Lu ) dx

    grad Pi = G^T P   + R u
    hess Pi = G^T C G + R        with  R = k_r (H^T H - L^T L/dim)

The inner Steihaug CG is preconditioned with the Green operator
M = G^T C_ref G + R, built once from a fixed reference material.

The load is applied in equal increments of lam from 0 to 1, and each
increment is one plain call to `tr_newton_bounded`.  No trust-region radius
tuning, no step-length control, no cutbacks, no imperfection seeding, no
diagnostics.
"""

import os
import sys
import inspect

import numpy as np
from mpi4py import MPI
from matplotlib import pyplot as plt

from NuMPI.Optimization import tr_newton_bounded

# `precond` exists only in a hand-patched BoundedTRNewtonCG.py; NuMPI reports
# version 0.15.1 either way, so fail here instead of with a bare TypeError
# several hundred lines down.
if 'precond' not in inspect.signature(tr_newton_bounded).parameters:
    raise RuntimeError(
        'Installed NuMPI has no `precond` support; this script needs the '
        'patched BoundedTRNewtonCG.py. Stock NuMPI 0.15.1 will not work.')

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from muFFTTO import domain, tensor_operations
from muFFTTO import microstructure_library
from muFFTTO import material_models
from muFFTTO import visualization_utils

# ============================================================================
# problem setup
# ============================================================================
nnn = 32
ninc = 100

# plotting: deformed mesh every `plot_every` increments (0 = only the final
# response curves).  Serial runs only -- the fields are distributed under MPI.
plot_every = 10

number_of_pixels = (nnn, nnn)
domain_size = [1, 1]
dim = len(domain_size)
problem_type = 'elasticity'
discretization_type = 'finite_element'
element_type = 'bilinear_rectangle'
formulation = 'finite_strain'

my_cell = domain.PeriodicUnitCell(domain_size=domain_size,
                                  problem_type=problem_type)

discretization = domain.Discretization(cell=my_cell,
                                       nb_of_pixels_global=number_of_pixels,
                                       discretization_type=discretization_type,
                                       element_type=element_type)

comm = MPI.COMM_WORLD
rank = discretization.communicator.rank


def root_print(*a, **kw):
    if rank == 0:
        print(*a, **kw)


# ============================================================================
# material parameters
# ============================================================================
# matrix: soft neo-Hookean
E_matrix = 100.0
nu_matrix = 0.3
lam_matrix = E_matrix * nu_matrix / ((1 + nu_matrix) * (1 - 2 * nu_matrix))
mu_matrix = E_matrix / (2 * (1 + nu_matrix))
K, G = material_models.get_bulk_and_shear_modulus(E_matrix, nu_matrix)

# third medium ("void"): same neo-Hookean, k_v times softer
k_v = 1e-5
E_void = k_v * E_matrix
nu_void = nu_matrix
lam_void = E_void * nu_void / ((1 + nu_void) * (1 - 2 * nu_void))
mu_void = E_void / (2 * (1 + nu_void))

# HuHu-LuLu regularization
alpha = 1e-6
k_r = alpha * domain_size[0] ** 2 * (K + G * 4 / 3)
inv_tr_I = 1.0 / dim

# reference material for the Green preconditioner
_i = np.eye(dim)
II = np.einsum('ij,kl->ijkl', _i, _i)
I4rt = np.einsum('ik,jl->ijkl', _i, _i)
I4s = 0.5 * (I4rt + np.einsum('il,jk->ijkl', _i, _i))
ref_mat = lam_matrix * II + 2.0 * mu_matrix * I4s

# ============================================================================
# geometry
# ============================================================================
geometry_name = 'contact_test_geometry_2'

phase_field = discretization.get_scalar_field(name='phase_field')
phase_field.s[0, 0] = microstructure_library.get_geometry(
    nb_voxels=discretization.nb_of_pixels,
    microstructure_name=geometry_name,
    coordinates=discretization.fft.coords
)

matrix_mask = phase_field.s[0, 0] > 0
inc_mask = phase_field.s[0, 0] == 0

# ============================================================================
# material fields
# ============================================================================
lam_field = discretization.get_quad_field_scalar(name='lam_field')
mu_field = discretization.get_quad_field_scalar(name='mu_field')

lam_field.s[0, 0, :, matrix_mask] = lam_matrix
lam_field.s[0, 0, :, inc_mask] = lam_void
mu_field.s[0, 0, :, matrix_mask] = mu_matrix
mu_field.s[0, 0, :, inc_mask] = mu_void

material = material_models.NeoHookean(
    discretization=discretization,
    lam_1qxyz=lam_field,
    mu_1qxyz=mu_field,
    name='neo_hookean_two_phase'
)

# ============================================================================
# fields
#
# CONVENTION: macro_gradient_full_field holds H_macro WITHOUT the identity;
# `set_F` adds I.
# ============================================================================
macro_gradient_full_field = discretization.get_gradient_size_field(
    name='macro_gradient_full_field')
grad_u_field = discretization.get_displacement_gradient_sized_field(name='grad_u')
total_strain_field = discretization.get_displacement_gradient_sized_field(name='F')
stress_field = discretization.get_displacement_gradient_sized_field(name='P')
tangent_field = discretization.get_material_data_size_field_mugrid(name='C')
energy_field = discretization.get_quad_field_scalar(name='energy_density')
J_1qxyz = discretization.get_quad_field_scalar(name='J')

x_field = discretization.get_unknown_size_field(name='x_field')
grad_field = discretization.get_unknown_size_field(name='grad_field')
Ru_field = discretization.get_unknown_size_field(name='Ru_field')

v_field = discretization.get_unknown_size_field(name='v_field')
Bv_field = discretization.get_unknown_size_field(name='Bv_field')

# preconditioner scratch, kept separate from the hessp path
r_field = discretization.get_unknown_size_field(name='r_field')
Pv_field = discretization.get_unknown_size_field(name='Pv_field')

hess_u_ijkqxyz = discretization.get_displacement_hessian_size_field(name='hess_u')
HtH_field = discretization.get_unknown_size_field(name='HtH')
lap_u_inxyz = discretization.get_displacement_laplacian_at_quad_field(name='lap_u')
LtL_field = discretization.get_displacement_sized_field(name='LtL')

displacement_fluctuation_field = discretization.get_unknown_size_field(
    name='u_fluc')

u_shape = np.asarray(displacement_fluctuation_field.s).shape
_qw = np.asarray(discretization.quadrature_weights).reshape((-1,) + (1,) * dim)


# ============================================================================
# reductions
# ============================================================================
def global_sum(a):
    return comm.allreduce(float(a), op=MPI.SUM)


def global_min(a):
    return comm.allreduce(float(a), op=MPI.MIN)


def global_max(a):
    return comm.allreduce(float(a), op=MPI.MAX)


def dot_global(a, b):
    return global_sum(np.dot(np.asarray(a).ravel(), np.asarray(b).ravel()))


# ============================================================================
# deformation gradient
# ============================================================================
def min_J(F_in):
    tensor_operations.det2(F_in, J_1qxyz)
    return global_min(np.min(J_1qxyz.s))


def set_F(lam, u_field, out):
    """F = I + lam * H_macro + grad(u~), rebuilt from scratch."""
    discretization.fft.communicate_ghosts(u_field)
    discretization.apply_gradient_operator_mugrid(u_inxyz=u_field,
                                                  grad_u_ijqxyz=grad_u_field)
    out.s[...] = grad_u_field.s + lam * macro_gradient_full_field.s
    for d in range(dim):
        out.s[d, d] += 1.0


# ============================================================================
# regularization operator  R = k_r (H^T H - L^T L / dim)
# ============================================================================
def add_regularization(x, out, scale=1.0):
    """out += scale * R x.  Ghosts of x must already be current."""
    discretization.apply_hessian_operator_to_vector_field_mugrid(
        u_inxyz=x, hess_u_ijkqxyz=hess_u_ijkqxyz)
    discretization.apply_hessian_operator_transposed_to_vector_field_mugrid(
        hess_u_ijkqxyz=hess_u_ijkqxyz, nodal_field_inxyz=HtH_field,
        apply_weights=True)

    discretization.laplacian.apply(nodal_field=x,
                                   quadrature_point_field=lap_u_inxyz)
    discretization.laplacian.transpose(quadrature_point_field=lap_u_inxyz,
                                       nodal_field=LtL_field,
                                       weights=discretization.quadrature_weights)

    out.s[...] += scale * k_r * (HtH_field.s - inv_tr_I * LtL_field.s)


# ============================================================================
# Green preconditioner:  M = G^T C_ref G + R,  applied as M^-1
#
# Built ONCE from the fixed reference material.  With a preconditioner the
# trust region is measured in the M-norm, so M must not change mid-run or the
# meaning of the radius changes with it.
# ============================================================================
def operator_for_preconditioner(input_field_inxyz, output_field_inxyz):
    """The operator whose Fourier symbol the Green blocks invert."""
    discretization.fft.communicate_ghosts(field=input_field_inxyz)
    discretization.apply_system_matrix_mugrid(
        material_data_field=ref_mat,
        input_field_inxyz=input_field_inxyz,
        output_field_inxyz=output_field_inxyz,
        formulation=formulation)
    add_regularization(input_field_inxyz, output_field_inxyz)


preconditioner = discretization.get_preconditioner_Green_mugrid(
    reference_material_data_ijkl=ref_mat,
    operator=operator_for_preconditioner,
)


def precond(v):
    """Return M^-1 v."""
    r_field.s[...] = v
    discretization.fft.communicate_ghosts(r_field)
    discretization.apply_preconditioner_mugrid(
        preconditioner_Fourier_fnfnqks=preconditioner,
        input_nodal_field_fnxyz=r_field,
        output_nodal_field_fnxyz=Pv_field)
    return np.array(Pv_field.s, copy=True)


# ============================================================================
# objective, gradient, Hessian-vector product
#
# The load parameter is read from the module-level `lam_current`, which the
# increment loop sets before each solve.
# ============================================================================
lam_current = 0.0

# Large but FINITE: np.inf would make ared = inf - inf = nan, and a nan rho is
# never rejected *and* never shrinks the radius, so the solver spins to
# maxiter.  A finite penalty gives rho << 0, hence rejection and delta/4.
INADMISSIBLE_ENERGY = 1e30


def _set_F_from(x):
    x_field.s[...] = x
    set_F(lam_current, x_field, total_strain_field)
    return min_J(total_strain_field)


def fun_grad(x):
    """(Pi, grad Pi), globally reduced."""
    J_min = _set_F_from(x)
    if not np.isfinite(J_min) or J_min <= 0.0:
        return INADMISSIBLE_ENERGY, np.zeros(u_shape)

    material.get_energy_density(total_strain_field, energy_field)
    Pi = global_sum(np.sum(_qw * energy_field.s[0, 0]))
    if not np.isfinite(Pi):
        return INADMISSIBLE_ENERGY, np.zeros(u_shape)

    # regularization energy 1/2 <u, R u>, reusing R u for the gradient
    Ru_field.s[...] = 0.0
    add_regularization(x_field, Ru_field)
    Pi += 0.5 * dot_global(x_field.s, Ru_field.s)

    material.get_stress(total_strain_field, stress_field)
    discretization.fft.communicate_ghosts(stress_field)
    discretization.apply_gradient_transposed_operator_mugrid(
        gradient_field_ijqxyz=stress_field, div_u_fnxyz=grad_field,
        apply_weights=True)
    grad_field.s[...] += Ru_field.s

    return Pi, np.array(grad_field.s, copy=True)


def hessp(x, v):
    """B v = (G^T C G + R) v, with C evaluated at x."""
    _set_F_from(x)
    material.get_algorithmic_tangent(total_strain_field, tangent_field)

    v_field.s[...] = v
    discretization.fft.communicate_ghosts(v_field)
    discretization.apply_system_matrix_mugrid(
        material_data_field=tangent_field,
        input_field_inxyz=v_field,
        output_field_inxyz=Bv_field,
        formulation=formulation)
    add_regularization(v_field, Bv_field)
    discretization.fft.communicate_ghosts(Bv_field)

    return np.array(Bv_field.s, copy=True)


# ============================================================================
# macroscopic loading:  F_bar = I + lam * H_macro,  lam: 0 -> 1
# ============================================================================
H_macro = np.zeros((dim, dim))
H_macro[1, 0] = -0.6

discretization.get_macro_gradient_field_mugrid(
    macro_gradient_ij=H_macro,
    macro_gradient_field_ijqxyz=macro_gradient_full_field
)

# ============================================================================
# incremental loading
# ============================================================================
u = np.zeros(u_shape)

hist_lam = []
hist_F10 = []
hist_P10 = []
hist_Pxx = []
hist_minJ = []
hist_energy = []

serial = discretization.communicator.size == 1

root_print('=' * 70)
root_print(f'geometry     : {geometry_name}')
root_print(f'element_type : {element_type}')
root_print(f'pixels       : {number_of_pixels}')
root_print(f'increments   : {ninc}')

for inc in range(1, ninc + 1):
    lam_current = inc / float(ninc)

    res = tr_newton_bounded(
        fun=fun_grad,
        x0=u,
        hessp=hessp,
        precond=precond,
        jac=True,
        gtol=1e-4,
        maxiter=500,
        inner_tol=1e-6,
        inner_maxiter=None,
        comm=comm,
    )

    u = np.asarray(res.x)

    J_min = _set_F_from(u)
    material.get_stress(total_strain_field, stress_field)

    root_print('-' * 70)
    root_print(f'increment {inc:4d}   lam = {lam_current:.4f}   '
               f'lam*H_10 = {lam_current * H_macro[1, 0]:+.5f}')
    root_print(f'  {res.message}')
    root_print(f'  outer its {res.nit:4d} | hessp {res.nb_hessp:6d} '
               f'| |grad|_inf {res.max_grad:10.3e} | Pi {res.fun:12.6e}')
    root_print(f'  min(det F) {J_min:10.3e} | '
               f'P_xx {stress_field.s[0, 0].mean():10.3e} | '
               f'P_yx {stress_field.s[1, 0].mean():10.3e}')

    hist_lam.append(lam_current)
    hist_F10.append(lam_current * H_macro[1, 0])
    hist_P10.append(float(stress_field.s[1, 0].mean()))
    hist_Pxx.append(float(stress_field.s[0, 0].mean()))
    hist_minJ.append(J_min)
    hist_energy.append(float(res.fun))

    # ---- deformed mesh -----------------------------------------------------
    if serial and plot_every and inc % plot_every == 0:
        displacement_fluctuation_field.s[...] = u
        x_plot_ixyz = visualization_utils.get_deformed_grid_coords_two_dim(
            discretization,
            macro_gradient_ij=lam_current * H_macro,
            displacement_fluctuation=displacement_fluctuation_field)
        visualization_utils.plot_field_on_grid(
            coordinates_for_plot=x_plot_ixyz,
            field_to_plot=phase_field.s[0, 0],
            name=f'phase, increment {inc}   lam = {lam_current:.4f}')

root_print('=' * 70)
root_print(f'final min J : {min_J(total_strain_field):.6e}')

# ============================================================================
# response curves
# ============================================================================
if rank == 0 and len(hist_lam) > 1:
    fig, ax = plt.subplots(1, 4, figsize=(19, 4))

    ax[0].plot(hist_F10, hist_P10, '-o', ms=3, color='k')
    ax[0].set_xlabel(r'$\bar{F}_{10}$')
    ax[0].set_ylabel(r'$\bar{P}_{10}$')

    ax[1].plot(hist_F10, hist_Pxx, '-o', ms=3, color='C0')
    ax[1].set_xlabel(r'$\bar{F}_{10}$')
    ax[1].set_ylabel(r'$\bar{P}_{xx}$')

    ax[2].semilogy(hist_lam, hist_minJ, '-o', ms=3, color='C3')
    ax[2].set_xlabel(r'$\lambda$')
    ax[2].set_ylabel(r'$\min \det F$')

    ax[3].plot(hist_lam, hist_energy, '-o', ms=3, color='C2')
    ax[3].set_xlabel(r'$\lambda$')
    ax[3].set_ylabel(r'$\Pi$')

    for a in ax:
        a.grid(alpha=.3)

    fig.tight_layout()
    plt.show()