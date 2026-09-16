"""
2D third-medium contact with full HuHu-LuLu regularization, solved with plain
Newton-CG and explicit control over the imposed deformation gradient.

    Pi(u) = int_Omega W(F) dx
          + k_r/2 int_Omega ( Hu : Hu - (1/dim) Lu . Lu ) dx

    grad Pi = G^T P   + R u
    hess Pi = G^T C G + R        with  R = k_r (H^T H - L^T L/dim)

The third medium is k_v = 1e-5 times softer than the matrix, so as the two
surfaces approach, the thin void layer absorbs nearly all the compression and
det F there drives toward zero.  A fixed load increment or a full Newton step
overshoots through zero and NeoHookean then takes log(J) of a negative number.

Two controls prevent that, both built on the same exact closed form
(`max_admissible_scale`):

  * the load increment dlam is capped at LOAD_SAFETY of the distance to
    det F = 0, measured at the *predictor* (u frozen);
  * the Newton step is capped at J_STEP_FRAC of the distance to det F = 0.

F is rebuilt from scratch every evaluation (`set_F`) rather than accumulated,
so the imposed macro gradient is always exactly lam * H_macro and a load step
can be halved or rolled back.

NOTE on the element type: the discrete Laplacian is identically zero for Q1
elements on axis-aligned pixels (see the RuntimeWarning in
discretization_library.py), which would silently reduce HuHu-LuLu to HuHu-only.
The LuLu term therefore requires 'biquadratic_rectangle'.
"""

import time
import os
import sys
import argparse

import numpy as np
from mpi4py import MPI
from NuMPI.IO import save_npy
from matplotlib import pyplot as plt

# Add the project root to sys.path relatively
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from muFFTTO import domain, tensor_operations
from muFFTTO import solvers
from muFFTTO import microstructure_library
from muFFTTO import material_models
from muFFTTO import visualization_utils

# ============================================================================
# argument parsing
# ============================================================================
parser = argparse.ArgumentParser(
    prog='example_2D_third_medium_contact_newton_CG_HuHu_LuLu_stepctrl.py',
    description='Finite strain neo-Hookean TMC in 2D, Newton-CG with '
                'deformation gradient step control'
)
parser.add_argument('-n', '--nb_pixel', default='32')
parser.add_argument('-inc', '--nb_increments', default='100')
parser.add_argument('--save_per_it', action='store_true',
                    help='Enable saving every iteration')
parser.add_argument('--plot_figures', action='store_true',
                    help='Plot per-increment meshes and the final response')

script_name = os.path.splitext(os.path.basename(__file__))[0]
args = parser.parse_args()
nnn = int(args.nb_pixel)
ninc = int(args.nb_increments)
save_per_it = args.save_per_it
plot_figures = args.plot_figures

# ============================================================================
# problem setup
# ============================================================================
number_of_pixels = (nnn, nnn)
domain_size = [1, 1]
dim = len(domain_size)
tol_newton = 1e-4
problem_type = 'elasticity'
discretization_type = 'finite_element'
element_type = 'biquadratic_rectangle'  # required for a nonzero LuLu term
formulation = 'finite_strain'
preconditioner_type = "Green"  # Options: 'Green', 'Green_Jacobi'

# --- deformation gradient step control --------------------------------------
LOAD_SAFETY = 0.5      # fraction of the way to det F = 0 for the load step
J_STEP_FRAC = 0.5      # fraction of the way to det F = 0 for the Newton step
MAX_NEWTON_ITS = 100
MAX_INCREMENTS = 10 * ninc
DLAM_MIN = 1e-9        # below this the load path is considered stalled

_info = {
    'problem_type': problem_type,
    'discretization_type': discretization_type,
    'element_type': element_type,
    'formulation': formulation,
    'preconditioner_type': preconditioner_type,
    'nb_of_pixels': number_of_pixels,
    'domain_size': domain_size,
    'ninc': ninc,
    'LOAD_SAFETY': LOAD_SAFETY,
    'J_STEP_FRAC': J_STEP_FRAC,
}

# ============================================================================
# discretization
# ============================================================================
my_cell = domain.PeriodicUnitCell(domain_size=domain_size,
                                  problem_type=problem_type)

discretization = domain.Discretization(cell=my_cell,
                                       nb_of_pixels_global=number_of_pixels,
                                       discretization_type=discretization_type,
                                       element_type=element_type)

rank = discretization.communicator.rank


def root_print(*a, **kw):
    if rank == 0:
        print(*a, **kw)


# ============================================================================
# output folders
# ============================================================================
file_folder_path = os.path.dirname(os.path.realpath(__file__))
data_folder_path = (file_folder_path + '/exp_data/' + script_name + '/'
                    + f'Nx={nnn}Ny={nnn}_{preconditioner_type}/')
figure_folder_path = (file_folder_path + '/figures/' + script_name + '/'
                      + f'Nx={nnn}Ny={nnn}_{preconditioner_type}/')

if rank == 0:
    os.makedirs(data_folder_path, exist_ok=True)
    os.makedirs(figure_folder_path, exist_ok=True)

# ============================================================================
# material parameters
# ============================================================================
# matrix: soft neo-Hookean
E_matrix = 100.0
nu_matrix = 0.3
lam_matrix = E_matrix * nu_matrix / ((1 + nu_matrix) * (1 - 2 * nu_matrix))
mu_matrix = E_matrix / (2 * (1 + nu_matrix))
K, G = material_models.get_bulk_and_shear_modulus(E_matrix, nu_matrix)

# third medium ("void"): same neo-Hookean as the matrix, k_v times softer
k_v = 1e-5  # TMC contrast, Table 1
E_void = k_v * E_matrix
nu_void = nu_matrix  # keep the solid's Poisson ratio
lam_void = E_void * nu_void / ((1 + nu_void) * (1 - 2 * nu_void))
mu_void = E_void / (2 * (1 + nu_void))

# HuHu-LuLu regularization
alpha = 1e-6  # NOTE: the HuHu coefficient. The step length is `alpha_step`.
k_r = alpha * domain_size[0] ** 2 * (K + G * 4 / 3)
inv_tr_I = 1.0 / dim  # = 1/Tr(I), Eq. (4)

# reference material for Green preconditioner
i = np.eye(dim)
II = np.einsum('ij,kl->ijkl', i, i)
I4rt = np.einsum('ik,jl->ijkl', i, i)
I4s = 0.5 * (I4rt + np.einsum('il,jk->ijkl', i, i))
ref_mat = lam_matrix * II + 2.0 * mu_matrix * I4s

_info['lam_matrix'] = lam_matrix
_info['mu_matrix'] = mu_matrix
_info['lam_inc'] = lam_void
_info['mu_inc'] = mu_void
_info['k_r'] = k_r

# ============================================================================
# geometry
# ============================================================================
phase_field = discretization.get_scalar_field(name='phase_field')
phase_field.s[0, 0] = microstructure_library.get_geometry(
    nb_voxels=discretization.nb_of_pixels,
    microstructure_name='contact_test_geometry_1',
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

root_print(f'lam min/max : {lam_field.s.min():.4e} / {lam_field.s.max():.4e}')
root_print(f'mu  min/max : {mu_field.s.min():.4e} / {mu_field.s.max():.4e}')
root_print(f'k_r (HuHu)  : {k_r:.6e}')

# ============================================================================
# material model
# ============================================================================
material = material_models.NeoHookean(
    discretization=discretization,
    lam_1qxyz=lam_field,
    mu_1qxyz=mu_field,
    name='neo_hookean_two_phase'
)

# ============================================================================
# fields
# ============================================================================
macro_gradient_full_field = discretization.get_gradient_size_field(name='macro_gradient_full_field')
displacement_fluctuation_field = discretization.get_unknown_size_field(name='displacement_fluctuation_field')
displacement_increment_field = discretization.get_unknown_size_field(name='displacement_increment_field')
strain_fluc_field = discretization.get_displacement_gradient_sized_field(name='strain_fluctuation_field')
grad_u_field = discretization.get_displacement_gradient_sized_field(name='grad_u_field')
total_strain_field = discretization.get_displacement_gradient_sized_field(name='total_strain_field')
stress_field = discretization.get_displacement_gradient_sized_field(name='stress_field')
tangent_field = discretization.get_material_data_size_field_mugrid(name='tangent_field')
rhs_field = discretization.get_unknown_size_field(name='rhs_field')
energy_field = discretization.get_quad_field_scalar(name='energy_field')

hess_u_ijkqxyz = discretization.get_displacement_hessian_size_field(name='hess_u')
HtH_field = discretization.get_unknown_size_field(name='HtH')

lap_u_inxyz = discretization.get_displacement_laplacian_at_quad_field(name='lap_u_ijnxyz')
LtL_field = discretization.get_displacement_sized_field(name='LtL_field')

J_1qxyz = discretization.get_quad_field_scalar(name='J')


# ============================================================================
# deformation gradient bookkeeping and admissibility
# ============================================================================
def min_J(F_in):
    """Global min over all quadrature points of det F."""
    tensor_operations.det2(F_in, J_1qxyz)
    return MPI.COMM_WORLD.allreduce(float(np.min(J_1qxyz.s)), op=MPI.MIN)


def set_F(lam, u_field, out):
    """F = I + lam * H_macro + grad(u~).

    Rebuilt from scratch, never accumulated, so the imposed macro gradient is
    always exactly lam * H_macro and a load step can be halved or rolled back.
    """
    discretization.fft.communicate_ghosts(u_field)
    discretization.apply_gradient_operator_mugrid(u_inxyz=u_field,
                                                  grad_u_ijqxyz=grad_u_field)
    out.s[...] = grad_u_field.s + lam * macro_gradient_full_field.s
    for d in range(dim):
        out.s[d, d] += 1.0


def max_admissible_scale(F_in, dF_in):
    """Largest s > 0 keeping det(F + s*dF) > 0 at every quadrature point.

    In 2D this is exact rather than a bound: for a 2x2 F,

        det(F + s dF) = det F + s*b + s^2 * det dF
        b = F00 dF11 + dF00 F11 - F01 dF10 - dF01 F10

    so the first crossing of zero is the smallest positive root of a quadratic
    and needs no bisection.  Returns +inf when the ray never crosses (the step
    is unconstrained) and 0.0 if F is already inadmissible.

    This is the fraction-to-the-boundary rule, and it serves both the load step
    (dF = H_macro, u frozen, so F is affine in the load parameter) and the
    Newton step (dF = grad of the displacement increment).  Unlike a
    determinant-lemma expression specialised to one component, it holds for an
    arbitrary load direction.
    """
    F, D = F_in.s, dF_in.s

    c = F[0, 0] * F[1, 1] - F[0, 1] * F[1, 0]
    b = (F[0, 0] * D[1, 1] + D[0, 0] * F[1, 1]
         - F[0, 1] * D[1, 0] - D[0, 1] * F[1, 0])
    a = D[0, 0] * D[1, 1] - D[0, 1] * D[1, 0]

    if MPI.COMM_WORLD.allreduce(float(np.min(c)), op=MPI.MIN) <= 0.0:
        return 0.0

    with np.errstate(divide='ignore', invalid='ignore'):
        # degenerate (rank-one or uniform) dF: det dF = 0, so det F is linear
        s_linear = np.where(b < 0.0, -c / b, np.inf)

        disc = b * b - 4.0 * a * c
        root = np.sqrt(np.maximum(disc, 0.0))
        r_minus = (-b - root) / (2.0 * a)
        r_plus = (-b + root) / (2.0 * a)
        first_positive = np.minimum(np.where(r_minus > 0.0, r_minus, np.inf),
                                    np.where(r_plus > 0.0, r_plus, np.inf))
        s_quadratic = np.where(disc >= 0.0, first_positive, np.inf)

    s_qxy = np.where(np.abs(a) <= 1e-30 * (np.abs(b) + 1.0),
                     s_linear, s_quadratic)

    return MPI.COMM_WORLD.allreduce(float(np.min(s_qxy)), op=MPI.MIN)


# ============================================================================
# regularization operator  R = k_r (H^T H - L^T L / dim)
# ============================================================================
def add_regularization(x, out, scale=1.0):
    """out += scale * k_r * (HtH - LtL/dim) applied to x. Mirrors K_fun."""
    discretization.apply_hessian_operator_to_vector_field_mugrid(
        u_inxyz=x, hess_u_ijkqxyz=hess_u_ijkqxyz)
    discretization.apply_hessian_operator_transposed_to_vector_field_mugrid(
        hess_u_ijkqxyz=hess_u_ijkqxyz, nodal_field_inxyz=HtH_field, apply_weights=True)

    discretization.laplacian.apply(nodal_field=x, quadrature_point_field=lap_u_inxyz)
    discretization.laplacian.transpose(quadrature_point_field=lap_u_inxyz,
                                       nodal_field=LtL_field,
                                       weights=discretization.quadrature_weights)

    out.s[...] += scale * k_r * (HtH_field.s - inv_tr_I * LtL_field.s)


# ============================================================================
# preconditioner
# ============================================================================
def operator_for_preconditioner(input_field_inxyz, output_field_inxyz):
    """The operator whose Fourier symbol the Green blocks invert."""
    discretization.fft.communicate_ghosts(field=input_field_inxyz)
    discretization.apply_system_matrix_mugrid(
        material_data_field=ref_mat,
        input_field_inxyz=input_field_inxyz,
        output_field_inxyz=output_field_inxyz,
        formulation=formulation
    )
    add_regularization(input_field_inxyz, output_field_inxyz)


preconditioner = discretization.get_preconditioner_Green_mugrid(
    reference_material_data_ijkl=ref_mat,
    operator=operator_for_preconditioner,
)


def M_fun_Green(x, Px):
    discretization.fft.communicate_ghosts(x)
    discretization.apply_preconditioner_mugrid(
        preconditioner_Fourier_fnfnqks=preconditioner,
        input_nodal_field_fnxyz=x,
        output_nodal_field_fnxyz=Px
    )


# ============================================================================
# residual and Hessian
# ============================================================================
def assemble_rhs(F_in, u_field):
    """rhs = -grad Pi = -(G^T P + R u).  Refreshes stress and tangent.

    The regularization is ALWAYS included, so norm(rhs) at the start of an
    increment and inside the Newton loop measure the same functional.
    """
    material.get_stress(F_in, stress_field)
    material.get_algorithmic_tangent(F_in, tangent_field)

    discretization.fft.communicate_ghosts(stress_field)
    discretization.apply_gradient_transposed_operator_mugrid(
        gradient_field_ijqxyz=stress_field, div_u_fnxyz=rhs_field, apply_weights=True)
    rhs_field.s[...] *= -1

    discretization.fft.communicate_ghosts(u_field)
    add_regularization(x=u_field, out=rhs_field, scale=-1.0)

    return np.sqrt(discretization.communicator.sum(
        np.dot(rhs_field.s.ravel(), rhs_field.s.ravel())
    ))


def K_fun(x, Ax):
    discretization.fft.communicate_ghosts(field=x)
    discretization.apply_system_matrix_mugrid(
        material_data_field=tangent_field,
        input_field_inxyz=x,
        output_field_inxyz=Ax,
        formulation=formulation
    )
    add_regularization(x, Ax)
    discretization.fft.communicate_ghosts(Ax)


# ============================================================================
# macroscopic loading
#
# H_macro is the TOTAL target macro gradient (F - I).  The load parameter lam
# runs 0 -> 1, so the imposed deformation gradient is exactly I + lam*H_macro
# at every point of the path.
# ============================================================================
H_macro = np.zeros((dim, dim))
H_macro[1, 0] = -0.6


discretization.get_macro_gradient_field_mugrid(
    macro_gradient_ij=H_macro,
    macro_gradient_field_ijqxyz=macro_gradient_full_field
)

dlam_nominal = 1.0 / float(ninc)

_info['H_macro'] = H_macro
_info['norm_strain_fluc_field'] = []
_info['mean_stress_field'] = []
_info['total_macro_gradient'] = []
_info['min_J'] = []
_info['load_scale'] = []
_info['lam'] = []

# ============================================================================
# initial state: lam = 0, u = 0  =>  F = I
# ============================================================================
displacement_fluctuation_field.s.fill(0)
set_F(0.0, displacement_fluctuation_field, total_strain_field)

material.get_stress(total_strain_field, stress_field)
material.get_algorithmic_tangent(total_strain_field, tangent_field)

root_print(f'tangent min/max : {tangent_field.s.min():.4e} / {tangent_field.s.max():.4e}')
root_print(f'min J at F = I  : {min_J(total_strain_field):.6e}')

# ============================================================================
# incremental loading — Newton-CG loop with step control
# ============================================================================
sum_CG_its = 0
sum_Newton_its = 0
iteration_total = 0
start_time = time.time()

lam = 0.0
inc = -1

while lam < 1.0 - 1e-12 and inc < MAX_INCREMENTS:
    inc += 1

    # ---- how much load can the worst quadrature point take? ----------------
    # u is frozen, so F is affine in dlam:  F + dlam * H_macro.
    s_star_load = max_admissible_scale(total_strain_field, macro_gradient_full_field)
    dlam = min(dlam_nominal, LOAD_SAFETY * s_star_load, 1.0 - lam)

    if dlam < DLAM_MIN:
        root_print(f'\nLoad increment underflow at lam = {lam:.6f}; stopping')
        break

    lam += dlam

    root_print(f'\nIncrement {inc}')
    root_print('=' * 70)
    root_print(f'  lam {lam:.6f}   dlam {dlam:.3e} (nominal {dlam_nominal:.3e})')
    root_print(f'  max admissible dlam {s_star_load:.3e}'
               f'   min J before load step {min_J(total_strain_field):.6e}')

    # apply the load step and evaluate the constitutive response at the new F
    set_F(lam, displacement_fluctuation_field, total_strain_field)

    norm_rhs_0 = assemble_rhs(total_strain_field, displacement_fluctuation_field)

    # Recompute the Green preconditioner from the current mean tangent
    ref_mat = tangent_field.s.mean(axis=tuple(range(4, tangent_field.s.ndim)))
    preconditioner = discretization.get_preconditioner_Green_mugrid(
        reference_material_data_ijkl=ref_mat,
        operator=operator_for_preconditioner,
    )

    En = np.sqrt(discretization.communicator.sum(
        np.dot(total_strain_field.s.ravel(), total_strain_field.s.ravel())
    ))

    root_print(f'  min J after load step   {min_J(total_strain_field):.6e}')
    root_print(f'  Rhs at new load step    {norm_rhs_0:10.2e}')
    root_print(f'  En  at new load step    {En:10.2e}')

    # ---- configure preconditioner -----------------------------------------
    if preconditioner_type == 'Green':
        M_fun = M_fun_Green

    elif preconditioner_type == 'Green_Jacobi':
        K_diag_alg = discretization.get_preconditioner_Jacobi_mugrid(
            material_data_field_ijklqxyz=tangent_field)

        x_jacobi_temp = discretization.get_unknown_size_field(name='x_jacobi_temp')

        def M_fun_Green_Jacobi(x, Px):
            discretization.fft.communicate_ghosts(x)
            x_jacobi_temp.s[...] = K_diag_alg.s * x.s
            discretization.apply_preconditioner_mugrid(
                preconditioner_Fourier_fnfnqks=preconditioner,
                input_nodal_field_fnxyz=x_jacobi_temp,
                output_nodal_field_fnxyz=Px)
            Px.s[...] = K_diag_alg.s * Px.s
            discretization.fft.communicate_ghosts(Px)

        M_fun = M_fun_Green_Jacobi

    # ------------------------------------------------------------------
    # Newton loop
    # ------------------------------------------------------------------
    iiter = 0
    norm_rhs = norm_rhs_0

    while True:
        norms = {'residual_rr': [], 'residual_rz': []}

        def callback(it, x, r, p, z, stop_crit_norm):
            norm_rr = discretization.communicator.sum(np.dot(r.ravel(), r.ravel()))
            norm_rz = discretization.communicator.sum(np.dot(r.ravel(), z.ravel()))
            norms['residual_rr'].append(norm_rr)
            norms['residual_rz'].append(norm_rz)

        displacement_increment_field.s.fill(0)

        solvers.conjugate_gradients_mugrid(
            comm=discretization.communicator,
            fc=discretization.field_collection,
            hessp=K_fun,
            b=rhs_field,
            x=displacement_increment_field,
            P=M_fun,
            tol=1e-6,
            maxiter=4000,
            callback=callback,
            rtol=True,
        )

        nb_it_cg = len(norms['residual_rr'])
        sum_CG_its += nb_it_cg
        iiter += 1
        sum_Newton_its += 1
        iteration_total += 1

        # strain from the displacement increment
        discretization.fft.communicate_ghosts(displacement_increment_field)
        discretization.apply_gradient_operator_mugrid(
            u_inxyz=displacement_increment_field,
            grad_u_ijqxyz=strain_fluc_field
        )

        norm_strain_fluc = np.sqrt(discretization.communicator.sum(
            np.dot(strain_fluc_field.s.ravel(), strain_fluc_field.s.ravel())
        ))

        # ================================================================
        # NEWTON STEP CONTROL.  `alpha` is the HuHu coefficient, so the
        # step length must not shadow it.
        # ================================================================
        s_star_step = max_admissible_scale(total_strain_field, strain_fluc_field)
        alpha_step = min(1.0, J_STEP_FRAC * s_star_step)

        if alpha_step <= 0.0:
            root_print('  min J <= 0 already; stopping increment')
            break

        displacement_fluctuation_field.s[...] += (
            alpha_step * displacement_increment_field.s[...])
        set_F(lam, displacement_fluctuation_field, total_strain_field)

        # re-evaluate the residual (also refreshes stress and tangent)
        norm_rhs = assemble_rhs(total_strain_field, displacement_fluctuation_field)
        material.get_energy_density(total_strain_field, energy_field)

        _info['norm_strain_fluc_field'].append(norm_strain_fluc)

        root_print(f'  Newton it {iiter}  |  CG its = {nb_it_cg}  '
                   f'|  alpha_step {alpha_step:8.5f}'
                   + ('' if alpha_step == 1.0 else '   [limited]'))
        root_print(f'    min J                         {min_J(total_strain_field):10.3e}')
        root_print(f'    norm(strain_fluc) / En        {norm_strain_fluc / En:10.2e}')
        root_print(f'    norm(rhs) / norm(rhs_0)       {norm_rhs / norm_rhs_0:10.2e}')
        root_print(f'    P_xx {stress_field.s[0, 0].mean():10.2e} | '
                   f'P_yx {stress_field.s[1, 0].mean():10.2e}')

        # save per iteration
        if save_per_it:
            save_npy(
                data_folder_path + f'displacement_fluctuation_field_it{iteration_total}.npy',
                displacement_fluctuation_field.s.mean(axis=1),
                tuple(discretization.subdomain_locations_no_buffers),
                tuple(discretization.nb_of_pixels_global),
                MPI.COMM_WORLD
            )
            save_npy(
                data_folder_path + f'energy_field_it{iteration_total}.npy',
                energy_field.s[0, 0].mean(axis=0),
                tuple(discretization.subdomain_locations_no_buffers),
                tuple(discretization.nb_of_pixels_global),
                MPI.COMM_WORLD
            )
            save_npy(
                data_folder_path + f'stress_field_it{iteration_total}.npy',
                stress_field.s.mean(axis=2),
                tuple(discretization.subdomain_locations_no_buffers),
                tuple(discretization.nb_of_pixels_global),
                MPI.COMM_WORLD
            )

        # convergence check
        if not np.isfinite(norm_rhs):
            root_print('  non-finite residual; stopping increment')
            break
        if norm_rhs < tol_newton * norm_rhs_0:
            break
        if iiter >= MAX_NEWTON_ITS:
            root_print(f'  Newton did not converge in {MAX_NEWTON_ITS} its '
                       f'(rel res {norm_rhs / norm_rhs_0:.2e})')
            break

    # ---- increment done ---------------------------------------------------
    total_macro_gradient = lam * H_macro
    _info['mean_stress_field'].append(float(stress_field.s[1, 0].mean()))
    _info['total_macro_gradient'].append(total_macro_gradient)
    _info['min_J'].append(min_J(total_strain_field))
    _info['load_scale'].append(dlam / dlam_nominal)
    _info['lam'].append(lam)

    if discretization.communicator.size == 1 and plot_figures:
        x_plot_ixyz = visualization_utils.get_deformed_grid_coords_two_dim(
            discretization,
            macro_gradient_ij=total_macro_gradient,
            displacement_fluctuation=displacement_fluctuation_field)
        visualization_utils.plot_field_on_grid(
            coordinates_for_plot=x_plot_ixyz,
            field_to_plot=phase_field.s[0, 0],
            name=fr'$\tilde{{u}}_{{x}}$' + f'load increment {inc}   ')

# ============================================================================
# response curves
# ============================================================================
if rank == 0 and plot_figures and len(_info['mean_stress_field']) > 1:
    F_plot = np.asarray(_info['total_macro_gradient'])[..., 1, 0]
    P_plot = np.asarray(_info['mean_stress_field'])

    fig, ax = plt.subplots(1, 3, figsize=(15, 5))

    ax[0].plot(F_plot, P_plot, '-o', ms=3, color='k')
    ax[0].set_xlabel(r'$\bar{F}_{10}$')
    ax[0].set_ylabel(r'$\bar{P}_{10}$')
    ax[0].grid(alpha=.3)

    ax[1].semilogy(F_plot, np.asarray(_info['min_J']), '-o', ms=3, color='C3')
    ax[1].set_xlabel(r'$\bar{F}_{10}$')
    ax[1].set_ylabel(r'$\min \det F$')
    ax[1].grid(alpha=.3)

    ax[2].plot(F_plot, np.asarray(_info['load_scale']), '-o', ms=3, color='C0')
    ax[2].set_xlabel(r'$\bar{F}_{10}$')
    ax[2].set_ylabel('load step / nominal')
    ax[2].grid(alpha=.3)

    fig.tight_layout()
    fig.savefig(figure_folder_path + 'response.png', dpi=150)
    plt.show()

# ============================================================================
# timing and summary
# ============================================================================
elapsed_time = time.time() - start_time

_info['sum_Newton_its'] = sum_Newton_its
_info['sum_CG_its'] = sum_CG_its
_info['iteration_total'] = iteration_total
_info['elapsed_time'] = elapsed_time
_info['lam_applied'] = lam

if rank == 0:
    print('=' * 70)
    print(f'element_type     : {element_type}')
    print(f'number_of_pixels : {number_of_pixels}')
    print(f'preconditioner   : {preconditioner_type}')
    print(f'reached          : lam = {lam:.6f} of 1.0   ({100 * lam:.1f}%)')
    if _info['min_J']:
        print(f'final min J      : {_info["min_J"][-1]:.6e}')
    print(f'increments       : {inc + 1}')
    print(f'Total CG its     : {sum_CG_its}')
    print(f'Total Newton its : {sum_Newton_its}')
    print(f'Elapsed time     : {elapsed_time:.2f} s  ({elapsed_time / 60:.2f} min)')
    np.savez(data_folder_path + 'info_log_final.npz', **_info)
