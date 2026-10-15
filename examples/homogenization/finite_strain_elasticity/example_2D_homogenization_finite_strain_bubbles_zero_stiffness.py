"""
Finite strain homogenization of a porous neo-Hookean solid: Green vs Green-Jacobi preconditioner.

Geometry: muFFTTO.geometry 'bubble_array' (2 x 2 round bubbles per cell). The bubbles have exactly zero
stiffness (lambda = mu = 0) by default, or --void_ratio times the solid stiffness: no third medium, no contact. Solver: incremental Newton, every Newton step solves
K du = -r with the preconditioned conjugate gradient method (pure Newton-CG, full steps, as in
example_2D_homogenization_finite_strain_elasticity_NeoHookean.py).

Every CG solve stops on the Meurant-Papez-Tichy (Hestenes-Stiefel) lower estimate of the energy error,
||x - x_k||_K <= cg_tol ||x - x_0||_K (solvers.conjugate_gradients_mugrid_experimental, stop_crit_norm=
'energy_lower_estim', energy_rtol=True, minimal delay d_min): the energy norm does not depend on the
preconditioner, so both preconditioners are stopped at the same accuracy.

Zero stiffness makes K singular: the nodes inside a bubble (all four neighbouring pixels void) have zero rows
and columns, and their displacement is not determined. The system is consistent (the residual is zero there),
so CG still converges. The Jacobi diagonal is zero at those nodes; Green-Jacobi stores 0 there
(zero_threshold=0), i.e. the preconditioner does not move them. Stress and tangent are set to zero in the void
after the material evaluation (the neo-Hookean law evaluates ln J, which is undefined if an undetermined void
pixel inverts).

Preconditioners (applied to the residual r):
    Green          z = G r,                 G = (B^T C_ref B)^-1 of the solid (FFT)
    Green_Jacobi   z = D^-1/2 G D^-1/2 r,   D = diag(K) of the current tangent (recomputed every Newton step)

usage:
    python example_2D_homogenization_finite_strain_bubbles_zero_stiffness.py -n 64 -inc 20 -p Green_Jacobi
writes exp_data/<script>/Nx=<n>Ny=<n>_<preconditioner>/info_log_final.npz
"""
import argparse
import os
import sys
import time

import numpy as np

# Add the project root to sys.path relatively
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from muFFTTO import domain
from muFFTTO import geometry
from muFFTTO import material_models
from muFFTTO import solvers

# ============================================================================
# argument parsing
# ============================================================================
parser = argparse.ArgumentParser(description='finite strain homogenization of a bubble array with void bubbles')
parser.add_argument('-n', '--nb_pixel', type=int, default=64, help='pixels per direction (default 64)')
parser.add_argument('-inc', '--nb_increments', type=int, default=20, help='load increments (default 20)')
parser.add_argument('-p', '--preconditioner', choices=['Green', 'Green_Jacobi'], default='Green_Jacobi')
parser.add_argument('--cg_tol', type=float, default=1e-6,
                    help='relative CG tolerance on the energy-norm error (default 1e-6)')
parser.add_argument('--cg_delay', type=int, default=10, help='minimal delay d_min of the energy estimate (default 10)')
parser.add_argument('--void_ratio', type=float, default=0.0,
                    help='stiffness of the bubbles / stiffness of the solid (default 0: no stiffness)')
parser.add_argument('--save_fields', action='store_true',
                    help='store the displacement fluctuation and det F of every increment in fields.npz')
parser.add_argument('--load', type=float, nargs=4, default=[0.0, 0.2, 0.0, 0.2], metavar=('H00', 'H01', 'H10', 'H11'),
                    help='total macroscopic displacement gradient F - I (default: shear 0.2 + stretch 0.2)')

script_name = os.path.splitext(os.path.basename(__file__))[0]
args = parser.parse_args()
nnn = args.nb_pixel
ninc = args.nb_increments
preconditioner_type = args.preconditioner

# ============================================================================
# problem setup
# ============================================================================
number_of_pixels = (nnn, nnn)
domain_size = [1, 1]
dim = len(domain_size)

problem_type = 'elasticity'
discretization_type = 'finite_element'
element_type = 'bilinear_rectangle'
formulation = 'finite_strain'

# total macroscopic displacement gradient F - I
macro_gradient_total = np.array(args.load).reshape(dim, dim)

newton_tol = 1e-8          # |grad du| / |F| (as in the two-phase NeoHookean example)
newton_maxiter = 100
cg_maxiter = 20000

_info = {
    'problem_type': problem_type,
    'discretization_type': discretization_type,
    'element_type': element_type,
    'formulation': formulation,
    'preconditioner_type': preconditioner_type,
    'nb_of_pixels': number_of_pixels,
    'domain_size': domain_size,
    'ninc': ninc,
    'geometry': 'bubble_array',
    'macro_gradient_total': macro_gradient_total,
    'cg_tol': args.cg_tol,
    'cg_delay': args.cg_delay,
    'cg_stop': 'energy_lower_estim, energy_rtol',
    'void_ratio': args.void_ratio,
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
comm = discretization.communicator

# ============================================================================
# output folders
# ============================================================================
file_folder_path = os.path.dirname(os.path.realpath(__file__))
data_folder_path = (file_folder_path + '/exp_data/' + script_name + '/'
                    + f'Nx={nnn}Ny={nnn}_{preconditioner_type}'
                    + (f'_void{args.void_ratio:g}' if args.void_ratio > 0 else '')
                    + ('' if args.load == [0.0, 0.2, 0.0, 0.2] else '_load' + '_'.join(f'{h:g}' for h in args.load))
                    + (f'_inc{ninc}' if ninc != 10 else '') + '/')
if comm.rank == 0:
    os.makedirs(data_folder_path, exist_ok=True)

# ============================================================================
# material parameters
# ============================================================================
# solid: neo-Hookean
E_solid = 1.0
nu_solid = 0.3
lam_solid = E_solid * nu_solid / ((1 + nu_solid) * (1 - 2 * nu_solid))
mu_solid = E_solid / (2 * (1 + nu_solid))
# bubbles: zero stiffness by default, or void_ratio x solid (a very soft neo-Hookean phase)
void_ratio = args.void_ratio
lam_void, mu_void = void_ratio * lam_solid, void_ratio * mu_solid

# reference material for the Green preconditioner: the solid
i = np.eye(dim)
II = np.einsum('ij,kl->ijkl', i, i)
I4rt = np.einsum('ik,jl->ijkl', i, i)
I4s = 0.5 * (I4rt + np.einsum('il,jk->ijkl', i, i))
ref_mat = lam_solid * II + 2.0 * mu_solid * I4s

_info['lam_solid'] = lam_solid
_info['mu_solid'] = mu_solid

# ============================================================================
# geometry: solid (1) with void bubbles (0), one value per pixel
# ============================================================================
phase_field = discretization.get_scalar_field(name='phase_field')
phase_field.s[0, 0] = geometry.get('bubble_array', discretization.fft.coords)

solid_mask = phase_field.s[0, 0] > 0
void_mask = ~solid_mask
porosity = comm.sum(float(np.sum(void_mask))) / np.prod(number_of_pixels)
_info['porosity'] = porosity

# ============================================================================
# material fields
# ============================================================================
lam_field = discretization.get_quad_field_scalar(name='lam_field')
mu_field = discretization.get_quad_field_scalar(name='mu_field')

lam_field.s[0, 0, :, solid_mask] = lam_solid
lam_field.s[0, 0, :, void_mask] = lam_void
mu_field.s[0, 0, :, solid_mask] = mu_solid
mu_field.s[0, 0, :, void_mask] = mu_void

material = material_models.NeoHookean(
    discretization=discretization,
    lam_1qxyz=lam_field,
    mu_1qxyz=mu_field,
    name='neo_hookean_bubbles'
)

# ============================================================================
# fields
# ============================================================================
macro_gradient_inc_field = discretization.get_gradient_size_field(name='macro_gradient_inc_field')
displacement_fluctuation_field = discretization.get_unknown_size_field(name='displacement_fluctuation_field')
displacement_increment_field = discretization.get_unknown_size_field(name='displacement_increment_field')
strain_fluc_field = discretization.get_displacement_gradient_sized_field(name='strain_fluctuation_field')
total_strain_field = discretization.get_displacement_gradient_sized_field(name='total_strain_field')
stress_field = discretization.get_displacement_gradient_sized_field(name='stress_field')
tangent_field = discretization.get_material_data_size_field_mugrid(name='tangent_field')
rhs_field = discretization.get_unknown_size_field(name='rhs_field')


def evaluate_material():
    """Stress and tangent at the current F; with zero void stiffness exactly zero in the bubbles."""
    with np.errstate(invalid='ignore', divide='ignore'):  # ln J of an undetermined (inverted) void pixel
        material.get_stress(total_strain_field, stress_field)
        material.get_algorithmic_tangent(total_strain_field, tangent_field)
    if void_ratio == 0:
        stress_field.s[:, :, :, void_mask] = 0.0
        tangent_field.s[:, :, :, :, :, void_mask] = 0.0


def compute_residual():
    """rhs = -B^T W P(F) and its Euclidean norm."""
    discretization.fft.communicate_ghosts(stress_field)
    discretization.apply_gradient_transposed_operator_mugrid(
        gradient_field_ijqxyz=stress_field,
        div_u_fnxyz=rhs_field,
        apply_weights=True
    )
    rhs_field.s[...] *= -1
    return np.sqrt(comm.sum(np.dot(rhs_field.s.ravel(), rhs_field.s.ravel())))


def mean_stress():
    """Volume average of P over all quadrature points (equal weights for Q1 on a regular grid)."""
    local = stress_field.s.sum(axis=tuple(range(2, stress_field.s.ndim)))
    return np.array([[comm.sum(float(local[a, b])) for b in range(dim)] for a in range(dim)]) / (
        np.prod(number_of_pixels) * stress_field.s.shape[2])


# ============================================================================
# initialize F = I (reference configuration)
# ============================================================================
total_strain_field.s[...] = 0.0
for d in range(dim):
    total_strain_field.s[d, d] = 1.0

# ============================================================================
# preconditioner
# ============================================================================
preconditioner = discretization.get_preconditioner_Green_mugrid(reference_material_data_ijkl=ref_mat)
jacobi_field = None   # D^-1/2, Green_Jacobi only
scaled_residual_field = discretization.get_unknown_size_field(name='scaled_residual_field')


def update_jacobi():
    """D^-1/2 of the current tangent; 0 where diag(K) = 0 (nodes inside a bubble)."""
    global jacobi_field
    if preconditioner_type == 'Green_Jacobi':
        jacobi_field = discretization.get_preconditioner_Jacobi_mugrid(
            material_data_field_ijklqxyz=tangent_field, formulation=formulation, zero_threshold=0.0)


def M_fun(x, Px):
    if preconditioner_type == 'Green_Jacobi':
        scaled_residual_field.s[...] = jacobi_field.s * x.s
        x = scaled_residual_field
    discretization.fft.communicate_ghosts(x)
    discretization.apply_preconditioner_mugrid(
        preconditioner_Fourier_fnfnqks=preconditioner,
        input_nodal_field_fnxyz=x,
        output_nodal_field_fnxyz=Px
    )
    if preconditioner_type == 'Green_Jacobi':
        Px.s[...] *= jacobi_field.s


def K_fun(x, Ax):
    discretization.apply_system_matrix_mugrid(
        material_data_field=tangent_field,
        input_field_inxyz=x,
        output_field_inxyz=Ax,
        formulation=formulation
    )
    discretization.fft.communicate_ghosts(Ax)


# ============================================================================
# macroscopic loading
# ============================================================================
macro_gradient_inc = macro_gradient_total / float(ninc)
discretization.get_macro_gradient_field_mugrid(
    macro_gradient_ij=macro_gradient_inc,
    macro_gradient_field_ijqxyz=macro_gradient_inc_field
)
_info['macro_gradient_inc'] = macro_gradient_inc

if comm.rank == 0:
    print(f'bubble_array {nnn}x{nnn}, porosity {porosity:.3f}, void stiffness {void_ratio:g} x solid, preconditioner {preconditioner_type}')

# ============================================================================
# incremental loading - Newton-CG loop
# ============================================================================
cg_per_newton = []        # per increment: list of CG iterations of its Newton steps
displacement_history = []  # per increment (--save_fields): displacement fluctuation at the nodes, (2, n, n)
det_F_history = []         # per increment (--save_fields): det F averaged over the quadrature points of a pixel
mean_P = []               # per increment: homogenized first Piola-Kirchhoff stress
newton_residuals = []
start_time = time.time()

loading_failed = None      # reason, if the load path stops early (the tangent lost positive definiteness)
for inc in range(ninc):
    # apply macroscopic deformation gradient increment
    total_strain_field.s[...] += macro_gradient_inc_field.s[...]
    evaluate_material()
    norm_rhs_0 = compute_residual()
    En = np.sqrt(comm.sum(np.dot(total_strain_field.s.ravel(), total_strain_field.s.ravel())))

    cg_its, residuals = [], [norm_rhs_0]
    for newton_it in range(newton_maxiter):
        update_jacobi()
        nb_cg = [0]

        def count(it, x, r, p, z, stop_crit_norm):
            nb_cg[0] = it

        displacement_increment_field.s.fill(0)
        try:
            solvers.conjugate_gradients_mugrid_experimental(
                comm=comm,
                fc=discretization.field_collection,
                hessp=K_fun,
                b=rhs_field,
                x=displacement_increment_field,
                P=M_fun,
                tol=args.cg_tol,
                maxiter=cg_maxiter,
                callback=count,
                stop_crit_norm='energy_lower_estim',
                energy_rtol=True,
                d_min=args.cg_delay,
            )
        except RuntimeError as error:    # e.g. buckling: plain CG needs a positive definite tangent
            loading_failed = f'increment {inc + 1}, Newton step {newton_it + 1}: {error}'
            break
        cg_its.append(nb_cg[0])

        # full Newton step: F <- F + grad(du)
        discretization.apply_gradient_operator_mugrid(
            u_inxyz=displacement_increment_field,
            grad_u_ijqxyz=strain_fluc_field
        )
        if void_ratio == 0:
            strain_fluc_field.s[:, :, :, void_mask] = 0.0      # F in the bubbles is irrelevant (zero stiffness)
        norm_strain_fluc = np.sqrt(comm.sum(np.dot(strain_fluc_field.s.ravel(), strain_fluc_field.s.ravel())))
        total_strain_field.s[...] += strain_fluc_field.s[...]
        displacement_fluctuation_field.s[...] += displacement_increment_field.s[...]

        evaluate_material()
        residuals.append(compute_residual())

        if norm_strain_fluc / En < newton_tol:
            break
    else:
        if comm.rank == 0:
            print(f'  WARNING: Newton did not converge in {newton_maxiter} iterations')
    if loading_failed is not None:
        if comm.rank == 0:
            print(f'load path stopped at {loading_failed}')
        break

    cg_per_newton.append(cg_its)
    if args.save_fields:
        F = total_strain_field.s                                              # (2, 2, q, x, y)
        displacement_history.append(np.array(displacement_fluctuation_field.s[:, 0]))
        det_F_history.append((F[0, 0] * F[1, 1] - F[0, 1] * F[1, 0]).mean(axis=0))
    newton_residuals.append(residuals)
    mean_P.append(mean_stress())
    if comm.rank == 0:
        print(f'increment {inc + 1:3d}/{ninc}  Newton its {len(cg_its):2d}  CG its {sum(cg_its):6d} '
              f'({", ".join(str(c) for c in cg_its)})  |r|/|r0| {residuals[-1] / residuals[0]:.1e}  '
              f'P01 {mean_P[-1][0, 1]:+.4e}  P11 {mean_P[-1][1, 1]:+.4e}')

# ============================================================================
# timing and summary
# ============================================================================
elapsed_time = time.time() - start_time
sum_CG_its = int(sum(sum(c) for c in cg_per_newton))
sum_Newton_its = int(sum(len(c) for c in cg_per_newton))

_info['sum_Newton_its'] = sum_Newton_its
_info['sum_CG_its'] = sum_CG_its
_info['elapsed_time'] = elapsed_time
_info['cg_per_increment'] = np.array([sum(c) for c in cg_per_newton])
_info['newton_per_increment'] = np.array([len(c) for c in cg_per_newton])
_info['cg_per_newton_flat'] = np.concatenate([np.asarray(c, dtype=int) for c in cg_per_newton])
_info['mean_P'] = np.array(mean_P)
_info['completed_increments'] = len(cg_per_newton)
_info['loading_failed'] = '' if loading_failed is None else loading_failed

if comm.rank == 0:
    print('=' * 70)
    print(f'number_of_pixels : {number_of_pixels}')
    print(f'preconditioner   : {preconditioner_type}')
    print(f'Total CG its     : {sum_CG_its}')
    print(f'Total Newton its : {sum_Newton_its}')
    print(f'Elapsed time     : {elapsed_time:.2f} s')
    np.savez(data_folder_path + 'info_log_final.npz', **_info)
    if args.save_fields:
        np.savez(data_folder_path + 'fields.npz', displacement=np.array(displacement_history),
                 det_F=np.array(det_F_history), solid=solid_mask,
                 macro_gradient=np.array([(k + 1) * macro_gradient_inc for k in range(len(cg_per_newton))]))
        print(f'wrote {data_folder_path}fields.npz')
    print(f'wrote {data_folder_path}info_log_final.npz')
