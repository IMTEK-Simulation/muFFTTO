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
    prog='exp_finite_strain_2D_NeoHookean.py',
    description='Solve finite strain NeoHookean elasticity in 2D'
)
parser.add_argument('-n', '--nb_pixel', default='64')
parser.add_argument('-inc', '--nb_increments', default='100')
parser.add_argument(
    '--save_per_it',
    action='store_true',
    help='Enable saving every iteration'
)

script_name = os.path.splitext(os.path.basename(__file__))[0]
args = parser.parse_args()
nnn = int(args.nb_pixel)
ninc = int(args.nb_increments)
save_per_it = args.save_per_it
# ============================================================================
# problem setup
# ============================================================================
number_of_pixels = (nnn, nnn)
domain_size = [1, 1]
dim = len(domain_size)
tol_newton = 1e-4
problem_type = 'elasticity'
discretization_type = 'finite_element'
element_type = 'bilinear_rectangle'
formulation = 'finite_strain'
preconditioner_type = "Green"  # Options: 'Green', 'Jacobi', 'Green_Jacobi'

DEBUG_PLOTS = False   # per-Newton-iteration plots; True is very slow

# --- step control settings --------------------------------------------------
J_STEP_FRAC = 0.5     # a Newton step may not take min J below this fraction
LOAD_SAFETY = 0.5     # fraction of the way to det F = 0 for the load step
MAX_INCREMENTS = 10 * ninc

_info = {
    'problem_type': problem_type,
    'discretization_type': discretization_type,
    'element_type': element_type,
    'formulation': formulation,
    'preconditioner_type': preconditioner_type,
    'nb_of_pixels': number_of_pixels,
    'domain_size': domain_size,
    'ninc': ninc,
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

# ============================================================================
# output folders
# ============================================================================
file_folder_path = os.path.dirname(os.path.realpath(__file__))
data_folder_path = (file_folder_path + '/exp_data/' + script_name + '/'
                    + f'Nx={nnn}Ny={nnn}_{preconditioner_type}/')
figure_folder_path = (file_folder_path + '/figures/' + script_name + '/'
                      + f'Nx={nnn}Ny={nnn}_{preconditioner_type}/')

if discretization.communicator.rank == 0:
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
k_v = 1e-4  # TMC contrast, Table 1
E_void = k_v * E_matrix
nu_void = nu_matrix  # keep the solid's Poisson ratio
lam_void = E_void * nu_void / ((1 + nu_void) * (1 - 2 * nu_void))
mu_void = E_void / (2 * (1 + nu_void))

# HuHu regularization
alpha = 1e-6
k_r = alpha * domain_size[0] ** 2 * (K + G * 4 / 3)

# reference material for Green preconditioner
i = np.eye(dim)
II = np.einsum('ij,kl->ijkl', i, i)
I4rt = np.einsum('ik,jl->ijkl', i, i)
I4s = 0.5 * (I4rt + np.einsum('il,jk->ijkl', i, i))
I4d = I4s - II / 2.0  # 2D: divide by dim=2 not 3
ref_mat = lam_matrix * II + 2.0 * mu_matrix * I4s

_info['lam_matrix'] = lam_matrix
_info['mu_matrix'] = mu_matrix
_info['lam_inc'] = lam_void
_info['mu_inc'] = mu_void

# ============================================================================
# geometry — square inclusion
# ============================================================================
phase_field = discretization.get_scalar_field(name='phase_field')
phase_field.s[0, 0] = microstructure_library.get_geometry(
    nb_voxels=discretization.nb_of_pixels,
    microstructure_name='contact_test_geometry_3',
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

if discretization.communicator.rank == 0:
    print(f'lam min/max : {lam_field.s.min():.4f} / {lam_field.s.max():.4f}')
    print(f'mu  min/max : {mu_field.s.min():.4f}  / {mu_field.s.max():.4f}')
    print(f'k_r (HuHu)  : {k_r:.6e}   (lam_void + mu_void = {lam_void + mu_void:.6e})')

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
macro_gradient_inc_field = discretization.get_gradient_size_field(name='macro_gradient_inc_field')
displacement_fluctuation_field = discretization.get_unknown_size_field(name='displacement_fluctuation_field')
displacement_increment_field = discretization.get_unknown_size_field(name='displacement_increment_field')
strain_fluc_field = discretization.get_displacement_gradient_sized_field(name='strain_fluctuation_field')
total_strain_field = discretization.get_displacement_gradient_sized_field(name='total_strain_field')
stress_field = discretization.get_displacement_gradient_sized_field(name='stress_field')
tangent_field = discretization.get_material_data_size_field_mugrid(name='tangent_field')
rhs_field = discretization.get_unknown_size_field(name='rhs_field')
energy_field = discretization.get_quad_field_scalar(name='energy_field')
hess_u_ijkqxyz = discretization.get_displacement_hessian_size_field(name='hess_u')
HtH_field = discretization.get_unknown_size_field(name='HtH')

# --- needed by the step length control ---------------------------------------
J_1qxyz = discretization.get_quad_field_scalar(name='J')
F_trial = discretization.get_displacement_gradient_sized_field(name='F_trial')


# ============================================================================
# helpers for the step length control
# ============================================================================
def min_J(F_in):
    """min over all quadrature points of det F."""
    tensor_operations.det2(F_in, J_1qxyz)
    return MPI.COMM_WORLD.allreduce(float(np.min(J_1qxyz.s)), op=MPI.MIN)


def max_J(F_in):
    tensor_operations.det2(F_in, J_1qxyz)
    return MPI.COMM_WORLD.allreduce(float(np.max(J_1qxyz.s)), op=MPI.MAX)


def admissible_dF00(F_in):
    """Largest |dF00| that keeps det F > 0 at every quadrature point.

    dF is rank one (only component [0,0]), so by the matrix determinant lemma
        det(F + dF) = det F + dF00 * F11
    EXACTLY -- no higher-order terms.  Staying positive therefore needs
        |dF00| < J / F11
    Only F11 > 0 constrains; where F11 < 0 the increment raises det F.

    Returns a SCALAR.  Multiplying the macro gradient field by the [q,x,y]
    array instead broadcasts against its [2,2,q,x,y] shape, giving every
    quadrature point a different macro gradient -- the load is then no longer
    uniform and its average is no longer the prescribed value.
    """
    Fs = F_in.s
    Jq = Fs[0, 0] * Fs[1, 1] - Fs[0, 1] * Fs[1, 0]
    lim_qxy = np.where(Fs[1, 1] > 0, Jq / np.maximum(Fs[1, 1], 1e-300), np.inf)
    return MPI.COMM_WORLD.allreduce(float(lim_qxy.min()), op=MPI.MIN)


def limited_step_length(F_cur, dF, frac=J_STEP_FRAC, max_halvings=40):
    """Largest alpha in (0, 1] with  min det(F_cur + alpha*dF) > frac * min det(F_cur).

    Costs a handful of det2 evaluations per Newton iteration and returns 1.0
    unchanged whenever the full step is harmless, so early increments are
    unaffected.  Without it, the Newton step at increment 81 of the previous
    run took min J from 3.5e-3 to -9.8e-2.
    """
    J_now = min_J(F_cur)
    if J_now <= 0.0:
        return 0.0
    alpha_step = 1.0
    for _ in range(max_halvings):
        F_trial.s[...] = F_cur.s + alpha_step * dF.s
        if min_J(F_trial) > frac * J_now:
            return alpha_step
        alpha_step *= 0.5
    return alpha_step


# ============================================================================
# initialize F = I (reference configuration)
# ============================================================================
total_strain_field.s[...] = 0.0
for d in range(dim):
    total_strain_field.s[d, d] = 1.0

# ============================================================================
# preconditioner
# ============================================================================
preconditioner = discretization.get_preconditioner_Green_mugrid(
    reference_material_data_ijkl=ref_mat
)


def M_fun_Green(x, Px):
    discretization.fft.communicate_ghosts(x)
    discretization.apply_preconditioner_mugrid(
        preconditioner_Fourier_fnfnqks=preconditioner,
        input_nodal_field_fnxyz=x,
        output_nodal_field_fnxyz=Px
    )


# ============================================================================
# macroscopic loading
#
# NOTE: the load is applied to component [0,0].  The original reported and
# plotted [1,0], which is identically zero -- hence "Load 0.0" every increment
# and a response curve of zeros against zeros.  LOAD_IJ keeps driving and
# reporting in sync.
# ============================================================================
LOAD_IJ = (0, 0)
F_TARGET = -0.3
macro_gradient_inc = np.zeros((dim, dim))
macro_gradient_inc[LOAD_IJ] += F_TARGET / float(ninc)

discretization.get_macro_gradient_field_mugrid(
    macro_gradient_ij=macro_gradient_inc,
    macro_gradient_field_ijqxyz=macro_gradient_inc_field
)

dF00_nominal = abs(macro_gradient_inc[LOAD_IJ])

_info['macro_gradient_inc'] = macro_gradient_inc
_info['norm_strain_fluc_field'] = []
_info['mean_stress_field'] = []
_info['total_macro_gradient'] = []
_info['min_J'] = []
_info['load_scale'] = []

# initial constitutive evaluation
material.get_stress(total_strain_field, stress_field)
material.get_algorithmic_tangent(total_strain_field, tangent_field)

if discretization.communicator.rank == 0:
    print(f'tangent min/max : {tangent_field.s.min():.4f} / {tangent_field.s.max():.4f}')

# ============================================================================
# incremental loading — Newton-CG loop
# ============================================================================
sum_CG_its = 0
sum_Newton_its = 0
iteration_total = 0
start_time = time.time()

lam_applied = 0.0     # fraction of the TOTAL load actually applied so far
inc = -1

while lam_applied < 1.0 - 1e-12 and inc < MAX_INCREMENTS:
    inc += 1

    # ---- how much load can the worst quadrature point take? ---------------
    lim = admissible_dF00(total_strain_field)
    scale = min(1.0, LOAD_SAFETY * lim / dF00_nominal)
    scale = min(scale, (1.0 - lam_applied) * ninc)      # do not overshoot target

    if scale * dF00_nominal < 1e-9:
        if discretization.communicator.rank == 0:
            print(f'\nLoad increment underflow at lam = {lam_applied:.6f}; stopping')
        break

    if discretization.communicator.rank == 0:
        print(f'\nIncrement {inc}')
        print(f'Load {lam_applied * F_TARGET + scale * macro_gradient_inc[LOAD_IJ]:.6f}')
        print('=' * 70)
        print(f'Minimum of J {min_J(total_strain_field)}')
        print(f'Max of J {max_J(total_strain_field)}')
        print(f'max admissible |dF00| = {lim:.3e}   '
              f'(applying {scale * dF00_nominal:.3e}, scale = {scale:.4f})')

    # apply macroscopic deformation gradient increment (scale is a SCALAR)
    total_strain_field.s[...] += scale * macro_gradient_inc_field.s[...]
    lam_applied += scale / ninc

    # constitutive response at new F
    material.get_stress(total_strain_field, stress_field)
    material.get_algorithmic_tangent(total_strain_field, tangent_field)

    # assemble rhs = -div(P)
    discretization.fft.communicate_ghosts(stress_field)
    discretization.apply_gradient_transposed_operator_mugrid(
        gradient_field_ijqxyz=stress_field,
        div_u_fnxyz=rhs_field,
        apply_weights=True
    )
    rhs_field.s[...] *= -1

    En = np.sqrt(discretization.communicator.sum(
        np.dot(total_strain_field.s.ravel(), total_strain_field.s.ravel())
    ))
    norm_rhs_0 = np.sqrt(discretization.communicator.sum(
        np.dot(rhs_field.s.ravel(), rhs_field.s.ravel())
    ))

    if discretization.communicator.rank == 0:
        print(f'Rhs at new load step    {norm_rhs_0:10.2e}')
        print(f'En  at new load step    {En:10.2e}')

    # 4. Configure preconditioner
    if preconditioner_type == 'Green':
        M_fun = M_fun_Green

    elif preconditioner_type == 'Green_Jacobi':
        K_diag_alg = discretization.get_preconditioner_Jacobi_mugrid(
            material_data_field_ijklqxyz=tangent_field)


        def M_fun_Green_Jacobi(x, Px):
            discretization.fft.communicate_ghosts(x)
            x_jacobi_temp = discretization.get_unknown_size_field(name='x_jacobi_temp')

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

        def K_fun(x, Ax):
            discretization.apply_system_matrix_mugrid(
                material_data_field=tangent_field,
                input_field_inxyz=x,
                output_field_inxyz=Ax,
                formulation=formulation
            )
            discretization.apply_hessian_operator_to_vector_field_mugrid(
                u_inxyz=x, hess_u_ijkqxyz=hess_u_ijkqxyz)
            discretization.apply_hessian_operator_transposed_to_vector_field_mugrid(
                hess_u_ijkqxyz=hess_u_ijkqxyz, nodal_field_inxyz=HtH_field, apply_weights=True)
            Ax.s[...] += k_r * HtH_field.s

            discretization.fft.communicate_ghosts(Ax)


        norms = {'residual_rr': [], 'residual_rz': []}


        def callback(it, x, r, p, z, stop_crit_norm):
            norm_rr = discretization.communicator.sum(np.dot(r.ravel(), r.ravel()))
            norm_rz = discretization.communicator.sum(np.dot(r.ravel(), z.ravel()))
            norms['residual_rr'].append(norm_rr)
            norms['residual_rz'].append(norm_rz)


        displacement_increment_field.s.fill(0)

        # muFFTTO's CG raises on negative curvature.  That is real information
        # (the medium is buckling), not a reason to abort the run.
        cg_failed = False
        try:
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
        except RuntimeError as exc:
            cg_failed = True
            if discretization.communicator.rank == 0:
                print(f'  CG: {exc}  -> preconditioned steepest descent')
            M_fun(rhs_field, displacement_increment_field)

        nb_it_cg = len(norms['residual_rr'])
        sum_CG_its += nb_it_cg
        iiter += 1
        sum_Newton_its += 1
        iteration_total += 1

        if discretization.communicator.rank == 0:
            print(f'  Newton it {iiter}  |  CG its = {nb_it_cg}')

        # strain from displacement increment
        discretization.apply_gradient_operator_mugrid(
            u_inxyz=displacement_increment_field,
            grad_u_ijqxyz=strain_fluc_field
        )

        norm_strain_fluc = np.sqrt(discretization.communicator.sum(
            np.dot(strain_fluc_field.s.ravel(), strain_fluc_field.s.ravel())
        ))

        # ================================================================
        # STEP LENGTH CONTROL -- applied to BOTH fields.  Note the name:
        # `alpha` above is the HuHu coefficient and must not be shadowed.
        # ================================================================
        alpha_step = limited_step_length(total_strain_field, strain_fluc_field)

        if alpha_step == 0.0:
            if discretization.communicator.rank == 0:
                print('  min J <= 0 already; stopping increment')
            break

        total_strain_field.s[...] += alpha_step * strain_fluc_field.s[...]
        displacement_fluctuation_field.s[...] += alpha_step * displacement_increment_field.s[...]

        if discretization.communicator.rank == 0:
            print(f'  alpha_step {alpha_step:8.5f}'
                  + ('   [neg curvature]' if cg_failed else ''))
            print(f'Minimum of J {min_J(total_strain_field)}')

        if DEBUG_PLOTS:
            tensor_operations.det2(total_strain_field, J_1qxyz)
            visualization_utils.plot_field_on_grid(
                coordinates_for_plot=discretization.get_nodal_points_coordinates_with_periodic_nodes()[:, 0],
                field_to_plot=J_1qxyz.s.mean(axis=2)[0, 0],
                name=f'load increment {inc}' + f' Newton it {iiter}  ')

            total_macro_gradient = lam_applied * macro_gradient_inc * ninc
            x_plot_ixyz = visualization_utils.get_deformed_grid_coords_two_dim(
                discretization,
                macro_gradient_ij=total_macro_gradient,
                displacement_fluctuation=displacement_fluctuation_field)
            visualization_utils.plot_field_on_grid(
                coordinates_for_plot=x_plot_ixyz,
                field_to_plot=phase_field.s[0, 0],
                name=f'load increment {inc}' + f' Newton it {iiter}  ')

        # re-evaluate constitutive response
        material.get_stress(total_strain_field, stress_field)
        material.get_algorithmic_tangent(total_strain_field, tangent_field)
        material.get_energy_density(total_strain_field, energy_field)

        # recompute rhs
        discretization.fft.communicate_ghosts(stress_field)
        discretization.apply_gradient_transposed_operator_mugrid(
            gradient_field_ijqxyz=stress_field,
            div_u_fnxyz=rhs_field,
            apply_weights=True
        )
        rhs_field.s[...] *= -1
        discretization.apply_hessian_operator_to_vector_field_mugrid(
            u_inxyz=displacement_fluctuation_field, hess_u_ijkqxyz=hess_u_ijkqxyz)
        discretization.apply_hessian_operator_transposed_to_vector_field_mugrid(
            hess_u_ijkqxyz=hess_u_ijkqxyz, nodal_field_inxyz=HtH_field, apply_weights=True)
        rhs_field.s[...] -= k_r * HtH_field.s
        norm_rhs = np.sqrt(discretization.communicator.sum(
            np.dot(rhs_field.s.ravel(), rhs_field.s.ravel())
        ))

        _info['norm_strain_fluc_field'].append(norm_strain_fluc)

        if discretization.communicator.rank == 0:
            print(f'  norm(strain_fluc) / En          {norm_strain_fluc / En:10.2e}')
            print(f'  norm(rhs) / norm(rhs_0)         {norm_rhs / norm_rhs_0:10.2e}')
            print(f'  norm(rhs)                       {norm_rhs:10.2e}')
            print(fr' $\sigma_{{xx}}$         {stress_field.s[0, 0].mean():10.2e}')
            print(fr' $\sigma_{{yy}}$         {stress_field.s[1, 1].mean():10.2e}')

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
            if discretization.communicator.rank == 0:
                print('  non-finite residual; stopping increment')
            break
        if norm_rhs < tol_newton * norm_rhs_0:
            break
        if iiter >= 100:
            break

    # ---- increment done ---------------------------------------------------
    total_macro_gradient = lam_applied * macro_gradient_inc * ninc
    _info['mean_stress_field'].append(float(stress_field.s[LOAD_IJ].mean()))
    _info['total_macro_gradient'].append(total_macro_gradient)
    _info['min_J'].append(min_J(total_strain_field))
    _info['load_scale'].append(scale)

    if discretization.communicator.size == 1:
        x_plot_ixyz = visualization_utils.get_deformed_grid_coords_two_dim(
            discretization,
            macro_gradient_ij=total_macro_gradient,
            displacement_fluctuation=displacement_fluctuation_field)
        tensor_operations.det2(total_strain_field, J_1qxyz)
        visualization_utils.plot_field_on_grid(
            coordinates_for_plot=x_plot_ixyz,
            field_to_plot=phase_field.s[0, 0],
            name=fr'$\tilde{{u}}_{{x}}$' + f'load increment {inc}   ')
        print(f'Minimum of J {np.min(J_1qxyz.s[...])}')

# ============================================================================
# response curve
# ============================================================================
F = np.asarray(_info['total_macro_gradient'])[..., LOAD_IJ[0], LOAD_IJ[1]]
P = np.asarray(_info['mean_stress_field'])

fig, ax = plt.subplots(1, 3, figsize=(15, 5))

ax[0].plot(F, P, '-o', ms=3, color='k')
ax[0].set_xlabel(rf'$\bar{{F}}_{{{LOAD_IJ[0]}{LOAD_IJ[1]}}}$')
ax[0].set_ylabel(rf'$\bar{{P}}_{{{LOAD_IJ[0]}{LOAD_IJ[1]}}}$')
ax[0].grid(alpha=.3)

ax[1].semilogy(F, np.asarray(_info['min_J']), '-o', ms=3, color='k')
ax[1].set_xlabel(rf'$\bar{{F}}_{{{LOAD_IJ[0]}{LOAD_IJ[1]}}}$')
ax[1].set_ylabel(r'$\min J$')
ax[1].grid(alpha=.3)

ax[2].plot(F, np.asarray(_info['load_scale']), '-o', ms=3, color='k')
ax[2].set_xlabel(rf'$\bar{{F}}_{{{LOAD_IJ[0]}{LOAD_IJ[1]}}}$')
ax[2].set_ylabel('load step scale')
ax[2].grid(alpha=.3)

fig.tight_layout()
fig.savefig(figure_folder_path + 'response.png', dpi=150)
plt.show()
# ============================================================================
# timing and summary
# ============================================================================
end_time = time.time()
elapsed_time = end_time - start_time

_info['sum_Newton_its'] = sum_Newton_its
_info['sum_CG_its'] = sum_CG_its
_info['iteration_total'] = iteration_total
_info['elapsed_time'] = elapsed_time
_info['lam_applied'] = lam_applied

if discretization.communicator.rank == 0:
    print('=' * 70)
    print(f'element_type     : {element_type}')
    print(f'number_of_pixels : {number_of_pixels}')
    print(f'preconditioner   : {preconditioner_type}')
    print(f'reached          : F_{LOAD_IJ[0]}{LOAD_IJ[1]} = '
          f'{lam_applied * F_TARGET:+.5f} of {F_TARGET:+.5f}   '
          f'({100 * lam_applied:.1f}%)')
    if _info['min_J']:
        print(f'final min J      : {_info["min_J"][-1]:.6e}')
    print(f'Total CG its     : {sum_CG_its}')
    print(f'Total Newton its : {sum_Newton_its}')
    print(f'Elapsed time     : {elapsed_time:.2f} s  ({elapsed_time / 60:.2f} min)')
    np.savez(data_folder_path + 'info_log_final.npz', **_info)