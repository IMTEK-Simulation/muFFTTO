import time
import os
import sys
import argparse

import numpy as np
from mpi4py import MPI
from NuMPI.IO import save_npy
from matplotlib import pyplot as plt

# Add the project root to sys.path relatively
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

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

problem_type = 'elasticity'
discretization_type = 'finite_element'
element_type = 'bilinear_rectangle'
formulation = 'finite_strain'
preconditioner_type = 'Green'

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
K,G =material_models.get_bulk_and_shear_modulus(E_matrix, nu_matrix)

# third medium ("void"): same neo-Hookean as the matrix, k_v times softer
k_v = 1e-3  # TMC contrast, Table 1
E_void = k_v * E_matrix
nu_void = nu_matrix  # keep the solid's Poisson ratio
lam_void = E_void * nu_void / ((1 + nu_void) * (1 - 2 * nu_void))
mu_void = E_void / (2 * (1 + nu_void))

# HuHu regularization
alpha=1e-6
k_r = alpha * domain_size[0]* 2* (K + G*4/3 )


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

if discretization.communicator.rank == 0:
    print(f'lam min/max : {lam_field.s.min():.4f} / {lam_field.s.max():.4f}')
    print(f'mu  min/max : {mu_field.s.min():.4f}  / {mu_field.s.max():.4f}')

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
# ============================================================================
macro_gradient_inc = np.zeros((dim, dim))
macro_gradient_inc[0, 0] += -0.2 / float(ninc)
# macro_gradient_inc[1, 1] += 0.3 / float(ninc)

discretization.get_macro_gradient_field_mugrid(
    macro_gradient_ij=macro_gradient_inc,
    macro_gradient_field_ijqxyz=macro_gradient_inc_field
)

_info['macro_gradient_inc'] = macro_gradient_inc
_info['norm_strain_fluc_field'] = []
_info['mean_stress_field'] = []
_info['total_macro_gradient'] = []

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

for inc in range(ninc):
    if discretization.communicator.rank == 0:
        print(f'Increment {inc}')
        print(f'Load {inc*macro_gradient_inc[0, 0] }')

        print('=' * 70)

    # apply macroscopic deformation gradient increment
    total_strain_field.s[...] += macro_gradient_inc_field.s[...]

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
            discretization.fft.communicate_ghosts(Ax)


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
            P=M_fun_Green,
            tol=1e-5,
            maxiter=1000,
            callback=callback,
            # rtol=True,
        )

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

        # update total strain and displacement
        total_strain_field.s[...] += strain_fluc_field.s[...]
        displacement_fluctuation_field.s[...] += displacement_increment_field.s[...]

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
        if norm_strain_fluc / En < 1e-8 and iiter > 0:
            break
        if iiter >= 100:
            break
    _info['mean_stress_field'].append(stress_field.s[1, 1].mean())
    total_macro_gradient = (inc + 1) * macro_gradient_inc
    _info['total_macro_gradient'].append(total_macro_gradient)
    if discretization.communicator.size == 1:
        # Plot the first two components of the solution field
        # Calculate total macroscopic gradient (F - I)

        x_plot_ixyz = visualization_utils.get_deformed_grid_coords_two_dim(discretization,
                                                                           macro_gradient_ij=total_macro_gradient,
                                                                           displacement_fluctuation=displacement_fluctuation_field)
        J_1qxyz = discretization.get_quad_field_scalar(name='J')
        tensor_operations.det2(total_strain_field, J_1qxyz)
        visualization_utils.plot_field_on_grid(
            coordinates_for_plot=x_plot_ixyz,
            field_to_plot=J_1qxyz.s.mean(2)[0,0],
            name=fr'$\tilde{{u}}_{{x}}$' + f'load increment {inc}   ')
    F = np.asarray(_info['total_macro_gradient'])[..., 0, 0]
    P = np.asarray(_info['mean_stress_field'])

    fig, ax = plt.subplots(1, 1, figsize=(5, 5), sharex=True)

    ax.plot(F, P, '-o', ms=3, color='k')
    ax.set_ylabel(r'$\bar{P}_{00}$')
    ax.grid(alpha=.3)

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

if discretization.communicator.rank == 0:
    print('=' * 70)
    print(f'element_type     : {element_type}')
    print(f'number_of_pixels : {number_of_pixels}')
    print(f'preconditioner   : {preconditioner_type}')
    print(f'Total CG its     : {sum_CG_its}')
    print(f'Total Newton its : {sum_Newton_its}')
    print(f'Elapsed time     : {elapsed_time:.2f} s  ({elapsed_time / 60:.2f} min)')
    np.savez(data_folder_path + 'info_log_final.npz', **_info)
