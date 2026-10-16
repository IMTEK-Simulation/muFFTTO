import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import numpy as np
import time

from mpi4py import MPI
from muFFTTO import domain
from muFFTTO import solvers
from muFFTTO import topology_optimization
from muFFTTO import material_models
from muFFTTO import io_utils
from muFFTTO import solvers_nonlinear

# Problem Configuration
problem_type = 'elasticity'
discretization_type = 'finite_element'
element_type = 'linear_triangles'
formulation = 'small_strain'

# Domain and Discretization
domain_size = [1, 1]
n_pix = int(os.environ.get('NPIX', 64))  # grid (environment NPIX overrides)
number_of_pixels = (n_pix, n_pix)
dim = np.size(number_of_pixels)
pixel_size = np.asarray(domain_size) / np.asarray(number_of_pixels)

# Optimization Parameters
soft_phase_exponent = int(os.environ.get('SOFT_EXP', 5))  # void = 10^-SOFT_EXP C_0
# Options: 'Green', 'Jacobi', 'Green_Jacobi', 'dual' (G K(C_0 A^-1 C_0) G); first command-line argument overrides
preconditioner_type = sys.argv[1] if len(sys.argv) > 1 else "Green_Jacobi"
show_plots = os.environ.get('SHOW_PLOTS', '1') == '1'  # phase field figure after every optimizer iteration (serial)
eta = float(os.environ['ETA']) if 'ETA' in os.environ else max(1 * pixel_size)  # Filter width (ETA: fixed in physical units)
init_noise_grid = int(os.environ.get('INIT_NOISE_GRID', '0'))  # >0: start noise drawn on this grid (see below)
weight = 5.  # Weight for the stress match term
cg_setup = {'cg_tol': 1e-3}
# CG_MODE='energy': every forward and adjoint solve from zero, stopped on the energy-error estimate relative to the
# initial error (1e-6, delay 10) -> accurate gradients, the same for every preconditioner. Default: warm starts and
# the residual tolerance cg_tol (absolute for the forward problem, relative to |b| for the adjoint)
cg_mode = os.environ.get('CG_MODE', 'residual')
if cg_mode == 'energy':
    cg_setup = {'cg_tol': 1e-6, 'cg_stop': 'energy', 'cg_warm_start': False, 'cg_delay': 10}
optimizer = 'l_bfgs'  # Options: 'l_bfgs', 'mma', 'tr_newton' (box constraints only), see solvers_nonlinear.minimize
volume_fraction = None  # None: box constraints only; e.g. 0.6: mean(phase) <= 0.6 (MMA) / = 0.6 (projected L-BFGS)
fd_step = 1e-3  # tr_newton: Hessian-vector products by central differences of the gradient, step in phase units
if optimizer == 'tr_newton':
    if volume_fraction is not None:
        raise ValueError('tr_newton_bounded supports box constraints only; use l_bfgs or mma for a volume fraction')
    cg_setup['cg_tol'] = 1e-7  # finite-difference Hessian products need accurate gradients

# Initialize periodic unit cell and discretization
my_cell = domain.PeriodicUnitCell(domain_size=domain_size,
                                  problem_type=problem_type)
start_time = time.time()

if MPI.COMM_WORLD.rank == 0:
    print(f'Domain resolution: {number_of_pixels}')
    print('  Rank   Size          Domain       Subdomain        Location')
    print('  ----   ----          ------       ---------        --------')
MPI.COMM_WORLD.Barrier()

discretization = domain.Discretization(cell=my_cell,
                                       nb_of_pixels_global=number_of_pixels,
                                       discretization_type=discretization_type,
                                       element_type=element_type)
print(f'{MPI.COMM_WORLD.rank:6} {MPI.COMM_WORLD.size:6} {str(discretization.fft.nb_domain_grid_pts):>15} '
      f'{str(discretization.fft.nb_subdomain_grid_pts):>15} {str(discretization.fft.subdomain_locations):>15}')

# Base Material Properties
K_0, G_0 = 1.0, 0.5
elastic_C_0 = material_models.get_elastic_material_tensor(dim=discretization.domain_dimension,
                                                 K=K_0,
                                                 mu=G_0,
                                                 kind='linear')
soft_phase = 10 ** (-soft_phase_exponent) if soft_phase_exponent > 0 else 0
elastic_C_void = elastic_C_0 * soft_phase

# Preconditioner setup. Reference material: Green and dual C_0; Green-Jacobi 2 mu I (the Jacobi diagonal carries the
# local magnitude, the Green part only the structure)
green_reference_ijkl = (2 * G_0 * np.einsum('ik,jl->ijkl', np.eye(dim), np.eye(dim))
                        if preconditioner_type == 'Green_Jacobi' else elastic_C_0)
preconditioner_fnfnqks = discretization.get_preconditioner_Green_mugrid(reference_material_data_ijkl=green_reference_ijkl)


def M_fun_Green(x, Px):
    """Green's operator based preconditioner."""
    discretization.fft.communicate_ghosts(x)
    discretization.apply_preconditioner_mugrid(preconditioner_Fourier_fnfnqks=preconditioner_fnfnqks,
                                               input_nodal_field_fnxyz=x,
                                               output_nodal_field_fnxyz=Px)


# Define load cases (macroscopic gradients)
nb_load_cases = 3
macro_gradients = np.zeros([nb_load_cases, dim, dim])
macro_gradients[0] = np.array([[1.0, 0.0], [0.0, 0.0]])
macro_gradients[1] = np.array([[0.0, 0.0], [0.0, 1.0]])
macro_gradients[2] = np.array([[0.0, 0.5], [0.5, 0.0]])

# Left macroscopic gradients for energy calculation
left_macro_gradients = np.zeros([nb_load_cases, dim, dim])
left_macro_gradients[0] = np.array([[0.0, 0.0], [0.0, 1.0]])
left_macro_gradients[1] = np.array([[1.0, 0.0], [0.0, 0.0]])
left_macro_gradients[2] = np.array([[0.0, 0.5], [0.5, 0.0]])

if MPI.COMM_WORLD.rank == 0:
    print(f'Load cases (macro gradients):\n{macro_gradients}')

# Macro gradient field allocation
macro_gradient_field_ijqxyz = discretization.get_gradient_size_field(name='macro_gradient_field')

# Target properties (Auxetic behavior)
poison_target = -0.0
E_0 = 9 * K_0 * G_0 / (3 * K_0 + G_0)
G_target_auxet = (3 / 20) * E_0
E_target = 2 * G_target_auxet * (1 + poison_target)
K_target, G_target = material_models.get_bulk_and_shear_modulus(E=E_target, poisson=poison_target)

elastic_C_target = material_models.get_elastic_material_tensor(dim=discretization.domain_dimension,
                                                      K=K_target,
                                                      mu=G_target,
                                                      kind='linear')

if MPI.COMM_WORLD.rank == 0:
    print(f'Target elastic tangent (Voigt):\n{material_models.compute_Voigt_notation_4order(elastic_C_target)}')

# Target stresses and energies
target_stresses = np.zeros([nb_load_cases, dim, dim])
target_energy = np.zeros([nb_load_cases])

for load_case in range(nb_load_cases):
    target_stresses[load_case] = np.einsum('ijkl,kl->ij', elastic_C_target, macro_gradients[load_case])
    target_energy[load_case] = np.einsum('ij,ijkl,kl->', left_macro_gradients[load_case], elastic_C_target,
                                         macro_gradients[load_case])
    if MPI.COMM_WORLD.rank == 0:
        print(f'Load case {load_case}: target stress = {target_stresses[load_case].tolist()}')

# Optimization state variables
displacement_field_load_case = [discretization.get_unknown_size_field(name=f'u_{i}') for i in range(nb_load_cases)]
adjoint_field_load_case = [discretization.get_unknown_size_field(name=f'adj_{i}') for i in range(nb_load_cases)]

p = 2  # SIMP penalty exponent
double_well_depth_test = 1
norms_sigma, norms_pf, norms_adjoint_energy = [], [], []

info_mech = {'num_iteration_adjoint': [], 'residual_rz': []}
info_adjoint = {'num_iteration_adjoint': [], 'residual_rz': []}

# Allocation of common fields used in the objective function
phase_field_1nxyz = discretization.get_scalar_field(name='phase_field')
phase_field_at_quad_poits_1qxyz = discretization.get_quad_field_scalar(name='phase_field_quad')
material_data_field_C_0_rho_ijklqxyz = discretization.get_material_data_size_field_mugrid(name='mat_data_field')
s_sensitivity_field = discretization.get_scalar_field(name='sensitivity_field')
rhs_load_case_inxyz = discretization.get_unknown_size_field(name='rhs_field')
s_stress_and_adjoint_load_case = discretization.get_scalar_field(name='stress_adj_sensitivity')
dual_material = discretization.get_material_data_size_field_mugrid(name='dual_material')   # C_0 A^-1 C_0 (dual)
dual_temp_1 = discretization.get_unknown_size_field(name='dual_temp_1')
dual_temp_2 = discretization.get_unknown_size_field(name='dual_temp_2')

w = weight / nb_load_cases

if MPI.COMM_WORLD.rank == 0:
    print(f'Penalty p: {p}, Weight w: {w}, Eta: {eta}')


def objective_function_multiple_load_cases(phase_field_1nxyz_flat):
    """
    Main objective function for topology optimization.
    Calculates the combined objective (target stress potential + phase field energy)
    and its sensitivity with respect to the phase field.
    """
    phase_field_1nxyz.s[...] = phase_field_1nxyz_flat.reshape([1, 1, *discretization.nb_of_pixels])

    # 1. Project phase field to quadrature points
    discretization.apply_N_operator_mugrid(phase_field_1nxyz, phase_field_at_quad_poits_1qxyz)

    # 2. Update material properties based on phase field (SIMP interpolation)
    material_data_field_C_0_rho_ijklqxyz.s[...] = (elastic_C_0 - elastic_C_void)[
                                                      ..., np.newaxis, np.newaxis, np.newaxis] * \
                                                  np.power(phase_field_at_quad_poits_1qxyz.s, p)[0, 0, :, ...] + \
                                                  elastic_C_void[..., np.newaxis, np.newaxis, np.newaxis]

    # 3. Calculate phase field contribution to objective and its sensitivity
    f_phase_field = topology_optimization.objective_function_phase_field(discretization=discretization,
                                                                         phase_field_1nxyz=phase_field_1nxyz,
                                                                         eta=eta,
                                                                         double_well_depth=double_well_depth_test)
    s_sensitivity_field.s.fill(0)

    topology_optimization.sensitivity_phase_field_term_FE_NEW(
        discretization=discretization,
        phase_field_1nxyz=phase_field_1nxyz,
        p=p,
        eta=eta,
        output_array=s_sensitivity_field,
        double_well_depth=1)

    objective_function = f_phase_field
    norms_pf.append(objective_function)

    # 4. Configure preconditioner
    if preconditioner_type == 'Green':
        M_fun = M_fun_Green
    elif preconditioner_type == 'Jacobi':
        K_diag_alg = discretization.get_preconditioner_Jacobi_mugrid(
            material_data_field_ijklqxyz=material_data_field_C_0_rho_ijklqxyz)

        def M_fun_Jacobi(x, Px):
            Px.s[...] = K_diag_alg.s * K_diag_alg.s * x.s
            discretization.fft.communicate_ghosts(Px)

        M_fun = M_fun_Jacobi

    elif preconditioner_type == 'Green_Jacobi':
        K_diag_alg = discretization.get_preconditioner_Jacobi_mugrid(
            material_data_field_ijklqxyz=material_data_field_C_0_rho_ijklqxyz)

        def M_fun_Green_Jacobi(x, Px):
            discretization.fft.communicate_ghosts(x)
            # Temporary field for Jacobi scaling
            x_jacobi_temp = discretization.get_unknown_size_field(name='x_jacobi_temp')

            x_jacobi_temp.s[...] = K_diag_alg.s * x.s
            discretization.apply_preconditioner_mugrid(
                preconditioner_Fourier_fnfnqks=preconditioner_fnfnqks,
                input_nodal_field_fnxyz=x_jacobi_temp,
                output_nodal_field_fnxyz=Px)

            Px.s[...] = K_diag_alg.s * Px.s
            discretization.fft.communicate_ghosts(Px)

        M_fun = M_fun_Green_Jacobi

    elif preconditioner_type == 'dual':
        # P = G K(C_0 A^-1 C_0) G. The SIMP material is A = s(x) C_0, s = rho^p (1 - soft_phase) + soft_phase, so the
        # middle material is exactly C_0 A^-1 C_0 = C_0 / s(x)
        dual_material.s[...] = elastic_C_0[..., np.newaxis, np.newaxis, np.newaxis] / (
            np.power(phase_field_at_quad_poits_1qxyz.s, p)[0, 0] * (1 - soft_phase) + soft_phase)

        def M_fun_dual(x, Px):
            discretization.fft.communicate_ghosts(x)
            discretization.apply_preconditioner_mugrid(preconditioner_Fourier_fnfnqks=preconditioner_fnfnqks,
                                                       input_nodal_field_fnxyz=x, output_nodal_field_fnxyz=dual_temp_1)
            discretization.apply_system_matrix_mugrid(material_data_field=dual_material, input_field_inxyz=dual_temp_1,
                                                      output_field_inxyz=dual_temp_2, formulation='small_strain')
            discretization.fft.communicate_ghosts(dual_temp_2)
            discretization.apply_preconditioner_mugrid(preconditioner_Fourier_fnfnqks=preconditioner_fnfnqks,
                                                       input_nodal_field_fnxyz=dual_temp_2, output_nodal_field_fnxyz=Px)
            discretization.fft.communicate_ghosts(Px)

        M_fun = M_fun_dual

    def K_fun(x, Ax):
        discretization.apply_system_matrix_mugrid(material_data_field=material_data_field_C_0_rho_ijklqxyz,
                                                  input_field_inxyz=x,
                                                  output_field_inxyz=Ax,
                                                  formulation='small_strain')

    disp = False
    # 5. Solve mechanical equilibrium and adjoint problems for each load case
    homogenized_stresses = np.zeros([nb_load_cases, dim, dim])

    f_sigmas = np.zeros([nb_load_cases, 1])
    adjoint_energies = np.zeros([nb_load_cases, 1])
    norm_sigma_step = 0
    adjoint_energies_step = 0
    for load_case in np.arange(nb_load_cases):

        discretization.get_macro_gradient_field_mugrid(macro_gradient_ij=macro_gradients[load_case],
                                                       macro_gradient_field_ijqxyz=macro_gradient_field_ijqxyz)

        rhs_load_case_inxyz.s.fill(0)
        discretization.get_rhs_mugrid(
            material_data_field_ijklqxyz=material_data_field_C_0_rho_ijklqxyz,
            macro_gradient_field_ijqxyz=macro_gradient_field_ijqxyz,
            rhs_inxyz=rhs_load_case_inxyz)

        if MPI.COMM_WORLD.rank == 0:
            norms_cg_mech = dict()
            norms_cg_mech['residual_rr'] = []
            norms_cg_mech['residual_rz'] = []

        def callback(it, x, r, p, z, stop_crit_norm):
            norm_of_rr = discretization.communicator.sum(np.dot(r.ravel(), r.ravel()))
            norm_of_rz = discretization.communicator.sum(np.dot(r.ravel(), z.ravel()))
            if MPI.COMM_WORLD.rank == 0:
                norms_cg_mech['residual_rr'].append(norm_of_rr)
                norms_cg_mech['residual_rz'].append(norm_of_rz)

        if cg_mode == 'energy':
            displacement_field_load_case[load_case].s[...] = 0.0
            solvers.conjugate_gradients_mugrid_experimental(
                comm=discretization.communicator,
                fc=discretization.field_collection,
                hessp=K_fun,
                b=rhs_load_case_inxyz,
                x=displacement_field_load_case[load_case],
                P=M_fun,
                tol=cg_setup['cg_tol'],
                maxiter=100000,
                callback=callback,
                stop_crit_norm='energy_lower_estim',
                energy_rtol=True,
                d_min=cg_setup['cg_delay'])
        else:
            solvers.conjugate_gradients_mugrid(
                comm=discretization.communicator,
                fc=discretization.field_collection,
                hessp=K_fun,  # linear operator
                b=rhs_load_case_inxyz,
                x=displacement_field_load_case[load_case],
                P=M_fun,
                tol=cg_setup['cg_tol'],
                maxiter=10000,
                callback=callback,
            )

        if MPI.COMM_WORLD.rank == 0:
            nb_it = len(norms_cg_mech['residual_rr'])
            try:
                norm_rz = norms_cg_mech['residual_rz'][-1]
                norm_rr = norms_cg_mech['residual_rr'][-1]
            except IndexError:
                norm_rz = 0
                norm_rr = 0
            info_mech['num_iteration_adjoint'].append(nb_it)

            print(
                'load case ' f'{load_case},  nb_ steps CG mech    =' f'{nb_it}, residual_rz = {norm_rz}, residual_rr = {norm_rr}')
            del norms_cg_mech

        # compute homogenized stress field corresponding to current displacement
        homogenized_stresses[load_case] = discretization.get_homogenized_stress_mugrid(
            material_data_field_ijklqxyz=material_data_field_C_0_rho_ijklqxyz,
            displacement_field_inxyz=displacement_field_load_case[load_case],
            macro_gradient_field_ijqxyz=macro_gradient_field_ijqxyz,
            formulation='small_strain')

        f_sigmas[load_case] = topology_optimization.compute_stress_equivalence_potential(
            actual_stress_ij=homogenized_stresses[load_case],
            target_stress_ij=target_stresses[load_case])

        s_stress_and_adjoint_load_case.s[0, 0], adjoint_field_load_case[
            load_case], adjoint_energies[
            load_case], info_adjoint_current = topology_optimization.sensitivity_stress_and_adjoint_FE_NEW(
            discretization=discretization,
            base_material_data_ijkl=elastic_C_0,
            void_material_data_ijkl=elastic_C_void,
            displacement_field_inxyz=displacement_field_load_case[load_case],
            adjoint_field_inxyz=adjoint_field_load_case[load_case],
            macro_gradient_field_ijqxyz=macro_gradient_field_ijqxyz,
            phase_field_1nxyz=phase_field_1nxyz,
            target_stress_ij=target_stresses[load_case],
            actual_stress_ij=homogenized_stresses[load_case],
            preconditioner_fun=M_fun,
            system_matrix_fun=K_fun,
            formulation='small_strain',
            p=p,
            weight=w,
            disp=disp,
            **cg_setup)

        s_sensitivity_field.s[0, 0] += s_stress_and_adjoint_load_case.s[0, 0]

        objective_function += w * f_sigmas[load_case]
        objective_function += adjoint_energies[load_case]
        if MPI.COMM_WORLD.rank == 0:
            info_adjoint['num_iteration_adjoint'].append(info_adjoint_current['num_iteration_adjoint'])
            nb_it_adjoint = info_adjoint_current['num_iteration_adjoint']
            print(
                'load case ' f'{load_case},  nb_steps CG adjoint  =' f'{nb_it_adjoint} ')
            if disp:
                print(
                    'load case ' f'{load_case},  f_sigmas =' f'{f_sigmas[load_case]}')
                print(
                    'load case ' f'{load_case},  objective_function =' f'{objective_function}')
        norm_sigma_step += f_sigmas[load_case]
        adjoint_energies_step += adjoint_energies[load_case]
    discretization.fft.communicate_ghosts(s_sensitivity_field)
    norms_sigma.append(norm_sigma_step)
    norms_adjoint_energy.append(adjoint_energies_step)

    return objective_function[0], s_sensitivity_field.s[0, 0].reshape(-1)


if __name__ == '__main__':
    script_name = os.path.splitext(os.path.basename(__file__))[0]
    file_folder_path = os.path.dirname(os.path.realpath(__file__))
    data_folder_path = os.path.join(file_folder_path, 'data', script_name) + '/'
    figure_folder_path = os.path.join(file_folder_path, 'figures', script_name) + '/'

    random_init = False
    save_history = True  # store the phase field in <output_name>_history.nc ...
    history_every = int(os.environ.get('HISTORY_EVERY', '1'))  # ... of every HISTORY_EVERY-th iteration

    if MPI.COMM_WORLD.rank == 0:
        os.makedirs(data_folder_path, exist_ok=True)
        os.makedirs(figure_folder_path, exist_ok=True)
    MPI.COMM_WORLD.Barrier()

    # Output files (NetCDF, see docs/io.md): <output_name>_history.nc and <output_name>_final.nc
    output_name = data_folder_path + (f'N{n_pix}_soft{soft_phase:g}_{cg_mode}_' if (n_pix != 64 or soft_phase_exponent != 5
                                                                              or cg_mode != 'residual') else '') \
        + f'{preconditioner_type}_eta_{eta}_w_{weight}_{optimizer}' + (
        f'_vol_{volume_fraction}' if volume_fraction is not None else '') + (
        f'_noise{init_noise_grid}' if init_noise_grid else '')
    parameters = {'nb_of_pixels': number_of_pixels, 'domain_size': domain_size, 'element_type': element_type,
                  'preconditioner_type': preconditioner_type, 'eta': eta, 'weight': weight, 'p': p,
                  'soft_phase': soft_phase, 'optimizer': optimizer, 'cg_mode': cg_mode,
                  'volume_fraction': volume_fraction if volume_fraction is not None else -1.0,
                  'target_C_voigt': material_models.compute_Voigt_notation_4order(elastic_C_target)}


    def apply_filter(phase):
        """Applies a simple frequency filter to the phase field."""
        f_field = discretization.ffield_collection.complex_field(
            name='f_field_phase_for_filter',
            components=(1,))
        discretization.fft.fft(phase, f_field)
        f_field.s[0, 0, np.logical_or(discretization.fft.ifftfreq[0] > 8,
                                      discretization.fft.ifftfreq[1] > 8)] = 0
        discretization.fft.ifft(f_field, phase)
        phase.s[...] *= discretization.fft.normalisation
        phase.s[phase.s > 1] = 1
        phase.s[phase.s < 0] = 0


    phase_field_0 = discretization.get_scalar_field(name='phase_field_initial')

    if random_init:
        phase_field_0.s[...] += np.random.rand(*phase_field_0.s.shape)
    else:
        np.random.seed(MPI.COMM_WORLD.rank)
        coords = discretization.fft.coords
        phase_field_0.s[0, 0] = (np.sin(coords[0] * 4 * np.pi) + np.sin(coords[1] * 4 * np.pi) + 2) / 4
        noise_grid = init_noise_grid
        if noise_grid:  # the noise of a serial run on noise_grid^2, pixel-replicated: same start on every grid / MPI split
            np.random.seed(0)
            factor = number_of_pixels[0] // noise_grid
            noise = np.kron(np.random.rand(1, 1, noise_grid, noise_grid), np.ones((factor, factor)))
            (x0, y0), (nx, ny) = discretization.fft.subdomain_locations, discretization.fft.nb_subdomain_grid_pts
            phase_field_0.s[0, 0] += 0.5 * noise[0, 0, x0:x0 + nx, y0:y0 + ny]
        else:
            phase_field_0.s[...] += 0.5 * np.random.rand(*phase_field_0.s.shape)

    iterat = 0
    phase_field_iterate = discretization.get_scalar_field(name='phase_field_iterate')
    history_writer = io_utils.FieldWriter(output_name + '_history.nc', [phase_field_iterate],
                                          attributes=parameters,
                                          frame_variables={'objective': (), 'volume_fraction': (), 'nb_evaluations': ()}
                                          ) if save_history else None
    evaluations = {'count': 0, 'last_f': np.nan}


    def counted_objective(phase_field_flat):
        """Objective with a counter of evaluations (each one = FE solves of all load cases)."""
        f, gradient = objective_function_multiple_load_cases(phase_field_flat)
        evaluations['count'] += 1
        evaluations['last_f'] = f
        return f, gradient


    def fd_hessian_product(phase_field_flat, direction):
        """Hessian-vector product by central differences of the adjoint gradient (2 evaluations)."""
        max_direction = MPI.COMM_WORLD.allreduce(float(np.max(np.abs(direction))), op=MPI.MAX)
        if max_direction == 0:
            return np.zeros_like(direction)
        step = fd_step / max_direction  # largest change of the phase field = fd_step
        _, gradient_plus = objective_function_multiple_load_cases(phase_field_flat + step * direction)
        _, gradient_minus = objective_function_multiple_load_cases(phase_field_flat - step * direction)
        evaluations['count'] += 2
        return (gradient_plus - gradient_minus) / (2 * step)


    def my_callback(x_current):
        """Callback to store and visualize progress during optimization."""
        global iterat
        iterat += 1
        if history_writer is not None and (iterat - 1) % history_every == 0:
            phase_field_iterate.s[0, 0] = x_current.reshape(discretization.nb_of_pixels)
            # the optimizers evaluate the accepted iterate last, so last_f is f(x_current)
            history_writer.write(objective=evaluations['last_f'],
                                 volume_fraction=MPI.COMM_WORLD.allreduce(np.sum(x_current)) / np.prod(
                                     number_of_pixels),
                                 nb_evaluations=evaluations['count'])
        if MPI.COMM_WORLD.size == 1 and show_plots:
            import matplotlib as mpl
            import matplotlib.pyplot as plt
            plt.figure()
            plt.pcolormesh(discretization.fft.coords[0],
                           discretization.fft.coords[1],
                           x_current.reshape(discretization.nb_of_pixels),
                           cmap=mpl.cm.Greys)
            #plt.clim(0, 1)
            plt.title(f'Iteration {iterat}')
            plt.colorbar()
            plt.show()


    # Run optimization
    # volume weights: <a, phase> = mean(phase) = volume fraction (a is the local slice under MPI)
    a = np.full(phase_field_0.s.size, 1.0 / np.prod(number_of_pixels))
    # stop when max |gradient| < gtol. The gradient w.r.t. the nodal phase values scales with the pixel area ~ 1/N^2,
    # so the 64^2 value 1e-3 is scaled: the same criterion on every grid (unscaled, 128^2 stopped after 44 evaluations)
    gtol = 1e-3 * (64 / n_pix) ** 2
    # GTOL overrides it, e.g. GTOL=0: stop only on the step test (xtol: largest change of the phase < 1e-3) or maxiter
    # -- the max-norm of the gradient does not scale exactly with h^2 (scaled, 128^2 still stopped at 52 evaluations)
    gtol = float(os.environ.get('GTOL', gtol))
    optimizer_options = {'l_bfgs': dict(xtol=1e-3, maxiter=500, maxcor=20, c1=1e-4, max_halvings=40),
                         'mma': dict(xtol=1e-3, maxiter=500),
                         'tr_newton': dict(maxiter=200)}[optimizer]
    xopt_FE_MPI = solvers_nonlinear.minimize(
        fun=counted_objective,
        x0=phase_field_0.s.ravel(),
        method=optimizer,
        bounds=(0., 1.),
        constraint_weights=a if volume_fraction is not None else None,
        constraint_target=volume_fraction,
        constraint_type='ineq' if optimizer == 'mma' else 'eq',
        hessp=fd_hessian_product if optimizer == 'tr_newton' else None,
        gtol=gtol,
        comm=MPI.COMM_WORLD,
        callback=my_callback,
        disp=True,
        **optimizer_options,
    )
    if history_writer is not None:
        history_writer.close()

    solution_phase = discretization.get_scalar_field(name='phase_field_solution')
    solution_phase.s[...] = xopt_FE_MPI.x.reshape([1, 1, *discretization.nb_of_pixels])

    ######## Postprocess for FE linear solver with NuMPI ########
    solution_phase_at_quad_poits_1qxyz = discretization.get_quad_field_scalar(
        name='solution_phase_at_quad_poits_1qxyz')
    discretization.apply_N_operator_mugrid(solution_phase, solution_phase_at_quad_poits_1qxyz)

    material_data_field_C_0_rho_quad = discretization.get_material_data_size_field_mugrid(
        name='material_data_field_C_0_rho_quad')
    material_data_field_C_0_rho_quad.s[...] = (elastic_C_0 - elastic_C_void)[..., np.newaxis, np.newaxis, np.newaxis] * \
                                              np.power(solution_phase_at_quad_poits_1qxyz.s, p)[0, 0, :, ...] + \
                                              elastic_C_void[..., np.newaxis, np.newaxis, np.newaxis]

    homogenized_stresses = np.zeros([nb_load_cases, dim, dim])

    for load_case in np.arange(nb_load_cases):
        # Solve mechanical equilibrium constraint
        rhs_field_final = discretization.get_unknown_size_field(name='rhs_field_final')
        rhs_field_final.s.fill(0)

        discretization.get_macro_gradient_field_mugrid(macro_gradient_ij=macro_gradients[load_case],
                                                       macro_gradient_field_ijqxyz=macro_gradient_field_ijqxyz)

        discretization.get_rhs_mugrid(
            material_data_field_ijklqxyz=material_data_field_C_0_rho_quad,
            macro_gradient_field_ijqxyz=macro_gradient_field_ijqxyz,
            rhs_inxyz=rhs_field_final)


        def K_fun_final(x, Ax):
            discretization.apply_system_matrix_mugrid(material_data_field=material_data_field_C_0_rho_quad,
                                                      input_field_inxyz=x,
                                                      output_field_inxyz=Ax,
                                                      formulation='small_strain')


        M_fun = M_fun_Green

        displacement_field = discretization.get_unknown_size_field(name='displacement_field_postprocess')

        solvers.conjugate_gradients_mugrid(
            comm=discretization.communicator,
            fc=discretization.field_collection,
            hessp=K_fun_final,  # linear operator
            b=rhs_field_final,
            x=displacement_field,
            P=M_fun,
            tol=1e-5,
            maxiter=10000,
        )

        # compute homogenized stress field corresponding to final displacement
        homogenized_stresses[load_case] = discretization.get_homogenized_stress_mugrid(
            material_data_field_ijklqxyz=material_data_field_C_0_rho_quad,
            displacement_field_inxyz=displacement_field,
            macro_gradient_field_ijqxyz=macro_gradient_field_ijqxyz,
            formulation='small_strain')

        if MPI.COMM_WORLD.rank == 0:
            print(f'target_stresses[load_case {load_case}] = {target_stresses[load_case]}')
            print(f'homogenized_stresses[load_case {load_case}]= {homogenized_stresses[load_case]}')

    homogenized_C_ijkl = np.zeros(np.array(4 * [dim, ]))
    macro_gradient_field_ijqxyz = discretization.get_gradient_size_field(name=f'macro_gradient_field_ij')
    rhs_field_final = discretization.get_unknown_size_field(name='rhs_field_final')

    # compute whole homogenized elastic tangent
    for i in range(2):
        for j in range(2):
            macro_gradient_ij = np.zeros([dim, dim])
            macro_gradient_ij[i, j] = 1

            discretization.get_macro_gradient_field_mugrid(macro_gradient_ij=macro_gradient_ij,
                                                           macro_gradient_field_ijqxyz=macro_gradient_field_ijqxyz)

            discretization.get_rhs_mugrid(material_data_field_ijklqxyz=material_data_field_C_0_rho_quad,
                                          macro_gradient_field_ijqxyz=macro_gradient_field_ijqxyz,
                                          rhs_inxyz=rhs_field_final)

            solvers.conjugate_gradients_mugrid(
                comm=discretization.communicator,
                fc=discretization.field_collection,
                hessp=K_fun_final,  # linear operator
                b=rhs_field_final,
                x=displacement_field,
                P=M_fun,
                tol=1e-5,
                maxiter=10000,
            )

            homogenized_C_ijkl[:, :, i, j] = discretization.get_homogenized_stress_mugrid(
                material_data_field_ijklqxyz=material_data_field_C_0_rho_quad,
                displacement_field_inxyz=displacement_field,
                macro_gradient_field_ijqxyz=macro_gradient_field_ijqxyz,
                formulation='small_strain')
    if MPI.COMM_WORLD.rank == 0:
        print('Optimized elastic tangent =  :\n' +
              np.array2string(material_models.compute_Voigt_notation_4order(homogenized_C_ijkl),
                              formatter={'float_kind': lambda x: f"{x:0.5f}"}))
        print(f'Target elastic tangent (Voigt):\n' +
              np.array2string(material_models.compute_Voigt_notation_4order(elastic_C_target),
                              formatter={'float_kind': lambda x: f"{x:0.5f}"}))

    # Save the optimized phase field with the run parameters and the optimization log
    # (attributes are stored flattened; the per-evaluation lists have one entry per objective call)
    io_utils.save_fields(output_name + '_final.nc', [solution_phase], attributes={
        **parameters,
        'nb_iterations': iterat,
        'elapsed_time': time.time() - start_time,
        'homogenized_C_voigt': material_models.compute_Voigt_notation_4order(homogenized_C_ijkl),
        'target_stresses': target_stresses,
        'homogenized_stresses': homogenized_stresses,
        'norms_sigma': norms_sigma,
        'norms_pf': norms_pf,
        'norms_adjoint_energy': norms_adjoint_energy,
        'num_iteration_mech': MPI.COMM_WORLD.bcast(info_mech['num_iteration_adjoint']),  # filled on rank 0
        'num_iteration_adjoint': MPI.COMM_WORLD.bcast(info_adjoint['num_iteration_adjoint'])})
    if MPI.COMM_WORLD.rank == 0:
        print(f'Data saved to: {output_name}_final.nc' + (' and _history.nc' if save_history else ''))
