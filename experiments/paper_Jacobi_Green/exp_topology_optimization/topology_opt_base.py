import numpy as np
import time
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))


import argparse
import matplotlib.pyplot as plt
import matplotlib as mpl

from NuMPI import Optimization
from NuMPI.IO import save_npy
from mpi4py import MPI
from muFFTTO import domain
from muFFTTO import solvers
from muFFTTO import topology_optimization


def run_topology_optimization(
        nb_pixels,
        cg_tol_exponent,
        soft_phase_exponent,
        preconditioner_type,
        eta,
        weight,
        poison_target,
        K_0,
        G_0,
        maxiter=1000,
        random_init=False,
        save_results=True,
        data_folder_path=None,
        figure_folder_path=None
):
    # Problem Configuration
    problem_type = 'elasticity'
    discretization_type = 'finite_element'
    element_type = 'linear_triangles'
    formulation = 'small_strain'

    # Domain and Discretization
    domain_size = [1, 1]
    number_of_pixels = (nb_pixels, nb_pixels)
    dim = np.size(number_of_pixels)



    # Optimization Parameters
    cg_setup = {'cg_tol': 10 ** (-cg_tol_exponent),
                'cg_max_it': 10000}

    # Initialize periodic unit cell and discretization
    my_cell = domain.PeriodicUnitCell(domain_size=domain_size,
                                      problem_type=problem_type)

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
    elastic_C_0 = domain.get_elastic_material_tensor(dim=discretization.domain_dimension,
                                                     K=K_0,
                                                     mu=G_0,
                                                     kind='linear')

    soft_phase = 10 ** (-soft_phase_exponent) if soft_phase_exponent > 0 else 0
    elastic_C_void = elastic_C_0 * soft_phase

    # Preconditioner setup
    preconditioner_fnfnqks = discretization.get_preconditioner_Green_mugrid(reference_material_data_ijkl=elastic_C_0)

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

    if MPI.COMM_WORLD.rank == 0:
        print(f'Load cases (macro gradients):\n{macro_gradients}')

    # Macro gradient field allocation
    macro_gradient_field_ijqxyz = discretization.get_gradient_size_field(name='macro_gradient_field')

    # Target properties (Auxetic behavior)
    E_0 = 9 * K_0 * G_0 / (3 * K_0 + G_0)
    G_target_auxet = (3 / 20) * E_0
    E_target = 2 * G_target_auxet * (1 + poison_target)
    K_target, G_target = domain.get_bulk_and_shear_modulus(E=E_target, poison=poison_target)

    elastic_C_target = domain.get_elastic_material_tensor(dim=discretization.domain_dimension,
                                                          K=K_target,
                                                          mu=G_target,
                                                          kind='linear')

    if MPI.COMM_WORLD.rank == 0:
        print(f'Target elastic tangent (Voigt):\n{domain.compute_Voigt_notation_4order(elastic_C_target)}')

    # Target stresses and energies
    target_stresses = np.zeros([nb_load_cases, dim, dim])

    for load_case in range(nb_load_cases):
        target_stresses[load_case] = np.einsum('ijkl,lk->ij', elastic_C_target, macro_gradients[load_case])

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

    w = weight / nb_load_cases

    if MPI.COMM_WORLD.rank == 0:
        print(f'Penalty p: {p}, Weight w: {w}, Eta: {eta}')

    def objective_function_multiple_load_cases(phase_field_1nxyz_flat):
        phase_field_1nxyz.s[...] = phase_field_1nxyz_flat.reshape([1, 1, *discretization.nb_of_pixels])
        discretization.apply_N_operator_mugrid(phase_field_1nxyz, phase_field_at_quad_poits_1qxyz)
        material_data_field_C_0_rho_ijklqxyz.s[...] = (elastic_C_0 - elastic_C_void)[
                                                          ..., np.newaxis, np.newaxis, np.newaxis] * \
                                                      np.power(phase_field_at_quad_poits_1qxyz.s, p)[0, 0, :, ...] + \
                                                      elastic_C_void[..., np.newaxis, np.newaxis, np.newaxis]

        f_phase_field = topology_optimization.objective_function_phase_field(discretization=discretization,
                                                                             phase_field_1nxyz=phase_field_1nxyz,
                                                                             eta=eta,
                                                                             double_well_depth=double_well_depth_test)
        s_sensitivity_field.s.fill(0)
        topology_optimization.sensitivity_phase_field_term_FE_NEW(
            discretization=discretization,
            base_material_data_ijkl=elastic_C_0,
            void_material_data_ijkl=elastic_C_void,
            phase_field_1nxyz=phase_field_1nxyz,
            p=p,
            eta=eta,
            output_array=s_sensitivity_field,
            double_well_depth=1)

        objective_val = f_phase_field
        norms_pf.append(objective_val)

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
                x_jacobi_temp = discretization.get_unknown_size_field(name='x_jacobi_temp')
                x_jacobi_temp.s[...] = K_diag_alg.s * x.s
                discretization.apply_preconditioner_mugrid(
                    preconditioner_Fourier_fnfnqks=preconditioner_fnfnqks,
                    input_nodal_field_fnxyz=x_jacobi_temp,
                    output_nodal_field_fnxyz=Px)
                Px.s[...] = K_diag_alg.s * Px.s
                discretization.fft.communicate_ghosts(Px)

            M_fun = M_fun_Green_Jacobi
        else:
            M_fun = M_fun_Green

        def K_fun(x, Ax):
            discretization.apply_system_matrix_mugrid(material_data_field=material_data_field_C_0_rho_ijklqxyz,
                                                      input_field_inxyz=x,
                                                      output_field_inxyz=Ax,
                                                      formulation='small_strain')

        homogenized_stresses = np.zeros([nb_load_cases, dim, dim])
        f_sigmas = np.zeros([nb_load_cases, 1])
        adjoint_energies = np.zeros([nb_load_cases, 1])
        norm_sigma_step = 0
        adjoint_energies_step = 0

        for load_case in range(nb_load_cases):
            discretization.get_macro_gradient_field_mugrid(macro_gradient_ij=macro_gradients[load_case],
                                                           macro_gradient_field_ijqxyz=macro_gradient_field_ijqxyz)
            rhs_load_case_inxyz.s.fill(0)
            discretization.get_rhs_mugrid(
                material_data_field_ijklqxyz=material_data_field_C_0_rho_ijklqxyz,
                macro_gradient_field_ijqxyz=macro_gradient_field_ijqxyz,
                rhs_inxyz=rhs_load_case_inxyz)

            if MPI.COMM_WORLD.rank == 0:
                norms_cg_mech = {'residual_rr': [], 'residual_rz': []}

            def callback(it, x, r, p, z, stop_crit_norm):
                norm_of_rr = discretization.communicator.sum(np.dot(r.ravel(), r.ravel()))
                norm_of_rz = discretization.communicator.sum(np.dot(r.ravel(), z.ravel()))
                if MPI.COMM_WORLD.rank == 0:
                    norms_cg_mech['residual_rr'].append(norm_of_rr)
                    norms_cg_mech['residual_rz'].append(norm_of_rz)

            #displacement_field_load_case[load_case].s.fill(0)
            solvers.conjugate_gradients_mugrid(
                comm=discretization.communicator,
                fc=discretization.field_collection,
                hessp=K_fun,
                b=rhs_load_case_inxyz,
                x=displacement_field_load_case[load_case],
                P=M_fun,
                rtol= False,
                tol=cg_setup['cg_tol'],
                maxiter=cg_setup['cg_max_it'],
                callback=callback)

            if MPI.COMM_WORLD.rank == 0:
                nb_it = len(norms_cg_mech['residual_rr'])
                norm_rz = norms_cg_mech['residual_rz'][-1] if nb_it > 0 else 0
                norm_rr = norms_cg_mech['residual_rr'][-1] if nb_it > 0 else 0
                info_mech['num_iteration_adjoint'].append(nb_it)
                print(
                    f'load case {load_case}, nb_ steps CG mech = {nb_it}, residual_rz = {norm_rz}, residual_rr = {norm_rr}')

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
                disp=False,
                **cg_setup)

            s_sensitivity_field.s[0, 0] += s_stress_and_adjoint_load_case.s[0, 0]
            objective_val += w * f_sigmas[load_case]
            objective_val += adjoint_energies[load_case]
            if MPI.COMM_WORLD.rank == 0:
                info_adjoint['num_iteration_adjoint'].append(info_adjoint_current['num_iteration_adjoint'])
                print(f'load case {load_case}, nb_steps CG adjoint = {info_adjoint_current["num_iteration_adjoint"]}')
            norm_sigma_step += f_sigmas[load_case]
            adjoint_energies_step += adjoint_energies[load_case]

        discretization.fft.communicate_ghosts(s_sensitivity_field)
        norms_sigma.append(norm_sigma_step)
        norms_adjoint_energy.append(adjoint_energies_step)
        return objective_val[0], s_sensitivity_field.s[0, 0].reshape(-1)

    # Initial phase field
    phase_field_0 = discretization.get_scalar_field(name='phase_field_initial')

    # Use rank-dependent seed for reproducibility in parallel
    np.random.seed(42 + MPI.COMM_WORLD.rank)

    if random_init:
        phase_field_0.s[...] += np.random.rand(*phase_field_0.s.shape)
    else:
        coords = discretization.fft.coords
        phase_field_0.s[0, 0] = (np.sin(coords[0] * 4 * np.pi) + np.sin(coords[1] * 4 * np.pi) + 2) / 4
        phase_field_0.s[...] += 0.5 * np.random.rand(*phase_field_0.s.shape)

    global_iterat = [0]

    def my_callback(x_current):
        global_iterat[0] += 1
        # No plotting in parallel callback to avoid hanging or issues with display
        if MPI.COMM_WORLD.size == 1:
            plt.figure()
            plt.pcolormesh(discretization.fft.coords[0],
                           discretization.fft.coords[1],
                           x_current.reshape(discretization.nb_of_pixels),
                           cmap=mpl.cm.Greys)
            plt.clim(0, 1)
            plt.title(f'Iteration {global_iterat[0]}')
            plt.colorbar()
            plt.show()
        elif MPI.COMM_WORLD.rank == 0:
            print(f"BFGS Iteration {global_iterat[0]} completed.")

    # Optimization
    xopt_FE_MPI = Optimization.l_bfgs(fun=objective_function_multiple_load_cases,
                                      x=phase_field_0.s.ravel(),
                                      jac=True,
                                      maxcor=20,
                                      gtol=1e-3,
                                      ftol=1e-5,
                                      maxiter=maxiter,
                                      comm=MPI.COMM_WORLD,
                                      disp=True,
                                      callback=my_callback
                                      )

    solution_phase = discretization.get_scalar_field(name='phase_field_solution')
    solution_phase.s[...] = xopt_FE_MPI.x.reshape([1, 1, *discretization.nb_of_pixels])

    _info = {}
    if MPI.COMM_WORLD.rank == 0:
        _info["num_iteration_mech"] = np.array(info_mech["num_iteration_adjoint"], dtype=object)
        _info["num_iteration_adjoint"] = np.array(info_adjoint["num_iteration_adjoint"], dtype=object)
        _info['nb_of_pixels'] = discretization.nb_of_pixels_global
        _info['norms_sigma'] = norms_sigma
        _info['norms_pf'] = norms_pf
        _info['norms_adjoint_energy'] = norms_adjoint_energy
        _info['nb_iterations'] = global_iterat[0]

    # Save results
    if save_results and data_folder_path:
        if MPI.COMM_WORLD.rank == 0:
            os.makedirs(data_folder_path, exist_ok=True)
        MPI.COMM_WORLD.Barrier()
        file_data_name = f'_eta_{eta}' + f'_w_{weight}' + f'_final'
        save_npy(os.path.join(data_folder_path, f'{preconditioner_type}{file_data_name}.npy'),
                 solution_phase.s[0].mean(axis=0),
                 tuple(discretization.fft.subdomain_locations),
                 tuple(discretization.nb_of_pixels_global), MPI.COMM_WORLD)
        if MPI.COMM_WORLD.rank == 0:
            print(f"Data saved to: {data_folder_path}")
            np.savez(os.path.join(data_folder_path, f'{preconditioner_type}_eta_{eta}_w_{weight}_log.npz'), **_info)

    return solution_phase, _info


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Base script for topology optimization experiments")
    parser.add_argument("-n", "--nb_pixels", type=int, default=16)
    parser.add_argument("-cg_tol", "--cg_tol_exponent", type=int, default=8)
    parser.add_argument("-soft", "--soft_phase_exponent", type=int, default=5)
    parser.add_argument("-eta", "--eta_parameter", type=float, default=0.01)
    parser.add_argument("-w", "--weight_parameter", type=float, default=5.)
    parser.add_argument("-poisson", "--target_poisson", type=float, default=-0.5)
    parser.add_argument("-K_0", "--base_bulk_modulus", type=float, default=1.)
    parser.add_argument("-G_0", "--base_shear_modulus", type=float, default=0.5)
    parser.add_argument("-p", "--preconditioner_type", type=str, choices=["Green", "Jacobi", "Green_Jacobi"],
                        default="Green_Jacobi")
    parser.add_argument("-r", "--random_init", action="store_true")
    parser.add_argument("-maxit", "--max_iterations", type=int, default=1000)
    parser.add_argument("--save_results", action="store_true", default=True, help="Save results to disk")
    parser.add_argument("--no_save", action="store_false", dest="save_results", help="Do not save results to disk")

    args = parser.parse_args()

    script_name = os.path.splitext(os.path.basename(__file__))[0]
    file_folder_path = os.path.dirname(os.path.realpath(__file__))
    data_folder_path = os.path.join(file_folder_path, 'data', script_name)
    figure_folder_path = os.path.join(file_folder_path, 'figures', script_name)

    run_topology_optimization(
        nb_pixels=args.nb_pixels,
        cg_tol_exponent=args.cg_tol_exponent,
        soft_phase_exponent=args.soft_phase_exponent,
        preconditioner_type=args.preconditioner_type,
        eta=args.eta_parameter,
        weight=args.weight_parameter,
        poison_target=args.target_poisson,
        K_0=args.base_bulk_modulus,
        G_0=args.base_shear_modulus,
        random_init=args.random_init,
        save_results=args.save_results,
        data_folder_path=data_folder_path,
        figure_folder_path=figure_folder_path,
        maxiter=args.max_iterations
    )
