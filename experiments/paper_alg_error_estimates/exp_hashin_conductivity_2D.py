import numpy as np
import scipy as sp
import matplotlib as mpl
import matplotlib.pyplot as plt
import time
import os

# from netCDF4 import Dataset
from NuMPI.IO import save_npy, load_npy
from mpi4py import MPI

from muFFTTO import domain
from muFFTTO import solvers
from muFFTTO import microstructure_library

script_name = os.path.splitext(os.path.basename(__file__))[0]
file_folder_path = os.path.dirname(os.path.realpath(__file__))  # script directory
data_folder_path = file_folder_path + '/exp_data/' + script_name + '/'
figure_folder_path = file_folder_path + '/figures/' + script_name + '/'
if not os.path.exists(file_folder_path):
    os.makedirs(file_folder_path)
if not os.path.exists(data_folder_path):
    os.makedirs(data_folder_path)
if not os.path.exists(figure_folder_path):
    os.makedirs(figure_folder_path)


problem_type = 'conductivity'
discretization_type = 'finite_element'
element_type = 'linear_triangles'
geometry_ID = 'hashin_inclusion_2D'

domain_size = [1, 1]
dim = len(domain_size)




grids_sizes =[3, 4, 5,6,7]  # [8,9,10 ]# [3, 4, 5,6,7]  # [3, 4, 5, 6, 7, 8, 9]  # [8,9,10,11], 10
rhos = [-3, -2, -1, 1, 2, 3]#[1, 3]  # [-3, -2, -1, 1, 2, 3]
for anisotropy in [False, True]:  # ,True
    for rho in rhos:
        for n in grids_sizes:
            number_of_pixels = (2 ** n, 2 ** n)

            my_cell = domain.PeriodicUnitCell(domain_size=domain_size,
                                              problem_type=problem_type)

            discretization = domain.Discretization(cell=my_cell,
                                                   nb_of_pixels_global=number_of_pixels,
                                                   discretization_type=discretization_type,
                                                   element_type=element_type)
            start_time = time.time()

            # set macroscopic gradient
            macro_gradient = np.array([1.0, .0])

            # create material data field
            mat_contrast_1 = 10 ** rho  # inclusion
            mat_contrast_2 = 1.  # coating
            if anisotropy:
                a_ani = 10
            else:
                a_ani = 1

            conductivity_C_1 = mat_contrast_1 * np.array([[a_ani, 0], [0, 1.0]])
            conductivity_C_2 = mat_contrast_2 * np.array([[a_ani, 0], [0, 1.0]])

            conductivity_C_ref = np.array([[1., 0], [0, 1.0]])

            eigen_C1 = sp.linalg.eigh(a=conductivity_C_1, b=conductivity_C_ref, eigvals_only=True)
            eigen_C2 = sp.linalg.eigh(a=conductivity_C_2, b=conductivity_C_ref, eigvals_only=True)
            eigen_LB = np.min([eigen_C1, eigen_C2])
            # seigen_LB *=0.9
            eigen_UB = np.max([eigen_C1, eigen_C2])
            print(f'eigen_LB = {eigen_LB}')
            print(f'eigen_UB = {eigen_UB}')
            total_phase_contrast = eigen_UB / eigen_LB

            # Analytical solution
            r1 = 0.2
            r2 = 0.4
            center = 0.5

            ϕ = (r1 / r2) ** dim
            α = (mat_contrast_2 - mat_contrast_1) / ((dim - 1) * mat_contrast_2 + mat_contrast_1)

            Chom = mat_contrast_2 * (1 - dim * α * ϕ / (1 + α * ϕ))
            conductivity_C_0 = Chom * np.array([[1., 0], [0, 1.0]])
            A_eff = Chom
            print("J_eff : ", A_eff)

            material_data_field = discretization.get_material_data_size_field_mugrid(name='material_data_field')
            material_data_field_C_1 = discretization.get_material_data_size_field_mugrid(name='material_data_field_C_1')
            material_data_field_C_2 = discretization.get_material_data_size_field_mugrid(name='material_data_field_C_2')

            material_data_field.s[...] = conductivity_C_0[:, :, np.newaxis, np.newaxis, np.newaxis]
            material_data_field_C_1.s[...] = conductivity_C_1[:, :, np.newaxis, np.newaxis, np.newaxis]

            material_data_field_C_2.s[...] = conductivity_C_2[:, :, np.newaxis, np.newaxis, np.newaxis]

            # material distribution
            pars_geometry = {'rad_1': r1,
                             'rad_2': r2,
                             'center': center}

            coordinates = discretization.fft.coords

            r_center = coordinates - center
            squares = 0
            squares += sum(r_center[d] ** 2 for d in range(dim))
            distances = np.sqrt(squares)
            # phase_field[np.logical_and(np.logical_and(coordinates[0] < 0.75, coordinates[1] < 0.75),
            #                                        np.logical_and(coordinates[0] >= 0.25, coordinates[1] >= 0.25))] = 0
            #

            # apply material distribution

            material_data_field.s[..., :, distances < r2] = material_data_field_C_2.s[..., distances < r2]
            material_data_field.s[..., distances < r1] = material_data_field_C_1.s[..., distances < r1]

            # Set up the equilibrium system
            macro_gradient_field = discretization.get_gradient_size_field(name=f'macro_gradient_field')
            discretization.get_macro_gradient_field_mugrid(macro_gradient_ij=macro_gradient,
                                                           macro_gradient_field_ijqxyz=macro_gradient_field)

            # Solve mechanical equilibrium constrain
            rhs_load_case_inxyz = discretization.get_unknown_size_field(name='rhs_field_at_load_case')
            discretization.get_rhs_mugrid(
                material_data_field_ijklqxyz=material_data_field,
                macro_gradient_field_ijqxyz=macro_gradient_field,
                rhs_inxyz=rhs_load_case_inxyz)

            def K_fun(x, Ax):
                discretization.apply_system_matrix_mugrid(material_data_field=material_data_field,
                                                          input_field_inxyz=x,
                                                          output_field_inxyz=Ax )
            # M_fun = lambda x: 1 * x

            preconditioner_fnfnqks = discretization.get_preconditioner_Green_mugrid(
                reference_material_data_ijkl=conductivity_C_ref)

            def M_fun_Green(x, Px):
                """
                Function to compute the product of the Preconditioner matrix with a vector.
                The Preconditioner is represented by the convolution operator.
                """
                discretization.fft.communicate_ghosts(x)
                discretization.apply_preconditioner_mugrid(preconditioner_Fourier_fnfnqks=preconditioner_fnfnqks,
                                                           input_nodal_field_fnxyz=x,
                                                           output_nodal_field_fnxyz=Px)




            norms_cg_mech = dict()
            norms_cg_mech['residual_rr'] = []
            norms_cg_mech['residual_rz'] = []
            def callback(it, x, r, p, z, stop_crit_norm):
                # global norms_cg_mech
                norm_of_rr = discretization.communicator.sum(np.dot(r.ravel(), r.ravel()))
                norm_of_rz = discretization.communicator.sum(np.dot(r.ravel(), z.ravel()))
                norms_cg_mech['residual_rr'].append(norm_of_rr)
                norms_cg_mech['residual_rz'].append(norm_of_rz)

            temperatute_field_precise = discretization.get_unknown_size_field(name=f'temperatute_field_precise')
            solvers.conjugate_gradients_mugrid(
                comm=discretization.communicator,
                fc=discretization.field_collection,
                hessp=K_fun,  # linear operator
                b=rhs_load_case_inxyz,
                x=temperatute_field_precise,
                P=M_fun_Green,
                tol=1e-14,
                maxiter=10000,
                callback=callback,
                # norm_metric=res_norm
            )



            error_in_Aeff_hk = []
            Aeff_hk = []
            # compute homogenized stress field corresponding to displacement
            Aeff_h_precise   = discretization.get_homogenized_stress_mugrid(
                material_data_field_ijklqxyz=material_data_field,
                displacement_field_inxyz=temperatute_field_precise,
                macro_gradient_field_ijqxyz=macro_gradient_field)
            callback_x = discretization.get_unknown_size_field(name=f'callback_x')

            def my_callback(it, x, r, p, z, stop_crit_norm):
                # compute homogenized stress field corresponding to displacement
                callback_x.s[:]=x[:]
                homogenized_flux =  discretization.get_homogenized_stress_mugrid(
                material_data_field_ijklqxyz=material_data_field,
                displacement_field_inxyz=callback_x,
                macro_gradient_field_ijqxyz=macro_gradient_field)
                Aeff_hk.append(homogenized_flux[0, 0])
                error_in_Aeff_hk.append(homogenized_flux[0, 0] - A_eff)  # J_eff_computed if J_eff is not available


            parameters_CG = {'exact_solution': temperatute_field_precise,
                             'energy_lower_bound': True,
                             'tau': 0.25}
            temperatute_field_  = discretization.get_unknown_size_field(name=f'temperatute_field_')

            solvers.conjugate_gradients_mugrid(
                comm=discretization.communicator,
                fc=discretization.field_collection,
                hessp=K_fun,  # linear operator
                b=rhs_load_case_inxyz,
                x=temperatute_field_,
                P=M_fun_Green,
                tol=1e-5,
                maxiter=10000,
                callback=my_callback,
                # norm_metric=res_norm
            )

            # true_e_error = np.asarray(norms['energy_iter_error'])
            # lower_bound = np.asarray(norms['energy_lower_bound'])
            # upper_estim = lower_bound / (1 - parameters_CG['tau'])
            # upper_bound = np.asarray(norms['energy_upper_bound'])
            # trivial_lower_bound = np.asarray(norms['residual_rz'] / eigen_UB)
            # trivial_upper_bound = np.asarray(norms['residual_rz'] / eigen_LB)

            # ----------------------------------------------------------------------
            # compute homogenized stress field corresponding to displacement
            homogenized_flux =  discretization.get_homogenized_stress_mugrid(
                material_data_field_ijklqxyz=material_data_field,
                displacement_field_inxyz=temperatute_field_,
                macro_gradient_field_ijqxyz=macro_gradient_field)

            print(homogenized_flux)

            end_time = time.time()
            elapsed_time = end_time - start_time
            print("Elapsed time: ", elapsed_time)

            _info = {}

            # _info['true_e_error'] = true_e_error
            # _info['lower_bound'] = lower_bound
            # _info['upper_bound'] = upper_bound
            # _info['upper_estim'] = upper_estim
            # _info['trivial_lower_bound'] = trivial_lower_bound
            # _info['trivial_upper_bound'] = trivial_upper_bound
            _info['total_phase_contrast'] = total_phase_contrast

            _info['homogenized_flux'] = homogenized_flux
            _info['Aeff_h_precise'] = Aeff_h_precise
            _info['A_eff'] = A_eff
            _info['Aeff_hk'] = Aeff_hk
            _info['error_in_Aeff_hk'] = error_in_Aeff_hk

            results_name = f'N{number_of_pixels[0]}_rho_inc{mat_contrast_1:.2e}_mat{mat_contrast_2:.2e}_ani{a_ani:.2e}'
            save_npy(data_folder_path + results_name + f'.npy',
                     temperatute_field_.s[0].mean(axis=0),
                     tuple(discretization.subdomain_locations_no_buffers),
                     tuple(discretization.nb_of_pixels_global),
                     MPI.COMM_WORLD)

            np.savez(data_folder_path + results_name + f'_log.npz', **_info)
            print(data_folder_path + results_name + f'.npy')
