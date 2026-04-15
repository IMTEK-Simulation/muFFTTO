import numpy as np
import scipy as sp
import matplotlib as mpl
import matplotlib.pyplot as plt
import time
import os

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

from NuMPI.IO import save_npy, load_npy
from mpi4py import MPI

from muFFTTO import domain
from muFFTTO import solvers
from muFFTTO import microstructure_library

from experiments.paper_alg_error_estimates import _labels, _colors, _markers

problem_type = 'conductivity'
discretization_type = 'finite_element'
element_type = 'linear_triangles'  # #'linear_triangles'# linear_triangles_tilled
# formulation = 'small_strain'
geometry_ID = 'square_inclusion'

domain_size = [1, 1]

grids_sizes = [10]#3, 4, 5, 6, 7, 8, 9]  # [3, 4, 5, 6, 7, 8, 9]  # [4, 6, 8], 10  7, 8, 9, 10
rhos = [3, ]

for anisotropy in [ True]:  #False,
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
            mat_contrast = 1  # matrix
            mat_contrast_2 = 10 ** rho  # inclusion

            if anisotropy:
                a_ani = 10
            else:
                a_ani = 1
            conductivity_C_1 = mat_contrast * np.array([[a_ani, 0], [0, 1.0]])
            conductivity_C_2 = mat_contrast_2 * np.array([[a_ani, 0], [0, 1.0]])
            conductivity_C_ref = np.array([[1., 0], [0, 1.0]])

            eigen_C1 = sp.linalg.eigh(a=conductivity_C_1, b=conductivity_C_ref, eigvals_only=True)
            eigen_C2 = sp.linalg.eigh(a=conductivity_C_2, b=conductivity_C_ref, eigvals_only=True)

            eigen_LB = np.min([eigen_C1, eigen_C2])
            # seigen_LB *=0.9
            eigen_UB = np.max([eigen_C1, eigen_C2])
            total_phase_contrast = eigen_UB / eigen_LB

            print(f'eigen_LB = {eigen_LB}')
            print(f'eigen_UB = {eigen_UB}')
            # J_eff = mat_contrast_2 * np.sqrt((mat_contrast_2 + 3 * mat_contrast) / (3 * mat_contrast_2 + mat_contrast))
            # print("J_eff : ", J_eff)
            A_eff = mat_contrast * np.sqrt((mat_contrast + 3 * mat_contrast_2) / (3 * mat_contrast + mat_contrast_2))
            print("A_eff : ", A_eff)

            material_data_field_C_1 = discretization.get_material_data_size_field_mugrid(name='material_data_field_C_1')
            material_data_field_C_2 = discretization.get_material_data_size_field_mugrid(name='material_data_field_C_2')
            material_data_field_C_ref = discretization.get_material_data_size_field_mugrid(
                name='material_data_field_C_ref')

            material_data_field_C_1.s = np.einsum('ij,qxy->ijqxy', conductivity_C_1,
                                                  np.ones(np.array([discretization.nb_quad_points_per_pixel,
                                                                    *discretization.nb_of_pixels])))

            material_data_field_C_2.s = np.einsum('ij,qxy->ijqxy', conductivity_C_2,
                                                  np.ones(np.array([discretization.nb_quad_points_per_pixel,
                                                                    *discretization.nb_of_pixels])))

            material_data_field_C_ref.s = np.einsum('ij,qxy->ijqxy', conductivity_C_ref,
                                                    np.ones(np.array([discretization.nb_quad_points_per_pixel,
                                                                      *discretization.nb_of_pixels])))
            # material distribution
            phase_field = discretization.get_scalar_field(name='phase_field')

            phase_field.s[0, 0] = microstructure_library.get_geometry(nb_voxels=discretization.nb_of_pixels,
                                                                      microstructure_name=geometry_ID,
                                                                      coordinates=discretization.fft.coords)
            matrix_mask = phase_field.s[0, 0] > 0
            inc_mask = phase_field.s[0, 0] == 0

            # apply material distribution
            # material_data_field_C_0_rho = material_data_field_C_1[..., :, :] * np.power(phase_field,                                                                                        1)
            # material_data_field_C_0_rho += material_data_field_C_2[..., :, :] * np.power(1 - phase_field, 2)
            material_data_field_C_0_rho = discretization.get_material_data_size_field_mugrid(
                name='material_data_field_C_0')

            material_data_field_C_0_rho.s[..., matrix_mask] = material_data_field_C_1.s[
                ..., matrix_mask]
            material_data_field_C_0_rho.s[..., inc_mask] = material_data_field_C_2.s[..., inc_mask]
            # Set up the equilibrium system
            macro_gradient_field = discretization.get_gradient_size_field(name='macro_gradient_field')
            discretization.get_macro_gradient_field_mugrid(macro_gradient_ij=macro_gradient,
                                                           macro_gradient_field_ijqxyz=macro_gradient_field)

            # Solve mechanical equilibrium constrain
            rhs_field = discretization.get_unknown_size_field(name='rhs_field')
            discretization.get_rhs_mugrid(material_data_field_ijklqxyz=material_data_field_C_0_rho,
                                          macro_gradient_field_ijqxyz=macro_gradient_field,
                                          rhs_inxyz=rhs_field)


            # K_fun = lambda x: discretization.apply_system_matrix(material_data_field_C_0_rho, x)

            def K_fun(x, Ax):

                discretization.apply_system_matrix_mugrid(material_data_field=material_data_field_C_0_rho,
                                                          input_field_inxyz=x,
                                                          output_field_inxyz=Ax)
                discretization.fft.communicate_ghosts(Ax)


            # M_fun = lambda x: 1 * x
            #
            # preconditioner = discretization.get_preconditioner_NEW(
            #     reference_material_data_field_ijklqxyz=material_data_field_C_ref)
            # # preconditioner_old = discretization.get_preconditioner(reference_material_data_field_ijklqxyz=material_data_field_C_0)
            #
            # M_fun = lambda x: discretization.apply_preconditioner_NEW(preconditioner, x)
            preconditioner = discretization.get_preconditioner_Green_mugrid(
                reference_material_data_ijkl=conductivity_C_ref)


            def M_fun(x, Px):
                """
                Function to compute the product of the Preconditioner matrix with a vector.
                The Preconditioner is represented by the convolution operator.
                """
                discretization.fft.communicate_ghosts(x)
                discretization.apply_preconditioner_mugrid(preconditioner_Fourier_fnfnqks=preconditioner,
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


            # temperatute_field_precise, norms_precise = solvers.PCG(Afun=K_fun,
            #                                                        B=rhs_field,
            #                                                        x0=None,
            #                                                        P=M_fun,
            #                                                        steps=int(1000),
            #                                                        toler=1e-14,
            #                                                        norm_energy_upper_bound=True,
            #                                                        lambda_min=eigen_LB)
            temperatute_field_precise = discretization.get_unknown_size_field(name=f'temperatute_field_precise')
            solvers.conjugate_gradients_mugrid_experimental(
                comm=discretization.communicator,
                fc=discretization.field_collection,
                hessp=K_fun,  # linear operator
                b=rhs_field,
                x=temperatute_field_precise,
                P=M_fun,
                tol=1e-14,
                maxiter=10000,
                callback=callback,
                # norm_metric=res_norm
            )

            # compute homogenized stress field corresponding to displacement
            Aeff_h_precise = discretization.get_homogenized_stress_mugrid(
                material_data_field_ijklqxyz=material_data_field_C_0_rho,
                displacement_field_inxyz=temperatute_field_precise,
                macro_gradient_field_ijqxyz=macro_gradient_field)

            error_in_Aeff_hk = []
            Aeff_hk = []

            callback_x = discretization.get_unknown_size_field(name=f'callback_x')

            norms_cg_mech = dict()
            norms_cg_mech['residual_rr'] = []
            norms_cg_mech['residual_rz'] = []


            def my_callback(it, x, r, p, z, stop_crit_norm):
                # compute homogenized stress field corresponding to displacement
                callback_x.s[:] = x[:]
                homogenized_flux = discretization.get_homogenized_stress_mugrid(
                    material_data_field_ijklqxyz=material_data_field_C_0_rho,
                    displacement_field_inxyz=callback_x,
                    macro_gradient_field_ijqxyz=macro_gradient_field)
                Aeff_hk.append(homogenized_flux[0, 0])
                error_in_Aeff_hk.append(homogenized_flux[0, 0] - A_eff)  # J_eff_computed if J_eff is not available
                norm_of_rr = discretization.communicator.sum(np.dot(r.ravel(), r.ravel()))
                norm_of_rz = discretization.communicator.sum(np.dot(r.ravel(), z.ravel()))
                norms_cg_mech['residual_rr'].append(norm_of_rr)
                norms_cg_mech['residual_rz'].append(norm_of_rz)


            # parameters_CG = {'exact_solution': temperatute_field_precise,
            #                  'energy_lower_bound': True,
            #                  'tau': 0.25}
            # temperatute_field, norms = solvers.PCG(Afun=K_fun,
            #                                        B=rhs_field,
            #                                        x0=None,
            #                                        P=M_fun,
            #                                        steps=int(1000), toler=1e-14,
            #                                        norm_energy_upper_bound=True,
            #                                        lambda_min=eigen_LB,
            #                                        callback=my_callback,
            #                                        **parameters_CG)
            temperatute_field_ = discretization.get_unknown_size_field(name=f'temperatute_field_')

            temperatute_field_, norms = solvers.conjugate_gradients_mugrid_experimental(
                comm=discretization.communicator,
                fc=discretization.field_collection,
                hessp=K_fun,  # linear operator
                b=rhs_field,
                x=temperatute_field_,
                P=M_fun,
                tol=1e-14,
                maxiter=10000,
                callback=my_callback,
                # norm_metric=res_norm
            )
            # nb_it = len(norms['residual_rz'])
            # print(' nb_ steps CG =' f'{nb_it}')
            #
            # true_e_error = np.asarray(norms['energy_iter_error'])
            # lower_bound = np.asarray(norms['energy_lower_bound'])
            # upper_estim = lower_bound / (1 - parameters_CG['tau'])
            # upper_bound = np.asarray(norms['energy_upper_bound'])
            # trivial_lower_bound = np.asarray(norms['residual_rz'] / eigen_UB)
            # trivial_upper_bound = np.asarray(norms['residual_rz'] / eigen_LB)

            # ----------------------------------------------------------------------
            # compute homogenized stress field corresponding to displacement
            homogenized_flux = discretization.get_homogenized_stress_mugrid(
                material_data_field_ijklqxyz=material_data_field_C_0_rho,
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
            # _info['total_phase_contrast'] = total_phase_contrast
            _info['lower_estim'] = norms['energy_lower_bound']
            _info['residual_rz'] = norms_cg_mech['residual_rz']
            _info['residual_rr'] = norms_cg_mech['residual_rr']

            _info['homogenized_flux'] = homogenized_flux
            _info['Aeff_h_precise'] = Aeff_h_precise
            _info['A_eff'] = A_eff
            _info['Aeff_hk'] = Aeff_hk
            _info['error_in_Aeff_hk'] = error_in_Aeff_hk

            results_name = f'N{number_of_pixels[0]}_rho_inc{mat_contrast_2:.0f}_mat{mat_contrast:.0f}_ani{a_ani:.0f}'
            save_npy(data_folder_path + results_name + f'.npy',
                     temperatute_field_.s[0].mean(axis=0),
                     tuple(discretization.subdomain_locations_no_buffers),
                     tuple(discretization.nb_of_pixels_global),
                     MPI.COMM_WORLD)

            np.savez(data_folder_path + results_name + f'_log.npz', **_info)
            print(data_folder_path + results_name + f'.npy')
