import numpy as np
import scipy as sc
import time
import os
import sys

sys.path.append('../..')  # Add parent directory to path

from mpi4py import MPI
from NuMPI.IO import save_npy, load_npy

import matplotlib as mpl
from matplotlib import pyplot as plt

from muFFTTO import domain
from muFFTTO import solvers
from muFFTTO import microstructure_library

# script_name = 'exp_paper_JG_nonlinear_elasticity_JZ'
script_name = os.path.splitext(os.path.basename(__file__))[0]

file_folder_path = os.path.dirname(os.path.realpath(__file__))  # script directory
data_folder_path = file_folder_path + '/exp_data/' + script_name + '/'
figure_folder_path = file_folder_path + '/figures/' + script_name + '/'

enforce_mean = False
for preconditioner_type in ['Jacobi_Green', 'Green', ]:  #
    for nnn in 2 ** np.array([8]):  # [32, ]:,7,8,9  #  # 3, 4, 5, 6, 7, 8, 9 5, 6, 7, 8, 95, 6, 7, 8, 9
        start_time = time.time()
        number_of_pixels = (nnn, nnn, nnn)  # (128, 128, 1)  # (32, 32, 1) # (64, 64, 1)  # (128, 128, 1) #
        domain_size = [1, 1, 1]
        Nx = number_of_pixels[0]
        Ny = number_of_pixels[1]
        Nz = number_of_pixels[2]

        save_results = True
        _info = {}

        problem_type = 'elasticity'
        discretization_type = 'finite_element'
        element_type = 'trilinear_hexahedron'  # 'trilinear_hexahedron' #'trilinear_hexahedron_1Q'
        formulation = 'small_strain'
        print(f'preconditioer {preconditioner_type}')

        _info['problem_type'] = problem_type
        _info['discretization_type'] = discretization_type
        _info['element_type'] = element_type
        _info['formulation'] = formulation
        _info['preconditioner_type'] = preconditioner_type

        my_cell = domain.PeriodicUnitCell(domain_size=domain_size,
                                          problem_type=problem_type)

        discretization = domain.Discretization(cell=my_cell,
                                               nb_of_pixels_global=number_of_pixels,
                                               discretization_type=discretization_type,
                                               element_type=element_type)

        file_folder_path = os.path.dirname(os.path.realpath(__file__))  # script directory
        data_folder_path = (
                file_folder_path + '/exp_data/' + script_name + '/' + f'Nx={Nx}' + f'Ny={Ny}' + f'Nz={Nz}'
                + f'_{preconditioner_type}' + '/')
        figure_folder_path = (file_folder_path + '/figures/' + script_name + '/' f'Nx={Nx}' + f'Ny={Ny}' + f'Nz={Nz}'
                              + f'_{preconditioner_type}' + '/')
        if discretization.fft.communicator.rank == 0:
            if not os.path.exists(file_folder_path):
                os.makedirs(file_folder_path)
            if not os.path.exists(data_folder_path):
                os.makedirs(data_folder_path)
            if not os.path.exists(figure_folder_path):
                os.makedirs(figure_folder_path)

        _info['nb_of_pixels'] = discretization.nb_of_pixels_global
        _info['domain_size'] = domain_size

        start_time = time.time()

        # identity tensor                                               [single tensor]
        i = np.eye(discretization.domain_dimension)
        I = np.einsum('ij,xyz', i, np.ones(number_of_pixels))

        # identity tensors                                            [grid of tensors]
        I4 = np.einsum('il,jk', i, i)
        I4rt = np.einsum('ik,jl', i, i)
        II = np.einsum('ij...  ,kl...  ->ijkl...', i, i)
        I4s = (I4 + I4rt) / 2.
        I4d = (I4s - II / 3.)

        model_parameters_non_linear = {'K': 2,
                                       'mu': 1,
                                       'sig0': 1.,
                                       'eps0': 0.1,
                                       'n': 10.0}

        model_parameters_linear = {'K': 2,
                                   'mu': 1}

        _info['model_parameters_non_linear'] = model_parameters_non_linear
        _info['model_parameters_linear'] = model_parameters_linear

        phase_field = discretization.get_scalar_field(name='phase_field')

        geometry_ID = 'square_inclusion'
        phase_field.s[0, 0] = microstructure_library.get_geometry(nb_voxels=discretization.nb_of_pixels,
                                                                  microstructure_name=geometry_ID,
                                                                  coordinates=discretization.fft.coords)
        # TODO delete
        phase_field.s[0, 0,
        1 * number_of_pixels[0] // 4:3 * number_of_pixels[0] // 4,
        1 * number_of_pixels[1] // 4:3 * number_of_pixels[1] // 4,
        :  # 1 * number_of_pixels[2] // 4:3 * number_of_pixels[2] // 4
        ] = 0

        matrix_mask = phase_field.s[0, 0] == 0
        inc_mask = phase_field.s[0, 0] > 0


        # phase_field[:26, :, :] = 1.
        # linear elasticity
        # -----------------
        def linear_elastic_q_points(strain_ijqxyz,
                                    tangent_ijklqxyz,
                                    stress_ijqxyz,
                                    phase_xyz,
                                    **kwargs):
            # parameters
            K = kwargs['K']
            mu = kwargs['mu']  # mu = 1.  # shear modulus
            # bulk  modulus

            # elastic stiffness tensor, and stress response
            # C4 = K * II_qxyz + 2. * mu * I4d_qxyz
            tangent_ijklqxyz.s[..., phase_xyz] = (K * II + 2. * mu * I4d)[..., None, None]
            stress_ijqxyz.s[..., phase_xyz] = np.einsum('ijklqx...,lkqx...  ->ijqx...  ',
                                                        tangent_ijklqxyz.s[..., phase_xyz],
                                                        strain_ijqxyz.s[..., phase_xyz])
            # sig = ddot42(C4, strain)


        # print()

        ###
        def nonlinear_elastic_q_points(strain_ijqxyz,
                                       tangent_ijklqxyz,
                                       stress_ijqxyz,
                                       phase_xyz,
                                       **kwargs):
            # K = 2.  # bulk modulus
            # sigma = K*trace(small_strain)*I_ij  + sigma_0* (strain_eq/epsilon_0)^n * N_ijkl
            K = kwargs['K']
            sig0 = kwargs['sig0']  # 1e3  # 0.25 #* K  # reference stress # 1e5              # 0.5
            eps0 = kwargs['eps0']  # = 0.03  # 0.2  # reference strain #    # 0.03                  # 0.1
            n = kwargs['n']  # 5.0  # 3.0  # hardening exponent  # # 5.0               # 10.0

            strain_trace_qx = np.einsum('ii...', strain_ijqxyz.s[..., phase_xyz]) / 3  # todo{2 or 3 in 2D }
            # strain_trace_xyz = np.einsum('ijxyz,ji ->xyz', strain, I) / 3  # todo{2 or 3 in 2D }

            # volumetric strain
            strain_vol_ijqxyz = discretization.get_gradient_size_field(name='strain_vol_ijqxyz')
            # strain_vol_ijqxyz = np.ndarray(shape=strain.shape)
            strain_vol_ijqxyz.s.fill(0)
            for d in np.arange(discretization.domain_dimension):
                strain_vol_ijqxyz.s[..., phase_xyz][d, d] = strain_trace_qx

            # deviatoric strain
            strain_dev_ijqxyz = discretization.get_gradient_size_field(name='strain_dev_ijqxyz')
            strain_dev_ijqxyz.s[..., phase_xyz] = strain_ijqxyz.s[..., phase_xyz] - strain_vol_ijqxyz.s[..., phase_xyz]

            # equivalent strain
            strain_dev_ddot = np.einsum('ijqx...,jiqx...-> qx...', strain_dev_ijqxyz.s[..., phase_xyz],
                                        strain_dev_ijqxyz.s[..., phase_xyz])
            strain_eq_qx = np.sqrt((2. / 3.) * strain_dev_ddot)

            #
            stress_ijqxyz.s[..., phase_xyz] = (3. * K * strain_vol_ijqxyz.s[..., phase_xyz]
                                               + 2. / 3. * sig0 / (eps0 ** n) *
                                               (strain_eq_qx ** (n - 1.)) * strain_dev_ijqxyz.s[..., phase_xyz])
            #
            # sig = 3. * K * strain_vol_ijqxyz * (strain_eq_qxyz == 0.).astype(float) + sig * (
            #         strain_eq_qxyz != 0.).astype(float)

            # K4_d = discretization.get_material_data_size_field(name='alg_tangent')
            strain_dev_dyad = np.einsum('ijqx...,klqx...->ijklqx...', strain_dev_ijqxyz.s[..., phase_xyz],
                                        strain_dev_ijqxyz.s[..., phase_xyz])

            K4_d = 2. / 3. * sig0 / (eps0 ** n) * (strain_dev_dyad * 2. / 3. * (n - 1.) * strain_eq_qx ** (
                    n - 3.) + strain_eq_qx ** (n - 1.) * I4d[..., np.newaxis, np.newaxis])

            # threshold = 1e-15
            # mask = (np.abs(strain_eq_qxyz) > threshold).astype(float)

            tangent_ijklqxyz.s[..., phase_xyz] = K * II[..., np.newaxis, np.newaxis] + K4_d
            # * mask  # *(strain_equivalent_qxyz != 0.).astype(float)


        def constitutive_q_points(strain_ijqxyz, tangent_ijklqxyz, stress_ijqxyz):
            #            phase_field = np.zeros([*number_of_pixels])
            global matrix_mask, inc_mask
            linear_elastic_q_points(strain_ijqxyz=strain_ijqxyz,
                                    tangent_ijklqxyz=tangent_ijklqxyz,
                                    stress_ijqxyz=stress_ijqxyz,
                                    phase_xyz=matrix_mask,
                                    **model_parameters_linear)

            nonlinear_elastic_q_points(strain_ijqxyz=strain_ijqxyz,
                                       tangent_ijklqxyz=tangent_ijklqxyz,
                                       stress_ijqxyz=stress_ijqxyz,
                                       phase_xyz=inc_mask,
                                       **model_parameters_non_linear)

            print()


        def constitutive(strain_ijqxyz,
                         sig_ijqxyz,
                         K4_ijklqxyz):

            constitutive_q_points(strain_ijqxyz=strain_ijqxyz,
                                  tangent_ijklqxyz=K4_ijklqxyz,
                                  stress_ijqxyz=sig_ijqxyz)


        macro_gradient_field = discretization.get_gradient_size_field(name='macro_gradient_field')
        macro_gradient_inc_field = discretization.get_gradient_size_field(name='macro_gradient_inc_field')

        displacement_fluctuation_field = discretization.get_unknown_size_field(name='displacement_fluctuation_field')
        displacement_increment_field = discretization.get_unknown_size_field(name='displacement_increment_field')

        strain_fluc_field = discretization.get_displacement_gradient_sized_field(name='strain_fluctuation_field')
        total_strain_field = discretization.get_displacement_gradient_sized_field(name='strain_field')
        rhs_field = discretization.get_unknown_size_field(name='rhs_field')

        stress_field = discretization.get_displacement_gradient_sized_field(name='stress_field')
        K4_ijklqyz = discretization.get_material_data_size_field_mugrid(name='K4_ijklqxyz')

        x = np.linspace(start=0, stop=domain_size[0], num=number_of_pixels[0])
        y = np.linspace(start=0, stop=domain_size[1], num=number_of_pixels[1])
        X, Y = np.meshgrid(x, y, indexing='ij')

        # evaluate material law
        constitutive(total_strain_field, stress_field, K4_ijklqyz)

        if save_results:
            # save strain fluctuation
            i = 0
            temp_max_size_ = {'nb_max_subdomain_grid_pts': discretization.nb_max_subdomain_grid_pts}

            results_name = (f'init_K_{0, 0}')
            to_save = np.copy(K4_ijklqyz.s.mean(axis=4)[0, 0, 0, 0])
            # np.save(data_folder_path + results_name + f'.npy', strain_fluc_field.s.mean(axis=2))
            save_npy(data_folder_path + results_name + f'.npy', to_save,
                     tuple(discretization.subdomain_locations_no_buffers),
                     tuple(discretization.nb_of_pixels_global), MPI.COMM_WORLD, **temp_max_size_)
        # set macroscopic loading increment
        ninc = 1
        _info['ninc'] = ninc

        macro_gradient_inc = np.zeros(shape=(3, 3))
        # macro_gradient_inc[0, 0] += 0.05 / float(ninc)
        macro_gradient_inc[0, 1] += 0.05 / float(ninc)
        macro_gradient_inc[1, 0] += 0.05 / float(ninc)
        dt = 1. / float(ninc)

        # set macroscopic gradient
        # macro_gradient_inc_field = discretization.get_macro_gradient_field(macro_gradient_ij=macro_gradient_inc,
        #                                                                    macro_gradient_field_ijqxyz=macro_gradient_inc_field)
        discretization.get_macro_gradient_field_mugrid(macro_gradient_ij=macro_gradient_inc,
                                                       macro_gradient_field_ijqxyz=macro_gradient_inc_field)
        # assembly preconditioner
        preconditioner = discretization.get_preconditioner_Green_mugrid(
            reference_material_data_ijkl=I4s)  # K4_ijklqyz.mean(axis=(4, 5, 6, 7))


        #
        # M_fun_Green = lambda x: discretization.apply_preconditioner_NEW(preconditioner_Fourier_fnfnqks=preconditioner,
        #                                                                 nodal_field_fnxyz=x)

        def M_fun_Green(x, Px):
            discretization.fft.communicate_ghosts(x)
            discretization.apply_preconditioner_mugrid(preconditioner_Fourier_fnfnqks=preconditioner,
                                                       input_nodal_field_fnxyz=x,
                                                       output_nodal_field_fnxyz=Px)


        sum_CG_its = 0
        sum_Newton_its = 0
        start_time = time.time()
        iteration_total = 0

        # incremental loading
        for inc in range(ninc):
            print(f'Increment {inc}')
            print(f'==========================================================================')

            # strain-hardening exponent
            total_strain_field.s[...] += macro_gradient_inc_field.s[...]

            # evaluate material law
            constitutive(total_strain_field, stress_field, K4_ijklqyz)

            # Solve mechanical equilibrium constrain
            # rhs_field = discretization.get_rhs_explicit_stress(stress_function=constitutive,  # constitutive_pixel
            #                                                    gradient_field_ijqxyz=total_strain_field,
            #                                                    rhs_inxyz=rhs_field)
            # discretization.get_rhs_mugrid(material_data_field_ijklqxyz=K4_ijklqyz,  # constitutive_pixel
            #                               macro_gradient_field_ijqxyz=total_strain_field,
            #                               rhs_inxyz=rhs_field)
            discretization.fft.communicate_ghosts(stress_field)
            discretization.apply_gradient_transposed_operator_mugrid(gradient_field_ijqxyz=stress_field,
                                                                     div_u_fnxyz=rhs_field,
                                                                     apply_weights=True)
            rhs_field.s *= -1

            # print('rank' f'{MPI.COMM_WORLD.rank:6} get_rhs_mugrid rhs_inxyz shape.  =' f'{rhs_inxyz.s.shape}')
            #       print('rank' f'{MPI.COMM_WORLD.rank:6} get_rhs_mugrid rhs_inxyz.s[0] =' f'{rhs_inxyz.s[0]}')
            #

            #

            # discretization.get_rhs_mugrid(material_data_field_ijklqxyz=material_data_field_C_0,
            #                               macro_gradient_field_ijqxyz=macro_gradient_field,
            #                               rhs_inxyz=rhs_field)
            # evaluate material law
            # stress, K4_ijklqyz = constitutive(total_strain_field)  #
            # print('save_results')

            if save_results:
                temp_max_size_ = {'nb_max_subdomain_grid_pts': discretization.nb_max_subdomain_grid_pts}
                i = 0
                j = 1
                # save strain fluctuation
                print('discretization.nb_of_pixels_global', discretization.nb_of_pixels_global)

                results_name = (f'strain_fluc_field_{i, j}' + f'_it{iteration_total}')
                to_save = np.copy(strain_fluc_field.s.mean(axis=2))[i, j]
                to_save = np.array(to_save, dtype=np.float64)
                # np.save(data_folder_path + results_name + f'.npy', strain_fluc_field.s.mean(axis=2))
                save_npy(data_folder_path + results_name + f'.npy', to_save,
                         tuple(discretization.subdomain_locations_no_buffers),
                         tuple(discretization.nb_of_pixels_global), MPI.COMM_WORLD,
                         **temp_max_size_)

                # save total  strain
                results_name = (f'total_strain_field_{i, j}' + f'_it{iteration_total}')
                to_save = np.copy(total_strain_field.s.mean(axis=2)[i, j])
                # np.save(data_folder_path + results_name + f'.npy', strain_fluc_field.s.mean(axis=2))
                save_npy(data_folder_path + results_name + f'.npy', to_save,
                         tuple(discretization.subdomain_locations_no_buffers),
                         tuple(discretization.nb_of_pixels_global), MPI.COMM_WORLD, **temp_max_size_)

                # save stress
                results_name = (f'stress_{i, j}' + f'_it{iteration_total}')
                to_save = np.copy(stress_field.s.mean(axis=2)[i, j])
                # np.save(data_folder_path + results_name + f'.npy', strain_fluc_field.s.mean(axis=2))
                save_npy(data_folder_path + results_name + f'.npy', to_save,
                         tuple(discretization.subdomain_locations_no_buffers),
                         tuple(discretization.nb_of_pixels_global), MPI.COMM_WORLD, **temp_max_size_)
                # save K4_ijklqyz
                results_name = (f'K4_ijklqyz_{0, 0}' + f'_it{iteration_total}')
                to_save = np.copy(K4_ijklqyz.s.mean(axis=4)[0, 0, 0, 0])
                # np.save(data_folder_path + results_name + f'.npy', strain_fluc_field.s.mean(axis=2))
                save_npy(data_folder_path + results_name + f'.npy', to_save,
                         tuple(discretization.subdomain_locations_no_buffers),
                         tuple(discretization.nb_of_pixels_global), MPI.COMM_WORLD, **temp_max_size_)

                results_name = (f'rhs_field_{i}' + f'_it{iteration_total}')
                to_save = np.copy(rhs_field.s.mean(axis=1)[i])
                # np.save(data_folder_path + results_name + f'.npy', strain_fluc_field.s.mean(axis=2))
                save_npy(data_folder_path + results_name + f'.npy', to_save,
                         tuple(discretization.subdomain_locations_no_buffers),
                         tuple(discretization.nb_of_pixels_global), MPI.COMM_WORLD, **temp_max_size_)
            # print('save_results')
            # En = np.sqrt(np.linalg.norm(total_strain_field.s.mean(axis=2)))
            En = np.sqrt(
                discretization.fft.communicator.sum(np.dot(total_strain_field.s.ravel(), total_strain_field.s.ravel())))

            # rhs_t_norm = np.linalg.norm(rhs_field.s)
            rhs_t_norm = np.sqrt(discretization.fft.communicator.sum(np.dot(rhs_field.s.ravel(), rhs_field.s.ravel())))
            # print(f'rhs_t_norm {rhs_t_norm}')
            # print(f'norm_rhs {norm_rhs}')
            # incremental deformation  newton loop
            iiter = 0

            norm_rhs = np.sqrt(discretization.fft.communicator.sum(np.dot(rhs_field.s.ravel(), rhs_field.s.ravel())))
            if discretization.fft.communicator.rank == 0:
                print('Rhs at new laod step {0:10.2e}'.format(norm_rhs))

            # preconditioer = 'Green'

            # iterate as long as the iterative update does not vanish
            while True:
                # Set up preconditioner

                if preconditioner_type == 'Green':
                    M_fun = M_fun_Green
                elif preconditioner_type == 'Jacobi_Green':
                    # K_ref = discretization.get_system_matrix(I4s)
                    # K_tot=discretization.get_system_matrix_mugrid(material_data_field=K4_ijklqyz, formulation=formulation)

                    K_diag_alg = discretization.get_preconditioner_Jacobi_mugrid(
                        material_data_field_ijklqxyz=K4_ijklqyz, formulation=formulation)  # K4_ijklqyz


                    # results_name = (f'K4_ijklqyz_full_it0')
                    # K4_initaa = np.load(data_folder_path + results_name + f'.npy', allow_pickle=True)

                    # GJ_matrix = np.diag(K_diag_alg.flatten()) @ K_ref @ np.diag(K_diag_alg.flatten())
                    def M_fun_Jacobi(x, Px):
                        discretization.fft.communicate_ghosts(x)
                        x_jacobi_temp = discretization.get_unknown_size_field(name='x_jacobi_temp')

                        x_jacobi_temp.s = K_diag_alg.s * x.s
                        discretization.apply_preconditioner_mugrid(preconditioner_Fourier_fnfnqks=preconditioner,
                                                                   input_nodal_field_fnxyz=x_jacobi_temp,
                                                                   output_nodal_field_fnxyz=Px)

                        Px.s = K_diag_alg.s * Px.s
                        discretization.fft.communicate_ghosts(Px)


                    M_fun = M_fun_Jacobi


                def M_fun_none(x, Px):
                    Px.s = x.s


                # M_fun = M_fun_none

                # mat_model_pars = {'mat_model': 'power_law_elasticity'}

                def K_fun(x, Ax):
                    discretization.apply_system_matrix_mugrid(material_data_field=K4_ijklqyz,
                                                              input_field_inxyz=x,
                                                              output_field_inxyz=Ax,
                                                              formulation=formulation)
                    discretization.fft.communicate_ghosts(Ax)


                norms = dict()
                norms['residual_rr'] = []
                norms['residual_rz'] = []


                def callback(it, x, r, p, z, stop_crit_norm):
                    global norms
                    """
                    Callback function to print the current solution, residual, and search direction.
                    """
                    #   for d in range(3):
                    #      x[d] -= np.mean(x[d], axis=(-1, -2, -3), keepdims=True)
                    # discretization.fft.communicate_ghosts(x)
                    norm_of_rr = discretization.fft.communicator.sum(np.dot(r.ravel(), r.ravel()))
                    norm_of_rz = discretization.fft.communicator.sum(np.dot(r.ravel(), z.ravel()))
                    norms['residual_rr'].append(norm_of_rr)
                    norms['residual_rz'].append(norm_of_rz)

                    if discretization.fft.communicator.rank == 0:
                        print(f"{it:5} norm of rr = {norm_of_rr:.5}")
                        print(f"{it:5} norm of rz = {norm_of_rz:.5}")
                        print(f"{it:5} stop_crit_norm = {stop_crit_norm:.5}")


                displacement_increment_field.s.fill(0)
                print('solvers')
                solvers.conjugate_gradients_mugrid(
                    comm=discretization.fft.communicator,
                    fc=discretization.field_collection,
                    hessp=K_fun,  # linear operator
                    b=rhs_field,
                    x=displacement_increment_field,
                    P=M_fun,
                    tol=1e-6,
                    maxiter=5000,
                    callback=callback,
                    norm_metric=M_fun_Green
                )

                nb_it_comb = len(norms['residual_rr'])
                if len(norms['residual_rr']) > 1:
                    norm_rz = norms['residual_rz'][-1]
                    norm_rr = norms['residual_rr'][-1]
                else:
                    norm_rz = 0
                    norm_rr = 0
                if discretization.fft.communicator.rank == 0:
                    print(f'nb iteration CG = {nb_it_comb}')
                sum_CG_its += nb_it_comb

                _info['norm_rr'] = norms['residual_rr']
                _info['norm_rz'] = norms['residual_rz']
                _info['nb_it_comb'] = nb_it_comb

                # phase_field_sol_FE_MPI = xopt.x.reshape([1, 1, *discretization.nb_of_pixels])
                # update Newton iteration counter
                iiter += 1
                sum_Newton_its += 1
                iteration_total += 1

                # compute strain from the displacement increment
                discretization.apply_gradient_operator_symmetrized_mugrid(
                    u_inxyz=displacement_increment_field,
                    grad_u_ijqxyz=strain_fluc_field)

                total_strain_field.s += strain_fluc_field.s
                displacement_fluctuation_field.s += displacement_increment_field.s
                # evaluate material law
                # stress, K4_ijklqyz = constitutive(total_strain_field)  #
                constitutive(total_strain_field, stress_field, K4_ijklqyz)

                # Recompute right hand side
                # rhs_field = discretization.get_rhs_explicit_stress(stress_function=constitutive,  # constitutive_pixel,
                #                                                    gradient_field_ijqxyz=total_strain_field,
                #                                                    rhs_inxyz=rhs_field)
                # discretization.get_rhs_mugrid(material_data_field_ijklqxyz=K4_ijklqyz,  # constitutive_pixel
                #                               macro_gradient_field_ijqxyz=total_strain_field,
                #                               rhs_inxyz=rhs_field)
                discretization.apply_gradient_transposed_operator_mugrid(gradient_field_ijqxyz=stress_field,
                                                                         div_u_fnxyz=rhs_field,
                                                                         apply_weights=True)
                rhs_field.s *= -1

                if save_results:
                    results_name = (f'displacement_increment_field_{i}' + f'_it{iteration_total}')
                    # np.save(data_folder_path + results_name + f'.npy', displacement_increment_field.s)
                    to_save = np.copy(displacement_increment_field.s.mean(axis=1)[i])
                    # np.save(data_folder_path + results_name + f'.npy', strain_fluc_field.s.mean(axis=2))
                    save_npy(data_folder_path + results_name + f'.npy', to_save,
                             tuple(discretization.subdomain_locations_no_buffers),
                             tuple(discretization.nb_of_pixels_global), MPI.COMM_WORLD, **temp_max_size_)

                    # save strain fluctuation
                    results_name = (f'strain_fluc_field_{i, j}' + f'_it{iteration_total}')
                    to_save = np.copy(strain_fluc_field.s.mean(axis=2)[i, j])
                    save_npy(data_folder_path + results_name + f'.npy', to_save,
                             tuple(discretization.subdomain_locations_no_buffers),
                             tuple(discretization.nb_of_pixels_global), MPI.COMM_WORLD, **temp_max_size_)

                    # save total  strain
                    results_name = (f'total_strain_field_{i, j}' + f'_it{iteration_total}')
                    to_save = np.copy(total_strain_field.s.mean(axis=2)[i, j])
                    save_npy(data_folder_path + results_name + f'.npy', to_save,
                             tuple(discretization.subdomain_locations_no_buffers),
                             tuple(discretization.nb_of_pixels_global), MPI.COMM_WORLD, **temp_max_size_)
                    # save stress
                    results_name = (f'stress_{i, j}' + f'_it{iteration_total}')
                    to_save = np.copy(stress_field.s.mean(axis=2)[i, j])
                    save_npy(data_folder_path + results_name + f'.npy', to_save,
                             tuple(discretization.subdomain_locations_no_buffers),
                             tuple(discretization.nb_of_pixels_global), MPI.COMM_WORLD, **temp_max_size_)

                    # save K4_ijklqyz
                    results_name = (f'K4_ijklqyz_{0, 0}' + f'_it{iteration_total}')
                    to_save = np.copy(K4_ijklqyz.s.mean(axis=4)[0, 0, 0, 0])
                    # np.save(data_folder_path + results_name + f'.npy', strain_fluc_field.s.mean(axis=2))
                    save_npy(data_folder_path + results_name + f'.npy', to_save,
                             tuple(discretization.subdomain_locations_no_buffers),
                             tuple(discretization.nb_of_pixels_global), MPI.COMM_WORLD, **temp_max_size_)

                    results_name = (f'rhs_field_{i}' + f'_it{iteration_total}')
                    to_save = np.copy(rhs_field.s.mean(axis=1)[i])
                    save_npy(data_folder_path + results_name + f'.npy', to_save,
                             tuple(discretization.subdomain_locations_no_buffers),
                             tuple(discretization.nb_of_pixels_global), MPI.COMM_WORLD, **temp_max_size_)

                # rhs *= -1

                # g_norm_div_stress = np.sum(rhs_field * M_fun_Green(rhs_field))
                # g_norm_div_stress_rel = np.sum(rhs_field * M_fun_Green(rhs_field)) / np.sum(stress_field.s)
                #
                # print('=====================')
                # print('g_norm_stress {}'.format(g_norm_div_stress))
                # print('g_norm_div_stress_rel {}'.format(g_norm_div_stress_rel))
                # En = np.linalg.norm(total_strain_field.s)
                En = np.sqrt(discretization.fft.communicator.sum(
                    np.dot(total_strain_field.s.ravel(), total_strain_field.s.ravel())))
                norm_rhs = np.sqrt(discretization.fft.communicator.sum(
                    np.dot(rhs_field.s.ravel(), rhs_field.s.ravel())))
                norm_strain_fluc = np.sqrt(discretization.fft.communicator.sum(
                    np.dot(strain_fluc_field.s.ravel(), strain_fluc_field.s.ravel())))
                _info['norm_strain_fluc_field'] = norm_strain_fluc
                _info['norm_En'] = En
                _info['rhs_t_norm'] = rhs_t_norm
                _info['norm_rhs_field'] = norm_rhs

                if discretization.fft.communicator.rank == 0:
                    print('=====================')
                    print('np.linalg.norm(strain_fluc_field.s) / En {0:10.2e}'.format(
                        norm_strain_fluc / En))
                    print('np.linalg.norm(rhs_field.s) / rhs_t_norm  {0:10.2e}'.format(
                        norm_rhs / rhs_t_norm))
                    print('norm_rhs {0:10.2e}'.format(norm_rhs))
                    print('strain_fluc_field {0:10.2e}'.format(norm_strain_fluc))

                if MPI.COMM_WORLD.rank == 0:
                    np.savez(data_folder_path + f'info_log_it{iteration_total - 1}.npz', **_info)
                    print(data_folder_path + f'info_log_it{iteration_total}.npz')

                # if np.linalg.norm(strain_fluc_field.s) / En < 1.e-6 and iiter > 0: break
                # if np.linalg.norm(rhs_field.s) / rhs_t_norm < 1.e-6 and iiter > 0: break

                if norm_rhs < 1.e-4 and iiter > 0: break

                if iiter == 100:
                    break

            # # linear part of displacement(X-domain_size[0]/2)
            disp_linear_x = ((X - domain_size[0] / 2) * macro_gradient_inc[0, 0] * inc +
                             Y * macro_gradient_inc[0, 1] * inc)  # (X - domain_size[0] / 2)
            disp_linear_y = ((X - domain_size[0] / 2) * macro_gradient_inc[1, 0] * inc
                             + Y * macro_gradient_inc[1, 1] * inc)
            # displacement in voids should be zero
            # displacement_fluctuation_field.s[:, 0, :, :5] = 0.0
            end_time = time.time()
            elapsed_time = end_time - start_time

            if discretization.fft.communicator.rank == 0:
                print("element_type : ", element_type)
                print("number_of_pixels: ", number_of_pixels)
                print(f'preconditioner_type: {preconditioner_type}')

                print(f'Total number of CG {sum_CG_its}')
                print(f'Total number of sum_Newton_its {sum_Newton_its}')

                print("Elapsed time : ", elapsed_time)
                print("Elapsed time: ", elapsed_time / 60)

            if save_results:
                # x_deformed = X + disp_linear_x + displacement_fluctuation_field.s[0, 0, :, :, 0]
                # y_deformed = Y + disp_linear_y + displacement_fluctuation_field.s[1, 0, :, :, 0]
                # # save deformed positions
                # results_name = (f'x_deformed' + f'_it{iteration_total}')
                # np.save(data_folder_path + results_name + f'.npy', x_deformed)
                #
                # results_name = (f'y_deformed' + f'_it{iteration_total}')
                # np.save(data_folder_path + results_name + f'.npy', y_deformed)

                _info['sum_Newton_its'] = sum_Newton_its
                _info['iteration_total'] = iteration_total
                _info['sum_CG_its'] = sum_CG_its
                _info['elapsed_time'] = elapsed_time
                if MPI.COMM_WORLD.rank == 0:
                    np.savez(data_folder_path + f'info_log_final.npz', **_info)
                    print(data_folder_path + f'info_log_final.npz')

            plot_sol_field = False
            if plot_sol_field:
                x_deformed = X + disp_linear_x + displacement_fluctuation_field.s[0, 0, :, :, 0]
                y_deformed = Y + disp_linear_y + displacement_fluctuation_field.s[1, 0, :, :, 0]
                fig = plt.figure(figsize=(9, 3.0))
                gs = fig.add_gridspec(2, 2, hspace=0.5, wspace=0.5, width_ratios=[1, 1],
                                      height_ratios=[1, 1])

                ax_strain = fig.add_subplot(gs[1, 0])
                pcm = ax_strain.pcolormesh(x_deformed, y_deformed, total_strain_field.s.mean(axis=2)[0, 1, ..., 0],
                                           cmap=mpl.cm.cividis,  # vmin=1, vmax=3,
                                           rasterized=True)
                plt.colorbar(pcm, ax=ax_strain)
                plt.title('total_strain_field   ')
                ax_strain.spines['top'].set_visible(False)
                ax_strain.spines['right'].set_visible(False)
                ax_strain.spines['bottom'].set_visible(False)
                ax_strain.spines['left'].set_visible(False)

                ax_strain.get_xaxis().set_visible(False)  # hides x-axis only
                ax_strain.get_yaxis().set_visible(False)  # hides y-axis only

                ax_strain = fig.add_subplot(gs[0, 0])
                pcm = ax_strain.pcolormesh(x_deformed, y_deformed, strain_fluc_field.s.mean(axis=2)[0, 1, ..., 0],
                                           cmap=mpl.cm.cividis,  # vmin=1, vmax=3,
                                           rasterized=True)
                plt.colorbar(pcm, ax=ax_strain)
                ax_strain.spines['top'].set_visible(False)
                ax_strain.spines['right'].set_visible(False)
                ax_strain.spines['bottom'].set_visible(False)
                ax_strain.spines['left'].set_visible(False)

                ax_strain.get_xaxis().set_visible(False)  # hides x-axis only
                ax_strain.get_yaxis().set_visible(False)  # hides y-axis only

                plt.title('strain_fluc_field')
                max_stress = stress_field.s.mean(axis=2)[0, 1, ..., 0].max()
                min_stress = stress_field.s.mean(axis=2)[0, 1, ..., 0].min()

                print('stress min ={}'.format(min_stress))
                print('stress max ={}'.format(max_stress))
                ax_stress = fig.add_subplot(gs[1, 1])
                pcm = ax_stress.pcolormesh(x_deformed, y_deformed, stress_field.s.mean(axis=2)[0, 1, ..., 0],
                                           cmap=mpl.cm.cividis, vmin=min_stress, vmax=max_stress,
                                           rasterized=True)
                ax_stress.spines['top'].set_visible(False)
                ax_stress.spines['right'].set_visible(False)
                ax_stress.spines['bottom'].set_visible(False)
                ax_stress.spines['left'].set_visible(False)

                ax_stress.get_xaxis().set_visible(False)  # hides x-axis only
                ax_stress.get_yaxis().set_visible(False)  # hides y-axis only
                plt.colorbar(pcm, ax=ax_stress)
                plt.title('stress   ')

                # plot constitutive tangent
                ax_tangent = fig.add_subplot(gs[0, 1])
                pcm = ax_tangent.pcolormesh(x_deformed, y_deformed, K4_ijklqyz.s.mean(axis=4)[0, 1, 0, 0, ..., 0],
                                            cmap=mpl.cm.cividis,  # vmin=0, vmax=1500,
                                            rasterized=True)
                ax_tangent.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
                # Hide all four spines individually
                ax_tangent.spines['top'].set_visible(False)
                ax_tangent.spines['right'].set_visible(False)
                ax_tangent.spines['bottom'].set_visible(False)
                ax_tangent.spines['left'].set_visible(False)

                ax_tangent.get_xaxis().set_visible(False)  # hides x-axis only
                ax_tangent.get_yaxis().set_visible(False)  # hides y-axis only
                # x_deformed[:, :], y_deformed[:, :],
                plt.colorbar(pcm, ax=ax_tangent)
                plt.title('material_data_field_C_0')

            plt.show()
        end_time = time.time()
        elapsed_time = end_time - start_time
        print("  time: ", elapsed_time)
