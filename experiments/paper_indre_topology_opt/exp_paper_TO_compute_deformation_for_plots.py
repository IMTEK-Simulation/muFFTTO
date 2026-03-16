import os

import numpy as np
import scipy as sc
import time
import sys
import matplotlib as mpl
from matplotlib.animation import FFMpegWriter

from matplotlib import pyplot as plt

sys.path.append('..')  # Add parent directory to path

from mpi4py import MPI
from NuMPI.IO import save_npy, load_npy

from muFFTTO import domain
from muFFTTO import solvers
from muFFTTO import microstructure_library


if MPI.COMM_WORLD.size == 1:
    plot = True
    compute = False
else:
    plot = False
    compute = True
grid_type ='square' # 'hex'  # 'square'

problem_type = 'elasticity'
discretization_type = 'finite_element'
formulation = 'small_strain'

if grid_type == 'hex':
    element_type = 'linear_triangles_tilled'

    domain_size = [1, np.sqrt(3) / 2]
elif grid_type == 'square':
    element_type = 'linear_triangles'

    domain_size = [1, 1]
nb_tiles = 1
number_of_pixel_PUC = 1024 # this is resolution of original unit cell
number_of_pixels = 2 * (nb_tiles*number_of_pixel_PUC,)# this is resolution with repeated images

my_cell = domain.PeriodicUnitCell(domain_size=domain_size,
                                  problem_type=problem_type)

discretization = domain.Discretization(cell=my_cell,
                                       nb_of_pixels_global=number_of_pixels,
                                       discretization_type=discretization_type,
                                       element_type=element_type)
start_time = time.time()
print(f'{MPI.COMM_WORLD.rank:6} {MPI.COMM_WORLD.size:6} {str(discretization.fft.nb_domain_grid_pts):>15} '
      f'{str(discretization.fft.nb_subdomain_grid_pts):>15} {str(discretization.fft.subdomain_locations):>15}')

K_0, G_0 = 1, 0.5

elastic_C_1 = domain.get_elastic_material_tensor(dim=discretization.domain_dimension,
                                                 K=K_0,
                                                 mu=G_0,
                                                 kind='linear')
print(domain.compute_Voigt_notation_4order(elastic_C_1))

material_data_field_C_0 = discretization.get_material_data_size_field_mugrid(name='elastic_tensor')

# populate the field with C_1 material
material_data_field_C_0.s = np.einsum('ijkl,qxy->ijklqxy', elastic_C_1,
                                      np.ones(np.array([discretization.nb_quad_points_per_pixel,
                                                        *discretization.nb_of_pixels])))
if MPI.COMM_WORLD.rank == 0:
    print('elastic tangent = \n {}'.format(domain.compute_Voigt_notation_4order(elastic_C_1)))

# material distribution
phase_field = discretization.get_scalar_field(name='phase_field')

eta_mult = 0.01
weight = 3.0
N = number_of_pixel_PUC
cg_tol_exponent = 8
soft_phase_exponent = 5
random_init = False
poison_target = -0.5

if grid_type == 'hex':
    script_name = 'exp_paper_TO_exp_4_hexa' + f'_random_{random_init}' + f'_N_{N}' + f'_cgtol_{cg_tol_exponent}' + f'_soft_{soft_phase_exponent}'
elif grid_type == 'square':
    script_name = 'exp_paper_TO_exp_4_square' + f'_random_{random_init}' + f'_N_{N}' + f'_cgtol_{cg_tol_exponent}' + f'_soft_{soft_phase_exponent}'

preconditioner_type_data = "Green_Jacobi" #Green_Jacobi
preconditioner_type = "Green" #Green_Jacobi

file_folder_path = os.path.dirname(os.path.realpath(__file__))  # script directory
data_folder_path = file_folder_path + '/exp_data/' + script_name + '/'
figure_folder_path = file_folder_path + '/figures/' + script_name + '/'
if not os.path.exists(figure_folder_path):
    os.makedirs(figure_folder_path)

name = data_folder_path + f'{preconditioner_type_data}' + f'_eta_{eta_mult}' + f'_w_{weight:.1f}' + f'_p_{poison_target}' + '_final' + f'.npy'
name_tilled = data_folder_path + f'{preconditioner_type_data}' + f'_eta_{eta_mult}' + f'_w_{weight:.1f}' + f'_p_{poison_target}' + f'_final_tilled{nb_tiles}' + f'.npy'
name_log =  data_folder_path + f'{preconditioner_type_data}' + f'_eta_{eta_mult}' + f'_w_{weight:.1f}' + f'_p_{poison_target}' + '_final' + f'_log.npz'
# Load, tile, and save in serial computation only (rank 0)
if MPI.COMM_WORLD.rank == 0:
    if not os.path.exists(name_tilled) and os.path.exists(name):
        if not os.path.exists(data_folder_path):
            os.makedirs(data_folder_path)

        # Load original unit cell
        phase_field_original = np.load(name)

        # Tile the field
        phase_field_tilled = np.tile(phase_field_original, (nb_tiles, nb_tiles))

        # Save tiled field
        np.save(name_tilled, phase_field_tilled)
        print(f'Tiled phase field saved to: {name_tilled}')

# Wait for rank 0 to finish tiling and saving
MPI.COMM_WORLD.Barrier()

# Now all ranks can load the tiled field
if os.path.exists(name_tilled):
    phase_field.s[0, 0] = load_npy(name_tilled,
                                   subdomain_locations=tuple(discretization.subdomain_locations_no_buffers),
                                   nb_subdomain_grid_pts=tuple(discretization.nb_of_pixels),
                                   components_are_leading=True,
                                   comm=MPI.COMM_WORLD)
    info_ = np.load(name_log, allow_pickle=True)
    C_eff_ij=info_.f.homogenized_C_ijkl


# remove layer
mask_free_layer = np.zeros_like(phase_field.s, dtype=bool)


dx = discretization.fft.coords[0, 1, 0] - discretization.fft.coords[0, 0, 0]
mask_free_layer[0, 0,discretization.fft.icoords[1]== 0] = 1
mask_free_layer[0, 0,discretization.fft.icoords[1]== 1] = 1
mask_free_layer[0, 0, discretization.fft.icoords[1]== number_of_pixels[0]-1] = 1


# phase_field.s[mask_free_layer]=0
phase_field.s[phase_field.s >= 0.5] = 1
phase_field.s[phase_field.s < 0.5] = 0
discretization.fft.communicate_ghosts(phase_field)

if compute:
    inc_contrast = 0.

    # Material data in quadrature points
    elastic_C_void = elastic_C_1 * 1e-5

    # Phase field  in quadrature points
    phase_field_at_quad_poits_1qxyz = discretization.get_quad_field_scalar(
        name='phase_field_at_quads_in_objective_function_multiple_load_cases')
    discretization.apply_N_operator_mugrid(phase_field, phase_field_at_quad_poits_1qxyz)
    material_data_field_C_0.s = (elastic_C_1 - elastic_C_void)[..., np.newaxis, np.newaxis, np.newaxis] * \
                                np.power(phase_field_at_quad_poits_1qxyz.s, 2)[0, 0, :, ...] + \
                                elastic_C_void[..., np.newaxis, np.newaxis, np.newaxis]


    def K_fun(x, Ax):
        discretization.apply_system_matrix_mugrid(material_data_field=material_data_field_C_0,
                                                  input_field_inxyz=x,
                                                  output_field_inxyz=Ax,
                                                  formulation='small_strain')
        discretization.fft.communicate_ghosts(Ax)


    preconditioner = discretization.get_preconditioner_Green_mugrid(reference_material_data_ijkl=elastic_C_1)


    def M_fun_Green(x, Px):
        """
        Function to compute the product of the Preconditioner matrix with a vector.
        The Preconditioner is represented by the convolution operator.
        """
        discretization.fft.communicate_ghosts(x)
        discretization.apply_preconditioner_mugrid(preconditioner_Fourier_fnfnqks=preconditioner,
                                                   input_nodal_field_fnxyz=x,
                                                   output_nodal_field_fnxyz=Px)
        # Px.s[...] = 1 * x.s[...]
        # print()


    # preconditioner_type = 'Green'
    if preconditioner_type == 'Green':
        M_fun = M_fun_Green
    elif preconditioner_type == 'Jacobi':
        K_diag_alg = discretization.get_preconditioner_Jacobi_mugrid(
            material_data_field_ijklqxyz=material_data_field_C_0)


        def M_fun_Jacobi(x, Px):
            Px.s = K_diag_alg.s * K_diag_alg.s * x.s
            discretization.fft.communicate_ghosts(Px)


        M_fun = M_fun_Jacobi

    elif preconditioner_type == 'Green_Jacobi':
        K_diag_alg = discretization.get_preconditioner_Jacobi_mugrid(
            material_data_field_ijklqxyz=material_data_field_C_0)


        def M_fun_Green_Jacobi(x, Px):
            discretization.fft.communicate_ghosts(x)
            x_jacobi_temp = discretization.get_unknown_size_field(name='x_jacobi_temp')

            x_jacobi_temp.s = K_diag_alg.s * x.s
            discretization.apply_preconditioner_mugrid(
                preconditioner_Fourier_fnfnqks=preconditioner,
                input_nodal_field_fnxyz=x_jacobi_temp,
                output_nodal_field_fnxyz=Px)

            Px.s = K_diag_alg.s * Px.s
            discretization.fft.communicate_ghosts(Px)


        M_fun = M_fun_Green_Jacobi

    for inc_index in range(0,10):
        load_increment = inc_index / 10
        # Set up right hand side
        # set macroscopic gradient
        macro_gradient = np.array([[0.2, 0.0],
                                   [0.0, 0.0]]) * load_increment
        # compute stress relaxation
        macro_gradient_voigt=np.array([macro_gradient[0,0],macro_gradient[1,1],macro_gradient[1,0]+macro_gradient[0,1]])* load_increment

        macro_stress_voigt= C_eff_ij @ macro_gradient_voigt
        S_eff_ij=np.linalg.inv(C_eff_ij)
        macro_stress_voigt_x=np.zeros_like(macro_stress_voigt)
        macro_stress_voigt_x[1]=macro_stress_voigt[1]
        strain_correction=S_eff_ij @ macro_stress_voigt_x
        # add stain correction to reduce stress in y direction

        macro_gradient[1,1]=-strain_correction[1]




        macro_gradient_field = discretization.get_gradient_size_field(name='macro_gradient_field')
        discretization.get_macro_gradient_field_mugrid(macro_gradient_ij=macro_gradient,
                                                       macro_gradient_field_ijqxyz=macro_gradient_field)

        # Solve mechanical equilibrium constrain
        rhs_field = discretization.get_unknown_size_field(name='rhs_field')
        discretization.get_rhs_mugrid(material_data_field_ijklqxyz=material_data_field_C_0,
                                      macro_gradient_field_ijqxyz=macro_gradient_field,
                                      rhs_inxyz=rhs_field)


        def callback(it, x, r, p, z, stop_crit_norm):
            """
            Callback function to print the current solution, residual, and search direction.
            """
            norm_of_rr = discretization.fft.communicator.sum(np.dot(r.ravel(), r.ravel()))
            if discretization.fft.communicator.rank == 0:
                print(f"{it:5} norm of residual = {norm_of_rr:.5}")


        solution_field = discretization.get_unknown_size_field(name='solution')

        solvers.conjugate_gradients_mugrid(
            comm=discretization.fft.communicator,
            fc=discretization.field_collection,
            hessp=K_fun,  # linear operator
            b=rhs_field,
            x=solution_field,
            P=M_fun,
            tol=1e-4,
            maxiter=2000,
            callback=callback,
        )

        file_data_name_it = f'_w_{weight}' + f'_p_{poison_target}' + f'_load_increment_{load_increment}' + f'_tilled{nb_tiles}'
        save_npy(data_folder_path + f'{preconditioner_type_data}' + file_data_name_it + f'.npy',
                 solution_field.s.mean(axis=1),
                 tuple(discretization.subdomain_locations_no_buffers),
                 tuple(discretization.nb_of_pixels_global), MPI.COMM_WORLD)

        _info = {}
        _info['macro_grads_corrected' ] = macro_gradient

        # compute homogenized stress field corresponding t
        homogenized_stress  = discretization.get_homogenized_stress_mugrid(
            material_data_field_ijklqxyz=material_data_field_C_0,
            displacement_field_inxyz=solution_field,
            macro_gradient_field_ijqxyz=macro_gradient_field,
            formulation='small_strain')

        _info['homogenized_stress' ] =homogenized_stress

        if MPI.COMM_WORLD.rank == 0:
            print(data_folder_path + f'{preconditioner_type_data}' + file_data_name_it + f'.npy')
            print(f'homogenized_stress = {homogenized_stress}')


    # np.save(folder_name + file_data_name+f'xopt_log.npz', xopt_FE_MPI)
        if MPI.COMM_WORLD.rank == 0:
            np.savez(data_folder_path + f'{preconditioner_type_data}' + file_data_name_it + f'_log_plotting.npz',
                     **_info)

if plot:
    if MPI.COMM_WORLD.rank == 0:
        fig = plt.figure(figsize=(11, 6.5))
        gs = fig.add_gridspec(1, 1, hspace=0.1)
        ax1 = fig.add_subplot(gs[0])
        metadata = dict(title=f'Deformation Movie w_{weight}_p_{poison_target}', artist='Junie', comment='Deformation evolution')
        writer = FFMpegWriter(fps=5, metadata=metadata)
        movie_name = figure_folder_path + f'movie_w_{weight}_p_{poison_target}.mp4'

    for inc_index in range(0,10):
        load_increment = inc_index / 10
        # load_increment=0.1
        # for inc_index in range(20):
        #     load_increment = inc_index / 20
        #     # Set up right hand side
        #     # set macroscopic gradient
        #macro_gradient = np.array([[0.5, 0], [0, 0.0]]) * load_increment
        # macro_gradient = np.array([[0.2, 0.0],
        #                            [0.0, 0.0]]) * load_increment


        file_data_name_it = f'_w_{weight}' + f'_p_{poison_target}' + f'_load_increment_{load_increment}' + f'_tilled{nb_tiles}'
        #name_tilled = data_folder_path + f'{preconditioner_type}' + f'_eta_{eta_mult}' + f'_w_{weight:.1f}' + f'_p_{poison_target}' + f'_final_tilled{nb_tiles}' + f'.npy'

        displacement = load_npy(data_folder_path + f'{preconditioner_type_data}' + file_data_name_it + f'.npy',
                                subdomain_locations=tuple(discretization.subdomain_locations_no_buffers),
                                nb_subdomain_grid_pts=tuple(discretization.nb_of_pixels),
                                components_are_leading=True,
                                comm=MPI.COMM_WORLD)

        #log_file_data_name_it = f'_w_{weight}' + f'_p_{poison_target}' + f'_load_increment_{0.2}' + f'_tilled{nb_tiles}'
        _info=np.load(data_folder_path + f'{preconditioner_type_data}' + file_data_name_it + f'_log_plotting.npz', allow_pickle=True)
        macro_gradient = _info.f.macro_grads_corrected
        #'Green_Jacobi_w_1.0_p_-0.5_load_increment_0.2_tilled1_log_plotting'

        if MPI.COMM_WORLD.rank == 0:
           # nb_tiles = 1
           #  x_ref = np.zeros([2, nb_tiles * (N) + 1, nb_tiles * (N) + 1])
           #  x_ref[0], x_ref[1] = np.meshgrid(np.linspace(0, nb_tiles, nb_tiles * (N) + 1),
           #                                   np.linspace(0, nb_tiles, nb_tiles * (N) + 1), indexing='ij')
           #  shift = 0.5 * np.linspace(0, nb_tiles, nb_tiles * (N) + 1)
            x_ref = np.zeros([2,number_of_pixels[0] + 1, number_of_pixels[1]  + 1])
            x_ref[0], x_ref[1] = np.meshgrid(np.linspace(0, 1,number_of_pixels[0]  + 1),
                                             np.linspace(0, 1, number_of_pixels[0]  + 1), indexing='ij')
            shift = 0.5 * np.linspace(0, 1,number_of_pixels[0]  + 1)

            x_coords = np.copy(x_ref)
            if grid_type == 'hex':
                # Apply shift to each row
                x_coords[0] += shift[None, :] - 2
                x_coords[1] *= np.sqrt(3) / 2

            # Add linear displacement from macro gradient x*macro_grad

            lin_disp_ixy = np.einsum('ij...,j...->i...', macro_gradient, x_coords)

            x_coords[0] += lin_disp_ixy[0]
            x_coords[1] += lin_disp_ixy[1]

            # add fluctuation of displacement
            #build a periodic displacement
            tilled_disp_x =  displacement[0]
            tilled_disp_y =  displacement[1]

            # extend to [N+1, N+1] to simulate periodicity
            nx, ny = tilled_disp_x.shape
            tilled_disp_x_ext = np.zeros((nx + 1, ny + 1))
            tilled_disp_y_ext = np.zeros((nx + 1, ny + 1))

            tilled_disp_x_ext[:-1, :-1] = tilled_disp_x
            tilled_disp_y_ext[:-1, :-1] = tilled_disp_y

            # Fill last row and column with first row and column
            tilled_disp_x_ext[-1, :-1] = tilled_disp_x[0, :]
            tilled_disp_x_ext[:-1, -1] = tilled_disp_x[:, 0]
            tilled_disp_x_ext[-1, -1] = tilled_disp_x[0, 0]

            tilled_disp_y_ext[-1, :-1] = tilled_disp_y[0, :]
            tilled_disp_y_ext[:-1, -1] = tilled_disp_y[:, 0]
            tilled_disp_y_ext[-1, -1] = tilled_disp_y[0, 0]

            x_coords[0] += tilled_disp_x_ext
            x_coords[1] += tilled_disp_y_ext

            ax1.clear()
            pcm = ax1.pcolormesh(x_coords[0], x_coords[1], np.tile(phase_field.s[0, 0], (1, 1)),
                                 shading='flat',
                                 edgecolors='none',
                                 lw=0.01,
                                 cmap=mpl.cm.Greys,
                                 vmin=0, vmax=1,
                                 rasterized=True)
            if inc_index == 0:
                fig.colorbar(pcm, ax=ax1)

            ax1.xaxis.set_ticks_position('none')
            ax1.yaxis.set_ticks_position('none')
            ax1.set_xlim(-0.5, nb_tiles+0.5 )
            ax1.set_ylim(-0.5, nb_tiles + 0.5)
            # ax1.set_aspect('equal')
            ax1.set_title(r"Poisson's ratio" + f' Load_incerment {load_increment:.1f}')

            # Add horizontal and vertical liner indicated initial shape
            for i in range(nb_tiles + 1):
                if grid_type == 'hex':
                    # Initial horizontal lines in hex grid
                    y_val = i * np.sqrt(3) / 2
                    ax1.plot([0 - 2, nb_tiles + 0.5 * nb_tiles - 2], [y_val, y_val], color='k', linestyle='--', linewidth=1, alpha=0.5)

                    # Initial vertical lines in hex grid (tilted)
                    # x_coords[0] += shift[None, :] - 2
                    # shift = 0.5 * linspace(0, nb_tiles, ...)
                    x_start = i - 2
                    x_end = i + 0.5 * nb_tiles - 2
                    ax1.plot([x_start, x_end], [0, nb_tiles * np.sqrt(3) / 2], color='k', linestyle='--', linewidth=1, alpha=0.5)

                else:
                    ax1.axhline(y=i, color='k', linestyle='--', linewidth=1, alpha=0.5)
                    ax1.axvline(x=i, color='k', linestyle='--', linewidth=1, alpha=0.5)

            if inc_index == 0:
                writer.setup(fig, movie_name, dpi=100)

            print(f'Adding frame for load increment: {load_increment:.2f}')
            writer.grab_frame()

    if MPI.COMM_WORLD.rank == 0:
        writer.finish()
        plt.close(fig)
        print(f'Movie saved: {movie_name}')
