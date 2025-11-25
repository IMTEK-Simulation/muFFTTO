import os

import numpy as np
import scipy as sc
import time
from mpi4py import MPI
from NuMPI.IO import save_npy, load_npy
import matplotlib as mpl
import matplotlib.pyplot as plt

from muFFTTO import domain
from muFFTTO import solvers
from muFFTTO import microstructure_library

script_name = 'surface_deformation'

file_folder_path = os.path.dirname(os.path.realpath(__file__))  # script directory
if not os.path.exists(file_folder_path):
    os.makedirs(file_folder_path)
data_folder_path = file_folder_path + '/exp_data/' + script_name + '/'
if not os.path.exists(data_folder_path):
    os.makedirs(data_folder_path)
figure_folder_path = file_folder_path + '/figures/' + script_name + '/'
if not os.path.exists(figure_folder_path):
    os.makedirs(figure_folder_path)

problem_type = 'elasticity'
discretization_type = 'finite_element'
element_type = 'linear_triangles'
formulation = 'small_strain'

domain_size = [2, 1]
number_of_pixels = (256, 256)

my_cell = domain.PeriodicUnitCell(domain_size=domain_size,
                                  problem_type=problem_type)

discretization = domain.Discretization(cell=my_cell,
                                       nb_of_pixels_global=number_of_pixels,
                                       discretization_type=discretization_type,
                                       element_type=element_type)
start_time = time.time()
print(f'{MPI.COMM_WORLD.rank:6} {MPI.COMM_WORLD.size:6} {str(discretization.fft.nb_domain_grid_pts):>15} '
      f'{str(discretization.fft.nb_subdomain_grid_pts):>15} {str(discretization.fft.subdomain_locations):>15}')
# set macroscopic gradient
step=0
load='tension' #  'tension' compression
if load=='compression':
    max_load=-0.4
elif load=='tension':
    max_load=0.4

file_data_name = f'N{number_of_pixels[0]}_DS{domain_size[0]}{domain_size[1]}_step{step}_{load}_max_{max_load}_'
x = np.linspace(start=0, stop=domain_size[0], num=number_of_pixels[0])
y = np.linspace(start=0, stop=domain_size[1], num=number_of_pixels[1])
X, Y = np.meshgrid(x, y, indexing='ij')
np.save(data_folder_path + file_data_name + 'X' + f'.npy', X)
np.save(data_folder_path + file_data_name + 'Y' + f'.npy', Y)



for load_inc in np.linspace(0, max_load, 50)[1:]:
    step += 1
    print(f'step {step}')
    macro_gradient = np.array([[ load_inc, 0], [0, 0.]])

    # create material data field
    # K_0, G_0 = 1, 0.5 #domain.get_bulk_and_shear_modulus(E=1, poison=0.2)
    K_0, G_0 = domain.get_bulk_and_shear_modulus(E=1, poison=0.32)

    elastic_C_1 = domain.get_elastic_material_tensor(dim=discretization.domain_dimension,
                                                     K=K_0,
                                                     mu=G_0,
                                                     kind='linear')

    material_data_field_C_0 = np.einsum('ijkl,qxy->ijklqxy', elastic_C_1,
                                        np.ones(np.array([discretization.nb_quad_points_per_pixel,
                                                          *discretization.nb_of_pixels])))

    #print('elastic tangent = \n {}'.format(domain.compute_Voigt_notation_4order(elastic_C_1)))

    # material distribution
    geometry_ID = '2_circles'  # 'circle_inclusion'# 'circles'#2_circles
    phase_field_00 = microstructure_library.get_geometry(nb_voxels=discretization.nb_of_pixels,
                                                         microstructure_name=geometry_ID,
                                                         coordinates=discretization.fft.coords)
    folder_name = 'experiments/exp_data/'  # s'exp_data/'
    # phase_field = np.random.rand(*discretization.get_scalar_sized_field().shape)  # set random distribution#
    # add stiff inclusion
    # mask_lt=phase_field_00[:number_of_pixels[0]//2,number_of_pixels[1]//2:]
    # mask_lt[mask_lt==0]=2
    # phase_field_00[:number_of_pixels[0]//2,number_of_pixels[1]//2:]=mask_lt
    #
    # mask_rb=phase_field_00[number_of_pixels[0]//2:,:number_of_pixels[1]//2]
    # mask_rb[mask_rb==0]=2
    # phase_field_00[number_of_pixels[0]//2:,:number_of_pixels[1]//2]=mask_rb

    # mask_lt=phase_field_00[number_of_pixels[0]//3:2*number_of_pixels[0]//3,number_of_pixels[1]//2:]
    # mask_lt[mask_lt==0]=0
    # phase_field_00[number_of_pixels[0]//3:2*number_of_pixels[0]//3,number_of_pixels[1]//2:]=mask_lt


    phase_field_00 += 0.0
    phase_field_00[:, :5] = 0.0
    # phase_field_00[0,:]=0


    phase_field = discretization.get_scalar_sized_field()
    phase_field[0, 0] = phase_field_00


    #fig = plt.figure(figsize=(8.3, 5.0))
    # gs = fig.add_gridspec(2, 1, hspace=0.2, wspace=0.1, width_ratios=[1],
    #                       height_ratios=[1, 1])
    # ax_original = fig.add_subplot(gs[0, 0])
    # pcm = ax_original.pcolormesh(X, Y, phase_field[0, 0], cmap=mpl.cm.Greys, vmin=0, vmax=2, linewidth=0,
    #                              rasterized=True)
    # ax_original.set_aspect('equal')

    # phase_field[0,0]=phase_field[0,0]/np.min(phase_field[0,0])

    # np.save('geometry_jacobi.npy', np.power(phase_field_l, 2),)
    # sc.io.savemat('geometry_jacobi.mat', {'data':  np.power(phase_field_l, 2)})

    material_data_field_C_0_rho = material_data_field_C_0[..., :, :] * phase_field
    # Set up right hand side
    macro_gradient_field = discretization.get_macro_gradient_field(macro_gradient)

    # Solve mechanical equilibrium constrain
    rhs = discretization.get_rhs(material_data_field_C_0_rho, macro_gradient_field)

    K_fun = lambda x: discretization.apply_system_matrix(material_data_field_C_0_rho, x,
                                                         formulation='small_strain')
    # M_fun = lambda x: 1 * x
    # K= discretization.get_system_matrix(material_data_field=material_data_field_C_0_rho)

    preconditioner = discretization.get_preconditioner_NEW(
        reference_material_data_field_ijklqxyz=material_data_field_C_0)

    M_fun = lambda x: discretization.apply_preconditioner_NEW(preconditioner_Fourier_fnfnqks=preconditioner,
                                                              nodal_field_fnxyz=x)

    # K_diag_alg = discretization.get_preconditioner_Jacoby_fast(
    #     material_data_field_ijklqxyz=material_data_field_C_0_rho)
    #
    # M_fun = lambda x: K_diag_alg * discretization.apply_preconditioner_NEW(
    #     preconditioner_Fourier_fnfnqks=preconditioner,
    #     nodal_field_fnxyz=K_diag_alg * x)
    # #
    # M_fun = lambda x: K_diag_alg *  K_diag_alg * x

    displacement_fluctuation_field, norms = solvers.PCG(K_fun, rhs, x0=None, P=M_fun, steps=int(1000), toler=1e-6)
    nb_it_comb = len(norms['residual_rz'])
    norm_rz = norms['residual_rz'][-1]
    norm_rr = norms['residual_rr'][-1]

   # ax_deformed = fig.add_subplot(gs[1, 0])
    # linear part of displacement
    disp_linear_x = (X-domain_size[0]/2) * macro_gradient[0, 0]
    disp_linear_y = Y * macro_gradient[1, 1]

    # displacement in voids should be zero
    displacement_fluctuation_field[:, 0, :, :5] = 0.0

    x_deformed = X +disp_linear_x+ displacement_fluctuation_field[0, 0]
    y_deformed = Y +disp_linear_y+ displacement_fluctuation_field[1, 0]
    # pcm = ax_deformed.pcolormesh(x_deformed, y_deformed, phase_field[0, 0], cmap=mpl.cm.Greys,
    #                              rasterized=True)
    # ax_deformed.set_ylim(0.2,1.2)
    # ax_deformed.set_xlim(-1.0, 3.)
    # ax_deformed.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    # # Hide all four spines individually
    # ax_deformed.spines['top'].set_visible(False)
    # ax_deformed.spines['right'].set_visible(False)
    # ax_deformed.spines['bottom'].set_visible(False)
    # ax_deformed.spines['left'].set_visible(False)
    #
    # ax_deformed.get_xaxis().set_visible(False)  # hides x-axis only
    # ax_deformed.get_yaxis().set_visible(False)  # hides y-axis only
    #
    # plt.show()

    file_data_name=f'N{number_of_pixels[0]}_DS{domain_size[0]}{domain_size[1]}_step{step}_{load}_max_{max_load}_'

    np.save(data_folder_path +file_data_name+'phase_field'+f'.npy', phase_field[0, 0])
    np.save(data_folder_path +file_data_name+'x_deformed'+f'.npy', x_deformed)
    np.save(data_folder_path +file_data_name+'y_deformed'+f'.npy', y_deformed)
    np.save(data_folder_path +file_data_name+'macro_gradient'+f'.npy', macro_gradient)
