import numpy as np
import scipy as sc
import time
import matplotlib.pyplot as plt
import matplotlib as mpl

import matplotlib.pyplot as plt
from PySide2.examples.opengl.contextinfo import colors
from matplotlib.animation import FuncAnimation, PillowWriter

from muFFTTO import domain
from muFFTTO import solvers
from muFFTTO import microstructure_library
from mpl_toolkits import mplot3d

problem_type = 'elasticity'
discretization_type = 'finite_element'
element_type = 'linear_triangles'
formulation = 'small_strain'

domain_size = [1, 1]
nb_pix_multips = [2]  # ,3,2,
small = np.arange(0., .1, 0.005)
middle = np.arange(0.1, 0.9, 0.03)

large = np.arange(0.9, 1.0 + 0.005, 0.005)
ratios = np.concatenate((small, middle, large))
ratios = np.arange(0., 1.1, 0.2)
ratios = np.arange(0., 1.1, 0.2)
ratios = np.arange(100)

nb_it = np.zeros((len(nb_pix_multips), ratios.size), )
nb_it_combi = np.zeros((len(nb_pix_multips), ratios.size), )
nb_it_Jacobi = np.zeros((len(nb_pix_multips), ratios.size), )

# nb_it = np.random.rand(len(nb_pix_multips), ratios.size )
# nb_it_combi = np.random.rand(len(nb_pix_multips), ratios.size )
# # nb_it_Jacobi = np.random.rand(len(nb_pix_multips), ratios.size )
# nb_it = np.array([[  25., 1000., 1000.,  995.,  932.,  898.,  860.,  824.,  799.,  781.,  757.,  733.,
#                     722.,  708.,  692.,  677.,  659.,  645.,  634.,  624.,  616.,  560.,  512.,  477.,
#                     448.,  423.,  403.,  384.,  369.,  354.,  351.,  339.,  330.,  320.,  311.,  303.,
#                     295.,  288.,  282.,  276.,  270.,  265.,  260.,  255.,  251.,  247.,  243.,  241.,
#                     241.,  242.,  242.,  238.,  238.,  237.,  240.,  235.,  235.,  235.,  237.,  237.,
#                     233.,  232.,  231.,  232.,  233.,  241.,  261.,  650.,]] )
# nb_it_combi = np.array([[223., 256., 260., 261., 262., 263., 264., 264., 265., 265., 266., 266., 266., 267.,
#                          267., 267., 267., 267., 268., 268., 269., 271., 273., 274., 276., 277., 278., 282.,
#                          285., 286., 289., 291., 293., 294., 297., 300., 301., 303., 304., 306., 306., 307.,
#                          308., 310., 312., 311., 322., 322., 325., 326., 326., 327., 328., 330., 330., 332.,
#                          332., 333., 334., 336., 339., 343., 346., 351., 348., 354., 357., 360.,]])
# nb_it_Jacobi = np.array([[103.,  68.,  65.,  63.,  62.,  59.,  58.,  60.,  59.,  58. , 58. , 58.,  57.,  57.,
#                            57.,  56.,  56.,  56.,  55.,  55.,  55.,  52.,  53.,  52. , 50. , 49.,  48.,  48.,
#                            48.,  48.,  48.,  47.,  47.,  47.,  46.,  46.,  47.,  48. , 48. , 49.,  49.,  50.,
#                            51.,  52.,  53.,  55.,  58.,  60.,  61.,  62.,  62.,  64. , 64. , 64.,  65.,  66.,
#                            66.,  67.,  68.,  69.,  71.,  72.,  74.,  75.,  77.,  79. , 82. , 91.,]])
#
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
#
#
# # Plot each line with a different z offset
# for i in np.arange(len(nb_pix_multips)):
#     ax.plot(ratios,  nb_pix_multips[i]*32, zs=nb_it[i],label='DGO 1', color='blue')
#     ax.plot(ratios,  nb_pix_multips[i]*32,zs=nb_it_combi[i], label='nb_it_combi 1', color='red')
#     ax.plot(ratios,  nb_pix_multips[i]*32,zs=nb_it_Jacobi[i], label='nb_it_Jacobi', color='black')
#
# ax.set_xlabel('ratio: ratio*smooth + (1-ratio)*pwconst')
# ax.set_ylabel('size')
# ax.set_zlabel('# CG iterations')
# plt.legend(['DGO', 'Jacoby', 'DGO + Jacoby' ])
# plt.show()


for kk in np.arange(np.size(nb_pix_multips)):
    nb_pix_multip = nb_pix_multips[kk]
    number_of_pixels = (nb_pix_multip * 32, nb_pix_multip * 32)

    my_cell = domain.PeriodicUnitCell(domain_size=domain_size,
                                      problem_type=problem_type)

    discretization = domain.Discretization(cell=my_cell,
                                           nb_of_pixels_global=number_of_pixels,
                                           discretization_type=discretization_type,
                                           element_type=element_type)
    start_time = time.time()

    # set macroscopic gradient
    macro_gradient = np.array([[1.0, 0], [0, 1.0]])

    # create material data field
    K_0, G_0 = 1, 0.5  # domain.get_bulk_and_shear_modulus(E=1, poison=0.2)

    # identity tensor                                               [single tensor]
    ii = np.eye(2)

    shape = tuple((number_of_pixels[0] for _ in range(2)))


    def expand(arr):
        new_shape = (np.prod(arr.shape), np.prod(shape))
        ret_arr = np.zeros(new_shape)
        ret_arr[:] = arr.reshape(-1)[:, np.newaxis]
        return ret_arr.reshape((*arr.shape, *shape))


    # identity tensors                                            [grid of tensors]
    I = ii
    I4 = np.einsum('il,jk', ii, ii)
    I4rt = np.einsum('ik,jl', ii, ii)
    I4s = (I4 + I4rt) / 2.

    elastic_C_1 = domain.get_elastic_material_tensor(dim=discretization.domain_dimension,
                                                     K=K_0,
                                                     mu=G_0,
                                                     kind='linear')

    material_data_field_C_0 = np.einsum('ijkl,qxy->ijklqxy', elastic_C_1,
                                        np.ones(np.array([discretization.nb_quad_points_per_pixel,
                                                          *discretization.nb_of_pixels])))

    refmaterial_data_field_I4s = np.einsum('ijkl,qxy->ijklqxy', I4s,
                                           np.ones(np.array([discretization.nb_quad_points_per_pixel,
                                                             *discretization.nb_of_pixels])))

    print('elastic tangent = \n {}'.format(domain.compute_Voigt_notation_4order(elastic_C_1)))

    # material distribution
    # 'sine_wave',
    phase_field_smooth = microstructure_library.get_geometry(nb_voxels=discretization.nb_of_pixels,
                                                             microstructure_name='circle_inclusion',
                                                             coordinates=discretization.fft.coords)
    phase_field_smooth = np.abs(phase_field_smooth)
    # phase_field = np.random.rand(*discretization.get_scalar_sized_field().shape)  # set random distribution#
    load_micro = True
    if load_micro:
        phase_field_smooth_32 = np.load(
            '../experiments/exp_data/lbfg_muFFTTO_elasticity_exp_2D_elasticity_TO_indre_3exp_N32_E_target_0.15_Poisson_-0.5_Poisson0_0.0_w4.0_eta0.0203_p2_bounds=False_FE_NuMPI6_nb_load_cases_3_energy_objective_False_random_True_it20.npy',
            allow_pickle=True)

        phase_field_smooth_32 = np.power(phase_field_smooth_32, 2)

        phase_field_smooth = sc.ndimage.zoom(phase_field_smooth_32, zoom=nb_pix_multip, order=1)

    geometry_ID = 'square_inclusion'  # 'square_inclusion'#,'random_distribution' sine_wave
    phase_field_pwconst = microstructure_library.get_geometry(nb_voxels=discretization.nb_of_pixels,
                                                              microstructure_name=geometry_ID,  # 'circle_inclusions',
                                                              coordinates=discretization.fft.coords)
    # scaling to 1 - 1e7
    scaling = False
    if scaling:
        phase_field_pwconst = phase_field_pwconst / np.min(phase_field_smooth)
        phase_field_smooth = phase_field_smooth / np.min(phase_field_smooth)
    # phase_field_pwconst[phase_field_pwconst>=0.5]=1
    # phase_field_pwconst[phase_field_pwconst<0.5]=0

    # phase = 1 * np.ones(number_of_pixels)
    inc_contrast = 0.

    # nb_it=[]
    # nb_it_combi=[]
    # nb_it_Jacobi=[]
    phase_field = phase_field_smooth


    def apply_smoother(phase):
        # Define a 2D smoothing kernel
        kernel = np.array([[0.0625, 0.125, 0.0625],
                           [0.125, 0.25, 0.125],
                           [0.0625, 0.125, 0.0625]])

        # Apply convolution for smoothing
        smoothed_arr = sc.signal.convolve2d(phase, kernel, mode='same', boundary='wrap')
        return smoothed_arr

fourier_ = np.array([159.,  154., 174., 161., 170., 187., 194., 201., 206., 217., 218., 223., 236., 234.,
                     235., 243., 232., 202., 174., 155., 141., 124., 112., 102., 92., 81., 76., 70.,
                     66., 59., 56., 53., 49., 47., 44., 42., 39., 37., 36., 35., 33., 32.,
                     31., 30., 29., 28., 27., 26., 25., 24., 24., 23., 22., 22., 21., 21.,
                     20., 20., 19., 19., 19., 18., 18., 18., 17., 17., 17., 17., 16., 16.,
                     16., 16., 15., 15., 15., 15., 15., 14., 14., 14., 14., 14., 14., 13.,
                     13., 13., 13., 13., 13., 13., 12., 12., 12., 12., 12., 12., 12., 12.,
                     12., 12.])


Fem_Green = np.array([10.,21., 44., 48., 55., 58., 60., 62., 71., 66., 72., 68., 74., 76., 72., 74., 72., 73.,
                      69.,66., 61., 59., 55., 51., 47., 44., 40., 38., 34., 32., 30., 28., 27., 26., 24., 23.,
                      22.,21., 20., 20., 19., 18., 18., 17., 16., 16., 16., 15., 15., 14., 14., 14., 13., 13.,
                      13.,12., 12., 12., 12., 12., 11., 11., 11., 11., 11., 10., 10., 10., 10., 10., 10., 10.,
                      9. ,9. , 9. , 9. , 9. , 9. , 9. , 9. , 9. , 9. , 8. , 8. , 8. , 8. , 8. , 8. , 8. , 8. ,
                      8. ,8. , 8. , 8. , 8. , 8. , 8. , 7. , 7. , 7.])
Fem_Jacobi= np.array([79., 77., 75., 74., 74., 73., 72., 72., 71., 70., 69., 69., 68., 67., 67., 66., 66., 66.,
                      66., 64., 65., 64., 64., 64., 64., 64., 64., 64., 64., 63., 63., 63., 63., 63., 62., 62.,
                      62., 62., 62., 62., 61., 61., 61., 60., 60., 60., 59., 59., 59., 59., 58., 58., 58., 58.,
                      58., 57., 57., 57., 57., 56., 56., 56., 56., 56., 55., 55., 55., 55., 55., 54., 54., 54.,
                      54., 53., 53., 53., 53., 53., 52., 52., 52., 52., 52., 51., 51., 51., 51., 51., 50., 50.,
                      50., 50., 49., 49., 49., 49., 48., 48., 48., 48.])
Fem_Combi= np.array( [21. ,15. ,13., 12., 10., 10.,  9.,  9.,  8.,  8.,  8.,  8.,  8.,  7.,  7.,  7.,  7.,  7.,
                       6. , 6. , 6.,  6.,  6.,  6.,  6.,  6.,  6.,  6.,  6.,  5.,  5.,  5.,  5.,  5.,  5.,  5.,
                       5. , 5. , 5.,  4.,  4.,  4.,  4.,  4.,  4.,  4.,  4.,  4.,  4.,  4.,  4.,  4.,  4.,  4.,
                       4. , 4. , 4.,  4.,  4.,  4.,  4.,  4.,  4.,  4.,  4.,  4.,  4.,  4.,  4.,  4.,  4.,  4.,
                       4. , 4. , 4.,  4.,  4.,  4.,  4.,  4.,  4.,  4.,  4.,  4.,  4.,  4.,  4.,  4.,  4.,  4.,
                       4. , 4. , 4.,  4.,  4.,  4.,  4.,  4.,  3.,  3.])



ratios = np.arange(100)



plot_evolion = True
if plot_evolion:
    for nb_tiles in [1, ]:
        # fig = plt.figure()

        #
        # fig, axs = plt.subplots(nrows=2, ncols=2,
        #                         figsize=(6, 6)  )
        fig = plt.figure()
        gs = fig.add_gridspec(2, 2)
        ax1 = fig.add_subplot(gs[0, 0])
        ax3 = fig.add_subplot(gs[0, 1])
        ax2 = fig.add_subplot(gs[1, :])
        # axs[0] = plt.axes(xlim=(0, nb_tiles * N), ylim=(0, nb_tiles * N))
        ax1.imshow(phase_field, cmap=mpl.cm.Greys, vmin=1e-4, vmax=1)
        ax3.semilogy(phase_field[:, phase_field.shape[0] // 2], linewidth=0)
        # ax3.plot(phase_field[:,phase_field.shape[0]//2], linewidth=0)

        print(ratios)

        print(nb_it)
        ax2.plot(ratios, fourier_, label='nb_it_Laplace', linewidth=0)

        # axs[1].plot(xopt.f.num_iteration_.transpose()[::3], 'w'  , linewidth=0)
        # axs[1].plot(xopt3.f.num_iteration_.transpose(), "b", label='Jacoby', linewidth=0)
        # axs[1].plot(xopt.f.num_iteration_.transpose(), "k", label='DGO + Jacoby', linewidth=0)
        # legend = plt.legend()
        # Animation function to update the image
        # ax2.set_xlabel('')
        ax2.set_ylabel('# PCG iterations')


        def update(i):
            ratio = ratios[i]
            phase_field = microstructure_library.get_geometry(nb_voxels=discretization.nb_of_pixels,
                                                              microstructure_name='circle_inclusion',
                                                              coordinates=discretization.fft.coords)
            phase_field += 1e-4
            for a in np.arange(i):
                phase_field = apply_smoother(phase_field)
            # min_val = np.min(phase_field)
            # max_val = np.max(phase_field)
            # phase_field = 1e-4 + (phase_field - min_val) * (1 - 1e-4) / (max_val - min_val)
            # phase_field = ratio * phase_field_smooth + (1 - ratio) * phase_field_pwconst

            ax1.clear()
            ax1.imshow(phase_field, cmap=mpl.cm.Greys, vmin=1e-4, vmax=1)
            ax1.set_title(r'Density $\rho$', wrap=True)
            #: {np.max(phase_field)/np.min(phase_field):.1e}  \n'                          f'  min = {np.min(phase_field):.1e}
            ax3.clear()
            ax3.semilogy(np.abs(phase_field[:, phase_field.shape[0] // 2]), linewidth=1)
            # ax3.plot(phase_field[:, phase_field.shape[0] // 2], linewidth=1)
            ax3.set_ylim([1e-4, 1.1])

            # ax3.set_ylim([1e-4, 1])

            ax3.set_title(f'Cross section')

            ax2.plot(ratios[0:i + 1], Fem_Green[ 0:i + 1], 'r', label='FEM: Green', linewidth=1)
            # axs[1].plot(xopt2.f.num_iteration_.transpose()[1:3*i+1:3],"r", label='DGO ',linewidth=1)
            # axs[1].plot(xopt2.f.num_iteration_.transpose()[2:3*i+2:3],"r", label='DGO ',linewidth=1)

            ax2.plot(ratios[0:i + 1], Fem_Jacobi[ 0:i + 1], "b", label='FEM: Jacobi', linewidth=1)
            ax2.plot(ratios[0:i + 1], Fem_Combi[ 0:i + 1], "k", label='FEM: Green + Jacobi', linewidth=1)
            ax2.plot(ratios[0:i + 1], fourier_[0:i + 1], "g", label='Fourier: Green ', linewidth=1)

            # axs[1].legend()
            plt.legend(['', 'FEM: Green', 'FEM: Jacobi', 'FEM: Green + Jacobi', 'Fourier: Green'])
            # plt.legend([ '', 'Green', 'Jacobi'  ])

            # img.set_array(xopt_it)


        # Create animation
        # ani = FuncAnimation(fig, update, frames=xopt.f.norms_f.size - 1, blit=False)

        ani = FuncAnimation(fig, update, frames=ratios.size, blit=False)
        #ani = FuncAnimation(fig, update, frames=100, blit=False)

        # axs[1].legend()middlemiddle
        # Save as a GIF
        ani.save(f"./figures/movie_{number_of_pixels[0]}_fix_comparison_{geometry_ID}_circle_inc_to_smooth_semiloplots3.gif",
                 writer=PillowWriter(fps=4))

    plt.show()
