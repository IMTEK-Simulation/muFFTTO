import numpy as np
import scipy as sc
import time
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

import matplotlib.pyplot as plt
from PySide2.examples.opengl.contextinfo import colors
from matplotlib.animation import FuncAnimation, PillowWriter

from experiments.exp_2D_elasticity_TO_indre_1exp import load_case
from muFFTTO import domain
from muFFTTO import solvers
from muFFTTO import microstructure_library
from mpl_toolkits import mplot3d


problem_type = 'elasticity'
discretization_type = 'finite_element'
element_type = 'linear_triangles'
formulation = 'small_strain'


domain_size = [1, 1]
nb_pix_multips=[1]#,
small = np.arange(0., .1, 0.005)
middle = np.arange(0.1, 0.9, 0.03)

large = np.arange(0.9, 1.0 + 0.005, 0.005)
ratios = np.concatenate((small, middle, large))
ratios = np.arange(0., 1.1, 0.2)
ratios = np.arange(0.6, 1.1, 0.2)

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


for nb_pix_multip in nb_pix_multips:

    number_of_pixels = (nb_pix_multip*32, nb_pix_multip*32)

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
    K_0, G_0 = 1, 0.5 #domain.get_bulk_and_shear_modulus(E=1, poison=0.2)

    elastic_C_1 = domain.get_elastic_material_tensor(dim=discretization.domain_dimension,
                                                     K=K_0,
                                                     mu=G_0,
                                                     kind='linear')

    material_data_field_C_0 = np.einsum('ijkl,qxy->ijklqxy', elastic_C_1,
                                        np.ones(np.array([discretization.nb_quad_points_per_pixel,
                                                          *discretization.nb_of_pixels])))

    print('elastic tangent = \n {}'.format(domain.compute_Voigt_notation_4order(elastic_C_1)))

    # material distribution
    #'sine_wave',
    phase_field_smooth = microstructure_library.get_geometry(nb_voxels=discretization.nb_of_pixels,
                                                      microstructure_name='tanh',
                                                      coordinates=discretization.fft.coords)
    phase_field_smooth=np.abs(phase_field_smooth)
    #phase_field = np.random.rand(*discretization.get_scalar_sized_field().shape)  # set random distribution#
    load_micro=False
    if load_micro:
        phase_field_smooth_32 = np.load('../experiments/exp_data/lbfg_muFFTTO_elasticity_exp_2D_elasticity_TO_indre_3exp_N32_E_target_0.15_Poisson_-0.5_Poisson0_0.0_w4.0_eta0.0203_p2_bounds=False_FE_NuMPI6_nb_load_cases_3_energy_objective_False_random_True_it20.npy', allow_pickle=True)

        phase_field_smooth_32=np.power(phase_field_smooth_32, 2)

        phase_field_smooth = sc.ndimage.zoom(phase_field_smooth_32, zoom=nb_pix_multip, order=1)


    geometry_ID = 'square_inclusion' #'square_inclusion'#,'random_distribution' sine_wave
    phase_field_pwconst = microstructure_library.get_geometry(nb_voxels=discretization.nb_of_pixels,
                                                      microstructure_name=geometry_ID,#'circle_inclusions',
                                                      coordinates=discretization.fft.coords)
    # scaling to 1 - 1e7
    scaling=False
    if scaling:
        phase_field_pwconst = phase_field_pwconst / np.min(phase_field_smooth)
        phase_field_smooth = phase_field_smooth / np.min(phase_field_smooth)
    # phase_field_pwconst[phase_field_pwconst>=0.5]=1
    # phase_field_pwconst[phase_field_pwconst<0.5]=0

    #phase = 1 * np.ones(number_of_pixels)
    inc_contrast = 0.



    # nb_it=[]
    # nb_it_combi=[]
    # nb_it_Jacobi=[]

    for i in np.arange(ratios.size):
        ratio=ratios[i]
        phase_field =  ratio*phase_field_smooth + (1-ratio)*phase_field_pwconst
        #phase_field =phase_field_pwconst + 1e-5*np.random.random(phase_field_pwconst.shape)
        phase_field =phase_field_pwconst  + 1e-4*phase_field_smooth
        #phase_field = phase_field_smooth +  phase_field_pwconst
        # phase[10:30, 10:30]ith: Obsonov solution
        # phase[phase.shape[0] * 1 // 4:phase.shape[0] * 3 // 4,
        # phase.shape[1] * 1 // 4:phase.shape[1] * 3 // 4] *= inc_contrast

        # min_ = discretization.
        phase_fem = np.zeros([2, *number_of_pixels])
        phase_fnxyz=discretization.get_scalar_sized_field()
        phase_fnxyz[0,0,...] = phase_field

        f_field = discretization.fft.fft(phase_fnxyz)
        # f_field[0, 0, np.logical_and(np.abs(discretization.fft.fftfreq[0]) > 0.25,
        #                              np.abs(discretization.fft.fftfreq[1]) > 0.25)] = 0
        f_field[0, 0, np.logical_or(np.abs(discretization.fft.ifftfreq[0]) > 10,
                                     np.abs(discretization.fft.ifftfreq[1]) > 10)] = 0
        # f_field[0, 0, 12:, 24:] = 0
        phase = discretization.fft.ifft(f_field) * discretization.fft.normalisation
       # phase[phase > 1] = 1
        phase[phase < 0] = phase[phase < 0] **2
        phase_fnxyz[0, 0, ...] =phase

        #phase_field[0,0]=np.power(phase_field, 2)
        #np.save('geometry_jacobi.npy', np.power(phase_field_l, 2),)
        #sc.io.savemat('geometry_jacobi.mat', {'data':  np.power(phase_field_l, 2)})

        phase_field_at_quad_poits_1qnxyz = \
                            discretization.evaluate_field_at_quad_points(nodal_field_fnxyz=phase_fnxyz,
                                                                         quad_field_fqnxyz=None,
                                                                         quad_points_coords_dq=None)[0]

        # apply material distribution
        #material_data_field_C_0_rho = material_data_field_C_0[..., :, :] * np.power(phase_field[0, 0], 1)
        #material_data_field_C_0_rho=material_data_field_C_0[..., :, :] * phase_fem
        #material_data_field_C_0_rho +=100*material_data_field_C_0[..., :, :] * (1-phase_fem)
        material_data_field_C_0_rho = material_data_field_C_0[..., :, :, :] * np.power(
                            phase_field_at_quad_poits_1qnxyz, 1)[0, :, 0, ...]

        # Set up right hand side
        macro_gradient_field = discretization.get_macro_gradient_field(macro_gradient)



        # Solve mechanical equilibrium constrain
        rhs = discretization.get_rhs(material_data_field_C_0_rho, macro_gradient_field)

        K_fun = lambda x: discretization.apply_system_matrix(material_data_field_C_0_rho, x,
                                                             formulation='small_strain')
        K=discretization.get_system_matrix(material_data_field_C_0_rho)
        # M_fun = lambda x: 1 * x
        eig=np.linalg.eigvals(K)
        plt.figure()
        plt.semilogy(sorted(eig))
        plt.show()
        K= discretization.get_system_matrix(material_data_field=material_data_field_C_0_rho)
        #material_data_field_C_0=np.mean(material_data_field_C_0_rho,axis=(4,5,6))
        mean_material=np.mean(material_data_field_C_0_rho,axis=(4,5,6))
        material_data_field_C_0_ratio = np.einsum('ijkl,qxy->ijklqxy', mean_material,
                                            np.ones(np.array([discretization.nb_quad_points_per_pixel,
                                                              *discretization.nb_of_pixels])))

        preconditioner = discretization.get_preconditioner_NEW(
            reference_material_data_field_ijklqxyz=material_data_field_C_0)

        M_fun = lambda x: discretization.apply_preconditioner_NEW(preconditioner_Fourier_fnfnqks=preconditioner,
                                                              nodal_field_fnxyz=x)


        K_diag_alg = discretization.get_preconditioner_Jacoby_fast(
            material_data_field_ijklqxyz=material_data_field_C_0_rho)

        M_fun_combi = lambda x: K_diag_alg * discretization.apply_preconditioner_NEW(
                               preconditioner_Fourier_fnfnqks=preconditioner,
                                nodal_field_fnxyz=K_diag_alg * x)
        # #
        M_fun_Jacobi = lambda x: K_diag_alg *  K_diag_alg * x

        displacement_field, norms = solvers.PCG(K_fun, rhs, x0=None, P=M_fun, steps=int(1000), toler=1e-6)
        nb_it[nb_pix_multip-1,i]=( len(norms['residual_rz']))
        norm_rz = norms['residual_rz'][-1]
        norm_rr = norms['residual_rr'][-1]
        print(nb_it)

        displacement_field, norms_combi = solvers.PCG(K_fun, rhs, x0=None, P=M_fun_combi, steps=int(1000), toler=1e-6)
        nb_it_combi[nb_pix_multip-1,i]=(len(norms_combi['residual_rz']))
        norm_rz_combi = norms_combi['residual_rz'][-1]
        norm_rr_combi = norms_combi['residual_rr'][-1]

        displacement_field, norms_Jacobi = solvers.PCG(K_fun, rhs, x0=None, P=M_fun_Jacobi, steps=int(1000), toler=1e-6)
        nb_it_Jacobi[nb_pix_multip-1,i]=(len(norms_Jacobi['residual_rz']))
        norm_rz_Jacobi = norms_Jacobi['residual_rz'][-1]
        norm_rr_Jacobi = norms_Jacobi['residual_rr'][-1]

        print(ratio)
        print(
            '   nb_ steps CG Green of =' f'{nb_it}, residual_rz = {norm_rz}, residual_rr = {norm_rr},\n \
                nb_ steps CG Jacobi of =' f'{nb_it_Jacobi}, residual_rz = {norm_rz_Jacobi}, residual_rr = {norm_rr_Jacobi},\n\
                nb_ steps CG combi of =' f'{nb_it_combi}, residual_rz = {norm_rz_combi}, residual_rr = {norm_rr_combi}'
        )
        #print(norms)


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')


# Plot each line with a different z offset
for i in np.arange(len(nb_pix_multips)):
    ax.plot(ratios,  nb_pix_multips[i]*32, zs=nb_it[i],label='DGO 1', color='blue')
    ax.plot(ratios,  nb_pix_multips[i]*32,zs=nb_it_combi[i], label='nb_it_combi 1', color='red')
    ax.plot(ratios,  nb_pix_multips[i]*32,zs=nb_it_Jacobi[i], label='nb_it_Jacobi', color='black')

ax.set_xlabel('ratio: ratio*smooth + (1-ratio)*pwconst')
ax.set_ylabel('size')
ax.set_zlabel('# CG iterations')
plt.legend(['DGO', 'Jacoby', 'DGO + Jacoby' ])

plt.show()


plot_evolion=False
if plot_evolion:
    for nb_tiles in [1,]:
        # fig = plt.figure()
        fig, axs = plt.subplots(nrows=2, ncols=1,
                                figsize=(6, 6)  )
        # axs[0] = plt.axes(xlim=(0, nb_tiles * N), ylim=(0, nb_tiles * N))
        axs[0].imshow(phase_field, cmap=mpl.cm.Greys, vmin=0, vmax=1)

        print(ratios)

        print(nb_it)
        axs[1].plot(ratios, nb_it, label='nb_it_Laplace', linewidth=0)

        #axs[1].plot(xopt.f.num_iteration_.transpose()[::3], 'w'  , linewidth=0)
        #axs[1].plot(xopt3.f.num_iteration_.transpose(), "b", label='Jacoby', linewidth=0)
        #axs[1].plot(xopt.f.num_iteration_.transpose(), "k", label='DGO + Jacoby', linewidth=0)
        #legend = plt.legend()
        # Animation function to update the image
        axs[1].set_xlabel('ratio: ratio*phase_field_smooth + (1-ratio)*phase_field_pwconst')
        axs[1].set_ylabel('nb steps')

        def update(i):
            ratio=ratios[i]
            phase_field = ratio * phase_field_smooth + (1 - ratio) * phase_field_pwconst

            axs[0].clear()
            axs[0].imshow(phase_field, cmap=mpl.cm.Greys)#, vmin=0, vmax=1
            axs[0].set_title(f'Phase_contrast: min = {np.min(phase_field)},max = {np.max(phase_field)}')

            axs[1].plot(ratios[0:i+1],nb_it[0:i+1], 'r', label='DGO',linewidth=1)
            #axs[1].plot(xopt2.f.num_iteration_.transpose()[1:3*i+1:3],"r", label='DGO ',linewidth=1)
            #axs[1].plot(xopt2.f.num_iteration_.transpose()[2:3*i+2:3],"r", label='DGO ',linewidth=1)

            axs[1].plot(ratios[0:i+1], nb_it_Jacobi[0:i+1],"b", label='Jacoby', linewidth=1)
            axs[1].plot(ratios[0:i+1], nb_it_combi[0:i+1], "k", label='DGO + Jacoby', linewidth=1)

            #axs[1].legend()
            plt.legend([ '', 'DGO', 'Jacoby', 'DGO + Jacoby' ])
                # img.set_array(xopt_it)


        # Create animation
        #ani = FuncAnimation(fig, update, frames=xopt.f.norms_f.size - 1, blit=False)

        ani = FuncAnimation(fig, update, frames=ratios.size, blit=False)
        #axs[1].legend()middlemiddle
        # Save as a GIF
        ani.save(f"./figures/movie_comparison_{geometry_ID}_square_inc_to_pf_scaled{scaling}.gif", writer=PillowWriter(fps=4))


    plt.show()