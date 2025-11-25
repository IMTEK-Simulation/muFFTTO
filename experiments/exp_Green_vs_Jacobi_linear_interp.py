from cProfile import label

import numpy as np
import scipy as sc
import time
import matplotlib.pyplot as plt
import matplotlib as mpl

import matplotlib.pyplot as plt
from IPython.terminal.shortcuts.filters import KEYBINDING_FILTERS
from PySide2.examples.opengl.contextinfo import colors
from matplotlib.animation import FuncAnimation, PillowWriter
from sympy.physics.quantum.sho1d import omega

from muFFTTO import domain
from muFFTTO import solvers
from muFFTTO import microstructure_library
from mpl_toolkits import mplot3d

problem_type = 'elasticity'
discretization_type = 'finite_element'
element_type = 'linear_triangles'
formulation = 'small_strain'





domain_size = [1, 1]
nb_pix_multips = [1]  # ,3,2,
small = np.arange(0., .1, 0.005)
middle = np.arange(0.1, 0.9, 0.03)

large = np.arange(0.9, 1.0 + 0.005, 0.005)
ratios = np.concatenate((small, middle, large))
ratios = np.arange(0., 1.1, 0.2)
ratios = np.arange(0., 1.1, 0.2)
ratios = np.arange(10)


nb_it = np.zeros((len(nb_pix_multips), ratios.size), )
nb_it_combi = np.zeros((len(nb_pix_multips), ratios.size), )
nb_it_Jacobi = np.zeros((len(nb_pix_multips), ratios.size), )
nb_it_Richardson= np.zeros((len(nb_pix_multips), ratios.size), )
nb_it_Richardson_combi= np.zeros((len(nb_pix_multips), ratios.size), )
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
norm_rr_combi=[]
norm_rz_combi=[]
norm_rr_Jacobi=[]
norm_rz_Jacobi=[]
norm_rr=[]
norm_rz=[]

kontrast=[]
kontrast_2=[]
eigen_LB=[]


for kk in np.arange(np.size(nb_pix_multips)):
    nb_pix_multip = nb_pix_multips[kk]
    number_of_pixels = (nb_pix_multip * 32, nb_pix_multip * 32)
    number_of_pixels = (16,16)

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
    C_1=domain.compute_Voigt_notation_4order(elastic_C_1)

    C_1=domain.compute_Voigt_notation_4order(elastic_C_1)

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
    load_micro = False
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
    phase_field = np.abs(phase_field_smooth-1)
    #flipped_arr = 1 - phase_field

    # Method 2: Using subtraction
   # flipped_arr_alt = np.logical_not(flipped_arr).astype(int)
    # plt.figure()
    # fig = plt.figure()
    # gs = fig.add_gridspec(1, 2)
    # ax1 = fig.add_subplot(gs[0, 0])
    # ax2 = fig.add_subplot(gs[0, 1])
    for i in np.arange(ratios.size):
        ratio = ratios[i]


        # phase_field =  ratio*phase_field_smooth + (1-ratio)*phase_field_pwconst
        # phase_field =phase_field_pwconst + 1e-5*np.random.random(phase_field_pwconst.shape)
        #        phase_field =phase_field_pwconst  + 1e-4*phase_field_smooth

        def apply_smoother(phase):
            # Define a 2D smoothing kernel
            kernel = np.array([[0.0625, 0.125, 0.0625],
                               [0.125, 0.25, 0.125],
                               [0.0625, 0.125, 0.0625]])

            # Apply convolution for smoothing
            smoothed_arr = sc.signal.convolve2d(phase, kernel, mode='same', boundary='wrap')
            return smoothed_arr


        if i == 0:
            phase_field = phase_field_smooth + 1e-4
        if i > 0:
            phase_field = apply_smoother(phase_field)


        # phase_field = 1e-4 + (phase_field - min_val) * (1 - 1e-4) / (max_val - min_val)

        # scaled_arr = min_val + (arr - arr.min()) * (max_val - min_val) / (arr.max() - arr.min())

        # for a in np.arange(20):
        #     phase_field = apply_smoother(phase_field)
        #     #print(np.min(phase_field))
        #     #(phase_field / np.min(phase_field)) / 1e4
        #     min_val=np.min(phase_field)
        #     max_val=np.max(phase_field)
        #     scaled_arr = 1e-4 + (phase_field - min_val) * (1 - 1e-4) / (max_val - min_val)
        #
        # np.min(phase_field)

        # phase_field = phase_field_smooth +  phase_field_pwconst
        # phase[10:30, 10:30]ith: Obsonov solution
        # phase[phase.shape[0] * 1 // 4:phase.shape[0] * 3 // 4,
        # phase.shape[1] * 1 // 4:phase.shape[1] * 3 // 4] *= inc_contrast
        # fig=plt.figure(i)
        # plt.imshow(phase_field, cmap=mpl.cm.Greys)  # , vmin=0, vmax=1
        # plt.show()
        # min_ = discretization.
        phase_fem = np.zeros([2, *number_of_pixels])
        phase_fnxyz = discretization.get_scalar_sized_field()
        phase_fnxyz[0, 0, ...] = phase_field


        def apply_filter(phase):
            f_field = discretization.fft.fft(phase)
            # f_field[0, 0, np.logical_and(np.abs(discretization.fft.fftfreq[0]) > 0.25,
            #                              np.abs(discretization.fft.fftfreq[1]) > 0.25)] = 0
            f_field[0, 0, np.logical_or(np.abs(discretization.fft.ifftfreq[0]) > 10,
                                        np.abs(discretization.fft.ifftfreq[1]) > 10)] = 0
            # f_field[0, 0, 12:, 24:] = 0
            phase = discretization.fft.ifft(f_field) * discretization.fft.normalisation
            # phase[phase > 1] = 1
            phase[phase < 0] = phase[phase < 0] ** 2
            phase_fnxyz[0, 0, ...] = phase
            return phase


        # np.save('geometry_jacobi.npy', np.power(phase_field_l, 2),)
        # sc.io.savemat('geometry_jacobi.mat', {'data':  np.power(phase_field_l, 2)})

        phase_field_at_quad_poits_1qnxyz = \
            discretization.evaluate_field_at_quad_points(nodal_field_fnxyz=phase_fnxyz,
                                                         quad_field_fqnxyz=None,
                                                         quad_points_coords_iq=None)[0]

        phase_field_at_quad_poits_1qnxyz[0, :, 0, ...] = phase_fnxyz
        # apply material distribution
        # material_data_field_C_0_rho = material_data_field_C_0[..., :, :] * np.power(phase_field[0, 0], 1)
        # material_data_field_C_0_rho=material_data_field_C_0[..., :, :] * phase_fem
        # material_data_field_C_0_rho +=100*material_data_field_C_0[..., :, :] * (1-phase_fem)
        material_data_field_C_0_rho = material_data_field_C_0[..., :, :, :] * np.power(
            phase_field_at_quad_poits_1qnxyz, 1)[0, :, 0, ...]
        # material_data_field_C_0_rho=phase_field_at_quad_poits_1qnxyz
        # Set up right hand side
        macro_gradient_field = discretization.get_macro_gradient_field(macro_gradient)
        # perturb=np.random.random(macro_gradient_field.shape)
        # macro_gradient_field += perturb#-np.mean(perturb)

        # Solve mechanical equilibrium constrain
        rhs = discretization.get_rhs(material_data_field_C_0_rho, macro_gradient_field)

        K_fun = lambda x: discretization.apply_system_matrix(material_data_field_C_0_rho, x,
                                                             formulation='small_strain')

        # plotting eigenvalues
      ##  K = discretization.get_system_matrix(material_data_field_C_0_rho)
       ## M = discretization.get_system_matrix(refmaterial_data_field_I4s)

       ## eig = sc.linalg.eigh(a=K, b=M, eigvals_only=True)

        min_val = np.min(phase_field)
        max_val = np.max(phase_field)

        kontrast.append(max_val / min_val)
        eigen_LB.append(min_val)

       # kontrast_2.append(eig[-3] / eig[np.argmax(eig > 0)])
        kontrast_2.append((max_val / min_val)/10)

        omega =1# 2 / ( eig[-1]+eig[np.argmax(eig>0)])
        # ax1.loglog(sorted(eig)[1:],label=f'{i}',marker='.', linewidth=0, markersize=1)
        # ax1.set_ylim([1e-5, 1e1])
        #
        # K_diag_half = np.copy(np.diag(K))
        # K_diag_half[K_diag_half < 1e-16] = 0
        # K_diag_half[K_diag_half != 0] = 1/np.sqrt(K_diag_half[K_diag_half != 0])
        #
        # DKDsym = np.matmul(np.diag(K_diag_half),np.matmul(K,np.diag(K_diag_half)))
        # eig = sc.linalg.eigh(a=DKDsym, b=M, eigvals_only=True)
        #
        # ax2.loglog(sorted(eig)[1:-2], label=f'{i}',marker='.', linewidth=0, markersize=1)
        # ax2.set_ylim([1e-5, 1e1])

        K = discretization.get_system_matrix(material_data_field=material_data_field_C_0_rho)
        # material_data_field_C_0=np.mean(material_data_field_C_0_rho,axis=(4,5,6))
        # mean_material=np.mean(material_data_field_C_0_rho,axis=(4,5,6))
        # material_data_field_C_0_ratio = np.einsum('ijkl,qxy->ijklqxy', mean_material,
        #                                     np.ones(np.array([discretization.nb_quad_points_per_pixel,
        #                                                       *discretization.nb_of_pixels])))

        preconditioner = discretization.get_preconditioner_NEW(
            reference_material_data_field_ijklqxyz=refmaterial_data_field_I4s)

        M_fun = lambda x: discretization.apply_preconditioner_NEW(preconditioner_Fourier_fnfnqks=preconditioner,
                                                                  nodal_field_fnxyz=x)



        K_diag_alg = discretization.get_preconditioner_Jacoby_fast(
            material_data_field_ijklqxyz=material_data_field_C_0_rho)

        M_fun_combi = lambda x: K_diag_alg * discretization.apply_preconditioner_NEW(
            preconditioner_Fourier_fnfnqks=preconditioner,
            nodal_field_fnxyz=K_diag_alg * x)
        # #
        M_fun_Jacobi = lambda x: K_diag_alg * K_diag_alg * x

        displacement_field, norms = solvers.PCG(K_fun, rhs, x0=None, P=M_fun, steps=int(1000), toler=1e-6)
        nb_it[kk - 1, i] = (len(norms['residual_rz']))
        norm_rz.append(norms['residual_rz'])
        norm_rr.append(norms['residual_rr'])
        print(nb_it)
#########
        displacement_field_combi, norms_combi = solvers.PCG(K_fun, rhs, x0=None, P=M_fun_combi, steps=int(1000),
                                                            toler=1e-6)
        nb_it_combi[kk - 1, i] = (len(norms_combi['residual_rz']))
        norm_rz_combi.append(norms_combi['residual_rz'])
        norm_rr_combi.append(norms_combi['residual_rr'])
        #
        displacement_field_Jacobi, norms_Jacobi = solvers.PCG(K_fun, rhs, x0=None, P=M_fun_Jacobi, steps=int(1000),
                                                              toler=1e-6)
        nb_it_Jacobi[kk - 1, i] = (len(norms_Jacobi['residual_rz']))
        norm_rz_Jacobi.append(norms_Jacobi['residual_rz'])
        norm_rr_Jacobi.append(norms_Jacobi['residual_rr'])
        displacement_field_Richardson, norms_Richardson = solvers.Richardson(K_fun, rhs, x0=None, P=M_fun,
                                                                      omega=omega,
                                                                      steps=int(1000),
                                                                      toler=1e-6)
        # nb_it_Richardson[kk - 1, i] = (len(norms_Richardson['residual_rr']))
        # norm_rr_Richardson= norms_Richardson['residual_rr'][-1]
        #
        # displacement_field_Richardson_combi, norms_Richardson_combi = solvers.Richardson(K_fun, rhs, x0=None, P=M_fun_combi,
        #                                                                      omega=omega*0.4,
        #                                                                      steps=int(3000),
        #                                                                      toler=1e-6)
        # nb_it_Richardson_combi[kk - 1, i] = (len(norms_Richardson_combi['residual_rr']))
        # norm_rr_Richardson_combi = norms_Richardson_combi['residual_rr'][-1]
        # kujacobi=K_fun(displacement_field_combi)-rhs
        # plt.figure()
        # plt.imshow(kujacobi[0,0])
        # plt.title('rez Jacobi Green')
        # plt.colorbar()
        # plt.show()
        #
        # kugreen= K_fun(displacement_field) - rhs
        # plt.figure()
        # plt.imshow(kugreen[0, 0])
        # plt.title('rez greens')
        #
        # plt.colorbar()
        # plt.show()
        # plt.figure()
        # plt.imshow((displacement_field_combi-displacement_field)[0,0])
        # plt.colorbar()
        # plt.show()
        #
        # plt.figure()
        # plt.imshow(displacement_field_combi[0, 0])
        # plt.show()
        #print(f'norm = {np.linalg.norm(displacement_field_combi[0, 0] - displacement_field[0, 0])}')
##################
        print(ratio)
        # print(
        #     '   nb_ steps CG Green of =' f'{nb_it}, residual_rz = {norm_rz}, residual_rr = {norm_rr},\n \
        #         nb_ steps CG Jacobi of =' f'{nb_it_Jacobi}, residual_rz = {norm_rz_Jacobi}, residual_rr = {norm_rr_Jacobi},\n\
        #         nb_ steps CG combi of =' f'{nb_it_combi}, residual_rz = {norm_rz_combi}, residual_rr = {norm_rr_combi},\n\
        #         nb_ steps Richardson of =' f'{nb_it_Richardson} , residual_rr = {norm_rr_Richardson},\n\
        #         nb_ steps Richardson of =' f'{nb_it_Richardson_combi} , residual_rr = {norm_rr_Richardson_combi}'
        # )
##################

        # print(norms)
# box = ax2.get_position()
# ax2.set_position([box.x0, box.y0, box.width * 0.8, box.height])
#
# # Put a legend to the right of the current axis
# ax2.legend(loc='center left', bbox_to_anchor=(1, 0.5))
# plt.show()

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot each line with a different z offset
for i in np.arange(len(nb_pix_multips)):
    ax.plot(ratios, nb_pix_multips[i] * 32, zs=nb_it[i], label='PCG: Green', color='blue')
    ax.plot(ratios, nb_pix_multips[i] * 32, zs=nb_it_Jacobi[i], label='PCG: Jacobi', color='black')
    ax.plot(ratios, nb_pix_multips[i] * 32, zs=nb_it_combi[i], label='PCG: Green + Jacobi', color='red')
    ax.plot(ratios, nb_pix_multips[i] * 32, zs=nb_it_Richardson[i], label='Richardson Green', color='green')
    ax.plot(ratios, nb_pix_multips[i] * 32, zs=nb_it_Richardson_combi[i], label='Richardson Green+Jacobi')
ax.set_xlabel('nb of filter aplications')
ax.set_ylabel('size')
ax.set_zlabel('# CG iterations')
plt.legend(['DGO', 'Jacobi', 'DGO + Jacobi','Richardson'])
plt.show()

for i in np.arange(ratios.size,step=1):
    kappa=kontrast[i]
    k=np.arange(max(map(len, norm_rr)))
    print(f'k \n {k}')
    lb=eigen_LB[i]
    print(f'lb \n {lb}')

    convergence=((np.sqrt(kappa)-1)/(np.sqrt(kappa)+1))**k
    convergence=convergence*norm_rr[i][0]


    print(f'convergecnce \n {convergence}')
    fig = plt.figure()
    gs = fig.add_gridspec(1, 1)
    ax_1 = fig.add_subplot(gs[0, 0])
    ax_1.set_title(f'{i}', wrap=True)
    ax_1.semilogy(convergence, '--',label='estim', color='k')

    ax_1.semilogy(norm_rr[i], label='PCG: Green', color='r')
    ax_1.semilogy(norm_rr_Jacobi[i], label='PCG: Jacobi', color='b')

    #x_1.plot(ratios, nb_pix_multips[i] * 32, zs=nb_it_Richardson[i], label='Richardson Green', color='green')
    #ax_1.plot(ratios, nb_pix_multips[i] * 32, zs=nb_it_Richardson_combi[i], label='Richardson Green+Jacobi')
    ax_1.set_xlabel('CG iterations')
    ax_1.set_ylabel('Norm of residua')
    plt.legend([r'$\kappa$ upper bound','Green', 'Jacobi', 'Green + Jacobi','Richardson'])
    ax_1.set_ylim([1e-7, norm_rr[i][0]])#norm_rz[i][0]]/lb)
    print(max(map(len, norm_rr)))
    ax_1.set_xlim([0, max(map(len, norm_rr))])

    plt.show()

plt.show()
fig = plt.figure()
gs = fig.add_gridspec(1, 1)
ax_1 = fig.add_subplot(gs[0, 0])
ax_1.semilogy(norm_rr[0], label='PCG: Green', color='blue', linewidth=0)

#ax_1.set_ylim([1e-7, 1e0])
ax_1.set_ylim([1e-7, norm_rr[0][0]])  # norm_rz[i][0]]/lb)

print(max(map(len, norm_rz)))
ax_1.set_xlim([0, max(map(len, norm_rr))])

def convergence_gif_rz(i):
    kappa=kontrast[i]
    k=np.arange(max(map(len, norm_rr)))
    print(f'k \n {k}')
    lb=eigen_LB[i]
    print(f'lb \n {lb}')

    convergence=((np.sqrt(kappa)-1)/(np.sqrt(kappa)+1))**k
    convergence=convergence*norm_rr[i][0]

    ax_1.clear()

    ax_1.set_title(f'{i}', wrap=True)
    ax_1.semilogy(convergence, '--',label='estim', color='g')

    ax_1.semilogy(norm_rr[i], label='PCG: Green', color='g')
    ax_1.semilogy(norm_rr_Jacobi[i], label='PCG: Jacobi', color='b')
    ax_1.semilogy(norm_rr_combi[i], label='PCG: Jacobi', color='k')

    # x_1.plot(ratios, nb_pix_multips[i] * 32, zs=nb_it_Richardson[i], label='Richardson Green', color='green')
    # ax_1.plot(ratios, nb_pix_multips[i] * 32, zs=nb_it_Richardson_combi[i], label='Richardson Green+Jacobi')
    ax_1.set_xlabel('CG iterations')
    ax_1.set_ylabel('Norm of residua')
    plt.legend([r'$\kappa$ upper bound','Green', 'Jacobi', 'DGO + Jacobi','Richardson'])
    ax_1.set_ylim([1e-7, norm_rr[i][0]])#norm_rz[i][0]]/lb)
    print(max(map(len, norm_rr)))
    ax_1.set_xlim([0, max(map(len, norm_rr))])
    # axs[1].legend()
    # plt.legend(['', 'FEM: Green', 'FEM: Jacobi', 'FEM: Green + Jacobi','FEM: Richardson'])
    plt.legend([r'$\kappa$ upper bound','Green', 'Jacobi', 'Green + Jacobi','Richardson Green', 'Richardson Green + Jacobi'],
               loc='center left', bbox_to_anchor=(0.8, 0.5))

ani = FuncAnimation(fig, convergence_gif_rz, frames=ratios.size, blit=False)
# axs[1].legend()middlemiddle
# Save as a GIF
ani.save(f"./figures/convergence__es2tgif_{number_of_pixels[0]}comparison{ratios[-1]}_RichardsonJacobi{geometry_ID}_circle_inc_to_smooth_semiloplots3.gif",
         writer=PillowWriter(fps=1))

plt.show()
#-------------------------------------------------------------------------------------------------------
# for i in np.arange(ratios.size,step=1):
#     kappa=kontrast[i]
#     kappa_2=kontrast_2[i]
#     k=np.arange(len(norm_rr_Jacobi[i]))
#     print(f'k \n {k}')
#
#     convergence=((np.sqrt(kappa)-1)/(np.sqrt(kappa)+1))**k
#     convergence=convergence*norm_rr[i][0]
#     convergence2 = ((np.sqrt(kappa_2) - 1) / (np.sqrt(kappa_2) + 1)) ** k
#     convergence2 = convergence2 * norm_rr[i][0]
#
#
#     print(f'convergecnce \n {convergence}')
#     fig = plt.figure()
#     gs = fig.add_gridspec(1, 1)
#     ax_1 = fig.add_subplot(gs[0, 0])
#     ax_1.set_title(f'{i}', wrap=True)
#     ax_1.semilogy(convergence, '-',label='estim', color='green')
#     #ax_1.semilogy(convergence2,'.-', label='estim2', color='green')
#
#     ax_1.semilogy(norm_rr[i], label='PCG: Green', color='blue')
#     ax_1.semilogy(norm_rr_Jacobi[i], label='PCG: Jacobi', color='black')
#     #ax_1.semilogy(norm_rr_combi[i], label='PCG: Green + Jacobi', color='red')
#     #x_1.plot(ratios, nb_pix_multips[i] * 32, zs=nb_it_Richardson[i], label='Richardson Green', color='green')
#     #ax_1.plot(ratios, nb_pix_multips[i] * 32, zs=nb_it_Richardson_combi[i], label='Richardson Green+Jacobi')
#     ax_1.set_xlabel('CG iterations')
#     ax_1.set_ylabel('Norm of residuals')
#     plt.legend([r'$\kappa$ upper bound','Green', 'Jacobi', 'Green + Jacobi','Richardson'])
#     ax_1.set_ylim([1e-7, norm_rr[i][0]])
#     print(max(map(len, norm_rr)))
#     ax_1.set_xlim([0, max(map(len, norm_rr))])
#
#     plt.show()

fig = plt.figure()
gs = fig.add_gridspec(1, 1)
ax_1 = fig.add_subplot(gs[0, 0])
ax_1.semilogy(norm_rr[0], label='PCG: Green', color='blue', linewidth=0)

ax_1.set_ylim([1e-7, 1e0])
print(max(map(len, norm_rr)))
ax_1.set_xlim([0, max(map(len, norm_rr))])

def convergence_gif(i):
    kappa = kontrast[i]
    k = np.arange( max(map(len, norm_rr)))
    print(f'k \n {k}')

    convergence = ((np.sqrt(kappa) - 1) / (np.sqrt(kappa) + 1)) ** k
    convergence = convergence * norm_rr[i][0]
    print(f'convergecnce \n {convergence}')
    ax_1.clear()

    ax_1.set_title(f'{i}', wrap=True)
    ax_1.semilogy(convergence, '--',label='estim', color='k')

    ax_1.semilogy(norm_rr[i], label='PCG: Green', color='r')
    ax_1.semilogy(norm_rr_Jacobi[i], label='PCG: Jacobi', color='b')
    #ax_1.semilogy(norm_rr_combi[i], label='PCG: Green + Jacobi', color='red')
    # x_1.plot(ratios, nb_pix_multips[i] * 32, zs=nb_it_Richardson[i], label='Richardson Green', color='green')
    # ax_1.plot(ratios, nb_pix_multips[i] * 32, zs=nb_it_Richardson_combi[i], label='Richardson Green+Jacobi')
    ax_1.set_xlabel('CG iterations')
    ax_1.set_ylabel('Norm of residuals')
    plt.legend([r'$\kappa$ upper bound','Green', 'Jacobi', 'Green + Jacobi', 'Richardson'])
    ax_1.set_ylim([1e-7, 1e0])
    print(max(map(len, norm_rr)))
    ax_1.set_xlim([0, max(map(len, norm_rr))])
    # axs[1].legend()
    # plt.legend(['', 'FEM: Green', 'FEM: Jacobi', 'FEM: Green + Jacobi','FEM: Richardson'])
    plt.legend([r'$\kappa$ upper bound','Green', 'Jacobi', 'Green + Jacobi', 'Richardson Green', 'Richardson Green + Jacobi'],
               loc='center left', bbox_to_anchor=(0.8, 0.5))


ani = FuncAnimation(fig, convergence_gif, frames=ratios.size, blit=False)
# axs[1].legend()middlemiddle
# Save as a GIF
ani.save(f"./figures/convergence_gif_{number_of_pixels[0]}comparison{ratios[-1]}_RichardsonJacobi{geometry_ID}_circle_inc_to_smooth_semiloplots3.gif",
         writer=PillowWriter(fps=1))

plt.show()



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
        ax3.set_ylim([1e-4, 1])
        print(ratios)

        print(nb_it)
        ax2.plot(ratios, nb_it[0], label='nb_it_Laplace', linewidth=0)
        ax3.set_ylim([1e0, 1e3 ])

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
            phase_field=np.abs(phase_field-1)
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
            ax3.semilogy(phase_field[:, phase_field.shape[0] // 2], linewidth=1)
            # ax3.plot(phase_field[:, phase_field.shape[0] // 2], linewidth=1)
            ax3.set_ylim([1e-4, 1])
            ax3.set_title(f'Cross section')

            ax2.plot(ratios[0:i + 1], nb_it[0, 0:i + 1], 'g', label=' Green', linewidth=1)
            # axs[1].plot(xopt2.f.num_iteration_.transpose()[1:3*i+1:3],"r", label='DGO ',linewidth=1)
            # axs[1].plot(xopt2.f.num_iteration_.transpose()[2:3*i+2:3],"r", label='DGO ',linewidth=1)

            ax2.plot(ratios[0:i + 1], nb_it_Jacobi[0, 0:i + 1], "b", label='PCG Jacobi', linewidth=1)
            ax2.semilogy(ratios[0:i + 1], nb_it_combi[0, 0:i + 1], "k", label='PCG Green + Jacobi', linewidth=1)
          #  ax2.semilogy(ratios[0:i + 1], nb_it_Richardson[0, 0:i + 1], "g", label=' Richardson Green ', linewidth=1)
          #  ax2.semilogy(ratios[0:i + 1], nb_it_Richardson_combi[0, 0:i + 1], "y",  label=' Richardson Green + Jacobi ', linewidth=1)

            # axs[1].legend()
            ax2.legend([ '','Green', 'Jacobi' ])
            #plt.legend(['', 'FEM: Green', 'FEM: Jacobi', 'FEM: Green + Jacobi','FEM: Richardson'])
           # plt.legend(['', ' Green', 'Jacobi', 'Green + Jacobi','Richardson Green','Richardson Green + Jacobi'],loc='best', bbox_to_anchor=(0.7, 0.5))
#        ax2.legend([r'$\kappa$ upper bound', 'Green', 'Jacobi', 'Green + Jacobi', 'Richardson'])

            # plt.legend([ '', 'Green', 'Jacobi'  ])

            # img.set_array(xopt_it)
        #ax2.legend([r'$\kappa$ upper bound', 'Green', 'Jacobi', 'Green + Jacobi', 'Richardson'])

        # box = ax2.get_position()
        # ax2.set_position([box.x0, box.y0, box.width * 0.8, box.height])
        #
        # # Put a legend to the right of the current axis
        # ax2.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        # Create animation
        # ani = FuncAnimation(fig, update, frames=xopt.f.norms_f.size - 1, blit=False)

        ani = FuncAnimation(fig, update, frames=ratios.size, blit=False)
        # axs[1].legend()middlemiddle
        # Save as a GIF
        ani.save(f"./figures/movie2222_{number_of_pixels[0]}comparison{ratios[-1]}_RichardsonJacobi{geometry_ID}_circle_inc_to_smooth_semiloplots3.gif",
                 writer=PillowWriter(fps=4))

    plt.show()
