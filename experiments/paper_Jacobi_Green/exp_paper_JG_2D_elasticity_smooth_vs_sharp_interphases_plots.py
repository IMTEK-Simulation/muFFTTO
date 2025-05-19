from cProfile import label

import numpy as np
import scipy as sc
import time

import matplotlib.pyplot as plt
import matplotlib as mpl
from muFFTTO import domain
from muFFTTO import solvers
from muFFTTO import microstructure_library
from mpl_toolkits import mplot3d

problem_type = 'elasticity'
discretization_type = 'finite_element'
element_type = 'linear_triangles'
formulation = 'small_strain'
src = '../figures/'  # source folder\

domain_size = [1, 1]
nb_pix_multips = [4]  # ,2,3,3,2,

ratios = np.array([2,4,6,8])# 4,6,8

nb_it = np.zeros((  ratios.size,2) )
nb_it_combi = np.zeros((  ratios.size,2) )
nb_it_Jacobi = np.zeros(( ratios.size,2) )
nb_it_Richardson = np.zeros( ( ratios.size,2) )
nb_it_Richardson_combi = np.zeros((  ratios.size,2) )

norm_rr_combi = []
norm_rz_combi = []
norm_rr_Jacobi = []

norm_rz_Jacobi = []
norm_rr = []
norm_rz = []

norm_rMr = []
norm_rMr_combi = []
norm_rMr_Jacobi = []

kontrast = []
kontrast_2 = []
eigen_LB = []

for kk in np.arange(np.size(nb_pix_multips)):
    nb_pix_multip = nb_pix_multips[kk]
    # number_of_pixels = (nb_pix_multip * 32, nb_pix_multip * 32)
    number_of_pixels = (nb_pix_multip * 16, nb_pix_multip * 16)

    # number_of_pixels = (16,16)

    my_cell = domain.PeriodicUnitCell(domain_size=domain_size,
                                      problem_type=problem_type)

    discretization = domain.Discretization(cell=my_cell,
                                           nb_of_pixels_global=number_of_pixels,
                                           discretization_type=discretization_type,
                                           element_type=element_type)
    start_time = time.time()

    # set macroscopic gradient
    macro_gradient = np.array([[1.0, 0.5], [0.5, 1.0]])

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
    C_1 = domain.compute_Voigt_notation_4order(elastic_C_1)

    material_data_field_C_0 = np.einsum('ijkl,qxy->ijklqxy', elastic_C_1,
                                        np.ones(np.array([discretization.nb_quad_points_per_pixel,
                                                          *discretization.nb_of_pixels])))

    refmaterial_data_field_I4s = np.einsum('ijkl,qxy->ijklqxy', elastic_C_1,
                                           np.ones(np.array([discretization.nb_quad_points_per_pixel,
                                                             *discretization.nb_of_pixels])))

    print('elastic tangent = \n {}'.format(domain.compute_Voigt_notation_4order(elastic_C_1)))

    # material distribution

    name = 'lbfg_muFFTTO_elasticity_exp_paper_JG_2D_elasticity_TO_N64_E_target_0.15_Poisson_-0.50_Poisson0_0.29_w5.00_eta0.02_mac_1.0_p2_prec=Green_bounds=False_FE_NuMPI6_nb_load_cases_3_e_obj_False_random_True'
    iteration = 1200
    geometry = np.load('../exp_data/' + name + f'_it{iteration}.npy', allow_pickle=True)
    phase_field_origin = np.abs(geometry)
    # phase_field = np.random.rand(*discretization.get_scalar_sized_field().shape)  # set random distribution#

    # phase = 1 * np.ones(number_of_pixels)
    inc_contrast = 0.

    # nb_it=[]
    # nb_it_combi=[]
    # nb_it_Jacobi=[]
    #phase_field_origin =# np.abs(phase_field_smooth - 1)
    # flipped_arr = 1 - phase_field
    phase_field_min = np.min(phase_field_origin)
    phase_field_max = np.max(phase_field_origin)


    def scale_field(field, min_val, max_val):
        """Scales a 2D random field to be within [min_val, max_val]."""
        field_min, field_max = np.min(field), np.max(field)
        scaled_field = (field - field_min) / (field_max - field_min)  # Normalize to [0,1]
        return scaled_field * (max_val - min_val) + min_val  # Scale to [min_val, max_val]

    def scale_field_log(field, min_val, max_val):
        """Scales a 2D random field to be within [min_val, max_val]."""
        field_log = np.log10(field)
        field_min, field_max = np.min(field_log), np.max(
            field_log)

        scaled_field = (field_log - field_min) / (field_max - field_min)  # Normalize to [0,1]
        return 10 ** (scaled_field * (np.log10(max_val) - np.log10(min_val)) + np.log10(min_val))  # Scale to [min_val, max_val]


    for i in np.arange(ratios.shape[0])  :
        ratio=ratios[i]


        counter = 0
        for sharp in [False, True]:
            print(f'ratio={ratio} ')
            phase_field = scale_field_log(np.copy(phase_field_origin), min_val=1 / (10 ** ratio), max_val=phase_field_max)
            print(f'min ={np.min(phase_field)} ')
            print(f'max ={np.max(phase_field)} ')

            #
            # if ratio == 0:
            #     phase_field = scale_field(phase_field, min_val=0, max_val=1.0)
            # else:
            #     phase_field = scale_field(phase_field, min_val=1 / 10 ** ratio, max_val=1.0)

            #phase_field = np.copy(phase_field_origin)

            if sharp:
                #phase_field = scale_field(phase_field_origin, min_val=1 / 10 ** ratio, max_val=1.0)
                phase_field[phase_field < 0.5] = 1 / 10 ** ratio#phase_field_min#
                phase_field[phase_field > 0.49] = phase_field_max#1

            print(f'min ={np.min(phase_field)} ')
            print(f'max ={np.max(phase_field)} ')

            material_data_field_C_0_rho = np.copy(material_data_field_C_0[..., :, :, :]) * np.power(
                phase_field, 1)

            # plt.figure()
            # plt.semilogy(np.power(
            #     phase_field, 1)[10,:])
            # plt.semilogy(np.power(
            #     phase_field, 2)[10,:])
            #
            #
            # plt.show()

            print(f'min ={np.min(material_data_field_C_0_rho)} ')
            print(f'max ={np.max(material_data_field_C_0_rho)} ')
            # print(np.max(np.power(
            #     phase_field, 2)))
            # material_data_field_C_0_rho_ijklqxyz = material_data_field_C_0[..., :, :, :] * np.power(
            #     material_data_field_C_0_rho, 2)[0, :, 0, ...]

            # apply material distribution

            # Set up right hand side
            macro_gradient_field = discretization.get_macro_gradient_field(macro_gradient)
            # perturb=np.random.random(macro_gradient_field.shape)
            # macro_gradient_field += perturb#-np.mean(perturb)

            # Solve mechanical equilibrium constrain
            rhs = discretization.get_rhs(material_data_field_C_0_rho, macro_gradient_field)

            K_fun = lambda x: discretization.apply_system_matrix(material_data_field_C_0_rho, x,
                                                                 formulation='small_strain')

            # plotting eigenvalues

            omega = 1  # 2 / ( eig[-1]+eig[np.argmax(eig>0)])

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
            x_init=discretization.get_displacement_sized_field()
            #x_init=np.random.random(discretization.get_displacement_sized_field().shape)

            displacement_field, norms = solvers.PCG(K_fun, rhs, x0=x_init, P=M_fun, steps=int(10000), toler=1e-10,
                                                    norm_type='data_scaled_rr',
                                                    norm_metric=M_fun)
            nb_it[i, counter] = (len(norms['residual_rz']))
            norm_rz.append(norms['residual_rz'])
            norm_rr.append(norms['residual_rr'])
            norm_rMr.append(norms['data_scaled_rr'])
            print(f'i={i} ')
            print(f'counter    ={counter} ')

            print(nb_it)
            #########
            displacement_field_combi, norms_combi = solvers.PCG(K_fun, rhs, x0=x_init, P=M_fun_combi, steps=int(4000),
                                                                toler=1e-10,
                                                                norm_type='data_scaled_rr',
                                                                norm_metric=M_fun)
            nb_it_combi[i, counter] = (len(norms_combi['residual_rz']))
            norm_rz_combi.append(norms_combi['residual_rz'])
            norm_rr_combi.append(norms_combi['residual_rr'])
            norm_rMr_combi.append(norms_combi['data_scaled_rr'])

            #
            displacement_field_Jacobi, norms_Jacobi = solvers.PCG(K_fun, rhs, x0=None, P=M_fun_Jacobi, steps=int(4000),
                                                                  toler=1e-10,
                                                                  norm_type='data_scaled_rr',
                                                                  norm_metric=M_fun)
            nb_it_Jacobi[i, counter] = (len(norms_Jacobi['residual_rz']))
            norm_rz_Jacobi.append(norms_Jacobi['residual_rz'])
            norm_rr_Jacobi.append(norms_Jacobi['residual_rr'])
            norm_rMr_Jacobi.append(norms_Jacobi['data_scaled_rr'])

            displacement_field_Richardson, norms_Richardson = solvers.Richardson(K_fun, rhs, x0=None, P=M_fun,
                                                                                 omega=omega,
                                                                                 steps=int(1000),
                                                                                 toler=1e-1)

            counter += 1

fig = plt.figure(figsize=(5.5, 6))
gs = fig.add_gridspec(3, 1)
ax_1 = fig.add_subplot(gs[2, 0])
ax_cross = fig.add_subplot(gs[1, 0])
ax_geom = fig.add_subplot(gs[0, 0])
lines = ['-', '-.','--',':']
for i in np.arange(ratios.size, step=1):
    kappa = 10 ** ratios[i]

    k = np.arange(max(map(len, norm_rr)))
    print(f'k \n {k}')

    convergence = ((np.sqrt(kappa) - 1) / (np.sqrt(kappa) + 1)) ** k
    convergence = convergence  # *norm_rr[i][0]


    divnorm = mpl.colors.Normalize(vmin=1e-8, vmax=1)
    cmap_ = mpl.cm.seismic# mpl.cm.seismic #mpl.cm.Greys
    geometry = np.load('../exp_data/' + name + f'_it{iteration}.npy', allow_pickle=True)
    phase_field_origin = np.abs(geometry)
    phase_field = scale_field_log(np.copy(phase_field_origin), min_val=1 / (10 ** ratios[i]), max_val=phase_field_max)
#np.unravel_index(phase_field_origin.argmin(), phase_field_origin.shape)
    pcm = ax_geom.pcolormesh(np.tile(phase_field, (1, 1)),
                             cmap=cmap_, linewidth=0,
                             rasterized=True, norm=divnorm)
    ax_geom.axhline(y=30, color='k', linestyle='-.')
    #ax_geom.set_aspect('equal', 'box')



    ax_cross.semilogy(phase_field[30,:], label=r'$\kappa=10^'+f'{{{-ratios[i]}}}$', color='k', linestyle=lines[i])
    #ax_geom.set_title(r'Initial ', wrap=True)
    ax_cross.set_ylabel(r'Density $\rho$')
    ax_cross.set_xlim([0 , 64])
    ax_cross.set_xticks([0 ,32, 64])
    ax_cross.set_ylim([1 / (10 ** ratios[i]), 1.1])

    ax_geom.set_xticks([0 ,32, 64])
    ax_geom.set_yticks([0 ,32, 64])

    # print(f'convergecnce \n {convergence}')
    #ax_1.set_title(f'Smooth', wrap=True)
    #ax_1.semilogy(convergence,  label=f'estim {kappa}', color='k', linestyle=lines[i])
    #ax_geom.set_xticks([])
    #ax_geom.set_xticks([])
    ax_1.semilogy(norm_rMr[2*i]/norm_rMr[2*i][0], label=r'$\kappa=10^'+f'{{{-ratios[i]}}}$', color='g', linestyle=lines[i])
  #  ax_1.semilogy(norm_rMr[2*i+1]/norm_rMr[2*i+1][0], label=f'Green ' +r'$\kappa=10^'+f'{{{ratios[i]}}}$', color='r', linestyle=lines[i])
    ax_1.semilogy(norm_rMr_combi[2*i]/norm_rMr_combi[2*i][0], label=r'$JG \kappa=10^'+f'{{{-ratios[i]}}}$', color='r', linestyle=lines[i])

    #ax_1.semilogy(norm_rMr_Jacobi[i]/norm_rMr_Jacobi[i][0], label=f' Jacobi {kappa}', color='b', linestyle=lines[i])
    #ax_1.semilogy(norm_rMr_combi[i]/norm_rMr_combi[i][0], label=f' Jacobi-Green {kappa}', color='r', linestyle=lines[i])

    # x_1.plot(ratios, nb_pix_multips[i] * 32, zs=nb_it_Richardson[i], label='Richardson Green', color='green')
    # ax_1.plot(ratios, nb_pix_multips[i] * 32, zs=nb_it_Richardson_combi[i], label='Richardson Green+Jacobi')
    ax_1.set_xlabel('# CG iterations')
    ax_1.set_ylabel('Relative error')
    # plt.legend([r'$\kappa$ upper bound','Green', 'Jacobi', 'Green + Jacobi','Richardson'])
    ax_1.set_ylim([1e-10, 1])  # norm_rz[i][0]]/lb)
    print(max(map(len, norm_rr)))
    ax_1.set_xlim([0, max(map(len, norm_rr))])
    ax_1.legend(['Green','Jacobi-Green'], loc='best')
    fig.tight_layout()
    fname = src + 'exp_paper_JG_2D_elasticity_TO_64_smooth' + '{}'.format('.pdf')
    print(('create figure: {}'.format(fname)))
    plt.savefig(fname, bbox_inches='tight')
plt.show()

fig = plt.figure(figsize=(5.5, 6))
gs = fig.add_gridspec(3, 1)
ax_1 = fig.add_subplot(gs[2, 0])
ax_cross = fig.add_subplot(gs[1, 0])
ax_geom = fig.add_subplot(gs[0, 0])

lines = ['-', '-.','--',':']
for i in np.arange(ratios.size, step=1):
    kappa = 10 ** ratios[i]

    k = np.arange(max(map(len, norm_rr)))
    print(f'k \n {k}')

    convergence = ((np.sqrt(kappa) - 1) / (np.sqrt(kappa) + 1)) ** k
    convergence = convergence  # *norm_rr[i][0]
    divnorm = mpl.colors.Normalize(vmin=1e-8, vmax=1)
    cmap_ = mpl.cm.seismic# mpl.cm.seismic #mpl.cm.Greys
    geometry = np.load('../exp_data/' + name + f'_it{iteration}.npy', allow_pickle=True)
    phase_field_origin = np.abs(geometry)
    phase_field = scale_field_log(np.copy(phase_field_origin), min_val=1 / (10 ** ratios[i]), max_val=phase_field_max)

    phase_field[phase_field < 0.5] = 1 / 10 ** ratios[i]  # phase_field_min#
    phase_field[phase_field > 0.49] = phase_field_max  # 1

    pcm = ax_geom.pcolormesh(np.tile(phase_field, (1, 1)),
                             cmap=cmap_, linewidth=0,
                             rasterized=True, norm=divnorm)
    ax_geom.axhline(y=30, color='k', linestyle='-.')

    ax_cross.semilogy(phase_field[30,:], label=r'$\kappa=10^'+f'{{{-ratios[i]}}}$', color='k', linestyle=lines[i])
    #ax_geom.set_title(r'Initial ', wrap=True) # :,15
    ax_cross.set_ylabel(r'Density $\rho$')
    ax_cross.set_ylim([1 / (10 ** ratios[i]), 1.1])
    ax_cross.set_xlim([0 , 64])
    ax_cross.set_xticks([0 ,32, 64])

    ax_geom.set_xticks([0 ,32, 64])
    ax_geom.set_yticks([0 ,32, 64])

    #print(f'convergecnce \n {convergence}')
    #ax_1.set_title(f'{i}', wrap=True)
    #ax_1.semilogy(convergence,  label=f'estim {kappa}', color='k', linestyle=lines[i])

   # ax_1.semilogy(norm_rMr[2*i]/norm_rMr[2*i][0], label=f'Green ' +r'$\kappa=10^'+f'{{{ratios[i]}}}$', color='g', linestyle=lines[i])
    ax_1.semilogy(norm_rMr[2*i+1]/norm_rMr[2*i+1][0], label=r'$\kappa=10^'+f'{{{-ratios[i]}}}$', color='g', linestyle=lines[i])
    ax_1.semilogy(norm_rMr_combi[2*i+1]/norm_rMr_combi[2*i+1][0], label=r'$JG \kappa=10^'+f'{{{-ratios[i]}}}$', color='r', linestyle=lines[i])

    #ax_1.semilogy(norm_rMr_Jacobi[i]/norm_rMr_Jacobi[i][0], label=f' Jacobi {kappa}', color='b', linestyle=lines[i])
    #ax_1.semilogy(norm_rMr_combi[i]/norm_rMr_combi[i][0], label=f' Jacobi-Green {kappa}', color='r', linestyle=lines[i])

    # x_1.plot(ratios, nb_pix_multips[i] * 32, zs=nb_it_Richardson[i], label='Richardson Green', color='green')
    # ax_1.plot(ratios, nb_pix_multips[i] * 32, zs=nb_it_Richardson_combi[i], label='Richardson Green+Jacobi')
    ax_1.set_xlabel('# CG iterations')
    ax_1.set_ylabel('Relative error')
    # plt.legend([r'$\kappa$ upper bound','Green', 'Jacobi', 'Green + Jacobi','Richardson'])
    ax_1.set_ylim([1e-10, 1])  # norm_rz[i][0]]/lb)
    print(max(map(len, norm_rr)))
    ax_1.set_xlim([0, max(map(len, norm_rr))])
    #ax_1.legend(loc='best',ncol=2)
    ax_1.legend(['Green','Jacobi-Green'], loc='best')

    fig.tight_layout()

    fname = src + 'exp_paper_JG_2D_elasticity_TO_64_sharp' + '{}'.format('.pdf')
    print(('create figure: {}'.format(fname)))
    plt.savefig(fname, bbox_inches='tight')

plt.show()

quit()

fig = plt.figure()
gs = fig.add_gridspec(1, 1)
ax_1 = fig.add_subplot(gs[0, 0])
# ax_1.semilogy(norm_rr[0], label='PCG: Green', color='blue', linewidth=0)
for kk in np.arange(np.size(nb_pix_multips)):
    ax_1.plot(ratios[0:], nb_it[kk], 'g', marker='|', label=' Green', linewidth=1)
    # axs[1].plot(xopt2.f.num_iteration_.transpose()[1:3*i+1:3],"r", label='DGO ',linewidth=1)
    # axs[1].plot(xopt2.f.num_iteration_.transpose()[2:3*i+2:3],"r", label='DGO ',linewidth=1)

    ax_1.plot(ratios[0:], nb_it_Jacobi[kk], "b", marker='o', label='PCG Jacobi', linewidth=1)  # [0, 0:]
    ax_1.plot(ratios[0:], nb_it_combi[kk], "k", marker='x', label='PCG Green + Jacobi', linewidth=1)
#  ax2.semilogy(ratios[0:i + 1], nb_it_Richardson[0, 0:i + 1], "g", label=' Richardson Green ', linewidth=1)
#  ax2.semilogy(ratios[0:i + 1], nb_it_Richardson_combi[0, 0:i + 1], "y",  label=' Richardson Green + Jacobi ', linewidth=1)

# axs[1].legend()
ax_1.set_ylim(bottom=0)
ax_1.legend(['Green', 'Jacobi', 'Green + Jacobi'])
plt.show()
# quit()
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
plt.legend(['DGO', 'Jacobi', 'DGO + Jacobi', 'Richardson'])
plt.show()

for i in np.arange(ratios.size, step=1):
    kappa = 10 ** kontrast[i]

    k = np.arange(max(map(len, norm_rr)))
    print(f'k \n {k}')
    lb = eigen_LB[i]
    print(f'lb \n {lb}')

    convergence = ((np.sqrt(kappa) - 1) / (np.sqrt(kappa) + 1)) ** k
    convergence = convergence * norm_rr[i][0]

    print(f'convergecnce \n {convergence}')
    fig = plt.figure()
    gs = fig.add_gridspec(1, 1)
    ax_1 = fig.add_subplot(gs[0, 0])
    ax_1.set_title(f'{i}', wrap=True)
    ax_1.semilogy(convergence, '--', label='estim', color='k')

    ax_1.semilogy(norm_rr[i], label=' Green', color='g')
    ax_1.semilogy(norm_rr_Jacobi[i], label=' Jacobi', color='b')
    ax_1.semilogy(norm_rr_combi[i], label=' Jacobi-Green', color='r')

    # x_1.plot(ratios, nb_pix_multips[i] * 32, zs=nb_it_Richardson[i], label='Richardson Green', color='green')
    # ax_1.plot(ratios, nb_pix_multips[i] * 32, zs=nb_it_Richardson_combi[i], label='Richardson Green+Jacobi')
    ax_1.set_xlabel('CG iterations')
    ax_1.set_ylabel('Norm of residua')
    plt.legend([r'$\kappa$ upper bound', 'Green', 'Jacobi', 'Green + Jacobi', 'Richardson'])
    ax_1.set_ylim([1e-10, norm_rr[i][0]])  # norm_rz[i][0]]/lb)
    print(max(map(len, norm_rr)))
    ax_1.set_xlim([0, max(map(len, norm_rr))])

    plt.show()
