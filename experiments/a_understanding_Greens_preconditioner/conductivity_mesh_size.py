

import os
import numpy as np
import time
import matplotlib as mpl
from mpi4py import MPI
from NuMPI.Tools import Reduction
import matplotlib.pyplot as plt

from NuMPI.IO import save_npy, load_npy
from matplotlib.animation import FuncAnimation, PillowWriter

from muFFTTO import domain
from muFFTTO import solvers
from muFFTTO import microstructure_library
from mpl_toolkits import mplot3d

script_name = os.path.splitext(os.path.basename(__file__))[0]
file_folder_path = os.path.dirname(os.path.realpath(__file__))
os.makedirs(file_folder_path, exist_ok=True)
data_folder_path = file_folder_path + '/exp_data/' + script_name + '/'
os.makedirs(data_folder_path, exist_ok=True)
figure_folder_path = file_folder_path + '/figures/' + script_name + '/'
os.makedirs(figure_folder_path, exist_ok=True)

problem_type = 'conductivity'
discretization_type = 'finite_element'
element_type = 'linear_triangles'

domain_size = [1, 1]
nb_pix_multips = [2]  # ,2,3,3,2,
tol_cg = 1e-7

ratios = np.arange(2,33)  # 65 17  33

nb_it = np.zeros((len(nb_pix_multips), ratios.size), )
nb_it_combi = np.zeros((len(nb_pix_multips), ratios.size), )
nb_it_Jacobi = np.zeros((len(nb_pix_multips), ratios.size), )
nb_it_Richardson = np.zeros((len(nb_pix_multips), ratios.size), )
nb_it_Richardson_combi = np.zeros((len(nb_pix_multips), ratios.size), )

norm_rr_combi = []
norm_rz_combi = []
norm_rr_Jacobi = []
norm_rz_Jacobi = []
norm_rr = []
norm_rz = []
norm_rMr_combi = []
norm_rMr = []
norm_rMr_Jacobi = []

# kontrast = []
# kontrast_2 = []
eigen_LB = []
kontrast=100
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
    macro_gradient = np.array([1.0, 1.0])
    # create material data field
    conductivity_C_0 = np.array([[1., 0], [0, 1.0]])
    refmaterial_data_ = np.copy(conductivity_C_0)


    # material distribution
    geometry_ID = 'n_laminate'  # 'square_inclusion'#'circle_inclusion'#


    # ax2 = fig.add_subplot(gs[0, 1])
    def scale_field(field, min_val, max_val):
        """Scales a 2D random field to be within [min_val, max_val]."""
        field_min, field_max = Reduction(MPI.COMM_WORLD).min(field), Reduction(MPI.COMM_WORLD).max(field)
        scaled_field = (field - field_min) / (field_max - field_min)  # Normalize to [0,1]
        return scaled_field * (max_val - min_val) + min_val  # Scale to [min_val, max_val]


    for i,ratio in enumerate(ratios):

        phase_field_smooth = microstructure_library.get_geometry(nb_voxels=discretization.nb_of_pixels,
                                                                 microstructure_name=geometry_ID,
                                                                 coordinates=discretization.fft.coords,
                                                                 parameter=ratio,
                                                                 contrast=1e-1
                                                                 )
        print(i + 2)
        print(f'parametr = {i + 2}')
        phase_field_smooth = np.abs(phase_field_smooth)
        # phase_field = np.random.rand(*discretization.get_scalar_sized_field().shape)  # set random distribution#

        # phase = 1 * np.ones(number_of_pixels)
        inc_contrast = 0.

        # nb_it=[]
        # nb_it_combi=[]
        # nb_it_Jacobi=[]
        phase_field_smooth = np.abs(phase_field_smooth)
        #phase_field = scale_field(phase_field, min_val=1, max_val=1e2)
#        phase_field = scale_field(phase_field, min_val=np.power(10, 1), max_val=10 ** 2)
        if kontrast == 100:
            phase_field_smooth = scale_field(phase_field_smooth, min_val=1, max_val=10 ** 2)
        elif kontrast == 10:
            phase_field_smooth = scale_field(phase_field_smooth, min_val=10, max_val=10 ** 2)
        elif kontrast == 2 :
            phase_field_smooth = scale_field(phase_field_smooth, min_val=50 , max_val=10 ** 2)

        # phase_field[phase_field<=0.001]= phase_field + 1e-4

        #phase_fem = np.zeros([2, *number_of_pixels])
        #phase_fnxyz = discretization.get_scalar_sized_field()
        phase_field = discretization.get_scalar_field(name='phase_field')
        phase_field.s[0, 0, ...] = phase_field_smooth


        material_data_field_C_0 = discretization.get_material_data_size_field_mugrid(name='algortihmic_tangent')
        material_data_field_C_0.s[...] = conductivity_C_0[..., np.newaxis, np.newaxis, np.newaxis] * \
                                         phase_field.s[np.newaxis, ...]
        # material_data_field_C_0_rho=phase_field_at_quad_poits_1qnxyz
        # Set up macro gradient field
        macro_gradient_field = discretization.get_gradient_size_field(name='macro_gradient_field')
        macro_gradient_field.sg.fill(0)
        discretization.get_macro_gradient_field_mugrid(macro_gradient_ij=macro_gradient,
                                                       macro_gradient_field_ijqxyz=macro_gradient_field)
        discretization.fft.communicate_ghosts(field=macro_gradient_field)

        # Set up right hand side
        rhs_field = discretization.get_unknown_size_field(name='rhs_field')
        rhs_field.sg.fill(0)
        discretization.get_rhs_mugrid(material_data_field_ijklqxyz=material_data_field_C_0,
                                      macro_gradient_field_ijqxyz=macro_gradient_field,
                                      rhs_inxyz=rhs_field)
        def K_fun(x, Ax):
            """
            Function to compute the product of the Hessian matrix with a vector.
            The Hessian is represented by the convolution operator.
            """

            discretization.apply_system_matrix_mugrid(material_data_field=material_data_field_C_0,
                                                      input_field_inxyz=x,
                                                      output_field_inxyz=Ax)
            discretization.fft.communicate_ghosts(Ax)

        # plotting eigenvalues
        ##  K = discretization.get_system_matrix(material_data_field_C_0_rho)
        ## M = discretization.get_system_matrix(refmaterial_data_field_I4s)

        ## eig = sc.linalg.eigh(a=K, b=M, eigvals_only=True)

        min_val = np.min(phase_field)
        max_val = np.max(phase_field)

        #kontrast.append(max_val / min_val)
        eigen_LB.append(min_val)

        # kontrast_2.append(eig[-3] / eig[np.argmax(eig > 0)])
        #kontrast_2.append((max_val / min_val) / 10)

        omega = 1  # 2 / ( eig[-1]+eig[np.argmax(eig>0)])
        # ax1.loglog(sorted(eig)[1:],label=f'{i}',marker='.', linewidth=0, markersize=1)
        # ax1.set_ylim([1e-5, 1e1])
        #
        # K_diag_half = np.copy(np.diag(K))
        # K_diag_half[K_diag_half < 9.99e-16] = 0
        # K_diag_half[K_diag_half != 0] = 1/np.sqrt(K_diag_half[K_diag_half != 0])
        #
        # DKDsym = np.matmul(np.diag(K_diag_half),np.matmul(K,np.diag(K_diag_half)))
        # eig = sc.linalg.eigh(a=DKDsym, b=M, eigvals_only=True)
        #
        # ax2.loglog(sorted(eig)[1:-2], label=f'{i}',marker='.', linewidth=0, markersize=1)
        # ax2.set_ylim([1e-5, 1e1])

        K = discretization.get_system_matrix_mugrid(material_data_field=material_data_field_C_0)
        # material_data_field_C_0=np.mean(material_data_field_C_0_rho,axis=(4,5,6))
        # mean_material=np.mean(material_data_field_C_0_rho,axis=(4,5,6))
        # material_data_field_C_0_ratio = np.einsum('ijkl,qxy->ijklqxy', mean_material,
        #                                     np.ones(np.array([discretization.nb_quad_points_per_pixel,
        #                                                       *discretization.nb_of_pixels])))

        preconditioner = discretization.get_preconditioner_Green_mugrid(reference_material_data_ijkl=refmaterial_data_)
        def M_fun_Green(x, Px):
            """
            Function to compute the product of the Preconditioner matrix with a vector.
            The Preconditioner is represented by the convolution operator.
            """
            discretization.fft.communicate_ghosts(x)
            discretization.apply_preconditioner_mugrid(preconditioner_Fourier_fnfnqks=preconditioner,
                                                       input_nodal_field_fnxyz=x,
                                                       output_nodal_field_fnxyz=Px)


        K_diag_alg = discretization.get_preconditioner_Jacobi_mugrid(
            material_data_field_ijklqxyz=material_data_field_C_0 )
        def M_fun_Jacobi(x, Px):
            Px.s[...] = K_diag_alg.s * K_diag_alg.s * x.s

        def M_fun_Green_Jacobi(x, Px):
            # discretization.fft.communicate_ghosts(x)
            x_jacobi_temp = discretization.get_unknown_size_field(name='x_jacobi_temp')

            x_jacobi_temp.s[...] = K_diag_alg.s * x.s
            discretization.apply_preconditioner_mugrid(preconditioner_Fourier_fnfnqks=preconditioner,
                                                       input_nodal_field_fnxyz=x_jacobi_temp,
                                                       output_nodal_field_fnxyz=Px)

            Px.s[...] = K_diag_alg.s * Px.s
        # # #
        # M_fun_Jacobi = lambda x: K_diag_alg * K_diag_alg * x

       # x0 = np.random.random(discretization.get_displacement_sized_field().shape)
        #x0 = np.zeros(discretization.get_displacement_sized_field().shape)
        x0_Green=discretization.get_unknown_size_field(name=f'x0_Green')
        norms_G = dict()
        norms_G['residual_rr'] = []
        norms_G['residual_rz'] = []

        def callback_G(it, x, r, p, z, stop_crit_norm):
            # global norms_cg_mech
            norm_of_rr = discretization.communicator.sum(np.dot(r.ravel(), r.ravel()))
            norm_of_rz = discretization.communicator.sum(np.dot(r.ravel(), z.ravel()))
            norms_G['residual_rr'].append(norm_of_rr)
            norms_G['residual_rz'].append(norm_of_rz)


        x0_Green.s.fill(0)
        x0_Green, norms = solvers.conjugate_gradients_mugrid_experimental(
            comm=discretization.communicator,
            fc=discretization.field_collection,
            hessp=K_fun,  # linear operator
            b=rhs_field,
            x=x0_Green,
            P=M_fun_Green ,
            tol=tol_cg,
            maxiter=10000,
            callback=callback_G,
            # rtol=True,
           # norm_metric=M_fun_Green
        )
        nb_it[kk - 1, i] = (len(norms_G['residual_rz']))
        norm_rz.append(norms_G['residual_rz'])
        norm_rr.append(norms_G['residual_rr'])
        plt.semilogy(norms_G['residual_rr'])
        plt.show()
        print(nb_it)
        #########
        # displacement_field_combi, norms_combi = solvers.PCG(K_fun, rhs, x0=x0, P=M_fun_combi, steps=int(1000),
        #                                                     toler=1e-6,
        #                                                     norm_type='data_scaled_rr',
        #                                                     norm_metric=M_fun)

        norms_GJ = dict()
        norms_GJ['residual_rr'] = []
        norms_GJ['residual_rz'] = []
        def callback_GJ(it, x, r, p, z, stop_crit_norm):
            # global norms_cg_mech
            norm_of_rr = discretization.communicator.sum(np.dot(r.ravel(), r.ravel()))
            norm_of_rz = discretization.communicator.sum(np.dot(r.ravel(), z.ravel()))
            norms_GJ['residual_rr'].append(norm_of_rr)
            norms_GJ['residual_rz'].append(norm_of_rz)
        x0_GreenJacobi=discretization.get_unknown_size_field(name=f'x0_GreenJacobi')
        x0_GreenJacobi.s.fill(0)

        x0_GreenJacobi, norms = solvers.conjugate_gradients_mugrid_experimental(
            comm=discretization.communicator,
            fc=discretization.field_collection,
            hessp=K_fun,  # linear operator
            b=rhs_field,
            x=x0_GreenJacobi,
            P=M_fun_Green_Jacobi,
            tol=tol_cg,
            maxiter=10000,
            callback=callback_GJ,
            # rtol=True,
            # norm_metric=M_fun_Green
        )



        nb_it_combi[kk - 1, i] = (len(norms_GJ['residual_rz']))
        norm_rz_combi.append(norms_GJ['residual_rz'])
        norm_rr_combi.append(norms_GJ['residual_rr'])
        #norm_rMr_combi.append(norms_GJ['data_scaled_rr'])
        print(nb_it_combi)

        #
        # displacement_field_Jacobi, norms_Jacobi = solvers.PCG(K_fun, rhs, x0=x0, P=M_fun_Jacobi, steps=int(1000),
        #                                                       toler=1e-6,
        #                                                     norm_type='data_scaled_rr',
        #                                                     norm_metric=M_fun)
        # nb_it_Jacobi[kk - 1, i] = (len(norms_Jacobi['residual_rz']))
        # norm_rz_Jacobi.append(norms_Jacobi['residual_rz'])
        # norm_rr_Jacobi.append(norms_Jacobi['residual_rr'])
        # norm_rMr_Jacobi.append(norms_Jacobi['data_scaled_rr'])
        #
        # displacement_field_Richardson, norms_Richardson = solvers.Richardson(K_fun, rhs, x0=x0, P=M_fun,
        #                                                                      omega=omega,
        #                                                                      steps=int(1000),
        #                                                                      toler=1e-6)
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
        # print(f'norm = {np.linalg.norm(displacement_field_combi[0, 0] - displacement_field[0, 0])}')
        ##################
        print(ratio)
        # print(
        #     '   nb_ steps CG Green of =' f'{nb_it}, residual_rz = {norm_rz}, residual_rr = {norm_rr},\n \
        #         nb_ steps CG Jacobi of =' f'{nb_it_Jacobi}, residual_rz = {norm_rz_Jacobi}, residual_rr = {norm_rr_Jacobi},\n\
        #         nb_ steps CG combi of =' f'{nb_it_combi}, residual_rz = {norm_rz_combi}, residual_rr = {norm_rr_combi},\n\
        #         nb_ steps Richardson of =' f'{nb_it_Richardson} , residual_rr = {norm_rr_Richardson},\n\
        #         nb_ steps Richardson of =' f'{nb_it_Richardson_combi} , residual_rr = {norm_rr_Richardson_combi}'
        # )
        _info = {}

        _info['nb_of_pixels'] = discretization.nb_of_pixels_global
        _info['nb_of_sampling_points'] = ratio
        # phase_field_sol_FE_MPI = xopt.x.reshape([1, 1, *discretization.nb_of_pixels])
#        _info['norm_rMr_G'] = norms_G['data_scaled_rr']
       #_info['norm_rMr_J'] = norms_Jacobi['data_scaled_rr']
     #   _info['norm_rMr_JG'] = norms_GJ['data_scaled_rr']
        _info['nb_it_G'] = nb_it
        _info['nb_it_J'] = nb_it_Jacobi
        _info['nb_it_JG'] = nb_it_combi
        file_data_name = (
            f'{script_name}_gID{geometry_ID}_T{number_of_pixels[0]}_G{ratio}_kappa{kontrast}.npy')
        save_npy(data_folder_path + file_data_name + f'.npy', phase_field.s[0].mean(axis=0),
                 tuple(discretization.subdomain_locations_no_buffers),
                 tuple(discretization.nb_of_pixels_global), MPI.COMM_WORLD)



        print(data_folder_path + file_data_name + f'.npy')

        if MPI.COMM_WORLD.rank == 0:
            np.savez(data_folder_path + file_data_name + f'xopt_log.npz', **_info)
            print(data_folder_path + file_data_name + f'.xopt_log.npz')
##################









fig = plt.figure()
gs = fig.add_gridspec(1, 1)
ax_1 = fig.add_subplot(gs[0, 0])
# ax_1.semilogy(norm_rr[0], label='PCG: Green', color='blue', linewidth=0)
for kk in np.arange(np.size(nb_pix_multips)):
    ax_1.plot(ratios[0:], nb_it[kk], 'g', marker='|', label=' Green', linewidth=1)
    # axs[1].plot(xopt2.f.num_iteration_.transpose()[1:3*i+1:3],"r", label='DGO ',linewidth=1)
    # axs[1].plot(xopt2.f.num_iteration_.transpose()[2:3*i+2:3],"r", label='DGO ',linewidth=1)

    # ax_1.plot(ratios[0:], nb_it_Jacobi[kk], "b", marker='o', label='PCG Jacobi', linewidth=1)#[0, 0:]
    # ax_1.plot(ratios[0:], nb_it_combi [kk], "k", marker='x', label='PCG Green + Jacobi', linewidth=1)
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
    kappa = kontrast[i]
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

    ax_1.semilogy(norm_rr[i]/norm_rr[i][0], label='PCG: Green', color='g')
    ax_1.semilogy(norm_rr_Jacobi[i]/norm_rr_Jacobi[i][0], label='PCG: Jacobi', color='b')
    ax_1.semilogy(norm_rMr_combi[i]/norm_rMr_combi[i][0], label='PCG: Jacobi Green', color='r')

    # x_1.plot(ratios, nb_pix_multips[i] * 32, zs=nb_it_Richardson[i], label='Richardson Green', color='green')
    # ax_1.plot(ratios, nb_pix_multips[i] * 32, zs=nb_it_Richardson_combi[i], label='Richardson Green+Jacobi')
    ax_1.set_xlabel('CG iterations')
    ax_1.set_ylabel('Norm of residua')
    plt.legend([r'$\kappa$ upper bound', 'Green', 'Jacobi', 'Green + Jacobi', 'Richardson'])
    ax_1.set_ylim([1e-19, 1e1])  # norm_rz[i][0]]/lb)  norm_rr[i][0]
    print(max(map(len, norm_rr)))
    ax_1.set_xlim([0, max(map(len, norm_rr))])

    plt.show()

plt.show()
fig = plt.figure()
gs = fig.add_gridspec(1, 1)
ax_1 = fig.add_subplot(gs[0, 0])
ax_1.semilogy(norm_rr[0], label='PCG: Green', color='blue', linewidth=0)

# ax_1.set_ylim([1e-7, 1e0])
ax_1.set_ylim([1e-7, 1e4])  # norm_rz[i][0]]/lb) norm_rr[0][0]

print(max(map(len, norm_rz)))
ax_1.set_xlim([0, max(map(len, norm_rr))])


def convergence_gif_rz(i):
    kappa = kontrast[i]
    k = np.arange(max(map(len, norm_rr)))
    print(f'k \n {k}')
    lb = eigen_LB[i]
    print(f'lb \n {lb}')

    convergence = ((np.sqrt(kappa) - 1) / (np.sqrt(kappa) + 1)) ** k
    convergence = convergence * norm_rr[i][0]

    ax_1.clear()

    ax_1.set_title(f'{i}', wrap=True)
    ax_1.semilogy(convergence, '--', label='estim', color='g')

    ax_1.semilogy(norm_rr[i], label='PCG: Green', color='g')
    ax_1.semilogy(norm_rr_Jacobi[i], label='PCG: Jacobi', color='b')
    ax_1.semilogy(norm_rr_combi[i], label='PCG: Jacobi', color='k')

    # x_1.plot(ratios, nb_pix_multips[i] * 32, zs=nb_it_Richardson[i], label='Richardson Green', color='green')
    # ax_1.plot(ratios, nb_pix_multips[i] * 32, zs=nb_it_Richardson_combi[i], label='Richardson Green+Jacobi')
    ax_1.set_xlabel('CG iterations')
    ax_1.set_ylabel('Norm of residua')
    plt.legend([r'$\kappa$ upper bound', 'Green', 'Jacobi', 'DGO + Jacobi', 'Richardson'])
    ax_1.set_ylim([1e-7, 1e5])  # norm_rr[i][0] # norm_rz[i][0]]/lb)
    print(max(map(len, norm_rr)))
    ax_1.set_xlim([0, max(map(len, norm_rr))])
    # axs[1].legend()
    # plt.legend(['', 'FEM: Green', 'FEM: Jacobi', 'FEM: Green + Jacobi','FEM: Richardson'])
    plt.legend(
        [r'$\kappa$ upper bound', 'Green', 'Jacobi', 'Green + Jacobi', 'Richardson Green', 'Richardson Green + Jacobi'],
        loc='center left', bbox_to_anchor=(0.8, 0.5))


ani = FuncAnimation(fig, convergence_gif_rz, frames=ratios.size, blit=False)
# axs[1].legend()middlemiddle
# Save as a GIF
ani.save(
    f"../figures/convergence__es2tgif_{number_of_pixels[0]}comparison{ratios[-1]}_RichardsonJacobi{geometry_ID}_circle_inc_to_smooth_semiloplots3.gif",
    writer=PillowWriter(fps=1))

plt.show()
# -------------------------------------------------------------------------------------------------------
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

ax_1.set_ylim([1e-7, 1e5])  # 1e0
print(max(map(len, norm_rr)))
ax_1.set_xlim([0, max(map(len, norm_rr))])


def convergence_gif(i):
    kappa = kontrast[i]
    k = np.arange(max(map(len, norm_rr)))
    print(f'k \n {k}')

    convergence = ((np.sqrt(kappa) - 1) / (np.sqrt(kappa) + 1)) ** k
    convergence = convergence * norm_rr[i][0]
    print(f'convergecnce \n {convergence}')
    ax_1.clear()

    ax_1.set_title(f'{i}', wrap=True)
    ax_1.semilogy(convergence, '--', label='estim', color='k')

    ax_1.semilogy(norm_rr[i], label='PCG: Green', color='r')
    ax_1.semilogy(norm_rr_Jacobi[i], label='PCG: Jacobi', color='b')
    # ax_1.semilogy(norm_rr_combi[i], label='PCG: Green + Jacobi', color='red')
    # x_1.plot(ratios, nb_pix_multips[i] * 32, zs=nb_it_Richardson[i], label='Richardson Green', color='green')
    # ax_1.plot(ratios, nb_pix_multips[i] * 32, zs=nb_it_Richardson_combi[i], label='Richardson Green+Jacobi')
    ax_1.set_xlabel('CG iterations')
    ax_1.set_ylabel('Norm of residuals')
    plt.legend([r'$\kappa$ upper bound', 'Green', 'Jacobi', 'Green + Jacobi', 'Richardson'])
    ax_1.set_ylim([1e-7, 1e5])  # 1e0
    print(max(map(len, norm_rr)))
    ax_1.set_xlim([0, max(map(len, norm_rr))])
    # axs[1].legend()
    # plt.legend(['', 'FEM: Green', 'FEM: Jacobi', 'FEM: Green + Jacobi','FEM: Richardson'])
    plt.legend(
        [r'$\kappa$ upper bound', 'Green', 'Jacobi', 'Green + Jacobi', 'Richardson Green', 'Richardson Green + Jacobi'],
        loc='center left', bbox_to_anchor=(0.8, 0.5))


ani = FuncAnimation(fig, convergence_gif, frames=ratios.size, blit=False)
# axs[1].legend()middlemiddle
# Save as a GIF
ani.save(
    f"../figures/convergence_gif_{number_of_pixels[0]}comparison{ratios[-1]}_RichardsonJacobi{geometry_ID}_circle_inc_to_smooth_semiloplots3.gif",
    writer=PillowWriter(fps=1))

plt.show()

plot_evolion = True
if plot_evolion:
    for nb_tiles in [1, ]:
        fig = plt.figure(figsize=(11, 4.5))
        gs = fig.add_gridspec(1, 1)
        # ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[0, 0])
        # ax1.step(np.arange(phase_field[:, phase_field.shape[0] // 2].size), phase_field[:, phase_field.shape[0] // 2],
        #         linewidth=0)
        # ax1.set_ylim([1e-4, 1])
        ax2.plot(ratios, nb_it_Jacobi[0], label='nb_it_Laplace', linewidth=0)
        ax2.set_ylabel('# PCG iterations')
        ax2.set_xlabel('# material phases')

        counter = 0
        # for i in np.array([0, ratios.size // 4 - 1, ratios.size - 1]):
        # ratio = ratios[i]

        # phase_field = microstructure_library.get_geometry(nb_voxels=discretization.nb_of_pixels,
        #                                                   microstructure_name=geometry_ID,
        #                                                   coordinates=discretization.fft.coords,
        #                                                   parameter=ratio,
        #                                                   contrast=1e-4
        #                                                   )
        #
        # linestyles = ['-', '--', ':']
        # extended_x = np.arange(phase_field[:, phase_field.shape[0] // 2].size + 1)
        # extended_y = np.append(phase_field[:, phase_field.shape[0] // 2],
        #                        phase_field[:, phase_field.shape[0] // 2][-1])
        # ax1.step(extended_x,extended_y
        #          ,where='post',
        #          linewidth=1, color='k', linestyle=linestyles[counter], marker='|',
        #          label=f'{ratios[i]} phases')
        # ax1.set_ylim([-0.1, 1.1])
        # ax1.set_xlim([0, phase_field.shape[0]])
        # ax1.set_yticks([0.001, 0.25, 0.50, 0.75, 1.0001])
        # ax1.set_yticklabels([0.001, 0.25, 0.50, 0.75, 1.00])
        # ax1.legend(loc="upper left")
        #
        # ax1.set_title(f'Cross sections')
        # ax1.set_ylabel('Young modulus (Pa)')
        # ax1.set_xlabel('Nodal point index')

        ax2.plot(ratios, nb_it[0], 'g', marker='o', label=' Green', linewidth=1, markerfacecolor='white')
        ax2.plot(ratios, nb_it_Jacobi[0], "b", marker='^', label='PCG Jacobi', linewidth=1, markerfacecolor='white')
        ax2.plot(ratios, nb_it_combi[0], "k", marker='x', label='PCG Green + Jacobi', linewidth=1,
                 markerfacecolor='white')

        ax2.set_ylim(bottom=1)
        ax2.set_xlim([2, ratios.size + 1])
        ax2.set_xticks(np.concatenate(([2, ], np.arange(5, ratios.size + 1, 5), [ratios.size + 1, ])))
        ax2.set_xticklabels(np.concatenate(([2, ], np.arange(5, ratios.size + 1, 5), [ratios.size + 1, ])))
        ax2.legend(['', 'Green', 'Jacobi', 'Green + Jacobi'])

        counter += 1
        plt.tight_layout()
    fname = src + 'introduction_1_{}{}'.format(number_of_pixels[0], '.pdf')
    print(('create figure: {}'.format(fname)))
    plt.savefig(fname, bbox_inches='tight')
    plt.show()

    for nb_tiles in [1, ]:
        # fig = plt.figure()

        #
        # fig, axs = plt.subplots(nrows=2, ncols=2,
        #                         figsize=(6, 6)  )
        fig = plt.figure()
        gs = fig.add_gridspec(2, 3)

        ax1 = fig.add_subplot(gs[1, :])
        # axs[0] = plt.axes(xlim=(0, nb_tiles * N), ylim=(0, nb_tiles * N))
        # ax1.imshow(phase_field, cmap=mpl.cm.Greys, vmin=1e-4, vmax=1)

        ax1.step(np.arange(phase_field[:, phase_field.shape[0] // 2].size), phase_field[:, phase_field.shape[0] // 2],
                 linewidth=0)
        # ax3.plot(phase_field[:,phase_field.shape[0]//2], linewidth=0)
        ax1.set_ylim([1e-4, 1e5])  # 1e5 1
        # print(ratios)

        # print(nb_it)
        ax2.plot(ratios, nb_it_Jacobi[0], label='nb_it_Laplace', linewidth=0)
        # ax1.set_ylim([1e0, 1e3 ])

        # axs[1].plot(xopt.f.num_iteration_.transpose()[::3], 'w'  , linewidth=0)
        # axs[1].plot(xopt3.f.num_iteration_.transpose(), "b", label='Jacoby', linewidth=0)
        # axs[1].plot(xopt.f.num_iteration_.transpose(), "k", label='DGO + Jacoby', linewidth=0)
        # legend = plt.legend()
        # Animation function to update the image
        # ax2.set_xlabel('')
        ax2.set_ylabel('# PCG iterations')
        ax2.set_xlabel('# material phases')
        x = np.arange(0, nb_tiles * number_of_pixels[0])
        y = np.arange(0, nb_tiles * number_of_pixels[1])
        X, Y = np.meshgrid(x, y)
        linestyles = ['-', '--', ':']
        colors = ['red', 'blue', 'green', 'orange', 'purple']

        counter = 0
        for i in np.array([0, ratios.size // 4 - 1, ratios.size - 1]):

            ratio = ratios[i]

            phase_field = microstructure_library.get_geometry(nb_voxels=discretization.nb_of_pixels,
                                                              microstructure_name=geometry_ID,
                                                              coordinates=discretization.fft.coords,
                                                              parameter=ratio,
                                                              contrast=1e-4
                                                              )

            ax0 = fig.add_subplot(gs[0, counter])

            ax0.pcolormesh(X, Y, np.transpose(phase_field), cmap=mpl.cm.Greys, vmin=1e-4, vmax=1, linewidth=0,
                           rasterized=True)
            ax0.set_xticks(np.arange(-.5, number_of_pixels[0], int(number_of_pixels[0] / 4)))
            ax0.set_yticks(np.arange(-.5, number_of_pixels[1], int(number_of_pixels[1] / 4)))
            ax0.set_xticklabels(np.arange(0, number_of_pixels[0] + 1, int(number_of_pixels[0] / 4)))
            ax0.set_yticklabels(np.arange(0, number_of_pixels[1] + 1, int(number_of_pixels[1] / 4)))
            ax0.set_title(f'{ratios[i]} phases')
            ax0.hlines(y=10, xmin=-0.5, xmax=number_of_pixels[0] - 0.5, color=colors[counter],
                       linestyle=linestyles[counter], linewidth=1.)
            if counter == 0:
                ax0.set_ylabel('y coordinate')
                ax0.set_xlabel('x coordinate')
            # ax0.hlines(y=1, xmin=0, xmax=number_of_pixels[0], colors='black', linestyles='--', linewidth=1.)
            # phase_field = np.abs(phase_field)  # -1
            # phase_field += 1e-4
            # min_val = np.min(phase_field)
            # max_val = np.max(phase_field)
            # phase_field = 9.99e-1 + (phase_field - min_val) * (1 - 9.99e-1) / (max_val - min_val)
            # phase_field = ratio * phase_field_smooth + (1 - ratio) * phase_field_pwconst

            # ax1.clear()
            # ax1.imshow(np.transpose( phase_field), cmap=mpl.cm.Greys, vmin=1e-4, vmax=1)
            # ax1.set_title(r'Density $\rho$', wrap=True)

            # #: {np.max(phase_field)/np.min(phase_field):.1e}  \n'                          f'  min = {np.min(phase_field):.1e}
            # ax3.clear()

            extended_x = np.arange(phase_field[:, phase_field.shape[0] // 2].size + 1)
            extended_y = np.append(phase_field[:, phase_field.shape[0] // 2],
                                   phase_field[:, phase_field.shape[0] // 2][-1])
            ax1.step(extended_x, extended_y
                     , where='post',
                     linewidth=1, color=colors[counter], linestyle=linestyles[counter], marker='|',
                     label=f'{ratios[i]} phases')
            # ax3.plot(phase_field[:, phase_field.shape[0] // 2], linewidth=1)
            ax1.set_ylim([-0.1, 1.1])
            ax1.set_xlim([0, phase_field.shape[0]])
            ax1.set_yticks([0.001, 0.25, 0.50, 0.75, 1.0001])
            ax1.set_yticklabels([0.001, 0.25, 0.50, 0.75, 1.00])
            # ax1.yaxis.set_ticks_position([0.001,0.25,0.5,0.75, 1])
            # ax2.legend(['2 phases', f'{ratio} phases', 'Jacobi', 'Green + Jacobi'])
            ax1.legend(loc="upper left")

            ax1.set_title(f'Cross sections')
            ax1.set_ylabel('Young modulus (Pa)')
            ax1.set_xlabel('x coordinate')

            # ax2.plot(ratios[0:i + 1], nb_it[0, 0:i + 1], 'g', marker='|', label=' Green', linewidth=1)
            # axs[1].plot(xopt2.f.num_iteration_.transpose()[1:3*i+1:3],"r", label='DGO ',linewidth=1)
            # axs[1].plot(xopt2.f.num_iteration_.transpose()[2:3*i+2:3],"r", label='DGO ',linewidth=1)

            # ax2.plot(ratios[0:i + 1], nb_it_Jacobi[0, 0:i + 1], "b", marker='o', label='PCG Jacobi', linewidth=1)
            # ax2.plot(ratios[0:i + 1], nb_it_combi[0, 0:i + 1], "k", marker='x', label='PCG Green + Jacobi', linewidth=1)
            #  ax2.semilogy(ratios[0:i + 1], nb_it_Richardson[0, 0:i + 1], "g", label=' Richardson Green ', linewidth=1)
            #  ax2.semilogy(ratios[0:i + 1], nb_it_Richardson_combi[0, 0:i + 1], "y",  label=' Richardson Green + Jacobi ', linewidth=1)
            # ax2.set_ylim(bottom=0)
            # axs[1].legend()
            # ax2.legend(['', 'Green', 'Jacobi', 'Green + Jacobi'])
            counter += 1
        plt.tight_layout()
    fname = src + 'introduction_geometry_1_{}{}'.format(number_of_pixels[0], '.pdf')
    print(('create figure: {}'.format(fname)))
    plt.savefig(fname, bbox_inches='tight')
    plt.show()

fig = plt.figure(figsize=(11, 4.5))
gs = fig.add_gridspec(1, 1)
ax_1 = fig.add_subplot(gs[0, 0])
ax_1.set_title(f'Convergence', wrap=True)
counter = 0

for i in np.array([0, ratios.size // 4 - 1, ratios.size - 1]):
    # kappa=kontrast[i]
    # k=np.arange(max(map(len, norm_rr)))
    # print(f'k \n {k}')
    # lb=eigen_LB[i]
    # print(f'lb \n {lb}')
    #
    # convergence=((np.sqrt(kappa)-1)/(np.sqrt(kappa)+1))**k
    # convergence=convergence*norm_rr[i][0]

    # print(f'convergecnce \n {convergence}')

    # ax_1.semilogy(convergence, '--',label='estim', color='k')

    ax_1.semilogy(np.arange(1, np.size(norm_rr[i]) + 1), norm_rr[i], label=f'Green - {ratios[i]} phases', color='green',
                  linestyle=linestyles[counter], marker='o', markerfacecolor='white')
    ax_1.semilogy(np.arange(1, np.size(norm_rr_Jacobi[i]) + 1), norm_rr_Jacobi[i], label=f'Jacobi - {ratios[i]} phases',
                  color='blue', linestyle=linestyles[counter],
                  marker='^', markerfacecolor='white')
    ax_1.semilogy(np.arange(1, np.size(norm_rr_combi[i]) + 1), norm_rr_combi[i],
                  label=f'Green + Jacobi - {ratios[i]} phases ', color='black', linestyle=linestyles[counter],
                  marker='x', markerfacecolor='white')
    # x_1.plot(ratios, nb_pix_multips[i] * 32, zs=nb_it_Richardson[i], label='Richardson Green', color='green')
    # ax_1.plot(ratios, nb_pix_multips[i] * 32, zs=nb_it_Richardson_combi[i], label='Richardson Green+Jacobi')
    ax_1.set_xlabel('PCG iteration')
    ax_1.set_ylabel('Norm of residua')
    plt.legend(ncol=3, loc='lower center')
    # ['Green - 2 phases', 'Jacobi - 2 phases', 'Green + Jacobi - 2 phases',
    #  f'Green - {ratios[i]} phases', f'  Jacobi - {ratios[i]} phases', f'Green + Jacobi - {ratios.size // 4 - 1} phases',
    #  f'Green - {ratios.size - 1} phases', f'Jacobi - {ratios[i]} phases', f'Green + Jacobi - {ratios.size - 1} phases']
    ax.set_yscale('symlog')
    # ax.set_xscale('symlog')
    ax_1.set_ylim([1e-12, 1e5])  # 1e1  norm_rz[i][0]]/lb) norm_rr[i][0]
    # ax_1.set_ylim([0, 1e1])
    print(max(map(len, norm_rr)))
    ax_1.set_xlim([1, max(map(len, norm_rr))])
    counter += 1
fname = src + 'introduction_convergence_1_{}{}'.format(number_of_pixels[0], '.pdf')
print(('create figure: {}'.format(fname)))
plt.savefig(fname, bbox_inches='tight')
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
        ax3.step(np.arange(phase_field[:, phase_field.shape[0] // 2].size), phase_field[:, phase_field.shape[0] // 2],
                 linewidth=0)
        # ax3.plot(phase_field[:,phase_field.shape[0]//2], linewidth=0)
        ax3.set_ylim([1e-4, 1])
        # print(ratios)

        # print(nb_it)
        ax2.plot(ratios, nb_it_Jacobi[0], label='nb_it_Laplace', linewidth=0)
        ax3.set_ylim([1e0, 1e3])

        # axs[1].plot(xopt.f.num_iteration_.transpose()[::3], 'w'  , linewidth=0)
        # axs[1].plot(xopt3.f.num_iteration_.transpose(), "b", label='Jacoby', linewidth=0)
        # axs[1].plot(xopt.f.num_iteration_.transpose(), "k", label='DGO + Jacoby', linewidth=0)
        # legend = plt.legend()
        # Animation function to update the image
        # ax2.set_xlabel('')
        ax2.set_ylabel('# PCG iterations')
        ax2.set_xlabel('# material phases')


        def update(i):
            ratio = ratios[i]

            phase_field = microstructure_library.get_geometry(nb_voxels=discretization.nb_of_pixels,
                                                              microstructure_name=geometry_ID,
                                                              coordinates=discretization.fft.coords,
                                                              parameter=i + 2,
                                                              contrast=1e-4
                                                              )
            # phase_field = np.abs(phase_field)  # -1
            # phase_field += 1e-4
            # min_val = np.min(phase_field)
            # max_val = np.max(phase_field)
            # phase_field = 9.99e-1 + (phase_field - min_val) * (1 - 9.99e-1) / (max_val - min_val)
            # phase_field = ratio * phase_field_smooth + (1 - ratio) * phase_field_pwconst

            ax1.clear()
            ax1.imshow(np.transpose(phase_field), cmap=mpl.cm.Greys, vmin=1e-4, vmax=1)
            ax1.set_title(r'Density $\rho$', wrap=True)
            #: {np.max(phase_field)/np.min(phase_field):.1e}  \n'                          f'  min = {np.min(phase_field):.1e}
            ax3.clear()
            ax3.step(np.arange(phase_field[:, phase_field.shape[0] // 2].size),
                     phase_field[:, phase_field.shape[0] // 2], linewidth=1)
            # ax3.plot(phase_field[:, phase_field.shape[0] // 2], linewidth=1)
            ax3.set_ylim([1e-4, 1])
            ax3.set_title(f'Cross section')

            ax2.plot(ratios[0:i + 1], nb_it[0, 0:i + 1], 'g', marker='|', label=' Green', linewidth=1)
            # axs[1].plot(xopt2.f.num_iteration_.transpose()[1:3*i+1:3],"r", label='DGO ',linewidth=1)
            # axs[1].plot(xopt2.f.num_iteration_.transpose()[2:3*i+2:3],"r", label='DGO ',linewidth=1)

            ax2.plot(ratios[0:i + 1], nb_it_Jacobi[0, 0:i + 1], "b", marker='o', label='PCG Jacobi', linewidth=1)
            ax2.plot(ratios[0:i + 1], nb_it_combi[0, 0:i + 1], "k", marker='x', label='PCG Green + Jacobi', linewidth=1)
            #  ax2.semilogy(ratios[0:i + 1], nb_it_Richardson[0, 0:i + 1], "g", label=' Richardson Green ', linewidth=1)
            #  ax2.semilogy(ratios[0:i + 1], nb_it_Richardson_combi[0, 0:i + 1], "y",  label=' Richardson Green + Jacobi ', linewidth=1)
            ax2.set_ylim(bottom=0)
            # axs[1].legend()
            ax2.legend(['', 'Green', 'Jacobi', 'Green + Jacobi'])
            # plt.legend(['', 'FEM: Green', 'FEM: Jacobi', 'FEM: Green + Jacobi','FEM: Richardson'])


        # plt.legend(['', ' Green', 'Jacobi', 'Green + Jacobi','Richardson Green','Richardson Green + Jacobi'],loc='best', bbox_to_anchor=(0.7, 0.5))
        #        ax2.legend([r'$\kappa$ upper bound', 'Green', 'Jacobi', 'Green + Jacobi', 'Richardson'])

        # plt.legend([ '', 'Green', 'Jacobi'  ])

        # img.set_array(xopt_it)
        # ax2.legend([r'$\kappa$ upper bound', 'Green', 'Jacobi', 'Green + Jacobi', 'Richardson'])

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
        ani.save(
            f"../figures/movie_exp_paper_JG_intro_{number_of_pixels[0]}comparison{ratios[-1]}_RichardsonJacobi{geometry_ID}_circle_inc_to_smooth_semiloplots3.gif",
            writer=PillowWriter(fps=4))

    plt.show()
