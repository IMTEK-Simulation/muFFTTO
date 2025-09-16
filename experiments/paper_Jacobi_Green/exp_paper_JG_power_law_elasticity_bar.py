import numpy as np
import scipy as sc
import time

from keras.src.ops import arange
from mpi4py import MPI
from NuMPI.IO import save_npy, load_npy
import matplotlib as mpl
from matplotlib import pyplot as plt

from muFFTTO import domain
from muFFTTO import solvers
from muFFTTO import microstructure_library

problem_type = 'elasticity'
discretization_type = 'finite_element'
element_type = 'trilinear_hexahedron'
formulation = 'small_strain'

number_of_pixels = (31, 31, 1)
domain_size = [1, 1, 1]
my_cell = domain.PeriodicUnitCell(domain_size=domain_size,
                                  problem_type=problem_type)

discretization = domain.Discretization(cell=my_cell,
                                       nb_of_pixels_global=number_of_pixels,
                                       discretization_type=discretization_type,
                                       element_type=element_type)
start_time = time.time()
print(f'{MPI.COMM_WORLD.rank:6} {MPI.COMM_WORLD.size:6} {str(discretization.fft.nb_domain_grid_pts):>15} '
      f'{str(discretization.fft.nb_subdomain_grid_pts):>15} {str(discretization.fft.subdomain_locations):>15}')

# create material data field
# K_0, G_0 = 1, 0.5 #domain.get_bulk_and_shear_modulus(E=1, poison=0.2)
K_0, G_0 = domain.get_bulk_and_shear_modulus(E=1, poison=0.2)

elastic_C_1 = domain.get_elastic_material_tensor(dim=discretization.domain_dimension,
                                                 K=K_0,
                                                 mu=G_0,
                                                 kind='linear')

material_data_field_C_0 = discretization.get_material_data_size_field(name='elastic_tensor')

# material distribution
# phase_field_smooth = microstructure_library.get_geometry(nb_voxels=discretization.nb_of_pixels,
#                                                          microstructure_name=geometry_ID,
#                                                          coordinates=discretization.fft.coords)
#
# phase_field = discretization.get_custom_size_nodal_field(name='phase_field', shape=(1,))
# # phase_field[0,0]= phase_field_l
# phase_field.s[0, 0] = phase_field_smooth

# phase_field[0,0]=phase_field[0,0]/np.min(phase_field[0,0])

# np.save('geometry_jacobi.npy', np.power(phase_field_l, 2),)
# sc.io.savemat('geometry_jacobi.mat', {'data':  np.power(phase_field_l, 2)})


# identity tensor                                               [single tensor]
i = np.eye(discretization.domain_dimension)
I = np.einsum('ij,xyz', i, np.ones(number_of_pixels))

# identity tensors                                            [grid of tensors]
I4 = np.einsum('il,jk', i, i)
I4rt = np.einsum('ik,jl', i, i)
II = np.einsum('ij...  ,kl...  ->ijkl...', i, i)
I4s = (I4 + I4rt) / 2.
I4d = (I4s - II / 3.)

II_xyz = np.broadcast_to(II[..., np.newaxis, np.newaxis, np.newaxis],
                         (3, 3, 3, 3, *number_of_pixels))
II_qxyz = np.broadcast_to(II[..., np.newaxis, np.newaxis, np.newaxis, np.newaxis],
                          (3, 3, 3, 3, discretization.nb_quad_points_per_pixel, *number_of_pixels))
I4d_xyz = np.broadcast_to(I4d[..., np.newaxis, np.newaxis, np.newaxis],
                          (3, 3, 3, 3, *number_of_pixels))
I4d_qxyz = np.broadcast_to(I4d[..., np.newaxis, np.newaxis, np.newaxis, np.newaxis],
                           (3, 3, 3, 3, discretization.nb_quad_points_per_pixel, *number_of_pixels))

# assembly preconditioner
preconditioner = discretization.get_preconditioner_NEW(reference_material_data_ijkl=I4s)

M_fun = lambda x: discretization.apply_preconditioner_NEW(preconditioner_Fourier_fnfnqks=preconditioner,
                                                          nodal_field_fnxyz=x)


# M_fun = lambda x: 1 * x

# linear elasticity
# -----------------

def linear_elastic_pixel(strain_ijxyz, K):
    # parameters
    # bulk  modulus
    mu = 1.  # shear modulus

    # elastic stiffness tensor, and stress response
    C4 = K * II_xyz + 2. * mu * I4d_xyz

    sig = np.einsum('ijklxyz,lkxyz  ->ijxyz  ', C4, strain_ijxyz)
    # sig = ddot42(C4, strain)

    return sig, C4


def linear_elastic_q_points(strain_ijqxyz, K):
    # parameters
    # bulk  modulus
    mu = 1.  # shear modulus

    # elastic stiffness tensor, and stress response
    C4 = K * II_qxyz + 2. * mu * I4d_qxyz

    sig = np.einsum('ijklqxyz,lkqxyz  ->ijqxyz  ', C4, strain_ijqxyz)
    # sig = ddot42(C4, strain)

    return sig, C4


def nonlinear_elastic_pixel(eps, K):
    # K = 2.  # bulk modulus
    sig0 = 0.5  # 0.25 #* K  # reference stress
    eps0 = 0.1  # 0.2  # reference strain
    n = 10.0  # 3.0  # hardening exponent

    ddot22_I = lambda A2, B2: np.einsum('ijxyz  ,ji  ->xyz    ', A2, B2)
    epsm = ddot22_I(eps, i) / 3.

    epsd = eps - epsm * I
    ddot22 = lambda A2, B2: np.einsum('ijxyz  ,jixyz  ->xyz    ', A2, B2)

    epseq = np.sqrt(2. / 3. * ddot22(epsd, epsd))

    sig = 3. * K * epsm * I + 2. / 3. * sig0 / (eps0 ** n) * (epseq ** (n - 1.)) * epsd
    # sig = 3. * K * epsm * I * (epseq == 0.).astype(float) + sig * (epseq != 0.).astype(float)

    dyad22 = lambda A2, B2: np.einsum('ijxyz  ,klxyz  ->ijklxyz', A2, B2)

    K4_d = 2. / 3. * sig0 / (eps0 ** n) * (
            dyad22(epsd, epsd) * 2. / 3. * (n - 1.) * epseq ** (n - 3.) + epseq ** (n - 1.) * I4d_xyz)

    threshold = 1e-15
    mask = (np.abs(epseq) > threshold).astype(float)

    K4 = K * II_xyz + K4_d * mask

    return sig, K4


###
def nonlinear_elastic_q_points(strain, K):
    # K = 2.  # bulk modulus
    # sigma = K*trace(small_strain)*I_ij  + sigma_0* (strain_eq/epsilon_0)^n * N_ijkl
    sig0 = 0.5  # 0.25 #* K  # reference stress
    eps0 = 0.1  # 0.2  # reference strain
    n = 10.0  # 3.0  # hardening exponent

    strain_trace_qxyz = np.einsum('ii...', strain) / 3  # todo{2 or 3 in 2D }
    # strain_trace_xyz = np.einsum('ijxyz,ji ->xyz', strain, I) / 3  # todo{2 or 3 in 2D }

    # volumetric strain
    strain_vol_ijqxyz = np.ndarray(shape=strain.shape)
    strain_vol_ijqxyz.fill(0)
    for d in arange(discretization.domain_dimension):
        strain_vol_ijqxyz[d, d, ...] = strain_trace_qxyz

    # deviatoric strain
    strain_dev_ijqxyz = strain - strain_vol_ijqxyz

    # equivalent strain
    strain_dev_ddot = np.einsum('ijqxyz,jiqxyz-> qxyz', strain_dev_ijqxyz, strain_dev_ijqxyz)
    strain_eq_qxyz = np.sqrt((2. / 3.) * strain_dev_ddot)

    #
    sig = (3. * K * strain_vol_ijqxyz
           + 2. / 3. * sig0 / (eps0 ** n) *
           (strain_eq_qxyz ** (n - 1.)) * strain_dev_ijqxyz)
    #
    # sig = 3. * K * strain_vol_ijqxyz * (strain_eq_qxyz == 0.).astype(float) + sig * (
    #         strain_eq_qxyz != 0.).astype(float)

    # K4_d = discretization.get_material_data_size_field(name='alg_tangent')
    strain_dev_dyad = np.einsum('ijqxyz,klqxyz->ijklqxyz', strain_dev_ijqxyz, strain_dev_ijqxyz)

    K4_d = 2. / 3. * sig0 / (eps0 ** n) * (strain_dev_dyad * 2. / 3. * (n - 1.) * strain_eq_qxyz ** (
            n - 3.) + strain_eq_qxyz ** (n - 1.) * I4d_qxyz)

    threshold = 1e-15
    mask = (np.abs(strain_eq_qxyz) > threshold).astype(float)

    K4 = K * II_qxyz + K4_d * mask  # *(strain_equivalent_qxyz != 0.).astype(float)

    return sig, K4


def constitutive_pixel(strain_ijqxyz):
    phase_field = np.zeros([*number_of_pixels])
    # phase_field[:number_of_pixels[0] // 2, :number_of_pixels[0] // 2, :] = 1.
    phase_field[:26, :, :] = 1.
    # sig_P1, K4_P1 = nonlin_elastic(strain_ijqxyz.s.mean(axis=2), K=100)
    # sig_P2, K4_P2 = nonlin_elastic(strain_ijqxyz.s.mean(axis=2), K=1)
    sig_P1, K4_P1 = nonlinear_elastic_pixel(strain_ijqxyz.s.mean(axis=2), K=2)
    # sig_P2, K4_P2 = nonlinear_elastic_pixel(strain_ijqxyz.s.mean(axis=2), K=2)

    # sig_P1, K4_P1 = linear_elastic_pixel(strain_ijqxyz.s.mean(axis=2), K=2)
    sig_P2, K4_P2 = linear_elastic_pixel(strain_ijqxyz.s.mean(axis=2), K=2)

    sig_ijxyz = phase_field * sig_P1 + (1. - phase_field) * sig_P2
    K4_ijklxyz = phase_field * K4_P1 + (1. - phase_field) * K4_P2

    sig_ijqxyz = np.broadcast_to(sig_ijxyz[:, :, np.newaxis, ...],
                                 (3, 3, discretization.nb_quad_points_per_pixel, *number_of_pixels))

    K4_ijklqxyz = np.broadcast_to(K4_ijklxyz[:, :, :, :, np.newaxis, ...],
                                  (3, 3, 3, 3, discretization.nb_quad_points_per_pixel, *number_of_pixels))
    return sig_ijqxyz, K4_ijklqxyz


def constitutive_q_points(strain_ijqxyz):
    phase_field = np.zeros([*number_of_pixels])
    # phase_field[:number_of_pixels[0] // 2, :number_of_pixels[0] // 2, :] = 1.
    phase_field[:26, :, :] = 1.

    sig_P1, K4_P1 = nonlinear_elastic_q_points(strain_ijqxyz.s, K=2)
    # sig_P2, K4_P2 = nonlinear_elastic_q_points(strain_ijqxyz.s, K=2)

    # sig_P1, K4_P1 = linear_elastic_q_points(strain_ijqxyz.s, K=2)
    sig_P2, K4_P2 = linear_elastic_q_points(strain_ijqxyz.s, K=2)

    sig_ijqxyz = phase_field * sig_P1 + (1. - phase_field) * sig_P2
    K4_ijklqxyz = phase_field * K4_P1 + (1. - phase_field) * K4_P2

    return sig_ijqxyz, K4_ijklqxyz


def constitutive(strain_ijqxyz):
    pixel_constant = False
    if pixel_constant:
        sig_ijqxyz, K4_ijklqxyz = constitutive_pixel(strain_ijqxyz)
    else:
        sig_ijqxyz, K4_ijklqxyz = constitutive_q_points(strain_ijqxyz)

    return sig_ijqxyz, K4_ijklqxyz


macro_gradient_inc_field = discretization.get_gradient_size_field(name='macro_gradient_inc_field')

displacement_fluctuation_field = discretization.get_unknown_size_field(name='displacement_fluctuation_field')
displacement_increment_field = discretization.get_unknown_size_field(name='displacement_increment_field')

strain_fluc_field = discretization.get_displacement_gradient_sized_field(name='strain_fluctuation_field')
total_strain_field = discretization.get_displacement_gradient_sized_field(name='strain_field')
rhs_field = discretization.get_unknown_size_field(name='rhs_field')

# evaluate material law
# stress, material_data_field_C_0_np = constitutive_temp(total_strain_field)
#
# material_data_field_C_0.s = np.broadcast_to(material_data_field_C_0_np[:, :, :, :, np.newaxis, ...],
#                                             (3, 3, 3, 3, discretization.nb_quad_points_per_pixel, *number_of_pixels))
# set macroscopic loading increment
ninc = 20
macro_gradient_inc = np.zeros(shape=(3, 3))
macro_gradient_inc[0, 1] += 0.5 / float(ninc)
macro_gradient_inc[1, 0] += 0.5 / float(ninc)
dt = 1. / float(ninc)

# set macroscopic gradient
macro_gradient_inc_field = discretization.get_macro_gradient_field(macro_gradient_ij=macro_gradient_inc,
                                                                   macro_gradient_field_ijqxyz=macro_gradient_inc_field)
# incremental loading
for inc in range(ninc):
    print(f'Increment {inc}')
    print(f'==========================================================================')
    # strain-hardening exponent
    total_strain_field.s[...] += macro_gradient_inc_field.s

    # Solve mechanical equilibrium constrain
    rhs_field = discretization.get_rhs_explicit_stress(stress_function=constitutive,  # constitutive_pixel
                                                       gradient_field_ijqxyz=total_strain_field,
                                                       rhs_inxyz=rhs_field)
    # evaluate material law
    stress, K4_ijklqyz = constitutive(total_strain_field)  #

    En = np.linalg.norm(total_strain_field.s.mean(axis=2))
    # incremental deformation  newton loop
    iiter = 0

    # iterate as long as the iterative update does not vanish
    while True:
        # Set up right hand side
        # mat_model_pars = {'mat_model': 'power_law_elasticity'}
        K_fun = lambda x: discretization.apply_system_matrix(
            material_data_field=K4_ijklqyz,  # constitutive_pixel
            displacement_field=x,
            formulation='small_strain')

        # K_fun = lambda x: discretization.apply_system_matrix_explicit_stress(
        #     stress_function=constitutive,  # constitutive_pixel
        #     displacement_field=x,
        #     formulation='small_strain',
        #     **mat_model_pars)

        displacement_increment_field.s.fill(0)
        displacement_increment_field.s, norms = solvers.PCG(Afun=K_fun,
                                                            B=rhs_field.s,
                                                            x0=displacement_increment_field.s,
                                                            P=M_fun, steps=int(1000),
                                                            toler=1e-14)
        nb_it_comb = len(norms['residual_rz'])
        norm_rz = norms['residual_rz'][-1]
        norm_rr = norms['residual_rr'][-1]
        print(f'nb iteration CG = {nb_it_comb}')
        # compute strain from the displacement increment
        strain_fluc_field.s = discretization.apply_gradient_operator_symmetrized(u_inxyz=displacement_increment_field,
                                                                                 grad_u_ijqxyz=strain_fluc_field)

        total_strain_field.s += strain_fluc_field.s
        displacement_fluctuation_field.s += displacement_increment_field.s
        # evaluate material law
        stress, K4_ijklqyz = constitutive(total_strain_field)  #
        # Recompute right hand side
        rhs_field = discretization.get_rhs_explicit_stress(stress_function=constitutive,  # constitutive_pixel,
                                                           gradient_field_ijqxyz=total_strain_field,
                                                           rhs_inxyz=rhs_field)

        # rhs *= -1

        if np.linalg.norm(displacement_increment_field.s) / En < 1.e-6 and iiter > 0: break
        # print('=====================')
        print('Rhs {0:10.2e}'.format(np.linalg.norm(rhs_field.s)))
        # print('Norm of disp displacement_increment_field {0:10.2e}'.format(
        #     np.linalg.norm(displacement_increment_field.s.mean(axis=1))))
        # print('Norm of disp displacement_increment_field/ EN {0:10.2e}'.format(
        #     np.linalg.norm(displacement_increment_field.s) / En))

        # update Newton iteration counter
        iiter += 1

        if iiter == 100:
            break

    plot_sol_field = True
    if plot_sol_field:
        fig = plt.figure(figsize=(9, 3.0))
        gs = fig.add_gridspec(2, 2, hspace=0.5, wspace=0.5, width_ratios=[1, 1],
                              height_ratios=[1, 1])
        ax_strain = fig.add_subplot(gs[1, 0])
        pcm = ax_strain.pcolormesh(total_strain_field.s.mean(axis=2)[0, 1, ..., 0],
                                   cmap=mpl.cm.cividis,  # vmin=1, vmax=3,
                                   rasterized=True)
        plt.colorbar(pcm, ax=ax_strain)
        plt.title('total_strain_field   ')
        ax_strain = fig.add_subplot(gs[0, 0])
        pcm = ax_strain.pcolormesh(strain_fluc_field.s.mean(axis=2)[0, 1, ..., 0],
                                   cmap=mpl.cm.cividis,  # vmin=1, vmax=3,
                                   rasterized=True)
        plt.colorbar(pcm, ax=ax_strain)

        plt.title('strain_fluc_field')
        max_stress = stress.mean(axis=2)[0, 1, ..., 0].max()
        min_stress = stress.mean(axis=2)[0, 1, ..., 0].min()

        print('stress min ={}'.format(min_stress))
        print('stress max ={}'.format(max_stress))
        ax_stress = fig.add_subplot(gs[1, 1])
        pcm = ax_stress.pcolormesh(stress.mean(axis=2)[0, 1, ..., 0],
                                   cmap=mpl.cm.cividis, vmin=min_stress, vmax=max_stress,
                                   rasterized=True)

        plt.colorbar(pcm, ax=ax_stress)
        plt.title('stress   ')

        # plot constitutive tangent
        ax_tangent = fig.add_subplot(gs[0, 1])
        # pcm = ax_tangent.pcolormesh(material_data_field_C_0.s.mean(axis=4)[0, 0, 0, 0, ..., 0],
        #                             cmap=mpl.cm.cividis,  # vmin=0, vmax=1500,
        #                             rasterized=True)
        # x_deformed[:, :], y_deformed[:, :],
        plt.colorbar(pcm, ax=ax_tangent)
        plt.title('material_data_field_C_0')

    plt.show()

# fig = plt.figure(figsize=(9, 3.0))
# gs = fig.add_gridspec(2, 2, hspace=0.5, wspace=0.5, width_ratios=[1, 1],
#                       height_ratios=[1, 1])
# ax_strain = fig.add_subplot(gs[1, 0])
# pcm = ax_strain.pcolormesh(total_strain_field.s.mean(axis=2)[0, 0, ..., 0],
#                            cmap=mpl.cm.cividis,  # vmin=1, vmax=3,
#                            rasterized=True)
# plt.colorbar(pcm, ax=ax_strain)
# plt.title('init total_strain_field   ')
# ax_strain = fig.add_subplot(gs[0, 0])
# pcm = ax_strain.pcolormesh(strain_fluc_field.s.mean(axis=2)[0, 0, ..., 0],
#                            cmap=mpl.cm.cividis,  # vmin=1, vmax=3,
#                            rasterized=True)
# plt.colorbar(pcm, ax=ax_strain)
#
# plt.title('strain_fluc_field')
#
# ax_strain = fig.add_subplot(gs[1, 1])
# pcm = ax_strain.pcolormesh(total_strain_field.s.mean(axis=2)[0, 0, ..., 0],
#                            cmap=mpl.cm.cividis,  # vmin=1, vmax=3,
#                            rasterized=True)
# plt.colorbar(pcm, ax=ax_strain)
# plt.title('total_strain_field late ')
#
# # plot constitutive tangent
# ax_tangent = fig.add_subplot(gs[0, 1])
# pcm = ax_tangent.pcolormesh(material_data_field_C_0.s.mean(axis=4)[0, 0, 0, 0, ..., 0],
#                             cmap=mpl.cm.cividis,  # vmin=0, vmax=1500,
#                             rasterized=True)
# # x_deformed[:, :], y_deformed[:, :],
# plt.colorbar(pcm, ax=ax_tangent)
# plt.title('material_data_field_C_0')
# plt.show()
print(
    '   nb_ steps CG of =' f'{nb_it_comb}, residual_rz = {norm_rz}, residual_rr = {norm_rr}')
# print(norms)
# ----------------------------------------------------------------------
# compute homogenized stress field corresponding to displacement
homogenized_stress = discretization.get_homogenized_stress(
    material_data_field_ijklqxyz=material_data_field_C_0,
    displacement_field_inxyz=displacement_fluctuation_field,
    macro_gradient_field_ijqxyz=macro_gradient_field,
    formulation='small_strain')

print('homogenized stress = \n {}'.format(homogenized_stress))
print('homogenized stress in Voigt notation = \n {}'.format(domain.compute_Voigt_notation_2order(homogenized_stress)))

end_time = time.time()
elapsed_time = end_time - start_time
print("Elapsed time: ", elapsed_time)

#
# fig = plt.figure(figsize=(6, 3.0))
#
# gs = fig.add_gridspec(1, 1, hspace=0., wspace=0., width_ratios=[1],
#                       height_ratios=[1])
#
# ax_deformed = fig.add_subplot(gs[0, 0])
# pcm = ax_deformed.pcolormesh(X, Y, stress[0, 0, 0, ..., 0], cmap=mpl.cm.cividis,  # vmin=-5, vmax=5,
#                              rasterized=True)
#
# plt.show()
# # plotting :

# # linear part of displacement
# disp_linear_x = X * macro_gradient[0, 0] + Y * macro_gradient[0, 1]  # (X - domain_size[0] / 2)
# disp_linear_y = X * macro_gradient[1, 0] + Y * macro_gradient[1, 1]
# # displacement in voids should be zero
# # displacement_fluctuation_field.s[:, 0, :, :5] = 0.0
#
# x_deformed = X + disp_linear_x + displacement_fluctuation_field.s[0, 0, :, :, 0]
# y_deformed = Y + disp_linear_y + displacement_fluctuation_field.s[1, 0, :, :, 0]


# plt.show()
# material_model_info = {
#     "mat_model": "power_law_elasticity",
# }
