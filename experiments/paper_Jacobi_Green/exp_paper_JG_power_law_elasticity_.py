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

domain_size = [1, 1, 1]
number_of_pixels = (32, 32, 1)

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
K_0, G_0 = domain.get_bulk_and_shear_modulus(E=1, poison=0.0)

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
I = np.eye(discretization.domain_dimension)
# identity tensors                                            [grid of tensors]
I4 = np.einsum('il,jk', I, I)
I4rt = np.einsum('ik,jl', I, I)
II = np.einsum('ij...  ,kl...  ->ijkl...', I, I)
I4s = (I4 + I4rt) / 2.
I4d = (I4s - II / 3.)


###
def nonlin_elastic_tangent(strain, K):
    # K = 2.  # bulk modulus
    sig0 = 0.5  # reference stress
    eps0 = 0.1  # reference strain
    n = 10.  # hardening exponent

    strain_trace_qxyz = np.einsum('ii...', strain.s) / 3  # todo{2 or 3 in 2D }

    strain_hydro_ijqxyz = discretization.get_displacement_gradient_sized_field(name='strain_hydro')
    for d in arange(discretization.domain_dimension):
        strain_hydro_ijqxyz.s[d, d, ...] = strain_trace_qxyz
    # strain_hydro_ijqxyz.s[1, 1, ...] = strain_trace_qxyz

    strain_dev_ijqxyz = strain.s - strain_hydro_ijqxyz.s
    # equivalent strain
    strain_dev_ddot = np.einsum('ijqxyz,jiqxyz->qxyz', strain_dev_ijqxyz, strain_dev_ijqxyz)
    strain_equivalent_qxyz = np.sqrt(2. / 3. * strain_dev_ddot)

    sig = (3. * K * strain_hydro_ijqxyz.s + 2. / 3. * sig0 / (eps0 ** n) *
           (strain_equivalent_qxyz ** (n - 1.)) * strain_dev_ijqxyz)
    #
    sig = 3. * K * strain_hydro_ijqxyz.s * (strain_equivalent_qxyz == 0.).astype(float) + sig * (
            strain_equivalent_qxyz != 0.).astype(float)

    # K4_d = discretization.get_material_data_size_field(name='alg_tangent')
    strain_dev_dyad = np.einsum('ijqxyz,klqxyz->ijklqxyz', strain_dev_ijqxyz, strain_dev_ijqxyz)

    K4_d = np.broadcast_to(I4d[..., np.newaxis, np.newaxis, np.newaxis, np.newaxis],
                           (3, 3, 3, 3, *strain_hydro_ijqxyz.s.shape[-4:]))

    K4_d = 2. / 3. * sig0 / (eps0 ** n) * (strain_dev_dyad * 2. / 3. * (n - 1.) * strain_equivalent_qxyz ** (
            n - 3.) + strain_equivalent_qxyz ** (n - 1.) * K4_d)

    K4 = K * II[..., np.newaxis, np.newaxis, np.newaxis, np.newaxis] + K4_d * (strain_equivalent_qxyz != 0.).astype(
        float)
    return sig, K4


def constitutive(strain):
    phase_field = np.zeros([*number_of_pixels])
    phase_field[12:26, 12:26] = 1.

    sig_P1, K4_P1 = nonlin_elastic_tangent(strain, K=1)
    sig_P2, K4_P2 = nonlin_elastic_tangent(strain, K=100)

    sig = phase_field * sig_P1 + (1. - phase_field) * sig_P2
    K4 = phase_field * K4_P1 + (1. - phase_field) * K4_P2

    return sig, K4


# assembly preconditioner
preconditioner = discretization.get_preconditioner_NEW(reference_material_data_ijkl=I4s)

M_fun = lambda x: discretization.apply_preconditioner_NEW(preconditioner_Fourier_fnfnqks=preconditioner,
                                                          nodal_field_fnxyz=x)

macro_gradient_field = discretization.get_gradient_size_field(name='macro_gradient_field')
displacement_fluctuation_field = discretization.get_unknown_size_field(name='displacement_fluctuation_field')
displacement_increment_field = discretization.get_unknown_size_field(name='displacement_increment_field')

strain_fluc_field = discretization.get_displacement_gradient_sized_field(name='strain_fluctuation_field')
total_strain_field = discretization.get_displacement_gradient_sized_field(name='strain_field')
rhs_field = discretization.get_unknown_size_field(name='rhs_field')

# set macroscopic gradient
macro_gradient = np.array([[0.2, 0.0, 0.00],
                           [0.0, 0.00, 0.00],
                           [0.0, 0.00, 0.00]])

macro_gradient_field = discretization.get_macro_gradient_field(macro_gradient_ij=macro_gradient,
                                                               macro_gradient_field_ijqxyz=macro_gradient_field)

# strain-hardening exponent
total_strain_field.s = strain_fluc_field.s + macro_gradient_field.s
# evaluate material law
stress, material_data_field_C_0.s = constitutive(total_strain_field)

# Solve mechanical equilibrium constrain
rhs = discretization.get_rhs(material_data_field_ijklqxyz=material_data_field_C_0,
                             macro_gradient_field_ijqxyz=total_strain_field,
                             rhs_inxyz=rhs_field)



En = np.linalg.norm(total_strain_field.s)
# incremental deformation  newton loop
iiter = 0

# iterate as long as the iterative update does not vanish
while True:
    # Set up right hand side

    # material_model_info = {
    #     "mat_model": "power_law_elasticity",
    # }
    K_fun = lambda x: discretization.apply_system_matrix(material_data_field_C_0, x,
                                                         formulation='small_strain')
    displacement_increment_field.s.fill(0)
    displacement_increment_field.s, norms = solvers.PCG(K_fun, rhs.s, x0=None, P=M_fun, steps=int(1000), toler=1e-8)
    nb_it_comb = len(norms['residual_rz'])
    norm_rz = norms['residual_rz'][-1]
    norm_rr = norms['residual_rr'][-1]
    print(f'nb iteration CG = {nb_it_comb}')

    displacement_fluctuation_field.s += displacement_increment_field.s

    strain_fluc_field.s = discretization.apply_gradient_operator_symmetrized(u_inxyz=displacement_increment_field,
                                                                             grad_u_ijqxyz=strain_fluc_field)
    total_strain_field.s += strain_fluc_field.s

    # evaluate material law
    stress, material_data_field_C_0.s = constitutive(total_strain_field)

    # Recompute right hand side
    rhs = discretization.get_rhs(material_data_field_ijklqxyz=material_data_field_C_0,
                                 macro_gradient_field_ijqxyz=total_strain_field,
                                 rhs_inxyz=rhs_field)

    plot_stress = True
    if plot_stress:
        x = np.linspace(start=0, stop=domain_size[0], num=number_of_pixels[0])
        y = np.linspace(start=0, stop=domain_size[1], num=number_of_pixels[1])
        X, Y = np.meshgrid(x, y, indexing='ij')

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

        # linear part of displacement
        disp_linear_x = X * macro_gradient[0, 0] + Y * macro_gradient[0, 1]  # (X - domain_size[0] / 2)
        disp_linear_y = X * macro_gradient[1, 0] + Y * macro_gradient[1, 1]
        # displacement in voids should be zero
        # displacement_fluctuation_field.s[:, 0, :, :5] = 0.0

        x_deformed = X + disp_linear_x + displacement_fluctuation_field.s[0, 0, :, :, 0]
        y_deformed = Y + disp_linear_y + displacement_fluctuation_field.s[1, 0, :, :, 0]

        fig = plt.figure(figsize=(6, 3.0))
        gs = fig.add_gridspec(1, 1, hspace=0., wspace=0., width_ratios=[1],
                              height_ratios=[1])

        ax_deformed = fig.add_subplot(gs[0, 0])
        pcm = ax_deformed.pcolormesh(x_deformed[:, :], y_deformed[:, :], stress[0, 1, 0, ..., 0],
                                     cmap=mpl.cm.cividis,  # vmin=0, vmax=2,
                                     rasterized=True)

        plt.show()

    if np.linalg.norm(displacement_increment_field.s) / En < 1.e-6 and iiter > 0: break
    print('=====================')
    print('Rhs {0:10.2e}'.format(np.linalg.norm(rhs.s)))
    print('Norm of disp displacement_increment_field {0:10.2e}'.format(np.linalg.norm(displacement_increment_field.s)))
    print('Norm of disp displacement_increment_field/ EN {0:10.2e}'.format(
        np.linalg.norm(displacement_increment_field.s) / En))

    # update Newton iteration counter
    iiter += 1
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
