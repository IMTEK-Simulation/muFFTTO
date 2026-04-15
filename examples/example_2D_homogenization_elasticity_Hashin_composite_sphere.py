import numpy as np
import matplotlib.pyplot as plt

import scipy as sc
import time
import sys

sys.path.append('..')  # Add parent directory to path

from mpi4py import MPI
from NuMPI.IO import save_npy, load_npy

from muFFTTO import domain
from muFFTTO import solvers
from muFFTTO import microstructure_library

problem_type = 'elasticity'
discretization_type = 'finite_element'
element_type = 'linear_triangles'
formulation = 'small_strain'

domain_size = [1, 1]
dim = len(domain_size)
number_of_pixels = (64,64)

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
macro_gradient = np.array([[1.0, 0], [0, 1.0]])

# inclusion radii
r_1 = 0.2
r_2 = 0.4
# create material data
# core
lambda_1 = 0.001  # first Lamé
mu_1 = 0.005  # second Lamé
kappa_1 = lambda_1 + 2 * mu_1 / dim  # bulk
C_core=domain.get_elastic_tensor_from_lame(dim=2,  lam=lambda_1, mu=mu_1)
# shell
lambda_2 = 1.
mu_2 = 0.5
kappa_2 = lambda_2 + 2 * mu_2 / dim
C_shell=domain.get_elastic_tensor_from_lame(dim=2,  lam=lambda_2, mu=mu_2)

# matrix = should be equal to homogenized data
phi = (r_1 / r_2) ** dim
alpha = dim * (kappa_2 - kappa_1) / ((dim - 1) * 2.0 * mu_2 + dim * kappa_1) #
beta = 1 + 2 * (dim - 1) * mu_2 / (dim * kappa_2)
kappa_3 = kappa_2 * (1.0 - beta * (alpha * phi) / (1.0 + alpha * phi))

# phi = (r_1 / r_2) ** dim
# alpha = dim * (kappa_2 - kappa_1) / ((dim - 1) * 2.0 * mu_2 + dim * kappa_2)
# kappa_32 = kappa_2 * (1.0 - (dim * alpha * phi) / (1.0 + alpha * phi))
# kappa_3=0.9
mu_3= 0.3
lambda_3 = kappa_3 -   2 * mu_3/ dim
C_matrix=domain.get_elastic_tensor_from_lame(dim=2,  lam=lambda_3, mu=mu_3)
print('C_matrix = \n {}'.format(domain.compute_Voigt_notation_4order(C_matrix)))

# kappa_1 = 0.1
# lambda_1 = mu_1= 3*  kappa_1/5#   Lamé
# C_core = domain.get_elastic_tensor_from_lame(dim=2, lam=lambda_1, mu=mu_1)
#
# kappa_2 = 1.5
# lambda_2 = mu_2= 3*  kappa_2/5#   Lamé
# C_shell = domain.get_elastic_tensor_from_lame(dim=2, lam=lambda_2, mu=mu_2)
#
# phi_1=(r_1 / r_2) ** dim
# phi_2=1-phi_1
#
# k_eff=kappa_2+phi_1/((1/(kappa_1-kappa_2)+phi_2/(kappa_2+4*mu_2/3)))
# lambda_3 = mu_3= 3*  k_eff/5#   Lamé
# C_matrix=domain.get_elastic_tensor_from_lame(dim=2,  lam=lambda_3, mu=mu_3)
# print('C_matrix = \n {}'.format(domain.compute_Voigt_notation_4order(C_matrix)))





# reference material data
C_0_ref = np.sqrt(np.einsum('ijkl,klmn->ijmn', C_core, C_shell))
# get the geometry
coordinates = discretization.fft.coords
center=np.array([0.5,0.5])#-1/np.array( number_of_pixels)
r = np.sqrt(np.sum((coordinates - center[:, None, None]) ** 2, axis=0))

#
# def compute_displacement_matti(r, r1, r2,
#                          K1, mu1,
#                          K2, mu2,
#                          K_eff, mu_eff):
#     """
#     Compute radial displacement u(r) for a coated inclusion
#     under macroscopic strain <eps> = Id.
#
#     Parameters
#     ----------
#     r : array_like
#         Radial coordinates.
#     r1, r2 : float
#         Core radius and outer coating radius.
#     K1, mu1 : float
#         Bulk and shear modulus of the core.
#     K2, mu2 : float
#         Bulk and shear modulus of the coating.
#     K_eff, mu_eff : float
#         Bulk and shear modulus of the matrix.
#
#     Returns
#     -------
#     u : ndarray
#         Radial displacement field.
#     """
#
#     r = np.asarray(r)
#
#     # Coefficients from the analytical solution
#     denom = 3 * K2 + 4 * mu2
#     a2 = (3 * K_eff + 4 * mu2) / denom
#     b2 = 3 * r2**dim * (K2 - K_eff) / denom
#     a1 = a2 + b2 / r1**dim
#     a_eff = 1.0  # macroscopic strain = 1
#
#     # Allocate displacement array
#     u = np.zeros_like(r)
#
#     # Region 1: core
#     mask1 = r < r1
#     u[mask1] = a1 * r[mask1]
#
#     # Region 2: coating
#     mask2 = (r >= r1) & (r < r2)
#     u[mask2] = a2 * r[mask2] + b2 / r[mask2]**2
#
#     # Region 3: matrix
#     mask3 = r >= r2
#     u[mask3] = a_eff * r[mask3]
#
#     return u
#
# def radial_to_cartesian_displacement(x, y, u_r):
#     """
#     Convert radial displacement u(r) to Cartesian components u_x, u_y.
#
#     Parameters
#     ----------
#     x, y : array_like
#         Cartesian coordinates.
#     u_r : array_like
#         Radial displacement evaluated at r = sqrt(x^2 + y^2).
#
#     Returns
#     -------
#     u_x, u_y : ndarray
#         Cartesian displacement components.
#     """
#     x = np.asarray(x)
#     y = np.asarray(y)
#     u_r = np.asarray(u_r)
#
#     r = np.sqrt(x**2 + y**2)
#
#     # Avoid division by zero at r = 0
#     u_x = np.zeros_like(r)
#     u_y = np.zeros_like(r)
#
#     mask = r > 0
#     u_x[mask] = u_r[mask] * x[mask] / r[mask]
#     u_y[mask] = u_r[mask] * y[mask] / r[mask]
#
#     return u_x, u_y

# # Compute radial displacement
# u_r = compute_displacement_matti(r, r_1, r_2, kappa_1, mu_1, kappa_2, mu_2, k_eff, mu_3)
#
# # Convert to Cartesian
# u_x, u_y = radial_to_cartesian_displacement(coordinates[0]-center[0, None, None], coordinates[1]-center[1, None, None], u_r)


mask_core = r <= r_1
mask_shell =  (r > r_1) & (r < r_2)
# populate the global data field
material_data_field_C = discretization.get_material_data_size_field_mugrid(name='elastic_tensor')
# populate the field with C_1 material
material_data_field_C.s[...] = np.einsum('ijkl,qxy->ijklqxy', C_matrix,
                                      np.ones(np.array([discretization.nb_quad_points_per_pixel,
                                                        *discretization.nb_of_pixels])))

# # apply material distribution to all quadrature points at masked locations
material_data_field_C.s[..., mask_core] = C_core[..., None, None]
material_data_field_C.s[..., mask_shell] = C_shell[..., None, None]



def K_fun(x, Ax):
    discretization.apply_system_matrix_mugrid(material_data_field=material_data_field_C,
                                              input_field_inxyz=x,
                                              output_field_inxyz=Ax,
                                              formulation='small_strain')
    discretization.fft.communicate_ghosts(Ax)


preconditioner = discretization.get_preconditioner_Green_mugrid(reference_material_data_ijkl=C_0_ref)


def M_fun(x, Px):
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


# Set up right hand side
macro_gradient_field = discretization.get_gradient_size_field(name='macro_gradient_field')
discretization.get_macro_gradient_field_mugrid(macro_gradient_ij=macro_gradient,
                                               macro_gradient_field_ijqxyz=macro_gradient_field)

# Solve mechanical equilibrium constrain
rhs_field = discretization.get_unknown_size_field(name='rhs_field')
discretization.get_rhs_mugrid(material_data_field_ijklqxyz=material_data_field_C,
                              macro_gradient_field_ijqxyz=macro_gradient_field,
                              rhs_inxyz=rhs_field)
def callback(it, x, r, p, z, stop_crit_norm):
    """
    Callback function to print the current solution, residual, and search direction.
    """
    norm_of_rr = discretization.communicator.sum(np.dot(r.ravel(), r.ravel()))
    # if discretization.communicator.rank == 0:
    #     print(f"{it:5} norm of residual = {norm_of_rr:.5}")


solution_field = discretization.get_unknown_size_field(name='solution')

solvers.conjugate_gradients_mugrid(
    comm=discretization.communicator,
    fc=discretization.field_collection,
    hessp=K_fun,  # linear operator
    b=rhs_field,
    x=solution_field,
    P=M_fun,
    tol=1e-8,
    maxiter=2000,
    callback=callback,
)
sol_r = np.sqrt(np.sum(( solution_field.s[:,0] ) ** 2, axis=0))

# print(norms)
total_strain_field = discretization.get_gradient_size_field(name='total_strain_field')

# compute strain from the displacement increment
discretization.apply_gradient_operator_symmetrized_mugrid(
    u_inxyz=solution_field,
    grad_u_ijqxyz=total_strain_field)

total_strain_field.s += macro_gradient_field.s

# Plot the strain field components
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Plot each component of the strain tensor
components = [
    (total_strain_field.s [ 0, 0,...].mean(axis=0), r'$\varepsilon_{xx}$'),
    (total_strain_field.s [ 0, 1,...].mean(axis=0), r'$\varepsilon_{xy}$'),
    (total_strain_field.s [1, 0,...].mean(axis=0), r'$\varepsilon_{yx}$'),
    (total_strain_field.s [ 1, 1,...].mean(axis=0), r'$\varepsilon_{yy}$')
]

for ax, (component, label) in zip(axes.flat, components):
    im = ax.imshow(component, extent=[0, 1, 0, 1], origin='lower', cmap='RdBu_r'
                   ,vmin=0.2, vmax=1.3)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title(label)
    plt.colorbar(im, ax=ax)

    # Draw circles for core and shell boundaries
    circle1 = plt.Circle(center+0.5/np.array( number_of_pixels), r_1, fill=False, edgecolor='black', linestyle='--', linewidth=1.5)
    circle2 = plt.Circle(center+0.5/np.array( number_of_pixels), r_2, fill=False, edgecolor='black', linestyle='-', linewidth=1.5)
    ax.add_patch(circle1)
    ax.add_patch(circle2)

plt.tight_layout()
plt.savefig('hashin_strain_field_2d_FFT_solver.png', dpi=150, bbox_inches='tight')
print("Plot saved as 'hashin_strain_field_2d_FFT_solver.png'")
plt.show()


# ----------------------------------------------------------------------
# compute homogenized stress field corresponding to displacement
homogenized_stress = discretization.get_homogenized_stress_mugrid(
    material_data_field_ijklqxyz=material_data_field_C ,
    displacement_field_inxyz=solution_field,
    macro_gradient_field_ijqxyz=macro_gradient_field,
    formulation='small_strain')

print('homogenized stress = \n {}'.format(homogenized_stress))
print('homogenized stress in Voigt notation = \n {}'.format(domain.compute_Voigt_notation_2order(homogenized_stress)))

end_time = time.time()
elapsed_time = end_time - start_time
print("Elapsed time: ", elapsed_time)

start_time = time.time()
dim = discretization.domain_dimension
homogenized_C_ijkl = np.zeros(np.array(4 * [dim, ]))
# compute whole homogenized elastic tangent
for i in range(dim):
    for j in range(dim):
        # set macroscopic gradient
        macro_gradient = np.zeros([dim, dim])
        macro_gradient[i, j] = 1

        discretization.get_macro_gradient_field_mugrid(macro_gradient_ij=macro_gradient,
                                                       macro_gradient_field_ijqxyz=macro_gradient_field)
        # Set up right hand side
        # Solve mechanical equilibrium constrain
        discretization.get_rhs_mugrid(material_data_field_ijklqxyz=material_data_field_C,
                                      macro_gradient_field_ijqxyz=macro_gradient_field,
                                      rhs_inxyz=rhs_field)
        # rhs_ij = discretization.get_rhs(material_data_field_C_0_rh, macro_gradient_field)

        solvers.conjugate_gradients_mugrid(
            comm=discretization.communicator,
            fc=discretization.field_collection,
            hessp=K_fun,  # linear operator
            b=rhs_field,
            x=solution_field,
            P=M_fun,
            tol=1e-6,
            maxiter=2000,
            callback=callback,
        )
        # ----------------------------------------------------------------------
        # compute homogenized stress field corresponding
        homogenized_C_ijkl[i, j] = discretization.get_homogenized_stress_mugrid(
            material_data_field_ijklqxyz=material_data_field_C,
            displacement_field_inxyz=solution_field,
            macro_gradient_field_ijqxyz=macro_gradient_field,
            formulation='small_strain')

print('homogenized elastic tangent = \n {}'.format(domain.compute_Voigt_notation_4order(homogenized_C_ijkl)))
end_time = time.time()
elapsed_time = end_time - start_time
print("Elapsed time: ", elapsed_time)
