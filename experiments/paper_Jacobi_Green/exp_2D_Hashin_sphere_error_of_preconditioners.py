import numpy as np
import matplotlib.pyplot as plt

import scipy as sc
import time
import sys
import  os
import argparse

sys.path.append('..')  # Add parent directory to path
script_name = os.path.splitext(os.path.basename(__file__))[0]

file_folder_path = os.path.dirname(os.path.realpath(__file__))  # script directory
data_folder_path = file_folder_path + '/exp_data/' + script_name + '/'
figure_folder_path = file_folder_path + '/figures/' + script_name + '/'

if not os.path.exists(file_folder_path):
    os.makedirs(file_folder_path)
if not os.path.exists(data_folder_path):
    os.makedirs(data_folder_path)
if not os.path.exists(figure_folder_path):
    os.makedirs(figure_folder_path)


from mpi4py import MPI
from NuMPI.IO import save_npy, load_npy

from muFFTTO import domain
from muFFTTO import solvers
from muFFTTO import microstructure_library


parser = argparse.ArgumentParser(
    prog="exp_paper_JG_nonlinear_elasticity_JZ.py",
    description="Solve non-linear elasticity example "
                "from J.Zeman et al., Int. J. Numer. Meth. Engng 111, 903–926 (2017)."
)
parser.add_argument("-n", "--nb_pixel", default="32")
parser.add_argument(
    "-p", "--preconditioner_type",
    type=str,
    choices=["Green", "Jacobi", "Green_Jacobi"],  # example options
    default="Green_Jacobi",
    help="Type of preconditioner to use"
)
parser.add_argument("-cg_tol", "--cg_tol_exponent", default="8")

args = parser.parse_args()
nnn = int(args.nb_pixel)
preconditioner_type = args.preconditioner_type
cg_tol_exponent = int(args.cg_tol_exponent)

cg_setup = {'cg_tol': 10 ** (-cg_tol_exponent)}

problem_type = 'elasticity'
discretization_type = 'finite_element'
element_type = 'linear_triangles'
formulation = 'small_strain'

domain_size = [1, 1]
dim = len(domain_size)
number_of_pixels = (nnn,nnn)

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


mu_3= 0.3
lambda_3 = kappa_3 -   2 * mu_3/ dim
C_matrix=domain.get_elastic_tensor_from_lame(dim=2,  lam=lambda_3, mu=mu_3)
print('C_matrix = \n {}'.format(domain.compute_Voigt_notation_4order(C_matrix)))

# reference material data
C_0_ref = np.sqrt(np.einsum('ijkl,klmn->ijmn', C_core, C_shell))
# get the geometry
coordinates = discretization.fft.coords
center=np.array([0.5,0.5])#-1/np.array( number_of_pixels)
r = np.sqrt(np.sum((coordinates - center[:, None, None]) ** 2, axis=0))

mask_core = r <= r_1
mask_shell =  (r > r_1) & (r < r_2)
# populate the global data field
material_data_field_C = discretization.get_material_data_size_field_mugrid(name='elastic_tensor')
# populate the field with C_1 material
material_data_field_C.s = np.einsum('ijkl,qxy->ijklqxy', C_matrix,
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
if preconditioner_type == 'Green':
    M_fun = M_fun_Green
elif preconditioner_type == 'Jacobi':
    K_diag_alg = discretization.get_preconditioner_Jacobi_mugrid(
        material_data_field_ijklqxyz=material_data_field_C)

    def M_fun_Jacobi(x, Px):
        Px.s = K_diag_alg.s * K_diag_alg.s * x.s
        discretization.fft.communicate_ghosts(Px)

    M_fun = M_fun_Jacobi

elif preconditioner_type == 'Green_Jacobi':
    K_diag_alg = discretization.get_preconditioner_Jacobi_mugrid(
        material_data_field_ijklqxyz=material_data_field_C)

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

# Set up right hand side
macro_gradient_field = discretization.get_gradient_size_field(name='macro_gradient_field')
discretization.get_macro_gradient_field_mugrid(macro_gradient_ij=macro_gradient,
                                               macro_gradient_field_ijqxyz=macro_gradient_field)

# Solve mechanical equilibrium constrain
rhs_field = discretization.get_unknown_size_field(name='rhs_field')
discretization.get_rhs_mugrid(material_data_field_ijklqxyz=material_data_field_C,
                              macro_gradient_field_ijqxyz=macro_gradient_field,
                              rhs_inxyz=rhs_field)

_info = {}
_info['norm_rr']=[]
def callback(it, x, r, p, z, stop_crit_norm):
    """
    Callback function to print the current solution, residual, and search direction.
    """
    norm_of_rr = discretization.fft.communicator.sum(np.dot(r.ravel(), r.ravel()))
    _info['norm_rr'].append(norm_of_rr)

    # if discretization.fft.communicator.rank == 0:
    #     print(f"{it:5} norm of residual = {norm_of_rr:.5}")


solution_field = discretization.get_unknown_size_field(name='solution')

solvers.conjugate_gradients_mugrid(
    comm=discretization.fft.communicator,
    fc=discretization.field_collection,
    hessp=K_fun,  # linear operator
    b=rhs_field,
    x=solution_field,
    P=M_fun,
    tol= cg_setup['cg_tol'],
    maxiter=2000,
    callback=callback,
)

print(len(  _info['norm_rr']))

total_strain_field = discretization.get_gradient_size_field(name='total_strain_field')

# compute strain from the displacement increment
discretization.apply_gradient_operator_symmetrized_mugrid(
    u_inxyz=solution_field,
    grad_u_ijqxyz=total_strain_field)

total_strain_field.s += macro_gradient_field.s

file_data_name_it = f'{preconditioner_type}' +f'_n_{number_of_pixels[0]}' + f'_cgtol_{cg_tol_exponent}'
# save_npy(data_folder_path +  file_data_name_it + f'.npy',
#          result_norms.reshape([*discretization.nb_of_pixels]),
#          tuple(discretization.subdomain_locations_no_buffers),
#          tuple(discretization.nb_of_pixels_global), MPI.COMM_WORLD)
data_to_save= np.array( total_strain_field.s[...])
save_npy(fn=data_folder_path + file_data_name_it  + f'.npy',
         data=data_to_save,
         subdomain_locations=tuple(discretization.subdomain_locations_no_buffers),
         nb_grid_pts=tuple(discretization.nb_of_pixels_global),
         components_are_leading=True,
         comm=MPI.COMM_WORLD)
# save_npy(fn=data_folder_path + file_data_name_it +'q1'+ f'.npy',
#          data=total_strain_field.s[0, 0 ,1],
#          subdomain_locations=tuple(discretization.subdomain_locations_no_buffers),
#          nb_grid_pts=tuple(discretization.nb_of_pixels_global),
#          components_are_leading=True,
#          comm=MPI.COMM_WORLD)
_info['nb_of_pixels'] = discretization.nb_of_pixels_global




# _info['homogenized_C_ijkl'] = domain.compute_Voigt_notation_4order(homogenized_C_ijkl)
# _info['target_C_ijkl'] = domain.compute_Voigt_notation_4order(elastic_C_target)

# np.save(folder_name + file_data_name+f'xopt_log.npz', xopt_FE_MPI)
if MPI.COMM_WORLD.rank == 0:
    np.savez(data_folder_path + file_data_name_it + f'_log.npz',
             **_info)  # + f'_its_{start}_{start + iterat}'

quit()

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
            comm=discretization.fft.communicator,
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


