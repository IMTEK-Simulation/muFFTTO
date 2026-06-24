import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import numpy as np
import time
from muGrid import Solvers

from muFFTTO import domain
from muFFTTO import solvers
from muFFTTO import microstructure_library

problem_type = 'conductivity'
discretization_type = 'finite_element'
element_type = 'trilinear_hexahedron'

domain_size = [1, 1, 1]
number_of_pixels = 3 * (32,)

geometry_ID = 'cos_wave' #'square_inclusion'
# set up the system
my_cell = domain.PeriodicUnitCell(domain_size=domain_size,
                                  problem_type=problem_type)

discretization = domain.Discretization(cell=my_cell,
                                       nb_of_pixels_global=number_of_pixels,
                                       discretization_type=discretization_type,
                                       element_type=element_type)


# create material data field
mat_contrast = 1
mat_contrast_2 = 1e2
conductivity_C_1 = 1 * np.array([[1, 0, 0],
                                 [0, 1, 0],
                                 [0, 0, 1]])

material_data_field_C_0 = discretization.get_material_data_size_field_mugrid(name=' conductivity_tensor')
material_data_field_C_0.s[...] = conductivity_C_1[:, :, np.newaxis, np.newaxis, np.newaxis, np.newaxis]

# material distribution
phase_field = discretization.get_scalar_field(name='phase_field')

phase_field.s[0, 0] = microstructure_library.get_geometry(nb_voxels=discretization.nb_of_pixels,
                                                  microstructure_name=geometry_ID,
                                                  coordinates=discretization.fft.coords)

matrix_mask = phase_field.s[0, 0] > 0
inc_mask = phase_field.s[0, 0] == 0

# apply material distribution
# apply material distribution
if geometry_ID == 'square_inclusion':
    material_data_field_C_0.s[..., matrix_mask] = mat_contrast_2 * material_data_field_C_0.s[..., matrix_mask]
    material_data_field_C_0.s[..., inc_mask] = mat_contrast * material_data_field_C_0.s[..., inc_mask]
else:
    material_data_field_C_0.s[... ] = phase_field.s[0, 0] * material_data_field_C_0.s[...]

# linear system
def K_fun(x, Ax):
    """
    Function to compute the product of the Hessian matrix with a vector.
    The Hessian is represented by the convolution operator.
    """

    discretization.apply_system_matrix_mugrid(material_data_field=material_data_field_C_0,
                                              input_field_inxyz=x,
                                              output_field_inxyz=Ax)
    discretization.fft.communicate_ghosts(Ax)



preconditioner = discretization.get_preconditioner_Green_mugrid(reference_material_data_ijkl=conductivity_C_1)

# preconditioner
def M_fun(x, Px):
    """
    Function to compute the product of the Preconditioner matrix with a vector.
    The Preconditioner is represented by the convolution operator.
    """
    discretization.fft.communicate_ghosts(x)
    discretization.apply_preconditioner_mugrid(preconditioner_Fourier_fnfnqks=preconditioner,
                                               input_nodal_field_fnxyz=x,
                                               output_nodal_field_fnxyz=Px)

dim = discretization.domain_dimension
homogenized_A_ij = np.zeros(np.array(2 * [dim, ]))
# compute whole homogenized elastic tangent

init_x_0 = discretization.get_unknown_size_field(name='init_solution')
solution_field = discretization.get_unknown_size_field(name='solution')
macro_gradient_field = discretization.get_gradient_size_field(name='macro_gradient_field')
rhs_field  = discretization.get_unknown_size_field(name='rhs_field')
start_time = time.time()
for i in range(dim):
    # set macroscopic gradient
    macro_gradient = np.zeros([dim])
    macro_gradient[i] = 1
    # Set up right hand side
    macro_gradient_field.s.fill(0)
    discretization.get_macro_gradient_field_mugrid(macro_gradient_ij=macro_gradient,
                                                   macro_gradient_field_ijqxyz=macro_gradient_field)
    discretization.fft.communicate_ghosts(field=macro_gradient_field)

    # Solve equilibrium
    rhs_field.s.fill(0)
    discretization.get_rhs_mugrid(material_data_field_ijklqxyz=material_data_field_C_0,
                                  macro_gradient_field_ijqxyz=macro_gradient_field,
                                  rhs_inxyz=rhs_field)
    # solver
    def callback(it, x, r, p, z, stop_crit_norm):
        norm_of_rr = discretization.communicator.sum(np.dot(r.ravel(), r.ravel()))
        norm_of_rz = discretization.communicator.sum(np.dot(r.ravel(), z.ravel()))

        if discretization.communicator.rank == 0:
            print(f"{it:5} norm of residual = {norm_of_rr:.5}")


    solution_field.sg.fill(0)
    solvers.conjugate_gradients_mugrid(
        comm=discretization.communicator,
        fc=discretization.field_collection,
        hessp=K_fun,  # linear operator
        b=rhs_field,  # right-hand side
        x=solution_field,
        P=M_fun,
        tol=1e-5,
        maxiter=2000,
        callback=callback, rtol=True
    )

    discretization.fft.communicate_ghosts(field=solution_field)

    # ----------------------------------------------------------------------
    # compute homogenized flux
    homogenized_A_ij[i, :] = discretization.get_homogenized_stress_mugrid(
        material_data_field_ijklqxyz=material_data_field_C_0,
        displacement_field_inxyz=solution_field,
        macro_gradient_field_ijqxyz=macro_gradient_field)

    if discretization.communicator.size == 1:
        # Plot the first component of the solution field
        import matplotlib.pyplot as plt
        plt.figure()
        plt.pcolormesh(discretization.fft.coords[0,...,number_of_pixels[2]//2],
                       discretization.fft.coords[1,...,number_of_pixels[2]//2],
                       solution_field.s[0,0, ...,number_of_pixels[2]//2,:])

        plt.title(f'Solution field - macro gradient {macro_gradient} ')
        plt.xlabel('x  / L')
        plt.ylabel('y  / L')
        plt.colorbar(label='Temperature / Potential')
        plt.show()
# ----------------------------------------------------------------------
if discretization.communicator.rank == 0:
    print(
        "homogenized conductivity tangent =\n" +
        np.array2string(homogenized_A_ij, formatter={'float_kind': lambda x: f"{x:0.8f}"})
    )
end_time = time.time()
elapsed_time = end_time - start_time

print("Elapsed time: ", elapsed_time)
print("Elapsed time: ", elapsed_time / 60)



# solving using block CG

list_of_solution_field = [discretization.get_unknown_size_field(name=f'solution-{j}') for j in range(dim)]
list_of_macro_gradient_field = [discretization.get_gradient_size_field(name=f'macro_gradient_field-{j}') for j
                                in
                                range(dim)]
list_of_rhs_field = [discretization.get_unknown_size_field(name=f'rhs_field-{j}') for j in range(dim)]

homogenized_A_ij = np.zeros(np.array(2 * [dim, ]))

start_time = time.time()
for i in range(dim):
    # set macroscopic gradient
    macro_gradient = np.zeros([dim])
    macro_gradient[i] = 1
    list_of_macro_gradient_field[i].sg.fill(0)
    discretization.get_macro_gradient_field_mugrid(macro_gradient_ij=macro_gradient,
                                                   macro_gradient_field_ijqxyz=list_of_macro_gradient_field[i])
    discretization.fft.communicate_ghosts(field=list_of_macro_gradient_field[i])
    # Solve equilibrium
    list_of_rhs_field[i].sg.fill(0)
    discretization.get_rhs_mugrid(material_data_field_ijklqxyz=material_data_field_C_0,
                                  macro_gradient_field_ijqxyz=list_of_macro_gradient_field[i],
                                  rhs_inxyz=list_of_rhs_field[i])

_, norms = solvers.dr_pbcg_mugrid(comm=discretization.communicator,
                                  fc=discretization.field_collection,
                                  hessp=K_fun,
                                  b_list=list_of_rhs_field,
                                  x_list=list_of_solution_field,
                                  P=M_fun,
                                  tol=1e-5,
                                  rtol=True)
if discretization.communicator.rank == 0:
    norms = np.asarray(norms['residual_frobenius'])
    print(f"{len(norms):1} norm of residual = {', '.join(f'{v}' for v in norms)}")

for i in range(dim):
    homogenized_A_ij[i, :] = discretization.get_homogenized_stress_mugrid(
        material_data_field_ijklqxyz=material_data_field_C_0,
        displacement_field_inxyz=list_of_solution_field[i],
        macro_gradient_field_ijqxyz=list_of_macro_gradient_field[i])
    if discretization.communicator.size == 1:
        # Plot the first component of the solution field
        import matplotlib.pyplot as plt
        plt.figure()
        plt.pcolormesh(discretization.fft.coords[0,...,number_of_pixels[2]//2],
                       discretization.fft.coords[1,...,number_of_pixels[2]//2],
                       list_of_solution_field[i].s[0, 0,  :, number_of_pixels[2] // 2, :])
        macro_gradient=list_of_macro_gradient_field[i].s[0,:,0,0,0,0]
        plt.title(f'Solution field - macro gradient { macro_gradient} ')
        plt.xlabel('x  / L')
        plt.ylabel('y  / L')
        plt.colorbar(label='Temperature / Potential')
        plt.show()
if discretization.communicator.rank == 0:
    print(
        "homogenized conductivity tangent =\n" +
        np.array2string(homogenized_A_ij, formatter={'float_kind': lambda x: f"{x:0.8f}"})
    )
end_time = time.time()
elapsed_time = end_time - start_time
if discretization.communicator.rank == 0:
    print("Elapsed time: ", elapsed_time, 'seconds')
    print("Elapsed time: ", elapsed_time / 60, 'minutes')
    J_eff = mat_contrast_2 * np.sqrt((mat_contrast_2 + 3 * mat_contrast) / (3 * mat_contrast_2 + mat_contrast))
    print(f'Analytical solution conductivity - A^eff_11  : {J_eff:0.8f}')
    print(f'Numerical solution  conductivity - A^eff_11  : {homogenized_A_ij[0, 0]:0.8f}')
