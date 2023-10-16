import numpy as np
import scipy as sc

from muFFTTO import domain
from muFFTTO import solvers


def objective_function_small_strain(discretization, actual_stress, target_stress, phase_field, eta, w):
    # evaluate objective functions
    # f = (flux_h -flux_target)^2 + w*eta* int (  (grad(rho))^2 )dx  +    int ( rho^2(1-rho)^2 ) / eta   dx
    # f =  f_sigma + w*eta* f_rho_grad  + f_dw/eta

    # stress difference potential
    stress_difference = (actual_stress - target_stress[
        (...,) + (np.newaxis,) * (actual_stress.ndim - 2)])

    f_sigma = np.sum(discretization.integrate_over_cell(stress_difference ** 2))

    # double - well potential
    integrant = (phase_field ** 2) * (1 - phase_field) ** 2
    f_dw = (np.sum(integrant) / np.prod(integrant.shape)) * discretization.cell.domain_volume

    phase_field_gradient = discretization.apply_gradient_operator(phase_field)
    f_rho_grad = np.sum(discretization.integrate_over_cell(phase_field_gradient ** 2))

    # gradient_of_phase_field = compute_gradient_of_phase_field(phase_field_gradient)

    f_rho = eta * f_rho_grad + f_dw / eta

    return (f_sigma + w * f_rho) / discretization.cell.domain_volume


def solve_adjoint_problem(discretization, material_data_field, stress_difference):
    # Solve adjoint problem ∂f/∂u=-∂g/∂u
    # Dt C D lambda = -Dt: C : sigma_diff

    # stress difference potential

    df_du_field = 2 * discretization.get_rhs(material_data_field, stress_difference)  # minus sign is already there

    K_fun = lambda x: discretization.apply_system_matrix(material_data_field, displacement_field=x)
    M_fun = lambda x: 1 * x
    # solve the system
    adjoint_field, adjoint_norms = solvers.PCG(K_fun, df_du_field, x0=None, P=M_fun, steps=int(500), toler=1e-6)

    return adjoint_field


def compute_double_well_potential(discretization, phase_field, eta=1):
    # The double-well potential
    # phase field potential = int ( rho^2(1-rho)^2 ) / eta   dx
    # double - well potential
    integrant = (phase_field ** 2) * (1 - phase_field) ** 2
    integral = (np.sum(integrant) / np.prod(integrant.shape)) * discretization.cell.domain_volume
    return eta * integral


def partial_der_of_double_well_potential_wrt_density(discretization, phase_field, eta=1):
    # Derivative of the double-well potential with respect to phase-field
    # phase field potential = int ( rho^2(1-rho)^2 )/eta   dx
    # gradient phase field potential = int ((2 * phase_field( + 2 * phase_field^2  -  3 * phase_field +1 )) )/eta   dx
    # d/dρ(ρ^2 (1 - ρ)^2) = 2 ρ (2 ρ^2 - 3 ρ + 1)

    integrant = (2 * phase_field * (2 * phase_field * phase_field - 3 * phase_field + 1))

    integral = (integrant / np.prod(integrant.shape)) * discretization.cell.domain_volume
    # there is no sum here, as the
    return eta * integral


def compute_gradient_of_phase_field_potential(discretization, phase_field, eta=1):
    # Input: phase_field [1,n,x,y,z]
    # Output: potential [1]
    # phase field gradient potential = int (  (grad(rho))^2 )    dx
    phase_field_gradient = discretization.apply_gradient_operator(phase_field)
    f_rho_grad = np.sum(discretization.integrate_over_cell(phase_field_gradient ** 2))
    return f_rho_grad / eta


def partial_derivative_of_gradient_of_phase_field_potential(discretization, phase_field, eta=1):
    # Input: phase_field [1,n,x,y,z]
    # Output: ∂ potential/ ∂ phase_field [1,n,x,y,z] # Note: one potential per phase field DOF

    # partial derivative of  phase field gradient potential = 2/eta int (  (grad(rho))^2 )    dx
    # Compute       grad (rho). grad I  without using I
    #
    #  (D rho, D I) ==  ( D I, D rho) and thus  == I D_t D rho
    # I try to implement it in the way = 2/eta (int I D_t D rho )
    phase_field_gradient = discretization.apply_gradient_operator(phase_field)
    phase_field_gradient = discretization.apply_quadrature_weights_on_gradient_field(phase_field_gradient)
    Dt_D_rho = discretization.apply_gradient_transposed_operator(phase_field_gradient)

    # integrated_Dt_D_rho =  #/ np.prod(Dt_D_rho.shape)) * discretization.cell.domain_volume
    return 2 * Dt_D_rho / eta


##
def objective_function_stress_equivalence(discretization, actual_stress, target_stress):
    # Input: phase_field [1,n,x,y,z]
    # Output: f_sigma  [1]   == stress difference =  (Sigma_target-Sigma_homogenized,Sigma_target-Sigma_homogenized)

    stress_difference = target_stress[(...,) + (np.newaxis,) * (actual_stress.ndim - 2)] - actual_stress

    f_sigma = np.sum(stress_difference ** 2)
    # can be done np.tensordot(stress_difference, stress_difference,axes=2)
    return f_sigma


def partial_derivative_of_objective_function_stress_equivalence(discretization,
                                                                phase_field,
                                                                target_stress_ij,
                                                                actual_stress_ij,
                                                                material_data_field,
                                                                displacement_field,
                                                                macro_gradient_field):
    # TODO  partial_derivative_of_objective_function_stress_equivalence DOES NOT  work!
    # Input: phase_field [1,n,x,y,z]
    #        material_data_field [d,d,d,d,q,x,y,z] - elasticity
    #        target_stress_ij [d,d]
    #        actual_stress_ij [d,d]
    #
    # Output: ∂ f_sigma/ ∂ rho  == 2*stress difference : int ∂ C/ ∂ rho : grad_s u dx [1,n,x,y,z]
    # TODO: Write it clearly. I do not know what part should do what
    # TODO: CLEAN UP!
    stress_difference_ij = -2 * (target_stress_ij - actual_stress_ij)

    # Gradient of material data with respect to phasse field   % interpolation of rho into quad points
    # I consider linear interpolation of material  C_ijkl= p*rho**(p-1) C^0_ijkl
    # so  ∂ C_ijkl/ ∂ rho = 1* C^0_ijkl
    p = 1
    material_data_field_C_0_rho = material_data_field[..., :, :] * (p * phase_field[0, 0]) ** (p - 1)

    # int(∂ C/ ∂ rho_i  * (macro_grad + micro_grad)) dx / | domain |
    # strain_ijqxyz = discretization.apply_gradient_operator(displacement_field)
    strain_ijqxyz = discretization.apply_gradient_operator_symmetrized(displacement_field)
    strain_ijqxyz = strain_ijqxyz + macro_gradient_field

    material_data_field_C_0_rho = discretization.apply_quadrature_weights(material_data_field_C_0_rho)

    stress_ijqxyz = discretization.apply_material_data(material_data_field_C_0_rho, strain_ijqxyz)

    # stress_C_0 = discretization.apply_material_data_elasticity(material_data_field_C_0_rho, strain)

    # stress_rho_homo_ij = discretization.get_homogenized_stress(material_data_field_C_0_rho,
    #                                                            displacement_field,
    #                                                            macro_gradient_field)
    # partial_derivative = (stress_ij * stress_difference_ij[(...,) + (np.newaxis,) * (stress_ij.ndim - 2)])

    double_contraction_stress_qxyz = np.einsum('ij...,ij->...', stress_ijqxyz, stress_difference_ij)
    # np.tensordot(stress_ijqxyz[...,0,0,0], stress_difference_ij,axes=2)

    # sum_actual_stress = np.einsum('ij...,ij...->...', partial_derivative, partial_derivative)
    # TODO: proper interpolation of phase field to quad points ???
    # partial_derivative = partial_derivative.mean(axis=0)  # Average over quad points in pixel !!!
    partial_derivative_xyz = double_contraction_stress_qxyz.sum(
        axis=0) / discretization.cell.domain_volume  # Average over quad points in pixel !!!
    # compute sum(sigma_ij*sigma_ij)
    # integrant = sum(stress_rho_homo_ij * stress_difference_ij)

    return partial_derivative_xyz


def adjoint_potential(discretization, stress_field_ijqxyz, adjoint_field_inxyz):
    # g = (grad lambda, stress)
    # g = (grad lambda, C grad displacement)
    # g = (grad lambda, C grad displacement)  == lambda_transpose grad_transpose C grad u

    # Input: adjoint_field [f,n,x,y,z]
    #        stress_field  [d,d,q,x,y,z]

    # Output: g  [1] == 0
    # -- -- -- -- -- -- -- -- -- -- --
    # apply quadrature weights
    stress_field_ijqxyz = discretization.apply_quadrature_weights_on_gradient_field(stress_field_ijqxyz)
    force_field_inxyz = discretization.apply_gradient_transposed_operator(stress_field_ijqxyz)
    adjoint_potential_field = np.einsum('i...,i...->...', adjoint_field_inxyz, force_field_inxyz)

    return np.sum(adjoint_potential_field)


def partial_derivative_of_adjoint_potential_wrt_displacement(discretization,
                                                             material_data_field_ijklqxyz,
                                                             adjoint_field_fnxyz):
    # Input: adjoint_field_inxyz [f,n,x,y,z]
    #        material_data_field_C_0_ijklqxyz  [d,d,d,d,q,x,y,z]

    # Output:  ∂ g/ ∂ u [f,n,x,y,z] = ( grad_transpose: C : grad lambda)
    # -- -- -- -- -- -- -- -- -- -- --
    # apply quadrature weights

    force_field_fnxyz = discretization.apply_system_matrix(material_data_field_ijklqxyz, adjoint_field_fnxyz,
                                                           formulation='small_strain')

    return force_field_fnxyz
