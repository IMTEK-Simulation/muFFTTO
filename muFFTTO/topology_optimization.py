import numpy as np
import scipy as sc

from muFFTTO import domain
from muFFTTO import solvers


def objective_function_small_strain(discretization,
                                    actual_stress_ij,
                                    target_stress_ij,
                                    phase_field_1nxyz,
                                    eta=1,
                                    w=1):
    # evaluate objective functions
    # f = (flux_h -flux_target)^2 + w*eta* int (  (grad(rho))^2 )dx  +    int ( rho^2(1-rho)^2 ) / eta   dx
    # f =  f_sigma + w*eta* f_rho_grad  + f_dw/eta

    # stress difference potential: actual_stress_ij is homogenized stress
    stress_difference_ij = actual_stress_ij - target_stress_ij

    f_sigma = np.sum(stress_difference_ij ** 2)

    # double - well potential
    integrant = (phase_field_1nxyz ** 2) * (1 - phase_field_1nxyz) ** 2
    f_dw = (np.sum(integrant) / np.prod(integrant.shape)) * discretization.cell.domain_volume

    phase_field_gradient = discretization.apply_gradient_operator(phase_field_1nxyz)
    f_rho_grad = np.sum(discretization.integrate_over_cell(phase_field_gradient ** 2))

    # gradient_of_phase_field = compute_gradient_of_phase_field(phase_field_gradient)

    f_rho = eta * f_rho_grad + f_dw / eta
    print('f_rho_grad =  {} '.format((f_rho_grad)))

    print('f_dw =  {} '.format((f_dw)))
    print('f_sigma =  {} '.format((f_sigma)))
    print('objective_function = {} '.format((f_sigma + w * f_rho)))

    return (f_sigma + w * f_rho)  # / discretization.cell.domain_volume


def compute_double_well_potential(discretization, phase_field, eta=1):
    # The double-well potential
    # phase field potential = int ( rho^2(1-rho)^2 ) / eta   dx
    # double - well potential
    integrant = (phase_field ** 2) * (1 - phase_field) ** 2
    integral = (np.sum(integrant) / np.prod(integrant.shape)) * discretization.cell.domain_volume
    return integral / eta


def partial_der_of_double_well_potential_wrt_density(discretization, phase_field_1nxyz, eta=1):
    # Derivative of the double-well potential with respect to phase-field
    # phase field potential = int ( rho^2(1-rho)^2 )/eta   dx
    # gradient phase field potential = int ((2 * phase_field( + 2 * phase_field^2  -  3 * phase_field +1 )) )/eta   dx
    # d/dρ(ρ^2 (1 - ρ)^2) = 2 ρ (2 ρ^2 - 3 ρ + 1)

    integrant_fnxyz = (2 * phase_field_1nxyz * (2 * phase_field_1nxyz * phase_field_1nxyz - 3 * phase_field_1nxyz + 1))

    integral_fnxyz = (integrant_fnxyz / np.prod(integrant_fnxyz.shape)) * discretization.cell.domain_volume
    # there is no sum here
    return integral_fnxyz / eta


def compute_gradient_of_phase_field_potential(discretization, phase_field_1nxyz, eta=1):
    # Input: phase_field [1,n,x,y,z]
    # Output: potential [1]
    # phase field gradient potential = int (  (grad(rho))^2 )    dx
    phase_field_gradient = discretization.apply_gradient_operator(phase_field_1nxyz)
    f_rho_grad = np.sum(discretization.integrate_over_cell(phase_field_gradient ** 2))
    return eta * f_rho_grad


def partial_derivative_of_gradient_of_phase_field_potential(discretization, phase_field_1nxyz, eta=1):
    # Input: phase_field [1,n,x,y,z]
    # Output: ∂ potential/ ∂ pha adjoint_potential ase_field [1,n,x,y,z] # Note: one potential per phase field DOF

    # partial derivative of  phase field gradient potential = 2/eta int (  (grad(rho))^2 )    dx
    # Compute       grad (rho). grad I  without using I
    #
    #  (D rho, D I) ==  ( D I, D rho) and thus  == I D_t D rho
    # I try to implement it in the way = 2/eta (int I D_t D rho )
    phase_field_gradient = discretization.apply_gradient_operator(phase_field_1nxyz)
    phase_field_gradient = discretization.apply_quadrature_weights_on_gradient_field(phase_field_gradient)
    Dt_D_rho = discretization.apply_gradient_transposed_operator(phase_field_gradient)

    # integrated_Dt_D_rho =  #/ np.prod(Dt_D_rho.shape)) * discretization.cell.domain_volume
    return 2 * Dt_D_rho * eta


##
def objective_function_stress_equivalence(discretization, actual_stress_ij, target_stress_ij):
    # Input: phase_field [1,n,x,y,z]
    # Output: f_sigma  [1]   == stress difference =  (Sigma_target-Sigma_homogenized,Sigma_target-Sigma_homogenized)

    # stress difference potential: actual_stress_ij is homogenized stress
    stress_difference_ij = actual_stress_ij - target_stress_ij

    # f_sigma = np.sum(stress_difference_ij ** 2)
    f_sigma = np.einsum('ij,ij->', stress_difference_ij, stress_difference_ij)
    # can be done np.tensordot(stress_difference, stress_difference,axes=2)
    return f_sigma


def partial_derivative_of_objective_function_stress_equivalence_wrt_phase_field(discretization,
                                                                                material_data_field_ijklqxyz,
                                                                                displacement_field_fnxyz,
                                                                                macro_gradient_field_ijqxyz,
                                                                                phase_field_1nxyz,
                                                                                target_stress_ij,
                                                                                actual_stress_ij,
                                                                                p):
    # Input: phase_field [1,n,x,y,z]
    #        material_data_field [d,d,d,d,q,x,y,z] - elasticity
    #        target_stress_ij [d,d]
    #        actual_stress_ij [d,d]
    # -- -- -- -- -- -- -- -- -- -- --

    # Gradient of material data with respect to phase field
    # % interpolation of rho into quad points
    # I consider linear interpolation of material  C_ijkl= p*rho**(p-1) C^0_ijkl
    # so  ∂ C_ijkl/ ∂ rho = 1* C^0_ijkl
    # p = 2
    material_data_field_ijklqxyz = material_data_field_ijklqxyz[..., :, :] * (
            p * np.power(phase_field_1nxyz[0, 0], (p - 1)))

    # int(∂ C/ ∂ rho_i  * (macro_grad + micro_grad)) dx / | domain |

    # compute strain field from to displacement and macro gradient
    strain_ijqxyz = discretization.apply_gradient_operator_symmetrized(displacement_field_fnxyz)
    strain_ijqxyz = macro_gradient_field_ijqxyz + strain_ijqxyz

    # compute stress field
    # ddot42 = lambda A4, B2: np.einsum('ijklxyz,lkxyz  ->ijxyz  ', A4, B2)
    stress_field_ijqxyz = np.einsum('ijkl...,lk...->ij...', material_data_field_ijklqxyz, strain_ijqxyz)

    # apply quadrature weights
    stress_ijqxyz = discretization.apply_quadrature_weights_on_gradient_field(
        stress_field_ijqxyz)

    # stress difference
    stress_difference_ij = actual_stress_ij - target_stress_ij

    double_contraction_stress_qxyz = np.einsum('ij,ijqxy...->qxy...',
                                               stress_difference_ij,
                                               stress_ijqxyz)
    # np.einsum('ijqxyz  ,jiqxyz  ->xyz    ', A2, B2)
    # Average over quad points in pixel !!!
    # partial_derivative = partial_derivative.mean(axis=0)
    partial_derivative_xyz = double_contraction_stress_qxyz.sum(axis=0)

    return 2 * partial_derivative_xyz / discretization.cell.domain_volume


def partial_derivative_of_objective_function_wrt_phase_field_OLD(discretization,
                                                                 material_data_field_ijklqxyz,
                                                                 displacement_field_fnxyz,
                                                                 macro_gradient_field_ijqxyz,
                                                                 phase_field_1nxyz,
                                                                 target_stress_ij,
                                                                 actual_stress_ij,
                                                                 p):
    df_drho = partial_derivative_of_objective_function_stress_equivalence_wrt_phase_field(discretization,
                                                                                          material_data_field_ijklqxyz,
                                                                                          displacement_field_fnxyz,
                                                                                          macro_gradient_field_ijqxyz,
                                                                                          phase_field_1nxyz,
                                                                                          target_stress_ij,
                                                                                          actual_stress_ij,
                                                                                          p)

    dgradrho_drho = partial_derivative_of_gradient_of_phase_field_potential(discretization,
                                                                            phase_field_1nxyz,
                                                                            eta=1)

    ddouble_well_drho_drho = partial_der_of_double_well_potential_wrt_density(discretization,
                                                                              phase_field_1nxyz,
                                                                              eta=1)

    return df_drho + dgradrho_drho + ddouble_well_drho_drho


def partial_derivative_of_objective_function_wrt_phase_field(discretization,
                                                             material_data_field_ijklqxyz,
                                                             displacement_field_fnxyz,
                                                             macro_gradient_field_ijqxyz,
                                                             phase_field_1nxyz,
                                                             target_stress_ij,
                                                             actual_stress_ij,
                                                             p,
                                                             eta=1):
    # Input:
    #        material_data_field_ijklqxyz [d,d,d,d,q,x,y,z] - elasticity without applied phase field -- C_0
    #        displacement_field_fnxyz [f,n,x,y,z]
    #        phase_field_1nxyz [1,n,x,y,z]
    #        macro_gradient_field_ijqxyz [d,d,q,x,y,z]
    #        target_stress_ij [d,d]
    #        actual_stress_ij [d,d] # homogenized stress
    #        p [1]  # polynomial order of a material interpolation
    #        eta [1] # weight parameter for balancing the phase field terms

    # Output:
    #        df_drho_fnxyz [1,n,x,y,z]
    # -- -- -- -- -- -- -- -- -- -- --

    # -----    stress difference potential ----- #
    # Gradient of material data with respect to phase field
    material_data_field_ijklqxyz = material_data_field_ijklqxyz[..., :, :] * (
            p * np.power(phase_field_1nxyz[0, 0], (p - 1)))

    # compute strain field from to displacement and macro gradient
    strain_ijqxyz = discretization.apply_gradient_operator_symmetrized(displacement_field_fnxyz)
    strain_ijqxyz = macro_gradient_field_ijqxyz + strain_ijqxyz

    # compute stress field
    stress_field_ijqxyz = np.einsum('ijkl...,lk...->ij...', material_data_field_ijklqxyz, strain_ijqxyz)

    # apply quadrature weights
    stress_ijqxyz = discretization.apply_quadrature_weights_on_gradient_field(
        stress_field_ijqxyz)

    # stress difference
    stress_difference_ij = actual_stress_ij - target_stress_ij

    double_contraction_stress_qxyz = np.einsum('ij,ijqxy...->qxy...',
                                               stress_difference_ij,
                                               stress_ijqxyz)
    # Average over quad points in pixel !!!
    partial_derivative_xyz = double_contraction_stress_qxyz.sum(axis=0)

    dfstress_drho = 2 * partial_derivative_xyz / discretization.cell.domain_volume

    # -----    phase field gradient potential ----- #
    # partial derivative of  phase field gradient potential = 2/eta int (  (grad(rho))^2 )    dx
    #  (D rho, D I) ==  ( D I, D rho) and thus  == I D_t D rho
    # I implement it in the way = 2/eta (  I D_t D rho )
    phase_field_gradient = discretization.apply_gradient_operator(phase_field_1nxyz)
    phase_field_gradient = discretization.apply_quadrature_weights_on_gradient_field(phase_field_gradient)
    Dt_D_rho = discretization.apply_gradient_transposed_operator(phase_field_gradient)

    dgradrho_drho = 2 * Dt_D_rho

    # -----    Double well potential ----- #

    # Derivative of the double-well potential with respect to phase-field
    # phase field potential = int ( rho^2(1-rho)^2 )/eta   dx
    # gradient phase field potential = int ((2 * phase_field( + 2 * phase_field^2  -  3 * phase_field +1 )) )/eta   dx
    # d/dρ(ρ^2 (1 - ρ)^2) = 2 ρ (2 ρ^2 - 3 ρ + 1)

    integrant_fnxyz = (2 * phase_field_1nxyz * (2 * phase_field_1nxyz * phase_field_1nxyz - 3 * phase_field_1nxyz + 1))

    integral_fnxyz = (integrant_fnxyz / np.prod(integrant_fnxyz.shape)) * discretization.cell.domain_volume
    # there is no sum here
    ddouble_well_drho_drho = integral_fnxyz

    return dfstress_drho + dgradrho_drho * eta + ddouble_well_drho_drho / eta


def partial_der_of_objective_function_wrt_displacement_small_strain(discretization,
                                                                    material_data_field_ijklqxyz,
                                                                    stress_difference_ij,
                                                                    eta=1,
                                                                    w=1):
    # Input: material_data_field [d,d,d,d,q,x,y,z] - elasticity
    #        stress_diff_ij [d,d] # difference between homogenized stress and target stress
    #
    # Output: ∂ f_sigma/ ∂ u  = - (2 / |domain size|) int  grad_transpose : C: sigma_diff dx [f,n,x,y,z]

    stress_field_ijqxyz = discretization.apply_material_data(material_data_field_ijklqxyz, stress_difference_ij)

    stress_field_ijqxyz = discretization.apply_quadrature_weights_on_gradient_field(stress_field_ijqxyz)

    df_sigma_du_fnxyz = discretization.apply_gradient_transposed_operator(stress_field_ijqxyz)

    return -2 * df_sigma_du_fnxyz / discretization.cell.domain_volume


def solve_adjoint_problem(discretization, material_data_field_ijklqxyz,
                          stress_difference_ij,
                          formulation='small_strain'):
    # Solve adjoint problem ∂f/∂u=-∂g/∂u
    # Dt C D lambda = - 2/|omega| Dt: C : sigma_diff

    # stress difference potential
    # rhs=-Dt*wA*E  -- we can use it to assemble df_du_field

    stress_difference_ijqxyz = discretization.get_gradient_size_field()
    stress_difference_ijqxyz[:, :, ...] = stress_difference_ij[
        (...,) + (np.newaxis,) * (stress_difference_ijqxyz.ndim - 2)]
    df_du_field = 2 * discretization.get_rhs(material_data_field_ijklqxyz=material_data_field_ijklqxyz,
                                             macro_gradient_field_ijqxyz=stress_difference_ijqxyz) / discretization.cell.domain_volume  # minus sign is already there
    #
    K_fun = lambda x: discretization.apply_system_matrix(material_data_field=material_data_field_ijklqxyz,
                                                         displacement_field=x,
                                                         formulation=formulation)
    M_fun = lambda x: 1 * x

    # solve the system
    adjoint_field, adjoint_norms = solvers.PCG(K_fun, df_du_field, x0=None, P=M_fun, steps=int(500), toler=1e-6)

    return adjoint_field


def adjoint_potential(discretization, stress_field_ijqxyz, adjoint_field_fnxyz):
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
    adjoint_potential_field = np.einsum('i...,i...->...', adjoint_field_fnxyz, force_field_inxyz)

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


def partial_derivative_of_adjoint_potential_wrt_phase_field(discretization,
                                                            material_data_field_ijklqxyz,
                                                            displacement_field_fnxyz,
                                                            macro_gradient_field_ijqxyz,
                                                            phase_field_1nxyz,
                                                            adjoint_field_fnxyz,
                                                            p=1):
    # Input:
    #        material_data_field_ijklqxyz [d,d,d,d,q,x,y,z] - elasticity without applied phase field -- C_0
    #        displacement_field_fnxyz [f,n,x,y,z]
    #        macro_gradient_field_ijqxyz [d,d,q,x,y,z]
    #        phase_field_1nxyz [1,n,x,y,z]
    #        adjoint_field_fnxyz  [f,n,x,y,z]
    #        p [1]  # polynomial order of a material interpolation

    # Output:
    #        dg_drho_fnxyz [1, n, x, y, z]
    # -- -- -- -- -- -- -- -- -- -- --

    # Gradient of material data with respect to phasse field   % interpolation of rho into quad points
    # I consider linear interpolation of material  C_ijkl= p*rho**(p-1) C^0_ijkl
    # so  ∂ C_ijkl/ ∂ rho = 1* C^0_ijkl
    material_data_field_ijklqxyz = material_data_field_ijklqxyz[..., :, :] * (
            p * np.power(phase_field_1nxyz[0, 0], (p - 1)))

    # compute strain field from to displacement and macro gradient
    strain_ijqxyz = discretization.apply_gradient_operator_symmetrized(displacement_field_fnxyz)
    strain_ijqxyz = macro_gradient_field_ijqxyz + strain_ijqxyz

    # compute stress field
    # ddot42 = lambda A4, B2: np.einsum('ijklxyz,lkxyz  ->ijxyz  ', A4, B2)
    stress_field_ijqxyz = np.einsum('ijkl...,lk...->ij...', material_data_field_ijklqxyz, strain_ijqxyz)

    # local_stress=np.einsum('ijkl,lk->ij',material_data_field_ijklqxyz[...,0,0,0] , strain_ijqxyz[...,0,0,0])
    # apply quadrature weights
    stress_field_ijqxyz = discretization.apply_quadrature_weights_on_gradient_field(stress_field_ijqxyz)

    # gradient of adjoint_field
    adjoint_field_gradient_ijqxyz = discretization.apply_gradient_operator_symmetrized(adjoint_field_fnxyz)

    # ddot22 = lambda A2, B2:  np.einsum('ijqxyz  ,jiqxyz  ->qxyz    ', A2, B2)
    double_contraction_stress_qxyz = np.einsum('ij...,ij...->...',
                                               adjoint_field_gradient_ijqxyz,
                                               stress_field_ijqxyz)

    return double_contraction_stress_qxyz.sum(axis=0)


def sensitivity_OLD(discretization,
                    material_data_field_ijklqxyz,
                    displacement_field_fnxyz,
                    macro_gradient_field_ijqxyz,
                    phase_field_1nxyz,
                    adjoint_field_fnxyz,
                    target_stress_ij,
                    actual_stress_ij,
                    p,
                    eta=1):
    df_drho = partial_derivative_of_objective_function_wrt_phase_field(
        discretization=discretization,
        material_data_field_ijklqxyz=material_data_field_ijklqxyz,
        displacement_field_fnxyz=displacement_field_fnxyz,
        macro_gradient_field_ijqxyz=macro_gradient_field_ijqxyz,
        phase_field_1nxyz=phase_field_1nxyz,
        target_stress_ij=target_stress_ij,
        actual_stress_ij=actual_stress_ij,
        p=p,
        eta=eta)

    dg_drho = partial_derivative_of_adjoint_potential_wrt_phase_field(
        discretization=discretization,
        material_data_field_ijklqxyz=material_data_field_ijklqxyz,
        displacement_field_fnxyz=displacement_field_fnxyz,
        macro_gradient_field_ijqxyz=macro_gradient_field_ijqxyz,
        phase_field_1nxyz=phase_field_1nxyz,
        adjoint_field_fnxyz=adjoint_field_fnxyz,
        p=p)

    return df_drho + dg_drho


def sensitivity(discretization,
                material_data_field_ijklqxyz,
                displacement_field_fnxyz,
                macro_gradient_field_ijqxyz,
                phase_field_1nxyz,
                adjoint_field_fnxyz,
                target_stress_ij,
                actual_stress_ij,
                p,
                eta=1):
    # Input:
    #        material_data_field_ijklqxyz [d,d,d,d,q,x,y,z] - elasticity without applied phase field -- C_0
    #        displacement_field_fnxyz [f,n,x,y,z]
    #        phase_field_1nxyz [1,n,x,y,z]
    #        macro_gradient_field_ijqxyz [d,d,q,x,y,z]
    #        adjoint_field_fnxyz  [f,n,x,y,z]
    #        target_stress_ij [d,d]
    #        actual_stress_ij [d,d] # homogenized stress
    #        p [1]  # polynomial order of a material interpolation
    #        eta [1] # weight parameter for balancing the phase field terms

    # Output:
    #        df_drho_fnxyz [1,n,x,y,z]
    # -- -- -- -- -- -- -- -- -- -- --

    # -----    stress difference potential ----- #
    # Gradient of material data with respect to phase field
    material_data_field_ijklqxyz = material_data_field_ijklqxyz[..., :, :] * (
            p * np.power(phase_field_1nxyz[0, 0], (p - 1)))

    # compute strain field from to displacement and macro gradient
    strain_ijqxyz = discretization.apply_gradient_operator_symmetrized(displacement_field_fnxyz)
    strain_ijqxyz = macro_gradient_field_ijqxyz + strain_ijqxyz

    # compute stress field
    stress_field_ijqxyz = np.einsum('ijkl...,lk...->ij...', material_data_field_ijklqxyz, strain_ijqxyz)

    # apply quadrature weights
    stress_field_ijqxyz = discretization.apply_quadrature_weights_on_gradient_field(stress_field_ijqxyz)

    # ---  part that is unique for  df_drho ---
    # stress difference
    stress_difference_ij = actual_stress_ij - target_stress_ij

    double_contraction_stress_qxyz = np.einsum('ij,ijqxy...->qxy...',
                                               stress_difference_ij,
                                               stress_field_ijqxyz)
    # Average over quad points in pixel !!!
    partial_derivative_xyz = double_contraction_stress_qxyz.sum(axis=0)

    dfstress_drho = 2 * partial_derivative_xyz / discretization.cell.domain_volume

    # -----    phase field gradient potential ----- #
    # partial derivative of  phase field gradient potential = 2/eta int (  (grad(rho))^2 )    dx
    #  (D rho, D I) ==  ( D I, D rho) and thus  == I D_t D rho
    # I implement it in the way = 2/eta (  I D_t D rho )
    phase_field_gradient = discretization.apply_gradient_operator(phase_field_1nxyz)
    phase_field_gradient = discretization.apply_quadrature_weights_on_gradient_field(phase_field_gradient)
    Dt_D_rho = discretization.apply_gradient_transposed_operator(phase_field_gradient)

    dgradrho_drho = 2 * Dt_D_rho

    # -----    Double well potential ----- #

    # Derivative of the double-well potential with respect to phase-field
    # phase field potential = int ( rho^2(1-rho)^2 )/eta   dx
    # gradient phase field potential = int ((2 * phase_field( + 2 * phase_field^2  -  3 * phase_field +1 )) )/eta   dx
    # d/dρ(ρ^2 (1 - ρ)^2) = 2 ρ (2 ρ^2 - 3 ρ + 1)

    integrant_fnxyz = (2 * phase_field_1nxyz * (2 * phase_field_1nxyz * phase_field_1nxyz - 3 * phase_field_1nxyz + 1))

    integral_fnxyz = (integrant_fnxyz / np.prod(integrant_fnxyz.shape)) * discretization.cell.domain_volume
    # there is no sum here
    ddouble_well_drho_drho = integral_fnxyz

    # sum of all parts of df_drho
    df_drho = dfstress_drho + dgradrho_drho * eta + ddouble_well_drho_drho / eta

    # --------------------------------------

    # gradient of adjoint_field
    adjoint_field_gradient_ijqxyz = discretization.apply_gradient_operator_symmetrized(adjoint_field_fnxyz)

    # ddot22 = lambda A2, B2:  np.einsum('ijqxyz  ,jiqxyz  ->qxyz    ', A2, B2)
    double_contraction_stress_qxyz = np.einsum('ij...,ij...->...',
                                               adjoint_field_gradient_ijqxyz,
                                               stress_field_ijqxyz)

    dg_drho = double_contraction_stress_qxyz.sum(axis=0)

    return df_drho + dg_drho


def sensitivity_with_adjoint_problem(discretization,
                                     material_data_field_ijklqxyz,
                                     displacement_field_fnxyz,
                                     macro_gradient_field_ijqxyz,
                                     phase_field_1nxyz,
                                     target_stress_ij,
                                     actual_stress_ij,
                                     formulation,
                                     p,
                                     eta=1,
                                     weight=1):
    # Input:
    #        material_data_field_ijklqxyz [d,d,d,d,q,x,y,z] - elasticity without applied phase field -- C_0
    #        displacement_field_fnxyz [f,n,x,y,z]
    #        phase_field_1nxyz [1,n,x,y,z]
    #        macro_gradient_field_ijqxyz [d,d,q,x,y,z]
    #        target_stress_ij [d,d]
    #        actual_stress_ij [d,d] # homogenized stress
    #        formulation  - 'finite_strain', 'small_strain'
    #        p [1]  # polynomial order of a material interpolation
    #        eta [1] # weight parameter for balancing the phase field terms

    # Output:
    #        df_drho_fnxyz [1,n,x,y,z]
    # -- -- -- -- -- -- -- -- -- -- --

    # -----    stress difference potential ----- #
    # Gradient of material data with respect to phase field
    dmaterial_data_field_drho_ijklqxyz = material_data_field_ijklqxyz[..., :, :] * (
            p * np.power(phase_field_1nxyz[0, 0], (p - 1)))

    # compute strain field from to displacement and macro gradient
    strain_ijqxyz = discretization.apply_gradient_operator_symmetrized(displacement_field_fnxyz)
    strain_ijqxyz = macro_gradient_field_ijqxyz + strain_ijqxyz

    # compute stress field
    stress_field_ijqxyz = np.einsum('ijkl...,lk...->ij...', dmaterial_data_field_drho_ijklqxyz, strain_ijqxyz)

    # apply quadrature weights
    stress_field_ijqxyz = discretization.apply_quadrature_weights_on_gradient_field(stress_field_ijqxyz)

    # ---  part that is unique for  df_drho ---
    # stress difference
    stress_difference_ij = actual_stress_ij - target_stress_ij

    double_contraction_stress_qxyz = np.einsum('ij,ijqxy...->qxy...',
                                               stress_difference_ij,
                                               stress_field_ijqxyz)
    # Average over quad points in pixel !!!
    partial_derivative_xyz = double_contraction_stress_qxyz.sum(axis=0)

    dfstress_drho = 2 * partial_derivative_xyz / discretization.cell.domain_volume

    # -----    phase field gradient potential ----- #
    # partial derivative of  phase field gradient potential = 2/eta int (  (grad(rho))^2 )    dx
    #  (D rho, D I) ==  ( D I, D rho) and thus  == I D_t D rho
    # I implement it in the way = 2/eta (  I D_t D rho )
    phase_field_gradient = discretization.apply_gradient_operator(phase_field_1nxyz)
    phase_field_gradient = discretization.apply_quadrature_weights_on_gradient_field(phase_field_gradient)
    Dt_D_rho = discretization.apply_gradient_transposed_operator(phase_field_gradient)

    dgradrho_drho = 2 * Dt_D_rho

    # -----    Double well potential ----- #

    # Derivative of the double-well potential with respect to phase-field
    # phase field potential = int ( rho^2(1-rho)^2 )/eta   dx
    # gradient phase field potential = int ((2 * phase_field( + 2 * phase_field^2  -  3 * phase_field +1 )) )/eta   dx
    # d/dρ(ρ^2 (1 - ρ)^2) = 2 ρ (2 ρ^2 - 3 ρ + 1)

    integrant_fnxyz = (2 * phase_field_1nxyz * (2 * phase_field_1nxyz * phase_field_1nxyz - 3 * phase_field_1nxyz + 1))

    integral_fnxyz = (integrant_fnxyz / np.prod(integrant_fnxyz.shape)) * discretization.cell.domain_volume
    # there is no sum here
    ddouble_well_drho_drho = integral_fnxyz

    # sum of all parts of df_drho
    df_drho = dfstress_drho + weight * (dgradrho_drho * eta + ddouble_well_drho_drho / eta)

    # --------------------------------------
    # Solve adjoint problem ∂f/∂u=-∂g/∂u
    # Dt C D lambda = - 2/|omega| Dt: C : sigma_diff
    material_data_field_C_0_rho_ijklqxyz = material_data_field_ijklqxyz[..., :, :] * np.power(phase_field_1nxyz,
                                                                                              p)
    # stress difference potential
    # rhs=-Dt*wA*E  -- we can use it to assemble df_du_field

    stress_difference_ijqxyz = discretization.get_gradient_size_field()
    stress_difference_ijqxyz[:, :, ...] = stress_difference_ij[
        (...,) + (np.newaxis,) * (stress_difference_ijqxyz.ndim - 2)]

    df_du_field = 2 * discretization.get_rhs(material_data_field_ijklqxyz=material_data_field_C_0_rho_ijklqxyz,
                                             macro_gradient_field_ijqxyz=stress_difference_ijqxyz) / discretization.cell.domain_volume  # minus sign is already there
    #
    K_fun = lambda x: discretization.apply_system_matrix(material_data_field=material_data_field_C_0_rho_ijklqxyz,
                                                         displacement_field=x,
                                                         formulation=formulation)
    M_fun = lambda x: 1 * x

    # solve the system
    adjoint_field_fnxyz, adjoint_norms = solvers.PCG(K_fun, df_du_field, x0=None, P=M_fun, steps=int(500), toler=1e-6)

    # gradient of adjoint_field
    adjoint_field_gradient_ijqxyz = discretization.apply_gradient_operator_symmetrized(adjoint_field_fnxyz)

    # ddot22 = lambda A2, B2:  np.einsum('ijqxyz  ,jiqxyz  ->qxyz    ', A2, B2)
    double_contraction_stress_qxyz = np.einsum('ij...,ij...->...',
                                               adjoint_field_gradient_ijqxyz,
                                               stress_field_ijqxyz)

    dg_drho = double_contraction_stress_qxyz.sum(axis=0)

    return df_drho + dg_drho
