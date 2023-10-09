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

    integrant = eta * (2 * phase_field * (2 * phase_field * phase_field - 3 * phase_field + 1))

    integral = (integrant / np.prod(integrant.shape)) * discretization.cell.domain_volume
    # there is no sum here, as the
    return eta * integral


def compute_gradient_of_phase_field_potential(discretization, phase_field, eta=1):
    # phase field gradient potential = int (  (grad(rho))^2 )    dx
    phase_field_gradient = discretization.apply_gradient_operator(phase_field)
    f_rho_grad = np.sum(discretization.integrate_over_cell(phase_field_gradient ** 2))
    return f_rho_grad / eta


def partial_derivative_of_gradient_of_phase_field_potential(discretization, phase_field, eta=1):
    # partial derivative of  phase field gradient potential = 2/eta int (  (grad(rho))^2 )    dx
    # TODO: find a proper way how to computa       grad (rho). grad I without using I
    phase_field_gradient = discretization.apply_gradient_operator(phase_field)
    f_rho_grad = np.sum(discretization.integrate_over_cell(phase_field_gradient ** 2))
    return f_rho_grad / eta
