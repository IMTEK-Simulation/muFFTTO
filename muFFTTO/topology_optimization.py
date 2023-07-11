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


def solve_adjoint_problem(discretization, material_data_field, flux_difference):
    # Solve adjoint problem ∂f/∂u=-∂g/∂u
    # Dt C D lambda = -Dt: C : sigma_diff

    # stress difference potential

    df_du_field = 2 * discretization.get_rhs(material_data_field, flux_difference)  # minus sign is already there

    K_fun = lambda x: discretization.apply_system_matrix(material_data_field, displacement_field=x)
    M_fun = lambda x: 1 * x
    # solve the system
    adjoint_field, adjoint_norms = solvers.PCG(K_fun, df_du_field, x0=None, P=M_fun, steps=int(500), toler=1e-6)

    return adjoint_field
