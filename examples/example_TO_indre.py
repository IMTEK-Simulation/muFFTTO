import numpy as np
import scipy as sc

from muFFTTO import domain
from muFFTTO import solvers


def evaluate_objective_function(actual_flux, target_flux, material_data_field, phase_field, phase_field_gradient,
                                domain_volume, eta, w):
    flux_difference = (actual_flux - target_flux[
        (...,) + (np.newaxis,) * (actual_flux.ndim - 2)]) ** 2

    f_sigma = np.sum(flux_difference)

    double_well_potential = compute_double_well_potential(phase_field, domain_volume)  # set eta to 1 for simplicity
    gradient_of_phase_field = compute_gradient_of_phase_field(phase_field_gradient)

    f_rho = eta * gradient_of_phase_field + double_well_potential / eta

    f = f_sigma + w * f_rho
    return f




def objective_function(discretization, actual_flux, target_flux, phase_field):
    flux_difference = (actual_flux - target_flux[
        (...,) + (np.newaxis,) * (actual_flux.ndim - 2)])

    f_sigma = np.sum(discretization.integrate_over_cell(flux_difference ** 2))

    # double - well potential
    # f_dw = (phase_field ** 2) * (1 - phase_field) ** 2
    integrant = (phase_field ** 2) * (1 - phase_field) ** 2
    f_dw = (np.sum(integrant) / np.prod(integrant.shape)) * discretization.cell.domain_volume

    phase_field_gradient = discretization.apply_gradient_operator(phase_field)
    f_rho_grad = np.sum(discretization.integrate_over_cell(phase_field_gradient ** 2))

    # gradient_of_phase_field = compute_gradient_of_phase_field(phase_field_gradient)

    f_rho = eta * f_rho_grad + f_dw / eta

    return f_sigma + w * f_rho






def compute_double_well_potential(phase_field, domain_volume):
    # The double-well potential
    # phase field potential = int ( rho^2(1-rho)^2 ) / eta   dx

    integrant = (phase_field ** 2) * (1 - phase_field) ** 2

    integral = (np.sum(integrant) / np.prod(integrant.shape)) * domain_volume

    return integral


def compute_gradient_of_phase_field(phase_field_gradient):
    # phase field gradient potential = int (  (grad(rho))^2 )    dx

    integral = discretization.integrate_over_cell(phase_field_gradient ** 2)
    return np.sum(integral)


def compute_df_du(actual_stress, target_stress, material_data_field):
    # df_sigma_du = int( 2*(Sigma-Sigma_target):C:grad_sym )d_Omega # TODO missing grad_sym operator
    stress_difference = 2 * (actual_stress - target_stress[
        (...,) + (np.newaxis,) * (actual_stress.ndim - 2)])
    ### TODO this needs adjustment -----
    stress_difference = discretization.apply_material_data(material_data=material_data_field,  ###
                                                           gradient_field=stress_difference)

    df_sigma_du = discretization.integrate_over_cell(stress_difference)
    return np.sum(df_sigma_du)


def compute_df_sigma_drho(actual_stress, target_stress, material_data_field, phase_field):  # todo dadasdsadassdasdadasd
    # df_sigma_drho = int( 2*(Sigma-Sigma_target):dK/drho )d_Omega
    # dK/drho =

    stress_difference = 2 * actual_stress - target_stress[
        (...,) + (np.newaxis,) * (actual_stress.ndim - 2)]

    material_data_phase_field = 5

    stress_difference = discretization.apply_material_data(material_data=material_data_field,
                                                           gradient_field=stress_difference)

    integral = discretization.integrate_over_cell(stress_difference)
    return integral


def compute_gradient_of_double_well_potential(phase_field, w=1, eta=1):
    # Derivative of the double-well potential with respect to phase-field
    # phase field potential = int ( rho^2(1-rho)^2 )/eta   dx
    # gradient phase field potential = int ((2 * phase_field( + 2 * phase_field^2  -  3 * phase_field +1 )) )/eta   dx
    # d/dρ(ρ^2 (1 - ρ)^2) = 2 ρ (2 ρ^2 - 3 ρ + 1)

    # phase field gradient  =( |grad (rgo)|^2 ) *eta

    integrant = (2 * phase_field*(2 * phase_field * phase_field - 3 * phase_field + 1))
    # INDRE  derivative = w / eta * 2 * phase * (1 - phase) * (1 - 2 * phase) * lengths[0] * lengths[1] / nb_pixels
    integral = discretization.integrate_over_cell(integrant)

    return integral





def adjoin_problem(material_data_field_i, K_fun, flux_difference):
    ####  solve the adjoin problem    K lambda = -df_du

    df_du_field = discretization.get_rhs(material_data_field_i, 2 * flux_difference)  # minus sign is already there
    # solve the system
    adjoin_field, adjoin_norms = solvers.PCG(K_fun, df_du_field, x0=None, P=M_fun, steps=int(500), toler=1e-6)

    return adjoin_field


def sensitivity(material_data_field_i, K_fun, actual_flux, target_flux):
    flux_difference = (actual_flux - target_flux[
        (...,) + (np.newaxis,) * (actual_flux.ndim - 2)])

    adjoin_field = adjoin_problem(material_data_field_i, K_fun, flux_difference)
    #  lambda * dK_drho* u  =====  3*rho^2 *  lambda^t * K* u # TODO don't be lazy and write it in index notation !!!!!!!!

    ## Gradient of material data with respect to prahse field   Sadjoin_m =   lambda_j *  K_ij,m u_i
    grad_material_data_field_i = 3 * (
            phase_field ** 2) * material_data_field_i  # this is missing dimension and exploiting the diagonal structure of partial dervative
    dK_drho_fun = lambda x: discretization.apply_system_matrix(grad_material_data_field_i, x)

    #
    lambda_dK_drho_u = adjoin_field * dK_drho_fun(
        solution_i)  # here is hiden a hamadard product and diagonal tensors

    #  compute df_drho
    # gradient phase field potential = int ((2 * phase_field( + 2 * phase_field^2  -  3 * phase_field +1 )) )/eta   dx

    dfdw_drho_field = 2 * phase_field * (
            2 * phase_field * phase_field - 3 * phase_field + 1)  # TODO check which one is valid
    int_dfdw_drho = ((dfdw_drho_field) / np.prod(dfdw_drho_field.shape)) * discretization.cell.domain_volume

    strain_field = discretization.apply_gradient_operator(solution_i)

    grad_material_data_field_i_weight = discretization.apply_quadrature_weights(grad_material_data_field_i)

    dstress_rho_field = discretization.apply_material_data(grad_material_data_field_i_weight,
                                                           strain_field)  # huuuuuge block diagonal field. We store only diagonals
    dfsigma_drho_field = np.sum(2 * flux_difference * dstress_rho_field, axis=1)
    # discretization.get_rhs(grad_material_data_field_i, 2 * flux_difference)
    int_dfsigma_drho = dfsigma_drho_field

    df_drho = int_dfsigma_drho + w * int_dfdw_drho
    sensitivity = df_drho + lambda_dK_drho_u

    return sensitivity


if __name__ == "__main__":
    w = 1
    eta = 1
    for problem_type in ['conductivity']:
        domain_size = [3, 4]
        my_cell = domain.PeriodicUnitCell(domain_size=domain_size,
                                          problem_type=problem_type)
        number_of_pixels = (8, 9)
        discretization_type = 'finite_element'
        element_type = 'linear_triangles'
        discretization = domain.Discretization(cell=my_cell,
                                               number_of_pixels=number_of_pixels,
                                               discretization_type=discretization_type,
                                               element_type=element_type)
        # Material set up
        K_1, G_1 = domain.get_bulk_and_shear_modulus(E=1, poison=0.2)

        mat_1 = np.array([[1, 0], [0, 1]])
        material_data_field_0 = np.einsum('ij,qxy->ijqxy', mat_1,
                                          np.ones(np.array([discretization.nb_quad_points_per_pixel,
                                                            *discretization.nb_of_pixels])))
        macro_gradient = np.zeros([1, discretization.domain_dimension])
        macro_gradient[0, :] = np.array([1, 0])

        target_flux = np.array([1, 0.3])

        phase_field = np.random.rand(
            *discretization.get_unknown_size_field().shape)  # Phase field has  oone  value per pixel

        # Update material data based on current Phase-field
        material_data_field_i = (phase_field ** 3) * material_data_field_0

        ##### solve equilibrium constrain
        # set up system
        macro_gradient_field = discretization.get_macro_gradient_field(macro_gradient)
        rhs = discretization.get_rhs(material_data_field_i, macro_gradient_field)

        K_fun = lambda x: discretization.apply_system_matrix(material_data_field_i, x)
        M_fun = lambda x: 1 * x
        # solve the system
        solution_i, norms = solvers.PCG(K_fun, rhs, x0=None, P=M_fun, steps=int(500), toler=1e-6)

        # get homogenized flux
        homogenized_flux = discretization.get_homogenized_stress(material_data_field_i,
                                                                 displacement_field_fnxyz=solution_i,
                                                                 macro_gradient_field_ijqxyz=macro_gradient_field)
        actual_flux = discretization.get_stress_field(material_data_field_i,
                                                      displacement_field_fnxyz=solution_i,
                                                      macro_gradient_field_ijqxyz=macro_gradient_field)

        ############################ evaluate objective functions
        # f = (flux_h -flux_target)^2 + w*eta* int (  (grad(rho))^2 )dx  +    int ( rho^2(1-rho)^2 ) / eta   dx
        # f =  f_sigma + w*eta* f_rho_grad  + f_dw/eta

        f = objective_function(discretization, actual_flux, target_flux, phase_field)

        ########################### Sensitivity analysis

        # df_du = compute_df_du(flux, target_flux, material_data_field_i)

        # integrant_df_du = discretization.apply_material_data(material_data=material_data_field_i,  ###
        #                                                    gradient_field=2 * flux_difference)

        #### !!!!!!!!!! this is just gues what it should looks like ####### !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        # = discretization.apply_gradient_transposed_operator(integrant_df_du)

        # df_du = np.sum(discretization.integrate_over_cell(integrant_df_du))

        ####  solve the adjoin problem    K lambda = -df_du
        # set up system
        # df_du_field = np.ones(discretization.get_unknown_size_field().shape) * df_du
        # TODO this is Handwaving expresion becous I do not know how to implement *C*D

        sensitivity = sensitivity(material_data_field_i, K_fun, actual_flux, target_flux)

        xopt = sc.optimize.minimize(objective_function(discretization, actual_flux, target_flux, phase_field), phase_field, method='bfgs', jac=sensitivity(material_data_field_i, K_fun, actual_flux, target_flux), options={'disp': 1})


        print(5)
