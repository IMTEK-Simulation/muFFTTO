import warnings
import gc

import muGrid
import numpy as np
import scipy as sc
import time

from muFFTTO import domain
from muFFTTO import solvers

from NuMPI.Tools import Reduction
from mpi4py import MPI

# phase field part is identical to the elasticity part
from muFFTTO.topology_optimization import objective_function_phase_field
from muFFTTO.topology_optimization import sensitivity_phase_field_term_FE_NEW


def compute_flux_equivalence_potential(actual_flux_ij: np.ndarray,
                                       target_flux_ij: np.ndarray,
                                       disp: bool = False):
    # evaluate objective functions
    # f_sigma = (flux_target-flux_h)^2/flux_target^2

    # stress difference potential: actual_stress_ij is homogenized stress
    # stress_difference_ij = actual_stress_ij - target_stress_ij
    stress_difference_ij = target_flux_ij - actual_flux_ij

    f_sigma = np.sum(stress_difference_ij ** 2) / np.sum(target_flux_ij ** 2)
    if disp and MPI.COMM_WORLD.rank == 0:
        print('f_sigma = '          ' {} '.format(f_sigma))  # good in MPI
    return f_sigma


def adjoint_potential(discretization,
                      flux_field_ijqxyz,
                      adjoint_field_inxyz):
    # g = (grad lambda, flux)
    # g = (grad lambda, C grad displacement)
    # g = (grad lambda, C grad displacement)  == lambda_transpose grad_transpose C grad u

    # Input: adjoint_field [f,n,x,y,z]
    #        stress_field  [d,d,q,x,y,z]

    # Output: g  [1] == 0
    # -- -- -- -- -- -- -- -- -- -- --
    # apply quadrature weights

    weights = discretization.quadrature_weights
    # apply B^transposed via the convolution operator "div (flux)"
    force_field_inxyz = discretization.get_scalar_field(
        name='temporary_field_inxyz_in_adjoint_potential_temporary')
    discretization.fft.communicate_ghosts(force_field_inxyz)

    discretization.conv_op.transpose(quadrature_point_field=flux_field_ijqxyz,
                                     nodal_field=force_field_inxyz,
                                     weights=weights)

    adjoint_potential_field = np.einsum('ui...,ui...->...', adjoint_field_inxyz.s, force_field_inxyz.s)

    # Reductor_numpi = discretization.mpi_reduction(MPI.COMM_WORLD)
    integral = discretization.mpi_reduction.sum(adjoint_potential_field)  #

    return integral


def sensitivity_flux_and_adjoint(discretization,
                                 base_material_data_ijkl,
                                 void_material_data_ijkl,
                                 displacement_field_inxyz,
                                 adjoint_field_inxyz,
                                 macro_gradient_field_ijqxyz,
                                 phase_field_1nxyz,
                                 target_flux_ij,
                                 actual_flux_ij,
                                 preconditioner_fun,
                                 system_matrix_fun, p,
                                 weight,
                                 formulation=None,
                                 disp=False,
                                 **kwargs):
    cg_tol = kwargs.get('cg_tol', 1e-7)
    r_tol = kwargs.get('r_tol', True)
    # Input:
    #        material_data_field_ijklqxyz [d,d,d,d,q,x,y,z] - elasticity tensors without applied phase field -- C_0
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
    dim = discretization.domain_dimension
    # -----    stress difference potential ----- #
    # Gradient of material data with respect to phase field
    phase_field_at_quad_poits_1qxyz = discretization.get_quad_field_scalar(
        name='phase_field_at_quads_reusable')
    discretization.apply_N_operator_mugrid(phase_field_1nxyz, phase_field_at_quad_poits_1qxyz)

    material_data_field_rho_ijklqxyz = discretization.get_material_data_size_field_mugrid(
        name='data_field_in_sensitivity_sensitivity_stress_and_adjoint_FE_NEW')

    # dim = 2 or 3 (number of spatial dimensions)
    # quad axis (q) + spatial axes (x, y[, z]) -> dim + 1 trailing axes
    expand = (...,) + (np.newaxis,) * (dim + 1)

    material_data_field_rho_ijklqxyz.s[...] = \
        (base_material_data_ijkl - void_material_data_ijkl)[expand] \
        * np.power(phase_field_at_quad_poits_1qxyz.s, p)[0, 0, :, ...] \
        + void_material_data_ijkl[expand]

    # d_stress_d_rho phase field gradient potential for a phase field without perturbation
    dstress_drho = partial_derivative_of_objective_function_flux_equivalence_wrt_phase_field(
        discretization=discretization,
        phase_field_1nxyz=phase_field_1nxyz,
        target_flux_ij=target_flux_ij,
        actual_flux_ij=actual_flux_ij,
        base_material_data_ij=base_material_data_ijkl,
        void_material_data_ij=void_material_data_ijkl,
        temperature_field_inxyz=displacement_field_inxyz,
        macro_gradient_field_ijqxyz=macro_gradient_field_ijqxyz,
        p=p)

    # Adjoint problem
    # compute strain field from to displacement and macro gradient
    # strain_fluctuation_ijqxyz = discretization.apply_gradient_operator_symmetrized(displacement_field_fnxyz)
    # stress_difference_ij = target_stress_ij - actual_stress_ij
    flux_difference_ij = target_flux_ij - actual_flux_ij

    flux_difference_ijqxyz = discretization.get_gradient_size_field(
        name='flux_field_in_sensitivity_flux_and_adjoint')
    discretization.get_macro_gradient_field_mugrid(macro_gradient_ij=flux_difference_ij,
                                                   macro_gradient_field_ijqxyz=flux_difference_ijqxyz
                                                   )
    # minus sign is already there
    df_du_field = discretization.get_unknown_size_field(
        name='adjoint_problem_rhs_in_sensitivity_flux_and_adjoint')
    discretization.get_rhs_mugrid(material_data_field_ijklqxyz=material_data_field_rho_ijklqxyz,
                                  macro_gradient_field_ijqxyz=flux_difference_ijqxyz,
                                  rhs_inxyz=df_du_field)
    # minus sign is already there
    df_du_field.s[...] = -2 * df_du_field.s / discretization.cell.domain_volume
    # Normalization
    df_du_field.s[...] = weight * df_du_field.s / np.sum(target_flux_ij ** 2)

    info_adjoint_ = {}
    # if MPI.COMM_WORLD.rank == 0:
    norms_cg_adjoint = dict()
    norms_cg_adjoint['residual_rr'] = []
    norms_cg_adjoint['residual_rz'] = []

    def callback_adjoint(it, x, r, p, z, stop_crit_norm):
        # global norms_cg_mech
        norm_of_rr = discretization.communicator.sum(np.dot(r.ravel(), r.ravel()))
        norm_of_rz = discretization.communicator.sum(np.dot(r.ravel(), z.ravel()))
        if MPI.COMM_WORLD.rank == 0:
            norms_cg_adjoint['residual_rr'].append(norm_of_rr)
            norms_cg_adjoint['residual_rz'].append(norm_of_rz)

    solvers.conjugate_gradients_mugrid(
        comm=discretization.communicator,
        fc=discretization.field_collection,
        hessp=system_matrix_fun,  # linear operator
        b=df_du_field,
        x=adjoint_field_inxyz,
        P=preconditioner_fun,
        tol=cg_tol,
        rtol=r_tol,
        maxiter=int(10000),
        callback=callback_adjoint,
        # norm_metric=res_norm
    )

    info_adjoint_['residual_rz'] = norms_cg_adjoint['residual_rz']
    if MPI.COMM_WORLD.rank == 0:
        nb_it = len(norms_cg_adjoint['residual_rr'])
        info_adjoint_['num_iteration_adjoint'] = nb_it

        del norms_cg_adjoint
        gc.collect()

        if disp:
            print(f" nb_steps CG adjoint = {nb_it}, residual_rz = {info_adjoint_['residual_rz']}")
    dadjoin_drho = discretization.get_scalar_field(name='dadjoin_drho_in_sensitivity_stress_and_adjoint_FE_NEW')
    dadjoin_drho = partial_derivative_of_adjoint_potential_wrt_phase_field(
        discretization=discretization,
        base_material_data_ijkl=base_material_data_ijkl,
        void_material_data_ijkl=void_material_data_ijkl,
        displacement_field_fnxyz=displacement_field_inxyz,
        macro_gradient_field_ijqxyz=macro_gradient_field_ijqxyz,
        phase_field_1nxyz=phase_field_1nxyz,
        adjoint_field_inxyz=adjoint_field_inxyz,
        output_field_inxyz=dadjoin_drho,
        p=p)

    flux_field_ijqxyz = discretization.get_gradient_size_field(
        name='flux_field_in_sensitivity_flux_and_adjoint_FE_NEW')

    discretization.get_flux_field_mugrid(
        material_data_field_ijqxyz=material_data_field_rho_ijklqxyz,
        temperature_field_inxyz=displacement_field_inxyz,
        macro_gradient_field_ijqxyz=macro_gradient_field_ijqxyz,
        output_flux_field_ijqxyz=flux_field_ijqxyz)

    adjoint_energy = adjoint_potential(
        discretization=discretization,
        flux_field_ijqxyz=flux_field_ijqxyz,
        adjoint_field_inxyz=adjoint_field_inxyz)

    return weight * dstress_drho.s + dadjoin_drho.s, adjoint_field_inxyz, adjoint_energy, info_adjoint_


def partial_derivative_of_objective_function_flux_equivalence_wrt_phase_field(discretization,
                                                                              base_material_data_ij: np.ndarray,
                                                                              void_material_data_ij: np.ndarray,
                                                                              temperature_field_inxyz,
                                                                              macro_gradient_field_ijqxyz,
                                                                              phase_field_1nxyz,
                                                                              target_flux_ij: np.ndarray,
                                                                              actual_flux_ij: np.ndarray,
                                                                              p: int):
    # Input: phase_field [1,n,x,y,z]
    #        material_data_field [d,d,d,d,q,x,y,z] - elasticity
    #        target_stress_ij [d,d]
    #        actual_stress_ij [d,d]
    # -- -- -- -- -- -- -- -- -- -- --
    dim = discretization.domain_dimension
    # Gradient of material data with respect to phase field
    # % interpolation of rho into quad points
    # -----    stress difference potential ----- #
    # Gradient of material data with respect to phase field
    phase_field_at_quad_poits_1qxyz = discretization.get_quad_field_scalar(
        name='phase_field_at_quads_reusable')
    discretization.apply_N_operator_mugrid(phase_field_1nxyz, phase_field_at_quad_poits_1qxyz)

    dmaterial_data_field_drho_ijklqxyz_FE = discretization.get_material_data_size_field_mugrid(
        name='data_field_drho_partial_derivative_of_objective_function_stress_equivalence_wrt_phase_field_FE')

    # I consider polynomial interpolation of material
    # C_ij= rho**(p) (C_base_ij-C_void_ij) + C_void_ij
    # ∂ C_ijkl/ ∂ rho = p*rho**(p-1) (C_base_ij-C_void_ij)
    expand = (...,) + (np.newaxis,) * (dim + 1)
    dmaterial_data_field_drho_ijklqxyz_FE.s[...] = \
        (base_material_data_ij - void_material_data_ij)[expand] \
        * (p * np.power(phase_field_at_quad_poits_1qxyz.s, p - 1))[0, 0, :, ...]

    # compute gradient of temperature field from to temperature and macro gradient
    flux_ijqxyz = discretization.get_gradient_size_field(name='temperature_ijqxyz_local_at_pdofsewpf')
    discretization.apply_gradient_operator_mugrid(u_inxyz=temperature_field_inxyz,
                                                  grad_u_ijqxyz=flux_ijqxyz)
    flux_ijqxyz.s[...] = macro_gradient_field_ijqxyz.s + flux_ijqxyz.s

    # compute heat flux field
    flux_ijqxyz.s[...] = np.einsum('ij...,uj...->ui...', dmaterial_data_field_drho_ijklqxyz_FE.s,
                                   flux_ijqxyz.s)
    # this is actually flux

    # flux difference
    flux_difference_ij = target_flux_ij - actual_flux_ij

    double_contraction_flux_qxyz_FE = discretization.get_quad_field_scalar(name='temp_at_quads')
    double_contraction_flux_qxyz_FE.s[0, 0] = np.einsum('uj,ujqxy...->qxy...',
                                                        flux_difference_ij,
                                                        flux_ijqxyz.s)
    dfflux_drho = discretization.get_scalar_field(name='dfflux_drho_output')
    discretization.apply_N_transposed_operator_mugrid(
        quad_field_ijqxyz=double_contraction_flux_qxyz_FE,
        nodal_field_inxyz=dfflux_drho,
        apply_weights=True)
    dfflux_drho.s[...] = -2 * dfflux_drho.s / discretization.cell.domain_volume / np.sum(target_flux_ij ** 2)
    return dfflux_drho


def partial_derivative_of_adjoint_potential_wrt_phase_field(discretization,
                                                            base_material_data_ijkl,
                                                            void_material_data_ijkl,
                                                            displacement_field_fnxyz,
                                                            macro_gradient_field_ijqxyz,
                                                            phase_field_1nxyz,
                                                            adjoint_field_inxyz,
                                                            output_field_inxyz,
                                                            p=1
                                                            ):
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
    dim = discretization.domain_dimension
    # Gradient of material data with respect to phasse field   % interpolation of rho into quad points
    # I consider linear interpolation of material  C_ijkl= p*rho**(p-1) C^0_ijkl
    # so  ∂ C_ijkl/ ∂ rho = 1* C^0_ijkl
    # Gradient of material data with respect to phase field
    phase_field_at_quad_poits_1qxyz = discretization.get_quad_field_scalar(
        name='phase_field_at_quads_reusable')
    discretization.apply_N_operator_mugrid(phase_field_1nxyz, phase_field_at_quad_poits_1qxyz)

    dmaterial_data_field_drho_ijklqxyz = discretization.get_material_data_size_field_mugrid(
        name='data_field_in_sensitivity_reusable')

    # I consider polynomial interpolation of material
    # C_ij= rho**(p) (C_base_ij-C_void_ij) + C_void_ij
    # ∂ C_ijkl/ ∂ rho = p*rho**(p-1) (C_base_ij-C_void_ij)
    expand = (...,) + (np.newaxis,) * (dim + 1)

    dmaterial_data_field_drho_ijklqxyz.s[...] = \
        (base_material_data_ijkl - void_material_data_ijkl)[expand] \
        * (p * np.power(phase_field_at_quad_poits_1qxyz.s[0, 0, :, ...], p - 1))

    # compute strain field from to displacement and macro gradient
    flux_ijqxyz = discretization.get_gradient_size_field(name='flux_ijqxyz_local_at_pdapwpf')

    discretization.get_flux_field_mugrid(
        material_data_field_ijqxyz=dmaterial_data_field_drho_ijklqxyz,
        temperature_field_inxyz=displacement_field_fnxyz,
        macro_gradient_field_ijqxyz=macro_gradient_field_ijqxyz,
        output_flux_field_ijqxyz=flux_ijqxyz)

    # local_stress=np.einsum('ijkl,lk->ij',material_data_field_ijklqxyz[...,0,0,0] , strain_ijqxyz[...,0,0,0])
    # apply quadrature weights
    # discretization.apply_quadrature_weights_on_gradient_field_mugrid(strain_ijqxyz)

    # gradient of adjoint_field
    adjoint_field_gradient_ijqxyz = discretization.get_gradient_of_scalar_field(
        name='adjoint_field_gradient_ijqxyz__pdapwpf')
    discretization.apply_gradient_operator_mugrid(u_inxyz=adjoint_field_inxyz,
                                                  grad_u_ijqxyz=adjoint_field_gradient_ijqxyz)

    double_contraction_stress_qxyz = discretization.get_quad_field_scalar(name='double_contraction_stress_qxyz')
    double_contraction_stress_qxyz.s[0, 0] = np.einsum('ij...,ij...->...',
                                                       adjoint_field_gradient_ijqxyz.s,
                                                       flux_ijqxyz.s)

    output_field_inxyz.s.fill(0)
    discretization.fft.communicate_ghosts(double_contraction_stress_qxyz)

    discretization.apply_N_transposed_operator_mugrid(
        quad_field_ijqxyz=double_contraction_stress_qxyz,
        nodal_field_inxyz=output_field_inxyz,
        apply_weights=True)

    return output_field_inxyz
