"""
Regression test: the adjoint solve of the sensitivity functions must not depend on the warm start.

Until 2026-10-10 ``sensitivity_stress_and_adjoint_FE_NEW`` (and the conductivity variant) passed ``rtol=r_tol`` to
CG, i.e. a tolerance relative to the residual of the initial guess. The adjoint is warm started; when the initial
guess was already the solution (objective evaluated twice at one design, or at the tiny steps of a backtracking
line search), that residual was round-off, the relative criterion could not be met, and CG returned a wrong adjoint
or raised "Hessian is not positive definite". Gradients off by an order of magnitude stalled L-BFGS.
The tolerance is now relative to the right-hand side.
"""
import numpy as np
import pytest

from muFFTTO import domain, material_models, solvers, topology_optimization


def setup_problem(n=16, p=3, contrast=1e-5, seed=3):
    cell = domain.PeriodicUnitCell(domain_size=[1.0, 1.0], problem_type='elasticity')
    discretization = domain.Discretization(cell=cell, nb_of_pixels_global=(n, n),
                                           discretization_type='finite_element', element_type='linear_triangles')
    C_solid = material_models.get_elastic_material_tensor(dim=2, K=1.0, mu=0.5, kind='linear')
    C_void = contrast * C_solid
    # high-contrast design: mostly 0/1 with a grey band (as in a projected filter design)
    rng = np.random.default_rng(seed)
    x = np.arange(n) / n
    X, Y = np.meshgrid(x, x, indexing='ij')
    rho = discretization.get_scalar_field(name='rho')
    smooth = np.sin(2 * np.pi * X) * np.cos(2 * np.pi * Y) + 0.2 * rng.standard_normal((n, n))
    rho.s[0, 0] = np.clip(0.5 + 2.0 * smooth, 0.0, 1.0)
    rho_quad = discretization.get_quad_field_scalar(name='rho_quad')
    discretization.apply_N_operator_mugrid(rho, rho_quad)
    C_field = discretization.get_material_data_size_field_mugrid(name='C_rho')
    C_field.s[...] = ((C_solid - C_void)[..., np.newaxis, np.newaxis, np.newaxis]
                      * np.power(rho_quad.s, p)[0, 0] + C_void[..., np.newaxis, np.newaxis, np.newaxis])
    green = discretization.get_preconditioner_Green_mugrid(reference_material_data_ijkl=C_solid)

    def K_fun(x, Ax):
        discretization.apply_system_matrix_mugrid(material_data_field=C_field, input_field_inxyz=x,
                                                  output_field_inxyz=Ax, formulation='small_strain')

    def M_fun(x, Px):
        discretization.fft.communicate_ghosts(x)
        discretization.apply_preconditioner_mugrid(preconditioner_Fourier_fnfnqks=green, input_nodal_field_fnxyz=x,
                                                   output_nodal_field_fnxyz=Px)

    E = discretization.get_gradient_size_field(name='E')
    discretization.get_macro_gradient_field_mugrid(macro_gradient_ij=np.array([[0.0, 0.025], [0.025, 0.0]]),
                                                   macro_gradient_field_ijqxyz=E)
    rhs = discretization.get_unknown_size_field(name='rhs')
    discretization.get_rhs_mugrid(material_data_field_ijklqxyz=C_field, macro_gradient_field_ijqxyz=E, rhs_inxyz=rhs)
    u = discretization.get_unknown_size_field(name='u')
    solvers.conjugate_gradients_mugrid(comm=discretization.communicator, fc=discretization.field_collection,
                                       hessp=K_fun, b=rhs, x=u, P=M_fun, tol=1e-12, maxiter=20000)
    stress = discretization.get_homogenized_stress_mugrid(material_data_field_ijklqxyz=C_field,
                                                          displacement_field_inxyz=u, macro_gradient_field_ijqxyz=E,
                                                          formulation='small_strain')
    target = 3.0 * stress + np.array([[0.0, 1e-3], [1e-3, 0.0]])
    return dict(discretization=discretization, C_solid=C_solid, C_void=C_void, rho=rho, u=u, E=E, stress=stress,
                target=target, K_fun=K_fun, M_fun=M_fun, p=p)


def sensitivity(problem, adjoint, cg_tol, r_tol=True):
    s, adjoint, _, _ = topology_optimization.sensitivity_stress_and_adjoint_FE_NEW(
        discretization=problem['discretization'], base_material_data_ijkl=problem['C_solid'],
        void_material_data_ijkl=problem['C_void'], displacement_field_inxyz=problem['u'], adjoint_field_inxyz=adjoint,
        macro_gradient_field_ijqxyz=problem['E'], phase_field_1nxyz=problem['rho'],
        target_stress_ij=problem['target'], actual_stress_ij=problem['stress'],
        preconditioner_fun=problem['M_fun'], system_matrix_fun=problem['K_fun'], formulation='small_strain',
        p=problem['p'], weight=10.0, cg_tol=cg_tol, r_tol=r_tol)
    return np.array(s, copy=True), adjoint


@pytest.mark.parametrize('cg_tol', [1e-8, 1e-10, 1e-12])
def test_adjoint_warm_start_at_the_solution(cg_tol):
    """Re-evaluating at the same design (warm start = solution) must give the same sensitivity."""
    problem = setup_problem()
    adjoint = problem['discretization'].get_unknown_size_field(name='adjoint')
    adjoint.s[...] = 0.0
    s_cold, adjoint = sensitivity(problem, adjoint, cg_tol)
    s_warm, adjoint = sensitivity(problem, adjoint, cg_tol)          # warm start: the solution itself
    s_warm_again, adjoint = sensitivity(problem, adjoint, cg_tol)
    scale = np.abs(s_cold).max()
    assert np.abs(s_warm - s_cold).max() <= 10 * cg_tol * scale + 1e-12 * scale
    assert np.abs(s_warm_again - s_cold).max() <= 10 * cg_tol * scale + 1e-12 * scale


def test_adjoint_warm_start_near_the_solution():
    """A warm start that is close to the solution (tiny line-search step) must not change the sensitivity."""
    problem = setup_problem()
    discretization = problem['discretization']
    adjoint = discretization.get_unknown_size_field(name='adjoint')
    adjoint.s[...] = 0.0
    s_cold, adjoint = sensitivity(problem, adjoint, 1e-11)
    rng = np.random.default_rng(0)
    adjoint.s[...] *= 1.0 + 1e-10 * rng.standard_normal(adjoint.s.shape)
    s_warm, adjoint = sensitivity(problem, adjoint, 1e-11)
    assert np.abs(s_warm - s_cold).max() <= 1e-8 * np.abs(s_cold).max()


def test_relative_tolerance_is_relative_to_rhs():
    """r_tol=True: the cold and a far-off warm start reach the same accuracy (tolerance relative to |b|)."""
    problem = setup_problem()
    discretization = problem['discretization']
    adjoint = discretization.get_unknown_size_field(name='adjoint')
    adjoint.s[...] = 0.0
    s_reference, _ = sensitivity(problem, adjoint, 1e-12)
    for start in (0.0, 1e3):
        adjoint.s[...] = start
        s, _ = sensitivity(problem, adjoint, 1e-7)
        assert np.abs(s - s_reference).max() <= 1e-5 * np.abs(s_reference).max()
