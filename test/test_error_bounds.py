"""
Tests of :mod:`muFFTTO.error_bounds` and of the total-error stopping rule of the PCG solvers.

The reference is the code of the paper "Certified homogenized coefficients in FFT-based homogenization"
(repository _error_estimates_for_fft_based_homogenization, simulations_scripts/paper_energy_norm/common). The
tests that compare with it are skipped if that repository is not available. The problem is the paper's P1: a
square inclusion (volume fraction 1/4) with A_mat = I, A_inc = 10^rho I, reference material I, linear triangles,
macroscopic gradient E = e_1, for which Obnosov's formula gives the exact effective conductivity.
"""
import os
import sys

import numpy as np
import pytest

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from muFFTTO import domain, microstructure_library, solvers
from muFFTTO.error_bounds import DelayedEnergyErrorEstimate, TotalErrorStoppingRule, TwoSidedBounds2D

PAPER_REPOSITORY = os.path.expanduser('~/Projects/_error_estimates_for_fft_based_homogenization')
E = np.array([1.0, 0.0])


def paper_code():
    """The paper's two_sided module, or skip."""
    if not os.path.isdir(PAPER_REPOSITORY):
        pytest.skip('repository of the paper not available')
    if PAPER_REPOSITORY not in sys.path:
        sys.path.insert(0, PAPER_REPOSITORY)
    from simulations_scripts.paper_energy_norm.common import two_sided
    return two_sided


class SquareInclusion:
    """The paper's problem P1 on an n x n grid, with the operators of a PCG solve."""

    def __init__(self, n, rho):
        cell = domain.PeriodicUnitCell(domain_size=[1, 1], problem_type='conductivity')
        self.discretization = domain.Discretization(cell=cell, nb_of_pixels_global=(n, n),
                                                    discretization_type='finite_element',
                                                    element_type='linear_triangles')
        d = self.discretization
        self.material = d.get_material_data_size_field_mugrid(name='material_data')
        phase = microstructure_library.get_geometry(nb_voxels=d.nb_of_pixels,
                                                    microstructure_name='square_inclusion',
                                                    coordinates=d.fft.coords)
        self.material.s[..., phase > 0] = np.eye(2)[..., np.newaxis, np.newaxis]
        self.material.s[..., phase == 0] = 10.0 ** rho * np.eye(2)[..., np.newaxis, np.newaxis]
        inclusion = 10.0 ** rho
        self.A_exact = np.sqrt((1 + 3 * inclusion) / (3 + inclusion))       # Obnosov, matrix conductivity 1

        self.E_field = d.get_gradient_size_field(name='macro_gradient_field')
        d.get_macro_gradient_field_mugrid(macro_gradient_ij=E, macro_gradient_field_ijqxyz=self.E_field)
        self.rhs = d.get_unknown_size_field(name='rhs_field')
        d.get_rhs_mugrid(material_data_field_ijklqxyz=self.material, macro_gradient_field_ijqxyz=self.E_field,
                         rhs_inxyz=self.rhs)
        self.green = np.array(d.get_preconditioner_Green_mugrid(reference_material_data_ijkl=np.eye(2)).s)

    def K(self, x, Ax):
        self.discretization.apply_system_matrix_mugrid(material_data_field=self.material, input_field_inxyz=x,
                                                       output_field_inxyz=Ax)
        self.discretization.fft.communicate_ghosts(Ax)

    def M(self, x, Px):
        self.discretization.fft.communicate_ghosts(x)
        green = type('Field', (), {'s': self.green})()
        self.discretization.apply_preconditioner_mugrid(preconditioner_Fourier_fnfnqks=green,
                                                        input_nodal_field_fnxyz=x, output_nodal_field_fnxyz=Px)

    def field(self, name, values=None):
        field = self.discretization.get_unknown_size_field(name=name)
        field.s[...] = 0.0 if values is None else values
        return field

    def solve(self, solver=solvers.conjugate_gradients_mugrid, name='solution', **options):
        """PCG from u_0 = 0; returns the solution field and the solver output."""
        u = self.field(name)
        out = solver(comm=self.discretization.communicator, fc=self.discretization.field_collection,
                     hessp=self.K, b=self.rhs, x=u, P=self.M, **options)
        return u, out

    def iterates(self, maxiter):
        """Every PCG iterate u_0 ... u_m and the step coefficients (alpha_k, (r_k, z_k)), m <= maxiter. The run stops
        at a relative residual of 1e-12: beyond round-off the steps vanish and the delayed estimate is undefined."""
        snapshots, steps, previous = [], [], {}

        def callback(iteration, x, r, p, z, stop_crit):
            # at iteration k the solver passes x_k, r_k, z_k and p_{k-1}, the direction of the step just taken:
            # x_k = x_{k-1} + alpha_{k-1} p_{k-1}; (r_{k-1}, z_{k-1}) was stored at the previous call
            if iteration > 0:
                direction = p.ravel()
                alpha = float((x - snapshots[-1]).ravel() @ direction / (direction @ direction))
                steps.append((alpha, previous['rz']))
            snapshots.append(np.array(x, copy=True))
            previous['rz'] = float(np.dot(r.ravel(), z.ravel()))

        self.solve(name='iterates', tol=1e-12, rtol=True, maxiter=maxiter, callback=callback)
        return snapshots, steps


def spread(snapshots, count=7):
    """Indices of `count` iterates spread over the run, first and last included."""
    return np.unique(np.linspace(0, len(snapshots) - 1, count).astype(int))


@pytest.fixture(params=[3, -3], ids=['stiff inclusion', 'soft inclusion'])
def problem(request):
    return SquareInclusion(32, request.param)


def test_two_sided_bounds_match_paper(problem):
    two_sided = paper_code()
    paper = two_sided.TwoSidedBounds(problem.discretization, problem.material, problem.E_field, E)
    ours = TwoSidedBounds2D(problem.discretization, problem.material, problem.E_field, E)
    snapshots, _ = problem.iterates(maxiter=40)
    u = problem.field('bounds_iterate')
    for k in spread(snapshots):
        u.s[...] = snapshots[k]
        upper_paper, _, lower_paper_optimized = paper(u)
        upper, lower = ours(u)
        assert upper == pytest.approx(upper_paper, rel=1e-13)
        assert lower == pytest.approx(lower_paper_optimized, rel=1e-12, abs=1e-14)


def test_two_sided_bounds_bracket_the_exact_coefficient(problem):
    bounds = TwoSidedBounds2D(problem.discretization, problem.material, problem.E_field, E)
    snapshots, _ = problem.iterates(maxiter=60)
    u = problem.field('bracket_iterate')
    for k in spread(snapshots):
        u.s[...] = snapshots[k]
        upper, lower = bounds(u)
        assert max(lower, bounds.reuss) <= problem.A_exact <= upper


def test_delayed_estimate_matches_paper_and_solver(problem):
    two_sided = paper_code()
    _, steps = problem.iterates(maxiter=60)
    ours = DelayedEnergyErrorEstimate()
    paper = two_sided.DelayedEstimate()
    for alpha, rz in steps:
        assert ours.update(alpha, rz) == pytest.approx(paper.update(alpha, rz), rel=1e-12)
    _, (_, norms) = problem.solve(solver=solvers.conjugate_gradients_mugrid_experimental, name='experimental',
                                  tol=1e-12, rtol=True, maxiter=60, stop_crit_norm='rr')
    estimates = [value for _, value in ours.estimates]
    assert estimates == pytest.approx(norms['energy_lower_estim'][:len(estimates)], rel=1e-10)


def eager_stop_iteration(problem, theta, maxiter):
    """The rule replayed on the recorded iterates of a separate PCG run."""
    bounds = TwoSidedBounds2D(problem.discretization, problem.material, problem.E_field, E)
    estimate = DelayedEnergyErrorEstimate()
    snapshots, steps = problem.iterates(maxiter)
    u = problem.field('eager_iterate')
    best_lower, previous_best_lower = bounds.reuss, None
    for k in range(1, len(snapshots)):
        estimate.update(*steps[k - 1])
        u.s[...] = snapshots[k]
        upper, lower = bounds(u)
        best_lower = max(best_lower, lower)
        width = upper - best_lower
        algebraic = estimate.latest
        if (algebraic is not None and algebraic <= theta * width and previous_best_lower is not None
                and abs(best_lower - previous_best_lower) <= theta * width):
            return k
        previous_best_lower = best_lower
    return None


def test_total_error_stop(problem):
    theta = 0.1
    bounds = TwoSidedBounds2D(problem.discretization, problem.material, problem.E_field, E)
    rule = TotalErrorStoppingRule(bounds, theta=theta)
    u, _ = problem.solve(name='total_error_stop', stop_crit_norm='total_error', stopping_rule=rule, maxiter=500)
    stop = rule.history['iteration'][-1]
    upper, lower = rule.history['upper'][-1], rule.best_lower

    # the rule inside the solver stops where the rule replayed on the recorded iterates stops
    assert stop == eager_stop_iteration(problem, theta, maxiter=stop + 5)
    # the certified interval contains the exact coefficient
    assert lower <= problem.A_exact <= upper
    # the true algebraic error at the stop, U_k - A_h (A_h from a tight reference solve), is below theta (U - L*)
    reference, _ = problem.solve(name='reference', tol=1e-14, rtol=True, maxiter=5000)
    A_h, _ = bounds(reference)
    assert upper - A_h <= theta * (upper - lower)
    # the experimental solver stops at the same iterate
    rule_experimental = TotalErrorStoppingRule(bounds, theta=theta)
    problem.solve(solver=solvers.conjugate_gradients_mugrid_experimental, name='total_error_experimental',
                  stop_crit_norm='total_error', stopping_rule=rule_experimental, maxiter=500)
    assert rule_experimental.history['iteration'][-1] == stop


def test_total_error_stop_requires_rule(problem):
    with pytest.raises(ValueError):
        problem.solve(name='no_rule', stop_crit_norm='total_error')


def test_stop_matches_paper_rule(problem):
    """With the PT upper estimate (alg='estimate' in the paper's code) and without dual steps, the rule stops at
    the same iteration as the paper's AdaptivePrimalDual."""
    two_sided = paper_code()
    paper_bounds = two_sided.TwoSidedBounds(problem.discretization, problem.material, problem.E_field, E)
    spectrum_lower_bound = min(1.0, float(problem.material.s[0, 0].min()))     # C_ref = I, isotropic phases
    paper_rule = two_sided.AdaptivePrimalDual(paper_bounds, spectrum_lower_bound, theta=0.1, max_burst=0,
                                              alg='estimate')
    iterate = problem.field('paper_rule_iterate')

    def drive_paper_rule(iteration, x, r, p, z, stop_crit):
        iterate.s[...] = x
        upper, _, lower_optimized = paper_bounds(iterate)
        paper_rule(iteration, iterate, x, p, float(np.dot(r.ravel(), z.ravel())), upper, lower_optimized)

    ours = TotalErrorStoppingRule(TwoSidedBounds2D(problem.discretization, problem.material, problem.E_field, E),
                                  theta=0.1, algebraic_estimate='upper')
    problem.solve(name='paper_rule', stop_crit_norm='total_error', stopping_rule=ours, callback=drive_paper_rule,
                  maxiter=500)
    assert paper_rule.stop == ours.history['iteration'][-1]
