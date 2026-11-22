import unittest
import warnings

import numpy as np
import muGrid
from mpi4py import MPI

from muFFTTO import solvers
from muFFTTO import domain


def _make_fc(nb_pixels=(4, 5), nb_nodal_pts=1):
    """Create a fresh FFTEngine and field collection for a test."""
    fft = muGrid.FFTEngine(
        nb_domain_grid_pts=nb_pixels,
        communicator=muGrid.Communicator(MPI.COMM_WORLD),
    )
    fc = fft.real_space_collection
    fc.set_nb_sub_pts('nodal_points', nb_nodal_pts)
    comm = muGrid.Communicator(MPI.COMM_WORLD)
    return fft, fc, comm  # fft must stay alive as long as fc is used


def _make_domain(nb_pixels=(4, 5), nb_nodal_pts=1):
    """Create a domain object."""
    domain_size = [1, 1]
    problem_type = 'conductivity'
    discretization_type = 'finite_element'
    element_type = 'linear_triangles'
    my_cell = domain.PeriodicUnitCell(domain_size=domain_size,
                                      problem_type=problem_type)
    discretization = domain.Discretization(cell=my_cell,
                                           nb_of_pixels_global=nb_pixels,
                                           discretization_type=discretization_type,
                                           element_type=element_type)
    comm = discretization.communicator
    return discretization


class TestConjugateGradients(unittest.TestCase):
    """Tests for solvers.conjugate_gradients (energy-error stopping test)."""

    @staticmethod
    def _identity_P(r, z):
        z.s[...] = r.s

    def _fields(self, fc, nb_components=2):
        b = fc.real_field(name='b', components=(nb_components,), sub_pt='nodal_points')
        x = fc.real_field(name='x', components=(nb_components,), sub_pt='nodal_points')
        return b, x

    def test_identity_system_converges(self):
        """A = I, b = 1, x0 = 0 -> x = 1 (exact after one step)."""
        fft, fc, comm = _make_fc()
        b, x = self._fields(fc)
        b.s[...] = 1.0
        x.s[...] = 0.0

        def hessp(p, Ap):
            Ap.s[...] = p.s

        result, info = solvers.conjugate_gradients(comm, fc, hessp, b, x, self._identity_P, tol=1e-10)
        np.testing.assert_allclose(result.s, b.s, atol=1e-12)
        self.assertEqual(info['iterations'], 1)
        self.assertTrue(info['converged'])

    def test_scaled_identity_system(self):
        """A = scale I, b = scale -> x = 1."""
        fft, fc, comm = _make_fc()
        b, x = self._fields(fc)
        scale = 4.0
        b.s[...] = scale
        x.s[...] = 0.0

        def hessp(p, Ap):
            Ap.s[...] = scale * p.s

        result, info = solvers.conjugate_gradients(comm, fc, hessp, b, x, self._identity_P, tol=1e-10)
        np.testing.assert_allclose(result.s, np.ones(result.s.shape), atol=1e-12)

    def test_spatially_varying_diagonal_system(self):
        """A = diag(a(x)): the true energy error is below the tolerance (relative to the initial error)."""
        fft, fc, comm = _make_fc(nb_pixels=(16, 16))
        b, x = self._fields(fc)
        diag = fc.real_field(name='diag', components=(2,), sub_pt='nodal_points')
        rng = np.random.default_rng(42)
        diag.s[...] = rng.uniform(1.0, 1e3, diag.s.shape)
        b.s[...] = diag.s                           # b = A * ones
        x.s[...] = 0.0

        def hessp(p, Ap):
            Ap.s[...] = diag.s * p.s

        tol = 1e-6
        result, info = solvers.conjugate_gradients(comm, fc, hessp, b, x, self._identity_P, tol=tol, d_min=0,
                                                   maxiter=2000)
        error = result.s - 1.0
        energy_error = comm.sum(float(np.sum(diag.s * error ** 2)))
        initial_error = comm.sum(float(np.sum(diag.s)))      # ||x* - 0||_K^2
        self.assertTrue(info['converged'])
        self.assertLess(np.sqrt(energy_error / initial_error), 10 * tol)
        self.assertLessEqual(info['estimate'], tol ** 2 * info['estimator'].initial_error)

    def test_zero_initial_residual_returns_immediately(self):
        """x0 already the solution -> no iteration."""
        fft, fc, comm = _make_fc()
        b, x = self._fields(fc)
        b.s[...] = 1.0
        x.s[...] = 1.0
        calls = []

        def hessp(p, Ap):
            Ap.s[...] = p.s

        result, info = solvers.conjugate_gradients(comm, fc, hessp, b, x, self._identity_P,
                                                   callback=lambda *args: calls.append(args))
        self.assertEqual(calls, [])
        self.assertEqual(info['iterations'], 0)
        np.testing.assert_allclose(result.s, b.s)

    def test_exact_preconditioner_converges_immediately(self):
        """P = A^-1 -> one iteration."""
        fft, fc, comm = _make_fc()
        b, x = self._fields(fc)
        scale = 7.0
        b.s[...] = scale * 3.0
        x.s[...] = 0.0

        def hessp(p, Ap):
            Ap.s[...] = scale * p.s

        def P(r, z):
            z.s[...] = r.s / scale

        result, info = solvers.conjugate_gradients(comm, fc, hessp, b, x, P, tol=1e-10)
        np.testing.assert_allclose(result.s, b.s / scale, atol=1e-12)
        self.assertEqual(info['iterations'], 1)

    def test_callback_receives_iterations(self):
        """The callback gets (iteration, x, r, p, z, estimate) for the start and every iteration."""
        fft, fc, comm = _make_fc(nb_pixels=(8, 8))
        b, x = self._fields(fc)
        diag = fc.real_field(name='diag', components=(2,), sub_pt='nodal_points')
        diag.s[...] = np.random.default_rng(1).uniform(1.0, 10.0, diag.s.shape)
        b.s[...] = 1.0
        x.s[...] = 0.0
        calls = []

        def hessp(p, Ap):
            Ap.s[...] = diag.s * p.s

        def callback(iteration, x_s, r_s, p_s, z_s, estimate):
            calls.append((iteration, estimate))

        _, info = solvers.conjugate_gradients(comm, fc, hessp, b, x, self._identity_P, tol=1e-8, d_min=0,
                                              callback=callback)
        self.assertEqual([c[0] for c in calls], list(range(0, info['iterations'] + 1)))
        self.assertIsNone(calls[1][1])               # no accepted estimate after one step
        self.assertIsNotNone(calls[-1][1])

    def test_minimal_delay(self):
        """With d_min the first estimate (and so the stop) comes at iteration > d_min at the earliest."""
        fft, fc, comm = _make_fc(nb_pixels=(16, 16))
        b, x = self._fields(fc)
        diag = fc.real_field(name='diag', components=(2,), sub_pt='nodal_points')
        diag.s[...] = np.random.default_rng(3).uniform(1.0, 100.0, diag.s.shape)
        b.s[...] = 1.0
        x.s[...] = 0.0

        def hessp(p, Ap):
            Ap.s[...] = diag.s * p.s

        _, info = solvers.conjugate_gradients(comm, fc, hessp, b, x, self._identity_P, tol=1e-2, d_min=10)
        self.assertGreaterEqual(info['iterations'], 10)
        self.assertTrue(all(l <= info['iterations'] - 10 for l, _ in info['estimator'].estimates))

    def test_warm_start_is_relative_to_start_energy(self):
        """Warm start: the stop compares with tol^2 ||x_0||_K^2, so a start close to the solution stops early."""
        fft, fc, comm = _make_fc(nb_pixels=(16, 16))
        b, x = self._fields(fc)
        diag = fc.real_field(name='diag', components=(2,), sub_pt='nodal_points')
        diag.s[...] = np.random.default_rng(4).uniform(1.0, 100.0, diag.s.shape)
        b.s[...] = diag.s                            # x* = 1

        def hessp(p, Ap):
            Ap.s[...] = diag.s * p.s

        x.s[...] = 0.0
        _, cold = solvers.conjugate_gradients(comm, fc, hessp, b, x, self._identity_P, tol=1e-6, d_min=0)
        x.s[...] = 1.0 + 1e-5 * np.random.default_rng(5).standard_normal(x.s.shape)
        _, warm = solvers.conjugate_gradients(comm, fc, hessp, b, x, self._identity_P, tol=1e-6, d_min=0)
        self.assertLess(warm['iterations'], cold['iterations'])
        energy_error = comm.sum(float(np.sum(diag.s * (x.s - 1.0) ** 2)))
        energy = comm.sum(float(np.sum(diag.s)))
        self.assertLess(np.sqrt(energy_error / energy), 10 * 1e-6)

    def test_stopping_rule_replaces_energy_test(self):
        """A stopping_rule decides when to stop."""
        fft, fc, comm = _make_fc(nb_pixels=(16, 16))
        b, x = self._fields(fc)
        diag = fc.real_field(name='diag', components=(2,), sub_pt='nodal_points')
        diag.s[...] = np.random.default_rng(6).uniform(1.0, 100.0, diag.s.shape)
        b.s[...] = 1.0
        x.s[...] = 0.0

        def hessp(p, Ap):
            Ap.s[...] = diag.s * p.s

        _, info = solvers.conjugate_gradients(comm, fc, hessp, b, x, self._identity_P,
                                              stopping_rule=lambda iteration, x, alpha, rz: iteration == 3)
        self.assertEqual(info['iterations'], 3)

    def test_non_positive_definite_raises_runtime_error(self):
        """A = -I -> RuntimeError."""
        fft, fc, comm = _make_fc()
        b, x = self._fields(fc)
        b.s[...] = 1.0
        x.s[...] = 0.0

        def hessp(p, Ap):
            Ap.s[...] = -p.s

        with self.assertRaises(RuntimeError):
            solvers.conjugate_gradients(comm, fc, hessp, b, x, self._identity_P)

    def test_maxiter_exceeded_issues_warning(self):
        """maxiter = 0 with a nonzero residual -> RuntimeWarning, converged False."""
        fft, fc, comm = _make_fc()
        b, x = self._fields(fc)
        b.s[...] = 1.0
        x.s[...] = 0.0

        def hessp(p, Ap):
            Ap.s[...] = p.s

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            _, info = solvers.conjugate_gradients(comm, fc, hessp, b, x, self._identity_P, maxiter=0)
        self.assertFalse(info['converged'])
        if comm.rank == 0:
            self.assertTrue(any(issubclass(wi.category, RuntimeWarning) for wi in w))

    def test_3d_domain(self):
        """CG on a 3D grid."""
        fft, fc, comm = _make_fc(nb_pixels=(3, 3, 3))
        b, x = self._fields(fc, nb_components=3)
        b.s[...] = 2.0
        x.s[...] = 0.0

        def hessp(p, Ap):
            Ap.s[...] = p.s

        result, _ = solvers.conjugate_gradients(comm, fc, hessp, b, x, self._identity_P, tol=1e-10)
        np.testing.assert_allclose(result.s, b.s, atol=1e-12)

    def test_homogenization_problem_anal_solution(self):
        """Solver scalar homogenization problem with analytical solution."""
        nb_pixels = (64, 64)
        dim = len(nb_pixels)
        discretization = _make_domain(nb_pixels=nb_pixels)

        # create material data field
        mat_contrast = 1
        mat_contrast_2 = 1e2
        conductivity_C_1 = np.array([[1., 0], [0, 1.0]])

        material_data_field_C_0 = discretization.get_material_data_size_field_mugrid(name='conductivity_tensor')

        # populate the field with C_1 material
        material_data_field_C_0.s[...] = conductivity_C_1[:, :, np.newaxis, np.newaxis, np.newaxis]

        # material distribution
        # phase_field_geom = microstructure_library.get_geometry(nb_voxels=discretization.nb_of_pixels,
        #                                                        microstructure_name=geometry_ID,
        #                                                        coordinates=discretization.fft.coords)

        phase_field = discretization.get_scalar_field(name='phase_field')
        phase_field.s[...].fill(1)
        phase_field.s[0, 0, nb_pixels[0] // 4:3 * nb_pixels[0] // 4, nb_pixels[1] // 4:3 * nb_pixels[1] // 4] = 0
        matrix_mask = phase_field.s[0, 0] > 0
        inc_mask = phase_field.s[0, 0] == 0

        # apply material distribution

        material_data_field_C_0.s[..., matrix_mask] = mat_contrast_2 * material_data_field_C_0.s[..., matrix_mask]
        material_data_field_C_0.s[..., inc_mask] = mat_contrast * material_data_field_C_0.s[..., inc_mask]

        norm_calls = [0]

        def hessp(p, Ap):
            discretization.apply_system_matrix_mugrid(material_data_field=material_data_field_C_0,
                                                      input_field_inxyz=p,
                                                      output_field_inxyz=Ap)
            discretization.fft.communicate_ghosts(Ap)

        def P(r, z):
            z.s[...] = r.s

        solution_field = discretization.get_unknown_size_field(name='solution')
        macro_gradient_field = discretization.get_gradient_size_field(name='macro_gradient_field')
        rhs_field = discretization.get_unknown_size_field(name='rhs_field')

        homogenized_A_ij = np.zeros(np.array(2 * [dim, ]))
        for i in range(dim):
            # set macroscopic gradient
            macro_gradient = np.zeros([dim])
            macro_gradient[i] = 1
            macro_gradient_field.sg.fill(0)
            discretization.get_macro_gradient_field_mugrid(macro_gradient_ij=macro_gradient,
                                                           macro_gradient_field_ijqxyz=macro_gradient_field)
            discretization.fft.communicate_ghosts(field=macro_gradient_field)
            # Solve equilibrium
            rhs_field.sg.fill(0)
            discretization.get_rhs_mugrid(material_data_field_ijklqxyz=material_data_field_C_0,
                                          macro_gradient_field_ijqxyz=macro_gradient_field,
                                          rhs_inxyz=rhs_field)

            solvers.conjugate_gradients(comm=discretization.communicator,
                                        fc=discretization.field_collection,
                                        hessp=hessp,
                                        b=rhs_field,
                                        x=solution_field,
                                        P=P,
                                        tol=1e-10)
            homogenized_A_ij[i, :] = discretization.get_homogenized_stress_mugrid(
                material_data_field_ijklqxyz=material_data_field_C_0,
                displacement_field_inxyz=solution_field,
                macro_gradient_field_ijqxyz=macro_gradient_field)
        J_eff = mat_contrast_2 * np.sqrt((mat_contrast_2 + 3 * mat_contrast) / (3 * mat_contrast_2 + mat_contrast))
        # for coarse grid, the tolerance must be large. due to discretizatiion error

        np.testing.assert_allclose(homogenized_A_ij[0, 0], J_eff, atol=0,
                                   rtol=1e-2)
        np.testing.assert_allclose(homogenized_A_ij[1, 1], J_eff, atol=0,
                                   rtol=1e-2)
        self.assertGreater(homogenized_A_ij[0, 0] - J_eff, 0)


class TestDRPBCGMuGrid(unittest.TestCase):
    """Tests for solvers.dr_pbcg_mugrid (Algorithm 5, Meurant & Tichy 2026)."""

    def _make_block(self, fc, m, components=(2,), prefix='b'):
        """Create m named fields in fc for use as a block RHS or solution."""
        return [
            fc.real_field(name=f'{prefix}-{j}', components=components, sub_pt='nodal_points')
            for j in range(m)
        ]

    def test_single_rhs_identity(self):
        """m=1, A=I, b=1, x0=0 → solution x=1."""
        fft, fc, comm = _make_fc()
        b_list = self._make_block(fc, m=1, prefix='b')
        x_list = self._make_block(fc, m=1, prefix='x')
        b_list[0].s[...] = 1.0
        x_list[0].s[...] = 0.0

        def hessp(p, Ap):
            Ap.s[...] = p.s

        def P(r, z):
            z.s[...] = r.s

        result, norms = solvers.dr_pbcg_mugrid(comm, fc, hessp, b_list, x_list, P, tol=1e-10)
        np.testing.assert_allclose(result[0].s, b_list[0].s, atol=1e-8)

    def test_block_rhs_identity(self):
        """m=3, A=I, random linearly-independent b[j] → x[j] = b[j]."""
        fft, fc, comm = _make_fc()
        m = 3
        b_list = self._make_block(fc, m, prefix='b')
        x_list = self._make_block(fc, m, prefix='x')
        rng = np.random.default_rng(42)
        x_true = [rng.standard_normal(b_list[j].s.shape) for j in range(m)]
        for j in range(m):
            b_list[j].s[...] = x_true[j]  # b = A * x_true = I * x_true = x_true
            x_list[j].s[...] = 0.0

        def hessp(p, Ap):
            Ap.s[...] = p.s

        def P(r, z):
            z.s[...] = r.s

        result, norms = solvers.dr_pbcg_mugrid(comm, fc, hessp, b_list, x_list, P, tol=1e-10)
        for j in range(m):
            np.testing.assert_allclose(result[j].s, x_true[j], atol=1e-8,
                                       err_msg=f"RHS {j} incorrect")

    def test_block_rhs_scaled_system(self):
        """m=2, A=scale*I, random x_true → b[j]=scale*x_true[j], solution x[j]=x_true[j]."""
        fft, fc, comm = _make_fc()
        m = 2
        scale = 5.0
        b_list = self._make_block(fc, m, prefix='b')
        x_list = self._make_block(fc, m, prefix='x')
        rng = np.random.default_rng(7)
        x_true = [rng.standard_normal(b_list[j].s.shape) for j in range(m)]
        for j in range(m):
            b_list[j].s[...] = scale * x_true[j]  # b = A * x_true = scale * x_true
            x_list[j].s[...] = 0.0

        def hessp(p, Ap):
            Ap.s[...] = scale * p.s

        def P(r, z):
            z.s[...] = r.s

        result, norms = solvers.dr_pbcg_mugrid(comm, fc, hessp, b_list, x_list, P, tol=1e-10)
        for j in range(m):
            np.testing.assert_allclose(result[j].s, x_true[j], atol=1e-8,
                                       err_msg=f"RHS {j} incorrect")

    def test_matches_scalar_cg_for_single_rhs(self):
        """m=1 DR-PBCG should give the same solution as conjugate_gradients."""
        # CG solve
        fft_cg, fc_cg, comm_cg = _make_fc()
        b_cg = fc_cg.real_field(name='b', components=(2,), sub_pt='nodal_points')
        x_cg = fc_cg.real_field(name='x', components=(2,), sub_pt='nodal_points')
        scale = 3.0
        b_cg.s[...] = 6.0
        x_cg.s[...] = 0.0

        def hessp(p, Ap): Ap.s[...] = scale * p.s

        def P(r, z):      z.s[...] = r.s

        x_cg_result, _ = solvers.conjugate_gradients(
            comm_cg, fc_cg, hessp, b_cg, x_cg, P, tol=1e-12)

        # DR-PBCG solve (m=1)
        fft_dr, fc_dr, comm_dr = _make_fc()
        b_dr = fc_dr.real_field(name='b', components=(2,), sub_pt='nodal_points')
        x_dr = fc_dr.real_field(name='x', components=(2,), sub_pt='nodal_points')
        b_dr.s[...] = 6.0
        x_dr.s[...] = 0.0

        result, _ = solvers.dr_pbcg_mugrid(
            comm_dr, fc_dr, hessp, [b_dr], [x_dr], P, tol=1e-12)

        np.testing.assert_allclose(result[0].s, x_cg_result.s, atol=1e-8)

    def test_returns_convergence_history(self):
        """norms dict should contain a decreasing 'residual_frobenius' list."""
        fft, fc, comm = _make_fc()
        m = 2
        b_list = self._make_block(fc, m, prefix='b')
        x_list = self._make_block(fc, m, prefix='x')
        rng = np.random.default_rng(99)
        for j in range(m):
            b_list[j].s[...] = rng.standard_normal(b_list[j].s.shape)
            x_list[j].s[...] = 0.0

        def hessp(p, Ap): Ap.s[...] = p.s

        def P(r, z):      z.s[...] = r.s

        _, norms = solvers.dr_pbcg_mugrid(comm, fc, hessp, b_list, x_list, P, tol=1e-10)

        self.assertIn('residual_frobenius', norms)
        hist = norms['residual_frobenius']
        self.assertGreater(len(hist), 0)
        self.assertGreater(hist[0], hist[-1])  # residual decreased

    def test_callback_is_called(self):
        """Callback should be invoked at iteration 0 and each subsequent step."""
        fft, fc, comm = _make_fc()
        b_list = self._make_block(fc, m=2, prefix='b')
        x_list = self._make_block(fc, m=2, prefix='x')
        rng = np.random.default_rng(11)
        for f in b_list: f.s[...] = rng.standard_normal(f.s.shape)
        for f in x_list: f.s[...] = 0.0

        calls = []

        def hessp(p, Ap):
            Ap.s[...] = p.s

        def P(r, z):
            z.s[...] = r.s

        def callback(it, xl, sc):
            calls.append(it)

        solvers.dr_pbcg_mugrid(comm, fc, hessp, b_list, x_list, P,
                               tol=1e-10, callback=callback)

        self.assertGreater(len(calls), 0)
        self.assertEqual(calls[0], 0)

    def test_zero_initial_residual_returns_immediately(self):
        """x0 already at the solution → returns after init without iterating."""
        fft, fc, comm = _make_fc()
        b_list = self._make_block(fc, m=2, prefix='b')
        x_list = self._make_block(fc, m=2, prefix='x')
        for f in b_list: f.s[...] = 1.0
        for f in x_list: f.s[...] = 1.0  # x0 = solution for A=I, b=1

        calls = []

        def hessp(p, Ap):
            Ap.s[...] = p.s

        def P(r, z):
            z.s[...] = r.s

        def callback(it, xl, sc):
            calls.append(it)

        result, norms = solvers.dr_pbcg_mugrid(comm, fc, hessp, b_list, x_list, P,
                                               tol=1e-10, callback=callback)
        self.assertEqual(len(calls), 0)
        for j in range(2):
            np.testing.assert_allclose(result[j].s, b_list[j].s, atol=1e-8)

    def test_maxiter_exceeded_warns(self):
        """maxiter=0 with non-zero residual should issue a RuntimeWarning."""
        fft, fc, comm = _make_fc()
        b_list = self._make_block(fc, m=2, prefix='b')
        x_list = self._make_block(fc, m=2, prefix='x')
        rng = np.random.default_rng(13)
        for f in b_list: f.s[...] = rng.standard_normal(f.s.shape)
        for f in x_list: f.s[...] = 0.0

        def hessp(p, Ap):
            Ap.s[...] = p.s

        def P(r, z):
            z.s[...] = r.s

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            solvers.dr_pbcg_mugrid(comm, fc, hessp, b_list, x_list, P,
                                   tol=1e-30, maxiter=0)
        runtime_warnings = [wi for wi in w if issubclass(wi.category, RuntimeWarning)]
        self.assertGreater(len(runtime_warnings), 0)

    def test_rtol_mode(self):
        """rtol=True scales tolerance by initial residual norm."""
        fft, fc, comm = _make_fc()
        b_list = self._make_block(fc, m=2, prefix='b')
        x_list = self._make_block(fc, m=2, prefix='x')
        rng = np.random.default_rng(17)
        for f in b_list: f.s[...] = 100.0 * rng.standard_normal(f.s.shape)
        for f in x_list: f.s[...] = 0.0

        def hessp(p, Ap):
            Ap.s[...] = p.s

        def P(r, z):
            z.s[...] = r.s

        result, _ = solvers.dr_pbcg_mugrid(comm, fc, hessp, b_list, x_list, P,
                                           tol=1e-6, rtol=True)
        for j in range(2):
            np.testing.assert_allclose(result[j].s, b_list[j].s, rtol=1e-4)

    def test_3d_domain_block(self):
        """DR-PBCG works correctly on a 3D grid with m=2 random RHS."""
        fft, fc, comm = _make_fc(nb_pixels=(3, 3, 3))
        m = 2
        b_list = self._make_block(fc, m, components=(3,), prefix='b')
        x_list = self._make_block(fc, m, components=(3,), prefix='x')
        rng = np.random.default_rng(55)
        x_true = [rng.standard_normal(f.s.shape) for f in b_list]
        for j, f in enumerate(b_list): f.s[...] = x_true[j]  # A=I → b=x_true
        for f in x_list:               f.s[...] = 0.0

        def hessp(p, Ap):
            Ap.s[...] = p.s

        def P(r, z):
            z.s[...] = r.s

        result, norms = solvers.dr_pbcg_mugrid(comm, fc, hessp, b_list, x_list, P, tol=1e-10)
        for j in range(m):
            np.testing.assert_allclose(result[j].s, x_true[j], atol=1e-8,
                                       err_msg=f"3D RHS {j} incorrect")

    def test_homogenization_problem_anal_solution(self):
        """Solver scalar homogenization problem with analytical solution."""
        nb_pixels = (84, 84)
        dim = len(nb_pixels)
        discretization = _make_domain(nb_pixels=nb_pixels)

        # create material data field
        mat_contrast = 1
        mat_contrast_2 = 1e2
        conductivity_C_1 = np.array([[1., 0], [0, 1.0]])

        material_data_field_C_0 = discretization.get_material_data_size_field_mugrid(name='conductivity_tensor')

        # populate the field with C_1 material
        material_data_field_C_0.s[...] = conductivity_C_1[:, :, np.newaxis, np.newaxis, np.newaxis]

        # material distribution
        # phase_field_geom = microstructure_library.get_geometry(nb_voxels=discretization.nb_of_pixels,
        #                                                        microstructure_name=geometry_ID,
        #                                                        coordinates=discretization.fft.coords)

        phase_field = discretization.get_scalar_field(name='phase_field')
        phase_field.s[...].fill(1)
        phase_field.s[0, 0, nb_pixels[0] // 4:3 * nb_pixels[0] // 4, nb_pixels[1] // 4:3 * nb_pixels[1] // 4] = 0
        matrix_mask = phase_field.s[0, 0] > 0
        inc_mask = phase_field.s[0, 0] == 0

        # apply material distribution

        material_data_field_C_0.s[..., matrix_mask] = mat_contrast_2 * material_data_field_C_0.s[..., matrix_mask]
        material_data_field_C_0.s[..., inc_mask] = mat_contrast * material_data_field_C_0.s[..., inc_mask]

        preconditioner = discretization.get_preconditioner_Green_mugrid(reference_material_data_ijkl=material_data_field_C_0)

        def hessp(p, Ap):
            discretization.apply_system_matrix_mugrid(material_data_field=material_data_field_C_0,
                                                      input_field_inxyz=p,
                                                      output_field_inxyz=Ap)
            discretization.fft.communicate_ghosts(Ap)

        def P(r, z):
            discretization.fft.communicate_ghosts(r)
            discretization.apply_preconditioner_mugrid(preconditioner_Fourier_fnfnqks=preconditioner,
                                                       input_nodal_field_fnxyz=r,
                                                       output_nodal_field_fnxyz=z)

        list_of_solution_field = [discretization.get_unknown_size_field(name=f'solution-{j}') for j in range(dim)]
        list_of_macro_gradient_field = [discretization.get_gradient_size_field(name=f'macro_gradient_field-{j}') for j
                                        in
                                        range(dim)]
        list_of_rhs_field = [discretization.get_unknown_size_field(name=f'rhs_field-{j}') for j in range(dim)]

        homogenized_A_ij = np.zeros(np.array(2 * [dim, ]))
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
                                          hessp=hessp,
                                          b_list=list_of_rhs_field,
                                          x_list=list_of_solution_field,
                                          P=P,
                                          tol=1e-10,
                                          rtol=True)
        # result, norms = solvers.dr_pbcg_mugrid(comm, fc, hessp, b_list, x_list, P, tol=1e-10)
        for i in range(dim):
            homogenized_A_ij[i, :] = discretization.get_homogenized_stress_mugrid(
                material_data_field_ijklqxyz=material_data_field_C_0,
                displacement_field_inxyz=list_of_solution_field[i],
                macro_gradient_field_ijqxyz=list_of_macro_gradient_field[i])
        J_eff = mat_contrast_2 * np.sqrt((mat_contrast_2 + 3 * mat_contrast) / (3 * mat_contrast_2 + mat_contrast))
        # for coarse grid, the tolerance must be large. due to discretizatiion error
        print(J_eff)
        print(homogenized_A_ij)
        print(len(norms['residual_frobenius']))
        np.testing.assert_allclose(homogenized_A_ij[0, 0], J_eff, atol=0,
                                   rtol=1e-2)
        np.testing.assert_allclose(homogenized_A_ij[1, 1], J_eff, atol=0,
                                   rtol=1e-2)
        self.assertGreater(homogenized_A_ij[0, 0] - J_eff, 0)


if __name__ == '__main__':
    unittest.main()
