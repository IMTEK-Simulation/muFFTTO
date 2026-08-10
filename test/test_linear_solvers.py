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


class TestConjugateGradientsMuGrid(unittest.TestCase):
    """Tests for solvers.conjugate_gradients_mugrid."""

    def test_identity_system_converges(self):
        """A=I, b=1, x0=0 → solution x=1 (converges in one step)."""
        fft, fc, comm = _make_fc()
        b = fc.real_field(name='b', components=(2,), sub_pt='nodal_points')
        x = fc.real_field(name='x', components=(2,), sub_pt='nodal_points')
        b.s[...] = 1.0
        x.s[...] = 0.0

        def hessp(p, Ap):
            Ap.s[...] = p.s

        def P(r, z):
            z.s[...] = r.s

        result = solvers.conjugate_gradients_mugrid(comm, fc, hessp, b, x, P, tol=1e-10)
        np.testing.assert_allclose(result.s, b.s, atol=1e-8)

    def test_scaled_identity_system(self):
        """A = scale*I, b = scale*ones → solution x = ones."""
        fft, fc, comm = _make_fc()
        b = fc.real_field(name='b', components=(2,), sub_pt='nodal_points')
        x = fc.real_field(name='x', components=(2,), sub_pt='nodal_points')
        scale = 4.0
        b.s[...] = scale
        x.s[...] = 0.0

        def hessp(p, Ap):
            Ap.s[...] = scale * p.s

        def P(r, z):
            z.s[...] = r.s

        result = solvers.conjugate_gradients_mugrid(comm, fc, hessp, b, x, P, tol=1e-10)
        np.testing.assert_allclose(result.s, np.ones(result.s.shape), atol=1e-8)

    def test_spatially_varying_diagonal_system(self):
        """A = diag(a(x)), b = a(x)*ones → solution x = ones."""
        fft, fc, comm = _make_fc()
        b = fc.real_field(name='b', components=(2,), sub_pt='nodal_points')
        x = fc.real_field(name='x', components=(2,), sub_pt='nodal_points')
        diag = fc.real_field(name='diag', components=(2,), sub_pt='nodal_points')

        rng = np.random.default_rng(42)
        diag.s[...] = rng.uniform(1.0, 5.0, diag.s.shape)
        b.s[...] = diag.s.copy()  # b = A * ones
        x.s[...] = 0.0

        def hessp(p, Ap):
            Ap.s[...] = diag.s * p.s

        def P(r, z):
            z.s[...] = r.s

        result = solvers.conjugate_gradients_mugrid(comm, fc, hessp, b, x, P,
                                                    tol=1e-10, maxiter=500)
        np.testing.assert_allclose(result.s, np.ones(result.s.shape), atol=1e-6)

    def test_zero_initial_residual_returns_immediately(self):
        """x0 already the solution → CG returns without iterating."""
        fft, fc, comm = _make_fc()
        b = fc.real_field(name='b', components=(2,), sub_pt='nodal_points')
        x = fc.real_field(name='x', components=(2,), sub_pt='nodal_points')
        b.s[...] = 1.0
        x.s[...] = 1.0  # x0 = solution for A=I, b=1

        iteration_count = [0]

        def hessp(p, Ap):
            Ap.s[...] = p.s

        def P(r, z):
            z.s[...] = r.s

        def callback(it, *args):
            iteration_count[0] += 1

        result = solvers.conjugate_gradients_mugrid(comm, fc, hessp, b, x, P,
                                                    tol=1e-10, callback=callback)
        # No CG iterations should have been performed (early return before loop)
        self.assertEqual(iteration_count[0], 0)
        np.testing.assert_allclose(result.s, b.s, atol=1e-8)

    def test_exact_preconditioner_converges_immediately(self):
        """P = A^{-1} (exact inverse) → converges in a single iteration."""
        fft, fc, comm = _make_fc()
        b = fc.real_field(name='b', components=(2,), sub_pt='nodal_points')
        x = fc.real_field(name='x', components=(2,), sub_pt='nodal_points')
        scale = 7.0
        b.s[...] = scale * 3.0
        x.s[...] = 0.0

        iteration_count = [0]

        def hessp(p, Ap):
            Ap.s[...] = scale * p.s

        def P(r, z):
            z.s[...] = r.s / scale  # exact inverse preconditioner

        def callback(it, *args):
            iteration_count[0] = it

        result = solvers.conjugate_gradients_mugrid(comm, fc, hessp, b, x, P,
                                                    tol=1e-10, callback=callback)
        np.testing.assert_allclose(result.s, b.s / scale, atol=1e-8)
        self.assertLessEqual(iteration_count[0], 1)

    def test_callback_receives_correct_arguments(self):
        """Callback should be called at iteration 0 and subsequent iterations."""
        fft, fc, comm = _make_fc()
        b = fc.real_field(name='b', components=(2,), sub_pt='nodal_points')
        x = fc.real_field(name='x', components=(2,), sub_pt='nodal_points')
        b.s[...] = 1.0
        x.s[...] = 0.0

        calls = []

        def hessp(p, Ap):
            Ap.s[...] = p.s

        def P(r, z):
            z.s[...] = r.s

        def callback(iteration, x_s, r_s, p_s, z_s, stop_crit):
            calls.append({
                'iteration': iteration,
                'stop_crit': stop_crit,
            })

        solvers.conjugate_gradients_mugrid(comm, fc, hessp, b, x, P,
                                           tol=1e-10, callback=callback)

        self.assertGreater(len(calls), 0)
        self.assertEqual(calls[0]['iteration'], 0)
        self.assertGreater(calls[0]['stop_crit'], 0.0)

    def test_non_positive_definite_raises_runtime_error(self):
        """A=-I (negative definite) → RuntimeError on first pAp check."""
        fft, fc, comm = _make_fc()
        b = fc.real_field(name='b', components=(2,), sub_pt='nodal_points')
        x = fc.real_field(name='x', components=(2,), sub_pt='nodal_points')
        b.s[...] = 1.0
        x.s[...] = 0.0

        def hessp(p, Ap):
            Ap.s[...] = -p.s  # A = -I

        def P(r, z):
            z.s[...] = r.s

        with self.assertRaises(RuntimeError):
            solvers.conjugate_gradients_mugrid(comm, fc, hessp, b, x, P)

    def test_maxiter_exceeded_issues_warning(self):
        """maxiter=0 with non-zero residual should trigger RuntimeWarning."""
        fft, fc, comm = _make_fc()
        b = fc.real_field(name='b', components=(2,), sub_pt='nodal_points')
        x = fc.real_field(name='x', components=(2,), sub_pt='nodal_points')
        b.s[...] = 1.0
        x.s[...] = 0.0

        def hessp(p, Ap):
            Ap.s[...] = p.s

        def P(r, z):
            z.s[...] = r.s

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            solvers.conjugate_gradients_mugrid(comm, fc, hessp, b, x, P,
                                               tol=1e-30, maxiter=0)
        runtime_warnings = [wi for wi in w if issubclass(wi.category, RuntimeWarning)]
        self.assertGreater(len(runtime_warnings), 0)

    def test_rtol_mode(self):
        """rtol=True should scale tolerance by initial residual norm."""
        fft, fc, comm = _make_fc()
        b = fc.real_field(name='b', components=(2,), sub_pt='nodal_points')
        x = fc.real_field(name='x', components=(2,), sub_pt='nodal_points')
        b.s[...] = 100.0
        x.s[...] = 0.0

        def hessp(p, Ap):
            Ap.s[...] = p.s

        def P(r, z):
            z.s[...] = r.s

        result = solvers.conjugate_gradients_mugrid(comm, fc, hessp, b, x, P,
                                                    tol=1e-6, rtol=True)
        np.testing.assert_allclose(result.s, b.s, atol=1e-3)

    def test_3d_domain(self):
        """CG works correctly on a 3D grid."""
        fft, fc, comm = _make_fc(nb_pixels=(3, 3, 3))
        b = fc.real_field(name='b', components=(3,), sub_pt='nodal_points')
        x = fc.real_field(name='x', components=(3,), sub_pt='nodal_points')
        b.s[...] = 2.0
        x.s[...] = 0.0

        def hessp(p, Ap):
            Ap.s[...] = p.s

        def P(r, z):
            z.s[...] = r.s

        result = solvers.conjugate_gradients_mugrid(comm, fc, hessp, b, x, P, tol=1e-10)
        np.testing.assert_allclose(result.s, b.s, atol=1e-8)

    def test_custom_norm_metric(self):
        """norm_metric parameter should be used as stopping criterion."""
        fft, fc, comm = _make_fc()
        b = fc.real_field(name='b', components=(2,), sub_pt='nodal_points')
        x = fc.real_field(name='x', components=(2,), sub_pt='nodal_points')
        b.s[...] = 1.0
        x.s[...] = 0.0

        norm_calls = [0]

        def hessp(p, Ap):
            Ap.s[...] = p.s

        def P(r, z):
            z.s[...] = r.s

        def norm_metric(r, Pr):
            Pr.s[...] = r.s  # identity metric (same as default)
            norm_calls[0] += 1

        result = solvers.conjugate_gradients_mugrid(comm, fc, hessp, b, x, P,
                                                    tol=1e-10, norm_metric=norm_metric)
        np.testing.assert_allclose(result.s, b.s, atol=1e-8)
        self.assertGreater(norm_calls[0], 0)

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

            solvers.conjugate_gradients_mugrid(comm=discretization.communicator,
                                               fc=discretization.field_collection,
                                               hessp=hessp,
                                               b=rhs_field,
                                               x=solution_field,
                                               P=P,
                                               tol=1e-10, rtol=True)
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
        """m=1 DR-PBCG should give the same solution as conjugate_gradients_mugrid."""
        # CG solve
        fft_cg, fc_cg, comm_cg = _make_fc()
        b_cg = fc_cg.real_field(name='b', components=(2,), sub_pt='nodal_points')
        x_cg = fc_cg.real_field(name='x', components=(2,), sub_pt='nodal_points')
        scale = 3.0
        b_cg.s[...] = 6.0
        x_cg.s[...] = 0.0

        def hessp(p, Ap): Ap.s[...] = scale * p.s

        def P(r, z):      z.s[...] = r.s

        x_cg_result = solvers.conjugate_gradients_mugrid(
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
