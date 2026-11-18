import unittest

import numpy as np
from mpi4py import MPI
from muGrid import Solvers

from muFFTTO import constraints
from muFFTTO import domain
from muFFTTO import material_models
from muFFTTO.tensor_operations import to_host

try:
    import cupy
    cupy.cuda.runtime.getDeviceCount()
    DEVICES = ('cpu', 'gpu')
except Exception:  # no cupy or no GPU
    DEVICES = ('cpu',)

NB_PIXELS = (16, 16)


def _make_discretization(device):
    my_cell = domain.PeriodicUnitCell(domain_size=[1, 1], problem_type='elasticity')
    return domain.Discretization(cell=my_cell,
                                 nb_of_pixels_global=NB_PIXELS,
                                 discretization_type='finite_element',
                                 element_type='bilinear_rectangle',
                                 device=device)


def _column_dofs():
    """Both displacement components of all nodes in the column x = 4, with varying prescribed values."""
    dofs = np.array([[f, 0, 4, y] for f in range(2) for y in range(NB_PIXELS[1])])
    values = 0.02 * (1 + dofs[:, 0]) * np.sin(2 * np.pi * dofs[:, 3] / NB_PIXELS[1])
    return dofs, values


def _local_constrained_mask(discretization, dofs_global):
    """Boolean mask in the .s layout of the local subdomain, True on the given global dofs."""
    shape = discretization.get_unknown_size_field(name='mask_shape').s.shape
    mask = np.zeros(shape, dtype=bool)
    local = dofs_global.copy()
    local[:, 2:] -= np.asarray(discretization.fft.subdomain_locations)
    owned = np.all((local[:, 2:] >= 0) & (local[:, 2:] < np.asarray(shape[2:])), axis=1)
    mask[tuple(local[owned].T)] = True
    return mask


def _elastic_operators(discretization):
    """Heterogeneous linear elasticity: material, K_fun (K x) and M_fun (Green preconditioner)."""
    xp = cupy if discretization.device == 'gpu' else np
    lam, mu = material_models.get_lame_parameters(E=1, poisson=0.2)
    lam_field = discretization.get_quad_field_scalar(name='lam')
    mu_field = discretization.get_quad_field_scalar(name='mu')
    coords = to_host(discretization.fft.coords)
    stiff = xp.asarray(np.where(np.abs(coords[0] - 0.5) < 0.25, 10.0, 1.0))
    lam_field.s[...] = lam * stiff
    mu_field.s[...] = mu * stiff
    material = material_models.LinearElastic(discretization, lam_field, mu_field)

    def K_fun(x, Ax):
        discretization.apply_system_matrix_mugrid_explicit_stress(
            constitutive=material.get_stress, input_field_inxyz=x, output_field_inxyz=Ax,
            formulation='small_strain')
        discretization.fft.communicate_ghosts(Ax)

    C_ref = material_models.get_elastic_tensor_from_lame(dim=2, lam=lam, mu=mu)
    preconditioner = discretization.get_preconditioner_Green_mugrid(reference_material_data_ijkl=C_ref)

    def M_fun(x, Px):
        discretization.fft.communicate_ghosts(x)
        discretization.apply_preconditioner_mugrid(preconditioner_Fourier_fnfnqks=preconditioner,
                                                   input_nodal_field_fnxyz=x,
                                                   output_nodal_field_fnxyz=Px)

    return material, K_fun, M_fun


def _constrained_solve(discretization, nodal_constraints, macro_gradient):
    """Solve Q K Q u = Q b - Q K P(u_D), u = Q(u) + P(u_D). Returns (u, b, K_fun) with b the unconstrained rhs."""
    material, K_fun, M_fun = _elastic_operators(discretization)
    macro_gradient_field = discretization.get_gradient_size_field(name='macro_gradient_field')
    discretization.get_macro_gradient_field_mugrid(macro_gradient_ij=macro_gradient,
                                                   macro_gradient_field_ijqxyz=macro_gradient_field)
    rhs = discretization.get_unknown_size_field(name='rhs')
    discretization.get_rhs_explicit_stress_mugrid(stress_function=material.get_stress,
                                                  gradient_field_ijqxyz=macro_gradient_field,
                                                  rhs_inxyz=rhs)
    b = discretization.get_unknown_size_field(name='b')
    b.s[...] = rhs.s
    u = discretization.get_unknown_size_field(name='u_solution')
    u.s[...] = 0

    nodal_constraints.apply_to_rhs(rhs, K_fun)
    Solvers.conjugate_gradients(comm=discretization.communicator,
                                fc=discretization.field_collection,
                                hessp=nodal_constraints.project_operator(K_fun),
                                b=rhs, x=u,
                                prec=nodal_constraints.project_preconditioner(M_fun),
                                tol=1e-10, maxiter=1000)
    nodal_constraints.impose_prescribed_values(u)
    return u, b, K_fun


def _max_free_residual(discretization, u, b, K_fun, constrained_mask):
    """max |K u - b| over the free dofs, over all MPI ranks."""
    residual = discretization.get_unknown_size_field(name='residual')
    K_fun(u, residual)
    residual_free = np.where(constrained_mask, 0.0, to_host(residual.s - b.s))
    return MPI.COMM_WORLD.allreduce(float(np.abs(residual_free).max()), op=MPI.MAX)


class TestNodalConstraints(unittest.TestCase):

    def _random_field(self, discretization, name):
        field = discretization.get_unknown_size_field(name=name)
        rng = np.random.default_rng(MPI.COMM_WORLD.rank)
        xp = cupy if discretization.device == 'gpu' else np
        field.s[...] = xp.asarray(rng.random(field.s.shape))
        return field

    def test_projections(self):
        """Q zeroes exactly the constrained dofs, P keeps exactly them, P + Q = I."""
        for device in DEVICES:
            with self.subTest(device=device):
                discretization = _make_discretization(device)
                dofs, values = _column_dofs()
                nodal_constraints = constraints.NodalConstraints(discretization, dofs, values)
                self.assertEqual(nodal_constraints.nb_constrained_dofs, dofs.shape[0])

                mask = _local_constrained_mask(discretization, dofs)
                u = self._random_field(discretization, 'u')
                u_ref = to_host(u.s).copy()

                nodal_constraints.apply_free_node_projection(u)
                np.testing.assert_array_equal(to_host(u.s), np.where(mask, 0.0, u_ref))

                u.s[...] = u.s * 0 + (cupy.asarray(u_ref) if device == 'gpu' else u_ref)
                nodal_constraints.apply_constrained_node_projection(u)
                np.testing.assert_array_equal(to_host(u.s), np.where(mask, u_ref, 0.0))

    def test_prescribed_values(self):
        """P(u_D) holds u_D on the constrained dofs and zero elsewhere; set_prescribed_values updates it."""
        for device in DEVICES:
            with self.subTest(device=device):
                discretization = _make_discretization(device)
                dofs, values = _column_dofs()
                nodal_constraints = constraints.NodalConstraints(discretization, dofs, values)
                mask = _local_constrained_mask(discretization, dofs)

                expected = np.zeros(mask.shape)
                local = dofs.copy()
                local[:, 2:] -= np.asarray(discretization.fft.subdomain_locations)
                owned = np.all((local[:, 2:] >= 0) & (local[:, 2:] < np.asarray(mask.shape[2:])), axis=1)
                expected[tuple(local[owned].T)] = values[owned]
                np.testing.assert_array_equal(to_host(nodal_constraints.prescribed_displacement_field.s), expected)

                nodal_constraints.set_prescribed_values(2 * values)
                np.testing.assert_array_equal(to_host(nodal_constraints.prescribed_displacement_field.s),
                                              2 * expected)

                u = self._random_field(discretization, 'u')
                u_ref = to_host(u.s).copy()
                nodal_constraints.impose_prescribed_values(u)
                np.testing.assert_array_equal(to_host(u.s), np.where(mask, 2 * expected, u_ref))

    def test_from_mask_equals_global_dofs(self):
        """from_mask with the same dofs and a field of prescribed values gives the same projections."""
        for device in DEVICES:
            with self.subTest(device=device):
                discretization = _make_discretization(device)
                dofs, values = _column_dofs()
                from_dofs = constraints.NodalConstraints(discretization, dofs, values, name='from_dofs')
                mask = _local_constrained_mask(discretization, dofs)
                from_mask = constraints.NodalConstraints.from_mask(
                    discretization, mask, from_dofs.prescribed_displacement_field, name='from_mask')
                self.assertEqual(from_mask.nb_constrained_dofs, from_dofs.nb_constrained_dofs)
                np.testing.assert_array_equal(to_host(from_mask.prescribed_displacement_field.s),
                                              to_host(from_dofs.prescribed_displacement_field.s))

                u = self._random_field(discretization, 'u')
                v = self._random_field(discretization, 'v')
                from_dofs.apply_free_node_projection(u)
                from_mask.apply_free_node_projection(v)
                np.testing.assert_array_equal(to_host(u.s), to_host(v.s))

    def test_pixel_mask_constrains_all_components(self):
        """A mask of shape [x, y] constrains all components of the selected pixels."""
        for device in DEVICES:
            with self.subTest(device=device):
                discretization = _make_discretization(device)
                shape = discretization.get_unknown_size_field(name='mask_shape').s.shape
                pixel_mask = np.zeros(shape[2:], dtype=bool)
                pixel_mask[1, :] = True
                nodal_constraints = constraints.NodalConstraints.from_mask(discretization, pixel_mask, 0.5)
                self.assertEqual(nodal_constraints.nb_local_constrained_dofs, 2 * shape[3])

                u = self._random_field(discretization, 'u')
                nodal_constraints.impose_prescribed_values(u)
                np.testing.assert_array_equal(to_host(u.s)[:, :, 1, :], 0.5)

    def test_prescribed_total_displacement(self):
        """u_D = u_total - E.x on the constrained dofs, x = physical nodal coordinates (non-unit cell)."""
        for device in DEVICES:
            with self.subTest(device=device):
                my_cell = domain.PeriodicUnitCell(domain_size=[2, 1], problem_type='elasticity')
                discretization = domain.Discretization(cell=my_cell, nb_of_pixels_global=NB_PIXELS,
                                                       discretization_type='finite_element',
                                                       element_type='bilinear_rectangle', device=device)
                dofs, values = _column_dofs()
                nodal_constraints = constraints.NodalConstraints(discretization, dofs, values)
                macro_gradient = np.array([[0.1, 0.3], [-0.2, 0.05]])
                nodal_constraints.set_prescribed_total_displacement(values, macro_gradient)

                local = dofs.copy()
                local[:, 2:] -= np.asarray(discretization.fft.subdomain_locations)
                shape = nodal_constraints.prescribed_displacement_field.s.shape
                owned = np.all((local[:, 2:] >= 0) & (local[:, 2:] < np.asarray(shape[2:])), axis=1)
                x = 2.0 * dofs[owned, 2] / NB_PIXELS[0]  # physical coordinates, cell 2 x 1
                y = 1.0 * dofs[owned, 3] / NB_PIXELS[1]
                f = dofs[owned, 0]
                expected = values[owned] - (macro_gradient[f, 0] * x + macro_gradient[f, 1] * y)
                np.testing.assert_allclose(to_host(nodal_constraints.prescribed_displacement_field.s)[
                                               tuple(local[owned].T)], expected, atol=1e-14)

    def test_constrained_solve(self):
        """Projected CG: u = u_D on constrained dofs, residual K u - b vanishes on free dofs."""
        for device in DEVICES:
            with self.subTest(device=device):
                discretization = _make_discretization(device)
                dofs, values = _column_dofs()
                nodal_constraints = constraints.NodalConstraints(discretization, dofs, values)
                u, b, K_fun = _constrained_solve(discretization, nodal_constraints,
                                                 macro_gradient=np.array([[0.0, 0.1], [0.1, 0.0]]))

                mask = _local_constrained_mask(discretization, dofs)
                np.testing.assert_array_equal(to_host(u.s)[mask],
                                              to_host(nodal_constraints.prescribed_displacement_field.s)[mask])
                self.assertLess(_max_free_residual(discretization, u, b, K_fun, mask), 1e-8)

    def test_project_operator_and_preconditioner(self):
        """hessp applies Q K Q, prec applies Q M, apply_to_rhs gives Q b - Q K P(u_D)."""
        for device in DEVICES:
            with self.subTest(device=device):
                discretization = _make_discretization(device)
                _, K_fun, M_fun = _elastic_operators(discretization)
                dofs, values = _column_dofs()
                nodal_constraints = constraints.NodalConstraints(discretization, dofs, values)
                mask = _local_constrained_mask(discretization, dofs)

                x = self._random_field(discretization, 'x')
                x_free = discretization.get_unknown_size_field(name='x_free')
                x_free.s[...] = x.s
                x_free.s[...] = x_free.s * (cupy.asarray(~mask) if device == 'gpu' else ~mask)
                expected = discretization.get_unknown_size_field(name='expected')

                # hessp = Q K Q
                K_fun(x_free, expected)
                Ax = discretization.get_unknown_size_field(name='Ax')
                nodal_constraints.project_operator(K_fun)(x, Ax)
                np.testing.assert_allclose(to_host(Ax.s), np.where(mask, 0.0, to_host(expected.s)), atol=1e-14)
                np.testing.assert_array_equal(to_host(x.s), to_host(x_free.s))  # Q(x) applied in place

                # prec = Q M (input already in range(Q))
                M_fun(x_free, expected)
                Mx = discretization.get_unknown_size_field(name='Mx')
                nodal_constraints.project_preconditioner(M_fun)(x_free, Mx)
                np.testing.assert_allclose(to_host(Mx.s), np.where(mask, 0.0, to_host(expected.s)), atol=1e-14)

                # apply_to_rhs: Q b - Q K P(u_D)
                rhs = self._random_field(discretization, 'rhs_random')
                b_host = to_host(rhs.s).copy()
                K_fun(nodal_constraints.prescribed_displacement_field, expected)
                nodal_constraints.apply_to_rhs(rhs, K_fun)
                np.testing.assert_allclose(to_host(rhs.s), np.where(mask, 0.0, b_host - to_host(expected.s)),
                                           atol=1e-14)

    def test_constrained_solve_total_displacement(self):
        """With set_prescribed_total_displacement(0, E), the constrained nodes do not move: E.x + u = 0."""
        for device in DEVICES:
            with self.subTest(device=device):
                discretization = _make_discretization(device)
                dofs, _ = _column_dofs()
                macro_gradient = np.array([[0.05, 0.1], [0.1, -0.02]])
                nodal_constraints = constraints.NodalConstraints(discretization, dofs)
                nodal_constraints.set_prescribed_total_displacement(0.0, macro_gradient)
                u, b, K_fun = _constrained_solve(discretization, nodal_constraints, macro_gradient)

                mask = _local_constrained_mask(discretization, dofs)
                coordinates = to_host(discretization.get_nodal_points_coordinates().s)  # [j, n, x, y]
                total = to_host(u.s) + np.einsum('fj,jnxy->fnxy', macro_gradient, coordinates)
                np.testing.assert_allclose(total[mask], 0.0, atol=1e-14)
                self.assertLess(_max_free_residual(discretization, u, b, K_fun, mask), 1e-8)

    def test_rank_without_constrained_dofs(self):
        """Dofs only in the row y = 0 (owned by one MPI rank); all methods work on ranks owning none."""
        for device in DEVICES:
            with self.subTest(device=device):
                discretization = _make_discretization(device)
                dofs = np.array([[f, 0, x, 0] for f in range(2) for x in range(NB_PIXELS[0])])
                values = 0.01 * (1 + dofs[:, 2])
                nodal_constraints = constraints.NodalConstraints(discretization, dofs, values)
                mask = _local_constrained_mask(discretization, dofs)
                self.assertEqual(nodal_constraints.nb_local_constrained_dofs, int(mask.sum()))
                self.assertEqual(nodal_constraints.nb_constrained_dofs, dofs.shape[0])

                u = self._random_field(discretization, 'u')
                u_ref = to_host(u.s).copy()
                nodal_constraints.apply_free_node_projection(u)
                np.testing.assert_array_equal(to_host(u.s), np.where(mask, 0.0, u_ref))
                nodal_constraints.set_prescribed_total_displacement(values, np.eye(2) * 0.1)

                u, b, K_fun = _constrained_solve(discretization, nodal_constraints,
                                                 macro_gradient=np.array([[0.0, 0.1], [0.1, 0.0]]))
                self.assertLess(_max_free_residual(discretization, u, b, K_fun, mask), 1e-8)

    def test_no_constrained_dofs(self):
        """An empty set of constrained dofs: projections are the identity, the solve is unconstrained."""
        for device in DEVICES:
            with self.subTest(device=device):
                discretization = _make_discretization(device)
                for nodal_constraints in (
                        constraints.NodalConstraints(discretization, np.zeros((0, 4), dtype=int), name='empty_dofs'),
                        constraints.NodalConstraints.from_mask(discretization,
                                                               np.zeros(tuple(discretization.nb_of_pixels), dtype=bool),  # local
                                                               name='empty_mask')):
                    self.assertEqual(nodal_constraints.nb_constrained_dofs, 0)
                    u = self._random_field(discretization, 'u')
                    u_ref = to_host(u.s).copy()
                    nodal_constraints.apply_free_node_projection(u)
                    nodal_constraints.impose_prescribed_values(u)
                    nodal_constraints.set_prescribed_values(1.0)
                    nodal_constraints.set_prescribed_total_displacement(0.0, np.eye(2))
                    np.testing.assert_array_equal(to_host(u.s), u_ref)

                    u, b, K_fun = _constrained_solve(discretization, nodal_constraints,
                                                     macro_gradient=np.array([[0.0, 0.1], [0.1, 0.0]]))
                    self.assertLess(_max_free_residual(discretization, u, b, K_fun,
                                                       np.zeros(u.s.shape, dtype=bool)), 1e-8)

    def test_from_mask_with_field(self):
        """from_mask accepts muGrid fields for the mask (nonzero = constrained) and the values."""
        for device in DEVICES:
            with self.subTest(device=device):
                discretization = _make_discretization(device)
                dofs, values = _column_dofs()
                from_dofs = constraints.NodalConstraints(discretization, dofs, values, name='from_dofs')
                mask = _local_constrained_mask(discretization, dofs)

                constrained_dofs_field = discretization.get_unknown_size_field(name='constrained_dofs_field')
                constrained_dofs_field.s[...] = cupy.asarray(mask) if device == 'gpu' else mask
                from_field = constraints.NodalConstraints.from_mask(
                    discretization, constrained_dofs_field, from_dofs.prescribed_displacement_field,
                    name='from_field')
                self.assertEqual(from_field.nb_constrained_dofs, dofs.shape[0])
                np.testing.assert_array_equal(to_host(from_field.prescribed_displacement_field.s),
                                              to_host(from_dofs.prescribed_displacement_field.s))


if __name__ == '__main__':
    unittest.main()
