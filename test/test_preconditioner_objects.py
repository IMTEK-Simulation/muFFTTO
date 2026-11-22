"""muFFTTO.preconditioners: Green (muGrid for one node per pixel), Jacobi, Green-Jacobi, dual."""
import numpy as np
import pytest

from muFFTTO import domain, material_models, preconditioners, solvers


def make(element, problem, nb_pixels):
    dim = len(nb_pixels)
    cell = domain.PeriodicUnitCell(domain_size=[1.0 + 0.3 * i for i in range(dim)], problem_type=problem)
    return domain.Discretization(cell=cell, nb_of_pixels_global=nb_pixels, discretization_type='finite_element',
                                 element_type=element)


def reference(d, problem):
    dim = d.domain_dimension
    if problem == 'elasticity':
        return material_models.get_elastic_material_tensor(dim=dim, K=2.0, mu=1.0, kind='linear'), 'small_strain'
    return np.eye(dim), None


def random_field(d, name, seed):
    field = d.get_unknown_size_field(name=name)
    field.s[...] = np.random.default_rng(seed).standard_normal(field.s.shape)
    field.s[...] -= field.s.mean(axis=tuple(range(2, field.s.ndim)), keepdims=True)
    return field


def scaled_two_phase(d, C_ref, contrast=1e-3, name='scale'):
    """C = s(x) C_ref with s = 1 in the matrix, `contrast` in a box inclusion."""
    scale = d.get_quad_field_scalar(name=name)
    scale.s[...] = 1.0
    box = tuple(slice(n // 4, 3 * n // 4) for n in d.nb_of_pixels)
    scale.s[(0, 0, slice(None)) + box] = contrast
    return scale, domain.ScaledMaterial([(scale, C_ref)])


CASES = [('linear_triangles', 'elasticity', (12, 10)), ('linear_triangles', 'conductivity', (12, 10)),
         ('bilinear_rectangle', 'elasticity', (8, 6)), ('trilinear_hexahedron', 'elasticity', (4, 6, 4))]


@pytest.mark.parametrize('element, problem, nb_pixels', CASES)
def test_green_equals_muffto_symbol(element, problem, nb_pixels):
    d = make(element, problem, nb_pixels)
    C_ref, formulation = reference(d, problem)
    green = preconditioners.Green(d, C_ref, formulation=formulation)
    assert green._mugrid is not None                     # one node per pixel: muGrid's preconditioner
    symbol = d.get_preconditioner_Green_mugrid(reference_material_data_ijkl=C_ref, formulation=formulation,
                                               name='symbol')
    r = random_field(d, 'r', 0)
    z, z_ref = d.get_unknown_size_field(name='z'), d.get_unknown_size_field(name='z_ref')
    green(r, z)
    d.apply_preconditioner_mugrid(preconditioner_Fourier_fnfnqks=symbol, input_nodal_field_fnxyz=r,
                                  output_nodal_field_fnxyz=z_ref)
    np.testing.assert_allclose(z.s, z_ref.s, atol=1e-13 * np.abs(z_ref.s).max())


def test_green_several_nodes_per_pixel_uses_block_inverse():
    d = make('biquadratic_rectangle', 'conductivity', (4, 6))
    green = preconditioners.Green(d, np.eye(2))
    assert green._mugrid is None and d.nb_nodes_per_pixel > 1
    r = random_field(d, 'r', 1)
    z = d.get_unknown_size_field(name='z')
    green(r, z)
    # G inverts K_ref on the zero-mean subspace
    Kz = d.get_unknown_size_field(name='Kz')
    d.apply_system_matrix_mugrid(material_data_field=np.eye(2), input_field_inxyz=z, output_field_inxyz=Kz)
    np.testing.assert_allclose(Kz.s, r.s, atol=1e-10 * np.abs(r.s).max())


def test_jacobi_divides_by_the_diagonal():
    d = make('linear_triangles', 'elasticity', (8, 8))
    C_ref, formulation = reference(d, 'elasticity')
    _, material = scaled_two_phase(d, C_ref)
    jacobi = preconditioners.Jacobi(d, material, formulation=formulation)
    K = d.get_system_matrix_mugrid(material, formulation=formulation)
    r = random_field(d, 'r', 2)
    z = d.get_unknown_size_field(name='z')
    jacobi(r, z)
    np.testing.assert_allclose(z.s.ravel(), r.s.ravel() / np.diag(K), rtol=1e-12)


def test_green_jacobi_is_scaled_green():
    d = make('linear_triangles', 'elasticity', (8, 8))
    C_ref, formulation = reference(d, 'elasticity')
    _, material = scaled_two_phase(d, C_ref)
    I_ref = 2.0 * preconditioners.isotropic_identity(2)
    gj = preconditioners.GreenJacobi(d, I_ref, material, formulation=formulation)
    green = preconditioners.Green(d, I_ref, formulation=formulation, name='plain-green')
    diagonal = np.diag(d.get_system_matrix_mugrid(material, formulation=formulation)).reshape(
        d.get_unknown_size_field(name='probe').s.shape)
    r = random_field(d, 'r', 3)
    z, scaled, inner = (d.get_unknown_size_field(name=n) for n in ('z', 'scaled', 'inner'))
    gj(r, z)
    scaled.s[...] = r.s / np.sqrt(diagonal)
    green(scaled, inner)
    np.testing.assert_allclose(z.s, inner.s / np.sqrt(diagonal), atol=1e-12 * np.abs(z.s).max())


@pytest.mark.parametrize('dim', [2, 3])
def test_dual_material_full_field_equals_scaled(dim):
    nb_pixels = (6, 4) if dim == 2 else (4, 4, 2)
    d = make('linear_triangles' if dim == 2 else 'trilinear_hexahedron', 'elasticity', nb_pixels)
    C_ref, _ = reference(d, 'elasticity')
    scale, scaled = scaled_two_phase(d, C_ref, contrast=0.01)
    full = d.get_material_data_size_field_mugrid(name='C_full')
    full.s[...] = np.einsum('ijkl,...->ijkl...', C_ref, scale.s[0, 0])
    dual_full = preconditioners.dual_material(d, C_ref, full)
    np.testing.assert_allclose(dual_full.s, np.einsum('ijkl,...->ijkl...', C_ref, 1.0 / scale.s[0, 0]),
                               rtol=1e-12, atol=1e-12)


def test_dual_material_conductivity():
    d = make('linear_triangles', 'conductivity', (6, 4))
    A_ref = np.array([[2.0, 0.5], [0.5, 1.0]])
    A = d.get_material_data_size_field_mugrid(name='A')
    A.s[...] = np.einsum('ij,...->ij...', A_ref, np.random.default_rng(4).uniform(0.1, 1.0, A.s.shape[2:]))
    dual = preconditioners.dual_material(d, A_ref, A)
    product = np.einsum('ij...,jk...->ik...', A.s, np.einsum('ij,jk...->ik...', np.linalg.inv(A_ref), dual.s))
    np.testing.assert_allclose(product, np.einsum('ij,...->ij...', A_ref, np.ones(A.s.shape[2:])), rtol=1e-12)


def test_dual_is_G_Kdual_G_and_solves_like_green():
    """z = G K(C_ref / s) G r, and CG with the dual gives the solution of CG with Green, in fewer iterations."""
    d = make('linear_triangles', 'elasticity', (32, 32))
    C_ref, formulation = reference(d, 'elasticity')
    scale, material = scaled_two_phase(d, C_ref, contrast=1e-3)
    dual = preconditioners.Dual(d, C_ref, material, formulation=formulation)
    green = dual.green
    r = random_field(d, 'r', 5)
    z, t1, t2 = (d.get_unknown_size_field(name=n) for n in ('z', 't1', 't2'))
    dual(r, z)
    green(r, t1)
    d.apply_system_matrix_mugrid(material_data_field=domain.ScaledMaterial([(1.0 / scale.s[0, 0], C_ref)]),
                                 input_field_inxyz=t1, output_field_inxyz=t2, formulation=formulation)
    green(t2, t1)
    np.testing.assert_allclose(z.s, t1.s, atol=1e-12 * np.abs(z.s).max())

    E = d.get_gradient_size_field(name='E')
    d.get_macro_gradient_field_mugrid(macro_gradient_ij=np.array([[1.0, 0.2], [0.2, 0.0]]),
                                      macro_gradient_field_ijqxyz=E)
    rhs = d.get_unknown_size_field(name='rhs')
    d.get_rhs_mugrid(material_data_field_ijklqxyz=material, macro_gradient_field_ijqxyz=E, rhs_inxyz=rhs)

    def K(x, Kx):
        d.apply_system_matrix_mugrid(material_data_field=material, input_field_inxyz=x, output_field_inxyz=Kx,
                                     formulation=formulation)

    results = {}
    for label, M in (('green', green), ('dual', dual)):
        u = d.get_unknown_size_field(name=f'u_{label}')
        u.s[...] = 0.0
        _, info = solvers.conjugate_gradients(d.communicator, d.field_collection, K, rhs, u, M, tol=1e-8)
        results[label] = (d.get_homogenized_stress_mugrid(material_data_field_ijklqxyz=material,
                                                         displacement_field_inxyz=u, macro_gradient_field_ijqxyz=E,
                                                         formulation=formulation), info['iterations'])
    np.testing.assert_allclose(results['dual'][0], results['green'][0], rtol=1e-6)
    assert results['dual'][1] < results['green'][1]
