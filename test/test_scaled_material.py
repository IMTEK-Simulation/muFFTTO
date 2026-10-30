"""Tests of domain.ScaledMaterial: C(x) = sum_k s_k(x) C_k.

The reference is always the plain contraction with the full tangent per quadrature point (einsum, or the
non-ScaledMaterial code path of the Discretization), for
  - apply() and apply(transposed=True) on elasticity and conductivity gradients, 2D and 3D, both on plain numpy
    arrays and on muGrid fields (muGrid's memory layout, where the fast path works on views),
  - apply(out=...), also in place (out = the input),
  - the scale types: float, ndarray, muGrid quadrature field, callable,
  - the minor_symmetric flag and the small-strain system matrix / stress field, which skip the symmetrisation of
    the gradient for a minor-symmetric C.
"""
import numpy as np
import pytest

from muFFTTO import domain
from muFFTTO import material_models

cases = [(2, 'bilinear_rectangle', [3, 4]),
         (3, 'trilinear_hexahedron', [3, 2, 4])]


def _discretization(dim, element_type, nb_pixels, problem_type):
    cell = domain.PeriodicUnitCell(domain_size=[1.3, 0.7, 1.1][:dim], problem_type=problem_type)
    return domain.Discretization(cell=cell, nb_of_pixels_global=nb_pixels, discretization_type='finite_element',
                                 element_type=element_type)


def _isotropic(dim, E=3.0, poisson=0.2):
    K, G = material_models.get_bulk_and_shear_modulus(E=E, poisson=poisson)
    return material_models.get_elastic_material_tensor(dim=dim, K=K, mu=G, kind='linear')


def _reference(tensors_and_scales, g, transposed):
    """sum_k s_k (C_k : g) by einsum; s_k broadcast over [q, x, y(, z)]."""
    out = 0
    for s, C in tensors_and_scales:
        if C.ndim == 4:
            term = np.einsum('ijkl,ij...->kl...' if transposed else 'ijkl,kl...->ij...', C, g)
        else:
            term = np.einsum('ij,ui...->uj...' if transposed else 'ij,uj...->ui...', C, g)
        out = out + s * term
    return out


@pytest.mark.parametrize('dim, element_type, nb_pixels', cases)
@pytest.mark.parametrize('problem_type', ['elasticity', 'conductivity'])
@pytest.mark.parametrize('transposed', [False, True])
@pytest.mark.parametrize('layout', ['numpy', 'mugrid'])
def test_apply_matches_einsum(dim, element_type, nb_pixels, problem_type, transposed, layout):
    """Two terms (a field scale and a float scale) with general (non-symmetric) tensors."""
    rng = np.random.default_rng(0)
    discretization = _discretization(dim, element_type, nb_pixels, problem_type)
    gradient = discretization.get_gradient_size_field(name='test_gradient')
    if layout == 'numpy':
        g = rng.normal(size=gradient.s.shape)             # C-ordered copy: the general (copying) path
    else:
        gradient.s[...] = rng.normal(size=gradient.s.shape)
        g = gradient.s                                    # muGrid layout: the view path
    tensor_shape = (dim,) * (4 if problem_type == 'elasticity' else 2)
    C_1, C_2 = rng.normal(size=tensor_shape), rng.normal(size=tensor_shape)
    s_1 = rng.uniform(0.1, 2.0, size=g.shape[2:])

    material = domain.ScaledMaterial([(s_1, C_1), (0.3, C_2)])
    expected = _reference([(s_1, C_1), (0.3, C_2)], np.array(g), transposed)

    np.testing.assert_allclose(material.apply(g, transposed=transposed), expected, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize('dim, element_type, nb_pixels', cases)
def test_apply_out(dim, element_type, nb_pixels):
    """out= gives the returned result, also when out is the input itself."""
    rng = np.random.default_rng(1)
    discretization = _discretization(dim, element_type, nb_pixels, 'elasticity')
    gradient = discretization.get_gradient_size_field(name='test_gradient')
    gradient.s[...] = rng.normal(size=gradient.s.shape)
    C = rng.normal(size=(dim,) * 4)
    s = rng.uniform(0.1, 2.0, size=gradient.s.shape[2:])
    material = domain.ScaledMaterial([(s, C)])
    expected = material.apply(gradient.s)

    out = np.zeros_like(expected)
    assert material.apply(gradient.s, out=out) is out
    np.testing.assert_allclose(out, expected, rtol=1e-14, atol=0)

    material.apply(gradient.s, out=gradient.s)                    # in place
    np.testing.assert_allclose(gradient.s, expected, rtol=1e-14, atol=0)


@pytest.mark.parametrize('dim, element_type, nb_pixels', cases)
def test_scale_types(dim, element_type, nb_pixels):
    """float, ndarray, muGrid quadrature field and callable scales give the same result; a field or callable is read
    at every application (not when the material is built)."""
    rng = np.random.default_rng(2)
    discretization = _discretization(dim, element_type, nb_pixels, 'elasticity')
    g = rng.normal(size=discretization.get_gradient_size_field(name='test_gradient').s.shape)
    C = _isotropic(dim)
    s = rng.uniform(0.1, 2.0, size=g.shape[2:])
    scale_field = discretization.get_quad_field_scalar(name='test_scale')

    by_array = domain.ScaledMaterial([(s, C)])
    by_field = domain.ScaledMaterial([(scale_field, C)])
    by_callable = domain.ScaledMaterial([(lambda: 1.0 / scale_field.s[0, 0], C)])
    scale_field.s[0, 0] = s                                       # set after the materials were built
    expected = _reference([(s, C)], g, transposed=False)

    np.testing.assert_allclose(by_array.apply(g), expected, rtol=1e-12)
    np.testing.assert_allclose(by_field.apply(g), expected, rtol=1e-12)
    np.testing.assert_allclose(by_callable.apply(g), _reference([(1.0 / s, C)], g, transposed=False), rtol=1e-12)
    np.testing.assert_allclose(domain.ScaledMaterial([(2.5, C)]).apply(g), 2.5 * _reference([(1.0, C)], g, False),
                               rtol=1e-12)


@pytest.mark.parametrize('dim', [2, 3])
def test_minor_symmetric_flag(dim):
    C = _isotropic(dim)
    rng = np.random.default_rng(3)
    assert domain.ScaledMaterial([(1.0, C)]).minor_symmetric
    assert domain.ScaledMaterial([(1.0, C), (0.5, 2 * C)]).minor_symmetric
    assert not domain.ScaledMaterial([(1.0, C), (0.5, rng.normal(size=(dim,) * 4))]).minor_symmetric
    assert not domain.ScaledMaterial([(1.0, np.eye(dim))]).minor_symmetric      # conductivity


def _full_field(discretization, s, C):
    """The same material as ScaledMaterial([(s, C)]) stored as a full tangent per quadrature point."""
    field = discretization.get_material_data_size_field_mugrid(name='test_material_full')
    field.s[...] = np.einsum('ijkl,...->ijkl...', C, s)
    return field


@pytest.mark.parametrize('dim, element_type, nb_pixels', cases)
@pytest.mark.parametrize('formulation', ['small_strain', 'finite_strain'])
def test_system_matrix_matches_full_field(dim, element_type, nb_pixels, formulation):
    """K u with a (minor-symmetric) ScaledMaterial, which skips sym(grad u) for small strain, equals K u with the
    full tangent field, which symmetrises."""
    rng = np.random.default_rng(4)
    discretization = _discretization(dim, element_type, nb_pixels, 'elasticity')
    C = _isotropic(dim)
    s = rng.uniform(0.1, 2.0, size=discretization.get_quad_field_scalar(name='test_shape').s.shape[2:])
    scaled, full = domain.ScaledMaterial([(s, C)]), _full_field(discretization, s, C)
    assert scaled.minor_symmetric

    u = discretization.get_unknown_size_field(name='test_u')
    u.s[...] = rng.normal(size=u.s.shape)
    Ku_scaled = discretization.get_unknown_size_field(name='test_Ku_scaled')
    Ku_full = discretization.get_unknown_size_field(name='test_Ku_full')
    discretization.apply_system_matrix_mugrid(scaled, u, Ku_scaled, formulation=formulation)
    discretization.apply_system_matrix_mugrid(full, u, Ku_full, formulation=formulation)

    np.testing.assert_allclose(Ku_scaled.s, Ku_full.s, rtol=1e-12, atol=1e-12 * np.abs(Ku_full.s).max())


@pytest.mark.parametrize('dim, element_type, nb_pixels', cases)
def test_stress_field_matches_full_field(dim, element_type, nb_pixels):
    """sigma = C : (E + sym(grad u)) with ScaledMaterial (written in place) equals the full tangent field result."""
    rng = np.random.default_rng(5)
    discretization = _discretization(dim, element_type, nb_pixels, 'elasticity')
    C = _isotropic(dim)
    s = rng.uniform(0.1, 2.0, size=discretization.get_quad_field_scalar(name='test_shape').s.shape[2:])
    scaled, full = domain.ScaledMaterial([(s, C)]), _full_field(discretization, s, C)

    u = discretization.get_unknown_size_field(name='test_u')
    u.s[...] = rng.normal(size=u.s.shape)
    E = discretization.get_gradient_size_field(name='test_E')
    E_macro = rng.normal(size=(dim, dim))
    E.s[...] = (0.5 * (E_macro + E_macro.T))[(...,) + (None,) * (E.s.ndim - 2)]
    stress_scaled = discretization.get_gradient_size_field(name='test_stress_scaled')
    stress_full = discretization.get_gradient_size_field(name='test_stress_full')
    for material, stress in ((scaled, stress_scaled), (full, stress_full)):
        discretization.get_stress_field_mugrid(material_data_field_ijklqxyz=material, displacement_field_inxyz=u,
                                               macro_gradient_field_ijqxyz=E, output_stress_field_ijqxyz=stress,
                                               formulation='small_strain')

    np.testing.assert_allclose(stress_scaled.s, stress_full.s, rtol=1e-12, atol=1e-12 * np.abs(stress_full.s).max())


@pytest.mark.parametrize('dim, element_type, nb_pixels', cases)
def test_system_matrix_small_strain_not_minor_symmetric(dim, element_type, nb_pixels):
    """For a C without minor symmetry the small-strain system matrix must still symmetrise the gradient:
    C : sym(grad u) = C_R : grad u with C_R,ijkl = (C_ijkl + C_ijlk) / 2, so K_small(C) u = K_full_gradient(C_R) u."""
    rng = np.random.default_rng(6)
    discretization = _discretization(dim, element_type, nb_pixels, 'elasticity')
    C = rng.normal(size=(dim,) * 4)
    C_R = 0.5 * (C + C.transpose(0, 1, 3, 2))
    s = rng.uniform(0.1, 2.0, size=discretization.get_quad_field_scalar(name='test_shape').s.shape[2:])
    assert not domain.ScaledMaterial([(s, C)]).minor_symmetric

    u = discretization.get_unknown_size_field(name='test_u')
    u.s[...] = rng.normal(size=u.s.shape)
    Ku_small = discretization.get_unknown_size_field(name='test_Ku_small')
    Ku_R = discretization.get_unknown_size_field(name='test_Ku_R')
    discretization.apply_system_matrix_mugrid(domain.ScaledMaterial([(s, C)]), u, Ku_small, formulation='small_strain')
    discretization.apply_system_matrix_mugrid(domain.ScaledMaterial([(s, C_R)]), u, Ku_R, formulation='finite_strain')

    np.testing.assert_allclose(Ku_small.s, Ku_R.s, rtol=1e-12, atol=1e-12 * np.abs(Ku_R.s).max())
