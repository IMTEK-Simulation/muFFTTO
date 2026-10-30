"""Tests of domain.ScaledMaterial: C(x) = sum_k s_k(x) C_k.

The reference is always the plain contraction with the full tangent per quadrature point (einsum, or the
non-ScaledMaterial code path of the Discretization), for
  - apply() and apply(transposed=True) on elasticity and conductivity gradients, 2D and 3D, both on plain numpy
    arrays and on muGrid fields (muGrid's memory layout, where the fast path works on views),
  - apply(out=...), also in place (out = the input),
  - the scale types: float, ndarray, muGrid quadrature field, callable,
  - the minor_symmetric flag and the small-strain system matrix / stress field, which skip the symmetrisation of
    the gradient for a minor-symmetric C.

Every test runs on the CPU and, if cupy finds a GPU, on the GPU. All checks are local to the MPI rank, so the file
also runs under mpirun (e.g. ``mpirun -n 2 python -m pytest test/test_scaled_material.py``); the system matrix and
stress field then include the ghost exchange between ranks.
"""
import numpy as np
import pytest

from muFFTTO import domain
from muFFTTO import material_models
from muFFTTO.tensor_operations import to_host

try:
    import cupy
    cupy.cuda.runtime.getDeviceCount()
    DEVICES = ('cpu', 'gpu')
except Exception:  # no cupy or no GPU
    cupy = None
    DEVICES = ('cpu',)

cases = [(2, 'bilinear_rectangle', [6, 8]),
         (3, 'trilinear_hexahedron', [4, 6, 8])]


def _discretization(dim, element_type, nb_pixels, problem_type, device):
    cell = domain.PeriodicUnitCell(domain_size=[1.3, 0.7, 1.1][:dim], problem_type=problem_type)
    return domain.Discretization(cell=cell, nb_of_pixels_global=nb_pixels, discretization_type='finite_element',
                                 element_type=element_type, device=device)


def _xp(device):
    return cupy if device == 'gpu' else np


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


@pytest.mark.parametrize('device', DEVICES)
@pytest.mark.parametrize('dim, element_type, nb_pixels', cases)
@pytest.mark.parametrize('problem_type', ['elasticity', 'conductivity'])
@pytest.mark.parametrize('transposed', [False, True])
@pytest.mark.parametrize('layout', ['contiguous', 'mugrid'])
def test_apply_matches_einsum(device, dim, element_type, nb_pixels, problem_type, transposed, layout):
    """Two terms (a field scale and a float scale) with general (non-symmetric) tensors."""
    xp = _xp(device)
    rng = np.random.default_rng(0)
    discretization = _discretization(dim, element_type, nb_pixels, problem_type, device)
    gradient = discretization.get_gradient_size_field(name='test_gradient')
    g_host = rng.normal(size=gradient.s.shape)
    if layout == 'contiguous':
        g = xp.asarray(g_host)                            # C-ordered array: the general (copying) path
    else:
        gradient.s[...] = xp.asarray(g_host)
        g = gradient.s                                    # muGrid layout: the view path
    tensor_shape = (dim,) * (4 if problem_type == 'elasticity' else 2)
    C_1, C_2 = rng.normal(size=tensor_shape), rng.normal(size=tensor_shape)
    s_1 = rng.uniform(0.1, 2.0, size=g.shape[2:])

    material = domain.ScaledMaterial([(xp.asarray(s_1), C_1), (0.3, C_2)])
    expected = _reference([(s_1, C_1), (0.3, C_2)], g_host, transposed)

    np.testing.assert_allclose(to_host(material.apply(g, transposed=transposed)), expected, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize('device', DEVICES)
@pytest.mark.parametrize('dim, element_type, nb_pixels', cases)
def test_apply_out(device, dim, element_type, nb_pixels):
    """out= gives the returned result, also when out is the input itself."""
    xp = _xp(device)
    rng = np.random.default_rng(1)
    discretization = _discretization(dim, element_type, nb_pixels, 'elasticity', device)
    gradient = discretization.get_gradient_size_field(name='test_gradient')
    gradient.s[...] = xp.asarray(rng.normal(size=gradient.s.shape))
    C = rng.normal(size=(dim,) * 4)
    s = xp.asarray(rng.uniform(0.1, 2.0, size=gradient.s.shape[2:]))
    material = domain.ScaledMaterial([(s, C)])
    expected = to_host(material.apply(gradient.s))

    out = xp.zeros(expected.shape)
    assert material.apply(gradient.s, out=out) is out
    np.testing.assert_allclose(to_host(out), expected, rtol=1e-14, atol=0)

    material.apply(gradient.s, out=gradient.s)                    # in place
    np.testing.assert_allclose(to_host(gradient.s), expected, rtol=1e-14, atol=0)


@pytest.mark.parametrize('device', DEVICES)
@pytest.mark.parametrize('dim, element_type, nb_pixels', cases)
def test_scale_types(device, dim, element_type, nb_pixels):
    """float, ndarray, muGrid quadrature field and callable scales give the same result; a field or callable is read
    at every application (not when the material is built)."""
    xp = _xp(device)
    rng = np.random.default_rng(2)
    discretization = _discretization(dim, element_type, nb_pixels, 'elasticity', device)
    g_host = rng.normal(size=discretization.get_gradient_size_field(name='test_gradient').s.shape)
    g = xp.asarray(g_host)
    C = _isotropic(dim)
    s = rng.uniform(0.1, 2.0, size=g.shape[2:])
    scale_field = discretization.get_quad_field_scalar(name='test_scale')

    by_array = domain.ScaledMaterial([(xp.asarray(s), C)])
    by_field = domain.ScaledMaterial([(scale_field, C)])
    by_callable = domain.ScaledMaterial([(lambda: 1.0 / scale_field.s[0, 0], C)])
    scale_field.s[0, 0] = xp.asarray(s)                           # set after the materials were built
    expected = _reference([(s, C)], g_host, transposed=False)

    np.testing.assert_allclose(to_host(by_array.apply(g)), expected, rtol=1e-12)
    np.testing.assert_allclose(to_host(by_field.apply(g)), expected, rtol=1e-12)
    np.testing.assert_allclose(to_host(by_callable.apply(g)), _reference([(1.0 / s, C)], g_host, transposed=False),
                               rtol=1e-12)
    np.testing.assert_allclose(to_host(domain.ScaledMaterial([(2.5, C)]).apply(g)),
                               2.5 * _reference([(1.0, C)], g_host, False), rtol=1e-12)


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
    field.s[...] = _xp(discretization.device).asarray(np.einsum('ijkl,...->ijkl...', C, s))
    return field


def _assert_fields_close(actual, expected):
    actual, expected = to_host(actual.s), to_host(expected.s)
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12 * np.abs(expected).max())


@pytest.mark.parametrize('device', DEVICES)
@pytest.mark.parametrize('dim, element_type, nb_pixels', cases)
@pytest.mark.parametrize('formulation', ['small_strain', 'finite_strain'])
def test_system_matrix_matches_full_field(device, dim, element_type, nb_pixels, formulation):
    """K u with a (minor-symmetric) ScaledMaterial, which skips sym(grad u) for small strain, equals K u with the
    full tangent field, which symmetrises."""
    rng = np.random.default_rng(4)
    discretization = _discretization(dim, element_type, nb_pixels, 'elasticity', device)
    C = _isotropic(dim)
    s = rng.uniform(0.1, 2.0, size=discretization.get_quad_field_scalar(name='test_shape').s.shape[2:])
    scaled, full = domain.ScaledMaterial([(_xp(device).asarray(s), C)]), _full_field(discretization, s, C)
    assert scaled.minor_symmetric

    u = discretization.get_unknown_size_field(name='test_u')
    u.s[...] = _xp(device).asarray(rng.normal(size=u.s.shape))
    Ku_scaled = discretization.get_unknown_size_field(name='test_Ku_scaled')
    Ku_full = discretization.get_unknown_size_field(name='test_Ku_full')
    discretization.apply_system_matrix_mugrid(scaled, u, Ku_scaled, formulation=formulation)
    discretization.apply_system_matrix_mugrid(full, u, Ku_full, formulation=formulation)

    _assert_fields_close(Ku_scaled, Ku_full)


@pytest.mark.parametrize('device', DEVICES)
@pytest.mark.parametrize('dim, element_type, nb_pixels', cases)
def test_stress_field_matches_full_field(device, dim, element_type, nb_pixels):
    """sigma = C : (E + sym(grad u)) with ScaledMaterial (written in place) equals the full tangent field result."""
    rng = np.random.default_rng(5)
    discretization = _discretization(dim, element_type, nb_pixels, 'elasticity', device)
    C = _isotropic(dim)
    s = rng.uniform(0.1, 2.0, size=discretization.get_quad_field_scalar(name='test_shape').s.shape[2:])
    scaled, full = domain.ScaledMaterial([(_xp(device).asarray(s), C)]), _full_field(discretization, s, C)

    u = discretization.get_unknown_size_field(name='test_u')
    u.s[...] = _xp(device).asarray(rng.normal(size=u.s.shape))
    E = discretization.get_gradient_size_field(name='test_E')
    E_macro = rng.normal(size=(dim, dim))
    E.s[...] = _xp(device).asarray((0.5 * (E_macro + E_macro.T))[(...,) + (None,) * (E.s.ndim - 2)])
    stress_scaled = discretization.get_gradient_size_field(name='test_stress_scaled')
    stress_full = discretization.get_gradient_size_field(name='test_stress_full')
    for material, stress in ((scaled, stress_scaled), (full, stress_full)):
        discretization.get_stress_field_mugrid(material_data_field_ijklqxyz=material, displacement_field_inxyz=u,
                                               macro_gradient_field_ijqxyz=E, output_stress_field_ijqxyz=stress,
                                               formulation='small_strain')

    _assert_fields_close(stress_scaled, stress_full)


@pytest.mark.parametrize('device', DEVICES)
@pytest.mark.parametrize('dim, element_type, nb_pixels', cases)
def test_system_matrix_small_strain_not_minor_symmetric(device, dim, element_type, nb_pixels):
    """For a C without minor symmetry the small-strain system matrix must still symmetrise the gradient:
    C : sym(grad u) = C_R : grad u with C_R,ijkl = (C_ijkl + C_ijlk) / 2, so K_small(C) u = K_full_gradient(C_R) u."""
    rng = np.random.default_rng(6)
    discretization = _discretization(dim, element_type, nb_pixels, 'elasticity', device)
    C = rng.normal(size=(dim,) * 4)
    C_R = 0.5 * (C + C.transpose(0, 1, 3, 2))
    s = rng.uniform(0.1, 2.0, size=discretization.get_quad_field_scalar(name='test_shape').s.shape[2:])
    s = _xp(device).asarray(s)
    assert not domain.ScaledMaterial([(s, C)]).minor_symmetric

    u = discretization.get_unknown_size_field(name='test_u')
    u.s[...] = _xp(device).asarray(rng.normal(size=u.s.shape))
    Ku_small = discretization.get_unknown_size_field(name='test_Ku_small')
    Ku_R = discretization.get_unknown_size_field(name='test_Ku_R')
    discretization.apply_system_matrix_mugrid(domain.ScaledMaterial([(s, C)]), u, Ku_small, formulation='small_strain')
    discretization.apply_system_matrix_mugrid(domain.ScaledMaterial([(s, C_R)]), u, Ku_R, formulation='finite_strain')

    _assert_fields_close(Ku_small, Ku_R)
