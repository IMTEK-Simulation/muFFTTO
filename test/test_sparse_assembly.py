"""muFFTTO.sparse_assembly: the assembled matrix equals the matrix-free operator B^T W tangent B."""
import numpy as np
import pytest

from muFFTTO import domain, sparse_assembly


def make_discretization(element_type, nb_pixels=(6, 5)):
    cell = domain.PeriodicUnitCell(domain_size=[1.2, 1.0], problem_type='elasticity')
    return domain.Discretization(cell=cell, nb_of_pixels_global=nb_pixels, element_type=element_type)


def heterogeneous_tangent(discretization, pixel_mean):
    """A nonsymmetric-in-ij but major-symmetric tangent C(x, q) (random, SPD) and, optionally, a volumetric part that
    acts on the pixel mean of dF (as selective reduced integration)."""
    rng = np.random.default_rng(3)
    shape = tuple(discretization.nb_of_pixels)
    nb_q = len(discretization.quadrature_weights)
    A = rng.standard_normal((4, 4) + (nb_q,) + shape)
    C = np.einsum('mk...,nk...->mn...', A, A).reshape((2, 2, 2, 2, nb_q) + shape)      # C_ijkl, major symmetric
    lam = 1.0 + rng.random(shape)

    def tangent(dF):
        dP = np.einsum('ijkl...,kl...->ij...', C, dF)
        if pixel_mean:
            w = np.asarray(discretization.quadrature_weights)
            dF0 = np.einsum('ijq...,q->ij...', dF, w / w.sum())[:, :, None]   # weighted pixel mean (symmetric)
            dP = dP + lam * np.einsum('ii...->...', dF0) * np.eye(2)[:, :, None, None, None]
        return dP
    return tangent


def matrix_free(discretization, assembly, tangent, v):
    """B^T W tangent(B v) with the discretization's own operators."""
    grad = assembly.gradient(v)
    field = discretization.get_gradient_size_field(name=f'dP_{np.random.default_rng().integers(1e9)}')
    field.s[...] = tangent(grad)
    out = discretization.get_unknown_size_field(name=f'out_{np.random.default_rng().integers(1e9)}')
    discretization.fft.communicate_ghosts(field)
    discretization.apply_gradient_transposed_operator_mugrid(gradient_field_ijqxyz=field, div_u_fnxyz=out,
                                                             apply_weights=True)
    return out.s.ravel().copy()


@pytest.mark.parametrize('element_type', ['bilinear_rectangle', 'biquadratic_rectangle'])
@pytest.mark.parametrize('pixel_mean', [False, True])
def test_assembled_equals_matrix_free(element_type, pixel_mean):
    discretization = make_discretization(element_type)
    assembly = sparse_assembly.SparseAssembly(discretization)
    tangent = heterogeneous_tangent(discretization, pixel_mean)
    K = assembly.assemble(tangent)
    assert len(assembly.element_unknowns) == {'bilinear_rectangle': 8, 'biquadratic_rectangle': 18}[element_type]
    assert abs(K - K.T).max() <= 1e-13 * abs(K).max()
    rng = np.random.default_rng(1)
    for _ in range(3):
        v = rng.standard_normal(assembly.nb_unknowns)
        Kv = matrix_free(discretization, assembly, tangent, v)
        np.testing.assert_allclose(K @ v, Kv, rtol=0, atol=1e-12 * np.abs(Kv).max())


def test_pattern_is_element_connectivity():
    """Q1: a node couples with the 9 nodes of the 4 pixels around it (2 components): 18 entries per row."""
    discretization = make_discretization('bilinear_rectangle')
    assembly = sparse_assembly.SparseAssembly(discretization)
    assert np.all(np.diff(assembly.pattern.indptr) == 18)


def test_bordered():
    K = make_discretization('bilinear_rectangle')
    K = sparse_assembly.SparseAssembly(K).assemble(lambda dF: dF)
    column = np.arange(K.shape[0] + 1, dtype=float)
    H = sparse_assembly.bordered(K, column)
    assert H.shape == (K.shape[0] + 1,) * 2
    np.testing.assert_array_equal(H[:, -1].toarray().ravel(), column)
    np.testing.assert_array_equal(H[-1, :].toarray().ravel(), column)
    assert abs(H[:-1, :-1] - K).max() == 0
