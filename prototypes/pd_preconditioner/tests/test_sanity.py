"""Sanity tests of the operators, preconditioners and PCG (run before any experiment)."""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from fem import B, BT, apply_K, gradient, gradient_transposed, mandel_isotropic, material_field, rhs, weight
from materials import cosine, disk, laminate
from pcg import pcg
from precond import (apply_green, assemble_green, jacobi_diagonal, jacobi_diagonal_brute, make_green,
                     make_green_jacobi, make_primal_dual, new_counter)

rng = np.random.default_rng(0)
C0_ELASTIC = mandel_isotropic(2.0, 1.0)


def setup(d, n, rho):
    C0 = np.eye(2) if d == 1 else C0_ELASTIC
    return C0, material_field(rho, C0)


def mean_free(a):
    return a - a.mean(axis=(1, 2), keepdims=True)


@pytest.mark.parametrize('d', [1, 2])
def test_gradient_and_B_adjoint(d):
    n = 8
    u = rng.normal(size=(d, n, n))
    g = rng.normal(size=(2, d, 2, n, n))
    assert np.isclose(np.vdot(gradient(u), g), np.vdot(u, gradient_transposed(g)), rtol=1e-12)
    s = rng.normal(size=(2, 2 if d == 1 else 3, n, n))
    assert np.isclose(np.vdot(B(u), s), np.vdot(u, BT(s)), rtol=1e-12)


@pytest.mark.parametrize('d', [1, 2])
def test_K_symmetric(d):
    n = 8
    _, C = setup(d, n, cosine(n, 4, 1e4))
    u, v = rng.normal(size=(2, d, n, n))
    assert np.isclose(np.vdot(apply_K(u, C), v), np.vdot(u, apply_K(v, C)), rtol=1e-12)


@pytest.mark.parametrize('d', [1, 2])
def test_green_inverts_Kref(d):
    n = 16
    C0, _ = setup(d, n, np.ones((n, n)))
    Ghat = assemble_green(C0, d, n)
    u = rng.normal(size=(d, n, n))
    np.testing.assert_allclose(apply_green(Ghat, apply_K(u, C0)), mean_free(u), atol=1e-11)


@pytest.mark.parametrize('d', [1, 2])
def test_gamma0_projector(d):
    """Gamma0 = B G B^T W with Cref = identity: projector onto compatible fields, Gamma0 B u = B u."""
    n = 16
    ds = 2 if d == 1 else 3
    I = np.eye(ds)
    Ghat = assemble_green(I, d, n)
    w = weight(n)

    def gamma0(e):
        return B(apply_green(Ghat, BT(w * e)))
    e = rng.normal(size=(2, ds, n, n))
    g = gamma0(e)
    np.testing.assert_allclose(gamma0(g), g, atol=1e-11)
    u = rng.normal(size=(d, n, n))
    np.testing.assert_allclose(gamma0(B(u)), B(u), atol=1e-11)


@pytest.mark.parametrize('d', [1, 2])
def test_jacobi_comb_equals_brute_force(d):
    n = 8
    _, C = setup(d, n, cosine(n, 4, 1e4))
    np.testing.assert_allclose(jacobi_diagonal(C, d, n), jacobi_diagonal_brute(C, d, n), rtol=1e-13)


@pytest.mark.parametrize('d', [1, 2])
@pytest.mark.parametrize('name', ['G', 'GJ', 'PD'])
def test_constant_material_one_iteration(d, name):
    n = 16
    rho = np.full((n, n), 3.0)
    C0, C = setup(d, n, rho)
    counter = new_counter()
    M = {'G': lambda: make_green(C0, d, n, counter), 'GJ': lambda: make_green_jacobi(C, C0, d, n, counter),
         'PD': lambda: make_primal_dual(rho, C0, C0, d, n, counter)}[name]()
    E = np.ones(2 if d == 1 else 3)
    f = rhs(C, E, d, n) + rng.normal(size=(d, n, n))     # nonzero rhs (constant material: E gives f = 0)
    f = mean_free(f)
    result = pcg(C, f, M, counter, tol=1e-10, track_energy=False)
    assert result['iterations'] == 1


@pytest.mark.parametrize('d', [1, 2])
@pytest.mark.parametrize('rho', [lambda n: cosine(n, 4, 1e4), lambda n: disk(n, 1e4, 0),
                                 lambda n: laminate(n, 8, 1e4)])
def test_primal_dual_symmetric_positive(d, rho):
    n = 16
    rho = rho(n)
    C0, _ = setup(d, n, rho)
    counter = new_counter()
    M = make_primal_dual(rho, C0, C0, d, n, counter)
    r, s = mean_free(rng.normal(size=(2, d, n, n)))
    assert np.isclose(np.vdot(M(r), s), np.vdot(r, M(s)), rtol=1e-10)
    assert np.vdot(M(r), r) > 0


@pytest.mark.parametrize('d', [1, 2])
def test_primal_dual_mean_free_without_subtraction(d):
    """G K_S G maps into the mean-free subspace by itself: the mean subtraction is a no-op up to round-off."""
    n = 16
    rho = cosine(n, 4, 1e4)
    C0, _ = setup(d, n, rho)
    M = make_primal_dual(rho, C0, C0, d, n, new_counter(), subtract_mean=False)
    z = M(rng.normal(size=(d, n, n)))
    assert np.abs(z.mean(axis=(1, 2))).max() <= 1e-12 * np.abs(z).max()


def test_modal_dual_reduces_to_old_dual_for_scaled_material():
    """For C = rho C0 (same Poisson ratio everywhere) the modal dual material equals C_ref C^-1 C_ref."""
    from precond import make_primal_dual_modal
    n = 16
    rho = cosine(n, 4, 1e4)
    C0, C = setup(2, n, rho)
    r = mean_free(rng.normal(size=(2, n, n)))
    old = make_primal_dual(rho, C0, C0, 2, n, new_counter())(r)
    modal = make_primal_dual_modal(C, C0, 2, n, new_counter())(r)
    np.testing.assert_allclose(modal, old, rtol=1e-10, atol=1e-12 * np.abs(old).max())


def test_isotropic_projection_exact_for_isotropic():
    from nonlinear import linear_isotropic
    from precond import isotropic_projection
    kappa, mu = isotropic_projection(linear_isotropic(20.0, 2000.0))
    assert np.isclose(kappa, 20.0) and np.isclose(mu, 2000.0)
