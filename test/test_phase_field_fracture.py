"""muFFTTO.phase_field_fracture: AT1/AT2 material models."""
import numpy as np
import pytest

from muFFTTO import domain, phase_field_fracture, preconditioners, solvers


def test_degradation():
    d = np.array([0.0, 0.5, 1.0])
    np.testing.assert_allclose(phase_field_fracture.degradation(d, k=1e-4), [1.0001, 0.2501, 1e-4])
    np.testing.assert_allclose(phase_field_fracture.degradation_derivative(d), [-2.0, -1.0, 0.0])


def test_crack_density_coefficients():
    Gc, l = 2.0, 0.1
    at2 = phase_field_fracture.CrackDensity('AT2', Gc, l)
    assert (at2.diffusion, at2.reaction, at2.rhs_offset, at2.threshold) == pytest.approx((Gc * l, Gc / l, 0, 0))
    at1 = phase_field_fracture.CrackDensity('AT1', Gc, l)
    assert (at1.diffusion, at1.reaction, at1.rhs_offset, at1.threshold) == pytest.approx(
        (0.75 * Gc * l, 0.0, 3 * Gc / (8 * l), 3 * Gc / (16 * l)))
    with pytest.raises(ValueError):
        phase_field_fracture.CrackDensity('AT3', Gc, l)


def test_energy_density_of_a_sharp_profile():
    """AT2 optimal 1D profile d = exp(-|x|/l) dissipates Gc per unit crack length."""
    Gc, l = 1.5, 0.05
    x = np.linspace(-1, 1, 200001)
    d = np.exp(-np.abs(x) / l)
    grad = np.gradient(d, x)[np.newaxis]
    density = phase_field_fracture.CrackDensity('AT2', Gc, l).energy_density(d, grad)
    assert np.trapezoid(density, x) == pytest.approx(Gc, rel=1e-3)


@pytest.mark.parametrize('dim', [2, 3])
def test_elastic_energy_splits(dim):
    rng = np.random.default_rng(0)
    A = rng.standard_normal((dim, dim, 50))
    strain = 0.5 * (A + A.transpose(1, 0, 2))
    lam, mu = 1.3, 0.7
    trace = np.einsum('ii...->...', strain)
    psi_0_ref = 0.5 * lam * trace ** 2 + mu * np.einsum('ij...,ij...->...', strain, strain)

    psi_0, psi_plus = phase_field_fracture.elastic_energy(strain, lam, mu, split='none')
    np.testing.assert_allclose(psi_0, psi_0_ref)
    np.testing.assert_allclose(psi_plus, psi_0_ref)

    # Amor: psi^- = K/2 <tr>_-^2 (only the compressive volumetric part)
    psi_0, psi_plus = phase_field_fracture.elastic_energy(strain, lam, mu, split='amor')
    bulk = lam + 2 * mu / dim
    np.testing.assert_allclose(psi_0 - psi_plus, 0.5 * bulk * np.minimum(trace, 0) ** 2, atol=1e-12)

    # spectral: psi^- = lam/2 <tr>_-^2 + mu sum <eps_i>_-^2
    psi_0, psi_plus = phase_field_fracture.elastic_energy(strain, lam, mu, split='spectral')
    principal = np.linalg.eigvalsh(np.moveaxis(strain, (0, 1), (-2, -1)))
    psi_minus = 0.5 * lam * np.minimum(trace, 0) ** 2 + mu * np.sum(np.minimum(principal, 0) ** 2, axis=-1)
    np.testing.assert_allclose(psi_0 - psi_plus, psi_minus, atol=1e-12)

    # pure compression does not drive the crack (Amor, spectral)
    compression = -np.eye(dim)[:, :, np.newaxis]
    for split in ('amor', 'spectral'):
        _, psi_plus = phase_field_fracture.elastic_energy(compression, lam, mu, split=split)
        np.testing.assert_allclose(psi_plus, 0.0)
    with pytest.raises(ValueError):
        phase_field_fracture.elastic_energy(strain, lam, mu, split='unknown')


def test_update_history():
    history = np.zeros(4)
    previous = np.array([0.0, 2.0, 0.5, 0.0])
    psi = np.array([1.0, 1.0, 0.1, 0.0])
    phase_field_fracture.update_history(history, previous, psi, threshold=0.2)
    np.testing.assert_allclose(history, [1.0, 2.0, 0.5, 0.2])


@pytest.mark.parametrize('model', ['AT2', 'AT1'])
def test_damage_equation_for_constant_history(model):
    """Constant H: the damage is constant, d = (2H - rhs_offset) / (reaction + 2H)."""
    cell = domain.PeriodicUnitCell(domain_size=[1.0, 1.0], problem_type='conductivity')
    disc = domain.Discretization(cell=cell, nb_of_pixels_global=(16, 16), discretization_type='finite_element',
                                 element_type='linear_triangles')
    density = phase_field_fracture.CrackDensity(model, Gc=1.0, length_scale=0.1)
    H = 7.0
    history = disc.get_quad_field_scalar(name='history')
    history.s[...] = H
    reaction = density.reaction_field(history.s)
    work = (disc.get_gradient_size_field(name='work_gradient'), disc.get_quad_field_scalar(name='work_quad'),
            disc.get_unknown_size_field(name='work_nodal'))

    def K(x, Kx):
        density.apply_operator(disc, x, Kx, reaction, *work)

    def K_reference(input_field_inxyz, output_field_inxyz):
        density.apply_operator(disc, input_field_inxyz, output_field_inxyz, density.preconditioner_reaction, *work)

    rhs = disc.get_unknown_size_field(name='rhs')
    density.rhs(disc, history.s, rhs, work[1])
    M = preconditioners.Green(disc, operator=K_reference, invert_zero_mode=True, name='green_damage')

    d = disc.get_unknown_size_field(name='damage')
    d.s[...] = 0.0
    solvers.conjugate_gradients(disc.communicator, disc.field_collection, K, rhs, d, M, tol=1e-10)
    np.testing.assert_allclose(d.s, (2 * H - density.rhs_offset) / (density.reaction + 2 * H), rtol=1e-8)
