"""
Tests of the one-point hexahedron with Flanagan-Belytschko hourglass control (element 'trilinear_hexahedron_1Q_hg',
muFFTTO.hourglass_control).
"""
import numpy as np
import pytest

from muFFTTO import domain, material_models
from muFFTTO.hourglass_control import HourglassControl, hourglass_base_vectors

C_0 = material_models.get_elastic_material_tensor(dim=3, K=1.0, mu=0.5, kind='linear')


def make(element, N, eps=None):
    cell = domain.PeriodicUnitCell(domain_size=[1, 1, 1], problem_type='elasticity')
    return domain.Discretization(cell=cell, nb_of_pixels_global=(N, N, N), discretization_type='finite_element',
                                 element_type=element, hourglass_eps=eps)


def K_apply(disc, material, u_array):
    u = disc.get_unknown_size_field(name='test_u')
    Ku = disc.get_unknown_size_field(name='test_Ku')
    u.s[...] = u_array
    disc.apply_system_matrix_mugrid(material_data_field=material, input_field_inxyz=u, output_field_inxyz=Ku,
                                    formulation='small_strain')
    return np.array(Ku.s)


def global_hourglass_modes(N):
    """the 4 x 3 periodic fields equal to +- h_alpha in one component on every voxel"""
    x = np.arange(N)
    X, Y, Z = np.meshgrid(x, x, x, indexing='ij')
    for sign in [(-1.0) ** (Y + Z), (-1.0) ** (Z + X), (-1.0) ** (X + Y), (-1.0) ** (X + Y + Z)]:
        for comp in range(3):
            u = np.zeros((3, 1, N, N, N))
            u[comp, 0] = sign
            yield u


def test_base_vectors_orthogonal_to_linear_fields():
    h = hourglass_base_vectors().reshape(4, 8)
    corners = np.array([[2 * i - 1, 2 * j - 1, 2 * k - 1] for i in (0, 1) for j in (0, 1) for k in (0, 1)])
    linear = np.column_stack([np.ones(8), corners])                      # 1, xi, eta, zeta at the corners
    np.testing.assert_allclose(h @ linear, 0.0, atol=1e-14)
    np.testing.assert_allclose(h @ h.T, 8 * np.eye(4), atol=1e-14)


@pytest.mark.parametrize('N', [4, 8])
def test_eps_one_reproduces_full_integration_per_mode(N):
    """eps = 1: one-point + control has exactly the hourglass energy of the 2x2x2 element, mode by mode, any grid"""
    full, hg, one = make('trilinear_hexahedron', N), make('trilinear_hexahedron_1Q_hg', N, eps=1.0), \
        make('trilinear_hexahedron_1Q', N)
    for u in global_hourglass_modes(N):
        e_full = np.sum(u * K_apply(full, C_0, u))
        e_one = np.sum(u * K_apply(one, C_0, u))
        e_hg = np.sum(u * K_apply(hg, C_0, u))
        assert abs(e_one) < 1e-10 * e_full                               # the one-point element: zero energy
        assert e_hg == pytest.approx(e_full, rel=1e-12)


def test_zero_on_constant_strain_and_symmetric_psd():
    N = 6
    disc = make('trilinear_hexahedron_1Q_hg', N)
    rng = np.random.default_rng(0)
    s = 10.0 ** rng.uniform(-3, 0, size=(1, N, N, N))
    scale = disc.get_quad_field_scalar(name='scale')
    scale.s[0, 0] = s
    material = domain.ScaledMaterial([(scale, C_0)])
    control = disc.hourglass_control
    q = control.strain
    # hourglass strain of a linear displacement vanishes on the voxels away from the periodic wrap
    x = np.arange(N) / N
    lin = np.zeros((3, 1, N, N, N))
    lin[0, 0] = 0.7 * x[:, None, None] + 0.2 * x[None, :, None] - 0.4 * x[None, None, :]
    lin[1, 0] = 0.3 * x[None, None, :] + 0.0 * x[:, None, None]
    u = disc.get_unknown_size_field(name='lin')
    u.s[...] = lin
    disc.fft.communicate_ghosts(u)
    control.operator.apply(nodal_field=u, quadrature_point_field=q)
    assert np.abs(np.array(q.s)[..., :-1, :-1, :-1]).max() < 1e-14
    # K_hg symmetric and positive semi-definite
    one = make('trilinear_hexahedron_1Q', N)
    scale_one = one.get_quad_field_scalar(name='scale')
    scale_one.s[0, 0] = s
    material_one = domain.ScaledMaterial([(scale_one, C_0)])
    a, b = rng.standard_normal((3, 1, N, N, N)), rng.standard_normal((3, 1, N, N, N))
    K_hg = lambda v: K_apply(disc, material, v) - K_apply(one, material_one, v)
    assert np.sum(b * K_hg(a)) == pytest.approx(np.sum(a * K_hg(b)), rel=1e-12)
    assert np.sum(a * K_hg(a)) > 0


def test_material_representations_agree():
    """ScaledMaterial, a full material field and a constant tensor give the same control"""
    N = 4
    disc = make('trilinear_hexahedron_1Q_hg', N)
    rng = np.random.default_rng(1)
    s = 10.0 ** rng.uniform(-3, 0, size=(1, N, N, N))
    scale = disc.get_quad_field_scalar(name='scale')
    scale.s[0, 0] = s
    scaled = domain.ScaledMaterial([(scale, C_0 - 0.01 * C_0), (1.0, 0.01 * C_0)])
    full = disc.get_material_data_size_field_mugrid(name='full')
    full.s[...] = np.einsum('ijkl,q...->ijklq...', C_0, 0.99 * s + 0.01)
    u = rng.standard_normal((3, 1, N, N, N))
    np.testing.assert_allclose(K_apply(disc, scaled, u), K_apply(disc, full, u), rtol=1e-12, atol=1e-14)
    const = disc.get_material_data_size_field_mugrid(name='const')
    const.s[...] = C_0[..., None, None, None, None]
    np.testing.assert_allclose(K_apply(disc, C_0, u), K_apply(disc, const, u), rtol=1e-12, atol=1e-14)


def test_plain_one_point_element_unchanged():
    assert make('trilinear_hexahedron_1Q', 4).hourglass_control is None
    assert make('trilinear_hexahedron', 4).hourglass_control is None
    assert isinstance(make('trilinear_hexahedron_1Q_hg', 4).hourglass_control, HourglassControl)
    assert make('trilinear_hexahedron_1Q_hg', 4).hourglass_control.eps == 0.3


def test_design_derivative_matches_finite_differences():
    """design_derivative(dC/ds, u, lambda) = d/ds_voxel (lambda^T K_hg(s) u), K_hg of the material s(x) C_0"""
    N = 4
    disc = make('trilinear_hexahedron_1Q_hg', N, eps=0.3)
    control = disc.hourglass_control
    rng = np.random.default_rng(2)
    s = 10.0 ** rng.uniform(-2, 0, size=(1, N, N, N))
    scale = disc.get_quad_field_scalar(name='scale')
    u, lam, out = (disc.get_unknown_size_field(name=n) for n in ('u_fd', 'lam_fd', 'out_fd'))
    u.s[...] = rng.standard_normal(u.s.shape)
    lam.s[...] = rng.standard_normal(lam.s.shape)

    def energy(s_array):
        scale.s[0, 0] = s_array
        out.s[...] = 0.0
        control.add_stiffness(domain.ScaledMaterial([(scale, C_0)]), u, out)
        return float(np.sum(np.array(lam.s) * np.array(out.s)))

    derivative = np.array(control.design_derivative(domain.ScaledMaterial([(1.0, C_0)]), u, lam))   # [1, x, y, z]
    for voxel in [(0, 0, 0, 0), (0, 1, 2, 3), (0, 3, 3, 1)]:
        delta = np.zeros_like(s)
        delta[voxel] = 1e-6
        fd = (energy(s + delta) - energy(s - delta)) / 2e-6
        assert derivative[voxel] == pytest.approx(fd, rel=1e-6)
