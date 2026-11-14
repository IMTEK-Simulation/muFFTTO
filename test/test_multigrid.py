"""
Tests of the geometric multigrid preconditioner muFFTTO.multigrid (V-cycle for A = s(x) C_0).
"""
import numpy as np
import pytest

from muFFTTO import domain, material_models, multigrid

C_0 = {2: material_models.get_elastic_material_tensor(dim=2, K=1.0, mu=0.5, kind='linear'),
       3: material_models.get_elastic_material_tensor(dim=3, K=1.0, mu=0.5, kind='linear')}
NB_QUAD = {'linear_triangles': 2, 'trilinear_hexahedron': 8, 'trilinear_hexahedron_1Q_hg': 1, 'fused': 1,
           'fused_single': 1}


FUSED = {'fused': ('fused', 'double'), 'fused_single': ('fused', 'single')}     # pseudo elements of the test cases


def make(element, N, dim):
    cell = domain.PeriodicUnitCell(domain_size=[1] * dim, problem_type='elasticity')
    operator, precision = FUSED.get(element, ('stencil', 'double'))
    grid_element = 'trilinear_hexahedron_1Q' if element in FUSED else element
    disc = domain.Discretization(cell=cell, nb_of_pixels_global=(N,) * dim, discretization_type='finite_element',
                                 element_type=grid_element, hourglass_eps=0.3 if element.endswith('_hg') else None)
    mg = multigrid.Multigrid(disc, C_0[dim], operator=operator, precision=precision)
    rng = np.random.default_rng(3)
    blocks = rng.uniform(-3, 0, size=(N // 4,) * dim)                       # contrast 1e3 in 4-pixel blocks
    scale = 10.0 ** np.kron(blocks, np.ones((4,) * dim))
    mg.scale.s[0, 0] = scale[None] * np.linspace(1.0, 1.2, NB_QUAD[element]).reshape((-1,) + (1,) * dim)
    mg.setup()
    return disc, mg


def zero_mean(rng, shape, dim):
    v = rng.standard_normal(shape)
    return v - v.mean(axis=tuple(range(-dim, 0)), keepdims=True)


def cycle(disc, mg, r_array):
    r, z = disc.get_unknown_size_field(name='test_r'), disc.get_unknown_size_field(name='test_z')
    r.s[...] = r_array
    mg.apply(r, z)
    return np.array(z.s)


CASES = [('linear_triangles', 16, 2), ('trilinear_hexahedron', 8, 3), ('trilinear_hexahedron_1Q_hg', 8, 3),
         ('fused', 8, 3), ('fused_single', 8, 3)]


@pytest.mark.parametrize('element, N, dim', CASES)
def test_cycle_is_symmetric_positive_definite(element, N, dim):
    """the V-cycle is a fixed SPD operator on the zero-mean vectors (what CG needs of a preconditioner)"""
    disc, mg = make(element, N, dim)
    rng = np.random.default_rng(0)
    shape = disc.get_unknown_size_field(name='test_r').s.shape
    a, b = zero_mean(rng, shape, dim), zero_mean(rng, shape, dim)
    Ma, Mb = cycle(disc, mg, a), cycle(disc, mg, b)
    accuracy = 1e-4 if element == 'fused_single' else 1e-9                      # the float32 cycle is SPD to round-off
    assert np.sum(b * Ma) == pytest.approx(np.sum(a * Mb), rel=10 * accuracy, abs=accuracy * np.sum(a * Ma))
    assert np.sum(a * Ma) > 0 and np.sum(b * Mb) > 0
    assert np.allclose(cycle(disc, mg, 2.0 * a - 3.0 * b), 2.0 * Ma - 3.0 * Mb, rtol=accuracy,
                       atol=accuracy * np.abs(Ma).max())


@pytest.mark.parametrize('element, N, dim', CASES)
def test_cg_converges_in_few_iterations(element, N, dim):
    """PCG with the cycle: residual reduced by 1e-8 in < 25 iterations at contrast 1e3 (Jacobi needs hundreds)"""
    disc, mg = make(element, N, dim)
    rng = np.random.default_rng(1)
    shape = disc.get_unknown_size_field(name='test_r').s.shape
    b = zero_mean(rng, shape, dim)
    p_field, Kp_field = disc.get_unknown_size_field(name='test_p'), disc.get_unknown_size_field(name='test_Kp')

    def K(v):
        p_field.s[...] = v
        mg.apply_K(p_field, Kp_field)
        return np.array(Kp_field.s)

    x, r = np.zeros(shape), b.copy()
    z = cycle(disc, mg, r)
    p, rz = z.copy(), np.sum(r * z)
    for iteration in range(1, 100):
        Kp = K(p)
        alpha = rz / np.sum(p * Kp)
        x, r = x + alpha * p, r - alpha * Kp
        if np.linalg.norm(r) < 1e-8 * np.linalg.norm(b):
            break
        z = cycle(disc, mg, r)
        rz_new = np.sum(r * z)
        p, rz = z + rz_new / rz * p, rz_new
    assert iteration < 25
    assert np.linalg.norm(K(x) - b) < 1e-7 * np.linalg.norm(b)


def test_grid_transfer_arrays_equal_kernel():
    """the array form of the grid transfer (device fields) equals muGrid's kernel, and R = P^T"""
    disc, mg = make('trilinear_hexahedron', 8, 3)
    fine, coarse = mg.levels[0], mg.levels[1]
    rng = np.random.default_rng(2)
    f1, f2 = (fine.disc.get_unknown_size_field(name=n) for n in ('test_f1', 'test_f2'))
    c1, c2 = (coarse.disc.get_unknown_size_field(name=n) for n in ('test_c1', 'test_c2'))
    f1.s[...] = rng.standard_normal(f1.s.shape)
    c1.s[...] = rng.standard_normal(c1.s.shape)
    fine.disc.fft.communicate_ghosts(f1)
    coarse.disc.fft.communicate_ghosts(c1)
    arrays = multigrid.GridTransfer(3)
    arrays.use_kernel = False
    mg.transfer.restrict(f1, c2)
    restricted = np.array(c2.s)
    arrays.restrict(f1, c2)
    assert np.allclose(c2.s, restricted, rtol=1e-13, atol=1e-13)
    mg.transfer.prolong(c1, f2)
    prolonged = np.array(f2.s)
    arrays.prolong(c1, f2)
    assert np.allclose(f2.s, prolonged, rtol=1e-13, atol=1e-13)
    assert np.sum(restricted * c1.s) == pytest.approx(np.sum(f1.s * prolonged), rel=1e-12)       # <R f, c> = <f, P c>


def test_one_point_coarse_levels():
    """coarse_element: one-point hexahedra on all levels still give a convergent SPD cycle"""
    cell = domain.PeriodicUnitCell(domain_size=[1, 1, 1], problem_type='elasticity')
    disc = domain.Discretization(cell=cell, nb_of_pixels_global=(8, 8, 8), discretization_type='finite_element',
                                 element_type='trilinear_hexahedron_1Q_hg', hourglass_eps=0.3)
    mg = multigrid.Multigrid(disc, C_0[3], coarse_element='trilinear_hexahedron_1Q_hg')
    mg.scale.s[...] = 1.0
    mg.setup()
    assert all(level.disc.element_type == 'trilinear_hexahedron_1Q_hg' for level in mg.levels)
    rng = np.random.default_rng(0)
    a = zero_mean(rng, disc.get_unknown_size_field(name='test_r').s.shape, 3)
    assert np.sum(a * cycle(disc, mg, a)) > 0


def test_fused_operator_is_the_voxelwise_8_point_hexahedron():
    """operator='fused' applies the stiffness of 'trilinear_hexahedron' with one material value per voxel"""
    N = 8
    cell = domain.PeriodicUnitCell(domain_size=[1, 1, 1], problem_type='elasticity')
    rng = np.random.default_rng(4)
    s = 10.0 ** rng.uniform(-3, 0, size=(N, N, N))
    u_array = rng.standard_normal((3, 1, N, N, N))
    results = []
    for element, operator in (('trilinear_hexahedron', 'stencil'), ('trilinear_hexahedron_1Q', 'fused')):
        disc = domain.Discretization(cell=cell, nb_of_pixels_global=(N,) * 3, discretization_type='finite_element',
                                     element_type=element)
        mg = multigrid.Multigrid(disc, C_0[3], operator=operator)
        mg.scale.s[0, 0] = s[None]
        mg.setup()
        u, Ku = disc.get_unknown_size_field(name='test_u'), disc.get_unknown_size_field(name='test_Ku')
        u.s[...] = u_array
        mg.apply_K(u, Ku)
        results.append((np.array(Ku.s), np.array(mg.levels[0].dinv.s)))
    assert np.allclose(results[0][0], results[1][0], rtol=1e-12, atol=1e-12 * np.abs(results[0][0]).max())
    assert np.allclose(results[0][1], results[1][1], rtol=1e-12)
