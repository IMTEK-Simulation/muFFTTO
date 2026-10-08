"""Tests for muFFTTO.geometry and the microstructure_library compatibility wrapper."""
import hashlib

import numpy as np
import pytest

from muFFTTO import geometry
from muFFTTO import microstructure_library


def grid_coords(nb_pixels):
    """Fractional pixel-corner coordinates, as returned by ``FFTEngine.coords``."""
    return np.asarray(np.meshgrid(*[np.arange(n) / n for n in nb_pixels], indexing='ij'))


def digest(array):
    return hashlib.sha256(np.ascontiguousarray(array, dtype=float).tobytes()).hexdigest()[:16]


# Hashes of the arrays produced by the former microstructure_library
# implementation. The examples' geometries must stay bit-identical.
REFERENCE_DIGESTS = [
    ('square_inclusion', (64, 64), 'a8df043470dc9c4a'),
    ('square_inclusion', (17, 23), '1bceeb1d457051ab'),
    ('square_inclusion', (16, 16, 16), '64098bfbeac6ea77'),
    ('circle_inclusion', (64, 64), '85d0e17b53568200'),
    ('circle_inclusion', (1000, 1000), 'c99ff20f2fae2e63'),
    ('circle_inclusion', (32, 1), '883eb66bb2f4722a'),
    ('circle_inclusion', (30, 30, 30), 'eb32ee6b7b473db9'),
    ('contact_test_geometry_1', (100, 100), '0edd3296077fb7e3'),
    ('contact_test_geometry_2', (100, 100), 'e88a9a5c088241bc'),
    ('contact_test_geometry_2', (37, 41), 'c4bee9e4864275cb'),
]


@pytest.mark.parametrize('name, nb_pixels, expected', REFERENCE_DIGESTS)
def test_named_geometries_match_former_implementation(name, nb_pixels, expected):
    assert digest(geometry.get(name, grid_coords(nb_pixels))) == expected


@pytest.mark.parametrize('name, nb_pixels', [
    ('square_inclusion', (24, 18)),
    ('circle_inclusion', (20, 20, 12)),
    ('contact_test_geometry_1', (40, 30)),
    ('contact_fracture_s_gap', (48, 40)),
    ('reentrant_honeycomb', (40, 40)),
    ('sinusoidal_ligaments', (36, 36)),
    ('random_distribution', (16, 12, 8)),
])
def test_subdomains_give_the_same_field(name, nb_pixels):
    """Evaluating on pieces of the grid (as MPI ranks do) gives the global field."""
    coords = grid_coords(nb_pixels)
    params = {'seed': 7} if name == 'random_distribution' else {}
    full = geometry.get(name, coords, **params)
    half = nb_pixels[0] // 2
    left = geometry.get(name, coords[:, :half], **params)
    right = geometry.get(name, coords[:, half:], **params)
    np.testing.assert_array_equal(np.concatenate([left, right], axis=0), full)


def test_compatibility_wrapper():
    nb_pixels = np.array([12, 10])
    coords = grid_coords(nb_pixels)
    phase = microstructure_library.get_geometry(nb_voxels=nb_pixels,
                                                microstructure_name='square_inclusion',
                                                coordinates=coords)
    np.testing.assert_array_equal(phase, geometry.get('square_inclusion', coords))
    with pytest.raises(ValueError):
        microstructure_library.get_geometry(nb_voxels=[12, 11],
                                            microstructure_name='square_inclusion',
                                            coordinates=coords)


def test_unknown_name_lists_available_geometries():
    with pytest.raises(ValueError, match='circle_inclusion'):
        geometry.get('no_such_geometry', grid_coords((4, 4)))


def test_square_inclusion_volume_fraction():
    phase = geometry.get('square_inclusion', grid_coords((40, 40)))
    assert phase.mean() == pytest.approx(0.75)
    phase = geometry.get('square_inclusion', grid_coords((8, 8, 8)))
    assert phase.mean() == pytest.approx(7 / 8)


def test_box_wraps_around_periodic_boundary():
    coords = grid_coords((10,))
    mask = geometry.box([0.8], [0.2])(coords)
    np.testing.assert_array_equal(np.flatnonzero(mask), [0, 1, 8, 9])


def test_ball_is_periodic():
    coords = grid_coords((20, 20))
    at_corner = geometry.ball([0.0, 0.0], 0.15)(coords)
    centred = geometry.ball([0.5, 0.5], 0.15)(coords)
    shifted = np.roll(centred, (10, 10), axis=(0, 1))
    np.testing.assert_array_equal(at_corner, shifted)


def test_shape_set_operations():
    coords = grid_coords((10, 10))
    a = geometry.box([0.0, 0.0], [0.5, 1.0])
    b = geometry.box([0.3, 0.0], [0.8, 1.0])
    np.testing.assert_array_equal((a | b)(coords), a(coords) | b(coords))
    np.testing.assert_array_equal((a & b)(coords), a(coords) & b(coords))
    np.testing.assert_array_equal((a - b)(coords), a(coords) & ~b(coords))
    np.testing.assert_array_equal((~a)(coords), ~a(coords))


def test_pixel_average_converges_to_disk_area():
    radius = 0.3
    disk = geometry.ball([0.5, 0.5], radius)
    errors = []
    for n in (16, 32, 64):
        fraction = geometry.pixel_average(disk, grid_coords((n, n)), pixel_size=[1 / n, 1 / n], samples=8)
        errors.append(abs(fraction.mean() - np.pi * radius ** 2))
    assert errors[-1] < 1e-3
    assert errors[-1] < errors[0]


def test_random_field_is_reproducible_and_uniform():
    coords = grid_coords((64, 64))
    a = geometry.random_field(coords, seed=3)
    np.testing.assert_array_equal(a, geometry.random_field(coords, seed=3))
    assert not np.array_equal(a, geometry.random_field(coords, seed=4))
    assert a.min() >= 0.0 and a.max() < 1.0
    assert a.mean() == pytest.approx(0.5, abs=0.02)


def test_register_custom_geometry():
    name = 'test_two_disks'

    @geometry.register(name)
    def two_disks(coords, radius=0.1):
        disks = geometry.ball((0.25, 0.5), radius) | geometry.ball((0.75, 0.5), radius)
        return geometry.indicator(~disks, coords)

    try:
        assert name in geometry.available()
        phase = geometry.get(name, grid_coords((20, 20)))
        assert phase.min() == 0.0 and phase.max() == 1.0
        with pytest.raises(ValueError):
            geometry.register(name)(two_disks)
    finally:
        geometry._REGISTRY.pop(name)


def test_reentrant_honeycomb():
    """Square cell by default, mirror-symmetric in x, solid grows with the wall thickness,
    and one whole bow-tie void sits in the centre."""
    from scipy import ndimage
    Lx, Ly = geometry.reentrant_honeycomb_cell_size(theta=30.0)
    assert Lx == pytest.approx(Ly)
    n = 64
    coords = grid_coords((n, n)) + 0.5 / n          # pixel centres: symmetric about x = 0.5
    thin = geometry.get('reentrant_honeycomb', coords, thickness=0.1)
    thick = geometry.get('reentrant_honeycomb', coords, thickness=0.2)
    np.testing.assert_array_equal(thin, thin[::-1])
    assert 0.1 < thin.mean() < thick.mean() < 0.6
    labels, _ = ndimage.label(thin == 0)
    centre = labels == labels[n // 2, n // 2]
    assert thin[n // 2, n // 2] == 0
    assert not (centre[0].any() or centre[-1].any() or centre[:, 0].any() or centre[:, -1].any())
    with pytest.raises(ValueError):
        geometry.get('reentrant_honeycomb', coords, theta=30.0, h_over_l=0.4)


def test_sinusoidal_ligaments_symmetry_and_gaps():
    """Mirror symmetric in x and y; the gap between the horizontal ligaments at the
    cell edge is open, and it closes for gap = 0."""
    n = 64
    coords = grid_coords((n, n)) + 0.5 / n          # pixel centres
    phase = geometry.get('sinusoidal_ligaments', coords)
    np.testing.assert_array_equal(phase, phase[::-1])
    np.testing.assert_array_equal(phase, phase[:, ::-1])
    edge = (np.abs(coords[0]) < 1.0 / n) & (np.abs(coords[1] - 0.5) < 0.05)   # inside the edge gap
    assert np.all(phase[edge] == 0)
    closed = geometry.get('sinusoidal_ligaments', coords, gap=0.0)
    assert np.all(closed[edge] == 1)
