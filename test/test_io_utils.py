"""Tests of muFFTTO.io_utils (save/load of muGrid fields with NetCDF).

They also run under MPI (``mpirun -n 2 python -m pytest test/test_io_utils.py``):
values are functions of the global coordinates, so every rank can check its
own subdomain.
"""
import numpy as np
import pytest
from mpi4py import MPI

import muGrid

from muFFTTO import domain
from muFFTTO import io_utils

pytestmark = pytest.mark.skipif(not muGrid.has_netcdf, reason='muGrid without NetCDF support')


def _discretization(problem_type, nb_pixels=(8, 6)):
    cell = domain.PeriodicUnitCell(domain_size=[1.0, 1.0], problem_type=problem_type)
    return domain.Discretization(cell=cell, nb_of_pixels_global=list(nb_pixels),
                                 discretization_type='finite_element', element_type='bilinear_rectangle')


def _fill(field, discretization, seed):
    """Deterministic values depending on the global position and the component index."""
    x, y = discretization.fft.coords
    leading = field.s.shape[:-2]
    for index in np.ndindex(*leading):
        k = seed + np.ravel_multi_index(index, leading)
        field.s[index] = np.sin(3 * x + k) * np.cos(2 * y - 0.5 * k)


def _shared_file_name(tmp_path, name):
    """Same file name on all ranks (tmp_path differs per process)."""
    return MPI.COMM_WORLD.bcast(str(tmp_path / name), root=0)


@pytest.fixture
def fields():
    disc = _discretization('elasticity')
    return disc, {
        'scalar_nodal': disc.get_scalar_field('io_scalar_nodal'),
        'vector_nodal': disc.get_unknown_size_field('io_vector_nodal'),
        'quad_scalar': disc.get_quad_field_scalar('io_quad_scalar'),
        'gradient': disc.get_gradient_size_field('io_gradient'),
        'material': disc.get_material_data_size_field_mugrid('io_material'),
    }


def test_save_and_load_all_field_types(fields, tmp_path):
    disc, field_dict = fields
    for seed, field in enumerate(field_dict.values()):
        _fill(field, disc, 10 * seed)
    reference = {name: np.copy(field.s) for name, field in field_dict.items()}
    file_name = _shared_file_name(tmp_path, 'all_types.nc')

    io_utils.save_fields(file_name, list(field_dict.values()), attributes={'Gc': 5e-4, 'name': 'test'})
    for field in field_dict.values():
        field.s[...] = 0
    attributes = io_utils.load_fields(file_name, list(field_dict.values()))

    for name, field in field_dict.items():
        np.testing.assert_array_equal(field.s, reference[name], err_msg=name)
    assert attributes['Gc'] == pytest.approx(5e-4)
    assert attributes['name'] == 'test'


def test_time_series_with_frame_variables_and_two_discretizations(tmp_path):
    disc_u, disc_d = _discretization('elasticity'), _discretization('conductivity')
    u = disc_u.get_unknown_size_field('io_u')
    d = disc_d.get_unknown_size_field('io_d')
    history = disc_d.get_quad_field_scalar('io_history')
    file_name = _shared_file_name(tmp_path, 'series.nc')

    with io_utils.FieldWriter(file_name, [u, d, history], attributes={'steps': [0, 1, 2]},
                              frame_variables={'load': (), 'stress': (2, 2)}) as writer:
        for frame in range(3):
            _fill(u, disc_u, frame)
            _fill(d, disc_d, frame + 100)
            _fill(history, disc_d, frame + 200)
            writer.write(load=0.1 * frame, stress=frame * np.eye(2))
    assert writer.nb_frames == 3

    for frame in (1, -1):
        io_utils.load_fields(file_name, [u, d, history], frame=frame)
        expected = frame % 3
        for field, disc, seed in ((u, disc_u, expected), (d, disc_d, expected + 100),
                                  (history, disc_d, expected + 200)):
            values = np.copy(field.s)
            _fill(field, disc, seed)
            np.testing.assert_array_equal(values, field.s)


def test_read_file_returns_global_arrays_in_field_layout(tmp_path):
    try:
        import netCDF4  # noqa: F401
    except (ImportError, ValueError) as error:  # ValueError: netCDF4 built against another numpy
        pytest.skip(f'netCDF4 not usable: {error}')
    disc = _discretization('elasticity')
    u = disc.get_unknown_size_field('io_read_u')
    history = disc.get_quad_field_scalar('io_read_history')
    file_name = _shared_file_name(tmp_path, 'read.nc')
    with io_utils.FieldWriter(file_name, [u, history], attributes={'Gc': 1.0},
                              frame_variables={'load': ()}) as writer:
        for frame in range(2):
            _fill(u, disc, frame)
            _fill(history, disc, frame + 50)
            writer.write(load=float(frame))

    content = io_utils.read_file(file_name)
    assert content.nb_frames == 2
    np.testing.assert_array_equal(content.frame_variables['load'], [0.0, 1.0])
    assert content.attributes['Gc'] == pytest.approx(1.0)
    # the global arrays have the layout of field.s; compare this rank's subdomain
    x0, y0 = disc.fft.subdomain_locations
    nx, ny = disc.fft.nb_subdomain_grid_pts
    assert content.fields['io_read_u'].shape == (2, 2, 1, 8, 6)
    assert content.fields['io_read_history'].shape == (2, 1, 1, history.s.shape[2], 8, 6)
    _fill(u, disc, 1)
    _fill(history, disc, 51)
    np.testing.assert_array_equal(content.fields['io_read_u'][1][..., x0:x0 + nx, y0:y0 + ny], u.s)
    np.testing.assert_array_equal(content.fields['io_read_history'][1][..., x0:x0 + nx, y0:y0 + ny], history.s)


def test_invalid_use_is_rejected(fields, tmp_path):
    disc, field_dict = fields
    file_name = _shared_file_name(tmp_path, 'invalid.nc')
    bad = disc.get_scalar_field('bad__name')
    with pytest.raises(ValueError, match="'__'"):
        io_utils.save_fields(file_name, [bad])

    io_utils.save_fields(file_name, [field_dict['scalar_nodal']])
    with pytest.raises(IndexError):
        io_utils.load_fields(file_name, [field_dict['scalar_nodal']], frame=5)

    with io_utils.FieldWriter(file_name, [field_dict['scalar_nodal']], frame_variables={'load': ()}) as writer:
        with pytest.raises(ValueError, match='not declared'):
            writer.write(lod=1.0)


def test_q2_scalar_and_vector_nodal_fields(tmp_path):
    """Q2 nodal fields have 4 nodes per pixel; the scalar one needs the helper field."""
    cell_u = domain.PeriodicUnitCell(domain_size=[1.0, 1.0], problem_type='elasticity')
    cell_d = domain.PeriodicUnitCell(domain_size=[1.0, 1.0], problem_type='conductivity')
    disc_u = domain.Discretization(cell=cell_u, nb_of_pixels_global=[8, 6], discretization_type='finite_element',
                                   element_type='biquadratic_rectangle')
    disc_d = domain.Discretization(cell=cell_d, nb_of_pixels_global=[8, 6], discretization_type='finite_element',
                                   element_type='biquadratic_rectangle')
    u = disc_u.get_unknown_size_field('io_q2_u')
    d = disc_d.get_unknown_size_field('io_q2_d')
    _fill(u, disc_u, 1)
    _fill(d, disc_d, 2)
    reference = np.copy(u.s), np.copy(d.s)
    file_name = _shared_file_name(tmp_path, 'q2.nc')
    io_utils.save_fields(file_name, [u, d])
    u.s[...] = 0
    d.s[...] = 0
    io_utils.load_fields(file_name, [u, d])
    np.testing.assert_array_equal(u.s, reference[0])
    np.testing.assert_array_equal(d.s, reference[1])
