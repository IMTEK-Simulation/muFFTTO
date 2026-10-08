"""
io_utils.py
===========
Save and load muGrid fields, MPI-parallel, with muGrid's NetCDF I/O
(``muGrid.FileIONetCDF``, the same mechanism muTopOpt uses).

Typical use::

    from muFFTTO import io_utils

    # one snapshot
    io_utils.save_fields('result.nc', [damage, u_fluct], attributes={'Gc': Gc})
    io_utils.load_fields('result.nc', [damage, u_fluct])          # last frame, in place

    # a time series: one frame per call of write()
    writer = io_utils.FieldWriter('run.nc', [damage, u_fluct], attributes={'Gc': Gc},
                                  frame_variables={'load': (), 'stress': (2, 2)})
    for load in loads:
        ...                                                       # solve
        writer.write(load=load, stress=sigma_ij)
    writer.close()

    io_utils.load_fields('run.nc', [damage, u_fluct], frame=3)   # restart from frame 3

    # post-processing in plain numpy, no discretization needed (serial, needs netCDF4)
    run = io_utils.read_file('run.nc')
    run.fields['damage'][-1]         # last frame, same layout as damage.s but global
    run.frame_variables['load']      # one value per frame
    run.attributes['Gc']

Notes
-----
- Each field is stored under its muGrid name, so names must be unique within a
  file. They must not contain ``'__'``, which muGrid uses as a separator in the
  NetCDF dimension names.
- Fields of several discretizations (e.g. a vector and a scalar one on the same
  grid) can be stored in one file.
- Under MPI every rank writes its own subdomain; a file can be read back with
  any number of ranks (the stored fields are global).
- ``load_fields`` fills the interior values (``.s``). The library operators
  refresh the ghost layers before use; call ``fft.communicate_ghosts`` yourself
  if you access ``.sg`` directly.
- muGrid bug worked around here: fields with a single component and more
  than one sub-point per pixel cannot be registered -- quadrature-point
  scalars (``get_quad_field_scalar``) and scalar nodal fields of elements with
  several nodes per pixel (e.g. Q2, ``biquadratic_rectangle``). They are
  written through a per-pixel helper field ``<name>_qp`` with one component
  per sub-point and converted back on loading; ``read_file`` undoes it too.
- Attributes are stored as global NetCDF attributes: strings, numbers and
  arrays (flattened). They are written when the file is created.
"""
import numpy as np
from mpi4py import MPI

import muGrid
from muGrid.Field import Field

__all__ = ['save_fields', 'load_fields', 'FieldWriter', 'read_file']

# global attribute holding the leading (component + sub-point) shape of a field
_LAYOUT_ATTRIBUTE = 'muFFTTO_layout_'
# name suffix of the helper field of a single-component field with several sub-points
_QUAD_SUFFIX = '_qp'


# ---------------------------------------------------------------------------
# Writing and reading
# ---------------------------------------------------------------------------
class FieldWriter:
    """Write fields to a NetCDF file, one frame per call of :meth:`write`.

    Parameters
    ----------
    file_name : str
        Output file; an existing file is overwritten.
    fields : list of muGrid.Field
        Fields to store. Their current values are written on every frame.
    attributes : dict, optional
        Run parameters stored once as global attributes (str, number or array).
    frame_variables : dict, optional
        ``{name: shape}`` of grid-less values stored per frame, e.g.
        ``{'load': (), 'stress': (2, 2)}``; passed to :meth:`write`.
    communicator : muGrid.Communicator, optional
        Default: all MPI ranks (``MPI.COMM_WORLD``), as in ``domain``.
    """

    def __init__(self, file_name, fields, attributes=None, frame_variables=None, communicator=None):
        self.file_name = file_name
        self._file = muGrid.FileIONetCDF(file_name, 'overwrite', _communicator(communicator))
        self._fields = [_StoredField(field) for field in fields]
        _register_fields(self._file, self._fields)

        self._frame_variables = {}
        for name, shape in (frame_variables or {}).items():
            _check_name(name)
            shape = [int(n) for n in np.atleast_1d(shape)] if np.size(shape) else []   # () -> scalar
            self._frame_variables[name] = self._file.register_frame_variable(name, shape, np.float64)

        # all attributes must exist before the first frame (muGrid freezes the header)
        layouts = {_LAYOUT_ATTRIBUTE + f.name: list(f.layout) for f in self._fields}
        _write_attributes(self._file, {**(attributes or {}), **layouts})
        self.nb_frames = 0

    def write(self, **frame_values):
        """Append one frame with the current field values.

        Parameters
        ----------
        **frame_values
            Values of the frame variables declared in the constructor;
            a missing one is stored as NaN.
        """
        unknown = set(frame_values) - set(self._frame_variables)
        if unknown:
            raise ValueError(f'frame variables {sorted(unknown)} were not declared in the constructor '
                             f'(declared: {sorted(self._frame_variables)})')
        for name, buffer in self._frame_variables.items():
            buffer[...] = frame_values.get(name, np.nan)
        for field in self._fields:
            field.copy_to_file_field()
        # every variable of a frame must go out in ONE write() call
        self._file.append_frame().write([f.file_name for f in self._fields] + list(self._frame_variables))
        if hasattr(self._file, 'sync'):
            self._file.sync()  # frames are on disk even if the run is killed later
        self.nb_frames += 1

    def close(self):
        """Close the file (collective under MPI)."""
        self._file.close()

    def __enter__(self):
        return self

    def __exit__(self, *exception):
        self.close()


def save_fields(file_name, fields, attributes=None, communicator=None):
    """Write the current values of ``fields`` to ``file_name`` (one frame).

    Parameters are those of :class:`FieldWriter`.
    """
    with FieldWriter(file_name, fields, attributes=attributes, communicator=communicator) as writer:
        writer.write()


def load_fields(file_name, fields, frame=-1, communicator=None):
    """Fill ``fields`` in place with the values stored in ``file_name``.

    The fields are matched by name and must have the same components and
    sub-points as when they were written (normally: the same script) and the
    same global grid; the number of MPI ranks may differ.

    Parameters
    ----------
    file_name : str
        File written by :func:`save_fields` or :class:`FieldWriter`.
    fields : list of muGrid.Field
        Fields to fill (``.s`` is overwritten).
    frame : int, optional
        Frame to read; negative values count from the end (default: last).
    communicator : muGrid.Communicator, optional
        Default: all MPI ranks.

    Returns
    -------
    dict
        The global attributes of the file (single values as scalars).
    """
    file = muGrid.FileIONetCDF(file_name, 'read', _communicator(communicator))
    try:
        stored_fields = [_StoredField(field) for field in fields]
        attributes = _read_attributes(file)
        for field in stored_fields:
            stored_layout = attributes.get(_LAYOUT_ATTRIBUTE + field.name)
            if stored_layout is not None and tuple(np.atleast_1d(stored_layout)) != field.layout:
                raise ValueError(f'field {field.name!r}: components/sub-points {field.layout} do not match '
                                 f'{tuple(np.atleast_1d(stored_layout))} in {file_name!r}')
        _register_fields(file, stored_fields)

        nb_frames = len(file)
        index = frame + nb_frames if frame < 0 else frame
        if not 0 <= index < nb_frames:
            raise IndexError(f'frame {frame} does not exist, {file_name!r} has {nb_frames} frames')
        file.read(index, [f.file_name for f in stored_fields])
        for field in stored_fields:
            field.copy_from_file_field()
    finally:
        file.close()
    return {key: value for key, value in attributes.items() if not key.startswith(_LAYOUT_ATTRIBUTE)}


class FileContent:
    """Content of a file read by :func:`read_file` (plain numpy arrays).

    Attributes
    ----------
    fields : dict
        ``name -> array [frame, *components, sub_pt, x, y(, z)]``, i.e. the
        layout of ``field.s`` on the global grid, for every frame.
    frame_variables : dict
        ``name -> array [frame, *shape]``.
    attributes : dict
        Global attributes (run parameters and muGrid's own metadata).
    nb_frames : int
    """

    def __init__(self, fields, frame_variables, attributes, nb_frames):
        self.fields = fields
        self.frame_variables = frame_variables
        self.attributes = attributes
        self.nb_frames = nb_frames

    def __repr__(self):
        return (f'FileContent({self.nb_frames} frames, fields {sorted(self.fields)}, '
                f'frame variables {sorted(self.frame_variables)})')


def read_file(file_name):
    """Read a whole file into numpy arrays for post-processing (serial).

    Needs the ``netCDF4`` package; no discretization is required. Every
    calling rank reads the whole file.

    Returns
    -------
    FileContent
    """
    import netCDF4

    with netCDF4.Dataset(file_name, 'r') as dataset:
        attributes = {key: dataset.getncattr(key) for key in dataset.ncattrs()}
        layouts = {key[len(_LAYOUT_ATTRIBUTE):]: tuple(int(n) for n in np.atleast_1d(value))
                   for key, value in attributes.items() if key.startswith(_LAYOUT_ATTRIBUTE)}
        nb_frames = len(dataset.dimensions['frame']) if 'frame' in dataset.dimensions else 0

        fields, frame_variables = {}, {}
        for name, layout in layouts.items():
            variable = dataset.variables[name if name in dataset.variables else name + _QUAD_SUFFIX]
            data = np.asarray(variable[:])
            nb_spatial = sum(dim in ('nx', 'ny', 'nz') for dim in variable.dimensions)
            fields[name] = data.reshape(data.shape[0], *layout, *data.shape[-nb_spatial:])
        stored_field_names = set(layouts) | {name + _QUAD_SUFFIX for name in layouts}
        for name, variable in dataset.variables.items():
            if name not in stored_field_names and variable.dimensions[:1] == ('frame',):
                frame_variables[name] = np.asarray(variable[:])

    attributes = {key: value for key, value in attributes.items() if not key.startswith(_LAYOUT_ATTRIBUTE)}
    return FileContent(fields, frame_variables, attributes, nb_frames)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
class _StoredField:
    """A user field and the field that is actually written to the file.

    They are the same field, except for single-component fields with more
    than one sub-point (quadrature scalars, Q2 nodal scalars), which muGrid
    cannot register: those go through a per-pixel helper field ``<name>_qp``
    with one component per sub-point in the same collection.
    """

    def __init__(self, field):
        self.field = field
        self.name = field.name
        _check_name(self.name)
        # components + sub-point axis of field.s (the grid axes follow)
        self.layout = tuple(int(n) for n in field.s.shape[:len(field.components_shape) + 1])
        if field.nb_components == 1 and self.layout[-1] > 1:
            collection = field.collection
            helper_name = self.name + _QUAD_SUFFIX
            nb_sub_pts = self.layout[-1]
            if collection.field_exists(helper_name):
                self.file_field = Field(collection.get_real_field(helper_name))
            else:
                self.file_field = Field(collection.real_field(helper_name, [nb_sub_pts], 'pixel'))
        else:
            self.file_field = field
        self.file_name = self.file_field.name

    def copy_to_file_field(self):
        if self.file_field is not self.field:
            # (1, 1, q, x, y) -> (q, 1, x, y): only singleton axes move, the data order is kept
            self.file_field.s[...] = self.field.s.reshape(self.file_field.s.shape)

    def copy_from_file_field(self):
        if self.file_field is not self.field:
            self.field.s[...] = self.file_field.s.reshape(self.field.s.shape)


def _register_fields(file, stored_fields):
    """Register the fields, grouped by their field collection (one call each)."""
    names = [f.file_name for f in stored_fields]
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise ValueError(f'field names must be unique within a file, duplicated: {duplicates}')
    groups = []  # [(collection, [names])]
    for stored in stored_fields:
        collection = stored.file_field.collection
        for group_collection, group_names in groups:
            if group_collection is collection:
                group_names.append(stored.file_name)
                break
        else:
            groups.append((collection, [stored.file_name]))
    for collection, group_names in groups:
        file.register_field_collection(collection, field_names=group_names)


def _write_attributes(file, attributes):
    for key, value in attributes.items():
        _check_name(key)
        if isinstance(value, str):
            file.write_global_attribute(key, value)
            continue
        values = np.atleast_1d(np.asarray(value)).ravel()
        if np.issubdtype(values.dtype, np.integer) or values.dtype == bool:
            file.write_global_attribute(key, [int(v) for v in values])
        else:
            file.write_global_attribute(key, [float(v) for v in values])


def _read_attributes(file):
    attributes = {}
    for key in file.read_global_attribute_names():
        value = file.read_global_attribute(key)
        if isinstance(value, (list, tuple)) and len(value) == 1:
            value = value[0]
        attributes[key] = value
    return attributes


def _check_name(name):
    if '__' in name:
        raise ValueError(f"name {name!r} contains '__', which muGrid's NetCDF I/O does not allow")


def _communicator(communicator):
    return muGrid.Communicator(MPI.COMM_WORLD) if communicator is None else communicator
