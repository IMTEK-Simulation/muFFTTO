"""
Backward-compatible entry point for microstructure geometries.

The geometries now live in :mod:`muFFTTO.geometry`. This module keeps the
old call used throughout the examples working::

    phase_field.s[0, 0] = microstructure_library.get_geometry(
        nb_voxels=discretization.nb_of_pixels,
        microstructure_name='square_inclusion',
        coordinates=discretization.fft.coords)

New code can call :func:`muFFTTO.geometry.get` directly. The former catalogue
of unused geometries (laminates, smooth test fields, 3D lattices, chiral
metamaterials, ...) was removed; it is available in the git history.
"""
import numpy as np

from muFFTTO import geometry


def get_geometry(nb_voxels, microstructure_name='random_distribution', coordinates=None,
                 parameter=None, contrast=None, **kwargs):
    """Phase field of the named microstructure on the given pixel coordinates.

    Parameters
    ----------
    nb_voxels : array_like of int, shape ``(dim,)``
        Number of pixels of the (local) grid, i.e.
        ``discretization.nb_of_pixels``. Only used to check ``coordinates``.
    microstructure_name : str, optional
        A name from :func:`muFFTTO.geometry.available`. Default
        ``'random_distribution'``.
    coordinates : ndarray, shape ``(dim, *nb_voxels)``
        Fractional pixel coordinates in ``[0, 1)``, e.g.
        ``discretization.fft.coords``.
    parameter, contrast
        Accepted for compatibility; none of the available geometries uses them.
    **kwargs
        Geometry parameters passed on to the geometry function, e.g.
        ``seed`` for ``'random_distribution'``.

    Returns
    -------
    ndarray, shape ``nb_voxels``
        The phase field (1 = matrix, 0 = inclusion, for the inclusion
        geometries).

    Raises
    ------
    ValueError
        If ``coordinates`` is missing or does not match ``nb_voxels``, or the
        name is unknown.
    """
    if coordinates is None:
        raise ValueError('get_geometry needs the pixel coordinates, e.g. '
                         'coordinates=discretization.fft.coords')
    coordinates = np.asarray(coordinates)
    nb_voxels = tuple(int(n) for n in np.atleast_1d(nb_voxels))
    if coordinates.shape != (len(nb_voxels),) + nb_voxels:
        raise ValueError(f'coordinates have shape {coordinates.shape}, expected '
                         f'{(len(nb_voxels),) + nb_voxels} for nb_voxels={nb_voxels}')
    return geometry.get(microstructure_name, coordinates, **kwargs)
