"""
Library of voxel/pixel microstructures (phase fields) for muFFTTO.

This module generates the material distribution ("phase field") of a periodic
unit cell on a regular grid. The resulting arrays are typically used as the
density / indicator field that scales the material stiffness or conductivity
in homogenization and topology-optimization examples.

Contents
--------
get_geometry
    Main dispatch function. Returns a phase field for a named microstructure
    (simple 2D/3D inclusions, laminates, smooth analytic test fields and a
    catalogue of 3D lattice / metamaterial unit cells).
Circle_A, Diagonal2D_A, Diagonal2D_B, Diagonal2D_FACE, Kite, square_frame2D,
square_frame2D_flexible, specialface, specialface1, height, height1, height2,
Structure2D, Structure2D_FACE
    2D building blocks (square ``Nx x Nx`` masks, mostly 0/1 valued) that are
    stacked/extruded into 3D cells.
HSCC, HFCC, HFCC_no_frame, HFDC, HBCC, Circle_Frame, Normalcube, Sphere,
SphereinCube, Metamaterial_1 ... Metamaterial_4
    3D voxel unit cells built from the 2D blocks (truss lattices, plates,
    metamaterials).
chiral_metamaterial, chiral_metamaterial_2
    Parametrised 3D chiral metamaterials (contributed by Indre Joedicke).
visualize_voxels
    Quick 3D voxel plot of a phase field with matplotlib.
check_*
    Input validation helpers used by :func:`get_geometry`.

Conventions
-----------
* Arrays are indexed ``phase_field[i_x, i_y]`` (2D) or
  ``phase_field[i_x, i_y, i_z]`` (3D), i.e. the first array axis is the
  x-direction.
* For the coordinate-based geometries in :func:`get_geometry`, ``coordinates``
  has shape ``(dim, *nb_voxels)`` and ``coordinates[d]`` holds the
  d-th coordinate of every voxel/pixel. Most thresholds (0.25, 0.5, ...) assume
  a unit cell of size 1 in every direction, i.e. coordinates in ``[0, 1)``.
* For the voxel-index based 3D geometries (HSCC, HBCC, ...), the value 1 means
  material (solid) and 0 means void, unless stated otherwise. These generators
  work on the full (global) grid and are not MPI-domain-decomposition aware.
"""
import warnings
import numpy as np
import matplotlib.pyplot as plt

# This import registers the 3D projection, but is otherwise unused.
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import


def get_geometry(nb_voxels,
                 microstructure_name='random_distribution',
                 coordinates=None,
                 parameter=None,
                 contrast=None,
                 **kwargs):
    """
    Create the phase field (material distribution) of a named microstructure.

    The function dispatches on ``microstructure_name`` and returns an array of
    shape ``nb_voxels`` with the phase/density value of each voxel (pixel in
    2D). Depending on the geometry the field is binary (0/1), piecewise
    constant with several levels (laminates, labelled inclusions) or smooth
    (analytic test functions).

    Parameters
    ----------
    nb_voxels : np.ndarray of int, shape (dim,)
        Number of voxels in each direction. Must be a NumPy array (several
        branches use ``nb_voxels.size``). For coordinate-based geometries it
        must match ``coordinates.shape[1:]``; when running with MPI, pass the
        shape of the local subdomain together with the local coordinates so
        that each rank creates only its own part of the field.
    microstructure_name : str, optional
        Name of the microstructure, default ``'random_distribution'``.
        Supported options (the phase value inside "inclusion" is given in
        brackets, the matrix is the other value):

        Random / coordinate-based 2D (and partly 3D) geometries:

        * ``'random_distribution'`` -- uniform random values in [0, 1).
          Optional ``kwargs['seed']`` seeds ``np.random``. Does not use
          ``coordinates``.
        * ``'square_inclusion'`` -- matrix 1, centred square/cube
          ``[0.25, 0.75)^dim`` set to 0 (2D and 3D).
        * ``'square_inclusion_equal_volfrac'`` -- matrix 1, centred square
          ``[0.15, 0.85)^2`` set to 0 (2D).
        * ``'hashin_inclusion_2D'`` -- currently identical to the 2D
          ``'square_inclusion'``; requires ``kwargs['rad_1']`` and
          ``kwargs['rad_2']`` which are read but not used.
        * ``'n_squares'`` -- currently identical to the 2D
          ``'square_inclusion'``.
        * ``'contact_test_geometry_1/2/3'`` -- 2D square frame of material 1
          (border width 0.15) around a void (0) with rectangular "sticks"
          (material 1) protruding into the void; used for contact tests.
        * ``'circles'`` -- matrix 1 with four circular voids (0) of radius
          ``y_lim/8`` centred at the quarter points of the cell (2D).
        * ``'2_circles'`` -- matrix 1 with three circular voids (0) of
          radius ``y_lim/10`` (2D; the name is historical).
        * ``'circle_inclusion'`` -- matrix 1, centred void (0): in 1D (or
          2D with ``nb_voxels[1] == 1``) the interval ``|x-0.5| < 0.2``, in
          2D a disk of radius 0.2, in 3D a ball with squared radius 0.1
          (radius ~0.316).
        * ``'circle_inclusions'`` -- 2D regular ``nb_circles x nb_circles``
          array of disk inclusions on a zero matrix, see ``kwargs`` below.
          Inclusions get value 1, or distinct integer labels
          ``1..nb_circles**2`` (randomly permuted) if ``random_density``.
          3D raises ``NotImplementedError``.

        Laminates (layers normal to x, i.e. depending on ``coordinates[0]``):

        * ``'laminate'`` -- 0 for ``x < 0.5``, 1 otherwise.
        * ``'n_laminate'`` -- ``parameter`` equally thick layers with values
          ``linspace(0, 1, parameter)``.
        * ``'laminate2'`` -- ``parameter`` layers with values
          ``linspace(contrast, 1, parameter)``; note that the first layer
          keeps the initial value 0 (not ``contrast``).
        * ``'laminate_log'`` -- ``parameter`` layers with logarithmically
          spaced values ``logspace(contrast, 0, parameter)``, i.e. from
          ``10**contrast`` to 1.

        Smooth analytic fields (mainly for testing solvers/preconditioners):

        * ``'linear'`` -- ``x``; ``'bilinear'`` -- ``x*y`` (``x*y*z`` in 3D).
        * ``'symmetric_linear'`` -- ``x`` in the first half of the x-index
          range, mirrored in the second half (2D only).
        * ``'right_cluster_x3'`` -- ``1 - (1 - x)**3``;
          ``'left_cluster_x3'`` -- ``(1 - x)**3``.
        * ``'abs_val'`` -- ``|x-0.5| + |y-0.5|`` (2D).
        * ``'sine_wave'`` -- ``0.5 + 0.25 cos(6 pi x) + 0.25 cos(6 pi y)``
          (+ a z-term in 3D).
        * ``'sine_wave_rapid'`` -- like ``'sine_wave'`` but with 30 and 10
          periods in x and y.
        * ``'sine_wave_'``, ``'sine_wave_inv'``, ``'cos_wave'`` -- other
          combinations of one-period cosines (see code).
        * ``'tanh'`` -- ``tanh((x-0.5)(y-0.5)/0.09)`` (2D).
        * ``'uniform_x1'`` -- accepted by the name check but **not
          implemented** in the ``match`` block (see Raises).

        3D voxel lattices / metamaterials (require ``dim == 3``; 1 = solid,
        0 = void). Category I (material in the faces of the cube):

        * ``'geometry_I_1_3D'`` -- cube edge frame (simple cubic truss),
          :func:`HSCC`. Requires equal ``Nx=Ny=Nz`` > 19.
        * ``'geometry_I_2_3D'`` -- edge frame + one diagonal in each face,
          :func:`HFDC`. Requires equal ``Nx=Ny=Nz`` > 19.
        * ``'geometry_I_3_3D'`` -- edge frame + both diagonals in each face
          (face-centred cubic truss), :func:`HFCC`.
        * ``'geometry_I_4_3D'`` -- both face diagonals only, no edges,
          :func:`HFCC_no_frame`.
        * ``'geometry_I_5_3D'`` -- hollow cube whose faces are plates with a
          circular hole, :func:`Circle_Frame`.

        Category II (material in the body):

        * ``'geometry_II_0_3D'`` -- completely filled cube,
          :func:`Normalcube`.
        * ``'geometry_II_1_3D'`` -- edge frame + four body diagonals
          (body-centred cubic truss), :func:`HBCC`.
        * ``'geometry_II_3_3D'`` -- outer edge frame, inner concentric
          square rings and body diagonals with the central cube removed,
          :func:`Metamaterial_1`.
        * ``'geometry_II_4_3D'`` -- filled cube with a (large) sphere
          removed, :func:`SphereinCube`.

        Category III (metamaterials):

        * ``'geometry_III_1_3D'`` -- :func:`Metamaterial_3`; expects
          ``Nz = 1.8 * Nx`` (``ratios=[1.8, 1.8, 1]``).
        * ``'geometry_III_2_3D'`` -- face diagonals + "kite" (diamond)
          mid-planes, :func:`Metamaterial_2`. Requires ``N > 39``.
        * ``'geometry_III_3_3D'`` -- extruded 2D structure,
          :func:`Metamaterial_4`.
        * ``'geometry_III_4_3D'`` -- :func:`chiral_metamaterial`, configured
          by ``parameter`` (dict with ``'lengths'``, ``'radius'``,
          ``'thickness'``, ``'alpha'``).
        * ``'geometry_III_5_3D'`` -- :func:`chiral_metamaterial_2`,
          configured by ``parameter`` (dict with ``'lengths'``,
          ``'radius_out'``, ``'radius_inn'``, ``'thickness'``, ``'alpha'``).

    coordinates : np.ndarray of float, shape (dim, *nb_voxels), optional
        Coordinates of the voxels (e.g. pixel/node coordinates of the
        discretization). Required by all coordinate-based geometries; not
        used by ``'random_distribution'`` and the 3D voxel lattices.
    parameter : int or dict, optional
        Geometry-specific parameter: number of layers for the laminates
        (``'n_laminate'``, ``'laminate2'``, ``'laminate_log'``) or a dict
        of geometric parameters for the chiral metamaterials. For the latter,
        missing keys are filled with defaults (but see Notes).
    contrast : float, optional
        Phase contrast for ``'laminate2'`` (lowest phase value) and
        ``'laminate_log'`` (exponent of the lowest phase value,
        ``10**contrast``).
    **kwargs
        Additional geometry-specific options:

        * ``seed`` (int) -- random seed for ``'random_distribution'``.
        * ``rad_1``, ``rad_2`` (float) -- required (but unused) by
          ``'hashin_inclusion_2D'``.
        * ``nb_circles`` (int), ``r_0`` (float), ``vol_frac`` (unused),
          ``random_density`` (bool), ``random_centers`` (bool) -- for
          ``'circle_inclusions'``. The inclusion radius is
          ``r_0 / nb_circles``; with ``random_centers`` each centre is shifted
          by a random amount bounded so that the disk stays inside its
          ``1/nb_circles`` sized box.

    Returns
    -------
    phase_field : np.ndarray of float, shape nb_voxels
        Phase field / density in every voxel.

    Raises
    ------
    ValueError
        If ``microstructure_name`` is not recognised, or if a 3D geometry is
        requested with an incompatible grid (see the ``check_*`` helpers).
    NotImplementedError
        For ``'circle_inclusions'`` in 3D.
    UnboundLocalError
        For ``'uniform_x1'``, which is listed but has no implementation, so
        ``phase_field`` is never assigned.

    Notes
    -----
    * For the chiral geometries, the defaults for a partially specified
      ``parameter`` dict are filled by an ``if/elif`` chain, so only the
      *first* missing key gets its default value.
    * ``'symmetric_linear'`` mirrors in index space using the full
      ``nb_voxels``; it is therefore only meaningful for a non-distributed
      (serial) field.
    """
    # Validate the requested name early, so typos fail with a clear message
    # instead of an UnboundLocalError at the end of the match block.
    if not microstructure_name in ['random_distribution', 'square_inclusion', 'circle_inclusion', 'circle_inclusions',
                                   'sine_wave', 'sine_wave_', 'linear', 'bilinear', 'tanh', 'sine_wave_inv', 'abs_val',
                                   'right_cluster_x3', 'left_cluster_x3', 'uniform_x1', 'n_laminate', 'circles',
                                   'cos_wave',
                                   '2_circles', 'contact_test_geometry_1', 'contact_test_geometry_2', 'contact_test_geometry_3',
                                   'symmetric_linear', 'hashin_inclusion_2D',
                                   'square_inclusion_equal_volfrac', 'sine_wave_rapid', 'n_squares',
                                   'laminate', 'laminate2', 'laminate_log',
                                   'geometry_I_1_3D', 'geometry_I_2_3D', 'geometry_I_3_3D', 'geometry_I_4_3D',
                                   'geometry_I_5_3D',
                                   'geometry_II_0_3D', 'geometry_II_1_3D', 'geometry_II_3_3D', 'geometry_II_4_3D',
                                   'geometry_III_1_3D', 'geometry_III_2_3D', 'geometry_III_3_3D', 'geometry_III_4_3D',
                                   'geometry_III_5_3D'
                                   ]:
        raise ValueError('Unrecognised microstructure_name {}'.format(microstructure_name))
    # Legacy size check kept for reference (now done per geometry by the check_* helpers).
    # if not nb_voxels[0] > 19 and nb_voxels[1] > 19 and nb_voxels[2] > 19 and nb_voxels[0]//5!=0 and nb_voxels[1]//5!=0 and nb_voxels[2]//5!=0:
    #     raise ValueError('Microstructure_name {} is implemented only when Size of any dimension is more than 10 and it is multiple of 5'.format(microstructure_name))

    match microstructure_name:
        case 'random_distribution':
            if 'seed' in kwargs:
                np.random.seed(kwargs['seed'])

            phase_field = np.random.rand(*nb_voxels)

        case 'square_inclusion':

            # Matrix = 1, centred square (2D) / cube (3D) [0.25, 0.75)^dim = 0.
            # The boolean masks are evaluated element-wise on the coordinate
            # arrays, so this also works on a local MPI subdomain.
            phase_field = np.ones(nb_voxels)
            if len(nb_voxels) == 2:
                phase_field[np.logical_and(np.logical_and(coordinates[0] < 0.75, coordinates[1] < 0.75),
                                           np.logical_and(coordinates[0] >= 0.25, coordinates[1] >= 0.25))] = 0
            elif len(nb_voxels) == 3:
                phase_field[np.logical_and(np.logical_and(coordinates[0] >= 0.25, coordinates[0] < 0.75),
                                           np.logical_and(np.logical_and(coordinates[1] < 0.75, coordinates[2] < 0.75),
                                                          np.logical_and(coordinates[1] >= 0.25,
                                                                         coordinates[2] >= 0.25)))] = 0
        case 'contact_test_geometry_1':
            # Start from solid (1), carve out the interior [0.15, 0.85)^2 to get
            # a square frame, then add solid "sticks" (1) reaching into the
            # void. Left and right sticks are vertically offset and nearly touch.
            phase_field = np.ones(nb_voxels)
            # frame
            phase_field[np.logical_and(np.logical_and(coordinates[0] < 0.85, coordinates[1] < 0.85),
                                       np.logical_and(coordinates[0] >= 0.15, coordinates[1] >= 0.15))] = 0
            # left stick

            phase_field[np.logical_and(np.logical_and(coordinates[0] < 0.45, coordinates[1] < 0.6),
                                       np.logical_and(coordinates[0] >= 0.15, coordinates[1] >= 0.5))] = 1
            # right stick
            phase_field[np.logical_and(np.logical_and(coordinates[0] < 0.95, coordinates[1] < 0.55),
                                       np.logical_and(coordinates[0] >= 0.55, coordinates[1] >= 0.45))] = 1

            # phase_field[:, :3] = 0 # remove boundary
            # phase_field[:, -3:] = 0

        case 'contact_test_geometry_2':
            phase_field = np.ones(nb_voxels)
            # frame
            phase_field[np.logical_and(np.logical_and(coordinates[0] < 0.85, coordinates[1] < 0.85),
                                       np.logical_and(coordinates[0] >= 0.15, coordinates[1] >= 0.15))] = 0
            # left lower stick
            phase_field[np.logical_and(np.logical_and(coordinates[0] < 0.52, coordinates[1] < 0.45),
                                       np.logical_and(coordinates[0] >= 0.15, coordinates[1] >= 0.35))] = 1
            # right upper stick
            phase_field[np.logical_and(np.logical_and(coordinates[0] < 0.95, coordinates[1] < 0.65),
                                       np.logical_and(coordinates[0] >= 0.47, coordinates[1] >= 0.55))] = 1
        case 'contact_test_geometry_3':
            phase_field = np.ones(nb_voxels)
            # frame
            phase_field[np.logical_and(np.logical_and(coordinates[0] < 0.85, coordinates[1] < 0.85),
                                       np.logical_and(coordinates[0] >= 0.15, coordinates[1] >= 0.15))] = 0
            # left middle stick
            phase_field[np.logical_and(np.logical_and(coordinates[0] < 0.6, coordinates[1] < 0.55),
                                       np.logical_and(coordinates[0] >= 0.15, coordinates[1] >= 0.45))] = 1
            # # right upper stick
            phase_field[np.logical_and(np.logical_and(coordinates[0] < 0.75, coordinates[1] < 0.65),
                                       np.logical_and(coordinates[0] >= 0.7, coordinates[1] >= 0.1))] = 1
            # remove boundary
            #phase_field[:, 1:5] = 0 # remove boundary
            #phase_field[:, -3:] = 0



        case 'hashin_inclusion_2D':
            # Radii are read (so they must be provided) but are not used yet:
            # the geometry below is the same centred square as 'square_inclusion'.
            r1 = kwargs['rad_1']
            r2 = kwargs['rad_2']

            phase_field = np.ones(nb_voxels)
            phase_field[np.logical_and(np.logical_and(coordinates[0] < 0.75, coordinates[1] < 0.75),
                                       np.logical_and(coordinates[0] >= 0.25, coordinates[1] >= 0.25))] = 0

        case 'n_squares':
            # Currently a single centred square void, identical to 'square_inclusion' (2D).
            phase_field = np.ones(nb_voxels)
            phase_field[np.logical_and(np.logical_and(coordinates[0] < 0.75, coordinates[1] < 0.75),
                                       np.logical_and(coordinates[0] >= 0.25, coordinates[1] >= 0.25))] = 0
        case 'circles':
            phase_field = np.ones(nb_voxels)
            # Extent of the cell taken from the coordinate of the last grid
            # point (i.e. L - h, not L). NOTE: with MPI this is the last point of
            # the *local* subdomain, so the circles would be placed per rank.
            x_lim = coordinates[0][-1, -1]
            y_lim = coordinates[1][-1, -1]
            # Define circle parameters (center coordinates and radius)
            circles = [
                (x_lim / 4, y_lim / 4, y_lim / 8),  # Circle 1
                (x_lim / 4, 3 * y_lim / 4, y_lim / 8),  # Circle 2
                (3 * x_lim / 4, 3 * y_lim / 4, y_lim / 8),  # Circle 3
                (3 * x_lim / 4, y_lim / 4, y_lim / 8)  # Circle 4
            ]
            # Apply circle masks
            for cx, cy, r in circles:
                mask = (coordinates[0] - cx) ** 2 + (coordinates[1] - cy) ** 2 <= r ** 2
                phase_field[mask] = 0  # Set pixels inside the circle to 0 (void)
        case '2_circles':
            # Despite the name, three circular voids (0) in a solid matrix (1),
            # all in the upper part of the cell (y ~ 2/3 .. 3/4 of the extent).
            phase_field = np.ones(nb_voxels)
            x_lim = coordinates[0][-1, -1]
            y_lim = coordinates[1][-1, -1]
            # Define circle parameters (center coordinates and radius)
            circles = [
                (x_lim / 6, 3 * y_lim / 4, y_lim / 10),  # Circle 1
                (3 * x_lim / 6, 4 * y_lim / 6, y_lim / 10),  # Circle 2
                (5 * x_lim / 6, 3 * y_lim / 4, y_lim / 10),
            ]
            # Apply circle masks
            for cx, cy, r in circles:
                mask = (coordinates[0] - cx) ** 2 + (coordinates[1] - cy) ** 2 <= r ** 2
                phase_field[mask] = 0  # Set pixels inside the circle to 0 (void)

        case 'laminate':

            phase_field = np.ones(nb_voxels)
            phase_field[coordinates[0] < 0.5] = 0
        case 'laminate2':
            # 'parameter' layers of equal thickness normal to x. Layer i covers
            # x in [positions[i], positions[i+1]) and gets phases[i]; this is
            # achieved by successively overwriting everything right of each
            # interface. The first layer is never overwritten and stays 0.
            phase_field = np.zeros(nb_voxels)
            # Legacy/alternative implementation kept for reference
            # division=1/parameter
            # divisions=np.arange(0, 1, 1 / parameter)
            # divisionss= np.linspace(0, 1, parameter, endpoint = False)

            # divisions2 = np.arange(0, 1, 1 /( parameter-1))
            phases = np.linspace(contrast, 1, parameter)

            # positions = np.arange(0, 1+1 / parameter, 1 / parameter)
            positions = np.linspace(0, 1, parameter + 1)
            for i in np.arange(phases.size - 1):
                # section=divisions[i]
                # phase_field[coordinates[0] >= section] = divisions2[i]
                phase_field[coordinates[0] >= positions[i + 1]] = phases[i + 1]

            # phase_field[coordinates[0] >= divisions[-1]] = positions[-1]
            # print()
        case 'n_laminate':
            # 'parameter' equally thick layers normal to x with values
            # linspace(0, 1, parameter) (first layer 0, last layer 1).
            phase_field = np.zeros(nb_voxels)
            # Legacy/alternative implementation kept for reference
            # division=1/parameter
            # divisions=np.arange(0, 1, 1 / parameter)
            # divisionss= np.linspace(0, 1, parameter, endpoint = False)

            # divisions2 = np.arange(0, 1, 1 /( parameter-1))
            phases = np.linspace(0, 1, parameter)

            # positions = np.arange(0, 1+1 / parameter, 1 / parameter)
            positions = np.linspace(0, 1, parameter + 1)
            for i in np.arange(phases.size - 1):
                # section=divisions[i]
                # phase_field[coordinates[0] >= section] = divisions2[i]
                phase_field[coordinates[0] >= positions[i + 1]] = phases[i + 1]

        case 'right_cluster_x3':
            # Smooth field increasing from 0 (x=0) to 1 (x=1), flat near x=1.

            phase_field = 1 - (1 - coordinates[0]) ** 3
        case 'left_cluster_x3':
            # Smooth field decreasing from 1 (x=0) to 0 (x=1), flat near x=1.

            phase_field = (1 - coordinates[0]) ** 3

        case 'laminate_log':
            # Like 'laminate2' but with logarithmically spaced layer values
            # 10**contrast, ..., 10**0 = 1. The field is initialised with
            # 10**contrast = phases[0], so the first layer is consistent here.
            phase_field = np.zeros(nb_voxels) + np.power(10., contrast)
            # Legacy/alternative implementation kept for reference
            # division=1/parameter
            # divisions=np.arange(0, 1, 1 / parameter)
            # divisionss= np.linspace(0, 1, parameter, endpoint = False)

            # divisions2 = np.arange(0, 1, 1 /( parameter-1))
            phases = np.logspace(contrast, 0, parameter)

            # positions = np.arange(0, 1+1 / parameter, 1 / parameter)
            positions = np.linspace(0, 1, parameter + 1)
            for i in np.arange(phases.size - 1):
                # section=divisions[i]
                # phase_field[coordinates[0] >= section] = divisions2[i]
                phase_field[coordinates[0] >= positions[i + 1]] = phases[i + 1]

            # phase_field[coordinates[0] >= divisions[-1]] = positions[-1]
            # print()
        case 'square_inclusion_equal_volfrac':
            # Larger centred square void [0.15, 0.85)^2 (void fraction 0.49).

            phase_field = np.ones(nb_voxels)
            phase_field[np.logical_and(np.logical_and(coordinates[0] < 0.85, coordinates[1] < 0.85),
                                       np.logical_and(coordinates[0] >= 0.15, coordinates[1] >= 0.15))] = 0
        case 'circle_inclusion':
            # Matrix 1 with a centred void (0). Cases: 1D, quasi-1D (2D grid with
            # a single voxel in y), 2D disk (radius 0.2), 3D ball. Note that the
            # 3D branch compares the *squared* distance with 0.1 (radius ~0.316).
            phase_field = np.ones(nb_voxels)
            if nb_voxels.size == 1:
                phase_field[(np.sqrt(np.power(coordinates[0] - 0.5, 2))) < 0.2] = 0
            elif nb_voxels.size == 2 and nb_voxels[1] == 1:
                phase_field[(np.sqrt(np.power(coordinates[0] - 0.5, 2))) < 0.2] = 0
            elif nb_voxels.size == 2:
                phase_field[(np.sqrt(np.power(coordinates[0] - 0.5, 2) + np.power(coordinates[1] - 0.5, 2))) < 0.2] = 0
            elif nb_voxels.size == 3:
                phase_field[
                    np.power(coordinates[0] - 0.5, 2) +
                    np.power(coordinates[1] - 0.5, 2) +
                    np.power(coordinates[2] - 0.5, 2) < 0.1] = 0

        case 'circle_inclusions':
            # Regular array of nb_circles x nb_circles disks in a zero matrix.
            nb_circles = kwargs['nb_circles']
            r_0 = kwargs['r_0']
            vol_frac = kwargs['vol_frac']  # read but currently unused
            # Radius scaled with the number of disks per direction so that the
            # total volume fraction stays constant (~ pi r_0^2) when refining.
            r_n = r_0 / nb_circles
            random_density = kwargs['random_density']
            random_centers = kwargs['random_centers']

            # Each disk lives in its own box of size 1/nb_circles; the maximal
            # centre perturbation keeps the disk inside that box.
            inclusion_box_size = 1 / nb_circles
            perturb_of_centers = inclusion_box_size / 2 - r_n

            phase_field = np.zeros(nb_voxels)
            dim = np.size(nb_voxels)
            # number of circles in one direction
            if dim == 2:
                nb_circles_i = np.asarray(nb_circles, dtype=int)

                # 1D centres at the midpoints of the nb_circles boxes in [0, 1).
                center = np.linspace(0, 1, nb_circles_i, endpoint=False)
                if nb_circles == 1:
                    center += 1 / 2
                else:
                    center += (center[1] - center[0]) / 2
                centers_x, centers_y = np.meshgrid(center, center)

                # Iterate over the results
                # Random permutation of the labels 1..nb_circles**dim: each disk
                # gets a distinct integer value when random_density is True.
                densities = np.random.permutation(np.arange(1, 1 + nb_circles ** dim))
                counter = 0
                for i in range(centers_x.shape[0]):
                    for j in range(centers_y.shape[1]):

                        center = np.array([centers_x[i, j], centers_y[i, j]])
                        if random_centers:
                            # A single random scalar is added to both
                            # components, i.e. the shift is along the (1,1) diagonal.
                            center += (perturb_of_centers * 0.99) * np.random.uniform(-1, 1)
                        # Distance of every pixel to the current centre
                        # (non-periodic: disks are not wrapped around the cell).
                        r_center = np.zeros_like(coordinates)
                        for d in np.arange(dim):
                            r_center[d] = coordinates[d, ...] - center[d]

                        squares = 0
                        squares += sum(r_center[d] ** 2 for d in range(dim))
                        distances = np.sqrt(squares)
                        if random_density:
                            # Create an array from 1 to 10

                            # Shuffle the array randomly

                            # phase_field[distances < r_n] = np.random.random()
                            phase_field[distances < r_n] = densities[counter]
                        else:
                            phase_field[distances < r_n] = 1
                        counter += 1
            elif dim == 3:
                raise (NotImplementedError)

        case 'abs_val':
            # "Pyramid" field |x-0.5| + |y-0.5| (0 at the centre, 1 at the corners).
            phase_field = np.zeros(nb_voxels)
            if nb_voxels.size == 2:
                phase_field = np.abs(coordinates[0] - 0.5) + np.abs(coordinates[1] - 0.5)
            elif nb_voxels.size == 3:
                # NOTE: the result is not assigned, so in 3D the field stays zero.
                np.abs(coordinates[0] - 0.5) + np.abs(coordinates[1] - 0.5) + np.abs(coordinates[2] - 0.5)

        case 'sine_wave_rapid':
            # Values in [0, 1]; 30 periods in x and 10 periods in y over the unit cell.
            phase_field = np.zeros(nb_voxels)
            if nb_voxels.size == 2:
                phase_field = 0.5 + 0.25 * np.cos(30 * 2 * np.pi * coordinates[0]) + 0.25 * np.cos(
                    10 * 2 * np.pi * coordinates[1])
            elif nb_voxels.size == 3:
                # Placeholder: returns sin of all coordinate components, i.e.
                # an array of shape (3, *nb_voxels), not (*nb_voxels).
                phase_field = np.sin(coordinates)
        case 'sine_wave':
            # Periodic field with 3 periods per direction; values in [0, 1] in
            # 2D and [-0.25, 1.25] in 3D (three cosine terms).
            phase_field = np.zeros(nb_voxels)
            if nb_voxels.size == 2:
                phase_field = 0.5 + 0.25 * np.cos(3 * 2 * np.pi * coordinates[0]) + 0.25 * np.cos(
                    3 * 2 * np.pi * coordinates[1])
            elif nb_voxels.size == 3:
                phase_field = 0.5 + 0.25 * np.cos(3 * 2 * np.pi * coordinates[0]) + 0.25 * np.cos(
                    3 * 2 * np.pi * coordinates[1]) + 0.25 * np.cos(
                    3 * 2 * np.pi * coordinates[2])
        case 'sine_wave_':
            # One-period cosines along the two diagonals (x-y) and (x+y);
            # values in [0, 1].
            phase_field = np.zeros(nb_voxels)
            if nb_voxels.size == 2:
                phase_field = 0.5 + 0.25 * np.cos(
                    2 * np.pi * coordinates[0] - 2 * np.pi * coordinates[1]) + 0.25 * np.cos(
                    2 * np.pi * coordinates[1] + 2 * np.pi * coordinates[0])
            elif nb_voxels.size == 3:
                phase_field = (0.5 + 0.25 * np.cos(
                    2 * np.pi * coordinates[0] - 2 * np.pi * coordinates[1] - 2 * np.pi * coordinates[2]) +
                               0.25 * np.cos(
                            2 * np.pi * coordinates[1] + 2 * np.pi * coordinates[0] + 2 * np.pi * coordinates[2]))
        case 'cos_wave':
            # 2D: one-period cosines along x and y; 3D: two oblique cosines with
            # different wave vectors (note: the 3*pi*z term in the second cosine
            # is not 1-periodic in z).
            phase_field = np.zeros(nb_voxels)
            if nb_voxels.size == 2:
                phase_field = (0.5 + 0.25 * np.cos(
                    2 * np.pi * coordinates[0])
                               + 0.25 * np.cos(
                            2 * np.pi * coordinates[1]))

            elif nb_voxels.size == 3:
                phase_field = (0.5 + 0.25 * np.cos(
                    2 * np.pi * coordinates[0] - 2 * np.pi * coordinates[1] - 4 * np.pi * coordinates[2]) +
                               0.25 * np.cos(
                            2 * np.pi * coordinates[1] + 2 * np.pi * coordinates[0] + 3 * np.pi * coordinates[2]))
        case 'sine_wave_inv':
            # 2D: 1 - 'sine_wave_' (inverted). The 3D branch is identical to
            # 'sine_wave_' (not inverted).
            phase_field = np.zeros(nb_voxels)
            if nb_voxels.size == 2:
                phase_field = 0.5 - 0.25 * np.cos(
                    2 * np.pi * coordinates[0] - 2 * np.pi * coordinates[1]) - 0.25 * np.cos(
                    2 * np.pi * coordinates[1] + 2 * np.pi * coordinates[0])
            elif nb_voxels.size == 3:
                phase_field = (0.5 + 0.25 * np.cos(
                    2 * np.pi * coordinates[0] - 2 * np.pi * coordinates[1] - 2 * np.pi * coordinates[2]) +
                               0.25 * np.cos(
                            2 * np.pi * coordinates[1] + 2 * np.pi * coordinates[0] + 2 * np.pi * coordinates[2]))
        case 'tanh':
            # Saddle-shaped field in (-1, 1): positive in two opposite
            # quadrants around the centre, negative in the other two.
            phase_field = np.zeros(nb_voxels)
            if nb_voxels.size == 2:
                phase_field = np.tanh((coordinates[0] - 0.5) / 0.3 * (coordinates[1] - 0.5) / 0.3)
            elif nb_voxels.size == 3:
                # Placeholder, returns an array of shape (3, *nb_voxels).
                phase_field = np.sin(coordinates)
        case 'linear':
            phase_field = np.zeros(nb_voxels)
            if nb_voxels.size == 2:
                phase_field = coordinates[0]
            elif nb_voxels.size == 3:
                phase_field = coordinates[0]
        case 'bilinear':
            phase_field = np.zeros(nb_voxels)
            if nb_voxels.size == 2:
                phase_field = coordinates[0] * coordinates[1]
            elif nb_voxels.size == 3:
                phase_field = coordinates[0] * coordinates[1] * coordinates[2]

        case 'symmetric_linear':
            phase_field = np.zeros(nb_voxels)
            if nb_voxels.size == 2:
                # NOTE: phase_field is a view of coordinates[0] (no copy), so
                # the mirroring below also modifies the caller's coordinates.
                phase_field = coordinates[0]
                # Mirror the first half (in x-index) onto the second half -> tent profile.
                phase_field[nb_voxels[0] // 2:] = np.flipud(phase_field[:nb_voxels[0] // 2])


            elif nb_voxels.size == 3:
                # NOTE: raising a str is itself a TypeError in Python 3.
                raise "Not IMPLEMENTED"
        # --- Category I : Material in faces
        # The 3D voxel geometries below are built in voxel-index space on the
        # full grid (coordinates are ignored); the check_* helpers validate
        # the grid before construction. Strut/plate thickness is typically
        # k = int(0.05 * Nx) voxels.
        case 'geometry_I_1_3D':
            check_dimension(nb_voxels=nb_voxels, microstructure_name=microstructure_name)
            check_equal_number_of_voxels(nb_voxels=nb_voxels, microstructure_name=microstructure_name)
            check_number_of_voxels(nb_voxels=nb_voxels, microstructure_name=microstructure_name, min_nb_voxels=19)
            #  Cube Frame
            phase_field = HSCC(*nb_voxels)

        case 'geometry_I_2_3D':
            #  Cube Frame with one diagonal in each face
            check_dimension(nb_voxels=nb_voxels, microstructure_name=microstructure_name)
            check_equal_number_of_voxels(nb_voxels=nb_voxels, microstructure_name=microstructure_name)
            check_number_of_voxels(nb_voxels=nb_voxels, microstructure_name=microstructure_name, min_nb_voxels=19)

            phase_field = HFDC(*nb_voxels)

        case 'geometry_I_3_3D':
            # Cube Frame with two diagonals in each face
            check_dimension(nb_voxels=nb_voxels, microstructure_name=microstructure_name)
            check_equal_number_of_voxels(nb_voxels=nb_voxels, microstructure_name=microstructure_name)

            phase_field = HFCC(*nb_voxels)

        case 'geometry_I_4_3D':
            # Just two diagonals in each face
            check_dimension(nb_voxels=nb_voxels, microstructure_name=microstructure_name)
            check_equal_number_of_voxels(nb_voxels=nb_voxels, microstructure_name=microstructure_name)

            phase_field = HFCC_no_frame(*nb_voxels)

        case 'geometry_I_5_3D':
            #  Hollow Cube  with the Circle removed from each face
            check_dimension(nb_voxels=nb_voxels, microstructure_name=microstructure_name)
            check_equal_number_of_voxels(nb_voxels=nb_voxels, microstructure_name=microstructure_name)

            phase_field = Circle_Frame(*nb_voxels)

        # --- Category II : in body geometries
        case 'geometry_II_0_3D':
            # Filled Cube
            check_dimension(nb_voxels=nb_voxels, microstructure_name=microstructure_name)
            # here should come your code
            phase_field = Normalcube(*nb_voxels)

        case 'geometry_II_1_3D':
            # Cube with the Body Diagonals
            check_dimension(nb_voxels=nb_voxels, microstructure_name=microstructure_name)
            check_equal_number_of_voxels(nb_voxels=nb_voxels, microstructure_name=microstructure_name)

            phase_field = HBCC(*nb_voxels)

        case 'geometry_II_3_3D':
            # Cube with a another isocentric connected with the diagonals of Both Cubes Subtracted
            check_dimension(nb_voxels=nb_voxels, microstructure_name=microstructure_name)
            check_equal_number_of_voxels(nb_voxels=nb_voxels, microstructure_name=microstructure_name)

            phase_field = Metamaterial_1(*nb_voxels)

        case 'geometry_II_4_3D':
            #  Filled Cube with a Sphere removed from it
            check_dimension(nb_voxels=nb_voxels, microstructure_name=microstructure_name)
            # here should come your code
            phase_field = SphereinCube(*nb_voxels)

        # --- Category III : Metamaterials
        case 'geometry_III_1_3D':
            #  lightweight strong metamaterial
            # Elongated cell: Metamaterial_3 builds Nz = int(1.8 * Nx) itself;
            # the ratio check requires the passed nb_voxels to satisfy
            # 1.8*Nx == Nz (given Nx == Ny).
            check_dimension(nb_voxels=nb_voxels, microstructure_name=microstructure_name)
            check_equal_number_of_voxels(nb_voxels=nb_voxels, microstructure_name=microstructure_name)
            check_unequal_number_of_voxels(nb_voxels=nb_voxels, microstructure_name=microstructure_name,
                                           ratios=[1.8, 1.8, 1])

            phase_field = Metamaterial_3(*nb_voxels)

        case 'geometry_III_2_3D':
            #  ligtweight strong metamaterial
            check_dimension(nb_voxels=nb_voxels, microstructure_name=microstructure_name)
            check_equal_number_of_voxels(nb_voxels=nb_voxels, microstructure_name=microstructure_name)
            check_number_of_voxels(nb_voxels=nb_voxels, microstructure_name=microstructure_name, min_nb_voxels=39)

            phase_field = Metamaterial_2(*nb_voxels)

        case 'geometry_III_3_3D':
            #  ligtweight strong metamaterial
            check_dimension(nb_voxels=nb_voxels, microstructure_name=microstructure_name)
            # check_equal_number_of_voxels(nb_voxels=nb_voxels, microstructure_name=microstructure_name)
            # check_number_of_voxels(nb_voxels=nb_voxels, microstructure_name=microstructure_name, min_nb_voxels=20)

            phase_field = Metamaterial_4(*nb_voxels)

        case 'geometry_III_4_3D':
            # Define a  chiral metamaterial.
            check_dimension(nb_voxels=nb_voxels, microstructure_name=microstructure_name)
            # check_equal_number_of_voxels(nb_voxels=nb_voxels, microstructure_name=microstructure_name)
            # check_number_of_voxels(nb_voxels=nb_voxels, microstructure_name=microstructure_name, min_nb_voxels=20)
            # Default parameters. NOTE: because of the elif chain only the
            # first missing key of a user-supplied dict is filled in.
            if parameter is None:
                parameter = {'lengths': [1., 1., 1],
                             'radius': 0.4,
                             'thickness': 0.1,
                             'alpha': 0.2}
            elif "lengths" not in parameter:
                parameter['lengths'] = [1., 1., 1]
            elif "radius" not in parameter:
                parameter['radius'] = 0.4
            elif "thickness" not in parameter:
                parameter['thickness'] = 0.1
            elif "alpha" not in parameter:
                parameter['alpha'] = 0.2

            phase_field = chiral_metamaterial(nb_grid_pts=nb_voxels,
                                              lengths=parameter['lengths'],
                                              radius=parameter['radius'],
                                              thickness=parameter['thickness'],
                                              alpha=parameter['alpha'])

        case 'geometry_III_5_3D':
            # Define a (more complex) chiral metamaterial.
            check_dimension(nb_voxels=nb_voxels, microstructure_name=microstructure_name)
            # check_equal_number_of_voxels(nb_voxels=nb_voxels, microstructure_name=microstructure_name)
            # check_number_of_voxels(nb_voxels=nb_voxels, microstructure_name=microstructure_name, min_nb_voxels=20)
            # Default parameters (lengths[0], lengths[1] > lengths[2] is
            # required by chiral_metamaterial_2). Only the first missing key
            # of a user-supplied dict is filled in (elif chain).
            if parameter is None:
                parameter = {
                    'lengths': [1.1, 1.1, 1],
                    'radius_out': 0.3,
                    'radius_inn': 0.2,
                    'thickness': 0.1,
                    'alpha': 0.25}

            elif "lengths" not in parameter:
                parameter['lengths'] = [1.1, 1.1, 1]
            elif "radius_out" not in parameter:
                parameter['radius_out'] = 0.3
            elif "radius_inn" not in parameter:
                parameter['radius_inn'] = 0.2
            elif "thickness" not in parameter:
                parameter['thickness'] = 0.1
            elif "alpha" not in parameter:
                parameter['alpha'] = 0.25

            phase_field = chiral_metamaterial_2(nb_grid_pts=nb_voxels,
                                                lengths=parameter['lengths'],
                                                radius_out=parameter['radius_out'],
                                                radius_inn=parameter['radius_inn'],
                                                thickness=parameter['thickness'],
                                                alpha=parameter['alpha'])

    return phase_field  # size is nb_voxels (except the placeholder 3D branches noted above)


def Circle_A(Nx, r):
    """
    2D square mask that is 1 *outside* a centred disk and 0 inside.

    Parameters
    ----------
    Nx : int
        Number of pixels per side; the mask is ``Nx x Nx``.
    r : float
        Disk radius in pixels.

    Returns
    -------
    geom : np.ndarray of float, shape (Nx, Nx)
        1 where the squared distance to the centre exceeds ``r**2``, else 0.

    Notes
    -----
    For odd ``Nx`` the centre is the middle pixel ``Nx // 2``; for even
    ``Nx`` the centre lies between the two middle pixels (``Nx/2 - 0.5``),
    so the disk is symmetric in both cases.
    """
    if Nx % 2 != 0:
        geom = np.zeros((Nx, Nx))
        for i in range(Nx):
            for j in range(Nx):
                if (i - (Nx) // 2) ** 2 + (j - (Nx) // 2) ** 2 > r ** 2:
                    geom[i, j] = 1
        geom = geom[0:Nx, 0:Nx]
    else:
        geom = np.zeros((Nx, Nx))
        for i in range(Nx):
            for j in range(Nx):
                if (i - (Nx) // 2 + 0.5) ** 2 + (j - (Nx) // 2 + 0.5) ** 2 > r ** 2:
                    geom[i, j] = 1
        geom = geom[0:Nx, 0:Nx]
    return geom


def Circle_Frame(*nb_voxels):
    """
    Hollow cube whose six faces are plates with a centred circular hole.

    Each face is a plate of thickness ``k = int(0.05 * Nx)`` voxels; the
    in-plane pattern is the union of :func:`Circle_A` (1 outside a disk of
    radius ``int(0.4 * Nx)``) and the square edge frame
    :func:`square_frame2D`. The interior of the cube is void.

    Parameters
    ----------
    *nb_voxels : int
        ``Nx, Ny, Nz``. Only ``Nx`` is used for the face patterns, so
        ``Nx == Ny == Nz`` is required.

    Returns
    -------
    G : np.ndarray of float, shape (Nx, Nx, Nx)
        1 = solid, 0 = void.
    """
    (Nx, Ny, Nz) = nb_voxels
    # Create cube
    Cube = np.zeros((Nx, Ny, Nz))
    k = int(0.05 * Nx)  # plate thickness in voxels
    r = int(0.4 * Nx)  # hole radius in voxels

    # Create meshgrid (unused, kept from the original implementation)
    I, J, K = np.meshgrid(np.arange(1, Nx + 1), np.arange(1, Ny + 1), np.arange(1, Nz + 1))

    G = np.zeros_like(Cube)
    Frame = square_frame2D(Nx, k)
    Circle = Circle_A(Nx, r)
    Overall = np.logical_or(Circle, Frame)

    # Stamp the 2D face pattern onto the k outermost layers of each of the
    # six faces (x=0/x=-1, y=0/y=-1, z=0/z=-1 sides).
    for i in range(k):
        G[i, :, :] = Overall
        G[-i - 1, :, :] = Overall
        G[:, i, :] = Overall
        G[:, -i - 1, :] = Overall
        G[:, :, i] = Overall
        G[:, :, -i - 1] = Overall

    # Restrict D to Nx x Nx x Nx
    G = G[0:Nx, 0:Nx, 0:Nx]
    return G


def Diagonal2D_A(Nx, k):
    """
    2D mask of a thick main diagonal (from pixel (0, 0) to (Nx-1, Nx-1)).

    Parameters
    ----------
    Nx : int
        Number of pixels per side.
    k : int
        Half-width of the band in pixels: pixels with ``|i - j| <= k`` are set.

    Returns
    -------
    geom : np.ndarray of float, shape (Nx, Nx)
        1 on the diagonal band, 0 elsewhere.
    """
    geom = np.zeros((Nx, Nx))
    for i in range(Nx):
        for j in range(Nx):
            if i == j or (i - k <= j <= i + k):
                geom[i, j] = 1
    geom = geom[0:Nx, 0:Nx]
    return geom


def Diagonal2D_B(Nx, k):
    """
    2D mask of a thick anti-diagonal (from pixel (0, Nx-1) to (Nx-1, 0)).

    Parameters
    ----------
    Nx : int
        Number of pixels per side.
    k : int
        Half-width of the band in pixels.

    Returns
    -------
    geom : np.ndarray of float, shape (Nx, Nx)
        1 on the anti-diagonal band, 0 elsewhere.

    Notes
    -----
    The band is ``Nx-1-i-k <= j <= Nx-1-i+k`` (centred on ``i + j = Nx-1``),
    plus the single extra line ``j = Nx - i`` (which lies inside the band for
    ``k >= 1``).
    """
    geom = np.zeros((Nx, Nx))
    for i in range(Nx):
        for j in range(Nx):
            if (Nx) - i == j or (Nx) - (i + k + 1) <= j <= (Nx) - (i - k + 1):
                geom[i, j] = 1
    geom = geom[0:Nx, 0:Nx];
    return geom


def Diagonal2D_FACE(Nx, k):
    """
    2D "X" mask: union of both thick diagonals.

    Parameters
    ----------
    Nx : int
        Number of pixels per side.
    k : int
        Half-width of each diagonal band in pixels.

    Returns
    -------
    C : np.ndarray of int, shape (Nx, Nx)
        1 on either diagonal (:func:`Diagonal2D_A` or :func:`Diagonal2D_B`),
        0 elsewhere.
    """
    A =Diagonal2D_A(Nx, k)
    B = Diagonal2D_B(Nx, k)
    C = np.logical_or(A, B).astype(int)
    return C


def HBCC(*nb_voxels):
    """
    Body-centred cubic (BCC) truss: cube edge frame plus the 4 body diagonals.

    The 12 cube edges are struts of thickness ``k = int(0.05 * Nx)`` voxels
    (obtained by stamping :func:`square_frame2D` on the six faces), and the
    four space diagonals connecting opposite corners are struts of half-width
    ``l = k * sqrt(3)`` (measured per coordinate).

    Parameters
    ----------
    *nb_voxels : int
        ``Nx, Ny, Nz``. The diagonals are built with ``Nx`` only, so
        ``Nx == Ny == Nz`` is required.

    Returns
    -------
    H : np.ndarray of float, shape (Nx, Ny, Nz)
        1 = solid, 0 = void.
    """
    (Nx, Ny, Nz) = nb_voxels
    # Create cube
    Cube = np.zeros((Nx, Ny, Nz))
    k = int(0.05 * Nx)  # strut thickness in voxels
    # Create meshgrid (unused, kept from the original implementation)
    I, J, K = np.meshgrid(np.arange(1, Nx + 1), np.arange(1, Ny + 1), np.arange(1, Nz + 1))

    H = np.zeros_like(Cube)
    Frame = square_frame2D(Nx, k)

    # TopandBottom=np.logical_or(Frame,Diagonal1)
    # Sides=np.logical_or(Frame,Diagonal2)
    l = k * np.sqrt(3)  # half-width of the diagonal struts
    # Edge frame: a square frame on each of the 6 faces -> 12 edge struts.
    for i in range(k):
        H[i, :, :] = Frame
        H[-i - 1, :, :] = Frame
        H[:, i, :] = Frame
        H[:, -i - 1, :] = Frame
        H[:, :, i] = Frame
        H[:, :, -i - 1] = Frame

    # Body diagonals, parametrised by j (the y-index). Each condition marks
    # voxels within l of one diagonal in both remaining coordinates:
    #   1) (j, j, j)              2) (Nx-1-j, j, j)
    #   3) (j, j, Nx-1-j)         4) (Nx-1-j, j, Nx-1-j)
    # NOTE: the loop variable k shadows the strut thickness k (no longer needed).
    for i in range(Nx):
        for j in range(Nx):
            for k in range(Nx):
                if i <= j + l and i >= j - l and k <= j + l and k >= j - l:
                    H[i, j, k] = 1
                if i <= Nx - j - 1 + l and i >= Nx - j - 1 - l and k <= j + l and k >= j - l:
                    H[i, j, k] = 1
                if i <= j + l and i >= j - l and k + 1 <= Nx - j + l and k + 1 >= Nx - j - l:
                    H[i, j, k] = 1
                if Nx - i - 1 <= j + l and Nx - i - 1 >= j - l and Nx - k - 1 <= j + l and Nx - k - 1 >= j - l:
                    H[i, j, k] = 1

    # Restrict D to Nx x Nx x Nx
    H = H[0:Nx, 0:Ny, 0:Nz]
    return H


def HFCC(*nb_voxels):
    """
    Face-centred cubic (FCC) truss: cube edge frame plus both diagonals on
    every face.

    The 2D face pattern (:func:`square_frame2D` OR :func:`Diagonal2D_FACE`)
    is stamped onto the ``k = int(0.05 * Nx)`` outermost voxel layers of each
    of the six faces; the interior is void.

    Parameters
    ----------
    *nb_voxels : int
        ``Nx, Ny, Nz``; ``Nx == Ny == Nz`` is required since the face
        pattern is ``Nx x Nx``.

    Returns
    -------
    D : np.ndarray of float, shape (Nx, Ny, Nz)
        1 = solid, 0 = void.
    """
    (Nx, Ny, Nz) = nb_voxels
    # Create cube
    Cube = np.zeros((Nx, Ny, Nz))
    k = int(0.05 * Nx)

    # Create meshgrid
    I, J, K = np.meshgrid(np.arange(1, Nx + 1), np.arange(1, Ny + 1), np.arange(1, Nz + 1))

    # Get Diagonal2D_FACE matrix
    Face = Diagonal2D_FACE(Nx, k)
    Frame = square_frame2D(Nx, k)
    Overall = np.logical_or(Face, Frame)
    # Assign values to D
    D = np.zeros_like(Cube)
    for i in range(k):
        D[i, :, :] = Overall
        D[-i - 1, :, :] = Overall
        D[:, i, :] = Overall
        D[:, -i - 1, :] = Overall
        D[:, :, i] = Overall
        D[:, :, -i - 1] = Overall

    # Restrict D to Nx x Nx x Nx
    D = D[0:Nx, 0:Ny, 0:Nz]
    return D


def HFCC_no_frame(*nb_voxels):
    """
    Like :func:`HFCC` but without the edge frame: only the two diagonals on
    each face (thickness ``k = int(0.05 * Nx)`` voxels).

    Parameters
    ----------
    *nb_voxels : int
        ``Nx, Ny, Nz``; ``Nx == Ny == Nz`` is required.

    Returns
    -------
    D : np.ndarray of float, shape (Nx, Ny, Nz)
        1 = solid, 0 = void.
    """
    (Nx, Ny, Nz) = nb_voxels
    # Create cube
    Cube = np.zeros((Nx, Ny, Nz))
    k = int(0.05 * Nx)

    # Create meshgrid
    I, J, K = np.meshgrid(np.arange(1, Nx + 1), np.arange(1, Ny + 1), np.arange(1, Nz + 1))

    # Get Diagonal2D_FACE matrix
    Face = Diagonal2D_FACE(Nx, k)
    Overall = np.logical_or(Face, Face)  # == Face (as bool); no frame added
    # Assign values to D
    D = np.zeros_like(Cube)
    for i in range(k):
        D[i, :, :] = Overall
        D[-i - 1, :, :] = Overall
        D[:, i, :] = Overall
        D[:, -i - 1, :] = Overall
        D[:, :, i] = Overall
        D[:, :, -i - 1] = Overall

    # Restrict D to Nx x Nx x Nx
    D = D[0:Nx, 0:Ny, 0:Nz]
    return D


def HSCC(*nb_voxels):
    """
    Simple cubic (SC) truss: only the 12 edges of the cube.

    Obtained by stamping the square frame :func:`square_frame2D` (border width
    ``k = int(0.05 * Nx)``) onto the ``k`` outermost layers of each face.
    Because the frames of adjacent faces coincide at the edges, the result
    consists of edge struts of cross-section ``k x k``.

    Parameters
    ----------
    *nb_voxels : int
        ``Nx, Ny, Nz``; ``Nx == Ny == Nz`` is required.

    Returns
    -------
    E : np.ndarray of float, shape (Nx, Ny, Nz)
        1 = solid, 0 = void.
    """
    (Nx, Ny, Nz) = nb_voxels
    # Create cube
    Cube = np.zeros((Nx, Ny, Nz))
    k = int(0.05 * Nx)
    Frame = square_frame2D(Nx, k)
    E = np.zeros_like(Cube)
    for i in range(k):
        E[i, :, :] = Frame
        E[-i - 1, :, :] = Frame
        E[:, i, :] = Frame
        E[:, -i - 1, :] = Frame
        E[:, :, i] = Frame
        E[:, :, -i - 1] = Frame

    # Restrict D to Nx x Nx x Nx
    E = E[0:Nx, 0:Ny, 0:Nz]
    return E


def HFDC(*nb_voxels):
    """
    Cube edge frame plus a single diagonal (:func:`Diagonal2D_A`) on each face.

    Parameters
    ----------
    *nb_voxels : int
        ``Nx, Ny, Nz``; ``Nx == Ny == Nz`` is required.

    Returns
    -------
    D : np.ndarray of float, shape (Nx, Ny, Nz)
        1 = solid, 0 = void. Strut thickness ``k = int(0.05 * Nx)`` voxels.
    """
    (Nx, Ny, Nz) = nb_voxels
    # Create cube
    Cube = np.zeros((Nx, Ny, Nz))
    k = int(0.05 * Nx)

    # Create meshgrid
    I, J, K = np.meshgrid(np.arange(1, Nx + 1), np.arange(1, Ny + 1), np.arange(1, Nz + 1))

    # Get the single-diagonal face matrix (Diagonal2D_A, not the X of Diagonal2D_FACE)
    Face = Diagonal2D_A(Nx, k)
    Frame = square_frame2D(Nx, k)
    Overall = np.logical_or(Face, Frame)
    # Assign values to D
    D = np.zeros_like(Cube)
    for i in range(k):
        D[i, :, :] = Overall
        D[-i - 1, :, :] = Overall
        D[:, i, :] = Overall
        D[:, -i - 1, :] = Overall
        D[:, :, i] = Overall
        D[:, :, -i - 1] = Overall

    # Restrict D to Nx x Nx x Nx
    D = D[0:Nx, 0:Ny, 0:Nz]
    return D


def Kite(Nx, k):
    """
    2D "kite"/diamond mask: four thick lines connecting the edge midpoints.

    The lines are ``j = t - 1 - i``, ``i = j + t``, ``j = i + t`` and
    ``j = Nx + t - 1 - i`` with ``t = Nx // 2``, i.e. a rhombus inscribed in
    the square with its vertices at the midpoints of the four edges.

    Parameters
    ----------
    Nx : int
        Number of pixels per side.
    k : int
        Line thickness parameter: each line pixel is also set at offsets
        ``0..k-1`` (in both directions along one axis).

    Returns
    -------
    geom : np.ndarray of float, shape (Nx, Nx)
        1 on the rhombus lines, 0 elsewhere.

    Notes
    -----
    The work array has size ``(Nx + k, Nx + k)`` so that ``i + k`` /
    ``j + k`` never go out of bounds; negative indices ``i - k`` wrap into
    this padding region. The padding is cropped at the end.
    """
    # Nx=100
    t = Nx // 2
    # k=5
    geom = np.zeros((Nx + k, Nx + k))

    # The outer loop variable shadows k: it runs over the thickness offsets
    # 0..k-1 (range(k) is evaluated before k is rebound).

    for k in range(k):
        for i in range(0, Nx):
            for j in range(0, Nx):
                if j == -i + t - 1:
                    geom[i, j] = 1
                    geom[i + k, j] = 1
                    geom[i - k, j] = 1

        for i in range(0, Nx):
            for j in range(0, Nx):
                if i == j + t:
                    geom[i, j] = 1
                    geom[i, j + k] = 1
                    geom[i, j - k] = 1

        for i in range(0, Nx):
            for j in range(0, Nx):
                if j == i + t:
                    geom[i, j] = 1
                    geom[i + k, j] = 1
                    geom[i - k, j] = 1

        for i in range(0, Nx):
            for j in range(0, Nx):
                if j == -i + Nx + t - 1:
                    geom[i, j] = 1
                    geom[i + k, j] = 1
                    geom[i - k, j] = 1

    geom = geom[0:Nx, 0:Nx]
    return geom


def Normalcube(*nb_voxels):
    """
    Completely filled cube (all voxels solid).

    Parameters
    ----------
    *nb_voxels : int
        ``Nx, Ny, Nz``.

    Returns
    -------
    geom : np.ndarray of float, shape (Nx, Ny, Nz)
        Array of ones.
    """
    (Nx, Ny, Nz) = nb_voxels
    # Create cube
    geom = np.ones((Nx, Ny, Nz))
    geom = geom[0:Nx, 0:Ny, 0:Nz]
    return geom


def Sphere(Nx, r):
    """
    3D mask of a solid ball centred in an ``Nx^3`` cube.

    Parameters
    ----------
    Nx : int
        Number of voxels per side.
    r : float
        Ball radius in voxels. May exceed ``Nx/2``, in which case the ball is
        truncated by the cube faces.

    Returns
    -------
    geom : np.ndarray of float, shape (Nx, Nx, Nx)
        1 inside the ball (distance < r), 0 outside. Centre convention as in
        :func:`Circle_A` (middle voxel for odd ``Nx``, between voxels for even).
    """
    geom = np.zeros((Nx, Nx, Nx))
    for i in range(Nx):
        for j in range(Nx):
            for k in range(Nx):
                if Nx % 2 != 0:
                    if np.sqrt((i - Nx // 2) ** 2 + (j - Nx // 2) ** 2 + (k - Nx // 2) ** 2) < r:
                        geom[i, j, k] = 1
                else:
                    if np.sqrt((i - Nx // 2 + 0.5) ** 2 + (j - Nx // 2 + 0.5) ** 2 + (k - Nx // 2 + 0.5) ** 2) < r:
                        geom[i, j, k] = 1
    geom = geom[0:Nx, 0:Nx]
    return geom


def SphereinCube(*nb_voxels):
    """
    Filled cube with a centred ball removed (1 outside the ball, 0 inside).

    The radius ``r = int(0.6 * Nx)`` is larger than half the cube size, so
    the ball cuts through all faces: the remaining solid consists of the
    eight corner regions, connected along the edges where the distance from
    the centre exceeds ``r``.

    Parameters
    ----------
    *nb_voxels : int
        ``Nx, Ny, Nz``; ``Nx == Ny == Nz`` is required (the ball uses ``Nx``).

    Returns
    -------
    N : np.ndarray of float, shape (Nx, Nx, Nx)
        1 = solid, 0 = void.
    """
    (Nx, Ny, Nz) = nb_voxels
    # Create cube
    Cube = np.zeros((Nx, Ny, Nz))
    r = int(0.6 * Nx)  # ball radius in voxels (> Nx/2)

    # Create meshgrid
    I, J, K = np.meshgrid(np.arange(1, Nx + 1), np.arange(1, Nx + 1), np.arange(1, Nx + 1))

    N = np.zeros_like(Cube)
    N[:, :, :] = Normalcube(*nb_voxels) - Sphere(Nx, r)
    # Restrict D to Nx x Nx x Nx
    N = N[0:Nx, 0:Nx, 0:Nx]
    return N


def square_frame2D_flexible(Nx, k1, k2):
    """
    2D square ring located between depths ``k1`` and ``k2`` from the border.

    Computed as the XOR of two frames :func:`square_frame2D` of widths
    ``k1`` and ``k2``; for ``k1 < k2`` it contains the pixels whose distance
    (in pixels) to the nearest edge lies in ``[k1, k2)``.

    Parameters
    ----------
    Nx : int
        Number of pixels per side.
    k1, k2 : int
        Inner and outer frame widths in pixels.

    Returns
    -------
    geom : np.ndarray of int, shape (Nx, Nx)
        1 on the ring, 0 elsewhere.
    """
    geom1 = square_frame2D(Nx, k1)
    geom2 = square_frame2D(Nx, k2)
    geom = np.logical_xor(geom1, geom2).astype(int)
    geom = geom[0:Nx, 0:Nx]
    return geom


def square_frame2D(Nx, k):
    """
    2D square frame: all pixels within ``k`` pixels of any edge.

    Parameters
    ----------
    Nx : int
        Number of pixels per side.
    k : int
        Frame (border) width in pixels.

    Returns
    -------
    geom : np.ndarray of float, shape (Nx, Nx)
        1 on the frame, 0 in the interior.
    """
    geom = np.zeros((Nx, Nx))
    for i in range(Nx):
        for j in range(Nx):
            if i < k or Nx - i <= k or j < k or Nx - j <= k:
                geom[i, j] = 1
    geom = geom[0:Nx, 0:Nx]
    return geom


def Metamaterial_1(*nb_voxels):
    """
    Nested-cube metamaterial ("geometry_II_3_3D").

    Built from three parts (1 = solid):

    1. square rings (:func:`square_frame2D_flexible` between depths
       ``k1 = int(0.30 Nx)`` and ``k2 = int(0.35 Nx)``) stamped on the
       ``k`` planes at depth ``k1..k1+k-1`` from each face -- these form the
       edges of a smaller, concentric inner cube;
    2. the outer cube edge frame (as in :func:`HSCC`);
    3. the four body diagonals (as in :func:`HBCC`), which connect the
       corners of the outer and the inner cube.

    Finally the core ``[k2, Nx-k2)^3`` is cleared, so the diagonals do not
    pass through the centre.

    Parameters
    ----------
    *nb_voxels : int
        ``Nx, Ny, Nz``; ``Nx == Ny == Nz`` is required.

    Returns
    -------
    M : np.ndarray of int, shape (Nx, Nx, Nx)
        1 = solid, 0 = void. Strut thickness ``k = int(0.05 * Nx)``.
    """
    (Nx, Ny, Nz) = nb_voxels
    # Create cube
    Cube = np.zeros((Nx, Ny, Nz))
    k = int(0.05 * Nx)  # strut thickness
    k2 = int(0.35 * Nx)  # outer depth of the inner-cube rings / half-size of the cleared core
    k1 = int(0.30 * Nx)  # inner depth (position) of the inner-cube planes
    l = k * np.sqrt(3)  # half-width of the diagonal struts
    # Create cube
    Cube = np.zeros((Nx, Ny, Nz), dtype=int)

    # Create meshgrid
    I, J, K = np.meshgrid(np.arange(1, Nx + 1), np.arange(1, Ny + 1), np.arange(1, Nz + 1))
    Frame1 = square_frame2D_flexible(Nx, k1, k2)
    Frame2 = square_frame2D(Nx, k)

    # Assign values to D
    M = np.zeros_like(Cube)
    # 1) inner cube: rings on planes at depth k1 .. k1+k-1 from every face
    for i in range(k):
        M[k1 + i, :, :] = Frame1
        M[-k1 - i - 1, :, :] = Frame1
        M[:, k1 + i, :] = Frame1
        M[:, -k1 - i - 1, :] = Frame1
        M[:, :, k1 + i] = Frame1
        M[:, :, -k1 - i - 1] = Frame1

    # 2) outer cube edge frame
    for i in range(k):
        M[i, :, :] = Frame2
        M[-i - 1, :, :] = Frame2
        M[:, i, :] = Frame2
        M[:, -i - 1, :] = Frame2
        M[:, :, i] = Frame2
        M[:, :, -i - 1] = Frame2

    # 3) four body diagonals (same conditions as in HBCC)
    for i in range(Nx):
        for j in range(Nx):
            for k in range(Nx):
                if i <= j + l and i >= j - l and k <= j + l and k >= j - l:
                    M[i, j, k] = 1
                if i <= Nx - j - 1 + l and i >= Nx - j - 1 - l and k <= j + l and k >= j - l:
                    M[i, j, k] = 1
                if i <= j + l and i >= j - l and k + 1 <= Nx - j + l and k + 1 >= Nx - j - l:
                    M[i, j, k] = 1
                if Nx - i - 1 <= j + l and Nx - i - 1 >= j - l and Nx - k - 1 <= j + l and Nx - k - 1 >= j - l:
                    M[i, j, k] = 1

    # 4) clear the central core (k is the loop variable here, not the thickness)
    for i in range(k2, Nx - k2):
        for j in range(k2, Nx - k2):
            for k in range(k2, Nx - k2):
                M[i, j, k] = 0

    # Restrict D to Nx x Nx x Nx
    M = M[0:Nx, 0:Nx, 0:Nx]
    return M


def Metamaterial_2(*nb_voxels):
    """
    Metamaterial with face diagonals and "kite" mid-planes ("geometry_III_2_3D").

    * Outer faces: both face diagonals (:func:`Diagonal2D_FACE`) stamped on
      the ``k = int(0.05 * Nx)`` outermost layers of each face (no edge frame).
    * Mid-planes: the rhombus pattern :func:`Kite` stamped on the planes
      ``h-l+1 .. h+l-1`` around the centre ``h = Nx // 2`` normal to each of
      the three axes.

    Parameters
    ----------
    *nb_voxels : int
        ``Nx, Ny, Nz``; ``Nx == Ny == Nz`` is required.

    Returns
    -------
    D : np.ndarray of float, shape (Nx, Nx, Nx)
        1 = solid, 0 = void.
    """
    (Nx, Ny, Nz) = nb_voxels
    # Create cube
    Cube = np.zeros((Nx, Ny, Nz))
    k = int(0.05 * Nx)  # thickness of the face diagonals
    # Create meshgrid (unused, kept from the original implementation)
    I, J, K = np.meshgrid(np.arange(1, Nx + 1), np.arange(1, Nx + 1), np.arange(1, Nx + 1))
    h = int(Nx / 2)  # index of the mid-plane
    l = int((k / 2) + 1)  # half-thickness of the kite mid-planes (also kite line thickness)
    # Get Diagonal2D_FACE matrix
    Face = Diagonal2D_FACE(Nx, k)
    kite = Kite(Nx, l)

    # Assign values to D
    D = np.zeros_like(Cube)
    for i in range(k):
        D[i, :, :] = Face
        D[-i - 1, :, :] = Face
        D[:, i, :] = Face
        D[:, -i - 1, :] = Face
        D[:, :, i] = Face
        D[:, :, -i - 1] = Face

    for i in range(l):
        D[h + i, :, :] = kite
        D[h - i, :, :] = kite
        D[:, h + i, :] = kite
        D[:, h - i, :] = kite
        D[:, :, h + i] = kite
        D[:, :, h - i] = kite

    # Restrict D to Nx x Nx x Nx
    D = D[0:Nx, 0:Nx, 0:Nx]
    return D


def specialface(Nx):
    """
    2D face pattern used for the outer walls of :func:`Metamaterial_3`.

    Union of (first array index = ``i``):

    * three thick bands along the second axis: ``i < 2k``,
      ``i >= Nx - 2k`` and the central band ``t-k-1 < i < t+k``;
    * the rhombus of :func:`Kite` (lines through the edge midpoints) with
      line thickness ``k``;

    followed by cutting notches into the two outer bands at
    ``j in [0.45 Nx, 0.55 Nx)``.

    Parameters
    ----------
    Nx : int
        Number of pixels per side.

    Returns
    -------
    geom : np.ndarray of float, shape (Nx, Nx)
        1 = solid, 0 = void. ``k = int(0.05 * Nx)``, ``t = Nx // 2``.

    Notes
    -----
    As in :func:`Kite`, the work array is padded to ``Nx + k`` so that
    shifted indices stay in bounds; it is cropped at the end.
    """
    k = int(0.05 * Nx)
    t = int(Nx // 2)
    geom = np.zeros((Nx + k, Nx + k))
    for i in range(Nx):
        for j in range(Nx):
            if i < 2 * k or Nx - i <= 2 * k or t - k - 1 < i < t + k:
                geom[i, j] = 1

            if j == -i + t - 1:
                geom[i, j] = 1
                geom[i + k, j] = 1
                geom[i - k, j] = 1
            if i == j + t:
                geom[i, j] = 1
                geom[i, j + k] = 1
                geom[i, j - k] = 1
            if j == i + t:
                geom[i, j] = 1
                geom[i + k, j] = 1
                geom[i - k, j] = 1
            if j == -i + Nx + t - 1:
                geom[i, j] = 1
                geom[i + k, j] = 1
                geom[i - k, j] = 1

    # Cut notches into the outer bands around the middle of the second axis.
    geom[0:2 * k, int(0.45 * Nx):int(0.55 * Nx)] = 0
    geom[Nx - 2 * k - 1:Nx, int(0.45 * Nx):int(0.55 * Nx)] = 0

    geom = geom[0:Nx, 0:Nx]
    return geom


# %%
def specialface1(Nx):
    """
    2D pattern used for the internal mid-planes of :func:`Metamaterial_3`.

    Union of both thick diagonals (same bands as :func:`Diagonal2D_A` and
    :func:`Diagonal2D_B`, half-width ``k = int(0.05 * Nx)``) and a central
    band ``t-k <= i < t+k`` (``t = Nx // 2``) along the second axis.

    Parameters
    ----------
    Nx : int
        Number of pixels per side.

    Returns
    -------
    geom : np.ndarray of float, shape (Nx, Nx)
        1 = solid, 0 = void.
    """
    k = int(0.05 * Nx)
    t = int(Nx // 2)
    geom = np.zeros((Nx, Nx))
    for i in range(Nx):
        for j in range(Nx):
            if i == j or (i - k <= j <= i + k):
                geom[i, j] = 1
    for i in range(Nx):
        for j in range(Nx):
            if (Nx) - i == j or (Nx) - (i + k + 1) <= j <= (Nx) - (i - k + 1):
                geom[i, j] = 1

    # Central band (the loop is redundant: the same slice is set k times).
    for i in range(k):
        geom[t - k:t + k, :] = 1

    geom = geom[0:Nx, 0:Nx]
    return geom


def height(Nx, h, k):
    """
    Rectangular ``Nx x h`` pattern with a central band (rows ``t-k .. t+k-1``,
    ``t = Nx // 2``) set to 1.

    Used in :func:`Metamaterial_3` to fill the gap between the two half-cells
    with a central pillar on the side walls.

    Parameters
    ----------
    Nx : int
        Size along the first axis.
    h : int
        Size along the second axis (height of the gap in voxels).
    k : int
        Half-width of the band.

    Returns
    -------
    geom : np.ndarray of float, shape (Nx, h)
    """
    t = Nx // 2
    geom = np.zeros((Nx, h))
    geom[t - k:t + k, :] = 1
    geom = geom[0:Nx, 0:h]
    return geom


def height1(Nx, h, k):
    """
    Rectangular ``Nx x h`` pattern with two border bands (rows ``0..2k-1``
    and ``Nx-2k..Nx-1``) set to 1 (corner pillars in :func:`Metamaterial_3`).

    Parameters
    ----------
    Nx : int
        Size along the first axis.
    h : int
        Size along the second axis.
    k : int
        Half of the band width (bands are ``2k`` wide).

    Returns
    -------
    geom : np.ndarray of float, shape (Nx, h)
    """
    geom = np.zeros((Nx, h))
    geom[0:2 * k, :] = 1
    geom[Nx - 2 * k:Nx, :] = 1
    geom = geom[0:Nx, 0:h]
    return geom


def height2(Nx, h, k):
    """
    Identical to :func:`height`: ``Nx x h`` pattern with a central band
    (rows ``t-k .. t+k-1``, ``t = Nx // 2``) set to 1.

    Parameters
    ----------
    Nx : int
        Size along the first axis.
    h : int
        Size along the second axis.
    k : int
        Half-width of the band.

    Returns
    -------
    geom : np.ndarray of float, shape (Nx, h)
    """
    t = Nx // 2
    geom = np.zeros((Nx, h))
    geom[t - k:t + k, :] = 1
    geom = geom[0:Nx, 0:h]
    return geom


def Metamaterial_3(*nb_voxels):
    """
    Elongated "lightweight strong" metamaterial cell ("geometry_III_1_3D").

    Construction (1 = solid):

    1. A cube ``D`` of size ``Nx^3`` with walls of thickness ``2k`` normal to
       x and y carrying :func:`specialface`, and mid-planes (thickness about
       ``2k`` around ``t = Nx // 2``) normal to x and y carrying
       :func:`specialface1`. No walls normal to z.
    2. The cube is split into its lower and upper halves along z, which are
       placed into an elongated cuboid ``E`` of size ``Nx x Nx x Nz``,
       ``Nz = int(1.8 * Nx)``, leaving gaps: ``h1 = h // 2`` slices at
       the bottom and top and about ``h = (Nz - Nx) // 2`` slices in the
       middle.
    3. The gaps are bridged by vertical pillars on the side walls: central
       bands (:func:`height`) in the middle gap, border bands
       (:func:`height1`) and central bands (:func:`height2`, on the mid-planes)
       in the top/bottom gaps.

    Parameters
    ----------
    *nb_voxels : int
        ``Nx, Ny, Nz``. Only ``Nx`` is used; ``Ny`` and the passed ``Nz`` are
        ignored and ``Nz`` is recomputed as ``int(1.8 * Nx)``.

    Returns
    -------
    E : np.ndarray of float, shape (Nx, Nx, int(1.8 * Nx))
        1 = solid, 0 = void.

    Notes
    -----
    The slices ``[..., -t-1:-1]`` and ``[..., -h1-1:-1]`` exclude the very
    last z-slice, so the last layer ``E[:, :, -1]`` stays void.
    """
    (Nx, Ny, Nz) = nb_voxels
    # Create cube
    Cube = np.zeros((Nx, Nx, Nx))
    Nz = int(Nx + 0.80 * Nx)  # elongated z-size (overrides the passed Nz)
    k = int(0.05 * Nx)  # base strut thickness
    t = Nx // 2  # mid-plane index
    h = (Nz - Nx) // 2  # height of the middle gap
    h1 = h // 2  # height of the top/bottom gaps
    # Create mesh grid (unused, kept from the original implementation)
    I, J, K = np.meshgrid(np.arange(1, Nx + 1), np.arange(1, Nx + 1), np.arange(1, Nz + 1))

    # Get Diagonal2D_FACE matrix
    Face = specialface(Nx)
    Lonee = specialface1(Nx)
    Height = height(Nx, h, k)
    Height1 = height1(Nx, h1, k)
    Height2 = height2(Nx, h1, k)

    D = np.zeros_like(Cube)
    # Assign values to D
    # Internal mid-planes normal to x and y (planes t-k+1 .. t+k-1).
    D[t, :, :] = Lonee
    D[:, t, :] = Lonee

    for i in range(k):
        D[t - i, :, :] = Lonee
        D[t + i, :, :] = Lonee
        D[:, t - i, :] = Lonee
        D[:, t + i, :] = Lonee

    # Outer side walls (thickness 2k) normal to x and y; no walls normal to z.
    for i in range(2 * k):
        D[i, :, :] = Face
        D[-i - 1, :, :] = Face
        D[:, i, :] = Face
        D[:, -i - 1, :] = Face
        # D[:, :, i] = Face
        # D[:, :, -i-1] = Face

    # Restrict D to Nx x Nx x Nx
    D = D[0:Nx, 0:Nx, 0:Nx]

    # Split the cube along z and place the halves into the elongated cuboid:
    # lower half after a bottom gap of h1 slices, upper half ending h1+1
    # slices before the top.
    Cuboid = np.zeros((Nx, Nx, Nz))
    E = np.zeros_like(Cuboid)
    E[:, :, h1:t + h1] = D[:, :, 0:t]
    E[:, :, -t - 1 - h1:-1 - h1] = D[:, :, -t - 1:-1]

    # Bridge the gaps with pillars on the outer side walls:
    # Height in the middle gap, Height1 (corner bands) in the bottom/top gaps.
    for i in range(2 * k):
        E[i, :, h1 + t:h1 + t + h] = Height
        E[-i - 1, :, h1 + t:h1 + t + h] = Height
        E[:, i, h1 + t:h1 + t + h] = Height
        E[:, -i - 1, h1 + t:h1 + t + h] = Height
        E[i, :, 0:h1] = Height1
        E[-i - 1, :, 0:h1] = Height1
        E[:, i, 0:h1] = Height1
        E[:, -i - 1, 0:h1] = Height1
        E[i, :, -h1 - 1:-1] = Height1
        E[-i - 1, :, -h1 - 1:-1] = Height1
        E[:, i, -h1 - 1:-1] = Height1
        E[:, -i - 1, -h1 - 1:-1] = Height1

    # Central pillars on the internal mid-planes in the bottom/top gaps.
    for i in range(k):
        E[t - i, :, 0:h1] = Height2
        E[t + i, :, 0:h1] = Height2
        E[:, t - i, 0:h1] = Height2
        E[:, t + i, 0:h1] = Height2
        E[:, t - i, -(h1 + 1):-1] = Height2
        E[:, t + i, -(h1 + 1):-1] = Height2
        E[t - i, :, -(h1 + 1):-1] = Height2
        E[t + i, :, -(h1 + 1):-1] = Height2

    E = E[0:Nx, 0:Nx, 0:Nz]

    return E


def Structure2D(Nx, k):
    """
    2D unit pattern used (transposed and tiled) by :func:`Structure2D_FACE`.

    Mechanically, the pattern consists of (first index ``i`` = row):

    * an "X" (:func:`Diagonal2D_FACE` of size ``N = int(0.6 Nx)``) whose
      upper half is placed in rows ``[0.1 Nx, 0.4 Nx)`` and lower half in
      rows ``[0.6 Nx, 0.9 Nx)``, columns ``[0.2 Nx, 0.8 Nx)`` -- i.e. the X is
      split horizontally and its halves pushed apart;
    * two vertical bars (width ~``2k``) at columns ``0.2 Nx`` and ``0.8 Nx``
      spanning rows ``[0.1 Nx, 0.9 Nx)``;
    * a central vertical bar at column ``0.5 Nx`` in rows ``[0, 0.4 Nx)`` and
      ``[0.6 Nx, Nx)``, connecting to the neighbouring cells.

    Parameters
    ----------
    Nx : int
        Number of pixels per side.
    k : int
        Line half-thickness in pixels.

    Returns
    -------
    geom : np.ndarray of float, shape (Nx, Nx)
        1 = solid, 0 = void.

    Notes
    -----
    The slice assignments assume that ``int(0.4 Nx) - int(0.1 Nx)`` equals
    ``int(N / 2)`` (and similarly for the lower half); for some ``Nx`` this
    may give a shape-mismatch error.
    """
    geom = np.zeros((Nx, Nx))
    N = int(0.6 * Nx)
    geom1 = Diagonal2D_FACE(N, k)
    geom[int(0.1 * Nx):int(0.4 * Nx), int(0.2 * Nx):int(0.8 * Nx)] = geom1[0:int(N / 2), 0:N]
    geom[int(0.6 * Nx):int(0.9 * Nx), int(0.2 * Nx):int(0.8 * Nx)] = geom1[int(N / 2):N, 0:N]
    for i in range(k):
        geom[int(0.1 * Nx):int(0.9 * Nx), int(0.2 * Nx) + i] = 1
        geom[int(0.1 * Nx):int(0.9 * Nx), int(0.8 * Nx) - i] = 1
        geom[int(0.1 * Nx):int(0.9 * Nx), int(0.2 * Nx) - i] = 1
        geom[int(0.1 * Nx):int(0.9 * Nx), int(0.8 * Nx) + i] = 1
        geom[0:int(0.4 * Nx), int(0.5 * Nx) + i] = 1
        geom[0:int(0.4 * Nx), int(0.5 * Nx) - i] = 1
        geom[int(0.6 * Nx):Nx, int(0.5 * Nx) + i] = 1
        geom[int(0.6 * Nx):Nx, int(0.5 * Nx) - i] = 1

    geom = geom[0:Nx, 0:Nx]
    return geom


def Structure2D_FACE(NX, k):
    """
    2D face pattern of :func:`Metamaterial_4`: a tiling of the
    :func:`Structure2D` unit.

    A work array of size ``Nx = int(1.5 * NX)`` is filled with a
    3 (along the first axis) x 2 (along the second axis) arrangement of the
    transposed unit of size ``Nx/3 = NX/2``. Along the first axis the copies
    are trimmed (``k+1`` rows at the start, ``k`` at the end) and overlap
    so that neighbouring cells share their connecting bars. The result is
    cropped to ``NX x NX``.

    Parameters
    ----------
    NX : int
        Size of the returned square pattern.
    k : int
        Line half-thickness in pixels, passed to :func:`Structure2D`.

    Returns
    -------
    geom : np.ndarray of float, shape (NX, NX)
        1 = solid, 0 = void.
    """
    Nx = int(3 * NX / 2)  # oversized work array, cropped to NX at the end

    geom = np.zeros((Nx, Nx))
    unit = np.transpose(Structure2D(int(Nx / 3), k))
    # Trim the unit along the first axis so that the stacked copies connect.
    unit = unit[k + 1:-k, :]
    geom[0 + 1:int(Nx / 3) - 2 * k, 0:int(Nx / 3)] = unit
    geom[int((Nx / 3) - 2 * k - 2 * k) + 1:int((2 * Nx / 3) - 4 * k - 2 * k), 0:int(Nx / 3)] = unit
    geom[int(2 * (Nx / 3 - 2 * k) - 4 * k) + 1:int(3 * (Nx / 3 - 2 * k) - 4 * k), 0:int(Nx / 3)] = unit
    geom[0 + 1:int(Nx / 3) - 2 * k, int(Nx / 3):int(2 * Nx / 3)] = unit
    geom[int((Nx / 3) - 2 * k - 2 * k) + 1:int((2 * Nx / 3) - 4 * k - 2 * k), int(Nx / 3):int(2 * Nx / 3)] = unit
    geom[int(2 * (Nx / 3 - 2 * k) - 4 * k) + 1:int(3 * (Nx / 3 - 2 * k) - 4 * k), int(Nx / 3):int(2 * Nx / 3)] = unit

    geom = geom[0:NX, 0:NX]
    return geom


def Metamaterial_4(*nb_voxels):
    """
    Prismatic metamaterial ("geometry_III_3_3D"): the 2D pattern
    :func:`Structure2D_FACE` (in the x-z plane) extruded along y.

    Parameters
    ----------
    *nb_voxels : int
        ``Nx, Ny, Nz``. ``Nz`` is ignored: the z-size equals ``Nx``.
        ``Ny`` (the extrusion length) may differ from ``Nx``.

    Returns
    -------
    D : np.ndarray of float, shape (Nx, Ny, Nx)
        1 = solid, 0 = void. Line thickness ``k = int(Nx / 20)``.
    """
    (Nx, Ny, Nz) = nb_voxels
    # Create cube
    Cube = np.zeros((Nx, Ny, Nx))
    k = int(Nx / 20)
    # Create meshgrid (unused, kept from the original implementation)
    I, J, K = np.meshgrid(np.arange(1, Nx + 1), np.arange(1, Nx + 1), np.arange(1, Nx + 1))

    # Get Diagonal2D_FACE matrix
    Face = Structure2D_FACE(Nx, k)
    # Assign values to D
    D = np.zeros_like(Cube)
    for i in range(Ny):
        D[:, i, :] = Face
    # Restrict D to Nx x Nx x Nx
    D = D[0:Nx, 0:Ny, 0:Nx]

    return D


def visualize_voxels(phase_field_xyz, figure=None, ax=None):
    """
    Plot a 3D phase field as coloured, semi-transparent voxels.

    Voxels with ``|value| / max|value| >= 0.1`` are drawn. Positive values
    are blue, negative values red, and the opacity of each voxel is
    ``|value| / max|value|``.

    Parameters
    ----------
    phase_field_xyz : np.ndarray, shape (Nx, Ny, Nz)
        Field to plot (e.g. the output of :func:`get_geometry`).
    figure : matplotlib.figure.Figure, optional
        Existing figure. If None, a new figure is created.
    ax : mpl_toolkits.mplot3d.axes3d.Axes3D, optional
        Existing 3D axes. If None, a 3D subplot is added to the figure.

    Returns
    -------
    fig : matplotlib.figure.Figure
    ax : mpl_toolkits.mplot3d.axes3d.Axes3D

    Notes
    -----
    The local variable ``fig`` is only defined when ``figure is None``;
    passing a ``figure`` currently leads to a ``NameError`` (at
    ``fig.add_subplot`` if ``ax`` is None, otherwise at the return).
    """
    # -----
    # phase_field_xyz  - indicator field in every voxel
    # plot voxelized geometry of the phase_field
    #
    # -----
    # phase_field_bool = phase_field_xyz.round(decimals=0).astype(int).astype(bool)
    # Which voxels to draw: those with at least 10 % of the maximal magnitude.
    phase_field_bool = np.empty(phase_field_xyz.shape, dtype=bool)
    phase_field_bool[np.abs(phase_field_xyz) / abs(phase_field_xyz).max() < 0.1] = False
    phase_field_bool[np.abs(phase_field_xyz) / abs(phase_field_xyz).max() >= 0.1] = True
    # test_bool = test.astype(int).astype(bool)

    # te=phase_field_xyz[abs(phase_field_xyz) >= 0.1] = 1
    negative_values = phase_field_xyz < 0
    positive_values = phase_field_xyz > 0

    # set the colors of each object (RGBA per voxel, last axis of size 4)
    face_colors = np.zeros(list(phase_field_xyz.shape) + [4], dtype=np.float32)
    alpha = 0.0
    # set possitive to blues
    face_colors[positive_values] = [0, 0, 1, alpha]
    face_colors[negative_values] = [1, 0, 0, alpha]
    # edge_colors = face_colors

    # set transparency --- opacity --- alpha     scale 0-1
    face_colors[..., -1] = abs(phase_field_xyz) / abs(phase_field_xyz).max()
    if figure is None:
        fig = plt.figure()
    if ax is None:
        ax = fig.add_subplot(projection='3d')
    ax.voxels(phase_field_bool, facecolors=face_colors, edgecolor='k', linewidth=0.01)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    # ax.voxels(np.full(phase_field_bool.shape, True), facecolors=face_colors, edgecolor='k', linewidth=0.01)
    return fig, ax


def check_equal_number_of_voxels(nb_voxels, microstructure_name):
    """
    Check that a 3D grid has the same number of voxels in each direction.

    Parameters
    ----------
    nb_voxels : array_like of int, shape (3,)
        Number of voxels ``(Nx, Ny, Nz)``.
    microstructure_name : str
        Name of the geometry (used in the error message).

    Raises
    ------
    ValueError
        If the grid is not cubic -- but see Notes.

    Notes
    -----
    The chained comparison ``a != b != c`` means ``a != b and b != c``, so
    the error is raised only if *both* pairs differ; e.g. ``(10, 10, 18)``
    passes the check.
    """
    if nb_voxels[0] != nb_voxels[1] != nb_voxels[2]:
        raise ValueError(
            'Microstructure_name {} is implemented only in Nx=Ny=Nz grids'.format(microstructure_name))


def check_unequal_number_of_voxels(nb_voxels, microstructure_name, ratios):
    """
    Check that a 3D grid has the prescribed aspect ratios.

    The intended condition is ``ratios[0]*Nx == ratios[1]*Ny == ratios[2]*Nz``.

    Parameters
    ----------
    nb_voxels : array_like of int, shape (3,)
        Number of voxels ``(Nx, Ny, Nz)``.
    microstructure_name : str
        Name of the geometry (used in the error message).
    ratios : sequence of 3 floats
        Ratio factors, e.g. ``[1.8, 1.8, 1]`` for ``Nz = 1.8 Nx = 1.8 Ny``.

    Raises
    ------
    ValueError
        Only if the x/y condition holds but the x/z condition does not (a
        grid violating the x/y condition is not rejected).
    """
    if (np.allclose(nb_voxels[0] * ratios[0], nb_voxels[1] * ratios[1])) and not (
            np.allclose(nb_voxels[0] * ratios[0], nb_voxels[2] * ratios[2])):
        raise ValueError(
            'Microstructure_name {} is implemented only in {} Nx = {} Ny = {} Nz grids'.format(microstructure_name,
                                                                                               ratios[0],
                                                                                               ratios[1], ratios[2]))


def check_number_of_voxels(nb_voxels, microstructure_name, min_nb_voxels):
    """
    Check that every direction has strictly more than ``min_nb_voxels`` voxels.

    Needed because strut thicknesses like ``int(0.05 * Nx)`` vanish on
    coarse grids.

    Parameters
    ----------
    nb_voxels : array_like of int, shape (3,)
        Number of voxels ``(Nx, Ny, Nz)``.
    microstructure_name : str
        Name of the geometry (used in the error message).
    min_nb_voxels : int
        Exclusive lower bound.

    Raises
    ------
    ValueError
        If any direction has ``<= min_nb_voxels`` voxels. (The message also
        mentions "multiple of 5", but that condition is commented out and
        not checked.)
    """
    if not (nb_voxels[0] > min_nb_voxels and nb_voxels[1] > min_nb_voxels and nb_voxels[2] > min_nb_voxels):
        # and nb_voxels[0] % 5 == 0 and nb_voxels[1] % 5 == 0 and nb_voxels[2] % 5 == 0
        raise ValueError('Microstructure_name {} is implemented only when Size '
                         'of any dimension is more than {} and it is a multiple of 5'.format(
            microstructure_name, min_nb_voxels))


def check_dimension(nb_voxels, microstructure_name):
    """
    Check that the grid is three-dimensional.

    Parameters
    ----------
    nb_voxels : np.ndarray of int
        Number of voxels per direction (must be a NumPy array, ``.size`` is used).
    microstructure_name : str
        Name of the geometry (used in the error message).

    Raises
    ------
    ValueError
        If ``nb_voxels.size != 3``.
    """
    if nb_voxels.size != 3:
        raise ValueError('Microstructure_name {} is implemented only in 3D'.format(microstructure_name))


### ----- Chiral metamaterials (contributed by Indre Joedicke) ----- ###

def chiral_metamaterial(nb_grid_pts, lengths, radius, thickness, alpha=0):
    """
    Define a (relatively simple) chiral metamaterial. It consists of two
    rings connected by four beams. Each beam is inclined by an angle alpha.

    The two rings (annuli of outer radius ``radius`` and width
    ``thickness``, centred on the vertical axis through the middle of the
    x-y cross-section) form the bottom and top layers (``thickness`` high)
    of the cell. Four beams of square cross-section ``thickness x thickness``
    start at the four "compass points" of the bottom ring and run upwards;
    for ``alpha != 0`` they are tilted tangentially (and pulled slightly
    inwards so that they stay on the ring), which makes the structure chiral.

    Parameters
    ----------
    nb_grid_pts : list of 3 ints
        Number of grid pts in each direction.
    lengths : list of 3 floats
        Lengths of unit cell in each direction.
    radius : float
        (Outer) radius of the circles.
    thickness : float
        Thickness of the circles and beams.
    alpha : float, optional
        Angle (in radians) at which the connecting beams are inclined.
        Default is 0, meaning the beams are vertical.

    Returns
    -------
    mask : np.ndarray of floats, shape nb_grid_pts
        Representation of the geometry with 0 corresponding
        to void and 1 corresponding to material.

    Notes
    -----
    Voxel ``(i, j, k)`` is represented by its centre
    ``((i + 0.5) hx, (j + 0.5) hy, (k + 0.5) hz)`` with ``h = lengths /
    nb_grid_pts``. The parameter checks only print warnings; the intended
    "Error" assertions are written such that they do not trigger in the
    erroneous case (see the comments in the code).
    """
    # Parameters
    # Voxel sizes
    hx = lengths[0] / nb_grid_pts[0]
    hy = lengths[1] / nb_grid_pts[1]
    hz = lengths[2] / nb_grid_pts[2]
    # Position of the vertical axis of the rings (at a voxel boundary near the centre)
    x_axis = nb_grid_pts[0] // 2 * hx  # + 0.5 * hx
    y_axis = nb_grid_pts[1] // 2 * hy  # + 0.5 * hy
    # Thickness expressed in number of voxels per direction
    thickness_x = round(thickness / hx)
    thickness_y = round(thickness / hy)
    thickness_z = round(thickness / hz)

    # Check wether the parameters are meaningful
    # NOTE: the asserts below assert the *valid* condition only inside the
    # branch where it is violated in the opposite sense, so they do not act as
    # intended guards (e.g. the first one passes exactly when radius is too
    # small); also (hy > thickness) is tested twice instead of hz.
    if (radius > lengths[0] / 2 - hx) or (radius > lengths[1] / 2 - hy):
        message = 'Attention: The diameter of the cylinder is larger '
        message += 'then the unit cell. THE PERIODIC BOUNDARIES ARE '
        message += 'NOT BROKEN.'
        print(message)
    if (radius < thickness + hx) or (radius < thickness + hy):
        message = 'Error: The radius is too small.'
        assert radius < thickness + hx, message
        assert radius < thickness + hy, message
    if (hx > thickness) or (hy > thickness) or (hy > thickness):
        message = 'Error: The pixels are larger than the thickness.'
        message += ' Please refine the discretization.'
        assert hx > thickness, message
        assert hy > thickness, message
        assert hz > thickness, message
    if (3 * hx > thickness) or (3 * hy > thickness):
        message = 'Attention: The thickness is represented by less then 3 pixels.'
        message += ' Please consider refining the discretization.'
        print(message)
    # Maximal tilt: the beam must not travel more than the ring's mid-radius
    # over the free height lengths[2] - 2*thickness between the rings.
    helper = np.arctan((radius - thickness / 2) / (lengths[2] - 2 * thickness))
    if (alpha > helper) or (alpha < - helper):
        message = f'Error: The angle must lie between {-helper} and {helper}'
        message += f'but it is {alpha}.'
        assert alpha > helper, message
        assert alpha < - helper, message

    # Circles at top and bottom
    # Annulus radius-thickness <= dist < radius in the x-y plane (evaluated at
    # voxel centres), extruded over the first and last thickness_z z-layers.
    mask = np.zeros(nb_grid_pts)
    for ind_x in range(nb_grid_pts[0]):
        for ind_y in range(nb_grid_pts[1]):
            x = ind_x * hx + 0.5 * hx
            y = ind_y * hy + 0.5 * hy
            dist = np.sqrt((x - x_axis) ** 2 + (y - y_axis) ** 2)
            if (dist < radius) and (dist >= radius - thickness):
                mask[ind_x, ind_y, 0:thickness_z] = 1
                mask[ind_x, ind_y, nb_grid_pts[2] - thickness_z:] = 1

    # Step in x- and y-direction of connecting beams
    # step_1: tangential offset per z-layer due to the tilt alpha.
    # step_2: radial (inward) offset per z-layer, chosen such that after the
    #         full free height the beam end lies again on the ring of
    #         mid-radius (radius - thickness/2) (chord geometry).
    step_1 = hz * np.tan(alpha)
    helper = (lengths[2] - 2 * thickness) * np.tan(alpha)
    helper = radius - thickness / 2 - \
             ((radius - thickness / 2) ** 2 - helper ** 2) ** 0.5
    step_2 = helper / (lengths[2] - 2 * thickness) * hz

    # Starting points for connecting beams (voxel index of the lower-left
    # corner of the beam cross-section on the bottom ring):
    # beam 1 at -y, beam 2 at +x, beam 3 at +y, beam 4 at -x of the ring.
    start1_x = nb_grid_pts[0] // 2 - thickness_x // 2
    start2_x = round((lengths[0] / 2 + radius - thickness) / hx)
    start3_x = nb_grid_pts[0] // 2 - thickness_x // 2
    start4_x = round((lengths[0] / 2 - radius) / hx)
    start1_y = round((lengths[1] / 2 - radius) / hy)
    start2_y = nb_grid_pts[1] // 2 - thickness_y // 2
    start3_y = round((lengths[1] / 2 + radius - thickness) / hy)
    start4_y = nb_grid_pts[1] // 2 - thickness_y // 2

    # Connecting beams
    # Loop over the z-layers between the rings; in each layer the beam
    # cross-section is shifted tangentially (help_*1) and radially (help_*2),
    # with signs rotating by 90 degrees from beam to beam.
    for ind_z in range(thickness_z, nb_grid_pts[2] - thickness_z):
        help_x1 = round((ind_z - thickness_z) * step_1 / hx)
        help_y1 = round((ind_z - thickness_z) * step_1 / hy)
        help_x2 = round((ind_z - thickness_z) * step_2 / hx)
        help_y2 = round((ind_z - thickness_z) * step_2 / hy)

        # 1. beam
        start_x = start1_x + help_x1
        start_y = start1_y + help_y2
        mask[start_x:start_x + thickness_x, start_y:start_y + thickness_y, ind_z] = 1
        # 2. beam
        start_x = start2_x - help_x2
        start_y = start2_y + help_y1
        mask[start_x:start_x + thickness_x, start_y:start_y + thickness_y, ind_z] = 1
        # 3. beam
        start_x = start3_x - help_x1
        start_y = start3_y - help_y2
        mask[start_x:start_x + thickness_x, start_y:start_y + thickness_y, ind_z] = 1
        # 4. beam
        start_x = start4_x + help_x2
        start_y = start4_y - help_y1
        mask[start_x:start_x + thickness_x, start_y:start_y + thickness_y, ind_z] = 1

    return mask


def chiral_metamaterial_2(nb_grid_pts, lengths, radius_out, radius_inn,
                          thickness, alpha=0):
    """
    Define a (more complex) chiral metamaterial. It consists of a beam on each
    face of the RVE connected to the edges by four beams. The beams are
    inclined with an angle alpha.

    More precisely, inside a cube of edge length ``a = lengths[2]`` (centred
    in x and y, surrounded by a void margin ``boundary = (lengths[0]-a)/2``)
    the structure consists of

    * eight corner blocks of size ``b = 1.5 * thickness``;
    * on each of the six cube faces a ring (annulus between ``radius_inn``
      and ``radius_out``) of thickness ``thickness`` (in z only
      ``thickness/2`` at the bottom and top, as they are split by
      periodicity);
    * on each face four inclined beams connecting the corner blocks to the
      ring, rotated by ``beta = pi/4 - alpha`` against the face diagonal --
      for ``alpha != 0`` this gives a chiral (handed) arrangement.

    Parameters
    ----------
    nb_grid_pts : list of 3 ints
        Number of grid pts in each direction.
    lengths : list of 3 floats
        Lengths of unit cell in each direction. Note that the size of
        the RVE corresponds to lengths[2], so that lengths[0] and
        lengths[1] must be larger than lengths[2] to break the periodicity.
    radius_out : float
        Outer radius of the circles.
    radius_inn : float
        Inner radius of the circles.
    thickness : float
        Thickness of the connecting beams.
    alpha : float, optional
        Angle (in radians) at which the connecting beams are inclined.
        Default is 0.

    Returns
    -------
    mask : np.ndarray of floats, shape nb_grid_pts
        Representation of the geometry with 0 corresponding
        to void and 1 corresponding to material.

    Raises
    ------
    AssertionError
        If ``lengths[0]`` or ``lengths[1]`` is not larger than ``lengths[2]``
        or if ``alpha`` is too large for the beams to reach the rings.

    Notes
    -----
    Negative-index slices (``-stop_y:-start_y`` etc.) are used to mirror the
    construction to the opposite faces of the cube.
    """
    ### ----- Parameters ----- ###
    hx = lengths[0] / nb_grid_pts[0]
    hy = lengths[1] / nb_grid_pts[1]
    hz = lengths[2] / nb_grid_pts[2]
    thickness_x = round(thickness / hx)
    thickness_y = round(thickness / hy)
    thickness_z = round(thickness / hz)
    a = lengths[2]  # edge length of the cube containing the structure
    b = 1.5 * thickness  # size of the corner blocks
    boundary = (lengths[0] - a) / 2  # void margin in x (and y) around the cube

    # Check wether the parameters are meaningful
    # NOTE: as in chiral_metamaterial, the "ERROR" asserts inside the if
    # branches are written such that they do not reject invalid input.
    if (radius_out > a / 2 - hx) or (radius_out > a / 2 - hy):
        message = 'ATTENTION: The diameter of the outer circle is larger '
        message += 'then the unit cell.'
        print(message)
    if (radius_inn < thickness + hx) or (radius_inn < thickness + hy):
        message = 'ERROR: The inner radius is too small.'
        assert radius_inn < thickness + hx, message
        assert radius_inn < thickness + hy, message
    if (hx > thickness) or (hy > thickness) or (hy > thickness):
        message = 'ERROR: The pixels are larger than the thickness.'
        message += ' Please refine the discretization.'
        assert hx > thickness, message
        assert hy > thickness, message
        assert hz > thickness, message
    if (3 * hx > thickness) or (3 * hy > thickness):
        message = 'ATTENTION: The thickness is represented by less then 3 pixels.'
        message += ' Please consider refining the discretization.'
        print(message)
    message = 'lengths[0] is not large enough to break the periodicity.'
    assert lengths[0] > a, message
    message = 'lengths[1] is not large enough to break the periodicity.'
    assert lengths[1] > a, message

    ### ----- Define the four corners ----- ###
    # (in fact eight blocks: four corners in the x-y plane, at the bottom
    # (0:bz) and top (-bz:) of the cell)
    mask = np.zeros(nb_grid_pts)
    bx = round(b / hx)
    by = round(b / hy)
    bz = round(b / hz)
    boundary_x = round(boundary / hx)
    boundary_y = round(boundary / hy)
    mask[boundary_x:boundary_x + bx, boundary_y:boundary_y + by, 0:bz] = 1
    mask[-bx - boundary_x:-boundary_x, boundary_y:boundary_y + by, 0:bz] = 1
    mask[boundary_x:boundary_x + bx, -by - boundary_y:-boundary_y, 0:bz] = 1
    mask[-bx - boundary_x:-boundary_x, -by - boundary_y:-boundary_y, 0:bz] = 1
    mask[boundary_x:boundary_x + bx, boundary_y:boundary_y + by, -bz:] = 1
    mask[-bx - boundary_x:-boundary_x, boundary_y:boundary_y + by, -bz:] = 1
    mask[boundary_x:boundary_x + bx, -by - boundary_y:-boundary_y, -bz:] = 1
    mask[-bx - boundary_x:-boundary_x, -by - boundary_y:-boundary_y, -bz:] = 1

    ### ----- Define the circle at each face ----- ###
    # Voxel-centre coordinates. np.meshgrid uses 'xy' indexing by default
    # (shape (ny, nx, nz)); the transpose (1, 0, 2) converts to the
    # [ix, iy, iz] layout of mask.
    x = np.arange(nb_grid_pts[0]) * hx
    y = np.arange(nb_grid_pts[1]) * hy
    z = np.arange(nb_grid_pts[2]) * hz
    X, Y, Z = np.meshgrid(x, y, z)
    X = X.transpose((1, 0, 2)) + 0.5 * hx
    Y = Y.transpose((1, 0, 2)) + 0.5 * hy
    Z = Z.transpose((1, 0, 2)) + 0.5 * hz

    # Circles in xz-planes
    x0 = lengths[0] / 2
    z0 = lengths[2] / 2
    dist = (X[:, 0, :] - x0) ** 2 + (Z[:, 0, :] - z0) ** 2
    dist = dist ** 0.5
    material = np.logical_and(dist < radius_out, dist > radius_inn)
    material = np.expand_dims(material, axis=1)
    material = np.broadcast_to(material, nb_grid_pts).copy()
    # Keep the annulus only in the two face layers y in [boundary_y,
    # boundary_y + thickness_y) and the mirrored one at the other side.
    material[:, 0:boundary_y, :] = False
    material[:, -boundary_y:, :] = False
    material[:, boundary_y + thickness_y:-thickness_y - boundary_y, :] = False
    mask[material] = 1

    # Circles in yz-planes
    y0 = lengths[1] / 2
    z0 = lengths[2] / 2
    dist = (Y[0, :, :] - y0) ** 2 + (Z[0, :, :] - z0) ** 2
    dist = dist ** 0.5
    material = np.logical_and(dist < radius_out, dist > radius_inn)
    material = np.expand_dims(material, axis=0)
    material = np.broadcast_to(material, nb_grid_pts).copy()
    material[0:boundary_x, :, :] = False
    material[-boundary_x:, :, :] = False
    material[boundary_x + thickness_x:-thickness_x - boundary_x, :, :] = False
    mask[material] = 1

    # Circles in xy-planes
    x0 = lengths[0] / 2
    y0 = lengths[1] / 2
    dist = (X[:, :, 0] - x0) ** 2 + (Y[:, :, 0] - y0) ** 2
    dist = dist ** 0.5
    material = np.logical_and(dist < radius_out, dist > radius_inn)
    material = np.expand_dims(material, axis=2)
    material = np.broadcast_to(material, nb_grid_pts).copy()
    # Keep only the bottom and top layers (about thickness/2 each); together
    # they form one ring of full thickness across the periodic z-boundary.
    material[:, :, thickness_z // 2:-thickness_z // 2] = False
    mask[material] = 1

    ### ----- Define the connecting beams ----- ###
    # beta: angle of the beams w.r.t. the face edges (pi/4 = along the face
    # diagonal for alpha = 0). step_exact: rise per voxel along the beam.
    beta = np.pi / 4 - alpha
    step_exact = hz * np.tan(beta)

    # Find the beam length 'stop' (projected onto the edge direction) at
    # which a beam starting at the corner block centre, (b/2, b/2) in face
    # coordinates, with slope tan(beta) hits the mid-radius circle
    # (radius_out + radius_inn)/2 around the face centre (a/2, a/2):
    # smaller root of the quadratic helper_a*s^2 + helper_b*s + helper_c = 0.

    helper_a = 1 + np.tan(beta) ** 2
    helper_b = - 2 * (np.tan(beta) + 1) * (a / 2 - b / 2)
    helper_c = 2 * (a / 2 - b / 2) ** 2 - (radius_out / 2 + radius_inn / 2) ** 2
    helper = helper_b ** 2 - 4 * helper_a * helper_c
    message = 'ERROR: The angle of the material is too large.'
    assert helper > 0, message
    stop = (- helper_b - helper ** 0.5) / 2 / helper_a

    # Beams in xz-planes
    # (on the two faces normal to y). Each face gets 4 beams: two marched
    # along x (first loop) and two marched along z (second loop); the
    # negative indices place the point-mirrored copies. The 'helper' branch
    # avoids an empty slice when the upper slice bound would be 0
    # (i.e. ``-n:0``); then the slice is open-ended instead.
    start_x = round((boundary + b / 2) / hx)
    stop_x = round((boundary + stop + b / 2) / hx)
    start_y = boundary_y
    stop_y = boundary_y + thickness_y
    start_z = round(b / 2 / hz)
    stop_z = round((b / 2 + stop) / hz)
    t_half_x = round(thickness / 2 / hx)
    t_half_y = round(thickness / 2 / hy)
    t_half_z = round(thickness / 2 / hz)
    for ind_x in range(start_x, stop_x):
        step = round((ind_x - start_x) * step_exact / hz)
        mask[ind_x - t_half_x + 1: ind_x + t_half_x + 1,
        start_y: stop_y,
        start_z + step - t_half_x: start_z + step + t_half_x] = 1
        mask[ind_x - t_half_x + 1: ind_x + t_half_x + 1,
        -stop_y: -start_y,
        start_z + step - t_half_z: start_z + step + t_half_z] = 1
        helper = -start_z - step + t_half_z
        if helper > -1:
            mask[-ind_x - t_half_x - 1: -ind_x + t_half_x - 1,
            start_y: stop_y,
            -start_z - step - t_half_z:] = 1
            mask[-ind_x - t_half_x - 1: -ind_x + t_half_x - 1,
            -stop_y: -start_y,
            -start_z - step - t_half_z:] = 1
        else:
            mask[-ind_x - t_half_x - 1: -ind_x + t_half_x - 1,
            start_y: stop_y,
            -start_z - step - t_half_z: helper] = 1
            mask[-ind_x - t_half_x - 1: -ind_x + t_half_x - 1,
            -stop_y: -start_y,
            -start_z - step - t_half_z: helper] = 1
    for ind_z in range(start_z, stop_z):
        step = round((ind_z - start_z) * step_exact / hx)
        mask[-start_x - step - t_half_x: -start_x - step + t_half_x,
        start_y: stop_y,
        ind_z - t_half_z: ind_z + t_half_z] = 1
        mask[start_x + step - t_half_x: start_x + step + t_half_x,
        start_y: stop_y,
        -ind_z - t_half_z: -ind_z + t_half_z] = 1
        mask[-start_x - step - t_half_x: -start_x - step + t_half_x,
        -stop_y: -start_y,
        ind_z - t_half_z: ind_z + t_half_z] = 1
        mask[start_x + step - t_half_x: start_x + step + t_half_x,
        -stop_y: -start_y,
        -ind_z - t_half_z: -ind_z + t_half_z] = 1

    # Beams in yz-planes (faces normal to x), same construction with x <-> y.
    start_x = boundary_x
    stop_x = boundary_x + thickness_x
    start_y = round((boundary + b / 2) / hy)
    stop_y = round((boundary + stop + b / 2) / hy)
    for ind_y in range(start_y, stop_y):
        step = round((ind_y - start_y) * step_exact / hz)
        mask[start_x: stop_x,
        ind_y - t_half_y + 1: ind_y + t_half_y + 1,
        start_z + step - t_half_z: start_z + step + t_half_z] = 1
        mask[-stop_x: -start_x,
        ind_y - t_half_y + 1: ind_y + t_half_y + 1,
        start_z + step - t_half_z: start_z + step + t_half_z] = 1
        helper = -start_z - step + t_half_z
        if helper > -1:
            mask[start_x: stop_x,
            -ind_y - t_half_y - 1: -ind_y + t_half_y - 1,
            -start_z - step - t_half_z:] = 1
            mask[-stop_x: -start_x,
            -ind_y - t_half_y - 1: -ind_y + t_half_y - 1,
            -start_z - step - t_half_z:] = 1
        else:
            mask[start_x: stop_x,
            -ind_y - t_half_y - 1: -ind_y + t_half_y - 1,
            -start_z - step - t_half_z: helper] = 1
            mask[-stop_x: -start_x,
            -ind_y - t_half_y - 1: -ind_y + t_half_y - 1,
            -start_z - step - t_half_z: helper] = 1
    for ind_z in range(start_z, stop_z):
        step = round((ind_z - start_z) * step_exact / hy)
        mask[start_x: stop_x,
        -start_y - step - t_half_y: -start_y - step + t_half_y,
        ind_z - t_half_z: ind_z + t_half_z] = 1
        mask[start_x: stop_x,
        start_y + step - t_half_y: start_y + step + t_half_y,
        -ind_z - t_half_z: -ind_z + t_half_z] = 1
        mask[-stop_x: -start_x,
        -start_y - step - t_half_y: -start_y - step + t_half_y,
        ind_z - t_half_z: ind_z + t_half_z] = 1
        mask[-stop_x: -start_x,
        start_y + step - t_half_y: start_y + step + t_half_y,
        -ind_z - t_half_z: -ind_z + t_half_z] = 1

    # Beams in xy-planes (faces normal to z): they are split across the
    # periodic z-boundary, so each beam is drawn in the bottom (:stop_z) and
    # top (-stop_z:) layers of about thickness/2.
    start_x = round((boundary + b / 2) / hx)
    stop_x = round((boundary + stop + b / 2) / hx)
    start_y = round((boundary + b / 2) / hy)
    stop_y = round((boundary + stop + b / 2) / hy)
    stop_z = round(thickness / 2 / hz)
    for ind_x in range(start_x, stop_x):
        step = round((ind_x - start_x) * step_exact / hy)
        mask[ind_x - t_half_x + 1: ind_x + t_half_x + 1,
        start_y + step - t_half_y: start_y + step + t_half_y,
        : stop_z] = 1
        mask[ind_x - t_half_x + 1: ind_x + t_half_x + 1,
        start_y + step - t_half_y: start_y + step + t_half_y,
        -stop_z:] = 1
        helper = -start_y - step + t_half_y
        if helper > -1:
            mask[-ind_x - t_half_x - 1: -ind_x + t_half_x - 1,
            -start_y - step - t_half_y:,
            : stop_z] = 1
            mask[-ind_x - t_half_x - 1: -ind_x + t_half_x - 1,
            -start_y - step - t_half_y:,
            -stop_z:] = 1
        else:
            mask[-ind_x - t_half_x - 1: -ind_x + t_half_x - 1,
            -start_y - step - t_half_y: helper,
            : stop_z] = 1
            mask[-ind_x - t_half_x - 1: -ind_x + t_half_x - 1,
            -start_y - step - t_half_y: helper,
            -stop_z:] = 1
    for ind_y in range(start_y, stop_y):
        step = round((ind_y - start_y) * step_exact / hx)
        mask[-start_x - step - t_half_x: -start_x - step + t_half_x,
        ind_y - t_half_y: ind_y + t_half_y,
        : stop_z] = 1
        mask[start_x + step - t_half_x: start_x + step + t_half_x,
        -ind_y - t_half_y: -ind_y + t_half_y,
        : stop_z] = 1
        mask[-start_x - step - t_half_x: -start_x - step + t_half_x,
        ind_y - t_half_y: ind_y + t_half_y,
        -stop_z:] = 1
        mask[start_x + step - t_half_x: start_x + step + t_half_x,
        -ind_y - t_half_y: -ind_y + t_half_y,
        -stop_z:] = 1

    return mask


# Small demo: build one geometry, plot it with visualize_voxels and save the
# figure (the output path is machine specific).
if __name__ == '__main__':
    import numpy as np
    import matplotlib.pyplot as plt


    # plot  geometry
    geometry_ID = 'geometry_III_5_3D'
    N = 60
    nb_of_pixels = np.asarray(3 * (N,), dtype=int)
    # nb_of_pixels = np.asarray( (N,N,1.8*N), dtype=int)

    phase_field = get_geometry(nb_voxels=nb_of_pixels,
                               microstructure_name=geometry_ID)

    fig, ax = visualize_voxels(phase_field_xyz=phase_field)
    ax.set_title(geometry_ID)
    save_plot = True
    if save_plot:
        src = '/home/martin/Programming/muFFTTO/experiments/figures/'  # source folder\
        fig_data_name = f'muFFTTO_{geometry_ID}_N{N}'

        fname = src + fig_data_name + '_geometry{}'.format('.pdf')
        print(('create figure: {}'.format(fname)))  # axes[1, 0].legend(loc='upper right')
        plt.savefig(fname, dpi=1000, pad_inches=0.02, bbox_inches='tight',
                    facecolor='auto', edgecolor='auto')
        print('END plot ')

    plt.show()
