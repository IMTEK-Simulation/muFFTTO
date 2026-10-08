"""
Periodic microstructure geometries.

A geometry is built from *shapes*: functions that map pixel coordinates to a
boolean mask (``True`` = inside). Shapes are combined with the set operators
``|`` (union), ``&`` (intersection), ``-`` (difference) and ``~`` (complement)
and turned into a phase field with :func:`indicator` or :func:`pixel_average`.

Every function here works point-wise on the coordinates it is given. With
``coordinates = discretization.fft.coords`` (the coordinates of the pixels
owned by the current MPI rank) each rank therefore builds exactly its own part
of the field, and the result does not depend on the number of ranks. No
function needs the global grid.

Conventions
-----------
* ``coords`` has shape ``(dim, n_x, n_y[, n_z])`` and holds *fractional*
  coordinates in the unit cell ``[0, 1)^dim`` -- the format of
  ``muGrid.FFTEngine.coords``. Points are pixel corners (lower-left), as in
  muGrid.
* Distances are periodic (minimum image), so shapes wrap around the cell.
* Phase fields use ``1`` for the matrix / first phase and ``0`` for
  inclusions, matching the material assignment in the examples.

Named geometries
----------------
The geometries used by the examples are registered by name; :func:`get`
builds one and :func:`available` lists them::

    from muFFTTO import geometry
    phase = geometry.get('circle_inclusion', discretization.fft.coords)

New geometries are added with the :func:`register` decorator::

    @geometry.register('two_disks')
    def two_disks(coords, radius=0.1):
        disks = geometry.ball((0.25, 0.5), radius) | geometry.ball((0.75, 0.5), radius)
        return geometry.indicator(~disks, coords)
"""
import numpy as np

__all__ = ['Shape', 'box', 'ball', 'everywhere', 'periodic_distance_sq', 'indicator', 'pixel_average',
           'random_field', 'register', 'get', 'available', 'reentrant_honeycomb_cell_size']


# ---------------------------------------------------------------------------
# Shapes
# ---------------------------------------------------------------------------

class Shape:
    """A region of the periodic unit cell.

    Wraps a function ``contains(coords) -> bool array`` of shape
    ``coords.shape[1:]``. Shapes support ``a | b`` (union), ``a & b``
    (intersection), ``a - b`` (difference) and ``~a`` (complement).
    """

    def __init__(self, contains):
        self._contains = contains

    def __call__(self, coords):
        """Boolean mask of the points of ``coords`` that lie in the shape."""
        return self._contains(np.asarray(coords))

    def __or__(self, other):
        return Shape(lambda c: self(c) | other(c))

    def __and__(self, other):
        return Shape(lambda c: self(c) & other(c))

    def __sub__(self, other):
        return Shape(lambda c: self(c) & ~other(c))

    def __invert__(self):
        return Shape(lambda c: ~self(c))


def everywhere():
    """The whole unit cell."""
    return Shape(lambda c: np.ones(c.shape[1:], dtype=bool))


def box(lower, upper):
    """Axis-aligned periodic box ``[lower_k, upper_k)`` in every direction ``k``.

    Parameters
    ----------
    lower, upper : sequence of float, length ``dim``
        Corners in fractional coordinates. If ``lower_k > upper_k`` the box
        wraps around the periodic boundary in direction ``k``, i.e. it covers
        ``[lower_k, 1) U [0, upper_k)``.
    """
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)

    def contains(c):
        mask = np.ones(c.shape[1:], dtype=bool)
        for k in range(len(lower)):
            if lower[k] <= upper[k]:
                mask &= (c[k] >= lower[k]) & (c[k] < upper[k])
            else:  # wraps around x_k = 1 -> 0
                mask &= (c[k] >= lower[k]) | (c[k] < upper[k])
        return mask

    return Shape(contains)


def ball(center, radius, axes=None):
    """Periodic ball (disk in 2D): points with distance ``< radius`` to ``center``.

    Parameters
    ----------
    center : sequence of float, length ``dim``
        Centre in fractional coordinates.
    radius : float
        Radius in fractional coordinates (``< 0.5`` so the ball does not
        overlap its periodic images).
    axes : sequence of int, optional
        Directions included in the distance. Default: all. Leaving a direction
        out gives a cylinder (2D: a band) along it.
    """
    return Shape(lambda c: np.sqrt(periodic_distance_sq(c, center, axes)) < radius)


def periodic_distance_sq(coords, center, axes=None):
    """Squared periodic (minimum-image) distance of every point to ``center``.

    Parameters
    ----------
    coords : ndarray, shape ``(dim, n_x, n_y[, n_z])``
    center : sequence of float, length ``dim``
    axes : sequence of int, optional
        Directions included in the distance. Default: all.

    Returns
    -------
    ndarray, shape ``coords.shape[1:]``
    """
    coords = np.asarray(coords)
    center = np.asarray(center, dtype=float)
    directions = range(coords.shape[0]) if axes is None else axes
    distance_sq = np.zeros(coords.shape[1:])
    for k in directions:
        d = np.abs(coords[k] - center[k])
        d = np.minimum(d, 1.0 - d)  # minimum image: the nearer of the two periodic copies
        distance_sq += d * d
    return distance_sq


# ---------------------------------------------------------------------------
# Shapes -> phase fields
# ---------------------------------------------------------------------------

def indicator(shape, coords, inside=1.0, outside=0.0):
    """Phase field with value ``inside`` in ``shape`` and ``outside`` elsewhere.

    Returns
    -------
    ndarray, shape ``coords.shape[1:]``
    """
    return np.where(shape(coords), inside, outside).astype(float)


def pixel_average(shape, coords, pixel_size, samples=4, inside=1.0, outside=0.0):
    """Pixel-averaged (anti-aliased) phase field.

    Each pixel gets the fraction of its area that lies in ``shape``, estimated
    with ``samples**dim`` points per pixel. Compared to :func:`indicator`, the
    effective properties then change smoothly when the geometry or the grid
    resolution changes, instead of jumping whenever a pixel flips.

    Parameters
    ----------
    pixel_size : sequence of float, length ``dim``
        Pixel size in fractional coordinates, i.e. ``1 / nb_pixels_global``.
    samples : int, optional
        Sample points per pixel and direction. Default 4.
    """
    coords = np.asarray(coords, dtype=float)
    dim = coords.shape[0]
    pixel_size = np.asarray(pixel_size, dtype=float).reshape((dim,) + (1,) * dim)
    offsets_1d = (np.arange(samples) + 0.5) / samples  # sample points inside [0, 1)
    fraction = np.zeros(coords.shape[1:])
    for offset in np.ndindex(*(samples,) * dim):
        shift = offsets_1d[list(offset)].reshape((dim,) + (1,) * dim)
        fraction += shape(np.mod(coords + shift * pixel_size, 1.0))
    fraction /= samples ** dim
    return outside + (inside - outside) * fraction


def random_field(coords, seed=0):
    """Uniform random values in ``[0, 1)``, one per point, reproducible under MPI.

    The value of a point is a hash of ``seed`` and its coordinates, so it is
    the same whichever rank owns the point and however many ranks there are
    (unlike ``np.random``, whose stream would depend on the decomposition).
    """
    coords = np.asarray(coords, dtype=float)
    # quantise the coordinates to integers (2**30 steps per unit cell)
    keys = np.round(coords * 2 ** 30).astype(np.uint64)
    with np.errstate(over='ignore'):  # uint64 wrap-around is intended
        h = np.full(coords.shape[1:], np.uint64(seed) * np.uint64(0x9E3779B97F4A7C15))
        for k in range(coords.shape[0]):
            h = _splitmix64(h ^ (keys[k] + np.uint64(k + 1) * np.uint64(0xBF58476D1CE4E5B9)))
    return (h >> np.uint64(11)).astype(float) / float(2 ** 53)  # top 53 bits -> [0, 1)


def _splitmix64(x):
    """SplitMix64 finaliser: a fast, well-mixing 64-bit integer hash."""
    x = (x ^ (x >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    x = (x ^ (x >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    return x ^ (x >> np.uint64(31))


# ---------------------------------------------------------------------------
# Registry of named geometries
# ---------------------------------------------------------------------------

_REGISTRY = {}


def register(name):
    """Decorator that registers ``function(coords, **parameters)`` under ``name``."""

    def decorator(function):
        if name in _REGISTRY:
            raise ValueError(f'Geometry {name!r} is already registered')
        _REGISTRY[name] = function
        return function

    return decorator


def available():
    """Sorted list of the registered geometry names."""
    return sorted(_REGISTRY)


def get(name, coords, **parameters):
    """Build the registered geometry ``name`` on ``coords``.

    Parameters
    ----------
    name : str
        One of :func:`available`.
    coords : ndarray, shape ``(dim, n_x, n_y[, n_z])``
        Fractional pixel coordinates, e.g. ``discretization.fft.coords``.
    **parameters
        Geometry parameters; see the individual geometry functions.

    Returns
    -------
    ndarray, shape ``coords.shape[1:]``
    """
    if name not in _REGISTRY:
        raise ValueError(f'Unknown geometry {name!r}. Available: {", ".join(available())}')
    return _REGISTRY[name](np.asarray(coords), **parameters)


# ---------------------------------------------------------------------------
# Geometries used by the examples
# ---------------------------------------------------------------------------

@register('square_inclusion')
def square_inclusion(coords, lower=0.25, upper=0.75):
    """Matrix (1) with a centred square/cube inclusion (0) ``[lower, upper)^dim``.

    The default inclusion fills a quarter of the cell in 2D and an eighth in 3D.
    """
    dim = coords.shape[0]
    return indicator(~box([lower] * dim, [upper] * dim), coords)


@register('circle_inclusion')
def circle_inclusion(coords, radius=None):
    """Matrix (1) with a centred disk/ball inclusion (0).

    A 2D grid with a single pixel in ``y`` is treated as 1D (only ``x`` enters
    the distance), giving a layer. The default radius is 0.2 in 1D/2D and
    ``sqrt(0.1) ~ 0.316`` in 3D (inclusion volume fraction ~0.13 in both).
    """
    dim = coords.shape[0]
    center = [0.5] * dim
    if dim == 3 and radius is None:
        # historical definition: squared distance < 0.1 (kept bit-identical)
        inclusion = Shape(lambda c: periodic_distance_sq(c, center) < 0.1)
    else:
        quasi_1d = dim == 2 and coords.shape[2] == 1
        inclusion = ball(center, 0.2 if radius is None else radius,
                         axes=[0] if quasi_1d else None)
    return indicator(~inclusion, coords)


@register('random_distribution')
def random_distribution(coords, seed=None):
    """Independent uniform random value in ``[0, 1)`` in every pixel.

    With ``seed=None`` a fresh seed is drawn, so every run differs (as before).
    Pass a seed for a reproducible field; it is the same for any number of
    MPI ranks.
    """
    if seed is None:
        seed = int(np.random.default_rng().integers(2 ** 63))
    return random_field(coords, seed=seed)


def _square_frame(inner_lower=0.15, inner_upper=0.85):
    """Solid frame: the cell minus the inner square ``[inner_lower, inner_upper)^2``."""
    return everywhere() - box([inner_lower] * 2, [inner_upper] * 2)


@register('contact_test_geometry_1')
def contact_test_geometry_1(coords):
    """2D square frame (1) around a void (0), with two solid sticks reaching inwards.

    The left stick ``[0.15, 0.45) x [0.5, 0.6)`` and the right stick
    ``[0.55, 0.95) x [0.45, 0.55)`` are vertically offset and nearly touch:
    a test cell for third-medium contact.
    """
    solid = (_square_frame()
             | box([0.15, 0.5], [0.45, 0.6])
             | box([0.55, 0.45], [0.95, 0.55]))
    return indicator(solid, coords)


@register('contact_test_geometry_2')
def contact_test_geometry_2(coords):
    """2D square frame (1) around a void (0), with two overlapping solid sticks.

    The lower-left stick ``[0.15, 0.52) x [0.35, 0.45)`` and the upper-right
    stick ``[0.47, 0.95) x [0.55, 0.65)`` overlap in ``x``, so they come into
    contact when the cell is sheared.
    """
    solid = (_square_frame()
             | box([0.15, 0.35], [0.52, 0.45])
             | box([0.47, 0.55], [0.95, 0.65]))
    return indicator(solid, coords)


@register('contact_fracture_s_gap')
def contact_fracture_s_gap(coords, corner=(0.42, 0.57), gap=0.05, overlap=0.12,
                           upper_radii=(0.26, 0.28), lower_radius_x=0.31, lower_bottom=0.18,
                           notch_length=0.0, notch_angle=-90.0, notch_width=0.035):
    """2D solid (1) with an S-shaped cavity (0) between two interlocking teeth.

    The cavity is the union of

    * a channel ``[cx, cx + overlap) x [cy, cy + gap)`` between the top face of
      the right tooth (corner ``(cx, cy)``) and the bottom face of the
      upper-left tooth (corner ``(cx + overlap, cy + gap)``);
    * an upper lobe: the quarter ellipse with centre ``(cx + overlap, cy)`` and
      radii ``upper_radii`` that opens up and to the right;
    * a lower lobe: the quarter ellipse with centre ``(cx, lower_bottom)``,
      radii ``(lower_radius_x, cy + gap - lower_bottom)``, that opens up and
      to the left.

    Under shear the channel closes and the teeth come into contact; the
    defaults were fitted to a sketch of the contact-fracture example. The
    cavity must lie inside the cell (no wrapping).

    Optionally a straight notch (slit) of ``notch_length`` and ``notch_width``
    starts at the bottom-right corner of the lower lobe, ``(cx, lower_bottom)``
    -- the re-entrant corner where the crack initiates under shear -- in the
    direction ``notch_angle`` (degrees from the x axis; -90 = straight down
    into the bottom ligament). ``notch_length = 0`` (default): no notch.
    """
    cx, cy = corner
    c = np.asarray(coords)
    x, y = c[0], c[1]

    def quarter_ellipse(center, radii, sx, sy):
        u = (x - center[0]) / radii[0]
        v = (y - center[1]) / radii[1]
        return (sx * u >= 0) & (sy * v >= 0) & (u ** 2 + v ** 2 <= 1)

    channel = (x >= cx) & (x < cx + overlap) & (y >= cy) & (y < cy + gap)
    upper = quarter_ellipse((cx + overlap, cy), upper_radii, +1, +1)
    lower = quarter_ellipse((cx, lower_bottom), (lower_radius_x, cy + gap - lower_bottom), -1, +1)
    cavity = channel | upper | lower
    if notch_length > 0:
        # points within notch_width / 2 of the segment from the corner, along notch_angle
        direction = np.array([np.cos(np.radians(notch_angle)), np.sin(np.radians(notch_angle))])
        rx, ry = x - cx, y - lower_bottom
        along = rx * direction[0] + ry * direction[1]
        across = -rx * direction[1] + ry * direction[0]
        cavity |= (along >= 0) & (along <= notch_length) & (np.abs(across) <= notch_width / 2)
    return np.where(cavity, 0.0, 1.0)


def reentrant_honeycomb_cell_size(theta=30.0, h_over_l=None):
    """Size ``(Lx, Ly)`` of the periodic cell of :func:`reentrant_honeycomb`, for ``l = 1``.

    ``Lx = 2 cos(theta)``, ``Ly = 2 (h/l - sin(theta))``. Use a ``domain_size``
    proportional to it for an undistorted lattice. ``h_over_l = None`` gives
    the square cell, ``h/l = cos(theta) + sin(theta)``.
    """
    t = np.radians(theta)
    if h_over_l is None:
        h_over_l = np.cos(t) + np.sin(t)
    return 2.0 * np.cos(t), 2.0 * (h_over_l - np.sin(t))


def _distance_to_segment(px, py, a, b):
    """Euclidean distance of the points (px, py) to the segment a-b."""
    d = np.subtract(b, a)
    s = np.clip(((px - a[0]) * d[0] + (py - a[1]) * d[1]) / (d[0] ** 2 + d[1] ** 2), 0.0, 1.0)
    return np.hypot(px - a[0] - s * d[0], py - a[1] - s * d[1])


@register('reentrant_honeycomb')
def reentrant_honeycomb(coords, theta=30.0, h_over_l=None, thickness=0.1):
    """2D re-entrant (bow-tie) auxetic honeycomb: solid walls (1), bow-tie voids (0).

    Walls of the classic re-entrant lattice (Gibson & Ashby; Masters & Evans):
    vertical walls of length ``h`` and inclined walls of length ``l = 1``,
    tilted inwards by the re-entrant angle ``theta`` (degrees), with wall
    thickness ``thickness`` (in units of ``l``). A point is solid if it is
    closer than ``thickness / 2`` to a wall centreline (periodic images
    included).

    The periodic cell is ``Lx x Ly`` = :func:`reentrant_honeycomb_cell_size`
    and holds one whole bow-tie void plus two halves. ``coords`` (fractional,
    ``[0, 1)``) are mapped onto it, so ``domain_size`` should be proportional
    to ``(Lx, Ly)``; the default ``h_over_l = None`` gives a square cell. A
    whole void sits in the cell centre. Requires ``h > l sin(theta)``.
    """
    t = np.radians(theta)
    if h_over_l is None:
        h_over_l = np.cos(t) + np.sin(t)
    if not h_over_l > np.sin(t):
        raise ValueError('reentrant_honeycomb needs h_over_l > sin(theta)')
    Lx, Ly = reentrant_honeycomb_cell_size(theta, h_over_l)
    h, cx, sy = h_over_l, np.cos(t), np.sin(t)
    # wall centrelines (nodes A, B, C, D of one cell; see the docs)
    A, B = (0.0, 0.0), (0.0, h)
    C, D = (cx, h - sy), (cx, 2 * h - sy)
    walls = [(A, B), (C, D), (B, C), (B, (C[0] - Lx, C[1])), (D, (0.0, Ly)), (D, (Lx, Ly))]
    c = np.asarray(coords)
    # map the cell centre (0.5, 0.5) onto the centre (0, (h + Ly) / 2) of a whole bow-tie void
    px = (c[0] + 0.5) % 1.0 * Lx
    py = (c[1] + 0.5 * h / Ly) % 1.0 * Ly
    distance = np.full(c.shape[1:], np.inf)
    for shift_x in (-Lx, 0.0, Lx):
        for shift_y in (-Ly, 0.0, Ly):
            for a, b in walls:
                distance = np.minimum(distance, _distance_to_segment(px - shift_x, py - shift_y, a, b))
    return np.where(distance < thickness / 2.0, 1.0, 0.0)


def _distance_to_polyline(px, py, points):
    """Distance of the points (px, py) to the polyline through ``points`` (shape (m, 2))."""
    distance = np.full(np.shape(px), np.inf)
    for a, b in zip(points[:-1], points[1:]):
        distance = np.minimum(distance, _distance_to_segment(px, py, a, b))
    return distance


@register('sinusoidal_ligaments')
def sinusoidal_ligaments(coords, amplitude=0.11, thickness=0.14, gap=0.14, vertical_amplitude=0.11,
                         vertical_gap=0.2, nb_samples=64):
    """2D auxetic lattice of curved (sinusoidal) ligaments of constant thickness: solid (1), voids (0).

    Lengths in units of the wavelength = cell size (square cell, ``[0, 1)^2``).

    * Two horizontal ligaments, one wavelength long, in opposite phase:
      centrelines ``y = 1/2 +- ((gap + thickness)/2 + amplitude (1 - cos 2 pi x))``.
      They are closest at the cell edges ``x = 0`` (gap ``gap`` between their
      surfaces) and ``4 amplitude`` further apart at ``x = 1/2``.
    * Two vertical ligaments, bowed towards each other:
      ``x = 1/2 -+ ((vertical_gap + thickness)/2 + vertical_amplitude (1 - cos 2 pi y'))``
      with ``y' = y - 1/2``; closest (gap ``vertical_gap``) at mid-height.

    ``thickness`` is measured normal to the centrelines. The ligaments cross
    each other, enclosing a central void; the narrow gaps between the
    ligaments of neighbouring cells close under compression (self-contact).
    """
    c = np.asarray(coords)
    x, y = c[0] % 1.0, c[1] % 1.0
    s = np.linspace(0.0, 1.0, nb_samples + 1)
    bump = 1.0 - np.cos(2 * np.pi * s)                             # 0 at the ends, 2 in the middle
    curves = []
    for sign in (+1.0, -1.0):
        curves.append(np.c_[s, 0.5 + sign * ((gap + thickness) / 2 + amplitude * bump)])
        offset_x = (vertical_gap + thickness) / 2 + vertical_amplitude * (1.0 - np.cos(2 * np.pi * (s - 0.5)))
        curves.append(np.c_[0.5 - sign * offset_x, s])
    distance = np.full(c.shape[1:], np.inf)
    for shift_x in (-1.0, 0.0, 1.0):
        for shift_y in (-1.0, 0.0, 1.0):
            for curve in curves:
                distance = np.minimum(distance, _distance_to_polyline(x - shift_x, y - shift_y, curve))
    return np.where(distance < thickness / 2.0, 1.0, 0.0)
