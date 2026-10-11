"""
grid_adaptation.py
==================
Conformal (interface-fitted) periodic grids for FFT-based solvers.

The grid keeps its topology (nx x ny pixels = bilinear elements, periodic); only
the nodes move, ``x = phi(X) = X + d_g(X)``, so that element edges follow the
interface between two phases. A solver keeps its unknowns, FFT and
preconditioner on the regular grid ``X`` and pulls its operators back with
``F_g = I + grad_X d_g`` and ``J_g = det F_g`` at the quadrature points; this
needs ``J_g > 0`` at the (2 x 2 Gauss) quadrature points only, so elements may
degenerate towards triangles (three nodes on one line).

Usage::

    from muFFTTO import grid_adaptation
    d_g, phase, info = grid_adaptation.adapt_grid((64, 64), 'sinusoidal_ligaments', method='zecevic')

``geometry`` is a name from :func:`muFFTTO.geometry.available` (with
``geometry_parameters``) or a callable ``phase(points)`` returning 0 / 1 at
points ``[2, ...]`` of the unit cell ``[0, 1)^2`` (periodic).

Methods
-------
``'zecevic'``
    Spring-based conformal grid of Zecevic, Lebensohn, Capolungo, Mech. Mater.
    212 (2026) 105512. Every pixel keeps the phase of its centre on the regular
    grid; the nodes on the resulting staircase boundary are moved onto the true
    interface:

    1. boundary chains: grid edges between pixels of different phase; nodes
       with more than two such edges and nodes at sharp corners of the
       interface (detected on its contour, here an addition to the paper) are
       vertices and stay fixed (corners: moved onto the corner);
    2. boundary nodes: Jacobi iterations of springs along the chains (with
       rest length 0: each node moves to the mean of its two chain neighbours,
       step ``relaxation``) followed by a projection onto the interface (signed
       distance image; exact by bisection at the end), until the update is
       below ``tolerance`` h; this distributes the nodes evenly along the curve
       (paper Fig. 2);
    3. interior nodes: weighted springs ``k / (dist + a)^b`` with ``dist`` the
       L1 (grid) distance to the boundary (paper Eq. 14), boundary fixed;
    4. safeguard: J at the Gauss points >= ``min_jacobian`` h^2 (paper: det F
       of the voxel >= 0.01); a move that violates it is undone within the
       iteration. The corners of an element are not checked, so elements may
       become triangles or slightly non-convex (needed where a wedge of one
       phase is narrower than a pixel).

    The boundary and corner nodes move gradually (step ``relaxation`` h) while
    the interior relaxes in every iteration; single-pixel features are flipped
    to the other phase first.

    Since no pixel changes phase, no element is cut by the interface after
    the projection (up to the curvature within one element).

``'snap'``
    Nodes snapped onto the interface: detected corners get the nearest node,
    every grid edge crossed by the interface moves its closer end node along
    the edge onto the crossing; the other nodes are relaxed (Laplace). The
    phase of an element is the majority of its area. Cheaper, but elements
    crossed diagonally by the interface stay cut.

Array layout as in muFFTTO: nodal fields ``[xy, x, y]``, node ``(i, j)`` at
``X = (i, j) h``; pixel ``(i, j)`` has the corners ``(i, j), (i+1, j), (i, j+1),
(i+1, j+1)`` (periodic). ``d_g`` is ``[2, nx, ny]``, ``phase`` ``[nx, ny]``.
Only square grids on the unit cell are supported.
"""
import numpy as np
from scipy import ndimage

from muFFTTO import geometry as geometry_library

__all__ = ['adapt_grid', 'project_nodes_to_interface', 'grid_quality', 'plot_grid', 'METHODS']

GAUSS = 0.5 - 0.5 / np.sqrt(3.0)          # 2-point Gauss abscissae in [0, 1]: GAUSS, 1 - GAUSS


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------
def _indicator(geometry, points, geometry_parameters):
    """Phase (0 / 1) at points [2, ...] (periodic in [0, 1))."""
    points = np.mod(points, 1.0)
    if callable(geometry):
        return np.asarray(geometry(points), dtype=float)
    return geometry_library.get(geometry, points, **geometry_parameters)


def _signed_distance_image(geometry, fine, geometry_parameters):
    """Signed distance to the interface (> 0 in phase 1) on a fine x fine image of pixel centres."""
    centres = (np.arange(fine) + 0.5) / fine
    phase = _indicator(geometry, np.array(np.meshgrid(centres, centres, indexing='ij')), geometry_parameters) > 0.5
    tiled = np.tile(phase, (3, 3))
    inside = ndimage.distance_transform_edt(tiled)[fine:2 * fine, fine:2 * fine]
    outside = ndimage.distance_transform_edt(~tiled)[fine:2 * fine, fine:2 * fine]
    return np.where(phase, inside - 0.5, -(outside - 0.5)) / fine


def _periodic_interpolation(image, points):
    """Bilinear interpolation of a periodic image of pixel-centre values at points [2, ...]."""
    fine = image.shape[0]
    u = np.mod(points, 1.0) * fine - 0.5
    i0 = np.floor(u).astype(int)
    t = u - i0
    value = 0.0
    for di in (0, 1):
        for dj in (0, 1):
            weight = (t[0] if di else 1 - t[0]) * (t[1] if dj else 1 - t[1])
            value = value + weight * image[(i0[0] + di) % fine, (i0[1] + dj) % fine]
    return value


class _Interface:
    """Signed distance, normal and exact projection onto the interface of a geometry."""

    def __init__(self, geometry, geometry_parameters, fine):
        self.geometry, self.parameters = geometry, geometry_parameters
        self.sdf = _signed_distance_image(geometry, fine, geometry_parameters)
        self.normal = np.array(np.gradient(np.pad(self.sdf, 1, mode='wrap'), 1.0 / fine))[:, 1:-1, 1:-1]

    def phase(self, points):
        return _indicator(self.geometry, points, self.parameters) > 0.5

    def project(self, points, h):
        """Nearest interface points of points [2, k]: along the normal, refined by bisection."""
        s = _periodic_interpolation(self.sdf, points)
        n = np.array([_periodic_interpolation(self.normal[k], points) for k in range(2)])
        n /= np.maximum(np.linalg.norm(n, axis=0), 1e-12)
        a, b = -s - 0.5 * h, -s + 0.5 * h                         # points + t n, t in [a, b]
        phase_a, phase_b = self.phase(points + a * n), self.phase(points + b * n)
        for _ in range(30):
            mid = 0.5 * (a + b)
            same = self.phase(points + mid * n) == phase_a
            a, b = np.where(same, mid, a), np.where(same, b, mid)
        found = phase_a != phase_b
        return np.where(found, points + 0.5 * (a + b) * n, points - s * n)


def _interface_corners(sdf, h, corner_angle):
    """Sharp corners of the zero contour of the periodic image sdf, as points [k, 2] in [0, 1)."""
    import contourpy
    fine = sdf.shape[0]
    pad = fine // 4
    coordinate = (np.arange(-pad, fine + pad) + 0.5) / fine
    lines = contourpy.contour_generator(coordinate, coordinate, np.pad(sdf, pad, mode='wrap').T).lines(0.0)
    window, step = h / 3.0, 1.0 / fine
    corners = []
    for line in lines:
        length = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(line, axis=0), axis=1))]
        if length[-1] < 4 * window:
            continue
        s = np.arange(0.0, length[-1], step)
        points = np.c_[np.interp(s, length, line[:, 0]), np.interp(s, length, line[:, 1])]
        k = max(3, int(round(window / step)))
        if np.linalg.norm(line[0] - line[-1]) < 2 * step:          # closed loop: continue over the start
            points = np.r_[points[-k:], points, points[:k]]
        before = points[k:-k] - points[:-2 * k]
        after = points[2 * k:] - points[k:-k]
        cosine = np.sum(before * after, axis=1) / (np.linalg.norm(before, axis=1) * np.linalg.norm(after, axis=1))
        turn = np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))
        for m in np.nonzero(turn > corner_angle)[0]:
            if turn[m] >= turn[max(0, m - k):m + k + 1].max():     # local maximum
                corners.append(points[m + k])
    unique = []                                                     # the padding repeats corners
    for c in np.mod(np.array(corners).reshape(-1, 2), 1.0):
        if all(np.linalg.norm(np.mod(c - u + 0.5, 1.0) - 0.5) > h / 4 for u in unique):
            unique.append(c)
    return np.array(unique).reshape(-1, 2)


# ---------------------------------------------------------------------------
# Element geometry
# ---------------------------------------------------------------------------
def _element_corners(d_g, h):
    """Corner positions p00, p10, p01, p11 [2, nx, ny] of all elements of the grid X + d_g."""
    n = d_g.shape[1]
    X = np.array(np.meshgrid(np.arange(n) * h, np.arange(n) * h, indexing='ij'))
    shift = np.array([h, 0.0])[:, None, None], np.array([0.0, h])[:, None, None]
    p00 = X + d_g
    p10 = X + shift[0] + np.roll(d_g, -1, axis=1)
    p01 = X + shift[1] + np.roll(d_g, -1, axis=2)
    p11 = X + shift[0] + shift[1] + np.roll(np.roll(d_g, -1, axis=1), -1, axis=2)
    return p00, p10, p01, p11


def _bilinear(corners, a, b):
    """Position and Jacobian of the bilinear map at the local coordinates (a, b) in [0, 1]^2."""
    p00, p10, p01, p11 = corners
    x = (1 - a) * (1 - b) * p00 + a * (1 - b) * p10 + (1 - a) * b * p01 + a * b * p11
    dx_da = (1 - b) * (p10 - p00) + b * (p11 - p01)
    dx_db = (1 - a) * (p01 - p00) + a * (p11 - p10)
    return x, dx_da[0] * dx_db[1] - dx_da[1] * dx_db[0]


def _jacobians(d_g, h):
    """(J / h^2 at the 4 Gauss points, J / h^2 at the 4 corners) of all elements."""
    corners = _element_corners(d_g, h)
    gauss = np.array([_bilinear(corners, a, b)[1] for a in (GAUSS, 1 - GAUSS) for b in (GAUSS, 1 - GAUSS)])
    corner = np.array([_bilinear(corners, a, b)[1] for a in (0.0, 1.0) for b in (0.0, 1.0)])
    return gauss / h ** 2, corner / h ** 2


def _phase_fractions(d_g, h, interface, samples=8):
    """Area of every element and the fraction of it in phase 1 (samples^2 points of the bilinear map)."""
    corners = _element_corners(d_g, h)
    t = (np.arange(samples) + 0.5) / samples
    area, phase_area = 0.0, 0.0
    for a in t:
        for b in t:
            x, jac = _bilinear(corners, a, b)
            area = area + jac
            phase_area = phase_area + jac * interface.phase(x)
    return area / samples ** 2, phase_area / np.where(area != 0, area, 1.0)


def _invalid_elements(d_g, h, min_jacobian, min_corner_jacobian=None):
    """Elements with J / h^2 < min_jacobian at a Gauss point (the only requirement of the pull-back);
    optionally also J / h^2 < min_corner_jacobian at a corner (None: corners not checked, so slightly
    non-convex elements are allowed)."""
    gauss, corner = _jacobians(d_g, h)
    bad = gauss.min(axis=0) < min_jacobian
    if min_corner_jacobian is not None:
        bad |= corner.min(axis=0) < min_corner_jacobian
    return bad


def _keep_valid(new, old, h, min_jacobian, min_corner_jacobian=None):
    """new, with the nodes around invalid elements reset to old, widened until all elements are
    valid (old must be valid, so this ends at the latest with old); returns (grid, reset nodes)."""
    reset = np.zeros(new.shape[1:], dtype=bool)
    grid = new
    for _ in range(new.shape[1] * new.shape[2]):
        bad = _invalid_elements(grid, h, min_jacobian, min_corner_jacobian)
        if not bad.any():
            break
        reset |= _element_to_node(bad)
        grid = np.where(reset, old, new)
    return grid, reset


def _element_to_node(mask):
    """Nodes touching at least one element of the element mask [nx, ny]."""
    return mask | np.roll(mask, 1, axis=0) | np.roll(mask, 1, axis=1) | np.roll(mask, (1, 1), axis=(0, 1))


def _neighbour_offsets(d_g, h):
    """Relative positions x(neighbour) - x(node) for the neighbours +x, -x, +y, -y, [4, 2, nx, ny]."""
    out = []
    for axis, sign in ((1, -1), (1, 1), (2, -1), (2, 1)):
        step = np.zeros((2, 1, 1))
        step[axis - 1] = -sign * h
        out.append(step + np.roll(d_g, sign, axis=axis) - d_g)
    return np.array(out)


def _relax_interior(d_g, fixed, h, iterations, stiffness=None):
    """Weighted Laplace (springs with rest length 0) of the free nodes, fixed nodes kept."""
    weights = np.ones((4,) + fixed.shape) if stiffness is None else np.array(
        [np.roll(stiffness, sign, axis=axis - 1) for axis, sign in ((1, -1), (1, 1), (2, -1), (2, 1))])
    total = weights.sum(axis=0)
    for _ in range(iterations):
        offsets = _neighbour_offsets(d_g, h)
        update = np.sum(weights[:, None] * offsets, axis=0) / total
        d_g = np.where(fixed, d_g, d_g + update)
    return d_g


def _graph_distance(mask):
    """L1 (grid-graph) distance of every node to the nodes in mask, periodic (BFS by dilation)."""
    distance = np.where(mask, 0, -1)
    front, level = mask.copy(), 0
    while (distance < 0).any() and front.any():
        level += 1
        grown = (np.roll(front, 1, 0) | np.roll(front, -1, 0) | np.roll(front, 1, 1) | np.roll(front, -1, 1))
        front = grown & (distance < 0)
        distance[front] = level
    return np.where(distance < 0, level + 1, distance)


# ---------------------------------------------------------------------------
# Methods
# ---------------------------------------------------------------------------
def _assign_corners(corners, candidates, X, h, max_distance):
    """Nearest candidate node of every corner (each node once): (mask, target positions)."""
    n = X.shape[1]
    is_corner = np.zeros((n, n), dtype=bool)
    target = np.array(X, copy=True)
    for c in corners:
        offset = np.mod(c[:, None, None] - X + 0.5, 1.0) - 0.5     # minimum image
        distance = np.where(candidates & ~is_corner, np.linalg.norm(offset, axis=0), np.inf)
        i, j = np.unravel_index(np.argmin(distance), distance.shape)
        if distance[i, j] < max_distance:
            is_corner[i, j] = True
            target[:, i, j] = X[:, i, j] + offset[:, i, j]
    return is_corner, target


def _clean_pixel_phase(v):
    """Flip pixels with 3 or 4 of their 4 neighbours in the other phase (single-pixel features
    would collapse when the staircase is mapped onto the interface)."""
    v = v.copy()
    for _ in range(v.size):
        other = sum((np.roll(v, shift, axis) != v).astype(int) for axis in (0, 1) for shift in (1, -1))
        flip = other >= 3
        if not flip.any():
            break
        v[flip] = ~v[flip]
    return v


def _zecevic(n, interface, corners, relaxation=0.3, tolerance=1e-6, min_iterations=200, max_iterations=3000, stiffness_b=0.5,
             stiffness_a=1.0, interior_sweeps=10, final_interior_iterations=500, min_jacobian=0.01,
             min_corner_jacobian=None, corner_distance=1.5):
    h = 1.0 / n
    X = np.array(np.meshgrid(np.arange(n) * h, np.arange(n) * h, indexing='ij'))
    v = _clean_pixel_phase(interface.phase(X + 0.5 * h))           # pixel phase on the regular grid
    phase = v.astype(float)
    # boundary edges: x-edge node(i,j)-node(i+1,j) separates pixels (i,j-1),(i,j); y-edge: (i-1,j),(i,j)
    edge_x = v != np.roll(v, 1, axis=1)
    edge_y = v != np.roll(v, 1, axis=0)
    links = np.array([edge_x, np.roll(edge_x, 1, axis=0), edge_y, np.roll(edge_y, 1, axis=1)])  # +x, -x, +y, -y
    degree = links.sum(axis=0)
    boundary = degree > 0
    is_corner, corner_target = _assign_corners(corners, boundary & (degree == 2), X, h, corner_distance * h)
    junction = (degree > 2) & ~is_corner                            # pixels touching at a corner
    junction_target = np.array(X, copy=True)
    if junction.any():
        junction_target[:, junction] = interface.project(X[:, junction], h)
    target_fixed = np.where(is_corner, corner_target, junction_target)
    fixed_nodes = is_corner | junction
    chain = boundary & ~fixed_nodes
    distance = _graph_distance(boundary)
    stiffness = 1.0 / (distance + stiffness_a) ** stiffness_b
    d_g = np.zeros((2, n, n))
    # 2. boundary nodes move step by step (springs along the chains + projection), the corners and
    # junctions towards their targets, the interior relaxes in between; a move that makes an element
    # invalid is undone (safeguard)
    for iteration in range(max_iterations):
        old = d_g.copy()
        offsets = _neighbour_offsets(d_g, h)
        chain_mean = np.sum(links[:, None] * offsets, axis=0) / np.maximum(degree, 1)
        moved = X + d_g + relaxation * chain_mean
        s = _periodic_interpolation(interface.sdf, moved)
        normal = np.array([_periodic_interpolation(interface.normal[k], moved) for k in range(2)])
        normal /= np.maximum(np.linalg.norm(normal, axis=0), 1e-12)
        # projection with a step limited to relaxation * h, so that the nodes come in gradually
        step = np.clip(-s, -relaxation * h, relaxation * h) * normal
        d_g = np.where(chain, moved + step - X, d_g)
        towards = target_fixed - X - d_g
        d_g = np.where(fixed_nodes, d_g + np.clip(np.linalg.norm(towards, axis=0), None, relaxation * h)
                       * towards / np.maximum(np.linalg.norm(towards, axis=0), 1e-30), d_g)
        d_g = _relax_interior(d_g, boundary, h, interior_sweeps, stiffness)
        d_g, _ = _keep_valid(d_g, old, h, min_jacobian, min_corner_jacobian)
        change = np.max(np.linalg.norm((d_g - old)[:, boundary], axis=0)) / h
        if change < tolerance and iteration >= min_iterations:
            break
    # 3. exact projection of the chain nodes (small correction) and final interior relaxation
    exact = np.array(d_g, copy=True)
    exact[:, chain] = interface.project(X[:, chain] + d_g[:, chain], h) - X[:, chain]
    exact = _relax_interior(exact, boundary, h, final_interior_iterations, stiffness)
    d_g, _ = _keep_valid(exact, d_g, h, min_jacobian, min_corner_jacobian)
    x = X + d_g
    on = np.zeros_like(boundary)
    on[boundary] = np.linalg.norm(interface.project(x[:, boundary], h) - x[:, boundary], axis=0) < 1e-3 * h
    return d_g, phase, {'nb_boundary_nodes': int(boundary.sum()), 'nb_on_interface': int(on.sum()),
                        'nb_corner_nodes': int(is_corner.sum()), 'nb_junctions': int(junction.sum()),
                        'boundary_iterations': iteration + 1}


def _snap(n, interface, corners, smoothing_iterations=300, min_jacobian=0.05, corner_distance=0.75):
    h = 1.0 / n
    X = np.array(np.meshgrid(np.arange(n) * h, np.arange(n) * h, indexing='ij'))
    is_corner, target = _assign_corners(corners, np.ones((n, n), dtype=bool), X, h, corner_distance * h)
    # every grid edge crossed by the interface moves its end node closer to the crossing onto it
    phase_node = interface.phase(X)
    best = np.full((n, n), np.inf)
    for axis in (0, 1):
        unit = np.zeros(2)
        unit[axis] = h
        crossed = phase_node != np.roll(phase_node, -1, axis=axis)
        start = X[:, crossed]
        a, b = np.zeros(start.shape[1]), np.ones(start.shape[1])
        for _ in range(30):
            mid = 0.5 * (a + b)
            same = interface.phase(start + mid * unit[:, None]) == phase_node[crossed]
            a, b = np.where(same, mid, a), np.where(same, b, mid)
        for i, j, t in zip(*np.nonzero(crossed), 0.5 * (a + b)):
            node = (i, j) if t <= 0.5 else (((i + 1) % n, j) if axis == 0 else (i, (j + 1) % n))
            distance = min(t, 1 - t) * h
            if not is_corner[node] and distance < best[node]:
                best[node] = distance
                target[:, node[0], node[1]] = X[:, node[0], node[1]] + (t if node == (i, j) else t - 1) * unit
    snapped = is_corner | np.isfinite(best)
    released = 0
    while True:
        d_g = _relax_interior(np.where(snapped, target - X, 0.0), snapped, h, smoothing_iterations)
        bad = _invalid_elements(d_g, h, min_jacobian)
        if not bad.any():
            break
        nodes = _element_to_node(bad) & snapped
        if not nodes.any():
            break
        move = np.where(nodes, np.linalg.norm(target - X, axis=0), -1.0)
        worst = np.unravel_index(np.argmax(move), move.shape)
        snapped[worst] = False
        released += 1
    _, fraction = _phase_fractions(d_g, h, interface)
    return d_g, (fraction > 0.5).astype(float), {'nb_corner_nodes': int(is_corner.sum()),
                                                  'nb_snapped': int(snapped.sum()), 'nb_released': released}


METHODS = {'zecevic': _zecevic, 'snap': _snap}


def project_nodes_to_interface(d_g, geometry, radius=0.5, geometry_parameters=None, min_jacobian=0.01,
                               relax_iterations=200, max_rounds=20, interface=None, verbose=True):
    """Move every node closer than radius h to the interface onto it (nearest point).

    A post-processing step for any grid (e.g. the 'zecevic' grid), in rounds:

    1. nodes on the interface (exact projection moves them < 1e-6 h) are fixed; the free nodes closer than radius h
       are projected onto the interface and fixed as well;
    2. the remaining free nodes are relaxed (Laplace springs, fixed nodes kept), so that the
       elements around the projected nodes can take the new shape;
    3. if an element is invalid (J / h^2 < min_jacobian at a Gauss point, or a folded corner),
       the projected node of it with the largest move is released, and 2-3 are repeated;
    4. next round, until no free node is closer than radius h.

    The element phases are not changed. Returns (d_g, info).
    """
    geometry_parameters = geometry_parameters or {}
    n = d_g.shape[1]
    h = 1.0 / n
    interface = interface or _Interface(geometry, geometry_parameters, max(512, 8 * n))
    X = np.array(np.meshgrid(np.arange(n) * h, np.arange(n) * h, indexing='ij'))
    # exactly on the interface: the exact projection does not move the node (the signed distance
    # image is only accurate to ~1 / fine, too coarse for this test)
    x = (X + d_g).reshape(2, -1)
    on_interface = (np.linalg.norm(interface.project(x, h) - x, axis=0) < 1e-6 * h).reshape(n, n)
    # nodes already exactly on the interface (from the method) and released nodes are not touched again
    fixed = on_interface.copy()
    released = np.zeros_like(fixed)
    nb_projected, nb_released = 0, 0
    for round_ in range(max_rounds):
        x = X + d_g
        close = (np.abs(_periodic_interpolation(interface.sdf, x)) < radius * h) & ~fixed & ~released
        if not close.any():
            break
        new = np.array(d_g, copy=True)
        new[:, close] = interface.project(x[:, close], h) - X[:, close]
        move = np.where(close, np.linalg.norm(new - d_g, axis=0), 0.0)
        projected = close.copy()
        while True:
            candidate = _relax_interior(new, fixed | projected, h, relax_iterations)
            bad = _invalid_elements(candidate, h, min_jacobian)
            nodes = _element_to_node(bad) & projected
            if not bad.any() or not nodes.any():
                break
            # release, per invalid element, its projected node with the largest move
            for i, k in zip(*np.nonzero(bad)):
                corners = [((i + a) % n, (k + b) % n) for a in (0, 1) for b in (0, 1)]
                corners = [c for c in corners if projected[c]]
                if corners:
                    worst = max(corners, key=lambda c: move[c])
                    projected[worst] = False
                    released[worst] = True
                    new[:, worst[0], worst[1]] = d_g[:, worst[0], worst[1]]
        d_g = candidate
        fixed |= projected
        nb_projected += int(projected.sum())
    nb_released = int(released.sum())
    still_close = (np.abs(_periodic_interpolation(interface.sdf, X + d_g)) < radius * h) & ~fixed
    info = {'nb_projected': nb_projected, 'nb_released': nb_released, 'rounds': round_ + 1,
            'nb_close_not_on_interface': int(still_close.sum())}
    if verbose:
        print('projection onto the interface: ' + ', '.join(f'{k} {v}' for k, v in info.items()))
    return d_g, info


def adapt_grid(nb_pixels, geometry, method='zecevic', geometry_parameters=None, fine=None, corner_angle=35.0,
               verbose=True, **options):
    """Conformal grid for a two-phase geometry.

    Parameters
    ----------
    nb_pixels : int or (int, int)
        Grid size (square only).
    geometry : str or callable
        Name from :func:`muFFTTO.geometry.available` or ``phase(points [2, ...]) -> 0 / 1``.
    method : str
        ``'zecevic'`` or ``'snap'`` (see the module docstring).
    geometry_parameters : dict, optional
        Passed to the geometry.
    fine : int, optional
        Resolution of the image for the signed distance and the corner detection
        (default ``max(512, 8 n)``).
    corner_angle : float
        Minimal turning angle (degrees) of the interface contour over ~h/3 for a corner.
    **options
        Method options (see ``_zecevic`` / ``_snap``).

    Returns
    -------
    d_g : ndarray [2, n, n]
        Node displacement of the grid.
    phase : ndarray [n, n]
        Phase (0 / 1) of every element.
    info : dict
        Method statistics and :func:`grid_quality`.
    """
    n = int(np.atleast_1d(nb_pixels)[0])
    if np.any(np.atleast_1d(nb_pixels) != n):
        raise ValueError('only square grids are supported')
    geometry_parameters = geometry_parameters or {}
    interface = _Interface(geometry, geometry_parameters, fine or max(512, 8 * n))
    corners = _interface_corners(interface.sdf, 1.0 / n, corner_angle)
    d_g, phase, info = METHODS[method](n, interface, corners, **options)
    min_gauss = _jacobians(d_g, 1.0 / n)[0].min()
    if not min_gauss > 0.0:
        raise RuntimeError(f'{method} grid {n}^2 is not valid: min J / h^2 at the Gauss points {min_gauss:.3g}')
    info.update(grid_quality(d_g, phase, geometry, geometry_parameters, interface=interface))
    info.update(method=method, nb_corners_found=len(corners), corners=corners)
    if verbose:
        print(f'{method} grid {n}^2: ' + ', '.join(f'{key} {value:.4g}' if isinstance(value, float) else
                                                   f'{key} {value}' for key, value in info.items()
                                                   if key not in ('corners', 'method')))
    return d_g, phase, info


def grid_quality(d_g, phase, geometry, geometry_parameters=None, interface=None, samples=8):
    """Geometric quality of a grid with element phases.

    Returns a dict: ``solid_fraction`` (area of phase 1), ``exact_fraction``,
    ``misassigned_area`` (area of the elements in the wrong phase, by sampling the
    bilinear maps), ``nb_cut`` (elements with 2-98 % of the other phase),
    ``min_gauss_J`` / ``min_corner_J`` (J / h^2), ``max_move`` (max |d_g| / h).
    """
    geometry_parameters = geometry_parameters or {}
    n = d_g.shape[1]
    h = 1.0 / n
    interface = interface or _Interface(geometry, geometry_parameters, max(512, 8 * n))
    area, fraction = _phase_fractions(d_g, h, interface, samples)
    wrong = np.where(phase > 0.5, 1.0 - fraction, fraction)
    gauss, corner = _jacobians(d_g, h)
    return {'solid_fraction': float(np.sum(area * (phase > 0.5))), 'exact_fraction': float(np.mean(interface.sdf > 0)),
            'misassigned_area': float(np.sum(area * wrong)), 'nb_cut': int(np.sum((wrong > 0.02))),
            'min_gauss_J': float(gauss.min()), 'min_corner_J': float(corner.min()),
            'max_move': float(np.abs(d_g).max() / h)}


def plot_grid(ax, d_g, phase, color_solid=(0.35, 0.35, 0.35), color_medium=(0.75, 0.87, 1.0), linewidth=0.2):
    """Draw the elements of the grid X + d_g coloured by phase (periodic corners added)."""
    from matplotlib.colors import ListedColormap
    n = d_g.shape[1]
    h = 1.0 / n
    i, j = np.meshgrid(np.arange(n + 1), np.arange(n + 1), indexing='ij')
    x = i * h + d_g[0][i % n, j % n]
    y = j * h + d_g[1][i % n, j % n]
    ax.pcolormesh(x, y, phase, cmap=ListedColormap([color_medium, color_solid]), vmin=0, vmax=1,
                  edgecolors='k', linewidth=linewidth, shading='flat')
    ax.set_aspect('equal')
