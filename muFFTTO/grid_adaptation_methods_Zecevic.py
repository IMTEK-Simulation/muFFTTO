'''
(FIXED!)
Periodic Unit Grid Adaptation using spring relaxation method by Zecevic

Achieving geometric accuracy in FFT-based micromechanical models using
conformal grid

How to use:
coords_of_displaced_nodes,phase_indicator_array = adapt_grid_to_circle(ref_grid_coords_ixyz,center,radius)

Overview
--------
This module implements a conformal ("interface-fitted") grid adaptation for a
2D periodic unit cell containing a single circular inclusion, following the
idea of Zecevic et al. ("Achieving geometric accuracy in FFT-based
micromechanical models using conformal grid"). The topology of the regular
grid is kept (same number of nodes, same connectivity, same periodicity); only
the nodal positions are moved so that the discrete phase boundary follows the
circle. The algorithm is:

1. ``cell_labels`` -- classify every (pixel) cell of the *reference* regular
   grid as inside (1) or outside (0) of the circle, based on its cell center.
2. ``interface_node_mask`` -- mark nodes whose four surrounding cells carry
   different labels; these nodes lie on the staircase phase boundary.
3. ``project_points_to_circle`` -- move the interface nodes radially onto the
   exact circle. The cell labels are NOT recomputed, i.e. the phase of each
   cell stays the one of the reference grid; only its geometry changes.
4. ``manhattan_distance_to_interface`` + ``stiffness_from_distance`` -- build
   a nodal spring stiffness ``k = max(k0 / (d + a)**b, kmin)`` that is large
   close to the interface and decays with the (periodic, graph) distance ``d``.
5. ``spring_relax_weighted`` -- keep the interface nodes fixed and relax all
   other nodes with a weighted Laplacian (spring) smoothing, so that the
   distortion caused by the projection is spread smoothly into the grid.

Array conventions
-----------------
Nodal coordinates are stored as ``P[xy, i, j]`` with ``xy = 0`` (x) and
``xy = 1`` (y), ``i`` the node index along x and ``j`` along y. Only the
``N x N`` *stored* nodes of the periodic grid are kept; the node ``N`` is the
periodic image of node ``0`` shifted by the domain length ``N * dx``. Cell
``(i, j)`` is spanned by nodes ``(i, j), (i+1, j), (i, j+1), (i+1, j+1)``
(indices modulo ``N``). Several routines assume a square grid
(``nx == ny == N``) and a uniform reference spacing.
'''
import numpy as np
from collections import deque


def cell_labels(
        ref_grid_coords_ixyz: np.ndarray,
        center: tuple[float, float] = (0.0, 0.0),
        radius: float = 20.0,
) -> np.ndarray:
    """
    Function that labels cells as inside or outside a circular inclusion.

    Cell centers are computed from the four corner nodes of each periodic cell.
    A cell is marked as inside if its center lies inside or on the target circle.

    Parameters
    ----------
    ref_grid_coords_ixyz : numpy.ndarray
        Array of nodal coordinates with shape [xy, nx, ny].
    center : tuple[float, float]
        Coordinates of the circle center given as (x_center, y_center).
    radius : float
        Radius of the circular inclusion.

    Returns
    -------
    inside : numpy.ndarray
        Integer array with shape [nx, ny].
        A value of 1 indicates that the periodic cell center lies inside the circle,
        and a value of 0 indicates that it lies outside.
    """
    cx, cy = center
    P = ref_grid_coords_ixyz
    R = radius

    # Corner nodes of cell (i, j): P[i,j], P[i+1,j], P[i,j+1], P[i+1,j+1].
    # np.roll with shift=-1 brings node (i+1) to index i (periodic wrap).
    P_ip1 = np.roll(P, shift=-1, axis=1)
    P_jp1 = np.roll(P, shift=-1, axis=2)
    P_ip1_jp1 = np.roll(P_ip1, shift=-1, axis=2)

    P_ip1 = P_ip1.copy()
    P_jp1 = P_jp1.copy()
    P_ip1_jp1 = P_ip1_jp1.copy()

    # Reference grid spacing (assumed uniform); used to "unwrap" the rolled
    # coordinates of the last row/column, which otherwise would jump back to
    # the left/bottom boundary instead of being the periodic image at x = L.
    dx_grid = P[0, 1, 0] - P[0, 0, 0]
    dy_grid = P[1, 0, 1] - P[1, 0, 0]

    P_ip1[0, -1, :] += dx_grid * P.shape[1]
    P_ip1_jp1[0, -1, :] += dx_grid * P.shape[1]
    P_jp1[1, :, -1] += dy_grid * P.shape[2]
    P_ip1_jp1[1, :, -1] += dy_grid * P.shape[2]

    # Cell centers = average of the four (unwrapped) corner nodes, shape [xy, nx, ny]
    Pc = 0.25 * (P + P_ip1 + P_jp1 + P_ip1_jp1)
    dx = Pc[0] - cx
    dy = Pc[1] - cy
    inside_mask = (dx * dx + dy * dy) <= R * R
    return inside_mask.astype(np.int32)

def interface_node_mask(cell_inside: np.ndarray) -> np.ndarray:
    """
    Function that detects interface nodes on a periodic stored grid.

    A stored node is marked as an interface node if the surrounding periodic cells
    do not all have the same inside/outside label. In other words, the node lies
    on the discrete boundary between two material regions.

    Parameters
    ----------
    cell_inside : numpy.ndarray
        Integer array of cell labels with shape [nx, ny].
        Cells with value 1 are inside the inclusion and cells with value 0 are outside.

    Returns
    -------
    mask : numpy.ndarray
        Boolean array with shape [nx, ny].
        True indicates that the node is an interface node, and False otherwise.
    """
    # NOTE: assumes a square grid (nx == ny == N).
    N = cell_inside.shape[0]
    mask = np.zeros((N, N), dtype=bool) # 創一個紀錄node 是不是 interface node 的 array (array flagging interface nodes)

    for i in range(N):
        for j in range(N):
            # Node (i, j) is the shared corner of the four cells
            # (i-1, j-1), (i, j-1), (i-1, j), (i, j) (periodic indices).
            vals = np.array([
                cell_inside[(i - 1) % N, (j - 1) % N],
                cell_inside[i % N, (j - 1) % N],
                cell_inside[(i - 1) % N, j % N],
                cell_inside[i % N, j % N],
            ], dtype=int)
            if vals.min() != vals.max():
                mask[i, j] = True

    return mask

def project_points_to_circle(
        Ppts: np.ndarray,
        center: tuple[float, float] = (0.0, 0.0),
        R: float = 20.0,
) -> np.ndarray:
    """
    Function that projects points onto a target circle.

    Each input point is moved along the radial direction so that its distance
    from the circle center becomes exactly equal to the prescribed radius.
    The function accepts point arrays in either [xy, k] or [k, xy] format.

    Parameters
    ----------
    Ppts : numpy.ndarray
        Array of point coordinates with shape [xy, k] or [k, xy].
    center : tuple[float, float]
        Coordinates of the circle center given as (x_center, y_center).
    R : float
        Radius of the target circle.

    Returns
    -------
    proj : numpy.ndarray
        Array of projected point coordinates with the same shape convention
        as the input array `Ppts`.

    Raises
    ------
    ValueError
        If `Ppts` is not a two-dimensional array with shape [xy, k] or [k, xy].
    """
    c = np.asarray(center, dtype=float)

    transposed = False
    if Ppts.ndim != 2:
        raise ValueError("Ppts must be a 2D array")
    if Ppts.shape[0] == 2:
        pts = Ppts.T
        transposed = True
    elif Ppts.shape[1] == 2:
        pts = Ppts
    else:
        raise ValueError("Ppts must have shape [xy, k] or [k, xy]")

    # Radial vector from the center to every point, shape [k, 2].
    # NOTE: no periodic (minimum-image) correction is applied, so the circle is
    # assumed to lie fully inside the unit cell.
    V = pts - c[None, :]
    r = np.linalg.norm(V, axis=1, keepdims=True)
    r = np.maximum(r, 1e-12)  # guard against division by zero for a point at the center
    proj = c[None, :] + (R / r) * V
    return proj.T if transposed else proj

def manhattan_distance_to_interface(interface_mask: np.ndarray) -> np.ndarray:
    """
    Function that computes the periodic Manhattan distance to the nearest interface node.

    The distance is measured in the discrete grid sense using nearest-neighbor
    connectivity in the x- and y-directions. Periodic wrapping is applied at the
    domain boundaries.

    Parameters
    ----------
    interface_mask : numpy.ndarray
        Boolean array with shape [nx, ny].
        True marks interface nodes and False marks non-interface nodes.

    Returns
    -------
    dist : numpy.ndarray
        Floating-point array with shape [nx, ny].
        Each entry contains the periodic Manhattan distance from that node to
        the nearest interface node.
    """
    # Multi-source breadth-first search (BFS) on the periodic 4-neighbour
    # node graph, seeded with all interface nodes (distance 0). All edges have
    # unit weight, so BFS yields the exact graph (Manhattan) distance.
    # NOTE: assumes a square grid (nx == ny == N).
    N = interface_mask.shape[0]
    dist = np.full((N, N), np.inf, dtype=float)
    queue = deque()

    for i in range(N):
        for j in range(N):
            if interface_mask[i, j]:
                dist[i, j] = 0.0
                queue.append((i, j))

    neighbors = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    while queue:
        i, j = queue.popleft()
        for di, dj in neighbors:
            ii = (i + di) % N
            jj = (j + dj) % N
            if dist[ii, jj] > dist[i, j] + 1.0:
                dist[ii, jj] = dist[i, j] + 1.0
                queue.append((ii, jj))

    return dist


def stiffness_from_distance(
        dist: np.ndarray,
        k0: float = 1.0,
        b: float = 1.0,
        a: float = 1.0,
        kmin: float = 1e-3,
) -> np.ndarray:
    """
    Function that computes a nodal stiffness field from the distance to the interface.

    The stiffness is defined by a distance-dependent power law and is bounded
    from below by a prescribed minimum stiffness value.

    Parameters
    ----------
    dist : numpy.ndarray
        Array of nodal distances to the nearest interface with shape [nx, ny].
    k0 : float
        Reference stiffness factor.
    b : float
        Exponent controlling how fast the stiffness decays with distance.
    a : float
        Positive shift added to the distance to avoid division by zero and to
        control the stiffness near the interface.
    kmin : float
        Minimum admissible stiffness value.

    Returns
    -------
    k : numpy.ndarray
        Floating-point stiffness array with shape [nx, ny].
    """
    # k(d) = k0 / (d + a)^b, clipped from below by kmin
    k = k0 / np.power(dist + a, b)
    return np.maximum(k, kmin)

def spring_relax_weighted(
        P: np.ndarray,
        fixed_mask: np.ndarray,
        k_node: np.ndarray,
        iters: int = 600,
        omega: float = 1.0,
) -> np.ndarray:
    """
    Function that performs weighted spring relaxation on a periodic stored grid.

    Each non-fixed node is iteratively moved toward a weighted average of its
    four nearest neighbors. The nodal stiffness field controls the local strength
    of the spring interaction, and periodic wrapping is applied at the domain boundaries.

    Parameters
    ----------
    P : numpy.ndarray
        Array of nodal coordinates with shape [xy, nx, ny].
    fixed_mask : numpy.ndarray
        Boolean array with shape [nx, ny].
        True marks fixed nodes that remain unchanged during relaxation.
    k_node : numpy.ndarray
        Floating-point nodal stiffness array with shape [nx, ny].
    iters : int
        Number of relaxation iterations.
    omega : float
        Relaxation parameter. Values between 0 and 1 correspond to under-relaxation,
        and omega = 1 applies the full update at each iteration.

    Returns
    -------
    P_new : numpy.ndarray
        Relaxed nodal coordinates with shape [xy, nx, ny].

    Notes
    -----
    The update for a free node ``p_ij`` reads

    .. math::

        p_{ij} \\leftarrow (1-\\omega)\\, p_{ij}
            + \\omega \\frac{\\sum_{n} w_n\\, p_n}{\\sum_n w_n},
        \\qquad w_n = \\tfrac12 (k_{ij} + k_n),

    where the sum runs over the four nearest neighbours ``n``. This is the
    equilibrium position of the node attached to its neighbours by linear
    springs with stiffness ``w_n``. Updates are done in place (Gauss-Seidel
    style), i.e. already-updated neighbours are used within the same sweep.
    The spacings ``dx_grid``/``dy_grid`` are read from the first two nodes of
    the *input* grid ``P`` and used to shift neighbours across the periodic
    boundary by the domain length ``N * dx_grid``. Pure-Python triple loop:
    cost is ``O(iters * N**2)`` and can be slow for large grids.
    Assumes a square grid (nx == ny == N).
    """
    P_new = P.copy()
    N = P.shape[1]

    dx_grid = P[0, 1, 0] - P[0, 0, 0]
    dy_grid = P[1, 0, 1] - P[1, 0, 0]

    for _ in range(iters):
        for i in range(N):
            for j in range(N):
                if fixed_mask[i, j]:
                    continue

                neighbors = [
                    ((i - 1) % N, j),
                    ((i + 1) % N, j),
                    (i, (j - 1) % N),
                    (i, (j + 1) % N),
                ]
                kij = k_node[i, j]
                w = np.empty(4, dtype=float)
                pts = np.empty((4, 2), dtype=float)

                xij = P_new[0, i, j]
                yij = P_new[1, i, j]

                for idx, (ii, jj) in enumerate(neighbors):
                    xnb = P_new[0, ii, jj]
                    ynb = P_new[1, ii, jj]

                    # Periodic unwrapping: if the neighbour was reached by
                    # wrapping around the boundary, shift it by the domain
                    # length so that it is spatially adjacent to node (i, j).
                    if ii == 0 and i == N - 1:
                        xnb += dx_grid * N
                    elif ii == N - 1 and i == 0:
                        xnb -= dx_grid * N

                    if jj == 0 and j == N - 1:
                        ynb += dy_grid * N
                    elif jj == N - 1 and j == 0:
                        ynb -= dy_grid * N

                    # Spring stiffness of edge (i,j)-(ii,jj): mean of nodal stiffnesses
                    w[idx] = 0.5 * (kij + k_node[ii, jj])
                    pts[idx] = [xnb, ynb]

                # Weighted average of neighbours = spring equilibrium position;
                # then under-/over-relaxed update with factor omega.
                target = (w[:, None] * pts).sum(axis=0) / (w.sum() + 1e-15)
                P_new[:, i, j] = (1.0 - omega) * np.array([xij, yij]) + omega * target

    return P_new

# MAIN FUNCTION
def adapt_grid_to_circle(
        ref_grid_coords_ixyz: np.ndarray = None,
        center: tuple[float, float] = (0.5, 0.5),
        radius: float = 0.2,
        iters: int = 600,
        omega: float = 0.8,
        b: float = 0.5,
        k0: float = 0.25,
        kmin: float = 0.0,
        is_projection : bool = True,
):
    """
    Adapt a periodic regular grid to a circular inclusion (Zecevic-type method).

    This is the main driver of the module. It labels the cells of the reference
    grid, projects the staircase interface nodes onto the circle and smooths
    the remaining nodes by weighted spring relaxation (see module docstring).

    Parameters
    ----------
    ref_grid_coords_ixyz : numpy.ndarray
        Nodal coordinates of the regular (reference) periodic grid with shape
        [xy, nx, ny] (``nx == ny`` required by the helper routines). Must be
        provided despite the default ``None``.
    center : tuple[float, float], optional
        Circle center (x_center, y_center). Default (0.5, 0.5).
    radius : float, optional
        Circle radius. Default 0.2.
    iters : int, optional
        Number of spring-relaxation sweeps. Default 600.
    omega : float, optional
        Relaxation factor of the spring smoothing. Default 0.8.
    b : float, optional
        Decay exponent of the stiffness ``k0 / (d + 1)**b``. Default 0.5.
    k0 : float, optional
        Stiffness scale factor. Default 0.25.
    kmin : float, optional
        Lower bound on the nodal stiffness. Default 0.0 (no clipping).
    is_projection : bool, optional
        If True (default), perform the full grid adaptation. If False, only
        the cell labels are computed and returned (no grid deformation).

    Returns
    -------
    P1 : numpy.ndarray
        Only if ``is_projection`` is True. Adapted nodal coordinates with
        shape [xy, nx, ny]; interface nodes lie exactly on the circle.
    inside : numpy.ndarray
        Integer cell phase indicator with shape [nx, ny] (1 = inclusion,
        0 = matrix), evaluated on the *reference* grid. If ``is_projection``
        is False, this is the only return value.

    Notes
    -----
    The return type depends on ``is_projection`` (tuple vs. single array).
    """
    # 定義網格 geometry: inside cells
    # (Define the grid geometry: label inside cells)
    inside = cell_labels(ref_grid_coords_ixyz,center,radius)
    if is_projection:
        # 定義網格 interface nodes
        # (Define the interface nodes of the grid)
        interface = interface_node_mask(inside)

        # Snap interface nodes radially onto the exact circle

        P = ref_grid_coords_ixyz.copy()
        idx = np.argwhere(interface)
        if idx.size:
            pts = P[:, idx[:, 0], idx[:, 1]]
            P[:, idx[:, 0], idx[:, 1]] = project_points_to_circle(pts,center=center,R=radius,)
        # Distance-dependent spring stiffness: stiff near the interface, soft far away
        dist = manhattan_distance_to_interface(interface)
        k_node = stiffness_from_distance(dist, k0=k0, b=b, a=1.0, kmin=kmin)
        # Interface nodes stay fixed on the circle; all others are relaxed
        fixed = interface.copy()
        P1 = spring_relax_weighted(P, fixed_mask=fixed, k_node=k_node, iters=iters, omega=omega)

        return P1, inside
    else:
        return inside