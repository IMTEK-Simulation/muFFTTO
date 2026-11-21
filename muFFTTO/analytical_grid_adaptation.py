'''
Storage of adapted grid data into muGrid real_field containers
through the primary accessor `field.p`.

Reference:
Zecevic, M., Lebensohn, R. A., & Capolungo, L. (2026).
Achieving geometric accuracy in FFT-based micromechanical models
using conformal grid. Mechanics of Materials, 212, 105512.
https://doi.org/10.1016/j.mechmat.2025.105512

Overview
--------
Despite the first line above, the active code of this module does not store
anything into muGrid fields (that experiment survives only in the
commented-out block at the end of the file; ``muGrid`` is imported but unused
by the active functions). What the module actually provides is a
*non-periodic* reference implementation of the conformal-grid adaptation of
Zecevic et al. for a single circular inclusion in the square [-L, L]^2:

1. ``make_grid_nodes`` -- regular (N+1) x (N+1) node grid (both boundaries
   included, no periodicity).
2. ``cell_labels`` -- inside/outside label per cell from its cell center.
3. ``interface_node_mask`` -- nodes adjacent to cells of both labels.
4. ``project_points_to_circle`` -- radial snap of interface nodes onto the
   circle.
5. ``manhattan_distance_to_interface`` / ``stiffness_from_distance`` -- nodal
   spring stiffness ``k = max(k0 / (d + a)**b, kmin)`` decaying with the
   graph distance ``d`` to the interface.
6. ``spring_relax_weighted`` -- Gauss-Seidel spring (weighted Laplacian)
   smoothing of all free nodes; interface nodes and the outer boundary
   (``outer_boundary_mask``) are kept fixed (Dirichlet-type).
7. ``adapt_grid_to_circle`` -- driver returning all intermediate fields.

For the periodic variant operating on muFFTTO coordinate arrays of layout
``[xy, nx, ny]`` see ``muFFTTO/grid_adaptation_methods_Zecevic.py``.

Array conventions
-----------------
In *this* module nodal coordinates are stored node-major, ``P[i, j, xy]``
(shape ``(N+1, N+1, 2)``), with ``i`` the x-index and ``j`` the y-index
(``meshgrid(..., indexing="ij")``). Cell ``(i, j)`` is spanned by nodes
``(i, j), (i+1, j), (i, j+1), (i+1, j+1)``; cell arrays have shape
``(N, N)``. A square grid is assumed throughout.
'''

import numpy as np
from collections import deque
import muGrid

def make_grid_nodes(N: int = 64, L: float = 25.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Function that constructs a regular Cartesian node grid.

    The grid has (N+1) x (N+1) nodes that span a square domain [-L, L] x [-L, L].
    Node coordinates are stored as an array of shape [nx, ny, xy].

    Parameters
    --------
    N : int
        Number of cells along one spatial direction. The number of nodes is N + 1.
    L : float
        Half-size of the square computational domain measured in the same units as x and y.

    Returns
    -------
    [i,x,y] (x,y),nx,ny
    P : numpy ndarray
        Array of nodal coordinates with shape [nx, ny, xy] = (N+1, N+1, 2).
        - nx = N + 1 is the number of nodes in x-direction.
        - ny = N + 1 is the number of nodes in y-direction.
        - the last index 'xy' holds (x, y) coordinates at each node.
    x : numpy ndarray
        One-dimensional array of x-coordinates with shape [nx].
    y : numpy ndarray
        One-dimensional array of y-coordinates with shape [ny].

    Notes
    -----
    Both boundaries x = -L and x = +L are included as separate nodes, i.e. the
    grid is not periodic; the node spacing is 2 L / N.
    """

    x = np.linspace(-L, L, N+1)
    y = np.linspace(-L, L, N+1)
    X, Y = np.meshgrid(x, y, indexing="ij")
    P = np.stack([X, Y], axis=-1)
   
    # Legacy/alternative implementation kept for reference: periodic grid on
    # [0, L) with N nodes per direction in muFFTTO layout [xy, nx, ny].
    '''
    # (0,L)
    x = np.linspace(0, L, N, endpoint=False )
    y = np.linspace(0, L, N, endpoint=False )
    X, Y = np.meshgrid(x, y, indexing="ij")
    P = np.stack([X, Y])
    '''
    return P, x, y


# specifically for circle inclusion, but can be adapted for other shapes by changing the labeling logic
def cell_labels(
    P: np.ndarray,
    center: tuple[float, float] = (0.0, 0.0),
    R: float = 20.0,
) -> np.ndarray:
    """
    Function that labels grid cells as inside or outside a circular inclusion.

    The label is decided from the position of cell centers with respect to a circle.
    A value of one means inside the circle, zero means outside.

    Parameters
    ----------
    P : numpy ndarray
        Nodal coordinates with shape [nx, ny, xy] as returned by make_grid_nodes.
        - nx, ny are nodal counts in x- and y-direction.
        - xy index stores (x, y) coordinates per node.
    center : tuple of float
        Coordinates of the circle center (cx, cy).
    R : float
        Radius of the inclusion circle.

    Returns
    -------
    inside : numpy ndarray
        Integer array of cell labels with shape [cell_x, cell_y].
        - inside[i, j] = 1 if the cell center lies inside the circle.
        - inside[i, j] = 0 otherwise.
    """
    cx, cy = center
    # cell centers from average of four surrounding nodes
    Pc = 0.25 * (P[:-1, :-1] + P[1:, :-1] + P[:-1, 1:] + P[1:, 1:])
    dx = Pc[..., 0] - cx
    dy = Pc[..., 1] - cy
    inside_mask = (dx * dx + dy * dy) <= R * R
    return inside_mask.astype(np.int32)


def interface_node_mask(cell_inside: np.ndarray) -> np.ndarray:
    """
    Function that detects interface nodes from neighboring cell labels.

    A node is classified as interface node if the adjacent cells contain both inside and outside labels.

    Parameters
    ----------
    cell_inside : numpy ndarray
        Cell labels with shape [cell_x, cell_y] created by cell_labels.
        - value 1 corresponds to cells inside the circle.
        - value 0 corresponds to cells outside the circle.

    Returns
    -------
    interface_mask : numpy ndarray
        Boolean nodal mask with shape [nx, ny].
        - True indicates that the node lies on the interface between phases.
        - False indicates a node far from the interface.
    """
    N = cell_inside.shape[0]
    mask = np.zeros((N + 1, N + 1), dtype=bool)

    for i in range(N + 1):
        for j in range(N + 1):
            # Collect labels of the (up to four) cells sharing node (i, j):
            # (i-1, j-1), (i, j-1), (i-1, j), (i, j). Boundary nodes have only
            # one or two adjacent cells (no periodic wrap in this module).
            values = []
            if i > 0 and j > 0:
                values.append(cell_inside[i - 1, j - 1])
            if i < N and j > 0:
                values.append(cell_inside[i, j - 1])
            if i > 0 and j < N:
                values.append(cell_inside[i - 1, j])
            if i < N and j < N:
                values.append(cell_inside[i, j])
            if not values:
                continue
            vals = np.asarray(values, dtype=int)
            if vals.size >= 2 and vals.min() != vals.max():
                mask[i, j] = True

    return mask


def outer_boundary_mask(N: int) -> np.ndarray:
    """
    Function that marks the outer square boundary nodes as fixed.

    The boundary corresponds to the four edges of the structured grid.
    The mask can be used to impose Dirichlet-type constraints in relaxation.

    Parameters
    ----------
    N : int
        Number of cells along one spatial direction. The number of nodes is N + 1.

    Returns
    -------
    outer_mask : numpy ndarray
        Boolean nodal mask with shape [nx, ny].
        - True indicates a node on the outer boundary of the domain.
        - False indicates an interior node.
    """
    m = np.zeros((N + 1, N + 1), dtype=bool)
    m[0, :] = True
    m[N, :] = True
    m[:, 0] = True
    m[:, N] = True
    return m

#specifically for circle inclusion, but can be adapted for other shapes by changing the projection logic
def project_points_to_circle(
    Ppts: np.ndarray,
    center: tuple[float, float] = (0.0, 0.0),
    R: float = 20.0,
) -> np.ndarray:
    """
    Function that projects given points onto a target circle.

    Each point is shifted radially such that its distance to the center
    equals the prescribed radius R.

    Parameters
    ----------
    Ppts : numpy ndarray
        Array of point coordinates with shape [k, xy].
        - k is the number of points to project.
        - xy index stores (x, y) coordinates per point.
    center : tuple of float
        Coordinates (cx, cy) of the circle center.
    R : float
        Target radius of the circle.

    Returns
    -------
    Pproj : numpy ndarray
        Array of projected point coordinates with shape [k, xy].
        - all returned points satisfy the distance ||Pproj - center|| = R within numerical tolerance.
    """
    c = np.asarray(center, dtype=float)
    # Radial vector from the center; rescale it to length R.
    V = Ppts - c[None, :]
    r = np.linalg.norm(V, axis=1, keepdims=True)
    r = np.maximum(r, 1e-12)  # guard against division by zero at the center
    return c[None, :] + (R / r) * V


def manhattan_distance_to_interface(interface_mask: np.ndarray) -> np.ndarray:
    """
    Function that computes Manhattan distance to the nearest interface node.

    The distance is measured on the grid graph using unit steps in the
    four von Neumann directions (up, down, left, right).

    Parameters
    ----------
    interface_mask : numpy ndarray
        Boolean nodal mask with shape [nx, ny].
        - True marks nodes that belong to the interface.
        - False marks other nodes.

    Returns
    -------
    dist : numpy ndarray
        Floating-point distance field with shape [nx, ny].
        - dist[i, j] = 0 for interface nodes.
        - dist[i, j] is the minimum number of grid steps from node (i, j) to the interface.
    """
    # Multi-source BFS from all interface nodes on the (non-periodic)
    # 4-neighbour node graph; unit edge weights => exact graph distance.
    # Nodes not reachable (only if there is no interface at all) stay at inf.
    N = interface_mask.shape[0] - 1
    dist = np.full((N + 1, N + 1), np.inf, dtype=float)
    queue = deque()

    for i in range(N + 1):
        for j in range(N + 1):
            if interface_mask[i, j]:
                dist[i, j] = 0.0
                queue.append((i, j))

    neighbors = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    while queue:
        i, j = queue.popleft()
        for di, dj in neighbors:
            ii, jj = i + di, j + dj
            if 0 <= ii <= N and 0 <= jj <= N:
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
    Function that computes nodal stiffness from distance to the interface.

    The stiffness follows a decay law k(d) = k0 / (d + a)^b and is bounded from below
    by the stiffness floor kmin.

    Parameters
    ----------
    dist : numpy ndarray
        Manhattan distance field with shape [nx, ny].
        - dist[i, j] is the distance of node (i, j) to the interface.
    k0 : float
        Baseline stiffness scale factor.
    b : float
        Exponent that controls how fast stiffness decays away from the interface.
    a : float
        Positive offset that regularizes the stiffness at zero distance.
    kmin : float
        Minimum stiffness floor applied elementwise.

    Returns
    -------
    k_node : numpy ndarray
        Nodal stiffness field with shape [nx, ny].
        - k_node[i, j] is the effective stiffness assigned to node (i, j).
        - values are guaranteed to be >= kmin.
    """
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
    Function that performs weighted spring-based relaxation of nodal positions.

    The update is a Gauss–Seidel type iteration that moves each free node
    toward the weighted average of its four neighbors, where edge weights
    are computed from nodal stiffness values.

    Parameters
    ----------
    P : numpy ndarray
        Nodal coordinates before relaxation with shape [nx, ny, xy].
    fixed_mask : numpy ndarray
        Boolean nodal mask with shape [nx, ny] that marks fixed nodes.
        - True indicates that the node is held fixed during the update.
    k_node : numpy ndarray
        Nodal stiffness field with shape [nx, ny].
        - stiffness values control how strongly a node couples to its neighbors.
    iters : int
        Number of Gauss–Seidel sweeps over all interior nodes.
    omega : float
        Relaxation parameter. Values smaller than one yield under-relaxation.

    Returns
    -------
    P_relaxed : numpy ndarray
        Nodal coordinates after relaxation with shape [nx, ny, xy].
        - fixed nodes remain unchanged.
        - free nodes are updated to a spring-equilibrium configuration.

    Notes
    -----
    Update of a free node ``p_ij`` with neighbours ``n`` (left/right/down/up):
    ``p_ij <- (1 - omega) p_ij + omega * sum_n w_n p_n / sum_n w_n`` with edge
    weight ``w_n = (k_ij + k_n) / 2``. Only interior nodes
    (``1 <= i, j <= N-1``) are visited, so the outer boundary nodes are never
    moved regardless of ``fixed_mask``.
    """
    P_new = P.copy()
    N = P.shape[0] - 1

    for _ in range(iters):
        for i in range(1, N):
            for j in range(1, N):
                if fixed_mask[i, j]:
                    continue

                neighbors = [(i - 1, j), (i + 1, j), (i, j - 1), (i, j + 1)]
                kij = k_node[i, j]
                w = np.empty(4, dtype=float)
                pts = np.empty((4, 2), dtype=float)

                for idx, (ii, jj) in enumerate(neighbors):
                    w[idx] = 0.5 * (kij + k_node[ii, jj])
                    pts[idx] = P_new[ii, jj]

                # Spring equilibrium position (weighted neighbour average),
                # blended with the old position by the relaxation factor omega.
                target = (w[:, None] * pts).sum(axis=0) / (w.sum() + 1e-15)
                P_new[i, j] = (1.0 - omega) * P_new[i, j] + omega * target

    return P_new


def adapt_grid_to_circle(
    N: int = 64,
    L: float = 40.0,
    center: tuple[float, float] = (0.0, 0.0),
    R: float = 20.0,
    iters: int = 600,
    omega: float = 0.8,
    b: float = 1.0,
    k0: float = 0.25,
    kmin: float = 0.005,
) -> dict:
    """
    Function that adapts a structured grid to a circular inclusion.

    The algorithm performs cell labeling, interface detection, projection
    of interface nodes onto the circle and weighted spring relaxation of
    interior nodes.

    Parameters
    ----------
    N : int
        Number of cells along one spatial direction.
    L : float
        Half-size of the square computational domain.
    center : tuple of float
        Coordinates (cx, cy) of the circle center.
    R : float
        Radius of the inclusion circle.
    iters : int
        Number of relaxation sweeps in spring_relax_weighted.
    omega : float
        Relaxation parameter for spring_relax_weighted.
    b : float
        Exponent of the distance-based stiffness decay law.
    k0 : float
        Baseline stiffness parameter.
    kmin : float
        Minimum stiffness floor.

    Returns
    -------
    result : dict
        Dictionary that collects all relevant fields:
        - 'P0' : initial nodal coordinates [nx, ny, xy]
        - 'P' : adapted nodal coordinates [nx, ny, xy]
        - 'inside' : cell labels [cell_x, cell_y]
        - 'interface' : interface nodal mask [nx, ny]
        - 'outer' : outer boundary mask [nx, ny]
        - 'fixed' : fixed nodal mask [nx, ny]
        - 'dist' : Manhattan distance field [nx, ny]
        - 'k_node' : nodal stiffness values [nx, ny]
        - 'params' : dictionary of scalar parameters used in the run.
    """
    # Reference regular grid and its phase/interface classification
    P0, _, _ = make_grid_nodes(N=N, L=L)
    inside = cell_labels(P0, center=center, R=R)
    interface = interface_node_mask(inside)
    outer = outer_boundary_mask(N)
    # Nodes that are not moved by the relaxation: interface + outer boundary
    fixed = outer | interface

    # Snap the staircase interface nodes onto the exact circle
    P1 = P0.copy()
    idx = np.argwhere(interface)
    if idx.size:
        P_if = P1[idx[:, 0], idx[:, 1], :]
        P1[idx[:, 0], idx[:, 1], :] = project_points_to_circle(
            P_if,
            center=center,
            R=R,
        )

    # Stiff springs near the interface, soft far away -> distortion is spread
    # mainly into the region away from the inclusion boundary
    dist = manhattan_distance_to_interface(interface)
    k_node = stiffness_from_distance(dist, k0=k0, b=b, a=1.0, kmin=kmin)
    P2 = spring_relax_weighted(P1, fixed_mask=fixed, k_node=k_node, iters=iters, omega=omega)

    return {
        "P0": P0,
        "P": P2,
        "inside": inside,
        "interface": interface,
        "outer": outer,
        "fixed": fixed,
        "dist": dist,
        "k_node": k_node,
        "params": {
            "N": N,
            "L": L,
            "center": center,
            "R": R,
            "iters": iters,
            "omega": omega,
            "b": b,
            "k0": k0,
            "kmin": kmin,
        },
    }



# Legacy/experimental code kept for reference: test of storing grid data in a
# muGrid CartesianDecomposition real_field (not executed).
'''
from muGrid import Communicator, CartesianDecomposition 
import numpy as np

if __name__ == "__main__":
    coor,x,z = make_grid_nodes(N=4, L=1.0)
    print(coor[0,0])
    print(coor[1,0])

    #print(muGrid.__file__)
    #print([x for x in dir(muGrid) if not x.startswith("_")])
    
    nx = 8
    ny = 8
    g = 1
    print("before communicator")
    comm = Communicator()
    print("after communicator")
    print("before decomp")

    #help(muGrid.CartesianDecomposition)vu0 
    decomp = muGrid.CartesianDecomposition(
        communicator=comm,
        nb_domain_grid_pts=(nx, ny),
        nb_subdivisions=(1, 1),
        nb_ghosts_left=(g, g),
        nb_ghosts_right=(g, g),
    )
    print("after decomp")
    
    field = decomp.real_field("displacement", components=(3,))
    print(comm)
    print(field)
    print(type(field))
    print([x for x in dir(field) if not x.startswith("_")])
    
'''