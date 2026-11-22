"""
Periodic grid adaptation using the spring-relaxation method of Zecevic et al.

    Zecevic, M., Lebensohn, R. A., & Capolungo, L.
    "Achieving geometric accuracy in FFT-based micromechanical models using
    conformal grids." Los Alamos National Laboratory, Los Alamos, NM, USA.

This module contains
    1. shared periodic utilities used by both grid-adaptation workflows
       (geometry, minimum image, interface nodes, stiffness, spring relaxation,
       invalid-node filtering, cell-inversion check, plotting coordinates),
    2. WORKFLOW A: grid adaptation to a circular inclusion.

The arbitrary-image workflow (WORKFLOW B) lives in
``muFFTTO.grid_adaptation_methods_Zecevic_arbitrary`` and imports the shared
utilities from here.

Conventions
-----------
* Nodal coordinates: shape (2, Nx, Ny), indexing [xy, ix, iy], indexing="ij".
* Only the Nx x Ny stored nodes of the periodic grid are kept; node Nx is the
  periodic image of node 0 shifted by Lx (same in y).
* Coarse cell (i, j) is spanned by nodes (i, j), (i+1, j), (i+1, j+1), (i, j+1),
  indices taken modulo (Nx, Ny).
* The reference grid is uniform and starts at (0, 0), exactly like
  ``Discretization.get_nodal_points_coordinates()``.
* The periodic lengths Lx, Ly are ALWAYS taken from the reference grid or the
  user-given domain size, never from deformed coordinates.
* Stored deformed coordinates are never wrapped back into the domain;
  periodic distances/displacements are computed instead.

Error handling
--------------
This module never prints and never emits warnings. Invalid input raises
``ValueError`` with the name of the function in the message. Non-fatal
information is returned in a ``diagnostics`` dictionary.
"""
from __future__ import annotations

from collections import deque

import numpy as np

SPRING_METHODS = ("gauss_seidel", "jacobi_displacement")


# =============================================================
# SHARED PERIODIC UTILITIES
# =============================================================

def periodic_grid_geometry(
        ref_grid_coords_ixyz: np.ndarray,
        domain_size: tuple[float, float] | None = None,
) -> tuple[int, int, float, float, float, float]:
    """
    Read and validate the geometry of the periodic REFERENCE grid.

    Parameters
    ----------
    ref_grid_coords_ixyz : numpy.ndarray
        Reference nodal coordinates, shape (2, Nx, Ny). Must be uniform,
        start at (0, 0) and exclude the periodic end nodes.
    domain_size : tuple[float, float] or None
        Optional (Lx, Ly). If given, it must agree with the reference grid.

    Returns
    -------
    (Nx, Ny, dx, dy, Lx, Ly) : tuple
        Node counts, grid spacings and periodic domain lengths.
    """
    name = "periodic_grid_geometry"
    P = np.asarray(ref_grid_coords_ixyz, dtype=float)
    if P.ndim != 3 or P.shape[0] != 2:
        raise ValueError(
            f"[{name}] reference coordinates must have shape (2, Nx, Ny), got {P.shape}"
        )
    Nx, Ny = P.shape[1:]
    if Nx < 3 or Ny < 3:
        raise ValueError(f"[{name}] need at least 3 nodes per direction, got ({Nx}, {Ny})")
    if not np.all(np.isfinite(P)):
        raise ValueError(f"[{name}] reference coordinates contain NaN or inf")

    dx = float(P[0, 1, 0] - P[0, 0, 0])
    dy = float(P[1, 0, 1] - P[1, 0, 0])
    if dx <= 0 or dy <= 0:
        raise ValueError(f"[{name}] reference spacings must be positive, got dx={dx}, dy={dy}")
    Lx, Ly = Nx * dx, Ny * dy

    if domain_size is not None:
        Lx_given, Ly_given = map(float, domain_size)
        if Lx_given <= 0 or Ly_given <= 0:
            raise ValueError(f"[{name}] domain_size must be positive, got {domain_size}")
        if not (np.isclose(Lx, Lx_given, rtol=1e-8) and np.isclose(Ly, Ly_given, rtol=1e-8)):
            raise ValueError(
                f"[{name}] reference grid spans ({Lx}, {Ly}) but domain_size is {domain_size}. "
                "Use discretization.get_nodal_points_coordinates() (physical coordinates), "
                "not discretization.fft.coords (fractional coordinates)."
            )
        Lx, Ly = Lx_given, Ly_given
        dx, dy = Lx / Nx, Ly / Ny

    X, Y = np.meshgrid(np.arange(Nx) * dx, np.arange(Ny) * dy, indexing="ij")
    atol = 1e-9 * max(Lx, Ly)
    if not (np.allclose(P[0], X, atol=atol) and np.allclose(P[1], Y, atol=atol)):
        raise ValueError(
            f"[{name}] reference grid must be uniform, start at (0, 0), use indexing='ij' "
            "and exclude the periodic end nodes"
        )
    return Nx, Ny, dx, dy, Lx, Ly


def periodic_minimum_image(delta: np.ndarray, Lx: float, Ly: float) -> np.ndarray:
    """
    Apply the minimum-image convention to displacement vectors.

    Parameters
    ----------
    delta : numpy.ndarray
        Raw displacement vectors, shape (2, ...); delta[0] is x, delta[1] is y.
    Lx, Ly : float
        Fixed periodic domain lengths.

    Returns
    -------
    numpy.ndarray
        Displacements of the same shape with components in [-L/2, L/2].
    """
    if Lx <= 0 or Ly <= 0:
        raise ValueError(f"[periodic_minimum_image] Lx and Ly must be positive, got ({Lx}, {Ly})")
    out = np.array(delta, dtype=float, copy=True)
    if out.shape[0] != 2:
        raise ValueError(f"[periodic_minimum_image] delta must have shape (2, ...), got {out.shape}")
    # PERIODIC: use minimum-image distance
    out[0] -= Lx * np.round(out[0] / Lx)
    out[1] -= Ly * np.round(out[1] / Ly)
    return out


def interface_nodes_from_cells(cell_mask: np.ndarray) -> np.ndarray:
    """
    Mark every node that is a corner of at least one marked cell.

    Node (i, j) is the shared corner of cells (i-1, j-1), (i-1, j), (i, j-1)
    and (i, j), all indices taken periodically.

    Parameters
    ----------
    cell_mask : numpy.ndarray
        Boolean cell mask, shape (Nx, Ny).

    Returns
    -------
    numpy.ndarray
        Boolean node mask, shape (Nx, Ny).
    """
    m = np.asarray(cell_mask, dtype=bool)
    if m.ndim != 2:
        raise ValueError(f"[interface_nodes_from_cells] cell_mask must be 2D, got shape {m.shape}")
    # PERIODIC: np.roll wraps cell indices across the domain boundary
    m_im1 = np.roll(m, 1, axis=0)
    m_jm1 = np.roll(m, 1, axis=1)
    m_im1_jm1 = np.roll(m_im1, 1, axis=1)
    return m | m_im1 | m_jm1 | m_im1_jm1


def manhattan_distance_to_interface(interface_mask: np.ndarray) -> np.ndarray:
    """
    Periodic four-neighbour graph distance to the nearest interface node.

    Parameters
    ----------
    interface_mask : numpy.ndarray
        Boolean node mask, shape (Nx, Ny). True marks the (fixed) interface nodes.

    Returns
    -------
    numpy.ndarray
        Float array, shape (Nx, Ny), distance in number of node steps.
    """
    name = "manhattan_distance_to_interface"
    mask = np.asarray(interface_mask, dtype=bool)
    if mask.ndim != 2:
        raise ValueError(f"[{name}] interface_mask must be 2D, got shape {mask.shape}")
    if not np.any(mask):
        raise ValueError(f"[{name}] interface_mask contains no interface node")
    Nx, Ny = mask.shape
    dist = np.full((Nx, Ny), np.inf)
    dist[mask] = 0.0
    queue = deque(map(tuple, np.argwhere(mask)))
    while queue:
        i, j = queue.popleft()
        d_next = dist[i, j] + 1.0
        # PERIODIC: neighbours wrap around the domain boundary
        for ii, jj in (((i - 1) % Nx, j), ((i + 1) % Nx, j),
                       (i, (j - 1) % Ny), (i, (j + 1) % Ny)):
            if dist[ii, jj] > d_next:
                dist[ii, jj] = d_next
                queue.append((ii, jj))
    return dist


def stiffness_from_distance(
        dist: np.ndarray,
        k0: float = 0.25,
        b: float = 0.5,
        a: float = 1.0,
        kmin: float = 0.0,
) -> np.ndarray:
    """
    Nodal spring stiffness k = max(k0 / (dist + a)**b, kmin).

    Parameters
    ----------
    dist : numpy.ndarray
        Distance to the interface, shape (Nx, Ny).
    k0 : float
        Stiffness of the springs closest to the interface.
    b : float
        Decay exponent; larger b gives softer springs far from the interface.
    a : float
        Positive distance shift (avoids division by zero).
    kmin : float
        Lower bound of the stiffness.

    Returns
    -------
    numpy.ndarray
        Strictly positive stiffness, shape (Nx, Ny).
    """
    name = "stiffness_from_distance"
    if k0 <= 0 or a <= 0 or b < 0 or kmin < 0:
        raise ValueError(
            f"[{name}] require k0 > 0, a > 0, b >= 0, kmin >= 0; "
            f"got k0={k0}, a={a}, b={b}, kmin={kmin}"
        )
    k = np.maximum(k0 / np.power(np.asarray(dist, dtype=float) + a, b), kmin)
    if not np.all(np.isfinite(k)) or np.any(k <= 0):
        raise ValueError(f"[{name}] stiffness must be finite and strictly positive")
    return k


def periodic_cell_corner_jacobians(
        P: np.ndarray,
        P_ref: np.ndarray,
        Lx: float,
        Ly: float,
) -> tuple[np.ndarray, list[tuple[int, int]]]:
    """
    Corner Jacobian determinants of every bilinear coarse cell.

    The corners of cell (i, j) are unwrapped as
    reference corner + periodic nodal displacement, so cells crossing the
    periodic boundary are treated exactly like interior cells. A cell is
    inverted (or degenerate) if any corner determinant is <= 0.

    Parameters
    ----------
    P : numpy.ndarray
        Deformed nodal coordinates, shape (2, Nx, Ny).
    P_ref : numpy.ndarray
        Reference nodal coordinates, shape (2, Nx, Ny).
    Lx, Ly : float
        Fixed periodic domain lengths.

    Returns
    -------
    corner_det : numpy.ndarray
        Shape (4, Nx, Ny); corners ordered p00, p10, p11, p01.
    inverted_cells : list[tuple[int, int]]
        Cells (i, j) with min(corner_det) <= 0.
    """
    name = "periodic_cell_corner_jacobians"
    P = np.asarray(P, dtype=float)
    P_ref = np.asarray(P_ref, dtype=float)
    if P.shape != P_ref.shape or P.ndim != 3 or P.shape[0] != 2:
        raise ValueError(f"[{name}] P and P_ref must both have shape (2, Nx, Ny); got {P.shape}, {P_ref.shape}")
    Nx, Ny = P.shape[1:]
    dx, dy = Lx / Nx, Ly / Ny
    u = P - P_ref
    shift_x = np.array([dx, 0.0])[:, None, None]
    shift_y = np.array([0.0, dy])[:, None, None]
    # PERIODIC: neighbour displacement via np.roll, reference corner unwrapped by +dx / +dy
    p00 = P_ref + u
    p10 = P_ref + shift_x + np.roll(u, -1, axis=1)
    p01 = P_ref + shift_y + np.roll(u, -1, axis=2)
    p11 = P_ref + shift_x + shift_y + np.roll(np.roll(u, -1, axis=1), -1, axis=2)

    def cross(a, b):
        return a[0] * b[1] - a[1] * b[0]

    corner_det = np.stack((
        cross(p10 - p00, p01 - p00),
        cross(p11 - p10, p00 - p10),
        cross(p01 - p11, p10 - p11),
        cross(p00 - p01, p11 - p01),
    ))
    inverted = np.argwhere(corner_det.min(axis=0) <= 0.0)
    return corner_det, [(int(i), int(j)) for i, j in inverted]


def periodic_cell_quadrature_det_F(
        P: np.ndarray,
        P_ref: np.ndarray,
        Lx: float,
        Ly: float,
) -> tuple[np.ndarray, list[tuple[int, int]]]:
    """
    det(F) of the grid deformation at the quadrature points of every cell.

    Uses the same rule as the solver element ``bilinear_rectangle``
    (``Element.bilinear_quad`` in ``discretization_library``): 2x2
    Gauss-Legendre points xi, eta = +-1/sqrt(3) on the reference square
    [-1, 1]^2, ordered with xi running fastest. det(F) is the ratio of the
    deformed to the reference Jacobian determinant, i.e. the value the solver
    computes as det(I + grad u) at the same points.

    Parameters
    ----------
    P : numpy.ndarray
        Deformed nodal coordinates, shape (2, Nx, Ny).
    P_ref : numpy.ndarray
        Reference nodal coordinates, shape (2, Nx, Ny).
    Lx, Ly : float
        Fixed periodic domain lengths.

    Returns
    -------
    det_F : numpy.ndarray
        Shape (4, Nx, Ny); quadrature points ordered (xi, eta) =
        (-g, -g), (+g, -g), (-g, +g), (+g, +g) with g = 1/sqrt(3).
    inverted_cells : list[tuple[int, int]]
        Cells (i, j) with det(F) <= 0 at any quadrature point.
    """
    name = "periodic_cell_quadrature_det_F"
    P = np.asarray(P, dtype=float)
    P_ref = np.asarray(P_ref, dtype=float)
    if P.shape != P_ref.shape or P.ndim != 3 or P.shape[0] != 2:
        raise ValueError(f"[{name}] P and P_ref must both have shape (2, Nx, Ny); got {P.shape}, {P_ref.shape}")
    Nx, Ny = P.shape[1:]
    dx, dy = Lx / Nx, Ly / Ny
    u = P - P_ref
    shift_x = np.array([dx, 0.0])[:, None, None]
    shift_y = np.array([0.0, dy])[:, None, None]
    # PERIODIC: neighbour displacement via np.roll, reference corner unwrapped by +dx / +dy
    p00 = P_ref + u
    p10 = P_ref + shift_x + np.roll(u, -1, axis=1)
    p01 = P_ref + shift_y + np.roll(u, -1, axis=2)
    p11 = P_ref + shift_x + shift_y + np.roll(np.roll(u, -1, axis=1), -1, axis=2)

    g = 1.0 / np.sqrt(3.0)
    det_ref = 0.25 * dx * dy
    det_F = []
    for eta in (-g, g):
        for xi in (-g, g):
            dx_dxi = 0.25 * ((1.0 - eta) * (p10 - p00) + (1.0 + eta) * (p11 - p01))
            dx_deta = 0.25 * ((1.0 - xi) * (p01 - p00) + (1.0 + xi) * (p11 - p10))
            det_F.append((dx_dxi[0] * dx_deta[1] - dx_dxi[1] * dx_deta[0]) / det_ref)
    det_F = np.stack(det_F)
    inverted = np.argwhere(det_F.min(axis=0) <= 0.0)
    return det_F, [(int(i), int(j)) for i, j in inverted]


INVERSION_CHECKS = ("quadrature", "corner", "off")


def inverted_cells(P, P_ref, Lx, Ly, inversion_check="quadrature"):
    """
    List of inverted cells (i, j) under the chosen criterion.

    "quadrature": det(F) <= 0 at a solver quadrature point (same as the solver).
    "corner"    : Jacobian <= 0 at a cell corner (stricter; also flags concave cells).
    "off"       : evaluated like "quadrature" (used for reporting only).
    """
    if inversion_check not in INVERSION_CHECKS:
        raise ValueError(f"[inverted_cells] inversion_check must be one of {INVERSION_CHECKS}, got '{inversion_check}'")
    if inversion_check == "corner":
        return periodic_cell_corner_jacobians(P, P_ref, Lx, Ly)[1]
    return periodic_cell_quadrature_det_F(P, P_ref, Lx, Ly)[1]


def remove_invalid_projected_nodes(
        P_projected: np.ndarray,
        P_ref: np.ndarray,
        fixed_mask: np.ndarray,
        Lx: float,
        Ly: float,
        duplicate_tol: float | None = None,
        max_passes: int = 20,
        inversion_check: str = "quadrature",
) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    Remove invalid projected nodes BEFORE spring relaxation.

    1. Duplicates: two neighbouring projected nodes (edge or cell diagonal)
       closer than ``duplicate_tol`` (periodic distance) were projected onto
       the same point. The node with the shorter projection distance is kept,
       the other one is reset to its reference position and released.
    2. Inversion: for every inverted cell, the fixed corner node with the
       largest projection displacement is reset and released. Repeated until
       no cell is inverted or ``max_passes`` is reached.

    ``inversion_check`` selects the inversion criterion:
        "quadrature" : det(F) <= 0 at a solver quadrature point (default,
                       identical to the det(F) computed by the solver),
        "corner"     : Jacobian <= 0 at a cell corner (stricter; also releases
                       nodes of concave cells that the solver accepts),
        "off"        : no filtering at all (neither duplicates nor inversion);
                       equivalent to the original Zecevic implementation.

    Released nodes become free nodes of the spring relaxation.

    Parameters
    ----------
    P_projected : numpy.ndarray
        Projected nodal coordinates, shape (2, Nx, Ny).
    P_ref : numpy.ndarray
        Reference nodal coordinates, shape (2, Nx, Ny).
    fixed_mask : numpy.ndarray
        Boolean mask of projected nodes, shape (Nx, Ny).
    Lx, Ly : float
        Fixed periodic domain lengths.
    duplicate_tol : float or None
        Duplicate distance tolerance. None -> 1e-3 * min(dx, dy).
    max_passes : int
        Maximum number of inversion-removal passes.
    inversion_check : str
        "quadrature", "corner" or "off".

    Returns
    -------
    P_clean : numpy.ndarray
        Cleaned coordinates, shape (2, Nx, Ny).
    fixed_clean : numpy.ndarray
        Cleaned boolean fixed-node mask, shape (Nx, Ny).
    info : dict
        n_duplicate_removed, n_inversion_removed, duplicate_removed_nodes,
        inversion_removed_nodes, inverted_cells_after_projection.
    """
    name = "remove_invalid_projected_nodes"
    P_ref = np.asarray(P_ref, dtype=float)
    u = np.asarray(P_projected, dtype=float) - P_ref
    fixed = np.array(fixed_mask, dtype=bool, copy=True)
    if u.ndim != 3 or u.shape[0] != 2 or fixed.shape != u.shape[1:]:
        raise ValueError(f"[{name}] incompatible shapes {u.shape} and {fixed.shape}")
    if int(max_passes) != max_passes or max_passes < 0:
        raise ValueError(f"[{name}] max_passes must be a non-negative integer, got {max_passes}")
    Nx, Ny = fixed.shape
    dx, dy = Lx / Nx, Ly / Ny
    if duplicate_tol is None:
        duplicate_tol = 1e-3 * min(dx, dy)
    if duplicate_tol < 0:
        raise ValueError(f"[{name}] duplicate_tol must be non-negative, got {duplicate_tol}")
    if inversion_check not in INVERSION_CHECKS:
        raise ValueError(f"[{name}] inversion_check must be one of {INVERSION_CHECKS}, got '{inversion_check}'")
    if inversion_check == "off":
        info = {
            "n_duplicate_removed": 0,
            "n_inversion_removed": 0,
            "duplicate_removed_nodes": [],
            "inversion_removed_nodes": [],
            "inverted_cells_after_projection": inverted_cells(P_ref + u, P_ref, Lx, Ly, "quadrature"),
        }
        return P_ref + u, fixed, info

    u_norm = np.sqrt(np.sum(u ** 2, axis=0))

    def release(i, j):
        u[:, i, j] = 0.0
        u_norm[i, j] = 0.0
        fixed[i, j] = False

    # ---- 1. duplicates (periodic distance between neighbouring nodes) ----
    duplicate_removed = []
    for si, sj in ((1, 0), (0, 1), (1, 1), (1, -1)):
        ref_offset = np.array([si * dx, sj * dy])[:, None, None]
        u_nb = np.roll(np.roll(u, -si, axis=1), -sj, axis=2)
        # PERIODIC: use minimum-image distance
        sep = periodic_minimum_image(ref_offset + u_nb - u, Lx, Ly)
        close = np.sqrt(np.sum(sep ** 2, axis=0)) < duplicate_tol
        both_fixed = fixed & np.roll(np.roll(fixed, -si, axis=0), -sj, axis=1)
        for i, j in np.argwhere(close & both_fixed):
            ii, jj = (i + si) % Nx, (j + sj) % Ny
            if not (fixed[i, j] and fixed[ii, jj]):
                continue
            drop = (i, j) if u_norm[i, j] >= u_norm[ii, jj] else (ii, jj)
            release(*drop)
            duplicate_removed.append((int(drop[0]), int(drop[1])))

    # ---- 2. inverted cells ----
    inversion_removed = []
    for _pass in range(int(max_passes)):
        inverted = inverted_cells(P_ref + u, P_ref, Lx, Ly, inversion_check)
        if not inverted:
            break
        to_release = set()
        for i, j in inverted:
            corners = [(i, j), ((i + 1) % Nx, j), ((i + 1) % Nx, (j + 1) % Ny), (i, (j + 1) % Ny)]
            fixed_corners = [c for c in corners if fixed[c]]
            if fixed_corners:
                to_release.add(max(fixed_corners, key=lambda c: u_norm[c]))
        if not to_release:
            break
        for c in sorted(to_release):
            release(*c)
            inversion_removed.append((int(c[0]), int(c[1])))
    inverted = inverted_cells(P_ref + u, P_ref, Lx, Ly, inversion_check)

    info = {
        "n_duplicate_removed": len(duplicate_removed),
        "n_inversion_removed": len(inversion_removed),
        "duplicate_removed_nodes": duplicate_removed,
        "inversion_removed_nodes": inversion_removed,
        "inverted_cells_after_projection": inverted,
    }
    return P_ref + u, fixed, info


def _relax_gauss_seidel(P, fixed, k, Lx, Ly, iters, check_fn):
    """Coordinate-based in-place Gauss-Seidel sweep (original Zecevic implementation)."""
    Nx, Ny = fixed.shape
    x = P[0].tolist()
    y = P[1].tolist()
    # Precompute free nodes, periodic neighbours, image offsets and spring weights.
    stencil = []
    for i in range(Nx):
        for j in range(Ny):
            if fixed[i, j]:
                continue
            neighbours = []
            # PERIODIC: neighbours wrap, coordinates are shifted by the fixed Lx / Ly
            for ii, jj, sx, sy in (
                    ((i - 1) % Nx, j, -Lx if i == 0 else 0.0, 0.0),
                    ((i + 1) % Nx, j, Lx if i == Nx - 1 else 0.0, 0.0),
                    (i, (j - 1) % Ny, 0.0, -Ly if j == 0 else 0.0),
                    (i, (j + 1) % Ny, 0.0, Ly if j == Ny - 1 else 0.0),
            ):
                neighbours.append((ii, jj, sx, sy, float(0.5 * (k[i, j] + k[ii, jj]))))
            w_sum = sum(n[4] for n in neighbours) + 1e-15
            stencil.append((i, j, neighbours, w_sum))

    for it in range(iters):
        for i, j, neighbours, w_sum in stencil:
            tx = 0.0
            ty = 0.0
            for ii, jj, sx, sy, w in neighbours:
                tx += w * (x[ii][jj] + sx)
                ty += w * (y[ii][jj] + sy)
            x[i][j] = tx / w_sum
            y[i][j] = ty / w_sum
        if check_fn is not None:
            check_fn(it, np.array((x, y), dtype=float))
    return np.array((x, y), dtype=float)


def _relax_jacobi_displacement(P, P_ref, fixed, k, iters, check_fn):
    """Vectorized Jacobi smoothing of the periodic displacement u = P - P_ref."""
    u = P - P_ref
    # Edge weights: w_east[i, j] couples nodes (i, j) and (i+1, j); w_north couples (i, j) and (i, j+1).
    # PERIODIC: np.roll closes the springs across the domain boundary
    w_east = 0.5 * (k + np.roll(k, -1, axis=0))
    w_north = 0.5 * (k + np.roll(k, -1, axis=1))
    w_west = np.roll(w_east, 1, axis=0)
    w_south = np.roll(w_north, 1, axis=1)
    w_sum = w_east + w_west + w_north + w_south
    for it in range(iters):
        u_new = (w_east * np.roll(u, -1, axis=1) + w_west * np.roll(u, 1, axis=1)
                 + w_north * np.roll(u, -1, axis=2) + w_south * np.roll(u, 1, axis=2)) / w_sum
        u = np.where(fixed[None], u, u_new)
        if check_fn is not None:
            check_fn(it, P_ref + u)
    return P_ref + u


def spring_relax_weighted(
        P: np.ndarray,
        P_ref: np.ndarray,
        fixed_mask: np.ndarray,
        k_node: np.ndarray,
        Lx: float,
        Ly: float,
        iters: int = 600,
        spring_method: str = "gauss_seidel",
        check_inversion_each_iter: bool = False,
) -> tuple[np.ndarray, dict]:
    """
    Periodic weighted spring relaxation with a fixed number of iterations.

    Only the projected interface nodes (``fixed_mask``) are fixed; periodic
    boundary nodes are free. Spring stiffness of an edge is the mean of its two
    nodal stiffnesses, w = 0.5 * (k_i + k_n). Full update (no under-relaxation).

    spring_method
        "gauss_seidel"        : original Zecevic implementation. Coordinates of
                                every free node are replaced in place by the
                                weighted mean of its four neighbours (periodic
                                neighbours shifted by Lx / Ly).
        "jacobi_displacement" : vectorized Jacobi smoothing of the periodic
                                displacement u = P - P_ref. Much faster; the
                                undeformed reference grid is an exact
                                equilibrium for any stiffness field.

    Parameters
    ----------
    P : numpy.ndarray
        Start coordinates (projected grid), shape (2, Nx, Ny).
    P_ref : numpy.ndarray
        Reference coordinates, shape (2, Nx, Ny).
    fixed_mask : numpy.ndarray
        Boolean mask of fixed nodes, shape (Nx, Ny).
    k_node : numpy.ndarray
        Nodal stiffness, shape (Nx, Ny), strictly positive.
    Lx, Ly : float
        Fixed periodic domain lengths.
    iters : int
        Fixed number of iterations (sweeps).
    spring_method : str
        "gauss_seidel" or "jacobi_displacement".
    check_inversion_each_iter : bool
        If True, record cells with det(F) <= 0 at a quadrature point after
        every iteration (record only, the grid is not modified).

    Returns
    -------
    P_relaxed : numpy.ndarray
        Relaxed coordinates, shape (2, Nx, Ny).
    info : dict
        spring_method, first_inverted_iter, inverted_cells_first_iter,
        inverted_cells_per_iter (list of counts, or None if not checked).
    """
    name = "spring_relax_weighted"
    P = np.asarray(P, dtype=float)
    P_ref = np.asarray(P_ref, dtype=float)
    fixed = np.asarray(fixed_mask, dtype=bool)
    k = np.asarray(k_node, dtype=float)
    if P.ndim != 3 or P.shape[0] != 2 or P_ref.shape != P.shape:
        raise ValueError(f"[{name}] P and P_ref must have shape (2, Nx, Ny); got {P.shape}, {P_ref.shape}")
    Nx, Ny = P.shape[1:]
    if fixed.shape != (Nx, Ny) or k.shape != (Nx, Ny):
        raise ValueError(f"[{name}] fixed_mask {fixed.shape} and k_node {k.shape} must have shape {(Nx, Ny)}")
    if Nx < 3 or Ny < 3 or Lx <= 0 or Ly <= 0:
        raise ValueError(f"[{name}] need >= 3 nodes per direction and positive Lx, Ly")
    if isinstance(iters, bool) or int(iters) != iters or iters < 0:
        raise ValueError(f"[{name}] iters must be a non-negative integer, got {iters}")
    if not np.any(fixed):
        raise ValueError(f"[{name}] at least one fixed node is required")
    if not np.all(np.isfinite(k)) or np.any(k <= 0):
        raise ValueError(f"[{name}] k_node must be finite and strictly positive")
    if spring_method not in SPRING_METHODS:
        raise ValueError(f"[{name}] spring_method must be one of {SPRING_METHODS}, got '{spring_method}'")

    info = {
        "spring_method": spring_method,
        "first_inverted_iter": None,
        "inverted_cells_first_iter": None,
        "inverted_cells_per_iter": [] if check_inversion_each_iter else None,
    }

    check_fn = None
    if check_inversion_each_iter:
        def check_fn(it, P_now):
            # same criterion as the solver: det(F) at the quadrature points
            _, inverted = periodic_cell_quadrature_det_F(P_now, P_ref, Lx, Ly)
            info["inverted_cells_per_iter"].append(len(inverted))
            if inverted and info["first_inverted_iter"] is None:
                info["first_inverted_iter"] = it
                info["inverted_cells_first_iter"] = inverted

    if spring_method == "gauss_seidel":
        P_relaxed = _relax_gauss_seidel(P, fixed, k, float(Lx), float(Ly), int(iters), check_fn)
    else:
        P_relaxed = _relax_jacobi_displacement(P, P_ref, fixed, k, int(iters), check_fn)
    return P_relaxed, info


def build_periodic_plot_coordinates(
        def_coords: np.ndarray,
        ref_coords: np.ndarray,
        domain_size: tuple[float, float],
) -> np.ndarray:
    """
    Node coordinates for PLOTTING ONLY, including the periodic end nodes.

    The displacement of the left / bottom boundary nodes is copied to the
    right / top boundary nodes and to the corner, then added to a
    (Nx+1) x (Ny+1) reference grid scaled to the physical domain size.
    These coordinates must not be passed to the solver.

    Parameters
    ----------
    def_coords : numpy.ndarray
        Deformed nodal coordinates used by the solver, shape (2, Nx, Ny).
    ref_coords : numpy.ndarray
        Reference nodal coordinates, shape (2, Nx, Ny).
    domain_size : tuple[float, float]
        Physical domain size (Lx, Ly).

    Returns
    -------
    numpy.ndarray
        Plot coordinates, shape (2, Nx+1, Ny+1).
    """
    Nx, Ny, _, _, Lx, Ly = periodic_grid_geometry(ref_coords, domain_size)
    def_coords = np.asarray(def_coords, dtype=float)
    if def_coords.shape != (2, Nx, Ny):
        raise ValueError(
            f"[build_periodic_plot_coordinates] def_coords must have shape {(2, Nx, Ny)}, got {def_coords.shape}"
        )
    u = def_coords - np.asarray(ref_coords, dtype=float)
    # PERIODIC: copy left/bottom boundary displacement to right/top boundary and corner
    u_periodic = np.zeros((2, Nx + 1, Ny + 1))
    u_periodic[:, :-1, :-1] = u
    u_periodic[:, -1, :-1] = u[:, 0, :]
    u_periodic[:, :-1, -1] = u[:, :, 0]
    u_periodic[:, -1, -1] = u[:, 0, 0]
    x_ref = np.stack(np.meshgrid(np.linspace(0.0, Lx, Nx + 1),
                                 np.linspace(0.0, Ly, Ny + 1), indexing="ij"))
    return x_ref + u_periodic


def _validate_spring_parameters(name, iters, b, k0, kmin, spring_method):
    """Shared validation of the spring-relaxation parameters."""
    if isinstance(iters, bool) or int(iters) != iters or iters < 0:
        raise ValueError(f"[{name}] iters must be a non-negative integer, got {iters}")
    if k0 <= 0 or kmin < 0 or b < 0:
        raise ValueError(f"[{name}] require k0 > 0, kmin >= 0, b >= 0; got k0={k0}, kmin={kmin}, b={b}")
    if spring_method not in SPRING_METHODS:
        raise ValueError(f"[{name}] spring_method must be one of {SPRING_METHODS}, got '{spring_method}'")


def _relax_and_collect(name, P_ref, P_projected, fixed, Lx, Ly, iters, b, k0, kmin,
                       if_spring_relaxation, spring_method, check_inversion_each_iter,
                       inversion_check):
    """Shared tail of both workflows: invalid-node removal, stiffness, relaxation, diagnostics."""
    P_clean, fixed_clean, removal_info = remove_invalid_projected_nodes(
        P_projected, P_ref, fixed, Lx, Ly, inversion_check=inversion_check)
    if not np.any(fixed_clean):
        raise ValueError(f"[{name}] all projected nodes were removed as invalid; nothing to fix")

    relax_info = {"spring_method": None, "first_inverted_iter": None,
                  "inverted_cells_first_iter": None, "inverted_cells_per_iter": None}
    if if_spring_relaxation:
        dist = manhattan_distance_to_interface(fixed_clean)
        k_node = stiffness_from_distance(dist, k0=k0, b=b, a=1.0, kmin=kmin)
        P_final, relax_info = spring_relax_weighted(
            P_clean, P_ref, fixed_clean, k_node, Lx, Ly, iters=int(iters),
            spring_method=spring_method, check_inversion_each_iter=check_inversion_each_iter)
    else:
        P_final = P_clean.copy()

    det_F_final, inverted_final = periodic_cell_quadrature_det_F(P_final, P_ref, Lx, Ly)
    _, concave_final = periodic_cell_corner_jacobians(P_final, P_ref, Lx, Ly)
    diagnostics = {
        "inversion_check": inversion_check,
        "n_fixed_nodes": int(np.count_nonzero(fixed_clean)),
        **removal_info,
        "if_spring_relaxation": bool(if_spring_relaxation),
        **relax_info,
        "min_det_F_quadrature": float(det_F_final.min()),
        "inverted_cells_after_relaxation": inverted_final,
        "concave_cells_after_relaxation": concave_final,
    }
    return P_clean, P_final, fixed_clean, diagnostics


# =============================================================
# WORKFLOW A: CIRCULAR INCLUSION
# =============================================================

def cell_labels(
        ref_grid_coords_ixyz: np.ndarray,
        center: tuple[float, float] = (0.5, 0.5),
        radius: float = 0.2,
) -> np.ndarray:
    """
    Classify reference cells as inside (1) / outside (0) of the circle.

    A cell is inside if the periodic distance of its centre to the circle
    centre is <= radius.

    Parameters
    ----------
    ref_grid_coords_ixyz : numpy.ndarray
        Reference nodal coordinates, shape (2, Nx, Ny).
    center : tuple[float, float]
        Circle centre (x, y).
    radius : float
        Circle radius.

    Returns
    -------
    numpy.ndarray
        int32 cell labels, shape (Nx, Ny).
    """
    if radius <= 0:
        raise ValueError(f"[cell_labels] radius must be positive, got {radius}")
    _, _, dx, dy, Lx, Ly = periodic_grid_geometry(ref_grid_coords_ixyz)
    P = np.asarray(ref_grid_coords_ixyz, dtype=float)
    cell_centres = P + np.array([0.5 * dx, 0.5 * dy])[:, None, None]
    # PERIODIC: use minimum-image distance
    delta = periodic_minimum_image(cell_centres - np.asarray(center, float)[:, None, None], Lx, Ly)
    return (np.sum(delta ** 2, axis=0) <= radius ** 2).astype(np.int32)


def interface_node_mask(cell_inside: np.ndarray) -> np.ndarray:
    """
    Nodes whose four surrounding periodic cells do not all carry the same label.

    Parameters
    ----------
    cell_inside : numpy.ndarray
        Integer cell labels, shape (Nx, Ny).

    Returns
    -------
    numpy.ndarray
        Boolean node mask, shape (Nx, Ny).
    """
    c = np.asarray(cell_inside)
    if c.ndim != 2:
        raise ValueError(f"[interface_node_mask] cell_inside must be 2D, got shape {c.shape}")
    # PERIODIC: the four cells around node (i, j) via np.roll
    c_im1 = np.roll(c, 1, axis=0)
    c_jm1 = np.roll(c, 1, axis=1)
    c_im1_jm1 = np.roll(c_im1, 1, axis=1)
    return ~((c == c_im1) & (c == c_jm1) & (c == c_im1_jm1))


def circle_interface_cell_mask(
        ref_grid_coords_ixyz: np.ndarray,
        center: tuple[float, float],
        radius: float,
) -> np.ndarray:
    """
    Reference cells cut by the circle: some corners inside, some outside.

    A cell is an interface cell if min(r_corner) < radius <= max(r_corner),
    where r_corner is the periodic distance of a corner to the circle centre.

    Parameters
    ----------
    ref_grid_coords_ixyz : numpy.ndarray
        Reference nodal coordinates, shape (2, Nx, Ny).
    center : tuple[float, float]
        Circle centre (x, y).
    radius : float
        Circle radius.

    Returns
    -------
    numpy.ndarray
        Boolean cell mask, shape (Nx, Ny).
    """
    if radius <= 0:
        raise ValueError(f"[circle_interface_cell_mask] radius must be positive, got {radius}")
    _, _, _, _, Lx, Ly = periodic_grid_geometry(ref_grid_coords_ixyz)
    P = np.asarray(ref_grid_coords_ixyz, dtype=float)
    # PERIODIC: use minimum-image distance
    delta = periodic_minimum_image(P - np.asarray(center, float)[:, None, None], Lx, Ly)
    r = np.sqrt(np.sum(delta ** 2, axis=0))
    r_corners = np.stack((r, np.roll(r, -1, axis=0), np.roll(r, -1, axis=1),
                          np.roll(np.roll(r, -1, axis=0), -1, axis=1)))
    return (r_corners.min(axis=0) < radius) & (radius <= r_corners.max(axis=0))


def project_points_to_circle_periodic(
        Ppts: np.ndarray,
        center: tuple[float, float],
        R: float,
        Lx: float,
        Ly: float,
) -> np.ndarray:
    """
    Project points radially onto the nearest periodic image of the circle.

    Parameters
    ----------
    Ppts : numpy.ndarray
        Points, shape (2, k) or (k, 2).
    center : tuple[float, float]
        Circle centre (x, y).
    R : float
        Circle radius.
    Lx, Ly : float
        Fixed periodic domain lengths.

    Returns
    -------
    numpy.ndarray
        Projected points with the same shape as ``Ppts``. The points are moved
        by a local displacement; they are not wrapped into the domain.
    """
    name = "project_points_to_circle_periodic"
    Ppts = np.asarray(Ppts, dtype=float)
    if Ppts.ndim != 2:
        raise ValueError(f"[{name}] Ppts must be a 2D array, got shape {Ppts.shape}")
    if Ppts.shape[0] == 2:
        pts = Ppts.T
        transpose_back = True
    elif Ppts.shape[1] == 2:
        pts = Ppts
        transpose_back = False
    else:
        raise ValueError(f"[{name}] Ppts must have shape [2, k] or [k, 2], got {Ppts.shape}")

    c = np.asarray(center, dtype=float)
    lengths = np.array([Lx, Ly], dtype=float)
    if np.any(lengths <= 0):
        raise ValueError(f"[{name}] Lx and Ly must be positive")
    if R <= 0:
        raise ValueError(f"[{name}] R must be positive")

    # PERIODIC: vector from the nearest periodic circle-centre image to each point
    V = pts - c
    V -= lengths * np.round(V / lengths)
    r = np.linalg.norm(V, axis=1, keepdims=True)
    if np.any(r <= 1e-12):
        raise ValueError(f"[{name}] projection direction is undefined at the circle centre")
    projected = pts + (R / r - 1.0) * V
    return projected.T if transpose_back else projected


def adapt_grid_to_circle(
        ref_grid_coords_ixyz: np.ndarray = None,
        center: tuple[float, float] = (0.5, 0.5),
        radius: float = 0.2,
        iters: int = 600,
        b: float = 0.5,
        k0: float = 0.25,
        kmin: float = 0.0,
        if_spring_relaxation: bool = True,
        spring_method: str = "gauss_seidel",
        check_inversion_each_iter: bool = False,
        inversion_check: str = "quadrature",
) -> dict:
    """
    WORKFLOW A: adapt a periodic grid to a circular inclusion.

    Steps: classify reference cells -> interface nodes -> periodic radial
    projection -> remove invalid projected nodes -> stiffness -> (optional)
    spring relaxation. Cell phases are NOT recomputed after deformation.

    Parameters
    ----------
    ref_grid_coords_ixyz : numpy.ndarray
        Reference nodal coordinates, shape (2, Nx, Ny).
    center : tuple[float, float]
        Circle centre; the circle may cross the periodic boundary.
    radius : float
        Circle radius, must be < min(Lx, Ly) / 2.
    iters, b, k0, kmin : spring parameters (see ``spring_relax_weighted``
        and ``stiffness_from_distance``).
    if_spring_relaxation : bool
        If False, the projected grid is returned as final grid.
    spring_method : str
        "gauss_seidel" (original) or "jacobi_displacement".
    check_inversion_each_iter : bool
        Record inverted cells after every relaxation iteration.
    inversion_check : str
        Criterion of the invalid-node filter after projection:
        "quadrature" (solver det(F), default), "corner" (stricter) or
        "off" (no filtering, original method).

    Returns
    -------
    dict
        coarse_phase_field_geom             (Nx, Ny) int32, 1 = inclusion
        coarse_phase_field_binary           (Nx, Ny) uint8, same as geom
        coarse_interface_cell               (Nx, Ny) bool
        coarse_def_grid_coords_inxyz        (2, Nx, Ny) projected and filtered
        coarse_def_spring_grid_coords_inxyz (2, Nx, Ny) final coordinates
        fixed_node_mask                     (Nx, Ny) bool
        diagnostics                         dict
    """
    name = "adapt_grid_to_circle"
    if ref_grid_coords_ixyz is None:
        raise ValueError(f"[{name}] ref_grid_coords_ixyz must be provided")
    _validate_spring_parameters(name, iters, b, k0, kmin, spring_method)
    if inversion_check not in INVERSION_CHECKS:
        raise ValueError(f"[{name}] inversion_check must be one of {INVERSION_CHECKS}, got '{inversion_check}'")
    P_ref = np.asarray(ref_grid_coords_ixyz, dtype=float)
    Nx, Ny, dx, dy, Lx, Ly = periodic_grid_geometry(P_ref)
    if radius <= 0 or radius >= 0.5 * min(Lx, Ly):
        raise ValueError(
            f"[{name}] radius must satisfy 0 < radius < min(Lx, Ly)/2 = {0.5 * min(Lx, Ly)}, got {radius}"
        )

    inside = cell_labels(P_ref, center=center, radius=radius)
    interface_cells = circle_interface_cell_mask(P_ref, center, radius)
    interface_nodes = interface_node_mask(inside)
    if not np.any(interface_nodes):
        raise ValueError(
            f"[{name}] no interface node found; the circle (radius={radius}) is not resolved "
            f"by the grid (dx={dx}, dy={dy})"
        )

    P_projected = P_ref.copy()
    idx = np.argwhere(interface_nodes)
    P_projected[:, idx[:, 0], idx[:, 1]] = project_points_to_circle_periodic(
        P_ref[:, idx[:, 0], idx[:, 1]], center=center, R=radius, Lx=Lx, Ly=Ly)

    P_clean, P_final, fixed_clean, diagnostics = _relax_and_collect(
        name, P_ref, P_projected, interface_nodes, Lx, Ly, iters, b, k0, kmin,
        if_spring_relaxation, spring_method, check_inversion_each_iter, inversion_check)
    diagnostics = {"n_interface_nodes": int(np.count_nonzero(interface_nodes)), **diagnostics}

    return {
        "coarse_phase_field_geom": inside,
        "coarse_phase_field_binary": inside.astype(np.uint8),
        "coarse_interface_cell": interface_cells,
        "coarse_def_grid_coords_inxyz": P_clean,
        "coarse_def_spring_grid_coords_inxyz": P_final,
        "fixed_node_mask": fixed_clean,
        "diagnostics": diagnostics,
    }
