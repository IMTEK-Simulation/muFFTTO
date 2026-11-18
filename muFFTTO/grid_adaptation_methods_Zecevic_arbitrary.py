"""
WORKFLOW B: periodic grid adaptation to an arbitrary imported image.

The fine-image information (``edge_mask``, ``phase_mask_binary``,
``phase_mask_label``) comes from ``muFFTTO.otsu.otsu_edgeDetection_and_phaseIndicator``
(use ``periodic=True`` there, so that the edge mask has no artificial edges
on the image boundary and labels are merged across the periodic boundary).

Steps
-----
1. Fine edge-cell centres.
2. Coarse interface cells: coarse cells that contain at least one fine edge pixel.
3. Coarse interface nodes: corners of interface cells.
4. Refinement: keep interface nodes with a fine edge centre inside the
   periodic search region ("circle" or "square").
5. Periodic projection of the kept nodes onto the nearest fine edge centre.
6. Removal of invalid projected nodes (duplicates, inverted cells).
7. Stiffness from the distance to the fixed nodes, optional spring relaxation.
8. Majority vote of fine labels and fine binary phases on the FINAL grid.

Conventions are those of ``muFFTTO.grid_adaptation_methods_Zecevic``:
coordinates (2, Nx, Ny) with indexing="ij", fine arrays (nfx, nfy) with
fine index i along x and j along y, reference grid starting at (0, 0).
This module never prints and never emits warnings; invalid input raises
``ValueError``, non-fatal information is returned in ``diagnostics``.
"""
from __future__ import annotations

import numpy as np
from matplotlib.path import Path as MplPath
from scipy.spatial import cKDTree

from muFFTTO.grid_adaptation_methods_Zecevic import (
    periodic_grid_geometry,
    periodic_minimum_image,
    interface_nodes_from_cells,
    _validate_spring_parameters,
    _relax_and_collect,
    INVERSION_CHECKS,
)

SEARCH_SHAPES = ("circle", "square")


# =============================================================
# WORKFLOW B: ARBITRARY IMPORTED IMAGE
# =============================================================

def fine_edge_cell_centres(edge_mask: np.ndarray, domain_size: tuple[float, float]) -> np.ndarray:
    """
    Physical centre coordinates of all fine edge pixels.

    Parameters
    ----------
    edge_mask : numpy.ndarray
        Fine edge mask, shape (nfx, nfy); nonzero marks an edge pixel.
    domain_size : tuple[float, float]
        Physical domain size (Lx, Ly).

    Returns
    -------
    numpy.ndarray
        Edge centres, shape (2, n_e), all inside [0, Lx) x [0, Ly).
    """
    name = "fine_edge_cell_centres"
    edge_mask = np.asarray(edge_mask)
    if edge_mask.ndim != 2 or edge_mask.size == 0:
        raise ValueError(f"[{name}] edge_mask must be a nonempty 2D array, got shape {edge_mask.shape}")
    Lx, Ly = map(float, domain_size)
    nfx, nfy = edge_mask.shape
    fi, fj = np.nonzero(edge_mask > 0)
    if fi.size == 0:
        raise ValueError(f"[{name}] edge_mask contains no edge pixel")
    return np.stack(((fi + 0.5) * Lx / nfx, (fj + 0.5) * Ly / nfy))


def coarse_interface_cells_from_edges(
        edge_centres: np.ndarray,
        number_of_coarse_pixels: tuple[int, int],
        domain_size: tuple[float, float],
) -> np.ndarray:
    """
    Coarse interface cells: coarse cells containing at least one fine edge centre.

    Parameters
    ----------
    edge_centres : numpy.ndarray
        Fine edge centres, shape (2, n_e).
    number_of_coarse_pixels : tuple[int, int]
        (Nx, Ny).
    domain_size : tuple[float, float]
        (Lx, Ly).

    Returns
    -------
    numpy.ndarray
        Boolean cell mask, shape (Nx, Ny).
    """
    name = "coarse_interface_cells_from_edges"
    edge_centres = np.asarray(edge_centres, dtype=float)
    if edge_centres.ndim != 2 or edge_centres.shape[0] != 2:
        raise ValueError(f"[{name}] edge_centres must have shape (2, n_e), got {edge_centres.shape}")
    Nx, Ny = map(int, number_of_coarse_pixels)
    Lx, Ly = map(float, domain_size)
    # PERIODIC: cell index taken modulo the number of cells
    ci = np.floor(edge_centres[0] / (Lx / Nx)).astype(int) % Nx
    cj = np.floor(edge_centres[1] / (Ly / Ny)).astype(int) % Ny
    cells = np.zeros((Nx, Ny), dtype=bool)
    cells[ci, cj] = True
    return cells


def _periodic_edge_tree(edge_centres, dx, dy, Lx, Ly, search_shape):
    """
    Build a periodic KD-tree of the edge centres.

    For "circle" the tree uses physical coordinates; for "square" the
    coordinates are scaled by (1/dx, 1/dy) so that the rectangular search box
    becomes a Chebyshev (p = inf) ball of radius search_range.
    """
    if search_shape == "circle":
        scale = np.array([1.0, 1.0])
        box = np.array([Lx, Ly])
    else:
        scale = np.array([1.0 / dx, 1.0 / dy])
        box = np.array([Lx / dx, Ly / dy])
    pts = np.mod(edge_centres.T * scale, box)
    pts = np.where(pts >= box, 0.0, pts)
    # PERIODIC: boxsize makes all KD-tree distances minimum-image distances
    return cKDTree(pts, boxsize=box), scale, box


def _validate_search(name, search_range, search_shape):
    if search_range <= 0:
        raise ValueError(f"[{name}] search_range must be positive, got {search_range}")
    if search_shape not in SEARCH_SHAPES:
        raise ValueError(f"[{name}] search_shape must be one of {SEARCH_SHAPES}, got '{search_shape}'")


def refine_interface_nodes_periodic(
        coarse_interface_node_mask: np.ndarray,
        ref_coarse_grid_coords_ixyz: np.ndarray,
        fine_edge_cell_centre: np.ndarray,
        dx_coarse: float,
        dy_coarse: float,
        search_range: float = 0.7,
        search_shape: str = "circle",
) -> tuple[np.ndarray, int]:
    """
    Keep interface nodes that have a fine edge centre inside the periodic search region.

    "circle": periodic distance <= search_range * min(dx, dy).
    "square": |dx_per| <= search_range * dx and |dy_per| <= search_range * dy.

    Parameters
    ----------
    coarse_interface_node_mask : numpy.ndarray
        Boolean node mask, shape (Nx, Ny).
    ref_coarse_grid_coords_ixyz : numpy.ndarray
        Reference nodal coordinates, shape (2, Nx, Ny).
    fine_edge_cell_centre : numpy.ndarray
        Fine edge centres, shape (2, n_e).
    dx_coarse, dy_coarse : float
        Coarse grid spacings.
    search_range : float
        Search size in units of the coarse spacing.
    search_shape : str
        "circle" or "square".

    Returns
    -------
    refined : numpy.ndarray
        Boolean node mask, shape (Nx, Ny).
    removed : int
        Number of interface nodes removed.
    """
    name = "refine_interface_nodes_periodic"
    _validate_search(name, search_range, search_shape)
    mask = np.asarray(coarse_interface_node_mask, dtype=bool)
    P = np.asarray(ref_coarse_grid_coords_ixyz, dtype=float)
    edges = np.asarray(fine_edge_cell_centre, dtype=float)
    Nx, Ny = mask.shape
    if P.shape != (2, Nx, Ny) or edges.ndim != 2 or edges.shape[0] != 2:
        raise ValueError(f"[{name}] invalid shapes: mask {mask.shape}, coords {P.shape}, edges {edges.shape}")
    if dx_coarse <= 0 or dy_coarse <= 0:
        raise ValueError(f"[{name}] coarse spacings must be positive")
    Lx, Ly = Nx * dx_coarse, Ny * dy_coarse

    tree, scale, box = _periodic_edge_tree(edges, dx_coarse, dy_coarse, Lx, Ly, search_shape)
    idx = np.argwhere(mask)
    refined = np.zeros_like(mask)
    if idx.size:
        nodes = np.mod(P[:, idx[:, 0], idx[:, 1]].T * scale, box)
        if search_shape == "circle":
            counts = tree.query_ball_point(nodes, search_range * min(dx_coarse, dy_coarse),
                                           return_length=True)
        else:
            counts = tree.query_ball_point(nodes, search_range, p=np.inf, return_length=True)
        hit = np.asarray(counts) > 0
        refined[idx[hit, 0], idx[hit, 1]] = True
    return refined, int(np.count_nonzero(mask & ~refined))


def project_nodes_to_nearest_edge_periodic(
        ref_coarse_grid_coords_ixyz: np.ndarray,
        node_mask: np.ndarray,
        fine_edge_cell_centre: np.ndarray,
        domain_size: tuple[float, float],
        search_range: float = 0.7,
        search_shape: str = "circle",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Move each selected node onto its nearest fine edge centre (periodic).

    Only edge centres inside the search region are candidates, so nodes are
    never projected further than the search region. The new position is
    P_ref + minimum-image displacement; it is NOT wrapped into the domain.

    Parameters
    ----------
    ref_coarse_grid_coords_ixyz : numpy.ndarray
        Reference nodal coordinates, shape (2, Nx, Ny).
    node_mask : numpy.ndarray
        Boolean mask of nodes to project, shape (Nx, Ny).
    fine_edge_cell_centre : numpy.ndarray
        Fine edge centres, shape (2, n_e).
    domain_size : tuple[float, float]
        (Lx, Ly).
    search_range : float
        Search size in units of the coarse spacing.
    search_shape : str
        "circle" or "square".

    Returns
    -------
    P_projected : numpy.ndarray
        Coordinates, shape (2, Nx, Ny); unprojected nodes stay at reference.
    projected_mask : numpy.ndarray
        Boolean mask of projected nodes, shape (Nx, Ny).
    """
    name = "project_nodes_to_nearest_edge_periodic"
    _validate_search(name, search_range, search_shape)
    P_ref = np.asarray(ref_coarse_grid_coords_ixyz, dtype=float)
    Nx, Ny, dx, dy, Lx, Ly = periodic_grid_geometry(P_ref, domain_size)
    mask = np.asarray(node_mask, dtype=bool)
    edges = np.asarray(fine_edge_cell_centre, dtype=float)
    if mask.shape != (Nx, Ny):
        raise ValueError(f"[{name}] node_mask must have shape {(Nx, Ny)}, got {mask.shape}")

    P_projected = P_ref.copy()
    projected_mask = np.zeros((Nx, Ny), dtype=bool)
    idx = np.argwhere(mask)
    if idx.size == 0:
        return P_projected, projected_mask

    tree, scale, box = _periodic_edge_tree(edges, dx, dy, Lx, Ly, search_shape)
    node_xy = P_ref[:, idx[:, 0], idx[:, 1]]
    nodes = np.mod(node_xy.T * scale, box)

    if search_shape == "circle":
        candidate_lists = tree.query_ball_point(nodes, search_range * min(dx, dy))
    else:
        candidate_lists = tree.query_ball_point(nodes, search_range, p=np.inf)

    nearest = np.full(len(idx), -1, dtype=int)
    for n, candidates in enumerate(candidate_lists):
        if not candidates:
            continue
        candidates = np.asarray(candidates)
        # PERIODIC: use minimum-image distance
        d = periodic_minimum_image(edges[:, candidates] - node_xy[:, n, None], Lx, Ly)
        d2 = np.sum(d ** 2, axis=0)
        # Nearest candidate; exact ties are broken by the smallest (dx, dy)
        # displacement so that the result does not depend on KD-tree ordering
        # and is invariant under periodic shifts of the image.
        tied = np.flatnonzero(d2 <= d2.min() * (1.0 + 1e-9) + 1e-30)
        pick = tied[np.lexsort((d[1, tied], d[0, tied]))[0]]
        nearest[n] = candidates[pick]

    found = nearest >= 0
    if np.any(found):
        # PERIODIC: displacement to the nearest periodic image of the edge centre
        delta = periodic_minimum_image(edges[:, nearest[found]] - node_xy[:, found], Lx, Ly)
        if search_shape == "circle":
            ok = np.sqrt(np.sum(delta ** 2, axis=0)) <= search_range * min(dx, dy) * (1 + 1e-9)
        else:
            ok = (np.abs(delta[0]) <= search_range * dx * (1 + 1e-9)) & \
                 (np.abs(delta[1]) <= search_range * dy * (1 + 1e-9))
        if not np.all(ok):
            raise ValueError(f"[{name}] internal error: projection exceeded the search region")
        ii, jj = idx[found, 0], idx[found, 1]
        P_projected[:, ii, jj] = node_xy[:, found] + delta
        projected_mask[ii, jj] = True
    return P_projected, projected_mask


def fine_cells_inside_coarse_cell(
        corners: np.ndarray,
        fine_shape: tuple[int, int],
        domain_size: tuple[float, float],
) -> tuple[np.ndarray, np.ndarray]:
    """
    Fine pixels whose centres lie inside one (possibly deformed) coarse cell.

    The candidate fine indices are computed directly from the cell bounding box
    (indices may lie outside [0, nf) and are wrapped afterwards), so only a few
    thousand points are tested per cell instead of the whole fine image.

    Parameters
    ----------
    corners : numpy.ndarray
        Unwrapped corners p00, p10, p11, p01, shape (4, 2).
    fine_shape : tuple[int, int]
        (nfx, nfy).
    domain_size : tuple[float, float]
        (Lx, Ly).

    Returns
    -------
    fi, fj : numpy.ndarray
        Wrapped fine indices of the inside pixels, each shape (n_inside,).
    """
    name = "fine_cells_inside_coarse_cell"
    corners = np.asarray(corners, dtype=float)
    if corners.shape != (4, 2):
        raise ValueError(f"[{name}] corners must have shape (4, 2), got {corners.shape}")
    nfx, nfy = map(int, fine_shape)
    Lx, Ly = map(float, domain_size)
    if Lx <= 0 or Ly <= 0:
        raise ValueError(f"[{name}] domain lengths must be positive")
    hx, hy = Lx / nfx, Ly / nfy
    (xmin, ymin), (xmax, ymax) = corners.min(axis=0), corners.max(axis=0)
    # fine centre (i + 0.5) * h inside [min, max]
    i0, i1 = int(np.ceil(xmin / hx - 0.5)), int(np.floor(xmax / hx - 0.5))
    j0, j1 = int(np.ceil(ymin / hy - 0.5)), int(np.floor(ymax / hy - 0.5))
    if i1 - i0 + 1 > nfx or j1 - j0 + 1 > nfy:
        raise ValueError(f"[{name}] a coarse cell is larger than the periodic domain; the grid is broken")
    if i1 < i0 or j1 < j0:
        return np.empty(0, dtype=int), np.empty(0, dtype=int)
    ii, jj = np.meshgrid(np.arange(i0, i1 + 1), np.arange(j0, j1 + 1), indexing="ij")
    ii, jj = ii.ravel(), jj.ravel()
    points = np.column_stack(((ii + 0.5) * hx, (jj + 0.5) * hy))
    inside = MplPath(corners).contains_points(points)
    # PERIODIC: wrap unwrapped fine indices back into the image
    return ii[inside] % nfx, jj[inside] % nfy


def _majority_vote_core(fine_labels, def_coords, number_of_coarse_pixels, domain_size, empty_cell_label):
    """Majority vote returning the coarse labels and the list of empty cells."""
    name = "majority_vote_for_coarse_grids"
    Nx, Ny = map(int, number_of_coarse_pixels)
    Lx, Ly = map(float, domain_size)
    labels = np.asarray(fine_labels)
    P = np.asarray(def_coords, dtype=float)
    if P.shape != (2, Nx, Ny):
        raise ValueError(f"[{name}] coordinates must have shape {(2, Nx, Ny)}, got {P.shape}")
    if labels.ndim != 2 or labels.size == 0:
        raise ValueError(f"[{name}] fine labels must be a nonempty 2D array, got shape {labels.shape}")
    if Lx <= 0 or Ly <= 0:
        raise ValueError(f"[{name}] domain lengths must be positive")
    dx, dy = Lx / Nx, Ly / Ny
    P_ref = np.stack(np.meshgrid(np.arange(Nx) * dx, np.arange(Ny) * dy, indexing="ij"))
    u = P - P_ref

    out = np.full((Nx, Ny), empty_cell_label, dtype=labels.dtype)
    empty_cells = []
    for i in range(Nx):
        for j in range(Ny):
            ip, jp = (i + 1) % Nx, (j + 1) % Ny
            # PERIODIC: unwrapped corners = reference corner + periodic nodal displacement
            x0, y0 = i * dx, j * dy
            corners = np.array([
                [x0 + u[0, i, j], y0 + u[1, i, j]],
                [x0 + dx + u[0, ip, j], y0 + u[1, ip, j]],
                [x0 + dx + u[0, ip, jp], y0 + dy + u[1, ip, jp]],
                [x0 + u[0, i, jp], y0 + dy + u[1, i, jp]],
            ])
            fi, fj = fine_cells_inside_coarse_cell(corners, labels.shape, (Lx, Ly))
            if fi.size == 0:
                empty_cells.append((i, j))
                continue
            values, counts = np.unique(labels[fi, fj], return_counts=True)
            out[i, j] = values[np.argmax(counts)]  # ties -> smallest label
    return out, empty_cells


def majority_vote_for_coarse_grids(
        fine_phase_mask_label: np.ndarray,
        def_coarse_grid_coords_ixyz: np.ndarray,
        number_of_coarse_pixels: tuple[int, int],
        domain_size: tuple[float, float] = (1.0, 1.0),
        empty_cell_label: int = 0,
) -> np.ndarray:
    """
    Assign every deformed coarse cell the most frequent fine label inside it.

    Parameters
    ----------
    fine_phase_mask_label : numpy.ndarray
        Fine labels (or binary phases), shape (nfx, nfy).
    def_coarse_grid_coords_ixyz : numpy.ndarray
        Deformed coarse coordinates, shape (2, Nx, Ny) (no periodic end nodes).
    number_of_coarse_pixels : tuple[int, int]
        (Nx, Ny).
    domain_size : tuple[float, float]
        (Lx, Ly).
    empty_cell_label : int
        Label of a coarse cell that contains no fine pixel centre.

    Returns
    -------
    numpy.ndarray
        Coarse labels, shape (Nx, Ny), dtype of the fine labels.
    """
    out, _ = _majority_vote_core(fine_phase_mask_label, def_coarse_grid_coords_ixyz,
                                 number_of_coarse_pixels, domain_size, empty_cell_label)
    return out


def adapt_grid_to_arbitrary_shape(
        results: dict = None,
        ref_coarse_grid_coords_ixyz: np.ndarray = None,
        domain_size: tuple = (1, 1),
        number_of_coarse_pixels: tuple = (32, 32),
        iters: int = 600,
        b: float = 0.5,
        k0: float = 0.25,
        kmin: float = 0.0,
        if_spring_relaxation: bool = True,
        search_range: float = 0.7,
        search_shape: str = "circle",
        spring_method: str = "gauss_seidel",
        check_inversion_each_iter: bool = False,
        inversion_check: str = "quadrature",
) -> dict:
    """
    WORKFLOW B: adapt a periodic coarse grid to the interfaces of a fine image.

    Parameters
    ----------
    results : dict
        Output of ``otsu_edgeDetection_and_phaseIndicator``; needs
        "edge_mask", "phase_mask_binary" and "phase_mask_label", each (nfx, nfy).
    ref_coarse_grid_coords_ixyz : numpy.ndarray
        Reference coarse coordinates in PHYSICAL units, shape (2, Nx, Ny).
    domain_size : tuple
        (Lx, Ly).
    number_of_coarse_pixels : tuple
        (Nx, Ny).
    iters, b, k0, kmin : spring parameters.
    if_spring_relaxation : bool
        If False, the projected grid is the final grid.
    search_range, search_shape : node refinement / projection search region.
    spring_method : str
        "gauss_seidel" (original) or "jacobi_displacement".
    check_inversion_each_iter : bool
        Record inverted cells after every relaxation iteration.
    inversion_check : str
        Criterion of the invalid-node filter after projection:
        "quadrature" (solver det(F), default), "corner" (stricter) or
        "off" (no filtering).

    Returns
    -------
    dict
        coarse_phase_field_geom             (Nx, Ny) label majority vote
        coarse_phase_field_binary           (Nx, Ny) uint8 binary majority vote
        coarse_interface_cell               (Nx, Ny) bool
        coarse_def_grid_coords_inxyz        (2, Nx, Ny) projected and filtered
        coarse_def_spring_grid_coords_inxyz (2, Nx, Ny) final coordinates
        fixed_node_mask                     (Nx, Ny) bool
        diagnostics                         dict
    """
    name = "adapt_grid_to_arbitrary_shape"
    if results is None or ref_coarse_grid_coords_ixyz is None:
        raise ValueError(f"[{name}] results and ref_coarse_grid_coords_ixyz are required")
    _validate_spring_parameters(name, iters, b, k0, kmin, spring_method)
    _validate_search(name, search_range, search_shape)
    if inversion_check not in INVERSION_CHECKS:
        raise ValueError(f"[{name}] inversion_check must be one of {INVERSION_CHECKS}, got '{inversion_check}'")
    for key in ("edge_mask", "phase_mask_binary", "phase_mask_label"):
        if key not in results:
            raise ValueError(f"[{name}] results is missing the key '{key}'")

    P_ref = np.asarray(ref_coarse_grid_coords_ixyz, dtype=float)
    Nx, Ny, dx, dy, Lx, Ly = periodic_grid_geometry(P_ref, domain_size)
    if (Nx, Ny) != tuple(map(int, number_of_coarse_pixels)):
        raise ValueError(
            f"[{name}] number_of_coarse_pixels {tuple(number_of_coarse_pixels)} does not match "
            f"the reference grid {(Nx, Ny)}"
        )

    edge_mask = np.asarray(results["edge_mask"])
    binary = np.asarray(results["phase_mask_binary"])
    labels = np.asarray(results["phase_mask_label"])
    if edge_mask.ndim != 2 or edge_mask.shape != binary.shape or edge_mask.shape != labels.shape:
        raise ValueError(
            f"[{name}] edge_mask {edge_mask.shape}, phase_mask_binary {binary.shape} and "
            f"phase_mask_label {labels.shape} must be equal 2D shapes"
        )

    # 1-3. fine edges -> coarse interface cells -> coarse interface nodes
    edges = fine_edge_cell_centres(edge_mask, (Lx, Ly))
    interface_cells = coarse_interface_cells_from_edges(edges, (Nx, Ny), (Lx, Ly))
    interface_nodes = interface_nodes_from_cells(interface_cells)

    # 4. keep nodes close enough to a fine edge
    refined, n_refine_removed = refine_interface_nodes_periodic(
        interface_nodes, P_ref, edges, dx, dy, search_range, search_shape)
    if not np.any(refined):
        raise ValueError(
            f"[{name}] no interface node has a fine edge within search_range={search_range} "
            f"({search_shape}); increase search_range"
        )

    # 5. periodic projection
    P_projected, projected_mask = project_nodes_to_nearest_edge_periodic(
        P_ref, refined, edges, (Lx, Ly), search_range, search_shape)

    # 6-7. invalid nodes, stiffness, spring relaxation
    P_clean, P_final, fixed_clean, relax_diagnostics = _relax_and_collect(
        name, P_ref, P_projected, projected_mask, Lx, Ly, iters, b, k0, kmin,
        if_spring_relaxation, spring_method, check_inversion_each_iter, inversion_check)

    # 8. majority vote on the FINAL grid
    phase_geom, empty_label = _majority_vote_core(labels, P_final, (Nx, Ny), (Lx, Ly), 0)
    phase_binary, _ = _majority_vote_core(
        (binary > 0).astype(np.uint8), P_final, (Nx, Ny), (Lx, Ly), 0)

    diagnostics = {
        "n_fine_edge_pixels": int(edges.shape[1]),
        "n_interface_cells": int(np.count_nonzero(interface_cells)),
        "n_interface_nodes": int(np.count_nonzero(interface_nodes)),
        "n_refine_removed": n_refine_removed,
        **relax_diagnostics,
        "empty_cells_majority_vote": empty_label,
    }
    return {
        "coarse_phase_field_geom": phase_geom,
        "coarse_phase_field_binary": phase_binary.astype(np.uint8),
        "coarse_interface_cell": interface_cells,
        "coarse_def_grid_coords_inxyz": P_clean,
        "coarse_def_spring_grid_coords_inxyz": P_final,
        "fixed_node_mask": fixed_clean,
        "diagnostics": diagnostics,
    }
