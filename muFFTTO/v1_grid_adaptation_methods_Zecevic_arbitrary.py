from __future__ import annotations
import time
from pathlib import Path
from typing import Union
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.path import Path as MplPath
from muFFTTO.v0_otsu import otsu_edgeDetection_and_phaseIndicator
from muFFTTO.grid_adaptation_methods_Zecevic import *

def refine_interface_nodes_periodic(
    coarse_interface_node_mask: np.ndarray,
    ref_coarse_grid_coords_ixyz: tuple[np.ndarray, np.ndarray],
    fine_edge_cell_centre: tuple[np.ndarray, np.ndarray],
    dx_coarse: float,
    dy_coarse: float,
    search_range: float = 0.7,
    search_shape: str = "circle",
):
    """
    Keep coarse interface nodes only if at least one fine edge-cell centre
    lies within a periodic search region around the coarse node.

    Parameters
    ----------
    coarse_interface_node_mask : (Nx, Ny) bool array
    ref_coarse_grid_coords_ixyz : (Xn, Yn), each (Nx, Ny)
    fine_edge_cell_centre : (xn, yn), each (Nfine,)
    dx_coarse, dy_coarse : float
        Coarse-grid spacings.
    search_range : float
        Search radius/box size in units of coarse spacing.
    search_shape : {"circle", "square"}

    Returns
    -------
    refined : (Nx, Ny) bool array
    removed : int
    matched_dx, matched_dy : 1D arrays
        Periodic displacement from matched coarse node to fine cell centre.
    matched_i, matched_j : 1D int arrays
        Indices of the matched coarse nodes.
    """
    refined = coarse_interface_node_mask.copy()

    Xn, Yn = ref_coarse_grid_coords_ixyz
    xn, yn = fine_edge_cell_centre

    Nx, Ny = coarse_interface_node_mask.shape

    # Periodic domain lengths
    Lx = Nx * dx_coarse
    Ly = Ny * dy_coarse

    hx = search_range * dx_coarse
    hy = search_range * dy_coarse

    if search_shape == "circle":
        r2 = (search_range * min(dx_coarse, dy_coarse)) ** 2
    elif search_shape == "square":
        pass
    else:
        raise ValueError("search_shape must be 'circle' or 'square'")

    matched_dx = []
    matched_dy = []
    matched_i = []
    matched_j = []
    removed = 0

    for i in range(Nx):
        for j in range(Ny):
            if not coarse_interface_node_mask[i, j]:
                continue

            X_node = Xn[i, j]
            Y_node = Yn[i, j]

            # Raw displacement: coarse node -> fine cell centre
            dx_raw = xn - X_node
            dy_raw = yn - Y_node

            # Minimum-image correction: wrap into [-L/2, L/2)
            dx_per = dx_raw - Lx * np.round(dx_raw / Lx)
            dy_per = dy_raw - Ly * np.round(dy_raw / Ly)

            if search_shape == "square":
                in_search_range = (
                    (np.abs(dx_per) <= hx) &
                    (np.abs(dy_per) <= hy)
                )
            else:  # circle
                in_search_range = (
                    dx_per ** 2 + dy_per ** 2
                ) <= r2

            n_match = np.count_nonzero(in_search_range)

            if n_match > 0:
                matched_dx.append(dx_per[in_search_range])
                matched_dy.append(dy_per[in_search_range])
                matched_i.append(np.full(n_match, i, dtype=int))
                matched_j.append(np.full(n_match, j, dtype=int))
            else:
                refined[i, j] = False
                removed += 1

    if matched_dx:
        matched_dx = np.concatenate(matched_dx)
        matched_dy = np.concatenate(matched_dy)
        matched_i = np.concatenate(matched_i)
        matched_j = np.concatenate(matched_j)
    else:
        matched_dx = np.array([])
        matched_dy = np.array([])
        matched_i = np.array([], dtype=int)
        matched_j = np.array([], dtype=int)

    return refined, removed

# TODO: check the majority vote usage
'''
    1. obtain def_coarse_grid_coords
    2. calculate fine grids centre
    3. Test: which fine cells are inside the deformed grid
    4. inside fine cells voting
    5. define coarse cell label
'''

def majority_vote_for_coarse_grids(
    fine_phase_mask_label: np.ndarray,
    def_coarse_grid_coords_ixyz: np.ndarray,
    number_of_coarse_pixels: tuple,
    domain_size: tuple = (1.0, 1.0),
    empty_cell_label: int = 0,
):
    """
    Assign a phase label to every deformed coarse grid cell by majority vote
    of the fine-grid cells whose centres lie inside that coarse cell.

    Parameters
    ----------
    fine_phase_mask_label : np.ndarray
        Fine-grid phase labels, shape (nfx, nfy).

    def_coarse_grid_coords_ixyz : np.ndarray
        Deformed coarse grid node coordinates, shape (2, Nx + 1, Ny + 1).

    number_of_coarse_pixels : tuple
        Number of coarse cells, (Nx, Ny).

    domain_size : tuple
        Physical domain size, (Lx, Ly).

    empty_cell_label : int
        Label assigned to a coarse cell if no fine cell centre lies inside it.

    Returns
    -------
    coarse_phase_geom : np.ndarray
        Coarse-grid phase labels obtained by majority vote.
        Shape: (Nx, Ny).
    """
    Nx, Ny = number_of_coarse_pixels
    Lx, Ly = domain_size

    nfx, nfy = fine_phase_mask_label.shape

    # ------------------------------------------------------------------
    # Fine-grid cell centres
    # ------------------------------------------------------------------
    fine_cell_idx_x = np.arange(nfx)
    fine_cell_idx_y = np.arange(nfy)

    fine_cell_centre_x = (fine_cell_idx_x + 0.5) / nfx * Lx
    fine_cell_centre_y = (fine_cell_idx_y + 0.5) / nfy * Ly

    # xf[i, j] and yf[i, j] are the centre of fine cell (i, j).
    xf, yf = np.meshgrid(
        fine_cell_centre_x,
        fine_cell_centre_y,
        indexing="ij",
    )

    # ------------------------------------------------------------------
    # Majority vote for every coarse cell
    # ------------------------------------------------------------------
    coarse_phase_geom = np.zeros(
        (Nx, Ny),
        dtype=fine_phase_mask_label.dtype,
    )

    for ci in range(Nx):
        for cj in range(Ny):
            # Periodic neighbour indices.
            ci_next = (ci + 1) % Nx
            cj_next = (cj + 1) % Ny

            # Periodic coordinate offsets.
            offset_x_next = Lx if ci + 1 == Nx else 0.0
            offset_y_next = Ly if cj + 1 == Ny else 0.0

            # Coarse cell corners, including periodic offsets.
            p00 = def_coarse_grid_coords_ixyz[:, ci, cj].copy()

            p10 = def_coarse_grid_coords_ixyz[:, ci_next, cj].copy()
            p10[0] += offset_x_next

            p11 = def_coarse_grid_coords_ixyz[:, ci_next, cj_next].copy()
            p11[0] += offset_x_next
            p11[1] += offset_y_next

            p01 = def_coarse_grid_coords_ixyz[:, ci, cj_next].copy()
            p01[1] += offset_y_next

            # Find fine cells whose centres lie inside this coarse cell.
            inside_fine = fine_cells_inside_coarse_cell(
                xf=xf,
                yf=yf,
                p00=p00,
                p10=p10,
                p11=p11,
                p01=p01,
            )

            if not np.any(inside_fine):
                coarse_phase_geom[ci, cj] = empty_cell_label
                continue

            # Labels of all fine cells inside this coarse cell.
            labels_inside = fine_phase_mask_label[inside_fine]

            # Majority vote.
            unique_labels, counts = np.unique(
                labels_inside,
                return_counts=True,
            )

            coarse_phase_geom[ci, cj] = unique_labels[np.argmax(counts)]

    return coarse_phase_geom


def fine_cells_inside_coarse_cell(
    xf: np.ndarray,
    yf: np.ndarray,
    p00: np.ndarray,
    p10: np.ndarray,
    p11: np.ndarray,
    p01: np.ndarray,
):
    """
    Return a boolean mask of fine cell centres inside one deformed coarse cell.

    The coarse cell is treated as a polygon with corners:

        p00 -> p10 -> p11 -> p01 -> p00

    A bounding-box preselection is applied before the exact polygon test.

    Parameters
    ----------
    xf, yf : np.ndarray
        Fine cell centre coordinates, shape (nfx, nfy).

    p00, p10, p11, p01 : np.ndarray
        Coarse cell corner coordinates, each shape (2,).

        p00 : lower-left corner
        p10 : lower-right corner
        p11 : upper-right corner
        p01 : upper-left corner

    Returns
    -------
    inside_fine : np.ndarray
        Boolean array, shape (nfx, nfy).
        inside_fine[i, j] is True if the centre of fine cell (i, j)
        lies inside the coarse cell.
    """
    # ------------------------------------------------------------------
    # Bounding-box preselection
    # ------------------------------------------------------------------
    x_min = min(p00[0], p10[0], p01[0], p11[0])
    x_max = max(p00[0], p10[0], p01[0], p11[0])

    y_min = min(p00[1], p10[1], p01[1], p11[1])
    y_max = max(p00[1], p10[1], p01[1], p11[1])

    candidate_mask = (
        (xf >= x_min)
        & (xf <= x_max)
        & (yf >= y_min)
        & (yf <= y_max)
    )

    inside_fine = np.zeros(xf.shape, dtype=bool)

    candidate_i, candidate_j = np.nonzero(candidate_mask)

    if candidate_i.size == 0:
        return inside_fine

    # Coordinates of candidate fine cell centres.
    x_candidates = xf[candidate_i, candidate_j]
    y_candidates = yf[candidate_i, candidate_j]

    candidate_points = np.column_stack(
        (x_candidates, y_candidates)
    )

    # ------------------------------------------------------------------
    # Exact polygon containment test
    # ------------------------------------------------------------------
    # The vertices must be ordered around the quadrilateral.
    vertices = np.stack(
        (p00, p10, p11, p01),
        axis=0,
    )

    polygon = MplPath(vertices)

    inside_candidates = polygon.contains_points(candidate_points)

    # Write the result back into the full fine-grid-shaped mask.
    inside_fine[
        candidate_i[inside_candidates],
        candidate_j[inside_candidates],
    ] = True

    return inside_fine


# MAIN
def adapt_grid_to_arbitrary_shape(
        results: dict = None,
        ref_coarse_grid_coords_ixyz: np.ndarray = None,
        domain_size: tuple = (1,1),
        number_of_coarse_pixels: tuple = (32,32),
        iters: int = 600,
        omega: float = 1,
        b: float = 0.5,
        k0: float = 0.25,
        kmin: float = 0.0,
        if_spring_relaxation: bool = True,
):
    # Input: fine edge detection results, including fine edge mask, fine phase mask label & binary

    # ---------------------------------
    # 1. Fine cells: Otsu edge & phase detection
    # ---------------------------------
    fine_edge_mask = results["edge_mask"]
    fine_phase_mask_binary = results["phase_mask_binary"]
    fine_phase_mask_label = results["phase_mask_label"]

    # ------------------------------------------------------------------
    # 2. Define fine edge cells and its centre
    # ------------------------------------------------------------------
    nx,ny = fine_edge_mask.shape
    Lx, Ly = domain_size
    coarse_Nx, coarse_Ny = number_of_coarse_pixels
    edge_binary = (fine_edge_mask > 0).astype(np.uint8)
    # find edge index
    fine_edge_cell_idx_x, fine_edge_cell_idx_y = np.nonzero(edge_binary)
    # edge cells centre coords: used to tell if the fine edge cell is inside the coarse cell
    # fine_edge_cell_idx_x, fine_edge_cell_idx_y = np.meshgrid(
    #     np.arange(nx), np.arange(ny), indexing="xy"
    # )

    fine_edge_cell_centre_coords_x = (fine_edge_cell_idx_x + 0.5) / nx * Lx
    fine_edge_cell_centre_coords_y = (fine_edge_cell_idx_y + 0.5) / ny * Ly

    fine_edge_cell_centre = np.stack(
        (fine_edge_cell_centre_coords_x, fine_edge_cell_centre_coords_y),
        axis=0
    )

    # ---------------------------------
    # 3. Define the fine nodes that'll shift
    # ---------------------------------
    # (1)Coarse interface Cells (if contains fine edges)
    coarse_interface_cell_mask = np.zeros((coarse_Ny, coarse_Nx),dtype=bool)

    dx_coarse = Lx/coarse_Nx
    dy_coarse = Ly/coarse_Ny
    ref_x = ref_coarse_grid_coords_ixyz[0]
    ref_y = ref_coarse_grid_coords_ixyz[1]

    for i in range(coarse_Nx):
        for j in range(coarse_Ny):
            X0 = ref_x[i, j]
            Y0 = ref_y[i, j]
            X1 = X0 + dx_coarse
            Y1 = Y0 + dy_coarse

            inside = (
                    (fine_edge_cell_centre_coords_x >= X0) &
                    (fine_edge_cell_centre_coords_x < X1) &
                    (fine_edge_cell_centre_coords_y >= Y0) &
                    (fine_edge_cell_centre_coords_y < Y1)
            )
            coarse_interface_cell_mask[i, j] = np.any(inside)

    # (2)Coarse interface Nodes (has interface cell neighbor(s))
    coarse_interface_node_mask = np.zeros((coarse_Ny, coarse_Nx),dtype=bool)

    # Node (i,j) is the lower-left corner of cell (j, i) and the shared corner
    # of cells (i-1, j-1), (i-1, j), (i, j-1), (i, j) (periodic indices).
    for i in range(coarse_Nx):
        for j in range(coarse_Ny):
            surrounding_cells = [
                coarse_interface_cell_mask[(i - 1) % coarse_Nx,(j - 1) % coarse_Ny],
                coarse_interface_cell_mask[(i - 1) % coarse_Nx,j % coarse_Ny],
                coarse_interface_cell_mask[i % coarse_Nx,(j - 1) % coarse_Ny],
                coarse_interface_cell_mask[i % coarse_Nx,j % coarse_Ny],
            ]
            coarse_interface_node_mask[i, j] = np.any(surrounding_cells)

    # (3)Refine Coarse Interface Nodes: remove ones that are too far from the edge
    # TODO: should also include periodic cell nodes
    coarse_interface_node_refined, removed = refine_interface_nodes_periodic(coarse_interface_node_mask=coarse_interface_node_mask, ref_coarse_grid_coords_ixyz=ref_coarse_grid_coords_ixyz, fine_edge_cell_centre=fine_edge_cell_centre, dx_coarse=dx_coarse, dy_coarse=dy_coarse)
    removed_nodes_mask = (coarse_interface_node_mask & (~ coarse_interface_node_refined))

    # ---------------------------------
    # 5. Project Refined Coarse Interface Nodes onto nearest edge Points
    # ---------------------------------
    def_coarse_grid_coords_ixyz = ref_coarse_grid_coords_ixyz.copy()
    displacement = np.zeros_like(ref_coarse_grid_coords_ixyz)
    n_fine_edges = fine_edge_cell_centre.shape[1]

    # TODO: When Project, should consider periodic situation
    # TODO: 想想在邊界的，在週期性重複的狀況下，哪個鄰居才是最近的?
    for i in range (coarse_Nx):
        for j in range (coarse_Ny):
            if not coarse_interface_node_refined[i,j]:
                continue

            point = ref_coarse_grid_coords_ixyz[:, i, j]
            distance_squared = np.zeros(n_fine_edges, dtype=float)

            for k in range(n_fine_edges):
                fine_point = fine_edge_cell_centre[:, k]

                difference = fine_point - point
                distance_squared[k] = np.sum(difference ** 2)

            # Choose the closest fine edge centre
            nearest_index = np.argmin(distance_squared)
            nearest_point = fine_edge_cell_centre[:, nearest_index]

            # Move the coarse interface node directly onto that point
            def_coarse_grid_coords_ixyz[0, i, j] = nearest_point[0]
            def_coarse_grid_coords_ixyz[1, i, j] = nearest_point[1]

            displacement[0, i, j] = nearest_point[0] - point[0]
            displacement[1, i, j] = nearest_point[1] - point[1]

    # ---------------------------------
    # Define Coarse Cells: Phase Label Mask --> Majority Vote
    # ---------------------------------
    P0_coarse = ref_coarse_grid_coords_ixyz
    coarse_phase_field_geom = majority_vote_for_coarse_grids(fine_phase_mask_label,def_coarse_grid_coords_ixyz,number_of_coarse_pixels)

    def_spring_coarse_grid_coords_ixyz = def_coarse_grid_coords_ixyz.copy()

    if if_spring_relaxation:
        # ---------------------------------
        # 6. Compute stiffness field
        # ---------------------------------
        dist = manhattan_distance_to_interface(coarse_interface_cell_mask)
        k_node = stiffness_from_distance(dist, k0=k0, b=b, a=1.0, kmin=kmin)
        # ---------------------------------
        # 7. Spring-relaxed(Deformed) coarse nodes coords
        # ---------------------------------
        def_spring_coarse_grid_coords_ixyz = spring_relax_weighted(def_coarse_grid_coords_ixyz, fixed_mask=coarse_interface_node_refined, k_node=k_node, iters=iters, omega=omega)

    return coarse_phase_field_geom,coarse_interface_cell_mask,def_coarse_grid_coords_ixyz, def_spring_coarse_grid_coords_ixyz

    # return {
    #     "coords_of_displaced_nodes": P_relaxed,
    #     "full_plot_coords_of_displaced_nodes": P_full,
    #     "fine_label_mask": fine_label_mask,
    #     "fine_edge_mask": fine_edge_mask,
    #     "coarse_label_mask": coarse_label_mask,
    #     "coarse_interface_cell_mask": coarse_interface_cell_mask,
    # }