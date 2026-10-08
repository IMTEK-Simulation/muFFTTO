from __future__ import annotations
import time
from pathlib import Path
from typing import Union
import matplotlib.pyplot as plt
import numpy as np
from muFFTTO.otsu import otsu_edgeDetection_and_phaseIndicator

def majority_vote_for_coarse_grids(
        fine_phase_label_mask: np.ndarray,
        coarse_Nx: int,
        coarse_Ny: int,
):
    fine_phase_label_mask = np.asarray(fine_phase_label_mask)

    return coarse_phase_label_mask
# MAIN
def adapt_grid_to_arbitrary_shape(
        input_path: PathLike,
        ref_grid_coords_ixyz: np.ndarray = None,
        Lx = 1,
        Ly = 1,
        coarse_Nx: int = 32,
        coarse_Ny: int = 32,

):
    # ---------------------------------
    # 1. Load image
    # ---------------------------------
    input_path = Path(input_path)
    data = np.load(input_path).astype(np.float32)

    # ---------------------------------
    # 2. Fine cells: Otsu edge & phase detection
    # ---------------------------------
    results = otsu_edgeDetection_and_phaseIndicator(data)
    fine_edge_mask = results["edge_mask"]
    fine_phase_mask_binary = results["phase_mask_binary"]
    fine_phase_mask_label = results["phase_mask_label"]

    ny, nx = data.shape
    edge_binary = (fine_edge_mask > 0).astype(np.uint8)

    # ------------------------------------------------------------------
    # Define fine edge cells and its centre
    # ------------------------------------------------------------------
    # find edge index
    fine_edge_cell_idx_y, fine_edge_cell_idx_x = np.nonzero(edge_binary)
    # edge cells centre coords: used to tell if the fine edge cell is inside the coarse cell
    fine_edge_cell_centre_coords_x = (edge_idx_x + 0.5) / nx * Lx
    fine_edge_cell_centre_coords_y = (edge_idx_y + 0.5) / ny * Ly

    fine_edge_cell_centre = np.column_stack((fine_edge_cell_centre_coords_x,fine_edge_cell_centre_coords_y))
    print("hello")

    # ---------------------------------
    # 3. Define Coarse Cells: Phase Label Mask (Majority Vote)
    # ---------------------------------
    # get coarse node coords from discretization
    P0_coarse = ref_grid_coords_ixyz
    dx_coarse = Lx / coarse_Nx
    dy_coarse = Ly / coarse_Ny


    # ---------------------------------
    # 4. Define
    # (1)Interface Cells (if contains fine edges)
    # (2)Interface Nodes (has interface cell neighbor(s))
    # (3)Refine Coarse Interface Nodes: remove ones that are too far from the edge
    # ---------------------------------

    # ---------------------------------
    # 5. Project Refined Coarse Interface Nodes onto nearest edge Points
    # ---------------------------------

    # ---------------------------------
    # 6. Compute stiffness field
    # ---------------------------------

    # ---------------------------------
    # 7. Spring-relaxed(Deformed) coarse nodes coords
    # ---------------------------------


    return results

    # return {
    #     "coords_of_displaced_nodes": P_relaxed,
    #     "full_plot_coords_of_displaced_nodes": P_full,
    #     "fine_label_mask": fine_label_mask,
    #     "fine_edge_mask": fine_edge_mask,
    #     "coarse_label_mask": coarse_label_mask,
    #     "coarse_interface_cell": coarse_interface_cell,
    # }