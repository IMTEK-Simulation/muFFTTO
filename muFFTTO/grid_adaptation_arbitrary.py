"""
Simply use function "run_grid_adaptation_workflow", to gain the deformed grid coordinates
"""

from __future__ import annotations
import time
from pathlib import Path
from typing import Union
import matplotlib.pyplot as plt
import numpy as np
from muFFTTO.otsu import otsu_edgeDetection_and_phaseIndicator

def build_regular_coarse_grid(coarse_Nx, coarse_Ny, Lx, Ly):
    x_nodes = np.linspace(0.0, Lx, coarse_Nx, endpoint=False)
    y_nodes = np.linspace(0.0, Ly, coarse_Ny, endpoint=False)
    Xn, Yn = np.meshgrid(x_nodes, y_nodes)
    P0 = np.zeros((2, coarse_Ny, coarse_Nx), dtype=float)
    P0[0] = Xn
    P0[1] = Yn
    return P0, Xn, Yn

def make_plot_coords(P, Lx, Ly):
    Ny, Nx = P.shape[1], P.shape[2]
    Pplot = np.zeros((2, Ny + 1, Nx + 1), dtype=float)
    Pplot[:, :Ny, :Nx] = P
    Pplot[0, :Ny, Nx] = P[0, :, 0] + Lx
    Pplot[1, :Ny, Nx] = P[1, :, 0]
    Pplot[0, Ny, :Nx] = P[0, 0, :]
    Pplot[1, Ny, :Nx] = P[1, 0, :] + Ly
    Pplot[0, Ny, Nx] = P[0, 0, 0] + Lx
    Pplot[1, Ny, Nx] = P[1, 0, 0] + Ly
    return Pplot

def periodic_manhattan_distance_to_mask(mask):
    from collections import deque

    Ny, Nx = mask.shape
    dist = np.full((Ny, Nx), np.inf, dtype=float)
    q = deque()

    for j in range(Ny):
        for i in range(Nx):
            if mask[j, i]:
                dist[j, i] = 0.0
                q.append((j, i))

    neighbors = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    while q:
        j, i = q.popleft()
        for dj, di in neighbors:
            jj = (j + dj) % Ny
            ii = (i + di) % Nx
            if dist[jj, ii] > dist[j, i] + 1.0:
                dist[jj, ii] = dist[j, i] + 1.0
                q.append((jj, ii))
    return dist

def stiffness_from_distance(dist, k0=1.0, b=0.0, a=1.0, kmin=1e-3):
    k = k0 / np.power(dist + a, b)
    return np.maximum(k, kmin)

def spring_relax_projected_grid(P_init, fixed_mask, k_node, iters=400, omega=0.35):
    P = P_init.copy()
    Ny, Nx = P.shape[1], P.shape[2]
    dx_grid = P_init[0, 0, 1] - P_init[0, 0, 0] if Nx > 1 else 0.0
    dy_grid = P_init[1, 1, 0] - P_init[1, 0, 0] if Ny > 1 else 0.0

    for _ in range(iters):
        P_new = P.copy()
        for j in range(Ny):
            for i in range(Nx):
                if fixed_mask[j, i]:
                    continue

                neighbors = [
                    (j, (i - 1) % Nx),
                    (j, (i + 1) % Nx),
                    ((j - 1) % Ny, i),
                    ((j + 1) % Ny, i),
                ]

                kij = k_node[j, i]
                w_sum = 0.0
                target = np.zeros(2, dtype=float)
                xij = P[0, j, i]
                yij = P[1, j, i]

                for jj, ii in neighbors:
                    xnb = P[0, jj, ii]
                    ynb = P[1, jj, ii]

                    if ii == 0 and i == Nx - 1:
                        xnb += dx_grid * Nx
                    elif ii == Nx - 1 and i == 0:
                        xnb -= dx_grid * Nx

                    if jj == 0 and j == Ny - 1:
                        ynb += dy_grid * Ny
                    elif jj == Ny - 1 and j == 0:
                        ynb -= dy_grid * Ny

                    wij = 0.5 * (kij + k_node[jj, ii])
                    target += wij * np.array([xnb, ynb])
                    w_sum += wij

                if w_sum > 0:
                    target /= w_sum
                    P_new[:, j, i] = (1.0 - omega) * np.array([xij, yij]) + omega * target
        P = P_new
    return P

def refine_interface_nodes_by_local_square(
    coarse_interface_node,
    Xn,
    Yn,
    fine_edge_points,
    dx_coarse,
    dy_coarse,
):
    """
    Keep a coarse interface node only if there is at least one fine-edge point
    inside the local square centered at the node.

    The square size is one coarse-cell size:
        x in [x_node - dx_coarse/2, x_node + dx_coarse/2]
        y in [y_node - dy_coarse/2, y_node + dy_coarse/2]

    This matches the idea:
    'check the square region from the node to nearby cell centers'.
    """
    refined = coarse_interface_node.copy()
    removed = 0
    Ny, Nx = coarse_interface_node.shape

    hx = 0.5 * dx_coarse
    hy = 0.5 * dy_coarse

    for j in range(Ny):
        for i in range(Nx):
            if not coarse_interface_node[j, i]:
                continue

            x_node = Xn[j, i]
            y_node = Yn[j, i]

            x0 = x_node - hx
            x1 = x_node + hx
            y0 = y_node - hy
            y1 = y_node + hy

            in_square = (
                (fine_edge_points[:, 0] >= x0) & (fine_edge_points[:, 0] <= x1) &
                (fine_edge_points[:, 1] >= y0) & (fine_edge_points[:, 1] <= y1)
            )

            if not np.any(in_square):
                refined[j, i] = False
                removed += 1

    return refined, removed

def majority_downsample_phase_labels(
    fine_phase_labels: np.ndarray,
    coarse_Nx: int,
    coarse_Ny: int,
):
    """Downsample multiclass phase labels by majority voting.

    Parameters
    ----------
    fine_phase_labels:
        Fine-grid categorical labels with shape ``(ny, nx)``.
        Labels may be 0, 1, 2, 3, ...
    coarse_Nx, coarse_Ny:
        Number of coarse cells in x and y directions.

    Returns
    -------
    coarse_phase_label:
        Majority label for each coarse cell,
        shape ``(coarse_Ny, coarse_Nx)``.
    coarse_phase_confidence:
        Fraction of the winning label in each coarse cell,
        shape ``(coarse_Ny, coarse_Nx)``.
    """
    fine_phase_labels = np.asarray(fine_phase_labels)

    if fine_phase_labels.ndim != 2:
        raise ValueError(
            "fine_phase_labels must be a 2-D array, "
            f"but got shape {fine_phase_labels.shape}."
        )

    ny, nx = fine_phase_labels.shape

    x_idx = np.linspace(
        0,
        nx,
        coarse_Nx + 1,
        dtype=int,
    )
    y_idx = np.linspace(
        0,
        ny,
        coarse_Ny + 1,
        dtype=int,
    )

    coarse_phase_label = np.zeros(
        (coarse_Ny, coarse_Nx),
        dtype=fine_phase_labels.dtype,
    )

    coarse_phase_confidence = np.zeros(
        (coarse_Ny, coarse_Nx),
        dtype=np.float32,
    )

    for j in range(coarse_Ny):
        for i in range(coarse_Nx):
            x0, x1 = x_idx[i], x_idx[i + 1]
            y0, y1 = y_idx[j], y_idx[j + 1]

            phase_patch = fine_phase_labels[y0:y1, x0:x1]

            if phase_patch.size == 0:
                raise ValueError(
                    f"Empty phase patch at coarse cell ({j}, {i})."
                )

            labels, counts = np.unique(
                phase_patch,
                return_counts=True,
            )

            winning_count = np.max(counts)

            # If there is a tie, np.unique returns sorted labels,
            # so this selects the smallest label deterministically.
            winning_label = labels[
                np.flatnonzero(counts == winning_count)[0]
            ]

            coarse_phase_label[j, i] = winning_label
            coarse_phase_confidence[j, i] = (
                winning_count / phase_patch.size
            )

    return coarse_phase_label, coarse_phase_confidence
def build_periodic_plot_coordinates(
    stored_deformed_coordinates: np.ndarray,
    Lx: float,
    Ly: float,
) -> np.ndarray:
    """Create periodic plotting coordinates from stored coordinates."""

    P = np.asarray(
        stored_deformed_coordinates,
        dtype=float,
    )

    if P.ndim != 3 or P.shape[0] != 2:
        raise ValueError(
            "Expected coordinates with shape (2, Ny, Nx), "
            f"but got {P.shape}."
        )

    _, Ny, Nx = P.shape

    plot_coordinates = np.zeros(
        (2, Ny + 1, Nx + 1),
        dtype=P.dtype,
    )

    # Stored grid
    plot_coordinates[:, :-1, :-1] = P

    # Right boundary: copy left displacement + Lx
    plot_coordinates[:, :-1, -1] = (
        P[:, :, 0]
        + np.array([Lx, 0.0])[:, None]
    )

    # Top boundary: copy bottom displacement + Ly
    plot_coordinates[:, -1, :-1] = (
        P[:, 0, :]
        + np.array([0.0, Ly])[:, None]
    )

    # Top-right corner: copy bottom-left displacement + Lx and Ly
    plot_coordinates[:, -1, -1] = (
        P[:, 0, 0]
        + np.array([Lx, Ly])
    )

    return plot_coordinates

PathLike = Union[str, Path]
def run_grid_adaptation_workflow(
    input_path: PathLike,
    coarse_Nx: int = 16,
    coarse_Ny: int = 16,
    Lx: float = 1.0,
    Ly: float = 1.0,
    relax_iters: int = 400,
    relax_omega: float = 0.35,
    relax_b: float = 0.5,
    verbose: bool = True,
) -> dict[str, np.ndarray]:
    """Run the complete image-to-deformed-grid workflow.

    Parameters
    ----------
    input_path:
        Path to a 2-D image saved as a NumPy ``.npy`` file.
    coarse_Nx, coarse_Ny:
        Number of coarse cells in the x and y directions.
    Lx, Ly:
        Physical domain lengths in the x and y directions.
    relax_iters:
        Number of spring-relaxation iterations.
    relax_omega:
        Relaxation step size.
    relax_b:
        Parameter controlling distance-dependent node stiffness.
    verbose:
        If True, print workflow information.

    Returns
    -------
    dict[str, np.ndarray]
        Dictionary containing the selected solver, plotting, phase, and
        interface outputs.

        ``coords_of_displaced_nodes`` has shape ``(2, coarse_Ny, coarse_Nx)``
        and is intended for muFFTTO.

        ``full_plot_coords_of_displaced_nodes`` has shape
        ``(2, coarse_Ny + 1, coarse_Nx + 1)`` and is intended for plotting.
    """
    input_path = Path(input_path)

    if not input_path.is_file():
        raise FileNotFoundError(
            f"Input file does not exist: {input_path}"
        )

    if coarse_Nx < 1 or coarse_Ny < 1:
        raise ValueError(
            "coarse_Nx and coarse_Ny must be positive integers."
        )

    if Lx <= 0 or Ly <= 0:
        raise ValueError("Lx and Ly must be positive.")

    if relax_iters < 0:
        raise ValueError("relax_iters must be non-negative.")

    data = np.load(input_path).astype(np.float32)

    if data.ndim != 2:
        raise ValueError(
            "Expected a 2-D input image, "
            f"but got shape {data.shape}."
        )

    if verbose:
        print(f"Loaded data successfully: {input_path}")

    # ------------------------------------------------------------------
    # Fine-grid Otsu detection
    # ------------------------------------------------------------------
    start_time = time.time()
    results = otsu_edgeDetection_and_phaseIndicator(data)
    elapsed_time = time.time() - start_time

    edge_mask = results["edge_mask"]
    phase_mask_binary = results["phase_mask_binary"]
    phase_mask_label = results["phase_mask_label"]

    if verbose:
        print(f"Otsu processing time: {elapsed_time:.4f} s")
        print(f"Data shape: {data.shape}")
        print(f"Edge-mask shape: {edge_mask.shape}")
        print(f"Phase-label shape: {phase_mask_label.shape}")
        print(
            "Fine phase labels:",
            np.unique(phase_mask_label),
        )

        if "number_of_phase_regions" in results:
            print(
                "Number of phase regions:",
                results["number_of_phase_regions"],
            )

    ny, nx = data.shape
    edge_binary = (edge_mask > 0).astype(np.uint8)

    # ------------------------------------------------------------------
    # Fine edge points
    # ------------------------------------------------------------------
    edge_idx_y, edge_idx_x = np.nonzero(edge_binary)

    edge_pts_x = (edge_idx_x + 0.5) / nx * Lx
    edge_pts_y = (edge_idx_y + 0.5) / ny * Ly

    fine_edge_points = np.column_stack(
        (edge_pts_x, edge_pts_y)
    )

    if fine_edge_points.shape[0] == 0:
        raise ValueError(
            "No fine edge points were found in edge_mask."
        )

    # ------------------------------------------------------------------
    # Regular coarse grid
    # ------------------------------------------------------------------
    P0_coarse, Xn, Yn = build_regular_coarse_grid(
        coarse_Nx,
        coarse_Ny,
        Lx,
        Ly,
    )

    dx_coarse = Lx / coarse_Nx
    dy_coarse = Ly / coarse_Ny

    x_lines = np.linspace(0, Lx, coarse_Nx + 1)
    y_lines = np.linspace(0, Ly, coarse_Ny + 1)

    x_idx = np.linspace(
        0,
        nx,
        coarse_Nx + 1,
        dtype=int,
    )
    y_idx = np.linspace(
        0,
        ny,
        coarse_Ny + 1,
        dtype=int,
    )

    # ------------------------------------------------------------------
    # Coarse multiclass phase labels by majority vote
    # ------------------------------------------------------------------
    coarse_phase_label, _coarse_phase_confidence = (
        majority_downsample_phase_labels(
            fine_phase_labels=phase_mask_label,
            coarse_Nx=coarse_Nx,
            coarse_Ny=coarse_Ny,
        )
    )

    # ------------------------------------------------------------------
    # Coarse interface cells from the fine edge mask
    # ------------------------------------------------------------------
    coarse_interface_cell = np.zeros(
        (coarse_Ny, coarse_Nx),
        dtype=bool,
    )

    for j in range(coarse_Ny):
        for i in range(coarse_Nx):
            x0, x1 = x_idx[i], x_idx[i + 1]
            y0, y1 = y_idx[j], y_idx[j + 1]

            edge_patch = edge_binary[y0:y1, x0:x1]
            coarse_interface_cell[j, i] = np.any(edge_patch)

    # ------------------------------------------------------------------
    # Coarse interface nodes from surrounding interface cells
    # ------------------------------------------------------------------
    coarse_interface_node = np.zeros(
        (coarse_Ny, coarse_Nx),
        dtype=bool,
    )

    for j in range(coarse_Ny):
        for i in range(coarse_Nx):
            surrounding_cells = [
                coarse_interface_cell[
                    (j - 1) % coarse_Ny,
                    (i - 1) % coarse_Nx,
                ],
                coarse_interface_cell[
                    (j - 1) % coarse_Ny,
                    i % coarse_Nx,
                ],
                coarse_interface_cell[
                    j % coarse_Ny,
                    (i - 1) % coarse_Nx,
                ],
                coarse_interface_cell[
                    j % coarse_Ny,
                    i % coarse_Nx,
                ],
            ]

            coarse_interface_node[j, i] = np.any(
                surrounding_cells
            )

    # ------------------------------------------------------------------
    # Remove interface nodes that are not sufficiently close to fine edge
    # ------------------------------------------------------------------
    (
        coarse_interface_node_refined,
        removed_count,
    ) = refine_interface_nodes_by_local_square(
        coarse_interface_node=coarse_interface_node,
        Xn=Xn,
        Yn=Yn,
        fine_edge_points=fine_edge_points,
        dx_coarse=dx_coarse,
        dy_coarse=dy_coarse,
    )

    removed_nodes_mask = (
        coarse_interface_node
        & (~coarse_interface_node_refined)
    )

    if verbose:
        print(
            "Original interface nodes:",
            np.count_nonzero(coarse_interface_node),
        )
        print(
            "Refined interface nodes:",
            np.count_nonzero(
                coarse_interface_node_refined
            ),
        )
        print("Removed nodes:", removed_count)

    # ------------------------------------------------------------------
    # Project refined interface nodes to nearest fine edge points
    # ------------------------------------------------------------------
    interface_node_indices = np.argwhere(
        coarse_interface_node_refined
    )

    coarse_node_points = np.column_stack(
        (
            Xn[coarse_interface_node_refined],
            Yn[coarse_interface_node_refined],
        )
    )

    projected_points = np.zeros_like(coarse_node_points)
    projection_distance = np.zeros(
        coarse_node_points.shape[0],
        dtype=float,
    )

    for k, point in enumerate(coarse_node_points):
        difference = fine_edge_points - point[None, :]
        distance_squared = np.sum(difference**2, axis=1)
        nearest_index = np.argmin(distance_squared)

        projected_points[k] = fine_edge_points[nearest_index]
        projection_distance[k] = np.sqrt(
            distance_squared[nearest_index]
        )

    if verbose:
        print(
            "Projected interface nodes:",
            projected_points.shape[0],
        )

        if projection_distance.size:
            print("Projection distance statistics:")
            print("  min  =", projection_distance.min())
            print("  max  =", projection_distance.max())
            print("  mean =", projection_distance.mean())

    # ------------------------------------------------------------------
    # Set projected interface nodes and fixed boundaries
    # ------------------------------------------------------------------
    P_init = P0_coarse.copy()

    for k, (j, i) in enumerate(interface_node_indices):
        is_fixed_boundary = (j == 0) or (i == 0)

        if not is_fixed_boundary:
            P_init[0, j, i] = projected_points[k, 0]
            P_init[1, j, i] = projected_points[k, 1]

    fixed_mask = coarse_interface_node_refined.copy()
    fixed_mask[0, :] = True
    fixed_mask[:, 0] = True

    # ------------------------------------------------------------------
    # Distance-dependent stiffness and spring relaxation
    # ------------------------------------------------------------------
    dist_to_interface = periodic_manhattan_distance_to_mask(
        fixed_mask
    )

    k_node = stiffness_from_distance(
        dist_to_interface,
        k0=1.0,
        b=relax_b,
        a=1.0,
        kmin=0.05,
    )

    P_relaxed = spring_relax_projected_grid(
        P_init=P_init,
        fixed_mask=fixed_mask,
        k_node=k_node,
        iters=relax_iters,
        omega=relax_omega,
    )

    # ------------------------------------------------------------------
    # Periodically extended coordinates for plotting only
    # ------------------------------------------------------------------
    full_plot_coords_of_displaced_nodes = make_plot_coords(
        P_relaxed,
        Lx=Lx,
        Ly=Ly,
    )
    deformed_grid_coordinates_for_plot = (
        build_periodic_plot_coordinates(
            stored_deformed_coordinates=P_relaxed,
            Lx=Lx,
            Ly=Ly,
        )
    )

    if verbose:
        displacement = P_relaxed - P0_coarse

        print(
            "Stored deformed grid shape:",
            P_relaxed.shape,
        )
        print(
            "Plotting deformed grid shape:",
            full_plot_coords_of_displaced_nodes.shape,
        )
        print(
            "Coarse phase-label shape:",
            coarse_phase_label.shape,
        )
        print(
            "Coarse phase labels:",
            np.unique(coarse_phase_label),
        )
        print(
            "ux min/max:",
            displacement[0].min(),
            displacement[0].max(),
        )
        print(
            "uy min/max:",
            displacement[1].min(),
            displacement[1].max(),
        )

    return {
        "coords_of_displaced_nodes": P_relaxed,
        "full_plot_coords_of_displaced_nodes": (
            deformed_grid_coordinates_for_plot
        ),
        "phase_mask_label": phase_mask_label,
        "coarse_phase_label": coarse_phase_label,
        "coarse_interface_cell": coarse_interface_cell,
        "phase_mask_binary": phase_mask_binary,
        "edge_mask": edge_mask,
    }



if __name__ == "__main__":
    input_file = (
        r'C:\Users\Test\Desktop\JiaLing\HiWi\Simulation\Grain Boundaries Data\Green_Jacobi_eta_0.01_w_10.0_p_0.0_final.npy'
    )
    deformed_grid = run_grid_adaptation_workflow(input_file)