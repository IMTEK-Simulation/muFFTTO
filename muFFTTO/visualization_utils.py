"""
Plotting helpers for muFFTTO results.

Functions
---------
plot_field_on_grid
    Plot a pixel-wise constant field on a (possibly deformed) quadrilateral grid
    using ``matplotlib.pyplot.pcolormesh``.
get_deformed_grid_coords_two_dim
    Compute the node coordinates of a deformed 2D periodic grid (including the
    periodic boundary nodes) from a macroscopic gradient, a displacement
    fluctuation and an optional grid-adaptation displacement.
visualize_voxels
    3D voxel plot of a phase field.
"""
import matplotlib.pyplot as plt

import numpy as np


def plot_field_on_grid(
        coordinates_for_plot: np.ndarray,
        field_to_plot: np.ndarray,
        name='field',
        plot_grid: bool = True):
    """
    Function that generates 2D plot with grid lines. Pixel wise constant data
    works with numpy files
    x_plot has shape (number_of_pixels[0]+1, number_of_pixels[1]+1) # also periodic nodes
    field_to_plot has shape (number_of_pixels[0], number_of_pixels[1])

    Parameters
    ----------
    coordinates_for_plot : numpy.ndarray
        Node coordinates of shape ``(2, Nx+1, Ny+1)``; ``coordinates_for_plot[0]``
        are the x- and ``coordinates_for_plot[1]`` the y-coordinates of all grid
        nodes, including the periodic copies on the right/top boundary (e.g. the
        output of :func:`get_deformed_grid_coords_two_dim`).
    field_to_plot : numpy.ndarray
        Pixel-wise constant values of shape ``(Nx, Ny)``; one colour per cell
        (``shading='flat'``).
    name : str, optional
        Title of the plot. Default ``'field'``.
    plot_grid : bool, optional
        If ``True`` (default), cell edges are drawn as thin black lines
        (line width 0.3); otherwise line width 0 (no visible grid).

    Returns
    -------
    None

    Notes
    -----
    Side effect: creates a new figure and calls ``plt.show()`` (blocking in
    interactive back-ends). Axis labels assume coordinates normalised by the
    cell size ``L``.
    """

    # plot def_F in grid
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    if plot_grid:
        grid_lines_width=0.3
    else:
        grid_lines_width=0
    # 'flat' shading: field_to_plot[i, j] colours the quadrilateral spanned by
    # nodes (i, j), (i+1, j), (i+1, j+1), (i, j+1) -> needs one more node per axis
    pcm = ax.pcolormesh(coordinates_for_plot[0], coordinates_for_plot[1],
                        field_to_plot, shading='flat', edgecolors='k',
                        cmap='coolwarm', lw=grid_lines_width)
    plt.colorbar(pcm, ax=ax)
    plt.xlabel('x  / L')
    plt.ylabel('y  / L')
    ax.set_aspect('equal')
    ax.set_title(name)
    plt.tight_layout()
    plt.show()


def get_deformed_grid_coords_two_dim(discretization,
                                     macro_gradient_ij,
                                     displacement_fluctuation,
                                     grid_nodes_displacement_inxyz=None):
    """
    This function calculates deformed grid coordinates.
    It uses original coordinates with periodic extension for plotting.

     linear macroscopic deformation from macro gradient
    and a displacement fluctuation (displacement_fluctuation),
    Add grid-conforming deformation (grid_nodes_displacement_inxyz),
    '

    The deformed position of node ``p`` is assembled in three steps::

        x_p  = x_ref_p + d_grid(x_ref_p)            # (optional) adapted grid
        x_p += E . x_p                              # macroscopic deformation
        x_p += u_fluct(x_p)                         # periodic fluctuation

    Parameters
    ----------
    discretization : muFFTTO.domain.Discretization
        Discretization object; only
        ``get_nodal_points_coordinates_with_periodic_nodes()`` is used, which
        returns reference coordinates of shape ``(2, n, Nx+1, Ny+1)`` on the
        unit square ``[0, 1]^2`` (normalised coordinates).
    macro_gradient_ij : numpy.ndarray
        Macroscopic displacement gradient ``E`` of shape ``(2, 2)``.
    displacement_fluctuation : muGrid.Field
        Periodic displacement fluctuation; ``.s`` of shape ``(2, n, Nx, Ny)``.
        Only nodal point ``n = 0`` is used.
    grid_nodes_displacement_inxyz : muGrid.Field, optional
        Displacement of grid nodes from a grid-adaptation step, ``.s`` of shape
        ``(2, n, Nx, Ny)``. If ``None`` (default), the regular reference grid
        is used.

    Returns
    -------
    x_plot_ixyz : numpy.ndarray
        x_plot_ixyz positions of deformed grid nodes, shape ``(2, Nx+1, Ny+1)``,
        including the periodic boundary nodes; suitable as
        ``coordinates_for_plot`` in :func:`plot_field_on_grid`.

    Notes
    -----
    Serial only: the periodic-node copies are taken from local index 0 of the
    fluctuation field, which assumes the whole grid is on one MPI rank.

    Units: the reference coordinates are normalised by the cell size (in
    ``[0, 1]``), while the fluctuation and the grid displacement are in
    physical units, so the result is only consistent for a unit cell
    (``domain_size = [1, 1]``). The periodic copy nodes (last row/column) do
    not receive the grid displacement.
    """

    # Reference coordinates with periodic extension for plotting
    x_plot_inxyz = discretization.get_nodal_points_coordinates_with_periodic_nodes()
    # add deformation  # x_p = x̃_p + ũ_Φ(x̃_p)
    # (only the interior/non-periodic nodes [:-1, :-1] receive the grid
    # displacement; the periodic copies in the last row/column are not shifted)
    if grid_nodes_displacement_inxyz is not None:
        x_plot_inxyz[..., :-1, :-1] += grid_nodes_displacement_inxyz.s[...]
    # keep only the first nodal point per pixel -> shape (2, Nx+1, Ny+1)
    x_plot_ixyz = x_plot_inxyz[:, 0, ...]
    # macroscopic displacement of a deformed grid Ex_p = E * x_p
    macro_disp_of_a_deformed_grid = np.einsum('ij...,j...->i...', macro_gradient_ij, x_plot_ixyz)
    # add macroscopic displacement
    # u = x_p +  Ex_p
    x_plot_ixyz += macro_disp_of_a_deformed_grid
    # add  displacement fluctuation # u=Ex_p + ũ(x_p)
    x_plot_ixyz[..., :-1, :-1] += displacement_fluctuation.s[:, 0, ...]

    # add displacement fluctuation at periodic nodes
    # periodicity: u(x = L) = u(x = 0), so the last row/column/corner copy the
    # fluctuation of the first row/column/corner (component 0 = u_x)
    x_plot_ixyz[0, -1, :-1] += displacement_fluctuation.s[0, 0, 0, :]
    x_plot_ixyz[0, :-1, -1] += displacement_fluctuation.s[0, 0, :, 0]
    x_plot_ixyz[0, -1, -1] += displacement_fluctuation.s[0, 0, 0, 0]
    # same for component 1 = u_y
    x_plot_ixyz[1, -1, :-1] += displacement_fluctuation.s[1, 0, 0, :]
    x_plot_ixyz[1, :-1, -1] += displacement_fluctuation.s[1, 0, :, 0]
    x_plot_ixyz[1, -1, -1] += displacement_fluctuation.s[1, 0, 0, 0]

    return x_plot_ixyz


def visualize_voxels(phase_field_xyz, figure=None, ax=None):
    """
    Plot a 3D phase field as coloured, semi-transparent voxels.

    Voxels with ``|value| / max|value| >= 0.1`` are drawn. Positive values
    are blue, negative values red, and the opacity of each voxel is
    ``|value| / max|value|``.

    Parameters
    ----------
    phase_field_xyz : np.ndarray, shape (Nx, Ny, Nz)
        Field to plot (e.g. the output of :func:`muFFTTO.geometry.get`).
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
    If ``ax`` is given, its figure is used and ``figure`` is ignored.
    """
    magnitude = np.abs(phase_field_xyz) / np.abs(phase_field_xyz).max()
    # draw only voxels with at least 10 % of the maximal magnitude
    visible = magnitude >= 0.1

    # RGBA colour per voxel (last axis): blue for positive, red for negative,
    # opacity proportional to the magnitude
    face_colors = np.zeros(list(phase_field_xyz.shape) + [4], dtype=np.float32)
    face_colors[phase_field_xyz > 0] = [0, 0, 1, 0]
    face_colors[phase_field_xyz < 0] = [1, 0, 0, 0]
    face_colors[..., -1] = magnitude

    if ax is not None:
        fig = ax.figure
    else:
        fig = plt.figure() if figure is None else figure
        ax = fig.add_subplot(projection='3d')
    ax.voxels(visible, facecolors=face_colors, edgecolor='k', linewidth=0.01)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    return fig, ax
