"""
Plotting helpers for two-dimensional muFFTTO results.

Functions
---------
plot_field_on_grid
    Plot a pixel-wise constant field on a (possibly deformed) quadrilateral grid
    using ``matplotlib.pyplot.pcolormesh``.
get_deformed_grid_coords_two_dim
    Compute the node coordinates of a deformed 2D periodic grid (including the
    periodic boundary nodes) from a macroscopic gradient, a displacement
    fluctuation and an optional grid-adaptation displacement.
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
