import matplotlib.pyplot as plt

import numpy as np


def plot_field_on_grid(
        coordinates_for_plot: np.ndarray,
        field_to_plot: np.ndarray,
        name='field'):
    """
    Function that generates 2D plot with grid lines. Pixel wise constant data
    works with numpy files
    x_plot has shape (number_of_pixels[0]+1, number_of_pixels[1]+1) # also periodic nodes
    field_to_plot has shape (number_of_pixels[0], number_of_pixels[1])
    """

    # plot def_F in grid
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    pcm = ax.pcolormesh(coordinates_for_plot[0], coordinates_for_plot[1],
                        field_to_plot, shading='flat', edgecolors='k',
                        cmap='coolwarm', lw=0.3)
    plt.colorbar(pcm, ax=ax)
    plt.xlabel('x  / L')
    plt.ylabel('y  / L')
    ax.set_aspect('equal')
    ax.set_title(name)
    plt.tight_layout()
    plt.show()


def get_deformed_grid_coords_two_dim(discretization,
                                     grid_nodes_displacement_inxyz,
                                     macro_gradient_ij,
                                     displacement_fluctuation):
    """
    This function calculates deformed grid coordinates.
    It uses original coordinates with periodic extension for plotting.
    Add grid-conforming deformation (grid_nodes_displacement_inxyz), linear macroscopic deformation from macro gradient
    and a displacement fluctuation (displacement_fluctuation)
    '

    :param discretization:
    :param grid_nodes_displacement_inxyz:
    :param macro_gradient_ij:
    :param displacement_fluctuation:
    :return:
    x_plot_ixyz positions of deformed grid nodes
    """

    # Reference coordinates with periodic extension for plotting
    x_plot_inxyz = discretization.get_nodal_points_coordinates_with_periodic_nodes()
    # add deformation  # x_p = x̃_p + ũ_Φ(x̃_p)
    x_plot_inxyz[..., :-1, :-1] += grid_nodes_displacement_inxyz.s[...]
    x_plot_ixyz = np.squeeze(x_plot_inxyz, axis=1)
    # macroscopic displacement of a deformed grid Ex_p = E * x_p
    macro_disp_of_a_deformed_grid = np.einsum('ij...,j...->i...', macro_gradient_ij, x_plot_ixyz)
    # add macroscopic displacement
    # u = x_p +  Ex_p
    x_plot_ixyz += macro_disp_of_a_deformed_grid
    # add  displacement fluctuation # u=Ex_p + ũ(x_p)
    x_plot_ixyz[..., :-1, :-1] += displacement_fluctuation.s[:, 0, ...]

    # add displacement fluctuation at periodic nodes
    x_plot_ixyz[0, -1, :-1] += displacement_fluctuation.s[0, 0, 0, :]
    x_plot_ixyz[0, :-1, -1] += displacement_fluctuation.s[0, 0, :, 0]
    x_plot_ixyz[0, -1, -1] += displacement_fluctuation.s[0, 0, 0, 0]
    #
    x_plot_ixyz[1, -1, :-1] += displacement_fluctuation.s[1, 0, 0, :]
    x_plot_ixyz[1, :-1, -1] += displacement_fluctuation.s[1, 0, :, 0]
    x_plot_ixyz[1, -1, -1] += displacement_fluctuation.s[1, 0, 0, 0]

    return x_plot_ixyz
