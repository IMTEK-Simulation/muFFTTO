import os

import numpy as np
import scipy as sc
import time
import sys
import matplotlib as mpl

from matplotlib import pyplot as plt

sys.path.append('..')  # Add parent directory to path

from mpi4py import MPI
from NuMPI.IO import save_npy, load_npy

from muFFTTO import domain
from muFFTTO import solvers
from muFFTTO import microstructure_library

plt.rcParams.update({
    "text.usetex": True,  # Use LaTeX
    # "font.family": "helvetica",  # Use a serif font
})
plt.rcParams.update({'font.size': 12})
plt.rcParams["font.family"] = "Arial"

if MPI.COMM_WORLD.size == 1:
    plot = True
# Parse command line arguments first to get grid_type
import argparse

parser = argparse.ArgumentParser(description='Compute deformation for different poison_target and weight values')
parser.add_argument('--poison_target', type=float, default=  -0.3, help='Poison target value')
parser.add_argument('--weight', type=float, default=20.0, help='Weight value')
parser.add_argument('--grid_type', type=str, default='square', choices=['hex', 'square'], help='Grid type')
args = parser.parse_args()

grid_type = args.grid_type

problem_type = 'elasticity'
discretization_type = 'finite_element'
formulation = 'small_strain'

if grid_type == 'hex':
    element_type = 'linear_triangles_tilled'

    domain_size = [1, np.sqrt(3) / 2]
elif grid_type == 'square':
    element_type = 'linear_triangles'

    domain_size = [1, 1]
nb_tiles = 1
number_of_pixel_PUC = 1024  # this is resolution of original unit cell
number_of_pixels = 2 * (nb_tiles * number_of_pixel_PUC,)  # this is resolution with repeated images

my_cell = domain.PeriodicUnitCell(domain_size=domain_size,
                                  problem_type=problem_type)

discretization = domain.Discretization(cell=my_cell,
                                       nb_of_pixels_global=number_of_pixels,
                                       discretization_type=discretization_type,
                                       element_type=element_type)
start_time = time.time()
print(f'{MPI.COMM_WORLD.rank:6} {MPI.COMM_WORLD.size:6} {str(discretization.fft.nb_domain_grid_pts):>15} '
      f'{str(discretization.fft.nb_subdomain_grid_pts):>15} {str(discretization.fft.subdomain_locations):>15}')

phase_field = discretization.get_scalar_field(name='phase_field')

eta_mult = 0.01
N = number_of_pixel_PUC
cg_tol_exponent = 8
soft_phase_exponent = 5
random_init = False

# Get parsed arguments
poison_target = args.poison_target
weight = args.weight

experiment = 5
if grid_type == 'hex':
    script_name = f'exp_paper_TO_exp_{experiment}_hexa' + f'_random_{random_init}' + f'_N_{N}' + f'_cgtol_{cg_tol_exponent}' + f'_soft_{soft_phase_exponent}'
elif grid_type == 'square':
    script_name = f'exp_paper_TO_exp_{experiment}_square' + f'_random_{random_init}' + f'_N_{N}' + f'_cgtol_{cg_tol_exponent}' + f'_soft_{soft_phase_exponent}'

preconditioner_type_data = "Green_Jacobi"  # Green_Jacobi
preconditioner_type = "Green"  # Green_Jacobi

file_folder_path = os.path.dirname(os.path.realpath(__file__))  # script directory
data_folder_path = file_folder_path + '/exp_data/' + script_name + '/'
figure_folder_path = file_folder_path + '/figures/' + script_name + '/'
if not os.path.exists(figure_folder_path):
    os.makedirs(figure_folder_path)

name = data_folder_path + f'{preconditioner_type_data}' + f'_eta_{eta_mult}' + f'_w_{weight:.1f}' + f'_p_{poison_target}' + '_final' + f'.npy'
name_tilled = data_folder_path + f'{preconditioner_type_data}' + f'_eta_{eta_mult}' + f'_w_{weight:.1f}' + f'_p_{poison_target}' + f'_final_tilled{nb_tiles}' + f'.npy'
name_log = data_folder_path + f'{preconditioner_type_data}' + f'_eta_{eta_mult}' + f'_w_{weight:.1f}' + f'_p_{poison_target}' + f'_log.npz'  # + '_final'
# Load, tile, and save in serial computation only (rank 0)
if MPI.COMM_WORLD.rank == 0:
    if not os.path.exists(name_tilled) and os.path.exists(name):
        if not os.path.exists(data_folder_path):
            os.makedirs(data_folder_path)

        # Load original unit cell
        phase_field_original = np.load(name)

        # Tile the field
        phase_field_tilled = np.tile(phase_field_original, (nb_tiles, nb_tiles))

        # Save tiled field
        np.save(name_tilled, phase_field_tilled)
        print(f'Tiled phase field saved to: {name_tilled}')

# Wait for rank 0 to finish tiling and saving
MPI.COMM_WORLD.Barrier()

# Now all ranks can load the tiled field
if os.path.exists(name_tilled):
    phase_field.s[0, 0] = load_npy(name_tilled,
                                   subdomain_locations=tuple(discretization.subdomain_locations_no_buffers),
                                   nb_subdomain_grid_pts=tuple(discretization.nb_of_pixels),
                                   components_are_leading=True,
                                   comm=MPI.COMM_WORLD)
    info_ = np.load(name_log, allow_pickle=True)
    C_eff_ij = info_.f.homogenized_C_ijkl

# remove layer
mask_free_layer = np.zeros_like(phase_field.s, dtype=bool)

dx = discretization.fft.coords[0, 1, 0] - discretization.fft.coords[0, 0, 0]
mask_free_layer[0, 0, discretization.fft.icoords[1] == 0] = 1
mask_free_layer[0, 0, discretization.fft.icoords[1] == 1] = 1
mask_free_layer[0, 0, discretization.fft.icoords[1] == number_of_pixels[0] - 1] = 1

# phase_field.s[mask_free_layer]=0
phase_field.s[phase_field.s >= 0.5] = 1
phase_field.s[phase_field.s < 0.5] = 0
discretization.fft.communicate_ghosts(phase_field)


from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset

plot_deformation_with_insets = True
if plot_deformation_with_insets:
    fig = plt.figure(figsize=(11, 4.0))  # Height for 2 rows8.3
    if grid_type == 'hex':
        gs = fig.add_gridspec(2, 4, hspace=0.05, wspace=0.05, width_ratios=[1, 1, 1, 1], height_ratios=[1, 1. ])
    else:
        gs = fig.add_gridspec(2, 4, hspace=0.2, wspace=0.0 , width_ratios=[1, 1, 1, 1], height_ratios=[1, 1.])

    axes = [fig.add_subplot(gs[0, i]) for i in range(4)]
    axes_insets = [fig.add_subplot(gs[1, i]) for i in range(4)]

    # Select 3 load increments to plot
    load_increments_to_plot = [0.0, 0.3, 0.7, 1.0]

    for plot_idx, load_increment in enumerate(load_increments_to_plot):
        inc_index = int(load_increment * 10)

        file_data_name_it = f'_w_{weight}' + f'_p_{poison_target}' + f'_load_increment_{load_increment}' + f'_tilled{nb_tiles}'

        displacement = load_npy(data_folder_path + f'{preconditioner_type_data}' + file_data_name_it + f'.npy',
                                subdomain_locations=tuple(discretization.subdomain_locations_no_buffers),
                                nb_subdomain_grid_pts=tuple(discretization.nb_of_pixels),
                                components_are_leading=True,
                                comm=MPI.COMM_WORLD)

        _info = np.load(data_folder_path + f'{preconditioner_type_data}' + file_data_name_it + f'_log_plotting.npz',
                        allow_pickle=True)
        macro_gradient = _info.f.macro_grads_corrected
        print(f'macro_gradient = {macro_gradient}')

        repetition = 3
        x_ref = np.zeros([2, repetition * (N) + 1, repetition * (N) + 1])
        x_ref[0], x_ref[1] = np.meshgrid(np.linspace(0, repetition, repetition * (N) + 1),
                                         np.linspace(0, repetition, repetition * (N) + 1), indexing='ij')
        shift = 0.5 * np.linspace(0, repetition, repetition * (N) + 1)

        x_coords = np.copy(x_ref)
        if grid_type == 'hex':
            x_coords[0] += shift[None, :]
            x_coords[1] *= np.sqrt(3) / 2

        lin_disp_ixy = np.einsum('ij...,j...->i...', macro_gradient, x_coords)

        x_coords[0] += lin_disp_ixy[0]
        x_coords[1] += lin_disp_ixy[1]

        x_coords[0, :-1, :-1] += np.tile(displacement[0], (repetition, repetition))
        x_coords[1, :-1, :-1] += np.tile(displacement[1], (repetition, repetition))

        tilled_disp_x = np.tile(displacement[0], (repetition, repetition))
        tilled_disp_y = np.tile(displacement[1], (repetition, repetition))

        x_coords[0, -1, :-1] += tilled_disp_x[0, :]
        x_coords[0, :-1, -1] += tilled_disp_x[:, 0]
        x_coords[0, -1, -1] += tilled_disp_x[0, 0]

        x_coords[1, -1, :-1] += tilled_disp_y[0, :]
        x_coords[1, :-1, -1] += tilled_disp_y[:, 0]
        x_coords[1, -1, -1] += tilled_disp_y[0, 0]

        # Main plot
        ax = axes[plot_idx]
        pcm = ax.pcolormesh(x_coords[0], x_coords[1], np.tile(phase_field.s[0, 0], (repetition, repetition)),
                            shading='flat',
                            edgecolors='none',
                            lw=0.01,
                            cmap=mpl.cm.Greys,
                            vmin=0, vmax=1,
                            rasterized=True)

        if grid_type == 'hex':
            ax.set_xlim(-0.0, repetition + repetition * 0.8)
            ax.set_ylim(-0.0, repetition - repetition * 0.1)
        else:
            ax.set_xlim(-0.0, repetition + repetition * 0.2)
            ax.set_ylim(-0.0, repetition + repetition * 0.1)

        ax.set_title(rf'$\overline{{\varepsilon}}_1$ = {0.2 * load_increment:.2f}')
        if grid_type == 'hex':
            ax.set_xticks([0, 1, 2, 3])
            ax.set_yticks([0, 1, 2])
        else:
            ax.set_xticks([0, 1, 2, 3])
            ax.set_yticks([0, 1, 2, 3])
        # ax.set_xlabel('Unit cell size  -  L')

        if plot_idx == 0:
            ax.set_ylabel('Unit cell size  - L')
        ax.yaxis.set_label_position("left")
        ax.set_aspect('equal')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)

        # Determine hinge point location
        point_hinge = None
        # Define zoom window size around blue circles
        zoom_size = 0.4  # Adjust this to control zoom level

        if grid_type == 'hex':
            if weight == 20 and poison_target == 0.2:
                point_hinge = np.array([2.3, 1.06])
                point_hinge = point_hinge + macro_gradient @ point_hinge
            if weight == 20 and poison_target == -0.3:
                point_hinge = np.array([1.82, 1.09])
                point_hinge = point_hinge + macro_gradient @ point_hinge
            if weight == 20 and poison_target == -0.5:
                point_hinge = np.array([2.055, 0.92])
                point_hinge = point_hinge + macro_gradient @ point_hinge
            if weight == 10 and poison_target == -0.5:
                point_hinge = np.array([1.82, 1.07])#single
                #point_hinge = np.array([1.95, 1.12])

                point_hinge = point_hinge + macro_gradient @ point_hinge
            if weight == 10 and poison_target == 0.1:
                point_hinge = np.array([1.56, 0.93]) #single
               # point_hinge = np.array([1.8, 0.8])
                point_hinge = point_hinge + macro_gradient @ point_hinge
        else:
            if weight == 20 and poison_target == -0.5:
                point_hinge = np.array([1.8, 1.0])
                point_hinge = point_hinge + macro_gradient @ point_hinge
                zoom_size = 0.5

            if weight == 20 and poison_target == -0.3:
                point_hinge = np.array([1.37, 0.65])
                point_hinge = point_hinge + macro_gradient @ point_hinge

        import string

        if grid_type == 'hex':
            letter_offset = 0.1
        else:
            letter_offset = -0.05
        letter = string.ascii_lowercase[plot_idx]
        ax.text(letter_offset, 1.1, rf'$\mathbf{{({letter})}}$', transform=ax.transAxes)

        # Create zoomed detail in second row
        if point_hinge is not None:
            axins = axes_insets[plot_idx]
            tilled_phase = np.tile(phase_field.s[0, 0], (repetition, repetition))
            # Plot the same data in the inset
            axins.pcolormesh(x_coords[0], x_coords[1], tilled_phase,
                             shading='flat',
                             edgecolors='none',
                             lw=0.01,
                             cmap=mpl.cm.Greys,
                             vmin=0, vmax=1,
                             rasterized=True)

            # Triangle vertices
            if grid_type == 'hex':
                if weight == 20 and poison_target == 0.2:
                    A = point_hinge + (0.0, 0.03)
                    B = point_hinge + (-0.08, -0.05)
                    C = point_hinge + (0.08, -0.05)
                    # Draw the three edges
                    # axins.plot([A[0], B[0]], [A[1], B[1]], 'r--', lw=1)
                    # axins.plot([B[0], C[0]], [B[1], C[1]], 'r--', lw=1)
                    # axins.plot([C[0], A[0]], [C[1], A[1]], 'r--', lw=1)

                if weight == 20 and poison_target == -0.3:
                    A = point_hinge + (0.00, 0.03)
                    B = point_hinge + (-0.08, -0.05)
                    C = point_hinge + (0.08, -0.05)
                    # Draw the three edges
                    # axins.plot([A[0], B[0]], [A[1], B[1]], 'r--', lw=1)
                    # axins.plot([B[0], C[0]], [B[1], C[1]], 'r--', lw=1)
                    # axins.plot([C[0], A[0]], [C[1], A[1]], 'r--', lw=1)
                if weight == 20 and poison_target == -0.5:
                    A = point_hinge + (0.00, 0.03)
                    B = point_hinge + (-0.08, -0.05)
                    C = point_hinge + (0.08, -0.05)

                    # Draw the three edges
                    # axins.plot([A[0], B[0]], [A[1], B[1]], 'r--', lw=1)
                    # axins.plot([B[0], C[0]], [B[1], C[1]], 'r--', lw=1)
                    # axins.plot([C[0], A[0]], [C[1], A[1]], 'r--', lw=1)
                if weight == 10 and poison_target == -0.5:
                    A = point_hinge + (0.00, 0.03)
                    B = point_hinge + (-0.08, -0.05)
                    C = point_hinge + (0.08, -0.05)
                if weight == 10 and poison_target == 0.1:
                    A = point_hinge + (0.00, 0.03)
                    B = point_hinge + (-0.08, -0.05)
                    C = point_hinge + (0.08, -0.05)
            # Create mask for points inside triangle
            def point_in_triangle(x, y, A, B, C):
                """Check if point (x,y) is inside triangle ABC using barycentric coordinates"""

                def sign(p1, p2, p3):
                    return (p1[0] - p3[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p3[1])

                d1 = sign((x, y), A, B)
                d2 = sign((x, y), B, C)
                d3 = sign((x, y), C, A)

                has_neg = (d1 < 0) | (d2 < 0) | (d3 < 0)
                has_pos = (d1 > 0) | (d2 > 0) | (d3 > 0)

                return ~(has_neg & has_pos)


            def create_cross_mask(X, Y, center, length=0.07, width=0.01):
                """Create mask for cross (horizontal and vertical lines) at center point"""
                # Horizontal line mask
                mask_horizontal = (np.abs(Y - center[1]) < width) & (np.abs(X - center[0]) < length)
                # Vertical line mask
                mask_vertical = (np.abs(X - center[0]) < width) & (np.abs(Y - center[1]) < length)
                # Combine both lines
                mask_cross = mask_horizontal | mask_vertical
                return mask_cross


            if plot_idx == 0 and grid_type == 'hex':
                mask_marker = point_in_triangle(x_coords[0], x_coords[1], A, B, C)
                #mask_marker_left = create_cross_mask(x_coords[0], x_coords[1], center=point_hinge + (0., -0.05) ,
                #                                    width=0.01)
                # mask_marker_right = create_cross_mask(x_coords[0], x_coords[1], center=point_hinge + (0.12, -0.02),
                #                                       width=0.01)
                #mask_marker = mask_marker_left #| mask_marker_right
            if plot_idx == 0 and grid_type == 'square' and weight == 20 and poison_target == -0.3:
                mask_marker_left = create_cross_mask(x_coords[0], x_coords[1], center=point_hinge + (-0.12, -0.02),
                                                     width=0.01)
                mask_marker_right = create_cross_mask(x_coords[0], x_coords[1], center=point_hinge + (0.12, -0.02),
                                                      width=0.01)
                mask_marker = mask_marker_left | mask_marker_right
            if plot_idx == 0 and grid_type == 'square' and weight == 20 and poison_target == -0.5:
                mask_marker_bottom = create_cross_mask(x_coords[0], x_coords[1], center=point_hinge + (-0.16, -0.22) )
                mask_marker_top= create_cross_mask(x_coords[0], x_coords[1], center=point_hinge + (-0.16, 0.05) )
                mask_marker = mask_marker_bottom | mask_marker_top
            if grid_type == 'square' and weight == 20 and poison_target == -0.5:
                # Plot the circle around the hinge point
                axins.plot(point_hinge[0], point_hinge[1]+0.35, 'ro', linestyle='--', mew=1, ms=25, fillstyle='none')

            if plot_idx == 0 and grid_type == 'square' and weight == 10 and poison_target == -0.5:
                mask_marker_bottom = create_cross_mask(x_coords[0], x_coords[1], center=point_hinge + (-0.16, -0.22))
                mask_marker_top = create_cross_mask(x_coords[0], x_coords[1], center=point_hinge + (-0.16, 0.05))
                mask_marker = mask_marker_bottom | mask_marker_top
            if grid_type == 'square' and weight == 10 and poison_target == -0.5:
                # Plot the circle around the hinge point
                axins.plot(point_hinge[0], point_hinge[1] + 0.33, 'ro', linestyle='--', mew=1, ms=25, fillstyle='none')

            if plot_idx == 0 and grid_type == 'square' and weight == 10 and poison_target ==  0.1:
                mask_marker_bottom = create_cross_mask(x_coords[0], x_coords[1], center=point_hinge + (-0.16, -0.22))
                mask_marker_top = create_cross_mask(x_coords[0], x_coords[1], center=point_hinge + (-0.16, 0.05))
                mask_marker = mask_marker_bottom | mask_marker_top
            if grid_type == 'square' and weight == 10 and poison_target == 0.1:
                # Plot the circle around the hinge point
                axins.plot(point_hinge[0], point_hinge[1] + 0.33, 'ro', linestyle='--', mew=1, ms=25, fillstyle='none')


            # Apply mask to tilled_phase
            tilled_phase_masked = tilled_phase.copy()
            tilled_phase_masked[mask_marker[:-1, :-1]] = 2

            # add triangle with different color
            # Create custom colormap: white for 0, black for 1, red for 2
            from matplotlib.colors import ListedColormap

            colors = ['white', 'black', 'red']
            cmap_custom = ListedColormap(colors)
            axins.pcolormesh(x_coords[0], x_coords[1], tilled_phase_masked , # transpose because tilled phase masked is transpose.. no power to search why
                             shading='flat',
                             edgecolors='none',
                             lw=0.01,
                             cmap=cmap_custom,
                             vmin=0, vmax=2,
                             rasterized=True)

            # Set zoom limits
            axins.set_xlim(point_hinge[0] - zoom_size, point_hinge[0] + zoom_size)
            axins.set_ylim(point_hinge[1] - zoom_size, point_hinge[1] + zoom_size)

            # Set aspect ratio
            axins.set_aspect('equal')

            # Remove tick labels
            axins.set_xticks([])
            axins.set_yticks([])

            # Add border
            for spine in axins.spines.values():
                spine.set_edgecolor('blue')
                spine.set_linewidth(2)

            # Add label
            letter_bottom = string.ascii_uppercase[plot_idx ]
            axins.text(-0.15, 0.85, rf'$\mathbf{{{letter_bottom}}}$', transform=axins.transAxes)
            # Draw lines connecting inset to zoomed region
            mark_inset(ax, axins, loc1=1, loc2=2, fc="none", ec="blue", lw=1, ls='--')




        else:
            # Hide unused inset axes
            axes_insets[plot_idx].set_visible(False)

    plt.tight_layout()
    figure_name = figure_folder_path + f'grid_{grid_type}_comparison_w_{weight}_p_{poison_target}_N{N}_{nb_tiles}_with_insets.pdf'
    plt.savefig(figure_name, dpi=1200 , bbox_inches='tight')  #
    plt.close(fig)
    print(f'Figure saved: {figure_name}')
