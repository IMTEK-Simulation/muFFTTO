import os

import numpy as np
import scipy as sc
import time
import sys
import matplotlib as mpl
from matplotlib.animation import FFMpegWriter

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
parser.add_argument('--poison_target', type=float, default= -0.3, help='Poison target value')
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
number_of_pixel_PUC = 1024 # this is resolution of original unit cell
number_of_pixels = 2 * (nb_tiles*number_of_pixel_PUC,)# this is resolution with repeated images

my_cell = domain.PeriodicUnitCell(domain_size=domain_size,
                                  problem_type=problem_type)

discretization = domain.Discretization(cell=my_cell,
                                       nb_of_pixels_global=number_of_pixels,
                                       discretization_type=discretization_type,
                                       element_type=element_type)
start_time = time.time()
print(f'{MPI.COMM_WORLD.rank:6} {MPI.COMM_WORLD.size:6} {str(discretization.fft.nb_domain_grid_pts):>15} '
      f'{str(discretization.fft.nb_subdomain_grid_pts):>15} {str(discretization.fft.subdomain_locations):>15}')

K_0, G_0 = 1, 0.5

elastic_C_1 = domain.get_elastic_material_tensor(dim=discretization.domain_dimension,
                                                 K=K_0,
                                                 mu=G_0,
                                                 kind='linear')
print(domain.compute_Voigt_notation_4order(elastic_C_1))

material_data_field_C_0 = discretization.get_material_data_size_field_mugrid(name='elastic_tensor')

# populate the field with C_1 material
material_data_field_C_0.s = np.einsum('ijkl,qxy->ijklqxy', elastic_C_1,
                                      np.ones(np.array([discretization.nb_quad_points_per_pixel,
                                                        *discretization.nb_of_pixels])))
if MPI.COMM_WORLD.rank == 0:
    print('elastic tangent = \n {}'.format(domain.compute_Voigt_notation_4order(elastic_C_1)))

# material distribution
phase_field = discretization.get_scalar_field(name='phase_field')

eta_mult = 0.01
N = number_of_pixel_PUC
cg_tol_exponent = 8
soft_phase_exponent = 5
random_init = False

# Get parsed arguments
poison_target = args.poison_target
weight = args.weight

experiment=5
if grid_type == 'hex':
    script_name = f'exp_paper_TO_exp_{experiment}_hexa' + f'_random_{random_init}' + f'_N_{N}' + f'_cgtol_{cg_tol_exponent}' + f'_soft_{soft_phase_exponent}'
elif grid_type == 'square':
    script_name = f'exp_paper_TO_exp_{experiment}_square' + f'_random_{random_init}' + f'_N_{N}' + f'_cgtol_{cg_tol_exponent}' + f'_soft_{soft_phase_exponent}'



preconditioner_type_data = "Green_Jacobi" #Green_Jacobi
preconditioner_type = "Green" #Green_Jacobi

file_folder_path = os.path.dirname(os.path.realpath(__file__))  # script directory
data_folder_path = file_folder_path + '/exp_data/' + script_name + '/'
figure_folder_path = file_folder_path + '/figures/' + script_name + '/'
if not os.path.exists(figure_folder_path):
    os.makedirs(figure_folder_path)

name = data_folder_path + f'{preconditioner_type_data}' + f'_eta_{eta_mult}' + f'_w_{weight:.1f}' + f'_p_{poison_target}' + '_final' + f'.npy'
name_tilled = data_folder_path + f'{preconditioner_type_data}' + f'_eta_{eta_mult}' + f'_w_{weight:.1f}' + f'_p_{poison_target}' + f'_final_tilled{nb_tiles}' + f'.npy'
name_log =  data_folder_path + f'{preconditioner_type_data}' + f'_eta_{eta_mult}' + f'_w_{weight:.1f}' + f'_p_{poison_target}'  + f'_log.npz' #+ '_final'
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
    C_eff_ij=info_.f.homogenized_C_ijkl


# remove layer
mask_free_layer = np.zeros_like(phase_field.s, dtype=bool)


dx = discretization.fft.coords[0, 1, 0] - discretization.fft.coords[0, 0, 0]
mask_free_layer[0, 0,discretization.fft.icoords[1]== 0] = 1
mask_free_layer[0, 0,discretization.fft.icoords[1]== 1] = 1
mask_free_layer[0, 0, discretization.fft.icoords[1]== number_of_pixels[0]-1] = 1


# phase_field.s[mask_free_layer]=0
phase_field.s[phase_field.s >= 0.5] = 1
phase_field.s[phase_field.s < 0.5] = 0
discretization.fft.communicate_ghosts(phase_field)



if MPI.COMM_WORLD.rank == 0:
    fig = plt.figure(figsize=(8.3, 6.5))
    gs = fig.add_gridspec(1, 4, hspace=0.05, wspace=0.15,width_ratios=[1, 1, 1, 1])
    axes = [fig.add_subplot(gs[0, i]) for i in range(4)]

# Select 3 load increments to plot
load_increments_to_plot = [0.0, 0.3, 0.7, 1.0]  # Adjust these values as needed

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

    if MPI.COMM_WORLD.rank == 0:
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

        # add fluctuation of
        x_coords[0, :-1, :-1] += np.tile(displacement[0], (repetition, repetition))
        x_coords[1, :-1, :-1] += np.tile(displacement[1], (repetition, repetition))

        # build a periodic displacement
        tilled_disp_x = np.tile(displacement[0], (repetition, repetition))
        tilled_disp_y = np.tile(displacement[1], (repetition, repetition))

        # Fill last row and column with first row and column
        x_coords[0, -1, :-1] += tilled_disp_x[0, :]
        x_coords[0, :-1, -1] += tilled_disp_x[:, 0]
        x_coords[0, -1, -1] += tilled_disp_x[0, 0]

        x_coords[1, -1, :-1] += tilled_disp_y[0, :]
        x_coords[1, :-1, -1] += tilled_disp_y[:, 0]
        x_coords[1, -1, -1] += tilled_disp_y[0, 0]

        ax = axes[plot_idx]
        pcm = ax.pcolormesh(x_coords[0], x_coords[1], np.tile(phase_field.s[0, 0], (repetition, repetition)),
                            shading='flat',
                            edgecolors='none',
                            lw=0.01,
                            cmap=mpl.cm.Greys,
                            vmin=0, vmax=1,
                            rasterized=True)



        #ax.xaxis.set_ticks_position('none')
        #ax.yaxis.set_ticks_position('none')

        if grid_type == 'hex':
            ax.set_xlim(-0.05, repetition + repetition * 0.8)
            ax.set_ylim(-0.05, repetition + repetition * 0.05)
        else:
            ax.set_xlim(-0.05, repetition + repetition * 0.2)
            ax.set_ylim(-0.05, repetition + repetition * 0.1)

        ax.set_title(rf'$\overline{{\varepsilon}}_1$ = {0.2*load_increment:.2f}')
        #ax.set_aspect('equal')

        ax.set_xticks([0, 1 , 2  , 3 ])
        ax.set_yticks([0, 1 , 2  , 3 ])

        ax.set_xticklabels([0, 1, 2, 3])
        ax.set_yticklabels([0, 1, 2, 3])
        # if j > 1:
        ax.set_xlabel('Unit cell size  -  L')

        if plot_idx == 0:
            ax.set_ylabel('L')
        ax.yaxis.set_label_position("left")
        #ax.yaxis.tick_right()

        # ax.set_xlim(0, 3 * N)
        # ax.set_ylim(0, 3 * N)

        ax.set_aspect('equal')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        import string
        if weight ==20 and poison_target == -0.5:
            point_cross=np.array([1.6, 0.77])
            point_cross =  point_cross+macro_gradient@point_cross
            ax.plot(point_cross[0], point_cross[1], 'r+', mew=1, ms=20)

            point_hinge=np.array([0.8, 0.35])
            point_hinge = point_hinge + macro_gradient @ point_hinge
            ax.plot(point_hinge[0], point_hinge[1], 'bo', mew=1, ms=20, fillstyle='none')
        if weight == 20 and poison_target == -0.3:
            point_cross = np.array([1.5, 0.67])
            point_cross = point_cross + macro_gradient @ point_cross
            ax.plot(point_cross[0], point_cross[1], 'r+', mew=1, ms=20)

            point_hinge = np.array([0.35, 0.65])
            point_hinge = point_hinge + macro_gradient @ point_hinge
            ax.plot(point_hinge[0], point_hinge[1], 'bo', mew=1, ms=20, fillstyle='none')
            #ax.plot([0.6, 2], [0.75, 2], 'r+', mew=2, ms=40)
        letter = string.ascii_uppercase[plot_idx]

        letter_offset = -0.15
      # 'A', 'B', 'C', ...
        ax.text(letter_offset, 1.05, rf'$\mathbf{{{letter}}}$', transform=ax.transAxes)
        # # Add grid lines
        # for i in range(nb_tiles + 1):
        #     if grid_type == 'hex':
        #         y_val = i * np.sqrt(3) / 2
        #         ax.plot([0 - 2, nb_tiles + 0.5 * nb_tiles - 2], [y_val, y_val],
        #                 color='k', linestyle='--', linewidth=1, alpha=0.5)
        #         x_start = i - 2
        #         x_end = i + 0.5 * nb_tiles - 2
        #         ax.plot([x_start, x_end], [0, nb_tiles * np.sqrt(3) / 2],
        #                 color='k', linestyle='--', linewidth=1, alpha=0.5)
        #     else:
        #         ax.axhline(y=i, color='k', linestyle='--', linewidth=1, alpha=0.5)
        #         ax.axvline(x=i, color='k', linestyle='--', linewidth=1, alpha=0.5)

if MPI.COMM_WORLD.rank == 0:
    plt.tight_layout()
    figure_name = figure_folder_path + f'grid_{grid_type}_comparison_w_{weight}_p_{poison_target}_N{N}_{nb_tiles}.pdf'
    plt.savefig(figure_name, dpi=450, bbox_inches='tight')
    plt.close(fig)
    print(f'Figure saved: {figure_name}')
  #  plt.show()

