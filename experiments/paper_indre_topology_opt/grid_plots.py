import matplotlib as mpl
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import glob
import re

# Plotting parameters
plt.rcParams.update({
    "text.usetex": True,  # Use LaTeX
})
plt.rcParams.update({'font.size': 14})
plt.rcParams["font.family"] = "Arial"

# Setup parameters
N = 1024
nb_tiles = 5
random_init = False
cg_tol_exponent = 8
soft_phase_exponent = 5
eta_mult = 0.01
preconditioner_type = "Green_Jacobi"

weights = np.array([0.1,  0.3,  0.7,  1.0, 3.0, 7.0, 10.0, 30.0, 70.0, 100.0])

# Paths
script_dir = os.path.dirname(os.path.realpath(__file__))
script_name = f'exp_paper_TO_exp_4_hexa_random_{random_init}_N_{N}_cgtol_{cg_tol_exponent}_soft_{soft_phase_exponent}'
data_folder_path = os.path.join(script_dir, 'exp_data', script_name)
figure_folder_path = os.path.join(script_dir, 'figures', script_name)

if not os.path.exists(figure_folder_path):
    os.makedirs(figure_folder_path)

# Tiling logic
x_ref = np.zeros([2, nb_tiles * N + 1, nb_tiles * N + 1])
x_ref[0], x_ref[1] = np.meshgrid(np.linspace(0, nb_tiles, nb_tiles * N + 1),
                                 np.linspace(0, nb_tiles, nb_tiles * N + 1), indexing='ij')
shift = 0.5 * np.linspace(0, nb_tiles, nb_tiles * N + 1)
x_coords = np.copy(x_ref)
x_coords[0] += shift[None, :] - 2
x_coords[1] *= np.sqrt(3) / 2

def get_latest_iteration_file(weight):
    prefix = f"{preconditioner_type}_eta_{eta_mult}_w_{weight:.1f}_iteration_"
    suffix = ".npy"
    highest_iteration = -1
    latest_file = None

    if os.path.exists(data_folder_path):
        for filename in os.listdir(data_folder_path):
            if filename.startswith(prefix) and filename.endswith(suffix):
                try:
                    iteration = int(filename[len(prefix):-len(suffix)])
                    if iteration > highest_iteration:
                        highest_iteration = iteration
                        latest_file = os.path.join(data_folder_path, filename)
                except ValueError:
                    continue
    return latest_file

def create_grid_plot():
    num_plots = len(weights)
    cols = 5
    rows = (num_plots + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    plt.subplots_adjust(hspace=0.3, wspace=0.1)
    axes = axes.flatten()

    for i, w_mult in enumerate(weights):
        ax = axes[i]
        file_path = get_latest_iteration_file(w_mult)
        
        if file_path is None or not os.path.exists(file_path):
            print(f"Warning: No data found for weight {w_mult:.1f}")
            ax.text(0.5, 0.5, f'w={w_mult:.1f}\nNot Found', ha='center', va='center')
            ax.axis('off')
            continue

        print(f"Loading {file_path}")
        phase_field = np.load(file_path, allow_pickle=True)
        
        # Tiling for plot
        pf_tiled = np.tile(phase_field, (nb_tiles, nb_tiles))
        
        # Downsample if N is too large to save memory
        if N > 256:
            step = N // 256
            pf_tiled = pf_tiled[::step, ::step]
            x_c0 = x_coords[0][::step, ::step]
            x_c1 = x_coords[1][::step, ::step]
        else:
            x_c0 = x_coords[0]
            x_c1 = x_coords[1]

        ax.pcolormesh(x_c0, x_c1, pf_tiled, cmap='Greys', shading='flat', rasterized=True)
        ax.set_aspect('equal')
        ax.set_title(f'$w={w_mult:.1f}$')
        ax.axis('off')

    # Hide any unused axes
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    output_path = os.path.join(figure_folder_path, 'phase_field_grid.pdf')
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    print(f"Grid plot saved to {output_path}")

    # Also save as PNG for easier viewing
    output_path_png = os.path.join(figure_folder_path, 'phase_field_grid.png')
    plt.savefig(output_path_png, bbox_inches='tight', dpi=300)
    print(f"Grid plot saved to {output_path_png}")

if __name__ == "__main__":
    create_grid_plot()
