import os
import sys
import argparse
import numpy as np
from NuMPI.IO import load_npy
import matplotlib.pyplot as plt

# Add parent directory to path for muFFTTO imports
sys.path.append('../..')

def plot_results(iteration, total_strain_s, stress_s, figure_folder_path, nb_of_pixels_global, dmax_val, eps0_val):
    """
    Plots 2D slices of the 3D fields to visualize material property changes.
    """
    # Get middle slice index
    mid_z = nb_of_pixels_global[2] // 2
    domain_dimension = 3

    fig, axs = plt.subplots(1, 3, figsize=(15, 5))

    # 1. Equivalent Strain
    # total_strain_s shape is (3, 3, q, x, y, z)
    strain_trace = np.einsum('ii...', total_strain_s) / 3
    strain_vol = np.zeros_like(total_strain_s)
    for d in range(domain_dimension):
        strain_vol[d, d] = strain_trace
    strain_dev = total_strain_s - strain_vol
    strain_eq = np.sqrt((2. / 3.) * np.einsum('ij...,ji...->...', strain_dev, strain_dev))

    im0 = axs[0].imshow(strain_eq[0, mid_z, :, :], cmap='viridis')
    axs[0].set_title(f'Equivalent Strain (it {iteration})')
    plt.colorbar(im0, ax=axs[0])

    # 2. Stress component (e.g., sigma_xy)
    # stress_s shape is (3, 3, q, x, y, z)
    im1 = axs[1].imshow(stress_s[0, 1, 0, mid_z, :, :], cmap='magma')
    axs[1].set_title('Stress sigma_xy')
    plt.colorbar(im1, ax=axs[1])

    # 3. Damage Field
    damage = dmax_val * (1.0 - np.exp(-strain_eq / eps0_val))
    im2 = axs[2].imshow(damage[0, mid_z, :, :], cmap='Reds', vmin=0, vmax=dmax_val)
    axs[2].set_title('Damage Variable (d)')
    plt.colorbar(im2, ax=axs[2])
    
    plt.tight_layout()
    if not os.path.exists(figure_folder_path):
        os.makedirs(figure_folder_path)
    plt.savefig(f"{figure_folder_path}iteration_{iteration:02d}.png")
    plt.close()

def main():
    parser = argparse.ArgumentParser(
        prog="exp_paper_JG_softening_damage_JZ_bubles_plot.py",
        description="Load results and plot them."
    )
    parser.add_argument("-n", "--nb_pixel", default="32")
    parser.add_argument("-exp0", "--eps0_damage", default="0.05", type=float)
    parser.add_argument("-dmax", "--max_damage", default="0.99", type=float)
    parser.add_argument("-p", "--preconditioner_type", type=str, default="Green_Jacobi")
    
    args = parser.parse_args()
    nnn = int(args.nb_pixel)
    eps0_val = args.eps0_damage
    dmax_val = args.max_damage
    preconditioner_type = args.preconditioner_type
    
    script_name = "exp_paper_JG_softening_damage_JZ_bubles"
    Nx, Ny, Nz = nnn, nnn, nnn
    nb_of_pixels_global = (Nx, Ny, Nz)
    
    file_folder_path = os.path.dirname(os.path.realpath(__file__))
    data_folder_path = (
            file_folder_path + '/exp_data/' + script_name + '/' + f'Nx={Nx}' + f'Ny={Ny}' + f'Nz={Nz}'
            + f'_{preconditioner_type}' + '/')
    figure_folder_path = (
            file_folder_path + '/figures/' + script_name + '/' f'Nx={Nx}' + f'Ny={Ny}' + f'Nz={Nz}'
            + f'_{preconditioner_type}' + '/')
    
    if not os.path.exists(data_folder_path):
        print(f"Data folder {data_folder_path} does not exist.")
        return

    # Load info_log_final.npz to get total iterations
    info_path = data_folder_path + 'info_log_final.npz'
    if os.path.exists(info_path):
        info = np.load(info_path)
        iteration_total = info['iteration_total']
        print(f"Total iterations to plot: {iteration_total}")
    else:
        # Fallback: find max iteration by checking files
        iteration_total = 0
        while os.path.exists(data_folder_path + f'strain_it{iteration_total}.npy'):
            iteration_total += 1
        iteration_total -= 1
        print(f"Detected {iteration_total + 1} iterations to plot.")

    for it in range(iteration_total + 1):
        strain_file = data_folder_path + f'strain_it{it}.npy'
        stress_file = data_folder_path + f'stress_it{it}.npy'
        
        if os.path.exists(strain_file) and os.path.exists(stress_file):
            print(f"Plotting iteration {it}...")
            # Using load_npy from NuMPI.IO as used in original script
            # For plotting script, we probably run it on a single core, but we need to pass dummy subdomain info
            # or use np.load if it's a single file. NuMPI.IO.load_npy handles MPI.
            # However, if we saved it with save_npy, it might be partitioned if MPI was used.
            # To load it back as a whole, we might need NuMPI's load_npy with appropriate args.
            
            # Since we want a simple plot script, maybe we should assume it's loaded as one.
            # load_npy(fn, subdomain_locations, nb_pixels, comm)
            from mpi4py import MPI
            strain = load_npy(strain_file, (0,0,0), nb_of_pixels_global, MPI.COMM_SELF)
            stress = load_npy(stress_file, (0,0,0), nb_of_pixels_global, MPI.COMM_SELF)
            
            plot_results(it, strain, stress, figure_folder_path, nb_of_pixels_global, dmax_val, eps0_val)
        else:
            print(f"Files for iteration {it} not found, skipping.")

if __name__ == "__main__":
    main()
