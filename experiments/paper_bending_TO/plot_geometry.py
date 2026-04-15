import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import os

# Define the path to the file
file_path = '/home/martin/Programming/muFFTTO_paralellFFT_test/muFFTTO/experiments/paper_bending_TO/exp_data/bending_v1_random_False_N_32_cgtol_8_soft_5/Green_Jacobi_eta_0.02_w_2.0_p_0.1_final.npy'

def plot_geometry(path):
    if not os.path.exists(path):
        print(f"Error: File not found at {path}")
        return

    # Load the geometry data
    try:
        data = np.load(path)
    except Exception as e:
        print(f"Error loading file: {e}")
        return

    print(f"Loaded geometry with shape: {data.shape}")

    # Create the plot
    plt.figure(figsize=(6, 5))
    
    # Use contourf as seen in bending_v1.py, or pcolormesh
    im = plt.contourf(data, cmap=mpl.cm.Greys, levels=20)
    
    # Set color limits (assuming phase field values between 0 and 1)
    plt.clim(0, 1)
    
    plt.colorbar(im, label='Phase Field')
    plt.title(f'Geometry: {os.path.basename(path)}')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.gca().set_aspect('equal')
    
    # Save the plot
    output_filename = os.path.basename(path).replace('.npy', '.png')
    plt.savefig(output_filename, dpi=300)
    print(f"Plot saved to {output_filename}")
    
    # Optional: show the plot (requires a GUI or notebook)
    # plt.show()

if __name__ == "__main__":
    plot_geometry(file_path)
