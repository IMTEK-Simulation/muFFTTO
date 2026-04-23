import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import argparse

def plot_results(data_folder, output_file=None):
    # Search for available log files in the data folder
    log_files = [f for f in os.listdir(data_folder) if f.endswith('_log.npz')]
    
    if not log_files:
        print(f"No log files found in {data_folder}")
        return

    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f"Optimization Analysis: {os.path.basename(data_folder)}")

    for log_file in log_files:
        path = os.path.join(data_folder, log_file)
        data = np.load(path, allow_pickle=True)
        
        label = log_file.replace('_log.npz', '')
        
        # 1. Objective Function Convergence
        if 'norms_pf' in data:
            axs[0, 0].semilogy(data['norms_pf'], label=label)
        axs[0, 0].set_title("Objective Function (Phase Field)")
        axs[0, 0].set_xlabel("Iteration")
        axs[0, 0].set_ylabel("Value")
        axs[0, 0].grid(True)
        axs[0, 0].legend()

        # 2. Stress Error Convergence
        if 'norms_sigma' in data:
            axs[0, 1].semilogy(data['norms_sigma'], label=label)
        axs[0, 1].set_title("Stress Objective (f_sigma)")
        axs[0, 1].set_xlabel("Iteration")
        axs[0, 1].set_ylabel("Value")
        axs[0, 1].grid(True)

        # 3. CG Iterations (Mechanical)
        if 'num_iteration_mech' in data:
            # num_iteration_mech is often a list of lists (per load case)
            mech_its = data['num_iteration_mech']
            if len(mech_its) > 0:
                # Flatten if it's multiple load cases per optimization step
                # For plotting, we might take the average or max per step
                # Assuming it's recorded per load case per step
                # Let's try to plot the total or average
                avg_its = [np.mean(it) if isinstance(it, (list, np.ndarray)) else it for it in mech_its]
                axs[1, 0].plot(avg_its, label=label)
        axs[1, 0].set_title("Avg CG Iterations (Mechanical)")
        axs[1, 0].set_xlabel("BFGS Iteration")
        axs[1, 0].set_ylabel("CG Steps")
        axs[1, 0].grid(True)

        # 4. CG Iterations (Adjoint)
        if 'num_iteration_adjoint' in data:
            adj_its = data['num_iteration_adjoint']
            if len(adj_its) > 0:
                avg_adj_its = [np.mean(it) if isinstance(it, (list, np.ndarray)) else it for it in adj_its]
                axs[1, 1].plot(avg_adj_its, label=label)
        axs[1, 1].set_title("Avg CG Iterations (Adjoint)")
        axs[1, 1].set_xlabel("BFGS Iteration")
        axs[1, 1].set_ylabel("CG Steps")
        axs[1, 1].grid(True)

    # Plot the final phase field if available
    npy_files = [f for f in os.listdir(data_folder) if f.endswith('.npy')]
    if npy_files:
        plt.figure(figsize=(6, 5))
        # Load the first npy file found
        try:
            from NuMPI.IO import load_npy
            phase_field = load_npy(os.path.join(data_folder, npy_files[0]))
        except ImportError:
            phase_field = np.load(os.path.join(data_folder, npy_files[0]))
            
        plt.pcolormesh(phase_field, cmap='Greys')
        plt.colorbar(label='Phase Field')
        plt.title(f"Final Phase Field: {npy_files[0]}")
        plt.axis('equal')
        if output_file:
            plt.savefig(output_file.replace('.pdf', '_phase.pdf'))

    plt.tight_layout()
    if output_file:
        plt.savefig(output_file)
        print(f"Plot saved to {output_file}")
    else:
        plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot results from topology optimization experiments")
    parser.add_argument("script_name", help="Name of the script (experiment) to analyze")
    parser.add_argument("--data_root", default="data", help="Root directory for data")
    parser.add_argument("--output", help="Output PDF file name")
    
    args = parser.parse_args()
    
    script_dir = os.path.dirname(os.path.realpath(__file__))
    data_folder = os.path.join(script_dir, args.data_root, args.script_name)
    
    if not os.path.exists(data_folder):
        print(f"Data folder not found: {data_folder}")
    else:
        plot_results(data_folder, args.output)
