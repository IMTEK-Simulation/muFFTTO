import os
import numpy as np
import matplotlib.pyplot as plt
import argparse

def get_line_style(precond):
    """Returns a specific line style based on the preconditioner name."""
    if "Jacobi" in precond:
        return "-"  # Solid line for Green_Jacobi
    else:
        return "--" # Dashed line for Green

def plot_iteration_history(experiment_names, data_path, output_file=None):
    plt.figure(figsize=(10, 6))
    for name in experiment_names:
        data_folder = os.path.join(data_path, name)
        log_files = [f for f in os.listdir(data_folder) if f.endswith('_log.npz')]
        
        if not log_files:
            continue
            
        data = np.load(os.path.join(data_folder, log_files[0]), allow_pickle=True)
        
        if 'num_iteration_mech' in data:
            mech_its = data['num_iteration_mech']
            avg_its = [np.mean(it) if isinstance(it, (list, np.ndarray)) else it for it in mech_its]
            
            # Extract preconditioner for line style
            parts = name.split('_')
            precond = "Green" # Default
            try:
                if "soft" in parts:
                    soft_idx = parts.index("soft")
                    precond = "_".join(parts[3:soft_idx])
                else:
                    precond = "_".join(parts[3:])
            except (ValueError, IndexError):
                pass

            linestyle = get_line_style(precond)
            label = name.replace('exp_', '')
            plt.plot(avg_its, label=label, linestyle=linestyle)

    plt.title("Mechanical CG Iterations Comparison")
    plt.xlabel("BFGS Iteration")
    plt.ylabel("Average CG Steps")
    plt.yscale('log')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file)
        print(f"Iteration history plot saved to {output_file}")
    else:
        plt.show()

def plot_grid_convergence(experiment_names, data_path, output_file=None):
    # Data structures for the N vs Iterations plot
    # mapping: label (precond + soft) -> {N: total_iterations}
    label_vs_its = {}

    for name in experiment_names:
        data_folder = os.path.join(data_path, name)
        log_files = [f for f in os.listdir(data_folder) if f.endswith('_log.npz')]
        
        if not log_files:
            continue
            
        data = np.load(os.path.join(data_folder, log_files[0]), allow_pickle=True)
        
        if 'num_iteration_mech' in data:
            mech_its = data['num_iteration_mech']
            
            # Extract N, Preconditioner and Soft Phase Exponent for the plot
            # Folder format: exp_N_{N}_{Preconditioner}_soft_{soft}
            parts = name.split('_')
            try:
                # Basic parsing: exp_N_{N}_{Preconditioner} or exp_N_{N}_{Preconditioner}_soft_{soft}
                N = int(parts[2])
                
                # Check for "soft" in name
                if "soft" in parts:
                    soft_idx = parts.index("soft")
                    precond = "_".join(parts[3:soft_idx])
                    soft_val = parts[soft_idx+1]
                    label = f"{precond} (soft={soft_val})"
                else:
                    precond = "_".join(parts[3:])
                    label = precond
                
                total_its = np.sum([np.sum(it) if isinstance(it, (list, np.ndarray)) else it for it in mech_its])
                
                if label not in label_vs_its:
                    label_vs_its[label] = {}
                label_vs_its[label][N] = total_its
            except (ValueError, IndexError):
                print(f"Could not parse parameters from {name}")

    plt.figure(figsize=(10, 6))
    for label, values in label_vs_its.items():
        # Sort by N
        sorted_N = sorted(values.keys())
        sorted_its = [values[n] for n in sorted_N]
        
        # Determine linestyle from label (contains precond)
        linestyle = get_line_style(label)
        plt.plot(sorted_N, sorted_its, 'o', linestyle=linestyle, label=label)

    plt.title("Total Mechanical CG Iterations vs Grid Size")
    plt.xlabel("Grid Size (N)")
    plt.ylabel("Total CG Steps (all BFGS iterations)")
    plt.yscale('log')

    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file)
        print(f"Grid convergence plot saved to {output_file}")
    else:
        plt.show()

def plot_solution_fields(experiment_names, data_path, output_file=None):
    num_exps = len(experiment_names)
    if num_exps == 0:
        return

    # Determine grid layout
    cols = 3
    rows = (num_exps + cols - 1) // cols
    
    fig, axs = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))
    if rows == 1:
        axs = np.atleast_2d(axs)
    elif cols == 1:
        axs = np.atleast_2d(axs).T
    
    # Flatten axs for easy iteration
    axs_flat = axs.flatten()

    for i, name in enumerate(experiment_names):
        data_folder = os.path.join(data_path, name)
        npy_files = [f for f in os.listdir(data_folder) if f.endswith('.npy')]
        
        if not npy_files:
            axs_flat[i].set_title(f"No .npy found: {name}")
            continue
            
        # Use the first .npy file found
        npy_path = os.path.join(data_folder, npy_files[0])
        try:
            from NuMPI.IO import load_npy
            # load_npy expects a communicator or defaults to MPI.COMM_WORLD
            # Since we are plotting (usually Rank 0), we can just load it.
            # However, load_npy might require MPI to be initialized.
            phase_field = load_npy(npy_path)
        except (ImportError, Exception):
            # Fallback to numpy load if NuMPI fails or is not available
            phase_field = np.load(npy_path)
            
        im = axs_flat[i].pcolormesh(phase_field, cmap='Greys')
        axs_flat[i].set_title(name.replace('exp_', ''))
        axs_flat[i].axis('equal')
        plt.colorbar(im, ax=axs_flat[i], fraction=0.046, pad=0.04)

    # Hide unused axes
    for j in range(i + 1, len(axs_flat)):
        axs_flat[j].axis('off')

    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file)
        print(f"Solution fields plot saved to {output_file}")
    else:
        plt.show()

def plot_comparison(experiment_names, data_path, output_file=None):
    # Data structures for the N vs Iterations plot
    # mapping: label -> {N: total_iterations}
    label_vs_its = {}

    # Combined Plot: Both subplots in one figure
    plt.figure(figsize=(12, 10))
    
    plt.subplot(2, 1, 1)
    for name in experiment_names:
        data_folder = os.path.join(data_path, name)
        log_files = [f for f in os.listdir(data_folder) if f.endswith('_log.npz')]
        
        if not log_files:
            continue
            
        data = np.load(os.path.join(data_folder, log_files[0]), allow_pickle=True)
        
        if 'num_iteration_mech' in data:
            mech_its = data['num_iteration_mech']
            avg_its = [np.mean(it) if isinstance(it, (list, np.ndarray)) else it for it in mech_its]
            
            # Extract N, Precond and Soft for the plot and styles
            parts = name.split('_')
            try:
                N = int(parts[2])
                if "soft" in parts:
                    soft_idx = parts.index("soft")
                    precond = "_".join(parts[3:soft_idx])
                    soft_val = parts[soft_idx+1]
                    label = f"{precond} (soft={soft_val})"
                else:
                    precond = "_".join(parts[3:])
                    label = precond
                
                linestyle = get_line_style(precond)
                label_short = name.replace('exp_', '')
                plt.plot(avg_its, label=label_short, linestyle=linestyle)
                
                total_its = np.sum([np.sum(it) if isinstance(it, (list, np.ndarray)) else it for it in mech_its])
                if label not in label_vs_its:
                    label_vs_its[label] = {}
                label_vs_its[label][N] = total_its
            except (ValueError, IndexError):
                pass

    plt.title("Mechanical CG Iterations Comparison")
    plt.xlabel("BFGS Iteration")
    plt.ylabel("Average CG Steps")
    plt.yscale('log')

    plt.grid(True)
    plt.legend()

    plt.subplot(2, 1, 2)
    for label, values in label_vs_its.items():
        sorted_N = sorted(values.keys())
        sorted_its = [values[n] for n in sorted_N]
        linestyle = get_line_style(label)
        plt.plot(sorted_N, sorted_its, 'o', linestyle=linestyle, label=label)

    plt.title("Total Mechanical CG Iterations vs Grid Size")
    plt.xlabel("Grid Size (N)")
    plt.ylabel("Total CG Steps (all BFGS iterations)")
    plt.grid(True)
    plt.yscale('log')

    plt.legend()
    
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file)
        print(f"Comparison plot saved to {output_file}")
    else:
        plt.show()

if __name__ == "__main__":
    # Automatic detection of the experiment folders from the study
    script_dir = os.path.dirname(os.path.realpath(__file__))
    data_path = os.path.join(script_dir, "../data")
    
    # Filter folders created by the grid convergence script
    study_folders = sorted([f for f in os.listdir(data_path) if f.startswith('exp_N_')])
    
    if not study_folders:
        print(f"No study folders (exp_N_*) found in {data_path}")
    else:
        # You can call either function or both
        # plot_iteration_history(study_folders, data_path)
        # plot_grid_convergence(study_folders, data_path)
        # plot_solution_fields(study_folders, data_path)
        plot_comparison(study_folders, data_path)
        plot_solution_fields(study_folders, data_path)
