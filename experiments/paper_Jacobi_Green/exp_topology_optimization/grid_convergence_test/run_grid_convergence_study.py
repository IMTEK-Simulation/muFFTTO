import os
import sys
from mpi4py import MPI

# Add the directory containing topology_opt_base.py to sys.path
script_dir = os.path.dirname(os.path.realpath(__file__))
# If running from a subdirectory (like grid_convergence_test), add the parent directory
parent_dir = os.path.dirname(script_dir)

if script_dir not in sys.path:
    sys.path.append(script_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from topology_opt_base import run_topology_optimization

if __name__ == "__main__":
    grid_sizes = [int(os.environ.get("EXP_N", 32))]
    preconditioners = [os.environ.get("EXP_P", "Green_Jacobi")]
    soft_phase_exponents = [int(os.environ.get("EXP_SOFT", 5))]


    for n in grid_sizes:
        for p_type in preconditioners:
            for soft_exponent in soft_phase_exponents:
                # Create a unique script name for this run
                script_name = f"exp_N_{n}_{p_type}_soft_{soft_exponent}"
                
                # Define paths
                data_folder_path = os.path.join(script_dir, "../data", script_name)
                figure_folder_path = os.path.join(script_dir, "../figures", script_name)

                # Parameters for this specific run
                params = {
                    "nb_pixels": n,
                    "cg_tol_exponent": 5,
                    "soft_phase_exponent": soft_exponent,
                    "preconditioner_type": p_type,
                    "eta": 0.01,
                    "weight": 5.0,
                    "poison_target": -0.5,
                    "K_0": 1.0,
                    "G_0": 0.5,
                    "random_init": False, # Fixed init for comparison
                    "save_results": True,
                    "data_folder_path": data_folder_path,
                    "figure_folder_path": figure_folder_path,
                    "maxiter": 10000,
                }

                if MPI.COMM_WORLD.rank == 0:
                    print("\n" + "="*50)
                    print(f"RUNNING: N={n}, Preconditioner={p_type}, Soft Phase Exponent={soft_exponent}")
                    print(f"Data folder: {data_folder_path}")
                    print("="*50)

                run_topology_optimization(**params)

    if MPI.COMM_WORLD.rank == 0:
        print("\nAll experiments completed.")
        print("Use plot_results_comparison.py (if created) or run plot_results.py on individual folders.")
