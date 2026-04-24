import os
import sys

# Add the project root to sys.path to allow imports from muFFTTO
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
project_root = os.path.dirname(os.path.dirname(parent_dir))

sys.path.insert(0, project_root)
sys.path.insert(0, parent_dir)
sys.path.insert(0, script_dir)

from mpi4py import MPI

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
