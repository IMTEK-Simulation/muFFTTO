import os
import sys
from mpi4py import MPI

# Add the directory containing topology_opt_base.py to sys.path
script_dir = os.path.dirname(os.path.realpath(__file__))
if script_dir not in sys.path:
    sys.path.append(script_dir)

from topology_opt_base import run_topology_optimization

if __name__ == "__main__":
    # Automatically determine the script name to use for the data folder
    script_name = os.path.splitext(os.path.basename(__file__))[0]
    
    # Define paths relative to this script
    data_folder_path = os.path.join(script_dir, "data", script_name)
    figure_folder_path = os.path.join(script_dir, "figures", script_name)

    # Parameters for the experiment
    params = {
        "nb_pixels": 32,
        "cg_tol_exponent": 8,
        "soft_phase_exponent": 5,
        "preconditioner_type": "Green_Jacobi",
        "eta": 0.01,
        "weight": 5.0,
        "poison_target": -0.5,
        "K_0": 1.0,
        "G_0": 0.5,
        "random_init": True,
        "save_results": True,
        "data_folder_path": data_folder_path,
        "figure_folder_path": figure_folder_path,
        "maxiter": 500,
    }

    if MPI.COMM_WORLD.rank == 0:
        print(f"Running experiment: {script_name}")
        print(f"Data will be saved to: {data_folder_path}")
        print(f"Parameters: {params}")

    run_simulation = False
    plot_results = True


    if run_simulation:
        run_topology_optimization(**params)

    if plot_results:
        # Optional: Automatically plot results after optimization (Rank 0 only)
        if MPI.COMM_WORLD.rank == 0:
            print("\nOptimization finished. Plotting results...")
            from plot_results import plot_results
            plot_results(data_folder_path)
