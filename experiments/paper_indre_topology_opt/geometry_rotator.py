import matplotlib as mpl
import numpy as np
import matplotlib.pyplot as plt
import os
from NuMPI.IO import save_npy, load_npy

letter_offset = -0.18
etas = [0.005, 0.01, 0.02, 0.05]  #
cg_tol_exponent = 8
soft_phase_exponent = 5
random_init = False

#poisson_targets = np.array([-0.5, -0.4, -0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3])
poisson_targets = np.array([-0.4])

weight = 20.0

# weights=[5]
N = 1024
nb_tiles=1
x_ref = np.zeros([2, nb_tiles * (N) + 1, nb_tiles * (N) + 1])
x_ref[0], x_ref[1] = np.meshgrid(np.linspace(0, nb_tiles, nb_tiles * (N) + 1),
                                 np.linspace(0, nb_tiles, nb_tiles * (N) + 1), indexing='ij')
# shift = 0.5 * np.linspace(0, nb_tiles, nb_tiles * (N) + 1)
x_coords = np.copy(x_ref)

for poison_target in poisson_targets:

    eta_mult = 0.01

    script_name = 'exp_paper_TO_exp_5_square' + f'_random_{random_init}' + f'_N_{N}' + f'_cgtol_{cg_tol_exponent}' + f'_soft_{soft_phase_exponent}'

    preconditioner_type = "Green_Jacobi"
    file_folder_path = os.path.dirname(os.path.realpath(__file__))  # script directory
    data_folder_path = file_folder_path + '/exp_data/' + script_name + '/'
    figure_folder_path = file_folder_path + '/figures/' + script_name + '/'
    if not os.path.exists(figure_folder_path):
        os.makedirs(figure_folder_path)
    # name = (
    # f'{optimizer}_muFFTTO_elasticity_{element_type}_{script_name}_N{N}_E_target_{E_target_0}_Poisson_{poison_target}_Poisson0_0.0_w{w_mult:.2f}_eta{eta_mult}_p{p}_bounds={bounds}_FE_NuMPI{cores}_nb_load_cases_{nb_load_cases}_energy_objective_{energy_objective}_random_{random_initial_geometry}')
    prefix = f"{preconditioner_type}_eta_{eta_mult}_w_{weight:.1f}_p_{poison_target}_iteration_"
    suffix = ".npy"

    highest_iteration = -1
    latest_file = None

    # Scan directory for matching files
    if os.path.exists(data_folder_path):
        for filename in os.listdir(data_folder_path):
            if filename.startswith(prefix) and filename.endswith(suffix):
                try:
                    # Extract iteration number
                    iteration = int(filename[len(prefix):-len(suffix)])
                    if iteration > highest_iteration:
                        highest_iteration = iteration
                        latest_file = os.path.join(data_folder_path, filename)
                except ValueError:
                    continue
    if latest_file is None:
        raise FileNotFoundError("No phase field files found matching the pattern.")

    # name =  data_folder_path + f'{preconditioner_type}' + f'_eta_{eta_mult}'+ f'_w_{weight}'  +'_final' + f'.npy'
    name = data_folder_path + f'{preconditioner_type}' + f'_eta_{eta_mult}' + f'_w_{weight:.1f}' + f'_p_{poison_target}' + '_final'
    try:
        phase_field = np.load(name+ f'.npy', allow_pickle=True)
        print(name)
    except:
        print("No info")
    plt.figure(figsize=(10, 10))
    # plt.pcolormesh(phase_field, cmap='jet', vmin=0, vmax=1)
    data_to_save = np.copy( phase_field[:,:])#:-1.transpose()

    plt.pcolormesh(x_coords[0], x_coords[1], data_to_save,#.transpose()
                        shading='flat',
                        edgecolors='none',
                        lw=0.01,
                        cmap=mpl.cm.Greys,
                        vmin=0, vmax=1,
                        rasterized=True)
    plt.show()
    save_npy(fn=name+'rotated'+ f'.npy' ,
             data=data_to_save )
    print(name)
    # #p
    print(highest_iteration)
print("Done")