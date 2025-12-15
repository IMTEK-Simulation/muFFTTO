import matplotlib as mpl
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import os

script_name = 'exp_paper_JG_2D_elasticity_TO'
file_folder_path = os.path.dirname(os.path.realpath(__file__))  # script directory
data_folder_path = file_folder_path + '/exp_data/' + script_name + '/'
figure_folder_path = file_folder_path + '/figures/'

src = '../figures/'  # source folder\

# Enable LaTeX rendering
plt.rcParams.update({
    "text.usetex": True,  # Use LaTeX
    "font.family": "helvetica",  # Use a serif font
})



N = 32  # , 64, 128
plt.figure(figsize=[8, 6])
max_it = 70
# x_coords_def = x_coords + imposed_disp_ixy + total_displacement_fluctuation_ixy
fig, ax = plt.subplots(1, 1, figsize=(10, 10))

nbit_per_lbfgs_mech_G = []
nbit_per_lbfgs_adjoint_G = []
nbit_per_lbfgs_mech_GJ = []
nbit_per_lbfgs_adjoint_GJ = []

script_name = 'exp_paper_JG_2D_elasticity_TO_load_init' + f'_N_{N}/'
preconditioner_type = 'Green_Jacobi'
file_name = f'_log.npz'












# _info_log_GJ = np.load('./exp_data/' + script_name + f'{preconditioner_type}' + file_name,
#                        allow_pickle=True)

plt.figure()
for i in np.arange(max_it):

    preconditioner_type = 'Green_Jacobi'
    file_name = f'iteration{i}'
    phase_field_it = np.load('./exp_data/' + script_name + f'{preconditioner_type}' + file_name + '.npy',
                           allow_pickle=True)

    plt.contourf(phase_field_it, cmap=mpl.cm.Greys)
    # nodal_coordinates[0, 0] * number_of_pixels[0], nodal_coordinates[1, 0] * number_of_pixels[0],
    plt.clim(0, 1)
    plt.title(f'Phase field {i}')
    plt.colorbar()
    plt.show()

quit
Ns = [16, 32, 64, 128]  # , 64, 128
plt.figure(figsize=[8, 6])
max_it = 300
# x_coords_def = x_coords + imposed_disp_ixy + total_displacement_fluctuation_ixy
fig, ax = plt.subplots(1, 1, figsize=(10, 10))

nbit_per_lbfgs_mech_G = []
nbit_per_lbfgs_adjoint_G = []
nbit_per_lbfgs_mech_GJ = []
nbit_per_lbfgs_adjoint_GJ = []

for i in np.arange(len(Ns)):
    N = Ns[i]
    script_name = 'exp_paper_JG_2D_elasticity_TO' + f'_N_{N}/'

    file_name = f'_log.npz'
    try:
        preconditioner_type = 'Green'
        _info_log_G = np.load('./exp_data/' + script_name + f'{preconditioner_type}' + file_name, allow_pickle=True)

        _info_log_G.f.num_iteration_adjoint
        nb_it_G_adjoint = (_info_log_G.f.num_iteration_adjoint.transpose()[::3] +
                           _info_log_G.f.num_iteration_adjoint.transpose()[1::3] +
                           _info_log_G.f.num_iteration_adjoint.transpose()[2::3]) / 3
        nb_it_G_mech = (_info_log_G.f.num_iteration_mech.transpose()[::3] +
                        _info_log_G.f.num_iteration_mech.transpose()[1::3] +
                        _info_log_G.f.num_iteration_mech.transpose()[2::3]) / 3
        nb_lbfgs_steps_G = len(nb_it_G_mech)
        nbit_per_lbfgs_mech_G.append(np.sum(nb_it_G_mech) / nb_lbfgs_steps_G)
        nbit_per_lbfgs_adjoint_G.append(np.sum(nb_it_G_adjoint) / nb_lbfgs_steps_G)
    except:
        pass

    preconditioner_type = 'Green_Jacobi'
    _info_log_GJ = np.load('./exp_data/' + script_name + f'{preconditioner_type}' + file_name,
                           allow_pickle=True)
    nb_it_GJ_mech = (_info_log_GJ.f.num_iteration_mech.transpose()[::3] +
                     _info_log_GJ.f.num_iteration_mech.transpose()[1::3] +
                     _info_log_GJ.f.num_iteration_mech.transpose()[2::3]) / 3
    nb_it_GJ_adjoint = (_info_log_GJ.f.num_iteration_adjoint.transpose()[::3] +
                        _info_log_GJ.f.num_iteration_adjoint.transpose()[1::3] +
                        _info_log_GJ.f.num_iteration_adjoint.transpose()[2::3]) / 3
    nb_lbfgs_steps_GJ = len(nb_it_GJ_mech)
    nbit_per_lbfgs_mech_GJ.append(np.sum(nb_it_GJ_mech) / nb_lbfgs_steps_GJ)
    nbit_per_lbfgs_adjoint_GJ.append(np.sum(nb_it_GJ_adjoint) / nb_lbfgs_steps_GJ)
    # ax.plot(np.linspace(1, 10, nb_it_G_mech.shape[0]), nb_it_G_mech, "g", label=r'Green - $\nabla \sigma$'+f'N={N}',
    #         linewidth=2)
    ax.plot(np.linspace(1, 10, nb_it_GJ_mech.shape[0]), nb_it_GJ_mech, "k",
            label=r'Green-Jacobi - $\nabla \sigma$' + f'N={N}',
            linewidth=2)
    # ax.plot(np.linspace(1, max_it, nb_it_G_adjoint.shape[0]), nb_it_G_adjoint, "g--", label=r'Green - Adjoint'+f'N={N}',
    #         linewidth=2)
    ax.plot(np.linspace(1, 10, nb_it_GJ_adjoint.shape[0]), nb_it_GJ_adjoint, "k--",
            label='Green-Jacobi - Adjoint' + f'N={N}',
            linewidth=2)
ax.set_title(r'$F_{q0}$')
ax.legend()
# plt.savefig(fname, bbox_inches='tight')

plt.show()

fig, ax = plt.subplots(1, 1, figsize=(5, 5))
ax.plot(np.array(Ns[:2]) ** 2, nbit_per_lbfgs_adjoint_G, "g-.", marker='x',
        label=r'Adjoint problem - Green ',
        linewidth=2)
ax.plot(np.array(Ns[:2]) ** 2, nbit_per_lbfgs_mech_G, "g", marker='o',
        label=r'Mechanical equilibrium - Green ',
        linewidth=2)
ax.plot(np.array(Ns) ** 2, nbit_per_lbfgs_adjoint_GJ, "k-.", marker='x',
        label=r'Adjoint problem - Green-Jacobi',
        linewidth=2)
ax.plot(np.array(Ns) ** 2, nbit_per_lbfgs_mech_GJ, "k", marker='o',
        label=r'Mechanical equilibrium - Green-Jacobi',
        linewidth=2)

ax.set_xlabel(r"System size ")
ax.set_xticks(np.array(Ns) ** 2)
ax.set_xticklabels([f"${n}^2$" for n in Ns])

ax.set_ylabel(r"$\#$ PCG iterations per L-BFGS step ")

ax.legend()
fname = figure_folder_path + 'scaling{}'.format(  '.pdf')
print(('create figure: {}'.format(fname)))
plt.savefig(fname, bbox_inches='tight')
plt.show()



