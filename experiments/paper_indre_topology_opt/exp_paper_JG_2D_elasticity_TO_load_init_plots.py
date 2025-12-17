import matplotlib as mpl
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import os

script_name = 'exp_paper_JG_2D_elasticity_TO_load_init'
file_folder_path = os.path.dirname(os.path.realpath(__file__))  # script directory
data_folder_path = file_folder_path + '/exp_data/' + script_name + '/'
figure_folder_path = file_folder_path + '/figures/'

src = '../figures/'  # source folder\

# Enable LaTeX rendering
plt.rcParams.update({
    "text.usetex": True,  # Use LaTeX
    "font.family": "helvetica",  # Use a serif font
})
plt.rcParams.update({'font.size': 13})
plt.rcParams["font.family"] = "Arial"

cg_tol = 7
Ns = [16, 32, 64, 128, 256, 512, 1024]
# steps = np.arange(0, 1000, 10)
# for j in np.arange(len(Ns)):
#     N = Ns[j]
#     fig, ax = plt.subplots(1, 1, figsize=(5, 5))
#     for i in np.arange(len(steps) - 1):
#         iter = steps[i]
#         preconditioner_type = 'Green_Jacobi'
#         random = False
#         try:
#             script_name = f'exp_paper_JG_2D_elasticity_TO_load_init_random_{random}' + f'_N_{N}'+ f'_cgtol_{cg_tol}'+'/'
#             phase_field_it_F = np.load(
#                 './exp_data/' + script_name + f'{preconditioner_type}' + f'_iteration_{iter}' + '.npy',
#                 allow_pickle=True)
#
#             plt.contourf(phase_field_it_F, cmap=mpl.cm.Greys)
#             # nodal_coordinates[0, 0] * number_of_pixels[0], nodal_coordinates[1, 0] * number_of_pixels[0],
#             plt.clim(0, 1)
#             plt.title(f'Phase field {iter}, res {N}, tol {cg_tol}')
#             plt.colorbar()
#             plt.show()
#         except:
#             pass


nbit_per_lbfgs_mech_G = []
nbit_per_lbfgs_adjoint_G = []
nbit_per_lbfgs_mech_GJ = []
nbit_per_lbfgs_adjoint_GJ = []
nb_it_G_mech_Ns = []
nb_it_G_adjoint_Ns = []
nb_it_GJ_mech_Ns = []
nb_it_GJ_adjoint_Ns = []

iterations = np.arange(0, 500)
# , 64, 128
fig, ax = plt.subplots(1, 1, figsize=(5.5, 4.5))
random = False
for j in np.arange(len(Ns)):
    N = Ns[j]

    nb_it_G_mech_ = []
    nb_it_G_adjoint_ = []
    nb_it_GJ_mech_ = []
    nb_it_GJ_adjoint_ = []
    # we do not know how manny iteration we have. So we itrate and find the last one
    for i in np.arange(len(iterations)):
        iteration = iterations[i]
        preconditioner_type = 'Green'
        file_name = f'_log.npz'

        try:
            script_name = f'exp_paper_JG_2D_elasticity_TO_load_init_random_{random}' + f'_N_{N}' + f'_cgtol_{cg_tol}' + '/'
            _info_log_G = np.load(
                './exp_data/' + script_name + f'{preconditioner_type}' + f'_iteration_{iteration}' + file_name,
                allow_pickle=True)
            nb_it_G_mech = (_info_log_G.f.num_iteration_mech.transpose()[::3] +
                            _info_log_G.f.num_iteration_mech.transpose()[1::3] +
                            _info_log_G.f.num_iteration_mech.transpose()[2::3]) / 3
            nb_it_G_adjoint = (_info_log_G.f.num_iteration_adjoint.transpose()[::3] +
                               _info_log_G.f.num_iteration_adjoint.transpose()[1::3] +
                               _info_log_G.f.num_iteration_adjoint.transpose()[2::3]) / 3
            # nb_it_G_mech_.extend(nb_it_G_mech)
            # nb_it_G_adjoint_.extend(nb_it_G_adjoint)
            nb_it_G_mech_ = nb_it_G_mech
            nb_it_G_adjoint_ = nb_it_G_adjoint
        except:
            pass

        preconditioner_type = 'Green_Jacobi'
        try:
            script_name = f'exp_paper_JG_2D_elasticity_TO_load_init_random_{random}' + f'_N_{N}' + f'_cgtol_{cg_tol}' + '/'
            _info_log_GJ = np.load(
                './exp_data/' + script_name + f'{preconditioner_type}' + f'_iteration_{iteration}' + file_name,
                allow_pickle=True)
            nb_it_GJ_mech = (_info_log_GJ.f.num_iteration_mech.transpose()[::3] +
                             _info_log_GJ.f.num_iteration_mech.transpose()[1::3] +
                             _info_log_GJ.f.num_iteration_mech.transpose()[2::3]) / 3
            nb_it_GJ_adjoint = (_info_log_GJ.f.num_iteration_adjoint.transpose()[::3] +
                                _info_log_GJ.f.num_iteration_adjoint.transpose()[1::3] +
                                _info_log_GJ.f.num_iteration_adjoint.transpose()[2::3]) / 3
            # nb_it_GJ_mech_.extend(nb_it_GJ_mech)
            # nb_it_GJ_adjoint_.extend(nb_it_GJ_adjoint)
            nb_it_GJ_mech_ = nb_it_GJ_mech
            nb_it_GJ_adjoint_ = nb_it_GJ_adjoint
        except:
            pass
            # nb_lbfgs_steps_GJ = len(nb_it_GJ_mech)
    nb_lbfgs_steps_G = max(len(nb_it_G_mech_), 1)
    nbit_per_lbfgs_mech_G.append(np.sum(nb_it_G_mech_) / nb_lbfgs_steps_G)
    nbit_per_lbfgs_adjoint_G.append(np.sum(nb_it_G_adjoint_) / nb_lbfgs_steps_G)
    nb_it_G_mech_Ns.append(np.asarray(nb_it_G_mech_))
    nb_it_G_adjoint_Ns.append(np.asarray(nb_it_G_adjoint_))

    nb_lbfgs_steps_GJ = max(len(nb_it_GJ_mech_), 1)
    nbit_per_lbfgs_mech_GJ.append(np.sum(nb_it_GJ_mech_) / nb_lbfgs_steps_GJ)
    nbit_per_lbfgs_adjoint_GJ.append(np.sum(nb_it_GJ_adjoint_) / nb_lbfgs_steps_GJ)
    nb_it_GJ_mech_Ns.append(np.asarray(nb_it_GJ_mech_))
    nb_it_GJ_adjoint_Ns.append(np.asarray(nb_it_GJ_adjoint_))

    print()

    ax.plot(np.asarray(nb_it_GJ_mech_), "y",
            label=r'Green-Jacobi - $\nabla \sigma$' + f'N={N}',
            linewidth=2)  # np.linspace(1, 10, np.asarray( nb_it_GJ_mech_).shape[0]),
    # ax.plot(np.linspace(1, max_it, nb_it_G_adjoint.shape[0]), nb_it_G_adjoint, "g--", label=r'Green - Adjoint'+f'N={N}',
    #         linewidth=2)
    ax.plot(np.asarray(nb_it_GJ_adjoint_), "r--",
            label='Green-Jacobi - Adjoint' + f'N={N}',
            linewidth=2)  # np.linspace(1, 10, np.asarray(nb_it_GJ_adjoint_).shape[0]),

plt.show()

fig, ax = plt.subplots(1, 1, figsize=(5, 5))
# ax.loglog(np.array(Ns) ** 2, np.array(Ns) /1, "k-",
#         label=r'Scaling',
#         linewidth=1)
line1G, = ax.plot(np.array(Ns[:3]) ** 2, nbit_per_lbfgs_adjoint_G[:3], "g-.", marker='x',
                 label=r'Adjoint problem - Green ',
                 linewidth=2)
line2G, = ax.plot(np.array(Ns[:3]) ** 2, nbit_per_lbfgs_mech_G[:3], "g", marker='x',
                 label=r'Mechanical equilibrium - Green ',
                 linewidth=2)
line1GJ, = ax.semilogx(np.array(Ns) ** 2, nbit_per_lbfgs_adjoint_GJ, "k-.", marker='o',
                      label=r'Adjoint problem - Green-Jacobi',
                      linewidth=2)
line2GJ, = ax.loglog(np.array(Ns) ** 2, nbit_per_lbfgs_mech_GJ, "k", marker='o',
                    label=r'Mechanical equilibrium - Green-Jacobi',
                    linewidth=2)
# plt.title(f' CG tol =  $10^{{-{cg_tol}}}$')

ax.set_xlabel(r"Grid size")
ax.set_xticks(np.array(Ns) ** 2)
ax.set_xticklabels([f"${n}^2$" for n in Ns])
ax.set_xlim([16 ** 2 - 1, 1024 ** 2 + 1])

ax.set_ylabel(r"$\#$ PCG iterations per L-BFGS step ")
ax.set_ylim([10, 1e4])
# First legend
first_legend = ax.legend([line1G, line2G],
                         ["Adjoint problem - Green", "Mechanical equilibrium - Green"],
                         loc="upper right")
ax.add_artist(first_legend)

plt.legend([line1GJ, line2GJ],["Adjoint problem - Green-Jacobi","Mechanical equilibrium - Green-Jacobi"],
                          loc="lower right")

fname = figure_folder_path + 'scaling{}'.format('.pdf')
print(('create figure: {}'.format(fname)))
plt.savefig(fname, bbox_inches='tight')
plt.show()

quit()

Ns = [16, 32, 64, 128]  # , 64, 128
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
fname = figure_folder_path + 'scaling{}'.format('.pdf')
print(('create figure: {}'.format(fname)))
plt.savefig(fname, bbox_inches='tight')
plt.show()
