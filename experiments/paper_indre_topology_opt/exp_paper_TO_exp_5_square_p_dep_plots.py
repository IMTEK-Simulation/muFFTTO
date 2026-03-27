import matplotlib as mpl
import numpy as np
import matplotlib.pyplot as plt
import os

plt.rcParams.update({
    "text.usetex": True,  # Use LaTeX
    # "font.family": "helvetica",  # Use a serif font
})
plt.rcParams.update({'font.size': 14})
plt.rcParams["font.family"] = "Arial"

# fig = plt.figure(figsize=(11, 4.5))
# gs = fig.add_gridspec(2, 5, width_ratios=[0.1, 1, 1, 1, 1])
# ax0 = fig.add_subplot(gs[0, 0:])
colors = ['red', 'blue', 'green', 'orange', 'purple']
linestyles = ['-', '--', '-.', ':', (0, (3, 1, 1, 1))]
letter_offset = -0.18
etas = [0.005, 0.01, 0.02, 0.05]  #
cg_tol_exponent = 8
soft_phase_exponent = 5
random_init = False
f_sigmas = []
f_pfs = []
f_adjoint = []
zener_ratios = []
poison_ratios = []
E1_target = []
c13_ = []
c23_ = []
nu12_target = []
nu21_target= []
E1 = []
E2 = []
nu12 = []
nu21 = []

C_22 = []

shear_G_computed = []
bulk_K_computed = []
shear_G_target = []
bulk_K_target = []
# weights = np.concatenate(
#     [np.arange(0.1, 2., 0.1), np.arange(2, 3, 1), np.arange(3, 10, 2), np.arange(10, 110, 20), np.array([150.0, 200.0, 300.0, 400.0, 500.0])])
# weights = np.array([0.1, 0.5, 1.0, 1.5, 2.0, 5.0, 20.0, 30.0, 50.0, 100.0, 200.0, 300.0, 400.0, 500.0, 1000.])

poisson_targets = np.array([-0.5, -0.4, -0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3])  # , 0.4
# poisson_targets =poisson_targets[: :-1]

homogenized_Cij = np.zeros((3, 3, poisson_targets.shape[0]))

target_Cij = np.zeros((poisson_targets.shape[0], 3, 3))
weight = 20.0
index = 0

# weights=[5]
N = 1024
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
    print(f"Loading phase field data from {latest_file}")

    # Load the file if found
    if latest_file is None:
        raise FileNotFoundError("No phase field files found matching the pattern.")

    # name =  data_folder_path + f'{preconditioner_type}' + f'_eta_{eta_mult}'+ f'_w_{weight}'  +'_final' + f'.npy'
    name = data_folder_path + f'{preconditioner_type}' + f'_eta_{eta_mult}' + f'_w_{weight:.1f}' + f'_p_{poison_target}' + '_final' + f'.npy'
    try:
        phase_field = np.load(name, allow_pickle=True)
    except:
        print("No info")
    name_info = data_folder_path + f'{preconditioner_type}' + f'_eta_{eta_mult}' + f'_w_{weight:.1f}' + f'_p_{poison_target}' + '_log' + f'.npz'
    try:
        info_ = np.load(name_info, allow_pickle=True)
    except:
        print(f"No info for p={poison_target}")
    # continue
    # if True:
    f_sigmas.append(info_.f.norms_sigma[-1])
    f_pfs.append(info_.f.norms_pf[-1])
    f_adjoint.append(info_.f.norms_adjoint_energy[-1])
    # for Isotropic material
    Cij = info_.f.homogenized_C_ijkl
    homogenized_Cij[..., index] = info_.f.homogenized_C_ijkl

    # compliance tensors for Orthotropic material
    #Cij[0, 2] = Cij[1, 2] = Cij[2, 0] = Cij[2, 1] = 0

    Sij = np.linalg.inv(Cij)

    young_1_S = 1 / Sij[0, 0]
    young_2_S = 1 / Sij[1, 1]
    poison_ratios_1_S = - Sij[0, 1] / Sij[0, 0]
    poison_ratios_2_S = - Sij[1, 0] / Sij[1, 1]
    shear_S = 1 / Sij[2, 2]
    bulk_S = 1 / (Sij[0, 0]+2*Sij[1, 0]+Sij[1, 1])
    print(f'check symmetry condition {poison_target}  {poison_ratios_1_S / young_1_S - poison_ratios_2_S / young_2_S}')
    print(f'check shear coupling {poison_target}  S_13 {Sij[0, 2]}  S_23 {Sij[1, 2]}')
    print(f'check isotropy {poison_target}  S_11-S_22 {Sij[0, 0] - Sij[1, 1]}')

    bulk_K_e_poisson = young_1_S / (3 * (1 - 2 * poison_ratios_1_S))
    bulk_K_mu_poisson = (2 * shear_S * (1 + poison_ratios_1_S)) / (3 * (1 - 2 * poison_ratios_1_S))
    bulk_K_mu_e = (young_1_S * shear_S) / (3 * (3 * shear_S - young_1_S))



    K_E_nu = young_1_S / (3 * (1 - 2 * poison_ratios_1_S))

    K_G_nu = (2 * shear_S * (1 + poison_ratios_1_S)) / (3 * (1 - 2 * poison_ratios_1_S))

    K_E_G = (young_1_S * shear_S) / (3 * (3 * shear_S - young_1_S))

    shear_G_computed.append(shear_S)
    bulk_K_computed.append(bulk_S)

    nu12.append(poison_ratios_1_S)
    nu21.append(poison_ratios_2_S)
    E1.append(young_1_S)
    E2.append(young_2_S)

    zener_ratios.append(2 * Cij[2, 2] / (Cij[0, 0] - Cij[0, 1]))
    lam, mu = Cij[0, 1], Cij[2, 2]

    # for Orthotropic material
    Cij_target = info_.f.target_C_ijkl
    target_Cij[index, ...] = info_.f.target_C_ijkl

    S_compl_target = np.linalg.inv(Cij_target)

    nu12_target.append(- S_compl_target[0, 1] / S_compl_target[0, 0])
    nu21_target.append(- S_compl_target[1, 0] / S_compl_target[1, 1])
    E1_target.append(1 / S_compl_target[0, 0])
    shear_G_target.append(1/S_compl_target[2, 2])
    bulk_K_target.append(1 / (S_compl_target[0, 0] + 2 * S_compl_target[1, 0] + S_compl_target[1, 1]))

    index += 1

fig = plt.figure(figsize=(5.5, 3.5))
gs_global = fig.add_gridspec(1, 1, width_ratios=[1], hspace=0.00)

G_0 = 0.5
K_0 = 1.0
ax_modulus = fig.add_subplot(gs_global[0, 0])

# ax_modulus.plot(poisson_targets, np.asarray(E1_target) / K_0, '-.', color='r', linewidth=1, marker='x',
#                 label=r'Target - Shear G')
# ax_modulus.plot(poisson_targets, np.asarray(E1) / K_0, '-', color='r', linewidth=2, marker='x',
#                 label=r'Computed - Shear G')
# ax_modulus.annotate(r'$E_1^\mathrm{{eff}}/K^0$', color='red',
#                     xy=(0.1, E1[np.where(poisson_targets == 0.1)[0][0]]),
#                     xytext=(-0.1, 0.55),
#                     arrowprops=dict(arrowstyle='->',
#                                     color='red',
#                                     lw=1,
#                                     ls='-')
#                     )
# ax_modulus.annotate(r'$E_1^\mathrm{{target}}/K^0$', color='red',
#                     xy=(0.3, E1_target[np.where(poisson_targets == 0.3)[0][0]]),
#                     xytext=(0.05, 0.62),
#                     arrowprops=dict(arrowstyle='->',
#                                     color='red',
#                                     lw=1,
#                                     ls='--')
#                     )


ax_modulus.plot(poisson_targets, np.asarray(shear_G_target) / K_0, '-.', color='b', linewidth=1, marker='^',
                label=r'Target - Shear G')
ax_modulus.plot(poisson_targets, np.asarray(shear_G_computed) / K_0, '-', color='b', linewidth=2, marker='^',
                label=r'Computed - Shear G')
ax_modulus.annotate(r'$\mu^\mathrm{{eff}}/K^0$', color='blue',
                    xy=(-0.3, shear_G_computed[np.where(poisson_targets == -0.3)[0][0]] / K_0),
                    xytext=(-0.25, 0.03),
                    arrowprops=dict(arrowstyle='->',
                                    color='blue',
                                    lw=1,
                                    ls='-')
                    )
ax_modulus.annotate(r'$\mu^\mathrm{{target}}/K^0$', color='blue',
                    xy=(-0.4, shear_G_target[np.where(poisson_targets == -0.4)[0][0]] / K_0),
                    xytext=(-0.45, 0.32),
                    arrowprops=dict(arrowstyle='->',
                                    color='blue',
                                    lw=1,
                                    ls='--')
                    )

# ax_modulus.plot(poisson_targets, np.asarray(bulk_K_target) / K_0, '-.', color='k', linewidth=1, marker='|',
#                 label=r'Target - Bulk K')
# ax_modulus.plot(poisson_targets, np.asarray(bulk_K_computed) / K_0, '-', color='k', linewidth=2, marker='|',
#                 label=r'Computed - Bulk K')
# ax_modulus.annotate(r'$K^\mathrm{{eff}}/K^0$', color='k',
#                     xy=(-0.3, bulk_K_computed[np.where(poisson_targets == -0.3)[0][0]]),
#                     xytext=(-0.46, 0.16),
#                     arrowprops=dict(arrowstyle='->',
#                                     color='k',
#                                     lw=1,
#                                     ls='-')
#                     )
# ax_modulus.annotate(r'$K^\mathrm{{target}}/K^0$', color='k',
#                     xy=(-0.4, bulk_K_target[np.where(poisson_targets == -0.4)[0][0]]),
#                     xytext=(-0.43, -0.1),
#                     arrowprops=dict(arrowstyle='->',
#                                     color='k',
#                                     lw=1,
#                                     ls='--')
#                     )

ax_modulus.plot(poisson_targets, np.asarray(nu12_target), '-.', color='g', linewidth=1, marker='o',
                label=r'Target - Poisson')
ax_modulus.plot(poisson_targets, np.asarray(nu12), '-', color='g', linewidth=2, marker='o',
                label=r'Computed - Poisson')
ax_modulus.annotate(r'$\nu^\mathrm{{eff}}_{12}$', color='green',
                    xy=(-0.2, nu12[np.where(poisson_targets == -0.2)[0][0]]),
                    xytext=(-0.3, -0.1),
                    arrowprops=dict(arrowstyle='->',
                                    color='green',
                                    lw=1,
                                    ls='-')
                    )
ax_modulus.annotate(r'$\nu^\mathrm{{target}}_{12}$', color='green',
                    xy=(-0.3, nu12_target[np.where(poisson_targets == -0.3)[0][0]]),
                    xytext=(-0.2, -0.32),
                    arrowprops=dict(arrowstyle='->',
                                    color='green',
                                    lw=1,
                                    ls='--')
                    )

ax_modulus.annotate(r'$1024^2$ grid points'+'\n'+rf'$\eta={eta_mult}L$'+'\n'+rf'$a={weight}$',
             xy=(0.02, -0.32))
ax_modulus.set_title(r"Square grid")

ax_modulus.set_xlabel(r"Target Poisson's ratio " + fr'- $\nu^\mathrm{{target}}$')
ax_modulus.set_xlim(-0.5, 0.3)
ax_modulus.set_ylim(-0.4, 0.5)
# ax_modulus.set_yticks([-0.5, -0.25, 0.0, 0.25, 0.5])
ax_modulus.set_xticks([-0.5, -0.3, -0.1, 0.1, 0.3])
ax_modulus.grid(axis='y', which='major', linestyle='-', linewidth=0.5, alpha=0.7)
ax_modulus.text(0.01, 1.05,r'$\textbf{{(a)}}$', transform=ax_modulus.transAxes)

fname_pf = figure_folder_path + 'exp5_square_' + f'w{weight:.0f}' + '_graph.pdf'
print(f'create figure: {fname_pf}')
fig.savefig(fname_pf, bbox_inches='tight')
plt.show()

# --- New figure for phase fields of all poisson_targets ---
n_targets = len(poisson_targets)
n_cols = 3
fig = plt.figure(figsize=(11, 8.5))
gs_global = fig.add_gridspec(1, 1, width_ratios=[1], hspace=0.2)

# Top: 2×4 subgrid
gs = gs_global[0].subgridspec(nrows=3, ncols=3,
                              wspace=0.2,
                              hspace=0.35)

for i, poison_target in enumerate(poisson_targets):
    nu12_target_i=np.asarray(nu12_target)[i]

    print(f'i org = {i}')
    j = i // n_cols
    k = i % n_cols
    print(f'j = {j}')
    print(f'k = {k}')
    # ax = axes_pf[i]
    ax = fig.add_subplot(gs[j, k])

    # Try to load the final phase field
    # name = data_folder_path + f'{preconditioner_type}'+ f'_eta_{eta_mult}' + f'_w_{weight:.1f}' + f'_p_{poison_target}' + '_final' + f'.npy'
    name = data_folder_path + f'{preconditioner_type}'
    if not os.path.exists(name):
        # Fallback to searching for the highest iteration if _final doesn't exist
        prefix = f"{preconditioner_type}_eta_{eta_mult}_w_{weight:.1f}_p_{poison_target}_iteration_"
        suffix = ".npy"
        highest_iteration = -1
        latest_file = None
        if os.path.exists(data_folder_path):
            for filename in os.listdir(data_folder_path):
                if filename.startswith(prefix) and filename.endswith(suffix):
                    try:
                        iteration = int(filename[len(prefix):-len(suffix)])
                        if iteration > highest_iteration:
                            highest_iteration = iteration
                            latest_file = os.path.join(data_folder_path, filename)
                    except ValueError:
                        continue
        name = latest_file

    if name and os.path.exists(name):
        print(f"Loading phase field data from {name}")
        phase_field = np.load(name, allow_pickle=True)
        # if i==1:
        #     phase_field_0 = np.copy(phase_field )
        # elif i==0:
        #     phase_field_0 = np.copy(phase_field)
        nb_tiles = 3

        pcm = ax.pcolormesh(np.tile(phase_field, (nb_tiles, nb_tiles)).transpose(),
                            shading='flat',
                            edgecolors='none',
                            lw=0.01,
                            cmap=mpl.cm.Greys,
                            vmin=0, vmax=1,
                            rasterized=True)
    else:
        ax.text(0.5, 0.5, f'Data missing for\np={poison_target}', ha='center', va='center')
    computed_poisson = np.asarray(nu12)[i]
    ax.text(0.5, 1.07, fr'$\nu^\mathrm{{target}}=$ {poison_target:0.1f}', transform=ax.transAxes, ha='center', fontsize=16)

    # below is for computed quantities
    # ax.text(0.5, 1.07, fr'$\nu_{{12}}^\mathrm{{target}}=$ {nu12_target_i:0.2f}', transform=ax.transAxes, ha='center', fontsize=16)
    # ax.text(-.10, 0.5, fr'$\nu_{{12}} =$ {computed_poisson:0.2f}',
    #         transform=ax.transAxes,
    #         rotation=90,
    #         ha='center', va='center',
    #         fontsize=16)

    letter_offset = -0.17
    import string

    letter = string.ascii_lowercase[i]  # 'A', 'B', 'C', ...
    #ax.text(letter_offset, 1.05, rf'$\textbf{{{letter}}}$', transform=ax.transAxes)
    ax.text(letter_offset, 1.05,  rf'$\textbf{{({letter})}}$', transform=ax.transAxes,fontsize=14)

    ax.set_aspect('equal')
    # ax.set_yticklabels([])
    # ax.set_xticklabels([])
    # ax.xaxis.set_ticks_position('none')
    # ax.yaxis.set_ticks_position('none')
    # ax.axis('off')
    # Apply identical limits
    ax.set_xticks([0, 1 * N, 2 * N, 3 * N])
    ax.set_yticks([0, 1 * N, 2 * N, 3 * N])

    ax.set_xticklabels([0, 1, 2, 3])
    ax.set_yticklabels([0, 1, 2, 3])
    if j > 1:
        ax.set_xlabel('Unit cell size  -  L')

    ax.set_ylabel('L')
    ax.yaxis.set_label_position("right")
    ax.yaxis.tick_right()

    ax.set_xlim(0, 3 * N)
    ax.set_ylim(0, 3 * N)

plt.tight_layout()
fname_pf = figure_folder_path  + 'exp5_square_' + f'w{weight:0.0f}' +'.pdf'
print(f'create figure: {fname_pf}')
fig.savefig(fname_pf, bbox_inches='tight', dpi=1200 )

# --- New figure for phase fields of all poisson_targets ---
n_targets = len(poisson_targets)
n_cols = 5
n_rows = (n_targets + n_cols - 1) // n_cols
fig_pf, axes_pf = plt.subplots(n_rows, n_cols, figsize=(9, 3 * n_rows))
axes_pf = axes_pf.flatten()

for i, poison_target in enumerate(poisson_targets):
    ax = axes_pf[i]

    # Try to load the final phase field
    # name = data_folder_path + f'{preconditioner_type}'+ f'_eta_{eta_mult}' + f'_w_{weight:.1f}' + f'_p_{poison_target}' + '_final' + f'.npy'
    name = data_folder_path + f'{preconditioner_type}'
    if not os.path.exists(name):
        # Fallback to searching for the highest iteration if _final doesn't exist
        prefix = f"{preconditioner_type}_eta_{eta_mult}_w_{weight:.1f}_p_{poison_target}_iteration_"
        suffix = ".npy"
        highest_iteration = -1
        latest_file = None
        if os.path.exists(data_folder_path):
            for filename in os.listdir(data_folder_path):
                if filename.startswith(prefix) and filename.endswith(suffix):
                    try:
                        iteration = int(filename[len(prefix):-len(suffix)])
                        if iteration > highest_iteration:
                            highest_iteration = iteration
                            latest_file = os.path.join(data_folder_path, filename)
                    except ValueError:
                        continue
        name = latest_file

    if name and os.path.exists(name):
        print(f"Loading phase field data from {name}")
        phase_field = np.load(name, allow_pickle=True)
        # if i==1:
        #     phase_field_0 = np.copy(phase_field )
        # elif i==0:
        #     phase_field_0 = np.copy(phase_field)
        nb_tiles = 1
        pcm = ax.pcolormesh(np.tile(phase_field, (nb_tiles, nb_tiles)),
                            shading='flat',
                            edgecolors='none',
                            lw=0.01,
                            cmap=mpl.cm.Greys,
                            vmin=0, vmax=1,
                            rasterized=True)
        # ax.set_title(r'$\nu_{{\text{target}=$'+f'{poison_target}')
        ax.set_title(fr'$\nu^\mathrm{{target}}=$ {poison_target}')

    else:
        ax.text(0.5, 0.5, f'Data missing for\np={poison_target}', ha='center', va='center')
    computed_poisson = np.asarray(nu12)[i]
    ax.text(0.5, -0.1, fr'$\nu =$ {computed_poisson:0.2f}', transform=ax.transAxes, ha='center')
    ax.set_aspect('equal')
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    ax.xaxis.set_ticks_position('none')
    ax.yaxis.set_ticks_position('none')

# Hide any extra subplots
for j in range(i + 1, len(axes_pf)):
    axes_pf[j].axis('off')

# Add unified colorbar for phase fields
fig_pf.subplots_adjust(right=0.85)
cbar_ax = fig_pf.add_axes([0.88, 0.15, 0.02, 0.7])
fig_pf.colorbar(pcm, cax=cbar_ax)

fig_pf.tight_layout(rect=[0, 0, 0.85, 1])
fname_pf = figure_folder_path + f'{weight}' + 'exp5_square_all_phase_fields.pdf'
print(f'create figure: {fname_pf}')
fig_pf.savefig(fname_pf, bbox_inches='tight')

# Keep original combined plots but update them if needed or just let them be.
# The user specifically asked to change it such that it plots for all poisson_targets.
# I will keep the existing structure but maybe simplify or fix it.
# Actually, the original script was trying to plot only 5.

fig = plt.figure(figsize=(11, 6.5))  # slightly taller to fit the extra subplot

gs = fig.add_gridspec(4, 4, hspace=0.1)  # increase rows from 3 → 4
ax5 = fig.add_subplot(gs[0:3, :])  # keep original plot spanning first 3 rows

# fig, ax = plt.subplots(1, 1, figsize=(8, 4))
ax5.loglog(poisson_targets, f_sigmas, '-', color='r', linewidth=2, marker='|',
           label=r'stress difference -  $f_{\sigma}$')
ax5.loglog(poisson_targets, f_pfs, '--', color='k', linewidth=1, marker='|', label=r'phase field - $f_{\rho}$')

# ax5.legend([r'stress difference -  $f_{\sigma}$', r'phase field - $f_{\rho}$'], loc='lower center')
# ax.set_aspect('equal')
ax5.set_title(r'Square grid : 3 load cases' + f' N={N}, Poisson={eta_mult}')
ax5.set_xlabel(r'Weight $a$')
ax5.set_xlim(0.1, 100)
ax5.set_ylim(1e-4, 1e1)
ax5.set_xticklabels([])

ax5.annotate(r'Stress difference -  $f_{\sigma}$', color='red',
             xy=(0.6, f_sigmas[np.where(poisson_targets == 0.1)[0][0]]),
             xytext=(1.0, 5.),
             arrowprops=dict(arrowstyle='->',
                             color='red',
                             lw=1,
                             ls='-')
             )
ax5.annotate(r'Phase field - $f_{\rho}$',
             xy=(50., f_pfs[np.where(poisson_targets == 0.3)[0][0]]),
             xytext=(20., 5.),
             arrowprops=dict(arrowstyle='->',
                             color='black',
                             lw=1,
                             ls='-')
             )
ax5.text(0.01, 0.95, r'$\textbf{{(a)}}$', transform=ax5.transAxes)

letter_offset = -0.15

for i in np.arange(len(poisson_targets)):
    if i >= 5:
        continue
    poison_target = poisson_targets[i]
    if i == 0:
        # ax1 = fig.add_subplot(gs[0, upper_ax])
        ax1 = fig.add_axes([0.12, 0.5, 0.18, 0.18], transform=ax5.transAxes)
        roll_x = -20
        roll_y = 5
        ax5.annotate('',
                     xy=(poison_target, f_sigmas[np.where(poisson_targets == poison_target)[0][0]]),
                     xytext=(0.23, 0.1),
                     arrowprops=dict(arrowstyle='->',
                                     color='black',
                                     lw=1,
                                     ls='-')
                     )
        ax5.text(letter_offset, 0.9, r'$\textbf{{A}}$', transform=ax1.transAxes)  #

    elif i == 1:
        ax1 = fig.add_axes([0.28, 0.39, 0.18, 0.18], transform=ax5.transAxes)
        roll_x = -26
        roll_y = 2
        ax5.annotate('',
                     xy=(poison_target, f_sigmas[np.where(poisson_targets == poison_target)[0][0]]),
                     xytext=(0.8, 0.01),
                     arrowprops=dict(arrowstyle='->',
                                     color='black',
                                     lw=1,
                                     ls='-')
                     )
        ax5.text(letter_offset, 0.9, r'$\textbf{{B}}$', transform=ax1.transAxes)

    elif i == 2:
        ax1 = fig.add_axes([0.44, 0.32, 0.18, 0.18], transform=ax5.transAxes)
        roll_x = 30
        roll_y = 16
        ax5.annotate('',
                     xy=(poison_target, f_sigmas[np.where(poisson_targets == poison_target)[0][0]]),
                     xytext=(5., 5e-4),
                     arrowprops=dict(arrowstyle='->',
                                     color='black',
                                     lw=1,
                                     ls='-')
                     )
        ax5.text(letter_offset, 0.9, r'$\textbf{{C}}$', transform=ax1.transAxes)

    elif i == 3:
        ax1 = fig.add_axes([0.56, 0.55, 0.18, 0.18], transform=ax5.transAxes)
        roll_x = 25
        roll_y = 10
        ax5.annotate('',
                     xy=(poison_target, f_sigmas[np.where(poisson_targets == poison_target)[0][0]]),
                     xytext=(10., 3e-2),
                     arrowprops=dict(arrowstyle='->',
                                     color='black',
                                     lw=1,
                                     ls='-')
                     )
        ax5.text(letter_offset, 0.9, r'$\textbf{{D}}$', transform=ax1.transAxes)

    elif i == 4:
        ax1 = fig.add_axes([0.72, 0.5, 0.18, 0.18], transform=ax5.transAxes)
        roll_x = 0
        roll_y = 0
        ax5.annotate('',
                     xy=(poison_target, f_sigmas[np.where(poisson_targets == poison_target)[0][0]]),
                     xytext=(50., 1e-2),
                     arrowprops=dict(arrowstyle='->',
                                     color='black',
                                     lw=1,
                                     ls='-')
                     )
        ax5.text(letter_offset, 0.9, r'$\textbf{{E}}$', transform=ax1.transAxes)

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

    # Load the file if found
    if latest_file is None:
        raise FileNotFoundError("No phase field files found matching the pattern.")

    print(f"Loading phase field data from {latest_file}")
    phase_field = np.load(latest_file, allow_pickle=True)
    # center the inclusion
    phase_opt = phase_field
    print(f'min = {phase_opt.min()}, max = {phase_opt.max()}')

    phase_opt = np.roll(phase_opt, roll_x, axis=0)
    phase_opt = np.roll(phase_opt, roll_y, axis=1)
    phase_opt = phase_opt.transpose((1, 0)).flatten(order='F')
    # create repeatable cells
    nb_cells = [3, 3]
    nb_additional_cells = 2
    ax1.set_aspect('equal')
    # ax1.set_xlim(0, nb_cells[0])
    # ax1.set_ylim(0, nb_cells[1] * +ymax)
    # plot solution
    nb_tiles = 3
    pcm = ax1.pcolormesh(np.tile(phase_field, (nb_tiles, nb_tiles)),
                         shading='flat',
                         edgecolors='none',
                         lw=0.01,
                         cmap=mpl.cm.Greys,
                         vmin=0, vmax=1,
                         rasterized=True)

    ax1.set_yticklabels([])
    ax1.set_xticklabels([])
    ax1.xaxis.set_ticks_position('none')
    ax1.yaxis.set_ticks_position('none')
    ax1.set_aspect('equal')
    # ax1.set_xlabel(f'w={weight:.1f}'.rstrip('0').rstrip('.'))
    # ax1.xaxis.set_label_position('bottom')
    # ax1.set_ylabel(r'Position y')
    ax1.set_yticklabels([])
    ax1.set_xticklabels([])

# --- new subplot underneath ---
ax6 = fig.add_subplot(gs[3, :])  # bottom row
ax6.semilogx(poisson_targets, zener_ratios, '-', color='b', linewidth=2, marker='|', label=r'Zener ratio')
ax6.set_xlabel(r'Weight $a$')
# ax6.set_ylabel(r'$a_r$')
# ax6.legend(loc='best')
# ax6.grid(True, which="both", ls="--", linewidth=0.5)
# ax6.grid(axis='y', which="both", visible=False)  # remove y-grid
ax6.set_xlim(0.1, 100)
ax6.set_ylim(0.6, 1.2)
ax6.annotate(r'Zener ratio', color='b',
             xy=(1., zener_ratios[np.where(poisson_targets == 0.1)[0][0]]),
             xytext=(0.5, 0.7),
             arrowprops=dict(arrowstyle='->',
                             color='b',
                             lw=1,
                             ls='-')
             )
ax6.text(0.01, 0.82, r'$\textbf{{(b)}}$', transform=ax6.transAxes)

fname = figure_folder_path + f'{weight}' + 'exp5_square{}'.format('.pdf')
print(('create figure: {}'.format(fname)))
plt.savefig(fname, bbox_inches='tight')
# plt.show()
