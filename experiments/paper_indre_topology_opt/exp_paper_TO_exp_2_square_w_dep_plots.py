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

fig = plt.figure(figsize=(11, 4.5))
gs = fig.add_gridspec(2, 5, width_ratios=[0.1, 1, 1, 1, 1])
ax0 = fig.add_subplot(gs[0, 0:])
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
#     [np.arange(0.1, 2., 0.1), np.arange(2, 3, 1), np.arange(3, 10, 2), np.arange(10, 110, 20), np.array([100.,150.0, 200.0, 300.0, 400.0, 500.0])])
weights=np.array([0.1,  0.3,  0.7,  1.0, 3.0, 7.0, 10.0, 30.0, 70.0, 100.0])
homogenized_Cij = np.zeros((3, 3, weights.shape[0]))
target_Cij = np.zeros((3, 3,weights.shape[0], ))
index = 0
N = 1024
for w_mult in weights:

    eta_mult = 0.01
    weight = w_mult
    script_name = 'exp_paper_TO_exp_2_square' + f'_random_{random_init}' + f'_N_{N}' + f'_cgtol_{cg_tol_exponent}' + f'_soft_{soft_phase_exponent}'

    preconditioner_type = "Green_Jacobi"
    file_folder_path = os.path.dirname(os.path.realpath(__file__))  # script directory
    data_folder_path = file_folder_path + '/exp_data/' + script_name + '/'
    figure_folder_path = file_folder_path + '/figures/' + script_name + '/'
    if not os.path.exists(figure_folder_path):
        os.makedirs(figure_folder_path)
    # name = (
    # f'{optimizer}_muFFTTO_elasticity_{element_type}_{script_name}_N{N}_E_target_{E_target_0}_Poisson_{poison_target}_Poisson0_0.0_w{w_mult:.2f}_eta{eta_mult}_p{p}_bounds={bounds}_FE_NuMPI{cores}_nb_load_cases_{nb_load_cases}_energy_objective_{energy_objective}_random_{random_initial_geometry}')
    prefix = f"{preconditioner_type}_eta_{eta_mult}_w_{weight:.1f}_iteration_"
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
    name = data_folder_path + f'{preconditioner_type}' + f'_eta_{eta_mult}' + f'_w_{weight:.1f}' + '_final' + f'.npy'
    try:
        phase_field = np.load(name, allow_pickle=True)
    except:
        print("No info")
    name_info = data_folder_path + f'{preconditioner_type}' + f'_eta_{eta_mult}' + f'_w_{weight:.1f}' + '_log' + f'.npz'
    try:
        info_ = np.load(name_info, allow_pickle=True)
    except:
        print("No info")

    f_sigmas.append(info_.f.norms_sigma[-1])
    f_pfs.append(info_.f.norms_pf[-1])
    f_adjoint.append(info_.f.norms_adjoint_energy[-1])
    # for Isotropic material
    Cij = info_.f.homogenized_C_ijkl
    homogenized_Cij[..., index] = info_.f.homogenized_C_ijkl

    Sij = np.linalg.inv(Cij)

    young_1_S = 1 / Sij[0, 0]
    young_2_S = 1 / Sij[1, 1]
    poison_ratios_1_S = - Sij[0, 1] / Sij[0, 0]
    poison_ratios_2_S = - Sij[1, 0] / Sij[1, 1]
    shear_S = 1 / Sij[2, 2]
    bulk_S = 1 / (Sij[0, 0]+2*Sij[1, 0]+Sij[1, 1])
    print(f'check symmetry condition {weight}  {poison_ratios_1_S / young_1_S - poison_ratios_2_S / young_2_S}')
    print(f'check shear coupling {weight}  S_13 {Sij[0, 2]}  S_23 {Sij[1, 2]}')
    print(f'check isotropy {weight}  S_11-S_22 {Sij[0, 0] - Sij[1, 1]}')

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
    target_Cij[ ...,index] = info_.f.target_C_ijkl

    S_compl_target = np.linalg.inv(Cij_target)

    nu12_target.append(- S_compl_target[0, 1] / S_compl_target[0, 0])
    nu21_target.append(- S_compl_target[1, 0] / S_compl_target[1, 1])
    E1_target.append(1 / S_compl_target[0, 0])
    shear_G_target.append(1/S_compl_target[2, 2])
    bulk_K_target.append(1 / (S_compl_target[0, 0] + 2 * S_compl_target[1, 0] + S_compl_target[1, 1]))

    index += 1

# plt.figure()
# plt.semilogx(weights, np.asarray(nu12), '-', color='r', linewidth=2, marker='|',
#              label=r' - Poisson ratio difference')
# plt.axhline(y=nu12_target, color='r', linestyle='-.', linewidth=2, label=r'Target Poisson ratio difference')
#
# plt.semilogx(weights, np.asarray(E1), '-', color='k', linewidth=2, marker='|',
#              label=r'C_{11} - Young modulus difference')
# plt.axhline(y=E1_target, color='k', linestyle='-.', linewidth=2, label=r'Target Young modulus difference')
#
# plt.semilogx(weights, np.asarray(C_22), '-', color='b', linewidth=2, marker='|',
#              label=r'C_{33} - Shear modulus difference')
# plt.axhline(y=G_target, color='b', linestyle='-.', linewidth=2, label=r'Target C_{33} - Shear modulus difference')
#
# plt.yscale('linear')
# plt.legend(loc='best')
# plt.xlabel(r'Weight $a$')
# plt.xlim(0.1, 100)
# plt.ylim(-0.5, 2)
# plt.title(r'Square grid zero poisson: 3 load cases' + f' N={N}, eta={eta_mult}')
# fname = figure_folder_path + 'exp2_square_convergence{}'.format('.pdf')
# print(('create figure: {}'.format(fname)))
# plt.savefig(fname, bbox_inches='tight')
# # plt.show()
#
# plt.figure()
# plt.semilogx(weights, np.abs(np.asarray(poison_ratios) - poison_target), '-', color='r', linewidth=2, marker='|',
#              label=r'Poisson ratio difference')
#
# plt.axhline(y=0.0, color='r', linestyle='-.', linewidth=2, label=r'Target Poisson ratio difference')
# plt.semilogx(weights, np.abs(np.asarray(Young_modulus) - Young_target), '--', color='k', linewidth=2, marker='|',
#              label=r'Young modulus difference')
# plt.axhline(y=0.0, color='k', linestyle=':', linewidth=2, label=r'Target Young modulus difference')
#
# plt.semilogx(weights, np.abs(c13_), '--', color='k', linewidth=1, marker='|', label=r'$C_{1,3}$')
# plt.semilogx(weights, np.abs(c23_), '--', color='k', linewidth=1, marker='|', label=r'$C_{2,3}$')
#
# plt.yscale('log')
# plt.legend(loc='best')
# plt.xlabel(r'Weight $a$')
# plt.xlim(0.1, 100)
# plt.ylim(1e-5, 10)
# # plt.show()

# ----------------------------------------------------------------------------------------------------- #
fig = plt.figure(figsize=(5.5, 3.5))
gs_global = fig.add_gridspec(1, 1, width_ratios=[1], hspace=0.00)

G_0 = 0.5
K_0 = 1.0
ax_modulus = fig.add_subplot(gs_global[0, 0])

ax_modulus.plot(weights, np.asarray(E1_target) / K_0, '-.', color='r', linewidth=1, marker='x',
                label=r'Target - Shear G')
ax_modulus.plot(weights, np.asarray(E1) / K_0, '-', color='r', linewidth=2, marker='x',
                label=r'Computed - Shear G')
ax_modulus.annotate(r'$E_1^\mathrm{{eff}}/K^0$', color='red',
                    xy=(0.1, E1[np.where(weights == 0.1)[0][0]]),
                    xytext=(0.1, 0.55),
                    arrowprops=dict(arrowstyle='->',
                                    color='red',
                                    lw=1,
                                    ls='-')
                    )
ax_modulus.annotate(r'$E_1^\mathrm{{target}}/K^0$', color='red',
                    xy=(0.3, E1_target[np.where(weights == 0.3)[0][0]]),
                    xytext=(0.05, 0.62),
                    arrowprops=dict(arrowstyle='->',
                                    color='red',
                                    lw=1,
                                    ls='--')
                    )


ax_modulus.plot(weights, np.asarray(shear_G_target) / G_0, '-.', color='b', linewidth=1, marker='^',
                label=r'Target - Shear G')
ax_modulus.plot(weights, np.asarray(shear_G_computed) / G_0, '-', color='b', linewidth=2, marker='^',
                label=r'Computed - Shear G')
ax_modulus.annotate(r'$\mu^\mathrm{{eff}}/\mu^0$', color='blue',
                    xy=(0.3, shear_G_computed[np.where(weights ==0.3)[0][0]] / G_0),
                    xytext=(0.25, 0.6),
                    arrowprops=dict(arrowstyle='->',
                                    color='blue',
                                    lw=1,
                                    ls='-')
                    )
ax_modulus.annotate(r'$\mu^\mathrm{{target}}/\mu^0$', color='blue',
                    xy=(0.7, shear_G_target[np.where(weights == 0.7)[0][0]] / G_0),
                    xytext=(0.45, 0.57),
                    arrowprops=dict(arrowstyle='->',
                                    color='blue',
                                    lw=1,
                                    ls='--')
                    )

ax_modulus.plot(weights, np.asarray(bulk_K_target) / K_0, '-.', color='k', linewidth=1, marker='|',
                label=r'Target - Bulk K')
ax_modulus.plot(weights, np.asarray(bulk_K_computed) / K_0, '-', color='k', linewidth=2, marker='|',
                label=r'Computed - Bulk K')
ax_modulus.annotate(r'$K^\mathrm{{eff}}/K^0$', color='k',
                    xy=(0.3, bulk_K_computed[np.where(weights == 0.3)[0][0]]),
                    xytext=(0.46, 0.16),
                    arrowprops=dict(arrowstyle='->',
                                    color='k',
                                    lw=1,
                                    ls='-')
                    )
ax_modulus.annotate(r'$K^\mathrm{{target}}/K^0$', color='k',
                    xy=(0.7, bulk_K_target[np.where(weights == 0.7)[0][0]]),
                    xytext=(0.43, 0.1),
                    arrowprops=dict(arrowstyle='->',
                                    color='k',
                                    lw=1,
                                    ls='--')
                    )

ax_modulus.semilogx(weights, np.asarray(nu12_target), '-.', color='g', linewidth=1, marker='o',
                label=r'Target - Poisson')
ax_modulus.semilogx(weights, np.asarray(nu12), '-', color='g', linewidth=2, marker='o',
                label=r'Computed - Poisson')
ax_modulus.annotate(r'$\nu^\mathrm{{eff}}_{12}$', color='green',
                    xy=(0.2, nu12[np.where(weights == 3.0)[0][0]]),
                    xytext=(0.1, 0.22),
                    arrowprops=dict(arrowstyle='->',
                                    color='green',
                                    lw=1,
                                    ls='-')
                    )
ax_modulus.annotate(r'$\nu^\mathrm{{target}}_{12}$', color='green',
                    xy=(1.0, nu12_target[np.where(weights == 1.0)[0][0]]),
                    xytext=(1.2, 0.4),
                    arrowprops=dict(arrowstyle='->',
                                    color='green',
                                    lw=1,
                                    ls='--')
                    )

ax_modulus.annotate(r'$1024^2$ grid points' + '\n' + rf'$w={weight}$',
                    xy=(0.2, -0.35),
                    xytext=(0.0, -0.45))
ax_modulus.set_title(r"Square grid")

ax_modulus.set_xlabel(r'Weight $a$')
ax_modulus.set_xlim(weights[0], weights[-1])
ax_modulus.set_ylim(-0.1, 1.1)
# ax_modulus.set_yticks([-0.5, -0.25, 0.0, 0.25, 0.5])
# ax_modulus.set_xticks([-0.5, -0.3, -0.1, 0.1, 0.3])
ax_modulus.grid(axis='y', which='major', linestyle='-', linewidth=0.5, alpha=0.7)
# ax_modulus.set_xscale('log')
#ax_modulus.set_yscale('linear')
fname_pf = figure_folder_path + 'exp2_square_' + f'w{weight:.0f}' + '_graph.pdf'
print(f'create figure: {fname_pf}')
fig.savefig(fname_pf, bbox_inches='tight')
plt.show()


#- ---------------------------------

fig = plt.figure(figsize=(11, 7.5))  # slightly taller to fit the extra subplot

gs = fig.add_gridspec(5, 4, hspace=0.1)  # increase rows from 3 → 4
ax5 = fig.add_subplot(gs[0:3, :])  # keep original plot spanning first 3 rows

# fig, ax = plt.subplots(1, 1, figsize=(8, 4))
ax5.loglog(weights, f_sigmas, '-', color='r', linewidth=2, marker='|', label=r'stress difference -  $f_{\sigma}$')
ax5.loglog(weights, f_pfs, '--', color='k', linewidth=1, marker='|', label=r'phase field - $f_{\rho}$')

# ax5.legend([r'stress difference -  $f_{\sigma}$', r'phase field - $f_{\rho}$'], loc='lower center')
# ax.set_aspect('equal')
#plt.title(r'Square grid zero poisson: 3 load cases' + f' N={N}, eta={eta_mult}')

ax5.set_xlabel(r'Weight $a$')
ax5.set_xlim(0.1, 100)
ax5.set_ylim(1e-6, 3e1)
ax5.set_xticklabels([])

ax5.annotate(r'Stress difference -  $f_{\sigma}$', color='red',
             xy=(weights[3], f_sigmas[np.where(weights == 1)[0][0]]),
             xytext=(1.0, 5.),
             arrowprops=dict(arrowstyle='->',
                             color='red',
                             lw=1,
                             ls='-')
             )
ax5.annotate(r'Phase field - $f_{\rho}$',
             xy=(weights[8], f_pfs[np.where(weights == 70.0)[0][0]]),
             xytext=(20., 5.),
             arrowprops=dict(arrowstyle='->',
                             color='black',
                             lw=1,
                             ls='-')
             )
ax5.text(0.01, 0.95, r'$\textbf{{(a)}}$', transform=ax5.transAxes)

letter_offset = -0.15

inset_size=0.16
for upper_ax in np.arange(5):
    weight = np.array([weights[1], weights[2], weights[4], weights[6], weights[-2]])[upper_ax]
    #weight = weights[upper_ax] 10

    if upper_ax == 0:
        # ax1 = fig.add_subplot(gs[0, upper_ax])
        ax1 = fig.add_axes([0.13, 0.58, inset_size, inset_size], transform=ax5.transAxes)
        roll_x = 128
        roll_y = -356
        ax5.annotate('',
                     xy=(weight, f_sigmas[np.where(weights == weight)[0][0]]),
                     xytext=(0.23, 0.1),
                     arrowprops=dict(arrowstyle='->',
                                     color='black',
                                     lw=1,
                                     ls='-')
                     )
        ax5.text(letter_offset, 0.9, r'$\textbf{{A}}$', transform=ax1.transAxes)  #

    elif upper_ax == 1:
        ax1 = fig.add_axes([0.28, 0.53, inset_size, inset_size], transform=ax5.transAxes)
        roll_x = -128
        roll_y = -160
        ax5.annotate('',
                     xy=(weight, f_sigmas[np.where(weights == weight)[0][0]]),
                     xytext=(0.8, 0.01),
                     arrowprops=dict(arrowstyle='->',
                                     color='black',
                                     lw=1,
                                     ls='-')
                     )
        ax5.text(letter_offset, 0.9, r'$\textbf{{B}}$', transform=ax1.transAxes)

    elif upper_ax == 2:
        ax1 = fig.add_axes([0.44, 0.48, inset_size, inset_size], transform=ax5.transAxes)
        roll_x = -64
        roll_y = 332
        ax5.annotate('',
                     xy=(weight, f_sigmas[np.where(weights == weight)[0][0]]),
                     xytext=(5., 5e-4),
                     arrowprops=dict(arrowstyle='->',
                                     color='black',
                                     lw=1,
                                     ls='-')
                     )
        ax5.text(letter_offset, 0.9, r'$\textbf{{C}}$', transform=ax1.transAxes)

    elif upper_ax == 3:
        ax1 = fig.add_axes([0.60, 0.43, inset_size, inset_size], transform=ax5.transAxes)
        roll_x = 340
        roll_y = -180
        ax5.annotate('',
                     xy=(weight, f_sigmas[np.where(weights == weight)[0][0]]),
                     xytext=(14., 3e-4),
                     arrowprops=dict(arrowstyle='->',
                                     color='black',
                                     lw=1,
                                     ls='-')
                     )
        ax5.text(letter_offset, 0.9, r'$\textbf{{D}}$', transform=ax1.transAxes)

    elif upper_ax == 4:
        ax1 = fig.add_axes([0.76, 0.62, inset_size, inset_size], transform=ax5.transAxes)
        roll_x = 0
        roll_y = 0
        ax5.annotate('',
                     xy=(weight, f_sigmas[np.where(weights == weight)[0][0]]),
                     xytext=(50., 1e-2),
                     arrowprops=dict(arrowstyle='->',
                                     color='black',
                                     lw=1,
                                     ls='-')
                     )
        ax5.text(letter_offset, 0.9, r'$\textbf{{E}}$', transform=ax1.transAxes)

    prefix = f"{preconditioner_type}_eta_{eta_mult}_w_{weight:.1f}_iteration_"
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
    # plotting part
    # center the inclusion
    phase_opt = phase_field
    print(f'min = {phase_opt.min()}, max = {phase_opt.max()}')
    phase_opt = np.roll(phase_opt, roll_x, axis=0)
    phase_opt = np.roll(phase_opt, roll_y, axis=1)
    #phase_opt = phase_opt.transpose((1, 0)).flatten(order='F')
    # create repeatable cells
    nb_cells = [3, 3]
    nb_additional_cells = 2
    ax1.set_aspect('equal')
    # ax1.set_xlim(0, nb_cells[0])
    # ax1.set_ylim(0, nb_cells[1] * +ymax)
    # plot solution
    nb_tiles = 3
    pcm = ax1.pcolormesh(np.linspace(0, nb_tiles, phase_opt.shape[0] * nb_tiles + 1),
                         np.linspace(0, nb_tiles, phase_opt.shape[1] * nb_tiles + 1),
                         np.tile(phase_opt, (nb_tiles, nb_tiles)),
                         shading='flat',
                         edgecolors='none',
                         lw=0.01,
                         cmap=mpl.cm.Greys,
                         rasterized=True)

    # parall.set_alpha(1.0)  # Set alpha to fully opaque
    ax1.hlines(np.arange(1, nb_tiles), 0, nb_tiles, colors='w', linestyles='--', linewidth=0.5)
    ax1.vlines(np.arange(1, nb_tiles), 0, nb_tiles, colors='w', linestyles='--', linewidth=0.5)

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
ax6.semilogx(weights, zener_ratios, '-', color='b', linewidth=2, marker='|', label=r'Zener ratio')
#ax6.set_xlabel(r'Weight $a$')
# ax6.set_ylabel(r'$a_r$')
# ax6.legend(loc='best')
# ax6.grid(True, which="both", ls="--", linewidth=0.5)
# ax6.grid(axis='y', which="both", visible=False)  # remove y-grid
ax6.set_xlim(0.1, 100)
ax6.set_ylim(0.5, 1.2)
ax6.annotate(r'Zener ratio', color='b',
             xy=(1., zener_ratios[np.where(weights == 1.0)[0][0]]),
             xytext=(0.5, 1.0),
             arrowprops=dict(arrowstyle='->',
                             color='b',
                             lw=1,
                             ls='-')
             )
ax6.text(0.01, 0.82, r'$\textbf{{(b)}}$', transform=ax6.transAxes)
ax6.set_xticklabels([])

# --- new subplot underneath ---
ax_poisson = fig.add_subplot(gs[4, :])  # bottom row
ax_poisson.semilogx(weights, np.asarray(nu12), '-', color='olivedrab', linewidth=2, marker='|', label=r'Poisson ratio')
ax_poisson.set_xlabel(r'Weight $a$')
ax_poisson.set_xlim(0.1, 100)
ax_poisson.set_ylim(-0.2, 0.4)
ax_poisson.annotate(r"Poisson's ratio", color='olivedrab',
             xy=(3.,  np.asarray(nu12)[np.where(weights == 3.0)[0][0]]),
             xytext=(7.0, 0.25),
             arrowprops=dict(arrowstyle='->',
                             color='olivedrab',
                             lw=1,
                             ls='-')
             )
# ax_poisson.axhline(y=0, color='black', linestyle='--', linewidth=1)
ax_poisson.semilogx(weights, np.asarray(nu12_target),  color='black', linestyle='--', linewidth=1, label=r'Poisson ratio')

ax_poisson.annotate(r"Target Poisson's ratio", color='black',
             xy=(0.5, 0),
             xytext=(0.15, -0.15),
             arrowprops=dict(arrowstyle='->',
                             color='black',
                             lw=1,
                             ls='-')
             )

ax_poisson.text(0.01, 0.65, r'$\textbf{{(c)}}$', transform=ax_poisson.transAxes)


fname = figure_folder_path + 'exp2_square{}'.format('.pdf')
print(('create figure: {}'.format(fname)))
plt.savefig(fname, bbox_inches='tight')
# plt.show()

