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
Young_modulus= []
c13_= []
c23_= []
E1= []
nu12= []
C_22= []

# weights = np.concatenate(
#     [np.arange(0.1, 2., 0.1), np.arange(2, 3, 1), np.arange(3, 10, 2), np.arange(10, 110, 20), np.array([150.0, 200.0])])#, 300.0, 400.0, 500.0
#weights=np.array([0.1,  0.3,  0.7,  1.0, 3.0, 7.0, 10.0, 30.0, 70.0, 100.0,  300.0, 700.0, 1000.])
weights=np.array([0.1,  0.3,  0.7,  1.0, 3.0, 7.0, 10.0, 30.0, 70.0, 100.0])


# weights=[5]
N = 1024
nb_tiles = 4

# for domain size
x_ref = np.zeros([2, nb_tiles * (N) + 1, nb_tiles * (N) + 1])
x_ref[0], x_ref[1] = np.meshgrid(np.linspace(0, nb_tiles, nb_tiles * (N) + 1),
                                 np.linspace(0, nb_tiles, nb_tiles * (N) + 1), indexing='ij')
shift = 0.5 * np.linspace(0, nb_tiles, nb_tiles * (N) + 1)
x_coords = np.copy(x_ref)
# Apply shift to each row
x_coords[0] += shift[None, :]-2
x_coords[1] *= np.sqrt(3) / 2

for w_mult in weights:

    eta_mult = 0.01
    weight = w_mult
    script_name = 'exp_paper_TO_exp_3_hexa' + f'_random_{random_init}' + f'_N_{N}' + f'_cgtol_{cg_tol_exponent}' + f'_soft_{soft_phase_exponent}'

    preconditioner_type = "Green_Jacobi"
    file_folder_path = os.path.dirname(os.path.realpath(__file__))  # script directory
    data_folder_path = file_folder_path + '/exp_data/' + script_name + '/'
    figure_folder_path = file_folder_path + '/figures/' + script_name + '/'
    if not os.path.exists(figure_folder_path):
        os.makedirs(figure_folder_path)
    # name = (
    # f'{optimizer}_muFFTTO_elasticity_{element_type}_{script_name}_N{N}_E_target_{E_target_0}_Poisson_{poison_target}_Poisson0_0.0_w{w_mult:.2f}_eta{eta_mult}_p{p}_bounds={bounds}_FE_NuMPI{cores}_nb_load_cases_{nb_load_cases}_energy_objective_{energy_objective}_random_{random_initial_geometry}')

    # name =  data_folder_path + f'{preconditioner_type}' + f'_eta_{eta_mult}'+ f'_w_{weight}'  +'_final' + f'.npy'
    name = data_folder_path + f'{preconditioner_type}' + f'_eta_{eta_mult}' + f'_w_{weight:.1f}'+ '_final' + f'.npy'
    try:
        phase_field = np.load(name, allow_pickle=True)
    except:
        print(f"Failed to load phase field data from {name}. Using default phase field.")
        #phase_field = np.load(name, allow_pickle=True)
    name_info = data_folder_path + f'{preconditioner_type}' + f'_eta_{eta_mult}' + f'_w_{weight:.1f}'+ '_log' + f'.npz'
    try:
        info_ = np.load(name_info, allow_pickle=True)
    except:
        print(f"Loaded info from {name_info}.")

    f_sigmas.append(info_.f.norms_sigma[-1] )
    f_pfs.append(info_.f.norms_pf[-1])
    f_adjoint.append(info_.f.norms_adjoint_energy[-1])
    # for Isotropic material
    Cij = info_.f.homogenized_C_ijkl
    zener_ratios.append(2 * Cij[2, 2] / (Cij[0, 0] - Cij[0, 1]))
    lam, mu = Cij[0, 1], Cij[2, 2]

    poison_ratios.append(lam / (2 * (lam + mu)))
    Young_modulus.append(mu * (3 * lam + 2 * mu) / (lam + mu))

    c13_.append(Cij[0, 2])
    c23_.append(Cij[1, 2])
    # for Orthotropic material
    S_compl = np.linalg.inv(Cij)

    E1.append(1 / S_compl[0, 0])  # ≈ 0.25
    nu12.append(-S_compl[0, 1] / S_compl[0, 0])  # ≈ -0.33
    C_22.append(1 / S_compl[2, 2])
    # poison_ratios.append(nu12)
    # Young_modulus.append(E1)

Cij_target = info_.f.target_C_ijkl
lam, mu = Cij_target[0, 1], Cij_target[2, 2]

poison_target = lam / (2 * (lam + mu))
Young_target = mu * (3 * lam + 2 * mu) / (lam + mu)

S_compl_target = np.linalg.inv(Cij_target)

E1_target = 1 / S_compl_target[0, 0]  # ≈ 0.25
nu12_target = -S_compl_target[0, 1] / S_compl_target[0, 0]  # ≈ -0.33
G_target = 1 / S_compl_target[2, 2]

plt.figure()
plt.semilogx(weights, np.asarray(nu12), '-', color='r', linewidth=2, marker='|',
             label=r' - Poisson ratio difference')
plt.axhline(y=nu12_target, color='r', linestyle='-.', linewidth=2, label=r'Target Poisson ratio difference')

plt.semilogx(weights, np.asarray(E1), '-', color='k', linewidth=2, marker='|',
             label=r'C_{11} - Young modulus difference')
plt.axhline(y=E1_target, color='k', linestyle='-.', linewidth=2, label=r'Target Young modulus difference')

plt.semilogx(weights, np.asarray(C_22), '-', color='b', linewidth=2, marker='|',
             label=r'C_{33} - Shear modulus difference')
plt.axhline(y=G_target, color='b', linestyle='-.', linewidth=2, label=r'Target C_{33} - Shear modulus difference')

plt.yscale('linear')
plt.legend(loc='best')
plt.xlabel(r'Weight $a$')
plt.xlim(0.1, 100)
plt.ylim(-0.5, 2)
plt.title(r'Hexagonal grid : 2 load cases'+ f' N={N}, eta={eta_mult}')
fname = figure_folder_path + 'exp3_hexa_convergence{}'.format('.pdf')
print(('create figure: {}'.format(fname)))
plt.savefig(fname, bbox_inches='tight')
# plt.show()
plt.figure()
plt.semilogx(weights, np.abs(np.asarray(poison_ratios)-poison_target), '-', color='r', linewidth=2, marker='|', label=r'Poisson ratio difference')

plt.axhline(y=0.0, color='r', linestyle='-.', linewidth=2, label=r'Target Poisson ratio difference')
plt.semilogx(weights, np.abs(np.asarray(Young_modulus)-Young_target), '--', color='k', linewidth=2, marker='|', label=r'Young modulus difference')
plt.axhline(y=0.0, color='k', linestyle=':', linewidth=2, label=r'Target Young modulus difference')

plt.semilogx(weights, np.abs(c13_), '--', color='k', linewidth=1, marker='|', label=r'$C_{1,3}$')
plt.semilogx(weights, np.abs(c23_), '--', color='k', linewidth=1, marker='|', label=r'$C_{2,3}$')

plt.yscale('log')
plt.legend(loc='best')
plt.xlabel(r'Weight $a$')
plt.xlim(0.1, 100)
plt.ylim(1e-5, 10)
# plt.show()

fig = plt.figure(figsize=(11, 6.5))  # slightly taller to fit the extra subplot

gs = fig.add_gridspec(4, 4, hspace=0.1)  # increase rows from 3 → 4
ax5 = fig.add_subplot(gs[0:3, :])  # keep original plot spanning first 3 rows

# fig, ax = plt.subplots(1, 1, figsize=(8, 4))
ax5.loglog(weights, f_sigmas, '-', color='r', linewidth=2, marker='|', label=r'stress difference -  $f_{\sigma}$')
ax5.loglog(weights, f_pfs, '--', color='k', linewidth=1, marker='|', label=r'phase field - $f_{\rho}$')

# ax5.legend([r'stress difference -  $f_{\sigma}$', r'phase field - $f_{\rho}$'], loc='lower center')
# ax.set_aspect('equal')
ax5.set_title(r'Hexagonal grid : 2 load cases'+ f' N={N}, eta={eta_mult}')

ax5.set_xlabel(r'Weight $a$')
ax5.set_xlim(0.1, 100)
ax5.set_ylim(1e-5, 1e1)
ax5.set_xticklabels([])

ax5.annotate(r'Stress difference -  $f_{\sigma}$', color='red',
             xy=(0.6, f_sigmas[np.where(weights == 1.0)[0][0]]),
             xytext=(1.0, 5.),
             arrowprops=dict(arrowstyle='->',
                             color='red',
                             lw=1,
                             ls='-')
             )
ax5.annotate(r'Phase field - $f_{\rho}$',
             xy=(50., f_pfs[np.where(weights == 70.0)[0][0]]),
             xytext=(20., 5.),
             arrowprops=dict(arrowstyle='->',
                             color='black',
                             lw=1,
                             ls='-')
             )
ax5.text(0.01, 0.95, r'$\textbf{{(a)}}$', transform=ax5.transAxes)

letter_offset = -0.15

for upper_ax in np.arange(5):
    weight = np.array([weights[0], weights[2], weights[4], weights[6], weights[-1]])[upper_ax]
    #weight = weights[upper_ax] #10

    if upper_ax == 0:
        # ax1 = fig.add_subplot(gs[0, upper_ax])
        ax1 = fig.add_axes([0.12, 0.5, 0.18, 0.18], transform=ax5.transAxes)
        roll_x = -20
        roll_y = 5
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
        ax1 = fig.add_axes([0.28, 0.39, 0.18, 0.18], transform=ax5.transAxes)
        roll_x = -26
        roll_y = 2
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
        ax1 = fig.add_axes([0.44, 0.32, 0.18, 0.18], transform=ax5.transAxes)
        roll_x = 30
        roll_y = 16
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
        ax1 = fig.add_axes([0.56, 0.55, 0.18, 0.18], transform=ax5.transAxes)
        roll_x = 25
        roll_y = 10
        ax5.annotate('',
                     xy=(weight, f_sigmas[np.where(weights == weight)[0][0]]),
                     xytext=(10., 3e-2),
                     arrowprops=dict(arrowstyle='->',
                                     color='black',
                                     lw=1,
                                     ls='-')
                     )
        ax5.text(letter_offset, 0.9, r'$\textbf{{D}}$', transform=ax1.transAxes)

    elif upper_ax == 4:
        ax1 = fig.add_axes([0.72, 0.5, 0.18, 0.18], transform=ax5.transAxes)
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


    #name = data_folder_path + f'{preconditioner_type}' + f'_eta_{eta_mult}' + f'_w_{weight:.1f}'+ '_final' + f'.npy'
    # name = data_folder_path + f'{preconditioner_type}' + f'_eta_{eta_mult}' + f'_w_{weight:.1f}' + '_iteration_700' + f'.npy'
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
    # center the inclusion
    phase_opt = phase_field
    print(f'min = {phase_opt.min()}, max = {phase_opt.max()}')
    # center the inclusion
    phase_opt = phase_field

    phase_opt = np.roll(phase_opt, roll_x, axis=0)
    phase_opt = np.roll(phase_opt, roll_y, axis=1)
    phase_opt = phase_opt.transpose((1, 0)).flatten(order='F')
    # create repeatable cells
    ax1.set_aspect('equal')
    ax1.set_xlim(0, nb_tiles-2)
    ax1.set_ylim(0, nb_tiles-2)
    # plot solution
    pcm=ax1.pcolormesh( x_coords[0], x_coords[1],np.tile(phase_field, (nb_tiles, nb_tiles)),
                           shading='flat',
                           edgecolors='none',
                           lw=0.01,
                           cmap=mpl.cm.Greys,
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
ax6.semilogx(weights, zener_ratios, '-', color='b', linewidth=2, marker='|', label=r'Zener ratio')
ax6.set_xlabel(r'Weight $a$')
# ax6.set_ylabel(r'$a_r$')
# ax6.legend(loc='best')
# ax6.grid(True, which="both", ls="--", linewidth=0.5)
# ax6.grid(axis='y', which="both", visible=False)  # remove y-grid
ax6.set_xlim(0.1, 100)
ax6.set_ylim(0.6, 1.2)
ax6.annotate(r'Zener ratio', color='b',
             xy=(1., zener_ratios[np.where(weights == 1.0)[0][0]]),
             xytext=(0.5, 0.7),
             arrowprops=dict(arrowstyle='->',
                             color='b',
                             lw=1,
                             ls='-')
             )
ax6.text(0.01, 0.82, r'$\textbf{{(b)}}$', transform=ax6.transAxes)

fname = figure_folder_path + 'exp3_hexa{}'.format('.pdf')

print(('create figure: {}'.format(fname)))
plt.savefig(fname, bbox_inches='tight')
# plt.show()
