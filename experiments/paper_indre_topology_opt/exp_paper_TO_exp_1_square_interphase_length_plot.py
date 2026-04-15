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

colors = ['red', 'blue', 'green', 'orange', 'purple']
linestyles = ['-', '--', '-.', ':', (0, (3, 1, 1, 1))]



f_sigmas = []
f_pfs = []
f_adjoint = []

# problem parameters:
cg_tol_exponent = 8
soft_phase_exponent = 5
random_init = False

weights=[0.1]
# eta = 0.07
index = 0
N = 64
eta=3*np.sqrt(2)/N

for w_mult in weights:


    weight = w_mult
    script_name = 'exp_paper_TO_exp_1_square_interphase_length' + f'_random_{random_init}' + f'_N_{N}' + f'_cgtol_{cg_tol_exponent}' + f'_soft_{soft_phase_exponent}'

    preconditioner_type = "Green_Jacobi"
    file_folder_path = os.path.dirname(os.path.realpath(__file__))  # script directory
    data_folder_path = file_folder_path + '/exp_data/' + script_name + '/'
    figure_folder_path = file_folder_path + '/figures/' + script_name + '/'

    name = data_folder_path + f'{preconditioner_type}' + f'_eta_{eta:.2f}' + f'_w_{weight:.1f}' + '_final' + f'.npy'
    try:
        phase_field = np.load(name, allow_pickle=True)
        print(f'phase_field { phase_field.max()}')
        print(f'phase_field { phase_field.min()}')

    except:
        print("No info")
    name_info = data_folder_path + f'{preconditioner_type}' + f'_eta_{eta:.2f}' + f'_w_{weight:.1f}' + '_log' + f'.npz'
    try:
        info_ = np.load(name_info, allow_pickle=True)
    except:
        print("No info")

f_rho_grad=info_.f.norms_pf_grad[-1]
f_dw=info_.f.norms_pf_rho[-1]
print(f'phase_field norms_pf = { info_.f.norms_pf[-1]}')
print(f'phase_field f_rho_grad = {f_rho_grad }')
print(f'phase_field f_dw = {f_dw }')

f_rho = eta * f_rho_grad +  f_dw / eta
print(f'phase_field f_rho = {f_rho }')
print(f'Length = {eta * f_rho_grad }')
# ----------------------------------------------------------------------------------------------------- #
fig = plt.figure(figsize=(5.5, 3.5))
gs_global = fig.add_gridspec(1, 1, width_ratios=[1], hspace=0.00)

ax1 = fig.add_subplot(gs_global[:, :])  # keep original plot spanning first 3 rows

nb_tiles = 3
pcm = ax1.pcolormesh(np.linspace(0, nb_tiles, phase_field.shape[0] * nb_tiles + 1),
                     np.linspace(0, nb_tiles, phase_field.shape[1] * nb_tiles + 1),
                     np.tile(phase_field, (nb_tiles, nb_tiles)),
                     vmin=0, vmax=1,
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
plt.show()