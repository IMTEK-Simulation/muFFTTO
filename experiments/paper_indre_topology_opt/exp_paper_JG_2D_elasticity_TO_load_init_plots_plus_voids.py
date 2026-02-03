import matplotlib as mpl
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import os
from scipy.ndimage import gaussian_filter


script_name = 'exp_paper_JG_2D_elasticity_TO_load_init'
file_folder_path = os.path.dirname(os.path.realpath(__file__))  # script directory
data_folder_path = file_folder_path + '/exp_data/' + script_name + '/'
figure_folder_path = file_folder_path + '/figures/'

src = '../figures/'  # source folder\

# Enable LaTeX rendering
plt.rcParams.update({
})
plt.rcParams.update({'font.size': 10})
plt.rcParams["font.family"] = "Arial"
# Publication-quality settings
plt.rcParams.update({
    "text.usetex": True,  # Use LaTeX
    'font.family': 'serif',
    'font.size': 12,
    'axes.labelsize': 14,
    'figure.dpi': 300,
    'savefig.dpi': 300,
})

soft_phase_exponent = 5
cg_tol = 8
# Ns = [16, 32, 64, 128, 256, 512, 1024]
Ns = [1024, ]  #[32, ]  # 512, 1024
nb_tiles = 5
steps = np.arange(0, 8000, 200)
for j in np.arange(len(Ns)):
    N = Ns[j]
    # Reference coordinates

    x_ref = np.zeros([2, nb_tiles * (N) + 1, nb_tiles * (N) + 1])
    # for index position
    # x_ref[0], x_ref[1] = np.meshgrid(np.arange(0, nb_tiles * (N) + 1), np.arange(0, nb_tiles * (N) + 1), indexing='ij')
    # shift = 0.5 * np.arange(nb_tiles * (N) + 1)

    # for domain size
    x_ref[0], x_ref[1] = np.meshgrid(np.linspace(0, nb_tiles, nb_tiles * (N) + 1),
                                     np.linspace(0, nb_tiles, nb_tiles * (N) + 1), indexing='ij')
    shift = 0.5 * np.linspace(0, nb_tiles, nb_tiles * (N) + 1)
    x_coords = np.copy(x_ref)
    # Apply shift to each row
    x_coords[0] += shift[None, :]
    x_coords[1] *= np.sqrt(3) / 2

    iter = 1000#v310#
    preconditioner_type = 'Green_Jacobi'  #
    random = False
    try:
        script_name_load = f'exp_paper_JG_2D_elasticity_TO_load_init_random_{random}' + f'_N_{N}' + f'_cgtol_{cg_tol}' + f'_soft_{soft_phase_exponent}' + '/'
        # script_name = f'exp_paper_JG_2D_elasticity_TO_load_init_random_{random}' + f'_N_{N}'+ f'_cgtol_{cg_tol}'+'/'

        phase_field_it_F = np.load(
            './exp_data/' + script_name_load + f'{preconditioner_type}' + f'_iteration_{iter}' + '.npy',
            allow_pickle=True)
        iter_hex = 5620#640#2730
        script_name_load = f'exp_paper_JG_2D_elasticity_TO_load_init_hexa_random_{random}' + f'_N_{N}' + f'_cgtol_{cg_tol}' + f'_soft_{soft_phase_exponent}' + '/'
        # script_name = f'exp_paper_JG_2D_elasticity_TO_load_init_random_{random}' + f'_N_{N}'+ f'_cgtol_{cg_tol}'+'/'

        phase_field_it_F_hex = np.load(
            './exp_data/' + script_name_load + f'{preconditioner_type}' + f'_iteration_{iter_hex}' + '.npy',
            allow_pickle=True)
        #new=np.roll(phase_field_it_F_hex,shift=10, axis=1)+   phase_field_it_F_hex
        # #new[new > 1] = 1
        # # Apply Gaussian blur then threshold to remove small features
        # sigma = 1.0  # larger = more aggressive smoothing
        # threshold = 0.9
        #
        # smoothed = gaussian_filter(phase_field_it_F_hex.astype(float), sigma=sigma)
        # filtered = (smoothed > threshold).astype(int)
        # np.save(
        #     './exp_data/' + script_name_load + f'{preconditioner_type}' + f'_iteration_{650}' + '.npy',
        #     filtered)

    except:
        print()

    #fig, ax = plt.subplots(1, 3, figsize=(10, 5), width_ratios=[0.03 ,1, 1])
    fig, ax = plt.subplots(1, 3, figsize=(10, 4.5), width_ratios=[0.03, 1, 1],
                           constrained_layout=True)

    ax[1].set_aspect('equal')
#    ax[0].set_title(f'Square grid - L-BFGS iteration {iter}')
    ax[1].set_title(f'Square grid')

    # ax[0].contourf( np.tile(phase_field_it_F, (1, 1)), cmap=mpl.cm.Greys)
    pcm1 = ax[1].pcolormesh(x_ref[0], x_ref[1], np.tile(phase_field_it_F, (nb_tiles, nb_tiles)),
                           shading='flat',
                           edgecolors='none',
                           lw=0.01,
                           cmap=mpl.cm.Greys,
                            rasterized=True)
    ax[1].text(-0.08, 1.05, r'$\textbf{(a)}$', transform=ax[1].transAxes)

    it_hex = np.copy(iter)

    ax[2].set_aspect('equal')
    #ax[1].set_title(f'Hexagonal grid - L-BFGS iteration {iter_hex}')
    ax[2].set_title(f'Hexagonal grid')

    ax[2].text(-0.08, 1.05, r'$\textbf{(b)}$', transform=ax[2].transAxes)
    pcm2 = ax[2].pcolormesh(x_coords[0]-2, x_coords[1], np.tile(phase_field_it_F_hex, (nb_tiles, nb_tiles)),
                           shading='flat',
                           edgecolors='none',
                           lw=0.01,
                           cmap=mpl.cm.Greys,
                            rasterized=True)

    pcm2.set_edgecolor('face')
    ax[0].axis('off')

    cax = fig.add_axes([0.0, 0.1, 0.015, 0.8])# [left, bottom, width, height]
    norm = mpl.colors.Normalize(vmin=0, vmax=1.0)
    cmap = mpl.cm.Greys

    cbar = fig.colorbar(
        mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
        cax=cax,
        ticks=np.arange(0, 1.2, 0.25)
    )

    cax.set_ylabel(r'Phase $\rho$', rotation=90, labelpad=-65)
    cax.yaxis.set_ticks_position('left')



    # Apply identical limits
    for a in ax[1:3]:
        #a.set_xlim(xmin, xmax)
        #a.set_ylim(ymin, ymax)
        a.set_xticks([0,1,2,3])
        a.set_yticks([0,1,2,3])
        a.set_xlabel('Unit cell size  -  L')
        a.set_ylabel('L')

        a.set_xlim(0, 3)
        a.set_ylim(0, 3)

        a.set_aspect('equal')
    #fig.tight_layout()
    fname = f'square_hexa_spec_{iter}' + '{}'.format('.pdf')

    plt.savefig(figure_folder_path + script_name + fname, bbox_inches='tight', dpi=400)
    print(('create figure: {}'.format(figure_folder_path + script_name_load +'/' +  fname)))
    #plt.show()


cg_tol = 8
# Ns = [16, 32, 64, 128, 256, 512, 1024]
Ns = [1024, ]  # 512, 1024
nb_tiles = 3
steps = np.arange(1000,2000, 100)
for j in np.arange(len(Ns)):
    N = Ns[j]
    # Reference coordinates

    x_ref = np.zeros([2, nb_tiles * (N) + 1, nb_tiles * (N) + 1])
    # for index position
    #x_ref[0], x_ref[1] = np.meshgrid(np.arange(0, nb_tiles * (N) + 1), np.arange(0, nb_tiles * (N) + 1), indexing='ij')
    # shift = 0.5 * np.arange(nb_tiles * (N) + 1)

    # for domain size
    x_ref[0], x_ref[1] = np.meshgrid(np.linspace(0, nb_tiles,nb_tiles * (N) + 1), np.linspace(0,nb_tiles, nb_tiles * (N) + 1), indexing='ij')
    shift = 0.5 *  np.linspace(0, nb_tiles, nb_tiles * (N) + 1)
    x_coords = np.copy(x_ref)
    # Apply shift to each row
    x_coords[0] += shift[None, :]
    x_coords[1] *= np.sqrt(3) / 2

    # fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    for i in np.arange(len(steps) - 1):
        iter = steps[i]
        preconditioner_type = 'Green_Jacobi'  #
        random = False

        fig, ax = plt.subplots(1, 2, figsize=(10, 5), width_ratios=[1, 1.5])
        ax[0].set_aspect('equal')
        ax[0].set_title(f'Square grid {iter}')
        # ax[0].contourf( np.tile(phase_field_it_F, (1, 1)), cmap=mpl.cm.Greys)

        try:
            script_name_load = f'exp_paper_JG_2D_elasticity_TO_load_init_random_{random}' + f'_N_{N}' + f'_cgtol_{cg_tol}' + f'_soft_{soft_phase_exponent}' + '/'
            # script_name = f'exp_paper_JG_2D_elasticity_TO_load_init_random_{random}' + f'_N_{N}'+ f'_cgtol_{cg_tol}'+'/'

            phase_field_it_F = np.load(
                './exp_data/' + script_name_load + f'{preconditioner_type}' + f'_iteration_{iter}' + '.npy',
                allow_pickle=True)

        except:
            print()

        else:
            pcm = ax[0].pcolormesh(x_ref[0], x_ref[1], np.tile(phase_field_it_F, (3, 3)),
                                   shading='flat',
                                   edgecolors='none',
                                   lw=0.01,
                                   cmap=mpl.cm.Greys,
                            rasterized=True)
        # ax[0].colorbar(pcm, ax=ax[0])



        try:

            script_name_load = f'exp_paper_JG_2D_elasticity_TO_load_init_hexa_random_{random}' + f'_N_{N}' + f'_cgtol_{cg_tol}' + f'_soft_{soft_phase_exponent}' + '/'
            # script_name = f'exp_paper_JG_2D_elasticity_TO_load_init_random_{random}' + f'_N_{N}'+ f'_cgtol_{cg_tol}'+'/'

            phase_field_it_F_hex = np.load(
                './exp_data/' + script_name_load + f'{preconditioner_type}' + f'_iteration_{iter}' + '.npy',
                allow_pickle=True)
        except:
            print()
        else:
            it_hex = np.copy(iter)

        # ax[0].contourf( np.tile(phase_field_it_F, (1, 1)), cmap=mpl.cm.Greys)
            pcm = ax[1].pcolormesh(x_coords[0], x_coords[1], np.tile(phase_field_it_F_hex, (3, 3)),
                                   shading='flat',
                                   edgecolors='none',
                                   lw=0.01,
                                   cmap=mpl.cm.Greys,
                            rasterized=True)
        # ax[1].colorbar(pcm, ax=ax[1])
        ax[1].set_aspect('equal')
        ax[1].set_title(f'Hexagonal grid {iter}')
        # ax[1].set_ylim([0,32])
        # ax[1].set_xlim([0,50])
        # Determine global limits
        xmin = 0
        ymin = 0

        # Square grid extents
        xmax_sq = x_ref[0].max()
        ymax_sq = x_ref[1].max()

        # Hex grid extents
        xmax_hex = x_coords[0].max()
        ymax_hex = x_coords[1].max()

        # Use the larger extents so both plots match
        xmax = max(xmax_sq, xmax_hex)
        ymax = max(ymax_sq, ymax_hex)

        # Apply identical limits
        for a in ax:
           # a.set_xlim(xmin, xmax)
            a.set_ylim(ymin, ymax)
            a.set_aspect('equal')
        fig.tight_layout()
        fname = f'square_hexa_{iter}' + '{}'.format('.png')
        #plt.savefig(figure_folder_path + script_name_load+'/' + fname, bbox_inches='tight')
        #print(('create figure: {}'.format(figure_folder_path + script_name_load +'/' +  fname)))
        plt.show()






quit()
cg_tol = 7
# Ns = [16, 32, 64, 128, 256, 512, 1024]
Ns = [16, 32, 64, 128, 256, 512, 1024]  # ,128,256,  512, 1024
# Professional styling
plt.rcParams['text.usetex'] = True

plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 11,
    'axes.labelsize': 11,
    'legend.fontsize': 11,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'axes.linewidth': 0.8,
})

# Create figure and gridspec
# fig = plt.figure(figsize=(7,4))
# gs = fig.add_gridspec(2, 1,hspace=0.3, wspace=0.25, width_ratios=[1 ],
#                               height_ratios=[1, 3])
fig = plt.figure(figsize=(8.5, 3.2))
gs = fig.add_gridspec(1, 2, height_ratios=[1], hspace=0.1)

# Add subplot in the second column
ax_void = fig.add_subplot(gs[0, 0])
ax = fig.add_subplot(gs[0, 1])

# ax_legend = fig.add_subplot(gs[0, 1])
# ax_legend_void = fig.add_subplot(gs[0, 0])
# Store line handles for legend
lines_void = []
labels_void = []
lines = []
labels = []
# ax_legend = fig.add_subplot(gs[0, 1])
colors = ['red', 'blue', 'green', 'orange', 'purple']
linestyles = ['-', '--', '-.', ':', (0, (3, 1, 1, 1))]
markers = [
    "v",  # point
    ">",  # pixel
    "o",  # circle
    "|",  # triangle down
    "^",  # triangle up
    "<",  # triangle left
    ">",  # triangle right
]

# fig, ax = plt.subplots(1, 1, figsize=(5, 5))
soft_phase_exponents = [5]  # 3,
for e in np.arange(len(soft_phase_exponents)):
    soft_phase_exponent = soft_phase_exponents[e]
    nbit_per_lbfgs_mech_G = []
    nbit_per_lbfgs_adjoint_G = []
    nbit_per_lbfgs_mech_GJ = []
    nbit_per_lbfgs_adjoint_GJ = []
    nb_it_G_mech_Ns = []
    nb_it_G_adjoint_Ns = []
    nb_it_GJ_mech_Ns = []
    nb_it_GJ_adjoint_Ns = []

    iterations = np.arange(0, 15000)
    # , 64, 128
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
                script_name = f'exp_paper_JG_2D_elasticity_TO_load_init_random_{random}' + f'_N_{N}' + f'_cgtol_{cg_tol}' + f'_soft_{soft_phase_exponent}' + '/'
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
                script_name = f'exp_paper_JG_2D_elasticity_TO_load_init_random_{random}' + f'_N_{N}' + f'_cgtol_{cg_tol}' + f'_soft_{soft_phase_exponent}' + '/'
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

        # ax.plot(np.asarray(nb_it_GJ_mech_), "y",
        #         label=r'Green-Jacobi - $\nabla \sigma$' + f'N={N}',
        #         linewidth=2)  # np.linspace(1, 10, np.asarray( nb_it_GJ_mech_).shape[0]),
        # # ax.plot(np.linspace(1, max_it, nb_it_G_adjoint.shape[0]), nb_it_G_adjoint, "g--", label=r'Green - Adjoint'+f'N={N}',
        # #         linewidth=2)
        # ax.plot(np.asarray(nb_it_GJ_adjoint_), "r--",
        #         label='Green-Jacobi - Adjoint' + f'N={N}',
        #         linewidth=2)  # np.linspace(1, 10, np.asarray(nb_it_GJ_adjoint_).shape[0]),

    # plt.show()

    # ax.loglog(np.array(Ns) ** 2, np.array(Ns) /1, "k-",
    #         label=r'Scaling',
    #         linewidth=1)
    Ns_squared = np.array(Ns[:]) ** 2
    l1, = ax.plot(np.array(Ns[:]) ** 2, nbit_per_lbfgs_adjoint_G[:], "g", linestyle='-.', marker='^',
                  label=fr'Adjoint problem - Green -$\kappa^{{soft}}  =10^{{-{soft_phase_exponent}}}$ ',
                  linewidth=2)
    lines.append(l1)
    labels.append(fr'Adjoint – Green – $\kappa^{{\mathrm{{soft}}}} = 10^{{-{soft_phase_exponent}}}$')
    ax.annotate(text=fr'FFT - Adjoint pr.',
                xy=(Ns_squared[3], nbit_per_lbfgs_adjoint_G[3]),
                xytext=(Ns_squared[3], 5e3),
                arrowprops=dict(arrowstyle='->',
                                color='Green',
                                lw=1,
                                ls='-'),
                color='Green'
                )

    l2, = ax.plot(np.array(Ns[:]) ** 2, nbit_per_lbfgs_mech_G[:], "g", linestyle='-.', marker='x',
                  label=f'Mechanical equilibrium - Green  - $\kappa^{{soft}}  =10^{{-{soft_phase_exponent}}}$ ',
                  linewidth=2)
    lines.append(l2)
    labels.append(fr'Mechanical eq. – Green – $\kappa^{{\mathrm{{soft}}}} = 10^{{-{soft_phase_exponent}}}$')
    ax.annotate(text=fr'FFT - Mechanical eq.',
                xy=(Ns_squared[3], nbit_per_lbfgs_mech_G[3]),
                xytext=(Ns_squared[2], 6e2),
                arrowprops=dict(arrowstyle='->',
                                color='Green',
                                lw=1,
                                ls='-'),
                color='Green'
                )

    l3, = ax.semilogx(np.array(Ns[:]) ** 2, nbit_per_lbfgs_adjoint_GJ[:], "k", linestyle=linestyles[e], marker='^',
                      label=fr'Adjoint problem - Green-Jacobi - $\kappa^{{soft}}  =10^{{-{soft_phase_exponent}}}$ ',
                      linewidth=2)
    lines.append(l3)
    labels.append(fr'Adjoint – Green-Jacobi – $\kappa^{{\mathrm{{soft}}}} = 10^{{-{soft_phase_exponent}}}$')
    ax.annotate(text=fr'J-FFT - Adjoint pr.',
                xy=(Ns_squared[3], nbit_per_lbfgs_adjoint_GJ[3]),
                xytext=(Ns_squared[3], 7e1),
                arrowprops=dict(arrowstyle='->',
                                color='k',
                                lw=1,
                                ls='-'),
                color='k'
                )

    l4, = ax.loglog(np.array(Ns[:]) ** 2, nbit_per_lbfgs_mech_GJ[:], "k", linestyle=linestyles[e], marker='x',
                    label=fr'Mechanical equilibrium - Green-Jacobi - $\kappa^{{soft}}  =10^{{-{soft_phase_exponent}}}$ ',
                    linewidth=2)
    lines.append(l4)
    labels.append(fr'Mechanical eq. – Green-Jacobi – $\kappa^{{\mathrm{{soft}}}} = 10^{{-{soft_phase_exponent}}}$')
    ax.annotate(text=fr'J-FFT - Mechanical eq.',
                xy=(Ns_squared[3], nbit_per_lbfgs_mech_GJ[3]),
                xytext=(Ns_squared[2], 2e1),
                arrowprops=dict(arrowstyle='->',
                                color='k',
                                lw=1,
                                ls='-'),
                color='k'
                )
# plt.title(f' CG tol =  $10^{{-{cg_tol}}}$')


nbit_per_lbfgs_mech_G = []
nbit_per_lbfgs_adjoint_G = []
nbit_per_lbfgs_mech_GJ = []
nbit_per_lbfgs_adjoint_GJ = []
nb_it_G_mech_Ns = []
nb_it_G_adjoint_Ns = []
nb_it_GJ_mech_Ns = []
nb_it_GJ_adjoint_Ns = []
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

# κ^soft = 0 (thick lines, x markers)

l5, = ax_void.plot(np.array(Ns[:3]) ** 2, nbit_per_lbfgs_adjoint_G[:3], "g-.", marker='^',
                   label=rf'Adjoint problem - Green - $\kappa^{{soft}}  =0$',
                   linewidth=2)
lines_void.append(l5)
labels_void.append(r'Adjoint – Green – $\kappa^{\mathrm{soft}} = 0$')
ax_void.annotate(text=fr'FFT - Adjoint pr.',
                 xy=(Ns_squared[1], nbit_per_lbfgs_adjoint_G[1]),
                 xytext=(Ns_squared[2], 5e3),
                 arrowprops=dict(arrowstyle='->',
                                 color='Green',
                                 lw=1,
                                 ls='-'),
                 color='Green'
                 )

l6, = ax_void.plot(np.array(Ns[:3]) ** 2, nbit_per_lbfgs_mech_G[:3], "g-.", marker='x',
                   label=rf'Mechanical equilibrium - Green - $\kappa^{{soft}}  =0$',
                   linewidth=2)
lines_void.append(l6)
labels_void.append(r'Mechanical eq. – Green – $\kappa^{\mathrm{soft}} = 0$')
ax_void.annotate(text=fr'FFT - Mechanical eq.',
                 xy=(Ns_squared[2], nbit_per_lbfgs_mech_G[2]),
                 xytext=(Ns_squared[2] + 2000, 2e3),
                 arrowprops=dict(arrowstyle='->',
                                 color='Green',
                                 lw=1,
                                 ls='-'),
                 color='Green'
                 )

l7, = ax_void.semilogx(np.array(Ns) ** 2, nbit_per_lbfgs_adjoint_GJ, "k", marker='^',
                       label=fr'Adjoint problem - Green-Jacobi- $\kappa^{{soft}}  =0$',
                       linewidth=2)
lines_void.append(l7)
labels_void.append(r'Adjoint – Green-Jacobi – $\kappa^{\mathrm{soft}} = 0$')
ax_void.annotate(text=fr'J-FFT - Adjoint pr.',
                 xy=(Ns_squared[4], nbit_per_lbfgs_adjoint_GJ[4]),
                 xytext=(Ns_squared[3], 170),
                 arrowprops=dict(arrowstyle='->',
                                 color='k',
                                 lw=1,
                                 ls='-'),
                 color='k'
                 )

l8, = ax_void.loglog(np.array(Ns) ** 2, nbit_per_lbfgs_mech_GJ, "k", marker='x',
                     label=fr'Mechanical equilibrium - Green-Jacobi  - $\kappa^{{soft}}  =0$',
                     linewidth=2)
lines_void.append(l8)
labels_void.append(r'Mechanical eq. – Green-Jacobi – $\kappa^{\mathrm{soft}} = 0$')
ax_void.annotate(text=fr'J-FFT - Mechanical eq.',
                 xy=(Ns_squared[3], nbit_per_lbfgs_mech_GJ[3]),
                 xytext=(Ns_squared[2], 2e1),
                 arrowprops=dict(arrowstyle='->',
                                 color='k',
                                 lw=1,
                                 ls='-'),
                 color='k'
                 )

# Configure main plot
ax.set_xlabel(r"Grid size")
ax.set_xscale('log')
ax.set_xticks(np.array(Ns) ** 2)
ax.set_xticklabels([f"${n}^2$" for n in Ns])
ax.set_xlim([16 ** 2, 1024 ** 2])
ax.set_ylabel(r"\# PCG iterations per L-BFGS step")
ax.set_ylim([10, 1e4])

# ax.grid(True, alpha=0.3, which='both')
ax.grid(True, axis='y', alpha=0.3)
ax.minorticks_off()
ax.set_title(fr'$\kappa^{{\mathrm{{void}}}} = 10^{{-{soft_phase_exponent}}}$')
ax.text(0.0, 1.05, rf'\textbf{{(b)}}', transform=ax.transAxes)

ax_void.set_xlabel(r"Grid size")
ax_void.set_xscale('log')
ax_void.set_xticks(np.array(Ns) ** 2)
ax_void.set_xticklabels([f"${n}^2$" for n in Ns])
ax_void.set_xlim([16 ** 2, 1024 ** 2])
ax_void.set_ylabel(r"\# PCG iterations per L-BFGS step")
ax_void.set_ylim([10, 1e4])

ax_void.grid(True, axis='y', alpha=0.3)
ax_void.minorticks_off()
ax_void.set_title(fr'$\kappa^{{\mathrm{{void}}}}=0$')
ax_void.text(0.0, 1.05, rf'\textbf{{(a)}}', transform=ax_void.transAxes)

# Configure legend panel (no axes, just legend)
# ax_legend_void.axis('off')
# ax_legend_void.legend(lines_void, labels_void, loc='center', ncol=1, frameon=True,
#                  fancybox=True, shadow=False, fontsize=8.5,
#                  columnspacing=1.5, handlelength=2.5)
#
# ax_legend.axis('off')
# ax_legend.legend(lines, labels, loc='center', ncol=1, frameon=True,
#                  fancybox=True, shadow=False, fontsize=8.5,
#                  columnspacing=1.5, handlelength=2.5)

# ax.set_xlabel(r"Grid size")
# ax.set_xticks(np.array(Ns) ** 2)
# ax.set_xticklabels([f"${n}^2$" for n in Ns])
# ax.set_xlim([16 ** 2 - 1, 1024 ** 2 + 1])
#
# ax.set_ylabel(r"$\#$ PCG iterations per L-BFGS step ")
# ax.set_ylim([10, 1e4])
# ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))
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
