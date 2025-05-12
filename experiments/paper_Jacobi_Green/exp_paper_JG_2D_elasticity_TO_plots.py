import matplotlib as mpl
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

# Enable LaTeX rendering
plt.rcParams.update({
    "text.usetex": True,  # Use LaTeX
    "font.family": "helvetica",  # Use a serif font
})
# Define the dimensions of the 2D array
rows = 25  # or whatever size you want
cols = 25  # or whatever size you want

# Create a random 2D array with 0 and 1
# The probabilities can be adjusted to get a different distribution of bubbles (0) and matrix (1)
array = np.random.choice([0, 1], size=(rows, cols), p=[0.5, 0.5])  # equal probability for 0 and 1
plot_figs = True
plot_movie = True
for ration in [0, ]:  # 0.2,0.1,0.0,-0.1,-0.2,-0.3,-0.4,-0.5,-0.6,-0.7,-0.8,-0.9 ]:
    poison_target = ration
    plt.clf()
    for w_mult in [5.0]:  # np.arange(0.1, 1., 0.1):# [1]:
        for eta_mult in [0.02]:  # np.arange(0.05, 0.5, 0.05):#[0.1 ]:
            energy_objective = False
            print(w_mult, eta_mult)
            pixel_size = 0.0078125
            eta = 0.01  # eta_mult * pixel_size
            N = 64
            cores = 6
            p = 2
            nb_load_cases = 3
            random_initial_geometry = True
            bounds = False
            optimizer = 'lbfg'  # adam2
            script_name = 'exp_paper_JG_2D_elasticity_TO'
            macro_multip = 1.0

            E_target = 0.15
            poison_target = -0.5
            poison_0 = 0.29
            # name = (            f'{optimizer}_muFFTTO_elasticity_{script_name}_N{N}_E_target_0.15_Poisson_-0.5_Poisson0_0.2_w{w_mult}_eta{eta_mult}_p{p}_bounds={bounds}_FE_NuMPI{cores}_nb_load_cases_{nb_load_cases}_energy_objective_{energy_objective}_random_{random_initial_geometry}')
            # DGO
            name = (
                f'{optimizer}_muFFTTO_elasticity_{script_name}_N{N}_E_target_{E_target:.2f}_Poisson_{poison_target:.2f}_Poisson0_{poison_0:.2f}_w{w_mult:.2f}_eta{eta_mult}_mac_{macro_multip}_p{p}_prec=Green_bounds={bounds}_FE_NuMPI{cores}_nb_load_cases_{nb_load_cases}_e_obj_{energy_objective}_random_{random_initial_geometry}')
            # eta_mult = 0.0102
            # combined
            name2 = (
                f'{optimizer}_muFFTTO_elasticity_{script_name}_N{N}_E_target_{E_target:.2f}_Poisson_{poison_target:.2f}_Poisson0_{poison_0:.2f}_w{w_mult:.2f}_eta{eta_mult}_mac_{macro_multip}_p{p}_prec=Jacobi_Green_bounds={bounds}_FE_NuMPI{cores}_nb_load_cases_{nb_load_cases}_e_obj_{energy_objective}_random_{random_initial_geometry}')
            # eta_mult = 0.0103
            # Jakobi
            name3 = (
                f'{optimizer}_muFFTTO_elasticity_{script_name}_N{N}_E_target_{E_target:.2f}_Poisson_{poison_target:.2f}_Poisson0_{poison_0:.2f}_w{w_mult:.2f}_eta{eta_mult}_mac_{macro_multip}_p{p}_prec=Jacobi_bounds={bounds}_FE_NuMPI{cores}_nb_load_cases_{nb_load_cases}_e_obj_{energy_objective}_random_{random_initial_geometry}')

        if plot_figs:
            phase_field_G = np.load('../exp_data/' + name + f'.npy', allow_pickle=True)
            phase_field_JG = np.load('../exp_data/' + name2 + f'.npy', allow_pickle=True)
            phase_field_J = np.load('../exp_data/' + name3 + f'.npy', allow_pickle=True)

            xopt = np.load('../exp_data/' + name + f'xopt_log.npz', allow_pickle=True)
            xopt2 = np.load('../exp_data/' + name2 + f'xopt_log.npz', allow_pickle=True)
            xopt3 = np.load('../exp_data/' + name3 + f'xopt_log.npz', allow_pickle=True)
            ## 32 grid
            N = 32
            cores = 4
            # DGO
            name_32 = (
                f'{optimizer}_muFFTTO_elasticity_{script_name}_N{N}_E_target_{E_target:.2f}_Poisson_{poison_target:.2f}_Poisson0_{poison_0:.2f}_w{w_mult:.2f}_eta{eta_mult}_mac_{macro_multip}_p{p}_prec=Green_bounds={bounds}_FE_NuMPI{cores}_nb_load_cases_{nb_load_cases}_e_obj_{energy_objective}_random_{random_initial_geometry}')
            # combined
            name2_32 = (
                f'{optimizer}_muFFTTO_elasticity_{script_name}_N{N}_E_target_{E_target:.2f}_Poisson_{poison_target:.2f}_Poisson0_{poison_0:.2f}_w{w_mult:.2f}_eta{eta_mult}_mac_{macro_multip}_p{p}_prec=Jacobi_Green_bounds={bounds}_FE_NuMPI{cores}_nb_load_cases_{nb_load_cases}_e_obj_{energy_objective}_random_{random_initial_geometry}')
            # Jakobi
            name3_32 = (
                f'{optimizer}_muFFTTO_elasticity_{script_name}_N{N}_E_target_{E_target:.2f}_Poisson_{poison_target:.2f}_Poisson0_{poison_0:.2f}_w{w_mult:.2f}_eta{eta_mult}_mac_{macro_multip}_p{p}_prec=Jacobi_bounds={bounds}_FE_NuMPI{cores}_nb_load_cases_{nb_load_cases}_e_obj_{energy_objective}_random_{random_initial_geometry}')

            phase_field_G_32 = np.load('../exp_data/' + name_32 + f'.npy', allow_pickle=True)
            phase_field_JG_32 = np.load('../exp_data/' + name2_32 + f'.npy', allow_pickle=True)
            phase_field_J_32 = np.load('../exp_data/' + name3_32 + f'.npy', allow_pickle=True)

            xopt_32 = np.load('../exp_data/' + name_32 + f'xopt_log.npz', allow_pickle=True)
            xopt2_32 = np.load('../exp_data/' + name2_32 + f'xopt_log.npz', allow_pickle=True)
            xopt3_32 = np.load('../exp_data/' + name3_32 + f'xopt_log.npz', allow_pickle=True)

            ## 32 grid
            N = 128
            cores = 10
            eta_mult = 0.01
            random_initial_geometry = False
            # Green
            name_128 = (
                f'{optimizer}_muFFTTO_elasticity_{script_name}_N{N}_E_target_{E_target:.2f}_Poisson_{poison_target:.2f}_Poisson0_{poison_0:.2f}_w{w_mult:.2f}_eta{eta_mult}_mac_{macro_multip}_p{p}_prec=Green_bounds={bounds}_FE_NuMPI{cores}_nb_load_cases_{nb_load_cases}_e_obj_{energy_objective}_random_{random_initial_geometry}')
            xopt_128 = np.load('../exp_data/' + name_128 + f'xopt_log.npz', allow_pickle=True)
            # combined
            name2_128 = (
                f'{optimizer}_muFFTTO_elasticity_{script_name}_N{N}_E_target_{E_target:.2f}_Poisson_{poison_target:.2f}_Poisson0_{poison_0:.2f}_w{w_mult:.2f}_eta{eta_mult}_mac_{macro_multip}_p{p}_prec=Jacobi_Green_bounds={bounds}_FE_NuMPI{cores}_nb_load_cases_{nb_load_cases}_e_obj_{energy_objective}_random_{random_initial_geometry}')
            xopt2_128 = np.load('../exp_data/' + name2_128 + f'xopt_log.npz', allow_pickle=True)

            N = 64
            cores = 6
            eta_mult = 0.02
            src = '../figures/'  # source folder\
            fig_data_name = f'muFFTTO_{name}'  # print('rank' f'{MPI.COMM_WORLD.rank:6} ')

            # dgo = (xopt.f.num_iteration_.transpose()[::3] +
            #        xopt.f.num_iteration_.transpose()[1::3] +
            #        xopt.f.num_iteration_.transpose()[2::3]) / 3
            # jacoby = (xopt3.f.num_iteration_.transpose()[::3] +
            #           xopt3.f.num_iteration_.transpose()[1::3] +
            #           xopt3.f.num_iteration_.transpose()[2::3]) / 3
            # combi = (xopt2.f.num_iteration_.transpose()[::3] +
            #          xopt2.f.num_iteration_.transpose()[1::3] +
            #          xopt2.f.num_iteration_.transpose()[2::3]) / 3
            fig = plt.figure()
            x_points = np.arange(-0.3, 1.3, 0.01)
            dw = x_points ** 2 * (1 - x_points) ** 2

            plt.plot(x_points, dw)
            plt.xlabel(r'$\rho$')
            plt.ylabel(r'$f_{DW}(\rho)$')

            plt.show()
            plt.figure()
            fig_data_name = f'muFFTTO_nb_it_{phase_field_G.shape}_line'  # print('rank' f'{MPI.COMM_WORLD.rank:6} ')
            # plt.semilogy(xopt.f.norms_f-xopt.f.norms_f[-1], label='objective f')
            # plt.semilogy(xopt.f.norms_pf-xopt.f.norms_pf[-1], label='phase field')
            # plt.semilogy(np.abs(xopt.f.norms_sigma[:,0]-xopt.f.norms_pf - xopt.f.norms_sigma[-1,0]+xopt.f.norms_pf[-1]), label='stress')
            # plt.plot([305,321,458,724,831,1477,1454,2180,2457,3919,5880,7047,6154,5879,6405,7309,7241,7614,7566,10000,10000,9962], label='No preconditioner',linewidth=2)

            plt.plot(xopt.f.num_iteration_.transpose()[::], label='Green', linewidth=2)
            plt.plot(xopt2.f.num_iteration_.transpose()[::], label='Green + Jacobi', linewidth=2)

            plt.plot(xopt3.f.num_iteration_.transpose()[::], label='Jacobi', linewidth=2)

            # plt.semilogy(xopt.f.norms_pf, label='phase field')
            # plt.semilogy(xopt.f.norms_sigma[:, 0] - xopt.f.norms_pf,    label='stress')
            # -xopt.f.norms_sigma[-1,0]+xopt.f.norms_pf[-1]
            # nodal_coordinates[0, 0] * number_of_pixels[0], nodal_coordinates[1, 0] * number_of_pixels[0],
            # plt.grid(True)
            # plt.title('Number of CG iteration per L-BFGS step {}'.format(optimizer))
            plt.xlabel(r" L-BFGS step (iteration) ")
            plt.ylabel(r'number of CG iteration')

            # plt.minorticks_on()
            fname = src + fig_data_name + '{}'.format('.png')
            print(('create figure: {}'.format(fname)))
            plt.legend()
            plt.show()

            plt.figure()
            fig_data_name = f'muFFTTO_{phase_field_G.shape}_line relative'  # print('rank' f'{MPI.COMM_WORLD.rank:6} ')
            # plt.semilogy(xopt.f.norms_f-xopt.f.norms_f[-1], label='objective f')
            # plt.semilogy(xopt.f.norms_pf-xopt.f.norms_pf[-1], label='phase field')
            # plt.semilogy(np.abs(xopt.f.norms_sigma[:,0]-xopt.f.norms_pf - xopt.f.norms_sigma[-1,0]+xopt.f.norms_pf[-1]), label='stress')
            plt.semilogy(xopt.f.norms_f - xopt.f.norms_f[-1], label='objective f')
            # plt.semilogy(np.abs(xopt.f.norms_pf - xopt.f.norms_pf[-1]), label='phase field')
            plt.title('optimizer {}'.format(optimizer))

            plt.grid(True)
            # plt.minorticks_on()
            fname = src + fig_data_name + '{}'.format('.png')
            print(('create figure: {}'.format(fname)))
            plt.legend()
            plt.show()
            plt.figure()
            fig_data_name = f'muFFTTO_{phase_field_G.shape}_line'  # print('rank' f'{MPI.COMM_WORLD.rank:6} ')
            plt.semilogy(xopt.f.norms_f, label='objective f')

            plt.semilogy(xopt.f.norms_delta_f, label='$\delta$ f')
            plt.semilogy(xopt.f.norms_max_grad_f, label='max $\delta$ f')

            plt.semilogy(xopt.f.norms_norm_grad_f, label='|$\delta$ f|')
            # plt.semilogy(xopt.f.norms_max_delta_x, label='max Δx')
            # plt.semilogy(xopt.f.norms_norm_delta_x, label='|Δx|')
            # plt.semilogy(np.abs(xopt.f.norms_sigma[:,0]-xopt.f.norms_pf - xopt.f.norms_sigma[-1,0]+xopt.f.norms_pf[-1]), label='stress')
            plt.title('optimizer {}'.format(optimizer))

            # nodal_coordinates[0, 0] * number_of_pixels[0], nodal_coordinates[1, 0] * number_of_pixels[0],
            plt.grid(True)
            # plt.minorticks_on()
            fname = src + fig_data_name + '{}'.format('.png')
            print(('create figure: {}'.format(fname)))
            plt.legend()
            plt.show()

            fig = plt.figure()
            gs = fig.add_gridspec(2, 2)
            ax1 = fig.add_subplot(gs[0, 0])
            ax2 = fig.add_subplot(gs[0, 1])
            ax3 = fig.add_subplot(gs[1, :])
            if nb_load_cases == 1:
                dgo = xopt.f.num_iteration_.transpose()[::]
                jacoby = xopt3.f.num_iteration_.transpose()[::]
                combi = xopt2.f.num_iteration_.transpose()[::]
            elif nb_load_cases == 3:
                dgo = (xopt.f.num_iteration_.transpose()[::3] +
                       xopt.f.num_iteration_.transpose()[1::3] +
                       xopt.f.num_iteration_.transpose()[2::3]) / 3
                jacoby = (xopt3.f.num_iteration_.transpose()[::3] +
                          xopt3.f.num_iteration_.transpose()[1::3] +
                          xopt3.f.num_iteration_.transpose()[2::3]) / 3
                combi = (xopt2.f.num_iteration_.transpose()[::3] +
                         xopt2.f.num_iteration_.transpose()[1::3] +
                         xopt2.f.num_iteration_.transpose()[2::3]) / 3
            dgo_shape = np.copy(dgo.shape[0])
            jacobi_shape = np.copy(jacoby.shape[0])

            combi_shape = np.copy(combi.shape[0])

            # Compute padding sizes
            pad_width = dgo.shape[0] - jacoby.shape[0]
            # Find the longest array
            longest_array = max([dgo, jacoby, combi], key=len)
            # Pad A to match B's shape
            dgo = np.pad(dgo, (0, longest_array.shape[0] - dgo.shape[0]), mode='constant', constant_values=0)
            jacoby = np.pad(jacoby, (0, longest_array.shape[0] - jacoby.shape[0]), mode='constant', constant_values=0)
            combi = np.pad(combi, (0, longest_array.shape[0] - combi.shape[0]), mode='constant', constant_values=0)

            xopt_it = np.load('../exp_data/' + name + f'_it{30}.npy', allow_pickle=True)

            ax1.clear()
            ax1.imshow(np.tile(xopt_it, (1, 1)), cmap=mpl.cm.Greys, vmin=0, vmax=1)

            ax1.set_title(r'Density at iteration {}'.format(90), wrap=True)

            ax2.clear()
            ax2.plot(xopt_it[xopt_it.shape[0] // 2, :], linewidth=1)
            ax2.set_ylim([-0.1, 1.1])
            ax2.set_title(f'Cross section y=15')

            ax3.plot(dgo, "g", label='Green', linewidth=1)
            # axs[1].plot(xopt2.f.num_iteration_.transpose()[1:3*i+1:3],"r", label='DGO ',linewidth=1)
            # axs[1].plot(xopt2.f.num_iteration_.transpose()[2:3*i+2:3],"r", label='DGO ',linewidth=1)

            ax3.plot(jacoby, "k", label='Jacobi', linewidth=1)
            ax3.plot(combi, "b", label='Green + Jacobi', linewidth=1)
            # axs[1].legend()
            ax3.set_xlabel(" L-BFGS step (iteration) ")
            ax3.set_ylabel("  PCG iterations")
            plt.legend(['Green', 'Jacobi', 'Green + Jacobi'])
            plt.show()

    if nb_load_cases == 1:
        dgo = xopt.f.num_iteration_.transpose()[::]
        jacoby = xopt3.f.num_iteration_.transpose()[::]
        combi = xopt2.f.num_iteration_.transpose()[::]

    elif nb_load_cases == 3:
        dgo = (xopt.f.num_iteration_.transpose()[::3] +
               xopt.f.num_iteration_.transpose()[1::3] +
               xopt.f.num_iteration_.transpose()[2::3]) / 3
        jacoby = (xopt3.f.num_iteration_.transpose()[::3] +
                  xopt3.f.num_iteration_.transpose()[1::3] +
                  xopt3.f.num_iteration_.transpose()[2::3]) / 3
        combi = (xopt2.f.num_iteration_.transpose()[::3] +
                 xopt2.f.num_iteration_.transpose()[1::3] +
                 xopt2.f.num_iteration_.transpose()[2::3]) / 3
        dgo_32 = (xopt_32.f.num_iteration_.transpose()[::3] +
                  xopt_32.f.num_iteration_.transpose()[1::3] +
                  xopt_32.f.num_iteration_.transpose()[2::3]) / 3
        jacoby_32 = (xopt3_32.f.num_iteration_.transpose()[::3] +
                     xopt3_32.f.num_iteration_.transpose()[1::3] +
                     xopt3_32.f.num_iteration_.transpose()[2::3]) / 3
        combi_32 = (xopt2_32.f.num_iteration_.transpose()[::3] +
                    xopt2_32.f.num_iteration_.transpose()[1::3] +
                    xopt2_32.f.num_iteration_.transpose()[2::3]) / 3
        dgo_128 = (xopt_128.f.num_iteration_.transpose()[::3] +
                     xopt_128.f.num_iteration_.transpose()[1::3] +
                     xopt_128.f.num_iteration_.transpose()[2::3]) / 3
        combi_128 = (xopt2_128.f.num_iteration_.transpose()[::3] +
                     xopt2_128.f.num_iteration_.transpose()[1::3] +
                     xopt2_128.f.num_iteration_.transpose()[2::3]) / 3
        # dgo = xopt.f.num_iteration_.transpose()[::3]
        # jacoby = xopt3.f.num_iteration_.transpose()[::3]
        # combi = xopt2.f.num_iteration_.transpose()[::3]

    nb_tiles = 1
    fig = plt.figure(figsize=(11, 6.5))
    gs = fig.add_gridspec(3, 4, width_ratios=[3, 3, 3, 0.2])
    ax_iterations = fig.add_subplot(gs[1:, :])

    ax_iterations.plot(np.linspace(1, 1000, dgo.shape[0]), dgo, "g", label='Green N=64', linewidth=1)
    ax_iterations.plot(np.linspace(1, 1000, jacoby.shape[0]), jacoby, "b", label='Jacobi N=64', linewidth=1)
    ax_iterations.plot(np.linspace(1, 1000, combi.shape[0]), combi, "k", label='Jacobi - Green N=64', linewidth=2)
    ax_iterations.plot(np.linspace(1, 1000, dgo_32.shape[0]), dgo_32, "g", label='Green N=32', linewidth=1,
                       linestyle=':')
    ax_iterations.plot(np.linspace(1, 1000, dgo_128.shape[0]),  dgo_128, "g",
                       label='Green N=128', linewidth=1, linestyle='-.')
    ax_iterations.plot(np.linspace(1, 1000, jacoby_32.shape[0]), jacoby_32, "b", label='Jacobi N=32', linewidth=1,
                       linestyle=':')
    ax_iterations.plot(np.linspace(1, 1000, combi_32.shape[0]), combi_32, "k", label='Jacobi - Green N=32', linewidth=2,
                       linestyle=':')
    ax_iterations.plot(np.linspace(1, 1000, combi_128.shape[0]), combi_128, "k", label='Jacobi - Green N=128',
                       linewidth=2, linestyle='-.')
    ax_iterations.plot(np.linspace(1, 1000, combi_32.shape[0]), np.ones(combi_32.shape[0]) * 1228, "k",
                       label='Jacobi - Green N=1024', linewidth=2, linestyle='--')

    ax_iterations.set_xlim(1, 1000)
    ax_iterations.set_xticks([1, 1000])
    ax_iterations.set_xticklabels([f'Start', f'Converged'])
    ax_iterations.set_ylim([1, 2600])
    ax_iterations.set_yscale('linear')

    ax_iterations.set_yscale('log')
    ax_iterations.set_ylim([10, 1e4])
    ax_iterations.set_xlabel(" L-BFGS optimization process")
    ax_iterations.set_ylabel(r"$\#$ PCG iterations")
    # ax_iterations.legend(loc='upper right')  # ['Green', 'Green + Jacobi', 'Jacobi'],
    # Adding ticks to the right side
    ax_iterations.tick_params(labelbottom=True, labeltop=False, labelleft=True, labelright=True,
                              bottom=True, top=False, left=True, right=True)

    ax_iterations.annotate(text=r'Jacobi-Green - $\mathcal{T}$' + f'$_{{{32}}}$',
                           xy=(300, 62.0),
                           xytext=(350., 30.6),
                           arrowprops=dict(arrowstyle='->',
                                           color='Black',
                                           lw=1,
                                           ls='-'),
                           color='Black'
                           )
    ax_iterations.annotate(text=r'Jacobi-Green - $\mathcal{T}$' + f'$_{{{64}}}$',
                           xy=(380, 88.0),
                           xytext=(400., 120.6),
                           arrowprops=dict(arrowstyle='->',
                                           color='Black',
                                           lw=1,
                                           ls='-'),
                           color='Black'
                           )
    ax_iterations.annotate(text=r'Jacobi-Green - $\mathcal{T}$' + f'$_{{{128}}}$',
                           xy=(700, 160.0),
                           xytext=(750., 100.6),
                           arrowprops=dict(arrowstyle='->',
                                           color='Black',
                                           lw=1,
                                           ls='-'),
                           color='Black'
                           )
    ax_iterations.annotate(text=r'Jacobi-Green - $\mathcal{T}$' + f'$_{{{1024}}}$',
                           xy=(900, 1160.0),
                           xytext=(830., 550.6),
                           arrowprops=dict(arrowstyle='->',
                                           color='Black',
                                           lw=1,
                                           ls='-'),
                           color='Black'
                           )
    ax_iterations.annotate(text=r'Jacobi - $\mathcal{T}$' + f'$_{{{32}}}$',
                           xy=(500, 210.0),
                           xytext=(550., 290.0),
                           arrowprops=dict(arrowstyle='->',
                                           color='Blue',
                                           lw=1,
                                           ls='-'),
                           color='Blue'
                           )
    ax_iterations.annotate(text=r'Jacobi - $\mathcal{T}$' + f'$_{{{64}}}$',
                           xy=(500, 420.0),
                           xytext=(550., 550.6),
                           arrowprops=dict(arrowstyle='->',
                                           color='Blue',
                                           lw=1,
                                           ls='-'),
                           color='Blue'
                           )
    ax_iterations.annotate(text=r'Green - $\mathcal{T}$' + f'$_{{{32}}}$',
                           xy=(600, 1100.0),
                           xytext=(650., 700.0),
                           arrowprops=dict(arrowstyle='->',
                                           color='Green',
                                           lw=1,
                                           ls='-'),
                           color='Green'
                           )
    ax_iterations.annotate(text=r'Green - $\mathcal{T}$' + f'$_{{{64}}}$',
                           xy=(650, 2400.0),
                           xytext=(670., 1400.),
                           arrowprops=dict(arrowstyle='->',
                                           color='Green',
                                           lw=1,
                                           ls='-'),
                           color='Green'
                           )
    ax_iterations.annotate(text=r'Green - $\mathcal{T}$' + f'$_{{{128}}}$',
                           xy=(750, 3500.0),
                           xytext=(770., 4500.),
                           arrowprops=dict(arrowstyle='->',
                                           color='Green',
                                           lw=1,
                                           ls='-'),
                           color='Green'
                           )
    #####
    init = 1
    middle = 34
    end = 1200
    x = np.linspace(0, 1, N)
    extended_x = np.linspace(0, 1, N + 1)
    # divnorm = mpl.colors.LogNorm(vmin=1, vmax=100)
    # divnorm = mpl.colors.Normalize(vmin=0, vmax=100)
    # pcm = ax.pcolormesh(X, Y, nb_iterations, label='PCG: Green + Jacobi', cmap='Reds', norm=divnorm)
    #divnorm = mpl.colors.LogNorm(vmin=1e-4, vmax=1 )
    divnorm = mpl.colors.Normalize(vmin=1e-8, vmax=1)
    cmap_ = mpl.cm.seismic# mpl.cm.seismic #mpl.cm.Greys
    ax_init = fig.add_subplot(gs[0, 0])
    # ax_init= fig.add_axes([0.15, 0.6, 0.1, 0.2])
    xopt_init = np.load('../exp_data/' + name  + f'_it{init}.npy', allow_pickle=True)
    # xopt_init128 = np.load('../exp_data/' + name2_128 + f'_it{init}.npy', allow_pickle=True)
    pcm = ax_init.pcolormesh(np.tile(xopt_init**2, (nb_tiles, nb_tiles)),
                             cmap=cmap_, linewidth=0,
                             rasterized=True, norm=divnorm)
    ax_init.set_title(r'Initial ', wrap=True)
    ax_init.set_ylabel(r'Density $\rho$')

    # ax1 = fig.add_axes([0.13 , 0.45, 0.1, 0.15])
    # # ax1.set_aspect('equal')
    # letter_offset = 0
    # ax_iterations.text(letter_offset, 1.05, '(b)', transform=ax1.transAxes)
    # # ax1.semilogy(xopt_init[17, :] ** 2, "g", label=r'Green', linewidth=1)
    #
    # # ax1.semilogy(xopt_init[10, :] ** 2, "r",linestyle='--',  label=r'Green',linewidth=1)
    # # ax1.semilogy(xopt_init[20, :] ** 2, "r",linestyle='-.',  label=r'Green',linewidth=1)
    # # ax1.semilogy(xopt_init[30, :] ** 2, "r",linestyle=':',  label=r'Green',linewidth=1)
    # ax1.step(x, xopt_init[:, 12] ** 2, "k", linestyle='--', label=r'Green', linewidth=1)
    #
    # # ax1.axis('square')
    # # ax1.set_xlim(0, N - 1)
    # # ax1.set_ylim([1e-6, 5])
    # ax1.set_xlim(0, 1)
    # ax1.set_ylim([9e-7, 1.1])
    # ax1.set_yticklabels([])
    # ax1.set_xticklabels([])
    # ax1.xaxis.set_ticks_position('none')
    # ax1.yaxis.set_ticks_position('none')
    # ax1.set_aspect('equal', adjustable='box')

    ax_middle = fig.add_subplot(gs[0, 1])
    # ax_middle = fig.add_axes([0.5, 0.6, 0.1, 0.2])
    xopt_middle = np.load('../exp_data/' + name2 + f'_it{middle}.npy', allow_pickle=True)
    ax_middle.pcolormesh(np.tile(xopt_middle**2, (nb_tiles, nb_tiles)),
                         cmap=cmap_, norm=divnorm, linewidth=0,
                         rasterized=True)
    ax_middle.set_title(r'Intermediate', wrap=True)
    # ax1 = fig.add_axes([0.45, 0.45, 0.1, 0.15])
    # # ax1.set_aspect('equal')
    # letter_offset = 0
    # ax_iterations.text(letter_offset, 1.05, '(c)', transform=ax1.transAxes)
    # # ax1.semilogy(xopt_middle[17, :] ** 2, "g", label=r'Green', linewidth=1)
    # # ax1.semilogy(xopt_middle[10, :] ** 2, "r", linestyle='--', label=r'Green', linewidth=1)
    # # ax1.semilogy(xopt_middle[20, :] ** 2, "r", linestyle='-.', label=r'Green', linewidth=1)
    # # ax1.semilogy(xopt_middle[30, :] ** 2, "r", linestyle=':', label=r'Green', linewidth=1)
    # ax1.step(x, xopt_middle[:, 12] ** 2, "k", linestyle='--', label=r'Green', linewidth=1)
    #
    # # ax1.axis('square')
    # ax1.set_xlim(0, 1)
    # ax1.set_ylim([9e-7, 1.1])
    # # ax1.set_xlim(0, 1)
    # # ax1.set_ylim([1e-6, 5])
    # # ax1.set_yticklabels([])
    # ax1.set_xticklabels([])
    # ax1.xaxis.set_ticks_position('none')
    # ax1.yaxis.set_ticks_position('none')
    #
    ax_end = fig.add_subplot(gs[0, 2])
    # ax_end = fig.add_axes([0.7, 0.3, 0.1, 0.2])
    xopt_end = np.load('../exp_data/' + name2  + f'_it{end}.npy', allow_pickle=True)
    # xopt_init128t_end = np.load('../exp_data/' + name2_128 + f'_it{end}.npy', allow_pickle=True)

    pcm = ax_end.pcolormesh(np.tile(xopt_end**2, (nb_tiles, nb_tiles)),
                            cmap=cmap_, norm=divnorm, linewidth=0,
                            rasterized=True)
    ax_end.set_title(r'Converged', wrap=True)
    cbar_ax = fig.add_subplot(gs[0, 3])
    cbar = plt.colorbar(pcm, location='left', cax=cbar_ax)
    cbar.ax.yaxis.tick_right()
    # cbar.set_ticks(ticks=[1e-4,1e-2, 1])
    # cbar.set_ticklabels([f'$10^{{{-4}}}$', f'$10^{{{-2}}}$', 1])
    cbar.set_ticks(ticks=[1e-8, 0.5, 1])
    cbar.set_ticklabels([f'$10^{{{-8}}}$', 0.5, 1])
    #
    # ax1 = fig.add_axes([0.79, 0.20, 0.1, 0.15])
    # # ax1.set_aspect('equal')
    # letter_offset = 0
    # ax_iterations.text(letter_offset, 1.05, '(b)', transform=ax1.transAxes)
    # # extended_y = np.append(np.diag(phase_field), np.diag(phase_field)[-1])
    # extended_y = np.append(xopt_end[:, 17],
    #                        xopt_end[:, 17][-1])
    # # ax1.step(extended_x, extended_y17
    # # ax1.step(x,xopt_end[17, :] ** 2, "g",  label=r'Green',linewidth=1)
    # ax1.step(x, xopt_end[:, 12] ** 2, "k", linestyle='--', label=r'Green', linewidth=1)
    # # ax1.step(x,xopt_end[20, :] ** 2, "k", linestyle='-.', label=r'Green', linewidth=1)
    # # ax1.step(x,xopt_end[30, :] ** 2, "k", linestyle=':', label=r'Green', linewidth=1)
    # # ax1.axis('square')
    # ax1.set_xlim(0, 1)
    # ax1.set_ylim([0, 1.1])
    # # ax1.set_yticklabels([])
    # ax1.set_xticklabels([])
    # ax1.xaxis.set_ticks_position('none')
    # ax1.yaxis.set_ticks_position('none')

    ###    midlle images
    # ax1 = fig.add_axes([0.35, 0.23, 0.1, 0.15] )
    # xopt_end = np.load('../exp_data/' + name + f'_it{31}.npy', allow_pickle=True)
    #
    # letter_offset = 0
    # ax_iterations.text(letter_offset, 1.05, '(31)', transform=ax1.transAxes)
    # ax1.step(x,xopt_end[ :,12] ** 2, "k", linestyle='--', label=r'Green', linewidth=1)
    # ax1.set_xlim(0, 1)
    # ax1.set_ylim([9e-7, 1.1])
    # ax1.set_xticklabels([])
    # ax1.xaxis.set_ticks_position('none')
    # ax1.yaxis.set_ticks_position('none')
    # ####
    # ax1 = fig.add_axes([0.55, 0.23, 0.1, 0.15])
    # xopt_end = np.load('../exp_data/' + name + f'_it{170}.npy', allow_pickle=True)
    #
    # letter_offset = 0
    # ax_iterations.text(letter_offset, 1.05, '(170)', transform=ax1.transAxes)
    # ax1.step(x, xopt_end[:, 12] ** 2, "k", linestyle='--', label=r'Green', linewidth=1)
    # ax1.set_xlim(0, 1)
    # ax1.set_ylim([9e-7, 1.1])
    # ax1.set_xticklabels([])
    # ax1.xaxis.set_ticks_position('none')
    # ax1.yaxis.set_ticks_position('none')
    # ax_iterations.semilogy(xopt_middle[17, :] ** 2, "b", label=r'Jacobi-Green', linewidth=1)
    # ax_iterations.semilogy(xopt_end[17, :] ** 2, "k", label=r'Jacobi', linewidth=1)
    fname = src + 'exp_paper_JG_2D_elasticity_TO_iterations' + '{}'.format('.pdf')
    print(('create figure: {}'.format(fname)))
    plt.savefig(fname, bbox_inches='tight')
    plt.show()
    plot_movie = False
    if plot_movie:
        for nb_tiles in [1]:
            # fig = plt.figure()
            # fig, axs = plt.subplots(nrows=2, ncols=1,
            #                         figsize=(6, 6)  )
            fig = plt.figure()
            gs = fig.add_gridspec(2, 3)
            ax_G = fig.add_subplot(gs[0, 0])
            ax_JG = fig.add_subplot(gs[0, 1])
            ax_J = fig.add_subplot(gs[0, 2])

            ax2 = fig.add_subplot(gs[1, 2])
            ax3 = fig.add_subplot(gs[1, :2])
            # axs[0] = plt.axes(xlim=(0, nb_tiles * N), ylim=(0, nb_tiles * N))
            ax_G.imshow(np.tile(phase_field_G, (nb_tiles, nb_tiles)), cmap=mpl.cm.Greys, vmin=0, vmax=1)
            # ax_G.set_title(r'Density $\rho$', wrap=True)
            ax_G.set_title(r'Green', wrap=True)
            #
            ax_JG.imshow(np.tile(phase_field_JG, (nb_tiles, nb_tiles)), cmap=mpl.cm.Greys, vmin=0, vmax=1)
            # ax_G.set_title(r'Density $\rho$', wrap=True)
            ax_JG.set_title(r'Jacobi-Green', wrap=True)
            #
            ax_G.imshow(np.tile(phase_field_JG, (nb_tiles, nb_tiles)), cmap=mpl.cm.Greys, vmin=0, vmax=1)
            # ax_G.set_title(r'Density $\rho$', wrap=True)
            ax_G.set_title(r'Jacobi', wrap=True)
            #
            ax2.semilogy(phase_field_G[:, phase_field_G.shape[0] // 2], linewidth=0)
            ax2.set_title(f'Cross section')
            ax2.tick_params(right=True, top=False, labelright=True, labeltop=False, labelrotation=0)
            ax2.yaxis.set_label_position('right')

            ax3.plot(xopt.f.num_iteration_.transpose()[::], 'w', linewidth=0)
            # axs[1].plot(xopt3.f.num_iteration_.transpose(), "b", label='Jacoby', linewidth=0)
            # axs[1].plot(xopt.f.num_iteration_.transpose(), "k", label='DGO + Jacoby', linewidth=0)
            # legend = plt.legend()
            # Animation function to update the image
            ax3.set_xlabel(" L-BFGS step (iteration) ")
            ax3.set_ylabel("   PCG iterations")


            def update(i):
                if dgo_shape > (i + 10):
                    xopt_it_G = np.load('../exp_data/' + name + f'_it{i + 1}.npy', allow_pickle=True)
                else:
                    xopt_it_G = np.load('../exp_data/' + name + f'_it{5}.npy', allow_pickle=True)

                if combi_shape > (i + 10):
                    xopt_it_JG = np.load('../exp_data/' + name2 + f'_it{i + 1}.npy', allow_pickle=True)
                else:
                    xopt_it_JG = np.load('../exp_data/' + name2 + f'_it{5}.npy', allow_pickle=True)
                if jacobi_shape > (i + 10):
                    xopt_it_J = np.load('../exp_data/' + name3 + f'_it{i + 1}.npy', allow_pickle=True)
                else:
                    xopt_it_J = np.load('../exp_data/' + name3 + f'_it{5}.npy', allow_pickle=True)

                ax_G.clear()
                ax_G.imshow(np.tile(xopt_it_G, (nb_tiles, nb_tiles)), cmap=mpl.cm.Greys, vmin=0, vmax=1)
                ax_G.set_title(r'Green', wrap=True)
                ax_G.set_ylabel(r'Density $\rho$')

                ax_JG.clear()
                ax_JG.imshow(np.tile(xopt_it_JG, (nb_tiles, nb_tiles)), cmap=mpl.cm.Greys, vmin=0, vmax=1)
                ax_JG.set_title(r'Jacobi-Green', wrap=True)

                ax_J.clear()
                ax_J.imshow(np.tile(xopt_it_J, (nb_tiles, nb_tiles)), cmap=mpl.cm.Greys, vmin=0, vmax=1)
                ax_J.set_title(r'Jacobi', wrap=True)

                # ax_G.set_title('iteration {}, optimizer {}'.format(i, optimizer))

                ax2.clear()
                # ax2.semilogy(xopt_it_G[xopt_it_G.shape[0] // 2, :] ** 2, linewidth=1)
                ax2.semilogy(xopt_it_G[17, :] ** 2, "g", linewidth=1)
                ax2.semilogy(xopt_it_JG[17, :] ** 2, "b", linewidth=1)
                ax2.semilogy(xopt_it_J[17, :] ** 2, "k", linewidth=1)

                # ax2.semilogy(np.diag(np.fliplr(xopt_it_G)) ** 2, "g", linewidth=1)
                # ax2.semilogy(np.diag(np.fliplr(xopt_it_JG)) ** 2, "b", linewidth=1)
                # ax2.semilogy(np.diag(np.fliplr(xopt_it_J)) ** 2, "k", linewidth=1)
                # ax2.legend([  'Green', 'Jacobi-Green', 'Jacobi'])
                # ax2.set_yscale('symlog')
                ax2.tick_params(right=True, top=False, labelright=True, labeltop=False, labelrotation=0)
                ax2.yaxis.set_label_position('right')
                ax2.set_ylim([1e-6, 5])
                # ax2.set_ylim([0.1, 1.])

                ax2.set_title(f'Diagonal cross section ')

                ax3.plot(dgo[0:i + 1], "g", label='Green', linewidth=1)
                # axs[1].plot(xopt2.f.num_iteration_.transpose()[1:3*i+1:3],"r", label='DGO ',linewidth=1)
                # axs[1].plot(xopt2.f.num_iteration_.transpose()[2:3*i+2:3],"r", label='DGO ',linewidth=1)

                ax3.plot(combi[0:i + 1], "b", label='Green + Jacobi', linewidth=1)
                ax3.plot(jacoby[0:i + 1], "k", label='Jacobi', linewidth=1)
                # axs[1].legend()
                plt.legend(['', 'Green', 'Jacobi-Green', 'Jacobi'])
                # plt.legend(['', 'Green'])
                ax3.set_xlim([0, 350])
                # img.set_array(xopt_it)


            # Create animation
            # ani = FuncAnimation(fig, update, frames=xopt.f.norms_f.size - 1, blit=False)

            ani = FuncAnimation(fig, update, frames=330, blit=False)  # 316
            # axs[1].legend()
            # Save as a GIF
            ani.save(f"../figures/movie{nb_tiles}_exp2D_JG_single_load_iters_{name}.gif", writer=PillowWriter(fps=30))


    fig = plt.figure(figsize=(11, 6.5))
    gs = fig.add_gridspec(3, 4, width_ratios=[3, 3, 3, 0.2])
    ax_iterations = fig.add_subplot(gs[1:, :])

    ax_iterations.plot(np.linspace(1, 1000, dgo.shape[0]), dgo, "g", label='Green N=64', linewidth=1)
   # ax_iterations.plot(np.linspace(1, 1000, jacoby.shape[0]), jacoby, "b", label='Jacobi N=64', linewidth=1)
#    ax_iterations.plot(np.linspace(1, 1000, combi.shape[0]), combi, "k", label='Jacobi - Green N=64', linewidth=2)
    ax_iterations.plot(np.linspace(1, 1000, dgo_32.shape[0]), dgo_32, "g", label='Green N=32', linewidth=1,
                       linestyle=':')
    ax_iterations.plot(np.linspace(1, 1000, dgo_128.shape[0]),  dgo_128, "g",
                       label='Green N=128', linewidth=1, linestyle='-.')
 #   ax_iterations.plot(np.linspace(1, 1000, jacoby_32.shape[0]), jacoby_32, "b", label='Jacobi N=32', linewidth=1,
 #                      linestyle=':')
 #    ax_iterations.plot(np.linspace(1, 1000, combi_32.shape[0]), combi_32, "k", label='Jacobi - Green N=32', linewidth=2,
 #                       linestyle=':')
 #    ax_iterations.plot(np.linspace(1, 1000, combi_128.shape[0]), combi_128, "k", label='Jacobi - Green N=128',
 #                       linewidth=2, linestyle='-.')
 #    ax_iterations.plot(np.linspace(1, 1000, combi_32.shape[0]), np.ones(combi_32.shape[0]) * 1228, "k",
 #                       label='Jacobi - Green N=1024', linewidth=2, linestyle='--')

    ax_iterations.set_xlim(1, 1000)
    ax_iterations.set_xticks([1, 1000])
    ax_iterations.set_xticklabels([f'Start', f'Converged'])
    ax_iterations.set_ylim([1, 2600])
    ax_iterations.set_yscale('linear')

    ax_iterations.set_yscale('log')
    ax_iterations.set_ylim([10, 1e4])
    ax_iterations.set_xlabel(" L-BFGS optimization process")
    ax_iterations.set_ylabel(r"$\#$ PCG iterations")
    # ax_iterations.legend(loc='upper right')  # ['Green', 'Green + Jacobi', 'Jacobi'],
    # Adding ticks to the right side
    ax_iterations.tick_params(labelbottom=True, labeltop=False, labelleft=True, labelright=True,
                              bottom=True, top=False, left=True, right=True)

    # ax_iterations.annotate(text=r'Jacobi-Green - $\mathcal{T}$' + f'$_{{{32}}}$',
    #                        xy=(300, 62.0),
    #                        xytext=(350., 30.6),
    #                        arrowprops=dict(arrowstyle='->',
    #                                        color='Black',
    #                                        lw=1,
    #                                        ls='-'),
    #                        color='Black'
    #                        )
    # ax_iterations.annotate(text=r'Jacobi-Green - $\mathcal{T}$' + f'$_{{{64}}}$',
    #                        xy=(380, 88.0),
    #                        xytext=(400., 120.6),
    #                        arrowprops=dict(arrowstyle='->',
    #                                        color='Black',
    #                                        lw=1,
    #                                        ls='-'),
    #                        color='Black'
    #                        )
    # ax_iterations.annotate(text=r'Jacobi-Green - $\mathcal{T}$' + f'$_{{{128}}}$',
    #                        xy=(700, 160.0),
    #                        xytext=(750., 100.6),
    #                        arrowprops=dict(arrowstyle='->',
    #                                        color='Black',
    #                                        lw=1,
    #                                        ls='-'),
    #                        color='Black'
    #                        )
    # ax_iterations.annotate(text=r'Jacobi-Green - $\mathcal{T}$' + f'$_{{{1024}}}$',
    #                        xy=(900, 1160.0),
    #                        xytext=(830., 550.6),
    #                        arrowprops=dict(arrowstyle='->',
    #                                        color='Black',
    #                                        lw=1,
    #                                        ls='-'),
    #                        color='Black'
    #                        )
    # ax_iterations.annotate(text=r'Jacobi - $\mathcal{T}$' + f'$_{{{32}}}$',
    #                        xy=(500, 210.0),
    #                        xytext=(550., 290.0),
    #                        arrowprops=dict(arrowstyle='->',
    #                                        color='Blue',
    #                                        lw=1,
    #                                        ls='-'),
    #                        color='Blue'
    #                        )
    # ax_iterations.annotate(text=r'Jacobi - $\mathcal{T}$' + f'$_{{{64}}}$',
    #                        xy=(500, 420.0),
    #                        xytext=(550., 550.6),
    #                        arrowprops=dict(arrowstyle='->',
    #                                        color='Blue',
    #                                        lw=1,
    #                                        ls='-'),
    #                        color='Blue'
    #                        )
    ax_iterations.annotate(text=r'Green - $\mathcal{T}$' + f'$_{{{32}}}$',
                           xy=(600, 1100.0),
                           xytext=(650., 700.0),
                           arrowprops=dict(arrowstyle='->',
                                           color='Green',
                                           lw=1,
                                           ls='-'),
                           color='Green'
                           )
    ax_iterations.annotate(text=r'Green - $\mathcal{T}$' + f'$_{{{64}}}$',
                           xy=(650, 2400.0),
                           xytext=(670., 1400.),
                           arrowprops=dict(arrowstyle='->',
                                           color='Green',
                                           lw=1,
                                           ls='-'),
                           color='Green'
                           )
    ax_iterations.annotate(text=r'Green - $\mathcal{T}$' + f'$_{{{128}}}$',
                           xy=(750, 3500.0),
                           xytext=(770., 4500.),
                           arrowprops=dict(arrowstyle='->',
                                           color='Green',
                                           lw=1,
                                           ls='-'),
                           color='Green'
                           )
    #####
    init = 1
    middle = 34
    end = 1200
    x = np.linspace(0, 1, N)
    extended_x = np.linspace(0, 1, N + 1)
    # divnorm = mpl.colors.LogNorm(vmin=1, vmax=100)
    # divnorm = mpl.colors.Normalize(vmin=0, vmax=100)
    # pcm = ax.pcolormesh(X, Y, nb_iterations, label='PCG: Green + Jacobi', cmap='Reds', norm=divnorm)
    #divnorm = mpl.colors.LogNorm(vmin=1e-4, vmax=1 )
    divnorm = mpl.colors.Normalize(vmin=1e-8, vmax=1)
    cmap_ = mpl.cm.seismic# mpl.cm.seismic #mpl.cm.Greys
    ax_init = fig.add_subplot(gs[0, 0])
    # ax_init= fig.add_axes([0.15, 0.6, 0.1, 0.2])
    xopt_init = np.load('../exp_data/' + name  + f'_it{init}.npy', allow_pickle=True)
    # xopt_init128 = np.load('../exp_data/' + name2_128 + f'_it{init}.npy', allow_pickle=True)
    pcm = ax_init.pcolormesh(np.tile(xopt_init**2, (nb_tiles, nb_tiles)),
                             cmap=cmap_, linewidth=0,
                             rasterized=True, norm=divnorm)
    ax_init.set_title(r'Initial ', wrap=True)
    ax_init.set_ylabel(r'Density $\rho$')

    # ax1 = fig.add_axes([0.13 , 0.45, 0.1, 0.15])
    # # ax1.set_aspect('equal')
    # letter_offset = 0
    # ax_iterations.text(letter_offset, 1.05, '(b)', transform=ax1.transAxes)
    # # ax1.semilogy(xopt_init[17, :] ** 2, "g", label=r'Green', linewidth=1)
    #
    # # ax1.semilogy(xopt_init[10, :] ** 2, "r",linestyle='--',  label=r'Green',linewidth=1)
    # # ax1.semilogy(xopt_init[20, :] ** 2, "r",linestyle='-.',  label=r'Green',linewidth=1)
    # # ax1.semilogy(xopt_init[30, :] ** 2, "r",linestyle=':',  label=r'Green',linewidth=1)
    # ax1.step(x, xopt_init[:, 12] ** 2, "k", linestyle='--', label=r'Green', linewidth=1)
    #
    # # ax1.axis('square')
    # # ax1.set_xlim(0, N - 1)
    # # ax1.set_ylim([1e-6, 5])
    # ax1.set_xlim(0, 1)
    # ax1.set_ylim([9e-7, 1.1])
    # ax1.set_yticklabels([])
    # ax1.set_xticklabels([])
    # ax1.xaxis.set_ticks_position('none')
    # ax1.yaxis.set_ticks_position('none')
    # ax1.set_aspect('equal', adjustable='box')

    ax_middle = fig.add_subplot(gs[0, 1])
    # ax_middle = fig.add_axes([0.5, 0.6, 0.1, 0.2])
    xopt_middle = np.load('../exp_data/' + name2 + f'_it{middle}.npy', allow_pickle=True)
    ax_middle.pcolormesh(np.tile(xopt_middle**2, (nb_tiles, nb_tiles)),
                         cmap=cmap_, norm=divnorm, linewidth=0,
                         rasterized=True)
    ax_middle.set_title(r'Intermediate', wrap=True)
    # ax1 = fig.add_axes([0.45, 0.45, 0.1, 0.15])
    # # ax1.set_aspect('equal')
    # letter_offset = 0
    # ax_iterations.text(letter_offset, 1.05, '(c)', transform=ax1.transAxes)
    # # ax1.semilogy(xopt_middle[17, :] ** 2, "g", label=r'Green', linewidth=1)
    # # ax1.semilogy(xopt_middle[10, :] ** 2, "r", linestyle='--', label=r'Green', linewidth=1)
    # # ax1.semilogy(xopt_middle[20, :] ** 2, "r", linestyle='-.', label=r'Green', linewidth=1)
    # # ax1.semilogy(xopt_middle[30, :] ** 2, "r", linestyle=':', label=r'Green', linewidth=1)
    # ax1.step(x, xopt_middle[:, 12] ** 2, "k", linestyle='--', label=r'Green', linewidth=1)
    #
    # # ax1.axis('square')
    # ax1.set_xlim(0, 1)
    # ax1.set_ylim([9e-7, 1.1])
    # # ax1.set_xlim(0, 1)
    # # ax1.set_ylim([1e-6, 5])
    # # ax1.set_yticklabels([])
    # ax1.set_xticklabels([])
    # ax1.xaxis.set_ticks_position('none')
    # ax1.yaxis.set_ticks_position('none')
    #
    ax_end = fig.add_subplot(gs[0, 2])
    # ax_end = fig.add_axes([0.7, 0.3, 0.1, 0.2])
    xopt_end = np.load('../exp_data/' + name2  + f'_it{end}.npy', allow_pickle=True)
    # xopt_init128t_end = np.load('../exp_data/' + name2_128 + f'_it{end}.npy', allow_pickle=True)

    pcm = ax_end.pcolormesh(np.tile(xopt_end**2, (nb_tiles, nb_tiles)),
                            cmap=cmap_, norm=divnorm, linewidth=0,
                            rasterized=True)
    ax_end.set_title(r'Converged', wrap=True)
    cbar_ax = fig.add_subplot(gs[0, 3])
    cbar = plt.colorbar(pcm, location='left', cax=cbar_ax)
    cbar.ax.yaxis.tick_right()
    # cbar.set_ticks(ticks=[1e-4,1e-2, 1])
    # cbar.set_ticklabels([f'$10^{{{-4}}}$', f'$10^{{{-2}}}$', 1])
    cbar.set_ticks(ticks=[1e-8, 0.5, 1])
    cbar.set_ticklabels([f'$10^{{{-8}}}$', 0.5, 1])

    fname = src + 'exp_paper_JG_2D_elasticity_TO_iterations_green_only' + '{}'.format('.pdf')
    print(('create figure: {}'.format(fname)))
    plt.savefig(fname, bbox_inches='tight')
    plt.show()