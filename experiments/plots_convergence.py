import matplotlib as mpl
import numpy as np
import matplotlib.pyplot as plt
from PySide2.examples.opengl.contextinfo import colors
from matplotlib.animation import FuncAnimation, PillowWriter

# Define the dimensions of the 2D array
rows = 25  # or whatever size you want
cols = 25  # or whatever size you want

# Create a random 2D array with 0 and 1
# The probabilities can be adjusted to get a different distribution of bubbles (0) and matrix (1)
array = np.random.choice([0, 1], size=(rows, cols), p=[0.5, 0.5])  # equal probability for 0 and 1
plot_figs = True
plot_movie = True
for ration in [-0.5, ]:  # 0.2,0.1,0.0,-0.1,-0.2,-0.3,-0.4,-0.5,-0.6,-0.7,-0.8,-0.9 ]:
    poison_target = ration
    plt.clf()
    for w_mult in [4.00]:  # np.arange(0.1, 1., 0.1):# [1]:
        for eta_mult in [0.0101]:  # np.arange(0.05, 0.5, 0.05):#[0.1 ]:
            energy_objective = False
            print(w_mult, eta_mult)
            pixel_size = 0.0078125
            eta = 0.03125  # eta_mult * pixel_size
            N = 64
            cores = 6
            p = 2
            nb_load_cases = 3
            random_initial_geometry = True
            bounds = False
            optimizer = 'lbfg'  # adam2
            script_name = 'exp_2D_elasticity_TO_indre_3exp'

            # name = (            f'{optimizer}_muFFTTO_elasticity_{script_name}_N{N}_E_target_0.15_Poisson_-0.5_Poisson0_0.2_w{w_mult}_eta{eta_mult}_p{p}_bounds={bounds}_FE_NuMPI{cores}_nb_load_cases_{nb_load_cases}_energy_objective_{energy_objective}_random_{random_initial_geometry}')
            # DGO
            name = (
                f'{optimizer}_muFFTTO_elasticity_{script_name}_N{N}_E_target_0.15_Poisson_{poison_target}_Poisson0_0.0_w{w_mult}_eta{eta_mult}_p{p}_bounds={bounds}_FE_NuMPI{cores}_nb_load_cases_{nb_load_cases}_energy_objective_{energy_objective}_random_{random_initial_geometry}')
            eta_mult = 0.0102# combined
            name2 = (
                f'{optimizer}_muFFTTO_elasticity_{script_name}_N{N}_E_target_0.15_Poisson_{poison_target}_Poisson0_0.0_w{w_mult}_eta{eta_mult}_p{p}_bounds={bounds}_FE_NuMPI{cores}_nb_load_cases_{nb_load_cases}_energy_objective_{energy_objective}_random_{random_initial_geometry}')
            eta_mult = 0.0103 # Jakobi
            name3 = (
                f'{optimizer}_muFFTTO_elasticity_{script_name}_N{N}_E_target_0.15_Poisson_{poison_target}_Poisson0_0.0_w{w_mult}_eta{eta_mult}_p{p}_bounds={bounds}_FE_NuMPI{cores}_nb_load_cases_{nb_load_cases}_energy_objective_{energy_objective}_random_{random_initial_geometry}')

        if plot_figs:
            phase_field = np.load('exp_data/' + name + f'.npy', allow_pickle=True)

            xopt = np.load('exp_data/' + name + f'xopt_log.npz', allow_pickle=True)
            xopt2 = np.load('exp_data/' + name2 + f'xopt_log.npz', allow_pickle=True)
            xopt3 = np.load('exp_data/' + name3 + f'xopt_log.npz', allow_pickle=True)

            src = './figures/'  # source folder\
            fig_data_name = f'muFFTTO_{name}'  # print('rank' f'{MPI.COMM_WORLD.rank:6} ')

            plt.show()
            plt.figure()
            fig_data_name = f'muFFTTO_nb_it_{phase_field.shape}_line'  # print('rank' f'{MPI.COMM_WORLD.rank:6} ')
            # plt.semilogy(xopt.f.norms_f-xopt.f.norms_f[-1], label='objective f')
            # plt.semilogy(xopt.f.norms_pf-xopt.f.norms_pf[-1], label='phase field')
            # plt.semilogy(np.abs(xopt.f.norms_sigma[:,0]-xopt.f.norms_pf - xopt.f.norms_sigma[-1,0]+xopt.f.norms_pf[-1]), label='stress')
            # plt.plot([305,321,458,724,831,1477,1454,2180,2457,3919,5880,7047,6154,5879,6405,7309,7241,7614,7566,10000,10000,9962], label='No preconditioner',linewidth=2)

            plt.plot(xopt.f.num_iteration_.transpose(), label='FFT + Diagonal', linewidth=2)
            # plt.plot(xopt2.f.num_iteration_.transpose(), label='FFT',linewidth=2)

            plt.plot(xopt3.f.num_iteration_.transpose(), label='Diagonal', linewidth=2)
            # plt.semilogy(xopt.f.norms_pf, label='phase field')
            # plt.semilogy(xopt.f.norms_sigma[:, 0] - xopt.f.norms_pf,    label='stress')
            # -xopt.f.norms_sigma[-1,0]+xopt.f.norms_pf[-1]
            # nodal_coordinates[0, 0] * number_of_pixels[0], nodal_coordinates[1, 0] * number_of_pixels[0],
            # plt.grid(True)
            # plt.title('Number of CG iteration per L-BFGS step {}'.format(optimizer))
            plt.xlabel(" L-BFGS step (iteration) ")
            plt.ylabel("# CG iteration")

            # plt.minorticks_on()
            fname = src + fig_data_name + '{}'.format('.png')
            print(('create figure: {}'.format(fname)))
            plt.legend()
            plt.show()

            plt.figure()
            fig_data_name = f'muFFTTO_{phase_field.shape}_line relative'  # print('rank' f'{MPI.COMM_WORLD.rank:6} ')
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
            fig_data_name = f'muFFTTO_{phase_field.shape}_line'  # print('rank' f'{MPI.COMM_WORLD.rank:6} ')
            plt.semilogy(xopt.f.norms_f, label='objective f')

            plt.semilogy(xopt.f.norms_delta_f, label='Δf')
            plt.semilogy(xopt.f.norms_max_grad_f, label='max ∇ f')

            plt.semilogy(xopt.f.norms_norm_grad_f, label='|∇ f|')
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
    if plot_movie:
        for nb_tiles in [1, 3]:
            # fig = plt.figure()
            fig, axs = plt.subplots(nrows=2, ncols=1,
                                    figsize=(6, 6)  )
            # axs[0] = plt.axes(xlim=(0, nb_tiles * N), ylim=(0, nb_tiles * N))
            axs[0].imshow(np.tile(phase_field, (nb_tiles, nb_tiles)), cmap=mpl.cm.Greys, vmin=0, vmax=1)

            axs[1].plot(xopt.f.num_iteration_.transpose()[::3], 'w'  , linewidth=0)
            #axs[1].plot(xopt3.f.num_iteration_.transpose(), "b", label='Jacoby', linewidth=0)
            #axs[1].plot(xopt.f.num_iteration_.transpose(), "k", label='DGO + Jacoby', linewidth=0)
            #legend = plt.legend()
            # Animation function to update the image
            axs[1].set_xlabel(" L-BFGS step (iteration) ")
            axs[1].set_ylabel("# CG iteration")
            dgo = (xopt.f.num_iteration_.transpose()[::3] +
                   xopt.f.num_iteration_.transpose()[1::3] +
                   xopt.f.num_iteration_.transpose()[2::3])/3
            jacoby = (xopt3.f.num_iteration_.transpose()[::3] +
                   xopt3.f.num_iteration_.transpose()[1::3] +
                   xopt3.f.num_iteration_.transpose()[2::3]) / 3
            combi = (xopt2.f.num_iteration_.transpose()[::3] +
                      xopt2.f.num_iteration_.transpose()[1::3] +
                      xopt2.f.num_iteration_.transpose()[2::3]) / 3
            def update(i):
                xopt_it = np.load('exp_data/' + name + f'_it{i + 1}.npy', allow_pickle=True)

                axs[0].clear()
                axs[0].imshow(np.tile(xopt_it, (nb_tiles, nb_tiles)), cmap=mpl.cm.Greys, vmin=0, vmax=1)

                axs[0].set_title('iteration {}, optimizer {}'.format(i, optimizer))


                axs[1].plot(dgo[0:i+1],"r", label='DGO ',linewidth=1)
                #axs[1].plot(xopt2.f.num_iteration_.transpose()[1:3*i+1:3],"r", label='DGO ',linewidth=1)
                #axs[1].plot(xopt2.f.num_iteration_.transpose()[2:3*i+2:3],"r", label='DGO ',linewidth=1)

                axs[1].plot(jacoby[0:i+1], "b", label='Jacoby', linewidth=1)
                axs[1].plot(combi[0:i+1], "k", label='DGO + Jacoby', linewidth=1)
                #axs[1].legend()
                plt.legend([ '', 'DGO', 'Jacoby', 'DGO + Jacoby' ])
                # img.set_array(xopt_it)


            # Create animation
            #ani = FuncAnimation(fig, update, frames=xopt.f.norms_f.size - 1, blit=False)

            ani = FuncAnimation(fig, update, frames=500, blit=False)
            #axs[1].legend()
            # Save as a GIF
            ani.save(f"./figures/movie{nb_tiles}_exp2_imshow_{name}.gif", writer=PillowWriter(fps=10))
