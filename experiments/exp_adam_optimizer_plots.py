import matplotlib as mpl
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

# Define the dimensions of the 2D array
rows = 25  # or whatever size you want
cols = 25  # or whatever size you want

# Create a random 2D array with 0 and 1
# The probabilities can be adjusted to get a different distribution of bubbles (0) and matrix (1)
array = np.random.choice([0, 1], size=(rows, cols), p=[0.5, 0.5])  # equal probability for 0 and 1
plot_figs = True
plot_movie = True
for ration in [-0.5]:#0.2,0.1,0.0,-0.1,-0.2,-0.3,-0.4,-0.5,-0.6,-0.7,-0.8,-0.9 ]:
    poison_target = ration
    plt.clf()
    for w_mult in [4.00]:  # np.arange(0.1, 1., 0.1):# [1]:
        for eta_mult in [0.01]:  # np.arange(0.05, 0.5, 0.05):#[0.1 ]:
            energy_objective = False
            print(w_mult, eta_mult)
            pixel_size = 0.0078125
            eta = 0.03125  # eta_mult * pixel_size
            N = 16
            cores = 2
            p = 2
            nb_load_cases = 1
            random_initial_geometry = True
            bounds = False
            optimizer = 'lbfg'  # adam2
            script_name = 'exp_2D_elasticity_TO_indre_3exp'

            # name = (            f'{optimizer}_muFFTTO_elasticity_{script_name}_N{N}_E_target_0.15_Poisson_-0.5_Poisson0_0.2_w{w_mult}_eta{eta_mult}_p{p}_bounds={bounds}_FE_NuMPI{cores}_nb_load_cases_{nb_load_cases}_energy_objective_{energy_objective}_random_{random_initial_geometry}')
            name = (
                f'{optimizer}_muFFTTO_elasticity_{script_name}_N{N}_E_target_0.15_Poisson_{poison_target}_Poisson0_0.0_w{w_mult}_eta{eta_mult}_p{p}_bounds={bounds}_FE_NuMPI{cores}_nb_load_cases_{nb_load_cases}_energy_objective_{energy_objective}_random_{random_initial_geometry}')

        if plot_figs:
            phase_field = np.load('exp_data/' + name + f'.npy', allow_pickle=True)

            xopt = np.load('exp_data/' + name + f'xopt_log.npz', allow_pickle=True)

            src = './figures/'  # source folder\
            fig_data_name = f'muFFTTO_{name}'  # print('rank' f'{MPI.COMM_WORLD.rank:6} ')

            fig, ax = plt.subplots(1, 2)
            plt.contourf(np.tile(phase_field, (3, 3)), cmap=mpl.cm.Greys)
            # nodal_coordinates[0, 0] * number_of_pixels[0], nodal_coordinates[1, 0] * number_of_pixels[0],
            plt.clim(0, 1)
            plt.colorbar()
            ax[0].set_axis_off()
            ax[0].text(-0.3, 0.5, f'target =\n {xopt.f.target_stress0}\n,'
                                  f' optimized =\n {xopt.f.homogenized_stresses0}\n,'
                                  f' diff = \n{xopt.f.homogenized_stresses0 - xopt.f.target_stress0}\n,'
                                  f' homogenized_C_ijkl = \n{xopt.f.homogenized_C_ijkl}'
                       )

            fname = src + fig_data_name + f'{3}' + '{}'.format('.png')
            print(('create figure: {}'.format(fname)))  # axes[1, 0].legend(loc='upper right')
            plt.savefig(fname, bbox_inches='tight')
            plt.show()

            plt.figure()
            plt.contourf(np.tile(phase_field, (1, 1)), cmap=mpl.cm.Greys)
            # plt.title(f'w = {w_mult},eta= {eta_mult}\n, {xopt.f.homogenized_stresses}')

            # nodal_coordinates[0, 0] * number_of_pixels[0], nodal_coordinates[1, 0] * number_of_pixels[0],
            plt.clim(0, 1)
            plt.colorbar()
            # plt.title(f'w = {w},eta= {eta_mult}\n, {xopt.f.message}')
            fname = src + fig_data_name + '{}'.format('.png')
            print(('create figure: {}'.format(fname)))  # axes[1, 0].legend(loc='upper right')
            plt.savefig(fname, bbox_inches='tight')
            plt.show()
            plt.figure()
            fig_data_name = f'muFFTTO_{phase_field.shape}_line'  # print('rank' f'{MPI.COMM_WORLD.rank:6} ')

            plt.plot(np.tile(phase_field, (1, 1))[:, 3].transpose())
            # nodal_coordinates[0, 0] * number_of_pixels[0], nodal_coordinates[1, 0] * number_of_pixels[0],
            plt.grid(True)
            plt.minorticks_on()
            fname = src + fig_data_name + '{}'.format('.png')
            print(('create figure: {}'.format(fname)))  # axes[1, 0].legend(loc='upper right')
            # plt.savefig(fname, bbox_inches='tight')
            plt.show()
            plt.figure()
            fig_data_name = f'muFFTTO_{phase_field.shape}_line'  # print('rank' f'{MPI.COMM_WORLD.rank:6} ')
            # plt.semilogy(xopt.f.norms_f-xopt.f.norms_f[-1], label='objective f')
            # plt.semilogy(xopt.f.norms_pf-xopt.f.norms_pf[-1], label='phase field')
            # plt.semilogy(np.abs(xopt.f.norms_sigma[:,0]-xopt.f.norms_pf - xopt.f.norms_sigma[-1,0]+xopt.f.norms_pf[-1]), label='stress')
            plt.semilogy(xopt.f.norms_f, label='objective f')
            # plt.semilogy(xopt.f.norms_pf, label='phase field')
            # plt.semilogy(xopt.f.norms_sigma[:, 0] - xopt.f.norms_pf,    label='stress')
            # -xopt.f.norms_sigma[-1,0]+xopt.f.norms_pf[-1]
            # nodal_coordinates[0, 0] * number_of_pixels[0], nodal_coordinates[1, 0] * number_of_pixels[0],
            plt.grid(True)
            plt.title('optimizer {}'.format(optimizer))

            # plt.minorticks_on()
            fname = src + fig_data_name + '{}'.format('.png')
            print(('create figure: {}'.format(fname)))

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
            fig = plt.figure()
            axi = plt.axes(xlim=(0, nb_tiles * N), ylim=(0, nb_tiles * N))


            # Animation function to update the image
            def update(i):
                xopt_it = np.load('exp_data/' + name + f'_it{i + 1}.npy', allow_pickle=True)

                axi.clear()
                axi.imshow(np.tile(xopt_it, (nb_tiles, nb_tiles)), cmap=mpl.cm.Greys, vmin=0, vmax=1)

                axi.set_title('iteration {}, optimizer {}'.format(i, optimizer))
                # img.set_array(xopt_it)


            # Create animation
            ani = FuncAnimation(fig, update, frames=xopt.f.norms_f.size - 1, blit=False)
            # ani = FuncAnimation(fig, update, frames=115, blit=False)

            # Save as a GIF
            ani.save(f"./figures/movie{nb_tiles}_exp2_imshow_{name}.gif", writer=PillowWriter(fps=40))
