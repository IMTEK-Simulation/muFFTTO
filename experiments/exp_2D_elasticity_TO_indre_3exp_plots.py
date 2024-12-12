import matplotlib as mpl
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from mpl_toolkits.axes_grid1 import make_axes_locatable

from muFFTTO import domain
from muFFTTO import topology_optimization

# Define the dimensions of the 2D array
rows = 25  # or whatever size you want
cols = 25  # or whatever size you want

# Create a random 2D array with 0 and 1
# The probabilities can be adjusted to get a different distribution of bubbles (0) and matrix (1)
array = np.random.choice([0, 1], size=(rows, cols), p=[0.5, 0.5])  # equal probability for 0 and 1
plot_figs = True
plot_movie = True
domain_size = [1, 1]
problem_type = 'elasticity'
discretization_type = 'finite_element'
element_type = 'linear_triangles'  # 'bilinear_rectangle'##'linear_triangles' #
formulation = 'small_strain'

f_sigmas = []
f_pfs = []
# weights = np.concatenate(
#             [np.arange(0.1, 2., 0.1), np.arange(2, 3, 1),np.arange(3, 10, 2), np.arange(10, 110, 10)])
weights = np.concatenate([np.arange(0.1, 1., 1)])
for ration in [0.0]:  # 0.2,0.1,0.0,-0.1,-0.2,-0.3,-0.4,-0.5,-0.6,-0.7,-0.8,-0.9 ]:
    poison_target = ration
    plt.clf()
    for w_mult in weights:  # np.arange(0.1, 1., 0.1):# [1]:
        for eta_mult in [0.01]:  # np.arange(0.05, 0.5, 0.05):#[0.1 ]:
            energy_objective = False
            print(w_mult, eta_mult)
            pixel_size = 0.0078125
            # eta = 0.03125  # eta_mult * pixel_size
            E_target_0 = 0.3
            N = 32
            cores = 4
            p = 2
            nb_load_cases = 3
            random_initial_geometry = True
            bounds = False
            optimizer = 'lbfg'  # adam2
            script_name = 'exp_2D_elasticity_TO_indre_3exp'

            # name = (            f'{optimizer}_muFFTTO_elasticity_{script_name}_N{N}_E_target_0.15_Poisson_-0.5_Poisson0_0.2_w{w_mult}_eta{eta_mult}_p{p}_bounds={bounds}_FE_NuMPI{cores}_nb_load_cases_{nb_load_cases}_energy_objective_{energy_objective}_random_{random_initial_geometry}')
            name = (
                f'{optimizer}_muFFTTO_elasticity_{script_name}_N{N}_E_target_{E_target_0}_Poisson_{poison_target}_Poisson0_0.0_w{w_mult:.2f}_eta{eta_mult}_p{p}_bounds={bounds}_FE_NuMPI{cores}_nb_load_cases_{nb_load_cases}_energy_objective_{energy_objective}_random_{random_initial_geometry}')

        if plot_figs:
            phase_field = np.load('exp_data/' + name + f'.npy', allow_pickle=True)

            my_cell = domain.PeriodicUnitCell(domain_size=domain_size,
                                              problem_type=problem_type)
            discretization = domain.Discretization(cell=my_cell,
                                                   nb_of_pixels_global=(N, N),
                                                   discretization_type=discretization_type,
                                                   element_type=element_type)
            f_phase_field = topology_optimization.objective_function_phase_field(discretization=discretization,
                                                                                 phase_field_1nxyz=np.expand_dims(
                                                                                     np.expand_dims(phase_field,
                                                                                                    axis=0), axis=0),
                                                                                 eta=eta_mult,
                                                                                 double_well_depth=1)

            xopt = np.load('exp_data/' + name + f'xopt_log.npz', allow_pickle=True)

            src = './figures/'  # source folder\
            fig_data_name = f'muFFTTO_{name}'  # print('rank' f'{MPI.COMM_WORLD.rank:6} ')

            fig, ax = plt.subplots(1, 4, figsize=(16, 4))
            # plt.contourf(np.tile(phase_field, (3, 3)), cmap=mpl.cm.Greys)

            contour = ax[3].contourf(np.tile(phase_field, (3, 3)), cmap='jet', vmin=0, vmax=1)

            # Colorbar
            divider = make_axes_locatable(ax[3])

            ax_cb = divider.new_horizontal(size="5%", pad=0.05)
            fig.add_axes(ax_cb)

            # cbar = fig.colorbar(contour, cax=ax_cb)

            cbar = fig.colorbar(mpl.cm.ScalarMappable(norm=contour.norm, cmap=contour.cmap), cax=ax_cb,
                                ticks=np.arange(0, 1.2, 0.2))
            cbar.ax.set_ylabel(r'Phase $\rho$', rotation=90, labelpad=10)

            ax[3].set_aspect('equal')
            ax[3].set_xlabel(r'Position x')
            ax[3].set_ylabel(r'Position y')
            ax[3].set_yticklabels([])
            ax[3].set_xticklabels([])
            # nodal_coordinates[0, 0] * number_of_pixels[0], nodal_coordinates[1, 0] * number_of_pixels[0],
            #            plt.clim(0, 1)
            # plt.colorbar()
            target_stresses = []
            target_stresses.append(xopt.f.target_stress0)
            target_stresses.append(xopt.f.target_stress1)
            target_stresses.append(xopt.f.target_stress2)
            homo_stresses = []
            homo_stresses.append(xopt.f.homogenized_stresses0)
            homo_stresses.append(xopt.f.homogenized_stresses1)
            homo_stresses.append(xopt.f.homogenized_stresses2)
            stress_difference_ij = []
            f_sigma = []

            for lc in np.arange(nb_load_cases):
                stress_difference_ij.append(target_stresses[lc] - homo_stresses[lc])

                f_sigma.append(np.sum(stress_difference_ij[lc] ** 2) / np.sum(target_stresses[lc] ** 2))
                ax[lc].set_axis_off()
                np.set_printoptions(suppress=True, precision=5)
                ax[lc].text(-0.3, 0.5, f'target =\n {target_stresses[lc]}\n,'
                                       f' optimized =\n {homo_stresses[lc]}\n,'
                                       f' stress_difference_ij = \n{stress_difference_ij[lc]}\n,'
                                       f' f_sigma = {f_sigma[lc]:.5f}\n,')
            ax[0].text(-0.3, 0.3, f' f_sigma tot = \n{np.sum(f_sigma):.5f}\n,')
            ax[1].text(-0.3, 0.3, f' f_sigma tot = \n{f_phase_field:.5f}\n,')

            ax[0].text(-0.3, -0., f' homogenized_C_ij = \n{xopt.f.homogenized_C_ijkl}\n,')
            ax[1].text(-0.3, -0., f' target_C_ij = \n{xopt.f.target_C_ijkl}\n,')
            ax[2].text(-0.3, -0., f' diff in C_ij = \n{xopt.f.target_C_ijkl - xopt.f.homogenized_C_ijkl},')
            fname = src + f'{w_mult:.2f}' + fig_data_name + '{}'.format('.png')
            print(('create figure: {}'.format(fname)))  # axes[1, 0].legend(loc='upper right')
            plt.savefig(fname, bbox_inches='tight')
            plt.show()
            f_sigmas.append(np.sum(f_sigma))
            f_pfs.append(f_phase_field)


    fig = plt.figure(figsize=(12, 6))

    gs = fig.add_gridspec(3, 4)
    ax1 = fig.add_subplot(gs[0, 0])
    name = (
        f'{optimizer}_muFFTTO_elasticity_{script_name}_N{N}_E_target_{E_target_0}_Poisson_{poison_target}_Poisson0_0.0_w{0.10:.2f}_eta{eta_mult}_p{p}_bounds={bounds}_FE_NuMPI{cores}_nb_load_cases_{nb_load_cases}_energy_objective_{energy_objective}_random_{random_initial_geometry}')
    phase_field = np.load('exp_data/' + name + f'.npy', allow_pickle=True)
    contour = ax1.contourf(np.tile(phase_field, (3, 3)), cmap='jet', vmin=0, vmax=1)
    # Colorbar
    # divider = make_axes_locatable(ax1)
    # ax_cb = divider.new_horizontal(size="5%", pad=0.05)
    # fig.add_axes(ax_cb)
    # cbar = fig.colorbar(mpl.cm.ScalarMappable(norm=contour.norm, cmap=contour.cmap), cax=ax_cb,
    #                     ticks=np.arange(0, 1.2, 0.2))
    # cbar.ax.set_ylabel(r'Phase $\rho$', rotation=90, labelpad=10)
    ax1.set_aspect('equal')
    ax1.set_xlabel(r'w=0.1')
    ax1.xaxis.set_label_position('top')
    #ax1.set_ylabel(r'Position y')
    ax1.set_yticklabels([])
    ax1.set_xticklabels([])

    ax2 = fig.add_subplot(gs[0, 1])
    name = (
        f'{optimizer}_muFFTTO_elasticity_{script_name}_N{N}_E_target_{E_target_0}_Poisson_{poison_target}_Poisson0_0.0_w{1.0:.2f}_eta{eta_mult}_p{p}_bounds={bounds}_FE_NuMPI{cores}_nb_load_cases_{nb_load_cases}_energy_objective_{energy_objective}_random_{random_initial_geometry}')
    phase_field = np.load('exp_data/' + name + f'.npy', allow_pickle=True)
    contour = ax2.contourf(np.tile(phase_field, (3, 3)), cmap='jet', vmin=0, vmax=1)
    # Colorbar
    #divider = make_axes_locatable(ax2)
    #ax_cb = divider.new_horizontal(size="5%", pad=0.05)
    #fig.add_axes(ax_cb)
    # cbar = fig.colorbar(mpl.cm.ScalarMappable(norm=contour.norm, cmap=contour.cmap), cax=ax_cb,
    #                     ticks=np.arange(0, 1.2, 0.2))
    #cbar.ax.set_ylabel(r'Phase $\rho$', rotation=90, labelpad=10)
    ax2.set_aspect('equal')
    ax2.set_xlabel(r'w=1.0')
    ax2.xaxis.set_label_position('top')
    #ax2.set_ylabel(r'Position y')
    ax2.set_yticklabels([])
    ax2.set_xticklabels([])

    ax3 = fig.add_subplot(gs[0, 2])
    name = (
        f'{optimizer}_muFFTTO_elasticity_{script_name}_N{N}_E_target_{E_target_0}_Poisson_{poison_target}_Poisson0_0.0_w{10.0:.2f}_eta{eta_mult}_p{p}_bounds={bounds}_FE_NuMPI{cores}_nb_load_cases_{nb_load_cases}_energy_objective_{energy_objective}_random_{random_initial_geometry}')
    phase_field = np.load('exp_data/' + name + f'.npy', allow_pickle=True)
    contour = ax3.contourf(np.tile(phase_field, (3, 3)), cmap='jet', vmin=0, vmax=1)
    # Colorbar
    # divider = make_axes_locatable(ax3)
    # ax_cb = divider.new_horizontal(size="5%", pad=0.05)
    # fig.add_axes(ax_cb)
    # cbar = fig.colorbar(mpl.cm.ScalarMappable(norm=contour.norm, cmap=contour.cmap), cax=ax_cb,
    #                     ticks=np.arange(0, 1.2, 0.2))
    # cbar.ax.set_ylabel(r'Phase $\rho$', rotation=90, labelpad=10)
    ax3.set_aspect('equal')
    ax3.set_xlabel(r'w=10')
    ax3.xaxis.set_label_position('top')
    #ax2.set_ylabel(r'Position y')
    ax3.set_yticklabels([])
    ax3.set_xticklabels([])

    ax4 = fig.add_subplot(gs[0, 3])
    name = (
        f'{optimizer}_muFFTTO_elasticity_{script_name}_N{N}_E_target_{E_target_0}_Poisson_{poison_target}_Poisson0_0.0_w{100.0:.2f}_eta{eta_mult}_p{p}_bounds={bounds}_FE_NuMPI{cores}_nb_load_cases_{nb_load_cases}_energy_objective_{energy_objective}_random_{random_initial_geometry}')
    phase_field = np.load('exp_data/' + name + f'.npy', allow_pickle=True)
    contour = ax4.contourf(np.tile(phase_field, (3, 3)), cmap='jet', vmin=0, vmax=1)
    # Colorbar
    divider = make_axes_locatable(ax4)
    ax_cb = divider.new_horizontal(size="5%", pad=0.05)
    fig.add_axes(ax_cb)
    cbar = fig.colorbar(mpl.cm.ScalarMappable(norm=contour.norm, cmap=contour.cmap), cax=ax_cb,
                        ticks=np.arange(0, 1.2, 0.2))
    cbar.ax.set_ylabel(r'Phase $\rho$', rotation=90, labelpad=10)
    ax4.set_aspect('equal')
    ax4.set_xlabel(r'w=100')
    ax4.xaxis.set_label_position('top')
    # ax2.set_ylabel(r'Position y')
    ax4.set_yticklabels([])
    ax4.set_xticklabels([])

    ax5 = fig.add_subplot(gs[1:, :])

    # fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    ax5.loglog(weights, f_sigmas, '--', color='k', linewidth=1)
    ax5.loglog(weights, f_pfs, '-', color='r', linewidth=1)
    ax5.legend([r'$f_{\sigma}$', r'$f_{\rho}$'], loc='lower left')
    # ax.set_aspect('equal')
    ax5.set_xlabel(r'weight $w$')
    fname = src + fig_data_name + 'wdependency{}'.format('.png')
    print(('create figure: {}'.format(fname)))
    plt.savefig(fname, bbox_inches='tight')
    plt.show()
    quit()
    if plot_movie:
        if random_initial_geometry:
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
