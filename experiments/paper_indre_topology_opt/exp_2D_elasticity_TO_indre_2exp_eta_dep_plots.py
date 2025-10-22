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

from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
### ----- Define the hexagonal grid ----- ###
def make_parallelograms(displ ):
    parallelograms = []
    nx = displ.shape[1] - 1  # number of squares in x direction
    ny = displ.shape[2] - 1  # number of squares in y direction

    for x in range(nx):
        for y in range(ny):
            corner_1 = displ[:, x, y]
            corner_2 = displ[:, x + 1, y]
            corner_3 = displ[:, x + 1, y + 1]
            corner_4 = displ[:, x, y + 1]
            corners = np.stack([corner_1, corner_2, corner_3, corner_4],
                               axis=1).T
            parallelogram = Polygon(corners)
            parallelograms.append(parallelogram)
            #parallelograms.set_edgecolor('face')
    return PatchCollection(parallelograms, cmap='jet', linewidth=0, edgecolor='None' , antialiased=False, alpha=1.0 )

# Create a random 2D array with 0 and 1
# The probabilities can be adjusted to get a different distribution of bubbles (0) and matrix (1)
array = np.random.choice([0, 1], size=(rows, cols), p=[0.5, 0.5])  # equal probability for 0 and 1
plot_figs = True
plot_movie = True
domain_size = [1, 1]
problem_type = 'elasticity'
discretization_type = 'finite_element'
element_type = 'linear_triangles'  # 'bilinear_rectangle'##'linear_triangles' #linear_triangles_tilled
formulation = 'small_strain'

f_sigmas = []
f_pfs = []
# weights = np.concatenate(
#             [np.arange(0.1, 2., 0.1), np.arange(2, 3, 1),np.arange(3, 10, 2), np.arange(10, 110, 10)])
# #weights = np.concatenate(
 #           [np.arange(0.1, 2., 0.1), np.arange(2, 3, 1),np.arange(3, 10, 2), np.arange(10, 20, 10)])
#weights = np.concatenate([np.arange(0.1, 1., 1)])
#weights = np.concatenate([np.arange(0.1, 1., 1)])
weights = np.arange(1, 2., 1)
for ration in [0.0]:  # 0.2,0.1,0.0,-0.1,-0.2,-0.3,-0.4,-0.5,-0.6,-0.7,-0.8,-0.9 ]:
    poison_target = ration
    plt.clf()
    for w_mult in weights:  # np.arange(0.1, 1., 0.1):# [1]:
        etas=[0.005,0.01,0.02,0.05]#np.concatenate([np.arange(0.005, 0.03, 0.005)])
        for eta_mult in etas :  # np.arange(0.05, 0.5, 0.05):#[0.1 ]: [0.01]
            energy_objective = False
            print(w_mult, eta_mult)
            pixel_size = 0.0078125
            # eta = 0.03125  # eta_mult * pixel_size
            E_target_0 = 0.3
            N = 64
            cores = 6
            p = 2
            nb_load_cases = 3
            random_initial_geometry = True
            bounds = False
            optimizer = 'lbfg'  # adam2
            script_name = 'exp_2D_elasticity_TO_indre_2exp'

            # name = (            f'{optimizer}_muFFTTO_elasticity_{script_name}_N{N}_E_target_0.15_Poisson_-0.5_Poisson0_0.2_w{w_mult}_eta{eta_mult}_p{p}_bounds={bounds}_FE_NuMPI{cores}_nb_load_cases_{nb_load_cases}_energy_objective_{energy_objective}_random_{random_initial_geometry}')
            # name = (
            #     f'{optimizer}_muFFTTO_elasticity_{script_name}_N{N}_E_target_{E_target_0}_Poisson_{poison_target}_Poisson0_0.0_w{w_mult}_eta{eta_mult}_p{p}_bounds={bounds}_FE_NuMPI{cores}_nb_load_cases_{nb_load_cases}_energy_objective_{energy_objective}_random_{random_initial_geometry}')
            name = (
            f'{optimizer}_muFFTTO_elasticity_{element_type}_{script_name}_N{N}_E_target_{E_target_0}_Poisson_{poison_target}_Poisson0_0.0_w{w_mult:.2f}_eta{eta_mult}_p{p}_bounds={bounds}_FE_NuMPI{cores}_nb_load_cases_{nb_load_cases}_energy_objective_{energy_objective}_random_{random_initial_geometry}')

            if plot_figs:
                phase_field = np.load('../exp_data/' + name + f'.npy', allow_pickle=True)

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

                xopt = np.load('../exp_data/'+ name + f'xopt_log.npz', allow_pickle=True)

                src = './figures/'  # source folder\
                fig_data_name = f'muFFTTO_{name}'  # print('rank' f'{MPI.COMM_WORLD.rank:6} ')

                fig = plt.figure(figsize=(11, 4.5))
                gs = fig.add_gridspec(1, 2)
                               # nodal_coordinates[0, 0] * number_of_pixels[0], nodal_coordinates[1, 0] * number_of_pixels[0],
                #            plt.clim(0, 1)


                ax1 = fig.add_subplot(gs[0, 1])

                # fig, ax = plt.subplots(1, 2, figsize=(8, 4))
                # plt.contourf(np.tile(phase_field, (3, 3)), cmap=mpl.cm.Greys)
                #ax1.hlines(y=N//2, xmin=0, xmax=N, colors='black', linestyles='--', linewidth=1.)
                ax1.plot([0, 1], [0, 1], color='black', linestyle='--', linewidth=1.,transform=ax1.transAxes)
                nb_reps = 1
                roll_x = 0
                roll_y = -25
                #phase_field=phase_field**2
                phase_field = np.roll(phase_field, roll_y, axis=0)
                phase_field = np.roll(phase_field, roll_x, axis=1)
                contour = ax1.contourf(np.tile(phase_field, (nb_reps, nb_reps)), cmap='jet', vmin=0, vmax=1)

                # Colorbar
                divider = make_axes_locatable(ax1)

                ax_cb = divider.new_horizontal(size="5%", pad=0.05)
                fig.add_axes(ax_cb)

                # cbar = fig.colorbar(contour, cax=ax_cb)

                cbar = fig.colorbar(mpl.cm.ScalarMappable(norm=contour.norm, cmap=contour.cmap), cax=ax_cb,
                                    ticks=np.arange(0, 1.2, 0.2))
                cbar.ax.set_ylabel(r'Phase $\rho$', rotation=90, labelpad=10)

                ax1.set_aspect('equal')
                ax1.set_xlabel(r'Position x')
                ax1.set_ylabel(r'Position y')
                ax1.set_xlim(0, N-1)
                ax1.set_ylim(0, N-1)
                ax1.set_yticklabels([])
                ax1.set_xticklabels([])
                # plot cross section
                ax0 = fig.add_subplot(gs[0, 0])
                #ax0.plot(phase_field[N // 2, :])
                ax0.plot(np.diag(phase_field))

                ax0.hlines(y=1, xmin=0, xmax=N, colors='black', linestyles='--', linewidth=1.)
                ax0.hlines(y=0, xmin=0, xmax=N, colors='black', linestyles='--', linewidth=1.)

                ax0.set_title(r'Cross section at y=20')
                # ax0.set_aspect('equal')
                ax0.set_xlabel(r'Position x')
                ax0.set_xlim(0, N)
                ax0.set_ylim(-0.1, 1.1)

                fname = src + f'{w_mult:.2f}' + fig_data_name + '{}'.format('.png')
                print(('create figure: {}'.format(fname)))  # axes[1, 0].legend(loc='upper right')
                # plt.savefig(fname, bbox_inches='tight')

                #plt.show()
                #f_sigmas.append(np.sum(f_sigma))
                #f_pfs.append(f_phase_field)

    fig = plt.figure(figsize=(11, 4.5))
    gs = fig.add_gridspec(2, 5, width_ratios=[0.1,1,1, 1, 1])
    ax0 = fig.add_subplot(gs[0, 0:])
    colors = ['red', 'blue', 'green', 'orange', 'purple']
    linestyles = ['-', '--', '-.', ':', (0, (3, 1, 1, 1))]
    letter_offset=-0.18

    for i in np.arange(4):  # np.arange(0.05, 0.5, 0.05):#[0.1 ]: [0.01]
        eta_mult = etas[i]
        name = (
            f'{optimizer}_muFFTTO_elasticity_{element_type}_{script_name}_N{N}_E_target_{E_target_0}_Poisson_{poison_target}_Poisson0_0.0_w{w_mult:.2f}_eta{eta_mult}_p{p}_bounds={bounds}_FE_NuMPI{cores}_nb_load_cases_{nb_load_cases}_energy_objective_{energy_objective}_random_{random_initial_geometry}')
        phase_field = np.load('../exp_data/' + name + f'.npy', allow_pickle=True)

        nb_reps = 1
        if i == 0:
            roll_x = 16
            roll_y = -21
            ax1 = fig.add_subplot(gs[1, i+1])
            ax0.annotate(text=r'$\eta = $' + f'{eta_mult}'+r'$L$',
                         xy=(19, 1.0),
                         xytext=(23., 0.6),
                         arrowprops=dict(arrowstyle='->',
                                         color=colors[i],
                                         lw=1,
                                         ls='-'),
                         color=colors[i]
                         )
            ax0.text(letter_offset, 1.05, '(b)', transform=ax1.transAxes)
        if i == 1:
            roll_x = -4
            roll_y = -21
            ax1 = fig.add_subplot(gs[1, i+1])
            ax0.annotate(text=r'$\eta = $' + f'{eta_mult}'+r'$L$',
                     xy=(20, 0.3),
                     xytext=(12., 0.1),
                     arrowprops=dict(arrowstyle='->',
                                     color=colors[i],
                                     lw=1,
                                     ls='-'),
                     color=colors[i]
                     )
            ax0.text(letter_offset, 1.05, '(c)', transform=ax1.transAxes)

        elif i == 2:
            roll_x = -3
            roll_y = -23
            ax1 = fig.add_subplot(gs[1, i+1])
            ax0.annotate(text=r'$\eta = $' + f'{eta_mult}'+r'$L$',
                     xy=(18, 0.85),
                     xytext=(11., 0.4),
                     arrowprops=dict(arrowstyle='->',
                                     color=colors[i],
                                     lw=1,
                                     ls='-'),
                     color=colors[i]
                     )
            ax0.text(letter_offset, 1.05, '(d)', transform=ax1.transAxes)

        elif i == 3:
            roll_x = 1
            roll_y = -25
            ax1 = fig.add_subplot(gs[1, i+1])
            ax0.annotate(text=r'$\eta = $' +f'{eta_mult}'+r'$L$',
                         xy=(15, 0.85),
                         xytext=(5., 0.5),
                         arrowprops=dict(arrowstyle='->',
                                         color= colors[i],
                                         lw=1,
                                         ls='-'),
                         color=colors[i]
                         )
            ax0.text(letter_offset, 1.05, '(e)', transform=ax1.transAxes)

        # phase_field=phase_field**2
        phase_field = np.roll(phase_field, roll_x, axis=1)
        phase_field = np.roll(phase_field, roll_y, axis=0)
        x = np.arange(0, nb_reps*N)
        y = np.arange(0, nb_reps*N)
        X, Y = np.meshgrid(x, y)
        levels = [-0.1,0.2,0.3,0.4,0.5,0.6, 0.7, 0.8,0.9,1.1] #levels,
        #levels =np.arange(-0.1, 1.1,1.2/64)
        #contour = ax1.contourf(np.tile(phase_field, (nb_reps, nb_reps)),levels, cmap='gray_r', vmin=0, vmax=1)
        contour = ax1.pcolormesh(X,Y,np.tile(phase_field, (nb_reps, nb_reps)),  cmap='gray_r', vmin=0, vmax=1,linewidth=0,rasterized=True)
        contour.set_edgecolor('face')

        ax1.plot([0, 1], [0, 1], color=colors[i], linestyle=linestyles[i], linewidth=1., transform=ax1.transAxes)

        if i == 0:
            # Colorbar
            #divider = make_axes_locatable(ax1)
            # Prevent shrinking the original image
            #ax1.set_anchor('SE')
            #ax_cb = divider.new_horizontal(size="5%", pad=-0.5)
            #ax_cb = divider.append_axes("left", size="5%", pad=0.5)
            #fig.add_axes(ax_cb)
            pos0 = ax0.get_position()
            pos1 = ax1.get_position()

            ax_cb=fig.add_axes([pos0.x0, pos1.y0, 0.02, pos1.height])
                               #ax_cb.invert_xaxis()  # Invert the x-axis to match the positioning
            ax_cb.yaxis.tick_left()
            # cbar = fig.colorbar(contour, cax=ax_cb)

            cbar = fig.colorbar(mpl.cm.ScalarMappable(norm=contour.norm, cmap=contour.cmap), cax=ax_cb,
                                ticks=np.arange(0, 1.2, 0.25))
            ax_cb.set_ylabel(r'Phase $\rho$', rotation=90, labelpad=-65)
            ax_cb.yaxis.set_ticks_position('left')
             #ax_cb.xaxis.set_label_position('left')
        ax1.set_xlabel(r'$\eta = $' +f'{eta_mult}'+r'$L$')

        ax1.set_xlim(0, N - 1)
        ax1.set_ylim(0, N - 1)
        ax1.set_yticklabels([])
        ax1.set_xticklabels([])
        ax1.xaxis.set_ticks_position('none')
        ax1.yaxis.set_ticks_position('none')
        ax1.set_aspect('equal')
        #ax1.axis('off')

        #ax1.hlines(y=32, xmin=0, xmax=N, colors='black', linestyles='--', linewidth=1.)
        #ax1.vlines(x=32, ymin=0, ymax=N, colors='black', linestyles='--', linewidth=1.)

        ax0.plot(np.diag(phase_field),color=colors[i], linestyle=linestyles[i])
    #ax0.grid(axis='x')
    number_of_runs = range(1, N)  # use your actual number_of_runs
    ax0.set_xticks(number_of_runs, minor=False)
    ax0.xaxis.grid(True,color='grey', linestyle='-', linewidth=0.01)
    ax0.set_xticklabels([])
    ax0.xaxis.set_ticks_position('none')
    #ax0.yaxis.set_ticks_position('none')
    ax0.set_ylabel(r'Phase $\rho$', rotation=90, labelpad=10)
    ax0.set_xlim(0, N - 1)
    ax0.text(-0.05, 1.1, '(a)', transform=ax0.transAxes)
    #ax0.set_ylim(0, N - 1)
        #ax0.set_aspect('equal')
    #ax1.set_ylabel(r'Position y')
    ax1.yaxis.set_label_position("right")

    fname = src + 'exp1_rectangles{}'.format('.pdf')
    print(('create figure: {}'.format(fname)))
    plt.savefig(fname, bbox_inches='tight')
    # plt.show()
    plt.show()

##########
    fig = plt.figure(figsize=(11,4.5))
    gs = fig.add_gridspec(3, 4)
    ax5 = fig.add_subplot(gs[:, :])

    # fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    ax5.loglog(etas, f_sigmas, '-', color='r', linewidth=2, label=r'stress difference -  $f_{\sigma}$')
    ax5.loglog(etas, f_pfs, '--', color='k', linewidth=1, label=r'phase field - $f_{\rho}$')

    ax1 = fig.add_axes([0.09, 0.4, 0.25, 0.25])
    ax1.plot(np.diag(np.diag(phase_field)))

    #ax5.legend([r'stress difference -  $f_{\sigma}$', r'phase field - $f_{\rho}$'], loc='lower center')
    # ax.set_aspect('equal')
    #ax5.set_xlabel(r'Weight $w$')
    #ax5.set_xlim(0.1, 100)
    #ax5.set_ylim(1e-4, 1e1)
    # ax5.annotate(r'Stress difference -  $f_{\sigma}$', color='red',
    #                     xy=(0.6, f_sigmas[np.where(weights == 0.6)[0][0]]),
    #                     xytext=(1.0, 5.),
    #                     arrowprops=dict(arrowstyle='->',
    #                                     color='red',
    #                                     lw=1,
    #                                     ls='-')
    #                     )
    # ax5.annotate(r'Phase field - $f_{\rho}$',
    #              xy=(50., f_pfs[np.where(weights == 50.0)[0][0]]),
    #              xytext=(20., 5.),
    #              arrowprops=dict(arrowstyle='->',
    #                              color='black',
    #                              lw=1,
    #                              ls='-')
    #              )

    letter_offset=0


    # divider = make_axes_locatable(ax1)
    # ax_cb = divider.new_horizontal(size="5%", pad=0.05)
    # fig.add_axes(ax_cb)
    # cbar = fig.colorbar(mpl.cm.ScalarMappable(norm=contour.norm, cmap=contour.cmap), cax=ax_cb,
    #                     ticks=np.arange(0, 1.2, 0.2))
    # cbar.ax.set_ylabel(r'Phase $\rho$', rotation=90, labelpad=10)


    fname = src +  'exp2_eta_depen{}'.format('.pdf')
    print(('create figure: {}'.format(fname)))
    #plt.savefig(fname, bbox_inches='tight')
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
