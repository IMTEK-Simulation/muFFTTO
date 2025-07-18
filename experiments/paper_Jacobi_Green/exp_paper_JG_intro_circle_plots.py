import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt


# Enable LaTeX rendering
plt.rcParams.update({
    "text.usetex": True,  # Use LaTeX
    # "font.family": "helvetica",  # Use a serif font
})
plt.rcParams.update({'font.size': 11})
plt.rcParams["font.family"] = "Arial"

script_name_save = 'exp_paper_JG_intro_circle'
file_folder_path = os.path.dirname(os.path.realpath(__file__))  # script directory
data_folder_path = file_folder_path + '/exp_data/' + script_name_save + '/'
figures_folder_path = file_folder_path + '/figures/' + script_name_save + '/'

problem_type = 'elasticity'
discretization_type = 'finite_element'
element_type = 'linear_triangles'
formulation = 'small_strain'

plot_v1=False
plot_v2=False
plot_v3=True



if plot_v3:
    contrast = 1e-4
    nb_of_filters = 3801
    rhos=[0,600,3800,nb_of_filters-1]
    rhos_to_print=[0,'I','II',3]

    filter_ids = np.arange(nb_of_filters)
    number_of_pixels = (256, 256)

    # for filer_id in filter_ids:
    filer_id = filter_ids[-1]
    # material distribution
    geometry_ID = 'circle_inclusion'

    file_data_name = (
        f'{script_name_save}_gID{geometry_ID}_T{number_of_pixels[0]}_F{filer_id}_kappa{contrast}.npy')

    # file_data_name = (
    #     f'{script_name_save}_gID{geometry_ID}_T{T}_G{G}_kappa{ratio}.npy')
    #
    folder_name = '../exp_data/'

    _info = np.load('../exp_data/' + file_data_name + f'xopt_log.npz', allow_pickle=True)
    phase_field = np.load('../exp_data/' + file_data_name + f'.npy', allow_pickle=True)

    nb_of_pixels_global = _info['nb_of_pixels']
    nb_of_filters_aplication = _info['nb_of_filters_aplication']

    norm_rMr_G = _info['norm_rMr_G']
    norm_rMr_J = _info['norm_rMr_J']
    norm_rMr_JG = _info['norm_rMr_JG']

    nb_it = _info['nb_it_G']
    nb_it_Jacobi = _info['nb_it_J']
    nb_it_combi = _info['nb_it_JG']

    colors = ['red', 'blue', 'green', 'orange', 'purple','olive','brown','purple']
    linestyles = [':', '-.', '--', (0, (3, 1, 1, 1))]

    x = np.arange(0, number_of_pixels[0])
    y = np.arange(0, number_of_pixels[1])
    X, Y = np.meshgrid(x, y)


    # create a figure Version 1 with combineed intersection with 2D plots
    fig = plt.figure(figsize=(8.3, 5.0))
    gs = fig.add_gridspec(4, 3, hspace=0.2, wspace=0.1, width_ratios=[1, 1, 1],
                          height_ratios=[0.2, 1, 1, 1])

    ax_global = fig.add_subplot(gs[:, :])

    ax_global.plot(filter_ids, nb_it[0][:nb_of_filters], 'g', linestyle=(0, (10, 3)) ,markevery=1, label='Green', linewidth=3)
  #  ax_global.plot(filter_ids, nb_it_Jacobi[0][:nb_of_filters], 'blue', linestyle='-.',markevery=5, label='Jacobi', linewidth=2)
    ax_global.plot(filter_ids, nb_it_combi[0][:nb_of_filters], 'black', linestyle='-',markevery=1, label='Green-Jacobi ', linewidth=3)

    ax_global.set_ylim([1, 100])
    ax_global.set_xlim([-50, nb_of_filters])
    ax_global.set_xticks([0,rhos[1],rhos[2]])
    ax_global.set_xticklabels([0,'I','II'])
    #
    ax_global.set_xlabel(r'\# filter applications - $i$')
    ax_global.set_ylabel(r'\# preconditioned CG iterations')
    ax_global.annotate(text=f'Green',#\n contrast = 100
                      xy=(filter_ids[1000], nb_it[0][:nb_of_filters][1000]),
                      xytext=(1000., 80.),
                      arrowprops=dict(arrowstyle='->',
                                      color='Green',
                                      lw=1,
                                      ls='--'),
                     fontsize=14,
                      color='Green'
                      )
    # ax_global.annotate(text=f'Jacobi',#\n contrast = 100
    #                   xy=(filter_ids[60], nb_it_Jacobi[0][:nb_of_filters][60]),
    #                   xytext=(65., 40.),
    #                   arrowprops=dict(arrowstyle='->',
    #                                   color='Blue',
    #                                   lw=1,
    #                                   ls='-.'),
    #                  fontsize=14,
    #                   color='Blue'
    #                   )
    ax_global.annotate(text=f'Green-Jacobi',#\n contrast = 100
                      xy=(filter_ids[2750], nb_it_combi[0][:nb_of_filters][2750]),
                      xytext=(2750., 15.),
                      arrowprops=dict(arrowstyle='->',
                                      color='Black',
                                      lw=1,
                                      ls='-'),
                     fontsize=14,
                      color='Black',
                      )

    # axis for cross sections
    ax_cross = fig.add_axes([0.56, 0.45, 0.4, 0.4])
    #ax_cross.set_title(r' $n_\mathrm{f}=0 $', y=-0.2)
    ax_cross.text(-0.45, 1.0, rf'\textbf{{(b)}}', transform=ax_cross.transAxes)



    for geom_ax in np.arange(3):
        # for filer_id in filter_ids:
        if geom_ax == 0:
            filer_id = filter_ids[rhos[geom_ax]]
            ax_geom_1 = fig.add_axes([0.145, 0.3, 0.2, 0.2])
            ax_geom_1.set_title(r' $\rho_0 $')
            ax_geom_1.text(-0.1, 1.1, rf'\textbf{{(a.1)}}', transform=ax_geom_1.transAxes)
            ax_cross.annotate(text=fr'$\rho_{filer_id}$',  # \n contrast = 100
                              xy=(nb_of_pixels_global[0]//3-2, 1e-3),
                              xytext=(nb_of_pixels_global[0]//10, 1e-3),
                              arrowprops=dict(arrowstyle='->',
                                              color=colors[-geom_ax],
                                              lw=1,
                                              ls=linestyles[geom_ax]),
                              fontsize=12,
                              color=colors[-geom_ax]
                              )
        if geom_ax == 1:
            filer_id = filter_ids[rhos[geom_ax]]
            ax_geom_1 = fig.add_axes([0.29, 0.22, 0.2, 0.2])
            ax_geom_1.set_title(fr' $\rho_{{\mathrm{{{ rhos_to_print[geom_ax]}}}}} $')
            ax_geom_1.text(-0.1, 1.1, rf'\textbf{{(a.2)}}', transform=ax_geom_1.transAxes)
            ax_cross.annotate(text=fr'$\rho_{{\mathrm{{{ rhos_to_print[geom_ax]}}}}}$',  # \n contrast = 100
                              xy=(nb_of_pixels_global[0]//2,  1e-4),
                              xytext=(nb_of_pixels_global[0]//2-7, 5e-4),
                              arrowprops=dict(arrowstyle='->',
                                              color=colors[-geom_ax],
                                              lw=1,
                                              ls=linestyles[geom_ax]),
                              fontsize=12,
                              color=colors[-geom_ax]
                              )
        if geom_ax == 2:
            filer_id = filter_ids[rhos[geom_ax]]
            ax_geom_1 = fig.add_axes([0.445, 0.19, 0.2, 0.2])#
            ax_geom_1.set_title(fr' $\rho_{{\mathrm{{{ rhos_to_print[geom_ax]}}}}}$')
            ax_geom_1.text(-0.1, 1.1,rf'\textbf{{(a.3)}}', transform=ax_geom_1.transAxes)
            ax_cross.annotate(text=fr'$\rho_{{\mathrm{{{ rhos_to_print[geom_ax]}}}}}$',  # \n contrast = 100
                              xy=(nb_of_pixels_global[0]//2, 9e-3),
                              xytext=(nb_of_pixels_global[0]//2-10, 1.e-1),
                              arrowprops=dict(arrowstyle='->',
                                              color=colors[-geom_ax],
                                              lw=1,
                                              ls=linestyles[geom_ax]),
                              fontsize=12,
                              color=colors[-geom_ax]
                              )
        if geom_ax == 3:
            filer_id = filter_ids[rhos[geom_ax]]
            ax_geom_1 = fig.add_axes([0.72, 0.29, 0.2, 0.2])
            # ax_geom_1.set_title(r'$\phi_{i}\, (\lambda_{i}=34) $ ')
            ax_geom_1.set_title(fr'$\rho_{{{rhos_to_print[geom_ax]}}}$')
            ax_geom_1.text(-0.2, 1.1,'(a.4)', transform=ax_geom_1.transAxes)
            ax_cross.annotate(text=fr'$\rho_{{{rhos_to_print[geom_ax]}}}$',  # \n contrast = 100
                              xy=(nb_of_pixels_global[0]//2, 4e-2),
                              xytext=(nb_of_pixels_global[0]//2-nb_of_pixels_global[0]//12, 3e-1),
                              arrowprops=dict(arrowstyle='->',
                                              color=colors[-geom_ax],
                                              lw=1,
                                              ls=linestyles[geom_ax]),
                              fontsize=12,
                              color=colors[-geom_ax]
                              )
        file_data_name = (
            f'{script_name_save}_gID{geometry_ID}_T{number_of_pixels[0]}_F{filer_id}_kappa{contrast}.npy')

        phase_field = np.load('../exp_data/' + file_data_name + f'.npy', allow_pickle=True)





        pcm = ax_geom_1.pcolormesh(X, Y, np.transpose(phase_field), cmap=mpl.cm.Greys, vmin=contrast, vmax=1, linewidth=0,
                                 rasterized=True)
        ax_geom_1.set_aspect('equal')

        ax_geom_1.set_xticks([ ])
        ax_geom_1.set_xticklabels([ ])
        ax_geom_1.set_yticks([ ])
        ax_geom_1.set_yticklabels([ ])
        ax_geom_1.set_xlim([0,number_of_pixels[0]-1 ])
        ax_geom_1.set_ylim([0,number_of_pixels[1]-1 ])
        ax_geom_1.set_box_aspect(1)  # Maintain square aspect ratio
        ax_geom_1.hlines(y=number_of_pixels[1]//2, xmin=-0.5, xmax=number_of_pixels[0] - 0.5, color=colors[-geom_ax],
                       linestyle=linestyles[geom_ax], linewidth=1.)
        # Create secondary y-axis


        ax_cross.semilogy(phase_field[:, phase_field.shape[0] // 2], linewidth=1,  color=colors[-geom_ax],
                       linestyle=linestyles[geom_ax])
        #ax_cross.set_ylabel("Y2-axis (Cos)", color='red')
        ax_cross.tick_params(axis='y', labelcolor='black')
        ax_cross.set_xticks([ ])
        ax_cross.set_xticklabels([ ])

        ax_cross.set_xlim([0,number_of_pixels[0]-1])
        ax_cross.set_ylim([9e-5,1.1])
        ax_cross.set_yticks([ 1e-4,  1e-2,1e0])
        ax_cross.set_yticklabels([r'$10^{-4}$', r'$10^{-2}$',1])
        #ax_cross.set_aspect('equal', adjustable='datalim')
        #ax2.set_aspect('equal')
        ax_cross.set_box_aspect(1)  # Maintain square aspect ratio
        ax_cross.set_ylabel(r'Density $\rho_{i}$')

   #  ax_cbar1 = fig.add_axes([ 0.22, 0.63, 0.01, 0.2])
   # # 0.16, 0.22,
   #  cbar = plt.colorbar(pcm, location='left', cax=ax_cbar1)
   #  cbar.ax.yaxis.tick_left()
   #  # cbar.set_ticks(ticks=[1e-4,1e-2, 1])
   #  # cbar.set_ticklabels([f'$10^{{{-4}}}$', f'$10^{{{-2}}}$', 1])
   #  cbar.set_ticks(ticks=[1e-8, 0.5, 1])
   #  cbar.set_ticklabels([r'$10^{-4}$', 0.5, 1])
   #  ax_cbar1.set_ylabel(r'Density $\rho$')

    ax_cbar = fig.add_axes([0.15, 0.65, 0.01, 0.2])
    # 0.16, 0.22,
    cbar = plt.colorbar(pcm, location='right', cax=ax_cbar)
    cbar.ax.yaxis.tick_right()
    # cbar.set_ticks(ticks=[1e-4,1e-2, 1])
    # cbar.set_ticklabels([f'$10^{{{-4}}}$', f'$10^{{{-2}}}$', 1])
    cbar.set_ticks(ticks=[1e-8, 0.5, 1])
    cbar.set_ticklabels([r'$10^{-4}$', 0.5, 1])
    ax_cbar.set_ylabel(r'Density $\rho_{i}$')

    fig.tight_layout()
    fname = script_name_save+ f'_N{number_of_pixels[0]}_v3' + '{}'.format('.pdf')
    print(('create figure: {}'.format(fname)))
    plt.savefig(figures_folder_path + fname, bbox_inches='tight')
    plt.show()


contrast = 1e-4
nb_of_filters = 110
filter_ids = np.arange(nb_of_filters)
number_of_pixels = (32, 32)

# for filer_id in filter_ids:
filer_id = filter_ids[-1]
# material distribution
geometry_ID = 'circle_inclusion'

file_data_name = (
    f'{script_name_save}_gID{geometry_ID}_T{number_of_pixels[0]}_F{filer_id}_kappa{contrast}.npy')

# file_data_name = (
#     f'{script_name_save}_gID{geometry_ID}_T{T}_G{G}_kappa{ratio}.npy')
#
folder_name = '../exp_data/'

_info = np.load('../exp_data/' + file_data_name + f'xopt_log.npz', allow_pickle=True)
phase_field = np.load('../exp_data/' + file_data_name + f'.npy', allow_pickle=True)

nb_of_pixels_global = _info['nb_of_pixels']
nb_of_filters_aplication = _info['nb_of_filters_aplication']

norm_rMr_G = _info['norm_rMr_G']
norm_rMr_J = _info['norm_rMr_J']
norm_rMr_JG = _info['norm_rMr_JG']

nb_it = _info['nb_it_G']
nb_it_Jacobi = _info['nb_it_J']
nb_it_combi = _info['nb_it_JG']

colors = ['red', 'blue', 'green', 'orange', 'purple','olive','brown','purple']
linestyles = [':', '-.', '--', (0, (3, 1, 1, 1))]

x = np.arange(0, number_of_pixels[0])
y = np.arange(0, number_of_pixels[1])
X, Y = np.meshgrid(x, y)

if plot_v2:

    # create a figure Version 1 with combineed intersection with 2D plots
    fig = plt.figure(figsize=(8.3, 5.0))
    gs = fig.add_gridspec(4, 3, hspace=0.2, wspace=0.1, width_ratios=[1, 1, 1],
                          height_ratios=[0.2, 1, 1, 1])

    ax_global = fig.add_subplot(gs[:, :])

    ax_global.plot(filter_ids, nb_it[0][:nb_of_filters], 'g', linestyle='--',markevery=5, label='Green', linewidth=3)
    ax_global.plot(filter_ids, nb_it_Jacobi[0][:nb_of_filters], 'blue', linestyle='-.',markevery=5, label='Jacobi', linewidth=2)
    ax_global.plot(filter_ids, nb_it_combi[0][:nb_of_filters], 'black', linestyle='-',markevery=5, label='Green-Jacobi ', linewidth=3)

    ax_global.set_ylim([1, 100])
    ax_global.set_xlim([-1, 100])
    ax_global.set_xticks([0,25, 50,75 ,100])
    ax_global.set_xticklabels([0, 25, 50,75 ,100])
    #
    ax_global.set_xlabel(r'\# filter applications - $n_\mathrm{f}$')
    ax_global.set_ylabel(r'\# CG iterations')
    ax_global.annotate(text=f'Green',#\n contrast = 100
                      xy=(filter_ids[45], nb_it[0][:nb_of_filters][45]),
                      xytext=(50., 37.),
                      arrowprops=dict(arrowstyle='->',
                                      color='Green',
                                      lw=1,
                                      ls='--'),
                     fontsize=14,
                      color='Green'
                      )
    ax_global.annotate(text=f'Jacobi',#\n contrast = 100
                      xy=(filter_ids[60], nb_it_Jacobi[0][:nb_of_filters][60]),
                      xytext=(65., 40.),
                      arrowprops=dict(arrowstyle='->',
                                      color='Blue',
                                      lw=1,
                                      ls='-.'),
                     fontsize=14,
                      color='Blue'
                      )
    ax_global.annotate(text=f'Green-Jacobi',#\n contrast = 100
                      xy=(filter_ids[50], nb_it_combi[0][:nb_of_filters][50]),
                      xytext=(45., 15.),
                      arrowprops=dict(arrowstyle='->',
                                      color='Black',
                                      lw=1,
                                      ls='-'),
                     fontsize=14,
                      color='Black'
                      )

    # axis for cross sections
    ax_cross = fig.add_axes([0.65, 0.55, 0.3, 0.3])
    #ax_cross.set_title(r' $n_\mathrm{f}=0 $', y=-0.2)
    ax_cross.text(-0.45, 1.0, r'(b)  ', transform=ax_cross.transAxes)



    for geom_ax in np.arange(4):
        # for filer_id in filter_ids:
        if geom_ax == 0:
            filer_id = filter_ids[0]
            ax_geom_1 = fig.add_axes([0.16, 0.22, 0.2, 0.2])
            ax_geom_1.set_title(r' $\rho_0 $')
            ax_geom_1.text(-0.2, 1.1, r'(a.1)  ', transform=ax_geom_1.transAxes)
            ax_cross.annotate(text=r'$\rho_{0}$',  # \n contrast = 100
                              xy=(10, 1e-3),
                              xytext=(2, 1e-3),
                              arrowprops=dict(arrowstyle='->',
                                              color=colors[-geom_ax],
                                              lw=1,
                                              ls=linestyles[geom_ax]),
                              fontsize=12,
                              color=colors[-geom_ax]
                              )
        if geom_ax == 1:
            filer_id = filter_ids[10]
            ax_geom_1 = fig.add_axes([0.25, 0.61, 0.2, 0.2])
            ax_geom_1.set_title(r' $\rho_{10} $')
            ax_geom_1.text(-0.2, 1.1, r'(a.2)  ', transform=ax_geom_1.transAxes)
            ax_cross.annotate(text=r'$\rho_{10}$',  # \n contrast = 100
                              xy=(16, 1e-4),
                              xytext=(14, 5e-4),
                              arrowprops=dict(arrowstyle='->',
                                              color=colors[-geom_ax],
                                              lw=1,
                                              ls=linestyles[geom_ax]),
                              fontsize=12,
                              color=colors[-geom_ax]
                              )
        if geom_ax == 2:
            filer_id = filter_ids[50]
            ax_geom_1 = fig.add_axes([0.43, 0.61, 0.2, 0.2])
            ax_geom_1.set_title(r' $\rho_{50}$')
            ax_geom_1.text(-0.2, 1.1,'(a.3)', transform=ax_geom_1.transAxes)
            ax_cross.annotate(text=r'$\rho_{50}$',  # \n contrast = 100
                              xy=(16, 5e-3),
                              xytext=(14, 1.9e-2),
                              arrowprops=dict(arrowstyle='->',
                                              color=colors[-geom_ax],
                                              lw=1,
                                              ls=linestyles[geom_ax]),
                              fontsize=12,
                              color=colors[-geom_ax]
                              )
        if geom_ax == 3:
            filer_id = filter_ids[100]
            ax_geom_1 = fig.add_axes([0.72, 0.24, 0.2, 0.2])
            # ax_geom_1.set_title(r'$\phi_{i}\, (\lambda_{i}=34) $ ')
            ax_geom_1.set_title(r'$\rho_{100}$')
            ax_geom_1.text(-0.2, 1.1,'(a.4)', transform=ax_geom_1.transAxes)
            ax_cross.annotate(text=r'$\rho_{100}$',  # \n contrast = 100
                              xy=(16, 4e-2),
                              xytext=(13, 3e-1),
                              arrowprops=dict(arrowstyle='->',
                                              color=colors[-geom_ax],
                                              lw=1,
                                              ls=linestyles[geom_ax]),
                              fontsize=12,
                              color=colors[-geom_ax]
                              )
        file_data_name = (
            f'{script_name_save}_gID{geometry_ID}_T{number_of_pixels[0]}_F{filer_id}_kappa{contrast}.npy')

        phase_field = np.load('../exp_data/' + file_data_name + f'.npy', allow_pickle=True)





        pcm = ax_geom_1.pcolormesh(X, Y, np.transpose(phase_field), cmap=mpl.cm.Greys, vmin=contrast, vmax=1, linewidth=0,
                                 rasterized=True)
        ax_geom_1.set_aspect('equal')

        ax_geom_1.set_xticks([ ])
        ax_geom_1.set_xticklabels([ ])
        ax_geom_1.set_yticks([ ])
        ax_geom_1.set_yticklabels([ ])
        ax_geom_1.set_xlim([0,number_of_pixels[0]-1 ])
        ax_geom_1.set_ylim([0,number_of_pixels[1]-1 ])
        ax_geom_1.set_box_aspect(1)  # Maintain square aspect ratio
        ax_geom_1.hlines(y=number_of_pixels[1]//2, xmin=-0.5, xmax=number_of_pixels[0] - 0.5, color=colors[-geom_ax],
                       linestyle=linestyles[geom_ax], linewidth=1.)
        # Create secondary y-axis


        ax_cross.semilogy(phase_field[:, phase_field.shape[0] // 2], linewidth=1,  color=colors[-geom_ax],
                       linestyle=linestyles[geom_ax])
        #ax_cross.set_ylabel("Y2-axis (Cos)", color='red')
        ax_cross.tick_params(axis='y', labelcolor='black')
        ax_cross.set_xticks([ ])
        ax_cross.set_xticklabels([ ])

        ax_cross.set_xlim([0,number_of_pixels[0]-1])
        ax_cross.set_ylim([9e-5,1.1])
        ax_cross.set_yticks([ 1e-4,  1e-2,1e0])
        ax_cross.set_yticklabels([r'$10^{-4}$', r'$10^{-2}$',1])
        #ax_cross.set_aspect('equal', adjustable='datalim')
        #ax2.set_aspect('equal')
        ax_cross.set_box_aspect(1)  # Maintain square aspect ratio
        ax_cross.set_ylabel(r'Density $\rho_{n_{\mathrm{f}}}$')

   #  ax_cbar1 = fig.add_axes([ 0.22, 0.63, 0.01, 0.2])
   # # 0.16, 0.22,
   #  cbar = plt.colorbar(pcm, location='left', cax=ax_cbar1)
   #  cbar.ax.yaxis.tick_left()
   #  # cbar.set_ticks(ticks=[1e-4,1e-2, 1])
   #  # cbar.set_ticklabels([f'$10^{{{-4}}}$', f'$10^{{{-2}}}$', 1])
   #  cbar.set_ticks(ticks=[1e-8, 0.5, 1])
   #  cbar.set_ticklabels([r'$10^{-4}$', 0.5, 1])
   #  ax_cbar1.set_ylabel(r'Density $\rho$')

    ax_cbar = fig.add_axes([0.15, 0.65, 0.01, 0.2])
    # 0.16, 0.22,
    cbar = plt.colorbar(pcm, location='right', cax=ax_cbar)
    cbar.ax.yaxis.tick_right()
    # cbar.set_ticks(ticks=[1e-4,1e-2, 1])
    # cbar.set_ticklabels([f'$10^{{{-4}}}$', f'$10^{{{-2}}}$', 1])
    cbar.set_ticks(ticks=[1e-8, 0.5, 1])
    cbar.set_ticklabels([r'$10^{-4}$', 0.5, 1])
    ax_cbar.set_ylabel(r'Density $\rho_{n_{\mathrm{f}}}$')

    fig.tight_layout()
    fname = script_name_save+ f'_N{number_of_pixels[0]}_v2' + '{}'.format('.pdf')
    print(('create figure: {}'.format(fname)))
    plt.savefig(figures_folder_path + fname, bbox_inches='tight')
    plt.show()


if plot_v1:

    # create a figure Version 1 with combineed intersection with 2D plots
    fig = plt.figure(figsize=(8.3, 5.0))
    gs = fig.add_gridspec(4, 3, hspace=0.2, wspace=0.1, width_ratios=[1, 1, 1],
                          height_ratios=[0.2, 1, 1, 1])

    ax_global = fig.add_subplot(gs[:, :])

    ax_global.plot(filter_ids, nb_it[0][:nb_of_filters], 'g', linestyle='--',markevery=5, label='Green', linewidth=3)
    ax_global.plot(filter_ids, nb_it_Jacobi[0][:nb_of_filters], 'blue', linestyle='-.',markevery=5, label='Jacobi', linewidth=2)
    ax_global.plot(filter_ids, nb_it_combi[0][:nb_of_filters], 'black', linestyle='-',markevery=5, label='Green-Jacobi ', linewidth=3)

    ax_global.set_ylim([1, 100])
    ax_global.set_xlim([-1, 100])
    ax_global.set_xticks([0,25, 50,75 ,100])
    ax_global.set_xticklabels([0, 25, 50,75 ,100])
    #
    ax_global.set_xlabel(r'\# filter applications - $n_\mathrm{f}$')
    ax_global.set_ylabel(r'\# CG iterations')
    ax_global.annotate(text=f'Green',#\n contrast = 100
                      xy=(filter_ids[45], nb_it[0][:nb_of_filters][45]),
                      xytext=(50., 37.),
                      arrowprops=dict(arrowstyle='->',
                                      color='Green',
                                      lw=1,
                                      ls='--'),
                     fontsize=14,
                      color='Green'
                      )
    ax_global.annotate(text=f'Jacobi',#\n contrast = 100
                      xy=(filter_ids[60], nb_it_Jacobi[0][:nb_of_filters][60]),
                      xytext=(65., 40.),
                      arrowprops=dict(arrowstyle='->',
                                      color='Blue',
                                      lw=1,
                                      ls='-.'),
                     fontsize=14,
                      color='Blue'
                      )
    ax_global.annotate(text=f'Green-Jacobi',#\n contrast = 100
                      xy=(filter_ids[40], nb_it_combi[0][:nb_of_filters][40]),
                      xytext=(35., 15.),
                      arrowprops=dict(arrowstyle='->',
                                      color='Black',
                                      lw=1,
                                      ls='-'),
                     fontsize=14,
                      color='Black'
                      )

    for geom_ax in np.arange(4):
        # for filer_id in filter_ids:


        if geom_ax == 0:
            filer_id = filter_ids[0]
            ax_geom_1 = fig.add_axes([0.16, 0.22, 0.2, 0.2])
            ax_geom_1.set_title(r' $n_\mathrm{f}=0 $')
            ax_geom_1.text(-0.2, 1.1, r'(a)  ', transform=ax_geom_1.transAxes)

        if geom_ax == 1:
            filer_id = filter_ids[10]
            ax_geom_1 = fig.add_axes([0.25, 0.60, 0.2, 0.2])
            ax_geom_1.set_title(r' $n_\mathrm{f}=10$')
            ax_geom_1.text(-0.2, 1.1, r'(b)  ', transform=ax_geom_1.transAxes)

        if geom_ax == 2:
            filer_id = filter_ids[50]
            ax_geom_1 = fig.add_axes([0.47, 0.6, 0.2, 0.2])
            ax_geom_1.set_title(r' $n_\mathrm{f}=50$')
            ax_geom_1.text(-0.2, 1.1,'(c)', transform=ax_geom_1.transAxes)
        if geom_ax == 3:
            filer_id = filter_ids[99]
            ax_geom_1 = fig.add_axes([0.68, 0.60, 0.2, 0.2])
            # ax_geom_1.set_title(r'$\phi_{i}\, (\lambda_{i}=34) $ ')
            ax_geom_1.set_title(r' $n_\mathrm{f}=100$')
            ax_geom_1.text(-0.2, 1.1,'(d)', transform=ax_geom_1.transAxes)

        file_data_name = (
            f'{script_name_save}_gID{geometry_ID}_T{number_of_pixels[0]}_F{filer_id}_kappa{contrast}.npy')

        phase_field = np.load('../exp_data/' + file_data_name + f'.npy', allow_pickle=True)





        pcm = ax_geom_1.pcolormesh(X, Y, np.transpose(phase_field), cmap=mpl.cm.Greys, vmin=contrast, vmax=1, linewidth=0,
                                alpha=0.7, rasterized=True)
        ax_geom_1.set_aspect('equal')

        ax_geom_1.set_xticks([ ])
        ax_geom_1.set_xticklabels([ ])
        ax_geom_1.set_yticks([ ])
        ax_geom_1.set_yticklabels([ ])
        ax_geom_1.set_xlim([0,number_of_pixels[0]-1 ])
        ax_geom_1.set_ylim([0,number_of_pixels[1]-1 ])
        ax_geom_1.set_box_aspect(1)  # Maintain square aspect ratio

        # Create secondary y-axis
        ax_cross = ax_geom_1.twinx()
        ax_cross.semilogy(phase_field[:, phase_field.shape[0] // 2], linewidth=1,  color='red')
        #ax_cross.set_ylabel("Y2-axis (Cos)", color='red')
        ax_cross.tick_params(axis='y', labelcolor='red')
        ax_cross.set_xticks([ ])
        ax_cross.set_xticklabels([ ])

        ax_cross.set_ylim([9e-5,1.1])
        ax_cross.set_yticks([ 1e-4,  1e-2,1e0])
        ax_cross.set_yticklabels([r'$10^{-4}$', r'$10^{-2}$',1])
        #ax_cross.set_aspect('equal', adjustable='datalim')
        #ax2.set_aspect('equal')
        ax_cross.set_box_aspect(1)  # Maintain square aspect ratio

    ax_cbar = fig.add_axes([0.15, 0.65, 0.01, 0.2])

    cbar = plt.colorbar(pcm, location='left', cax=ax_cbar)
    cbar.ax.yaxis.tick_right()
    # cbar.set_ticks(ticks=[1e-4,1e-2, 1])
    # cbar.set_ticklabels([f'$10^{{{-4}}}$', f'$10^{{{-2}}}$', 1])
    cbar.set_ticks(ticks=[1e-8, 0.5, 1])
    cbar.set_ticklabels([r'$10^{-4}$', 0.5, 1])


    fig.tight_layout()
    fname = script_name_save+ f'_N{number_of_pixels[0]}_v1' + '{}'.format('.pdf')
    print(('create figure: {}'.format(fname)))
    plt.savefig(figures_folder_path + fname, bbox_inches='tight')
    plt.show()
