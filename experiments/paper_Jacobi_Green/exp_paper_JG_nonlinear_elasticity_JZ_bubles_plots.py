import numpy as np
import scipy as sc
import time
import os
from NuMPI.IO import save_npy, load_npy
from mpi4py import MPI

import matplotlib as mpl
from matplotlib import pyplot as plt

plt.rcParams.update({
    "text.usetex": True,  # Use LaTeX
    # "font.family": "helvetica",  # Use a serif font
})
plt.rcParams.update({'font.size': 11})
plt.rcParams["font.family"] = "Arial"

script_name = 'exp_paper_JG_nonlinear_elasticity_JZ_bubles'  # exp_paper_JG_nonlinear_elasticity_JZ
folder_name = '../exp_data/'
file_folder_path = os.path.dirname(os.path.realpath(__file__))  # script directory

figure_folder_path = file_folder_path + '/figures/' + script_name + '/'

plot_time_vs_dofs = False
plot_stress_field = False
plot_data_vs_CG = True
plot_data_vs_CG_3D= False
plot_3D_geometry = False

plot_iterations_vs_grids_size = True
if plot_iterations_vs_grids_size:

    # print time vs DOFS
    its_G = []
    its_GJ = []
    stress_diff_norm = []
    strain_fluc_norm = []
    strain_total_norm = []
    diff_rhs_norm = []
    norm_rhs_G = []
    norm_rhs_GJ = []
    rhs_inf_G = []
    rhs_inf_GJ = []
    norm_newrton_stop_G = []
    norm_newrton_stop_GJ = []


    it_max = 9
    n_exponents = np.array([3])
    iterations = np.arange(it_max)  # numbers of grids points

    grid_sizes= np.array( [ 32, 64, 128 , 256])#,200,128,200
    #grid_sizes= np.array( [ 50, 100, 150 ,200])#,200,128,200

    its_G = np.zeros([len(grid_sizes),it_max, len(n_exponents)])
    its_GJ = np.zeros([len(grid_sizes),it_max, len(n_exponents)])

    norm_newton_stop_G= np.zeros([len(grid_sizes),it_max,  len(n_exponents)])
    norm_newton_stop_GJ = np.zeros([len(grid_sizes),it_max, len(n_exponents)])

    unique_components = np.zeros([len(grid_sizes), it_max, len(n_exponents)])
    total_contrast = np.zeros([len(grid_sizes), it_max, len(n_exponents)])


    for i, n in enumerate(grid_sizes):
        print(i, n)

        Nx = n
        Ny = Nx
        Nz = Nx

        for j in np.arange(len(n_exponents)):
            n_exp = n_exponents[j]
            for iteration_total in iterations:

                preconditioner_type = 'Green'

                data_folder_path = (
                        file_folder_path + '/exp_data/' + script_name + '/' + f'Nx={Nx}' + f'Ny={Ny}' + f'Nz={Nz}'
                        + f'_{preconditioner_type}' + '/')
                if iteration_total < it_max:
                    # if Nx == 256:
                    #     _info_final_G = np.load(data_folder_path + f'info_log_it{iteration_total}.npz', allow_pickle=True)
                    #
                    # else:
                    _info_final_G = np.load(data_folder_path + f'info_log_exp_{n_exp}_it{iteration_total}.npz',
                                                allow_pickle=True)


                results_name = (f'K4_ijklqyz' + f'_exp_{n_exp}_it{iteration_total}')

                K4_xyz_G = np.load(data_folder_path + results_name + f'.npy', allow_pickle=True, mmap_mode='r')

                unique_components[i,iteration_total, j]  = np.unique(K4_xyz_G).size
                total_contrast[i,iteration_total, j]  = np.max(K4_xyz_G)/np.min(K4_xyz_G)
                #counts = np.bincount(K4_xyz_G.flatten())
                # hist_norm, bins = np.histogram(K4_xyz_G, density=True)
                #
                # plt.plot()
                # #plt.hist(K4_xyz_G.flatten(), bins='auto')
                # #plt.hist(K4_xyz_G.flatten(), bins='auto', density=True)
                # plt.title(f"Histogram of K4_ijklqyz -{n}, {n_exp}, {iteration_total} ")
                # plt.xlabel("Value")
                # plt.ylabel("Frequency")
                # plt.show()


                its_G[i,iteration_total, j] = _info_final_G.f.nb_it_comb
                norm_rhs_G.append(_info_final_G.f.norm_rhs_field)
                norm_newrton_stop_G.append(_info_final_G.f.newton_stop_crit)
                norm_newton_stop_G[i,iteration_total, j]=_info_final_G.f.newton_stop_crit
              #  info_log_final_G = np.load(data_folder_path + f'info_log_final_exp_{n_exp}.npz', allow_pickle=True)

                preconditioner_type = 'Green_Jacobi'

                data_folder_path = (
                        file_folder_path + '/exp_data/' + script_name + '/' + f'Nx={Nx}' + f'Ny={Ny}' + f'Nz={Nz}'
                        + f'_{preconditioner_type}' + '/')
                if iteration_total < it_max:
                    # if Nx == 256:
                    #     _info_final_GJ = np.load(data_folder_path + f'info_log_it{iteration_total}.npz',
                    #                              allow_pickle=True)
                    #
                    # else:
                    _info_final_GJ = np.load(data_folder_path + f'info_log_exp_{n_exp}_it{iteration_total}.npz',
                                                 allow_pickle=True)
                # stress_GJ = np.load(data_folder_path + f'stress' + f'_it{iteration_total}' + f'.npy', allow_pickle=True)
                # strain_fluc_GJ = np.load(data_folder_path + f'strain_fluc_field' + f'_it{iteration_total}' + f'.npy',
                #                          allow_pickle=True)
                # strain_total_GJ = np.load(data_folder_path + f'total_strain_field' + f'_it{iteration_total}' + f'.npy',
                #                           allow_pickle=True)
                # rhs_field_GJ = np.load(data_folder_path + f'rhs_field' + f'_it{iteration_total}' + f'.npy', allow_pickle=True)

                its_GJ[i,iteration_total, j] = _info_final_GJ.f.nb_it_comb
                norm_rhs_GJ.append(_info_final_GJ.f.norm_rhs_field)
                norm_newrton_stop_GJ.append(_info_final_GJ.f.newton_stop_crit)
                norm_newton_stop_GJ[i,iteration_total, j]=_info_final_GJ.f.newton_stop_crit

               # info_log_final_GJ = np.load(data_folder_path + f'info_log_final_exp_{n_exp}.npz', allow_pickle=True)
# plot convergence lines
                # plt.figure(figsize=(10, 4.50))
                # plt.semilogy(_info_final_G.f.norm_rr/_info_final_G.f.norm_rr[0], color='green', linestyle='-', label=f'rr Green-Jacobi - {n}')
                # plt.semilogy(_info_final_G.f.norm_rz/_info_final_G.f.norm_rz[0], color='green', linestyle='--', label=f'rz Green-Jacobi - {n}')
                #
                # plt.semilogy(_info_final_GJ.f.norm_rr/_info_final_GJ.f.norm_rr[0],color='black',linestyle='-', label=f'rr Green-Jacobi - {n}')
                # plt.semilogy(_info_final_GJ.f.norm_rz/_info_final_GJ.f.norm_rz[0],color='black', linestyle='--',label=f'rz Green-Jacobi - {n}')
                # plt.legend()
                # plt.ylim([1e-11, 10])
                # plt.xlim([0, 250])
                #
                # plt.show()
    fig = plt.figure(figsize=(8.3, 12.0))
    gs = fig.add_gridspec(4, 1, hspace=0.4, wspace=0.1, width_ratios=[1],
                          height_ratios=[1,1,1,1])
    gs_fnorm_vs_iteration = fig.add_subplot(gs[0, 0])
    plt.title(f' exponent = {n_exp}')
    for i, n in enumerate(grid_sizes):
        gs_fnorm_vs_iteration.semilogy(iterations, norm_newton_stop_G[i, :,0] , '-', marker='x', label=f'Green - {n}')
        gs_fnorm_vs_iteration.semilogy(iterations, norm_newton_stop_GJ[i, :,0], '--', marker='o', markerfacecolor='none', label=f'Green-Jacobi - {n}')
    gs_fnorm_vs_iteration.legend()

    gs_iter_vs_mesh_size = fig.add_subplot(gs[1, 0])
    for i, n in enumerate(grid_sizes):
        gs_iter_vs_mesh_size.plot(iterations, its_G[i, :,0], '-', marker='x', label=f'Green - {n}')
        gs_iter_vs_mesh_size.plot(iterations, its_GJ[i, :,0], '--', marker='o', markerfacecolor='none',
                                   label=f'Green-Jacobi - {n}')
    gs_iter_vs_mesh_size.legend()
    #gs_iter_vs_mesh_size.set_ylim([0,350])
    gs_iter_vs_mesh_size.set_title('Iterations vs Mesh Size')

    gs_iter_vs_unique_ = fig.add_subplot(gs[2, 0])
    for i, n in enumerate(grid_sizes):
        gs_iter_vs_unique_.semilogy(iterations, unique_components[i, :,0], '-', marker='x', label=f' {n}')
    gs_iter_vs_unique_.legend()
    gs_iter_vs_unique_.set_title('Unique vsIterations')

    gs_iter_vs_contrast = fig.add_subplot(gs[3, 0])
    for i, n in enumerate(grid_sizes):
        gs_iter_vs_contrast.semilogy(iterations, total_contrast[i, :, 0], '-', marker='x', label=f' {n}')
    gs_iter_vs_contrast.legend()
    gs_iter_vs_contrast.set_title('Contrast vs Iterations')

    fig.tight_layout()
    fname = f'fig_temp'+ f'ex{n_exp}' + '{}'.format('.pdf')
    plt.savefig(figure_folder_path + script_name + fname, bbox_inches='tight')
    print(('create figure: {}'.format(figure_folder_path + script_name + fname)))
    plt.show()

    print()



if plot_3D_geometry:
    from muFFTTO import domain

    number_of_pixels = 3*(32,)
    domain_size = [1, 1, 1]
    Nx = number_of_pixels[0]
    Ny = number_of_pixels[1]
    Nz = number_of_pixels[2]

    problem_type = 'elasticity'
    discretization_type = 'finite_element'
    element_type = 'trilinear_hexahedron'
    formulation = 'small_strain'

    my_cell = domain.PeriodicUnitCell(domain_size=domain_size,
                                      problem_type=problem_type)

    discretization = domain.Discretization(cell=my_cell,
                                           nb_of_pixels_global=number_of_pixels,
                                           discretization_type=discretization_type,
                                           element_type=element_type)

    def generate_circular_inclusions(coords, num_inclusions, radius, seed=None):
        """
        Generate a boolean array with randomly distributed spherical inclusions
        with periodic boundary conditions.

        Parameters
        ----------
        coords : ndarray, shape [3, Nx, Ny, Nz]
            Grid coordinates
        num_inclusions : int
            Number of inclusions to place
        radius : float
            Radius of each inclusion
        seed : int, optional
            Random seed for reproducibility

        Returns
        -------
        ndarray, shape [Nx, Ny, Nz]
            Boolean array where 1 indicates inclusion, 0 indicates matrix
        """
        if seed is not None:
            np.random.seed(seed)

        # Get domain bounds and size
        x_min, x_max = coords[0].min(), coords[0].max()
        y_min, y_max = coords[1].min(), coords[1].max()
        z_min, z_max = coords[2].min(), coords[2].max()

        # Domain lengths
        Lx = x_max - x_min
        Ly = y_max - y_min
        Lz = z_max - z_min

        # Initialize output array
        inclusions = np.zeros(coords.shape[1:], dtype=np.int32)

        # Generate random center positions
        centers_x = np.random.uniform(x_min, x_max, num_inclusions)
        centers_y = np.random.uniform(y_min, y_max, num_inclusions)
        centers_z = np.random.uniform(z_min, z_max, num_inclusions)

        # Mark points inside each inclusion with periodic images
        for cx, cy, cz in zip(centers_x, centers_y, centers_z):
            # Compute minimum distance considering periodic images
            dx = coords[0] - cx
            dy = coords[1] - cy
            dz = coords[2] - cz

            # Apply minimum image convention
            dx = dx - Lx * np.round(dx / Lx)
            dy = dy - Ly * np.round(dy / Ly)
            dz = dz - Lz * np.round(dz / Lz)

            distance_sq = dx ** 2 + dy ** 2 + dz ** 2
            inclusions[distance_sq <= radius ** 2] = 1

        return inclusions


    inclusions = generate_circular_inclusions(
        discretization.fft.coords,
        num_inclusions=10,
        radius=0.2,
        seed=42
    )


    def visualize_inclusions_voxels(inclusions, color='blue', edgecolor=None, figsize=(8, 8)):
        """
        Visualize 3D inclusion geometry using voxels.
        """

        import matplotlib.pyplot as plt
        import numpy as np
        from matplotlib.patches import Patch

        # Create cutaway version of the data
        cutoff_x = 2 * inclusions.shape[0] // 3 - 3
        cutoff_y = 2 * inclusions.shape[1] // 3 - 3
        cutoff_z = inclusions.shape[2] // 2

        inclusions_cut = inclusions.copy()
        inclusions_cut[:cutoff_x, : cutoff_y, cutoff_z:] = -1

        # Create boolean arrays
        voxelarray_inclusions = (inclusions_cut == 1)
        voxelarray_matrix = (inclusions_cut == 0)

        # Use string colors instead of RGBA - simpler and more reliable
        colors = np.empty(inclusions_cut.shape, dtype=object)
        colors[voxelarray_inclusions] = '#d3d3d3'  # Gray for inclusions
        colors[voxelarray_matrix] = '#1f77b4'  # Blue for matrix

        voxelarray_combined = voxelarray_inclusions | voxelarray_matrix

        # Publication-quality settings
        plt.rcParams.update({
            'font.family': 'serif',
            'font.size': 12,
            'axes.labelsize': 14,
            'figure.dpi': 300,
            'savefig.dpi': 300,
        })

        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(projection='3d')

        # ===== KEY FIX: Remove alpha parameter entirely =====
        # Don't pass alpha at all, or set shade=False to prevent any transparency effects
        ax.voxels(voxelarray_combined,
                  facecolors=colors,
                  edgecolor='none',  # <-- No edges
                  linewidth=0,  # <-- Zero line width
                  shade=False)  # shade=False prevents lighting effects that can look like transparency

        # Axis labels
        ax.set_xlabel(r'voxel index in $x_1$ direction', labelpad=10)
        ax.set_ylabel(r'voxel index in $x_2$ direction', labelpad=10)
        ax.set_zlabel(r'voxel index in $x_3$ direction', labelpad=10)

        ax.view_init(elev=25, azim=-135)
        ax.set_box_aspect([1, 1, 1])
        # Clean panes - REMOVE EVERYTHING
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False

        # Hide pane edges
        ax.xaxis.pane.set_visible(False)
        ax.yaxis.pane.set_visible(False)
        ax.zaxis.pane.set_visible(False)

        # Remove grid completely
        ax.grid(False)

        # Alternative: Hide gridlines via axinfo
        ax.xaxis._axinfo['grid']['linewidth'] = 0
        ax.yaxis._axinfo['grid']['linewidth'] = 0
        ax.zaxis._axinfo['grid']['linewidth'] = 0
        # Clean panes
        # ax.xaxis.pane.fill = False
        # ax.yaxis.pane.fill = False
        # ax.zaxis.pane.fill = False
        #
        # ax.xaxis.pane.set_edgecolor('gray')
        # ax.yaxis.pane.set_edgecolor('gray')
        # ax.zaxis.pane.set_edgecolor('gray')
        # ax.grid(True, linestyle='--', alpha=0.01)

        # Legend
        legend_elements = [
            Patch(facecolor='#d3d3d3', edgecolor='#d3d3d3', label='Inclusions'),
            Patch(facecolor='#1f77b4', edgecolor='#1f77b4', label='Matrix'),
        ]
        ax.legend(handles=legend_elements, loc='upper left')

        # Axis limits
        ax.set_xlim(0, inclusions.shape[0])
        ax.set_ylim(0, inclusions.shape[1])
        ax.set_zlim(0, inclusions.shape[2])

        # Ticks
        nx, ny, nz = inclusions.shape
        ax.set_xticks([0, nx // 2, nx])
        ax.set_yticks([0, ny // 2, ny])
        ax.set_zticks([0, nz // 2, nz])
        ax.set_xticklabels(['1', str(nx // 2), str(nx)])
        ax.set_yticklabels(['1', str(ny // 2), str(ny)])
        ax.set_zticklabels(['1', str(nz // 2), str(nz)])

        ax.set_box_aspect([
            inclusions.shape[0],
            inclusions.shape[1],
            inclusions.shape[2]
        ])

        plt.savefig('composite_cutaway.pdf', format='pdf', dpi=1200, bbox_inches='tight', pad_inches=0.4)
        plt.savefig('composite_cutaway.png', format='png', dpi=1200, bbox_inches='tight', pad_inches=0.4)
        plt.show()

        vf = inclusions.sum() / inclusions.size
        print(f"Inclusion volume fraction: {vf:.2%}")
        print(f"Matrix volume fraction: {1 - vf:.2%}")


    visualize_inclusions_voxels(inclusions)
    # Or with custom options
    #visualize_inclusions_voxels(inclusions, color='red', edgecolor=None, alpha=0.7)

if plot_time_vs_dofs:
    # print time vs DOFS
    time_G = []
    time_GJ = []
    its_G = []
    its_GJ = []
    Ns = 2 ** np.array([3, 4, 5, 6])  # 4, 5,6,7,8  # numbers of grids points
    for N in Ns:
        Nx = N
        Ny = N
        Nz = N  # N#
        preconditioner_type = 'Green'

        data_folder_path = (file_folder_path + '/exp_data/' + script_name + '/' + f'Nx={Nx}' + f'Ny={Ny}' + f'Nz={Nz}'
                            + f'_{preconditioner_type}' + '/')

        _info_final_G = np.load(data_folder_path + f'info_log_final.npz', allow_pickle=True)

        time_G.append(_info_final_G.f.elapsed_time)
        its_G.append(_info_final_G.f.sum_CG_its)

        preconditioner_type = 'Green_Jacobi'

        data_folder_path = (file_folder_path + '/exp_data/' + script_name + '/' + f'Nx={Nx}' + f'Ny={Ny}' + f'Nz={Nz}'
                            + f'_{preconditioner_type}' + '/')

        _info_final_GJ = np.load(data_folder_path + f'info_log_final.npz', allow_pickle=True)

        time_GJ.append(_info_final_GJ.f.elapsed_time)
        its_GJ.append(_info_final_GJ.f.sum_CG_its)
    time_G = np.array(time_G)
    time_GJ = np.array(time_GJ)
    its_G = np.array(its_G)
    its_GJ = np.array(its_GJ)

    fig = plt.figure(figsize=(9, 4.50))
    gs = fig.add_gridspec(1, 1, hspace=0.5, wspace=0.5, width_ratios=[1],
                          height_ratios=[1])
    nb_dofs = 3 * Ns ** 2
    line1, = plt.loglog(nb_dofs, time_G, '-x', color='Green', label='Green')
    line2, = plt.loglog(nb_dofs, time_GJ, 'k-', marker='o', markerfacecolor='none', label='Green-Jacobi')
    line3, = plt.loglog(nb_dofs, nb_dofs * np.log(nb_dofs) / (nb_dofs[0] * np.log(nb_dofs[0])) * time_G[0], ':',
                        label=r'Quasilinear - $ \mathcal{O} (N_{\mathrm{N}} \log  N_{\mathrm{N}}$)')
    # plt.loglog(nb_dofs, nb_dofs / (nb_dofs[0]) * time_G[0], '--', label='linear')

    line4, = plt.loglog(nb_dofs, time_G / its_G, 'g--x', label='Green')
    line5, = plt.loglog(nb_dofs, time_GJ / its_GJ, 'k--', marker='o', markerfacecolor='none', label='Green-Jacobi')
    # plt.loglog(nb_dofs,
    #            nb_dofs* np.log(nb_dofs) / (nb_dofs[0]  * np.log(nb_dofs)) * time_G[0] / its_G[0], ':',
    #            label='N log N')
    line6, = plt.loglog(nb_dofs, nb_dofs / (nb_dofs[0]) * time_G[0] / its_G[0], '--',
                        label=r'Linear - $\mathcal{O} (N_{\mathrm{N}})$')
    plt.loglog(np.linspace(1e1, 1e8), 1e-4 * np.linspace(1e1, 1e8), 'k-', linewidth=0.9)

    plt.xlabel(r' $\#$ of degrees of freedom (DOFs) - $d N_{\mathrm{N}}$')
    plt.ylabel('Time (s)')
    plt.gca().set_xlim([nb_dofs[0], nb_dofs[-1]])
    plt.gca().set_xticks([1e3, 1e4, 1e5, 1e6])
    plt.gca().set_ylim([1e-3, 1e4])

    # plt.gca().set_xticks(iterations)

    legend1 = plt.legend(handles=[line1, line2, line3], loc='upper left', title='Wall-clock time')

    plt.gca().add_artist(legend1)  # Add the first legend manually

    # Second legend (bottom right)
    plt.legend(handles=[line4, line5, line6], loc='lower right', title='Wall-clock time / $\#$ of PCG iteration')

    fig.tight_layout()
    fname = f'time_scaling' + '{}'.format('.pdf')
    plt.savefig(figure_folder_path + script_name + fname, bbox_inches='tight')
    print(('create figure: {}'.format(figure_folder_path + script_name + fname)))

    plt.show()

if plot_data_vs_CG_3D:
    # print time vs DOFS
    its_G = []
    its_GJ = []
    stress_diff_norm = []
    strain_fluc_norm = []
    strain_total_norm = []
    diff_rhs_norm = []
    norm_rhs_G = []
    norm_rhs_GJ = []
    rhs_inf_G = []
    rhs_inf_GJ = []
    norm_newrton_stop_G = []
    norm_newrton_stop_GJ = []

    Nx =128#3200 # 2 ** 7# 8
    Ny = Nx
    Nz = Nx
    it_max = 10
    n_exponents = np.array([5])
    iterations = np.arange(it_max)  # numbers of grids points

    its_G = np.zeros([it_max, len(n_exponents)])
    its_GJ = np.zeros([it_max, len(n_exponents)])

    for j in np.arange(len(n_exponents)):
        n_exp = n_exponents[j]
        for iteration_total in iterations:

            preconditioner_type = 'Green'

            data_folder_path = (
                    file_folder_path + '/exp_data/' + script_name + '/' + f'Nx={Nx}' + f'Ny={Ny}' + f'Nz={Nz}'
                    + f'_{preconditioner_type}' + '/')
            if iteration_total < it_max:
                if Nx == 256:
                    _info_final_G = np.load(data_folder_path + f'info_log_it{iteration_total}.npz', allow_pickle=True)

                else:
                    _info_final_G = np.load(data_folder_path + f'info_log_exp_{n_exp}_it{iteration_total}.npz',
                                            allow_pickle=True)

            # with open(data_folder_path + f'stress' + f'_exp_{n_exp}_it{iteration_total + 1}' + f'.npy', 'rb') as f:
            #     magic = f.read(6)
            #     print(f"Magic number: {magic}")

            # strain_fluc_G = np.load(data_folder_path + f'strain_fluc_field' + f'_it{iteration_total}' + f'.npy',
            #                         allow_pickle=True)
            # strain_total_G = np.load(data_folder_path + f'total_strain_field' + f'_it{iteration_total}' + f'.npy',
            #                          allow_pickle=True)
            # stress_G = np.load(data_folder_path + f'stress' + f'_it{iteration_total}' + f'.npy',
            #                    allow_pickle=True)  # , allow_pickle=True

            # rhs_field_G = np.load(data_folder_path + f'rhs_field' + f'_it{iteration_total}' + f'.npy', allow_pickle=True)

            its_G[iteration_total, j] = _info_final_G.f.nb_it_comb
            norm_rhs_G.append(_info_final_G.f.norm_rhs_field)
            norm_newrton_stop_G.append(_info_final_G.f.newton_stop_crit)
            info_log_final_G = np.load(data_folder_path + f'info_log_final_exp_{n_exp}.npz', allow_pickle=True)

            preconditioner_type = 'Green_Jacobi'

            data_folder_path = (
                    file_folder_path + '/exp_data/' + script_name + '/' + f'Nx={Nx}' + f'Ny={Ny}' + f'Nz={Nz}'
                    + f'_{preconditioner_type}' + '/')
            if iteration_total < it_max:
                if Nx == 256:
                    _info_final_GJ = np.load(data_folder_path + f'info_log_it{iteration_total}.npz',
                                             allow_pickle=True)

                else:
                    _info_final_GJ = np.load(data_folder_path + f'info_log_exp_{n_exp}_it{iteration_total}.npz',
                                             allow_pickle=True)
            # stress_GJ = np.load(data_folder_path + f'stress' + f'_it{iteration_total}' + f'.npy', allow_pickle=True)
            # strain_fluc_GJ = np.load(data_folder_path + f'strain_fluc_field' + f'_it{iteration_total}' + f'.npy',
            #                          allow_pickle=True)
            # strain_total_GJ = np.load(data_folder_path + f'total_strain_field' + f'_it{iteration_total}' + f'.npy',
            #                           allow_pickle=True)
            # rhs_field_GJ = np.load(data_folder_path + f'rhs_field' + f'_it{iteration_total}' + f'.npy', allow_pickle=True)

            its_GJ[iteration_total, j] = _info_final_GJ.f.nb_it_comb
            norm_rhs_GJ.append(_info_final_GJ.f.norm_rhs_field)
            norm_newrton_stop_GJ.append(_info_final_GJ.f.newton_stop_crit)
            # diff_stress = stress_G - stress_GJ
            # stress_diff_norm.append(
            #      np.linalg.norm(diff_stress.ravel(), ord=np.inf))  # / np.linalg.norm(stress_G))
            print(_info_final_G.f.norm_rr[0])

            # diff_strain_fluc = strain_fluc_G - strain_fluc_GJ
            # strain_fluc_norm.append(
            #     np.linalg.norm(diff_strain_fluc.ravel(), ord=np.inf))  # / _info_final_GJ.f.norm_En)
            print(_info_final_GJ.f.norm_rr[0])
            # diff_strain_total = strain_total_G - strain_total_GJ
            # strain_total_norm.append(
            #    np.linalg.norm(diff_strain_total.ravel(), ord=np.inf))  # / _info_final_GJ.f.norm_En)

            # diff_rhs = rhs_field_G - rhs_field_GJ
            # diff_rhs_norm.append(
            #    np.linalg.norm(diff_rhs.ravel()))
            # rhs_inf_G.append(
            #     np.linalg.norm(rhs_field_G.ravel(), ord=np.inf))
            # rhs_inf_GJ.append(
            #     np.linalg.norm(rhs_field_GJ.ravel(), ord=np.inf))

            info_log_final_GJ = np.load(data_folder_path + f'info_log_final_exp_{n_exp}.npz', allow_pickle=True)

    # del strain_fluc_G, strain_total_G, stress_G, rhs_field_G
    # del strain_fluc_GJ, strain_total_GJ, stress_GJ, rhs_field_GJ
    # del diff_stress

    # data = np.load('large_3d_array.npy', mmap_mode='r')
    K = 2  # _info_final_G.f.model_parameters_linear
    its_G = np.array(its_G)
    its_GJ = np.array(its_GJ)
    strain_0_norm = np.array(np.atleast_1d(_info_final_GJ.f.norm_En))
    # stress_diff_norm = np.array(stress_diff_norm)
    # strain_fluc_norm = np.array(strain_fluc_norm)
    # strain_total_norm = np.array(strain_total_norm)
    # diff_rhs_norm = np.array(diff_rhs_norm)

    norm_rhs_t_G = np.concatenate((np.array(np.atleast_1d(_info_final_G.f.rhs_t_norm)), np.array(norm_rhs_G)))
    norm_rhs_t_GJ = np.concatenate((np.array(np.atleast_1d(_info_final_GJ.f.rhs_t_norm)), np.array(norm_rhs_GJ)))

    fig = plt.figure(figsize=(8.3, 4.0))
    gs = fig.add_gridspec(2, 5, hspace=0.2, wspace=0.1, width_ratios=[1, 1, 1, 1, 0.05],
                          height_ratios=[1, 1.])

    gs_global = fig.add_subplot(gs[:, 0])

    gs_global.plot(iterations, its_G[:, 0], 'g-', marker='x', label='Green')
    gs_global.plot(iterations, its_GJ[:, 0], 'k-', marker='o', markerfacecolor='none', label='Green-Jacobi')

    # gs_global.plot(iterations, its_G[:, -2], 'g-', marker='x', label='Green')
    # gs_global.plot(iterations, its_GJ[:, -2], 'k-', marker='o', markerfacecolor='none', label='Green-Jacobi')
    #
    # gs_global.plot(iterations, its_G[:, -1], 'g-', marker='x', label='Green')
    # gs_global.plot(iterations, its_GJ[:, -1], 'k-', marker='o', markerfacecolor='none', label='Green-Jacobi')

    gs_global.set_xlabel(r'Newton iteration -  $i$')
    gs_global.set_ylabel(r'$\#$ of PCG iterations')
    # gs_global.legend(loc='best')
    gs_global.set_xlim(-0.05, iterations[-1] + .05)
    gs_global.set_ylim(0., 750)
    gs_global.set_xticks(iterations)

    gs_global.annotate(text=f'Green-Jacobi',  # \n contrast = 100
                       xy=(iterations[1], its_GJ[1]),
                       xytext=(3., 200.),
                       arrowprops=dict(arrowstyle='->',
                                       color='Black',
                                       lw=1,
                                       ls='-'),
                       fontsize=11,
                       color='Black',
                       )
    gs_global.annotate(text=f'Green',  # \n contrast = 100
                       xy=(iterations[2], its_G[2]),
                       xytext=(3, 400.),
                       arrowprops=dict(arrowstyle='->',
                                       color='green',
                                       lw=1,
                                       ls='-'),
                       fontsize=11,
                       color='green',
                       )
    gs_global.text(0.20, 0.95, r'$ \approx 24 \times 10^{6}$ DOFs ',
                   transform=gs_global.transAxes)



    # plot mat data 0 Newton iteration
    ijkl = (0, 0, 0, 0)
    cut_to_plot = Nz // 2 - 1

    preconditioner_type = 'Green'
    data_folder_path = (file_folder_path + '/exp_data/' + script_name + '/' + f'Nx={Nx}' + f'Ny={Ny}' + f'Nz={Nz}'
                        + f'_{preconditioner_type}' + '/')

    # first iteration
    iteration_total = 1
    i = 0
    results_name = (f'K4_ijklqyz' + f'_exp_{n_exp}_it{iteration_total}')

    K4_xyz_G = np.load(data_folder_path + results_name + f'.npy', allow_pickle=True, mmap_mode='r')
    # Normalize values
    # idx = np.unravel_index(np.argmax(values), values.shape)
    values = np.copy(K4_xyz_G) / K
    # Calculate normalization parameters
    max_K =10#values.max() #10#
    min_K =1# values.min()
    mid_K =1.66#max_K/2#  values.mean() #1.66

    # Set up colormap and normalization
    norm = mpl.colors.TwoSlopeNorm(vmin=min_K, vcenter=mid_K, vmax=max_K)

    def plot_voxels_colormap(fig, gs_position, K4_xyz_G, K, Nz, iteration_total,
                             cmap='cividis', cutaway_type='half_z',
                             label='(b.2)', cbar_gs_position=None, norm=norm,
                             cbar_label=r'$\mathrm{C}_{11}/\mathrm{K}$',
                             keep_ticks=False   ):
        """
        Plot 3D voxels with colormap based on continuous values.

        Parameters
        ----------
        fig : matplotlib.figure.Figure
            Figure object
        gs_position : tuple or GridSpec index
            Position in gridspec, e.g., gs[0, 2]
        K4_xyz_G : ndarray, shape [Nx, Ny, Nz]
            3D array of values to plot
        K : float
            Normalization constant
        Nz : int
            Size in z direction
        iteration_total : int
            Iteration number for title
        cmap : str or Colormap
            Colormap to use (default: 'cividis')
        cutaway_type : str
            Type of cutaway: 'half_z', 'quarter', 'none'
        label : str
            Subplot label, e.g., '(b.2)'
        cbar_gs_position : GridSpec index or None
            Position for colorbar, e.g., gs[0, 4]. If None, no colorbar is added.
        cbar_label : str
            Label for colorbar

        Returns
        -------
        ax : Axes3D
            The 3D axes object
        cbar : Colorbar or None
            The colorbar object (if cbar_gs_position is provided)
        """
        import matplotlib as mpl
        import matplotlib.pyplot as plt
        import numpy as np

        ax = fig.add_subplot(gs_position, projection='3d')

        # Normalize values
        values = np.copy(K4_xyz_G.transpose()) / K

        # Get dimensions
        nx, ny, nz = values.shape

        ax.dist = 2  # Lower value = closer/larger plot (default is ~10)

        # Remove padding around axes
        ax.margins(0)

        # Tighten the axis bounds
        ax.autoscale_view(tight=True)

        cmap_obj = mpl.cm.get_cmap(cmap) if isinstance(cmap, str) else cmap

        # Create mask for cutaway (True = show, False = hide)
        mask_cut = np.ones(values.shape, dtype=bool)

        if cutaway_type == 'half_z':
#            cutoff_z = nz // 2
#            mask_cut[:, :, cutoff_z:] = False
            cutoff_z =  74#nz // 2
            mask_cut[:-10, :-10, cutoff_z:] = False
            mask_cut[  :-9, :-9, cutoff_z:] = False
            #mask_cut[:-9, :, cutoff_z:] = False
            mask_cut[1: , 1: , :cutoff_z-1]= False
        elif cutaway_type == 'quarter':
            cutoff_x = 2 * nx // 3 - 3
            cutoff_y = 2 * ny // 3 - 3
            cutoff_z = nz // 2
            mask_cut[:cutoff_x, :cutoff_y, cutoff_z:] = False
        elif cutaway_type == 'corner':
            cutoff_x = nx // 2
            cutoff_y = ny // 2
            mask_cut[:cutoff_x, :cutoff_y, :] = False
        # 'none' = no cutaway, show all voxels

        # Convert values to colors using colormap
        colors = cmap_obj(norm(values))  # Returns RGBA array [Nx, Ny, Nz, 4]

        # Plot voxels with colormap colors
        ax.voxels(mask_cut,
                  facecolors=colors,
                 # edgecolor='none',
                  edgecolors=colors,  # Same as face colors
                  linewidth=0,
                  shade=False)

        # View and aspect
        ax.view_init(elev=35, azim=-135)
        ax.set_box_aspect([nx, ny, nz])


        # Clean panes and grid - SAFE VERSION
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False

        ax.xaxis.pane.set_edgecolor('none')
        ax.yaxis.pane.set_edgecolor('none')
        ax.zaxis.pane.set_edgecolor('none')

        # Remove grid
        ax.grid(False)

        # Make grid lines transparent (safer than setting linewidth to 0)
        ax.xaxis._axinfo['grid']['color'] = (1, 1, 1, 0)
        ax.yaxis._axinfo['grid']['color'] = (1, 1, 1, 0)
        ax.zaxis._axinfo['grid']['color'] = (1, 1, 1, 0)



        # Axis limits
        ax.set_xlim(0, nx)
        ax.set_ylim(0, ny)
        ax.set_zlim(0, nz)

        # Ticks: First, Middle, Last
        if keep_ticks:
            ax.set_xticks([0, nx // 2, nx])
            ax.set_yticks([0, ny // 2, ny])
            ax.set_zticks([0, nz // 2, nz])
            ax.set_xticklabels(['1', str(nx // 2), str(nx)])
            ax.set_yticklabels(['1', str(ny // 2), str(ny)])
            ax.set_zticklabels(['1', str(nz // 2), str(nz)])
            ax.zaxis._axinfo['juggled'] = (1, 2, 0)  # Change axis position
            # Axis labels
            ax.set_xlabel(r'$x_1$', labelpad=3)
            ax.set_ylabel(r'$x_2$', labelpad=3)
            ax.set_zlabel(f'voxel index in \n '+ r'$x_3$ dircetion', labelpad=5)
        else:
            ax.set_xticks([ ])
            ax.set_yticks([ ])
            ax.set_zticks([])
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_zticklabels([])
            ax.zaxis._axinfo['juggled'] = (1, 2, 0)  # Change axis position





        # Title and label
        ax.set_title(fr'$i={iteration_total}$')
        if label:
            ax.text2D(0.0, 1.05, rf'\textbf{{{label}}}', transform=ax.transAxes)

        # Add colorbar if position is provided
        cbar = None
        if cbar_gs_position is not None:
            ax_cbar = fig.add_subplot(cbar_gs_position)
            sm = mpl.cm.ScalarMappable(cmap=cmap_obj, norm=norm)
            sm.set_array([])

            cbar = plt.colorbar(sm, location='left', cax=ax_cbar)
            cbar.set_ticks([min_K, mid_K, max_K])
            cbar.set_ticklabels([f'{min_K:.1f}', f'{mid_K:.1f}', f'{max_K:.1f}'])
            ax_cbar.tick_params(right=True, top=False, labelright=False, labeltop=False, labelrotation=0)
            cbar.ax.yaxis.set_ticks_position('right')
            cbar.ax.yaxis.set_label_position('right')
            ax_cbar.set_ylabel(cbar_label)

        return ax, cbar, cmap


    # ===== USAGE EXAMPLE =====

    # Call the function
    ax_geom_0, cbar, cmap_  = plot_voxels_colormap(
        fig=fig,
        gs_position=gs[0, 2],
        K4_xyz_G=K4_xyz_G,
        K=K,
        Nz=Nz,
        iteration_total=iteration_total,
        cmap='cividis',
        cutaway_type='half_z',
        label='(b.2)',
        norm=norm,
        cbar_gs_position=gs[0, 4],
        cbar_label=r'$\mathrm{C}_{11}/\mathrm{K}$'
    )

    gs_global.text(-2.3, 1.05, rf'\textbf{{(a)}}', transform=ax_geom_0.transAxes)
    #ax.text2D(0.0, 1.05, rf'\textbf{{{label}}}', transform=ax.transAxes)
    # Add additional text if needed
    # gs_global.text(-2.3, 1.05, rf'\textbf{{(a)}}', transform=ax_geom_0.transAxes)

    # ----------------
    iteration_total = 0
    results_name = (f'K4_ijklqyz' + f'_exp_{n_exp}_it{iteration_total}')
    del K4_xyz_G
    K4_xyz_G = np.load(data_folder_path + results_name + f'.npy', allow_pickle=True, mmap_mode='r')

    # K4_to_plot_G = K4_xyz_G[..., cut_to_plot]  # K4_xyz_G[i,0,0,0, ..., cut_to_plot]
    #
    # ax_geom_0 = fig.add_subplot(gs[0, 1])
    # # ax_geom_0 = fig.add_axes([0.1, 0.75, 0.2, 0.2])
    # pcm = ax_geom_0.pcolormesh(np.tile(np.transpose(K4_to_plot_G) / K, (1, 1)),
    #                            cmap=cmap_, norm=norm,
    #                            linewidth=0,
    #                            rasterized=True)
    # ax_geom_0.set_aspect('equal')
    # ax_geom_0.set_title(fr'$i={iteration_total}$')
    # ax_geom_0.text(0.5, 0.4, r'$N_{z}$' + f'={Nx}', transform=ax_geom_0.transAxes)
    # ax_geom_0.text(-0., 1.05, rf'\textbf{{(b.1)}}', transform=ax_geom_0.transAxes)
    # ax_geom_0.set_aspect('equal')
    #
    # ax_geom_0.set_xticks([])
    # ax_geom_0.set_xticklabels([])
    # ax_geom_0.set_yticks([])
    # ax_geom_0.set_yticklabels([])
    # # ax_geom_0.set_xlim([0, Nz])
    # # ax_geom_0.set_ylim([0, Nz])
#    ax_geom_0.set_box_aspect(1)

    # Call the function
    ax_geom_0  = plot_voxels_colormap(
        fig=fig,
        gs_position=gs[0, 1],
        K4_xyz_G=K4_xyz_G,
        K=K,
        Nz=Nz,
        iteration_total=iteration_total,
        cmap='cividis',
        cutaway_type='half_z',
        label='(b.1)',
        norm=norm,
       #cbar_gs_position=gs[0, 4],
       # cbar_label=r'$\mathrm{C}_{11}/\mathrm{K}$'
    )[0]
    # ----------------

    # for iteration_total in 6:
    iteration_total = 2
    # iteration_total = 2
    del K4_xyz_G
    results_name = (f'K4_ijklqyz' + f'_exp_{n_exp}_it{iteration_total}')
    K4_xyz_G = np.load(data_folder_path + results_name + f'.npy', allow_pickle=True, mmap_mode='r')
    K4_to_plot_G = K4_xyz_G[..., cut_to_plot]  # K4_xyz_G[i,0,0,0, ..., cut_to_plot]

    #ax_geom_0 = fig.add_subplot(gs[0, 3])
    # Call the function
    ax_geom_0  = plot_voxels_colormap(
        fig=fig,
        gs_position=gs[0, 3],
        K4_xyz_G=K4_xyz_G,
        K=K,
        Nz=Nz,
        iteration_total=iteration_total,
        cmap='cividis',
        cutaway_type='half_z',
        label='(b.3)',
        norm=norm,
       #cbar_gs_position=gs[0, 4],
       # cbar_label=r'$\mathrm{C}_{11}/\mathrm{K}$'
    )[0]
    # ----------------
    # for iteration_total in 6:
    iteration_total = 3
    # iteration_total = 2
    results_name = (f'K4_ijklqyz' + f'_exp_{n_exp}_it{iteration_total}')
    del K4_xyz_G
    K4_xyz_G = np.load(data_folder_path + results_name + f'.npy', allow_pickle=True, mmap_mode='r')
   # K4_to_plot_G = K4_xyz_G[..., cut_to_plot]  # K4_xyz_G[i,0,0,0, ..., cut_to_plot]
    # ax_geom_0 = fig.add_axes([0.7, 0.75, 0.2, 0.2])
    ax_geom_0 = plot_voxels_colormap(
        fig=fig,
        gs_position=gs[1, 1],
        K4_xyz_G=K4_xyz_G,
        K=K,
        Nz=Nz,
        iteration_total=iteration_total,
        cmap='cividis',
        cutaway_type='half_z',
        label=f'(b.{iteration_total})',
        norm=norm,
        # cbar_gs_position=gs[0, 4],
        # cbar_label=r'$\mathrm{C}_{11}/\mathrm{K}$'
    )[0]

    iteration_total = 4
    # iteration_total = 2
    results_name = (f'K4_ijklqyz' + f'_exp_{n_exp}_it{iteration_total}')
    del K4_xyz_G
    K4_xyz_G = np.load(data_folder_path + results_name + f'.npy', allow_pickle=True, mmap_mode='r')

    ax_geom_0 = plot_voxels_colormap(
        fig=fig,
        gs_position=gs[1, 2],
        K4_xyz_G=K4_xyz_G,
        K=K,
        Nz=Nz,
        iteration_total=iteration_total,
        cmap='cividis',
        cutaway_type='half_z',
        label=f'(b.{5})',
        norm=norm,
        # cbar_gs_position=gs[0, 4],
        # cbar_label=r'$\mathrm{C}_{11}/\mathrm{K}$'
    )[0]

    iteration_total = 5
    # iteration_total = 2
    results_name = (f'K4_ijklqyz' + f'_exp_{n_exp}_it{iteration_total}')
    del K4_xyz_G
    K4_xyz_G = np.load(data_folder_path + results_name + f'.npy', allow_pickle=True, mmap_mode='r')
    ax_geom_0 = plot_voxels_colormap(
        fig=fig,
        gs_position=gs[1, 3],
        K4_xyz_G=K4_xyz_G,
        K=K,
        Nz=Nz,
        iteration_total=iteration_total,
        cmap='cividis',
        cutaway_type='half_z',
        label=f'(b.{6})',
        norm=norm,
        keep_ticks=True,
        # cbar_gs_position=gs[0, 4],
        # cbar_label=r'$\mathrm{C}_{11}/\mathrm{K}$'
    )[0]
    #ax_geom_0.yaxis.set_ticks_position('right')
    #ax_geom_0.yaxis.set_label_position('right')

    # axis for cross sections
    add_stress_plot = False
    if add_stress_plot:
        #### STRESSES
        ax_cross = fig.add_axes([0.65, 0.2, 0.2, 0.2])
        iteration_total = 6
        preconditioner_type = 'Green'

        data_folder_path = (file_folder_path + '/exp_data/' + script_name + '/' + f'Nx={Nx}' + f'Ny={Ny}' + f'Nz={Nz}'
                            + f'_{preconditioner_type}' + '/')

        stress_G = np.load(data_folder_path + f'stress' + f'_it{iteration_total}' + f'.npy', allow_pickle=True)
        strain_fluc_G = np.load(data_folder_path + f'strain_fluc_field' + f'_it{iteration_total}' + f'.npy',
                                allow_pickle=True)

        preconditioner_type = 'Jacobi_Green'

        data_folder_path = (file_folder_path + '/exp_data/' + script_name + '/' + f'Nx={Nx}' + f'Ny={Ny}' + f'Nz={Nz}'
                            + f'_{preconditioner_type}' + '/')
        stress_GJ = np.load(data_folder_path + f'stress' + f'_it{iteration_total}' + f'.npy', allow_pickle=True)
        strain_fluc_GJ = np.load(data_folder_path + f'strain_fluc_field' + f'_it{iteration_total}' + f'.npy',
                                 allow_pickle=True)

        ### compute the difference
        ij = (0, 0)

        stress_diff = abs((stress_GJ[ij + (..., 0)] - stress_G[ij + (..., 0)]))  # / stress_G[ij + (..., 0)])

        max_K = stress_diff.max()
        min_K = 0  # stress_diff.min()

        pcm = ax_cross.pcolormesh(np.tile(stress_diff, (1, 1)),
                                  cmap=cmap_,
                                  # norm=norm,
                                  linewidth=0,
                                  rasterized=True)
        ax_cross.set_aspect('equal')

        ax_cbar2 = fig.add_axes([0.9, 0.2, 0.01, 0.3])
        cbar = plt.colorbar(pcm, location='left', cax=ax_cbar2)
        cbar.set_ticks(ticks=[0, max_K])  # 0,min_K,
        # cbar.set_ticklabels([f'{min_K:.0f}', f'{0:.0f}', f'{max_K:.0f}'])
        cbar.set_ticklabels(
            ['0',
             f'$10^{{{int(np.log10(max_K))}}}$'])  # f'$10^{{{-int(np.log10(abs(min_K)))}}}$', f'$10^{{{int(np.log10(1))}}}$',
        ax_cbar2.tick_params(right=True, top=False, labelright=False, labeltop=False, labelrotation=0)
        cbar.ax.yaxis.set_ticks_position('right')  # move ticks to right
        cbar.ax.yaxis.set_label_position('right')  # move label to right
        ax_cbar2.set_ylabel(fr'$(\sigma^{{G}}_{{11}}-\sigma^{{GJ}}_{{11}})  $')

    fig.tight_layout()
    fname = f'fig_3D_{Nx}_exp_{n_exponents[0]}' + '{}'.format('.pdf')
    plt.savefig(figure_folder_path + script_name + fname, bbox_inches='tight')
    print(('create figure: {}'.format(figure_folder_path + script_name + fname)))
    plt.savefig('fig_3D.pdf', format='pdf', dpi=1200, bbox_inches='tight', pad_inches=0.1)
    plt.savefig('fig_3D.png', format='png', dpi=1200, bbox_inches='tight', pad_inches=0.1)
    plt.show()

if plot_data_vs_CG:
    # print time vs DOFS
    its_G = []
    its_GJ = []
    stress_diff_norm = []
    strain_fluc_norm = []
    strain_total_norm = []
    diff_rhs_norm = []
    norm_rhs_G = []
    norm_rhs_GJ = []
    rhs_inf_G = []
    rhs_inf_GJ = []
    norm_newrton_stop_G = []
    norm_newrton_stop_GJ = []
    #grid_sizes= np.array( [ 50, 100, 150,200])#,200,128,200

    Nx = 40#2 ** 4#8
    Ny = Nx
    Nz = Nx
    it_max = 10
    n_exponents = np.array([5])
    iterations = np.arange(it_max) # numbers of grids points

    its_G = np.zeros([it_max,  len(n_exponents)])
    its_GJ = np.zeros([it_max,  len(n_exponents)])

    for j in np.arange(int(len(n_exponents))):
        n_exp = n_exponents[j]
        for iteration_total in iterations:

            preconditioner_type = 'Green'

            data_folder_path = (
                    file_folder_path + '/exp_data/' + script_name + '/' + f'Nx={Nx}' + f'Ny={Ny}' + f'Nz={Nz}'
                    + f'_{preconditioner_type}' + '/')
            if iteration_total < it_max:
                if Nx == 256:
                    _info_final_G = np.load(data_folder_path + f'info_log_it{iteration_total}.npz', allow_pickle=True)

                else:
                    _info_final_G = np.load(data_folder_path + f'info_log_exp_{n_exp}_it{iteration_total}.npz',
                                            allow_pickle=True)

            # with open(data_folder_path + f'stress' + f'_exp_{n_exp}_it{iteration_total+1}' + f'.npy', 'rb') as f:
            #     magic = f.read(6)
            #     print(f"Magic number: {magic}")

            # strain_fluc_G = np.load(data_folder_path + f'strain_fluc_field' + f'_it{iteration_total}' + f'.npy',
            #                         allow_pickle=True)
            # strain_total_G = np.load(data_folder_path + f'total_strain_field' + f'_it{iteration_total}' + f'.npy',
            #                          allow_pickle=True)
            # stress_G = np.load(data_folder_path + f'stress' + f'_it{iteration_total}' + f'.npy',
            #                    allow_pickle=True)  # , allow_pickle=True

            # rhs_field_G = np.load(data_folder_path + f'rhs_field' + f'_it{iteration_total}' + f'.npy', allow_pickle=True)

            its_G[iteration_total, j] = _info_final_G.f.nb_it_comb
            norm_rhs_G.append(_info_final_G.f.norm_rhs_field)
            norm_newrton_stop_G.append(_info_final_G.f.newton_stop_crit)
            info_log_final_G = np.load(data_folder_path + f'info_log_final_exp_{n_exp}.npz', allow_pickle=True)

            preconditioner_type = 'Green_Jacobi'

            data_folder_path = (
                    file_folder_path + '/exp_data/' + script_name + '/' + f'Nx={Nx}' + f'Ny={Ny}' + f'Nz={Nz}'
                    + f'_{preconditioner_type}' + '/')
            if iteration_total < it_max:
                if Nx == 256:
                    _info_final_GJ = np.load(data_folder_path + f'info_log_it{iteration_total}.npz',
                                             allow_pickle=True)

                else:
                    _info_final_GJ = np.load(data_folder_path + f'info_log_exp_{n_exp}_it{iteration_total}.npz',
                                             allow_pickle=True)
            # stress_GJ = np.load(data_folder_path + f'stress' + f'_it{iteration_total}' + f'.npy', allow_pickle=True)
            # strain_fluc_GJ = np.load(data_folder_path + f'strain_fluc_field' + f'_it{iteration_total}' + f'.npy',
            #                          allow_pickle=True)
            # strain_total_GJ = np.load(data_folder_path + f'total_strain_field' + f'_it{iteration_total}' + f'.npy',
            #                           allow_pickle=True)
            # rhs_field_GJ = np.load(data_folder_path + f'rhs_field' + f'_it{iteration_total}' + f'.npy', allow_pickle=True)

            its_GJ[iteration_total, j] = _info_final_GJ.f.nb_it_comb
            norm_rhs_GJ.append(_info_final_GJ.f.norm_rhs_field)
            norm_newrton_stop_GJ.append(_info_final_GJ.f.newton_stop_crit)
            # diff_stress = stress_G - stress_GJ
            # stress_diff_norm.append(
            #      np.linalg.norm(diff_stress.ravel(), ord=np.inf))  # / np.linalg.norm(stress_G))
            print(_info_final_G.f.norm_rr[0])

            # diff_strain_fluc = strain_fluc_G - strain_fluc_GJ
            # strain_fluc_norm.append(
            #     np.linalg.norm(diff_strain_fluc.ravel(), ord=np.inf))  # / _info_final_GJ.f.norm_En)
            print(_info_final_GJ.f.norm_rr[0])
            # diff_strain_total = strain_total_G - strain_total_GJ
            # strain_total_norm.append(
            #    np.linalg.norm(diff_strain_total.ravel(), ord=np.inf))  # / _info_final_GJ.f.norm_En)

            # diff_rhs = rhs_field_G - rhs_field_GJ
            # diff_rhs_norm.append(
            #    np.linalg.norm(diff_rhs.ravel()))
            # rhs_inf_G.append(
            #     np.linalg.norm(rhs_field_G.ravel(), ord=np.inf))
            # rhs_inf_GJ.append(
            #     np.linalg.norm(rhs_field_GJ.ravel(), ord=np.inf))

            info_log_final_GJ = np.load(data_folder_path + f'info_log_final_exp_{n_exp}.npz', allow_pickle=True)

    # del strain_fluc_G, strain_total_G, stress_G, rhs_field_G
    # del strain_fluc_GJ, strain_total_GJ, stress_GJ, rhs_field_GJ
    # del diff_stress

    # data = np.load('large_3d_array.npy', mmap_mode='r')
    K = 2  # _info_final_G.f.model_parameters_linear
    its_G = np.array(its_G)
    its_GJ = np.array(its_GJ)
    strain_0_norm = np.array(np.atleast_1d(_info_final_GJ.f.norm_En))
    # stress_diff_norm = np.array(stress_diff_norm)
    # strain_fluc_norm = np.array(strain_fluc_norm)
    # strain_total_norm = np.array(strain_total_norm)
    # diff_rhs_norm = np.array(diff_rhs_norm)

    norm_rhs_t_G = np.concatenate((np.array(np.atleast_1d(_info_final_G.f.rhs_t_norm)), np.array(norm_rhs_G)))
    norm_rhs_t_GJ = np.concatenate((np.array(np.atleast_1d(_info_final_GJ.f.rhs_t_norm)), np.array(norm_rhs_GJ)))

    fig = plt.figure(figsize=(8.3, 4.0))
    gs = fig.add_gridspec(2, 5, hspace=0.2, wspace=0.1, width_ratios=[1, 1, 1, 1, 0.05],
                          height_ratios=[1, 1.])

    gs_global = fig.add_subplot(gs[:, 0])

    gs_global.plot(iterations, its_G[:, 0], 'g-', marker='x', label='Green')
    gs_global.plot(iterations, its_GJ[:, 0], 'k-', marker='o', markerfacecolor='none', label='Green-Jacobi')

    # gs_global.plot(iterations, its_G[:, -2], 'g-', marker='x', label='Green')
    # gs_global.plot(iterations, its_GJ[:, -2], 'k-', marker='o', markerfacecolor='none', label='Green-Jacobi')
    #
    # gs_global.plot(iterations, its_G[:, -1], 'g-', marker='x', label='Green')
    # gs_global.plot(iterations, its_GJ[:, -1], 'k-', marker='o', markerfacecolor='none', label='Green-Jacobi')

    gs_global.set_xlabel(r'Newton iteration -  $i$')
    gs_global.set_ylabel(r'$\#$ of PCG iterations')
    # gs_global.legend(loc='best')
    gs_global.set_xlim(-0.05, iterations[-1] + .05)
    gs_global.set_ylim(0., 300)
    gs_global.set_xticks(iterations)

    gs_global.annotate(text=f'Green-Jacobi',  # \n contrast = 100
                       xy=(iterations[1], its_GJ[1]),
                       xytext=(3., 100.),
                       arrowprops=dict(arrowstyle='->',
                                       color='Black',
                                       lw=1,
                                       ls='-'),
                       fontsize=11,
                       color='Black',
                       )
    gs_global.annotate(text=f'Green',  # \n contrast = 100
                       xy=(iterations[2], its_G[2]),
                       xytext=(3, 200.),
                       arrowprops=dict(arrowstyle='->',
                                       color='green',
                                       lw=1,
                                       ls='-'),
                       fontsize=11,
                       color='green',
                       )
    gs_global.text(0.20,  0.95, r'$ \approx  6 \times 10^{6}$ DOFs ',
                   transform=gs_global.transAxes)

    # gs_global.set_ylim(0, 800)

    # Right y-axis
    # ax2 = fig.add_subplot(gs10[0, 1])
    #
    # ax2.semilogy(np.arange(len(norm_newrton_stop_GJ)), norm_newrton_stop_GJ, 'g:', marker='o', markerfacecolor='none',
    #              label=r'$norm_newrton_stop_GJ$')
    # ax2.semilogy(np.arange(len(norm_newrton_stop_G)), norm_newrton_stop_G, 'g:', marker='o', markerfacecolor='none',
    #              label=r'$norm_newrton_stop_GJ$')
    # ax2.set_ylim([1e-10, 1e-1])
    # ax2.set_yticks([1e-10, 1e-7, 1e-4, 1e-1])
    # ax2.set_yticklabels([r'$10^{-10}$', r'$10^{-7}$', r'$10^{-4}$', r'$10^{-1}$'])
    # ax2.yaxis.set_ticks_position('right')  # move ticks to right
    # ax2.yaxis.set_label_position('right')  # move label to right
    # ax2.set_xlabel(r'Newton iteration - $i$')
    # ax2.set_ylabel(r'$||{X} ||_{\infty}$')
    # # ax2.legend(loc='best')
    # ax2.set_xlim(-0.05, iterations[-1] + .05)
    # ax2.set_xticks(iterations)
    # ax2.text(0.02, 0.92, rf'\textbf{{(b.2)}}', transform=ax2.transAxes)

    # plot mat data 0 Newton iteration
    ijkl = (0, 0, 0, 0)
    # results_name = (f'init_K')
    # K4_init = np.load(data_folder_path + results_name + f'.npy', allow_pickle=True, mmap_mode='r')
    # del K4_init
    # # plot mat data  in  Newton iterations

    preconditioner_type = 'Green'
    data_folder_path = (file_folder_path + '/exp_data/' + script_name + '/' + f'Nx={Nx}' + f'Ny={Ny}' + f'Nz={Nz}'
                        + f'_{preconditioner_type}' + '/')

    # first iteration
    iteration_total = 1
    i = 0
    results_name = (f'K4_ijklqyz' + f'_exp_{n_exp}_it{iteration_total}')

    K4_xyz_G = np.load(data_folder_path + results_name + f'.npy', allow_pickle=True, mmap_mode='r')
    K4_xyz_G_50= np.load(  file_folder_path + '/exp_data/' + script_name + '/' + f'Nx={50}' + f'Ny={50}' + f'Nz={50}'
                        + f'_{preconditioner_type}' + '/'  + results_name + f'.npy', allow_pickle=True, mmap_mode='r')

    # ax_geom_0 = fig.add_axes([0.3, 0.75, 0.2, 0.2])
    ax_geom_0 = fig.add_subplot(gs[0, 2])

    # max_K = K4_ijklqyz_G[ijkl + (..., 0)].max() / K
    # min_K = K4_ijklqyz_G[ijkl + (..., 0)].min() / K
    cut_to_plot = Nz // 2 - 1
    K4_to_plot_G = K4_xyz_G[..., cut_to_plot]  # K4_xyz_G[i,0,0,0, ..., cut_to_plot]
    max_K = K4_to_plot_G.max() / K
    min_K = K4_to_plot_G.min() / K

    mid_K = K4_to_plot_G.mean() / K  # K4_init[ (Nx // 2, Ny // 2,  Nz // 2)] ijkl +
    norm = mpl.colors.TwoSlopeNorm(vmin=min_K, vcenter=mid_K, vmax=max_K)
    cmap_ = mpl.cm.cividis  # mpl.cm.seismic

    pcm = ax_geom_0.pcolormesh(np.tile(np.transpose(K4_to_plot_G) / K, (1, 1)),
                               cmap=cmap_, norm=norm,
                               linewidth=0,
                               rasterized=True)
    ax_geom_0.set_aspect('equal')
    ax_geom_0.set_title(fr'$i={iteration_total}$')
    # ax_geom_0.text(0.32, 0.5, r'$N_{z}$=128', transform=ax_geom_0.transAxes)
    ax_geom_0.text(-0., 1.05, rf'\textbf{{(b.2)}}', transform=ax_geom_0.transAxes)
    ax_geom_0.set_aspect('equal')

    gs_global.text(-2.3, 1.05, rf'\textbf{{(a)}}', transform=ax_geom_0.transAxes)

    ax_geom_0.set_xticks([])
    ax_geom_0.set_xticklabels([])
    ax_geom_0.set_yticks([])
    ax_geom_0.set_yticklabels([])
    # ax_geom_0.set_xlim([0, Nz])
    # ax_geom_0.set_ylim([0, Nz])
    ax_geom_0.set_box_aspect(1)

    # ----------------
    # colobar is based on the first iteration
    # ax_cbar = fig.add_axes([0.8, 0.22, 0.02, 0.2])
    ax_cbar = fig.add_subplot(gs[0, 4])

    cbar = plt.colorbar(pcm, location='left', cax=ax_cbar)
    cbar.set_ticks(ticks=[min_K, mid_K, max_K])
    cbar.set_ticklabels([f'{min_K:.0f}', f'{mid_K:.0f}', f'{max_K:.0f}'])
    ax_cbar.tick_params(right=True, top=False, labelright=False, labeltop=False, labelrotation=0)
    cbar.ax.yaxis.set_ticks_position('right')  # move ticks to right
    cbar.ax.yaxis.set_label_position('right')  # move label to right
    ax_cbar.set_ylabel(r'$\mathrm{C}_{11}/\mathrm{K}$')
    # ----------------
    iteration_total = 0
    results_name = (f'K4_ijklqyz' + f'_exp_{n_exp}_it{iteration_total}')
    del K4_xyz_G
    K4_xyz_G = np.load(data_folder_path + results_name + f'.npy', allow_pickle=True, mmap_mode='r')
    # voxelino=np.copy(K4_xyz_G)
    # voxelino[voxelino < 2.5] = 0
    #
    # # Create boolean array
    # voxelarray = voxelino.astype(bool)
    #
    # # Set colors
    # colors = np.empty(voxelarray.shape, dtype=object)
    # colors[voxelarray] = 'blue'
    #
    # # Plot
    # fig3D = plt.figure(5)
    # ax_3D = fig3D.add_subplot(projection='3d')
    # ax_3D.voxels(voxelarray, facecolors=colors, edgecolor=None, alpha=0.9)
    #
    # ax_3D.set_xlabel('X')
    # ax_3D.set_ylabel('Y')
    # ax_3D.set_zlabel('Z')
    # ax_3D.set_title('3D Inclusions')
    #
    # plt.tight_layout()
    #plt.show()

    # Print volume fraction
    # vf = voxelino.sum() / voxelino.size
    # print(f"Volume fraction: {vf:.2%}")



    K4_to_plot_G = K4_xyz_G[..., cut_to_plot]  # K4_xyz_G[i,0,0,0, ..., cut_to_plot]




    ax_geom_0 = fig.add_subplot(gs[0, 1])
    # ax_geom_0 = fig.add_axes([0.1, 0.75, 0.2, 0.2])
    pcm = ax_geom_0.pcolormesh(np.tile(np.transpose(K4_to_plot_G) / K, (1, 1)),
                               cmap=cmap_, norm=norm,
                               linewidth=0,
                               rasterized=True)
    ax_geom_0.set_aspect('equal')
    ax_geom_0.set_title(fr'$i={iteration_total}$')
    ax_geom_0.text(0.5, 0.4, r'$N_{z}$'+f'={Nx}', transform=ax_geom_0.transAxes)
    ax_geom_0.text(-0., 1.05, rf'\textbf{{(b.1)}}', transform=ax_geom_0.transAxes)
    ax_geom_0.set_aspect('equal')

    ax_geom_0.set_xticks([])
    ax_geom_0.set_xticklabels([])
    ax_geom_0.set_yticks([])
    ax_geom_0.set_yticklabels([])
    # ax_geom_0.set_xlim([0, Nz])
    # ax_geom_0.set_ylim([0, Nz])
    ax_geom_0.set_box_aspect(1)

    # ----------------

    # for iteration_total in 6:
    iteration_total = 2
    # iteration_total = 2
    del K4_xyz_G
    results_name = (f'K4_ijklqyz' + f'_exp_{n_exp}_it{iteration_total}')
    K4_xyz_G = np.load(data_folder_path + results_name + f'.npy', allow_pickle=True, mmap_mode='r')
    K4_to_plot_G = K4_xyz_G[..., cut_to_plot]  # K4_xyz_G[i,0,0,0, ..., cut_to_plot]

    ax_geom_0 = fig.add_subplot(gs[0, 3])
    # ax_geom_0 = fig.add_axes([0.5, 0.75, 0.2, 0.2])
    pcm = ax_geom_0.pcolormesh(np.tile(np.transpose(K4_to_plot_G) / K, (1, 1)),
                               cmap=cmap_, norm=norm,
                               linewidth=0,
                               rasterized=True)
    ax_geom_0.set_aspect('equal')
    ax_geom_0.set_title(fr'$i={iteration_total}$')
    # ax_geom_0.text(0.30, 0.5, r'$N_{z}=128$', transform=ax_geom_0.transAxes)

    ax_geom_0.text(-0., 1.05, rf'\textbf{{(b.{3})}}', transform=ax_geom_0.transAxes)
    ax_geom_0.set_aspect('equal')

    ax_geom_0.set_xticks([])
    ax_geom_0.set_xticklabels([])
    ax_geom_0.set_yticks([])
    ax_geom_0.set_yticklabels([])
    # ax_geom_0.set_xlim([0, Nz])
    # ax_geom_0.set_ylim([0, Nz])
    ax_geom_0.set_box_aspect(1)
    # ----------------
    # for iteration_total in 6:
    iteration_total = 3
    # iteration_total = 2
    results_name = (f'K4_ijklqyz' + f'_exp_{n_exp}_it{iteration_total}')
    del K4_xyz_G
    K4_xyz_G = np.load(data_folder_path + results_name + f'.npy', allow_pickle=True, mmap_mode='r')
    K4_to_plot_G = K4_xyz_G[..., cut_to_plot]  # K4_xyz_G[i,0,0,0, ..., cut_to_plot]
    # ax_geom_0 = fig.add_axes([0.7, 0.75, 0.2, 0.2])
    ax_geom_0 = fig.add_subplot(gs[1, 1])

    pcm = ax_geom_0.pcolormesh(np.tile(np.transpose(K4_to_plot_G) / K, (1, 1)),
                               cmap=cmap_, norm=norm,
                               linewidth=0,
                               rasterized=True)
    ax_geom_0.set_aspect('equal')
    ax_geom_0.set_title(fr'$i={iteration_total}$')
    #ax_geom_0.text(0.32, -0.5, r'$3 \cdot 256^{3}$ DOFs', transform=ax_geom_0.transAxes)
    # ax_geom_0.text(0.2, -0.3, r'$ \approx  50 \times 10^{6}DOFs $', transform=ax_geom_0.transAxes)

    ax_geom_0.text(-0., 1.05, rf'\textbf{{(b.{4})}}', transform=ax_geom_0.transAxes)
    ax_geom_0.set_xticks([1, Nz // 2, Nz])

    ax_geom_0.set_yticks([])
    ax_geom_0.set_yticklabels([])
    # ax_geom_0.set_xlim([0, Nz])
    # ax_geom_0.set_ylim([0, Nz])
    ax_geom_0.set_box_aspect(1)

    iteration_total = 4
    # iteration_total = 2
    results_name = (f'K4_ijklqyz' + f'_exp_{n_exp}_it{iteration_total}')
    del K4_xyz_G
    K4_xyz_G = np.load(data_folder_path + results_name + f'.npy', allow_pickle=True, mmap_mode='r')
    K4_to_plot_G = K4_xyz_G[..., cut_to_plot]  # K4_xyz_G[i,0,0,0, ..., cut_to_plot]
    # ax_geom_0 = fig.add_axes([0.7, 0.75, 0.2, 0.2])
    ax_geom_0 = fig.add_subplot(gs[1, 2])

    pcm = ax_geom_0.pcolormesh(np.tile(np.transpose(K4_to_plot_G) / K, (1, 1)),
                               cmap=cmap_, norm=norm,
                               linewidth=0,
                               rasterized=True)
    ax_geom_0.set_aspect('equal')
    ax_geom_0.set_title(fr'$i={iteration_total}$')
    #ax_geom_0.text(0.32, -0.5, r'$3 \cdot 256^{3}$ DOFs', transform=ax_geom_0.transAxes)
    # ax_geom_0.text(0.2, -0.3, r'$ \approx  50 \times 10^{6}DOFs $', transform=ax_geom_0.transAxes)

    ax_geom_0.text(-0., 1.05, rf'\textbf{{(b.{5})}}', transform=ax_geom_0.transAxes)
    ax_geom_0.set_aspect('equal')
    ax_geom_0.set_xticks([1, Nz // 2, Nz])

    ax_geom_0.set_yticks([])
    ax_geom_0.set_yticklabels([])
    # ax_geom_0.set_xlim([0, Nz])
    # ax_geom_0.set_ylim([0, Nz])
    ax_geom_0.set_box_aspect(1)

    iteration_total = 5
    # iteration_total = 2
    results_name = (f'K4_ijklqyz' + f'_exp_{n_exp}_it{iteration_total}')
    del K4_xyz_G
    K4_xyz_G = np.load(data_folder_path + results_name + f'.npy', allow_pickle=True, mmap_mode='r')
    K4_to_plot_G = K4_xyz_G[..., cut_to_plot]  # K4_xyz_G[i,0,0,0, ..., cut_to_plot]
    # ax_geom_0 = fig.add_axes([0.7, 0.75, 0.2, 0.2])
    ax_geom_0 = fig.add_subplot(gs[1, 3])

    pcm = ax_geom_0.pcolormesh(np.tile(np.transpose(K4_to_plot_G) / K, (1, 1)),
                               cmap=cmap_, norm=norm,
                               linewidth=0,
                               rasterized=True)
    ax_geom_0.set_aspect('equal')
    ax_geom_0.set_title(fr'$i={iteration_total}$')
    #ax_geom_0.text(0.32, -0.5, r'$3 \cdot 256^{3}$ DOFs', transform=ax_geom_0.transAxes)
    # ax_geom_0.text(0.2, -0.3, r'$ \approx  50 \times 10^{6}DOFs $', transform=ax_geom_0.transAxes)

    ax_geom_0.text(-0., 1.05, rf'\textbf{{(b.{6})}}', transform=ax_geom_0.transAxes)
    ax_geom_0.set_aspect('equal')
    ax_geom_0.set_xticks([])
    ax_geom_0.set_ylabel('Pixel index')
    # ax_geom_0.set_xticks([0, Nx//2, Nx])

    ax_geom_0.set_yticks([1, Nz // 2, Nz])
    ax_geom_0.set_xticks([1, Nz // 2, Nz])
    #    ax_geom_0.set_xlim([0, Nz])
    #  ax_geom_0.set_ylim([0, Nz])
    ax_geom_0.set_box_aspect(1)  # Maintain square aspect ratio
    ax_geom_0.yaxis.set_ticks_position('right')
    ax_geom_0.yaxis.set_label_position('right')

    # axis for cross sections
    add_stress_plot = False
    if add_stress_plot:
        #### STRESSES
        ax_cross = fig.add_axes([0.65, 0.2, 0.2, 0.2])
        iteration_total = 6
        preconditioner_type = 'Green'

        data_folder_path = (file_folder_path + '/exp_data/' + script_name + '/' + f'Nx={Nx}' + f'Ny={Ny}' + f'Nz={Nz}'
                            + f'_{preconditioner_type}' + '/')

        stress_G = np.load(data_folder_path + f'stress' + f'_it{iteration_total}' + f'.npy', allow_pickle=True)
        strain_fluc_G = np.load(data_folder_path + f'strain_fluc_field' + f'_it{iteration_total}' + f'.npy',
                                allow_pickle=True)

        preconditioner_type = 'Jacobi_Green'

        data_folder_path = (file_folder_path + '/exp_data/' + script_name + '/' + f'Nx={Nx}' + f'Ny={Ny}' + f'Nz={Nz}'
                            + f'_{preconditioner_type}' + '/')
        stress_GJ = np.load(data_folder_path + f'stress' + f'_it{iteration_total}' + f'.npy', allow_pickle=True)
        strain_fluc_GJ = np.load(data_folder_path + f'strain_fluc_field' + f'_it{iteration_total}' + f'.npy',
                                 allow_pickle=True)

        ### compute the difference
        ij = (0, 0)

        stress_diff = abs((stress_GJ[ij + (..., 0)] - stress_G[ij + (..., 0)]))  # / stress_G[ij + (..., 0)])

        max_K = stress_diff.max()
        min_K = 0  # stress_diff.min()

        pcm = ax_cross.pcolormesh(np.tile(stress_diff, (1, 1)),
                                  cmap=cmap_,
                                  # norm=norm,
                                  linewidth=0,
                                  rasterized=True)
        ax_cross.set_aspect('equal')

        ax_cbar2 = fig.add_axes([0.9, 0.2, 0.01, 0.3])
        cbar = plt.colorbar(pcm, location='left', cax=ax_cbar2)
        cbar.set_ticks(ticks=[0, max_K])  # 0,min_K,
        # cbar.set_ticklabels([f'{min_K:.0f}', f'{0:.0f}', f'{max_K:.0f}'])
        cbar.set_ticklabels(
            ['0',
             f'$10^{{{int(np.log10(max_K))}}}$'])  # f'$10^{{{-int(np.log10(abs(min_K)))}}}$', f'$10^{{{int(np.log10(1))}}}$',
        ax_cbar2.tick_params(right=True, top=False, labelright=False, labeltop=False, labelrotation=0)
        cbar.ax.yaxis.set_ticks_position('right')  # move ticks to right
        cbar.ax.yaxis.set_label_position('right')  # move label to right
        ax_cbar2.set_ylabel(fr'$(\sigma^{{G}}_{{11}}-\sigma^{{GJ}}_{{11}})  $')

    fig.tight_layout()
    fname = f'fig_2' + '{}'.format('.pdf')
    plt.savefig(figure_folder_path + script_name + fname, bbox_inches='tight')
    print(('create figure: {}'.format(figure_folder_path + script_name + fname)))

    plt.show()
##############################################################################################
if plot_stress_field:

    def compute_eq_strain(strain):
        strain_trace_xyz = np.einsum('ii...', strain) / 3  # todo{2 or 3 in 2D }

        # volumetric strain
        strain_vol_ijxyz = np.ndarray(shape=strain.shape)
        strain_vol_ijxyz.fill(0)
        for d in np.arange(3):
            strain_vol_ijxyz[d, d, ...] = strain_trace_xyz

        # deviatoric strain
        strain_dev_ijxyz = strain - strain_vol_ijxyz

        # equivalent strain
        strain_dev_ddot = np.einsum('ijxyz,jixyz-> xyz', strain_dev_ijxyz, strain_dev_ijxyz)
        strain_eq_xyz = np.sqrt((2. / 3.) * strain_dev_ddot)

        return strain_eq_xyz


    # print time vs DOFS
    iteration_total = 6
    Nx = 3
    Ny = 256
    Nz = 256
    i = 0
    j = 1

    preconditioner_type = 'Green'
    data_folder_path = (file_folder_path + '/exp_data/' + script_name + '/' + f'Nx={Nx}' + f'Ny={Ny}' + f'Nz={Nz}'
                        + f'_{preconditioner_type}' + '/')

    _info_final_G = np.load(data_folder_path + f'info_log_final.npz', allow_pickle=True)

    results_name = (f'stress_{i, j}' + f'_it{iteration_total}')
    stress_field_G = np.load(data_folder_path + results_name + f'.npy', allow_pickle=True)
    results_name = (f'total_strain_field_{i, j}' + f'_it{iteration_total}')
    total_strain_field_G = np.load(data_folder_path + results_name + f'.npy', allow_pickle=True)
    results_name = (f'K4_ijklqyz_{i, 0}' + f'_it{iteration_total}')
    K4_ijklqyz_G = np.load(data_folder_path + results_name + f'.npy', allow_pickle=True)

    results_name = (f'init_K_{0, 0}')
    K4_init = np.load(data_folder_path + results_name + f'.npy', allow_pickle=True)

    preconditioner_type = 'Jacobi_Green'

    data_folder_path = (file_folder_path + '/exp_data/' + script_name + '/' + f'Nx={Nx}' + f'Ny={Ny}' + f'Nz={Nz}'
                        + f'_{preconditioner_type}' + '/')

    _info_final_GJ = np.load(data_folder_path + f'info_log_final.npz', allow_pickle=True)

    results_name = (f'stress_{i, j}' + f'_it{iteration_total}')
    stress_field_GJ = np.load(data_folder_path + results_name + f'.npy', allow_pickle=True)

    results_name = (f'total_strain_field_{i, j}' + f'_it{iteration_total}')
    total_strain_field_GJ = np.load(data_folder_path + results_name + f'.npy', allow_pickle=True)

    results_name = (f'K4_ijklqyz_{i, 0}' + f'_it{iteration_total}')
    K4_ijklqyz_GJ = np.load(data_folder_path + results_name + f'.npy', allow_pickle=True)

    fig = plt.figure(figsize=(9, 9.0))
    gs = fig.add_gridspec(3, 3, hspace=0.2, wspace=0.2, width_ratios=[1, 1, 0.1],
                          height_ratios=[1, 1, 1])

    gs_stress_G = fig.add_subplot(gs[0, 0])
    gs_stress_GJ = fig.add_subplot(gs[0, 1])
    ax_cbar_stress = fig.add_subplot(gs[0, 2])

    # compute max min stress
    ij = (0, 1)
    max_stress = stress_field_G[..., 0].max()  # ij + (..., 0)
    min_stress = stress_field_G[..., 0].min()
    divnorm = mpl.colors.Normalize(vmin=min_stress, vmax=max_stress)
    cmap_ = mpl.cm.cividis

    pcm = gs_stress_G.pcolormesh(np.tile(stress_field_G[..., 0], (1, 1)),
                                 cmap=cmap_, linewidth=0,
                                 rasterized=True,
                                 # norm=divnorm
                                 )
    pcm = gs_stress_GJ.pcolormesh(np.tile(stress_field_GJ[..., 0], (1, 1)),
                                  cmap=cmap_, linewidth=0,
                                  rasterized=True,
                                  # norm=divnorm
                                  )
    cbar = plt.colorbar(pcm, location='left', cax=ax_cbar_stress)

    ##### plot strains
    gs_tot_strain_G = fig.add_subplot(gs[1, 0])
    gs_tot_strain_GJ = fig.add_subplot(gs[1, 1])
    ax_cbar_tot_strain = fig.add_subplot(gs[1, 2])

    eq_strain_G = total_strain_field_G  # compute_eq_strain(total_strain_field_G)
    eq_strain_GJ = total_strain_field_GJ  # compute_eq_strain(total_strain_field_GJ)

    #
    max_strain = eq_strain_G[..., 0].max()
    min_strain = eq_strain_G[..., 0].min()
    divnorm = mpl.colors.Normalize(vmin=min_strain, vmax=max_strain)
    cmap_ = mpl.cm.cividis

    pcm = gs_tot_strain_G.pcolormesh(np.tile(eq_strain_G[..., 0], (1, 1)),
                                     cmap=cmap_, linewidth=0,
                                     rasterized=True,
                                     # norm=divnorm
                                     )
    pcm = gs_tot_strain_GJ.pcolormesh(np.tile(eq_strain_GJ[..., 0], (1, 1)),
                                      cmap=cmap_, linewidth=0,
                                      rasterized=True,
                                      # norm=divnorm
                                      )
    cbar = plt.colorbar(pcm, location='left', cax=ax_cbar_tot_strain)

    ##### plot material data
    gs_K_G = fig.add_subplot(gs[2, 0])
    gs_K_GJ = fig.add_subplot(gs[2, 1])
    ax_cbar_K = fig.add_subplot(gs[2, 2])

    # compute max min stress
    ijkl = (0, 0, 0, 0)
    max_K = K4_ijklqyz_G[..., 0].max()
    min_K = K4_ijklqyz_G[..., 0].min()
    divnorm = mpl.colors.Normalize(vmin=min_K, vmax=max_K)
    cmap_ = mpl.cm.cividis

    pcm = gs_K_G.pcolormesh(np.tile(K4_ijklqyz_G[..., 0], (1, 1)),
                            cmap=cmap_, linewidth=0,
                            rasterized=True,
                            # norm=divnorm
                            )
    pcm = gs_K_GJ.pcolormesh(np.tile(K4_ijklqyz_GJ[..., 0], (1, 1)),
                             cmap=cmap_, linewidth=0,
                             rasterized=True,
                             # norm=divnorm
                             )
    cbar = plt.colorbar(pcm, location='left', cax=ax_cbar_K)

    plt.show()

quit()

print(np.array(time_G))
print(np.array(time_GJ))

# phase_field = np.load('../exp_data/' + file_data_name + f'.npy', allow_pickle=True)

number_of_pixels = (32, 32, 1)  # (128, 128, 1)  # (32, 32, 1) # (64, 64, 1)  # (128, 128, 1) #
domain_size = [1, 1, 1]
Nx = number_of_pixels[0]
Ny = number_of_pixels[1]
Nz = number_of_pixels[2]

_info_final = np.load(data_folder_path + f'info_log_final.npz', allow_pickle=True)

iteration_total = 0
_info_at_it = np.load(data_folder_path + f'info_log_it{iteration_total}.npz', allow_pickle=True)

print(_info_at_it)
# file_data_name = (
#         f'{script_name_save}_gID{geometry_ID}_T{number_of_pixels[0]}_F{filer_id}_kappa{contrast}.npy')
