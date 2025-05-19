import matplotlib as mpl
import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.animation import FuncAnimation, PillowWriter

from mpl_toolkits.axes_grid1 import make_axes_locatable


"""
@file   plot_result_opt_hexagonal_grid.py

@author Indre Jödicke <indre.joedicke@imtek.uni-freiburg.de>

@date   14 Jan 2021

@brief  Plot result of topology optimization on a hexagonal grid
"""
import numpy as np
import matplotlib.pyplot as plt

import matplotlib as mpl
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection

from mpl_toolkits.axes_grid1 import make_axes_locatable

mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = ['Arial']
mpl.rcParams['font.serif'] = ['Arial']
mpl.rcParams['font.cursive'] = ['Arial']
mpl.rcParams['font.size'] = '10'
mpl.rcParams['legend.fontsize'] = '10'
mpl.rcParams['xtick.labelsize'] = '9'
mpl.rcParams['ytick.labelsize'] = '9'
mpl.rcParams['svg.fonttype'] = 'none'




### ----- Define the hexagonal grid ----- ###
def make_parallelograms(displ, cmap=mpl.cm.jet):
    parallelograms = []
    nx = displ.shape[1] - 1  # number of squares in x direction
    ny = displ.shape[2] - 1  # number of squares in y direction

    for x in range(nx):
        for y in range(ny):
            print(x, y)

            corner_1 = displ[:, x    , y    ]
            corner_2 = displ[:, x + 1, y    ]
            corner_3 = displ[:, x + 1, y + 1]
            corner_4 = displ[:, x    , y + 1]
            corners = np.stack([corner_1, corner_2, corner_3, corner_4],
                               axis=1).T #[corner_1, corner_2, corner_3, corner_4]
            parallelogram = Polygon(corners)
            parallelograms.append(parallelogram)

    return PatchCollection(parallelograms, cmap=cmap, linewidth=0, edgecolor='none')

### ----- Prepare figure ----- ###
fig = plt.figure()

gs = fig.add_gridspec(nrows=1, ncols=1)

ax = fig.add_subplot(gs[0, 0])
ax.set_aspect('equal')
#ax.set_xlabel('Position x')
#ax.set_ylabel('Position y')
ax.set_yticklabels([])
ax.set_xticklabels([])

### ----- Import data ----- ###


for w_mult in [5.00]:  # np.arange(0.1, 1., 0.1):# [1]:
    for eta_mult in [0.01]:  # np.arange(0.05, 0.5, 0.05):#[0.1 ]:
        energy_objective = False
        print(w_mult, eta_mult)
        pixel_size = 0.0078125
        eta = 0.03125  # eta_mult * pixel_size
        N = 1024  # 512
        cores = 90  # 40
        p = 2
        nb_load_cases = 3
        random_initial_geometry = True
        bounds = False
        optimizer = 'lbfg'  # adam
        script_name = 'exp_2D_elasticity_TO_indre_3exp'
        E_target = 0.15
        poison_target = -0.5
        poison_0 = 0.0
        # name = (    f'{optimizer}_muFFTTO_elasticity_{script_name}_N{N}_E_target_0.15_Poisson_-0.5_Poisson0_0.2_w{w_mult}_eta{eta_mult}_p{p}_bounds={bounds}_FE_NuMPI{cores}_nb_load_cases_{nb_load_cases}_energy_objective_{energy_objective}_random_{random_initial_geometry}')
        name = (
            f'{script_name}_N{N}_Et_{E_target}_Pt_{poison_target}_P0_{poison_0}_w{w_mult}_eta{eta_mult}_p{p}_mpi{cores}_nlc_{nb_load_cases}_e_{energy_objective}')
        #xopt_it = np.load(os.path.expanduser('~/exp_data/' + name + f'_it{1 + 1}.npy'), allow_pickle=True)
        phase_field = np.load(os.path.expanduser('~/exp_data/' + name + f'_it{8740}.npy'), allow_pickle=True)
        #phase_field = np.load(os.path.expanduser('~/exp_data/lbfg_elasticity_exp_2D_elasticity_TO_indre_3exp_N32_Et_0.15_Pt_-0.5_P0_0.2_w40.0_eta0.02_p2_NuMPI4_nb_load_cases_4_ener_obj_True_it8.npy'), allow_pickle=True)
        #phase_field = np.load(os.path.expanduser('~/exp_data/exp_2D_elasticity_TO_indre_3exp_N128_Et_0.15_Pt_-0.5_P0_0.2_w4.0_eta0.01_p2_mpi10_nlc_4_e_True_it25.npy'), allow_pickle=True)
        #xopt2_128 = np.load('../exp_data/' + name2_128 + f'xopt_log.npz', allow_pickle=True)

# Import data
#name = 'result_opt_negative_poisson.txt'
#metadata = np.loadtxt(name, skiprows=2, max_rows=1)
nb_grid_pts = [N,N]
Lx, Ly = [1,1]
#data = np.loadtxt(name, skiprows=4)

# Rename data
#phase_ini = data[:, 1].reshape(nb_grid_pts, order='F')
#phase_opt = data[:, 2].reshape(nb_grid_pts, order='F')
phase_opt = phase_field
### ----- Define x-, y- coordinates of hexagonal grid ----- ###
nx = nb_grid_pts[0] + 1
ny = nb_grid_pts[1] + 1
hx = Lx / nb_grid_pts[0]
hy = Ly / nb_grid_pts[1]

displ_x, displ_y = np.mgrid[:nx, :ny]
displ_x = displ_x * hx
displ_y = displ_y * hy
displ = np.stack((displ_x, displ_y))

xmin = np.amin(displ_x)
xmax = np.amax(displ_x)
ymin = np.amin(displ_y)
ymax = np.amax(displ_y)

### ----- Plot phase distribution ----- ###
phase_opt = phase_opt.transpose((1, 0)).flatten(order='F')

# Optimized phase for an assembly of unit cells
nb_tiles=3
nb_cells = [nb_tiles, nb_tiles]
nb_additional_cells = 0
ax.set_aspect('equal')

cell_points_x, cell_points_y = np.mgrid[:nb_cells[0]+2*nb_additional_cells+1,
                                        :nb_cells[1]+1]
cell_points_x = cell_points_x * Lx - nb_additional_cells * Lx
cell_points_y = cell_points_y * Ly
cell_points = np.stack((cell_points_x, cell_points_y))*N
# p = make_parallelograms(cell_points)
# p.set_edgecolor('black')
# p.set_linewidth(0.5)
# p.set_facecolor('none')

# p_movie= make_parallelograms(cell_points)
# p_movie.set_edgecolor('black')
# p_movie.set_linewidth(0.5)
# p_movie.set_facecolor('none')

#ax.add_collection(p)
### ----- Finish plot ----- ###
contour = ax.contourf(np.tile(phase_field, (3, 3)), cmap='gray_r', vmin=0, vmax=1)


# Colorbar
# divider = make_axes_locatable(ax)
#
# ax_cb = divider.new_horizontal(size="5%", pad=0.05)
# fig.add_axes(ax_cb)
#
# # cbar = fig.colorbar(contour, cax=ax_cb)
#
# cbar = fig.colorbar(mpl.cm.ScalarMappable(norm=contour.norm, cmap=contour.cmap), cax=ax_cb,
#                     ticks=np.arange(0, 1.2, 0.2))
# cbar.ax.set_ylabel(r'Phase $\rho$', rotation=90, labelpad=10)
#ax.add_collection(p)

for i in range(3):
    ax.hlines(y=1024 * i, xmin=0, xmax=N * nb_tiles,  colors='white', linestyles='--', linewidth=0.3)
    ax.vlines(x=1024 * i, ymin=0, ymax=N * nb_tiles,  colors='white', linestyles='--', linewidth=0.3)
ax.set_xlim(0, 1024 * nb_tiles)
ax.set_ylim(0, 1024 * nb_tiles)
ax.set_yticklabels([])
ax.set_xticklabels([])
ax.xaxis.set_ticks_position('none')
ax.yaxis.set_ticks_position('none')
# ax.set_yticks([0, 1024, 2*1024, 3*1024])
# ax.set_xticks([0, 1024, 2*1024, 3*1024])
# ax.xaxis.set_ticks(np.arange(0, 4*1024, 1024))
# ax.xaxis.set_major_formatter(mpl.ticker.FormatStrFormatter('%0.0f'))
# ax.yaxis.set_ticks(np.arange(0, 4*1024, 1024))
# ax.yaxis.set_major_formatter(mpl.ticker.FormatStrFormatter('%0.0f'))
# nodal_coordinates[0, 0] * number_of_pixels[0], nodal_coordinates[1, 0] * number_of_pixels[0],
# plt.clim(0, 1)
# Add a colorbar
# Colorbar
ax.text(-0.17, 0.97, '(a)', transform=ax.transAxes)
#
# divider = make_axes_locatable(ax)
# ax_cb = divider.new_horizontal(size="5%", pad=0.05)
# fig.add_axes(ax_cb)
# cbar = fig.colorbar(mpl.cm.ScalarMappable(norm=contour.norm, cmap=contour.cmap), cax=ax_cb,
#                     ticks=np.arange(0, 1.2, 0.2))
# cbar.ax.set_ylabel(r'Phase $\rho$', rotation=90, labelpad=10)
#

src = os.getcwd()  + '/figures/'
fname = src + 'exp3_rect{}'.format('.pdf')
print(('create figure: {}'.format(fname)))
plt.savefig(fname, bbox_inches='tight')


print(('create figure: {}'.format(fname)))  # axes[1, 0].legend(loc='upper right')

fig.savefig(fname, bbox_inches='tight')
plt.show()


plot_figs = False
plot_movie = False
if plot_movie:
    for nb_tiles in [1,3 ]:
        fig = plt.figure()
        ax = plt.axes(xlim=(0, nb_tiles * N), ylim=(0, nb_tiles * N))


        # Animation function to update the image
        def update(i):
            i=i*20
            xopt_it = np.load(os.path.expanduser('~/exp_data/' + name + f'_it{i + 1}.npy'), allow_pickle=True)

            # xopt_it = np.load(os.getcwd() + '/muFFTTO_test/experiments/exp_data/' + name + f'_it{i + 1}.npy',
            #                   allow_pickle=True)
            ax.clear()
            contour = ax.contourf(np.tile(xopt_it, (nb_tiles, nb_tiles)), cmap='jet', vmin=0, vmax=1)

            #ax.imshow(np.tile(xopt_it, (nb_tiles, nb_tiles)), cmap=mpl.cm.Greys, vmin=0, vmax=1)
            ax.set_title('L-BFGS iteration {}'.format(i))

            divider = make_axes_locatable(ax)
            ax_cb = divider.new_horizontal(size="5%", pad=0.05)
            fig.add_axes(ax_cb)

            # cbar = fig.colorbar(contour, cax=ax_cb)

            cbar = fig.colorbar(mpl.cm.ScalarMappable(norm=contour.norm, cmap=contour.cmap), cax=ax_cb,
                                ticks=np.arange(0, 1.2, 0.2))
            cbar.ax.set_ylabel(r'Phase $\rho$', rotation=90, labelpad=10)

            ax.set_aspect('equal')
            ax.set_xlabel(r'Position x')
            ax.set_ylabel(r'Position y')
            ax.set_yticklabels([])
            ax.set_xticklabels([])

            # p_movie = make_parallelograms(cell_points)
            # p_movie.set_edgecolor('black')
            # p_movie.set_linewidth(0.5)
            # p_movie.set_facecolor('none')
            # ax.add_collection(p_movie)
            if nb_tiles>1:
                for i in range(nb_tiles):
                    ax.hlines(y=1024*i, xmin=0, xmax=N*nb_tiles, colors='black', linestyles='-', linewidth=1)
                    ax.vlines(x=1024*i, ymin=0, ymax=N*nb_tiles, colors='black', linestyles='-', linewidth=1)
            ax.set_xlim(0, 1024*nb_tiles)
            ax.set_ylim(0, 1024*nb_tiles)

            # img.set_array(xopt_it)


        # Create animation
        # ani = FuncAnimation(fig, update, frames=xopt.f.norms_f.size-1, blit=False)
        ani = FuncAnimation(fig, update, frames=200, blit=False)

        # Save as a GIF
        ani.save(src + f'/movie{nb_tiles}_exp2_imshow_{name}.gif',
                 writer=PillowWriter(fps=5))

name = (
    f'{script_name}_N{N}_Et_{E_target}_Pt_{poison_target}_P0_{poison_0}_w{w_mult}_eta{eta_mult}_p{p}_mpi{cores}_nlc_{nb_load_cases}_e_{energy_objective}')
# xopt_it = np.load(os.path.expanduser('~/exp_data/' + name + f'_it{1 + 1}.npy'), allow_pickle=True)
phase_field = np.load(os.path.expanduser('~/exp_data/' + name + f'_it{8740}.npy'), allow_pickle=True)

##################################################################--------------------------------------------------------#############################################33

from NuMPI import Optimization
from NuMPI.IO import save_npy, load_npy
import time

from mpi4py import MPI
from muFFTTO import domain
from muFFTTO import solvers
from muFFTTO import topology_optimization
from muFFTTO import microstructure_library

problem_type = 'elasticity'
discretization_type = 'finite_element'
element_type = 'linear_triangles'  # 'bilinear_rectangle'##'linear_triangles' #
formulation = 'small_strain'

domain_size = [1, 1]
number_of_pixels = (1024,1024)
dim = np.size(number_of_pixels)
my_cell = domain.PeriodicUnitCell(domain_size=domain_size,
                                  problem_type=problem_type)

discretization = domain.Discretization(cell=my_cell,
                                       nb_of_pixels_global=number_of_pixels,
                                       discretization_type=discretization_type,
                                       element_type=element_type)

E_0 = 1
poison_0 = 0.0
K_0, G_0 = domain.get_bulk_and_shear_modulus(E=E_0, poison=poison_0)

elastic_C_0 = domain.get_elastic_material_tensor(dim=discretization.domain_dimension,
                                                 K=K_0,
                                                 mu=G_0,
                                                 kind='linear')

material_data_field_C_0 = np.einsum('ijkl,qxy->ijklqxy', elastic_C_0,
                                    np.ones(np.array([discretization.nb_quad_points_per_pixel,
                                                      *discretization.nb_of_pixels])))


preconditioner_fnfnqks = discretization.get_preconditioner_NEW(
    reference_material_data_field_ijklqxyz=material_data_field_C_0)

M_fun = lambda x: discretization.apply_preconditioner_NEW(
    preconditioner_Fourier_fnfnqks=preconditioner_fnfnqks,
    nodal_field_fnxyz=x)

# set up load cases
nb_load_cases = 3
macro_gradients = np.zeros([nb_load_cases, dim, dim])
macro_gradients[0] = np.array([[1.0, 0.],
                               [ 0., .0]])*0.01
macro_gradients[1] = np.array([[  .0, .0],
                               [ .0, 1.0]])*0.01
macro_gradients[2] = np.array([[.0, 0.5],
                               [0.5, .0]])*0.01

left_macro_gradients = np.zeros([nb_load_cases, dim, dim])
left_macro_gradients[0] = np.array([[.0,  .0],
                                    [.0, 1.0]])
left_macro_gradients[1] = np.array([[1.0, .0],
                                    [ .0, .0]])
left_macro_gradients[2] = np.array([[.0, .5],
                                    [0.5, 0.0]])

print('macro_gradients = \n {}'.format(macro_gradients))

# Set up  macroscopic gradients
macro_gradient_fields = np.zeros([nb_load_cases, *discretization.get_gradient_size_field().shape])
for load_case in np.arange(nb_load_cases):
    macro_gradient_fields[load_case] = discretization.get_macro_gradient_field(macro_gradients[load_case])
    stress = np.einsum('ijkl,lk->ij', elastic_C_0, macro_gradients[load_case])
    print('init_stress for load case {} = \n {}'.format(load_case, stress))

##### create target material data

for ration in [0.0 ]:
    poison_target  = -0.5
    G_target_auxet = (3 / 20) * E_0  # (3 / 10) * E_0  #
    # G_target_auxet = (1 / 4) * E_0
    E_target = 2 * G_target_auxet * (1 + poison_target)
    #E_target = 0.5
    # Auxetic metamaterials
    # G_target_auxet = (1 / 4) * E_0  #23   25
    # E_target=2*G_target_auxet*(1+poison_target)
    # test materials


    K_targer, G_target = domain.get_bulk_and_shear_modulus(E=E_target, poison=poison_target)

    elastic_C_target = domain.get_elastic_material_tensor(dim=discretization.domain_dimension,
                                                          K=K_targer,
                                                          mu=G_target,
                                                          kind='linear')
    print('Target elastic tangent = \n {}'.format(domain.compute_Voigt_notation_4order(elastic_C_target)))
    ##### create target stresses
    target_stresses = np.zeros([nb_load_cases, dim, dim])

    displacement_field_load_case = np.zeros([nb_load_cases, *discretization.get_displacement_sized_field().shape])
    adjoint_field_load_case = np.zeros([nb_load_cases, *discretization.get_displacement_sized_field().shape])

    # Auxetic metamaterials
    p = 2
    double_well_depth_test = 1
    energy_objective = False
    norms_sigma = []
    norms_pf = []
    num_iteration_ = []


    for w_mult in [1., ]:
        for eta_mult in [0.01   , ]:
            pixel_diameter = np.sqrt(np.sum(discretization.pixel_size ** 2))
            w = w_mult / nb_load_cases  # / discretization.pixel_size[0]
            eta = eta_mult  # * discretization.pixel_size[0]  # pixel_diameter#

            def objective_function_multiple_load_cases(phase_field_1nxyz):
                # reshape the field
                phase_field_1nxyz = phase_field_1nxyz.reshape([1, 1, *discretization.nb_of_pixels])

                # objective function phase field terms
                f_phase_field = topology_optimization.objective_function_phase_field(discretization=discretization,
                                                                                     phase_field_1nxyz=phase_field_1nxyz,
                                                                                     eta=eta,
                                                                                     double_well_depth=double_well_depth_test)
                #  sensitivity phase field terms
                s_phase_field = topology_optimization.sensitivity_phase_field_term_FE_NEW(discretization=discretization,
                                                                                          material_data_field_ijklqxyz=material_data_field_C_0,
                                                                                          phase_field_1nxyz=phase_field_1nxyz,
                                                                                          p=p,
                                                                                          eta=eta,
                                                                                          double_well_depth=1)
                objective_function = f_phase_field

                norms_pf.append(objective_function)
                # Material data in quadrature points
                phase_field_at_quad_poits_1qnxyz = \
                    discretization.evaluate_field_at_quad_points(nodal_field_fnxyz=phase_field_1nxyz,
                                                                 quad_field_fqnxyz=None,
                                                                 quad_points_coords_iq=None)[0]

                material_data_field_C_0_rho_ijklqxyz = material_data_field_C_0[..., :, :, :] * np.power(
                    phase_field_at_quad_poits_1qnxyz, p)[0, :, 0, ...]

                K_diag_alg = discretization.get_preconditioner_Jacoby_fast(
                    material_data_field_ijklqxyz=material_data_field_C_0_rho_ijklqxyz)
                M_fun = lambda x: K_diag_alg * discretization.apply_preconditioner_NEW(
                    preconditioner_Fourier_fnfnqks=preconditioner_fnfnqks,
                    nodal_field_fnxyz=K_diag_alg * x)

                K_fun = lambda x: discretization.apply_system_matrix(
                    material_data_field=material_data_field_C_0_rho_ijklqxyz,
                    displacement_field=x,
                    formulation='small_strain')
                # Solve mechanical equilibrium constrain
                homogenized_stresses = np.zeros([nb_load_cases, dim, dim])

                f_sigmas = np.zeros([nb_load_cases, 1])
                f_sigmas_energy = np.zeros([nb_load_cases, 1])
                adjoint_energies = np.zeros([nb_load_cases, 1])
                s_stress_and_adjoint_load_cases = np.zeros([nb_load_cases, *s_phase_field.shape])
                s_energy_and_adjoint_load_cases = np.zeros([nb_load_cases, *s_phase_field.shape])
                for load_case in np.arange(nb_load_cases):
                    rhs_load_case = discretization.get_rhs(
                        material_data_field_ijklqxyz=material_data_field_C_0_rho_ijklqxyz,
                        macro_gradient_field_ijqxyz=macro_gradient_fields[load_case])
                    # if MPI.COMM_WORLD.size == 1:
                    #     print('rhs Of = {}'.format(np.linalg.norm(rhs_load_case)))

                    displacement_field_load_case[load_case], norms = solvers.PCG(Afun=K_fun,
                                                                                 B=rhs_load_case,
                                                                                 x0=displacement_field_load_case[load_case],
                                                                                 P=M_fun,
                                                                                 steps=int(10000),
                                                                                 toler=1e-10)
                    if MPI.COMM_WORLD.rank == 0:
                        nb_it_comb = len(norms['residual_rz'])
                        norm_rz = norms['residual_rz'][-1]
                        norm_rr = norms['residual_rr'][-1]
                        num_iteration_.append(nb_it_comb)

                        print(
                            'load case ' f'{load_case},  nb_ steps CG of =' f'{nb_it_comb}, residual_rz = {norm_rz}, residual_rr = {norm_rr}')
                        # compute homogenized stress field corresponding t
                    homogenized_stresses[load_case] = discretization.get_homogenized_stress(
                        material_data_field_ijklqxyz=material_data_field_C_0_rho_ijklqxyz,
                        displacement_field_fnxyz=displacement_field_load_case[load_case],
                        macro_gradient_field_ijqxyz=macro_gradient_fields[load_case],
                        formulation='small_strain')


                    f_sigmas[load_case] = w * (topology_optimization.compute_stress_equivalence_potential(
                        actual_stress_ij=homogenized_stresses[load_case],
                        target_stress_ij=target_stresses[load_case]))
                    # if MPI.COMM_WORLD.rank == 0:
                    #     print('w*f_sigmas  = '          ' {} '.format(f_sigmas[load_case]))  # good in MPI
                    #     print('sum of w*f_sigmas  = '          ' {} '.format(np.sum(f_sigmas)))
                    s_stress_and_adjoint_load_cases[load_case], adjoint_field_load_case[
                        load_case], adjoint_energies[
                        load_case] = topology_optimization.sensitivity_stress_and_adjoint_FE_NEW(
                        discretization=discretization,
                        material_data_field_ijklqxyz=material_data_field_C_0,
                        displacement_field_fnxyz=displacement_field_load_case[load_case],
                        adjoint_field_last_step_fnxyz=adjoint_field_load_case[load_case],
                        macro_gradient_field_ijqxyz=macro_gradient_fields[load_case],
                        phase_field_1nxyz=phase_field_1nxyz,
                        target_stress_ij=target_stresses[load_case],
                        actual_stress_ij=homogenized_stresses[load_case],
                        preconditioner_fun=M_fun,
                        system_matrix_fun=K_fun,
                        formulation='small_strain',
                        p=p,
                        weight=w)
                    s_phase_field += s_stress_and_adjoint_load_cases[load_case]

                    f_sigmas[load_case] += adjoint_energies[load_case]

                    objective_function += f_sigmas[load_case]

                norms_sigma.append(objective_function)
                return objective_function[0], s_phase_field.reshape(-1)


phase_field_a=np.expand_dims(np.expand_dims(phase_field, axis=0), axis=0)
print('Init objective function FE  = {}'.format(objective_function_multiple_load_cases(phase_field_a)[0]))




# Define the dimensions of the 2D array
rows = 25  # or whatever size you want
cols = 25  # or whatever size you want

# Create a random 2D array with 0 and 1
# The probabilities can be adjusted to get a different distribution of bubbles (0) and matrix (1)
array = np.random.choice([0, 1], size=(rows, cols), p=[0.5, 0.5])  # equal probability for 0 and 1

for w_mult in [5.00]:  # np.arange(0.1, 1., 0.1):# [1]:
    for eta_mult in [0.01]:  # np.arange(0.05, 0.5, 0.05):#[0.1 ]:
        energy_objective = False
        print(w_mult, eta_mult)
        pixel_size = 0.0078125
        eta = 0.03125  # eta_mult * pixel_size
        N = 1024  # 512
        cores = 90  # 40
        p = 2
        nb_load_cases = 3
        random_initial_geometry = True
        bounds = False
        optimizer = 'lbfg'  # adam
        script_name = 'exp_2D_elasticity_TO_indre_3exp'
        E_target = 0.15
        poison_target = -0.5
        poison_0 = 0.0
        # name = (    f'{optimizer}_muFFTTO_elasticity_{script_name}_N{N}_E_target_0.15_Poisson_-0.5_Poisson0_0.2_w{w_mult}_eta{eta_mult}_p{p}_bounds={bounds}_FE_NuMPI{cores}_nb_load_cases_{nb_load_cases}_energy_objective_{energy_objective}_random_{random_initial_geometry}')
        name = (
            f'{script_name}_N{N}_Et_{E_target}_Pt_{poison_target}_P0_{poison_0}_w{w_mult}_eta{eta_mult}_p{p}_mpi{cores}_nlc_{nb_load_cases}_e_{energy_objective}')
        #xopt_it = np.load(os.path.expanduser('~/exp_data/' + name + f'_it{1 + 1}.npy'), allow_pickle=True)
    if plot_figs:
        phase_field = np.load(os.path.expanduser('~/exp_data/' + name + f'_it{8740}.npy'), allow_pickle=True)

        src = os.getcwd()  # + '/muFFTTO_test/experiments/figures/'  # source folder\
        fig_data_name = f'muFFTTO_{name}_geom'  # print('rank' f'{MPI.COMM_WORLD.rank:6} ')

        fig, ax = plt.subplots(1, 1)
        ax.set_aspect('equal')
        ax.set_xlabel('Position x')
        ax.set_ylabel('Position y')
        ax.set_yticklabels([])
        ax.set_xticklabels([])

        contour = ax.contourf(np.tile(phase_field, (3, 3)), cmap='jet', vmin=0, vmax=1)
        # nodal_coordinates[0, 0] * number_of_pixels[0], nodal_coordinates[1, 0] * number_of_pixels[0],
        # plt.clim(0, 1)
        # Add a colorbar
        # Colorbar


        divider = make_axes_locatable(ax)
        ax_cb = divider.new_horizontal(size="5%", pad=0.05)
        fig.add_axes(ax_cb)
        cbar = fig.colorbar(mpl.cm.ScalarMappable(norm=contour.norm, cmap=contour.cmap), cax=ax_cb,  ticks=np.arange(0, 1.2, 0.2))
        cbar.ax.set_ylabel(r'Phase $\rho$', rotation=90, labelpad=10)
        #cbar.ax.set_ylabel(r'Phase $\rho$', rotation=90, labelpad=10)

        #cbar = plt.colorbar(contour, boundaries=np.linspace(0, 1))
        #cbar = fig.colorbar(mpl.cm.ScalarMappable(norm=contour.norm, cmap=contour.cmap), ticks=np.arange(0, 1, 0.2), ax=ax)

        # cbar.set_label("Value")
        # cbar.set_clim(0, 1)  # Ensure colorbar matches contou
        # cbar.set_clim(vmin=0, vmax=1)
          # contour.set_clim(vmin=0, vmax=1)contour
        # plt.colorbar()
        # ax[0].set_axis_off()
        # ax[0].text(-0.1, 0.5, f'target =\n {xopt.f.target_stress0}\n,'
        #                       f' optimized =\n {xopt.f.homogenized_stresses0}\n,'
        #                       f' diff = \n{xopt.f.homogenized_stresses0 - xopt.f.target_stress0}')

        fname = src + fig_data_name + '{}'.format('1.png')
        print(('create figure: {}'.format(fname)))  # axes[1, 0].legend(loc='upper right')
        plt.savefig(fname, bbox_inches='tight')
        plt.show()
        quit()

        # phase_field = np.load(os.getcwd() +'/muFFTTO_test/experiments/exp_data/' + name + '.npy', allow_pickle=True, max_header_size=100000)

        xopt = np.load(os.getcwd() + '/muFFTTO_test/experiments/exp_data/' + name + 'xopt_log.npz', allow_pickle=True,
                       max_header_size=100000)

        src = os.getcwd() + '/muFFTTO_test/experiments/figures/'  # source folder\
        fig_data_name = f'muFFTTO_{name}'  # print('rank' f'{MPI.COMM_WORLD.rank:6} ')

        fig, ax = plt.subplots(1, 2)
        plt.contourf(np.tile(phase_field, (3, 3)), cmap=mpl.cm.Greys)
        # nodal_coordinates[0, 0] * number_of_pixels[0], nodal_coordinates[1, 0] * number_of_pixels[0],
        plt.clim(0, 1)
        plt.colorbar()
        ax[0].set_axis_off()
        ax[0].text(-0.1, 0.5, f'target =\n {xopt.f.target_stress0}\n,'
                              f' optimized =\n {xopt.f.homogenized_stresses0}\n,'
                              f' diff = \n{xopt.f.homogenized_stresses0 - xopt.f.target_stress0}')

        fname = src + fig_data_name + '{}'.format('1.png')
        print(('create figure: {}'.format(fname)))  # axes[1, 0].legend(loc='upper right')
        plt.savefig(fname, bbox_inches='tight')
        # plt.show()

        plt.figure()
        plt.contourf(np.tile(phase_field, (1, 1)), cmap=mpl.cm.Greys)
        # plt.title(f'w = {w_mult},eta= {eta_mult}\n, {xopt.f.homogenized_stresses}')

        # nodal_coordinates[0, 0] * number_of_pixels[0], nodal_coordinates[1, 0] * number_of_pixels[0],
        plt.clim(0, 1)
        plt.colorbar()
        # plt.title(f'w = {w},eta= {eta_mult}\n, {xopt.f.message}')
        fname = src + fig_data_name + '{}'.format('2.png')
        print(('create figure: {}'.format(fname)))  # axes[1, 0].legend(loc='upper right')
        plt.savefig(fname, bbox_inches='tight')
        plt.show()
        plt.figure()
        fig_data_name = f'muFFTTO_{phase_field.shape}_line'  # print('rank' f'{MPI.COMM_WORLD.rank:6} ')

        plt.plot(np.tile(phase_field, (1, 1))[:, 3].transpose())
        # nodal_coordinates[0, 0] * number_of_pixels[0], nodal_coordinates[1, 0] * number_of_pixels[0],
        plt.grid(True)
        plt.minorticks_on()
        fname = src + fig_data_name + '{}'.format('3.png')
        print(('create figure: {}'.format(fname)))  # axes[1, 0].legend(loc='upper right')
        # plt.savefig(fname, bbox_inches='tight')
        # plt.show()
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
        fname = src + fig_data_name + '{}'.format('4.png')
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
        # plt.show()
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
        # plt.show()
if plot_movie:
    for nb_tiles in [1, 3]:
        fig = plt.figure()
        ax = plt.axes(xlim=(0, nb_tiles * N), ylim=(0, nb_tiles * N))


        # Animation function to update the image
        def update(i):
            xopt_it = np.load(os.getcwd() + '/muFFTTO_test/experiments/exp_data/' + name + f'_it{i + 1}.npy',
                              allow_pickle=True)
            ax.clear()
            ax.imshow(np.tile(xopt_it, (nb_tiles, nb_tiles)), cmap=mpl.cm.Greys, vmin=0, vmax=1)

            ax.set_title('iteration {}'.format(i))
            # img.set_array(xopt_it)


        # Create animation
        # ani = FuncAnimation(fig, update, frames=xopt.f.norms_f.size-1, blit=False)
        ani = FuncAnimation(fig, update, frames=8420, blit=False)

        # Save as a GIF
        ani.save(os.getcwd() + f'/muFFTTO_test/experiments/figures/movie{nb_tiles}_exp2_imshow_{name}.gif',
                 writer=PillowWriter(fps=50))

    xopt_it = np.load(os.getcwd() + '/muFFTTO_test/experiments/exp_data/' + name + f'_it{8420}.npy', allow_pickle=True)
    for nb_tiles in [1, 3]:
        fig = plt.figure()
        ax = plt.axes(xlim=(0, nb_tiles * N), ylim=(0, nb_tiles * N))
        ax.imshow(np.tile(xopt_it, (nb_tiles, nb_tiles)), cmap=mpl.cm.Greys, vmin=0, vmax=1)

        # fname = src + fig_data_name + '{}'.format('os.getcwd() +f'/.png')
        # print(('create figure: {}'.format(fname)))  # axes[1, 0].legend(loc='upper right')
        plt.savefig(os.getcwd() + f'/muFFTTO_test/experiments/figures/result{nb_tiles}_exp2_imshow_{name}.png',
                    bbox_inches='tight')
