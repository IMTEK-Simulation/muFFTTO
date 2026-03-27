import matplotlib as mpl
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from mpl_toolkits.axes_grid1 import make_axes_locatable
import os

from muFFTTO import domain
from muFFTTO import topology_optimization

# Define the dimensions of the 2D array
rows = 25  # or whatever size you want
cols = 25  # or whatever size you want

from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection


### ----- Define the hexagonal grid ----- ###
def make_parallelograms(displ):
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
            # parallelograms.set_edgecolor('face')
    return PatchCollection(parallelograms, cmap='jet', linewidth=0, edgecolor='None', antialiased=False, alpha=1.0)


# Create a random 2D array with 0 and 1
# The probabilities can be adjusted to get a different distribution of bubbles (0) and matrix (1)
array = np.random.choice([0, 1], size=(rows, cols), p=[0.5, 0.5])  # equal probability for 0 and 1
plot_figs = False
domain_size = [1, 1]
problem_type = 'elasticity'
discretization_type = 'finite_element'
element_type = 'linear_triangles'  # 'bilinear_rectangle'##'linear_triangles' #linear_triangles_tilled
formulation = 'small_strain'



fig = plt.figure(figsize=(11, 4.5))
gs = fig.add_gridspec(2, 5, width_ratios=[0.1, 1, 1, 1, 1])
ax0 = fig.add_subplot(gs[0, 0:])
colors = ['red', 'blue', 'green', 'orange', 'purple']
linestyles = ['-', '--', '-.', ':', (0, (3, 1, 1, 1))]
letter_offset = -0.18
etas = [0.005, 0.01, 0.02, 0.05]  #
cg_tol_exponent = 8
soft_phase_exponent = 5
random_init = False

N = 32
for i in np.arange(0, len(etas)):  # np.arange(0.05, 0.5, 0.05):#[0.1 ]: [0.01]
    eta_mult = etas[i]
    weight = 0.1
    script_name = 'exp_paper_TO_exp_2_square' + f'_random_{random_init}' + f'_N_{N}' + f'_cgtol_{cg_tol_exponent}' + f'_soft_{soft_phase_exponent}'

    preconditioner_type = "Green_Jacobi"
    file_folder_path = os.path.dirname(os.path.realpath(__file__))  # script directory
    data_folder_path = file_folder_path + '/exp_data/' + script_name + '/'
    figure_folder_path = file_folder_path + '/figures/' + script_name + '/'
    if not os.path.exists(figure_folder_path):
        os.makedirs(figure_folder_path)
    # name = (
    # f'{optimizer}_muFFTTO_elasticity_{element_type}_{script_name}_N{N}_E_target_{E_target_0}_Poisson_{poison_target}_Poisson0_0.0_w{w_mult:.2f}_eta{eta_mult}_p{p}_bounds={bounds}_FE_NuMPI{cores}_nb_load_cases_{nb_load_cases}_energy_objective_{energy_objective}_random_{random_initial_geometry}')

    name =  data_folder_path + f'{preconditioner_type}' + f'_eta_{eta_mult}'+ f'_w_{weight}'  +'_final' + f'.npy'
    #name = data_folder_path + f'{preconditioner_type}' + f'_eta_{eta_mult}' + '_final' + f'.npy'

    phase_field = np.load(name, allow_pickle=True)

    nb_reps = 1
    if i == 0:
        roll_x = 55
        roll_y = 79
        ax1 = fig.add_subplot(gs[1, i + 1])
        ax0.annotate(text=r'$\eta = $' + f'{eta_mult}' + r'$L$',
                     xy=(18, 1.0),
                     xytext=(23., 0.6),
                     arrowprops=dict(arrowstyle='->',
                                     color=colors[i],
                                     lw=1,
                                     ls='-'),
                     color=colors[i],
                     bbox=dict(
                         facecolor='white',  # background color
                         edgecolor='none',  # border color (set to 'black' if you want a frame)
                         alpha=0.8,  # transparency
                         boxstyle='round,pad=0.3'  # rounded box with padding
                     )

                     )
        ax0.text(letter_offset, 1.05, '(b)', transform=ax1.transAxes)
    if i == 1:
        roll_x = 9
        roll_y = 10
        ax1 = fig.add_subplot(gs[1, i + 1])
        ax0.annotate(text=r'$\eta = $' + f'{eta_mult}' + r'$L$',
                     xy=(20, 0.3),
                     xytext=(12., 0.1),
                     arrowprops=dict(arrowstyle='->',
                                     color=colors[i],
                                     lw=1,
                                     ls='-'),
                     color=colors[i],
                     bbox=dict(
                         facecolor='white',  # background color
                         edgecolor='none',  # border color (set to 'black' if you want a frame)
                         alpha=0.8,  # transparency
                         boxstyle='round,pad=0.3'  # rounded box with padding
                     )
                     )
        ax0.text(letter_offset, 1.05, '(c)', transform=ax1.transAxes)

    elif i == 2:
        roll_x = 28
        roll_y = 95
        ax1 = fig.add_subplot(gs[1, i + 1])
        ax0.annotate(text=r'$\eta = $' + f'{eta_mult}' + r'$L$',
                     xy=(18, 0.85),
                     xytext=(11., 0.4),
                     arrowprops=dict(arrowstyle='->',
                                     color=colors[i],
                                     lw=1,
                                     ls='-'),
                     color=colors[i],
                     bbox=dict(
                         facecolor='white',  # background color
                         edgecolor='none',  # border color (set to 'black' if you want a frame)
                         alpha=0.8,  # transparency
                         boxstyle='round,pad=0.3'  # rounded box with padding
                     )
                     )
        ax0.text(letter_offset, 1.05, '(d)', transform=ax1.transAxes)

    elif i == 3:
        roll_x = 25
        roll_y = 8
        ax1 = fig.add_subplot(gs[1, i + 1])
        ax0.annotate(text=r'$\eta = $' + f'{eta_mult}' + r'$L$',
                     xy=(15, 0.85),
                     xytext=(5., 0.5),
                     arrowprops=dict(arrowstyle='->',
                                     color=colors[i],
                                     lw=1,
                                     ls='-'),
                     color=colors[i],
                     bbox=dict(
                         facecolor='white',  # background color
                         edgecolor='none',  # border color (set to 'black' if you want a frame)
                         alpha=0.8,  # transparency
                         boxstyle='round,pad=0.3'  # rounded box with padding
                     )
                     )
        ax0.text(letter_offset, 1.05, '(e)', transform=ax1.transAxes)

    # phase_field=phase_field**2
    phase_field = np.roll(phase_field, roll_x, axis=1)
    phase_field = np.roll(phase_field, roll_y, axis=0)
    x = np.arange(0, nb_reps * N)
    y = np.arange(0, nb_reps * N)
    X, Y = np.meshgrid(x, y)
    levels = [-0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.1]  # levels,
    # levels =np.arange(-0.1, 1.1,1.2/64)
    # contour = ax1.contourf(np.tile(phase_field, (nb_reps, nb_reps)),levels, cmap='gray_r', vmin=0, vmax=1)
    contour = ax1.pcolormesh(X, Y, np.tile(phase_field, (nb_reps, nb_reps)), cmap='gray_r', vmin=0, vmax=1,
                             linewidth=0, rasterized=True)
    contour.set_edgecolor('face')

    ax1.plot([0, 1], [0, 1], color=colors[i], linestyle=linestyles[i], linewidth=1., transform=ax1.transAxes)

    if i == 0:
        # Colorbar
        # divider = make_axes_locatable(ax1)
        # Prevent shrinking the original image
        # ax1.set_anchor('SE')
        # ax_cb = divider.new_horizontal(size="5%", pad=-0.5)
        # ax_cb = divider.append_axes("left", size="5%", pad=0.5)
        # fig.add_axes(ax_cb)
        pos0 = ax0.get_position()
        pos1 = ax1.get_position()

        ax_cb = fig.add_axes([pos0.x0, pos1.y0, 0.02, pos1.height])
        # ax_cb.invert_xaxis()  # Invert the x-axis to match the positioning
        ax_cb.yaxis.tick_left()
        # cbar = fig.colorbar(contour, cax=ax_cb)

        cbar = fig.colorbar(mpl.cm.ScalarMappable(norm=contour.norm, cmap=contour.cmap), cax=ax_cb,
                            ticks=np.arange(0, 1.2, 0.25))
        ax_cb.set_ylabel(r'Phase $\rho$', rotation=90, labelpad=-65)
        ax_cb.yaxis.set_ticks_position('left')
        # ax_cb.xaxis.set_label_position('left')
    ax1.set_xlabel(r'$\eta = $' + f'{eta_mult}' + r'$L$')

    ax1.set_xlim(0, N - 1)
    ax1.set_ylim(0, N - 1)
    ax1.set_yticklabels([])
    ax1.set_xticklabels([])
    ax1.xaxis.set_ticks_position('none')
    ax1.yaxis.set_ticks_position('none')
    ax1.set_aspect('equal')
    # ax1.axis('off')

    # ax1.hlines(y=32, xmin=0, xmax=N, colors='black', linestyles='--', linewidth=1.)
    # ax1.vlines(x=32, ymin=0, ymax=N, colors='black', linestyles='--', linewidth=1.)

    ax0.plot(np.diag(phase_field), color=colors[i], linestyle=linestyles[i])

    # ax0.grid(axis='x')
number_of_runs = range(1, N)  # use your actual number_of_runs
ax0.set_xticks(number_of_runs, minor=False)
ax0.xaxis.grid(True, color='grey', linestyle='-', linewidth=0.01)
ax0.set_xticklabels([])
ax0.xaxis.set_ticks_position('none')
# ax0.yaxis.set_ticks_position('none')
ax0.set_ylabel(r'Phase $\rho$', rotation=90, labelpad=10)
ax0.set_xlim(0, N - 1)
ax0.text(-0.05, 1.1, '(a)', transform=ax0.transAxes)
# ax0.set_ylim(0, N - 1)
# ax0.set_aspect('equal')
# ax1.set_ylabel(r'Position y')
ax1.yaxis.set_label_position("right")
# Annotate distance with arrow
ax0.annotate(
    "", xy=(0.2658, 1.05), xytext=(0.352, 1.05),
    xycoords="axes fraction", textcoords="axes fraction",
    arrowprops=dict(arrowstyle='<->', color='black')
)
ax0.text(
    0.31, 1.12, "5h",
    ha="center", va="center",
    # bbox=dict(boxstyle="round,pad=0.5", facecolor="white", edgecolor="black"),
    transform=ax0.transAxes
)

fname = figure_folder_path + 'exp1_squares{}'.format('.pdf')
print(('create figure: {}'.format(fname)))
plt.savefig(fname, bbox_inches='tight')
# plt.show()
plt.show()

