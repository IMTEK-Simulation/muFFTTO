import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from scipy.constants import precision

# plot geometries
from muFFTTO import domain
from muFFTTO import solvers
from muFFTTO import microstructure_library


def scale_field(field, min_val, max_val):
    """Scales a 2D random field to be within [min_val, max_val]."""
    field_min, field_max = field.min(), field.max()
    scaled_field = (field - field_min) / (field_max - field_min)  # Normalize to [0,1]
    return scaled_field * (max_val - min_val) + min_val  # Scale to [min_val, max_val]


problem_type = 'elasticity'
discretization_type = 'finite_element'
element_type = 'linear_triangles'
formulation = 'small_strain'

domain_size = [1, 1]

src = '../figures/'

#### numbers of iterations
################## ################## LINEAR  FOR 10^6 precision ################## ################## ##################
nb_it_Green_linear_2_e06 = np.array(
    [[13., 0., 0., 0., 0., 0., 0., 0., 0.],
     [19., 22., 0., 0., 0., 0., 0., 0., 0.],
     [20., 30., 32., 0., 0., 0., 0., 0., 0.],
     [21., 30., 41., 43., 0., 0., 0., 0., 0.],
     [22., 32., 43., 52., 52., 0., 0., 0., 0.],
     [22., 33., 43., 53., 59., 59., 0., 0., 0.],
     [22., 33., 43., 54., 61., 65., 66., 0., 0.],
     [23., 33., 43., 55., 64., 67., 70., 69., 0.],
     [24., 34., 45., 56., 65., 69., 73., 74., 73.]])

nb_it_Jacobi_linear_2_e06 = np.array(
    [[1., 0., 0., 0., 0., 0., 0., 0., 0.],
     [1., 1., 0., 0., 0., 0., 0., 0., 0.],
     [1., 1., 1., 0., 0., 0., 0., 0., 0.],
     [1., 1., 1., 1., 0., 0., 0., 0., 0.],
     [1., 1., 1., 1., 1., 0., 0., 0., 0.],
     [1., 1., 1., 1., 1., 1., 0., 0., 0.],
     [1., 1., 1., 1., 1., 1., 1., 0., 0.],
     [1., 1., 1., 1., 1., 1., 1., 1., 0.],
     [1., 1., 1., 1., 1., 1., 1., 1., 1.]])

nb_it_combi_linear_2_e06 = np.array([[10., 0., 0., 0., 0., 0., 0., 0., 0.],
                                     [15., 14., 0., 0., 0., 0., 0., 0., 0.],
                                     [21., 20., 19., 0., 0., 0., 0., 0., 0.],
                                     [30., 28., 29., 27., 0., 0., 0., 0., 0.],
                                     [50., 45., 43., 44., 40., 0., 0., 0., 0.],
                                     [78., 73., 71., 70., 67., 64., 0., 0., 0.],
                                     [138., 129., 121., 116., 118., 117.,  109., 0., 0.],
                                     [252., 236., 220., 214., 205., 212.,  209., 197., 0.],
                                     [499., 446., 420., 403., 398., 390.,  382., 379., 365.]])

nb_it_Green_linear_4_e06 = np.array([[16., 0., 0., 0., 0., 0., 0., 0., 0.],
                                     [26., 29., 0., 0., 0., 0., 0., 0., 0.],
                                     [30., 46., 51., 0., 0., 0., 0., 0., 0.],
                                     [33., 49., 73., 75., 0., 0., 0., 0., 0.],
                                     [33., 52., 75., 109., 117., 0., 0., 0., 0.],
                                     [35., 52., 79., 115., 164., 173., 0., 0., 0.],
                                     [35., 53., 79., 115., 165., 234., 256., 0., 0.],
                                     [35., 53., 79., 116., 168., 234., 313. ,338. , 0.],
                                     [36., 54., 80., 116., 168.,  234., 321., 420., 460.]])

nb_it_Jacobi_linear_4_e06 = np.array(
    [[1., 0., 0., 0., 0., 0., 0., 0., 0.],
     [1., 1., 0., 0., 0., 0., 0., 0., 0.],
     [1., 1., 1., 0., 0., 0., 0., 0., 0.],
     [1., 1., 1., 1., 0., 0., 0., 0., 0.],
     [1., 1., 1., 1., 1., 0., 0., 0., 0.],
     [1., 1., 1., 1., 1., 1., 0., 0., 0.],
     [1., 1., 1., 1., 1., 1., 1., 0., 0.],
     [1., 1., 1., 1., 1., 1., 1., 1., 0.],
     [1., 1., 1., 1., 1., 1., 1., 1., 1.]])

nb_it_combi_linear_4_e06 = np.array([[12., 0., 0., 0., 0., 0., 0., 0., 0.],
                                     [19., 17., 0., 0., 0., 0., 0., 0., 0.],
                                     [29., 28., 25., 0., 0., 0., 0., 0., 0.],
                                     [45., 42., 42., 36., 0., 0., 0., 0., 0.],
                                     [75., 72., 65., 69., 56., 0., 0., 0., 0.],
                                     [130., 126., 117., 112., 117., 90., 0., 0., 0.],
                                     [230., 229., 212., 200., 205., 216.,153., 0., 0.],
                                     [447., 428., 406., 378., 386., 388.,  394., 267., 0.],
                                     [852., 845., 801., 727., 734., 744., 757., 740., 502.]])
################## ################## ##################  SINE WAVE FOR 10^6 precision   ################## ################## ##################
nb_it_Green_sine_wave_0_e06 = np.array(
    [[8., 0., 0., 0., 0., 0., 0., 0., 0.],
     [11., 13., 0., 0., 0., 0., 0., 0., 0.],
     [12., 18., 21., 0., 0., 0., 0., 0., 0.],
     [13., 19., 31., 36., 0., 0., 0., 0., 0.],
     [13., 20., 33., 54., 63., 0., 0., 0., 0.],
     [13., 21., 35., 61., 89., 100., 0., 0., 0.],
     [14., 22., 36., 63., 91., 125., 131., 0., 0.],
     [15., 23., 40., 67., 98., 157., 177., 193., 0.],
     [14., 24., 41., 69., 101., 168., 224., 247., 253.]])

nb_it_Jacobi_sine_wave_0_e06 = np.array(
    [[1., 0., 0., 0., 0., 0., 0., 0., 0.],
     [1., 1., 0., 0., 0., 0., 0., 0., 0.],
     [1., 1., 1., 0., 0., 0., 0., 0., 0.],
     [1., 1., 1., 1., 0., 0., 0., 0., 0.],
     [1., 1., 1., 1., 1., 0., 0., 0., 0.],
     [1., 1., 1., 1., 1., 1., 0., 0., 0.],
     [1., 1., 1., 1., 1., 1., 1., 0., 0.],
     [1., 1., 1., 1., 1., 1., 1., 1., 0.],
     [1., 1., 1., 1., 1., 1., 1., 1., 1.]])

nb_it_combi_sine_wave_0_e06 = np.array(
    [[7., 0., 0., 0., 0., 0., 0., 0., 0.],
     [9., 8., 0., 0., 0., 0., 0., 0., 0.],
     [11., 10., 9., 0., 0., 0., 0., 0., 0.],
     [15., 13., 11., 10., 0., 0., 0., 0., 0.],
     [19., 17., 14., 11., 11., 0., 0., 0., 0.],
     [28., 25., 20., 16., 10., 10., 0., 0., 0.],
     [43., 34., 30., 24., 17., 10., 10., 0., 0.],
     [59., 61., 50., 36., 27., 18., 10., 10., 0.],
     [119., 96., 81., 59., 40., 27., 22., 10., 10.]])

nb_it_Green_sine_wave_4_e06 = np.array(
    [[8., 0., 0., 0., 0., 0., 0., 0., 0.],
     [11., 12., 0., 0., 0., 0., 0., 0., 0.],
     [13., 17., 21., 0., 0., 0., 0., 0., 0.],
     [25., 20., 31., 38., 0., 0., 0., 0., 0.],
     [25., 38., 35., 50., 61., 0., 0., 0., 0.],
     [26., 39., 64., 99., 83., 96., 0., 0., 0.],
     [27., 41., 65., 103., 151., 133., 126., 0., 0.],
     [28., 42., 68., 109., 156., 199., 169., 173., 0.],
     [27., 42., 71., 113., 159., 216., 206., 207., 205.]])

nb_it_Jacobi_sine_wave_4_e06 = np.array(
    [[1., 0., 0., 0., 0., 0., 0., 0., 0.],
     [1., 1., 0., 0., 0., 0., 0., 0., 0.],
     [1., 1., 1., 0., 0., 0., 0., 0., 0.],
     [1., 1., 1., 1., 0., 0., 0., 0., 0.],
     [1., 1., 1., 1., 1., 0., 0., 0., 0.],
     [1., 1., 1., 1., 1., 1., 0., 0., 0.],
     [1., 1., 1., 1., 1., 1., 1., 0., 0.],
     [1., 1., 1., 1., 1., 1., 1., 1., 0.],
     [1., 1., 1., 1., 1., 1., 1., 1., 1.]])

nb_it_combi_sine_wave_4_e06 = np.array(
    [[7., 0., 0., 0., 0., 0., 0., 0., 0.],
     [11., 7., 0., 0., 0., 0., 0., 0., 0.],
     [17., 11., 9., 0., 0., 0., 0., 0., 0.],
     [24., 15., 11., 10., 0., 0., 0., 0., 0.],
     [35., 26., 16., 11., 11., 0., 0., 0., 0.],
     [72., 38., 26., 16., 10., 10., 0., 0., 0.],
     [114., 64., 39., 27., 18., 10., 9., 0., 0.],
     [197., 115., 67., 41., 27., 17., 10., 9., 0.],
     [384., 205., 118., 67., 43., 28., 16., 9., 8.]])

#
nb_pix_multips = [2, 3, 4, 5, 6, 7, 8, 9, 10]
Nx = (np.asarray(nb_pix_multips))
X, Y = np.meshgrid(Nx, Nx, indexing='ij')

nb_pix_multips = [2, 4, 5, 6, 7, 8]
# material distribution
geometry_ID = 'linear'  # linear  sine_wave_
# rhs = 'sin_wave'
rhs = False
linestyles = ['-', '--', ':', '-.', '--', ':', '-.']
colors = ['red', 'blue', 'green', 'orange', 'purple', 'orange', 'purple']
precc = 6
fig = plt.figure(figsize=(11, 7.0))
gs = fig.add_gridspec(2, 4, hspace=0.2, wspace=0.25, width_ratios=[1.2, 1.2, 1.2, 0.03],
                      height_ratios=[1, 1])
row = 0
for phase_contrast in [2, 4]:  # 1, 4
    ratio = phase_contrast
    if geometry_ID == 'linear':
        divnorm = mpl.colors.Normalize(vmin=0, vmax=500)
        white_lim = 250
    elif geometry_ID == 'sine_wave_':
        divnorm = mpl.colors.Normalize(vmin=0, vmax=200)
        white_lim = 100
    # Green graph
    gs0 = gs[row, 0].subgridspec(1, 1, wspace=0.1, width_ratios=[1])
    ax = fig.add_subplot(gs0[0, 0])
    # ax.set_aspect('equal')
    if phase_contrast == 2:
        if geometry_ID == 'linear':
            if precc == 6:
                nb_iterations = (nb_it_Green_linear_2_e06[:, :])
        elif geometry_ID == 'sine_wave_':
            if precc == 6:
                nb_iterations = nb_it_Green_sine_wave_0_e06[:, :]
    elif phase_contrast == 4:
        if geometry_ID == 'linear':
            if precc == 6:
                nb_iterations = nb_it_Green_linear_4_e06[:, :]
        elif geometry_ID == 'sine_wave_':
            if precc == 6:
                nb_iterations = (nb_it_Green_sine_wave_4_e06[:, :])

    nb_iterations = np.nan_to_num(nb_iterations, nan=1.0)
    for i in range(nb_iterations.shape[0]):
        for j in range(nb_iterations.shape[1]):
            if nb_iterations[i, j] == 0:
                pass
            elif nb_iterations[i, j] < white_lim:
                ax.text(i + Nx[0], j + Nx[0], f'{nb_iterations[i, j]:.0f}', size=8,
                        ha='center', va='center', color='black')
            else:
                ax.text(i + Nx[0], j + Nx[0], f'{nb_iterations[i, j]:.0f}', size=8,
                        ha='center', va='center', color='white')

    pcm = ax.pcolormesh(X, Y, nb_iterations, label='PCG: Green + Jacobi', cmap='Reds', norm=divnorm)

    # ax.text(0.05, 0.92, f'Total phase contrast $\kappa=10^{phase_contrast}$', transform=ax.transAxes)
    if geometry_ID == 'sine_wave_' and phase_contrast == 2:
        ax.text(0.05, 0.82, f'Total phase contrast \n' + r'$\kappa=\infty$', transform=ax.transAxes)
    elif geometry_ID == 'sine_wave_':
        ax.text(0.05, 0.82, f'Total phase contrast \n $\kappa=10^{{{-phase_contrast}}}$', transform=ax.transAxes)
    else:
        ax.text(0.05, 0.82, f'Total phase contrast \n $\kappa=10^{phase_contrast}$', transform=ax.transAxes)

    if row == 0:
        ax.set_title('Total number of iteration \n Green ')
    # ax.set_zlim(1 ,100)
    # ax.set_ylabel('# data/geometry sampling points (x direction)')

    # ax.yaxis.set_label_position('right')
    # ax.yaxis.tick_right()
    ax.set_ylabel('# of material phases')
    if row == 1:
        ax.set_xlabel('# of nodal points (x direction)')
    ax.set_xticks(Nx)
    ax.set_xticklabels([f'$2^{{{i}}}$' for i in Nx])
    # ax2 = ax.twinx()
    ax.set_yticks(Nx)
    ax.set_yticklabels([f'$2^{{{i}}}$' for i in Nx])
    ax.tick_params(right=True, top=False, labelright=False, labeltop=False, labelrotation=0)
    #    ax.set_aspect('equal')
    if row == 0:
        ax.text(-0.25, 1.15, f'(a.{row + 1})', transform=ax.transAxes)
    elif row == 1:
        ax.text(-0.25, 1.05, f'(a.{row + 1})', transform=ax.transAxes)

    # jacobi  graph
    gs1 = gs[row, 1].subgridspec(1, 1, wspace=0.1, width_ratios=[5])
    ax = fig.add_subplot(gs1[0, 0])
    #    ax.set_aspect('equal')
    if phase_contrast == 2:
        if geometry_ID == 'linear':
            if precc == 6:
                nb_iterations = (nb_it_Jacobi_linear_2_e06[:, :])
        elif geometry_ID == 'sine_wave_':
            if precc == 6:
                nb_iterations = nb_it_Jacobi_sine_wave_0_e06[:, :]
    elif phase_contrast == 4:
        if geometry_ID == 'linear':
            if precc == 6:
                nb_iterations = nb_it_Jacobi_linear_4_e06[:, :]
        elif geometry_ID == 'sine_wave_':
            if precc == 6:
                nb_iterations = (nb_it_Jacobi_sine_wave_4_e06[:, :])

    nb_iterations = np.nan_to_num(nb_iterations, nan=1.0)
    for i in range(nb_iterations.shape[0]):
        for j in range(nb_iterations.shape[1]):
            if nb_iterations[i, j] == 0:
                pass
            elif nb_iterations[i, j] < white_lim:
                ax.text(i + Nx[0], j + Nx[0], f'{nb_iterations[i, j]:.0f}', size=8,
                        ha='center', va='center', color='black')
            else:
                ax.text(i + Nx[0], j + Nx[0], f'{nb_iterations[i, j]:.0f}', size=8,
                        ha='center', va='center', color='white')

    pcm = ax.pcolormesh(X, Y, nb_iterations, label='PCG: Green + Jacobi', cmap='Reds', norm=divnorm)

    # ax.text(0.05, 0.92, f'Total phase contrast $\kappa=10^{phase_contrast}$', transform=ax.transAxes)
    if geometry_ID == 'sine_wave_' and phase_contrast == 2:
        ax.text(0.05, 0.82, f'Total phase contrast \n' + r'$\kappa=\infty$', transform=ax.transAxes)
    elif geometry_ID == 'sine_wave_':
        ax.text(0.05, 0.82, f'Total phase contrast \n $\kappa=10^{{{-phase_contrast}}}$', transform=ax.transAxes)
    else:
        ax.text(0.05, 0.82, f'Total phase contrast \n $\kappa=10^{phase_contrast}$', transform=ax.transAxes)

    if row == 0:
        ax.set_title('Total number of iteration \n Jacobi ')
    # ax.set_zlim(1 ,100)
    # ax.set_ylabel('# data/geometry sampling points (x direction)')

    # ax.yaxis.set_label_position('right')
    # ax.yaxis.tick_right()
    if row == 1:
        ax.set_xlabel('# of nodal points (x direction)')
    ax.set_xticks(Nx)
    ax.set_xticklabels([f'$2^{{{i}}}$' for i in Nx])
    # ax2 = ax.twinx()
    ax.set_yticks(Nx)
    ax.set_yticklabels([f'$2^{{{i}}}$' for i in Nx])
    ax.tick_params(right=True, top=False, labelright=False, labeltop=False, labelrotation=0)
    #    ax.set_aspect('equal')

    # ax.set_zlabel('# CG iterations')
    if row == 0:
        ax.text(-0.15, 1.15, f'(b.{row + 1})', transform=ax.transAxes)
    elif row == 1:
        ax.text(-0.15, 1.05, f'(b.{row + 1})', transform=ax.transAxes)
    # plot Jacobi green
    gs2 = gs[row, 2].subgridspec(1, 1, wspace=0.1, width_ratios=[5])
    ax = fig.add_subplot(gs2[0, 0])
    if phase_contrast == 2:
        if geometry_ID == 'linear':
            if precc == 6:
                nb_iterations = (nb_it_combi_linear_2_e06[:, :])
        elif geometry_ID == 'sine_wave_':
            if precc == 6:
                nb_iterations = nb_it_combi_sine_wave_0_e06[:, :]
    elif phase_contrast == 4:
        if geometry_ID == 'linear':
            if precc == 6:
                nb_iterations = nb_it_combi_linear_4_e06[:, :]
        elif geometry_ID == 'sine_wave_':
            if precc == 6:
                nb_iterations = (nb_it_combi_sine_wave_4_e06[:, :])
    nb_iterations = np.nan_to_num(nb_iterations, nan=1.0)
    for i in range(nb_iterations.shape[0]):
        for j in range(nb_iterations.shape[1]):
            if nb_iterations[i, j] == 0:
                pass
            elif nb_iterations[i, j] < white_lim:
                ax.text(i + Nx[0], j + Nx[0], f'{nb_iterations[i, j]:.0f}', size=8,
                        ha='center', va='center', color='black')
            else:
                ax.text(i + Nx[0], j + Nx[0], f'{nb_iterations[i, j]:.0f}', size=8,
                        ha='center', va='center', color='white')
    # Replace NaN values with zero

    pcm = ax.pcolormesh(X, Y, nb_iterations, label='PCG: Green + Jacobi', cmap='Reds', norm=divnorm)

    if geometry_ID == 'sine_wave_' and phase_contrast == 2:
        ax.text(0.05, 0.82, f'Total phase contrast \n' + r'$\kappa=\infty$', transform=ax.transAxes)
    elif geometry_ID == 'sine_wave_':
        ax.text(0.05, 0.82, f'Total phase contrast \n $\kappa=10^{{{-phase_contrast}}}$', transform=ax.transAxes)
    else:
        ax.text(0.05, 0.82, f'Total phase contrast \n  $\kappa=10^{phase_contrast}$', transform=ax.transAxes)

    if row == 0:
        ax.set_title('Total number of iteration \n Jacobi-Green  ')
    # ax.set_zlim(1 ,100)
    # ax.set_ylabel('# of material phases')

    # ax.yaxis.set_label_position('right')
    # ax.yaxis.tick_right()
    if row == 1:
        ax.set_xlabel('# of nodal points (x direction)')
    ax.set_xticks(Nx)
    ax.set_xticklabels([f'$2^{{{i}}}$' for i in Nx])
    # ax2 = ax.twinx()
    ax.set_yticks(Nx)
    ax.set_yticklabels([f'$2^{{{i}}}$' for i in Nx])
    ax.tick_params(right=True, top=False, labelright=False, labeltop=False, labelrotation=0)
    #   ax.set_aspect('equal')

    if row == 0:
        ax.text(-0.15, 1.15, f'(c.{row + 1})', transform=ax.transAxes)
    elif row == 1:
        ax.text(-0.15, 1.05, f'(c.{row + 1})', transform=ax.transAxes)
    # Adding a color bar with custom ticks and labels
    cbar_ax = fig.add_subplot(gs[row, 3])
    cbar = plt.colorbar(pcm, location='left', cax=cbar_ax, ticklocation='right')  # Specify the ticks
    # cbar.ax.invert_yaxis()
    # # cbar.set_ticks(ticks=[  0, 1,10])
    # cbar.set_ticks([10, 5, 2, 1, 1 / 2, 1 / 5, 1 / 10])
    # cbar.ax.set_yticklabels(
    #     ['Jacobi-Green \n needs less', '5 times', '2 times', 'Equal', '2 times', '5 times',
    #      'Jacobi-Green \n needs more'])

    #

    row += 1

fname = src + 'JG_exp_tables_random_x0_{}_geom_{}_rho_{}{}'.format(geometry_ID, precc, phase_contrast, '.pdf')
print(('create figure: {}'.format(fname)))
plt.savefig(fname, bbox_inches='tight')

plt.show()




nb_pix_multips = [2, 3, 4, 5, 6, 7, 8, 9, 10]
Nx = (np.asarray(nb_pix_multips))
X, Y = np.meshgrid(Nx, Nx, indexing='ij')

# material distribution
geometry_ID = 'linear'  # linear  sine_wave_
# rhs = 'sin_wave'
rhs = False
linestyles = ['-', '--', ':', '-.', '--', ':', '-.']
colors = ['red', 'blue', 'green', 'orange', 'purple', 'orange', 'purple']
precc = 6


row = 0
for phase_contrast in [2, 4]:  # 1, 4
    ratio = phase_contrast
    fig = plt.figure(figsize=(8, 3.5))
    gs = fig.add_gridspec(1, 3, hspace=0.2, wspace=0.25, width_ratios=[1.2, 1.2, 0.03],
                          height_ratios=[1])
    if geometry_ID == 'linear':
        divnorm = mpl.colors.Normalize(vmin=0, vmax=500)
        white_lim = 250
    elif geometry_ID == 'sine_wave_':
        divnorm = mpl.colors.Normalize(vmin=0, vmax=200)
        white_lim = 100
    # Green graph
    gs0 = gs[row, 0].subgridspec(1, 1, wspace=0.1, width_ratios=[1])
    ax = fig.add_subplot(gs0[0, 0])
    # ax.set_aspect('equal')
    if phase_contrast == 2:
        if geometry_ID == 'linear':
            if precc == 6:
                nb_iterations = (nb_it_Green_linear_2_e06[:, :])
        elif geometry_ID == 'sine_wave_':
            if precc == 6:
                nb_iterations = nb_it_Green_sine_wave_0_e06[:, :]
    elif phase_contrast == 4:
        if geometry_ID == 'linear':
            if precc == 6:
                nb_iterations = nb_it_Green_linear_4_e06[:, :]
        elif geometry_ID == 'sine_wave_':
            if precc == 6:
                nb_iterations = (nb_it_Green_sine_wave_4_e06[:, :])

    nb_iterations = np.nan_to_num(nb_iterations, nan=1.0)
    for i in range(nb_iterations.shape[0]):
        for j in range(nb_iterations.shape[1]):
            if nb_iterations[i, j] == 0:
                pass
            elif nb_iterations[i, j] < white_lim:
                ax.text(i + Nx[0], j + Nx[0], f'{nb_iterations[i, j]:.0f}', size=8,
                        ha='center', va='center', color='black')
            else:
                ax.text(i + Nx[0], j + Nx[0], f'{nb_iterations[i, j]:.0f}', size=8,
                        ha='center', va='center', color='white')

    pcm = ax.pcolormesh(X, Y, nb_iterations, label='PCG: Green + Jacobi', cmap='Reds', norm=divnorm)

    # ax.text(0.05, 0.92, f'Total phase contrast $\kappa=10^{phase_contrast}$', transform=ax.transAxes)
    if geometry_ID == 'sine_wave_' and phase_contrast == 2:
        ax.text(0.05, 0.82, f'Total phase contrast \n' + r'$\kappa=\infty$', transform=ax.transAxes)
    elif geometry_ID == 'sine_wave_':
        ax.text(0.05, 0.82, f'Total phase contrast \n $\kappa=10^{{{-phase_contrast}}}$', transform=ax.transAxes)
    else:
        ax.text(0.05, 0.82, f'Total phase contrast \n $\kappa=10^{phase_contrast}$', transform=ax.transAxes)

    if row == 0:
        ax.set_title('Total number of iteration \n Green ')
    # ax.set_zlim(1 ,100)
    # ax.set_ylabel('# data/geometry sampling points (x direction)')

    # ax.yaxis.set_label_position('right')
    # ax.yaxis.tick_right()
    ax.set_ylabel('# of material phases')
    ax.set_xlabel('# of nodal points (x direction)')
    ax.set_xticks(Nx)
    ax.set_xticklabels([f'$2^{{{i}}}$' for i in Nx])
    # ax2 = ax.twinx()
    ax.set_yticks(Nx)
    ax.set_yticklabels([f'$2^{{{i}}}$' for i in Nx])
    ax.tick_params(right=True, top=False, labelright=False, labeltop=False, labelrotation=0)

    # plot Jacobi green
    gs2 = gs[row, 1].subgridspec(1, 1, wspace=0.1, width_ratios=[5])
    ax = fig.add_subplot(gs2[0, 0])
    if phase_contrast == 2:
        if geometry_ID == 'linear':
            if precc == 6:
                nb_iterations = (nb_it_combi_linear_2_e06[:, :])
        elif geometry_ID == 'sine_wave_':
            if precc == 6:
                nb_iterations = nb_it_combi_sine_wave_0_e06[:, :]
    elif phase_contrast == 4:
        if geometry_ID == 'linear':
            if precc == 6:
                nb_iterations = nb_it_combi_linear_4_e06[:, :]
        elif geometry_ID == 'sine_wave_':
            if precc == 6:
                nb_iterations = (nb_it_combi_sine_wave_4_e06[:, :])
    nb_iterations = np.nan_to_num(nb_iterations, nan=1.0)
    for i in range(nb_iterations.shape[0]):
        for j in range(nb_iterations.shape[1]):
            if nb_iterations[i, j] == 0:
                pass
            elif nb_iterations[i, j] < white_lim:
                ax.text(i + Nx[0], j + Nx[0], f'{nb_iterations[i, j]:.0f}', size=8,
                        ha='center', va='center', color='black')
            else:
                ax.text(i + Nx[0], j + Nx[0], f'{nb_iterations[i, j]:.0f}', size=8,
                        ha='center', va='center', color='white')
    # Replace NaN values with zero

    pcm = ax.pcolormesh(X, Y, nb_iterations, label='PCG: Green + Jacobi', cmap='Reds', norm=divnorm)

    if geometry_ID == 'sine_wave_' and phase_contrast == 2:
        ax.text(0.05, 0.82, f'Total phase contrast \n' + r'$\kappa=\infty$', transform=ax.transAxes)
    elif geometry_ID == 'sine_wave_':
        ax.text(0.05, 0.82, f'Total phase contrast \n $\kappa=10^{{{-phase_contrast}}}$', transform=ax.transAxes)
    else:
        ax.text(0.05, 0.82, f'Total phase contrast \n  $\kappa=10^{phase_contrast}$', transform=ax.transAxes)

    if row == 0:
        ax.set_title('Total number of iteration \n Jacobi-Green  ')
    # ax.set_zlim(1 ,100)
    # ax.set_ylabel('# of material phases')

    # ax.yaxis.set_label_position('right')
    # ax.yaxis.tick_right()
    ax.set_xlabel('# of nodal points (x direction)')
    ax.set_xticks(Nx)
    ax.set_xticklabels([f'$2^{{{i}}}$' for i in Nx])
    # ax2 = ax.twinx()
    ax.set_yticks(Nx)
    ax.set_yticklabels([f'$2^{{{i}}}$' for i in Nx])
    ax.tick_params(right=True, top=False, labelright=False, labeltop=False, labelrotation=0)
    #   ax.set_aspect('equal')

    # Adding a color bar with custom ticks and labels
    cbar_ax = fig.add_subplot(gs[row, 2])
    cbar = plt.colorbar(pcm, location='left', cax=cbar_ax, ticklocation='right')  # Specify the ticks
    # cbar.ax.invert_yaxis()
    # # cbar.set_ticks(ticks=[  0, 1,10])
    # cbar.set_ticks([10, 5, 2, 1, 1 / 2, 1 / 5, 1 / 10])
    # cbar.ax.set_yticklabels(
    #     ['Jacobi-Green \n needs less', '5 times', '2 times', 'Equal', '2 times', '5 times',
    #      'Jacobi-Green \n needs more'])

    #


    fname = src + 'JG_exp_tables_random_x0_{}_geom_{}_rho_{}_contrast{}'.format(geometry_ID, precc, phase_contrast, '.pdf')
    print(('create figure: {}'.format(fname)))
    plt.savefig(fname, bbox_inches='tight')

plt.show()


