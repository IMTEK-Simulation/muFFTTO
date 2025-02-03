import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
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


src = './figures/'

nb_it_semi_continuous_16 = np.array([20., 32., 31., 30., 29., 28.])
nb_it_Jacobi_semi_continuous_16 = np.array([16., 47., 86., 167., 310., 588.])
nb_it_combi_semi_continuous_16 = np.array([5., 8., 12., 20., 35., 64.])

nb_it_continuous = np.array([20., 53., 87., 84., 52., 20.])
nb_it_Jacobi_continuous = np.array([16., 46., 88., 166., 310., 585.])
nb_it_combi_continuous = np.array([5., 3., 3., 3., 3., 2.])

Nx = np.array([1, 4, 8, 16, 32, 64]) * 16

fig = plt.figure()
gs = fig.add_gridspec(1, 1)
ax = fig.add_subplot(gs[0, 0])
# Plot each line with a different z offset
# for i in np.arange(len(nb_pix_multips)):
ax.plot(Nx, nb_it_semi_continuous_16, label='PCG: Green', color='blue')
ax.plot(Nx, nb_it_Jacobi_semi_continuous_16, label='PCG: Jacobi', color='black')
ax.plot(Nx, nb_it_combi_semi_continuous_16, label='PCG: Green + Jacobi', color='red')
ax.plot(Nx, nb_it_continuous, label='continuous PCG: Green', color='blue', linestyle='--')
ax.plot(Nx, nb_it_Jacobi_continuous, label='continuous PCG: Jacobi', color='black', linestyle='--')
ax.plot(Nx, nb_it_combi_continuous, label='continuous PCG: Green + Jacobi', color='red', linestyle='--')

ax.set_xlabel('Grid size')
ax.set_ylabel('# PCG iterations')
# plt.legend(['PCG: Green', 'PCG: Jacobi', 'PCG: Green + Jacobi', 'Richardson Green', 'Richardson Green+Jacobi'])
plt.legend()

fname = src + 'JG_exp4_GRID_DEP_comparison_{}{}'.format(0, '.pdf')
print(('create figure: {}'.format(fname)))
plt.savefig(fname, bbox_inches='tight')

plt.show()

nb_it_Green_4 = np.array(
    [[5., 0., 0., 0., 0., 0., 0., 0.],
     [8., 11., 0., 0., 0., 0., 0., 0.],
     [10., 16., 20., 0., 0., 0., 0., 0.],
     [11., 18., 27., 35., 0., 0., 0., 0.],
     [11., 17., 32., 47., 53., 0., 0., 0.],
     [11., 17., 31., 48., 73., 87., 0., 0.],
     [11., 19., 30., 49., 69., 85., 84., 0.],
     [11., 19., 29., 48., 65., 79., 66., 52.]])

nb_it_Green_1 = np.array([[5., 0., 0., 0., 0., 0., 0., 0.],
                          [7., 9., 0., 0., 0., 0., 0., 0.],
                          [9., 10., 11., 0., 0., 0., 0., 0.],
                          [9., 10., 11., 11., 0., 0., 0., 0.],
                          [10., 10., 10., 10., 10., 0., 0., 0.],
                          [10., 10., 9., 10., 9., 8., 0., 0.],
                          [10., 9., 9., 9., 8., 8., 7., 0.],
                          [10., 9., 9., 8., 8., 7., 7., 6.]])

nb_it_Green_1_0 = np.array([[5., 0., 0., 0., 0., 0., 0., 0.],
                            [8., 9., 0., 0., 0., 0., 0., 0.],
                            [10., 13., 14., 0., 0., 0., 0., 0.],
                            [11., 15., 17., 16., 0., 0., 0., 0.],
                            [11., 15., 18., 18., 14., 0., 0., 0.],
                            [11., 15., 18., 18., 17., 12., 0., 0.],
                            [11., 15., 17., 18., 17., 15., 11., 0.],
                            [11., 15., 17., 17., 17., 14., 12., 9.]])
nb_it_Green_4_0 = np.array([[5., 0., 0., 0., 0., 0., 0., 0.],
                            [8., 11., 0., 0., 0., 0., 0., 0.],
                            [10., 16., 20., 0., 0., 0., 0., 0.],
                            [11., 18., 27., 35., 0., 0., 0., 0.],
                            [11., 17., 32., 47., 54., 0., 0., 0.],
                            [11., 17., 31., 48., 76., 89., 0., 0.],
                            [11., 19., 30., 49., 72., 97., 100., 0.],
                            [11., 19., 29., 49., 68., 88., 73., 53.]])
nb_it_Green_4_0_inv = np.array(
    [[5., 0., 0., 0., 0., 0., 0., 0.],
     [8., 11., 0., 0., 0., 0., 0., 0.],
     [10., 16., 20., 0., 0., 0., 0., 0.],
     [11., 18., 27., 35., 0., 0., 0., 0.],
     [11., 17., 32., 47., 54., 0., 0., 0.],
     [11., 17., 31., 48., 76., 89., 0., 0.],
     [11., 19., 30., 49., 72., 97., 100., 0.],
     [11., 19., 29., 49., 68., 88., 73., 53.]])
nb_it_combi_4 = np.array([[5., 0., 0., 0., 0., 0., 0., 0.],
                          [6., 5., 0., 0., 0., 0., 0., 0.],
                          [8., 6., 5., 0., 0., 0., 0., 0.],
                          [10., 9., 5., 4., 0., 0., 0., 0.],
                          [15., 11., 8., 5., 3., 0., 0., 0.],
                          [24., 19., 12., 8., 5., 3., 0., 0.],
                          [45., 32., 20., 13., 8., 4., 3., 0.],
                          [82., 56., 35., 20., 11., 8., 4., 3.]])

nb_it_combi_1 = np.array([[5., 0., 0., 0., 0., 0., 0., 0.],
                          [5., 4., 0., 0., 0., 0., 0., 0.],
                          [6., 5., 4., 0., 0., 0., 0., 0.],
                          [7., 6., 5., 3., 0., 0., 0., 0.],
                          [9., 6., 5., 5., 3., 0., 0., 0.],
                          [11., 7., 6., 5., 4., 3., 0., 0.],
                          [15., 10., 7., 6., 5., 4., 3., 0.],
                          [24., 13., 9., 7., 6., 5., 3., 3.]])

nb_it_combi_1_0 = np.array([[5., 0., 0., 0., 0., 0., 0., 0.],
                            [6., 5., 0., 0., 0., 0., 0., 0.],
                            [7., 6., 5., 0., 0., 0., 0., 0.],
                            [8., 7., 6., 5., 0., 0., 0., 0.],
                            [9., 8., 7., 6., 4., 0., 0., 0.],
                            [12., 10., 8., 6., 5., 4., 0., 0.],
                            [15., 14., 11., 8., 6., 4., 4., 0.],
                            [23., 20., 15., 11., 7., 6., 4., 3.]])

nb_it_combi_4_0 = np.array([[5., 0., 0., 0., 0., 0., 0., 0.],
                            [6., 5., 0., 0., 0., 0., 0., 0.],
                            [7., 6., 5., 0., 0., 0., 0., 0.],
                            [8., 7., 5., 4., 0., 0., 0., 0.],
                            [9., 8., 7., 5., 3., 0., 0., 0.],
                            [12., 11., 9., 8., 5., 3., 0., 0.],
                            [16., 16., 14., 10., 8., 4., 3., 0.],
                            [24., 24., 20., 16., 10., 8., 4., 3.]])

nb_it_combi_4_0_inv = np.array([[5., 0., 0., 0., 0., 0., 0., 0.],
                                [6., 5., 0., 0., 0., 0., 0., 0.],
                                [7., 6., 5., 0., 0., 0., 0., 0.],
                                [8., 7., 5., 4., 0., 0., 0., 0.],
                                [9., 8., 7., 5., 3., 0., 0., 0.],
                                [11., 11., 9., 8., 5., 3., 0., 0.],
                                [14., 16., 14., 10., 8., 4., 3., 0.],
                                [22., 24., 20., 16., 10., 8., 4., 3.]])

nb_it_Jacobi_4 = np.array([[5., 0., 0., 0., 0., 0., 0., 0.],
                           [11., 10., 0., 0., 0., 0., 0., 0.],
                           [20., 17., 16., 0., 0., 0., 0., 0.],
                           [32., 25., 25., 24., 0., 0., 0., 0.],
                           [62., 47., 47., 46., 46., 0., 0., 0.],
                           [117., 89., 86., 89., 88., 88., 0., 0.],
                           [220., 171., 167., 165., 166., 165., 166., 0.],
                           [423., 318., 310., 313., 312., 312., 308., 310.]])

nb_it_Jacobi_1 = np.array([[5., 0., 0., 0., 0., 0., 0., 0.],
                           [10., 8., 0., 0., 0., 0., 0., 0.],
                           [18., 15., 14., 0., 0., 0., 0., 0.],
                           [29., 24., 21., 18., 0., 0., 0., 0.],
                           [54., 45., 40., 36., 30., 0., 0., 0.],
                           [104., 85., 78., 70., 60., 50., 0., 0.],
                           [201., 154., 144., 134., 119., 97., 92., 0.],
                           [352., 299., 265., 243., 211., 176., 156., 156.]])

nb_pix_multips = [2, 3, 4, 5, 6, 7, 8, 9]
Nx = (np.asarray(nb_pix_multips))
X, Y = np.meshgrid(Nx, Nx, indexing='ij')

fig = plt.figure(figsize=(5.5, 4.5))
# ax = fig.add_subplot(111, projection='3d')
ax = fig.add_subplot(111)
# Setting the view angle
# ax.view_init(elev=-0, azim=0)  # Adjust these values as needed
# Plotting the surface
# ax.plot_surface(X, Y, nb_it_Green[:, :], label='PCG: Green', color='green')
# ax.plot_wireframe(X, Y, nb_it_Jacobi[:, :], label='PCG: Jacobi', color='black')
# ax.plot_surface(X, Y, nb_it_combi[:, :], label='PCG: Green + Jacobi', color='red')
relative_nb_iterations = (nb_it_combi_4[:, :]) / nb_it_Green_4[:, :]
relative_nb_iterations = np.nan_to_num(relative_nb_iterations, nan=1.0)

divnorm = mpl.colors.TwoSlopeNorm(vmin=0., vcenter=1., vmax=relative_nb_iterations.max())
# Replace NaN values with zero

pcm = ax.pcolormesh(X, Y, relative_nb_iterations, label='PCG: Green + Jacobi', cmap='seismic', norm=divnorm)

ax.text(0.1, 0.8, r'Total phase contrast $\kappa=10^4$', transform=ax.transAxes)
ax.set_title('Relative number of iteration \n' + r' nb_{JG}/nb_{G}'
             '\n cosine function')
# ax.set_zlim(1 ,100)
ax.set_ylabel('# data/geometry sampling points (x direction)')
ax.set_xlabel('# of nodal points (x direction)')
ax.set_xticks(Nx)
ax.set_xticklabels([f'$2^{i}$' for i in Nx])
ax.set_yticks(Nx)
ax.set_yticklabels([f'$2^{i}$' for i in Nx])
# ax.set_zlabel('# CG iterations')
# Adding a color bar with custom ticks and labels
cbar = plt.colorbar(pcm)  # Specify the ticks
# cbar.set_ticks(ticks=[  0, 1,10])
cbar.set_ticks([0, 0.2, 0.5, 1, 2, 5, relative_nb_iterations.max()])
cbar.ax.set_yticklabels(
    ['Jacobi-Green \n is better', '5 times', '2 times', 'Equal', '2 times', '5 times', 'Green \n is better'])
# plt.legend(['PCG: Green'])
fname = src + 'JG_exp4_GRID_DEP_matrix_rho4{}'.format( '.pdf')
print(('create figure: {}'.format(fname)))
plt.savefig(fname, bbox_inches='tight')
plt.show()

fig = plt.figure(figsize=(5.5, 4.5))
# ax = fig.add_subplot(111, projection='3d')
ax = fig.add_subplot(111)
# Setting the view angle
# ax.view_init(elev=-0, azim=0)  # Adjust these values as needed
# Plotting the surface
# ax.plot_surface(X, Y, nb_it_Green[:, :], label='PCG: Green', color='green')
# ax.plot_wireframe(X, Y, nb_it_Jacobi[:, :], label='PCG: Jacobi', color='black')
# ax.plot_surface(X, Y, nb_it_combi[:, :], label='PCG: Green + Jacobi', color='red')
relative_nb_iterations_1 = (nb_it_combi_1[:, :]) / nb_it_Green_1[:, :]
relative_nb_iterations_1 = np.nan_to_num(relative_nb_iterations_1, nan=1.0)

divnorm = mpl.colors.TwoSlopeNorm(vmin=0., vcenter=1., vmax=relative_nb_iterations.max())
# Replace NaN values with zero

pcm = ax.pcolormesh(X, Y, relative_nb_iterations_1, label='PCG: Green + Jacobi', cmap='seismic', norm=divnorm)

ax.text(0.1, 0.8, r'Total phase contrast $\kappa=10^1$', transform=ax.transAxes)
ax.set_title('Relative number of iteration \n cosine function')
# ax.set_zlim(1 ,100)
ax.set_ylabel('# data/geometry sampling points (x direction)')
ax.set_xlabel('# of nodal points (x direction)')
ax.set_xticks(Nx)
ax.set_xticklabels([f'$2^{i}$' for i in Nx])
ax.set_yticks(Nx)
ax.set_yticklabels([f'$2^{i}$' for i in Nx])
# ax.set_zlabel('# CG iterations')
# Adding a color bar with custom ticks and labels
cbar = plt.colorbar(pcm)  # Specify the ticks
# cbar.set_ticks(ticks=[  0, 1,10])
cbar.set_ticks([0, 0.2, 0.5, 1, 2, 5, relative_nb_iterations.max()])
cbar.ax.set_yticklabels(
    ['Jacobi-Green \n is better', '5 times', '2 times', 'Equal', '2 times', '5 times', 'Green \n is better'])
# plt.legend(['PCG: Green'])
fname = src + 'JG_exp4_GRID_DEP_matrix_rho1{}'.format( '.pdf')
print(('create figure: {}'.format(fname)))
plt.savefig(fname, bbox_inches='tight')
plt.show()

fig = plt.figure(figsize=(5.5, 4.5))
# ax = fig.add_subplot(111, projection='3d')
ax = fig.add_subplot(111)
# Setting the view angle
# ax.view_init(elev=-0, azim=0)  # Adjust these values as needed
# Plotting the surface
# ax.plot_surface(X, Y, nb_it_Green[:, :], label='PCG: Green', color='green')
# ax.plot_wireframe(X, Y, nb_it_Jacobi[:, :], label='PCG: Jacobi', color='black')
# ax.plot_surface(X, Y, nb_it_combi[:, :], label='PCG: Green + Jacobi', color='red')
relative_nb_iterations_1 = (nb_it_combi_1_0[:, :]) / nb_it_Green_1_0[:, :]
relative_nb_iterations_1 = np.nan_to_num(relative_nb_iterations_1, nan=1.0)

divnorm = mpl.colors.TwoSlopeNorm(vmin=0., vcenter=1., vmax=relative_nb_iterations.max())
# Replace NaN values with zero

pcm = ax.pcolormesh(X, Y, relative_nb_iterations_1, label='PCG: Green + Jacobi', cmap='seismic', norm=divnorm)

ax.text(0.1, 0.8, r'Total phase contrast $\kappa=10^1$', transform=ax.transAxes)
ax.set_title('Relative number of iteration \n cosine function _ zeros')
# ax.set_zlim(1 ,100)
ax.set_ylabel('# data/geometry sampling points (x direction)')
ax.set_xlabel('# of nodal points (x direction)')
ax.set_xticks(Nx)
ax.set_xticklabels([f'$2^{i}$' for i in Nx])
ax.set_yticks(Nx)
ax.set_yticklabels([f'$2^{i}$' for i in Nx])
# ax.set_zlabel('# CG iterations')
# Adding a color bar with custom ticks and labels
cbar = plt.colorbar(pcm)  # Specify the ticks
# cbar.set_ticks(ticks=[  0, 1,10])
cbar.set_ticks([0, 0.2, 0.5, 1, 2, 5, relative_nb_iterations.max()])
cbar.ax.set_yticklabels(
    ['Jacobi-Green \n is better', '5 times', '2 times', 'Equal', '2 times', '5 times', 'Green \n is better'])
# plt.legend(['PCG: Green'])
fname = src + 'JG_exp4_GRID_DEP_matrix_rho1_inf{}'.format(0, '.pdf')
print(('create figure: {}'.format(fname)))
plt.savefig(fname, bbox_inches='tight')
plt.show()

fig = plt.figure(figsize=(5.5, 4.5))
# ax = fig.add_subplot(111, projection='3d')
ax = fig.add_subplot()
# Setting the view angle
# ax.view_init(elev=-0, azim=0)  # Adjust these values as needed
# Plotting the surface
# ax.plot_surface(X, Y, nb_it_Green[:, :], label='PCG: Green', color='green')
# ax.plot_wireframe(X, Y, nb_it_Jacobi[:, :], label='PCG: Jacobi', color='black')
# ax.plot_surface(X, Y, nb_it_combi[:, :], label='PCG: Green + Jacobi', color='red')
relative_nb_iterations_1 = (nb_it_combi_4_0[:, :]) / nb_it_Green_4_0[:, :]
relative_nb_iterations_1 = np.nan_to_num(relative_nb_iterations_1, nan=1.0)

divnorm = mpl.colors.TwoSlopeNorm(vmin=0., vcenter=1., vmax=relative_nb_iterations.max())
# Replace NaN values with zero

pcm = ax.pcolormesh(X, Y, relative_nb_iterations_1, label='PCG: Green + Jacobi', cmap='seismic', norm=divnorm)

ax.text(0.1, 0.8, r'Total phase contrast $\kappa=10^4$', transform=ax.transAxes)
ax.set_title('Relative number of iteration \n cosine function _ zeros')
# ax.set_zlim(1 ,100)
ax.set_ylabel('# data/geometry sampling points (x direction)')
ax.set_xlabel('# of nodal points (x direction)')
ax.set_xticks(Nx)
ax.set_xticklabels([f'$2^{i}$' for i in Nx])
ax.set_yticks(Nx)
ax.set_yticklabels([f'$2^{i}$' for i in Nx])
# ax.set_zlabel('# CG iterations')
# Adding a color bar with custom ticks and labels
cbar = plt.colorbar(pcm)  # Specify the ticks
# cbar.set_ticks(ticks=[  0, 1,10])
cbar.set_ticks([0, 0.2, 0.5, 1, 2, 5, relative_nb_iterations.max()])
cbar.ax.set_yticklabels(
    ['Jacobi-Green \n is better', '5 times', '2 times', 'Equal', '2 times', '5 times', 'Green \n is better'])
# plt.legend(['PCG: Green'])
fname = src + 'JG_exp4_GRID_DEP_matrix_rho4_inf{}'.format(0, '.pdf')
print(('create figure: {}'.format(fname)))
plt.savefig(fname, bbox_inches='tight')
plt.show()



























fig = plt.figure(figsize=(5.5, 4.5))
# ax = fig.add_subplot(111, projection='3d')
ax = fig.add_subplot()
relative_nb_iterations_1 = (nb_it_combi_4_0_inv[:, :]) / nb_it_Green_4_0_inv[:, :]
relative_nb_iterations_1 = np.nan_to_num(relative_nb_iterations_1, nan=1.0)

divnorm = mpl.colors.TwoSlopeNorm(vmin=0., vcenter=1., vmax=relative_nb_iterations.max())
# Replace NaN values with zero

pcm = ax.pcolormesh(X, Y, relative_nb_iterations_1, label='PCG: Green + Jacobi', cmap='seismic', norm=divnorm)

ax.text(0.1, 0.8, r'Total phase contrast $\kappa=10^4$', transform=ax.transAxes)
ax.set_title('Relative number of iteration \n cosine function _ zeros inverse')
# ax.set_zlim(1 ,100)
ax.set_ylabel('# data/geometry sampling points (x direction)')
ax.set_xlabel('# of nodal points (x direction)')
ax.set_xticks(Nx)
ax.set_xticklabels([f'$2^{i}$' for i in Nx])
ax.set_yticks(Nx)
ax.set_yticklabels([f'$2^{i}$' for i in Nx])
# ax.set_zlabel('# CG iterations')
# Adding a color bar with custom ticks and labels
cbar = plt.colorbar(pcm)  # Specify the ticks
# cbar.set_ticks(ticks=[  0, 1,10])
cbar.set_ticks([0, 0.2, 0.5, 1, 2, 5, relative_nb_iterations.max()])
cbar.ax.set_yticklabels(
    ['Jacobi-Green \n is better', '5 times', '2 times', 'Equal', '2 times', '5 times', 'Green \n is better'])
# plt.legend(['PCG: Green'])
fname = src + 'JG_exp4_GRID_DEP_matrix_rho4_inf_{}{}'.format(0, '.pdf')
print(('create figure: {}'.format(fname)))
plt.savefig(fname, bbox_inches='tight')
plt.show()

## number of iteration in 3D
# [[ 5.  0.  0.  0.  0.]
#  [ 7.  9.  0.  0.  0.]
#  [ 8. 11. 13.  0.  0.]
#  [ 9. 12. 17. 17.  0.]
#  [ 9. 12. 16. 15. 14.]]
# [[ 6.  0.  0.  0.  0.]
#  [ 8.  8.  0.  0.  0.]
#  [14. 14. 13.  0.  0.]
#  [25. 22. 22. 20.  0.]
#  [44. 34. 33. 33. 36.]]
# [[ 5.  0.  0.  0.  0.]
#  [ 5.  5.  0.  0.  0.]
#  [ 7.  5.  5.  0.  0.]
#  [10.  6.  5.  4.  0.]
#  [15.  9.  6.  4.  3.]]
### plot geometry
nb_pix_multips = [2, 4, 9]

# material distribution
geometry_ID = 'sine_wave_'

ratio = np.arange(1, 2)

fig = plt.figure()
gs = fig.add_gridspec(3, 3)

ax1 = fig.add_subplot(gs[1, :])
ax2 = fig.add_subplot(gs[2, :])
linestyles = ['-', '--', ':']
colors = ['red', 'blue', 'green', 'orange', 'purple']
counter = 0
for kk in np.arange(np.size(nb_pix_multips)):
    nb_pix_multip = nb_pix_multips[kk]
    print(f'kk = {kk}')
    print(f'nb_pix_multip = {nb_pix_multip}')
    # system set up
    number_of_pixels = (2 ** nb_pix_multip, 2 ** nb_pix_multip)

    my_cell = domain.PeriodicUnitCell(domain_size=domain_size,
                                      problem_type=problem_type)

    discretization = domain.Discretization(cell=my_cell,
                                           nb_of_pixels_global=number_of_pixels,
                                           discretization_type=discretization_type,
                                           element_type=element_type)

    # if kk == 0:
    phase_field = microstructure_library.get_geometry(nb_voxels=discretization.nb_of_pixels,
                                                      microstructure_name=geometry_ID,
                                                      coordinates=discretization.fft.coords,
                                                      seed=1)
    phase_field += 1 / 10 ** ratio
    phase_field = scale_field(phase_field, min_val=1 / 10 ** ratio, max_val=1.0)
    # phase_fied_small_grid=np.copy(phase_field_smooth)
    phase_field = np.copy(phase_field)
    # if kk > 0:
    #     # phase_field_smooth = sc.ndimage.zoom(phase_fied_small_grid, zoom=nb_pix_multip, order=0)
    #     phase_field_smooth = np.repeat(phase_fied_small_grid, 2 ** (kk), axis=0)
    #     phase_field_smooth = np.repeat(phase_field_smooth, 2 ** (kk), axis=1)

    # print(phase_field)

    x = np.arange(0, 1 * number_of_pixels[0])
    y = np.arange(0, 1 * number_of_pixels[1])
    X, Y = np.meshgrid(x, y)

    print(f'kk = {kk}')
    ax0 = fig.add_subplot(gs[0, kk])
    ax0.pcolormesh(X, Y, np.transpose(phase_field),
                   cmap=mpl.cm.Greys, vmin=1e-4, vmax=1, linewidth=0,
                   rasterized=True)
    ax0.hlines(y=number_of_pixels[1] // 2, xmin=-0.5, xmax=number_of_pixels[0] - 0.5, color=colors[counter],
               linestyle=linestyles[counter], linewidth=1.)
    ax0.set_title(f'Resolution $(2^{nb_pix_multip})^{2}$')
    extended_x = np.linspace(0, 1, phase_field[:, phase_field.shape[0] // 2].size + 1)
    extended_y = np.append(phase_field[:, phase_field.shape[0] // 2],
                           phase_field[:, phase_field.shape[0] // 2][-1])

    ax0.set_xticks( [])
    ax0.set_yticks([])
    if kk == 0:
        ax0.set_ylabel(r'Geometries')

    ax1.step(extended_x, extended_y
             , where='post',
             linewidth=1, color=colors[counter], linestyle=linestyles[counter],  # marker='|',
             label=r'phase contrast -' + f'1e{nb_pix_multips[kk]} ')
    # ax3.plot(phase_field[:, phase_field.shape[0] // 2], linewidth=1)
    ax1.set_ylim([0.05, 1.1])
    ax1.set_xlim([0, 1])
    ax1.set_xticks( [])
    ax1.set_xticklabels([ ])

    ax1.set_yticks([0.1, 0.25, 0.50, 0.75, 1.0001])
    ax1.set_yticklabels([0.1, 0.25, 0.50, 0.75, 1.00])
    ax1.set_ylabel(f'phase' + r' $\rho$' f' \n  (linear scale)')
    ax1.set_title(f'Cross sections')
    # log scale plot
    ax2.step(extended_x, extended_y
             , where='post',
             linewidth=1, color=colors[counter], linestyle=linestyles[counter],  # marker='|',
             label=r'phase contrast -' + f'1e{nb_pix_multips[kk]} ')
    # ax3.plot(phase_field[:, phase_field.shape[0] // 2], linewidth=1)
    ax2.set_ylim([0.05, 1.1])
    ax2.set_xlim([0, 1]) 

    ax2.set_yticks([0.1, 0.25, 0.50, 0.75, 1.0001])
    ax2.set_yticklabels([0.1, 0.25, 0.50, 0.75, 1.00])
    ax2.set_yscale('log')
    ax2.set_xlabel('x coordinate')
    ax2.set_ylabel(f'phase' + r' $\rho$' f' \n  (log scale)')
    #
    counter += 1
fname = src + 'JG_exp4_GRID_DEP_geometry_rho1{}'.format( '.pdf')
print(('create figure: {}'.format(fname)))
plt.savefig(fname, bbox_inches='tight')

plt.show()

### plot geometry
nb_pix_multips = [2, 4, 9]

# material distribution
geometry_ID = 'sine_wave_'

ratio = np.arange(4, 5)

fig = plt.figure()
gs = fig.add_gridspec(3, 3)

ax1 = fig.add_subplot(gs[1, :])
ax2 = fig.add_subplot(gs[2, :])
linestyles = ['-', '--', ':']
colors = ['red', 'blue', 'green', 'orange', 'purple']
counter = 0
for kk in np.arange(np.size(nb_pix_multips)):
    nb_pix_multip = nb_pix_multips[kk]
    print(f'kk = {kk}')
    print(f'nb_pix_multip = {nb_pix_multip}')
    # system set up
    number_of_pixels = (2 ** nb_pix_multip, 2 ** nb_pix_multip)

    my_cell = domain.PeriodicUnitCell(domain_size=domain_size,
                                      problem_type=problem_type)

    discretization = domain.Discretization(cell=my_cell,
                                           nb_of_pixels_global=number_of_pixels,
                                           discretization_type=discretization_type,
                                           element_type=element_type)

    # if kk == 0:
    phase_field = microstructure_library.get_geometry(nb_voxels=discretization.nb_of_pixels,
                                                      microstructure_name=geometry_ID,
                                                      coordinates=discretization.fft.coords,
                                                      seed=1)
    phase_field += 1 / 10 ** ratio
    phase_field = scale_field(phase_field, min_val=1 / 10 ** ratio, max_val=1.0)
    # phase_fied_small_grid=np.copy(phase_field_smooth)
    phase_field = np.copy(phase_field)
    # if kk > 0:
    #     # phase_field_smooth = sc.ndimage.zoom(phase_fied_small_grid, zoom=nb_pix_multip, order=0)
    #     phase_field_smooth = np.repeat(phase_fied_small_grid, 2 ** (kk), axis=0)
    #     phase_field_smooth = np.repeat(phase_field_smooth, 2 ** (kk), axis=1)

    # print(phase_field)

    x = np.arange(0, 1 * number_of_pixels[0])
    y = np.arange(0, 1 * number_of_pixels[1])
    X, Y = np.meshgrid(x, y)

    print(f'kk = {kk}')
    ax0 = fig.add_subplot(gs[0, kk])
    ax0.pcolormesh(X, Y, np.transpose(phase_field),
                   cmap=mpl.cm.Greys, vmin=1e-4, vmax=1, linewidth=0,
                   rasterized=True)
    ax0.hlines(y=number_of_pixels[1] // 2, xmin=-0.5, xmax=number_of_pixels[0] - 0.5, color=colors[counter],
               linestyle=linestyles[counter], linewidth=1.)
    ax0.set_title(f'Resolution $(2^{nb_pix_multip})^{2}$')
    extended_x = np.linspace(0, 1, phase_field[:, phase_field.shape[0] // 2].size + 1)
    extended_y = np.append(phase_field[:, phase_field.shape[0] // 2],
                           phase_field[:, phase_field.shape[0] // 2][-1])

    ax0.set_xticks( [])
    ax0.set_yticks([])
    if kk == 0:
        ax0.set_ylabel(r'Geometries')

    ax1.step(extended_x, extended_y
             , where='post',
             linewidth=1, color=colors[counter], linestyle=linestyles[counter],  # marker='|',
             label=r'phase contrast -' + f'1e{nb_pix_multips[kk]} ')
    # ax3.plot(phase_field[:, phase_field.shape[0] // 2], linewidth=1)
    ax1.set_ylim([0.0, 1.1])
    ax1.set_xlim([0, 1])
    ax1.set_xticks( [])
    ax1.set_xticklabels([ ])

    ax1.set_yticks([0.0001, 0.25, 0.50, 0.75, 1.0001])
    ax1.set_yticklabels([0.0001, 0.25, 0.50, 0.75, 1.00])
    ax1.set_ylabel(f'phase' + r' $\rho$' f' \n  (linear scale)')
    ax1.set_title(f'Cross sections')
    # log scale plot
    ax2.step(extended_x, extended_y
             , where='post',
             linewidth=1, color=colors[counter], linestyle=linestyles[counter],  # marker='|',
             label=r'phase contrast -' + f'1e{nb_pix_multips[kk]} ')
    # ax3.plot(phase_field[:, phase_field.shape[0] // 2], linewidth=1)
    ax2.set_ylim([0.000009, 1.1])
    ax2.set_xlim([0, 1])

    ax2.set_yticks([0.001, 0.25, 0.50, 0.75, 1.0001])
    ax2.set_yticklabels([0.001, 0.25, 0.50, 0.75, 1.00])

    ax2.set_yscale('log')
    ax2.set_xlabel('x coordinate')
    ax2.set_ylabel(f'phase' + r' $\rho$' f' \n  (log scale)')
    #
    counter += 1
fname = src + 'JG_exp4_GRID_DEP_geometry_rho4{}'.format( '.pdf')
print(('create figure: {}'.format(fname)))
plt.savefig(fname, bbox_inches='tight')
plt.show()




### plot geometry
nb_pix_multips = [2, 4, 9]

# material distribution
geometry_ID = 'sine_wave_'

ratio = np.arange(4, 5)

fig = plt.figure()
gs = fig.add_gridspec(3, 3)

ax1 = fig.add_subplot(gs[1, :])
ax2 = fig.add_subplot(gs[2, :])
linestyles = ['-', '--', ':']
colors = ['red', 'blue', 'green', 'orange', 'purple']
counter = 0
for kk in np.arange(np.size(nb_pix_multips)):
    nb_pix_multip = nb_pix_multips[kk]
    print(f'kk = {kk}')
    print(f'nb_pix_multip = {nb_pix_multip}')
    # system set up
    number_of_pixels = (2 ** nb_pix_multip, 2 ** nb_pix_multip)

    my_cell = domain.PeriodicUnitCell(domain_size=domain_size,
                                      problem_type=problem_type)

    discretization = domain.Discretization(cell=my_cell,
                                           nb_of_pixels_global=number_of_pixels,
                                           discretization_type=discretization_type,
                                           element_type=element_type)

    # if kk == 0:
    phase_field = microstructure_library.get_geometry(nb_voxels=discretization.nb_of_pixels,
                                                      microstructure_name=geometry_ID,
                                                      coordinates=discretization.fft.coords,
                                                      seed=1)
    #phase_field += 1 / 10 ** ratio
    phase_field = scale_field(phase_field, min_val=1 / 10 ** ratio, max_val=1.0)
    # phase_fied_small_grid=np.copy(phase_field_smooth)
    phase_field = np.copy(phase_field)
    # if kk > 0:
    #     # phase_field_smooth = sc.ndimage.zoom(phase_fied_small_grid, zoom=nb_pix_multip, order=0)
    #     phase_field_smooth = np.repeat(phase_fied_small_grid, 2 ** (kk), axis=0)
    #     phase_field_smooth = np.repeat(phase_field_smooth, 2 ** (kk), axis=1)

    # print(phase_field)

    x = np.arange(0, 1 * number_of_pixels[0])
    y = np.arange(0, 1 * number_of_pixels[1])
    X, Y = np.meshgrid(x, y)

    print(f'kk = {kk}')
    ax0 = fig.add_subplot(gs[0, kk])
    ax0.pcolormesh(X, Y, np.transpose(phase_field),
                   cmap=mpl.cm.Greys, vmin=1e-4, vmax=1, linewidth=0,
                   rasterized=True)
    ax0.hlines(y=number_of_pixels[1] // 2, xmin=-0.5, xmax=number_of_pixels[0] - 0.5, color=colors[counter],
               linestyle=linestyles[counter], linewidth=1.)
    ax0.set_title(f'Resolution $(2^{nb_pix_multip})^{2}$')
    extended_x = np.linspace(0, 1, phase_field[:, phase_field.shape[0] // 2].size + 1)
    extended_y = np.append(phase_field[:, phase_field.shape[0] // 2],
                           phase_field[:, phase_field.shape[0] // 2][-1])

    ax0.set_xticks( [])
    ax0.set_yticks([])
    if kk == 0:
        ax0.set_ylabel(r'Geometries')

    ax1.step(extended_x, extended_y
             , where='post',
             linewidth=1, color=colors[counter], linestyle=linestyles[counter],  # marker='|',
             label=r'phase contrast -' + f'1e{nb_pix_multips[kk]} ')
    # ax3.plot(phase_field[:, phase_field.shape[0] // 2], linewidth=1)
    ax1.set_ylim([0.0, 1.1])
    ax1.set_xlim([0, 1])
    ax1.set_xticks( [])
    ax1.set_xticklabels([ ])

    ax1.set_yticks([0.00, 0.25, 0.50, 0.75, 1.0001])
    ax1.set_yticklabels([0.0, 0.25, 0.50, 0.75, 1.00])
    ax1.set_ylabel(f'phase' + r' $\rho$' f' \n  (linear scale)')
    ax1.set_title(f'Cross sections')
    # log scale plot
    ax2.step(extended_x, extended_y
             , where='post',
             linewidth=1, color=colors[counter], linestyle=linestyles[counter],  # marker='|',
             label=r'phase contrast -' + f'1e{nb_pix_multips[kk]} ')
    # ax3.plot(phase_field[:, phase_field.shape[0] // 2], linewidth=1)
    # ax2.set_ylim([0.0, 1.1])
    ax2.set_xlim([0, 1])
    #
    ax2.set_yticks([0.0, 0.25, 0.50, 0.75, 1.0001])
    ax2.set_yticklabels([0.0, 0.25, 0.50, 0.75, 1.00])

    ax2.set_yscale('log')
    ax2.set_xlabel('x coordinate')
    ax2.set_ylabel(f'phase' + r' $\rho$' f' \n  (semilog scale)')
    #
    counter += 1
fname = src + 'JG_exp4_GRID_DEP_geometry_rho4_0{}'.format( '.pdf')
print(('create figure: {}'.format(fname)))
plt.savefig(fname, bbox_inches='tight')
plt.show()




### plot geometry
nb_pix_multips = [2, 5, 9]

# material distribution
geometry_ID = 'sine_wave_inv'

ratio = np.arange(4, 5)

fig = plt.figure()
gs = fig.add_gridspec(2, 3)

ax1 = fig.add_subplot(gs[1, :])
linestyles = ['-', '--', ':']
colors = ['red', 'blue', 'green', 'orange', 'purple']
counter = 0
for kk in np.arange(np.size(nb_pix_multips)):
    nb_pix_multip = nb_pix_multips[kk]
    print(f'kk = {kk}')
    print(f'nb_pix_multip = {nb_pix_multip}')
    # system set up
    number_of_pixels = (2 ** nb_pix_multip, 2 ** nb_pix_multip)

    my_cell = domain.PeriodicUnitCell(domain_size=domain_size,
                                      problem_type=problem_type)

    discretization = domain.Discretization(cell=my_cell,
                                           nb_of_pixels_global=number_of_pixels,
                                           discretization_type=discretization_type,
                                           element_type=element_type)

    # if kk == 0:
    phase_field = microstructure_library.get_geometry(nb_voxels=discretization.nb_of_pixels,
                                                      microstructure_name=geometry_ID,
                                                      coordinates=discretization.fft.coords,
                                                      seed=1)
    # phase_field += 1 / 10 ** ratio
    # phase_fied_small_grid=np.copy(phase_field_smooth)
    phase_field = np.copy(phase_field)
    # if kk > 0:
    #     # phase_field_smooth = sc.ndimage.zoom(phase_fied_small_grid, zoom=nb_pix_multip, order=0)
    #     phase_field_smooth = np.repeat(phase_fied_small_grid, 2 ** (kk), axis=0)
    #     phase_field_smooth = np.repeat(phase_field_smooth, 2 ** (kk), axis=1)

    # print(phase_field)

    x = np.arange(0, 1 * number_of_pixels[0])
    y = np.arange(0, 1 * number_of_pixels[1])
    X, Y = np.meshgrid(x, y)

    print(f'kk = {kk}')
    ax0 = fig.add_subplot(gs[0, kk])
    ax0.pcolormesh(X, Y, np.transpose(phase_field),
                   cmap=mpl.cm.Greys, vmin=1e-4, vmax=1, linewidth=0,
                   rasterized=True)
    ax0.hlines(y=number_of_pixels[1] // 2, xmin=-0.5, xmax=number_of_pixels[0] - 0.5, color=colors[counter],
               linestyle=linestyles[counter], linewidth=1.)
    ax0.set_title(f'Resolution $(2^{nb_pix_multip})^{2}$')
    #
    # ax0.set_xticks(np.arange(-.5, number_of_pixels[0], int(number_of_pixels[0] / 4)))
    # ax0.set_yticks(np.arange(-.5, number_of_pixels[1], int(number_of_pixels[1] / 4)))
    # ax0.set_xticklabels(np.arange(0, number_of_pixels[0] + 1, int(number_of_pixels[0] / 4)))
    # ax0.set_yticklabels(np.arange(0, number_of_pixels[1] + 1, int(number_of_pixels[1] / 4)))
    extended_x = np.linspace(0, 1, phase_field[:, phase_field.shape[0] // 2].size + 1)
    extended_y = np.append(phase_field[:, phase_field.shape[0] // 2],
                           phase_field[:, phase_field.shape[0] // 2][-1])
    ax1.step(extended_x, extended_y
             , where='post',
             linewidth=1, color=colors[counter], linestyle=linestyles[counter],  # marker='|',
             label=r'phase contrast -' + f'1e{nb_pix_multips[kk]} ')
    # ax3.plot(phase_field[:, phase_field.shape[0] // 2], linewidth=1)
    ax1.set_ylim([0.000009, 1.1])
    ax1.set_xlim([0, 1])
    ax1.set_yticks([0.001, 0.25, 0.50, 0.75, 1.0001])
    ax1.set_yticklabels([0.001, 0.25, 0.50, 0.75, 1.00])
    ax1.set_yscale('log')
    counter += 1
plt.show()
#
# io= 1
# greeen
# [[ 5.  0.  0.  0.  0.  0.  0.  0.]
#  [ 7.  9.  0.  0.  0.  0.  0.  0.]
#  [ 9. 10. 11.  0.  0.  0.  0.  0.]
#  [ 9. 10. 11. 11.  0.  0.  0.  0.]
#  [10. 10. 10. 10. 10.  0.  0.  0.]
#  [10. 10.  9. 10.  9.  8.  0.  0.]
#  [10.  9.  9.  9.  8.  8.  7.  0.]
#  [10.  9.  9.  8.  8.  7.  7.  6.]]
# jacobi
# [[  5.   0.   0.   0.   0.   0.   0.   0.]
#  [ 10.   8.   0.   0.   0.   0.   0.   0.]
#  [ 18.  15.  14.   0.   0.   0.   0.   0.]
#  [ 29.  24.  21.  18.   0.   0.   0.   0.]
#  [ 54.  45.  40.  36.  30.   0.   0.   0.]
#  [104.  85.  78.  70.  60.  50.   0.   0.]
#  [201. 154. 144. 134. 119.  97.  92.   0.]
#  [352. 299. 265. 243. 211. 176. 156. 156.]]
# combi
# [[ 5.  0.  0.  0.  0.  0.  0.  0.]
#  [ 5.  4.  0.  0.  0.  0.  0.  0.]
#  [ 6.  5.  4.  0.  0.  0.  0.  0.]
#  [ 7.  6.  5.  3.  0.  0.  0.  0.]
#  [ 9.  6.  5.  5.  3.  0.  0.  0.]
#  [11.  7.  6.  5.  4.  3.  0.  0.]
#  [15. 10.  7.  6.  5.  4.  3.  0.]
#  [24. 13.  9.  7.  6.  5.  3.  3.]]
# ratio= 2
# greeen
# [[ 5.  0.  0.  0.  0.  0.  0.  0.]
#  [ 8. 10.  0.  0.  0.  0.  0.  0.]
#  [10. 14. 18.  0.  0.  0.  0.  0.]
#  [11. 16. 23. 24.  0.  0.  0.  0.]
#  [12. 16. 23. 25. 26.  0.  0.  0.]
#  [17. 16. 23. 24. 24. 23.  0.  0.]
#  [17. 16. 23. 22. 22. 21. 19.  0.]
#  [17. 16. 21. 21. 20. 19. 17. 15.]]
# jacobi
# [[  5.   0.   0.   0.   0.   0.   0.   0.]
#  [ 11.  10.   0.   0.   0.   0.   0.   0.]
#  [ 20.  17.  15.   0.   0.   0.   0.   0.]
#  [ 32.  24.  24.  23.   0.   0.   0.   0.]
#  [ 61.  47.  46.  45.  43.   0.   0.   0.]
#  [115.  89.  86.  87.  83.  81.   0.   0.]
#  [216. 166. 166. 160. 159. 150. 145.   0.]
#  [418. 312. 308. 305. 295. 288. 269. 265.]]
# combi
# [[ 5.  0.  0.  0.  0.  0.  0.  0.]
#  [ 6.  5.  0.  0.  0.  0.  0.  0.]
#  [ 8.  6.  4.  0.  0.  0.  0.  0.]
#  [10.  7.  5.  4.  0.  0.  0.  0.]
#  [12. 10.  7.  5.  3.  0.  0.  0.]
#  [20. 13.  9.  6.  5.  3.  0.  0.]
#  [34. 21. 11.  8.  6.  4.  3.  0.]
#  [60. 34. 18. 10.  8.  6.  4.  3.]]
# ratio= 3
# greeen
# [[ 5.  0.  0.  0.  0.  0.  0.  0.]
#  [ 8. 11.  0.  0.  0.  0.  0.  0.]
#  [10. 16. 20.  0.  0.  0.  0.  0.]
#  [11. 17. 27. 33.  0.  0.  0.  0.]
#  [11. 17. 30. 43. 48.  0.  0.  0.]
#  [11. 17. 30. 43. 51. 54.  0.  0.]
#  [11. 17. 29. 42. 49. 49. 45.  0.]
#  [11. 19. 28. 42. 43. 42. 38. 32.]]
# jacobi
# [[  5.   0.   0.   0.   0.   0.   0.   0.]
#  [ 11.  10.   0.   0.   0.   0.   0.   0.]
#  [ 20.  17.  16.   0.   0.   0.   0.   0.]
#  [ 32.  25.  25.  24.   0.   0.   0.   0.]
#  [ 61.  47.  47.  46.  46.   0.   0.   0.]
#  [117.  89.  86.  89.  87.  87.   0.   0.]
#  [219. 170. 166. 165. 165. 163. 162.   0.]
#  [423. 315. 310. 313. 310. 309. 301. 300.]]
# combi
# [[ 5.  0.  0.  0.  0.  0.  0.  0.]
#  [ 6.  5.  0.  0.  0.  0.  0.  0.]
#  [ 8.  6.  5.  0.  0.  0.  0.  0.]
#  [10.  8.  5.  4.  0.  0.  0.  0.]
#  [14. 10.  8.  5.  3.  0.  0.  0.]
#  [22. 18. 11.  8.  5.  3.  0.  0.]
#  [42. 29. 19. 10.  7.  4.  3.  0.]
#  [77. 49. 27. 17. 10.  7.  4.  3.]]
# ratio= 4
# greeen
# [[ 5.  0.  0.  0.  0.  0.  0.  0.]
#  [ 8. 11.  0.  0.  0.  0.  0.  0.]
#  [10. 16. 20.  0.  0.  0.  0.  0.]
#  [11. 18. 27. 35.  0.  0.  0.  0.]
#  [11. 17. 32. 47. 53.  0.  0.  0.]
#  [11. 17. 31. 48. 73. 87.  0.  0.]
#  [11. 19. 30. 49. 69. 85. 84.  0.]
#  [11. 19. 29. 48. 65. 79. 66. 52.]]
# jacobi
# [[  5.   0.   0.   0.   0.   0.   0.   0.]
#  [ 11.  10.   0.   0.   0.   0.   0.   0.]
#  [ 20.  17.  16.   0.   0.   0.   0.   0.]
#  [ 32.  25.  25.  24.   0.   0.   0.   0.]
#  [ 62.  47.  47.  46.  46.   0.   0.   0.]
#  [117.  89.  86.  89.  88.  88.   0.   0.]
#  [220. 171. 167. 165. 166. 165. 166.   0.]
#  [423. 318. 310. 313. 312. 312. 308. 310.]]
# combi
# [[ 5.  0.  0.  0.  0.  0.  0.  0.]
#  [ 6.  5.  0.  0.  0.  0.  0.  0.]
#  [ 8.  6.  5.  0.  0.  0.  0.  0.]
#  [10.  9.  5.  4.  0.  0.  0.  0.]
#  [15. 11.  8.  5.  3.  0.  0.  0.]
#  [24. 19. 12.  8.  5.  3.  0.  0.]
#  [45. 32. 20. 13.  8.  4.  3.  0.]
#  [82. 56. 35. 20. 11.  8.  4.  3.]]


# 00000000
# ratio= 1
# greeen
# [[ 5.  0.  0.  0.  0.  0.  0.  0.]
#  [ 8.  9.  0.  0.  0.  0.  0.  0.]
#  [10. 13. 14.  0.  0.  0.  0.  0.]
#  [11. 15. 17. 16.  0.  0.  0.  0.]
#  [11. 15. 18. 18. 14.  0.  0.  0.]
#  [11. 15. 18. 18. 17. 12.  0.  0.]
#  [11. 15. 17. 18. 17. 15. 11.  0.]
#  [11. 15. 17. 17. 17. 14. 12.  9.]]
# jacobi
# [[  5.   0.   0.   0.   0.   0.   0.   0.]
#  [ 11.  10.   0.   0.   0.   0.   0.   0.]
#  [ 21.  18.  16.   0.   0.   0.   0.   0.]
#  [ 33.  28.  26.  24.   0.   0.   0.   0.]
#  [ 62.  51.  48.  46.  44.   0.   0.   0.]
#  [120.  96.  91.  87.  84.  80.   0.   0.]
#  [222. 180. 176. 165. 157. 152. 145.   0.]
#  [427. 351. 337. 315. 297. 279. 267. 250.]]
# combi
# [[ 5.  0.  0.  0.  0.  0.  0.  0.]
#  [ 6.  5.  0.  0.  0.  0.  0.  0.]
#  [ 7.  6.  5.  0.  0.  0.  0.  0.]
#  [ 8.  7.  6.  5.  0.  0.  0.  0.]
#  [ 9.  8.  7.  6.  4.  0.  0.  0.]
#  [12. 10.  8.  6.  5.  4.  0.  0.]
#  [15. 14. 11.  8.  6.  4.  4.  0.]
#  [23. 20. 15. 11.  7.  6.  4.  3.]]
# ratio= 2
# greeen
# [[ 5.  0.  0.  0.  0.  0.  0.  0.]
#  [ 8. 10.  0.  0.  0.  0.  0.  0.]
#  [10. 15. 19.  0.  0.  0.  0.  0.]
#  [11. 17. 25. 30.  0.  0.  0.  0.]
#  [11. 17. 28. 36. 35.  0.  0.  0.]
#  [11. 17. 28. 36. 43. 33.  0.  0.]
#  [11. 18. 27. 37. 42. 37. 24.  0.]
#  [11. 19. 27. 36. 42. 35. 24. 16.]]
# jacobi
# [[  5.   0.   0.   0.   0.   0.   0.   0.]
#  [ 11.  10.   0.   0.   0.   0.   0.   0.]
#  [ 20.  17.  16.   0.   0.   0.   0.   0.]
#  [ 32.  25.  25.  24.   0.   0.   0.   0.]
#  [ 61.  47.  47.  47.  46.   0.   0.   0.]
#  [117.  89.  89.  89.  88.  87.   0.   0.]
#  [219. 172. 166. 165. 165. 164. 162.   0.]
#  [423. 326. 313. 314. 312. 309. 304. 300.]]
# combi
# [[ 5.  0.  0.  0.  0.  0.  0.  0.]
#  [ 6.  5.  0.  0.  0.  0.  0.  0.]
#  [ 7.  6.  5.  0.  0.  0.  0.  0.]
#  [ 8.  7.  5.  4.  0.  0.  0.  0.]
#  [ 9.  8.  7.  5.  4.  0.  0.  0.]
#  [12. 11.  9.  7.  5.  3.  0.  0.]
#  [15. 16. 11. 10.  7.  4.  3.  0.]
#  [24. 22. 19. 15.  8.  7.  4.  3.]]
# ratio= 3
# greeen
# [[ 5.  0.  0.  0.  0.  0.  0.  0.]
#  [ 8. 11.  0.  0.  0.  0.  0.  0.]
#  [10. 16. 20.  0.  0.  0.  0.  0.]
#  [11. 18. 27. 34.  0.  0.  0.  0.]
#  [11. 17. 31. 45. 51.  0.  0.  0.]
#  [11. 17. 30. 46. 68. 71.  0.  0.]
#  [11. 19. 30. 48. 65. 76. 60.  0.]
#  [11. 19. 29. 46. 64. 71. 44. 32.]]
# jacobi
# [[  5.   0.   0.   0.   0.   0.   0.   0.]
#  [ 11.  10.   0.   0.   0.   0.   0.   0.]
#  [ 20.  17.  16.   0.   0.   0.   0.   0.]
#  [ 32.  25.  25.  24.   0.   0.   0.   0.]
#  [ 62.  47.  47.  46.  46.   0.   0.   0.]
#  [117.  89.  86.  89.  89.  89.   0.   0.]
#  [220. 171. 166. 165. 166. 166. 166.   0.]
#  [423. 318. 311. 313. 313. 312. 311. 310.]]
# combi
# [[ 5.  0.  0.  0.  0.  0.  0.  0.]
#  [ 6.  5.  0.  0.  0.  0.  0.  0.]
#  [ 7.  6.  5.  0.  0.  0.  0.  0.]
#  [ 8.  7.  5.  4.  0.  0.  0.  0.]
#  [ 9.  8.  7.  5.  3.  0.  0.  0.]
#  [12. 11.  9.  8.  5.  3.  0.  0.]
#  [15. 16. 14. 10.  7.  4.  3.  0.]
#  [23. 24. 20. 16. 12.  7.  4.  3.]]
# ratio= 4
# greeen
# [[  5.   0.   0.   0.   0.   0.   0.   0.]
#  [  8.  11.   0.   0.   0.   0.   0.   0.]
#  [ 10.  16.  20.   0.   0.   0.   0.   0.]
#  [ 11.  18.  27.  35.   0.   0.   0.   0.]
#  [ 11.  17.  32.  47.  54.   0.   0.   0.]
#  [ 11.  17.  31.  48.  76.  89.   0.   0.]
#  [ 11.  19.  30.  49.  72.  97. 100.   0.]
#  [ 11.  19.  29.  49.  68.  88.  73.  53.]]
# jacobi
# [[  5.   0.   0.   0.   0.   0.   0.   0.]
#  [ 11.  10.   0.   0.   0.   0.   0.   0.]
#  [ 20.  17.  16.   0.   0.   0.   0.   0.]
#  [ 32.  25.  25.  24.   0.   0.   0.   0.]
#  [ 62.  47.  47.  46.  46.   0.   0.   0.]
#  [117.  89.  86.  89.  88.  89.   0.   0.]
#  [220. 171. 167. 165. 166. 165. 167.   0.]
#  [423. 318. 310. 313. 311. 312. 310. 312.]]
# combi
# [[ 5.  0.  0.  0.  0.  0.  0.  0.]
#  [ 6.  5.  0.  0.  0.  0.  0.  0.]
#  [ 7.  6.  5.  0.  0.  0.  0.  0.]
#  [ 8.  7.  5.  4.  0.  0.  0.  0.]
#  [ 9.  8.  7.  5.  3.  0.  0.  0.]
#  [12. 11.  9.  8.  5.  3.  0.  0.]
#  [16. 16. 14. 10.  8.  4.  3.  0.]
#  [24. 24. 20. 16. 10.  8.  4.  3.]]


# naopak
# ratio= 1
# #greeen
# [[ 5.  0.  0.  0.  0.  0.  0.  0.]
#  [ 8.  9.  0.  0.  0.  0.  0.  0.]
#  [10. 13. 14.  0.  0.  0.  0.  0.]
#  [11. 15. 17. 16.  0.  0.  0.  0.]
#  [11. 15. 18. 18. 14.  0.  0.  0.]
#  [11. 15. 18. 18. 17. 12.  0.  0.]
#  [11. 15. 17. 18. 17. 15. 11.  0.]
#  [11. 15. 17. 17. 17. 14. 12.  9.]]
# jacobi
# [[  5.   0.   0.   0.   0.   0.   0.   0.]
#  [ 11.  10.   0.   0.   0.   0.   0.   0.]
#  [ 21.  18.  16.   0.   0.   0.   0.   0.]
#  [ 33.  28.  26.  24.   0.   0.   0.   0.]
#  [ 62.  51.  48.  46.  44.   0.   0.   0.]
#  [120.  96.  91.  87.  84.  80.   0.   0.]
#  [222. 180. 176. 165. 157. 152. 145.   0.]
#  [427. 351. 337. 315. 297. 279. 267. 250.]]
# combi
# [[ 5.  0.  0.  0.  0.  0.  0.  0.]
#  [ 6.  5.  0.  0.  0.  0.  0.  0.]
#  [ 7.  6.  5.  0.  0.  0.  0.  0.]
#  [ 8.  7.  6.  5.  0.  0.  0.  0.]
#  [ 9.  8.  7.  6.  4.  0.  0.  0.]
#  [12. 10.  8.  6.  5.  4.  0.  0.]
#  [16. 14. 11.  8.  6.  4.  4.  0.]
#  [24. 20. 16. 10.  7.  6.  4.  3.]]
# ratio= 2
# greeen
# [[ 5.  0.  0.  0.  0.  0.  0.  0.]
#  [ 8. 10.  0.  0.  0.  0.  0.  0.]
#  [10. 15. 19.  0.  0.  0.  0.  0.]
#  [11. 17. 25. 30.  0.  0.  0.  0.]
#  [11. 17. 28. 36. 35.  0.  0.  0.]
#  [11. 17. 28. 36. 43. 33.  0.  0.]
#  [11. 18. 27. 37. 42. 37. 24.  0.]
#  [11. 19. 27. 36. 42. 35. 24. 16.]]
# jacobi
# [[  5.   0.   0.   0.   0.   0.   0.   0.]
#  [ 11.  10.   0.   0.   0.   0.   0.   0.]
#  [ 20.  17.  16.   0.   0.   0.   0.   0.]
#  [ 32.  25.  25.  24.   0.   0.   0.   0.]
#  [ 61.  47.  47.  47.  46.   0.   0.   0.]
#  [117.  89.  89.  89.  88.  87.   0.   0.]
#  [219. 172. 166. 165. 165. 164. 162.   0.]
#  [423. 326. 313. 314. 312. 309. 304. 300.]]
# combi
# [[ 5.  0.  0.  0.  0.  0.  0.  0.]
#  [ 6.  5.  0.  0.  0.  0.  0.  0.]
#  [ 7.  6.  5.  0.  0.  0.  0.  0.]
#  [ 8.  7.  5.  4.  0.  0.  0.  0.]
#  [ 9.  8.  7.  5.  4.  0.  0.  0.]
#  [12. 10.  9.  7.  5.  3.  0.  0.]
#  [16. 15. 11.  9.  7.  4.  3.  0.]
#  [23. 22. 18. 15.  9.  7.  4.  3.]]
# ratio= 3
# greeen
# [[ 5.  0.  0.  0.  0.  0.  0.  0.]
#  [ 8. 11.  0.  0.  0.  0.  0.  0.]
#  [10. 16. 20.  0.  0.  0.  0.  0.]
#  [11. 18. 27. 34.  0.  0.  0.  0.]
#  [11. 17. 31. 45. 51.  0.  0.  0.]
#  [11. 17. 30. 46. 68. 71.  0.  0.]
#  [11. 19. 30. 48. 65. 76. 60.  0.]
#  [11. 19. 29. 46. 64. 71. 44. 32.]]
# jacobi
# [[  5.   0.   0.   0.   0.   0.   0.   0.]
#  [ 11.  10.   0.   0.   0.   0.   0.   0.]
#  [ 20.  17.  16.   0.   0.   0.   0.   0.]
#  [ 32.  25.  25.  24.   0.   0.   0.   0.]
#  [ 62.  47.  47.  46.  46.   0.   0.   0.]
#  [117.  89.  86.  89.  89.  89.   0.   0.]
#  [220. 171. 166. 165. 166. 166. 166.   0.]
#  [423. 318. 311. 313. 313. 312. 311. 310.]]
# combi
# [[ 5.  0.  0.  0.  0.  0.  0.  0.]
#  [ 6.  5.  0.  0.  0.  0.  0.  0.]
#  [ 7.  6.  5.  0.  0.  0.  0.  0.]
#  [ 8.  7.  5.  4.  0.  0.  0.  0.]
#  [ 9.  8.  7.  5.  3.  0.  0.  0.]
#  [11. 11.  9.  8.  5.  3.  0.  0.]
#  [14. 16. 14. 10.  7.  4.  3.  0.]
#  [22. 23. 20. 16. 12.  7.  4.  3.]]
# ratio= 4
# greeen
# [[  5.   0.   0.   0.   0.   0.   0.   0.]
#  [  8.  11.   0.   0.   0.   0.   0.   0.]
#  [ 10.  16.  20.   0.   0.   0.   0.   0.]
#  [ 11.  18.  27.  35.   0.   0.   0.   0.]
#  [ 11.  17.  32.  47.  54.   0.   0.   0.]
#  [ 11.  17.  31.  48.  76.  89.   0.   0.]
#  [ 11.  19.  30.  49.  72.  97. 100.   0.]
#  [ 11.  19.  29.  49.  68.  88.  73.  53.]]
# jacobi
# [[  5.   0.   0.   0.   0.   0.   0.   0.]
#  [ 11.  10.   0.   0.   0.   0.   0.   0.]
#  [ 20.  17.  16.   0.   0.   0.   0.   0.]
#  [ 32.  25.  25.  24.   0.   0.   0.   0.]
#  [ 62.  47.  47.  46.  46.   0.   0.   0.]
#  [117.  89.  86.  89.  88.  89.   0.   0.]
#  [220. 171. 167. 165. 166. 165. 167.   0.]
#  [423. 318. 310. 313. 311. 312. 310. 312.]]
# combi
# [[ 5.  0.  0.  0.  0.  0.  0.  0.]
#  [ 6.  5.  0.  0.  0.  0.  0.  0.]
#  [ 7.  6.  5.  0.  0.  0.  0.  0.]
#  [ 8.  7.  5.  4.  0.  0.  0.  0.]
#  [ 9.  8.  7.  5.  3.  0.  0.  0.]
#  [11. 11.  9.  8.  5.  3.  0.  0.]
#  [14. 16. 14. 10.  8.  4.  3.  0.]
#  [22. 24. 20. 16. 10.  8.  4.  3.]]
