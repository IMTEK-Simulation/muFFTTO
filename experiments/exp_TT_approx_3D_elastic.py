import numpy as np
from netCDF4 import Dataset
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from mpl_toolkits.mplot3d import Axes3D
from scipy.ndimage import gaussian_filter

# modify global setting
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
mpl.rcParams['font.family'] = 'sans-serif'

from muFFTTO import microstructure_library
from muFFTTO import TT_tools

problem_type = 'elasticity'
discretization_type = 'finite_element'
element_type = 'trilinear_hexahedron'
formulation = 'small_strain'

number_of_pixels = 3 * (30,)
geometry_ID = 'geometry_II_1_3D'

dataset_name = 'exp_data/' + f'muFFTTO_{problem_type}_{formulation}_{geometry_ID}_N{number_of_pixels[0]}_all.nc'

# read dataset
loaded_dataset = Dataset(dataset_name)
#
displacemet = loaded_dataset.variables['displacement_field']

phase_field = loaded_dataset.variables['phase_field']

# microstructure_library.visualize_voxels(phase_field_xyz=geometry)
# routine
field_name = 'phase_field'

field = np.asarray(loaded_dataset.variables[field_name])  # [1, 0]
field_norm = np.linalg.norm(field)
abs_error_norms = np.empty([number_of_pixels[0], number_of_pixels[0]])
rel_error_norms = np.empty([number_of_pixels[0], number_of_pixels[0]])
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(6, 6),
                             gridspec_kw={'wspace': 0.5, 'hspace': 0.5})
for filter_par in [ 0.1, 0.5, 1, 2, 4, 8]:
    field_filtered = gaussian_filter(field, sigma=filter_par)
    microstructure_library.visualize_voxels(phase_field_xyz=field_filtered)
    for rank_i in np.arange(0, number_of_pixels[0]):
        rank_j=rank_i
        #for rank_j in np.arange(0, number_of_pixels[0]):

        ranks = np.asarray([1, rank_i + 1, rank_j + 1, 1])
        tt_field = TT_tools.tt_decompose_rank(tensor_xyz=field_filtered,
                                              ranks=ranks)

        tt_reconstructed_field = TT_tools.tt_to_full_format(tt_cores=tt_field)

        abs_error_norms[rank_i, rank_j] = (np.linalg.norm(field - tt_reconstructed_field))
        rel_error_norms[rank_i, rank_j] = (abs_error_norms[rank_i, rank_j] / field_norm)
        #if rank_i == rank_j < 10:
            #microstructure_library.visualize_voxels(phase_field_xyz=tt_reconstructed_field)

    axes[1].semilogy(np.arange(0, number_of_pixels[0]), np.diag(rel_error_norms))
                     # ,  # data
                     # marker='o',  # each marker will be rendered as a circle
                     # markersize=8,  # marker size
                     # markerfacecolor='red',  # marker facecolor
                     # markeredgecolor='black',  # marker edgecolor
                     # markeredgewidth=2,  # marker edge width
                     # linestyle='--',  # line style will be dash line
                     # linewidth=3)  # line width


print(abs_error_norms)
print(rel_error_norms)

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(6, 6),
                         gridspec_kw={'wspace': 0.5, 'hspace': 0.5})
axes[1].semilogy(np.arange(0, number_of_pixels[0]), np.diag(rel_error_norms),  # data
                 marker='o',  # each marker will be rendered as a circle
                 markersize=8,  # marker size
                 markerfacecolor='red',  # marker facecolor
                 markeredgecolor='black',  # marker edgecolor
                 markeredgewidth=2,  # marker edge width
                 linestyle='--',  # line style will be dash line
                 linewidth=3)  # line width

X, Y = np.meshgrid(np.arange(0, number_of_pixels[0]), np.arange(0, number_of_pixels[0]))

fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')
# Set scale of Z axis as logarithmic
# ax.zaxis.set_major_formatter(ticker.LogFormatter())

# Set logarithmic scale for the z-axis
# ax.set_zscale('log')
surf = ax.plot_surface(X, Y, rel_error_norms, cmap=plt.cm.cividis)
# ax.set_zscale('log')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z (Log Scale)')

# ax.set_xlabel('rank i')
# ax.set_ylabel('rank j')
# ax.set_zlabel('rel error norm')
# ax.set_zticks([0,1e-4,1e-3,  1e-2,1e-1,1])

ax.set_zlim(0, 1)
plt.show()
