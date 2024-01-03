import matplotlib.pyplot as plt
import numpy as np



# This import registers the 3D projection, but is otherwise unused.
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import

# prepare some coordinates
x, y, z = np.indices((8, 8, 8))

# draw cuboids in the top left and bottom right corners, and a link between them
cube1 = (x < 3) & (y < 3) & (z < 3)
cube2 = (x >= 5) & (y >= 5) & (z >= 5)
link = abs(x - y) + abs(y - z) + abs(z - x) <= 2

# combine the objects into a single boolean array
voxels = cube1 | cube2 | link

# set the colors of each object
face_colors = np.empty(list(voxels.shape) + [4], dtype=np.float32)
alpha = 0
face_colors[link] = [1, 0, 0, alpha]
face_colors[cube1] = [0, 0, 0, 0.5]
face_colors[cube2] = [0, 0, 1, 1]

edge_colors = np.empty(list(voxels.shape) + [4], dtype=np.float32)
edge_colors=face_colors
# and plot everything
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.voxels(voxels, facecolors=face_colors, edgecolor='k', linewidth=0.1)

plt.show()

# import matplotlib.pyplot as plt
# import numpy as np
#
# # Fixing random state for reproducibility.
# np.random.seed(19680801)
#
# fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(8, 4))
#
# x_values = [n for n in range(20)]
# y_values = np.random.randn(20)
#
# facecolors = ['green' if y > 0 else 'red' for y in y_values]
# edgecolors = facecolors
#
# ax1.bar(x_values, y_values, color=facecolors, edgecolor=edgecolors, alpha=0.5)
# ax1.set_title("Explicit 'alpha' keyword value\nshared by all bars and edges")
#
# # Normalize y values to get distinct face alpha values.
# abs_y = [abs(y) for y in y_values]
# face_alphas = [n / max(abs_y) for n in abs_y]
# edge_alphas = [1 - alpha for alpha in face_alphas]
#
# colors_with_alphas = list(zip(facecolors, face_alphas))
# edgecolors_with_alphas = list(zip(edgecolors, edge_alphas))
#
# ax2.bar(x_values, y_values, color=colors_with_alphas,
#         edgecolor=edgecolors_with_alphas)
# ax2.set_title('Normalized alphas for\neach bar and each edge')
#
# plt.show()
