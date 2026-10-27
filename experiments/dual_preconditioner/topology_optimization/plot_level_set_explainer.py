"""Level-set method for dummies: three rows explaining the design representation of
example_2D_elasticity_TO_level_set.py, drawn from a real run (Green-Jacobi, eps continuation 2-1-0.5-0)
-> figures/level_set_explainer_N64.png

    1. the design is the shoreline of a landscape phi: solid where phi > 0, void where phi < 0
    2. optimizing = raising / lowering the landscape; the shoreline moves, holes can merge or vanish
    3. the edge: density rho = H_eps(phi), a band ~2 eps wide of intermediate stiffness, sharp for eps = 0

    python plot_level_set_explainer.py
"""
import os

import matplotlib
import netCDF4
import numpy as np

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

HERE = os.path.dirname(os.path.realpath(__file__))
RUN = os.path.join(HERE, 'data', 'example_2D_elasticity_TO_level_set',
                   'N64_soft1e-05_energy_Green_Jacobi_w_5.0_p1_eps2-1-0.5-0_alpha2_holes4')
FRAME_EARLY = 2  # landscape shown in rows 1 and 3
FRAME_BEFORE, FRAME_AFTER = 2, 10  # a few optimization steps (row 2)
CONTOUR_FRAMES = [0, 20, 242]  # shoreline over the run (row 2)
ROW_CUT = 21  # pixel row of the 1D cut (many edge moves there)

history = netCDF4.Dataset(RUN + '_history.nc')
phi_all = np.asarray(history['level_set_iterate'][:])[:, 0]
n = phi_all.shape[-1]
h = 1.0 / n
phi_all = phi_all / h  # in pixels: phi ~ signed distance to the edge
x = (np.arange(n) + 0.5) / n
SOLID, VOID = '#f2f2f2', '#1a1a1a'


def heaviside(phi, eps):
    """rho = H_eps(phi) of the level-set driver (eps in pixels, 0 = sharp)."""
    if eps == 0:
        return (phi > 0).astype(float)
    t = np.clip(phi / eps, -1, 1)
    return 0.5 * (1 + t + np.sin(np.pi * t) / np.pi)


def show_pixels(ax, field, **kwargs):
    ax.imshow(field.T, origin='lower', extent=[0, 1, 0, 1], interpolation='nearest', **kwargs)
    ax.set_xticks([]), ax.set_yticks([])


fig = plt.figure(figsize=(16, 15), dpi=150)
grid = fig.add_gridspec(3, 3, hspace=0.42, wspace=0.18)
phi = phi_all[FRAME_EARLY]
norm = TwoSlopeNorm(vcenter=0, vmin=phi.min(), vmax=phi.max())

# --- Row 1: landscape -> shoreline -> design
ax = fig.add_subplot(grid[0, 0], projection='3d')
quarter = slice(0, n // 2)  # one quarter of the cell, so the hills stay readable
X, Y = np.meshgrid(x[quarter], x[quarter], indexing='ij')
ax.plot_surface(X, Y, phi[quarter, quarter], cmap='RdBu_r', norm=norm, rstride=1, cstride=1, linewidth=0,
                antialiased=False)
ax.plot_surface(X, Y, np.zeros_like(X), color='0.5', alpha=0.3, linewidth=0)
ax.contour(X, Y, phi[quarter, quarter], levels=[0], colors='k', linewidths=1.5, offset=0)
ax.set_zlabel('φ  [pixels]')
ax.set_xticks([]), ax.set_yticks([])
ax.view_init(elev=35, azim=-60)
ax.set_title('1a. The unknown is a landscape φ(x, y)\n(one quarter; grey sheet = sea level φ = 0)', fontsize=11)

ax = fig.add_subplot(grid[0, 1])
show_pixels(ax, phi, cmap='RdBu_r', norm=norm)
ax.contour(x, x, phi.T, levels=[0], colors='k', linewidths=1.5)
ax.set_title('1b. From above: red = land (φ > 0),\nblue = sea (φ < 0), black = shoreline φ = 0', fontsize=11)

ax = fig.add_subplot(grid[0, 2])
show_pixels(ax, heaviside(phi, 0), cmap='Greys_r', vmin=0, vmax=1)
ax.contour(x, x, phi.T, levels=[0], colors='#eb6834', linewidths=1.5)
ax.set_title('1c. Design: land = solid (white), sea = void (black);\n'
             'only the shoreline matters, not the height', fontsize=11)

# --- Row 2: optimization moves the landscape
phi_0, phi_1 = phi_all[FRAME_BEFORE], phi_all[FRAME_AFTER]
ax = fig.add_subplot(grid[1, 0])
ax.axhline(0, color='0.5', lw=1)
ax.fill_between(x, -1e3, 1e3, where=phi_1[:, ROW_CUT] > 0, color='#eb6834', alpha=0.12, step='mid', lw=0)
ax.plot(x, phi_0[:, ROW_CUT], color='0.4', ls='--', label=f'before (iteration {FRAME_BEFORE})')
ax.plot(x, phi_1[:, ROW_CUT], color='#eb6834', label=f'after (iteration {FRAME_AFTER}), shaded = solid')
ax.set_ylim(min(phi_0[:, ROW_CUT].min(), phi_1[:, ROW_CUT].min()) - 0.5,
            max(phi_0[:, ROW_CUT].max(), phi_1[:, ROW_CUT].max()) + 0.5)
ax.set_xlim(0, 1)
ax.set_xlabel('x  (along one row of pixels)')
ax.set_ylabel('φ  [pixels]')
ax.legend(fontsize=8, loc='lower right')
ax.set_title(f'2a. Optimizing: the stiffness sensitivity says where to\n'
             f'lift / lower φ; where φ crosses 0 the edge moves', fontsize=11)

ax = fig.add_subplot(grid[1, 1])
change = heaviside(phi_1, 0) - heaviside(phi_0, 0)  # +1 material added, -1 removed
show_pixels(ax, heaviside(phi_0, 0), cmap='Greys_r', vmin=-0.4, vmax=1.1)  # before: dark grey void, light solid
added = np.ma.masked_where(change == 0, change)
show_pixels(ax, added, cmap=matplotlib.colors.ListedColormap(['#2a78d6', '#eb6834']), vmin=-1, vmax=1)
ax.contour(x, x, phi_0.T, levels=[0], colors='0.1', linewidths=1, linestyles='--')
ax.contour(x, x, phi_1.T, levels=[0], colors='k', linewidths=1.3)
ax.axhline(x[ROW_CUT], color='#eb6834', lw=0.8, ls=':')
ax.set_title(f'2b. Iterations {FRAME_BEFORE} → {FRAME_AFTER} in 2D: orange = became solid,\n'
             'blue = became void (dotted = the cut of 2a)', fontsize=11)

ax = fig.add_subplot(grid[1, 2])
show_pixels(ax, heaviside(phi_all[CONTOUR_FRAMES[-1]], 0), cmap='Greys_r', vmin=0, vmax=1, alpha=0.25)
colors = plt.cm.viridis(np.linspace(0, 0.9, len(CONTOUR_FRAMES)))
for frame, color in zip(CONTOUR_FRAMES, colors):
    ax.contour(x, x, phi_all[frame].T, levels=[0], colors=[color], linewidths=1.4)
    ax.plot([], [], color=color, label=f'iteration {frame}')
ax.legend(fontsize=8, loc='upper left', bbox_to_anchor=(1.01, 1))
ax.set_title('2c. Over the whole run: holes grow, merge and vanish\n'
             '(no new holes appear -- the start has many)', fontsize=11)

# --- Row 3: the edge, fuzzy (eps) or sharp
ax = fig.add_subplot(grid[2, 0])
phi_axis = np.linspace(-3, 3, 601)
for eps, color in zip([2, 1, 0.5, 0], ['#eb6834', '#d9a521', '#4a9b5c', 'k']):
    ax.plot(phi_axis, heaviside(phi_axis, eps), color=color, label='sharp (ε = 0)' if eps == 0 else f'ε = {eps:g} px')
ax.set_xlabel('φ  [pixels from the edge]')
ax.set_ylabel('density ρ  (stiffness = ρ C_solid)')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
ax.set_title('3a. Density from height: ρ = H_ε(φ); within ε of the\n'
             'shoreline ρ is in between (grey material)', fontsize=11)

for column, eps in [(1, 2), (2, 0)]:
    ax = fig.add_subplot(grid[2, column])
    zoom = slice(n // 4, 3 * n // 4)
    rho = heaviside(phi, eps)[zoom, zoom]
    ax.imshow(rho.T, origin='lower', extent=[0.25, 0.75, 0.25, 0.75], interpolation='nearest',
              cmap='Greys_r', vmin=0, vmax=1)
    ax.contour(x[zoom], x[zoom], phi[zoom, zoom].T, levels=[0], colors='#eb6834', linewidths=1.2)
    ax.set_xticks([]), ax.set_yticks([])
    grey = np.mean((heaviside(phi, eps) > 1e-3) & (heaviside(phi, eps) < 1 - 1e-3))
    ax.set_title(f'3{"bc"[column - 1]}. ' + (f'ε = {eps} px (zoom): {100 * grey:.0f}% of pixels grey\n'
                                             'smooth J, easy to optimize' if eps else
                                             'sharp ε = 0 (zoom): every pixel 0 or 1\n'
                                             'true two-phase; J jumps pixel by pixel'), fontsize=11)

fig.suptitle('Level-set topology optimization in three pictures  (64², real run: Green-Jacobi, ε = 2 → 1 → 0.5 → 0)',
             fontsize=14, y=0.93)
figure_name = os.path.join(HERE, 'figures', 'level_set_explainer_N64.png')
fig.savefig(figure_name, bbox_inches='tight')
print(f'figure: {figure_name}')
