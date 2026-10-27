"""Video: evolution of one level-set run (default: Green-Jacobi, eps continuation 2-1-0.5-0), one frame per accepted
iteration, in the pictures of plot_level_set_explainer.py:
    top       the landscape phi (3D, one quarter of the cell; grey sheet = phi = 0), phi from above with the
              shoreline phi = 0, and the density rho = H_eps(phi) the mechanics sees (exact pixels)
    bottom    objective J and CG iterations per solve (mean of the 3 forward + 3 adjoint solves of the accepted
              evaluation); continuation stages shaded, future part light

    python make_TO_level_set_video.py                -> figures/TO_level_set_evolution_<run>.mp4
    PREC=Green python make_TO_level_set_video.py     (once that run has finished)
    RUN_SUFFIX=_noreinit python make_TO_level_set_video.py   (the runs without reinitialization)
    RUN_PATH=example_2D_elasticity_TO_reaction_diffusion/<name> python make_TO_level_set_video.py
        any run under data/ (without _history.nc / _final.nc); for a reaction-diffusion run phi is the bounded
        level set in [-1, 1] and panel 3 shows the material at the 2 quadrature points (triangles) of each pixel
"""
import os

import imageio_ffmpeg
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import netCDF4
import numpy as np
from matplotlib import animation
from matplotlib.colors import TwoSlopeNorm

plt.rcParams['animation.ffmpeg_path'] = imageio_ffmpeg.get_ffmpeg_exe()
HERE = os.path.dirname(os.path.realpath(__file__))
PREC = os.environ.get('PREC', 'Green_Jacobi')
RUN_NAME = f'N64_soft1e-05_energy_{PREC}_w_5.0_p1_eps2-1-0.5-0_alpha2_holes4' + os.environ.get('RUN_SUFFIX', '')
RUN = os.path.join(HERE, 'data', 'example_2D_elasticity_TO_level_set', RUN_NAME)
if 'RUN_PATH' in os.environ:
    RUN = os.path.join(HERE, 'data', os.environ['RUN_PATH'])
    RUN_NAME = os.path.basename(RUN)
REACTION_DIFFUSION = 'reaction_diffusion' in RUN
FPS = 10
SURFACE, TEXT_2, GRID, FUTURE = '#fcfcfb', '#52514e', '#e6e5e1', '#d4d3cd'
STAGE_COLORS = {2: '#fbe3d6', 1: '#fbf0d0', 0.5: '#dcefe0', 0: '#e4e4e4'}

history = netCDF4.Dataset(RUN + '_history.nc')
final = netCDF4.Dataset(RUN + '_final.nc')
n = history['level_set_iterate'].shape[-1]
phi_all = np.asarray(history['level_set_iterate'][:])[:, 0] * (1 if REACTION_DIFFUSION else n)  # HJ: in pixels
rho_all = np.asarray(history['phase_field_iterate'][:])[:, 0]
objective = np.asarray(history['objective'][:])
eps_all = np.asarray(history['eps'][:])
evaluations = np.asarray(history['nb_evaluations'][:]).astype(int)
cg_per_solve = (np.asarray(final.num_iteration_mech, dtype=float) +
                np.asarray(final.num_iteration_adjoint, dtype=float)).reshape(-1, 3).sum(1) / 6
cg_frame = cg_per_solve[evaluations - 1]  # CG of the evaluation that produced each accepted design
nb_frames = len(objective)
iterations = np.arange(nb_frames)
x = (np.arange(n) + 0.5) / n
norm = TwoSlopeNorm(vcenter=0, vmin=np.percentile(phi_all, 0.5), vmax=np.percentile(phi_all, 99.5))
stages = [(eps, iterations[eps_all == eps].min(), iterations[eps_all == eps].max()) for eps in dict.fromkeys(eps_all)]

PREC = final.preconditioner_type


def quad_values(field):
    """Field at the 2 quadrature points of each pixel (linear triangles, as discretization.apply_N_operator_mugrid)."""
    f_10, f_01 = np.roll(field, -1, 0), np.roll(field, -1, 1)
    return np.stack([(field + f_10 + f_01) / 3, (f_10 + f_01 + np.roll(f_10, -1, 1)) / 3])


def triangulation(n):
    """Lower-left and upper-right triangle of each pixel, nodes at the pixel corners."""
    from matplotlib.tri import Triangulation
    i, j = np.meshgrid(np.arange(n + 1), np.arange(n + 1), indexing='ij')
    ii, jj = np.meshgrid(np.arange(n), np.arange(n), indexing='ij')
    node = lambda a, b: a * (n + 1) + b
    lower = np.stack([node(ii, jj), node(ii + 1, jj), node(ii, jj + 1)], -1).reshape(-1, 3)
    upper = np.stack([node(ii + 1, jj), node(ii, jj + 1), node(ii + 1, jj + 1)], -1).reshape(-1, 3)
    return Triangulation(i.ravel() / n, j.ravel() / n, np.concatenate([lower, upper]))


def solver_density(frame):
    """Material at the quadrature points of a reaction-diffusion run: threshold of the interpolated level set."""
    return (quad_values(phi_all[frame]) > 0).astype(float).reshape(2, -1).ravel()


fig = plt.figure(figsize=(15, 9.2), facecolor=SURFACE)
gs_top = fig.add_gridspec(1, 3, wspace=0.12, left=0.02, right=0.98, top=0.86, bottom=0.42)
gs_bottom = fig.add_gridspec(1, 2, wspace=0.22, left=0.07, right=0.97, top=0.32, bottom=0.07)

ax_3d = fig.add_subplot(gs_top[0, 0], projection='3d', facecolor=SURFACE)
quarter = slice(0, n // 2)
X, Y = np.meshgrid(x[quarter], x[quarter], indexing='ij')
ax_3d.set_title('1. the unknown: landscape φ  (one quarter)\ngrey sheet = sea level φ = 0', fontsize=11)

ax_top = fig.add_subplot(gs_top[0, 1])
phi_image = ax_top.imshow(phi_all[0].T, origin='lower', extent=[0, 1, 0, 1], interpolation='nearest',
                          cmap='RdBu_r', norm=norm)
ax_top.set_xticks([]), ax_top.set_yticks([])
ax_top.set_title('2. from above: red = land (φ > 0), blue = sea\nblack = shoreline φ = 0', fontsize=11)
shoreline = []

ax_design = fig.add_subplot(gs_top[0, 2])
if REACTION_DIFFUSION:
    rho_image = ax_design.tripcolor(triangulation(n), facecolors=solver_density(0), cmap='Greys_r', vmin=0, vmax=1,
                                    edgecolors='none')
    ax_design.set_aspect('equal')
    ax_design.set_xlim(0, 1), ax_design.set_ylim(0, 1)
else:
    rho_image = ax_design.imshow(rho_all[0].T, origin='lower', extent=[0, 1, 0, 1], interpolation='nearest',
                                 cmap='Greys_r', vmin=0, vmax=1)
ax_design.set_xticks([]), ax_design.set_yticks([])
design_title = ax_design.set_title('', fontsize=11)

ax_J = fig.add_subplot(gs_bottom[0, 0])
ax_cg = fig.add_subplot(gs_bottom[0, 1])
progress = []
for ax, values, label in ((ax_J, objective, 'objective J  (stress mismatch)'),
                          (ax_cg, cg_frame, 'CG iterations per solve')):
    for eps, first, last in ([] if REACTION_DIFFUSION else stages):  # RD: sharp throughout
        ax.axvspan(first - 0.5, last + 0.5, color=STAGE_COLORS.get(eps, '#eeeeee'), lw=0)
        ax.text(last if eps == 0 else (first + last) / 2, 0.89 if eps == 0 else 0.97, 'sharp' if eps == 0 else f'ε = {eps:g} px',
                transform=ax.get_xaxis_transform(), ha='right' if eps == 0 else 'center', va='top', fontsize=9,
                color=TEXT_2)
    ax.plot(iterations, values, color=FUTURE, lw=1.2)
    line, = ax.plot([], [], color='#eb6834', lw=1.8)
    dot, = ax.plot([], [], 'o', color='#eb6834', ms=6)
    progress.append((line, dot, values))
    ax.set(xlabel='accepted iteration', ylabel=label, xlim=(-1, nb_frames))
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID)
ax_J.set_yscale('log')
ax_cg.set_ylim(0, None)
ax_J.set_title('fit of the target (lower = better)'
               + ('; spikes = reinitialization' if not REACTION_DIFFUSION and 'noreinit' not in RUN_NAME else ''),
               fontsize=11)
ax_cg.set_title(f'cost of the elasticity solves ({PREC.replace("_", "-")} preconditioner)', fontsize=11)
suptitle = fig.suptitle('', fontsize=14, y=0.975)


def draw(frame):
    phi, eps = phi_all[frame], eps_all[frame]
    if REACTION_DIFFUSION:
        suptitle.set_text(f'Reaction-diffusion level set, sharp two-phase, 64²  —  iteration {frame},  '
                          f'J = {objective[frame]:.3g}  (best so far {objective[:frame + 1].min():.3g})')
    else:
        suptitle.set_text(f'Level-set topology optimization, 64²  —  iteration {frame},  '
                          f'{"sharp edge (ε = 0)" if eps == 0 else f"edge width ε = {eps:g} px"},  '
                          f'J = {objective[frame]:.3g}')
    ax_3d.clear()
    ax_3d.plot_surface(X, Y, phi[quarter, quarter], cmap='RdBu_r', norm=norm, rstride=1, cstride=1, linewidth=0,
                       antialiased=False)
    ax_3d.plot_surface(X, Y, np.zeros_like(X), color='0.5', alpha=0.3, linewidth=0)
    ax_3d.contour(X, Y, phi[quarter, quarter], levels=[0], colors='k', linewidths=1.5, offset=0)
    ax_3d.set_zlim(norm.vmin, norm.vmax)
    ax_3d.set_xticks([]), ax_3d.set_yticks([])
    ax_3d.set_zlabel('φ' if REACTION_DIFFUSION else 'φ  [pixels]')
    ax_3d.view_init(elev=35, azim=-60)
    ax_3d.set_title('1. the unknown: landscape φ  (one quarter)\ngrey sheet = sea level φ = 0', fontsize=11)
    phi_image.set_data(phi.T)
    for artist in shoreline:
        artist.remove()
    shoreline[:] = [ax_top.contour(x, x, phi.T, levels=[0], colors='k', linewidths=1.3)]
    if REACTION_DIFFUSION:
        rho_image.set_array(solver_density(frame))
        design_title.set_text('3. the design the mechanics sees: ρ = [φ > 0]\n'
                              'at the 2 quadrature points (triangles) of each pixel')
        draw_progress(frame)
        return
    rho = rho_all[frame]
    rho_image.set_data(rho.T)
    grey = np.mean((rho > 1e-3) & (rho < 1 - 1e-3))
    design_title.set_text(f'3. the design the mechanics sees: ρ = H_ε(φ)\n'
                          f'white = solid, black = void, {100 * grey:.0f}% of pixels grey')
    draw_progress(frame)


def draw_progress(frame):
    for line, dot, values in progress:
        line.set_data(iterations[:frame + 1], values[:frame + 1])
        dot.set_data(iterations[frame:frame + 1], values[frame:frame + 1])


out = os.path.join(HERE, 'figures', f'TO_level_set_evolution_{RUN_NAME}')
writer = animation.FFMpegWriter(fps=FPS, bitrate=6000, codec='libx264', extra_args=['-pix_fmt', 'yuv420p'])
with writer.saving(fig, out + '.mp4', dpi=110):
    for frame in range(nb_frames):
        draw(frame)
        writer.grab_frame(facecolor=SURFACE)
    for _ in range(2 * FPS):  # hold the final state for 2 s
        writer.grab_frame(facecolor=SURFACE)
fig.savefig(out + '_last_frame.png', facecolor=SURFACE, dpi=110)
print(f'video: {out}.mp4 ({nb_frames} frames)')
