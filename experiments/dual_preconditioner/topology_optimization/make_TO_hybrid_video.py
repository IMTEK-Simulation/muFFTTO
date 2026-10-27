"""Video: the hybrid run, reaction-diffusion level set (topology, sharp two-phase, Green) followed by the Hamilton-Jacobi
level set with the eps continuation 2-1-0.5-0 started from its best design (shape, Green-Jacobi), 64^2. One frame per
iteration of both runs, one after the other:
    top       the level set phi from above with the shoreline phi = 0; the design the mechanics sees, at the 2
              quadrature points (triangles) of each pixel (exact geometry); what changed since the previous frame
              (triangles that became solid / void)
    bottom    objective J and CG iterations per solve over the whole pipeline; stages shaded, future part light

    python make_TO_hybrid_video.py   -> figures/TO_hybrid_RD_to_HJ_evolution_N64.mp4 (+ _last_frame.png)
"""
import os

import imageio_ffmpeg
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import netCDF4
import numpy as np
from matplotlib import animation
from matplotlib.colors import ListedColormap, TwoSlopeNorm
from matplotlib.tri import Triangulation

plt.rcParams['animation.ffmpeg_path'] = imageio_ffmpeg.get_ffmpeg_exe()
HERE = os.path.dirname(os.path.realpath(__file__))
DATA = os.path.join(HERE, 'data')
RD_RUN = os.path.join(DATA, 'example_2D_elasticity_TO_reaction_diffusion',
                      'N64_soft1e-05_energy_Green_w_5.0_quad_L1_c0.2_all_initphase')
HJ_RUN = os.path.join(DATA, 'example_2D_elasticity_TO_level_set',
                      'N64_soft1e-05_energy_Green_Jacobi_w_5.0_p1_eps2-1-0.5-0_alpha2_initpf_final_noreinit_from_RD')
FPS = 10
SURFACE, TEXT_2, GRID, FUTURE = '#fcfcfb', '#52514e', '#e6e5e1', '#d4d3cd'
BLUE, ORANGE = '#2a78d6', '#eb6834'
STAGE_COLORS = {'rd': '#dbe8f8', 2: '#fbe3d6', 1: '#fbf0d0', 0.5: '#dcefe0', 0: '#e4e4e4'}


def quad_values(field):
    """Field at the 2 quadrature points of each pixel (linear triangles, as discretization.apply_N_operator_mugrid)."""
    f_10, f_01 = np.roll(field, -1, 0), np.roll(field, -1, 1)
    return np.stack([(field + f_10 + f_01) / 3, (f_10 + f_01 + np.roll(f_10, -1, 1)) / 3]).reshape(2, -1).ravel()


def triangulation(n):
    """Lower-left and upper-right triangle of each pixel, nodes at the pixel corners."""
    i, j = np.meshgrid(np.arange(n + 1), np.arange(n + 1), indexing='ij')
    ii, jj = np.meshgrid(np.arange(n), np.arange(n), indexing='ij')
    node = lambda a, b: a * (n + 1) + b
    lower = np.stack([node(ii, jj), node(ii + 1, jj), node(ii, jj + 1)], -1).reshape(-1, 3)
    upper = np.stack([node(ii + 1, jj), node(ii, jj + 1), node(ii + 1, jj + 1)], -1).reshape(-1, 3)
    return Triangulation(i.ravel() / n, j.ravel() / n, np.concatenate([lower, upper]))


def load(run, rd):
    history, final = netCDF4.Dataset(run + '_history.nc'), netCDF4.Dataset(run + '_final.nc')
    phi = np.asarray(history['level_set_iterate'][:])[:, 0]
    if rd:  # material = sharp threshold at the quadrature points
        density = np.array([(quad_values(p) > 0).astype(float) for p in phi])
        phi_scaled = phi  # bounded level set in [-1, 1]
        stage = np.array(['rd'] * len(phi), dtype=object)
    else:  # material = H_eps(phi) at the nodes, interpolated to the quadrature points
        density = np.array([quad_values(r) for r in np.asarray(history['phase_field_iterate'][:])[:, 0]])
        phi_scaled = phi * phi.shape[-1] / 2  # signed distance in pixels, /2 to share the colour scale
        stage = np.asarray(history['eps'][:]).astype(float).astype(object)
    cg = (np.asarray(final.num_iteration_mech, dtype=float) +
          np.asarray(final.num_iteration_adjoint, dtype=float)).reshape(-1, 3).sum(1) / 6
    evaluations = np.asarray(history['nb_evaluations'][:]).astype(int)
    return dict(phi=phi_scaled, density=density, objective=np.asarray(history['objective'][:]), stage=stage,
                cg=cg[evaluations - 1], time=float(final.elapsed_time), prec=final.preconditioner_type)


rd, hj = load(RD_RUN, True), load(HJ_RUN, False)
nb_rd = len(rd['objective'])
phi_all = np.concatenate([rd['phi'], hj['phi']])
density_all = np.concatenate([rd['density'], hj['density']])
objective = np.concatenate([rd['objective'], hj['objective']])
cg_frame = np.concatenate([rd['cg'], hj['cg']])
stage_all = np.concatenate([rd['stage'], hj['stage']])
nb_frames = len(objective)
iterations = np.arange(nb_frames)
n = phi_all.shape[-1]
x = (np.arange(n) + 0.5) / n
mesh = triangulation(n)
norm = TwoSlopeNorm(vcenter=0, vmin=-1, vmax=1)
stages = [(s, iterations[stage_all == s].min(), iterations[stage_all == s].max()) for s in dict.fromkeys(stage_all)]
STAGE_LABEL = {'rd': '1. reaction-diffusion\ntopology, sharp, Green', 2: '2. level set, ε = 2 px', 1: 'ε = 1',
               0.5: 'ε = ½', 0: 'sharp'}
best_rd = int(np.argmin(rd['objective']))  # stage 2 starts from the best stage-1 design

fig = plt.figure(figsize=(15, 9.2), facecolor=SURFACE)
gs_top = fig.add_gridspec(1, 3, wspace=0.08, left=0.03, right=0.97, top=0.85, bottom=0.42)
gs_bottom = fig.add_gridspec(1, 2, wspace=0.2, left=0.07, right=0.97, top=0.32, bottom=0.07)

ax_phi = fig.add_subplot(gs_top[0, 0])
phi_image = ax_phi.imshow(phi_all[0].T, origin='lower', extent=[0, 1, 0, 1], interpolation='nearest', cmap='RdBu_r',
                          norm=norm)
shoreline = []
ax_design = fig.add_subplot(gs_top[0, 1])
design_image = ax_design.tripcolor(mesh, facecolors=density_all[0], cmap='Greys_r', vmin=0, vmax=1, edgecolors='none')
ax_change = fig.add_subplot(gs_top[0, 2])
change_cmap = ListedColormap([BLUE, '#f1f0ec', ORANGE])  # -1 became void, 0 unchanged, +1 became solid
change_image = ax_change.tripcolor(mesh, facecolors=np.zeros(2 * n * n), cmap=change_cmap, vmin=-1.5, vmax=1.5,
                                   edgecolors='none')
ax_change.set_title('3. change since the previous frame\norange = became solid, blue = became void', fontsize=11)
for ax in (ax_phi, ax_design, ax_change):
    ax.set_aspect('equal')
    ax.set_xlim(0, 1), ax.set_ylim(0, 1)
    ax.set_xticks([]), ax.set_yticks([])
design_title = ax_design.set_title('', fontsize=11)
phi_title = ax_phi.set_title('', fontsize=11)

ax_J = fig.add_subplot(gs_bottom[0, 0])
ax_cg = fig.add_subplot(gs_bottom[0, 1])
progress = []
for ax, values, label in ((ax_J, objective, 'objective J  (stress mismatch)'),
                          (ax_cg, cg_frame, 'CG iterations per solve')):
    for k, (s, first, last) in enumerate(stages):
        ax.axvspan(first - 0.5, last + 0.5, color=STAGE_COLORS[s], lw=0)
        narrow = last - first < 25  # stagger the labels of the short stages
        ax.text((first + last) / 2, 0.97 - 0.09 * (narrow and k % 2 == 0), STAGE_LABEL[s],
                transform=ax.get_xaxis_transform(), ha='center', va='top', fontsize=8.5, color=TEXT_2)
    ax.axvline(nb_rd - 0.5, color=TEXT_2, lw=1)
    ax.plot(iterations, values, color=FUTURE, lw=1.2)
    line, = ax.plot([], [], color='k', lw=1.6)
    dot, = ax.plot([], [], 'o', color='k', ms=6)
    progress.append((line, dot, values))
    ax.set(xlabel='iteration (both stages)', ylabel=label, xlim=(-1, nb_frames))
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID)
ax_J.plot(best_rd, objective[best_rd], '*', color=BLUE, ms=12, zorder=5)
ax_J.annotate(f'best: J = {objective[best_rd]:.2f}\n→ start of stage 2', (best_rd, objective[best_rd]),
              xytext=(best_rd, objective[best_rd] / 40), ha='center', fontsize=8.5, color=BLUE,
              arrowprops=dict(arrowstyle='->', color=BLUE))
ax_J.set_yscale('log')
ax_J.set_ylim(None, objective.max() * 8)  # room for the stage labels
ax_cg.set_yscale('log')
ax_cg.set_ylim(None, cg_frame.max() * 6)
ax_J.set_title('fit of the target (lower = better)', fontsize=11)
ax_cg.set_title(f'cost per solve: {rd["prec"]} in stage 1 ({rd["time"] / 60:.1f} min), '
                f'{hj["prec"].replace("_", "-")} in stage 2 ({hj["time"] / 60:.1f} min)', fontsize=11)
suptitle = fig.suptitle('', fontsize=14, y=0.975)


def draw(frame):
    in_rd = frame < nb_rd
    stage = stage_all[frame]
    where = 'stage 1: reaction-diffusion (creates / removes features)' if in_rd else \
        f'stage 2: Hamilton-Jacobi level set (moves the boundary), {"sharp" if stage == 0 else f"ε = {stage:g} px"}'
    suptitle.set_text(f'Hybrid topology optimization, 64²  —  {where}  —  J = {objective[frame]:.3g}')
    phi_image.set_data(phi_all[frame].T)
    for artist in shoreline:
        artist.remove()
    shoreline[:] = [ax_phi.contour(x, x, phi_all[frame].T, levels=[0], colors='k', linewidths=1.3)]
    phi_title.set_text('1. level set φ, black = boundary φ = 0\n' +
                       ('bounded φ ∈ [−1, 1]' if in_rd else 'signed distance (colour: ±2 px)'))
    density = density_all[frame]
    design_image.set_array(density)
    grey = np.mean((density > 1e-3) & (density < 1 - 1e-3))
    design_title.set_text(('2. design the mechanics sees: ρ = [φ > 0]' if in_rd else
                           '2. design the mechanics sees: ρ = [φ > 0] at the nodes' if stage == 0 else
                           '2. design the mechanics sees: ρ = H_ε(φ)') +
                          f'\n2 triangles per pixel, {100 * grey:.0f}% grey')
    previous = density_all[best_rd if frame == nb_rd else max(frame - 1, 0)]  # hand-over: from the best design
    change = np.where(density - previous > 0.5, 1, np.where(density - previous < -0.5, -1, 0))
    change_image.set_array(change)
    for line, dot, values in progress:
        line.set_data(iterations[:frame + 1], values[:frame + 1])
        dot.set_data(iterations[frame:frame + 1], values[frame:frame + 1])


out = os.path.join(HERE, 'figures', 'TO_hybrid_RD_to_HJ_evolution_N64')
writer = animation.FFMpegWriter(fps=FPS, bitrate=6000, codec='libx264', extra_args=['-pix_fmt', 'yuv420p'])
with writer.saving(fig, out + '.mp4', dpi=110):
    for frame in range(nb_frames):
        draw(frame)
        writer.grab_frame(facecolor=SURFACE)
        if frame == nb_rd - 1:  # pause at the hand-over
            for _ in range(FPS):
                writer.grab_frame(facecolor=SURFACE)
    for _ in range(2 * FPS):  # hold the final state for 2 s
        writer.grab_frame(facecolor=SURFACE)
draw(nb_rd - 1)
fig.savefig(out + '_handover_frame.png', facecolor=SURFACE, dpi=110)
draw(nb_frames - 1)
fig.savefig(out + '_last_frame.png', facecolor=SURFACE, dpi=110)
print(f'video: {out}.mp4 ({nb_frames} frames, {nb_rd} reaction-diffusion)')
