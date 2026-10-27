"""Video: level set vs phase field (the runs of plot_TO_level_set_vs_phase_field.py, every finished one), one frame per
objective evaluation so that the runs advance in step:
    top       per run the design as the CG solves see it: log10 of the stiffness scale s = rho^p (1 - soft) + soft,
              C(x) = s C_0, on the exact pixels (STIFFNESS=0: the density rho instead)
    bottom    CG iterations per solve (mean of the 3 forward + 3 adjoint solves of the evaluation) and in total,
              and the fit of the target: stress mismatch sum_lc w |Sigma_h - Sigma_t|^2 / |Sigma_t|^2 of the
              accepted designs (phase field: without the regularization terms of its objective); future part light

    python make_TO_level_set_vs_phase_field_video.py   -> figures/TO_level_set_vs_phase_field_N64.mp4
"""
import os

import imageio_ffmpeg
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import netCDF4
import numpy as np
from matplotlib import animation

plt.rcParams['animation.ffmpeg_path'] = imageio_ffmpeg.get_ffmpeg_exe()
HERE = os.path.dirname(os.path.realpath(__file__))
PF = os.path.join(HERE, 'data', 'example_2D_elasticity_TO', 'N64_soft1e-05_energy-warm_{}_eta_0.03125_w_5.0_l_bfgs')
LS = os.path.join(HERE, 'data', 'example_2D_elasticity_TO_level_set',
                  'N64_soft1e-05_energy_{}_w_5.0_p1_eps2_alpha2_holes4')
LS_PF_START = LS.replace('_holes4', '_initphase')
PRECS = [('Green', 'Green (C_0)', '#2a78d6'), ('Green_Jacobi', 'Green-Jacobi (2μI)', '#eb6834')]
METHODS = [('phase field', PF, '-'), ('level set', LS, '--'), ('level set, PF start', LS_PF_START, ':'),
           ('level set from PF result', LS.replace('_holes4', '_initpf_final'), '-.')]  # LS_INIT=pf_final
STIFFNESS = os.environ.get('STIFFNESS', '1') == '1'
FPS = 10
SURFACE, TEXT_2, GRID, FUTURE = '#fcfcfb', '#52514e', '#e6e5e1', '#d4d3cd'

runs = []
for method, template, style in METHODS:
    for prec, label, color in PRECS:
        if not os.path.exists(template.format(prec) + '_final.nc'):  # unfinished run
            continue
        f = netCDF4.Dataset(template.format(prec) + '_final.nc')
        h = netCDF4.Dataset(template.format(prec) + '_history.nc')
        designs = np.asarray(h['phase_field_iterate'][:])[:, 0]
        if STIFFNESS:
            soft, p = float(f.soft_phase), float(f.p)
            designs = np.log10(np.clip(designs, 0, 1) ** p * (1 - soft) + soft)
        cg = (np.asarray(f.num_iteration_mech, dtype=float) +
              np.asarray(f.num_iteration_adjoint, dtype=float)).reshape(-1, 3).sum(1)
        if method.startswith('level set'):  # the last evaluation re-sets the final material (post-processing)
            cg = cg[:-1]
        evaluations = np.asarray(h['nb_evaluations'][:]).astype(int)
        if method.startswith('level set'):  # the level-set objective is the mismatch
            mismatch = np.asarray(h['objective'][:])
        else:  # mismatch of each accepted design = norms_sigma of its evaluation, times w = weight / 3 load cases
            mismatch = float(f.weight) / 3 * np.asarray(f.norms_sigma)[evaluations - 1]
        runs.append(dict(label=f'{method}\n{label}', color=color, style=style, designs=designs, cg=cg,
                         evaluations=evaluations, mismatch=mismatch))
nb_frames = max(len(r['cg']) for r in runs)

fig = plt.figure(figsize=(max(3.0 * len(runs) + 0.6, 13), 7.4), facecolor=SURFACE)
gs = fig.add_gridspec(1, len(runs), wspace=0.25, left=0.06, right=0.88, top=0.83, bottom=0.5)
gs_bottom = fig.add_gridspec(1, 3, wspace=0.3, left=0.06, right=0.98, top=0.4, bottom=0.08)
images, titles = [], []
for i, r in enumerate(runs):
    ax = fig.add_subplot(gs[0, i])
    images.append(ax.imshow(r['designs'][0].T, origin='lower', cmap='viridis' if STIFFNESS else 'Greys',
                            vmin=-5 if STIFFNESS else 0, vmax=0 if STIFFNESS else 1, interpolation='nearest',
                            extent=[0, 1, 0, 1]))
    ax.set_xticks([]), ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_edgecolor(r['color'])
        spine.set_linewidth(2.5)
    ax.text(0.5, 1.2, r['label'], transform=ax.transAxes, ha='center', va='bottom', fontsize=10)
    titles.append(ax.set_title('', fontsize=9, color=TEXT_2))
colorbar_ax = fig.add_axes([0.9, 0.5, 0.012, 0.33])
fig.colorbar(images[0], cax=colorbar_ax, label='log10 stiffness  (C = s C_0)' if STIFFNESS else 'density')

ax_solve = fig.add_subplot(gs_bottom[0, 0])
ax_total = fig.add_subplot(gs_bottom[0, 1])
ax_fit = fig.add_subplot(gs_bottom[0, 2])
lines = []
for r in runs:
    ev = np.arange(1, len(r['cg']) + 1)
    for ax, ev, values in ((ax_solve, ev, r['cg'] / 6), (ax_total, ev, np.cumsum(r['cg'])),
                           (ax_fit, r['evaluations'], r['mismatch'])):
        ax.semilogy(ev, values, color=FUTURE, ls=r['style'], lw=1.2)
        line, = ax.semilogy([], [], color=r['color'], ls=r['style'], lw=2, label=r['label'].replace('\n', ', '))
        dot, = ax.semilogy([], [], 'o', color=r['color'], ms=5)
        lines.append((line, dot, ev, values))
ax_solve.set(xlabel='objective evaluation', ylabel='CG iterations per solve', title='per solve')
ax_total.set(xlabel='objective evaluation', ylabel='CG iterations, cumulative', title='total')
ax_fit.set(xlabel='objective evaluation', ylabel='stress mismatch to target',
           title='fit of the target (0 = exact)')
ax_solve.legend(frameon=False, fontsize=7.5, loc='upper right')
for ax in (ax_solve, ax_total, ax_fit):
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID)
    ax.set_xlim(0, nb_frames + 1)
suptitle = fig.suptitle('', fontsize=12, y=0.985)


def draw(evaluation):
    """State after `evaluation` objective evaluations (a run that has finished stays at its final design)."""
    suptitle.set_text(f'objective evaluation {evaluation}')
    for r, image, title in zip(runs, images, titles):
        frame = max(np.searchsorted(r['evaluations'], evaluation, side='right') - 1, 0)
        image.set_data(r['designs'][frame].T)
        e = min(evaluation, len(r['cg']))
        title.set_text(f'{"finished, " if evaluation > len(r["cg"]) else ""}CG/solve {r["cg"][e - 1] / 6:.0f}, '
                       f'total {int(r["cg"][:e].sum()):,}\nmismatch {r["mismatch"][frame]:.3g}')
    for line, dot, ev, values in lines:
        n = max(np.searchsorted(ev, evaluation, side='right'), 1)
        line.set_data(ev[:n], values[:n])
        dot.set_data(ev[n - 1:n], values[n - 1:n])


out = os.path.join(HERE, 'figures', 'TO_level_set_vs_phase_field' + ('' if STIFFNESS else '_density') + '_N64')
writer = animation.FFMpegWriter(fps=FPS, bitrate=6000, codec='libx264', extra_args=['-pix_fmt', 'yuv420p'])
with writer.saving(fig, out + '.mp4', dpi=120):
    for evaluation in range(1, nb_frames + 1):
        draw(evaluation)
        writer.grab_frame(facecolor=SURFACE)
    for _ in range(2 * FPS):  # hold the final state for 2 s
        writer.grab_frame(facecolor=SURFACE)
fig.savefig(out + '_last_frame.png', facecolor=SURFACE, dpi=120)
print(f'video: {out}.mp4 ({nb_frames} frames)')
