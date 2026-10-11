"""
Plot the NetCDF output of example_2D_phase_field_fracture_AT2.py.

    python plot_phase_field_fracture_output.py [file.nc] [frame ...]
    python plot_phase_field_fracture_output.py [file.nc] --animate

Without frames: the frame at the peak stress and the last frame. With
--animate: an animation over all stored frames (every load step, or every
staggered iteration if the run used output_every = 'iteration'), saved next to
the file as <file>.mp4 (<file>.gif without ffmpeg) and then shown. Works for
files written by serial and parallel runs (the stored fields are global).
Needs the netCDF4 package.
"""
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import numpy as np
import matplotlib.pyplot as plt

from muFFTTO import io_utils

arguments = [a for a in sys.argv[1:] if not a.startswith('--')]
animate = '--animate' in sys.argv
file_name = arguments[0] if arguments and arguments[0].endswith('.nc') else \
    os.path.join(os.path.dirname(__file__), 'exp_data', 'phase_field_fracture_AT2.nc')
run = io_utils.read_file(file_name)
print(run)

load = run.frame_variables['load']
stress_xx = run.frame_variables['macro_stress'][:, 0, 0]
damage = run.fields['damage']                     # [frame, 1, 1, x, y]
# converged load steps (files without 'converged' store only those)
converged = run.frame_variables['converged'] == 1 if 'converged' in run.frame_variables \
    else np.ones(run.nb_frames, dtype=bool)
size = run.attributes['domain_size']


def frame_title(frame):
    title = rf'frame {frame}, $\bar{{\varepsilon}}_{{xx}}$ = {load[frame]:.4f}'
    if 'staggered_iteration' in run.frame_variables:
        title += f", iteration {int(run.frame_variables['staggered_iteration'][frame])}"
    return title


def plot_curve(ax, marked_frames):
    ax.plot(np.r_[0, load[converged]], np.r_[0, stress_xx[converged]], 'o-', markersize=3)
    ax.plot(load[marked_frames], stress_xx[marked_frames], 'rs')
    ax.set_xlabel(r'$\bar{\varepsilon}_{xx}$')
    ax.set_ylabel(r'$\bar{\sigma}_{xx}$')
    ax.set_title(f"{run.attributes['nb_grid_pts']} grid, Gc = {run.attributes['Gc']}")


def plot_damage(ax, frame):
    image = ax.imshow(damage[frame, 0, 0].T, origin='lower', vmin=0, vmax=1, cmap='inferno',
                      extent=(0, size[0], 0, size[1]))
    ax.set_title(frame_title(frame), fontsize=10)
    return image


if animate:
    from matplotlib import animation

    fig, (ax_d, ax_s) = plt.subplots(1, 2, figsize=(10, 4.3))

    def draw(frame):
        ax_d.clear()
        ax_s.clear()
        plot_damage(ax_d, frame)
        plot_curve(ax_s, [frame])

    movie = animation.FuncAnimation(fig, draw, frames=run.nb_frames, interval=100, repeat=False)
    # saved next to the .nc file: mp4 with ffmpeg, gif otherwise
    if animation.writers.is_available('ffmpeg'):
        movie_file = os.path.splitext(file_name)[0] + '.mp4'
        movie.save(movie_file, writer=animation.FFMpegWriter(fps=10), dpi=100)
    else:
        movie_file = os.path.splitext(file_name)[0] + '.gif'
        movie.save(movie_file, writer=animation.PillowWriter(fps=10), dpi=80)
    print(f'animation saved to {movie_file}')
    plt.show()
else:
    peak = int(np.flatnonzero(converged)[np.argmax(stress_xx[converged])])
    frames = [int(f) for f in arguments if not f.endswith('.nc')] or [peak, run.nb_frames - 1]
    fig, axes = plt.subplots(1, 1 + len(frames), figsize=(4.5 * (1 + len(frames)), 4))
    plot_curve(axes[0], frames)
    for ax, frame in zip(axes[1:], frames):
        fig.colorbar(plot_damage(ax, frame), ax=ax)
    fig.tight_layout()
    plt.show()
