"""
Plot the NetCDF output of example_2D_phase_field_fracture_contact.py.

    python plot_phase_field_fracture_contact_output.py [file.nc] [frame ...] [--save]
    python plot_phase_field_fracture_contact_output.py [file.nc] --animate [--save]

Left: mean stress and maximal damage over the applied deformation, for the
driven component (the largest entry of H_macro, e.g. F_10 in shear, F_11 in
compression).
Right: the damage on the deformed cell (solid pixels; the third medium is
drawn light blue), for the chosen frames -- default: the frame of the peak
stress and the last one. With --animate all frames are rendered into
<file>.mp4 (<file>.gif without ffmpeg) next to the file and then shown.
With --save nothing is shown; the static figure is saved as response.png in
the folder of the file. Without a file: the newest run in exp_data.
Needs the netCDF4 package.
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import numpy as np  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, ListedColormap  # noqa: E402

from muFFTTO import io_utils  # noqa: E402

arguments = [a for a in sys.argv[1:] if not a.startswith('--')]
animate = '--animate' in sys.argv
save = '--save' in sys.argv
if arguments and arguments[0].endswith('.nc'):
    file_name = arguments[0]
else:                                       # the newest run
    import glob
    runs = glob.glob(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'exp_data',
                                  'phase_field_fracture_contact', '*', 'contact_fracture.nc'))
    file_name = max(runs, key=os.path.getmtime)
run = io_utils.read_file(file_name)
print(run)

h = run.frame_variables
d_max, lam = h['damage_max'], h['lam']
H_macro = np.asarray(run.attributes['H_macro']).reshape(2, 2)
# driven component: the largest entry of H_macro
i_d, j_d = np.unravel_index(np.argmax(np.abs(H_macro)), H_macro.shape)
label = f'{i_d}{j_d}'
F_d = (i_d == j_d) + lam * H_macro[i_d, j_d]                # F_bar_ij = delta_ij + lam H_ij
# stress relative to the shear modulus of the solid, mu = E / (2 (1 + nu))
mu_solid = float(run.attributes['E_solid']) / (2.0 * (1.0 + float(run.attributes['nu_solid'])))
P_d = h['mean_P'][:, i_d, j_d] / mu_solid
size = np.asarray(run.attributes['domain_size'], dtype=float)
phase = run.fields['phase'][0, 0, 0]                       # [x, y], 1 solid / 0 third medium
nx, ny = phase.shape
medium_cmap = ListedColormap([(0.75, 0.87, 1.0)])
# damage: black (intact) -> red (damaged)
damage_cmap = LinearSegmentedColormap.from_list('damage', ['black', (0.9, 0.0, 0.0)])
# fully cracked solid pixels (damage above this) are not drawn -> the crack shows as a white gap
crack_threshold = 0.95


def deformed_corners(frame):
    """Positions of the (nx+1) x (ny+1) pixel corners: (I + lam H) X + u~(X), u~ periodic."""
    i, j = np.meshgrid(np.arange(nx + 1), np.arange(ny + 1), indexing='ij')
    X = np.array([i * size[0] / nx, j * size[1] / ny])
    u = run.fields['u_fluc'][frame][:, 0]                   # node 0 = pixel corner, [2, x, y]
    u = u[:, i % nx, j % ny]
    return np.einsum('ab,b...->a...', np.eye(2) + lam[frame] * H_macro, X) + u


def pixel_damage(frame):
    """Damage per pixel: mean of its four corner values (node 0 of the nodal field)."""
    d = run.fields['damage'][frame, 0, 0]
    return 0.25 * (d + np.roll(d, -1, 0) + np.roll(d, -1, 1) + np.roll(d, -1, (0, 1)))


def plot_cell(ax, frame):
    x, y = deformed_corners(frame)
    solid = phase > 0
    ax.pcolormesh(x, y, np.ma.masked_where(solid, np.ones_like(phase)), cmap=medium_cmap, shading='flat')
    damage = pixel_damage(frame)
    cracked = damage > crack_threshold
    image = ax.pcolormesh(x, y, np.ma.masked_where(~solid | cracked, damage), cmap=damage_cmap,
                          vmin=0, vmax=1, shading='flat')
    ax.set_aspect('equal')
    ax.set_title(f'frame {frame}: F{label} = {F_d[frame]:+.3f}, max d = {d_max[frame]:.2f}', fontsize=10)
    return image


def plot_response(ax, marked_frames):
    F0 = float(i_d == j_d)
    ax.plot(np.r_[F0, F_d], np.r_[0, P_d], 'k.-', ms=3)
    ax.plot(F_d[marked_frames], P_d[marked_frames], 'rs')
    ax.set_xlabel(rf'$\bar F_{{{label}}}$')
    ax.set_ylabel(rf'$\bar P_{{{label}}} \,/\, \mu$')
    twin = ax.twinx()
    twin.plot(np.r_[F0, F_d], np.r_[0, d_max], '-', color='C3', lw=1)
    twin.set_ylim(0, 1.05)
    twin.set_ylabel('max d', color='C3')
    ax.set_title(f"{run.attributes['geometry']}, {nx}x{ny}, Gc = {run.attributes['Gc']}", fontsize=10)


if animate:
    from matplotlib import animation

    # no system ffmpeg: use the one shipped with imageio-ffmpeg (if installed), so the movie is an mp4
    if not animation.writers.is_available('ffmpeg'):
        try:
            import imageio_ffmpeg
            plt.rcParams['animation.ffmpeg_path'] = imageio_ffmpeg.get_ffmpeg_exe()
        except ImportError:
            pass

    fig, (ax_r, ax_c) = plt.subplots(1, 2, figsize=(11, 4.6))

    def draw(frame):
        fig.clf()
        ax_r, ax_c = fig.subplots(1, 2)
        plot_response(ax_r, [frame])
        fig.colorbar(plot_cell(ax_c, frame), ax=ax_c, label='damage')
        fig.tight_layout()

    movie = animation.FuncAnimation(fig, draw, frames=run.nb_frames, interval=100, repeat=False)
    if animation.writers.is_available('ffmpeg'):
        movie_file = os.path.splitext(file_name)[0] + '.mp4'
        movie.save(movie_file, writer=animation.FFMpegWriter(fps=8), dpi=200)
    else:
        movie_file = os.path.splitext(file_name)[0] + '.gif'
        movie.save(movie_file, writer=animation.PillowWriter(fps=8), dpi=80)
    print(f'animation saved to {movie_file}')
    if not save:
        plt.show()
else:
    peak = int(np.argmax(np.abs(P_d)))
    frames = [int(f) for f in arguments if not f.endswith('.nc')] or sorted({peak, run.nb_frames - 1})
    fig, axes = plt.subplots(1, 1 + len(frames), figsize=(5.2 * (1 + len(frames)), 4.6))
    plot_response(axes[0], frames)
    for ax, frame in zip(axes[1:], frames):
        fig.colorbar(plot_cell(ax, frame), ax=ax, label='damage')
    fig.tight_layout()
    if save:
        fig.savefig(os.path.join(os.path.dirname(file_name), 'response.png'), dpi=200)
    else:
        plt.show()
