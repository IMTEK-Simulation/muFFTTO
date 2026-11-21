"""
Read the NetCDF output of the third-medium contact runs (muFFTTO.io_utils).

    from read_tmc import load

    r = load('tmc_run.nc')

    r.hist['min_det_F']      per-frame history (numpy array), one entry per frame
    r.attrs['k_v']           run parameters, as written
    r.frames                 increment number of each stored frame
    r.field('F_flat', -1)    a stored field at the last frame, [*components, 1, x, y]
    r.detF(-1)               det F of that frame, [x, y]
    r.phase                  the static phase field, [x, y]

To restart from a frame, fill the fields of a discretization instead:

    io_utils.load_fields('tmc_run.nc', [displacement_fluctuation_field], frame=-1)

Command line:

    python read_tmc.py tmc_run.nc              summary
    python read_tmc.py tmc_run.nc --plot       response curves
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from muFFTTO import io_utils  # noqa: E402


class TMCRun:
    """One run: global attributes, per-frame history and the stored fields."""

    def __init__(self, path):
        self.path = path
        content = io_utils.read_file(path)
        self.attrs = content.attributes
        self.fields = content.fields
        self.hist = {name: np.asarray(value) for name, value in content.frame_variables.items()}
        self.nb_frames = content.nb_frames
        self.frames = self.hist['increment'].astype(int)

    @property
    def converged(self):
        """True only if every stored increment converged."""
        return bool(np.all(self.hist['converged'] == 1))

    @property
    def bad_increments(self):
        """Stored increments that did NOT reach gtol -- not equilibria."""
        return self.frames[self.hist['converged'] != 1]

    def field(self, name, frame=-1):
        """One field at one frame. `frame` indexes the STORED frames, so -1
        is the last one written, not increment -1."""
        if name not in self.fields:
            raise KeyError(f'{name!r} not in {sorted(self.fields)}')
        return self.fields[name][frame]

    def detF(self, frame=-1):
        """det F (quadrature-averaged F) as an [x, y] array."""
        return self.field('detF', frame)[0, 0]

    @property
    def phase(self):
        """The static phase field: > 0 matrix, == 0 third medium (see the
        `phase_note` attribute)."""
        return self.field('phase_field', 0)[0, 0]

    def summary(self):
        a, h = self.attrs, self.hist
        out = [f'file        : {self.path}',
               f'command     : {a.get("command_line", "?")}',
               f'geometry    : {a.get("geometry", "?")}   element {a.get("element_type", "?")}',
               f'grid        : {np.atleast_1d(a.get("nb_grid_pts", [])).tolist()}',
               f'frames      : {self.nb_frames}, increments {self.frames.min()}..{self.frames.max()}'
               + (f' of {int(a["ninc"])}' if 'ninc' in a else ''),
               f'fields      : {sorted(self.fields)}',
               f'elapsed     : {h["elapsed_time"][-1]:.1f} s',
               f'converged   : {self.converged}']
        bad = self.bad_increments
        if bad.size:
            out.append(f'  NOT converged at {bad.size} increment(s): '
                       f'{bad[:12].tolist()}{" ..." if bad.size > 12 else ""}')
            out.append('  those states are not equilibria')
        out.append(f'min det F   : {h["min_det_F"].min():.4e} (final {h["min_det_F"][-1]:.4e})')
        out.append(f'total hessp : {int(h["nb_hessp"].sum())}')
        return '\n'.join(out)


def load(path):
    return TMCRun(path)


def plot_response(run, file_name=None):
    """F-P, F-Pxx, min det F and energy over the stored frames."""
    from matplotlib import pyplot as plt
    h = run.hist
    # the retract script drives a general component: F_driven / P_driven
    if 'F10' in h:
        F, P, label = h['F10'], h['P10'], '10'
    else:
        F, P = h['F_driven'], h['P_driven']
        label = ''.join(str(int(i)) for i in np.atleast_1d(run.attrs['driven_component']))
    fig, ax = plt.subplots(1, 4, figsize=(19, 4))
    ax[0].plot(F, P, '-o', ms=3, color='k')
    ax[0].set_xlabel(rf'$\bar{{F}}_{{{label}}}$')
    ax[0].set_ylabel(rf'$\bar{{P}}_{{{label}}}$')
    ax[1].plot(F, h['Pxx'], '-o', ms=3, color='C0')
    ax[1].set_xlabel(rf'$\bar{{F}}_{{{label}}}$')
    ax[1].set_ylabel(r'$\bar{P}_{xx}$')
    ax[2].semilogy(h['lam'], h['min_det_F'], '-o', ms=3, color='C3')
    ax[2].set_xlabel(r'$\lambda$')
    ax[2].set_ylabel(r'$\min \det F$')
    ax[3].plot(h['lam'], h['energy'], '-o', ms=3, color='C2')
    ax[3].set_xlabel(r'$\lambda$')
    ax[3].set_ylabel(r'$\Pi$')
    # mark the increments that never converged
    for lam in h['lam'][h['converged'] != 1]:
        for a in ax[2:]:
            a.axvline(lam, color='0.8', lw=0.6, zorder=0)
    for a in ax:
        a.grid(alpha=.3)
    fig.tight_layout()
    if file_name:
        fig.savefig(file_name, dpi=150)
    return fig


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('path')
    p.add_argument('--plot', action='store_true', help='response curves')
    args = p.parse_args()

    r = load(args.path)
    print(r.summary())
    if args.plot:
        from matplotlib import pyplot as plt
        plot_response(r)
        plt.show()


if __name__ == '__main__':
    main()
