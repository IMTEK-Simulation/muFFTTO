"""
Read the NetCDF output of the third-medium contact runs.

    from read_tmc import load

    r = load('tmc_run.nc')

    r.hist['min_J']          per-increment history (numpy array)
    r.attrs['k_v']           run parameters, as written
    r.frames                 increment number of each stored frame
    r.field('F', -1)         the F field of the last frame
    r.detF(-1)               det F of that frame, recomputed from F
    r.phase                  the static phase field, wherever it was stored

Command line:

    python read_tmc.py tmc_run.nc              summary
    python read_tmc.py tmc_run.nc --plot       response curves
"""

import argparse

import numpy as np
from netCDF4 import Dataset


class TMCRun:
    """One run: global attributes, scalar histories and the stored frames."""

    def __init__(self, path):
        self.path = path
        self._ds = Dataset(path, 'r')

        self.attrs = {k: self._ds.getncattr(k) for k in self._ds.ncattrs()}

        # Histories were written as fixed-length placeholders and overwritten
        # with the real values, so trailing slots may be unused. `hist_lam` is
        # strictly increasing while real, which gives the true length.
        self.hist = {k[len('hist_'):]: np.atleast_1d(v)
                     for k, v in self.attrs.items() if k.startswith('hist_')}
        self.n = self._true_length()
        self.hist = {k: v[:self.n] for k, v in self.hist.items()}

        # -1 marks unused frame slots.
        frames = np.atleast_1d(self.attrs.get('frame_increments', []))
        self.frames = frames[frames >= 0].astype(int)

        self.vars = list(self._ds.variables)

    # -- scalars --------------------------------------------------------
    def _true_length(self):
        lam = self.hist.get('lam')
        if lam is None or lam.size == 0:
            return 0
        nz = np.nonzero(lam)[0]
        return int(nz[-1]) + 1 if nz.size else 0

    @property
    def converged(self):
        """True only if every increment converged."""
        return bool(np.atleast_1d(self.attrs.get('converged', [0]))[0])

    @property
    def bad_increments(self):
        """1-based increments that did NOT reach gtol -- not equilibria."""
        c = self.hist.get('converged')
        if c is None:
            return np.array([], dtype=int)
        return np.nonzero(c == 0)[0] + 1

    # -- fields ---------------------------------------------------------
    def field(self, name, frame=-1):
        """One field at one frame. `frame` indexes the STORED frames, so -1
        is the last one written, not increment -1."""
        if name not in self._ds.variables:
            raise KeyError(f'{name!r} not in {self.vars}')
        return np.asarray(self._ds.variables[name][frame])

    def detF(self, frame=-1):
        """det F, recomputed from F: it is not stored (pure function of F)."""
        F = self.field('F', frame)
        F = np.squeeze(F)
        return F[0, 0] * F[1, 1] - F[0, 1] * F[1, 0]

    @property
    def phase(self):
        """The static phase field, from wherever it ended up.

        phase > 0 is the matrix, phase == 0 the third medium -- see the
        `phase_note` attribute, which travels with the file.
        """
        if 'phase_field' in self._ds.variables:
            return np.squeeze(np.asarray(self._ds.variables['phase_field'][0]))
        if 'phase_field_flat' in self.attrs:
            shape = [int(n) for n in
                     np.atleast_1d(self.attrs['phase_field_shape'])]
            return np.asarray(self.attrs['phase_field_flat']).reshape(shape)
        raise KeyError('no phase field in this file')

    def close(self):
        self._ds.close()

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()

    def summary(self):
        a = self.attrs
        out = [f'file        : {self.path}',
               f'command     : {a.get("command_line", "?")}',
               f'geometry    : {a.get("geometry", "?")}   '
               f'element {a.get("element_type", "?")}',
               f'grid        : {list(np.atleast_1d(a.get("nb_grid_pts", [])))}',
               f'increments  : {self.n} of {int(np.atleast_1d(a.get("ninc", [0]))[0])}',
               f'frames      : {len(self.frames)}  at increments '
               f'{self.frames.tolist()}',
               f'fields      : {self.vars}',
               f'elapsed     : {float(np.atleast_1d(a.get("elapsed_time", [0]))[0]):.1f} s',
               f'converged   : {self.converged}']
        bad = self.bad_increments
        if bad.size:
            out.append(f'  NOT converged at {bad.size} increment(s): '
                       f'{bad[:12].tolist()}{" ..." if bad.size > 12 else ""}')
            out.append('  those states are not equilibria')
        if self.n:
            out.append(f'min det F   : {self.hist["min_J"].min():.4e} '
                       f'(final {self.hist["min_J"][-1]:.4e})')
            out.append(f'total hessp : {int(self.hist["nb_hessp"].sum())}')
        return '\n'.join(out)


def load(path):
    return TMCRun(path)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('path')
    p.add_argument('--plot', action='store_true', help='response curves')
    args = p.parse_args()

    r = load(args.path)
    print(r.summary())

    if args.plot:
        from matplotlib import pyplot as plt
        h = r.hist
        fig, ax = plt.subplots(1, 4, figsize=(19, 4))
        ax[0].plot(h['F10'], h['P10'], '-o', ms=3, color='k')
        ax[0].set_xlabel(r'$\bar{F}_{10}$')
        ax[0].set_ylabel(r'$\bar{P}_{10}$')
        ax[1].plot(h['F10'], h['Pxx'], '-o', ms=3, color='C0')
        ax[1].set_xlabel(r'$\bar{F}_{10}$')
        ax[1].set_ylabel(r'$\bar{P}_{xx}$')
        ax[2].semilogy(h['lam'], h['min_J'], '-o', ms=3, color='C3')
        ax[2].set_xlabel(r'$\lambda$')
        ax[2].set_ylabel(r'$\min \det F$')
        ax[3].plot(h['lam'], h['energy'], '-o', ms=3, color='C2')
        ax[3].set_xlabel(r'$\lambda$')
        ax[3].set_ylabel(r'$\Pi$')
        # mark the increments that never converged
        for i in r.bad_increments:
            if i - 1 < len(h['lam']):
                for a in ax[2:]:
                    a.axvline(h['lam'][i - 1], color='0.8', lw=0.6, zorder=0)
        for a in ax:
            a.grid(alpha=.3)
        fig.tight_layout()
        plt.show()

    r.close()


if __name__ == '__main__':
    main()
