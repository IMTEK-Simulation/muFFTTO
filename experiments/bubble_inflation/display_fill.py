"""Display-only fill of pinned void nodes (e.g. a gas region): the discrete harmonic extension of the surrounding
displacement (6-neighbour Laplace equation on the masked nodes, periodic), solved by sparse CG with a warm start."""
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla


class HarmonicFill:
    def __init__(self, mask):
        self.mask = mask
        self.shape = mask.shape
        idx = -np.ones(mask.shape, int)
        idx[mask] = np.arange(mask.sum())
        self.idx = idx
        rows, cols = [], []
        self.nbr = []
        n = int(mask.sum())
        for ax in range(3):
            for s in (1, -1):
                nb = np.roll(idx, -s, ax)[mask]          # index of the neighbour (or -1 if it is not masked)
                self.nbr.append((ax, s))
                inside = nb >= 0
                rows.append(np.arange(n)[inside])
                cols.append(nb[inside])
        A = sp.csr_matrix((np.full(sum(len(r) for r in rows), -1.0), (np.concatenate(rows), np.concatenate(cols))),
                          shape=(n, n))
        self.A = (A + 6.0 * sp.identity(n)).tocsr()

    def __call__(self, u, start=None):
        """u [3, x, y, z]: returns u with the masked nodes replaced by the harmonic extension"""
        out = u.copy()
        for c in range(u.shape[0]):
            known = np.where(self.mask, 0.0, u[c])
            b = sum(np.roll(known, -s, ax)[self.mask] for ax, s in self.nbr)
            x0 = None if start is None else start[c][self.mask]
            x, _ = spla.cg(self.A, b, x0=x0, rtol=1e-8, maxiter=5000)
            out[c][self.mask] = x
        return out
