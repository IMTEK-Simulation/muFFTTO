"""1D check: what the Green and the dual preconditioner do to the spectrum (periodic bar, N cells, forward difference
D, K = D^T C D, a diffuse soft band with the AT2 profile, l = 4 cells, contrast 1/k; c_ref = 1, the solid).

On the mean-free strains e = D u (D is a bijection from mean-free u onto them), with Pi = projection onto mean-free
vectors:
    Green:  P_G K  ~  Pi C Pi                      eigenvalues = the stiffness values (compressed): many distinct ones
    dual:   P_D K  ~  Pi C^-1 Pi C  =  I - (Pi c^-1) (mean of c e)
            = identity except ONE eigenvalue mean(c) * mean(1/c)   (the macro-stress mode)
so in 1D the dual is exact up to a rank-1 term: CG converges in 2 iterations at any contrast.
(The nodal form with pinv(D^T D) is numerically useless here: pinv of the singular D loses ~1e11.)
    python dual_exact_1d.py
"""
import numpy as np
import scipy.linalg as sl

N, L = 128, 4.0
x = np.arange(N)
Q = sl.null_space(np.ones((1, N)))                                 # basis of the mean-free vectors
Pi = np.eye(N) - np.ones((N, N)) / N
for k in (1e-2, 1e-4, 1e-8):
    d = np.exp(-np.minimum(np.abs(x - N / 2), N - np.abs(x - N / 2)) / L)
    c = (1 - d) ** 2 * (1 - k) + k
    for name, M in (('Green', Q.T @ Pi @ np.diag(c) @ Pi @ Q),
                    ('dual', Q.T @ Pi @ np.diag(1 / c) @ Pi @ np.diag(c) @ Q)):
        ev = np.sort(np.real(np.linalg.eigvals(M)))
        n1 = int(np.sum(np.abs(ev - 1) > 1e-8))
        print(f'contrast {1 / k:6.0e} {name:5s}: eigenvalues in [{ev[0]:.4g}, {ev[-1]:.4g}], {n1} of {N - 1} differ '
              f'from 1' + (f' (prediction for the one: mean(c) mean(1/c) = {c.mean() * (1 / c).mean():.4g})'
                           if name == 'dual' else ''))
