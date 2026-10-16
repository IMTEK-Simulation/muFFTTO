"""
Why does the dual preconditioner G K(A^-1) G work for 2D conductivity and not for elasticity?

Small, self-contained study (numpy, linear triangles on a periodic N x N grid, FFT Green operators, dense matrices,
so spectra are exact). Scalar conductivity and small-strain linear elasticity (plane strain, Mandel notation) on
the same geometries, voids / inclusions of relative stiffness k_v.

Exact structure. The quadrature fields (strains / fluxes) split W-orthogonally into E = range(B) (compatible) and
S = {s : B^T W s = 0} (self-equilibrated: rotated gradients for the scalar problem, Airy stresses for elasticity).
With G_I = (B^T W B)^-1, sigma_0 = B G_I f and Q = A^-1, the complementary energy gives

    K^-1 = G_I B^T W [Q - Q Gamma_S (Gamma_S Q Gamma_S)^+ Gamma_S Q] B G_I .

The dual preconditioner keeps the first term; the neglected term is PSD, hence lambda_min(P K) >= 1.

Preconditioners (P ~ K^-1):
    green            G of the solid (C_ref = solid tangent / conductivity)
    green_jacobi     D^-1/2 G_I D^-1/2, D = diag K
    dual             G_I B^T W Q B G_I
    dual_solid       G_s B^T W C_s Q C_s B G_s        (reference with the bulk part, elasticity)
    schur_neumann_k  first term minus the correction with S~ = truncated Neumann series of (Gamma_S Q Gamma_S)^+
                     (k terms; every partial sum <= the inverse, so lambda_min >= 1 is kept)
    schur_A          S~ = Gamma_S A Gamma_S  (an upper bound of the inverse: captures the heterogeneity,
                     no guarantee, may be indefinite)
    schur_exact      S~ = exact pseudo-inverse (check: P = K^-1)

    python study.py 16 32            -> results/study_N16.json, results/study_N32.json
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import scipy.linalg as sl

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from muFFTTO import geometry  # noqa: E402

HERE = Path(__file__).resolve().parent
RTOL = 1e-6                     # PCG stop: ||e_k||_K <= RTOL ||e_0||_K (true energy error)
E_SOLID, NU = 100.0, 0.3        # elasticity: solid of the slide problems
MU = E_SOLID / (2 * (1 + NU))
LAM = E_SOLID * NU / ((1 + NU) * (1 - 2 * NU))


# ----------------------------------------------------------------------------------------------------------------
# Discretization: linear triangles, 2 per pixel (as in the paper-1 notebooks)
# ----------------------------------------------------------------------------------------------------------------
class Grid:
    def __init__(self, N, physics):
        self.N, self.physics = N, physics
        self.du = 1 if physics == 'scalar' else 2               # unknowns per node
        self.nc = 2 if physics == 'scalar' else 3               # strain components (Mandel for elasticity)
        h = 1.0 / N
        Bl = np.zeros([2, 2, 4])                                 # [dim, quad, node 0..3 = (0,0),(0,1),(1,0),(1,1)]
        Bl[:, 0, :] = [[-1 / h, 0, 1 / h, 0], [-1 / h, 1 / h, 0, 0]]
        Bl[:, 1, :] = [[0, -1 / h, 0, 1 / h], [0, 0, -1 / h, 1 / h]]
        self.Bl = Bl.reshape(2, 2, 2, 2)
        self.w = h * h / 2                                       # weight of each quadrature point
        # strain = M . grad u (grad u flattened as [d_u, dim])
        if physics == 'scalar':
            self.M = np.eye(2)
        else:
            s = 1 / np.sqrt(2)
            self.M = np.array([[1, 0, 0, 0], [0, 0, 0, 1], [0, s, s, 0]], dtype=float)   # e11, e22, sqrt2 e12
        self.n = self.du * N * N

    def grad(self, u):                                           # u [du, N, N] -> [du, dim, q, N, N]
        g = np.zeros((self.du, 2, 2, self.N, self.N))
        for a in range(2):
            for b in range(2):
                g += np.einsum('jq,ixy->ijqxy', self.Bl[:, :, a, b], np.roll(u, (-a, -b), axis=(1, 2)))
        return g

    def grad_T(self, g):                                         # weighted transpose
        out = np.zeros((self.du, self.N, self.N))
        for a in range(2):
            for b in range(2):
                out += np.roll(np.einsum('jq,ijqxy->ixy', self.Bl[:, :, a, b], g), (a, b), axis=(1, 2))
        return self.w * out

    def B(self, u):                                              # nodal -> strain [nc, q, N, N]
        g = self.grad(u.reshape(self.du, self.N, self.N)).reshape(self.du * 2, 2, self.N, self.N)
        return np.einsum('cg,gqxy->cqxy', self.M, g)

    def BtW(self, s):                                            # strain -> nodal (weighted)
        g = np.einsum('cg,cqxy->gqxy', self.M, s).reshape(self.du, 2, 2, self.N, self.N)
        return self.grad_T(g).reshape(-1)

    def K(self, A):                                              # A [nc, nc, q, N, N] or [nc, nc]
        if A.ndim == 2:
            return lambda u: self.BtW(np.einsum('cd,dqxy->cqxy', A, self.B(u)))
        return lambda u: self.BtW(np.einsum('cdqxy,dqxy->cqxy', A, self.B(u)))

    def green(self, C):
        """G = (B^T W C B)^-1 for a constant C (FFT, zero frequency -> 0)."""
        K = self.K(C)
        N, du = self.N, self.du
        symbol = np.zeros((du, du, N, N), dtype=complex)
        for d in range(du):
            e = np.zeros((du, N, N))
            e[d, 0, 0] = 1
            symbol[:, d] = np.fft.fft2(K(e).reshape(du, N, N))
        sym = symbol.transpose(2, 3, 0, 1).copy()
        sym[0, 0] = np.eye(du)
        inv = np.linalg.inv(sym)
        inv[0, 0] = 0
        inv = inv.transpose(2, 3, 0, 1)
        return lambda r: np.real(np.fft.ifft2(np.einsum('abxy,bxy->axy', inv, np.fft.fft2(r.reshape(du, N, N)))
                                              )).reshape(-1)


def material(grid, geometry_name, k_v):
    """A, A^-1 [nc, nc, q, N, N] and the solid tangent; phase on pixel centres, both triangles of a pixel equal."""
    N = grid.N
    coords = (np.indices((N, N)) + 0.5) / N
    phase = np.asarray(geometry.get(geometry_name, coords), dtype=float)          # 1 solid, 0 void / inclusion
    if grid.physics == 'scalar':
        C_s = np.eye(2)
    else:
        m = np.array([1.0, 1.0, 0.0])
        C_s = 2 * MU * np.eye(3) + LAM * np.outer(m, m)
    scale = phase + k_v * (1 - phase)
    A = np.einsum('cd,xy->cdxy', C_s, scale)[:, :, None].repeat(2, axis=2)
    Q = np.einsum('cd,xy->cdxy', np.linalg.inv(C_s), 1 / scale)[:, :, None].repeat(2, axis=2)
    return A, Q, C_s, phase


def preconditioners(grid, A, Q, C_s):
    nc = grid.nc
    I_ref = np.eye(nc)
    G_I, G_s = grid.green(I_ref), grid.green(C_s)
    K = grid.K(A)
    apply_Q = lambda s: np.einsum('cdqxy,dqxy->cqxy', Q, s)
    apply_A = lambda s: np.einsum('cdqxy,dqxy->cqxy', A, s)
    gamma_E = lambda s: grid.B(G_I(grid.BtW(s)))
    gamma_S = lambda s: s - gamma_E(s)
    q_max = float(np.max(np.linalg.eigvalsh(Q.transpose(2, 3, 4, 0, 1))))

    # Jacobi diagonal of K by Dirac combs (stencil within one pixel, even N)
    diag = np.zeros((grid.du, grid.N, grid.N))
    for d in range(grid.du):
        for a in range(2):
            for b in range(2):
                e = np.zeros((grid.du, grid.N, grid.N))
                e[d, a::2, b::2] = 1
                diag[d, a::2, b::2] = K(e).reshape(grid.du, grid.N, grid.N)[d, a::2, b::2]
    jac = (1 / np.sqrt(diag)).reshape(-1)

    def schur(r, s_tilde):
        sigma0 = grid.B(G_I(r))
        y = apply_Q(sigma0)
        if s_tilde is not None:
            y = y - apply_Q(gamma_S(s_tilde(gamma_S(y))))
        return G_I(grid.BtW(y))

    def neumann(k):
        def s_tilde(t):                                          # (1/q_max) sum_{j<k} (I - X)^j t, X = Gs Q Gs / q_max
            z, term = t / q_max, t / q_max
            for _ in range(k - 1):
                term = term - gamma_S(apply_Q(term)) / q_max
                z = z + term
            return z
        return s_tilde

    def exact_s_tilde():
        # dense pseudo-inverse of Gamma_S Q Gamma_S on the quadrature space (small N only)
        m = nc * 2 * grid.N * grid.N
        shape = (nc, 2, grid.N, grid.N)
        X = np.empty((m, m))
        for i in range(m):
            e = np.zeros(m)
            e[i] = 1
            X[:, i] = gamma_S(apply_Q(gamma_S(e.reshape(shape)))).reshape(-1)
        Xp = np.linalg.pinv(0.5 * (X + X.T), rcond=1e-10, hermitian=True)
        return lambda t: (Xp @ t.reshape(-1)).reshape(shape)

    P = {
        'green': (G_s, 1),
        'green_jacobi': (lambda r: jac * G_I(jac * r), 1),
        'dual': (lambda r: schur(r, None), 2),
        'dual_solid': (lambda r: G_s(grid.BtW(np.einsum('cd,dqxy->cqxy', C_s, apply_Q(
            np.einsum('cd,dqxy->cqxy', C_s, grid.B(G_s(r))))))), 2),
        'schur_neumann_1': (lambda r: schur(r, neumann(1)), 4),
        'schur_neumann_3': (lambda r: schur(r, neumann(3)), 6),
        'schur_A': (lambda r: schur(r, lambda t: apply_A(t)), 4),
    }
    if grid.N <= 16:
        s_ex = exact_s_tilde()
        P['schur_exact'] = (lambda r: schur(r, s_ex), np.nan)
    return K, P


def dense(op, n):
    M = np.empty((n, n))
    for i in range(n):
        e = np.zeros(n)
        e[i] = 1
        M[:, i] = op(e)
    return M


def pcg_iterations(K, P, b, x_exact, maxiter=5000):
    x = np.zeros_like(b)
    r = b.copy()
    z = P @ r
    p = z.copy()
    rz = r @ z
    e0 = np.sqrt((x_exact - x) @ K @ (x_exact - x))
    for k in range(1, maxiter + 1):
        Kp = K @ p
        pKp = p @ Kp
        if pKp <= 0:
            return None                                          # indefinite: CG breaks down
        alpha = rz / pKp
        x += alpha * p
        r -= alpha * Kp
        e = x_exact - x
        if np.sqrt(max(e @ K @ e, 0.0)) <= RTOL * e0:
            return k
        z = P @ r
        rz_new = r @ z
        p = z + (rz_new / rz) * p
        rz = rz_new
    return maxiter


def run_case(N, physics, geometry_name, k_v):
    grid = Grid(N, physics)
    A, Q, C_s, phase = material(grid, geometry_name, k_v)
    K_op, P_ops = preconditioners(grid, A, Q, C_s)
    n = grid.n
    Km = dense(K_op, n)
    Km = 0.5 * (Km + Km.T)
    # complement of the constants (rigid translations)
    consts = np.kron(np.eye(grid.du), np.ones((N * N, 1)))
    Z = sl.null_space(consts.T)
    # macroscopic strain e11 = 1 (scalar: gradient e1)
    E = np.zeros(grid.nc)
    E[0] = 1.0
    b = -grid.BtW(np.einsum('cdqxy,d->cqxy', A, E))
    Kz = Z.T @ Km @ Z
    x_exact = Z @ np.linalg.solve(Kz, Z.T @ b)
    out = {'N': N, 'physics': physics, 'geometry': geometry_name, 'k_v': k_v,
           'solid_fraction': float(phase.mean()), 'results': {}}
    for name, (op, cost) in P_ops.items():
        t = time.time()
        Pm = dense(op, n)
        Pz = Z.T @ (0.5 * (Pm + Pm.T)) @ Z
        p_eig = np.linalg.eigvalsh(Pz)
        res = {'cost_G': cost, 'P_min_eig': float(p_eig[0]), 'asym': float(np.abs(Pm - Pm.T).max() / np.abs(Pm).max())}
        if p_eig[0] > 0:
            L = np.linalg.cholesky(Pz)
            lam = np.linalg.eigvalsh(L.T @ Kz @ L)
            res.update(lam_min=float(lam[0]), lam_max=float(lam[-1]), kappa=float(lam[-1] / lam[0]),
                       eig_hist=np.histogram(np.log10(lam), bins=40, range=(-6, 6))[0].tolist())
            res['iterations'] = pcg_iterations(Km, 0.5 * (Pm + Pm.T), b, x_exact)
        else:
            res.update(lam_min=None, lam_max=None, kappa=None, iterations=None)
        res['time'] = time.time() - t
        out['results'][name] = res
        fmt = lambda v: '-' if v is None else f'{v:.3g}'
        print(f'  {name:16s} it {str(res["iterations"]):>5s}  lam [{fmt(res["lam_min"])}, {fmt(res["lam_max"])}]  '
              f'minEig(P) {p_eig[0]:.2e}', flush=True)
    return out


if __name__ == '__main__':
    (HERE / 'results').mkdir(exist_ok=True)
    for N in map(int, sys.argv[1:] or ['16']):
        cases = []
        for physics in ('scalar', 'elastic'):
            for geometry_name in ('bubble_array', 'square_inclusion'):
                for k_v in (1e-1, 1e-3, 1e-5):
                    print(f'N {N}  {physics}  {geometry_name}  k_v {k_v:g}', flush=True)
                    cases.append(run_case(N, physics, geometry_name, k_v))
                    json.dump(cases, open(HERE / 'results' / f'study_N{N}.json', 'w'), indent=1)
