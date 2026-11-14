"""The crack-orientation test (crack_orientation_test.py) with the classical SPECTRAL discretization of FFT-based solvers
(Moulinec-Suquet / Fourier-Galerkin with collocation): displacement u on the N^2 grid points, trigonometric polynomial
basis, gradient by the Fourier symbol i xi (xi = 2 pi k, Nyquist frequency set to zero), stiffness C evaluated at the
grid points. K = D^T C D (D^T the exact adjoint of D), Green P_G = K_ref^-1 (diagonal in Fourier space, 2 x 2 per
frequency), dual P_D = P_G D^T (C_ref C^-1 C_ref) D P_G, Green-Jacobi: C_ref = 2 mu I with the Jacobi scaling of K.
FFTs through muFFTTO's engine (disc.fft; muFFTTO's own 'Fourier' discretization is only a stub).
Same geometries, loads, contrast 1e4, l = 4h, CG energy-error stop 1e-6 with delay DMIN as the FE test.
    python spectral_orientation_test.py [N=128]  ->  spectral_orientation_results_N<N>_dmin<d>.csv
env: DMIN (10), PRECS (Green,Green_Jacobi,dual), MAXIT (2000), GEOMS (comma list, default all)
Green-Jacobi uses the exact diagonal of the spectral K: a unit displacement at one grid point gives the strain
e(r) = sym(d(r) x e_a) everywhere (d = kernel of the spectral derivative), so diag(K)_a(i) = sum_j s_j e:C0:e (j - i),
a periodic convolution of s with w_a = e:C0:e, done by FFT.
Geometries smooth (rho = 1/2 + 1/2 cos 2pi x cos 2pi y, s = rho^2 (1-k) + k) and wide_disc (soft disc, rim l = 16h):
smooth fields where Green-Jacobi wins in the FE studies.
"""
import csv
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

import numpy as np

from muFFTTO import domain, material_models, solvers

HERE = os.path.dirname(os.path.realpath(__file__))
N = int(sys.argv[1]) if len(sys.argv) > 1 else 128
K_RES, L = 1e-4, 4.0 / N
DMIN = int(os.environ.get('DMIN', 10))
PRECS = os.environ.get('PRECS', 'Green,Green_Jacobi,dual').split(',')
MAXIT = int(os.environ.get('MAXIT', 2000))
dim = 2
C_0 = material_models.get_elastic_material_tensor(dim=dim, K=1.0, mu=0.5, kind='linear')
C_2muI = 2 * 0.5 * np.einsum('ik,jl->ijkl', np.eye(dim), np.eye(dim))
LOADS = {'E_xx (across)': np.array([[1.0, 0.0], [0.0, 0.0]]), 'E_yy (along)': np.array([[0.0, 0.0], [0.0, 1.0]]),
         'E_xy (shear)': np.array([[0.0, 0.5], [0.5, 0.0]])}

# the FE discretization is used only for its grid, field collections and FFT engine (one node per pixel)
cell = domain.PeriodicUnitCell(domain_size=[1, 1], problem_type='elasticity')
disc = domain.Discretization(cell=cell, nb_of_pixels_global=(N, N), discretization_type='finite_element',
                             element_type='bilinear_rectangle')
fft = disc.fft
u_hat = disc.ffield_collection.complex_field(name='u_hat', components=(dim,), sub_pt='nodal_points')
w_hat = disc.ffield_collection.complex_field(name='w_hat', components=(dim,), sub_pt='nodal_points')
tmp = disc.get_unknown_size_field(name='tmp')
rhs = disc.get_unknown_size_field(name='rhs')
u = disc.get_unknown_size_field(name='u')
t1 = disc.get_unknown_size_field(name='t1')
t2 = disc.get_unknown_size_field(name='t2')
norm = fft.normalisation

# wave vectors xi = 2 pi k on the Fourier grid; Nyquist zeroed
freq = np.asarray(fft.fftfreq)                                   # normalised frequencies k / N, [dim, ...]
xi = 2 * np.pi * freq * N
xi[:, np.isclose(np.abs(freq).max(axis=0), 0.5)] = 0.0
nyq = np.isclose(np.abs(freq), 0.5).any(axis=0)
xi[:, nyq] = 0.0
fshape = u_hat.s.shape[2:]                                       # Fourier grid shape
xi = xi.reshape((dim,) + fshape)


def ref_inverse(Cref):
    """per-frequency inverse of K_ref(xi)_ac = Cref_abcd xi_b xi_d (zero at xi = 0 and Nyquist)"""
    A = np.einsum('abcd,b...,d...->ac...', Cref, xi, xi)
    A = np.moveaxis(A, (0, 1), (-2, -1))
    inv = np.zeros_like(A)
    ok = np.abs(np.linalg.det(A)) > 1e-12
    inv[ok] = np.linalg.inv(A[ok])
    return np.moveaxis(inv, (-2, -1), (0, 1))


Ginv = {'C0': ref_inverse(C_0), '2muI': ref_inverse(C_2muI)}


def grad(x):
    """strain at the grid points, sym(grad u), [dim, dim, N, N]"""
    fft.fft(x, u_hat)
    uh = u_hat.s[:, 0]
    e = np.empty((dim, dim) + uh.shape[1:], dtype=complex)
    for a in range(dim):
        for b in range(dim):
            e[a, b] = 0.5j * (xi[a] * uh[b] + xi[b] * uh[a])
    out = np.empty((dim, dim, N, N))
    for a in range(dim):
        for b in range(dim):
            w_hat.s[0, 0] = e[a, b]
            fft.ifft(w_hat, tmp)
            out[a, b] = tmp.s[0, 0] * norm
    return out


def div_T(sig, out):
    """D^T sigma (exact adjoint of grad): in Fourier space -i xi_b sigma_hat_ab ... with the same scaling"""
    acc = np.zeros((dim,) + fshape, dtype=complex)
    for a in range(dim):
        for b in range(dim):
            tmp.s[0, 0] = sig[a, b]
            fft.fft(tmp, w_hat)
            acc[a] += -1j * xi[b] * w_hat.s[0, 0] * 0.5
            acc[b] += -1j * xi[a] * w_hat.s[0, 0] * 0.5
    u_hat.s[:, 0] = acc
    fft.ifft(u_hat, out)
    out.s[...] *= norm


def apply_C(scale, Cten, e):
    return scale * np.einsum('abcd,cd...->ab...', Cten, e)


def K_fun(x, Ax):
    div_T(apply_C(state['s'], C_0, grad(x)), Ax)


def ref_solve(x, Px, key):
    fft.fft(x, u_hat)
    u_hat.s[:, 0] = np.einsum('ac...,c...->a...', Ginv[key], u_hat.s[:, 0])
    fft.ifft(u_hat, Px)
    Px.s[...] *= norm * norm                                     # K_ref = D^T Cref D carries norm^2 per FFT pair


def green(x, Px):
    ref_solve(x, Px, 'C0')


def dual(x, Px):
    ref_solve(x, t1, 'C0')
    div_T(apply_C(1.0 / state['s'], C_0, grad(t1)), t2)          # C_ref C^-1 C_ref = C_0 / s for C = s C_0
    ref_solve(t2, Px, 'C0')


def green_jacobi(x, Px):
    t1.s[...] = state['J'] * x.s
    ref_solve(t1, Px, '2muI')
    Px.s[...] *= state['J']


def jacobi_diag(s):
    """exact diag(K) of the spectral operator, [dim, N, N]"""
    dker = []                                                    # kernel of d/dx_b at the grid points
    for b in range(dim):
        w_hat.s[0, 0] = 1j * xi[b]
        fft.ifft(w_hat, tmp)
        dker.append(tmp.s[0, 0] * norm)
    out = []
    for a in range(dim):
        e = np.zeros((dim, dim, N, N))
        for b in range(dim):
            e[a, b] += 0.5 * dker[b]
            e[b, a] += 0.5 * dker[b]
        w = np.einsum('ab...,abcd,cd...->...', e, C_0, e)       # e:C0:e(r), r = j - i
        # diag(i) = sum_j s_j w(j - i): correlation = ifft(fft(s) * conj(fft(w)))
        tmp.s[0, 0] = s
        fft.fft(tmp, w_hat)
        sh = w_hat.s[0, 0].copy()
        tmp.s[0, 0] = w
        fft.fft(tmp, w_hat)
        w_hat.s[0, 0] = sh * np.conj(w_hat.s[0, 0])
        fft.ifft(w_hat, tmp)
        out.append(tmp.s[0, 0] * norm)
    return np.array(out)


def periodic_dist(a, b):
    d = np.abs(a - b)
    return np.minimum(d, 1 - d)


X, Y = np.meshgrid(np.arange(N) / N, np.arange(N) / N, indexing='ij')
dx, dy = periodic_dist(X, 0.5), periodic_dist(Y, 0.5)
seg = np.where(np.abs(Y - 0.5) <= 0.25, dx, np.hypot(dx, np.abs(Y - 0.5) - 0.25))
GEOM = {'homog': None, 'band': dx, 'band2ph': dx, 'finite': seg, 'cross': np.minimum(dx, dy),
        'disc': np.maximum(np.hypot(dx, dy) - 0.25, 0.0), 'smooth': 'smooth', 'wide_disc': 'wide_disc'}
if os.environ.get('GEOMS'):
    GEOM = {k: v for k, v in GEOM.items() if k in os.environ['GEOMS'].split(',')}
state = {}
rows = []
for g, dist in GEOM.items():
    if dist is None:
        s = np.ones((N, N))
    elif isinstance(dist, str) and dist == 'smooth':
        rho = 0.5 + 0.5 * np.cos(2 * np.pi * X) * np.cos(2 * np.pi * Y)
        s = rho ** 2 * (1 - K_RES) + K_RES
    elif isinstance(dist, str) and dist == 'wide_disc':
        d = np.exp(-np.maximum(np.hypot(dx, dy) - 0.25, 0.0) / (4 * L))
        s = (1 - d) ** 2 * (1 - K_RES) + K_RES
    else:
        d = np.exp(-dist / L)
        s = np.where(d > 0.5, K_RES, 1.0) if g == 'band2ph' else (1 - d) ** 2 * (1 - K_RES) + K_RES
    state['s'] = s
    # Jacobi scaling: diag(K)^-1/2 per component, diag computed by probing one unit vector per residue class is
    # costly; use the local estimate diag(K)_a ~ s(x) * sum_b C0_abab xi^2-average = s(x) * const (constant
    # cancels in the scaling), i.e. J = s^-1/2
    state['J'] = (1.0 / np.sqrt(jacobi_diag(s)))[:, None]
    for lname, E in LOADS.items():
        sigE = apply_C(s, C_0, np.broadcast_to(E[:, :, None, None], (dim, dim, N, N)))
        div_T(sigE, rhs)
        rhs.s[...] *= -1.0
        if g == 'homog':                                         # zero rhs for a homogeneous cell: use a random one
            u.s[...] = np.random.default_rng(0).standard_normal(u.s.shape)
            K_fun(u, rhs)
        res = {}
        for P in PRECS:
            u.s.fill(0)
            norms = solvers.conjugate_gradients_mugrid_experimental(
                comm=disc.communicator, fc=disc.field_collection, hessp=K_fun, b=rhs, x=u,
                P={'Green': green, 'dual': dual, 'Green_Jacobi': green_jacobi}[P], tol=1e-6, maxiter=MAXIT,
                stop_crit_norm='energy_lower_estim', energy_rtol=True, d_min=DMIN)[1]
            it = norms.get('stop_iteration', {}).get('energy_lower_estim')
            res[P] = int(it) if it is not None else -MAXIT
        rows.append([g, lname] + [res[p] for p in PRECS])
        print(f'{g:8s} {lname:15s} ' + '  '.join(f'{p} {res[p]:5d}' for p in PRECS), flush=True)
        if g == 'homog':
            break
with open(os.path.join(HERE, f'spectral_orientation_results_N{N}_dmin{DMIN}.csv'), 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['geometry', 'load'] + [f'cg_{p}' for p in PRECS])
    w.writerows(rows)
