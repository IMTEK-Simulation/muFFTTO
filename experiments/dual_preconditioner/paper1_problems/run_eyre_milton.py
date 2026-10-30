"""
Eyre-Milton against Green-PCG and dual-PCG on the conductivity problems P1-P3 of run_experiment.py (same grid,
material and discretization: linear triangles, reference A_0 = I for Green and dual).

Eyre-Milton (Eyre & Milton 1999), strain form with scalar reference c0 and Gamma0 = B G_I B^T W / c0:
    e^{k+1} = e^k + 2 c0 (A + c0 I)^-1 [ E - e^k - Gamma0 (A - c0 I) e^k ]
one Green application (FFT pair) per iteration; rate <= max |(g - c0)/(g + c0)|, g the eigenvalues of A.
References: c0 = 1 (the matrix, as Green) and c0 = sqrt(min g max g) (the optimal Eyre-Milton choice).

All methods are measured by the true relative energy error  ||e_k - e_h||_{W A} / ||e_0 - e_h||_{W A}  against the
reference solution u_h (dual-PCG until (r, z) < 1e-26); k_true = first iteration with error <= --tol.
For PCG also k_prod: the production stop (PT lower estimate, relative --tol, delay 10).
Cost in FFT pairs per iteration: Green 1, dual 2, Eyre-Milton 1.

    python run_eyre_milton.py --problem P1 --kappa-exp -3 [--grid 256]
writes exp_data_eyre_milton/<problem>_N<grid>_kappa1e<k>.npz
"""
import sys
import os
import argparse
import time
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

import numpy as np

from muFFTTO import domain
from muFFTTO import solvers

HERE = Path(__file__).resolve().parent

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument('--problem', choices=['P1', 'P2', 'P3'], required=True)
parser.add_argument('--kappa-exp', type=int, required=True, help='total phase contrast kappa = 10^kappa_exp')
parser.add_argument('--grid', type=int, default=256, help='grid N x N, a multiple of 256 for P2 and P3')
parser.add_argument('--tol', type=float, default=1e-6, help='relative energy-norm tolerance')
parser.add_argument('--d-min', type=int, default=10)
parser.add_argument('--maxiter', type=int, default=20000)
args = parser.parse_args()

kappa = 10.0 ** args.kappa_exp
N = args.grid

# ============================================================================
# DISCRETIZATION AND MATERIAL  (as run_experiment.py)
# ============================================================================
cell = domain.PeriodicUnitCell(domain_size=[1, 1], problem_type='conductivity')
discretization = domain.Discretization(cell=cell,
                                       nb_of_pixels_global=(N, N),
                                       discretization_type='finite_element',
                                       element_type='linear_triangles')
loc = discretization.subdomain_locations_no_buffers
nb = discretization.nb_of_pixels
local = (slice(loc[0], loc[0] + nb[0]), slice(loc[1], loc[1] + nb[1]))

I = np.eye(2)
A_global = np.zeros((2, 2, N, N))
if args.problem == 'P1':
    x_c = (np.arange(N) + 0.5) / N
    inside = np.abs(x_c - 0.5) < 0.25
    phase = 1 - np.outer(inside, inside)
    A_global = np.einsum('ij,xy->ijxy', I, phase + kappa * (1 - phase))
elif args.problem == 'P2':
    phase = np.load(HERE / 'data' / 'p2_circles_256.npz')['phase']
    phase = np.kron(phase, np.ones((N // 256, N // 256)))
    for value, A in [(0, I), (1, kappa * np.diag([1., 2.])), (2, kappa * np.diag([2., 1.]))]:
        A_global[:, :, phase == value] = A[:, :, None]
elif args.problem == 'P3':
    rho = np.load(HERE / 'data' / 'p3_density_256.npz')['density']
    rho = np.kron(rho, np.ones((N // 256, N // 256)))
    A_global = np.einsum('ij,xy->ijxy', I, rho + (1 - rho) * kappa)

eig = np.linalg.eigvalsh(A_global.transpose(2, 3, 0, 1))
eig_min, eig_max = eig.min(), eig.max()


def material_field(name, A_ijxy):
    field = discretization.get_material_data_size_field_mugrid(name=name)
    field.s[...] = A_ijxy[:, :, None][(..., *local)]
    return field


def pointwise(fun):
    """fun applied to every 2 x 2 matrix of A_global"""
    return np.moveaxis(fun(np.moveaxis(A_global, (0, 1), (-2, -1))), (-2, -1), (0, 1))


material = material_field('conductivity', A_global)
material_inv = material_field('conductivity_inverse', pointwise(np.linalg.inv))

macro_gradient_field = discretization.get_gradient_size_field(name='macro_gradient_field')
discretization.get_macro_gradient_field_mugrid(macro_gradient_ij=np.array([1.0, 0.0]),
                                               macro_gradient_field_ijqxyz=macro_gradient_field)
rhs = discretization.get_unknown_size_field(name='rhs')
discretization.get_rhs_mugrid(material_data_field_ijklqxyz=material,
                              macro_gradient_field_ijqxyz=macro_gradient_field,
                              rhs_inxyz=rhs)


def K_fun(x, Ax):
    discretization.apply_system_matrix_mugrid(material_data_field=material,
                                              input_field_inxyz=x, output_field_inxyz=Ax)
    discretization.fft.communicate_ghosts(Ax)


def K_inv_fun(x, Ax):
    discretization.apply_system_matrix_mugrid(material_data_field=material_inv,
                                              input_field_inxyz=x, output_field_inxyz=Ax)
    discretization.fft.communicate_ghosts(Ax)


green = discretization.get_preconditioner_Green_mugrid(reference_material_data_ijkl=I)


def G_fun(x, Px):
    discretization.fft.communicate_ghosts(x)
    discretization.apply_preconditioner_mugrid(preconditioner_Fourier_fnfnqks=green,
                                               input_nodal_field_fnxyz=x, output_nodal_field_fnxyz=Px)


tmp_1 = discretization.get_unknown_size_field(name='prec_tmp_1')
tmp_2 = discretization.get_unknown_size_field(name='prec_tmp_2')


def dual_fun(x, Px):
    """P = G K(A^-1) G"""
    G_fun(x, tmp_1)
    K_inv_fun(tmp_1, tmp_2)
    G_fun(tmp_2, Px)


def cg(x, P, tol, stop, cb=None, **kw):
    solvers.conjugate_gradients_mugrid_experimental(
        comm=discretization.communicator, fc=discretization.field_collection,
        hessp=K_fun, b=rhs, x=x, P=P, tol=tol, maxiter=args.maxiter, callback=cb,
        stop_crit_norm=stop, **kw)


# ============================================================================
# REFERENCE SOLUTION u_h and its gradient e_h = E + B u_h
# ============================================================================
u_h = discretization.get_unknown_size_field(name='u_h')
cg(u_h, dual_fun, 1e-13, 'rz')
e_h = discretization.get_gradient_size_field(name='e_h')
discretization.apply_gradient_operator_mugrid(u_inxyz=u_h, grad_u_ijqxyz=e_h)
e_h.s[...] += macro_gradient_field.s
A_h = discretization.get_homogenized_energy_mugrid(material_data_field_ijklqxyz=material,
                                                   displacement_field_inxyz=u_h,
                                                   macro_gradient_field_ijqxyz=macro_gradient_field)

# ============================================================================
# PCG: true error history (run past the tolerance) and production stop
# ============================================================================
err = discretization.get_unknown_size_field(name='err')
K_err = discretization.get_unknown_size_field(name='K_err')
results = {}
for name, P, nb_fft in [('green', G_fun, 1), ('dual', dual_fun, 2)]:
    hist = []

    def record(it, x, r, p, z, stop_crit):
        err.s[...] = u_h.s - x
        K_fun(err, K_err)
        hist.append(discretization.communicator.sum(np.dot(err.s.ravel(), K_err.s.ravel())))

    u = discretization.get_unknown_size_field(name=f'u_{name}')
    cg(u, P, args.tol * 1e-2, 'energy_lower_estim', cb=record, energy_rtol=True, d_min=args.d_min)
    rel = np.sqrt(np.abs(hist) / hist[0])
    u.s[...] = 0
    t0 = time.time()
    count = []
    cg(u, P, args.tol, 'energy_lower_estim', cb=lambda *a: count.append(0), energy_rtol=True, d_min=args.d_min)
    t = time.time() - t0
    results[name] = dict(error=rel, k_true=int(np.argmax(rel <= args.tol)) if np.any(rel <= args.tol) else -1,
                         k_prod=len(count) - 1, t_prod=t, nb_fft=nb_fft)

# ============================================================================
# EYRE-MILTON
# ============================================================================
e = discretization.get_gradient_size_field(name='em_e')
tau = discretization.get_gradient_size_field(name='em_tau')
diff = discretization.get_gradient_size_field(name='em_diff')
f = discretization.get_unknown_size_field(name='em_f')
v = discretization.get_unknown_size_field(name='em_v')
weighted = discretization.get_gradient_size_field(name='em_weighted')


def energy_norm_sq(g):
    weighted.s[...] = g.s
    discretization.apply_material_data_mugrid(material, weighted)
    discretization.apply_quadrature_weights_on_gradient_field_mugrid(grad_field=weighted)
    return discretization.communicator.sum(np.sum(g.s * weighted.s))


for name, c0 in [('em_c0_matrix', 1.0), ('em_c0_opt', np.sqrt(eig_min * eig_max))]:
    A_minus = material_field(f'{name}_A_minus', A_global - c0 * I[:, :, None, None])
    S = material_field(f'{name}_S', pointwise(lambda A: 2 * c0 * np.linalg.inv(A + c0 * I)))
    e.s[...] = macro_gradient_field.s
    diff.s[...] = e.s - e_h.s
    hist = [energy_norm_sq(diff)]
    t0 = time.time()
    for k in range(args.maxiter):
        tau.s[...] = e.s                                          # tau = (A - c0) e
        discretization.apply_material_data_mugrid(A_minus, tau)
        discretization.apply_gradient_transposed_operator_mugrid(gradient_field_ijqxyz=tau, div_u_fnxyz=f,
                                                                 apply_weights=True)
        G_fun(f, v)                                               # Gamma tau = B G B^T W tau
        discretization.apply_gradient_operator_mugrid(u_inxyz=v, grad_u_ijqxyz=tau)
        tau.s[...] = macro_gradient_field.s - e.s - tau.s / c0    # Lippmann-Schwinger residual
        discretization.apply_material_data_mugrid(S, tau)         # 2 c0 (A + c0)^-1 residual
        e.s[...] += tau.s
        diff.s[...] = e.s - e_h.s
        hist.append(energy_norm_sq(diff))
        if np.sqrt(hist[-1] / hist[0]) <= args.tol or not np.isfinite(hist[-1]):
            break
    t = time.time() - t0
    rel = np.sqrt(np.abs(hist) / hist[0])
    rate_bound = np.max(np.abs((eig - c0) / (eig + c0)))
    tail = rel[-11:] if len(rel) > 11 else rel
    results[name] = dict(error=rel, k_true=len(rel) - 1 if rel[-1] <= args.tol else -1, k_prod=-1, t_prod=t,
                         nb_fft=1, c0=c0, rate_bound=rate_bound,
                         rate_observed=(tail[-1] / tail[0]) ** (1 / (len(tail) - 1)))

# ============================================================================
# SAVE
# ============================================================================
out_dir = HERE / 'exp_data_eyre_milton'
out_dir.mkdir(exist_ok=True)
stem = f'{args.problem}_N{N}_kappa1e{args.kappa_exp:+d}'
np.savez(out_dir / f'{stem}.npz', problem=args.problem, N=N, kappa=kappa, tol=args.tol, A_h=A_h,
         eig_min=eig_min, eig_max=eig_max,
         results=np.array(results, dtype=object))
line = ', '.join(f'{m} {r["k_true"]}' + (f'/{r["k_prod"]}' if r['k_prod'] >= 0 else '') for m, r in results.items())
print(f'{stem}: k_true[/k_prod] {line}; EM rate obs/bound '
      + ', '.join(f'{results[m]["rate_observed"]:.4f}/{results[m]["rate_bound"]:.4f}'
                  for m in ['em_c0_matrix', 'em_c0_opt']), flush=True)
