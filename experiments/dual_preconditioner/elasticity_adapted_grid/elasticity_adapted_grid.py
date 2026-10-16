"""
Small-strain elasticity homogenization of the bubble array of the slide problems on the uniform grid and on the grid
adapted to the bubbles (Zecevic), with the preconditioners Green, Green-Jacobi and dual. Formulation of Bignonnet,
Ladecky, Pultarova, Zeman, C. R. Mecanique 354 (2026) 227-256, Section 2.3: the pulled-back problem on the uniform
computational grid,
    K~ = B~^T W~ C~ B~,   f~_E = -B~^T W~ C~ E~,
    C~_ijkl = J F^-1_jp C_ipkq F^-1_lq   (major symmetry only),   E~ = E . F,   F = I + grad~ u~_Phi,  J = det F,
applied with the full displacement gradient. The effective stiffness is the mean physical stress
C : (E + sym(grad~ u . F^-1)) over the physical cell (weights w_q J_q).

Material: solid (1) C_0 isotropic (K = 1, mu = 1/2), bubbles (0) kappa C_0. Preconditioners:
    Green          G_0 = K(C_0)^-1 of the uniform grid
    Green_Jacobi   D^-1/2 G_2muI D^-1/2, D = diag(K~), reference 2 mu I
    dual           G_0 K(C_0 C~^+ C_0) G_0, C~^+ the pseudo-inverse of the 4 x 4 pulled-back tangent (C~ is singular on
                   W . F, W skew); on the uniform grid C~ = C (minor symmetric)
Q1 elements, as the slide runs. CG from zero, energy-error estimate relative to the initial error (1e-6, delay 10).
--spectrum (small grids): eigenvalues of P K~ on the mean-free displacements.

    python elasticity_adapted_grid.py -n 32 -k -3 -p dual --grid adapted [--spectrum]
writes data/bubble_array_<grid>_N<n>_kappa1e<k>_<preconditioner>.npz
"""
import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from muFFTTO import domain
from muFFTTO import material_models
from muFFTTO import microstructure_library
from muFFTTO import solvers

HERE = os.path.dirname(os.path.realpath(__file__))
parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument('-n', '--nb_pixel', type=int, default=32)
parser.add_argument('-k', '--kappa_exp', type=int, required=True, help='bubble stiffness kappa C_0, kappa = 10^k')
parser.add_argument('-p', '--preconditioner', choices=['Green', 'Green_Jacobi', 'dual'], required=True)
parser.add_argument('--grid', choices=['regular', 'adapted'], default='regular')
parser.add_argument('--cg_tol', type=float, default=1e-6)
parser.add_argument('--cg_delay', type=int, default=10)
parser.add_argument('--spectrum', action='store_true', help='dense spectrum of P K (serial, small grids)')
parser.add_argument('--save_solution', action='store_true',
                    help='store u of the load case E_11, the grid displacement and the phase in solutions/ (serial)')
args = parser.parse_args()
nnn, kappa, preconditioner_type = args.nb_pixel, 10.0 ** args.kappa_exp, args.preconditioner
dim = 2
formulation = 'finite_strain'                     # full displacement gradient: C~ has no minor symmetry

disc = domain.Discretization(cell=domain.PeriodicUnitCell(domain_size=[1, 1], problem_type='elasticity'),
                             nb_of_pixels_global=(nnn, nnn), discretization_type='finite_element',
                             element_type='bilinear_rectangle')
comm = disc.communicator
coords = np.asarray(disc.fft.coords)
local_index = tuple(np.rint(coords[k] * nnn).astype(int) % nnn for k in range(dim))

# ============================================================================
# grid map and phase (copies of the grid files of the slide problems in grids/)
# ============================================================================
if args.grid == 'adapted':
    grid = np.load(os.path.join(HERE, 'grids', f'grid_bubble_array_N{nnn}_zecevic_min_jacobian0.01.npz'))
    u_phi = grid['d_g'][(slice(None),) + local_index]
    phase_pixel = grid['phase'][local_index]
else:
    u_phi = np.zeros((dim,) + coords.shape[1:])
    phase_pixel = microstructure_library.get_geometry(nb_voxels=disc.nb_of_pixels, microstructure_name='bubble_array',
                                                      coordinates=disc.fft.coords)
solid = phase_pixel > 0
u_phi_field = disc.get_unknown_size_field(name='u_phi')
u_phi_field.s[:, 0] = u_phi
disc.fft.communicate_ghosts(u_phi_field)
grad_phi = disc.get_gradient_size_field(name='grad_phi')
disc.apply_gradient_operator_mugrid(u_inxyz=u_phi_field, grad_u_ijqxyz=grad_phi)
F = grad_phi.s + np.eye(dim).reshape(dim, dim, 1, 1, 1)                      # [i, j, q, x, y]
J = F[0, 0] * F[1, 1] - F[0, 1] * F[1, 0]
if not comm.min(float(J.min())) > 0:
    raise ValueError('grid map not invertible')
F_inv = np.moveaxis(np.linalg.inv(np.moveaxis(F, (0, 1), (-2, -1))), (-2, -1), (0, 1))

C_0 = material_models.get_elastic_material_tensor(dim=dim, K=1.0, mu=0.5, kind='linear')
mu_0 = 0.5
C = C_0[..., None, None, None] * np.where(solid[None], 1.0, kappa)          # physical stiffness [i,j,k,l,q,x,y]
C_tilde = J * np.einsum('jp...,ipkq...,lq...->ijkl...', F_inv, C, F_inv)      # pulled back (eq. 22)

material = disc.get_material_data_size_field_mugrid(name='C_tilde')
material.s[...] = C_tilde
n4 = dim * dim
Ct = np.moveaxis(C_tilde.reshape((n4, n4) + C_tilde.shape[4:]), (0, 1), (-2, -1))   # [..., (ij), (kl)]
w, V = np.linalg.eigh(0.5 * (Ct + np.swapaxes(Ct, -1, -2)))
keep = np.abs(w) > 1e-10 * np.abs(w).max(axis=-1, keepdims=True)              # drop the kernel W . F
w_inv = np.where(keep, 1.0 / np.where(keep, w, 1.0), 0.0)
C0m = C_0.reshape(n4, n4)
middle = C0m @ np.einsum('...ab,...b,...cb->...ac', V, w_inv, V) @ C0m
material_dual = disc.get_material_data_size_field_mugrid(name='C_dual')
material_dual.s[...] = np.moveaxis(middle, (-2, -1), (0, 1)).reshape(C_tilde.shape)
# extremes of the generalized eigenvalues of C~ w.r.t. C_0 on the symmetric part (Mandel), for information
SYM = np.array([[1, 0, 0, 0], [0, 0, 0, 1], [0, 1 / np.sqrt(2), 1 / np.sqrt(2), 0]])
L_inv = np.linalg.inv(np.linalg.cholesky(SYM @ C0m @ SYM.T))
g = np.linalg.eigvalsh(L_inv @ (SYM @ (0.5 * (Ct + np.swapaxes(Ct, -1, -2))) @ SYM.T) @ L_inv.T)
g_min, g_max = comm.min(float(g.min())), comm.max(float(g.max()))


def K_fun(x, Ax):
    disc.apply_system_matrix_mugrid(material_data_field=material, input_field_inxyz=x, output_field_inxyz=Ax,
                                    formulation=formulation)
    disc.fft.communicate_ghosts(Ax)


green_0 = disc.get_preconditioner_Green_mugrid(reference_material_data_ijkl=C_0, name='green_C0')
green_2muI = disc.get_preconditioner_Green_mugrid(
    reference_material_data_ijkl=2 * mu_0 * np.einsum('ik,jl->ijkl', np.eye(dim), np.eye(dim)), name='green_2muI')
jacobi = disc.get_preconditioner_Jacobi_mugrid(material_data_field_ijklqxyz=material, formulation=formulation,
                                               name='jacobi_C_tilde')
temp_1 = disc.get_unknown_size_field(name='temp_1')
temp_2 = disc.get_unknown_size_field(name='temp_2')


def apply_green(G, x, Px):
    disc.fft.communicate_ghosts(x)
    disc.apply_preconditioner_mugrid(preconditioner_Fourier_fnfnqks=G, input_nodal_field_fnxyz=x,
                                     output_nodal_field_fnxyz=Px)


def M_Green(x, Px):
    apply_green(green_0, x, Px)


def M_Green_Jacobi(x, Px):
    temp_1.s[...] = jacobi.s * x.s
    apply_green(green_2muI, temp_1, Px)
    Px.s[...] *= jacobi.s


def M_dual(x, Px):
    apply_green(green_0, x, temp_1)
    disc.apply_system_matrix_mugrid(material_data_field=material_dual, input_field_inxyz=temp_1,
                                    output_field_inxyz=temp_2, formulation=formulation)
    disc.fft.communicate_ghosts(temp_2)
    apply_green(green_0, temp_2, Px)


PRECS = {'Green': M_Green, 'Green_Jacobi': M_Green_Jacobi, 'dual': M_dual}

# ============================================================================
# three load cases E_11, E_22, E_12
# ============================================================================
stress = disc.get_gradient_size_field(name='stress')
grad_u = disc.get_gradient_size_field(name='grad_u')
rhs = disc.get_unknown_size_field(name='rhs')
u = disc.get_unknown_size_field(name='u')
u_ref = disc.get_unknown_size_field(name='u_ref')
err = disc.get_unknown_size_field(name='err')
K_err = disc.get_unknown_size_field(name='K_err')
weights = np.asarray(disc.quadrature_weights).reshape((-1, 1, 1))


def energy(x):
    K_fun(x, K_err)
    return comm.sum(float(np.dot(x.s.ravel(), K_err.s.ravel())))


iterations, rel_errors, C_eff = [], [], np.zeros((dim, dim, dim, dim))
vol = comm.sum(float((weights * J).sum()))
t0 = time.time()
for i, j in [(0, 0), (1, 1), (0, 1)]:
    E = np.zeros((dim, dim))
    E[i, j] = E[j, i] = 1.0 if i == j else 0.5
    E_tilde = np.einsum('ik,kj...->ij...', E, F)                              # E . F
    stress.s[...] = np.einsum('ijkl...,kl...->ij...', C_tilde, E_tilde)
    disc.fft.communicate_ghosts(stress)
    disc.apply_gradient_transposed_operator_mugrid(gradient_field_ijqxyz=stress, div_u_fnxyz=rhs, apply_weights=True)
    rhs.s[...] *= -1
    u_ref.s.fill(0)
    solvers.conjugate_gradients_mugrid_experimental(comm=comm, fc=disc.field_collection, hessp=K_fun, b=rhs, x=u_ref,
                                                    P=M_Green_Jacobi, tol=1e-13, maxiter=50000, stop_crit_norm='rz')
    u.s.fill(0)
    norms = solvers.conjugate_gradients_mugrid_experimental(
        comm=comm, fc=disc.field_collection, hessp=K_fun, b=rhs, x=u, P=PRECS[preconditioner_type], tol=args.cg_tol,
        maxiter=50000, stop_crit_norm='energy_lower_estim', energy_rtol=True, d_min=args.cg_delay)[1]
    iterations.append(int(norms['stop_iteration']['energy_lower_estim']))
    err.s[...] = u_ref.s - u.s
    rel_errors.append(np.sqrt(abs(energy(err)) / energy(u_ref)))
    disc.fft.communicate_ghosts(u)
    disc.apply_gradient_operator_mugrid(u_inxyz=u, grad_u_ijqxyz=grad_u)
    strain = E.reshape(dim, dim, 1, 1, 1) + np.einsum('ik...,kj...->ij...', grad_u.s, F_inv)
    strain = 0.5 * (strain + np.swapaxes(strain, 0, 1))
    sigma = np.einsum('ijkl...,kl...->ij...', C, strain)
    mean_sigma = np.array([[comm.sum(float((weights * J * sigma[a, b]).sum())) for b in range(dim)]
                           for a in range(dim)]) / vol
    C_eff[:, :, i, j] = C_eff[:, :, j, i] = mean_sigma
    if args.save_solution and comm.size == 1 and (i, j) == (0, 0):
        os.makedirs(os.path.join(HERE, 'solutions'), exist_ok=True)
        np.savez(os.path.join(HERE, 'solutions', f'bubble_array_{args.grid}_N{nnn}_kappa1e{args.kappa_exp:+d}.npz'),
                 u=np.array(u.s[:, 0]), u_phi=u_phi, phase=phase_pixel, E=E)
elapsed = time.time() - t0

spectrum = None
if args.spectrum:
    import scipy.linalg as sl
    n = rhs.s.size
    e, o = disc.get_unknown_size_field(name='spec_e'), disc.get_unknown_size_field(name='spec_o')

    def dense(op):
        out = np.zeros((n, n))
        for col in range(n):
            e.s[...] = 0.0
            e.s.flat[col] = 1.0
            disc.fft.communicate_ghosts(e)
            op(e, o)
            out[:, col] = o.s.ravel()
        return out

    T = np.zeros((n, dim))
    for c in range(dim):
        t = np.zeros(rhs.s.shape)
        t[c] = 1.0
        T[:, c] = t.ravel()
    Q = sl.null_space(T.T)
    Kd = Q.T @ dense(K_fun) @ Q
    spectrum = {p: np.sort(np.real(sl.eigvals(Q.T @ dense(PRECS[p]) @ Q @ Kd))) for p in PRECS}

if comm.rank == 0:
    os.makedirs(os.path.join(HERE, 'data'), exist_ok=True)
    out = os.path.join(HERE, 'data', f'bubble_array_{args.grid}_N{nnn}_kappa1e{args.kappa_exp:+d}_'
                                     f'{preconditioner_type}.npz')
    voigt = material_models.compute_Voigt_notation_4order(C_eff)
    np.savez(out, grid=args.grid, N=nnn, kappa=kappa, preconditioner=preconditioner_type, iterations=iterations,
             rel_errors=rel_errors, C_eff=C_eff, C_eff_voigt=voigt, g_min=g_min, g_max=g_max,
             min_J=comm.min(float(J.min())), max_J=comm.max(float(J.max())), elapsed=elapsed,
             **({f'spectrum_{p}': ev for p, ev in spectrum.items()} if spectrum else {}))
    print(f'{args.grid:8s} N{nnn} kappa 1e{args.kappa_exp:+d} {preconditioner_type:12s}: CG {iterations} '
          f'(rel. energy error {max(rel_errors):.1e}), C_eff 11/12/33 {voigt[0, 0]:.5f} {voigt[0, 1]:.5f} '
          f'{voigt[2, 2]:.5f}, g in [{g_min:.3g}, {g_max:.3g}], J in [{comm.min(float(J.min())):.3f}, '
          f'{comm.max(float(J.max())):.3f}], {elapsed:.1f} s')
    if spectrum:
        for p, ev in spectrum.items():
            print(f'  spectrum {p:12s} [{ev[0]:.4g}, {ev[-1]:.4g}]  cond {ev[-1] / ev[0]:.3g}  '
                  f'#eig < 1: {np.sum(ev < 1 - 1e-8)}')
