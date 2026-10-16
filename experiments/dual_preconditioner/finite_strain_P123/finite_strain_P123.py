"""
Finite-strain homogenization (neo-Hookean, Newton-CG) on the geometries P1-P3 of the paper-1 problems, with the
preconditioners Green, Green-Jacobi and dual. Adapted from a copy of
examples/homogenization/finite_strain_elasticity/example_2D_homogenization_finite_strain_bubbles_zero_stiffness.py
(Newton loop, energy stop) and experiments/dual_preconditioner/paper1_problems/run_experiment_elasticity.py
(geometries).

Materials: neo-Hookean, matrix E = 1, nu = 0.3 (lam_0, mu_0); finite contrast kappa:
    P1  square inclusion [0.25, 0.75)^2:  kappa (lam_0, mu_0)
    P2  58 circles, two isotropic phases:  kappa (lam_0, mu_0) and 2 kappa (lam_0, mu_0)
    P3  topology-optimization density:     (rho + (1 - rho) kappa) (lam_0, mu_0)
Preconditioners (full displacement gradient, the finite-strain formulation):
    Green          G_0 = K(C_0)^-1, C_0 = lam_0 I(x)I + 2 mu_0 I_s, the matrix tangent at F = I
    Green_Jacobi   D^-1/2 G_2muI D^-1/2, D = diag(K) of the current tangent, reference 2 mu_0 I
    dual           G_0 K(C_0 A^+ C_0) G_0, A = the current tangent dP/dF (4 x 4 per quadrature point), rebuilt every
                   Newton step; --dual_variant full: A^+ the pseudo-inverse through |eigenvalues| (relative cutoff
                   --dual_cutoff, default 1e-6); sym: the inverse of the block of A on the symmetric gradients
Every Newton step solves K du = -B^T W P from du = 0; CG stops on the energy-error estimate relative to the initial
error (1e-6, delay 10). Full Newton steps; Newton stops on |grad du| / |F| < 1e-8.

    python finite_strain_P123.py -g P1 -k -3 -n 32 -p dual [-inc 10] [--load H00 H01 H10 H11]
writes data/<geometry>_N<n>_kappa1e<k>_<preconditioner>.npz
"""
import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from muFFTTO import domain
from muFFTTO import material_models
from muFFTTO import solvers

HERE = os.path.dirname(os.path.realpath(__file__))
parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument('-g', '--geometry', choices=['P1', 'P2', 'P3'], required=True)
parser.add_argument('-k', '--kappa_exp', type=int, required=True, help='phase contrast kappa = 10^kappa_exp')
parser.add_argument('-n', '--nb_pixel', type=int, default=32)
parser.add_argument('-p', '--preconditioner', choices=['Green', 'Green_Jacobi', 'dual'], required=True)
parser.add_argument('-inc', '--nb_increments', type=int, default=10)
parser.add_argument('--cg_tol', type=float, default=1e-6)
parser.add_argument('--cg_delay', type=int, default=10)
parser.add_argument('--load', type=float, nargs=4, default=[0.0, 0.2, 0.0, 0.2], metavar=('H00', 'H01', 'H10', 'H11'),
                    help='total macroscopic displacement gradient F - I (default: shear 0.2 + stretch 0.2)')
parser.add_argument('--dual_variant', choices=['full', 'sym', 'inv'], default='full',
                    help="dual middle material: 'full' C_0 A^+ C_0 with the pseudo-inverse of the 4 x 4 tangent, "
                         "'sym' C_0 (A_ss)^-1 C_0 with the inverse of its block on the symmetric gradients (Mandel), "
                         "'inv' C_0 A^-1 C_0 with the plain inverse of the 4 x 4 tangent")
parser.add_argument('--dual_cutoff', type=float, default=1e-6,
                    help="'full': relative cutoff of the pseudo-inverse (eigenvalues below cutoff x max are dropped)")
parser.add_argument('--save_increments', type=int, nargs='*', default=[],
                    help='store the displacement fluctuation after these increments in solutions/ (serial)')
parser.add_argument('--spectrum_increments', type=int, nargs='*', default=[],
                    help='after these increments: dense spectra of P K for all preconditioners (small grids only)')
args = parser.parse_args()
nnn, ninc, kappa = args.nb_pixel, args.nb_increments, 10.0 ** args.kappa_exp
preconditioner_type = args.preconditioner

number_of_pixels = (nnn, nnn)
dim = 2
formulation = 'finite_strain'
macro_gradient_total = np.array(args.load).reshape(dim, dim)
newton_tol, newton_maxiter, cg_maxiter = 1e-8, 100, 20000

discretization = domain.Discretization(
    cell=domain.PeriodicUnitCell(domain_size=[1, 1], problem_type='elasticity'),
    nb_of_pixels_global=number_of_pixels, discretization_type='finite_element', element_type='linear_triangles')
comm = discretization.communicator
loc = discretization.subdomain_locations_no_buffers
nb = discretization.nb_of_pixels
local = (slice(loc[0], loc[0] + nb[0]), slice(loc[1], loc[1] + nb[1]))

# ============================================================================
# geometry -> stiffness scaling s(x) of the matrix moduli
# ============================================================================
if args.geometry == 'P1':
    x_c = (np.arange(nnn) + 0.5) / nnn
    inside = np.abs(x_c - 0.5) < 0.25
    scale = np.where(np.outer(inside, inside), kappa, 1.0)
elif args.geometry == 'P2':
    phase = np.load(os.path.join(HERE, 'data', 'p2_circles_256.npz'))['phase']
    f = 256 // nnn if nnn < 256 else 1
    phase = phase[f // 2::f, f // 2::f] if nnn < 256 else np.kron(phase, np.ones((nnn // 256, nnn // 256)))
    scale = np.choose(phase.astype(int), [1.0, kappa, 2 * kappa])
else:
    rho = np.load(os.path.join(HERE, 'data', 'p3_density_256.npz'))['density']
    f = 256 // nnn if nnn < 256 else 1
    rho = rho.reshape(nnn, f, nnn, f).mean(axis=(1, 3)) if nnn < 256 else np.kron(rho, np.ones((nnn // 256,) * 2))
    scale = rho + (1 - rho) * kappa

E_0, nu_0 = 1.0, 0.3
lam_0 = E_0 * nu_0 / ((1 + nu_0) * (1 - 2 * nu_0))
mu_0 = E_0 / (2 * (1 + nu_0))
lam_field = discretization.get_quad_field_scalar(name='lam_field')
mu_field = discretization.get_quad_field_scalar(name='mu_field')
lam_field.s[0, 0] = lam_0 * scale[local][None]
mu_field.s[0, 0] = mu_0 * scale[local][None]
material = material_models.NeoHookean(discretization=discretization, lam_1qxyz=lam_field, mu_1qxyz=mu_field,
                                      name='neo_hookean')

i2 = np.eye(dim)
I4rt = np.einsum('ik,jl->ijkl', i2, i2)
I4s = 0.5 * (I4rt + np.einsum('il,jk->ijkl', i2, i2))
C_0 = lam_0 * np.einsum('ij,kl->ijkl', i2, i2) + 2.0 * mu_0 * I4s      # matrix tangent at F = I

# ============================================================================
# fields
# ============================================================================
macro_gradient_inc_field = discretization.get_gradient_size_field(name='macro_gradient_inc_field')
displacement_fluctuation_field = discretization.get_unknown_size_field(name='displacement_fluctuation_field')
displacement_increment_field = discretization.get_unknown_size_field(name='displacement_increment_field')
strain_fluc_field = discretization.get_displacement_gradient_sized_field(name='strain_fluctuation_field')
total_strain_field = discretization.get_displacement_gradient_sized_field(name='total_strain_field')
stress_field = discretization.get_displacement_gradient_sized_field(name='stress_field')
tangent_field = discretization.get_material_data_size_field_mugrid(name='tangent_field')
dual_field = discretization.get_material_data_size_field_mugrid(name='dual_field')
rhs_field = discretization.get_unknown_size_field(name='rhs_field')


def evaluate_material():
    material.get_stress(total_strain_field, stress_field)
    material.get_algorithmic_tangent(total_strain_field, tangent_field)


def compute_residual():
    discretization.fft.communicate_ghosts(stress_field)
    discretization.apply_gradient_transposed_operator_mugrid(gradient_field_ijqxyz=stress_field,
                                                             div_u_fnxyz=rhs_field, apply_weights=True)
    rhs_field.s[...] *= -1
    return np.sqrt(comm.sum(np.dot(rhs_field.s.ravel(), rhs_field.s.ravel())))


def mean_stress():
    local_sum = stress_field.s.sum(axis=tuple(range(2, stress_field.s.ndim)))
    return np.array([[comm.sum(float(local_sum[a, b])) for b in range(dim)] for a in range(dim)]) / (
        np.prod(number_of_pixels) * stress_field.s.shape[2])


total_strain_field.s[...] = 0.0
for d in range(dim):
    total_strain_field.s[d, d] = 1.0

# ============================================================================
# preconditioners (each Green operator with its own field name)
# ============================================================================
green_0 = discretization.get_preconditioner_Green_mugrid(reference_material_data_ijkl=C_0, name='green_C0')
green_2muI = discretization.get_preconditioner_Green_mugrid(reference_material_data_ijkl=2 * mu_0 * I4rt,
                                                            name='green_2muI')
jacobi_field = None
temp_1 = discretization.get_unknown_size_field(name='prec_temp_1')
temp_2 = discretization.get_unknown_size_field(name='prec_temp_2')
C0_mat = C_0.reshape(dim * dim, dim * dim)


def update_preconditioner():
    """Green-Jacobi: D^-1/2 of the current tangent; dual: the material C_0 A^+ C_0 of the current tangent."""
    global jacobi_field
    if preconditioner_type == 'Green_Jacobi':
        jacobi_field = discretization.get_preconditioner_Jacobi_mugrid(
            material_data_field_ijklqxyz=tangent_field, formulation=formulation)
    elif preconditioner_type == 'dual':
        dual_field.s[...] = dual_material(args.dual_variant, args.dual_cutoff)
        negative, below_cutoff = tangent_diagnostics()
        safeguard_counts.append((negative, below_cutoff))


# orthonormal basis of the symmetric 2 x 2 tensors (Mandel), as rows of a 3 x 4 matrix in the (ij) index
SYM_BASIS = np.array([[1, 0, 0, 0], [0, 0, 0, 1], [0, 1 / np.sqrt(2), 1 / np.sqrt(2), 0]])


def pinv_abs(A, cutoff):
    """Pseudo-inverse of the symmetric matrices A [..., m, m] through |eigenvalues| (SPD), relative cutoff."""
    w, V = np.linalg.eigh(0.5 * (A + np.swapaxes(A, -1, -2)))
    w_abs = np.abs(w)
    keep = w_abs > cutoff * w_abs.max(axis=-1, keepdims=True)
    w_inv = np.where(keep, 1.0 / np.where(keep, w_abs, 1.0), 0.0)
    return np.einsum('...ab,...b,...cb->...ac', V, w_inv, V)


def tangent_diagnostics():
    """(number of quadrature points with a negative eigenvalue, with an eigenvalue below the cutoff) of the tangent"""
    A = np.moveaxis(tangent_field.s.reshape((dim * dim, dim * dim) + tangent_field.s.shape[4:]), (0, 1), (-2, -1))
    w = np.linalg.eigvalsh(0.5 * (A + np.swapaxes(A, -1, -2)))
    small = np.abs(w).min(axis=-1) <= args.dual_cutoff * np.abs(w).max(axis=-1)
    return comm.sum(int(np.sum(w.min(axis=-1) < 0))), comm.sum(int(np.sum(small)))


def dual_material(variant, cutoff):
    """C_0 A^+ C_0 of the current tangent A = dP/dF at every quadrature point, shaped like the tangent field.
    'full': pseudo-inverse of the 4 x 4 tangent; 'sym': inverse of its block on the symmetric gradients."""
    A = np.moveaxis(tangent_field.s.reshape((dim * dim, dim * dim) + tangent_field.s.shape[4:]), (0, 1), (-2, -1))
    if variant == 'full':
        A_inv = pinv_abs(A, cutoff)
    elif variant == 'inv':
        A_inv = np.linalg.inv(0.5 * (A + np.swapaxes(A, -1, -2)))
    else:
        A_ss = SYM_BASIS @ A @ SYM_BASIS.T                          # [..., 3, 3]
        A_inv = SYM_BASIS.T @ pinv_abs(A_ss, 1e-12) @ SYM_BASIS     # back to the 4 x 4 gradient space
    middle = C0_mat @ A_inv @ C0_mat
    return np.moveaxis(middle, (-2, -1), (0, 1)).reshape(tangent_field.s.shape)


def apply_green(green, x, Px):
    discretization.fft.communicate_ghosts(x)
    discretization.apply_preconditioner_mugrid(preconditioner_Fourier_fnfnqks=green, input_nodal_field_fnxyz=x,
                                               output_nodal_field_fnxyz=Px)


def M_fun(x, Px):
    if preconditioner_type == 'Green':
        apply_green(green_0, x, Px)
    elif preconditioner_type == 'Green_Jacobi':
        temp_1.s[...] = jacobi_field.s * x.s
        apply_green(green_2muI, temp_1, Px)
        Px.s[...] *= jacobi_field.s
    else:
        apply_green(green_0, x, temp_1)
        discretization.apply_system_matrix_mugrid(material_data_field=dual_field, input_field_inxyz=temp_1,
                                                  output_field_inxyz=temp_2, formulation=formulation)
        discretization.fft.communicate_ghosts(temp_2)
        apply_green(green_0, temp_2, Px)


def K_fun(x, Ax):
    discretization.apply_system_matrix_mugrid(material_data_field=tangent_field, input_field_inxyz=x,
                                              output_field_inxyz=Ax, formulation=formulation)
    discretization.fft.communicate_ghosts(Ax)


macro_gradient_inc = macro_gradient_total / float(ninc)
discretization.get_macro_gradient_field_mugrid(macro_gradient_ij=macro_gradient_inc,
                                               macro_gradient_field_ijqxyz=macro_gradient_inc_field)

# ============================================================================
# incremental loading, Newton-CG
# ============================================================================


def dense_spectra():
    """Eigenvalues of P K on the mean-free displacements at the current state, for Green, Green-Jacobi and the dual
    variants (full 1e-6, full 1e-2, sym). Serial, small grids: K and P are assembled column by column."""
    global preconditioner_type, jacobi_field
    import scipy.linalg as sl
    n = rhs_field.s.size
    e = discretization.get_unknown_size_field(name='spectrum_e')
    o = discretization.get_unknown_size_field(name='spectrum_o')

    def dense(op):
        M = np.zeros((n, n))
        for col in range(n):
            e.s[...] = 0.0
            e.s.flat[col] = 1.0
            discretization.fft.communicate_ghosts(e)
            op(e, o)
            M[:, col] = o.s.ravel()
        return M

    T = np.zeros((n, dim))
    for c in range(dim):
        t = np.zeros(rhs_field.s.shape)
        t[c] = 1.0
        T[:, c] = t.ravel()
    Q = sl.null_space(T.T)                                          # mean-free displacements
    K = Q.T @ dense(K_fun) @ Q
    saved = preconditioner_type
    out = {}
    for label, variant, cutoff in [('Green', None, None), ('Green_Jacobi', None, None), ('dual_full_1e-6', 'full', 1e-6),
                                   ('dual_full_1e-2', 'full', 1e-2), ('dual_sym', 'sym', None)]:
        preconditioner_type = 'dual' if variant else label
        if label == 'Green_Jacobi':
            jacobi_field = discretization.get_preconditioner_Jacobi_mugrid(
                material_data_field_ijklqxyz=tangent_field, formulation=formulation)
        if variant:
            dual_field.s[...] = dual_material(variant, cutoff)
        P = Q.T @ dense(M_fun) @ Q
        ev = np.sort(np.real(sl.eigvals(P @ K)))
        out[label] = ev
        if comm.rank == 0:
            print(f'  spectrum {label:15s} lambda [{ev[0]:.4g}, {ev[-1]:.4g}]  cond {ev[-1] / ev[0]:.3g}  '
                  f'#eig < 1: {np.sum(ev < 1 - 1e-8)}', flush=True)
    preconditioner_type = saved
    return out


spectra = {}
safeguard_counts = []   # dual: (#quadrature points with a negative / below-cutoff eigenvalue) per Newton step
cg_per_newton, mean_P, loading_failed = [], [], ''
start_time = time.time()
for inc in range(ninc):
    total_strain_field.s[...] += macro_gradient_inc_field.s[...]
    evaluate_material()
    norm_rhs_0 = compute_residual()
    En = np.sqrt(comm.sum(np.dot(total_strain_field.s.ravel(), total_strain_field.s.ravel())))
    cg_its = []
    for newton_it in range(newton_maxiter):
        update_preconditioner()
        nb_cg = [0]

        def count(it, x, r, p, z, stop_crit_norm):
            nb_cg[0] = it

        displacement_increment_field.s.fill(0)
        try:
            solvers.conjugate_gradients_mugrid_experimental(
                comm=comm, fc=discretization.field_collection, hessp=K_fun, b=rhs_field,
                x=displacement_increment_field, P=M_fun, tol=args.cg_tol, maxiter=cg_maxiter, callback=count,
                stop_crit_norm='energy_lower_estim', energy_rtol=True, d_min=args.cg_delay)
        except RuntimeError as error:
            loading_failed = f'increment {inc + 1}, Newton step {newton_it + 1}: {error}'
            break
        cg_its.append(nb_cg[0])
        discretization.apply_gradient_operator_mugrid(u_inxyz=displacement_increment_field,
                                                      grad_u_ijqxyz=strain_fluc_field)
        norm_strain_fluc = np.sqrt(comm.sum(np.dot(strain_fluc_field.s.ravel(), strain_fluc_field.s.ravel())))
        total_strain_field.s[...] += strain_fluc_field.s[...]
        displacement_fluctuation_field.s[...] += displacement_increment_field.s[...]
        evaluate_material()
        compute_residual()
        if norm_strain_fluc / En < newton_tol:
            break
    if loading_failed:
        if comm.rank == 0:
            print(f'load path stopped at {loading_failed}')
        break
    cg_per_newton.append(cg_its)
    mean_P.append(mean_stress())
    if inc + 1 in args.save_increments and comm.size == 1:
        os.makedirs(os.path.join(HERE, 'solutions'), exist_ok=True)
        np.savez(os.path.join(HERE, 'solutions', f'{args.geometry}_N{nnn}_kappa1e{args.kappa_exp:+d}_inc{inc + 1}.npz'),
                 u=np.array(displacement_fluctuation_field.s[:, 0]), H=(inc + 1) * macro_gradient_inc, scale=scale,
                 load_fraction=(inc + 1) / ninc)
    if inc + 1 in args.spectrum_increments:
        spectra[inc + 1] = dense_spectra()
    if comm.rank == 0:
        print(f'increment {inc + 1:3d}/{ninc}  Newton its {len(cg_its):2d}  CG its {sum(cg_its):6d} '
              f'({", ".join(str(c) for c in cg_its)})  P01 {mean_P[-1][0, 1]:+.6e}  P11 {mean_P[-1][1, 1]:+.6e}',
              flush=True)

elapsed_time = time.time() - start_time
if comm.rank == 0:
    os.makedirs(os.path.join(HERE, 'data'), exist_ok=True)
    tag = preconditioner_type + (f'_{args.dual_variant}' + (f'{args.dual_cutoff:g}' if args.dual_variant == 'full'
                                                             else '') if preconditioner_type == 'dual'
                                 and (args.dual_variant, args.dual_cutoff) != ('full', 1e-6) else '')
    out = os.path.join(HERE, 'data', f'{args.geometry}_N{nnn}_kappa1e{args.kappa_exp:+d}_{tag}.npz')
    if spectra:
        np.savez(out.replace('.npz', '_spectra.npz'),
                 **{f'inc{k}_{label}': ev for k, sp in spectra.items() for label, ev in sp.items()})
    np.savez(out, geometry=args.geometry, N=nnn, kappa=kappa, preconditioner=preconditioner_type, ninc=ninc,
             macro_gradient_total=macro_gradient_total, cg_tol=args.cg_tol, cg_delay=args.cg_delay,
             cg_per_increment=np.array([sum(c) for c in cg_per_newton]),
             newton_per_increment=np.array([len(c) for c in cg_per_newton]),
             cg_per_newton_flat=np.concatenate([np.asarray(c, dtype=int) for c in cg_per_newton])
             if cg_per_newton else np.array([], dtype=int),
             mean_P=np.array(mean_P), completed_increments=len(cg_per_newton), loading_failed=loading_failed,
             safeguard_counts=np.array(safeguard_counts),
             elapsed_time=elapsed_time)
    total = int(sum(sum(c) for c in cg_per_newton))
    newton = int(sum(len(c) for c in cg_per_newton))
    print(f'{args.geometry} N{nnn} kappa 1e{args.kappa_exp:+d} {tag}: CG {total}, Newton {newton}, '
          f'{elapsed_time:.1f} s{"; FAILED " + loading_failed if loading_failed else ""}')
