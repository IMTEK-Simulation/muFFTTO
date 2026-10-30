"""
TEST COPY of ../stefanus_2D_finite_strain_NeoHookean_TR.py: --diffuse w smooths the phase interfaces over about w
pixels (tanh profile of the signed distance to the circles, geometric interpolation of mu and lambda); needs a third
medium in the voids (true holes have mu = 0). --diffuse 0: the original sharp pixel phases.

Finite-strain homogenization of Composite A (Li & Rudykh 2019, Composites Part B 172, 352-362): uniaxial compression
of the periodic cell in X or Y with the lateral sides stress free, nominal stress vs nominal strain (paper Fig. 4).

Fixed copy of stefanus_2D_homogenization_finite_strain_elasticity_NeoHookean.py:
  * equilibrium as energy minimization by trust-region Newton-CG (NuMPI tr_newton_bounded, Steihaug CG with the
    energy-error stop: fixed tolerance 1e-6, adaptive delay >= 10; Green preconditioner); the plain Newton of the original diverged (NaN)
  * void = true hole (zero energy, stress and tangent, as in the paper); the stand-in mu_void = 1e-4 made the Green
    CG fail (no convergence in 5000 iterations); --void_mu > 0 restores a soft stand-in
  * biquadratic elements by default (Lambda/mu = 1000 locks bilinear elements)
  * load as in the paper: F_kk = 1 - eps prescribed, the lateral F_ll an unknown of the energy minimization
    (stationarity in F_ll is <P_ll> = 0: lateral side stress free)
  * homogenized 1st Piola-Kirchhoff stress <P> written per increment (CSV), deformed cell saved as PNG

The 1 x 1 primitive cell cannot show the 1 x 2 buckling mode of Composite A (paper Sec. 4); beyond the critical strain
(eps_cr = 0.016 in X, 0.026 in Y) the response follows the unstable symmetric branch.

usage:  python stefanus_2D_finite_strain_NeoHookean_TR.py -n 32 -inc 10 --direction X --eps_max 0.02
"""
import argparse
import os
import sys
import time

import numpy as np
from mpi4py import MPI

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from NuMPI.Optimization import tr_newton_bounded

from muFFTTO import domain
from muFFTTO import material_models
from muFFTTO import microstructure_library

parser = argparse.ArgumentParser(description='Finite-strain uniaxial compression of Composite A (trust-region Newton)')
parser.add_argument('-n', '--nb_pixel', type=int, default=32, help='pixels along x (NY follows from the cell)')
parser.add_argument('-inc', '--nb_increments', type=int, default=10)
parser.add_argument('--direction', choices=['X', 'Y'], default='X', help='loading direction (paper Fig. 4)')
parser.add_argument('--eps_max', type=float, default=0.02, help='final nominal compressive strain')
parser.add_argument('--element', default='biquadratic_rectangle')
parser.add_argument('--void_mu', type=float, default=0.0, help='0: true hole; > 0: soft stand-in (mu / mu_m)')
parser.add_argument('--third_medium', type=float, default=0.0,
                    help='> 0: the voids are a third medium (compressible neo-Hookean, nu = 0.3, mu = k_v mu_m) with the '
                         'HuHu - LuLu/dim regularization inside the voids (third-medium contact, as in '
                         'examples/internal_contact/Example_2d_third_medium_contact_tr_minimal.py); 0: true holes')
parser.add_argument('--tm_lam_over_mu', type=float, default=1.5,
                    help='lambda / mu of the third medium (default 1.5: nu = 0.3; 1000: the ratio of the matrix and '
                         'the inclusion)')
parser.add_argument('--lam_over_mu', type=float, default=1000.0,
                    help='Lambda / mu of the matrix and the inclusion (paper: 1000, nu = 0.4995)')
parser.add_argument('--diffuse', type=float, default=0.0,
                    help='width w [pixels] of diffuse interfaces: phase fraction phi = (1 - tanh(d / (w h))) / 2 per circle '
                         '(d: signed distance to the circle at the pixel point, h: pixel size), ln mu = sum phi_p ln mu_p '
                         'and the same for lambda; 0: sharp pixel phases')
parser.add_argument('--interp', choices=['log', 'linear_N', 'log_inside'], default='log',
                    help='log: phase fractions at the pixel points, ln mu = sum phi_p ln mu_p (one value per pixel); '
                         'linear_N: phase fractions at the nodes (sharp for --diffuse 0, else the tanh profile), '
                         'interpolated to the Gauss points by the N operator, mu = sum phi_p mu_p (linear mixing); '
                         'log_inside: as log, but the profile is shifted into the circles by 3 w h, so the matrix stays '
                         'sharp (fraction 0.25 %% at the circle boundary) and only the void / inclusion interior ramps')
parser.add_argument('--void_grade', type=float, default=0.0,
                    help='> 0: graded third medium only (matrix and inclusion sharp, as the pixel phases): inside a void '
                         'ln mu rises from ln k (deep inside) to ln CAP at the wall over about 6 w pixels (--diffuse w), '
                         'lambda / mu of the medium kept; 0: off')
parser.add_argument('--dual_refresh', action='store_true',
                    help='--precond dual: rebuild the dual middle material at every load increment from the last '
                         'converged state, with the effective isotropic moduli mu_eff = max(mu - lam ln J0, 1e-3 mu) / J0, '
                         'lam_eff = lam / J0 per pixel (J0: det of the pixel-mean F); the trust region (Green norm) is fixed')
parser.add_argument('--dual_refresh_newton', action='store_true',
                    help='with --dual_refresh: rebuild the dual at every new Newton iterate (when the tangent is '
                         'evaluated in hessp) instead of once per load increment')
parser.add_argument('--dual_refresh_mode', choices=['J', 'b', 'full'], default='J',
                    help='J: mu_eff = max(mu - lam ln J0, 1e-3 mu) / J0; b: mu_eff = max(mu tr(b0) / 2 - lam ln J0, 1e-3 mu) / J0 '
                         '(b0 = F0 F0^T of the pixel-mean F); lam_eff = lam / J0 in both; full: the exact dual of the '
                         'current SRI tangent, C_ref |A|^+ C_ref per Gauss point (shear part, 4 x 4) and per pixel '
                         '(volumetric part), |A|^+ the pseudo-inverse with |eigenvalues| (keeps the dual positive)')
parser.add_argument('--restart_frame', default=None,
                    help='start from a saved frame (inc*.npz of --frames): its u and F_macro, the load from its eps on '
                         '(steps of eps_max / nb_increments up to eps_max)')
parser.add_argument('--linear_test_state', action='store_true',
                    help='with --restart_frame: solve one Newton system at the restart state (Hessian there, right-hand '
                         'side -g after applying the next load increment, Hessian at the converged state) by plain PCG with a cap of 100 000 iterations; '
                         'print the iteration count, the exit status and the energy-error estimate history, then exit')
parser.add_argument('--inner_maxiter', type=int, default=None,
                    help='iteration cap of the Steihaug CG (NuMPI default None: the number of unknowns)')
parser.add_argument('--ramp_shift', type=float, default=None,
                    help='shift of the tanh profile into the circles, in units of w h (default: 0 for log, 3 for '
                         'log_inside, where the matrix stays sharp); in between the void skin / ligament softening balance')
parser.add_argument('--time_precond', action='store_true',
                    help='time 20 applications of the preconditioner and of the Hessian at F = I, then exit')
parser.add_argument('--tr_metric', choices=['precond', 'green'], default='precond',
                    help='norm of the trust region: precond (the preconditioner norm, NuMPI default) or green (the Green '
                         'norm N = G^-1, for --precond dual: N z = K_dual G r is a by-product of the preconditioner), as in '
                         '../dual_green_norm/stefanus_dual_trnorm.py')
parser.add_argument('--dump_material', action='store_true',
                    help='save mu, lambda per pixel (first Gauss point) to exp_data/material_N<n>_w<diffuse>.npz and exit')
parser.add_argument('--ref_lam_over_mu', type=float, default=None,
                    help='lambda / mu of the reference medium of Green and dual (mu_ref = mu_m; default: --lam_over_mu, '
                         'the matrix). Between the matrix and the third medium it splits their bulk / shear mismatch')
parser.add_argument('--incl_contrast', type=float, default=1000.0,
                    help='shear modulus of the inclusion / matrix (paper: 1000)')
parser.add_argument('--incl_lam_over_mu', type=float, default=None,
                    help='lambda / mu of the inclusion (default: --lam_over_mu, as the matrix; 1/3: bulk modulus '
                         'K = lambda + 2 mu / 3 equal to the shear modulus, nu = 0.125)')
parser.add_argument('--reg_alpha', type=float, default=1e-6,
                    help='regularization k_r = alpha L^2 mu_m, L = primitive cell width (mu_m, not the p-wave modulus '
                         'lambda_m + 2 mu_m: with Lambda/mu = 1000 that would make the voids as stiff as the matrix)')
parser.add_argument('--precond', choices=['green', 'gj', 'dual'], default='green',
                    help="green: Green of the matrix; gj: Green-Jacobi D^-1/2 G D^-1/2, D = diag of the Hessian at F = I "
                         "(fixed, so the trust-region M-norm does not change during the run); dual (--sri only): "
                         "G K_dual G with K_dual the SRI Hessian at F = I with the dual moduli mu_m^2 / mu, "
                         "lam_m^2 / lam (also fixed)")
parser.add_argument('--gj_ref', choices=['identity', 'matrix'], default='identity',
                    help='reference of the Green operator inside Green-Jacobi: 2 I_sym or the matrix tangent')
parser.add_argument('--delta0', type=float, default=0.05, help='initial trust-region radius (M-norm, RMS units)')
parser.add_argument('--delta_max', type=float, default=0.5)
parser.add_argument('--inner_tol', type=float, default=1e-6,
                    help='Steihaug CG: relative energy-error tolerance (default 1e-6; <= 0: forcing sequence)')
parser.add_argument('--inner_tau', type=float, default=0.25,
                    help='Steihaug CG: adaptive delay of the energy-error estimate (Meurant-Papez-Tichy, minimum '
                         'delay 10); <= 0: fixed delay 10')
parser.add_argument('--cells', default='1x1', help='supercell of primitive cells, e.g. 1x2 (Composite A buckles '
                                                  'into a 1 x 2 pattern, paper Sec. 4)')
parser.add_argument('--imperfection', type=float, default=0.0,
                    help='random initial fluctuation, amplitude in pixel sizes (breaks the symmetry for buckling)')
parser.add_argument('--frames', action='store_true', help='save u, F_macro and the stress per increment (video)')
parser.add_argument('--fbar', action='store_true',
                    help='F-bar (Q1 only): W(Fbar_q), Fbar_q = (J_0 / J_q)^(1/2) F_q with J_0 = det of the element mean '
                         'of F (= F at the element centre for bilinear pixels) -- no volumetric locking')
parser.add_argument('--sri', action='store_true',
                    help='Q1 selective reduced integration: mu part at the 4 Gauss points, (lam/2) ln(J0)^2 at the pixel '
                         'centre (J0 = det of the pixel mean of F); same stencil, no locking')
parser.add_argument('--tr_maxiter', type=int, default=500, help='trust-region iterations per solve (then: halve)')
parser.add_argument('--device', choices=['cpu', 'gpu'], default='cpu', help="where the fields live ('gpu': CuPy)")
parser.add_argument('--tag', default='', help='suffix of the frame folder and the output files')
parser.add_argument('--verbose', action='store_true', help='print the trust-region result of every solve')
parser.add_argument('--linear_test', action='store_true',
                    help='solve H(F = I) s = -g (g: gradient of the first load increment at the initial state) by plain '
                         'PCG with the chosen preconditioner and the Steihaug energy stop, print the CG iterations, exit. '
                         'At F = I the Hessian is positive definite (at the compressed state the stiff inclusion makes it '
                         'indefinite: initial-stress softening)')
parser.add_argument('--trace', action='store_true',
                    help='print every energy evaluation (one per trust-region step): CG its of its Steihaug solve, '
                         'energy, inf-norm of the gradient')
parser.add_argument('--check', action='store_true', help='finite-difference check of the Hessian, then exit')
parser.add_argument('--gtol', type=float, default=1e-4,
                    help='TR Newton: inf-norm of the energy gradient relative to the matrix nodal force of one load '
                         'increment, mu_m h d_eps (not to the initial gradient, which the 1000x stiffer inclusion '
                         'dominates)')
args = parser.parse_args()
rank0 = MPI.COMM_WORLD.rank == 0
if args.device == 'gpu':
    import cupy as xp
else:
    xp = np


def to_host(a_):
    return a_.get() if hasattr(a_, 'get') else np.asarray(a_)

# Unit cell input: physical units (paper Sec. 2 and 3) =====================================================
R_MM, MU_M_MPA, C_MATRIX = 3.0, 0.6, 0.25
LAM_OVER_MU = args.lam_over_mu
REF_LAM_OVER_MU = LAM_OVER_MU if args.ref_lam_over_mu is None else args.ref_lam_over_mu
INCL_LAM_OVER_MU = LAM_OVER_MU if args.incl_lam_over_mu is None else args.incl_lam_over_mu
MU_I_MPA = args.incl_contrast * MU_M_MPA
S_REF = MU_M_MPA                     # stresses in units of mu_m, lengths in units of r
incl_vol_frac = 1.0 - C_MATRIX
a = np.sqrt(2 * np.pi / (np.sqrt(3) * incl_vol_frac))   # lattice spacing in units of r
cx, cy = (int(c) for c in args.cells.split('x'))
cell_size = [2 * a, np.sqrt(3) * a]                      # primitive cell
domain_size = [cx * cell_size[0], cy * cell_size[1]]
number_of_pixels = (cx * args.nb_pixel, cy * int(round(args.nb_pixel * cell_size[1] / cell_size[0])))
dim = 2
k, l = (0, 1) if args.direction == 'X' else (1, 0)      # loaded axis k, free lateral axis l
if rank0:
    print(f'a = {a * R_MM:.3f} mm | ligament = {(a - 2) * R_MM:.3f} mm | grid = {number_of_pixels} | '
          f'{(a - 2) / (domain_size[0] / number_of_pixels[0]):.1f} px across ligament | {args.element}')

cell = domain.PeriodicUnitCell(domain_size=domain_size, problem_type='elasticity')
disc = domain.Discretization(cell=cell, nb_of_pixels_global=number_of_pixels, discretization_type='finite_element',
                             element_type=args.element, device=args.device)
comm = disc.communicator

# geometry: 0 matrix, 1 void, 2 stiff inclusion ==============================================================
def geometry_stefanus(coords_frac, vol_frac, Lx, Ly):
    """Composite A primitive cell (paper Fig. 1a): 3 voids (phase 1) at (0, 0), (1/4, 1/2), (3/4, 1/2) and one stiff
    inclusion (phase 2) at (1/2, 0) in fractional coordinates, equal radii from the circle volume fraction (as
    microstructure_library 'geometry_stefanus' of the StefanusMichael_Dev branch)"""
    radius = np.sqrt(vol_frac * Lx * Ly / (4 * np.pi))
    out_ = np.zeros(coords_frac[0].shape)
    for (fx, fy), p_ in (((0.0, 0.0), 1), ((0.25, 0.5), 1), ((0.75, 0.5), 1), ((0.5, 0.0), 2)):
        dx_ = (coords_frac[0] - fx - np.round(coords_frac[0] - fx)) * Lx
        dy_ = (coords_frac[1] - fy - np.round(coords_frac[1] - fy)) * Ly
        out_[dx_ ** 2 + dy_ ** 2 <= radius ** 2] = p_
    return out_


coords = to_host(disc.fft.coords)
phase = geometry_stefanus((np.mod(coords[0] * cx, 1.0), np.mod(coords[1] * cy, 1.0)), incl_vol_frac, *cell_size)
lam_q, mu_q = disc.get_quad_field_scalar(name='lam_q'), disc.get_quad_field_scalar(name='mu_q')
void_lam_mu = (args.tm_lam_over_mu * args.third_medium, args.third_medium) if args.third_medium > 0 else (0.0, args.void_mu)
for p, (lam_p, mu_p) in enumerate([(LAM_OVER_MU, 1.0), void_lam_mu,
                                   (INCL_LAM_OVER_MU * MU_I_MPA / MU_M_MPA, MU_I_MPA / MU_M_MPA)]):
    lam_q.s[0, 0][:, xp.asarray(phase == p)] = lam_p
    mu_q.s[0, 0][:, xp.asarray(phase == p)] = mu_p
if (args.diffuse > 0 or args.interp == 'linear_N') and args.void_grade == 0:
    assert args.third_medium > 0 or args.void_mu > 0, '--diffuse needs a soft void phase (true holes have mu = 0)'
    h_pix = cell_size[0] / args.nb_pixel
    radius = np.sqrt(incl_vol_frac * cell_size[0] * cell_size[1] / (4 * np.pi))
    fx_all, fy_all = np.mod(coords[0] * cx, 1.0), np.mod(coords[1] * cy, 1.0)

    def circle_fraction(fx, fy):
        """smooth indicator of the circle at fractional position (fx, fy) of the primitive cell"""
        dx_ = (fx_all - fx - np.round(fx_all - fx)) * cell_size[0]
        dy_ = (fy_all - fy - np.round(fy_all - fy)) * cell_size[1]
        d_ = np.sqrt(dx_ ** 2 + dy_ ** 2) - radius
        shift = args.ramp_shift if args.ramp_shift is not None else (3.0 if args.interp == 'log_inside' else 0.0)
        d_ = d_ + shift * args.diffuse * h_pix                 # 3: the ramp lies inside the circle (matrix sharp)
        if args.diffuse == 0:                                  # sharp (linear_N: node inside the circle or not)
            return (d_ <= 0).astype(float)
        return 0.5 * (1.0 - np.tanh(d_ / (args.diffuse * h_pix)))

    phi_void = np.clip(sum(circle_fraction(*c) for c in ((0.0, 0.0), (0.25, 0.5), (0.75, 0.5))), 0.0, 1.0)
    phi_incl = circle_fraction(0.5, 0.0)
    phi_matrix = np.clip(1.0 - phi_void - phi_incl, 0.0, 1.0)
    moduli = [(LAM_OVER_MU, 1.0), void_lam_mu, (INCL_LAM_OVER_MU * MU_I_MPA / MU_M_MPA, MU_I_MPA / MU_M_MPA)]
    weights = [phi_matrix, phi_void, phi_incl]
    total = sum(weights)
    if args.interp in ('log', 'log_inside'):
        for i_, (name_, field_) in enumerate((('lam', lam_q), ('mu', mu_q))):
            log_mod = sum(w_ * np.log(m_[i_]) for w_, m_ in zip(weights, moduli)) / total
            field_.s[0, 0][...] = xp.asarray(np.exp(log_mod))[None]
    else:   # nodal fractions -> Gauss points by N, then linear mixing of the moduli
        phi_nodal = disc.get_scalar_field(name='phi_nodal')
        phi_quad = disc.get_quad_field_scalar(name='phi_quad')
        lam_q.s[...] = 0.0
        mu_q.s[...] = 0.0
        for w_, (lam_m, mu_m) in zip(weights, moduli):
            phi_nodal.s[0, 0] = xp.asarray(w_ / total)
            disc.apply_N_operator_mugrid(phi_nodal, phi_quad)
            lam_q.s[...] += lam_m * phi_quad.s
            mu_q.s[...] += mu_m * phi_quad.s
    if rank0:
        print(f'diffuse interfaces: width {args.diffuse:g} px, mu in [{float(mu_q.s.min()):.2e}, {float(mu_q.s.max()):.2e}]',
              flush=True)
if args.void_grade > 0:
    assert args.third_medium > 0 and args.diffuse > 0 and args.interp == 'log' and args.ramp_shift is None
    h_pix = cell_size[0] / args.nb_pixel
    radius = np.sqrt(incl_vol_frac * cell_size[0] * cell_size[1] / (4 * np.pi))
    fx_all, fy_all = np.mod(coords[0] * cx, 1.0), np.mod(coords[1] * cy, 1.0)
    d_void = np.full(phase.shape, np.inf)                 # signed distance to the nearest void boundary
    for fx, fy in ((0.0, 0.0), (0.25, 0.5), (0.75, 0.5)):
        dx_ = (fx_all - fx - np.round(fx_all - fx)) * cell_size[0]
        dy_ = (fy_all - fy - np.round(fy_all - fy)) * cell_size[1]
        d_void = np.minimum(d_void, np.sqrt(dx_ ** 2 + dy_ ** 2) - radius)
    # t = 1 at the wall, 0 deep inside: the ramp lies inside the void, over ~6 w pixels
    t = 0.5 * (1.0 + np.tanh((d_void + 3.0 * args.diffuse * h_pix) / (args.diffuse * h_pix)))
    grade = np.exp(t * np.log(args.void_grade / args.third_medium))      # mu_TM(x) / k, from 1 to CAP / k
    in_void = xp.asarray(phase == 1)
    mu_q.s[0, 0][:, in_void] = xp.asarray(args.third_medium * grade[phase == 1])
    lam_q.s[0, 0][:, in_void] = xp.asarray(args.tm_lam_over_mu * args.third_medium * grade[phase == 1])
    if rank0:
        print(f'graded third medium: mu in the voids from {args.third_medium:g} to {float(mu_q.s[0, 0][:, in_void].max()):.2e} '
              f'(cap {args.void_grade:g}), width {args.diffuse:g} px', flush=True)
if args.dump_material:
    np.savez(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'exp_data', f'material_N{args.nb_pixel}_w{args.diffuse:g}' + {'log': '', 'linear_N': '_linN', 'log_inside': '_login'}[args.interp]
             + (f'_grade{args.void_grade:g}' if args.void_grade > 0 else '') + '.npz'),
             mu=to_host(mu_q.s[0, 0, 0]), lam=to_host(lam_q.s[0, 0, 0]), phase=phase, domain_size=domain_size,
             R_MM=R_MM)
    sys.exit()
hole_h = (phase == 1) if args.void_mu == 0 and args.third_medium == 0 else np.zeros_like(phase, dtype=bool)
hole, hole_any = xp.asarray(hole_h), bool(hole_h.any())
material = material_models.NeoHookean(discretization=disc, lam_1qxyz=lam_q, mu_1qxyz=mu_q, name='neo_hookean')
# tangent contraction: the feat/fracture_and_contact library stores A_ijkl = dP_ij / dF_kl (standard order), the older
# StefanusMichael_Dev library the reversed order P_ij = A_ijkl F_lk
A_CONTRACT = ('ijkl...,kl...->ij...' if 'standard order' in (material_models.NeoHookean.get_algorithmic_tangent.__doc__
                                                            or '') else 'ijkl...,lk...->ij...')

# fields ======================================================================================================
F = disc.get_displacement_gradient_sized_field(name='F')
P = disc.get_displacement_gradient_sized_field(name='P')
A = disc.get_material_data_size_field_mugrid(name='A')
W = disc.get_quad_field_scalar(name='W')
grad_u = disc.get_displacement_gradient_sized_field(name='grad_u')
u = disc.get_unknown_size_field(name='u')
u.s[...] = xp.asarray(args.imperfection * domain_size[0] / number_of_pixels[0]
                      * np.random.default_rng(1).standard_normal(u.s.shape))
work, out = disc.get_unknown_size_field(name='work'), disc.get_unknown_size_field(name='out')
F_macro = np.eye(dim)                # macroscopic deformation gradient; F_macro[l, l] = 1 + f is an unknown
T = disc.get_displacement_gradient_sized_field(name='T')
w_q = xp.asarray(disc.quadrature_weights)
nb_u = u.s.size
V_total = float(np.prod(disc.nb_of_pixels_global) * float(w_q.sum()))
F_SCALE = 1.0 / np.prod(disc.nb_of_pixels_global)   # f = F_SCALE * x_f: its gradient V <P_ll> F_SCALE is then of
                                                    # the size of a nodal force, so one inf-norm test fits both


# unknowns x = (u, f): the periodic fluctuation u and the lateral macro stretch f = F_ll - 1 (on rank 0 only, so
# that the dot products of the optimizer count it once). Minimizing the energy over f gives <P_ll> = 0, i.e. the
# lateral side is stress free -- one trust-region solve per load increment.
def split(x):
    f = MPI.COMM_WORLD.bcast(float(x[nb_u]) if rank0 else None, root=0)
    return x[:nb_u], F_SCALE * f


def join(u_part, f_part):
    """f_part in physical units (stretch) for x, v; in dual units (gradient, H v) use join_dual"""
    return xp.concatenate([u_part, xp.asarray([f_part / F_SCALE])]) if rank0 else u_part.copy()


def join_dual(u_part, g_f):
    return xp.concatenate([u_part, xp.asarray([g_f * F_SCALE])]) if rank0 else u_part.copy()


def sum_ll(field):
    """sum over the cell of w_q field_ll (global)"""
    return comm.sum(float(xp.einsum('q...,q->...', field.s[l, l], w_q).sum()))


INADMISSIBLE_ENERGY = 1e30   # finite: inf would give a nan trust-region ratio (no rejection, no shrinking)

# third-medium regularization R = k_r (H^T H - L^T L / dim), restricted to the voids (in the solid it would leak
# force), acting on the fluctuation only
use_reg = args.third_medium > 0
if use_reg:
    k_r = args.reg_alpha * cell_size[0] ** 2 * 1.0                   # mu_m = 1
    hess_u = disc.get_displacement_hessian_size_field(name='hess_u')
    HtH = disc.get_unknown_size_field(name='HtH')
    lap_u = disc.get_displacement_laplacian_at_quad_field(name='lap_u')
    LtL = disc.get_displacement_sized_field(name='LtL')
    reg_x, reg_out = disc.get_unknown_size_field(name='reg_x'), disc.get_unknown_size_field(name='reg_out')
    gap = xp.asarray(phase == 1)
    # Q1 on axis-aligned pixels: the discrete Laplacian is identically zero (library warning), so the L^T L part of
    # the regularization vanishes; test it once and skip it then
    reg_x.s[...] = xp.asarray(np.random.default_rng(3).standard_normal(reg_x.s.shape))
    disc.fft.communicate_ghosts(reg_x)
    disc.laplacian.apply(nodal_field=reg_x, quadrature_point_field=lap_u)
    laplacian_zero = float(xp.abs(lap_u.s).max()) == 0.0


def regularization(u_part):
    """R u for the flat fluctuation u_part"""
    reg_x.s[...] = u_part.reshape(reg_x.s.shape)
    disc.fft.communicate_ghosts(reg_x)
    disc.apply_hessian_operator_to_vector_field_mugrid(u_inxyz=reg_x, hess_u_ijkqxyz=hess_u)
    hess_u.s[..., ~gap] = 0.0
    disc.fft.communicate_ghosts(hess_u)          # the transposed operators read the ghost layer
    disc.apply_hessian_operator_transposed_to_vector_field_mugrid(hess_u_ijkqxyz=hess_u, nodal_field_inxyz=HtH,
                                                                  apply_weights=True)
    if laplacian_zero:
        return (k_r * HtH.s).ravel()
    disc.laplacian.apply(nodal_field=reg_x, quadrature_point_field=lap_u)
    lap_u.s[..., ~gap] = 0.0
    disc.fft.communicate_ghosts(lap_u)
    disc.laplacian.transpose(quadrature_point_field=lap_u, nodal_field=LtL, weights=disc.quadrature_weights)
    return (k_r * (HtH.s - LtL.s.reshape(HtH.s.shape) / dim)).ravel()


def det2_(M):
    return M[0, 0] * M[1, 1] - M[0, 1] * M[1, 0]


def inv_T(M):
    """M^-T of 2 x 2 tensors (2, 2, ...)"""
    return xp.stack([xp.stack([M[1, 1], -M[1, 0]]), xp.stack([-M[0, 1], M[0, 0]])]) / det2_(M)


def contract(a_, b_):
    return xp.einsum('ij...,ij...->...', a_, b_)


SR = {}   # SRI state of the last evaluation


def AtBA(A_, B_):
    """A B^T A per point for 2 x 2 tensors (2, 2, ...), component by component (no einsum, no stacking)"""
    c00 = A_[0, 0] * B_[0, 0] + A_[0, 1] * B_[0, 1]          # C = A B^T: C_il = A_ik B_lk
    c01 = A_[0, 0] * B_[1, 0] + A_[0, 1] * B_[1, 1]
    c10 = A_[1, 0] * B_[0, 0] + A_[1, 1] * B_[0, 1]
    c11 = A_[1, 0] * B_[1, 0] + A_[1, 1] * B_[1, 1]
    out_ = xp.empty(xp.broadcast_shapes(A_.shape, B_.shape), dtype=B_.dtype)
    out_[0, 0] = c00 * A_[0, 0] + c01 * A_[1, 0]              # (C A)_ij = C_il A_lj
    out_[0, 1] = c00 * A_[0, 1] + c01 * A_[1, 1]
    out_[1, 0] = c10 * A_[0, 0] + c11 * A_[1, 0]
    out_[1, 1] = c10 * A_[0, 1] + c11 * A_[1, 1]
    return out_


def sri_dP(st, dF, lam_, mu_):
    """SRI tangent: mu (dF + F^-T dF^T F^-T) at the Gauss points + lam d(ln J0 F0^-T), dF0 = pixel mean of dF
    (the products F^-T dF^T F^-T written out for 2 x 2 tensors)"""
    dF0 = dF.mean(axis=2, keepdims=True)
    FiT, F0iT = st['FiT'], st['F0iT']
    out_ = AtBA(FiT, dF)
    out_ += dF
    out_ *= mu_
    vol = AtBA(F0iT, dF0)
    vol *= -st['lnJ0']
    vol += contract(F0iT, dF0) * F0iT
    vol *= lam_
    out_ += vol
    return out_


FB, FBH = {}, {}       # F-bar state of the last evaluation (FBH: the one the Hessian belongs to)


def evaluate(x, tangent=False):
    """F = F_macro + grad u at every quadrature point; P, W (and A) with the hole set to zero"""
    u_part, f = split(x)
    F_macro[l, l] = 1.0 + f
    work.s[...] = u_part.reshape(work.s.shape)
    disc.fft.communicate_ghosts(work)
    disc.apply_gradient_operator_mugrid(u_inxyz=work, grad_u_ijqxyz=grad_u)
    F.s[...] = grad_u.s + xp.asarray(F_macro)[(...,) + (None,) * (F.s.ndim - 2)]
    if hole_any:
        F.s[..., hole] = xp.eye(dim)[..., None, None]           # keeps log J finite in the hole, zeroed below
    FB['admissible'] = True
    if args.fbar:
        Fraw = F.s.copy()
        F0 = Fraw.mean(axis=2, keepdims=True)                   # element mean (4 Gauss points of the pixel)
        J, J0 = det2_(Fraw), det2_(F0)
        if comm.min(min(float(J.min()), float(J0.min()))) <= 0.0:
            FB['admissible'] = False
            F.s[...] = xp.eye(dim)[(...,) + (None,) * (F.s.ndim - 2)]
        else:
            alpha = xp.sqrt(J0 / J)
            FB.update(Fraw=Fraw, alpha=alpha, FinvT=inv_T(Fraw), F0invT=inv_T(xp.broadcast_to(F0, Fraw.shape)))
            F.s[...] = alpha * Fraw                             # the material sees Fbar
    if args.sri:
        F0 = xp.broadcast_to(F.s.mean(axis=2, keepdims=True), F.s.shape)   # F at the pixel centre (Q1)
        J, J0 = det2_(F.s), det2_(F0)
        if comm.min(min(float(J.min()), float(J0.min()))) <= 0.0:
            FB['admissible'] = False
            F.s[...] = xp.eye(dim)[(...,) + (None,) * (F.s.ndim - 2)]
            F0, J, J0 = F.s, det2_(F.s), det2_(F.s)
        lam_, mu_ = lam_q.s[0, 0], mu_q.s[0, 0]
        SR.update(FiT=inv_T(F.s), F0iT=inv_T(F0), lnJ0=xp.log(J0), lam=lam_, mu=mu_)
        W.s[0, 0] = 0.5 * mu_ * (contract(F.s, F.s) - dim) - mu_ * xp.log(J) + 0.5 * lam_ * SR['lnJ0'] ** 2
        P.s[...] = mu_ * (F.s - SR['FiT']) + lam_ * SR['lnJ0'] * SR['F0iT']
    else:
        material.get_stress(F, P)
        material.get_energy_density(F, W)
    if tangent and not args.sri:
        material.get_algorithmic_tangent(F, A)
    if hole_any:
        P.s[..., hole] = 0.0
        W.s[..., hole] = 0.0
        if tangent and not args.sri:
            A.s[..., hole] = 0.0
    if args.fbar and FB['admissible']:
        FB['Pbar'] = P.s.copy()
        if tangent:
            FBH.update(FB)


def fbar_T(st):
    """dPi = sum w T_q : dF_q:  T = alpha Pbar - s F^-T + <s> F0^-T,  s = alpha (Pbar : F) / d"""
    s_ = st['alpha'] * contract(st['Pbar'], st['Fraw']) / dim
    return st['alpha'] * st['Pbar'] - s_ * st['FinvT'] + s_.mean(axis=0, keepdims=True) * st['F0invT']


def fbar_dT(st, Afield, dF):
    """directional derivative of T in the direction dF (2, 2, q, ...)"""
    al, Fr, Pb, FiT, F0iT = st['alpha'], st['Fraw'], st['Pbar'], st['FinvT'], st['F0invT']
    dF0 = xp.broadcast_to(dF.mean(axis=2, keepdims=True), dF.shape)
    a_ = (contract(F0iT, dF0) - contract(FiT, dF)) / dim           # d alpha = alpha a_
    dFbar = al * (a_ * Fr + dF)
    dPb = xp.einsum(A_CONTRACT, Afield, dFbar)
    PF = contract(Pb, Fr)
    s_ = al * PF / dim
    ds = al * (a_ * PF + contract(dPb, Fr) + contract(Pb, dF)) / dim
    dFiT = -xp.einsum('ik...,lk...,lj...->ij...', FiT, dF, FiT)
    dF0iT = -xp.einsum('ik...,lk...,lj...->ij...', F0iT, dF0, F0iT)
    m_, dm = s_.mean(axis=0, keepdims=True), ds.mean(axis=0, keepdims=True)
    return al * a_ * Pb + al * dPb - ds * FiT - s_ * dFiT + dm * F0iT + m_ * dF0iT


def fun(x):
    """total energy and its gradient (B^T w P, sum w P_ll)"""
    evaluate(x)
    J = F.s[0, 0] * F.s[1, 1] - F.s[0, 1] * F.s[1, 0]
    if not FB['admissible'] or comm.min(float(J.min())) <= 0.0:
        return INADMISSIBLE_ENERGY, xp.zeros_like(x)
    energy = comm.sum(float(xp.einsum('q...,q->...', W.s[0, 0], w_q).sum()))
    if not np.isfinite(energy):
        return INADMISSIBLE_ENERGY, xp.zeros_like(x)
    if args.fbar:
        P.s[...] = fbar_T(FB)               # from here on P holds T = dW(Fbar)/dF (its mean is the macro stress)
    disc.fft.communicate_ghosts(P)
    disc.apply_gradient_transposed_operator_mugrid(gradient_field_ijqxyz=P, div_u_fnxyz=out, apply_weights=True)
    g_u = out.s.ravel().copy()
    if use_reg:
        Ru = regularization(x[:nb_u])
        energy += 0.5 * comm.sum(float(xp.dot(x[:nb_u], Ru)))
        g_u += Ru
    return energy, join_dual(g_u, sum_ll(P))


hess_at = [None]


def hessp(x, v):
    """Hessian of the energy in (u, f): dF = grad v_u + v_f e_ll, dP = A : dF, (B^T w dP, sum w dP_ll)"""
    if hess_at[0] is None or not bool(xp.array_equal(hess_at[0], x)):
        evaluate(x, tangent=True)
        hess_at[0] = x.copy()
        if args.dual_refresh and args.dual_refresh_newton and args.precond == 'dual':
            refresh_dual()
    v_u, v_f = split(v)
    work.s[...] = v_u.reshape(work.s.shape)
    disc.fft.communicate_ghosts(work)
    disc.apply_gradient_operator_mugrid(u_inxyz=work, grad_u_ijqxyz=grad_u)
    grad_u.s[l, l] += v_f
    if args.sri:
        T.s[...] = sri_dP(SR, grad_u.s, SR['lam'], SR['mu'])
    elif args.fbar:
        T.s[...] = fbar_dT(FBH, A.s, grad_u.s)
    else:
        T.s[...] = xp.einsum(A_CONTRACT, A.s, grad_u.s)
    disc.fft.communicate_ghosts(T)
    disc.apply_gradient_transposed_operator_mugrid(gradient_field_ijqxyz=T, div_u_fnxyz=out, apply_weights=True)
    Hv = out.s.ravel().copy()
    if use_reg:
        Hv += regularization(v_u)
    return join_dual(Hv, sum_ll(T))


I_sym = 0.5 * (np.einsum('ik,jl->ijkl', np.eye(dim), np.eye(dim)) + np.einsum('il,jk->ijkl', np.eye(dim), np.eye(dim)))
C_matrix = REF_LAM_OVER_MU * np.einsum('ij,kl->ijkl', np.eye(dim), np.eye(dim)) + 2.0 * I_sym
C_ref = 2.0 * I_sym if args.precond == 'gj' and args.gj_ref == 'identity' else C_matrix


def fbar_reference_operator(input_field_inxyz, output_field_inxyz):
    """the F-bar Hessian at F = I for the uniform reference C_ref (its Fourier symbol is what Green inverts)"""
    disc.fft.communicate_ghosts(input_field_inxyz)
    disc.apply_gradient_operator_mugrid(u_inxyz=input_field_inxyz, grad_u_ijqxyz=grad_u)
    st = dict(alpha=1.0, Fraw=xp.eye(dim)[(...,) + (None,) * (grad_u.s.ndim - 2)] + 0 * grad_u.s,
              Pbar=0 * grad_u.s)
    st['FinvT'] = st['F0invT'] = st['Fraw']
    T.s[...] = fbar_dT(st, xp.asarray(C_ref)[(...,) + (None,) * (grad_u.s.ndim - 2)], grad_u.s)
    disc.fft.communicate_ghosts(T)
    disc.apply_gradient_transposed_operator_mugrid(gradient_field_ijqxyz=T, div_u_fnxyz=output_field_inxyz,
                                                   apply_weights=True)


def sri_reference_operator(input_field_inxyz, output_field_inxyz):
    """the SRI Hessian at F = I for the matrix (lam, mu) (translation invariant: Green inverts its symbol)"""
    disc.fft.communicate_ghosts(input_field_inxyz)
    disc.apply_gradient_operator_mugrid(u_inxyz=input_field_inxyz, grad_u_ijqxyz=grad_u)
    I_ = xp.eye(dim)[(...,) + (None,) * (grad_u.s.ndim - 2)] + 0 * grad_u.s
    T.s[...] = sri_dP(dict(FiT=I_, F0iT=I_, lnJ0=0.0), grad_u.s, REF_LAM_OVER_MU, 1.0)
    disc.fft.communicate_ghosts(T)
    disc.apply_gradient_transposed_operator_mugrid(gradient_field_ijqxyz=T, div_u_fnxyz=output_field_inxyz,
                                                   apply_weights=True)


# dual preconditioner G K_dual G (--precond dual). At F = I the SRI Hessian is B^T C B with a block-diagonal C:
# mu(x) 2 I_sym at the Gauss points and lam(x) I x I at the pixel centre. Then C_ref C^-1 C_ref (pseudo-inverse on
# the range of each block) is again of this form, with mu_m^2 / mu and lam_m^2 / lam: K_dual is the SRI Hessian at
# F = I with these dual moduli
if args.precond == 'dual':
    assert args.sri, '--precond dual is implemented for --sri only'
    assert float(lam_q.s.min()) > 0 and float(mu_q.s.min()) > 0, '--precond dual needs lam, mu > 0 (no true holes)'
    lam_dual, mu_dual = REF_LAM_OVER_MU ** 2 / lam_q.s[0, 0], 1.0 / mu_q.s[0, 0]     # mu_m = 1
    green_out = disc.get_unknown_size_field(name='green_out')
    dual_out = disc.get_unknown_size_field(name='dual_out')


def sri_dual_operator(input_field_inxyz, output_field_inxyz):
    """K_dual: the SRI Hessian at F = I with the dual moduli (lam_dual, mu_dual). At F = I the SRI tangent is
    mu (grad v + grad v^T) at the Gauss points + lam tr(pixel mean of grad v) I (sri_dP with F = I), written out
    directly instead of the general contraction with F^-T (same result, fewer temporaries)"""
    disc.fft.communicate_ghosts(input_field_inxyz)
    disc.apply_gradient_operator_mugrid(u_inxyz=input_field_inxyz, grad_u_ijqxyz=grad_u)
    g = grad_u.s
    if DUAL_FULL:            # full anisotropic dual: 4 x 4 per Gauss point on the shear part, scalar per pixel on the volume
        T.s[...] = xp.einsum('ab...,b...->a...', DUAL_FULL['mu'], g.reshape((4,) + g.shape[2:])).reshape(g.shape)
        vol = DUAL_FULL['lam'] * (g[0, 0] + g[1, 1]).mean(axis=0, keepdims=True)
        T.s[0, 0] += vol
        T.s[1, 1] += vol
        disc.fft.communicate_ghosts(T)
        disc.apply_gradient_transposed_operator_mugrid(gradient_field_ijqxyz=T, div_u_fnxyz=output_field_inxyz,
                                                       apply_weights=True)
        return
    T.s[...] = mu_dual * (g + xp.swapaxes(g, 0, 1))
    vol = lam_dual * (g[0, 0] + g[1, 1]).mean(axis=0, keepdims=True)      # lam tr(mean_q grad v), per Gauss point
    T.s[0, 0] += vol
    T.s[1, 1] += vol
    disc.fft.communicate_ghosts(T)
    disc.apply_gradient_transposed_operator_mugrid(gradient_field_ijqxyz=T, div_u_fnxyz=output_field_inxyz,
                                                   apply_weights=True)


green = disc.get_preconditioner_Green_mugrid(
    reference_material_data_ijkl=C_ref,
    operator=sri_reference_operator if args.sri else (fbar_reference_operator if args.fbar else None))

if args.precond == 'gj':
    # diagonal of the Hessian at F = I (fixed): probing with combs of stride 2 per component and nodal sub-point --
    # the element stencil reaches one pixel, so the nodes of a comb never share an element
    x_I = join(xp.zeros(nb_u), 0.0)
    diag = xp.zeros(work.s.shape)
    comb = xp.zeros(work.s.shape)
    for d_ in range(work.s.shape[0]):
        for n_ in range(work.s.shape[1]):
            for ox in range(2):
                for oy in range(2):
                    comb[...] = 0.0
                    comb[d_, n_, ox::2, oy::2] = 1.0
                    Hc = hessp(x_I, join(comb.ravel(), 0.0))[:nb_u].reshape(work.s.shape)
                    diag[d_, n_, ox::2, oy::2] = Hc[d_, n_, ox::2, oy::2]
    hess_at[0] = None
    d_inv_half = xp.where(diag > 0, 1.0 / xp.sqrt(xp.abs(diag)), 0.0)
    if rank0:
        print(f'Green-Jacobi: diag of K in [{float(diag.min()):.2e}, {float(diag.max()):.2e}], '
              f'Green reference {args.gj_ref}')


metric_cache = {}


DUAL_FULL = {}            # full (anisotropic) dual middle material, see refresh_dual


def tangent_4x4(apply_map, shape):
    """the linear map dF -> apply_map(dF) on 2 x 2 tensors as a 4 x 4 matrix per point, [a, b, ...], a = 2 i + j"""
    M_ = xp.zeros((4, 4) + shape)
    for b_ in range(4):
        E_ = xp.zeros((2, 2) + (1,) * len(shape))
        E_[b_ // 2, b_ % 2] = 1.0
        M_[:, b_] = apply_map(E_).reshape((4,) + shape)
    return M_


def abs_pinv_sandwich(M_, C_):
    """C |M|^+ C per point (M symmetrized, |M|^+ = V diag(1/|lambda|) V^T, tiny eigenvalues dropped); [4, 4, ...]"""
    Mp = xp.moveaxis(0.5 * (M_ + xp.swapaxes(M_, 0, 1)), (0, 1), (-2, -1))
    w_, V_ = xp.linalg.eigh(Mp)
    aw = xp.abs(w_)
    inv_w = xp.where(aw > 1e-12 * aw.max(axis=-1, keepdims=True), 1.0 / xp.maximum(aw, 1e-300), 0.0)
    Mi = xp.einsum('...ab,...b,...cb->...ac', V_, inv_w, V_)
    return xp.moveaxis(xp.einsum('ab,...bc,cd->...ad', C_, Mi, C_), (-2, -1), (0, 1))


def refresh_dual():
    """dual middle material from the state of the last evaluate: effective isotropic moduli per pixel (J, b), or the
    exact dual of the current SRI tangent (full)"""
    global lam_dual, mu_dual
    if args.dual_refresh_mode == 'full':
        shape = mu_q.s[0, 0].shape
        mu_, lam_ = mu_q.s[0, 0], lam_q.s[0, 0]
        # shear part per Gauss point: dF -> mu (dF + F^-T dF^T F^-T); reference mu_ref (I + transposition), mu_ref = 1
        M_mu = tangent_4x4(lambda E_: mu_ * (E_ + AtBA(SR['FiT'], E_)), shape)
        C_mu = xp.asarray((np.eye(4) + np.eye(4)[[0, 2, 1, 3]]))
        DUAL_FULL['mu'] = abs_pinv_sandwich(M_mu, C_mu)
        # volumetric part per pixel: dF0 -> lam ((F0^-T : dF0) F0^-T - ln J0 F0^-T dF0^T F0^-T); reference lam_ref I x I
        F0iT = SR['F0iT'][:, :, :1]
        lam0 = lam_[:1]
        M_lam = tangent_4x4(lambda E_: lam0 * (contract(F0iT, E_) * F0iT - SR['lnJ0'][:1] * AtBA(F0iT, E_)),
                            lam0.shape)
        vecI = np.array([1.0, 0.0, 0.0, 1.0])
        C_lam = xp.asarray(REF_LAM_OVER_MU * np.outer(vecI, vecI))
        lam_d = abs_pinv_sandwich(M_lam, C_lam)[0, 0] / 1.0                 # = lam_ref^2 (I : |M|^+ : I) (I x I)_00
        DUAL_FULL['lam'] = xp.broadcast_to(lam_d, shape)
        return
    J0_ = xp.exp(SR['lnJ0'])
    stretch_ = 1.0
    if args.dual_refresh_mode == 'b':                     # tr(b0) / 2 of the pixel-mean F (1 at F = I)
        F0_ = F.s.mean(axis=2, keepdims=True)
        stretch_ = xp.broadcast_to(0.5 * contract(F0_, F0_), mu_q.s[0, 0].shape)
    mu_eff = xp.maximum(mu_q.s[0, 0] * stretch_ - lam_q.s[0, 0] * SR['lnJ0'], 1e-3 * mu_q.s[0, 0]) / J0_
    lam_dual, mu_dual = REF_LAM_OVER_MU ** 2 / (lam_q.s[0, 0] / J0_), 1.0 / mu_eff


def precond(r):
    """Green (or Green-Jacobi, or dual) on u; the macro unknown scaled by 1 / (V mu_m) (its Hessian is V times a modulus)"""
    r_u, r_f = r[:nb_u], (float(r[nb_u]) if rank0 else 0.0)
    work.s[...] = r_u.reshape(work.s.shape)
    if args.precond == 'gj':
        work.s[...] *= d_inv_half
    disc.fft.communicate_ghosts(work)
    if args.precond == 'dual':      # G K_dual G
        disc.apply_preconditioner_mugrid(preconditioner_Fourier_fnfnqks=green, input_nodal_field_fnxyz=work,
                                         output_nodal_field_fnxyz=green_out)
        sri_dual_operator(green_out, dual_out)
        # N z for the Green-norm trust region (N = G^-1): u part K_dual G r, macro part r_f (its z is r_f / c)
        metric_cache['Nz'] = (xp.concatenate([dual_out.s.ravel(), xp.asarray([r_f])]) if rank0
                              else dual_out.s.ravel().copy())
        disc.fft.communicate_ghosts(dual_out)
        disc.apply_preconditioner_mugrid(preconditioner_Fourier_fnfnqks=green, input_nodal_field_fnxyz=dual_out,
                                         output_nodal_field_fnxyz=out)
    else:
        disc.apply_preconditioner_mugrid(preconditioner_Fourier_fnfnqks=green, input_nodal_field_fnxyz=work,
                                         output_nodal_field_fnxyz=out)
    if args.precond == 'gj':
        out.s[...] *= d_inv_half
    return (xp.concatenate([out.s.ravel(), xp.asarray([r_f / (V_total * F_SCALE ** 2)])]) if rank0
            else out.s.ravel().copy())


def mean_P():
    """homogenized 1st Piola-Kirchhoff stress <P> (units of mu_m)"""
    s = to_host(xp.einsum('ijq...,q->ij...', P.s, w_q).reshape(dim, dim, -1).sum(axis=-1))
    return comm.sum(np.asfortranarray(s)) / V_total


if args.check:  # directional derivative of the gradient against hessp, at a random state
    rng = np.random.default_rng(0)
    F_macro[k, k] = 0.98
    x = join(xp.asarray(1e-3 * rng.standard_normal(nb_u)), 0.01)
    vv = join(xp.asarray(rng.standard_normal(nb_u)), 1.0)
    g0, Hv = fun(x)[1], hessp(x, vv)
    for e in (1e-5, 1e-6, 1e-7):
        print(f'check: eps {e:.0e}  |(g(x + eps v) - g(x)) / eps - H v| / |H v| = '
              f'{float(xp.linalg.norm((fun(x + e * vv)[1] - g0) / e - Hv) / xp.linalg.norm(Hv)):.2e}')
    sys.exit()

counts = dict(newton=0, cg=0)


def equilibrate(x0):
    """minimize the energy over (u, f) for the current F_kk from x0; returns (success, <P>, f, TR its, CG its).
    On success u holds the solution; on failure the caller restores the last converged state."""
    cg_its = [0]
    h = domain_size[0] / number_of_pixels[0]
    gtol = args.gtol * 1.0 * h * args.eps_max / args.nb_increments      # mu_m = 1, nominal increment

    def hessp_counted(x, v):
        cg_its[0] += 1
        return hessp(x, v)
    # NuMPI's trust-region solver works on host (numpy) vectors: the device fields are filled inside fun / hessp /
    # precond, only the vectors cross (a few 10^4 numbers per call)
    trace = {'evals': 0, 'cg_at_last_eval': 0}

    def fun_h(x_h):
        e_, g_ = fun(xp.asarray(x_h))
        if args.trace and rank0:
            print(f'    eval {trace["evals"]:4d}  CG {cg_its[0] - trace["cg_at_last_eval"]:6d}  energy {e_:.10e}  '
                  f'|g|_inf {float(abs(g_).max()):.2e}', flush=True)
            trace['evals'] += 1
            trace['cg_at_last_eval'] = cg_its[0]
        return e_, to_host(g_)
    res = tr_newton_bounded(fun_h, to_host(x0), lambda x_h, v_h: to_host(hessp_counted(xp.asarray(x_h), xp.asarray(v_h))),
                            precond=lambda r_h: to_host(precond(xp.asarray(r_h))), gtol=gtol,
                            precond_metric=(lambda z_h: to_host(metric_cache['Nz'])) if args.tr_metric == 'green' else None,
                            maxiter=args.tr_maxiter, delta0=args.delta0, delta_max=args.delta_max, inner_stop='energy',
                            inner_delay=10, inner_tol=args.inner_tol if args.inner_tol > 0 else None,
                            inner_tau=args.inner_tau if args.inner_tau > 0 else None, inner_maxiter=args.inner_maxiter,
                            comm=MPI.COMM_WORLD)
    res.x = xp.asarray(res.x)
    counts['newton'] += res.nit
    counts['cg'] += cg_its[0]
    ok = bool(res.success) and res.fun < 0.5 * INADMISSIBLE_ENERGY
    if rank0 and (args.verbose or not ok):
        print(f'   TR Newton: {res.message} | gtol {gtol:.2e}', flush=True)
    u_part, f = split(res.x)
    if ok:
        u.s[...] = u_part.reshape(u.s.shape)
        fun(res.x)
    return ok, (mean_P() if ok else None), f, res.nit, cg_its[0]


# uniaxial compression with an adaptive load step (as in the slide problems of
# examples/phase_field_fracture_adapted_grid): each increment starts from the last converged state (plain trust-region
# Newton-CG, no predictor); a failed trust-region solve halves the step, a success lets it grow back (x 2) up to the
# nominal step eps_max / nb_increments
max_step = args.eps_max / args.nb_increments
if args.time_precond:
    r_t = join(xp.asarray(np.random.default_rng(0).standard_normal(nb_u)), 0.0)
    x_t = join(xp.zeros(nb_u), 0.0)
    hessp(x_t, r_t)                                   # evaluate the tangent once
    v_t = r_t[:nb_u]

    def grad_only():
        work.s[...] = v_t.reshape(work.s.shape)
        disc.fft.communicate_ghosts(work)
        disc.apply_gradient_operator_mugrid(u_inxyz=work, grad_u_ijqxyz=grad_u)

    def transpose_only():
        disc.fft.communicate_ghosts(T)
        disc.apply_gradient_transposed_operator_mugrid(gradient_field_ijqxyz=T, div_u_fnxyz=out, apply_weights=True)
    parts = [('precond', lambda: precond(r_t)), ('hessp', lambda: hessp(x_t, r_t)), ('  gradient', grad_only),
             ('  sri_dP', lambda: sri_dP(SR, grad_u.s, SR['lam'], SR['mu'])), ('  transpose', transpose_only),
             ('  array_equal(x)', lambda: bool(xp.array_equal(hess_at[0], x_t)))]
    if use_reg:
        parts.append(('  regularization', lambda: regularization(v_t)))
    for name_, f_ in parts:
        f_()
        if args.device == 'gpu':
            xp.cuda.Device().synchronize()
        t_ = time.time()
        for _ in range(20):
            f_()
        if args.device == 'gpu':
            xp.cuda.Device().synchronize()
        if rank0:
            print(f'time {name_} ({args.precond}): {(time.time() - t_) / 20 * 1e3:.2f} ms', flush=True)
    sys.exit()
if args.linear_test:   # PCG = Steihaug CG with an infinite trust region
    from NuMPI.Optimization.BoundedTRNewtonCG import _steihaug_cg
    from NuMPI.Tools import Reduction
    F_macro[k, k] = 1.0 - max_step
    g_lin = fun(join(u.s.ravel(), 0.0))[1]
    F_macro[k, k] = 1.0
    x_lin = join(xp.zeros(nb_u), 0.0)      # Hessian at F = I
    if rank0:   # curvature along the first PCG direction z = M^-1 g, and along random directions
        z_lin = precond(g_lin)
        rng_lin = np.random.default_rng(0)
        print(f'linear test: g.z = {float(g_lin @ z_lin):.3e}, z.Hz = {float(z_lin @ hessp(x_lin, z_lin)):.3e}', flush=True)
        for _ in range(3):
            v_lin = xp.asarray(rng_lin.standard_normal(g_lin.shape))
            print(f'linear test: random v: v.Hv / v.v = {float(v_lin @ hessp(x_lin, v_lin)) / float(v_lin @ v_lin):.3e}, '
                  f'v.Mv (precond) / v.v = {float(v_lin @ precond(v_lin)) / float(v_lin @ v_lin):.3e}', flush=True)
    t_lin = time.time()
    curvatures = []   # v.Hv / v.v of every CG direction

    def hessp_lin(v):          # host vectors in and out (MPI reductions on host arrays, as in the trust region)
        Hv_ = to_host(hessp(x_lin, xp.asarray(v)))
        curvatures.append(float(v @ Hv_) / float(v @ v))
        return Hv_
    s_lin, _, _, status, nb_cg = _steihaug_cg(hessp_lin, to_host(g_lin), np.inf, args.inner_tol, 50000,
                                              Reduction(MPI.COMM_WORLD),
                                              precond=lambda r_h: to_host(precond(xp.asarray(r_h))), stop='energy', delay=10,
                                              tau=args.inner_tau if args.inner_tau > 0 else None)
    if rank0:
        print(f'linear test: precond {args.precond}, third medium {args.third_medium:g} (lam/mu {args.tm_lam_over_mu:g}), '
              f'inclusion {args.incl_contrast:g}, grid {number_of_pixels}: '
              f'{nb_cg} CG iterations ({status}), {time.time() - t_lin:.1f} s', flush=True)
        print(f'linear test: v.Hv / v.v of the CG directions: min {min(curvatures):.3e}, last {curvatures[-3:]}',
              flush=True)
    sys.exit()

rows = []
t0 = time.time()
min_step = max_step / 64
eps, d_eps, f_cur, inc = 0.0, max_step, 0.0, 0
if args.restart_frame:
    frame0 = np.load(args.restart_frame)
    u.s[:, 0] = xp.asarray(frame0['u'])
    F_macro[...] = frame0['F_macro']
    eps, f_cur = float(frame0['eps']), float(frame0['F_macro'][l, l]) - 1.0
    if rank0:
        print(f'restart from {args.restart_frame}: eps {eps:.4f}, lateral stretch {f_cur:+.5f}', flush=True)
    if args.linear_test_state:
        from NuMPI.Optimization.BoundedTRNewtonCG import _steihaug_cg
        from NuMPI.Tools import Reduction
        x_st = join(u.s.ravel(), f_cur)
        if args.dual_refresh and args.precond == 'dual':
            evaluate(x_st)
            refresh_dual()
        F_macro[k, k] = 1.0 - (eps + max_step)
        g_st = to_host(fun(x_st)[1])                  # right-hand side of the next increment
        F_macro[k, k] = 1.0 - eps                     # Hessian at the converged state (stable: positive definite)
        hess_at[0] = None
        t_st = time.time()
        for maxit in (100000,):
            _, _, _, status_st, nb_st = _steihaug_cg(
                lambda v: to_host(hessp(x_st, xp.asarray(v))), g_st, np.inf, args.inner_tol, maxit,
                Reduction(MPI.COMM_WORLD), precond=lambda r_h: to_host(precond(xp.asarray(r_h))), stop='energy',
                delay=10, tau=args.inner_tau if args.inner_tau > 0 else None)
        if rank0:
            print(f'linear test at state eps {eps:.4f}: precond {args.precond}: {nb_st} CG iterations ({status_st}), '
                  f'{time.time() - t_st:.0f} s', flush=True)
        sys.exit()
while eps < args.eps_max - 1e-12:
    d_eps = min(d_eps, args.eps_max - eps)
    F_macro[k, k] = 1.0 - (eps + d_eps)
    x_saved = join(u.s.ravel(), f_cur)
    if args.dual_refresh and args.precond == 'dual':    # effective moduli at the last converged state
        F_macro[k, k] = 1.0 - eps
        evaluate(x_saved)
        refresh_dual()
        hess_at[0] = None
        F_macro[k, k] = 1.0 - (eps + d_eps)
    x0 = x_saved                                      # plain continuation from the last converged state
    ok, Pm, f, nit, ncg = equilibrate(x0)
    if not ok:
        u_part, f_cur = split(x_saved)
        u.s[...] = u_part.reshape(u.s.shape)
        d_eps *= 0.5
        if rank0:
            print(f'  eps {eps + d_eps * 2:.5f}: trust region failed -> halve the step to {d_eps:.2e}', flush=True)
        if d_eps < min_step:
            if rank0:
                print('step below min_step: stopping', flush=True)
            break
        continue
    eps += d_eps
    inc += 1
    f_cur = f
    rows.append((eps, -Pm[k, k] * S_REF, Pm[l, l] * S_REF, f, nit, ncg))
    if args.frames and comm.size == 1:
        frame_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'exp_data',
                                 f'frames_TR_{args.direction}_{number_of_pixels[0]}x{number_of_pixels[1]}_cells{args.cells}'
                                 + (f'_TM{args.third_medium:g}' if args.third_medium > 0 else '') + ('_sri' if args.sri else '') + args.tag)
        os.makedirs(frame_dir, exist_ok=True)
        np.savez(os.path.join(frame_dir, f'inc{inc:04d}.npz'), u=to_host(u.s[:, 0]), F_macro=F_macro, eps=eps,
                 stress_MPa=-Pm[k, k] * S_REF,
                 P_kk=to_host(xp.einsum('q...,q->...', P.s[k, k], w_q) / w_q.sum() * S_REF),
                 lateral_strain=f, phase=phase, domain_size=domain_size, R_MM=R_MM, direction=args.direction)
    if rank0:
        print(f'inc {inc:3d}  eps_{args.direction} = {eps:.4f} (step {d_eps:.4f})  nominal stress '
              f'{-Pm[k, k] * S_REF:.5f} MPa  lateral strain {f:+.5f}  <P_ll>/<P_kk> = {Pm[l, l] / Pm[k, k]:.1e}  '
              f'(TR its {nit}, CG its {ncg})', flush=True)
    d_eps = min(2.0 * d_eps, max_step)

elapsed = time.time() - t0
here = os.path.dirname(os.path.abspath(__file__))
tag = f'{args.direction}_{number_of_pixels[0]}x{number_of_pixels[1]}_{args.element.split("_")[0]}_cells{args.cells}' + \
      (f'_TM{args.third_medium:g}' if args.third_medium > 0 else '') + args.tag
if rank0:
    rows = np.array(rows)
    os.makedirs(os.path.join(here, 'exp_data'), exist_ok=True)
    np.savetxt(os.path.join(here, 'exp_data', f'finite_strain_TR_{tag}.csv'), rows, delimiter=',', fmt='%.8g',
               comments='', header='eps [-],nominal stress [MPa],lateral P [MPa],lateral strain F_ll - 1 [-],'
                                   'TR its,CG its')
    E0 = rows[0, 1] / rows[0, 0]
    print(f'initial slope {E0:.4f} MPa (paper Fig. 4, periodic simulation: X 0.35, Y 0.50 MPa) | '
          f'{counts["newton"]} TR its, {counts["cg"]} CG its, {elapsed:.1f} s')

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap, BoundaryNorm
    # deformed cell: pixel corners mapped by F_macro x + u (u at the pixel corner nodes)
    nx, ny = number_of_pixels
    X, Y = np.meshgrid(np.arange(nx + 1) * domain_size[0] / nx, np.arange(ny + 1) * domain_size[1] / ny,
                       indexing='ij')
    ux = np.pad(to_host(u.s[0, 0]), ((0, 1), (0, 1)), mode='wrap')
    uy = np.pad(to_host(u.s[1, 0]), ((0, 1), (0, 1)), mode='wrap')
    xd = F_macro[0, 0] * X + F_macro[0, 1] * Y + ux
    yd = F_macro[1, 0] * X + F_macro[1, 1] * Y + uy
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), layout='constrained')
    axes[0].pcolormesh(xd * R_MM, yd * R_MM, phase, cmap=ListedColormap(['#7fbf7b', 'white', '#1f4e9c']),
                       norm=BoundaryNorm([-0.5, 0.5, 1.5, 2.5], 3), edgecolors='none')
    axes[0].plot(np.array([0, 1, 1, 0, 0]) * domain_size[0] * R_MM, np.array([0, 0, 1, 1, 0]) * domain_size[1] * R_MM,
                 'k--', lw=0.8)
    axes[0].set(aspect='equal', xlabel='x [mm]', ylabel='y [mm]',
                title=f'deformed cell at eps_{args.direction} = {rows[-1, 0]:.3f} (dashed: undeformed)')
    axes[1].plot(rows[:, 0], rows[:, 1], 'o-', color='tab:green', label='muFFTTO, periodic, finite strain')
    paper = {'X': 0.35, 'Y': 0.50}[args.direction]
    axes[1].plot(rows[:, 0], paper * rows[:, 0], 'k--', lw=1, label=f'paper initial slope {paper} MPa (digitised)')
    axes[1].set(xlabel=f'nominal strain $\\varepsilon_{args.direction}$', ylabel='nominal stress [MPa]',
                title=f'uniaxial compression in {args.direction}, lateral side free')
    axes[1].legend(frameon=False)
    axes[1].grid(alpha=0.3)
    os.makedirs(os.path.join(here, 'figures'), exist_ok=True)
    fig.savefig(os.path.join(here, 'figures', f'finite_strain_TR_{tag}.png'), dpi=110)
