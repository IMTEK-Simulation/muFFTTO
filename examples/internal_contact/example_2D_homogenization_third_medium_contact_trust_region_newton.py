import time
import os
import sys
import argparse

import numpy as np
from mpi4py import MPI
from NuMPI.IO import save_npy
from matplotlib import pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from muFFTTO import domain, tensor_operations
from muFFTTO import solvers
from muFFTTO import microstructure_library
from muFFTTO import material_models
from muFFTTO import visualization_utils

# ============================================================================
# argument parsing
# ============================================================================
parser = argparse.ArgumentParser(
    prog='exp_finite_strain_2D_NeoHookean.py',
    description='Solve finite strain NeoHookean elasticity in 2D'
)
parser.add_argument('-n', '--nb_pixel', default='64')
parser.add_argument('-inc', '--nb_increments', default='100')
parser.add_argument('--save_per_it', action='store_true')

script_name = os.path.splitext(os.path.basename(__file__))[0]
args = parser.parse_args()
nnn = int(args.nb_pixel)
ninc = int(args.nb_increments)
save_per_it = args.save_per_it

# ============================================================================
# problem setup
# ============================================================================
number_of_pixels = (nnn, nnn)
domain_size = [1, 1]
dim = len(domain_size)

problem_type = 'elasticity'
discretization_type = 'finite_element'
element_type = 'bilinear_rectangle'
formulation = 'finite_strain'
preconditioner_type = "Green"

_info = {
    'problem_type': problem_type,
    'discretization_type': discretization_type,
    'element_type': element_type,
    'formulation': formulation,
    'preconditioner_type': preconditioner_type,
    'nb_of_pixels': number_of_pixels,
    'domain_size': domain_size,
    'ninc': ninc,
}

my_cell = domain.PeriodicUnitCell(domain_size=domain_size,
                                  problem_type=problem_type)

discretization = domain.Discretization(cell=my_cell,
                                       nb_of_pixels_global=number_of_pixels,
                                       discretization_type=discretization_type,
                                       element_type=element_type)

comm = discretization.communicator
rank0 = (comm.rank == 0)

file_folder_path = os.path.dirname(os.path.realpath(__file__))
data_folder_path = (file_folder_path + '/exp_data/' + script_name + '/'
                    + f'Nx={nnn}Ny={nnn}_{preconditioner_type}/')
figure_folder_path = (file_folder_path + '/figures/' + script_name + '/'
                      + f'Nx={nnn}Ny={nnn}_{preconditioner_type}/')
if rank0:
    os.makedirs(data_folder_path, exist_ok=True)
    os.makedirs(figure_folder_path, exist_ok=True)

# ============================================================================
# material parameters
# ============================================================================
E_matrix = 100.0
nu_matrix = 0.3
lam_matrix = E_matrix * nu_matrix / ((1 + nu_matrix) * (1 - 2 * nu_matrix))
mu_matrix = E_matrix / (2 * (1 + nu_matrix))
K, G = material_models.get_bulk_and_shear_modulus(E_matrix, nu_matrix)

k_v = 1e-5
E_void = k_v * E_matrix
nu_void = nu_matrix
lam_void = E_void * nu_void / ((1 + nu_void) * (1 - 2 * nu_void))
mu_void = E_void / (2 * (1 + nu_void))

alpha = 1e-6
k_r = alpha * domain_size[0] ** 2 * (K + G * 4 / 3)

i = np.eye(dim)
II = np.einsum('ij,kl->ijkl', i, i)
I4rt = np.einsum('ik,jl->ijkl', i, i)
I4s = 0.5 * (I4rt + np.einsum('il,jk->ijkl', i, i))
ref_mat = lam_matrix * II + 2.0 * mu_matrix * I4s

_info['lam_matrix'] = lam_matrix
_info['mu_matrix'] = mu_matrix
_info['lam_inc'] = lam_void
_info['mu_inc'] = mu_void

# ============================================================================
# geometry
# ============================================================================
phase_field = discretization.get_scalar_field(name='phase_field')
phase_field.s[0, 0] = microstructure_library.get_geometry(
    nb_voxels=discretization.nb_of_pixels,
    microstructure_name='contact_test_geometry_2',
    coordinates=discretization.fft.coords
)
matrix_mask = phase_field.s[0, 0] > 0
inc_mask = phase_field.s[0, 0] == 0

lam_field = discretization.get_quad_field_scalar(name='lam_field')
mu_field = discretization.get_quad_field_scalar(name='mu_field')
lam_field.s[0, 0, :, matrix_mask] = lam_matrix
lam_field.s[0, 0, :, inc_mask] = lam_void
mu_field.s[0, 0, :, matrix_mask] = mu_matrix
mu_field.s[0, 0, :, inc_mask] = mu_void

if rank0:
    print(f'lam min/max : {lam_field.s.min():.4f} / {lam_field.s.max():.4f}')
    print(f'mu  min/max : {mu_field.s.min():.4f}  / {mu_field.s.max():.4f}')

material = material_models.NeoHookean(
    discretization=discretization,
    lam_1qxyz=lam_field,
    mu_1qxyz=mu_field,
    name='neo_hookean_two_phase'
)

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
rhs_field = discretization.get_unknown_size_field(name='rhs_field')
energy_field = discretization.get_quad_field_scalar(name='energy_field')
hess_u_ijkqxyz = discretization.get_displacement_hessian_size_field(name='hess_u')
HtH_field = discretization.get_unknown_size_field(name='HtH')

# --- NEW: scratch fields for the trust region -------------------------------
J_1qxyz = discretization.get_quad_field_scalar(name='J')
F_trial = discretization.get_displacement_gradient_sized_field(name='F_trial')
F_save = discretization.get_displacement_gradient_sized_field(name='F_save')
u_trial = discretization.get_unknown_size_field(name='u_trial')
u_save = discretization.get_unknown_size_field(name='u_save')
Kdu_field = discretization.get_unknown_size_field(name='Kdu')

total_strain_field.s[...] = 0.0
for d in range(dim):
    total_strain_field.s[d, d] = 1.0

preconditioner = discretization.get_preconditioner_Green_mugrid(
    reference_material_data_ijkl=ref_mat
)


def M_fun(x, Px):
    discretization.fft.communicate_ghosts(x)
    discretization.apply_preconditioner_mugrid(
        preconditioner_Fourier_fnfnqks=preconditioner,
        input_nodal_field_fnxyz=x,
        output_nodal_field_fnxyz=Px
    )


# ============================================================================
# loading
# NOTE: the original printed / plotted component [1, 0] while loading [0, 0],
#       which is why every increment showed "Load 0.0" and response.png was a
#       plot of zeros.  LOAD_IJ keeps driving and reporting in sync.
# ============================================================================
LOAD_IJ = (0, 0)
macro_gradient_inc = np.zeros((dim, dim))
macro_gradient_inc[LOAD_IJ] += -0.3 / float(ninc)

discretization.get_macro_gradient_field_mugrid(
    macro_gradient_ij=macro_gradient_inc,
    macro_gradient_field_ijqxyz=macro_gradient_inc_field
)

# NEW: the FULL macro gradient, so that F can be rebuilt from (lam, u) at any
# load level.  Accumulating F increment by increment makes it impossible to
# halve a load step or roll back after a failure.
macro_gradient_full = macro_gradient_inc * float(ninc)
macro_gradient_full_field = discretization.get_gradient_size_field(name='macro_full')
discretization.get_macro_gradient_field_mugrid(
    macro_gradient_ij=macro_gradient_full,
    macro_gradient_field_ijqxyz=macro_gradient_full_field
)

_info['macro_gradient_inc'] = macro_gradient_inc
_info['norm_strain_fluc_field'] = []
_info['mean_stress_field'] = []
_info['total_macro_gradient'] = []

material.get_stress(total_strain_field, stress_field)
material.get_algorithmic_tangent(total_strain_field, tangent_field)
if rank0:
    print(f'tangent min/max : {tangent_field.s.min():.4f} / {tangent_field.s.max():.4f}')


# ============================================================================
# NEW: energy, residual and operator helpers
# ============================================================================
def dot(a, b):
    return comm.sum(float(np.dot(a.s.ravel(), b.s.ravel())))


def nrm(a):
    return np.sqrt(dot(a, a))


def set_F(lam, u_field, out):
    """F = I + lam * H_macro + grad(u~).  Rebuilt from scratch, never
    accumulated, so that a load step can be halved or rolled back."""
    discretization.apply_gradient_operator_mugrid(u_inxyz=u_field,
                                                  grad_u_ijqxyz=strain_fluc_field)
    out.s[...] = strain_fluc_field.s + lam * macro_gradient_full_field.s
    for d in range(dim):
        out.s[d, d] += 1.0


def feasible_dlam(lam0, u0, du_pred, dlam_ref, dlam_want, tau=0.5,
                  n_scan=16, n_bisect=40):
    """Largest load increment whose PREDICTOR state is still admissible.

    Why this is needed at all: the macro increment is a uniform gradient, so it
    imposes a displacement Delta_F_00 * x_1 on every node.  The stiff phase
    hardly compresses, so the fluctuation must return |Delta_F_00| * w over each
    block of width w, and ALL of it is absorbed by the thin medium layer.  The
    thickness loss per increment is therefore ~ |Delta_F_00| * w, INDEPENDENT of
    the current thickness t, while J ~ t/t0.  A constant absolute decrement
    against a quantity going to zero crosses zero in finite load.

    Scaling dlam on the converged drop in J does not fix this: the converged
    drop is small because the barrier resists, whereas the predictor -- which
    has no barrier resistance -- follows the constant-decrement line.  The cap
    must be computed at the predictor.

    With u frozen or secant-extrapolated, F is affine in dlam, so det F is a
    low-degree polynomial in dlam and the first root follows from scan +
    bisection.  This is the load-step analogue of the fraction-to-the-boundary
    rule already applied to the Newton step.
    """
    scratch_u = discretization.get_unknown_size_field(name='u_dlam_scratch')

    def g(dl):
        scratch_u.s[...] = u0.s + (dl / dlam_ref) * du_pred.s
        set_F(lam0 + dl, scratch_u, F_trial)
        return min_J(F_trial)

    if g(dlam_want) > 0.0:
        return dlam_want

    lo, hi = 0.0, dlam_want
    for k in range(1, n_scan + 1):
        d = dlam_want * k / n_scan
        if g(d) <= 0.0:
            lo, hi = dlam_want * (k - 1) / n_scan, d
            break
    for _ in range(n_bisect):
        mid = 0.5 * (lo + hi)
        if g(mid) > 0.0:
            lo = mid
        else:
            hi = mid
        if hi - lo < 1e-16:
            break
    return tau * lo


def min_J(F_in):
    tensor_operations.det2(F_in, J_1qxyz)
    return MPI.COMM_WORLD.allreduce(float(np.min(J_1qxyz.s)), op=MPI.MIN)


def apply_HtH(u_field, out):
    discretization.apply_hessian_operator_to_vector_field_mugrid(
        u_inxyz=u_field, hess_u_ijkqxyz=hess_u_ijkqxyz)
    discretization.apply_hessian_operator_transposed_to_vector_field_mugrid(
        hess_u_ijkqxyz=hess_u_ijkqxyz, nodal_field_inxyz=out, apply_weights=True)


def assemble_rhs(F_in, u_field):
    """rhs = -dPi/du.  Unlike the original, the HuHu term is ALWAYS included,
    so norm(rhs) at the start of an increment and inside the Newton loop
    measure the same functional.  Also refreshes stress and tangent."""
    material.get_stress(F_in, stress_field)
    material.get_algorithmic_tangent(F_in, tangent_field)
    discretization.fft.communicate_ghosts(stress_field)
    discretization.apply_gradient_transposed_operator_mugrid(
        gradient_field_ijqxyz=stress_field, div_u_fnxyz=rhs_field, apply_weights=True)
    rhs_field.s[...] *= -1
    apply_HtH(u_field, HtH_field)
    rhs_field.s[...] -= k_r * HtH_field.s
    return nrm(rhs_field)


def K_fun(x, Ax):
    discretization.apply_system_matrix_mugrid(
        material_data_field=tangent_field, input_field_inxyz=x,
        output_field_inxyz=Ax, formulation=formulation)
    discretization.apply_hessian_operator_to_vector_field_mugrid(
        u_inxyz=x, hess_u_ijkqxyz=hess_u_ijkqxyz)
    discretization.apply_hessian_operator_transposed_to_vector_field_mugrid(
        hess_u_ijkqxyz=hess_u_ijkqxyz, nodal_field_inxyz=HtH_field, apply_weights=True)
    Ax.s[...] += k_r * HtH_field.s
    discretization.fft.communicate_ghosts(Ax)


W_MAT = float(np.prod(domain_size)) / (energy_field.s.shape[2]
                                       * float(np.prod(number_of_pixels)))


def total_energy(F_in, u_field):
    """Neo-Hookean + HuHu.  Returns +inf on an inadmissible state, so a step
    that would take det F through zero is rejected by rho < 0 and never
    reaches log(J)."""
    if min_J(F_in) <= 0.0:
        return np.inf
    material.get_energy_density(F_in, energy_field)
    E = W_MAT * comm.sum(float(energy_field.s.sum()))
    apply_HtH(u_field, HtH_field)
    E += 0.5 * k_r * dot(u_field, HtH_field)
    return E if np.isfinite(E) else np.inf


# --- one-time calibration of W_MAT against the discrete gradient ------------
# rho is only meaningful if the energy and the residual are the same
# functional.  This checks it and rescales W_MAT if the quadrature weight in
# apply_gradient_transposed_operator_mugrid differs from the guess above.
_rng = np.random.default_rng(0)
u_trial.s[...] = 1e-3 * _rng.standard_normal(u_trial.s.shape)
discretization.apply_gradient_operator_mugrid(u_inxyz=u_trial, grad_u_ijqxyz=strain_fluc_field)
F_trial.s[...] = strain_fluc_field.s
for d in range(dim):
    F_trial.s[d, d] += 1.0
material.get_stress(F_trial, stress_field)
discretization.fft.communicate_ghosts(stress_field)
discretization.apply_gradient_transposed_operator_mugrid(
    gradient_field_ijqxyz=stress_field, div_u_fnxyz=HtH_field, apply_weights=True)
_an = dot(HtH_field, u_trial)                       # dE_mat/du . u
_h = 1e-6
_E = []
for _s in (+1.0, -1.0):
    displacement_increment_field.s[...] = (1.0 + _s * _h) * u_trial.s
    discretization.apply_gradient_operator_mugrid(
        u_inxyz=displacement_increment_field, grad_u_ijqxyz=strain_fluc_field)
    F_trial.s[...] = strain_fluc_field.s
    for d in range(dim):
        F_trial.s[d, d] += 1.0
    material.get_energy_density(F_trial, energy_field)
    _E.append(comm.sum(float(energy_field.s.sum())))
_fd = (_E[0] - _E[1]) / (2 * _h)                    # d/ds E_raw(s*u) at s=1
if abs(_fd) > 0:
    W_MAT = _an / _fd
if rank0:
    print(f'energy quadrature weight calibrated: W_MAT = {W_MAT:.6e}')
u_trial.s.fill(0.0)
displacement_increment_field.s.fill(0.0)

# ============================================================================
# trust-region parameters
# ============================================================================
TR_ETA = 1e-4        # accept if rho > TR_ETA
TR_SHRINK = 0.25
TR_EXPAND = 2.0
TR_MAX = np.inf
TOL_REL = 1e-6       # relative to the predictor residual of THIS increment
TOL_ABS_FRAC = 1e-6  # ... but never tighter than this fraction of RHS_SCALE.
                     # Needed because a good predictor makes norm_rhs_0 tiny,
                     # and a purely relative test would then demand a residual
                     # below what CG and the energy sum can resolve.
RHS_SCALE = None     # fixed once, at the first increment
MAX_NEWTON = 30
CG_TOL = 1e-6        # FIXED, and tighter than the original 1e-4.  See below:
                     # muFFTTO's CG reports itself converged at iteration 0 if
                     # this is loose, which yields du = 0 and stalls the outer
                     # loop.  Do not make it adaptive.

Delta = None         # set from the first Newton step; carried across increments

# --- NEW: adaptive load stepping -------------------------------------------
DLAM_MIN = 1e-9
DLAM_MAX = 4.0 / ninc
J_DROP_TARGET = 0.25     # allow at most a 25% drop in min J per increment
MAX_CUTS = 14

# ============================================================================
# incremental loading
# ============================================================================
sum_CG_its = 0
sum_Newton_its = 0
iteration_total = 0
start_time = time.time()

lam = 0.0
dlam = 1.0 / ninc
dlam_prev = dlam
set_F(lam, displacement_fluctuation_field, total_strain_field)
J_prev = min_J(total_strain_field)

u_backup = discretization.get_unknown_size_field(name='u_backup')
du_secant = discretization.get_unknown_size_field(name='du_secant')   # last increment's Du
du_secant.s.fill(0.0)

inc = -1
while lam < 1.0 - 1e-12:
    inc += 1
    dlam = min(dlam, 1.0 - lam)
    u_backup.s[...] = displacement_fluctuation_field.s
    lam_safe, J_safe = lam, J_prev
    n_cuts = 0

    while True:
        # --- CAP THE LOAD INCREMENT BY PREDICTOR FEASIBILITY ------------
        # This replaces the reactive "try, fail, quarter" loop.  As the medium
        # closes, dlam_max shrinks in proportion to the remaining thickness,
        # which is the correct asymptotics.
        dlam = feasible_dlam(lam_safe, u_backup, du_secant, dlam_prev, dlam,
                             tau=0.5)
        if dlam < DLAM_MIN:
            n_cuts = MAX_CUTS + 1
            break

        lam_try = lam_safe + dlam
        if rank0:
            print(f'\nIncrement {inc}   '
                  f'F_{LOAD_IJ[0]}{LOAD_IJ[1]} = '
                  f'{lam_try * macro_gradient_full[LOAD_IJ]:+.5f}'
                  f'   (dlam = {dlam:.3e})')
            print('=' * 70)

        # --- SECANT PREDICTOR ------------------------------------------
        # Freezing the fluctuation while adding the macro increment forces the
        # medium to absorb the whole block compression.  Extrapolating the
        # fluctuation from the previous increment supplies most of that
        # redistribution in advance.
        displacement_fluctuation_field.s[...] = (
            u_backup.s + (dlam / dlam_prev) * du_secant.s)
        set_F(lam_try, displacement_fluctuation_field, total_strain_field)

        J_pred = min_J(total_strain_field)
        if J_pred <= 0.0:
            # should not happen after the cap, but guard anyway
            n_cuts += 1
            if rank0:
                print(f'  predictor inadmissible (min J = {J_pred:.3e});'
                      f' cutback {n_cuts}/{MAX_CUTS}')
            dlam *= 0.25
            if n_cuts > MAX_CUTS or dlam < DLAM_MIN:
                break
            continue

        norm_rhs_0 = assemble_rhs(total_strain_field, displacement_fluctuation_field)
        E_cur = total_energy(total_strain_field, displacement_fluctuation_field)

        if not np.isfinite(norm_rhs_0) or not np.isfinite(E_cur):
            n_cuts += 1
            if rank0:
                print(f'  non-finite predictor state; cutback {n_cuts}/{MAX_CUTS}')
            dlam *= 0.25
            if n_cuts > MAX_CUTS or dlam < DLAM_MIN:
                break
            continue

        if RHS_SCALE is None:
            RHS_SCALE = norm_rhs_0        # fixed reference force scale

        if rank0:
            print(f'Rhs at new load step    {norm_rhs_0:10.2e}   '
                  f'(predictor min J = {J_pred:.3e})')
            print(f'En  at new load step    {E_cur:10.4e}')
        newton_ok = True
        break

    tol_newton = max(TOL_REL * norm_rhs_0, TOL_ABS_FRAC * RHS_SCALE)
    stall = 0
    roundoff_exit = False
    if n_cuts > MAX_CUTS or dlam < DLAM_MIN:
        if rank0:
            print(f'\nStopping at lam = {lam_safe:.6f} '
                  f'(F_{LOAD_IJ[0]}{LOAD_IJ[1]} = '
                  f'{lam_safe * macro_gradient_full[LOAD_IJ]:+.5f}), '
                  f'min J = {J_safe:.3e}.\n'
                  f'The third medium is essentially fully compressed here. '
                  f'To go further, raise --alpha_huhu or accept a smaller J.')
        displacement_fluctuation_field.s[...] = u_backup.s
        set_F(lam_safe, displacement_fluctuation_field, total_strain_field)
        break

    # ------------------------------------------------------------------
    # trust-region Newton
    # ------------------------------------------------------------------
    for iiter in range(1, MAX_NEWTON + 1):
        norm_rhs = nrm(rhs_field)
        if norm_rhs < tol_newton:
            break

        # --- linear solve --------------------------------------------------
        # A zero-iteration return is a FAILED solve, not a converged one:
        # muFFTTO's CG satisfies its internal test at x = 0 when tol is loose.
        # Retry tighter before concluding anything about the Hessian.
        tol_try = CG_TOL
        for _attempt in range(3):
            cg_count = {'n': 0}

            def callback(it, x, r, p, z, stop_crit_norm):
                cg_count['n'] += 1

            displacement_increment_field.s.fill(0)
            solvers.conjugate_gradients_mugrid(
                comm=comm,
                fc=discretization.field_collection,
                hessp=K_fun,
                b=rhs_field,
                x=displacement_increment_field,
                P=M_fun,
                tol=tol_try,
                maxiter=4000,
                callback=callback,
                rtol=True,
            )
            dnorm = nrm(displacement_increment_field)
            if cg_count['n'] > 0 and dnorm > 0.0:
                break
            tol_try *= 1e-2

        nb_it_cg = cg_count['n']
        sum_CG_its += nb_it_cg
        sum_Newton_its += 1
        iteration_total += 1

        if dnorm == 0.0 or not np.isfinite(dnorm):
            if rank0:
                print(f'  it {iiter:2d} | CG returned a zero step even at tol '
                      f'{tol_try:.1e}; stopping increment at '
                      f'|rhs|/|rhs_0| = {norm_rhs / norm_rhs_0:.2e}')
            break

        # --- quadratic model data: one extra matvec ------------------------
        # rhs = -g, so descent needs g.du < 0  <=>  rhs.du > 0.
        # dnorm > 0 is already guaranteed above, so dKd <= 0 here really does
        # mean indefinite curvature rather than a null step.
        K_fun(displacement_increment_field, Kdu_field)
        gdu = -dot(rhs_field, displacement_increment_field)
        dKd = dot(displacement_increment_field, Kdu_field)

        neg_curv = (dKd <= 0.0)
        if neg_curv or gdu >= 0.0:
            # Indefinite tangent (the medium is buckling) or CG returned a
            # non-descent direction: fall back to preconditioned steepest
            # descent, which the trust region can always make progress on.
            M_fun(rhs_field, displacement_increment_field)
            K_fun(displacement_increment_field, Kdu_field)
            gdu = -dot(rhs_field, displacement_increment_field)
            dKd = dot(displacement_increment_field, Kdu_field)
            dnorm = nrm(displacement_increment_field)
            if dnorm == 0.0 or not np.isfinite(dnorm):
                if rank0:
                    print('  null steepest-descent direction; stopping increment')
                break

        if Delta is None:
            Delta = dnorm                      # first step is unconstrained

        # --- truncate to the trust region --------------------------------
        scale = min(1.0, Delta / dnorm)
        if scale < 1.0:
            displacement_increment_field.s[...] *= scale
            gdu *= scale
            dKd *= scale * scale
            dnorm = Delta
        on_boundary = (scale < 1.0)

        pred = -(gdu + 0.5 * dKd)

        # The energy is a sum over N_q*N_p terms, so it cannot resolve
        # differences below ~eps*N*|E| (~1e-12 here).  Once the predicted
        # decrease falls below that, rho is noise.  Do NOT stop -- the residual
        # is still perfectly well resolved at 1e-6, so switch the merit
        # function to ||rhs|| for this step.  (Stopping here contradicted the
        # increment acceptance test and produced an infinite cutback loop.)
        use_res_merit = (pred <= 1e-11 * max(abs(E_cur), 1e-300))

        # --- trial state ---------------------------------------------------
        u_trial.s[...] = displacement_fluctuation_field.s + displacement_increment_field.s
        discretization.apply_gradient_operator_mugrid(
            u_inxyz=displacement_increment_field, grad_u_ijqxyz=strain_fluc_field)
        F_trial.s[...] = total_strain_field.s + strain_fluc_field.s

        E_try = total_energy(F_trial, u_trial)

        if use_res_merit:
            # ---- residual merit: tentatively take the step, judge on ||rhs||
            u_save.s[...] = displacement_fluctuation_field.s
            F_save.s[...] = total_strain_field.s
            displacement_fluctuation_field.s[...] = u_trial.s
            total_strain_field.s[...] = F_trial.s
            new_rhs = assemble_rhs(total_strain_field, displacement_fluctuation_field)
            if (np.isfinite(new_rhs) and new_rhs < norm_rhs
                    and min_J(total_strain_field) > 0.0):
                if np.isfinite(E_try):
                    E_cur = E_try
                norm_rhs = new_rhs
                rho = 1.0
                accepted = 'accept*'
            else:
                displacement_fluctuation_field.s[...] = u_save.s
                total_strain_field.s[...] = F_save.s
                assemble_rhs(total_strain_field, displacement_fluctuation_field)
                rho = -1.0
                accepted = 'REJECT*'
        else:
            rho = (E_cur - E_try) / pred if (np.isfinite(E_try) and pred > 0) else -1.0
            if rho > TR_ETA:
                displacement_fluctuation_field.s[...] = u_trial.s
                total_strain_field.s[...] = F_trial.s
                E_cur = E_try
                norm_rhs = assemble_rhs(total_strain_field,
                                        displacement_fluctuation_field)
                accepted = 'accept'
            else:
                accepted = 'REJECT'

        # --- radius update -------------------------------------------------
        # The expansion no longer requires on_boundary.  With a good predictor
        # the Newton step is almost always well inside the region, so the old
        # rule let Delta ratchet down on every cutback and never recover; by
        # the second half of the run it was truncating full Newton steps to a
        # 2x residual reduction instead of 1e-5.
        if rho < 0.25:
            Delta = TR_SHRINK * dnorm
        elif rho > 0.75:
            Delta = min(max(Delta, TR_EXPAND * dnorm), TR_MAX)

        _J = min_J(total_strain_field)
        _dn = nrm(strain_fluc_field)
        _info['norm_strain_fluc_field'].append(_dn)

        if rank0:
            flag = '  [neg curvature -> buckling]' if neg_curv else ''
            print(f'  it {iiter:2d} | CG {nb_it_cg:5d} | {accepted} '
                  f'| rho {rho:+8.3f} | Delta {Delta:9.3e} '
                  f'| |rhs|/|rhs_0| {norm_rhs / norm_rhs_0:9.2e} '
                  f'| J {_J:9.3e}{flag}')

        # stagnation: the residual merit could not improve either
        if accepted == 'REJECT*':
            stall += 1
            if stall >= 2:
                if rank0:
                    print('  no further residual decrease attainable; '
                          'accepting increment')
                roundoff_exit = True
                break
        else:
            stall = 0

        if Delta < 1e-14:
            if rank0:
                print('  trust region collapsed; stopping increment')
            break

        if save_per_it and accepted == 'accept':
            save_npy(data_folder_path + f'displacement_fluctuation_field_it{iteration_total}.npy',
                     displacement_fluctuation_field.s.mean(axis=1),
                     tuple(discretization.subdomain_locations_no_buffers),
                     tuple(discretization.nb_of_pixels_global), MPI.COMM_WORLD)
            save_npy(data_folder_path + f'stress_field_it{iteration_total}.npy',
                     stress_field.s.mean(axis=2),
                     tuple(discretization.subdomain_locations_no_buffers),
                     tuple(discretization.nb_of_pixels_global), MPI.COMM_WORLD)

    # ------------------------------------------------------------------
    # increment accepted or not
    # ------------------------------------------------------------------
    J_new = min_J(total_strain_field)
    converged = (nrm(rhs_field) < tol_newton or roundoff_exit) \
        and np.isfinite(J_new) and J_new > 0.0

    if not converged:
        if rank0:
            print(f'  Newton failed at lam = {lam_try:.6f}; rolling back and '
                  f'cutting the load step')
        displacement_fluctuation_field.s[...] = u_backup.s
        set_F(lam_safe, displacement_fluctuation_field, total_strain_field)
        dlam *= 0.25
        Delta = max(0.25 * (Delta if Delta else 1.0), 1e-8)
        if dlam < DLAM_MIN:
            if rank0:
                print('  load increment underflow; stopping')
            break
        inc -= 1          # retry the same increment number
        continue

    # store the secant for the next predictor, then commit
    du_secant.s[...] = displacement_fluctuation_field.s - u_backup.s
    dlam_prev = dlam
    lam = lam_try

    _info['mean_stress_field'].append(float(stress_field.s[LOAD_IJ].mean()))
    total_macro_gradient = lam * macro_gradient_full
    _info['total_macro_gradient'].append(total_macro_gradient)

    dJ = J_safe - J_new
    if rank0:
        print(f'  ---> min J = {J_new:.6e}   (dJ = {dJ:+.3e})')

    # --- adaptive load increment ---------------------------------------
    # min J decays roughly geometrically once contact is established, so cap
    # the FRACTIONAL drop rather than the absolute one.  This makes dlam shrink
    # geometrically as J -> 0, which a fixed increment can never do.
    f_iter = np.sqrt(4.0 / max(iiter, 1))
    f_bar = (J_DROP_TARGET * J_new / abs(dJ)) if abs(dJ) > 1e-300 else 1.5
    dlam *= float(max(0.1, min(1.5, f_iter, f_bar)))
    dlam = float(min(dlam, DLAM_MAX))
    J_prev = J_new

    if comm.size == 1:
        x_plot_ixyz = visualization_utils.get_deformed_grid_coords_two_dim(
            discretization, macro_gradient_ij=total_macro_gradient,
            displacement_fluctuation=displacement_fluctuation_field)
        visualization_utils.plot_field_on_grid(
            coordinates_for_plot=x_plot_ixyz,
            field_to_plot=phase_field.s[0, 0],
            name=fr'$\tilde{{u}}_{{x}}$' + f'load increment {inc}   ')

# ============================================================================
# response curve
# ============================================================================
F = np.asarray(_info['total_macro_gradient'])[..., LOAD_IJ[0], LOAD_IJ[1]]
P = np.asarray(_info['mean_stress_field'])

fig, ax = plt.subplots(1, 1, figsize=(5, 5))
ax.plot(F, P, '-o', ms=3, color='k')
ax.set_xlabel(rf'$\bar F_{{{LOAD_IJ[0]}{LOAD_IJ[1]}}}$')
ax.set_ylabel(rf'$\bar P_{{{LOAD_IJ[0]}{LOAD_IJ[1]}}}$')
ax.grid(alpha=.3)
fig.tight_layout()
fig.savefig(figure_folder_path + 'response.png', dpi=150)
plt.show()

# ============================================================================
# summary
# ============================================================================
elapsed_time = time.time() - start_time
_info['sum_Newton_its'] = sum_Newton_its
_info['sum_CG_its'] = sum_CG_its
_info['iteration_total'] = iteration_total
_info['elapsed_time'] = elapsed_time

if rank0:
    print('=' * 70)
    print(f'element_type     : {element_type}')
    print(f'number_of_pixels : {number_of_pixels}')
    print(f'preconditioner   : {preconditioner_type}')
    print(f'Total CG its     : {sum_CG_its}')
    print(f'Total Newton its : {sum_Newton_its}')
    print(f'Elapsed time     : {elapsed_time:.2f} s  ({elapsed_time / 60:.2f} min)')
    np.savez(data_folder_path + 'info_log_final.npz', **_info)