"""
Topology optimization of a periodic unit cell in FINITE STRAIN (2D, plane strain).

This example combines
    examples/topology_optimization/example_2D_elasticity_TO.py          (phase field, SIMP, adjoint, L-BFGS-B)
    examples/homogenization/finite_strain_elasticity/...NeoHookean.py   (neo-Hookean cell problem, Newton-CG)

Problem
-------
Find a nodal phase field rho (1 = solid, 0 = void) such that the homogenized first Piola-Kirchhoff
stress of the cell, P_h(F_l; rho), matches a target P_t(F_l) for a few macroscopic deformation
gradients F_l = I + H_l (the load cases):

    min_rho  F(rho) = sum_l  w * |P_h^l - P_t^l|^2 / |P_t^l|^2                     (stress mismatch)
                     + eta int |grad rho|^2 dx + 1/eta int rho^2 (1 - rho)^2 dx     (phase-field term)
    s.t.     0 <= rho <= 1,   and for every load case the cell problem
             r_l(u_l, rho) = D^T W P(I + H_l + D u_l; rho) = 0                       (equilibrium)

Material: compressible neo-Hookean (muFFTTO.material_models.NeoHookean),
    W(F) = lam/2 ln(J)^2 + mu/2 (F:F - 2) - mu ln(J),    P = lam ln(J) F^-T + mu (F - F^-T).
SIMP interpolation of both Lame parameters with the same factor
    lam(rho) = s(rho) lam_1,   mu(rho) = s(rho) mu_1,   s(rho) = eps + (1 - eps) rho^p,
so the stress is simply P(F; rho) = s(rho) P_1(F) and dP/drho = s'(rho)/s(rho) P(F; rho).

Target: a homogeneous neo-Hookean material (lam_t, mu_t). For small H_l this reduces exactly to the
linear target of the small-strain example (same K_t, G_t), so the two examples can be compared.

Sensitivity (adjoint method, one adjoint solve per load case)
-------------------------------------------------------------
    K_T lambda_l = - df_l/du = - c_l D^T W A : (P_h^l - P_t^l),     c_l = 2 w / (|Omega| |P_t^l|^2)
    dF/drho_a    = sum_l N^T W [ s'/s ( c_l (P_h^l - P_t^l) + D lambda_l ) : P ]_a  +  d(phase-field term)/drho_a
with K_T = D^T W A D the tangent stiffness (A = dP/dF, symmetric for a hyperelastic material) at the
converged state. Check it with check_sensitivity.py (finite differences).

Solvers
-------
State problem: Newton's method with preconditioned CG for the Newton steps (inexact: the CG tolerance
follows the Newton residual), backtracking until all J = det F > 0 and the residual decreases. The
state of the previous design is the initial guess; if Newton fails from there, the load is applied
again from u = 0 in increments. Adjoint problem: CG with the tangent of the converged state.
Preconditioner: Green (reference = linearized solid) or Green-Jacobi (Jacobi scaling of the current
tangent). Optimizer: bound-constrained L-BFGS from NuMPI (as in the small-strain example).

Run:   python example_2D_finite_strain_TO.py                (serial; small grids)
       mpirun -n 4 python example_2D_finite_strain_TO.py    (parallel)
Output: data/<run name>/ (history and final design as NetCDF, figures), a new folder for every run.
"""
import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import numpy as np
from mpi4py import MPI

from muFFTTO import domain
from muFFTTO import solvers_nonlinear
from muFFTTO import solvers
from muFFTTO import material_models
from muFFTTO import topology_optimization
from muFFTTO import io_utils

# ============================================================================
# Parameters
# ============================================================================
# grid
domain_size = [1.0, 1.0]
nnn = 32                                    # pixels per direction
number_of_pixels = (nnn, nnn)
element_type = 'bilinear_rectangle'         # Q1, 4 quadrature points (as in the finite strain example)

# solid material (same moduli as the small-strain TO example: K = 1, G = 0.5, lam = K - 2/3 G)
K_solid, G_solid = 1.0, 0.5
soft_phase_exponent = 4                     # void = 10^-4 x solid
simp_p = 2                                  # SIMP exponent p

# target material: isotropic, nu_t = 0, G_t = 3/20 E_solid (the target of the small-strain example)
poisson_target = 0.0

# load cases: F_l = I + strain_magnitude * H_l (uniaxial x, uniaxial y, simple shear)
strain_magnitude = 0.10
load_directions = np.array([[[1.0, 0.0], [0.0, 0.0]],
                            [[0.0, 0.0], [0.0, 1.0]],
                            [[0.0, 1.0], [0.0, 0.0]]])

# objective
weight = 5.0                                # weight of the stress mismatch (per load case: weight / L)
double_well_depth = 1.0
eta_in_pixels = 1.0                         # phase-field interface width eta, in pixels

# solvers
preconditioner_type = 'Green_Jacobi'        # 'Green' or 'Green_Jacobi'
newton_rtol = 1e-8                          # Newton: |r| <= newton_rtol * |r(u = 0)|
newton_maxiter = 30
cg_forcing_max = 1e-1                       # inexact Newton: CG rtol = min(cg_forcing_max, sqrt(|r|/|r0|))
adjoint_rtol = 1e-8                         # adjoint CG: relative energy error
cg_delay = 10                               # CG stopping: minimal delay of the energy-error lower bound (PT)
cg_maxiter = 5000
min_load_fraction = 1.0 / 64                # smallest load increment of the fallback load stepping

# optimizer
lbfgs_maxiter = 300
lbfgs_gtol = 1e-4
lbfgs_xtol = 1e-4
random_seed = 0
plot_every = 10                             # design figure every n optimizer iterations (0 = never)

# ============================================================================
# Discretization
# ============================================================================
comm = MPI.COMM_WORLD
dim = len(domain_size)
cell = domain.PeriodicUnitCell(domain_size=domain_size, problem_type='elasticity')
discretization = domain.Discretization(cell=cell, nb_of_pixels_global=number_of_pixels,
                                       discretization_type='finite_element', element_type=element_type)
pixel_size = np.asarray(domain_size) / np.asarray(number_of_pixels)
cell_volume = float(np.prod(domain_size))
quad_weights = discretization.quadrature_weights      # [q], sum over the cell = cell volume
eta = eta_in_pixels * float(np.max(pixel_size))


def root_print(*args):
    if comm.rank == 0:
        print(*args, flush=True)


def global_dot(a, b):
    """Global scalar product of two fields (all MPI ranks)."""
    return discretization.communicator.sum(np.dot(a.s.ravel(), b.s.ravel()))


def global_norm(a):
    return np.sqrt(global_dot(a, a))


def global_min(value):
    return comm.allreduce(float(value), op=MPI.MIN)


def cell_average(field_ijqxyz):
    """(1/|Omega|) int field dx for a tensor field at the quadrature points -> [i, j]."""
    weighted = np.einsum('ijq...,q->ijq...', field_ijqxyz.s, quad_weights)
    return discretization.mpi_reduction.sum(weighted, axis=tuple(range(-dim - 1, 0))) / cell_volume


# ============================================================================
# Materials
# ============================================================================
lam_solid = K_solid - 2.0 / 3.0 * G_solid             # plane strain with the 3D relation (as get_elastic_material_tensor)
mu_solid = G_solid
eps_void = 10.0 ** (-soft_phase_exponent)


def simp(rho):
    """s(rho) = eps + (1 - eps) rho^p: stiffness factor of both Lame parameters."""
    return eps_void + (1.0 - eps_void) * rho ** simp_p


def simp_derivative(rho):
    """s'(rho) = (1 - eps) p rho^(p-1)."""
    return (1.0 - eps_void) * simp_p * rho ** (simp_p - 1)


def neo_hookean_stress(F, lam, mu):
    """P(F) of a homogeneous neo-Hookean material for a single 2x2 deformation gradient (the target)."""
    F_inv_T = np.linalg.inv(F).T
    return lam * np.log(np.linalg.det(F)) * F_inv_T + mu * (F - F_inv_T)


# target: E_t = 2 G_t (1 + nu_t), K_t from (E_t, nu_t), lam_t = K_t - 2/3 G_t
E_solid = 9.0 * K_solid * G_solid / (3.0 * K_solid + G_solid)
G_target = 3.0 / 20.0 * E_solid
E_target = 2.0 * G_target * (1.0 + poisson_target)
K_target, G_target = material_models.get_bulk_and_shear_modulus(E=E_target, poisson=poisson_target)
lam_target, mu_target = K_target - 2.0 / 3.0 * G_target, G_target

nb_load_cases = len(load_directions)
macro_gradients = strain_magnitude * load_directions                      # H_l = F_l - I
target_stresses = np.array([neo_hookean_stress(np.eye(dim) + H, lam_target, mu_target) for H in macro_gradients])
weight_per_case = weight / nb_load_cases

# ============================================================================
# Fields
# ============================================================================
rho_nodal = discretization.get_scalar_field(name='rho')                   # design variable (nodal)
rho_quad = discretization.get_quad_field_scalar(name='rho_quad')          # rho at the quadrature points
lam_field = discretization.get_quad_field_scalar(name='lam_field')        # lam(rho) at the quadrature points
mu_field = discretization.get_quad_field_scalar(name='mu_field')          # mu(rho)
dlog_s_field = discretization.get_quad_field_scalar(name='dlog_s')        # s'(rho) / s(rho)
material = material_models.NeoHookean(discretization=discretization, lam_1qxyz=lam_field, mu_1qxyz=mu_field)

macro_gradient_field = discretization.get_gradient_size_field(name='macro_gradient')    # H_l at all quad points
F_field = discretization.get_displacement_gradient_sized_field(name='F')
P_field = discretization.get_displacement_gradient_sized_field(name='P')
A_field = discretization.get_material_data_size_field_mugrid(name='A')                 # tangent dP/dF
work_quad = discretization.get_displacement_gradient_sized_field(name='work_quad')
scalar_quad = discretization.get_quad_field_scalar(name='scalar_quad')
scalar_nodal = discretization.get_scalar_field(name='scalar_nodal')

residual = discretization.get_unknown_size_field(name='residual')
newton_step = discretization.get_unknown_size_field(name='newton_step')
u_trial = discretization.get_unknown_size_field(name='u_trial')
adjoint_rhs = discretization.get_unknown_size_field(name='adjoint_rhs')
jacobi_work = discretization.get_unknown_size_field(name='jacobi_work')
sensitivity = discretization.get_scalar_field(name='sensitivity')

# state and adjoint of every load case; kept between objective evaluations (initial guesses)
displacements = [discretization.get_unknown_size_field(name=f'u_{l}') for l in range(nb_load_cases)]
adjoints = [discretization.get_unknown_size_field(name=f'adjoint_{l}') for l in range(nb_load_cases)]

# ============================================================================
# Preconditioner
# ============================================================================
# Green: the inverse of the system matrix of the homogeneous, linearized solid (FFT-diagonal).
reference_tangent = material_models.get_elastic_material_tensor(dim=dim, K=K_solid, mu=G_solid)
green_operator = discretization.get_preconditioner_Green_mugrid(reference_material_data_ijkl=reference_tangent)
jacobi_diagonal = None                      # diag(K_T)^-1/2, updated with the tangent (Green_Jacobi only)


def apply_preconditioner(x, Px):
    """Px = M^-1 x: Green, or Green-Jacobi M^-1 = J G J with J = diag(K_T)^-1/2."""
    discretization.fft.communicate_ghosts(x)
    if preconditioner_type == 'Green':
        discretization.apply_preconditioner_mugrid(preconditioner_Fourier_fnfnqks=green_operator,
                                                   input_nodal_field_fnxyz=x, output_nodal_field_fnxyz=Px)
        return
    jacobi_work.s[...] = jacobi_diagonal.s * x.s
    discretization.apply_preconditioner_mugrid(preconditioner_Fourier_fnfnqks=green_operator,
                                               input_nodal_field_fnxyz=jacobi_work, output_nodal_field_fnxyz=Px)
    Px.s[...] *= jacobi_diagonal.s
    discretization.fft.communicate_ghosts(Px)


def apply_tangent(x, Ax):
    """Ax = K_T x = D^T W A D x with the current tangent A."""
    discretization.apply_system_matrix_mugrid(material_data_field=A_field, input_field_inxyz=x,
                                              output_field_inxyz=Ax, formulation='finite_strain')


def update_tangent():
    """A = dP/dF at the current F (and the Jacobi scaling of the new tangent)."""
    global jacobi_diagonal
    material.get_algorithmic_tangent(F_field, A_field)
    if preconditioner_type == 'Green_Jacobi':
        jacobi_diagonal = discretization.get_preconditioner_Jacobi_mugrid(material_data_field_ijklqxyz=A_field,
                                                                         formulation='finite_strain')


def solve_cg(rhs, x, rtol):
    """Solve K_T x = rhs with PCG, starting from x.
    Stopping: the Meurant-Papez-Tichy lower bound of the energy error |x - x_k|_K^2 (Hestenes-Stiefel sum
    with adaptive delay, at least cg_delay iterations; solvers.conjugate_gradients_mugrid_experimental)
    below rtol^2 times the estimate of the initial energy error |x - x_0|_K^2 (energy_rtol: relative, like a
    relative residual; computed from the CG scalars, no extra operator application).
    Returns the number of iterations."""
    counter = [0]

    def count(*args):
        counter[0] += 1

    solvers.conjugate_gradients_mugrid_experimental(
        comm=discretization.communicator, fc=discretization.field_collection, hessp=apply_tangent, b=rhs, x=x,
        P=apply_preconditioner, tol=rtol, maxiter=cg_maxiter, callback=count,
        stop_crit_norm='energy_lower_estim', d_min=cg_delay, energy_rtol=True)
    return max(counter[0] - 1, 0)           # called once before the first iteration (not at all if x is converged)


# ============================================================================
# State problem: equilibrium of the cell for one load case
# ============================================================================
def set_deformation_gradient(u, load_fraction):
    """F = I + t H + D u at all quadrature points; returns the global min of det F."""
    discretization.apply_gradient_operator_mugrid(u_inxyz=u, grad_u_ijqxyz=F_field)
    F_field.s[...] += load_fraction * macro_gradient_field.s
    for k in range(dim):
        F_field.s[k, k] += 1.0
    F = F_field.s
    return global_min(np.min(F[0, 0] * F[1, 1] - F[0, 1] * F[1, 0]))


def compute_residual(u, load_fraction):
    """Stress P and residual r = D^T W P at (u, t); returns |r|, or inf if some det F <= 0."""
    if set_deformation_gradient(u, load_fraction) <= 0.0:
        return np.inf
    material.get_stress(F_field, P_field)
    discretization.apply_gradient_transposed_operator_mugrid(gradient_field_ijqxyz=P_field,
                                                             div_u_fnxyz=residual, apply_weights=True)
    return global_norm(residual)


def newton(u, load_fraction, reference_norm):
    """Newton's method for r(u) = 0 at load fraction t, starting from u (updated in place).
    Returns (converged, Newton iterations, CG iterations)."""
    norm = compute_residual(u, load_fraction)
    nb_cg = 0
    for iteration in range(newton_maxiter):
        if not np.isfinite(norm):
            return False, iteration, nb_cg
        if norm <= newton_rtol * reference_norm:
            return True, iteration, nb_cg
        # Newton step: K_T du = -r, solved inexactly (tolerance follows the residual)
        update_tangent()
        residual.s[...] *= -1.0
        newton_step.s[...] = 0.0
        nb_cg += solve_cg(residual, newton_step, rtol=min(cg_forcing_max, np.sqrt(norm / reference_norm)))
        # backtracking: all det F > 0 and a smaller residual
        step = 1.0
        while step > 1e-4:
            u_trial.s[...] = u.s + step * newton_step.s
            trial_norm = compute_residual(u_trial, load_fraction)
            if trial_norm < norm:
                break
            step *= 0.5
        else:
            compute_residual(u, load_fraction)   # restore P and the residual of u
            return False, iteration + 1, nb_cg
        u.s[...] = u_trial.s
        norm = trial_norm
    return norm <= newton_rtol * reference_norm, newton_maxiter, nb_cg


def solve_state(load_case):
    """Equilibrium for load case l: F, P (and the tangent A) of the converged state are left in the fields.
    Returns (Newton iterations, CG iterations, number of load increments)."""
    u = displacements[load_case]
    discretization.get_macro_gradient_field_mugrid(macro_gradient_ij=macro_gradients[load_case],
                                                   macro_gradient_field_ijqxyz=macro_gradient_field)
    # scale of the residual: |r| of the affine deformation (u = 0) at the full load
    u_trial.s[...] = 0.0
    reference_norm = compute_residual(u_trial, 1.0)

    # 1. from the state of the previous design (usually close)
    converged, nb_newton, nb_cg = newton(u, 1.0, reference_norm)
    nb_increments = 1
    if not converged:
        # 2. fallback: apply the load again from u = 0 in increments (halved when Newton fails)
        u.s[...] = 0.0
        load_fraction, increment = 0.0, 0.25
        while load_fraction < 1.0 - 1e-12:
            target = min(1.0, load_fraction + increment)
            u_trial.s[...] = u.s
            ok, it, cg = newton(u_trial, target, reference_norm)
            nb_newton, nb_cg = nb_newton + it, nb_cg + cg
            if ok:
                u.s[...] = u_trial.s
                load_fraction, nb_increments = target, nb_increments + 1
            else:
                increment *= 0.5
                if increment < min_load_fraction:
                    raise RuntimeError(f'load case {load_case}: no equilibrium found (load fraction {load_fraction})')
        compute_residual(u, 1.0)
    update_tangent()                        # tangent of the converged state (adjoint problem)
    return nb_newton, nb_cg, nb_increments


# ============================================================================
# Design -> material
# ============================================================================
def set_design(rho_flat):
    """rho (flat, rank-local) -> nodal field, quadrature values, Lame fields and s'/s."""
    rho_nodal.s[...] = rho_flat.reshape(rho_nodal.s.shape)
    discretization.apply_N_operator_mugrid(rho_nodal, rho_quad)
    rho = rho_quad.s[0, 0]
    lam_field.s[0, 0] = simp(rho) * lam_solid
    mu_field.s[0, 0] = simp(rho) * mu_solid
    dlog_s_field.s[0, 0] = simp_derivative(rho) / simp(rho)


# ============================================================================
# Objective and its gradient
# ============================================================================
log = {'objective': [], 'mismatch': [], 'phase_field': [], 'newton': [], 'cg_state': [], 'cg_adjoint': [],
       'homogenized_stresses': []}


def objective(rho_flat):
    """F(rho) and dF/drho (rank-local flat arrays), for NuMPI's L-BFGS-B."""
    set_design(rho_flat)

    # phase-field term and its gradient (library functions of the small-strain example)
    f_phase = topology_optimization.objective_function_phase_field(discretization=discretization,
                                                                   phase_field_1nxyz=rho_nodal, eta=eta,
                                                                   double_well_depth=double_well_depth)
    topology_optimization.sensitivity_phase_field_term_FE_NEW(discretization=discretization,
                                                              phase_field_1nxyz=rho_nodal, p=simp_p, eta=eta,
                                                              output_array=sensitivity,
                                                              double_well_depth=double_well_depth)
    f_mismatch = 0.0
    nb_newton = nb_cg_state = nb_cg_adjoint = 0
    stresses = np.zeros((nb_load_cases, dim, dim))
    for l in range(nb_load_cases):
        # state: equilibrium, homogenized stress, mismatch
        it, cg, _ = solve_state(l)
        nb_newton, nb_cg_state = nb_newton + it, nb_cg_state + cg
        stresses[l] = cell_average(P_field)
        difference = stresses[l] - target_stresses[l]                     # P_h - P_t
        scale = weight_per_case / np.sum(target_stresses[l] ** 2)
        f_mismatch += scale * np.sum(difference ** 2)
        c = 2.0 * scale / cell_volume

        # adjoint: K_T lambda = -c D^T W A : (P_h - P_t)
        work_quad.s[...] = difference[(...,) + (np.newaxis,) * (work_quad.s.ndim - 2)]
        work_quad.s[...] = np.einsum('ijkl...,kl...->ij...', A_field.s, work_quad.s)
        discretization.apply_gradient_transposed_operator_mugrid(gradient_field_ijqxyz=work_quad,
                                                                 div_u_fnxyz=adjoint_rhs, apply_weights=True)
        adjoint_rhs.s[...] *= -c
        nb_cg_adjoint += solve_cg(adjoint_rhs, adjoints[l], rtol=adjoint_rtol)

        # sensitivity: N^T W [ s'/s (c (P_h - P_t) + D lambda) : P ]
        discretization.apply_gradient_operator_mugrid(u_inxyz=adjoints[l], grad_u_ijqxyz=work_quad)
        work_quad.s[...] += c * difference[(...,) + (np.newaxis,) * (work_quad.s.ndim - 2)]
        scalar_quad.s[0, 0] = dlog_s_field.s[0, 0] * np.einsum('ij...,ij...->...', work_quad.s, P_field.s)
        discretization.apply_N_transposed_operator_mugrid(quad_field_ijqxyz=scalar_quad,
                                                          nodal_field_inxyz=scalar_nodal, apply_weights=True)
        sensitivity.s[...] += scalar_nodal.s

    discretization.fft.communicate_ghosts(sensitivity)
    f = f_mismatch + f_phase
    for key, value in [('objective', f), ('mismatch', f_mismatch), ('phase_field', f_phase), ('newton', nb_newton),
                       ('cg_state', nb_cg_state), ('cg_adjoint', nb_cg_adjoint), ('homogenized_stresses', stresses)]:
        log[key].append(value)
    root_print(f'  F {f:.6e}  (mismatch {f_mismatch:.3e}, phase field {f_phase:.3e})  Newton {nb_newton:3d}  '
               f'CG state {nb_cg_state:5d}  CG adjoint {nb_cg_adjoint:5d}')
    return f, sensitivity.s.reshape(-1).copy()


# ============================================================================
# Post-processing helpers
# ============================================================================
def response_curve(load_case, fractions):
    """Homogenized stress of the current design along F = I + t H_l, t in fractions (increasing, from 0)."""
    discretization.get_macro_gradient_field_mugrid(macro_gradient_ij=macro_gradients[load_case],
                                                   macro_gradient_field_ijqxyz=macro_gradient_field)
    u = discretization.get_unknown_size_field(name='u_curve')
    u.s[...] = 0.0
    u_trial.s[...] = 0.0
    reference_norm = compute_residual(u_trial, max(fractions))
    stresses = []
    for t in fractions:
        converged = newton(u, t, reference_norm)[0]
        stresses.append(cell_average(P_field) if converged else np.full((dim, dim), np.nan))
    return np.array(stresses)


def save_figures(folder, tag):
    """Design (3 x 3 cells) and the deformed cell of every load case (serial runs only)."""
    if comm.size > 1:
        return
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    rho = rho_nodal.s[0, 0]
    fig, axes = plt.subplots(1, 1 + nb_load_cases, figsize=(4 * (1 + nb_load_cases), 4))
    axes[0].imshow(np.tile(rho, (3, 3)).T, origin='lower', cmap='Greys', vmin=0, vmax=1)
    axes[0].set_title(f'design ({tag}), 3 x 3 cells')
    nx, ny = rho.shape
    i, j = np.meshgrid(np.arange(nx + 1), np.arange(ny + 1), indexing='ij')
    X = np.array([i * pixel_size[0], j * pixel_size[1]])
    for l in range(nb_load_cases):
        u = displacements[l].s[:, 0][:, i % nx, j % ny]                  # periodic node values at the corners
        x = np.einsum('ab,b...->a...', np.eye(dim) + macro_gradients[l], X) + u
        axes[1 + l].pcolormesh(x[0], x[1], rho, cmap='Greys', vmin=0, vmax=1, shading='flat')
        axes[1 + l].set_aspect('equal')
        axes[1 + l].set_title(f'load case {l}: F = I + {np.round(macro_gradients[l], 3).tolist()}', fontsize=8)
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(os.path.join(folder, f'design_{tag}.png'), dpi=110)
    plt.close(fig)


# ============================================================================
# Driver
# ============================================================================
if __name__ == '__main__':
    root_print(f'grid {number_of_pixels}, {element_type}, eta {eta:.4f}, SIMP p {simp_p}, void {eps_void:g}, '
               f'preconditioner {preconditioner_type}, ranks {comm.size}')
    root_print(f'solid lam {lam_solid:.4f} mu {mu_solid:.4f};  target lam {lam_target:.4f} mu {mu_target:.4f}')
    for l in range(nb_load_cases):
        root_print(f'load case {l}: H = {macro_gradients[l].tolist()}, target P = {target_stresses[l].round(5).tolist()}')

    # output folder: a new one for every run
    run_name = (f'N{nnn}_{preconditioner_type}_strain{strain_magnitude:g}_w{weight:g}_p{simp_p}_'
                + time.strftime('%Y%m%d-%H%M%S'))
    folder = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data', run_name)
    if comm.rank == 0:
        os.makedirs(folder)
    comm.Barrier()
    parameters = {'nb_of_pixels': number_of_pixels, 'domain_size': domain_size, 'element_type': element_type,
                  'K_solid': K_solid, 'G_solid': G_solid, 'void': eps_void, 'simp_p': simp_p, 'eta': eta,
                  'weight': weight, 'double_well_depth': double_well_depth, 'strain_magnitude': strain_magnitude,
                  'macro_gradients': macro_gradients, 'target_stresses': target_stresses,
                  'lam_target': lam_target, 'mu_target': mu_target, 'preconditioner_type': preconditioner_type,
                  'newton_rtol': newton_rtol, 'adjoint_rtol': adjoint_rtol, 'random_seed': random_seed}
    root_print(f'output {folder}')

    # initial design: the smooth pattern of the small-strain example plus noise (independent of the ranks)
    coords = discretization.fft.coords
    rng = np.random.default_rng(random_seed)
    noise = rng.random(number_of_pixels)
    loc = tuple(slice(o, o + n) for o, n in zip(discretization.fft.subdomain_locations,
                                               discretization.fft.nb_subdomain_grid_pts))
    rho_0 = (np.sin(coords[0] * 4 * np.pi) + np.sin(coords[1] * 4 * np.pi) + 2) / 4 + 0.5 * noise[loc]
    rho_0 = np.clip(rho_0, 0.0, 1.0)

    # history: the design of every optimizer iteration
    rho_iterate = discretization.get_scalar_field(name='rho_iterate')
    history = io_utils.FieldWriter(os.path.join(folder, 'history.nc'), [rho_iterate], attributes=parameters,
                                   frame_variables={'objective': (), 'mismatch': ()})
    iteration = [0]

    def callback(rho_flat):
        iteration[0] += 1
        rho_iterate.s[...] = rho_flat.reshape(rho_iterate.s.shape)
        history.write(objective=log['objective'][-1], mismatch=log['mismatch'][-1])
        root_print(f'iteration {iteration[0]}')
        if plot_every and iteration[0] % plot_every == 0:
            save_figures(folder, f'it{iteration[0]:04d}')

    start = time.time()
    result = solvers_nonlinear.minimize(fun=objective, x0=rho_0.ravel(), method='l_bfgs', bounds=(0.0, 1.0),
                                        gtol=lbfgs_gtol, xtol=lbfgs_xtol,
                                        maxiter=lbfgs_maxiter, maxcor=20, c1=1e-4, max_halvings=40,
                                        comm=comm, callback=callback, disp=True)
    history.close()
    elapsed = time.time() - start

    # final design: state, figures, response curves against the target
    objective(result.x)
    save_figures(folder, 'final')
    fractions = np.linspace(0.0, 2.0, 21)                  # up to twice the optimized strain
    curves = np.array([response_curve(l, fractions) for l in range(nb_load_cases)])
    targets = np.array([[neo_hookean_stress(np.eye(dim) + t * H, lam_target, mu_target) for t in fractions]
                        for H in macro_gradients])
    rho_final = discretization.get_scalar_field(name='rho_final')
    rho_final.s[...] = rho_nodal.s
    io_utils.save_fields(os.path.join(folder, 'final.nc'), [rho_final], attributes={
        **parameters, 'nb_iterations': iteration[0], 'elapsed': elapsed,
        'homogenized_stresses': log['homogenized_stresses'][-1], 'response_fractions': fractions,
        'response_curves': curves, 'response_targets': targets,
        **{f'log_{k}': np.asarray(v) for k, v in log.items()}})

    if comm.rank == 0:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, nb_load_cases + 1, figsize=(4.5 * (nb_load_cases + 1), 4))
        for l in range(nb_load_cases):
            i, j = np.unravel_index(np.argmax(np.abs(load_directions[l])), (dim, dim))
            for (a, b), style in [((0, 0), 'C0'), ((1, 1), 'C1'), ((0, 1), 'C2')]:
                axes[l].plot(fractions * strain_magnitude, curves[l, :, a, b], '-', color=style, label=f'P{a}{b} design')
                axes[l].plot(fractions * strain_magnitude, targets[l, :, a, b], '--', color=style, label=f'P{a}{b} target')
            axes[l].axvline(strain_magnitude, color='0.7', lw=0.8)
            axes[l].set_xlabel(f'H{i}{j} (load case {l})')
            axes[l].set_title(f'load case {l}: optimized at {strain_magnitude:g}', fontsize=9)
            axes[l].legend(fontsize=7)
        axes[-1].semilogy(log['objective'], label='objective')
        axes[-1].semilogy(log['mismatch'], label='stress mismatch')
        axes[-1].semilogy(log['phase_field'], label='phase-field term')
        axes[-1].set_xlabel('objective evaluation')
        axes[-1].legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(os.path.join(folder, 'response.png'), dpi=110)
        print(f'{iteration[0]} iterations, {elapsed:.1f} s; homogenized stresses\n'
              f'{log["homogenized_stresses"][-1].round(5)}\ntarget\n{target_stresses.round(5)}\nwrote {folder}')
