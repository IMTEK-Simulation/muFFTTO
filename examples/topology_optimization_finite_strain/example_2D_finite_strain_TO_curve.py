"""
Topology optimization in FINITE STRAIN for a prescribed NONLINEAR stress-strain curve (2D, plane strain),
optionally with the void as a THIRD MEDIUM so that the cell may come into internal contact.

Switch `third_medium_contact`:
  False (default): plain finite strain topology optimization. Soft SIMP void, no regularization; the
                   default target is a strain-stiffening J-curve in tension.
  True           : periodic cells that respond (almost) linearly at small strain and stiffen when their
                   inner surfaces come into contact at large compression.
The contact variant follows third medium contact (TMC) topology
optimization (Frederiksen, Dalklint, Sigmund, Poulios, CMAME 436 (2025) 117595): one neo-Hookean law for
solid and void, its stiffness interpolated with RAMP down to gamma_0 in the void, plus the HuHu-LuLu
regularization; the void then transmits contact forces when it is compressed to zero volume.
Homogenization (periodic cell, macroscopic deformation gradient) and the phase-field regularization of
the design are those of the muFFTTO topology optimization examples.

Problem
-------
Load case l deforms the cell along F(tau) = I + tau H_l (H_l a fixed direction, tau the strain level;
tau < 0 for compression). At the levels tau_lk (k = 1..K) the homogenized first Piola-Kirchhoff stress
P_h(tau_lk; rho) should match the target curve P_t(tau_lk) in the selected components (mask m_l; the
other components are free):

    min_rho  F(rho) = w / N  sum_l sum_k  |m_l o (P_h^lk - P_t^lk)|^2 / S_l                     (curve mismatch)
                     + eta int |grad rho|^2 dx + 1/eta int rho^2 (1 - rho)^2 dx                  (phase-field term)
    s.t.     0 <= rho <= 1,   u_lk = argmin Pi(u; tau_lk, rho)                                   (equilibrium)

    Pi(u; tau, rho) = int gamma(rho) W(I + tau H + D u) dx + k_r/2 (|H u|^2 - |L u|^2 / d)       (HuHu-LuLu)

N = number of (l, k) pairs, S_l = max_k |m_l o P_t^lk|^2. W is the compressible neo-Hookean energy of
muFFTTO.material_models.NeoHookean, W = lam/2 ln(J)^2 + mu/2 (F:F - 2) - mu ln(J): it grows without
bound for J -> 0, which is what makes a compressed void (third medium) stiff. gamma(rho) is RAMP,
    gamma(rho) = gamma_0 + (1 - gamma_0) rho / (1 + q (1 - rho)),
so P = gamma(rho) P_1(F) and dP/drho = gamma'/gamma P (SIMP gamma_0 + (1 - gamma_0) rho^p without contact). The regularization (k_r = alpha L^2 (lam + 2 mu)
of the solid) acts on the fluctuation only and does not depend on rho; with bilinear (Q1) pixel elements
the discrete Laplacian vanishes and only the HuHu part acts (as in the contact-fracture example).

The material is hyperelastic, so every level is an equilibrium problem of its own with its own state and
adjoint (K = D^T W A D + R the Hessian of Pi at the minimizer):
    K lambda_lk = - c_lk D^T W A : d_lk,     d_lk = m_l o (P_h^lk - P_t^lk),   c_lk = 2 w / (N S_l |Omega|)
    dF/drho_a = sum_lk N^T W [ gamma'/gamma (c_lk d_lk + D lambda_lk) : P ]_a + d(phase-field term)/drho_a
(the homogenized stress is the average of P: the regularization does not see the affine part).

Solvers
-------
Equilibrium: minimization of Pi with NuMPI's trust-region Newton-CG (Steihaug), which handles negative
curvature (buckling, contact); det F <= 0 is rejected (large finite energy). The inner CG stops on the
Hestenes-Stiefel lower bound of the energy error with delay cg_delay (tr_inner_stop = 'energy', patched
NuMPI), relative to the estimated initial energy error, with the forcing tolerance min(0.5, sqrt|g|). Preconditioner: Green
operator of the solid plus the regularization, optionally with Jacobi scaling of the current Hessian
(Green_Jacobi), as in examples/phase_field_fracture/example_2D_phase_field_fracture_contact.py. The
levels of a load case are solved in order (path following): the state of the previous design is the
first initial guess; if that fails, the strain is stepped up from the converged previous level.
Adjoint: CG with the same Hessian and preconditioner. Optimizer: NuMPI's L-BFGS-B.

Target curve (target_type)
--------------------------
'curve'      : P_t = sign(tau) (E_1 e + E_2 <e - e_c>^2), e = |tau|: linear up to the strain e_c (contact),
               then stiffening (tangent E_1 + 2 E_2 (e - e_c)); e_c = 0 gives a J-curve from the start
'neo_hookean': the stress of a homogeneous neo-Hookean material (lam_t, mu_t) at F(tau)

Run:   python example_2D_finite_strain_TO_curve.py [-n 64] [--weight 50] [--contact] [--double-well 10]
                                                   [--target E1 E2 e_c]
Output: data/<run name>/ (history.nc, final.nc, design_*.png, curves.png), a new folder for every run.
Needs the patched NuMPI with `precond` and `inner_stop` in tr_newton_bounded (see docs/examples/internal_contact.md).
"""
import os
import sys
import time
import inspect
import argparse

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import numpy as np
from mpi4py import MPI
from NuMPI import Optimization
from NuMPI.Optimization import tr_newton_bounded

from muFFTTO import domain
from muFFTTO import solvers
from muFFTTO import material_models
from muFFTTO import topology_optimization
from muFFTTO import io_utils
from muFFTTO import geometry

if not {'precond', 'inner_stop'} <= set(inspect.signature(tr_newton_bounded).parameters):
    raise RuntimeError('this example needs the patched NuMPI with `precond` and `inner_stop` in tr_newton_bounded')

# ============================================================================
# Parameters
# ============================================================================
# command line (optional): -n <pixels per direction>, --weight <weight of the curve mismatch>, --contact
parser = argparse.ArgumentParser(description='finite strain topology optimization for a stress-strain curve')
parser.add_argument('-n', type=int, default=32, help='pixels per direction (default 32)')
parser.add_argument('--weight', type=float, default=5.0, help='weight of the curve mismatch (default 5)')
parser.add_argument('--contact', action='store_true', help='third medium contact (default: plain finite strain)')
parser.add_argument('--normalize', action='store_true',
                    help='divide the objective by its initial value (moderate first optimizer step)')
parser.add_argument('--init', default='sin',
                    help="initial design: 'sin' (pattern of the small-strain example), 'holes' (large voids) or "
                         "any name of muFFTTO.geometry, e.g. 'sinusoidal_ligaments'")
parser.add_argument('--init-file', default=None,
                    help='start from the design of an earlier run (its folder or final.nc: final design; history.nc: '
                         'last iterate, also of an unfinished run); overrides --init')
parser.add_argument('--levels', type=int, default=6, help='number of strain levels per load case (default 6)')
parser.add_argument('--max-strain', type=float, default=0.30, help='largest strain |tau| (default 0.3)')
parser.add_argument('--double-well', type=float, default=1.0, help='weight of the double-well term (default 1)')
parser.add_argument('--target', type=float, nargs=3, metavar=('E1', 'E2', 'e_c'), default=None,
                    help='target curve parameters (default: per model, see below)')
arguments, _ = parser.parse_known_args()   # known only: check_sensitivity.py imports this module

# grid
domain_size = [1.0, 1.0]
nnn = arguments.n                           # pixels per direction
number_of_pixels = (nnn, nnn)
element_type = 'bilinear_rectangle'

# model of the void
#   False: plain finite strain topology optimization: soft SIMP void, no regularization (no contact)
#   True : third medium contact (TMC paper): RAMP down to gamma_0 = 1e-6 plus HuHu-LuLu regularization,
#          the void transmits contact forces when it is compressed to zero volume
third_medium_contact = arguments.contact

# solid material (K = 1, G = 0.5 as in the small-strain example; lam = K - 2/3 G)
K_solid, G_solid = 1.0, 0.5
ramp_q = 3.0                                # RAMP penalization q
simp_p = 2.0                                # SIMP exponent
if third_medium_contact:
    interpolation = 'ramp'
    gamma_0 = 1e-5                          # stiffness of the third medium relative to the solid (as k_v of
                                            # the contact-fracture example; 1e-6 hardly converges at contact)
    alpha_regularization = 1e-6             # HuHu-LuLu: k_r = alpha L^2 (lam + 2 mu)
else:
    interpolation = 'simp'
    gamma_0 = 1e-4                          # soft void
    alpha_regularization = 0.0

# load cases: direction H, matched stress components; strain levels tau_k (< 0: compression).
# Target curve P = sign(tau) (E1 e + E2 <e - e_c>^2), e = |tau|, in the matched components.
if third_medium_contact:
    # compression: linear up to the contact strain e_c, then stiffening by contact
    load_cases = [
        {'name': 'compression x', 'direction': [[1.0, 0.0], [0.0, 0.0]], 'match': [(0, 0)]},
        {'name': 'compression y', 'direction': [[0.0, 0.0], [0.0, 1.0]], 'match': [(1, 1)]},
    ]
    levels = -np.linspace(arguments.max_strain / arguments.levels, arguments.max_strain, arguments.levels)
    target_E1, target_E2, target_contact_strain = 0.05, 1.5, 0.15
else:
    # tension: strain-stiffening J-curve (soft at small strain, stiffer at large strain)
    load_cases = [
        {'name': 'tension x', 'direction': [[1.0, 0.0], [0.0, 0.0]], 'match': [(0, 0)]},
        {'name': 'tension y', 'direction': [[0.0, 0.0], [0.0, 1.0]], 'match': [(1, 1)]},
    ]
    levels = np.linspace(arguments.max_strain / arguments.levels, arguments.max_strain, arguments.levels)
    target_E1, target_E2, target_contact_strain = 0.05, 0.3, 0.0
if arguments.target is not None:
    target_E1, target_E2, target_contact_strain = arguments.target

# target curve
target_type = 'curve'                       # 'curve' (the curve above) or 'neo_hookean'
lam_target, mu_target = 0.0, 0.1            # neo_hookean target

# objective
weight = arguments.weight                   # weight of the curve mismatch
double_well_depth = arguments.double_well
eta_in_pixels = 1.0                         # phase-field interface width, in pixels

# solvers
preconditioner_type = 'Green_Jacobi'        # 'Green' or 'Green_Jacobi'
tr_gtol = 1e-8                              # trust region: max |nodal force| / force scale
tr_maxiter = 2000 if third_medium_contact else 500
tr_delta_max = 0.5                          # largest trust-region radius (NuMPI default 0.5)
tr_inner_stop = 'energy'                    # Steihaug CG: 'energy' (PT lower bound, delay cg_delay) or 'residual'
adjoint_rtol = 1e-8                         # adjoint CG: relative energy error
cg_delay = 10                               # CG stopping: minimal delay of the energy-error lower bound (PT)
cg_maxiter = 5000
min_level_step = 1e-3                       # smallest strain increment of the fallback path following
INADMISSIBLE_ENERGY = 1e30                  # energy of states with det F <= 0 (finite: the trust region shrinks)

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
quad_weights = discretization.quadrature_weights      # [q]
eta = eta_in_pixels * float(np.max(pixel_size))


def root_print(*args):
    if comm.rank == 0:
        print(*args, flush=True)


def global_dot(a, b):
    """Global scalar product of two fields (all MPI ranks)."""
    return discretization.communicator.sum(np.dot(a.s.ravel(), b.s.ravel()))


def global_sum(value):
    return comm.allreduce(float(value), op=MPI.SUM)


def global_min(value):
    return comm.allreduce(float(value), op=MPI.MIN)


def global_max(value):
    return comm.allreduce(float(value), op=MPI.MAX)


def cell_average(field_ijqxyz):
    """(1/|Omega|) int field dx for a tensor field at the quadrature points -> [i, j]."""
    weighted = np.einsum('ijq...,q->ijq...', field_ijqxyz.s, quad_weights)
    return discretization.mpi_reduction.sum(weighted, axis=tuple(range(-dim - 1, 0))) / cell_volume


# ============================================================================
# Materials and targets
# ============================================================================
lam_solid = K_solid - 2.0 / 3.0 * G_solid
mu_solid = G_solid
k_r = alpha_regularization * domain_size[0] ** 2 * (lam_solid + 2.0 * mu_solid)
force_scale = (lam_solid + 2.0 * mu_solid) * float(np.prod(pixel_size)) ** ((dim - 1) / dim)   # E h^(d-1)


def interpolation_factor(rho):
    """gamma(rho) and gamma'(rho): stiffness factor of both Lame parameters (solid 1, void gamma_0)."""
    if interpolation == 'ramp':
        denominator = 1.0 + ramp_q * (1.0 - rho)
        return (gamma_0 + (1.0 - gamma_0) * rho / denominator,
                (1.0 - gamma_0) * (1.0 + ramp_q) / denominator ** 2)
    return gamma_0 + (1.0 - gamma_0) * rho ** simp_p, (1.0 - gamma_0) * simp_p * rho ** (simp_p - 1)


def neo_hookean_stress(F, lam, mu):
    """P(F) of a homogeneous neo-Hookean material for a single 2x2 deformation gradient."""
    F_inv_T = np.linalg.inv(F).T
    return lam * np.log(np.linalg.det(F)) * F_inv_T + mu * (F - F_inv_T)


def target_stress(case, tau):
    """Target stress (2x2) of load case `case` at strain level tau; only the matched components are used."""
    if target_type == 'neo_hookean':
        return neo_hookean_stress(np.eye(dim) + tau * directions[case], lam_target, mu_target)
    e = abs(tau)
    value = np.sign(tau) * (target_E1 * e + target_E2 * max(e - target_contact_strain, 0.0) ** 2)
    P = np.zeros((dim, dim))
    for (i, j) in load_cases[case]['match']:
        P[i, j] = value
    return P


nb_cases, nb_levels = len(load_cases), len(levels)
directions = np.array([case['direction'] for case in load_cases], dtype=float)
masks = np.zeros((nb_cases, dim, dim))
for l, case in enumerate(load_cases):
    for (i, j) in case['match']:
        masks[l, i, j] = 1.0
target_stresses = np.array([[target_stress(l, tau) for tau in levels] for l in range(nb_cases)])    # [l, k, i, j]
stress_scales = np.array([np.max(np.sum((masks[l] * target_stresses[l]) ** 2, axis=(1, 2)))
                          for l in range(nb_cases)])                                              # S_l
weight_per_pair = weight / (nb_cases * nb_levels)

# ============================================================================
# Fields
# ============================================================================
rho_nodal = discretization.get_scalar_field(name='rho')                   # design variable (nodal)
rho_quad = discretization.get_quad_field_scalar(name='rho_quad')
lam_field = discretization.get_quad_field_scalar(name='lam_field')        # gamma(rho) lam at the quadrature points
mu_field = discretization.get_quad_field_scalar(name='mu_field')          # gamma(rho) mu
dlog_gamma_field = discretization.get_quad_field_scalar(name='dlog_gamma')  # gamma'(rho) / gamma(rho)
material = material_models.NeoHookean(discretization=discretization, lam_1qxyz=lam_field, mu_1qxyz=mu_field)

direction_field = discretization.get_gradient_size_field(name='direction')  # H_l at all quadrature points
F_field = discretization.get_displacement_gradient_sized_field(name='F')
P_field = discretization.get_displacement_gradient_sized_field(name='P')
A_field = discretization.get_material_data_size_field_mugrid(name='A')     # tangent dP/dF
energy_field = discretization.get_quad_field_scalar(name='energy')
work_quad = discretization.get_displacement_gradient_sized_field(name='work_quad')
scalar_quad = discretization.get_quad_field_scalar(name='scalar_quad')
scalar_nodal = discretization.get_scalar_field(name='scalar_nodal')
hess_u = discretization.get_displacement_hessian_size_field(name='hess_u')
lap_u = discretization.get_displacement_laplacian_at_quad_field(name='lap_u')

u_work = discretization.get_unknown_size_field(name='u_work')             # unknown of fun/hessp
v_work = discretization.get_unknown_size_field(name='v_work')
out_work = discretization.get_unknown_size_field(name='out_work')
reg_work = discretization.get_unknown_size_field(name='reg_work')
lap_work = discretization.get_unknown_size_field(name='lap_work')
jacobi_work = discretization.get_unknown_size_field(name='jacobi_work')
u_path = discretization.get_unknown_size_field(name='u_path')           # start candidate of a level
adjoint_rhs = discretization.get_unknown_size_field(name='adjoint_rhs')
sensitivity = discretization.get_scalar_field(name='sensitivity')
u_shape = u_work.s.shape

# state and adjoint of every (load case, level); kept between objective evaluations (initial guesses)
displacements = [[discretization.get_unknown_size_field(name=f'u_{l}_{k}') for k in range(nb_levels)]
                 for l in range(nb_cases)]
adjoints = [[discretization.get_unknown_size_field(name=f'adjoint_{l}_{k}') for k in range(nb_levels)]
            for l in range(nb_cases)]

# ============================================================================
# Regularization R = k_r (H^T W H - L^T W L / d)
# ============================================================================
# the discrete Laplacian of a bilinear (Q1) field on pixels is identically zero: skip it when it has no effect
lap_work.s[...] = np.random.default_rng(comm.rank).random(u_shape)
discretization.fft.communicate_ghosts(lap_work)
discretization.laplacian.apply(nodal_field=lap_work, quadrature_point_field=lap_u)
laplacian_acts = global_max(np.max(np.abs(lap_u.s))) > 0.0


def add_regularization(x, out):
    """out += R x (x a field with current ghosts)."""
    if k_r == 0.0:
        return
    discretization.apply_hessian_operator_to_vector_field_mugrid(u_inxyz=x, hess_u_ijkqxyz=hess_u)
    discretization.apply_hessian_operator_transposed_to_vector_field_mugrid(hess_u_ijkqxyz=hess_u,
                                                                           nodal_field_inxyz=reg_work,
                                                                           apply_weights=True)
    out.s[...] += k_r * reg_work.s
    if laplacian_acts:
        discretization.laplacian.apply(nodal_field=x, quadrature_point_field=lap_u)
        discretization.laplacian.transpose(quadrature_point_field=lap_u, nodal_field=lap_work,
                                           weights=quad_weights)
        out.s[...] -= (k_r / dim) * lap_work.s


# ============================================================================
# Energy, gradient, Hessian of Pi at strain level tau (for the trust-region solver)
# ============================================================================
current_tau = 0.0                           # strain level of the problem being solved
tangent_state = {'x': None, 'tau': None}    # (x, tau) the tangent A_field was built for


def set_deformation_gradient(u, tau):
    """F = I + tau H + D u at all quadrature points; returns the global min of det F."""
    discretization.apply_gradient_operator_mugrid(u_inxyz=u, grad_u_ijqxyz=F_field)
    F_field.s[...] += tau * direction_field.s
    for k in range(dim):
        F_field.s[k, k] += 1.0
    F = F_field.s
    return global_min(np.min(F[0, 0] * F[1, 1] - F[0, 1] * F[1, 0]))


def energy_and_gradient(x):
    """Pi and dPi/du (flat, scaled by force_scale) at the fluctuation x and level current_tau."""
    u_work.s[...] = x.reshape(u_shape)
    if not set_deformation_gradient(u_work, current_tau) > 0.0:
        return INADMISSIBLE_ENERGY, np.zeros(x.size)
    material.get_energy_density(F_field, energy_field)
    energy = global_sum(np.sum(np.einsum('q...,q->q...', energy_field.s[0, 0], quad_weights)))
    out_work.s[...] = 0.0
    add_regularization(u_work, out_work)
    energy += 0.5 * global_dot(u_work, out_work)
    if not np.isfinite(energy):
        return INADMISSIBLE_ENERGY, np.zeros(x.size)
    material.get_stress(F_field, P_field)
    discretization.apply_gradient_transposed_operator_mugrid(gradient_field_ijqxyz=P_field, div_u_fnxyz=v_work,
                                                             apply_weights=True)
    v_work.s[...] += out_work.s
    return energy / force_scale, v_work.s.reshape(-1) / force_scale


def update_tangent(x, tau):
    """Tangent A at (x, tau) into A_field, unless it is already there."""
    stale = tangent_state['tau'] != tau or tangent_state['x'] is None or not np.array_equal(tangent_state['x'], x)
    if global_max(float(stale)) == 0.0:
        return
    u_work.s[...] = x.reshape(u_shape)
    set_deformation_gradient(u_work, tau)
    material.get_algorithmic_tangent(F_field, A_field)
    tangent_state.update(x=np.array(x, copy=True), tau=tau)


def apply_hessian(input_field_inxyz, output_field_inxyz):
    """output = K input = D^T W A D input + R input with the current A_field."""
    discretization.apply_system_matrix_mugrid(material_data_field=A_field, input_field_inxyz=input_field_inxyz,
                                              output_field_inxyz=output_field_inxyz, formulation='finite_strain')
    add_regularization(input_field_inxyz, output_field_inxyz)
    discretization.fft.communicate_ghosts(output_field_inxyz)


def hessian_product(x, v):
    """K(x) v (flat, scaled), for the trust-region solver."""
    update_tangent(x, current_tau)
    v_work.s[...] = v.reshape(u_shape)
    discretization.fft.communicate_ghosts(v_work)
    apply_hessian(v_work, out_work)
    return out_work.s.reshape(-1) / force_scale


# ============================================================================
# Preconditioner: Green operator of the solid + R, optionally with Jacobi scaling of the current Hessian
# ============================================================================
reference_tangent = material_models.get_elastic_material_tensor(dim=dim, K=K_solid, mu=G_solid)


def reference_operator(input_field_inxyz, output_field_inxyz):
    discretization.fft.communicate_ghosts(input_field_inxyz)
    discretization.apply_system_matrix_mugrid(material_data_field=reference_tangent, input_field_inxyz=input_field_inxyz,
                                              output_field_inxyz=output_field_inxyz, formulation='finite_strain')
    add_regularization(input_field_inxyz, output_field_inxyz)


green_operator = discretization.get_preconditioner_Green_mugrid(reference_material_data_ijkl=reference_tangent,
                                                                operator=reference_operator)
# Jacobi scaling normalized by the (uniform) diagonal of the reference operator, so that the trust region
# (measured in the preconditioner norm) is the same for Green and Green_Jacobi
reference_diagonal = global_max(np.max(
    1.0 / discretization.get_preconditioner_Jacobi_mugrid(operator=reference_operator).s ** 2))
jacobi_scaling = None


def update_jacobi():
    """diag(K)^-1/2 of the Hessian with the current A_field (Green_Jacobi only)."""
    global jacobi_scaling
    if preconditioner_type == 'Green_Jacobi':
        jacobi_scaling = np.sqrt(reference_diagonal) * discretization.get_preconditioner_Jacobi_mugrid(
            operator=apply_hessian).s


def apply_preconditioner(r_field, z_field):
    """z = M^-1 r (fields)."""
    discretization.fft.communicate_ghosts(r_field)
    if preconditioner_type == 'Green_Jacobi':
        jacobi_work.s[...] = jacobi_scaling * r_field.s
        r_field = jacobi_work
    discretization.apply_preconditioner_mugrid(preconditioner_Fourier_fnfnqks=green_operator,
                                               input_nodal_field_fnxyz=r_field, output_nodal_field_fnxyz=z_field)
    if preconditioner_type == 'Green_Jacobi':
        z_field.s[...] *= jacobi_scaling
    discretization.fft.communicate_ghosts(z_field)


def preconditioner_flat(v):
    """M^-1 v (flat, scaled consistently with hessian_product), for the trust-region solver."""
    v_work.s[...] = v.reshape(u_shape)
    apply_preconditioner(v_work, out_work)
    return out_work.s.reshape(-1) * force_scale


# ============================================================================
# State problem: minimize Pi at one strain level
# ============================================================================
def minimize(u, tau):
    """Trust-region minimization of Pi(.; tau) starting from u (updated in place on success).
    Returns (success, trust-region iterations, Hessian products)."""
    global current_tau
    current_tau = tau
    if not set_deformation_gradient(u, tau) > 0.0:
        return False, 0, 0
    update_tangent(u.s.reshape(-1).copy(), tau)
    update_jacobi()
    result = tr_newton_bounded(fun=energy_and_gradient, x0=u.s.reshape(-1).copy(), hessp=hessian_product,
                               precond=preconditioner_flat, jac=True, gtol=tr_gtol, maxiter=tr_maxiter,
                               inner_tol=None, inner_stop=tr_inner_stop, inner_delay=cg_delay,
                               delta_max=tr_delta_max, comm=comm)
    if not result.success or result.fun >= 0.5 * INADMISSIBLE_ENERGY:
        return False, int(result.nit), int(result.nb_hessp)
    u.s[...] = np.asarray(result.x).reshape(u_shape)
    return True, int(result.nit), int(result.nb_hessp)


def predict_fluctuation(u, H, tau_0, tau_1):
    """Start for level tau_1 from the converged state u at tau_0 (in place): u <- A u with
    A = (I + tau_1 H)(I + tau_0 H)^-1, so that F_start = A F (all det F > 0 stay positive; adding the strain
    increment to F would invert third-medium elements that are compressed almost to zero volume)."""
    A = (np.eye(dim) + tau_1 * H) @ np.linalg.inv(np.eye(dim) + tau_0 * H)
    u.s[...] = np.einsum('ij,j...->i...', A, u.s)


def solve_level(l, k):
    """Equilibrium of load case l at level k (levels 0..k-1 of this design already solved). Leaves F, P,
    the tangent A and its Jacobi scaling of the converged state. Returns (TR iterations, Hessian products)."""
    tau = levels[k]
    u = displacements[l][k]
    # 1. start from the better (lower energy) of: the state of the previous design at this level, or the
    #    converged previous level of this design carried to tau (predictor). With contact the former can
    #    be a poor start (the contact configuration differs between designs).
    global current_tau
    current_tau = tau
    u_path.s[...] = displacements[l][k - 1].s if k > 0 else 0.0
    predict_fluctuation(u_path, directions[l], levels[k - 1] if k > 0 else 0.0, tau)
    if energy_and_gradient(u_path.s.reshape(-1))[0] < energy_and_gradient(u.s.reshape(-1))[0]:
        u.s[...] = u_path.s
    ok, nb_it, nb_hp = minimize(u, tau)
    if not ok:
        # 2. path following: from the converged previous level (or u = 0), step the strain to tau
        tau_0 = levels[k - 1] if k > 0 else 0.0
        u.s[...] = displacements[l][k - 1].s if k > 0 else 0.0
        step = (tau - tau_0) / 2.0
        while abs(tau - tau_0) > 1e-12:
            tau_1 = tau_0 + step if abs(step) < abs(tau - tau_0) else tau
            u_backup = u.s.copy()
            predict_fluctuation(u, directions[l], tau_0, tau_1)
            ok, it, hp = minimize(u, tau_1)
            nb_it, nb_hp = nb_it + it, nb_hp + hp
            if ok:
                tau_0 = tau_1
            else:
                u.s[...] = u_backup
                step *= 0.5
                if abs(step) < min_level_step:
                    raise RuntimeError(f'load case {l}: no equilibrium found beyond strain {tau_0:.4f}')
    # converged state: F, P, tangent and the Jacobi scaling for the adjoint problem
    x = u.s.reshape(-1).copy()
    update_tangent(x, tau)
    set_deformation_gradient(u, tau)
    material.get_stress(F_field, P_field)
    update_jacobi()
    return nb_it, nb_hp


def solve_adjoint(rhs, x):
    """Solve K x = rhs (Hessian at the converged state) with PCG, starting from x.
    Stopping: the Meurant-Papez-Tichy lower bound of the energy error |x - x_k|_K^2 (Hestenes-Stiefel sum
    with adaptive delay, at least cg_delay iterations; solvers.conjugate_gradients_mugrid_experimental)
    below adjoint_rtol^2 times the estimate of the initial energy error |x - x_0|_K^2 (energy_rtol: relative, like a
    relative residual; computed from the CG scalars, no extra operator application).
    Returns the number of iterations."""
    counter = [0]

    def count(*args):
        counter[0] += 1

    solvers.conjugate_gradients_mugrid_experimental(
        comm=discretization.communicator, fc=discretization.field_collection, hessp=apply_hessian, b=rhs, x=x,
        P=apply_preconditioner, tol=adjoint_rtol, maxiter=cg_maxiter, callback=count,
        stop_crit_norm='energy_lower_estim', d_min=cg_delay, energy_rtol=True)
    return max(counter[0] - 1, 0)           # called once before the first iteration (not at all if x is converged)


# ============================================================================
# Design -> material
# ============================================================================
def set_design(rho_flat):
    """rho (flat, rank-local) -> nodal field, quadrature values, Lame fields and gamma'/gamma."""
    rho_nodal.s[...] = rho_flat.reshape(rho_nodal.s.shape)
    discretization.apply_N_operator_mugrid(rho_nodal, rho_quad)
    gamma, dgamma = interpolation_factor(rho_quad.s[0, 0])
    lam_field.s[0, 0] = gamma * lam_solid
    mu_field.s[0, 0] = gamma * mu_solid
    dlog_gamma_field.s[0, 0] = dgamma / gamma
    tangent_state.update(x=None, tau=None)  # the material changed


# ============================================================================
# Objective and its gradient
# ============================================================================
log = {'objective': [], 'mismatch': [], 'phase_field': [], 'tr_iterations': [], 'hessian_products': [],
       'cg_adjoint': [], 'homogenized_stresses': []}


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
    nb_it = nb_hp = nb_cg_adjoint = 0
    stresses = np.zeros((nb_cases, nb_levels, dim, dim))
    expand = (...,) + (np.newaxis,) * (work_quad.s.ndim - 2)              # [i, j] -> [i, j, q, x, y]
    for l in range(nb_cases):
        discretization.get_macro_gradient_field_mugrid(macro_gradient_ij=directions[l],
                                                       macro_gradient_field_ijqxyz=direction_field)
        for k in range(nb_levels):
            # state: equilibrium, homogenized stress, mismatch of the matched components
            it, hp = solve_level(l, k)
            nb_it, nb_hp = nb_it + it, nb_hp + hp
            stresses[l, k] = cell_average(P_field)
            difference = masks[l] * (stresses[l, k] - target_stresses[l, k])          # d_lk
            f_mismatch += weight_per_pair * np.sum(difference ** 2) / stress_scales[l]
            c = 2.0 * weight_per_pair / (stress_scales[l] * cell_volume)

            # adjoint: K lambda = -c D^T W A : d
            work_quad.s[...] = np.einsum('ijkl...,kl->ij...', A_field.s, difference)
            discretization.apply_gradient_transposed_operator_mugrid(gradient_field_ijqxyz=work_quad,
                                                                     div_u_fnxyz=adjoint_rhs, apply_weights=True)
            adjoint_rhs.s[...] *= -c
            nb_cg_adjoint += solve_adjoint(adjoint_rhs, adjoints[l][k])

            # sensitivity: N^T W [ gamma'/gamma (c d + D lambda) : P ]
            discretization.apply_gradient_operator_mugrid(u_inxyz=adjoints[l][k], grad_u_ijqxyz=work_quad)
            work_quad.s[...] += c * difference[expand]
            scalar_quad.s[0, 0] = dlog_gamma_field.s[0, 0] * np.einsum('ij...,ij...->...', work_quad.s, P_field.s)
            discretization.apply_N_transposed_operator_mugrid(quad_field_ijqxyz=scalar_quad,
                                                              nodal_field_inxyz=scalar_nodal, apply_weights=True)
            sensitivity.s[...] += scalar_nodal.s

    discretization.fft.communicate_ghosts(sensitivity)
    f = f_mismatch + f_phase
    for key, value in [('objective', f), ('mismatch', f_mismatch), ('phase_field', f_phase),
                       ('tr_iterations', nb_it), ('hessian_products', nb_hp), ('cg_adjoint', nb_cg_adjoint),
                       ('homogenized_stresses', stresses)]:
        log[key].append(value)
    root_print(f'  F {f:.6e}  (mismatch {f_mismatch:.3e}, phase field {f_phase:.3e})  TR it {nb_it:4d}  '
               f'Hessian products {nb_hp:6d}  CG adjoint {nb_cg_adjoint:5d}')
    return f, sensitivity.s.reshape(-1).copy()


# ============================================================================
# Post-processing
# ============================================================================
def response_curve(l, taus):
    """Homogenized stress of the current design along F = I + tau H_l (taus monotone, from 0)."""
    discretization.get_macro_gradient_field_mugrid(macro_gradient_ij=directions[l],
                                                   macro_gradient_field_ijqxyz=direction_field)
    u = discretization.get_unknown_size_field(name='u_curve')
    u.s[...] = 0.0
    stresses = []
    tau_previous = 0.0
    for tau in taus:
        predict_fluctuation(u, directions[l], tau_previous, tau)
        tau_previous = tau
        if minimize(u, tau)[0]:
            set_deformation_gradient(u, tau)
            material.get_stress(F_field, P_field)
            stresses.append(cell_average(P_field))
        else:
            stresses.append(np.full((dim, dim), np.nan))
    return np.array(stresses)


def save_design_figure(folder, tag):
    """Design (3 x 3 cells) and the deformed cell (solid and third medium) of every load case at the
    largest level (serial runs only)."""
    if comm.size > 1:
        return
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    rho = rho_nodal.s[0, 0]
    fig, axes = plt.subplots(1, 1 + nb_cases, figsize=(4 * (1 + nb_cases), 4))
    axes[0].imshow(np.tile(rho, (3, 3)).T, origin='lower', cmap='Greys', vmin=0, vmax=1)
    axes[0].set_title(f'design ({tag}), 3 x 3 cells, black = solid', fontsize=9)
    nx, ny = rho.shape
    i, j = np.meshgrid(np.arange(nx + 1), np.arange(ny + 1), indexing='ij')
    X = np.array([i * pixel_size[0], j * pixel_size[1]])
    for l in range(nb_cases):
        u = displacements[l][-1].s[:, 0][:, i % nx, j % ny]               # periodic node values at the corners
        x = np.einsum('ab,b...->a...', np.eye(dim) + levels[-1] * directions[l], X) + u
        axes[1 + l].pcolormesh(x[0], x[1], rho, cmap='Greys', vmin=0, vmax=1, shading='flat',
                               edgecolors='0.75', linewidth=0.1)
        axes[1 + l].set_aspect('equal')
        axes[1 + l].set_title(f'{load_cases[l]["name"]}, strain {levels[-1]:g}', fontsize=9)
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
    root_print(f'grid {number_of_pixels}, {element_type}, eta {eta:.4f}, {interpolation} (void {gamma_0:g}), '
               f'k_r {k_r:.2e}, preconditioner {preconditioner_type}, ranks {comm.size}')
    root_print(f'target {target_type}, levels {levels.round(4).tolist()}')
    for l, case in enumerate(load_cases):
        root_print(f'load case {l} ({case["name"]}): H = {case["direction"]}, matched {case["match"]}, target '
                   f'{[round(float(np.sum(masks[l] * P)), 5) for P in target_stresses[l]]}')

    run_name = (f'curve_{"tmc" if third_medium_contact else "fs"}_N{nnn}_{interpolation}_{preconditioner_type}_tau{levels[-1]:g}x{nb_levels}'
                f'_w{weight:g}_dw{double_well_depth:g}_E{target_E1:g}-{target_E2:g}-{target_contact_strain:g}'
                f'{"_norm" if arguments.normalize else ""}_{"restart" if arguments.init_file else arguments.init}_'
                + time.strftime('%Y%m%d-%H%M%S'))
    folder = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data', run_name)
    if comm.rank == 0:
        os.makedirs(folder)
    comm.Barrier()
    parameters = {'nb_of_pixels': number_of_pixels, 'domain_size': domain_size, 'element_type': element_type,
                  'K_solid': K_solid, 'G_solid': G_solid, 'interpolation': interpolation, 'gamma_0': gamma_0,
                  'ramp_q': ramp_q, 'simp_p': simp_p, 'k_r': k_r, 'eta': eta, 'weight': weight,
                  'double_well_depth': double_well_depth, 'levels': levels, 'directions': directions,
                  'masks': masks, 'third_medium_contact': int(third_medium_contact), 'target_type': target_type, 'target_stresses': target_stresses,
                  'target_E1': target_E1, 'target_E2': target_E2, 'target_contact_strain': target_contact_strain,
                  'lam_target': lam_target, 'mu_target': mu_target, 'preconditioner_type': preconditioner_type,
                  'tr_gtol': tr_gtol, 'adjoint_rtol': adjoint_rtol, 'random_seed': random_seed,
                  'normalize': int(arguments.normalize), 'init': arguments.init,
                  'init_file': os.path.abspath(arguments.init_file) if arguments.init_file else ''}
    root_print(f'output {folder}')

    # initial design: the smooth pattern of the small-strain example plus noise (independent of the ranks)
    coords = discretization.fft.coords
    noise = np.random.default_rng(random_seed).random(number_of_pixels)
    loc = tuple(slice(o, o + n) for o, n in zip(discretization.fft.subdomain_locations,
                                               discretization.fft.nb_subdomain_grid_pts))
    if arguments.init_file:
        # continuation: the final design of an earlier run (same grid)
        init_path = arguments.init_file
        if os.path.isdir(init_path):
            init_path = os.path.join(init_path, 'final.nc')
        # final.nc holds 'rho_final'; history.nc (also of an unfinished run) holds 'rho_iterate', last frame used
        field_name = 'rho_iterate' if os.path.basename(init_path) == 'history.nc' else 'rho_final'
        rho_previous = discretization.get_scalar_field(name=field_name)
        io_utils.load_fields(init_path, [rho_previous])
        rho_0 = rho_previous.s[0, 0].copy()
    elif arguments.init not in ('sin', 'holes'):
        # a geometry of muFFTTO.geometry (solid 1, void 0) at the nodes, e.g. the sinusoidal ligaments
        rho_0 = geometry.get(arguments.init, np.asarray(coords) / np.asarray(domain_size)[:, None, None]).astype(float)
    elif arguments.init == 'holes':
        # one large void per cell (radius 0.4) between solid ligaments, plus noise to break the symmetry
        distance = np.sqrt((coords[0] - 0.5 * domain_size[0]) ** 2 + (coords[1] - 0.5 * domain_size[1]) ** 2)
        rho_0 = np.where(distance < 0.4 * domain_size[0], 0.0, 1.0) + 0.3 * (noise[loc] - 0.5)
    else:
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
            save_design_figure(folder, f'it{iteration[0]:04d}')

    # optional normalization by the initial objective value (the optimizer sees F / F_0)
    objective_scale = 1.0 / objective(rho_0.ravel())[0] if arguments.normalize else 1.0
    root_print(f'objective scale {objective_scale:.3e}')

    def scaled_objective(rho_flat):
        f, g = objective(rho_flat)
        return objective_scale * f, objective_scale * g

    start = time.time()
    result = Optimization.l_bfgs_bounded(fun=scaled_objective, x0=rho_0.ravel(), args=(), jac=True,
                                         bounds_lo=0.0, bounds_hi=1.0, gtol=lbfgs_gtol * objective_scale, xtol=lbfgs_xtol,
                                         maxiter=lbfgs_maxiter, maxcor=20, c1=1e-4, max_halvings=40,
                                         comm=comm, callback=callback, disp=True)
    history.close()
    elapsed = time.time() - start

    # final design: state, figures, the full curves (beyond the last level) against the target
    objective(result.x)
    save_design_figure(folder, 'final')
    taus = np.linspace(0.0, 1.2 * levels[-1], 25)
    curves = np.array([response_curve(l, taus) for l in range(nb_cases)])                 # [l, tau, i, j]
    target_curves = np.array([[target_stress(l, tau) for tau in taus] for l in range(nb_cases)])
    rho_final = discretization.get_scalar_field(name='rho_final')
    rho_final.s[...] = rho_nodal.s
    io_utils.save_fields(os.path.join(folder, 'final.nc'), [rho_final], attributes={
        **parameters, 'nb_iterations': iteration[0], 'elapsed': elapsed,
        'homogenized_stresses': log['homogenized_stresses'][-1], 'curve_taus': taus, 'curves': curves,
        'target_curves': target_curves, **{f'log_{k}': np.asarray(v) for k, v in log.items()}})

    if comm.rank == 0:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, nb_cases + 1, figsize=(4.8 * (nb_cases + 1), 4.2))
        sign = np.sign(levels[-1])                 # compression is shown with positive values
        for l, case in enumerate(load_cases):
            ax = axes[l]
            for n, (i, j) in enumerate(case['match']):
                ax.plot(sign * taus, sign * curves[l, :, i, j], '-', color=f'C{n}', label=f'P{i}{j} design')
                ax.plot(sign * taus, sign * target_curves[l, :, i, j], '--', color=f'C{n}', label=f'P{i}{j} target')
                ax.plot(sign * levels, sign * target_stresses[l, :, i, j], 'o', color=f'C{n}', mfc='none',
                        label='target levels')
            free = [(a, b) for a in range(dim) for b in range(dim) if (a, b) not in case['match']]
            for n, (i, j) in enumerate(free):
                ax.plot(sign * taus, sign * curves[l, :, i, j], ':', color='0.5', lw=0.8,
                        label='other components' if n == 0 else None)
            ax.set_xlabel('strain |tau|')
            ax.set_ylabel('homogenized stress' + (' (compression > 0)' if sign < 0 else ''))
            ax.set_title(f'{case["name"]}: F = I + tau {case["direction"]}', fontsize=9)
            ax.grid(alpha=0.3)
            ax.legend(fontsize=7)
        axes[-1].semilogy(log['objective'], label='objective')
        axes[-1].semilogy(log['mismatch'], label='curve mismatch')
        axes[-1].semilogy(log['phase_field'], label='phase-field term')
        axes[-1].set_xlabel('objective evaluation')
        axes[-1].legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(os.path.join(folder, 'curves.png'), dpi=110)
        print(f'{iteration[0]} iterations, {elapsed:.1f} s; wrote {folder}')
