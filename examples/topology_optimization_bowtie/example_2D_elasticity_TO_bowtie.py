"""
Linear elastic topology optimization of a periodic cell, starting from a bow-tie (re-entrant honeycomb) design.

Steps
-----
1. Initial design: the re-entrant (bow-tie) honeycomb of ``muFFTTO.geometry`` (square cell, re-entrant angle
   ``--theta``, wall thickness ``--thickness``), solid = 1, void = 0.
2. Homogenized properties of the initial design: the effective stiffness C_h (Voigt, engineering shear)
   from three macroscopic strains of amplitude ``--strain`` (default 0.05), and from its compliance S_h = C_h^-1 the engineering constants
       E_x = 1 / S_11,  E_y = 1 / S_22,  G_xy = 1 / S_33,  nu_xy = -S_21 / S_11,  nu_yx = -S_12 / S_22.
   With ``--evaluate-only`` the script stops here.
3. Target with the Poisson ratio ``--nu`` (nu_xy), everything else as close to the bow-tie as possible:
   ``--keep stiffness`` (default): the stiffness entries C_11, C_22, C_66 of the bow-tie, C_12 = nu_xy C_22
   (then nu_xy = C_12 / C_22); the stress targets differ from the bow-tie only through C_12.
   ``--keep moduli``: an orthotropic material with the moduli E_x, E_y, G_xy of the bow-tie (nu_yx =
   nu_xy E_y / E_x); for the strongly auxetic bow-tie this changes C_11 several times (E_x = C_11 - C_12^2/C_22).
   The objective is divided by its initial value (moderate first L-BFGS step: an unscaled first step went to
   rho = 0 everywhere, where the SIMP derivative vanishes).
4. Optimization (as examples/topology_optimization/example_2D_elasticity_TO.py): SIMP
   C(rho) = (C_1 - C_0) rho^p + C_0, objective
       F(rho) = w / 3 sum_k |Sigma_t^k - Sigma_h^k(rho)|^2 / |Sigma_t^k|^2  +  phase-field term
   over the three load cases (xx, yy, shear); adjoint sensitivity; L-BFGS-B with bounds 0 <= rho <= 1.

Regularization (``--regularization``)
-------------------------------------
``phase-field`` (default): the phase-field term eta int |grad rho|^2 + dw / eta int rho^2 (1 - rho)^2 is added to F.
``filter``: no phase-field term; the material sees the physical density rho_bar = H_beta(F rho) of the design rho
(Helmholtz filter of radius ``--filter-radius`` pixels + Heaviside projection, ``muFFTTO.density_filter``), with
beta continuation over ``--beta`` (one L-BFGS run of at most ``--stage-iterations`` per beta, each restarted from
the previous design). The filter radius is the minimum member size. ``--robust``: the objective is the mean of
the mismatches of the eroded (threshold 0.75), intermediate (0.5) and dilated (0.25) projections of the same
filtered field (Wang, Lazarov & Sigmund 2011), which excludes thin members and hinges of the intermediate design
(the reported design). ``--robust-shape``: the eroded and dilated designs are matched up to their own optimal stiffness
factor (as ``--free-scale``), the intermediate design absolutely. The eroded / dilated designs are necessarily softer /
stiffer than the intermediate one and cannot all match one absolute C; only the shape of C (Poisson ratio,
anisotropy) has to be robust.
``--length-scale KAPPA``: adds KAPPA (g_solid + g_void), the geometric minimum length scale constraints of Zhou,
Lazarov, Wang & Sigmund (2015) on the filtered field (``density_filter.length_scale_constraints``), active from
``--length-scale-beta`` on: the filtered field must exceed 0.75 on the skeleton of every solid member and stay below
0.25 on the skeleton of every void, i.e. the blueprint has the minimum feature sizes of the robust designs.

Run
---
    python example_2D_elasticity_TO_bowtie.py --evaluate-only             # properties of the bow-tie only
    python example_2D_elasticity_TO_bowtie.py --nu -0.3 [-n 64] [--weight 5]
    python example_2D_elasticity_TO_bowtie.py --nu -0.3 --regularization filter --filter-radius 3

Output: data/<run name>/ (history.nc, final.nc, design.png, properties.txt), a new folder for every run.
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import numpy as np
from mpi4py import MPI
from NuMPI import Optimization

from muFFTTO import density_filter, domain, geometry, io_utils, material_models, solvers, topology_optimization

parser = argparse.ArgumentParser(description='linear elastic TO from a bow-tie initial design')
parser.add_argument('-n', type=int, default=64, help='pixels per direction (default 64)')
parser.add_argument('--theta', type=float, default=30.0, help='re-entrant angle of the bow-tie, degrees (default 30)')
parser.add_argument('--thickness', type=float, default=0.15, help='wall thickness of the bow-tie, units of l (0.15)')
parser.add_argument('--nu', type=float, default=None, help='target Poisson ratio nu_xy (moduli of the bow-tie kept)')
parser.add_argument('--weight', type=float, default=5.0, help='weight of the stress mismatch (default 5)')
parser.add_argument('--double-well', type=float, default=1.0, help='weight of the double-well term (default 1)')
parser.add_argument('--init', choices=['bowtie', 'original'], default='bowtie',
                    help='initial design: the bow-tie (default) or the guess of topology_optimization/'
                         'example_2D_elasticity_TO.py, (sin 4 pi x + sin 4 pi y + 2) / 4 + 0.5 rand, clipped to [0, 1]')
parser.add_argument('--shear-fraction', type=float, default=3 / 20,
                    help='--keep isotropic: target shear modulus / E_solid (default 3/20 as in the original example)')
parser.add_argument('--gtol', type=float, default=1e-3, help='L-BFGS gradient tolerance (default 1e-3)')
parser.add_argument('--keep', choices=['stiffness', 'moduli', 'isotropic'], default='stiffness',
                    help='target: C_11, C_22, C_66 of the bow-tie (default) or its moduli E_x, E_y, G_xy')
parser.add_argument('--eta', type=float, default=1.0,
                    help='phase-field interface width in pixels (default 1); larger = thicker minimum members')
parser.add_argument('--strain', type=float, default=0.05, help='amplitude of the macroscopic strains (default 0.05)')
parser.add_argument('--cases', nargs='+', choices=['xx', 'yy', 'shear'], default=['xx', 'yy', 'shear'],
                    help='load cases matched in the objective (default all; the others are free)')
parser.add_argument('--free-scale', action='store_true',
                    help='free overall stiffness level: match C_target up to a factor (Poisson ratio, ratios)')
parser.add_argument('--regularization', choices=['phase-field', 'filter'], default='phase-field',
                    help='phase-field term (default) or density filter + Heaviside projection')
parser.add_argument('--filter-radius', type=float, default=3.0, help='filter radius in pixels (default 3)')
parser.add_argument('--beta', type=float, nargs='+', default=[1, 2, 4, 8, 16, 32],
                    help='projection sharpness per continuation stage (default 1 2 4 8 16 32)')
parser.add_argument('--robust', action='store_true',
                    help='with --regularization filter: robust formulation (eroded / intermediate / dilated designs '
                         'must all match; guarantees a minimum size of solid and void features)')
parser.add_argument('--preconditioner', choices=['green-jacobi', 'green', 'jacobi'], default='green-jacobi',
                    help='CG preconditioner: green-jacobi (default) D^1/2 G D^1/2, green G (solid reference), '
                         'jacobi diag(K)^-1')
parser.add_argument('--objective', choices=['stiffness', 'poisson'], default='stiffness',
                    help='stiffness (default): match the target stiffness C_target (stress mismatch of the load cases); '
                         'poisson: match only the Poisson ratio, f = weight (nu_xy(C) - nu)^2 with nu_xy = -S_12 / S_11, '
                         'S = C^-1 (stiffness level and anisotropy free)')
parser.add_argument('--length-scale', type=float, default=0.0, metavar='KAPPA',
                    help='with --regularization filter: add KAPPA (g_solid + g_void), the geometric minimum length scale '
                         'constraints of Zhou et al. 2015 on the filtered field (minimum solid / void size about the '
                         'filter radius); default 0 = off')
parser.add_argument('--length-scale-beta', type=float, default=4.0,
                    help='the length scale term is active from this beta on (default 4; the skeleton indicator needs '
                         'a fairly 0/1 design)')
parser.add_argument('--robust-shape', action='store_true',
                    help='with --robust: the eroded and dilated designs match C_target up to their own stiffness '
                         'factor (Poisson ratio and anisotropy robust); only the intermediate design matches the '
                         'absolute stiffness')
parser.add_argument('--xtol', type=float, default=None,
                    help='L-BFGS stops if no design value changes by more than xtol (default: 1e-3 phase field, '
                         '0 = off for the filter, whose small smooth steps would stop it early)')
parser.add_argument('--stage-iterations', type=int, default=100, help='L-BFGS iterations per beta stage (100)')
parser.add_argument('--simp-p', type=float, default=2.0,
                    help='SIMP exponent (default 2); 3 penalizes grey more (needed for compliant targets)')
parser.add_argument('--evaluate-only', action='store_true', help='only the properties of the initial design')
arguments, _ = parser.parse_known_args()

comm = MPI.COMM_WORLD


def root_print(*args):
    if comm.rank == 0:
        print(*args, flush=True)


# ----------------------------------------------------------------------------------------------------------------
# discretization and material
# ----------------------------------------------------------------------------------------------------------------
domain_size = [1.0, 1.0]                      # square cell: the bow-tie with h/l = cos(theta) + sin(theta)
number_of_pixels = (arguments.n, arguments.n)
dim = 2
pixel_size = np.asarray(domain_size) / np.asarray(number_of_pixels)
element_type = 'linear_triangles'

cell = domain.PeriodicUnitCell(domain_size=domain_size, problem_type='elasticity')
discretization = domain.Discretization(cell=cell, nb_of_pixels_global=number_of_pixels,
                                       discretization_type='finite_element', element_type=element_type)

K_solid, G_solid = 1.0, 0.5                   # solid phase (E = 1.2857, nu = 0.2857 in plane strain)
C_solid = material_models.get_elastic_material_tensor(dim=dim, K=K_solid, mu=G_solid, kind='linear')
void_contrast = 1e-5                          # void stiffness / solid stiffness
C_void = void_contrast * C_solid
simp_p = arguments.simp_p                     # SIMP exponent
eta = arguments.eta * float(np.max(pixel_size))   # phase-field interface width (sets the minimum member size)
double_well_depth = arguments.double_well
weight = arguments.weight
cg_tol = 1e-8                                 # CG tolerance in the optimization: tight, so that the objective is
                                              # smooth for the line search (1e-3 of the original example stalls)
cg_tol_homogenization = 1e-8                  # CG tolerance for the reported properties

# ----------------------------------------------------------------------------------------------------------------
# load cases: small strains of amplitude strain_amplitude in Voigt order (xx, yy, engineering shear xy).
# Linear elasticity: stresses scale with the amplitude and the mismatch is relative, so the result does not
# depend on it; the stiffness is the homogenized stress divided by the amplitude.
# ----------------------------------------------------------------------------------------------------------------
strain_amplitude = arguments.strain
macro_gradients = strain_amplitude * np.array([[[1.0, 0.0], [0.0, 0.0]],
                                               [[0.0, 0.0], [0.0, 1.0]],
                                               [[0.0, 0.5], [0.5, 0.0]]])   # symmetric: engineering shear
load_case_names = ['xx', 'yy', 'shear']
matched_cases = [load_case_names.index(name) for name in arguments.cases]   # cases in the objective
nb_load_cases = len(macro_gradients)


def voigt_stress(stress_ij):
    return np.array([stress_ij[0, 0], stress_ij[1, 1], stress_ij[0, 1]])


def tensor_stress(stress_voigt):
    return np.array([[stress_voigt[0], stress_voigt[2]], [stress_voigt[2], stress_voigt[1]]])


def engineering_constants(C_voigt):
    """E_x, E_y, G_xy, nu_xy, nu_yx of a 3 x 3 Voigt stiffness (engineering shear)."""
    S = np.linalg.inv(C_voigt)
    return {'E_x': 1 / S[0, 0], 'E_y': 1 / S[1, 1], 'G_xy': 1 / S[2, 2],
            'nu_xy': -S[1, 0] / S[0, 0], 'nu_yx': -S[0, 1] / S[1, 1]}


def orthotropic_stiffness(E_x, E_y, G_xy, nu_xy):
    """3 x 3 Voigt stiffness of an orthotropic material (engineering constants, nu_yx = nu_xy E_y / E_x)."""
    S = np.array([[1 / E_x, -nu_xy / E_x, 0.0],
                  [-nu_xy / E_x, 1 / E_y, 0.0],
                  [0.0, 0.0, 1 / G_xy]])
    return np.linalg.inv(S)


# ----------------------------------------------------------------------------------------------------------------
# fields
# ----------------------------------------------------------------------------------------------------------------
rho_nodal = discretization.get_scalar_field(name='rho')
rho_quad = discretization.get_quad_field_scalar(name='rho_quad')
C_field = discretization.get_material_data_size_field_mugrid(name='C_rho')
macro_gradient_field = discretization.get_gradient_size_field(name='macro_gradient_field')
rhs = discretization.get_unknown_size_field(name='rhs')
x_jacobi = discretization.get_unknown_size_field(name='x_jacobi')
sensitivity = discretization.get_scalar_field(name='sensitivity')
sensitivity_load_case = discretization.get_scalar_field(name='sensitivity_load_case')
# state and adjoint of every load case (warm starts); the robust formulation keeps one set per design
nb_designs = 3 if arguments.robust else 1
displacement_sets = [[discretization.get_unknown_size_field(name=f'u_{d}_{k}') for k in range(nb_load_cases)]
                     for d in range(nb_designs)]
adjoint_sets = [[discretization.get_unknown_size_field(name=f'adjoint_{d}_{k}') for k in range(nb_load_cases)]
                for d in range(nb_designs)]
displacements, adjoints = displacement_sets[0], adjoint_sets[0]         # the active set

green_preconditioner = discretization.get_preconditioner_Green_mugrid(reference_material_data_ijkl=C_solid)

# density filter + projection (--regularization filter): design rho -> physical density rho_bar
use_filter = arguments.regularization == 'filter'
if arguments.objective == 'poisson' and (len(matched_cases) != 3 or arguments.free_scale):
    raise SystemExit('--objective poisson needs all load cases and no --free-scale (it is scale free already)')
if arguments.robust_shape and not arguments.robust:
    raise SystemExit('--robust-shape needs --robust')
if arguments.robust and not use_filter:
    raise SystemExit('--robust needs --regularization filter')
filter_radius = arguments.filter_radius * float(np.max(pixel_size))
if arguments.robust:          # eroded, intermediate (reported), dilated
    density = density_filter.RobustFilterAndProjection(discretization, radius=filter_radius, beta=arguments.beta[0])
elif use_filter:
    density = density_filter.FilterAndProjection(discretization, radius=filter_radius, beta=arguments.beta[0])
else:
    density = None
design_names = ['eroded', 'intermediate', 'dilated'] if arguments.robust else ['design']
jacobi_diagonal = None


def set_design(rho_flat):
    """rho -> SIMP stiffness at the quadrature points and the Jacobi part of the preconditioner."""
    global jacobi_diagonal
    rho_nodal.s[...] = rho_flat.reshape(rho_nodal.s.shape)
    discretization.fft.communicate_ghosts(rho_nodal)
    discretization.apply_N_operator_mugrid(rho_nodal, rho_quad)
    C_field.s[...] = ((C_solid - C_void)[..., np.newaxis, np.newaxis, np.newaxis]
                      * np.power(rho_quad.s, simp_p)[0, 0] + C_void[..., np.newaxis, np.newaxis, np.newaxis])
    jacobi_diagonal = discretization.get_preconditioner_Jacobi_mugrid(material_data_field_ijklqxyz=C_field)


def apply_stiffness(x, Ax):
    discretization.apply_system_matrix_mugrid(material_data_field=C_field, input_field_inxyz=x,
                                              output_field_inxyz=Ax, formulation='small_strain')


def apply_preconditioner(x, Px):
    """--preconditioner: green-jacobi D^1/2 G D^1/2 (G: Green operator of the solid, D = diag(K)^-1, the Jacobi
    field stores D^1/2), green G alone, jacobi D alone."""
    discretization.fft.communicate_ghosts(x)
    if arguments.preconditioner == 'jacobi':
        Px.s[...] = jacobi_diagonal.s ** 2 * x.s
        discretization.fft.communicate_ghosts(Px)
        return
    if arguments.preconditioner == 'green':
        discretization.apply_preconditioner_mugrid(preconditioner_Fourier_fnfnqks=green_preconditioner,
                                                   input_nodal_field_fnxyz=x, output_nodal_field_fnxyz=Px)
        discretization.fft.communicate_ghosts(Px)
        return
    x_jacobi.s[...] = jacobi_diagonal.s * x.s
    discretization.apply_preconditioner_mugrid(preconditioner_Fourier_fnfnqks=green_preconditioner,
                                               input_nodal_field_fnxyz=x_jacobi, output_nodal_field_fnxyz=Px)
    Px.s[...] = jacobi_diagonal.s * Px.s
    discretization.fft.communicate_ghosts(Px)


cg_iterations = {'state': 0, 'adjoint': 0}       # per objective evaluation (reset in objective)


def count_state_iteration(*args):
    cg_iterations['state'] += 1


def solve_load_case(k, tol):
    """Equilibrium for the load case k (warm start from the previous solution); homogenized stress.

    ``tol`` is relative to the right-hand side: |r| <= tol |b| (scale free; with an absolute tolerance below the
    round-off level of |b| CG iterates in noise and returns a wrong state)."""
    discretization.get_macro_gradient_field_mugrid(macro_gradient_ij=macro_gradients[k],
                                                   macro_gradient_field_ijqxyz=macro_gradient_field)
    rhs.s.fill(0)
    discretization.get_rhs_mugrid(material_data_field_ijklqxyz=C_field, macro_gradient_field_ijqxyz=macro_gradient_field,
                                  rhs_inxyz=rhs)
    rhs_norm = np.sqrt(discretization.communicator.sum(np.dot(rhs.s.ravel(), rhs.s.ravel())))
    solvers.conjugate_gradients_mugrid(comm=discretization.communicator, fc=discretization.field_collection,
                                       hessp=apply_stiffness, b=rhs, x=displacements[k], P=apply_preconditioner,
                                       tol=tol * rhs_norm, rtol=False, maxiter=20000, callback=count_state_iteration)
    return discretization.get_homogenized_stress_mugrid(material_data_field_ijklqxyz=C_field,
                                                        displacement_field_inxyz=displacements[k],
                                                        macro_gradient_field_ijqxyz=macro_gradient_field,
                                                        formulation='small_strain')


def homogenized_stiffness(rho_flat):
    """3 x 3 Voigt stiffness of the design: column k = homogenized stress of load case k / strain amplitude."""
    set_design(rho_flat)
    return np.array([voigt_stress(solve_load_case(k, cg_tol_homogenization))
                     for k in range(nb_load_cases)]).T / strain_amplitude


# ----------------------------------------------------------------------------------------------------------------
# objective: stress mismatch of the three load cases + phase-field term, and its gradient (adjoint)
# ----------------------------------------------------------------------------------------------------------------
target_stresses = np.zeros([nb_load_cases, dim, dim])   # set in the driver from the target stiffness
log = {'objective': [], 'mismatch': [], 'phase_field': [], 'scale': [], 'length_scale': [], 'cg_state': [], 'cg_adjoint': []}


def physical_designs(rho_flat):
    """Densities the material sees: [rho] (phase field), [H_beta(F rho)] (filter) or the eroded, intermediate and
    dilated projections (robust)."""
    if not use_filter:
        return [rho_flat]
    designs = density.forward(rho_flat)
    return designs if arguments.robust else [designs]


def physical_density(rho_flat):
    """The reported design: the only one, or the intermediate one of the robust formulation."""
    designs = physical_designs(rho_flat)
    return designs[1] if arguments.robust else designs[0]


def mismatch_and_gradient(rho_physical_flat, design_index=0, free_scale=None):
    """Weighted stress mismatch of one physical design and its gradient w.r.t. that design (adjoint method).

    ``free_scale`` (default ``--free-scale``): match the target up to the optimal stiffness factor.

    Leaves the stiffness of this design in C_field and rho_nodal = rho_physical (used by the phase-field term).
    """
    global displacements, adjoints
    displacements, adjoints = displacement_sets[design_index], adjoint_sets[design_index]
    set_design(rho_physical_flat)
    w = weight / len(matched_cases)
    # state solves: tolerance relative to the right-hand side (warm start safe: stops at once if already converged)
    stresses = {k: solve_load_case(k, cg_tol) for k in matched_cases}
    if arguments.objective == 'poisson':
        return poisson_mismatch_and_gradient(stresses)
    # free stiffness scale (--free-scale): f = sum_k |S_t - t S_h|^2 / |S_t|^2 minimized over t in closed form,
    # t* = sum (S_t : S_h) / |S_t|^2 / sum |S_h|^2 / |S_t|^2. At t* the gradient is the one with t fixed (envelope
    # theorem), which is the plain mismatch with the target S_t / t*: |S_t / t* - S_h|^2 / |S_t / t*|^2.
    scale = 1.0
    if arguments.free_scale if free_scale is None else free_scale:
        numerator = sum(np.sum(target_stresses[k] * stresses[k]) / np.sum(target_stresses[k] ** 2) for k in matched_cases)
        denominator = sum(np.sum(stresses[k] ** 2) / np.sum(target_stresses[k] ** 2) for k in matched_cases)
        scale = max(numerator / denominator, 1e-12)
    f_mismatch = 0.0
    gradient = np.zeros(rho_nodal.s[0, 0].shape)
    for k in matched_cases:
        target = target_stresses[k] / scale
        f_mismatch += w * topology_optimization.compute_stress_equivalence_potential(
            actual_stress_ij=stresses[k], target_stress_ij=target)
        # gradient of w * f_sigma (explicit part + adjoint part); the returned adjoint energy lambda^T (K u - f) is
        # zero at the solution and is NOT added to the objective (with inexact solves it only adds noise)
        discretization.get_macro_gradient_field_mugrid(macro_gradient_ij=macro_gradients[k],
                                                       macro_gradient_field_ijqxyz=macro_gradient_field)
        # adjoint: warm start, tolerance relative to the right-hand side (r_tol). Until 2026-10-10 the library
        # measured it relative to the residual of the warm start, which near the solution is round-off: CG then
        # returned wrong adjoints (gradients off by up to 10x after short line-search steps; robust runs stalled).
        sensitivity_load_case.s[0, 0], adjoints[k], adjoint_energy, adjoint_info = \
            topology_optimization.sensitivity_stress_and_adjoint_FE_NEW(
                discretization=discretization, base_material_data_ijkl=C_solid, void_material_data_ijkl=C_void,
                displacement_field_inxyz=displacements[k], adjoint_field_inxyz=adjoints[k],
                macro_gradient_field_ijqxyz=macro_gradient_field, phase_field_1nxyz=rho_nodal,
                target_stress_ij=target, actual_stress_ij=stresses[k],
                preconditioner_fun=apply_preconditioner, system_matrix_fun=apply_stiffness,
                formulation='small_strain', p=simp_p, weight=w, cg_tol=cg_tol, r_tol=True)
        cg_iterations['adjoint'] += adjoint_info.get('num_iteration_adjoint', 0)
        gradient += sensitivity_load_case.s[0, 0]
    return f_mismatch, gradient.reshape(-1), scale


def stress_functional_gradient(k, stress, A):
    """d(A : Sigma_k)/drho for a fixed 2 x 2 tensor A (adjoint method), at the state of load case k.

    The library sensitivity of weight |S_t - Sigma|^2 / |S_t|^2 at fixed S_t is weight (-2 / |S_t|^2) (S_t - Sigma) :
    dSigma/drho; with S_t = Sigma + A and weight = -|S_t|^2 / 2 it is A : dSigma/drho.
    """
    target = stress + A
    discretization.get_macro_gradient_field_mugrid(macro_gradient_ij=macro_gradients[k],
                                                   macro_gradient_field_ijqxyz=macro_gradient_field)
    sensitivity_load_case.s[0, 0], adjoints[k], _, adjoint_info = topology_optimization.sensitivity_stress_and_adjoint_FE_NEW(
        discretization=discretization, base_material_data_ijkl=C_solid, void_material_data_ijkl=C_void,
        displacement_field_inxyz=displacements[k], adjoint_field_inxyz=adjoints[k],
        macro_gradient_field_ijqxyz=macro_gradient_field, phase_field_1nxyz=rho_nodal, target_stress_ij=target,
        actual_stress_ij=stress, preconditioner_fun=apply_preconditioner, system_matrix_fun=apply_stiffness,
        formulation='small_strain', p=simp_p, weight=-0.5 * np.sum(target ** 2), cg_tol=cg_tol, r_tol=True)
    cg_iterations['adjoint'] += adjoint_info.get('num_iteration_adjoint', 0)
    return sensitivity_load_case.s[0, 0].copy()


def poisson_mismatch_and_gradient(stresses):
    """--objective poisson: f = weight (nu_xy(C) - nu)^2, nu_xy = -S_12 / S_11 with S = C^-1 (C from all load cases).

    dnu/dC_cd = (S_1c S_d2 S_11 - S_12 S_1c S_d1) / S_11^2 (dS = -S dC S); C_cd = Sigma^d_c / strain, so
    df/dSigma^d = A_d with A_d = df/dC_.d / strain in tensor form (the shear entry halved: engineering shear).
    """
    C = np.array([voigt_stress(stresses[k]) for k in range(nb_load_cases)]).T / strain_amplitude
    S = np.linalg.inv(C)
    nu_xy = -S[0, 1] / S[0, 0]
    f = weight * (nu_xy - arguments.nu) ** 2
    dnu_dC = (np.outer(S[0, :], S[:, 1]) * S[0, 0] - S[0, 1] * np.outer(S[0, :], S[:, 0])) / S[0, 0] ** 2
    df_dC = 2.0 * weight * (nu_xy - arguments.nu) * dnu_dC
    gradient = np.zeros(rho_nodal.s[0, 0].shape)
    for k in range(nb_load_cases):
        column = df_dC[:, k] / strain_amplitude
        A = np.array([[column[0], 0.5 * column[2]], [0.5 * column[2], column[1]]])
        if np.any(A != 0):
            gradient += stress_functional_gradient(k, stresses[k], A)
    return f, gradient.reshape(-1), 1.0


def objective(rho_flat):
    cg_iterations['state'] = cg_iterations['adjoint'] = 0
    designs = physical_designs(rho_flat)
    # --robust-shape: eroded (0) and dilated (2) designs with a free stiffness factor, the intermediate one absolute
    results = [mismatch_and_gradient(design, d, free_scale=True if arguments.robust_shape and d != 1 else None)
               for d, design in enumerate(designs)]
    if arguments.robust:
        # mean over the eroded, intermediate and dilated designs
        weights = np.full(len(designs), 1.0 / len(designs))
        f_mismatch = float(np.dot(weights, [r[0] for r in results]))
        gradient = density.backward([r[1] for r in results], weights=weights)
    else:
        f_mismatch, gradient_physical, _ = results[0]
        gradient = density.backward(gradient_physical) if use_filter else gradient_physical.copy()

    f_phase_field = 0.0
    if not use_filter:            # phase-field term of rho (= rho_nodal, set by mismatch_and_gradient)
        f_phase_field = topology_optimization.objective_function_phase_field(
            discretization=discretization, phase_field_1nxyz=rho_nodal, eta=eta, double_well_depth=double_well_depth)
        sensitivity.s.fill(0)
        topology_optimization.sensitivity_phase_field_term_FE_NEW(
            discretization=discretization, phase_field_1nxyz=rho_nodal, p=simp_p, eta=eta, output_array=sensitivity,
            double_well_depth=double_well_depth)
        discretization.fft.communicate_ghosts(sensitivity)
        gradient = gradient + sensitivity.s[0, 0].reshape(-1)

    # geometric length scale constraints (--length-scale) on the filtered field F rho, as a penalty
    f_length_scale = 0.0
    if use_filter and arguments.length_scale > 0 and density.beta >= arguments.length_scale_beta:
        g_solid, g_void, dg_solid, dg_void = density_filter.length_scale_constraints(
            density._rho_tilde.reshape(rho_nodal.s.shape), radius_pixels=arguments.filter_radius, beta=density.beta)
        f_length_scale = arguments.length_scale * (g_solid + g_void)
        gradient = gradient + density.filter.apply_transpose(
            arguments.length_scale * (dg_solid + dg_void)).reshape(gradient.shape)

    f = f_phase_field + f_mismatch + f_length_scale
    scale = results[0][2]
    log['objective'].append(f)
    log['mismatch'].append(f_mismatch)
    log['phase_field'].append(f_phase_field)
    log['scale'].append(scale)
    log['length_scale'].append(f_length_scale)
    log['cg_state'].append(cg_iterations['state'])
    log['cg_adjoint'].append(cg_iterations['adjoint'])
    extra = ('  [' + ', '.join(f'{n} {r[0]:.3e}' for n, r in zip(design_names, results)) + ']') if arguments.robust else ''
    root_print(f'  F {f:.6e}  (mismatch {f_mismatch:.3e}, phase field {f_phase_field:.3e}, length scale {f_length_scale:.3e}, scale {scale:.4g}, CG state {cg_iterations["state"]} adjoint {cg_iterations["adjoint"]}){extra}')
    return f, gradient


# ----------------------------------------------------------------------------------------------------------------
# output helpers
# ----------------------------------------------------------------------------------------------------------------
def format_properties(name, C_voigt):
    p = engineering_constants(C_voigt)
    return (f'{name}:\n  C (Voigt, engineering shear) =\n'
            + '\n'.join('    ' + '  '.join(f'{c: .5f}' for c in row) for row in C_voigt)
            + f'\n  E_x {p["E_x"]:.5f}  E_y {p["E_y"]:.5f}  G_xy {p["G_xy"]:.5f}'
              f'  nu_xy {p["nu_xy"]:.4f}  nu_yx {p["nu_yx"]:.4f}\n')


def save_design_figure(path, designs):
    """designs: list of (title, nodal density) -> 3 x 3 cells each."""
    if comm.size > 1:
        return
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, len(designs), figsize=(4.2 * len(designs), 4.4), squeeze=False)
    for ax, (title, rho) in zip(axes[0], designs):
        ax.imshow(np.tile(rho, (3, 3)).T, origin='lower', cmap='Greys', vmin=0, vmax=1)
        ax.set_title(title, fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


# ----------------------------------------------------------------------------------------------------------------
# driver
# ----------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    # 1. the bow-tie design (nodal densities; coords are fractional because the cell is the unit square)
    coords = np.asarray(discretization.fft.coords) / np.asarray(domain_size)[:, None, None]
    rho_0 = geometry.get('reentrant_honeycomb', coords, theta=arguments.theta, thickness=arguments.thickness)
    rho_0 = rho_0.astype(float)
    rho_bowtie = rho_0
    if arguments.init == 'original':          # initial guess of the original example (fixed seed: reproducible)
        np.random.seed(comm.rank)
        rho_0 = (np.sin(4 * np.pi * coords[0]) + np.sin(4 * np.pi * coords[1]) + 2) / 4 \
                + 0.5 * np.random.rand(*coords[0].shape)
        rho_0 = np.clip(rho_0, 0.0, 1.0)

    # 2. its homogenized properties
    C_0 = homogenized_stiffness(rho_bowtie.reshape(-1))
    if arguments.init == 'original':
        C_initial_guess = homogenized_stiffness(rho_0.reshape(-1))
    properties_start = engineering_constants(C_initial_guess if arguments.init == 'original' else C_0)
    start_name = 'initial (bow-tie)' if arguments.init == 'bowtie' else 'initial guess (original example)'
    properties_0 = engineering_constants(C_0)
    report = (f'bow-tie: theta {arguments.theta:g} deg, thickness {arguments.thickness:g}, {arguments.n}^2 pixels, '
              f'solid fraction {comm.allreduce(rho_0.sum()) / np.prod(number_of_pixels):.3f}\n'
              + format_properties('initial design (bow-tie)', C_0))
    root_print(report)

    tag = f'theta{arguments.theta:g}_t{arguments.thickness:g}_N{arguments.n}'
    if arguments.init == 'original':
        report += format_properties('initial guess of the original example', C_initial_guess)
    if arguments.evaluate_only:
        folder = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data',
                              f'bowtie_properties_{tag}_' + time.strftime('%Y%m%d-%H%M%S'))
        if comm.rank == 0:
            os.makedirs(folder)
            open(os.path.join(folder, 'properties.txt'), 'w').write(report)
        save_design_figure(os.path.join(folder, 'design.png'),
                           [(f'bow-tie, nu_xy {properties_0["nu_xy"]:.3f}, nu_yx {properties_0["nu_yx"]:.3f}', rho_0)])
        root_print(f'wrote {folder}')
        sys.exit(0)

    # 3. target: moduli of the bow-tie, Poisson ratio --nu
    if arguments.nu is None:
        raise SystemExit('give a target Poisson ratio with --nu (or use --evaluate-only)')
    if arguments.keep == 'isotropic':        # 2D isotropic: G = --shear-fraction E_solid (3/20 in the original example)
        E_solid = 9 * K_solid * G_solid / (3 * K_solid + G_solid)   # and nu_xy = --nu exactly (the original example
        G_target = arguments.shear_fraction * E_solid               # converts with 3D formulas: nu 2D differs)
        C_11 = 2 * G_target / (1 - arguments.nu)                    # C_11 = E/(1 - nu^2) with E = 2 G (1 + nu)
        C_target = np.array([[C_11, arguments.nu * C_11, 0.0], [arguments.nu * C_11, C_11, 0.0], [0.0, 0.0, G_target]])
    elif arguments.keep == 'moduli':
        C_target = orthotropic_stiffness(properties_0['E_x'], properties_0['E_y'], properties_0['G_xy'], arguments.nu)
    else:
        C_target = np.diag(np.diag(C_0))                # C_11, C_22, C_66 of the bow-tie, no normal-shear coupling
        C_target[0, 1] = C_target[1, 0] = arguments.nu * C_0[1, 1]
    if np.min(np.linalg.eigvalsh(C_target)) <= 0:
        raise SystemExit(f'target with nu_xy = {arguments.nu} (--keep {arguments.keep}) is not positive definite')
    for k in range(nb_load_cases):
        target_stresses[k] = tensor_stress(strain_amplitude * C_target[:, k])
    report += format_properties(f'target ({f"isotropic, G = {arguments.shear_fraction:g} E_solid" if arguments.keep == "isotropic" else arguments.keep + " of the bow-tie kept"}, nu_xy = {arguments.nu:g})', C_target)
    root_print(format_properties('target', C_target))

    if use_filter:
        regularization_tag = f'{("robustshape" if arguments.robust_shape else "robust") if arguments.robust else "filter"}R{arguments.filter_radius:g}{f"_ls{arguments.length_scale:g}" if arguments.length_scale > 0 else ""}'
    else:
        regularization_tag = f'dw{double_well_depth:g}_eta{arguments.eta:g}'
    folder = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data',
                          f'bowtie_TO_{tag if arguments.init == "bowtie" else "init-original_N" + str(arguments.n)}_nu{arguments.nu:g}_{"poisson" if arguments.objective == "poisson" else arguments.keep}{f"{arguments.shear_fraction:.3g}" if arguments.keep == "isotropic" else ""}_w{weight:g}_{regularization_tag}_p{simp_p:g}_s{strain_amplitude:g}{"" if len(matched_cases) == 3 else "_" + "-".join(arguments.cases)}{"_free" if arguments.free_scale else ""}{"" if arguments.preconditioner == "green-jacobi" else "_" + arguments.preconditioner}_'
                          + time.strftime('%Y%m%d-%H%M%S'))
    if comm.rank == 0:
        os.makedirs(folder)
    comm.Barrier()
    parameters = {'nb_of_pixels': number_of_pixels, 'domain_size': domain_size, 'element_type': element_type,
                  'K_solid': K_solid, 'G_solid': G_solid, 'void_contrast': void_contrast, 'simp_p': simp_p,
                  'eta': eta, 'weight': weight, 'double_well_depth': double_well_depth, 'theta': arguments.theta,
                  'thickness': arguments.thickness, 'strain_amplitude': strain_amplitude, 'matched_cases': matched_cases, 'free_scale': int(arguments.free_scale), 'regularization': arguments.regularization,
                  'filter_radius_pixels': arguments.filter_radius, 'robust': int(arguments.robust), 'robust_shape': int(arguments.robust_shape), 'objective': arguments.objective, 'preconditioner': arguments.preconditioner, 'length_scale': arguments.length_scale, 'length_scale_beta': arguments.length_scale_beta, 'beta_schedule': arguments.beta, 'nu_target': arguments.nu, 'keep': arguments.keep, 'init': arguments.init, 'shear_fraction': arguments.shear_fraction, 'gtol': arguments.gtol, 'C_initial_voigt': C_initial_guess if arguments.init == 'original' else C_0,
                  'C_target_voigt': C_target, 'target_stresses': target_stresses}

    # 4. optimization (filter: one L-BFGS run per beta of the continuation, each from the previous design)
    rho_iterate = discretization.get_scalar_field(name='rho_iterate')           # design variable
    rho_physical = discretization.get_scalar_field(name='rho_physical')         # density the material sees
    history = io_utils.FieldWriter(os.path.join(folder, 'history.nc'), [rho_iterate, rho_physical],
                                   attributes=parameters, frame_variables={'objective': (), 'mismatch': ()})
    iteration = 0

    def callback(x):
        global iteration
        iteration += 1
        rho_iterate.s[0, 0] = x.reshape(rho_iterate.s.shape[2:])
        rho_physical.s[0, 0] = physical_density(x).reshape(rho_iterate.s.shape[2:])
        history.write(objective=log['objective'][-1], mismatch=log['mismatch'][-1])
        if iteration % 10 == 0:
            save_design_figure(os.path.join(folder, f'design_it{iteration:04d}.png'),
                               [(start_name, rho_0), (f'iteration {iteration}', rho_physical.s[0, 0])])

    start = time.time()
    # the optimizer sees F / F(rho_0) (scale of the first stage, kept for all stages)
    objective_scale = 1.0 / objective(rho_0.reshape(-1))[0]

    def scaled_objective(x):
        f, g = objective(x)
        return objective_scale * f, objective_scale * g

    stages = arguments.beta if use_filter else [None]
    xtol = arguments.xtol if arguments.xtol is not None else (0.0 if use_filter else 1e-3)
    x = rho_0.reshape(-1).copy()
    for beta in stages:
        if use_filter:
            density.beta = beta
            root_print(f'--- beta {beta:g} ---')
        result = Optimization.l_bfgs_bounded(fun=scaled_objective, x0=x, args=(), jac=True, bounds_lo=0.0,
                                             bounds_hi=1.0, gtol=arguments.gtol, xtol=xtol,
                                             maxiter=arguments.stage_iterations if use_filter else 500, maxcor=20,
                                             c1=1e-4, max_halvings=40, comm=comm, callback=callback, disp=True)
        x = result.x
        if use_filter:
            grey = density_filter.measure_of_non_discreteness(physical_density(x), discretization.communicator)
            root_print(f'beta {beta:g}: {iteration} iterations so far, mismatch {log["mismatch"][-1]:.3e}, '
                       f'grey measure M_nd {grey:.3f}')
    history.close()
    elapsed = time.time() - start

    # 5. properties of the optimized (physical) design, figures and data
    rho_opt = physical_density(x).reshape(rho_0.shape)
    C_opt = homogenized_stiffness(rho_opt.reshape(-1))
    properties_opt = engineering_constants(C_opt)
    grey_final = density_filter.measure_of_non_discreteness(rho_opt, discretization.communicator)
    report += format_properties(f'optimized design ({iteration} iterations, {elapsed:.0f} s, '
                                f'grey measure M_nd {grey_final:.3f})', C_opt)
    root_print(report)
    if comm.rank == 0:
        open(os.path.join(folder, 'properties.txt'), 'w').write(report)
    save_design_figure(os.path.join(folder, 'design.png'),
                       [(f'initial: nu_xy {properties_start["nu_xy"]:.3f}, nu_yx {properties_start["nu_yx"]:.3f}', rho_0),
                        (f'optimized: nu_xy {properties_opt["nu_xy"]:.3f}, nu_yx {properties_opt["nu_yx"]:.3f}'
                         f' (target {arguments.nu:g})', rho_opt)])
    rho_final = discretization.get_scalar_field(name='rho_final')                # physical density
    rho_final.s[0, 0] = rho_opt
    rho_design = discretization.get_scalar_field(name='rho_design')              # design variable
    rho_design.s[0, 0] = x.reshape(rho_0.shape)
    io_utils.save_fields(os.path.join(folder, 'final.nc'), [rho_final, rho_design], attributes={
        **parameters, 'nb_iterations': iteration, 'elapsed': elapsed, 'C_optimized_voigt': C_opt,
        'grey_measure': grey_final, 'log_objective': log['objective'], 'log_mismatch': log['mismatch'],
        'log_phase_field': log['phase_field'], 'log_scale': log['scale'], 'log_length_scale': log['length_scale'],
        'log_cg_state': log['cg_state'], 'log_cg_adjoint': log['cg_adjoint']})
    root_print(f'wrote {folder}')
