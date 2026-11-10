import time
import os
import sys
import argparse

import numpy as np
from mpi4py import MPI
from NuMPI.IO import save_npy

# Add the project root to sys.path relatively
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from muGrid import Solvers
from muFFTTO import domain
from muFFTTO import solvers
from muFFTTO import microstructure_library
from muFFTTO import visualization_utils

# Numerical settings ============================
NX = 64                     # Arbitrary number of pixels (default of -n; NY follows from the cell)
N_inc = 10                  # number of load steps (nominal; a failed trust-region solve halves the step)
NEWTON_TOL = 5e-7           # only for the small-strain probe of NU_LAT / E_eff below (|r| / |r0| of its Newton solve)
E_LOAD = 1                  # scales the total macro load below (finite strain: response is NOT linear in E_LOAD)
plot_load_cases = False     # plot the deformed cell after every load step
plot_fig4 = True           # plot nominal stress vs nominal strain at the end (comparison with paper Fig. 4)
MATERIAL = 'paper'          # 'paper' = Li & Rudykh (2019): contrast 1000, Lambda/mu 1000
                            # 'simplified' = contrast 10, Lambda/mu 100 (Martin's report: -3.5 % before buckling,
                            #                <= 1 % after, at 128 px; much cheaper CG). NOT the paper's material!
INTEGRATION = 'full'         # 'sri'  = selective reduced integration (Martin --sri): mu part at the 4 Gauss points,
                            #          lambda part once per pixel (J0 = det of the pixel-mean F) -> no volumetric locking
                            # 'full' = the same code with J0 -> J at every Gauss point = standard NeoHookean (bilinear
                            #          LOCKS for Lambda/mu = 1000)
SRI_TANGENT = 'exact'       # 'exact' = consistent tangent (Martin, lines 258-263); indefinite under compression, which
                            #           the trust region handles (Steihaug CG stops at negative curvature)
                            # 'spd'   = without the initial-stress part of the lam term (same solution, other path)
VOID        = 'hole'        # 'hole' = true hole: mu = lam = 0 and F = I inside (as Martin and the paper)
                            # 'soft' = stand-in mu = 1e-4 mu_m, lam = 0 (Martin --void_mu 1e-4)
LATERAL     = 'exact'       # 'exact'  = F_ll is an unknown of the energy minimization -> <P_ll> = 0 (as Martin)
                            # 'simple' = F_ll = 1 + NU_LAT eps prescribed (lateral stress up to ~1.6 % of axial)
# Trust-region Newton-CG (as Martin's stefanus_2D_finite_strain_NeoHookean_TR.py; needs NuMPI branch feat/tr_precond)
GTOL          = 1e-4        # Martin --gtol: stop when max|gradient| < GTOL mu_m h d_eps (nodal force of one load step)
TR_DELTA0     = 0.05        # Martin --delta0: initial trust-region radius (M-norm)
TR_DELTA_MAX  = 0.5         # Martin --delta_max
TR_INNER_TOL  = 1e-6        # Martin --inner_tol: Steihaug CG relative energy-error tolerance
TR_INNER_TAU  = 0.25        # Martin --inner_tau: adaptive delay of the energy-error estimate (minimum delay 10)
TR_MAXITER    = 500         # Martin --tr_maxiter: trust-region iterations per load step (then: halve the step)
# Load case: Li & Rudykh (2019) Fig. 4 = uniaxial COMPRESSION, lateral side free =============
LOAD_MODE = 'fig4'                              # 'fig4' = uniaxial compression below, 'manual' = your own load_pattern
LOAD_DIRECTION = 'Y'                            # 'X' = Fig. 4(a), 'Y' = Fig. 4(b)
EPS_MAX = {'X': 0.016, 'Y': 0.026}              # final nominal strain = critical strains eps_cr (paper Sec. 3; Martin's
                                                # docstring) = Martin's --eps_max (also sets gtol via d_eps)
Y_MAX_MPA = {'X': 0.015, 'Y': 0.026}            # y-axis top of the Fig. 4 plot [MPa] (same in the small-strain script)
# Small-strain probe: nu = small-strain Poisson ratio of the cell (used by LATERAL = 'simple') and E_eff (printed).
# NU_MODE = 'auto': computed in this script at F = I (2 linear solves) for exactly this NX, element, MATERIAL and
# INTEGRATION. 'table': the values below (small-strain script: bilinear, full integration only)
NU_MODE = 'auto'
NU_LAT_TABLE = {('paper',      64):  {'X': 0.2575, 'Y': 0.2797},
                ('paper',      128): {'X': 0.2990, 'Y': 0.3479},
                ('simplified', 64):  {'X': 0.3269, 'Y': 0.3933},
                ('simplified', 128): {'X': 0.3383, 'Y': 0.4217}}
PAPER_SLOPE_MPA = {'X': {'Exp.': 0.32, 'Sim. periodic': 0.35},   # initial slopes digitised from Fig. 4,
                   'Y': {'Exp.': 0.46, 'Sim. periodic': 0.50}}   # approx. +-5 %
# ====================================================================================================================================================

# ============================================================================
# argument parsing
# ============================================================================
parser = argparse.ArgumentParser(
    prog='exp_finite_strain_2D_NeoHookean.py',
    description='Solve finite strain NeoHookean elasticity in 2D'
)
parser.add_argument('-n',   '--nb_pixel',       default=str(NX))
parser.add_argument('-inc', '--nb_increments',  default=str(N_inc))
parser.add_argument(
    '--save_per_it',
    action='store_false',
    help='by default the fields of every converged load step are saved; this flag switches it off'
)


script_name = os.path.splitext(os.path.basename(__file__))[0]
args        = parser.parse_args()
nnn         = int(args.nb_pixel)
ninc        = int(args.nb_increments)
save_per_it =  args.save_per_it
# ============================================================================
# problem setup
# ============================================================================

# Unit Cells input: -- PHYSICAL units ===================
R_MM         = 3.0              # void & inclusion radius r                      [mm]
MU_M_MPA     = 0.6              # matrix shear modulus mu^(m) (FLX9860)           [MPa]
# MU_I_MPA     = 600.0            # stiff inclusion shear modulus mu^(i) (VeroWhite) [MPa]
# LAM_OVER_MU  = 1e3              # Lambda / mu, matrix and inclusion (Sec. 3, after Eq. 1) [-]
MU_I_MPA     = {'paper': 600.0, 'simplified': 6.0}[MATERIAL]     # inclusion shear modulus mu^(i) [MPa] (paper: 600)
LAM_OVER_MU  = {'paper': 1e3,   'simplified': 100.0}[MATERIAL]   # Lambda / mu, matrix and inclusion [-] (paper: 1000)
C_MATRIX     = 0.25             # target matrix volume fraction c^(m) ratio            [-]
SPECIMEN_MM  = (92.37, 91.42)   # specimen width x height (finite sample, check only) [mm]
THICKNESS_MM = 10.0             # out-of-plane thickness -> plane strain, not used in 2D [mm]
MU_V_MPA     = 1e-4 * MU_M_MPA if VOID == 'soft' else 0.0   # void: stand-in [MPa], or 0 for the true hole

# Reference scales (nondimensionalisation) ====================================================
L_REF = R_MM        # length scale  [mm]  -> lengths in units of r
S_REF = MU_M_MPA    # stress scale  [MPa] -> stresses / moduli in units of mu^(m)

# Nondimensional quantities used by the solver (derived, do not edit) =========================
c_matrix = C_MATRIX
r = R_MM / L_REF                        # = 1
CONTRAST = MU_I_MPA / S_REF             # = 1000
MU_VOID = MU_V_MPA / S_REF              # 'soft': 1e-4; 'hole': 0 (allowed: F = I is set inside the hole)
incl_vol_frac = 1.0 - c_matrix          # total circle fraction (voids + stiff)

# Domain: TRIANGULAR lattice (paper Fig. 1a), centre spacing a, periodic cell 2a x sqrt(3)a
a = r * np.sqrt(2 * np.pi / (np.sqrt(3) * incl_vol_frac))
ligament = a - 2 * r                                            # thinnest matrix ligament (void-void)

domain_size      = [2 * a, np.sqrt(3) * a]
number_of_pixels = (nnn, int(round(nnn * domain_size[1] / domain_size[0])))   # square pixels
dim              = len(domain_size)
h_x, h_y = domain_size[0] / number_of_pixels[0], domain_size[1] / number_of_pixels[1]   # actual sizes

if MPI.COMM_WORLD.rank == 0:
    print(f'a = {a * L_REF:.3f} mm | ligament = {ligament * L_REF:.3f} mm | '
          f'pixel = {h_x * L_REF:.4f} x {h_y * L_REF:.4f} mm | '
          f'{ligament / max(h_x, h_y):.1f} px across ligament | grid = {number_of_pixels}')


problem_type        = 'elasticity'
discretization_type = 'finite_element'
element_type        = 'bilinear_rectangle' #
formulation         = 'finite_strain'
preconditioner_type = 'Green'

_info = {
    'problem_type':        problem_type,
    'discretization_type': discretization_type,
    'element_type':        element_type,
    'formulation':         formulation,
    'preconditioner_type': preconditioner_type,
    'nb_of_pixels':        number_of_pixels,
    'domain_size':         domain_size,
    'ninc':                ninc,
}

# ============================================================================
# discretization
# ============================================================================
my_cell = domain.PeriodicUnitCell(domain_size=domain_size,
                                  problem_type=problem_type)

discretization = domain.Discretization(cell=my_cell,
                                       nb_of_pixels_global=number_of_pixels,
                                       discretization_type=discretization_type,
                                       element_type=element_type)

# ============================================================================
# output folders
# ============================================================================
file_folder_path   = os.path.dirname(os.path.realpath(__file__))
data_folder_path   = (file_folder_path + '/exp_data/' + script_name + '/'
                      + f'Nx={number_of_pixels[0]}Ny={number_of_pixels[1]}_{preconditioner_type}_{LOAD_DIRECTION}_{MATERIAL}_{INTEGRATION}/')
figure_folder_path = (file_folder_path + '/figures/' + script_name + '/'
                      + f'Nx={number_of_pixels[0]}Ny={number_of_pixels[1]}_{preconditioner_type}_{LOAD_DIRECTION}_{MATERIAL}_{INTEGRATION}/')

if discretization.communicator.rank == 0:
    os.makedirs(data_folder_path,   exist_ok=True)
    os.makedirs(figure_folder_path, exist_ok=True)

# ============================================================================
# material parameters
# ============================================================================
# matrix: soft neo-Hookean
# nondimensional Lame parameters (plane strain uses the 3D lambda), same as the small-strain script
# matrix: soft neo-Hookean (FLX9860)
mu_matrix  = 1.0
lam_matrix = LAM_OVER_MU * mu_matrix

# inclusion: stiff neo-Hookean (VeroWhite)
mu_inc  = CONTRAST
lam_inc = LAM_OVER_MU * mu_inc

# void stand-in (compressible)
mu_void  = MU_VOID
lam_void = 0.0

# reference material for Green preconditioner
i    = np.eye(dim)
II   = np.einsum('ij,kl->ijkl', i, i)
I4rt = np.einsum('ik,jl->ijkl', i, i)
I4s  = 0.5 * (I4rt + np.einsum('il,jk->ijkl', i, i))
I4d  = I4s - II / 2.0       # 2D: divide by dim=2 not 3
ref_mat = lam_matrix * II + 2.0 * mu_matrix * I4s

_info['lam_matrix'] = lam_matrix
_info['mu_matrix']  = mu_matrix
_info['lam_inc']    = lam_inc
_info['mu_inc']     = mu_inc
_info['mu_void']    = mu_void
_info['E_LOAD']     = E_LOAD

# ============================================================================
# geometry — Composite A (0 = matrix, 1 = void, 2 = stiff inclusion)
# ============================================================================
geometry_ID = 'geometry_stefanus'

phase_field = discretization.get_scalar_field(name='phase_field')
phase_field.s[0, 0] = microstructure_library.get_geometry(
    nb_voxels=discretization.nb_of_pixels,
    microstructure_name=geometry_ID,
    coordinates=discretization.fft.coords,
    vol_frac=incl_vol_frac,
    domain_size=domain_size
)

matrix_mask = phase_field.s[0, 0] == 0
void_mask   = phase_field.s[0, 0] == 1
inc_mask    = phase_field.s[0, 0] == 2

# ============================================================================
# material fields
# ============================================================================
lam_field = discretization.get_quad_field_scalar(name='lam_field')
mu_field  = discretization.get_quad_field_scalar(name='mu_field')

lam_field.s[0, 0, :, matrix_mask] = lam_matrix
lam_field.s[0, 0, :, void_mask]   = lam_void
lam_field.s[0, 0, :, inc_mask]    = lam_inc
mu_field.s[0, 0,  :, matrix_mask] = mu_matrix
mu_field.s[0, 0,  :, void_mask]   = mu_void
mu_field.s[0, 0,  :, inc_mask]    = mu_inc

if discretization.communicator.rank == 0:
    print(f'lam min/max : {lam_field.s.min():.4f} / {lam_field.s.max():.4f}')
    print(f'mu  min/max : {mu_field.s.min():.4f}  / {mu_field.s.max():.4f}')
    print(f'material: {MATERIAL} | contrast mu_i/mu_m = {CONTRAST:g} | Lambda/mu = {LAM_OVER_MU:g} | '
          f'integration: {INTEGRATION} | element: {element_type}')

# ============================================================================
# fields
# ============================================================================
macro_gradient_inc_field       = discretization.get_gradient_size_field(name='macro_gradient_inc_field')
displacement_fluctuation_field = discretization.get_unknown_size_field(name='displacement_fluctuation_field')
displacement_increment_field   = discretization.get_unknown_size_field(name='displacement_increment_field')
strain_fluc_field              = discretization.get_displacement_gradient_sized_field(name='strain_fluctuation_field')
total_strain_field             = discretization.get_displacement_gradient_sized_field(name='total_strain_field')
stress_field                   = discretization.get_displacement_gradient_sized_field(name='stress_field')
rhs_field                      = discretization.get_unknown_size_field(name='rhs_field')
energy_field                   = discretization.get_quad_field_scalar(name='energy_field')
u_unit_cell                    = discretization.get_unknown_size_field(name='u_unit_cell')

# ============================================================================
# initialize F = I (reference configuration)
# ============================================================================
total_strain_field.s[...] = 0.0
for d in range(dim):
    total_strain_field.s[d, d] = 1.0

# ============================================================================
# selective reduced integration (SRI) for bilinear pixels  (INTEGRATION = 'sri')
# ============================================================================
#   W = mu/2 (F:F - 2) - mu ln J    at each of the 4 Gauss points    (shear part: full integration)
#     + lam/2 (ln J0)^2             once per pixel, J0 = det F0      (volumetric part: reduced integration)
#   F0 = mean of F over the 4 Gauss points of the pixel (= F at the pixel centre for bilinear pixels)
#   P_q  = mu (F_q - F_q^-T) + lam ln J0 F0^-T                        (lam part identical at the 4 points)
#   dP_q = mu (dF_q + F_q^-T dF_q^T F_q^-T) + lam [(F0^-T : dF0) F0^-T - ln J0 F0^-T dF0^T F0^-T]
dF_field  = discretization.get_displacement_gradient_sized_field(name='dF_field')     # dF = grad(x) in K_fun
dP_field  = discretization.get_displacement_gradient_sized_field(name='dP_field')     # dP in K_fun
sri_state = {}                                                                          # F^-T, F0^-T, ln J0 of the last update


def det2(M):
    """det of 2x2 tensors stored as (2, 2, ...)"""
    return M[0, 0] * M[1, 1] - M[0, 1] * M[1, 0]


def inv_T(M):
    """M^-T of 2x2 tensors stored as (2, 2, ...)"""
    return np.stack([np.stack([M[1, 1], -M[1, 0]]), np.stack([-M[0, 1], M[0, 0]])]) / det2(M)


def ddot(A, B):
    """A : B for 2x2 tensors (2, 2, ...)"""
    return np.einsum('ij...,ij...->...', A, B)


def A_dBT_C(A, dB, C):
    """A dB^T C for 2x2 tensors (2, 2, ...)"""
    return np.einsum('ik...,lk...,lj...->ij...', A, dB, C)


def sri_update(F_field, P_field, W_field):
    """SRI stress P and energy W at the deformation gradient F_field; keeps what K_fun needs in sri_state"""
    F   = F_field.s                                   # (2, 2, q, x, y)
    F0  = np.broadcast_to(F.mean(axis=2, keepdims=True), F.shape) if INTEGRATION == 'sri' else F   # 'sri': pixel mean of F
    J, J0 = det2(F), det2(F0)
    J_min = MPI.COMM_WORLD.allreduce(float(min(J.min(), J0.min())), op=MPI.MIN)
    if J_min <= 0.0:
        raise RuntimeError(f'SRI: inverted element (min det F = {J_min:.3e}): use smaller load increments')
    lam, mu = lam_field.s[0, 0], mu_field.s[0, 0]     # (q, x, y), constant within a pixel
    FiT, F0iT, lnJ0 = inv_T(F), inv_T(F0), np.log(J0)
    P_field.s[...]   = mu * (F - FiT) + lam * lnJ0 * F0iT
    W_field.s[0, 0]  = 0.5 * mu * (ddot(F, F) - dim) - mu * np.log(J) + 0.5 * lam * lnJ0 ** 2
    sri_state.update(FiT=FiT, F0iT=F0iT, lnJ0=lnJ0, lam=lam, mu=mu)


def sri_dP(dF, st):
    """linearised SRI stress dP for the gradient increment dF (2, 2, q, x, y) at the state st"""
    dF0 = np.broadcast_to(dF.mean(axis=2, keepdims=True), dF.shape) if INTEGRATION == 'sri' else dF
    return (st['mu'] * (dF + A_dBT_C(st['FiT'], dF, st['FiT']))
            + st['lam'] * (ddot(st['F0iT'], dF0) * st['F0iT']
                           - (st['lnJ0'] * A_dBT_C(st['F0iT'], dF0, st['F0iT']) if SRI_TANGENT == 'exact' else 0.0)))


def sri_reference_operator(input_field_inxyz, output_field_inxyz):
    """SRI system matrix at F = I for the matrix material (homogeneous): what the Green preconditioner inverts"""
    discretization.fft.communicate_ghosts(input_field_inxyz)
    discretization.apply_gradient_operator_mugrid(u_inxyz=input_field_inxyz, grad_u_ijqxyz=dF_field)
    dF  = dF_field.s
    dF0 = dF.mean(axis=2, keepdims=True) if INTEGRATION == 'sri' else dF
    dP_field.s[...] = (mu_matrix * (dF + np.swapaxes(dF, 0, 1))
                       + lam_matrix * np.einsum('ii...->...', dF0) * np.eye(dim)[:, :, None, None, None])
    discretization.fft.communicate_ghosts(dP_field)
    discretization.apply_gradient_transposed_operator_mugrid(gradient_field_ijqxyz=dP_field,
                                                             div_u_fnxyz=output_field_inxyz, apply_weights=True)


def constitutive_update():
    """stress, energy and the state for the Hessian at the current total_strain_field"""
    if VOID == 'hole':                              # true hole: F = I inside keeps ln J finite; mu = lam = 0 there,
        total_strain_field.s[..., void_mask] = np.eye(dim)[:, :, None, None]   # so P = 0, W = 0, tangent = 0
    sri_update(total_strain_field, stress_field, energy_field)


def K_fun(x, Ax):
    """Hessian of the energy at the last constitutive_update applied to x (used by the small-strain probe)"""
    discretization.fft.communicate_ghosts(x)
    discretization.apply_gradient_operator_mugrid(u_inxyz=x, grad_u_ijqxyz=dF_field)
    dP_field.s[...] = sri_dP(dF_field.s, sri_state)
    discretization.fft.communicate_ghosts(dP_field)
    discretization.apply_gradient_transposed_operator_mugrid(gradient_field_ijqxyz=dP_field,
                                                             div_u_fnxyz=Ax, apply_weights=True)
    discretization.fft.communicate_ghosts(Ax)


# ============================================================================
# preconditioner
# ============================================================================
preconditioner = discretization.get_preconditioner_Green_mugrid(
    reference_material_data_ijkl=ref_mat,
    operator=sri_reference_operator    # Green of the operator actually used (SRI or full), Martin l. 406-442
)

def M_fun_Green(x, Px):
    discretization.fft.communicate_ghosts(x)
    discretization.apply_preconditioner_mugrid(
        preconditioner_Fourier_fnfnqks=preconditioner,
        input_nodal_field_fnxyz=x,
        output_nodal_field_fnxyz=Px
    )
P_weighted = discretization.get_displacement_gradient_sized_field(name='P_weighted')


def average_P():
    """volume average of the 1st Piola-Kirchhoff stress P (same averaging as get_homogenized_stress_mugrid)"""
    P_weighted.s[...] = stress_field.s[...]
    discretization.apply_quadrature_weights_on_gradient_field_mugrid(grad_field=P_weighted)
    return discretization.mpi_reduction.sum(P_weighted.s, axis=tuple(range(-dim - 1, 0))) / \
        discretization.cell.domain_volume


def set_F_identity():
    total_strain_field.s[...] = 0.0
    for d in range(dim):
        total_strain_field.s[d, d] = 1.0


def linear_response(E_macro, h=1e-6):
    """d<P>/dE for the macro strain direction E_macro at F = I: the small-strain response of THIS discretization
    (element, integration, material), units of mu_m. Newton-converged solve at h E; leaves the fields at F = I."""
    set_F_identity()
    total_strain_field.s[...] += h * E_macro[:, :, np.newaxis, np.newaxis, np.newaxis]
    constitutive_update()
    its = []
    for newton_it in range(10):                     # Newton to convergence: ONE step is not enough, because the
        discretization.fft.communicate_ghosts(stress_field)          # first residual is dominated by the stiff
        discretization.apply_gradient_transposed_operator_mugrid(   # inclusion (<P> was 23 % off after 1 step)
            gradient_field_ijqxyz=stress_field, div_u_fnxyz=rhs_field, apply_weights=True)
        rhs_field.s[...] *= -1
        norm_r = np.sqrt(discretization.communicator.sum(np.dot(rhs_field.s.ravel(), rhs_field.s.ravel())))
        if newton_it == 0:
            norm_r0 = norm_r
        elif norm_r / norm_r0 < NEWTON_TOL:
            break
        displacement_increment_field.s.fill(0)
        Solvers.conjugate_gradients(comm=discretization.communicator, fc=discretization.field_collection,
                                    hessp=K_fun, b=rhs_field, x=displacement_increment_field, prec=M_fun_Green,
                                    rtol=1e-6, atol=0.0, maxiter=50000, callback=lambda it, fields: its.append(it))
        discretization.apply_gradient_operator_mugrid(u_inxyz=displacement_increment_field,
                                                      grad_u_ijqxyz=strain_fluc_field)
        total_strain_field.s[...] += strain_fluc_field.s[...]
        constitutive_update()
    response = average_P() / h
    set_F_identity()
    displacement_increment_field.s.fill(0)
    return response, len(its)


# ============================================================================
# lateral Poisson ratio of the cell (free lateral side, simple version)
# ============================================================================
if NU_MODE == 'auto':
    C_dir = {}                                      # C_dir[d][i, j] = d<P_ij> / dE_dd  (small strain, units mu_m)
    for d in range(dim):
        E_unit = np.zeros((dim, dim))
        E_unit[d, d] = 1.0
        C_dir[d], its_lin = linear_response(E_unit)
        if MPI.COMM_WORLD.rank == 0:
            print(f'linear response for E_{d}{d} = 1: {its_lin} CG its')
    NU_LAT, E_EFF_MPA = {}, {}
    for name, (kk, ll) in {'X': (0, 1), 'Y': (1, 0)}.items():
        NU_LAT[name]    = C_dir[kk][ll, ll] / C_dir[ll][ll, ll]       # = C[k,k,l,l] / C[l,l,l,l] of the small-strain script
        E_EFF_MPA[name] = (C_dir[kk][kk, kk] - C_dir[kk][ll, ll] * C_dir[ll][kk, kk] / C_dir[ll][ll, ll]) * S_REF
else:
    if INTEGRATION != 'full' or (MATERIAL, nnn) not in NU_LAT_TABLE:
        raise ValueError(f'no table NU_LAT for {MATERIAL}, NX = {nnn}, {INTEGRATION}: use NU_MODE = "auto"')
    NU_LAT, E_EFF_MPA = NU_LAT_TABLE[(MATERIAL, nnn)], {'X': float('nan'), 'Y': float('nan')}
if MPI.COMM_WORLD.rank == 0:
    print(f'NU_LAT ({NU_MODE}) = X {NU_LAT["X"]:.4f}, Y {NU_LAT["Y"]:.4f} | small-strain E_eff = '
          f'X {E_EFF_MPA["X"]:.4f}, Y {E_EFF_MPA["Y"]:.4f} MPa (= expected initial slope)')
# ============================================================================
# macroscopic loading
# # ============================================================================
k, l = (0, 1) if LOAD_DIRECTION == 'X' else (1, 0)   # k = loaded axis, l = free lateral axis
load_pattern       = np.zeros((dim, dim))       # total macro gradient F - I
if LOAD_MODE == 'fig4':
    load_pattern[k, k] = -EPS_MAX[LOAD_DIRECTION]                           # uniaxial compression
    load_pattern[l, l] = +NU_LAT[LOAD_DIRECTION] * EPS_MAX[LOAD_DIRECTION]  # lateral expansion (free side, simple)
    load_pattern[l, k] = 0.01   # introduce shear for imperfection
else:                                           # 'manual': set any components here (index [i, j] = du_i/dx_j)
    load_pattern[0, 1] = 0.01                   # shear
    load_pattern[0, 0] = 0.04                   # stretch in x
    # load_pattern[1, 1] = 0.05                 # stretch in y
macro_gradient_inc = E_LOAD * load_pattern / float(ninc)
GTOL_ABS = GTOL * 1.0 * h_x * np.abs(macro_gradient_inc).max()   # Martin: gtol mu_m h d_eps (mu_m = 1)

if MPI.COMM_WORLD.rank == 0:
    if LOAD_MODE == 'fig4':
        print(f'load: compression in {LOAD_DIRECTION} up to eps = {E_LOAD * EPS_MAX[LOAD_DIRECTION]} in {ninc} increments '
              f'({E_LOAD * EPS_MAX[LOAD_DIRECTION] / ninc:.2e} per increment), lateral nu = {NU_LAT[LOAD_DIRECTION]}')
    else:
        print(f'load (manual): total F - I =\n{E_LOAD * load_pattern}\nin {ninc} increments, '
              f'largest component per increment = {np.abs(macro_gradient_inc).max():.2e}')

discretization.get_macro_gradient_field_mugrid(
    macro_gradient_ij=macro_gradient_inc,
    macro_gradient_field_ijqxyz=macro_gradient_inc_field
)

_info['macro_gradient_inc']     = macro_gradient_inc
_info['norm_strain_fluc_field'] = []

# initial constitutive evaluation
constitutive_update()


# ============================================================================
# incremental loading — trust-region Newton-CG
# ============================================================================
sum_CG_its      = 0
sum_Newton_its  = 0
iteration_total = 0
start_time      = time.time()

# ====================================================================================================================================================










# history per increment (index 0 = undeformed state)
hist = {'eps_nominal': [0.0], 'P_load': [0.0], 'P_lat': [0.0], 'lateral_stretch': [1.0],
        'H': [np.zeros((dim, dim))], 'P': [np.zeros((dim, dim))]}   # full macro F - I and average P# ====================================================================================================================================================

# ============================================================================
# trust-region Newton-CG, as Martin's stefanus_2D_finite_strain_NeoHookean_TR.py (lines 189-623)
# ============================================================================
#   unknowns x = (u, f): u = periodic fluctuation (all nodes), f = F_ll - 1 (only if LATERAL = 'exact', stored on
#   rank 0, scaled by 1 / F_SCALE). The trust region minimizes the total energy; stationarity in f is <P_ll> = 0.
#   load parameter t in [0, 1]: F_macro = I + t E_LOAD load_pattern (F_ll replaced by 1 + f if it is an unknown);
#   a failed trust-region solve halves the step, a success lets it grow back (x 2) up to 1 / ninc (Martin l. 586-624)
from NuMPI.Optimization import tr_newton_bounded      # NuMPI branch feat/tr_precond (precond, inner_stop)
comm        = discretization.communicator
rank0       = MPI.COMM_WORLD.rank == 0
LAT_UNKNOWN = LATERAL == 'exact' and LOAD_MODE == 'fig4'
u_work      = discretization.get_unknown_size_field(name='tr_u_work')
tr_in       = discretization.get_unknown_size_field(name='tr_in')
tr_out      = discretization.get_unknown_size_field(name='tr_out')
w_q         = np.asarray(discretization.quadrature_weights)
nb_u        = u_work.s.size
V_total     = float(np.prod(discretization.nb_of_pixels_global) * float(w_q.sum()))
F_SCALE     = 1.0 / np.prod(discretization.nb_of_pixels_global)   # f = F_SCALE x_f (Martin l. 185)
INADMISSIBLE_ENERGY = 1e30                                       # finite: inf would give a nan TR ratio
tr_F_macro  = np.eye(dim)                                         # prescribed macro F of the current load step
tr_state    = {'hess_at': None, 'F_macro': np.eye(dim), 'cg': 0}   # hess_at: x of the last Hessian state (Martin's hess_at)

def tr_split(x):
    if not LAT_UNKNOWN:
        return x, 0.0
    f = MPI.COMM_WORLD.bcast(float(x[nb_u]) if rank0 else None, root=0)
    return x[:nb_u], F_SCALE * f

def tr_join(u_part, f_part):                  # x, v: f in physical units (stretch)
    if LAT_UNKNOWN and rank0:
        return np.concatenate([u_part, [f_part / F_SCALE]])
    return u_part.copy()

def tr_join_dual(u_part, g_f):                # gradient, H v: f component in dual units
    if LAT_UNKNOWN and rank0:
        return np.concatenate([u_part, [g_f * F_SCALE]])
    return u_part.copy()

def sum_ll(field):
    """sum over the cell of w_q field_ll (global) = V <field_ll>"""
    return comm.sum(float(np.einsum('q...,q->...', field.s[l, l], w_q).sum()))

def tr_evaluate(x):
    """F = F_macro + grad u at every quadrature point, then stress, energy and the SRI state"""
    u_part, f = tr_split(x)
    F_mac = tr_F_macro.copy()
    if LAT_UNKNOWN:
        F_mac[l, l] = 1.0 + f
    tr_state['F_macro'] = F_mac
    u_work.s[...] = u_part.reshape(u_work.s.shape)
    discretization.fft.communicate_ghosts(u_work)
    discretization.apply_gradient_operator_mugrid(u_inxyz=u_work, grad_u_ijqxyz=strain_fluc_field)
    total_strain_field.s[...] = strain_fluc_field.s + F_mac[:, :, np.newaxis, np.newaxis, np.newaxis]
    constitutive_update()                     # raises RuntimeError for an inverted element

def tr_fun(x):
    """total energy and its gradient (B^T w P, sum w P_ll) -- Martin l. 340-358"""
    try:
        tr_evaluate(x)
    except RuntimeError:                      # inverted element: inadmissible, the trust region shrinks
        set_F_identity()                      # Martin l. 294-297: the state of an inadmissible point is F = I
        constitutive_update()
        return INADMISSIBLE_ENERGY, np.zeros_like(x)
    energy = comm.sum(float(np.einsum('q...,q->...', energy_field.s[0, 0], w_q).sum()))
    if not np.isfinite(energy):
        return INADMISSIBLE_ENERGY, np.zeros_like(x)
    discretization.fft.communicate_ghosts(stress_field)
    discretization.apply_gradient_transposed_operator_mugrid(gradient_field_ijqxyz=stress_field,
                                                             div_u_fnxyz=tr_out, apply_weights=True)
    return energy, tr_join_dual(tr_out.s.ravel().copy(), sum_ll(stress_field))

def tr_hessp(x, v):
    """Hessian of the energy in (u, f) applied to v: dF = grad v_u + v_f e_ll -- Martin l. 364-385"""
    if tr_state['hess_at'] is None or not np.array_equal(tr_state['hess_at'], x):   # as Martin l. 366-368: the
        tr_evaluate(x)                                                                # state is cached by x, fun()
        tr_state['hess_at'] = x.copy()                                                # at trial points overwrites it
    tr_state['cg'] += 1
    v_u, v_f = tr_split(v)
    tr_in.s[...] = v_u.reshape(tr_in.s.shape)
    discretization.fft.communicate_ghosts(tr_in)
    discretization.apply_gradient_operator_mugrid(u_inxyz=tr_in, grad_u_ijqxyz=dF_field)
    if LAT_UNKNOWN:
        dF_field.s[l, l] += v_f
    dP_field.s[...] = sri_dP(dF_field.s, sri_state)
    discretization.fft.communicate_ghosts(dP_field)
    discretization.apply_gradient_transposed_operator_mugrid(gradient_field_ijqxyz=dP_field,
                                                             div_u_fnxyz=tr_out, apply_weights=True)
    return tr_join_dual(tr_out.s.ravel().copy(), sum_ll(dP_field))

def tr_precond(r):
    """Green on u; the macro unknown scaled by 1 / (V mu_m) -- Martin l. 465-485"""
    tr_in.s[...] = r[:nb_u].reshape(tr_in.s.shape)
    M_fun_Green(tr_in, tr_out)
    if LAT_UNKNOWN and rank0:
        return np.concatenate([tr_out.s.ravel().copy(), [float(r[nb_u]) / (V_total * F_SCALE ** 2)]])
    return tr_out.s.ravel().copy()

max_step, t_load, inc = 1.0 / ninc, 0.0, 0
min_step = max_step / 64
d_t      = max_step
x_cur    = tr_join(np.zeros(nb_u), 0.0)
while t_load < 1.0 - 1e-12:
    d_t = min(d_t, 1.0 - t_load)
    tr_F_macro[...] = np.eye(dim) + (t_load + d_t) * E_LOAD * load_pattern
    tr_state['hess_at'], tr_state['cg'] = None, 0
    res = tr_newton_bounded(tr_fun, x_cur, tr_hessp, precond=tr_precond, gtol=GTOL_ABS, maxiter=TR_MAXITER,
                            delta0=TR_DELTA0, delta_max=TR_DELTA_MAX, inner_stop='energy', inner_delay=10,
                            inner_tol=TR_INNER_TOL, inner_tau=TR_INNER_TAU, comm=MPI.COMM_WORLD)
    sum_CG_its     += tr_state['cg']
    sum_Newton_its += res.nit
    ok = bool(res.success) and res.fun < 0.5 * INADMISSIBLE_ENERGY
    if not ok:                                # keep x_cur (last converged state), halve the step
        d_t *= 0.5
        if rank0:
            print(f'  t = {t_load + 2 * d_t:.4f}: trust region failed ({res.message}) -> halve the step to '
                  f'{d_t * E_LOAD * EPS_MAX.get(LOAD_DIRECTION, 1.0):.2e}', flush=True)
        if d_t < min_step:
            if rank0:
                print('  step below min_step: stopping', flush=True)
            break
        continue
    t_load += d_t
    x_cur   = np.asarray(res.x).copy()
    tr_fun(x_cur)                             # fields (stress, F) at the converged state
    u_part, _ = tr_split(x_cur)
    displacement_fluctuation_field.s[...] = u_part.reshape(displacement_fluctuation_field.s.shape)
    iteration_total += 1
    if save_per_it:                            # fields of every converged load step
        for name_, arr_ in (('displacement_fluctuation_field', displacement_fluctuation_field.s.mean(axis=1)),
                            ('energy_field', energy_field.s[0, 0].mean(axis=0)),
                            ('stress_field', stress_field.s.mean(axis=2))):
            save_npy(data_folder_path + f'{name_}_inc{inc}.npy', arr_,
                     tuple(discretization.subdomain_locations_no_buffers),
                     tuple(discretization.nb_of_pixels_global), MPI.COMM_WORLD)
    total_macro_gradient = tr_state['F_macro'] - np.eye(dim)
    P_avg = average_P()
    hist['eps_nominal'].append(-total_macro_gradient[k, k])
    hist['P_load'].append(P_avg[k, k])
    hist['P_lat'].append(P_avg[l, l])
    hist['lateral_stretch'].append(1.0 + total_macro_gradient[l, l])
    hist['H'].append(total_macro_gradient.copy())
    hist['P'].append(P_avg.copy())
    if rank0:
        if LOAD_MODE == 'fig4':
            print(f'inc {inc:3d}  eps = {hist["eps_nominal"][-1]:.5f} (step {d_t * E_LOAD * EPS_MAX[LOAD_DIRECTION]:.4f}) | '
                  f'nominal stress = {-P_avg[k, k] * S_REF:.6f} MPa | lateral stress = {P_avg[l, l] * S_REF:+.2e} MPa '
                  f'({abs(P_avg[l, l] / P_avg[k, k]):.2%} of axial) | TR its {res.nit}, CG its {tr_state["cg"]}',
                  flush=True)
        else:
            print(f'inc {inc:3d}  t = {t_load:.4f} | F - I = {np.array2string(total_macro_gradient.ravel(), precision=5)} | '
                  f'P [MPa] = {np.array2string(P_avg.ravel() * S_REF, precision=6)} | TR its {res.nit}, '
                  f'CG its {tr_state["cg"]}', flush=True)
    if plot_load_cases and discretization.communicator.size == 1:
        L_cell = np.diag(domain_size)
        u_unit_cell.s[0] = displacement_fluctuation_field.s[0] / domain_size[0]
        u_unit_cell.s[1] = displacement_fluctuation_field.s[1] / domain_size[1]
        x_plot_ixyz = visualization_utils.get_deformed_grid_coords_two_dim(
            discretization, macro_gradient_ij=np.linalg.inv(L_cell) @ total_macro_gradient @ L_cell,
            displacement_fluctuation=u_unit_cell)
        visualization_utils.plot_field_on_grid(coordinates_for_plot=x_plot_ixyz, field_to_plot=phase_field.s[0, 0],
                                               name=f'TR, increment {inc + 1}, t = {t_load:.3f}')
    inc += 1
    d_t = min(2.0 * d_t, max_step)
# ============================================================================
# timing and summary
# ============================================================================
end_time     = time.time()
elapsed_time = end_time - start_time

_info['sum_Newton_its']  = sum_Newton_its
_info['sum_CG_its']      = sum_CG_its
_info['iteration_total'] = iteration_total
_info['elapsed_time']    = elapsed_time

if discretization.communicator.rank == 0:
    print('=' * 70)
    print(f'element_type     : {element_type} | integration: {INTEGRATION} | tangent: {SRI_TANGENT} | material: {MATERIAL}')
    print(f'number_of_pixels : {number_of_pixels}')
    print(f'preconditioner   : {preconditioner_type}')
    print(f'Total CG its     : {sum_CG_its}')
    print(f'Total TR its     : {sum_Newton_its}  (trust-region Newton iterations)')
    print(f'Elapsed time     : {elapsed_time:.2f} s  ({elapsed_time / 60:.2f} min)')
    np.savez(data_folder_path + 'info_log_final.npz', **_info)

    # ------------------------------------------------------------------------
    # all load modes: full macro F - I and average P per increment -> CSV
    # ------------------------------------------------------------------------
    H_all = np.array(hist['H']).reshape(-1, dim * dim)
    P_all = np.array(hist['P']).reshape(-1, dim * dim) * S_REF
    fname_all = data_folder_path + f'stress_strain_{LOAD_MODE}.csv'
    np.savetxt(fname_all, np.column_stack([np.arange(len(H_all)), H_all, P_all]), delimiter=',', fmt='%.8f',
               comments='', header='increment,H00,H01,H10,H11,P00 [MPa],P01 [MPa],P10 [MPa],P11 [MPa]')
    print(f'saved {fname_all}   (H = F - I macro, P = 1st Piola-Kirchhoff average)')

if discretization.communicator.rank == 0 and LOAD_MODE == 'fig4':
    # ------------------------------------------------------------------------
    # Fig. 4 data: nominal strain / nominal stress (compression > 0) -> CSV
    # ------------------------------------------------------------------------
    eps_nom    = np.array(hist['eps_nominal'])
    stress_MPa = -np.array(hist['P_load']) * S_REF               # nominal stress = -P_kk (1st Piola-Kirchhoff)
    lat_MPa    = np.array(hist['P_lat']) * S_REF                 # should be ~0 (free lateral side)
    stretch    = np.array(hist['lateral_stretch'])
    nu_nom     = np.zeros_like(eps_nom)
    nu_nom[1:] = (stretch[1:] - 1.0) / eps_nom[1:]               # nominal Poisson ratio ('exact': computed; 'simple': = NU_LAT)
    fname = data_folder_path + f'Fig4_finite_strain_{LOAD_DIRECTION}.csv'
    np.savetxt(fname, np.column_stack([eps_nom, stress_MPa, lat_MPa, stretch, nu_nom]),
               delimiter=',', fmt='%.8f', comments='',
               header=f'nominal strain eps_{LOAD_DIRECTION} [-],nominal stress [MPa],lateral stress [MPa],'
                      f'lateral stretch [-],nominal Poisson ratio [-]')
    print(f'saved {fname}')
    print(f'initial slope = {stress_MPa[1] / eps_nom[1]:.4f} MPa | secant slope at eps = {eps_nom[-1]:.4f}: '
          f'{stress_MPa[-1] / eps_nom[-1]:.4f} MPa | max |lateral / axial stress| = '
          f'{np.max(np.abs(lat_MPa[1:] / stress_MPa[1:])):.2%}')

    if plot_fig4:
        from matplotlib import pyplot as plt
        fig, ax = plt.subplots(figsize=(5, 4.2))
        ax.plot(eps_nom, stress_MPa, 'o-', color='tab:green', ms=3,
                label=f'muFFTTO finite strain ({number_of_pixels[0]}x{number_of_pixels[1]}, {element_type}, {INTEGRATION}, {MATERIAL})')
        for (name, slope), style in zip(PAPER_SLOPE_MPA[LOAD_DIRECTION].items(), ['k-', 'r-.']):
            ax.plot(eps_nom, slope * eps_nom, style, lw=1, label=f'paper {name} (initial slope, digitised)')
        ax.set_xlim(0, EPS_MAX[LOAD_DIRECTION])
        ax.xaxis.set_major_locator(plt.MaxNLocator(5))                       # 0, 0.004, ..., 0.020 (as small strain)
        ax.yaxis.set_major_locator(plt.MultipleLocator(0.002))               # y ticks every 0.002 MPa (as small strain)
        ax.set_ylim(0, max(Y_MAX_MPA[LOAD_DIRECTION], 1.05 * stress_MPa.max()))   # fixed top, grows if the curve is higher
        ax.set_xlabel(rf'Nominal strain, $\varepsilon_{LOAD_DIRECTION}$')
        ax.set_ylabel('Nominal stress (MPa)')
        ax.set_title(f'Fig. 4{"a" if LOAD_DIRECTION == "X" else "b"}: uniaxial compression in {LOAD_DIRECTION} '
                     f'(up to eps_cr)', fontsize=10)
        ax.legend(fontsize=7, loc='upper left')
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(figure_folder_path + f'Fig4_finite_strain_{LOAD_DIRECTION}.png', dpi=200)
        plt.show()