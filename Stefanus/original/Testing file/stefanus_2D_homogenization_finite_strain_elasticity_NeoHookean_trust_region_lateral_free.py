"""
Uniaxial compression of the Composite A cell with a STRESS-FREE LATERAL SIDE, as in the paper's test
(Li & Rudykh 2019, Fig. 4).

Copy of stefanus_2D_homogenization_finite_strain_elasticity_NeoHookean_trust_region.py with one change:

  * there, the whole macroscopic deformation gradient F_macro is prescribed: compression in y AND F_xx = 1, so the
    cell cannot get wider on average -> a lateral stress <P_xx> != 0 appears and the response is too stiff;
  * here, only the loaded component F_kk is prescribed. The lateral stretch f = F_ll - 1 is an UNKNOWN, found
    together with the displacement fluctuation u by the same trust-region energy minimization.

Why this gives a stress-free side: the minimum of the energy E(u, f) has dE/df = 0, and

    dE/df = sum over quadrature points of w_q * P_ll  =  cell area * <P_ll>,

so at the solution <P_ll> = 0. (Same idea as a free boundary in a FE code: no force where nothing is prescribed.)
"""
import time
import os
import sys
import argparse

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpi4py import MPI
from NuMPI.IO import save_npy
from NuMPI.Optimization import tr_newton_bounded

# Add the project root to sys.path relatively
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from muFFTTO import domain
from muFFTTO import microstructure_library
from muFFTTO import material_models
from muFFTTO import visualization_utils
from muFFTTO.tensor_operations import to_host

# Numerical settings ============================
NX = 64                     # Arbitrary number of pixels (default of -n; NY follows from the cell)
N_inc = 30                  # Number of increment for the applied load
cg_tol = None               # CG tolerance of each Newton step (relative energy error); None: adaptive,
                            # min(0.5, sqrt(max |residual|)) -- rough far from equilibrium, tight close to it
E_LOAD = 1.0                # scales the total macro load below (finite strain: response is NOT linear in E_LOAD)
plot_load_cases = True      # plot the deformed cell after every load increment
# ===============================================

# ============================================================================
# argument parsing
# ============================================================================
parser = argparse.ArgumentParser(
    prog='stefanus_2D_homogenization_finite_strain_elasticity_NeoHookean_trust_region.py',
    description='Solve finite strain NeoHookean elasticity in 2D'
)
parser.add_argument('-n',   '--nb_pixel',       default=str(NX))
parser.add_argument('-inc', '--nb_increments',  default=str(N_inc))
parser.add_argument(
    '--save_per_it',
    action=argparse.BooleanOptionalAction, default=True,
    help='save the fields (.npy) after every load increment (default: on; --no-save_per_it switches it off)'
)
parser.add_argument('--show', action='store_true',
                    help='show the deformed cell in a window after every load increment (close it to continue); '
                         'by default the plots are only saved as PNG in figures/')
parser.add_argument('--direction', choices=['X', 'Y'], default='Y',
                    help='loaded direction (compression); the other direction is the stress-free lateral side')
parser.add_argument('--element', default='bilinear_rectangle',
                    help="finite element: 'bilinear_rectangle' (Q1) or 'biquadratic_rectangle' (Q2, no volumetric "
                         "locking for nearly incompressible materials, ~4x the cost)")
parser.add_argument('--device', choices=['cpu', 'gpu'], default='cpu',
                    help="where the fields live: 'cpu' (numpy) or 'gpu' (cupy, muGrid built with CUDA)")


script_name = os.path.splitext(os.path.basename(__file__))[0]
args        = parser.parse_args()
nnn         = int(args.nb_pixel)
ninc        = int(args.nb_increments)
save_per_it = args.save_per_it
if not args.show:
    matplotlib.use('Agg')       # no windows: plots are saved as PNG only

# array module of the fields: numpy on the CPU, cupy on the GPU. Small numpy constants (identity, weights, ...)
# are moved to the device with xp.asarray before they meet a field
if args.device == 'gpu':
    import cupy as xp
else:
    xp = np
# ============================================================================
# problem setup
# ============================================================================

# Unit Cells input: -- PHYSICAL units ===================
R_MM         = 3.0              # void & inclusion radius r                      [mm]
MU_M_MPA     = 0.6              # matrix shear modulus mu^(m) (FLX9860)           [MPa]
MU_I_MPA     = 6.0              # stiff inclusion shear modulus mu^(i) [MPa]; paper (VeroWhite): 600
LAM_OVER_MU  = 100.0            # Lambda / mu, matrix and inclusion [-]; paper (Sec. 3, after Eq. 1): 1000
# simplified material (contrast 10 instead of 1000, Lambda / mu 100 instead of 1000): within 3.5 % of the paper's
# material before buckling and 1 % after (128 px), and much cheaper to solve (see the Stefanus report)
C_MATRIX     = 0.25             # target matrix volume fraction c^(m) ratio            [-]
SPECIMEN_MM  = (92.37, 91.42)   # specimen width x height (finite sample, check only) [mm]
THICKNESS_MM = 10.0             # out-of-plane thickness -> plane strain, not used in 2D [mm]
MU_V_MPA     = 0.0              # void = true hole (no material, as in the paper) [MPa]

# Reference scales (nondimensionalisation) ====================================================
L_REF = R_MM        # length scale  [mm]  -> lengths in units of r
S_REF = MU_M_MPA    # stress scale  [MPa] -> stresses / moduli in units of mu^(m)

# Nondimensional quantities used by the solver (derived, do not edit) =========================
c_matrix = C_MATRIX
r = R_MM / L_REF                        # = 1
CONTRAST = MU_I_MPA / S_REF             # = 10 (paper: 1000)
MU_VOID = MU_V_MPA / S_REF              # = 0: the hole has no energy, stress or stiffness (F = I is set there,
                                        # see set_deformation_gradient, so that log J stays finite)
incl_vol_frac = 1.0 - c_matrix          # total circle fraction (voids + stiff)

# Domain: TRIANGULAR lattice (paper Fig. 1a), centre spacing a, periodic cell 2a x sqrt(3)a
a = r * np.sqrt(2 * np.pi / (np.sqrt(3) * incl_vol_frac))
ligament = a - 2 * r                                            # thinnest matrix ligament (void-void)

cell_size        = [2 * a, np.sqrt(3) * a]                       # the periodic cell of Composite A
cell_pixels      = (nnn, int(round(nnn * cell_size[1] / cell_size[0])))   # square pixels
h_x, h_y = cell_size[0] / cell_pixels[0], cell_size[1] / cell_pixels[1]   # actual sizes

domain_size      = cell_size                 # computational box = the periodic cell
number_of_pixels = cell_pixels
dim              = len(domain_size)

# loaded axis k (F_kk prescribed) and lateral axis l (F_ll = 1 + f, f unknown)
k, l = (0, 1) if args.direction == 'X' else (1, 0)

if MPI.COMM_WORLD.rank == 0:
    print(f'a = {a * L_REF:.3f} mm | ligament = {ligament * L_REF:.3f} mm | '
          f'pixel = {h_x * L_REF:.4f} x {h_y * L_REF:.4f} mm | '
          f'{ligament / max(h_x, h_y):.1f} px across ligament | grid = {number_of_pixels}')


problem_type        = 'elasticity'
discretization_type = 'finite_element'
element_type        = args.element
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
                                       element_type=element_type,
                                       device=args.device)

# ============================================================================
# output folders
# ============================================================================
file_folder_path   = os.path.dirname(os.path.realpath(__file__))
data_folder_path   = (file_folder_path + '/exp_data/' + script_name + '/'
                      + f'Nx={number_of_pixels[0]}Ny={number_of_pixels[1]}_{preconditioner_type}_{element_type}'
                      + f'_{args.direction}_lateral_free/')
figure_folder_path = (file_folder_path + '/figures/' + script_name + '/'
                      + f'Nx={number_of_pixels[0]}Ny={number_of_pixels[1]}_{preconditioner_type}_{element_type}'
                      + f'_{args.direction}_lateral_free/')

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
_info['cg_tol']     = cg_tol

# ============================================================================
# geometry — Composite A (0 = matrix, 1 = void, 2 = stiff inclusion)
# ============================================================================
geometry_ID = 'geometry_stefanus'

phase_host = microstructure_library.get_geometry(                             # built on the host (numpy)
    nb_voxels=discretization.nb_of_pixels,
    microstructure_name=geometry_ID,
    coordinates=to_host(discretization.fft.coords),
    vol_frac=incl_vol_frac,
    domain_size=cell_size
)
phase_field = discretization.get_scalar_field(name='phase_field')
phase_field.s[0, 0] = xp.asarray(phase_host)

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
    print(f'lam min/max : {float(lam_field.s.min()):.4f} / {float(lam_field.s.max()):.4f}')
    print(f'mu  min/max : {float(mu_field.s.min()):.4f}  / {float(mu_field.s.max()):.4f}')

# ============================================================================
# material model
# ============================================================================
material = material_models.NeoHookean(
    discretization=discretization,
    lam_1qxyz=lam_field,
    mu_1qxyz=mu_field,
    name='neo_hookean_two_phase'
)

# ============================================================================
# fields
# ============================================================================
macro_gradient_inc_field       = discretization.get_gradient_size_field(name='macro_gradient_inc_field')
displacement_fluctuation_field = discretization.get_unknown_size_field(name='displacement_fluctuation_field')
displacement_increment_field   = discretization.get_unknown_size_field(name='displacement_increment_field')
strain_fluc_field              = discretization.get_displacement_gradient_sized_field(name='strain_fluctuation_field')
total_strain_field             = discretization.get_displacement_gradient_sized_field(name='total_strain_field')
stress_field                   = discretization.get_displacement_gradient_sized_field(name='stress_field')
tangent_field                  = discretization.get_material_data_size_field_mugrid(name='tangent_field')
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
# preconditioner
# ============================================================================
preconditioner = discretization.get_preconditioner_Green_mugrid(
    reference_material_data_ijkl=ref_mat
)

def M_fun_Green(x, Px):
    discretization.fft.communicate_ghosts(x)
    discretization.apply_preconditioner_mugrid(
        preconditioner_Fourier_fnfnqks=preconditioner,
        input_nodal_field_fnxyz=x,
        output_nodal_field_fnxyz=Px
    )

# ============================================================================
# macroscopic loading
# ============================================================================
load_pattern       = np.zeros((dim, dim))       # total PRESCRIBED macro gradient F - I for E_LOAD = 1
load_pattern[k, k] = -0.2                       # 20 % compression in the loaded direction
                                                # (load_pattern[l, l] stays 0: F_ll is the unknown 1 + f)
macro_gradient_inc = E_LOAD * load_pattern / float(ninc)

# ============================================================================
# print the set-up
# ============================================================================
if discretization.communicator.rank == 0:
    print('=' * 70)
    print('SET-UP')
    print(f'  cell        : Composite A, triangular lattice, '
          f'{cell_size[0] * L_REF:.2f} x {cell_size[1] * L_REF:.2f} mm, '
          f'r = {R_MM} mm, circle fraction {incl_vol_frac:.2f}')
    print(f'  grid        : {number_of_pixels[0]} x {number_of_pixels[1]} pixels, {element_type}, '
          f'{ligament / max(h_x, h_y):.1f} px across the thinnest ligament')
    print(f'  {"phase":<12}{"mu [MPa]":>12}{"lambda [MPa]":>14}{"lambda / mu":>13}')
    for name, mu_, lam_ in (('matrix', mu_matrix, lam_matrix), ('inclusion', mu_inc, lam_inc),
                            ('void', mu_void, lam_void)):
        ratio = f'{lam_ / mu_:13.0f}' if mu_ > 0 else f'{"true hole":>13}'
        print(f'  {name:<12}{mu_ * S_REF:12.4g}{lam_ * S_REF:14.4g}{ratio}')
    print(f'  contrast    : mu_inclusion / mu_matrix = {CONTRAST:g}')
    print(f'  load        : F_{"xy"[k]}{"xy"[k]} - 1 = {load_pattern[k, k] * E_LOAD:g} in {ninc} increments, '
          f'F_{"xy"[l]}{"xy"[l]} free (lateral side stress free, <P_{"xy"[l]}{"xy"[l]}> = 0)')
    print(f'  solver      : trust-region Newton (NuMPI tr_newton_bounded), Green preconditioner '
          f'(reference: matrix), CG tolerance {"adaptive" if cg_tol is None else cg_tol}, '
          f'stop at residual drop 1e-8')
    print(f'  device      : {args.device}, {MPI.COMM_WORLD.size} MPI process(es)')
    print('=' * 70)

discretization.get_macro_gradient_field_mugrid(
    macro_gradient_ij=macro_gradient_inc,
    macro_gradient_field_ijqxyz=macro_gradient_inc_field
)

_info['macro_gradient_inc']     = macro_gradient_inc
_info['norm_strain_fluc_field'] = []

# initial constitutive evaluation
material.get_stress(total_strain_field, stress_field)
material.get_algorithmic_tangent(total_strain_field, tangent_field)

if discretization.communicator.rank == 0:
    print(f'tangent min/max : {float(tangent_field.s.min()):.4f} / {float(tangent_field.s.max()):.4f}')

# ============================================================================
# equilibrium as energy minimization (trust-region Newton)
# ============================================================================
# The displacement fluctuation u is in equilibrium where the total energy
#
#     E(u) = sum over quadrature points of  w_q * W(F_q),      F = F_macro + grad u
#
# is minimal. Its gradient is the (negative) residual  B^T w P  and its Hessian is the
# system matrix  B^T w A B  of the Newton loop in the original script.
#
# tr_newton_bounded (NuMPI) minimizes E with Newton steps, each one solved by preconditioned
# CG, but inside a trust region: a step that does not lower the energy is rejected and the
# region shrinks, a good step lets it grow. The plain Newton loop of the original takes
# every full step, which can overshoot and diverge for large load increments.
#
# The unknown u has the shape of displacement_fluctuation_field.s; tr_newton_bounded hands u (and the
# vectors v, r) to the functions below in that shape. They copy it into a muGrid field, apply the
# muFFTTO operators, and return the result in the same shape.
#
# NuMPI works on numpy arrays (CPU). With --device gpu the fields are cupy arrays (GPU): u, v, r are copied
# to the GPU with xp.asarray, and the results back to the CPU with host_copy. Only these vectors cross; the
# expensive part (gradients, material, FFTs of the preconditioner) stays on the GPU.

# NEW in this script: the unknowns are x = (u, f), u the displacement fluctuation and f = F_ll - 1 the lateral
# macro stretch. With  F = I + (F_macro - I) + f e_l (x) e_l + grad u  the energy E(u, f) has the gradient
#
#     dE/du = B^T w P                       (as before: the residual)
#     dE/df = sum_q w_q P_ll  = A <P_ll>     (A = cell area: the lateral force)
#
# and the Hessian (the system matrix) gets one extra row and column:
#
#     H [v_u, v_f] = ( B^T w A_t (B v_u + v_f E_ll) ,  sum_q w_q [A_t (B v_u + v_f E_ll)]_ll )
#
# (A_t the tangent dP/dF, E_ll the matrix with 1 at (l, l)). The trust-region solver minimizes over u and f
# together; its stop test max |dE/dx| <= gtol then also enforces <P_ll> ~ 0.
#
# x is one flat vector [u.ravel(), x_f]. f is scaled, f = F_SCALE * x_f, so that dE/dx_f = A <P_ll> F_SCALE has the
# size of a nodal force (one pixel's share of the cell): the same inf-norm stop test then fits both parts.
# With several MPI processes only rank 0 holds x_f, so that the dot products of the solver count it once.

INADMISSIBLE_ENERGY = 1e30          # energy of a state with J = det F <= 0 (inverted element): step is rejected

macro_gradient_total = np.zeros((dim, dim))   # prescribed F_macro - I (only [k, k]), grows every increment
rank0   = MPI.COMM_WORLD.rank == 0
u_shape = displacement_fluctuation_field.s.shape
nb_u    = int(np.prod(u_shape))
F_SCALE = 1.0 / np.prod(number_of_pixels)

hessian_vector_field = discretization.get_unknown_size_field(name='hessian_vector_field')
precond_input_field  = discretization.get_unknown_size_field(name='precond_input_field')
precond_output_field = discretization.get_unknown_size_field(name='precond_output_field')


quadrature_weights = xp.asarray(discretization.quadrature_weights)   # [q], include the pixel area


def host_copy(field_values):
    """a numpy (CPU) copy of field values, for NuMPI"""
    return np.array(to_host(field_values))


def split(x):
    """x -> (u in field shape, f).  f lives on rank 0 and is broadcast to all processes"""
    x_f = MPI.COMM_WORLD.bcast(float(x[nb_u]) if rank0 else None, root=0)
    return x[:nb_u].reshape(u_shape), F_SCALE * x_f


def join(u_part, f_part):
    """(u, f) -> x, the inverse of split"""
    if rank0:
        return np.concatenate([np.ravel(u_part), [f_part / F_SCALE]])
    return np.ravel(u_part).copy()


def join_gradient(du_part, df_part):
    """(dE/du, dE/df) -> dE/dx  (chain rule: dE/dx_f = F_SCALE dE/df)"""
    if rank0:
        return np.concatenate([np.ravel(du_part), [df_part * F_SCALE]])
    return np.ravel(du_part).copy()


def sum_ll(field):
    """sum over all quadrature points of w_q field_ll  (all processes)"""
    return discretization.communicator.sum(
        float(xp.einsum('qxy,q->', field.s[l, l], quadrature_weights)))


def set_deformation_gradient(u, f):
    """u -> displacement_fluctuation_field, and total_strain_field = F = I + (F_macro - I) + f e_l e_l + grad u"""
    displacement_fluctuation_field.s[...] = xp.asarray(u)
    discretization.fft.communicate_ghosts(displacement_fluctuation_field)
    discretization.apply_gradient_operator_mugrid(
        u_inxyz=displacement_fluctuation_field,
        grad_u_ijqxyz=strain_fluc_field
    )
    F_macro_host = np.eye(dim) + macro_gradient_total
    F_macro_host[l, l] += f                                     # the unknown lateral stretch
    F_macro = xp.asarray(F_macro_host)
    total_strain_field.s[...] = strain_fluc_field.s + F_macro[:, :, np.newaxis, np.newaxis, np.newaxis]

    # true holes: the hole may deform arbitrarily (even invert), it carries no material. F = I there keeps the
    # neo-Hookean log J finite; with mu = lam = 0 the hole then has zero energy, stress and tangent
    total_strain_field.s[:, :, :, void_mask] = xp.eye(dim)[:, :, np.newaxis, np.newaxis]


def energy_and_gradient(x):
    """total energy E(u, f) and its gradient (dE/du, dE/df) = (B^T w P, sum w P_ll)"""
    u, f = split(x)
    set_deformation_gradient(u, f)

    F = total_strain_field.s
    J = F[0, 0] * F[1, 1] - F[0, 1] * F[1, 0]
    if MPI.COMM_WORLD.allreduce(float(J.min()), op=MPI.MIN) <= 0.0:
        return INADMISSIBLE_ENERGY, np.zeros_like(x)

    material.get_energy_density(total_strain_field, energy_field)
    energy = discretization.communicator.sum(
        float(xp.einsum('qxy,q->', energy_field.s[0, 0], quadrature_weights))
    )

    material.get_stress(total_strain_field, stress_field)
    discretization.fft.communicate_ghosts(stress_field)
    discretization.apply_gradient_transposed_operator_mugrid(
        gradient_field_ijqxyz=stress_field,
        div_u_fnxyz=rhs_field,
        apply_weights=True
    )
    return energy, join_gradient(host_copy(rhs_field.s), sum_ll(stress_field))


tangent_state = {'x': None}         # the state at which tangent_field was last evaluated
dF_field = discretization.get_displacement_gradient_sized_field(name='dF_field')


def hessian_product(x, v):
    """Hessian of E at x applied to v = (v_u, v_f):
       dF = B v_u + v_f E_ll,  dP = A_t : dF,  H v = (B^T w dP, sum w dP_ll)"""
    if tangent_state['x'] is None or not np.array_equal(tangent_state['x'], x):
        set_deformation_gradient(*split(x))
        material.get_algorithmic_tangent(total_strain_field, tangent_field)
        tangent_state['x'] = x.copy()

    v_u, v_f = split(v)
    displacement_increment_field.s[...] = xp.asarray(v_u)
    discretization.fft.communicate_ghosts(displacement_increment_field)
    discretization.apply_gradient_operator_mugrid(u_inxyz=displacement_increment_field, grad_u_ijqxyz=dF_field)
    dF_field.s[l, l] += v_f                                         # the lateral stretch increment
    discretization.apply_material_data_mugrid(material_data=tangent_field,
                                              gradient_field=dF_field)  # dF_field now holds dP = A_t : dF
    discretization.fft.communicate_ghosts(dF_field)
    discretization.apply_gradient_transposed_operator_mugrid(gradient_field_ijqxyz=dF_field,
                                                             div_u_fnxyz=hessian_vector_field,
                                                             apply_weights=True)
    return join_gradient(host_copy(hessian_vector_field.s), sum_ll(dF_field))


# preconditioner for f: 1 / (d^2E/dx_f^2 of the reference material) = 1 / (F_SCALE^2 A C_ref_llll)
f_preconditioner = 1.0 / (F_SCALE ** 2 * cell_size[0] * cell_size[1] * ref_mat[l, l, l, l])


def preconditioner_Green(r):
    """Green preconditioner for the u part (as in the original), a scalar for the f part"""
    r_u, r_f = split(r)
    precond_input_field.s[...] = xp.asarray(r_u)
    M_fun_Green(precond_input_field, precond_output_field)
    return join(host_copy(precond_output_field.s), F_SCALE * r_f * f_preconditioner)


# ============================================================================
# incremental loading — trust-region Newton
# ============================================================================
sum_CG_its      = 0
sum_Newton_its  = 0
iteration_total = 0
start_time      = time.time()

x = join(np.zeros(u_shape), 0.0)                        # start from the undeformed state: u = 0, f = 0
homogenized_stress_log = []                             # per increment: inc, <F>, <P>

for inc in range(ninc):
    if discretization.communicator.rank == 0:
        print(f'Increment {inc}')
        print('=' * 70)

    # apply macroscopic deformation gradient increment
    macro_gradient_total += macro_gradient_inc

    # stop when the largest residual entry has dropped by a factor 1e8 within this increment
    # (the original: norm of the Newton update / norm of F < 1e-8). 1e-6 is not enough: the residual is
    # dominated by the stiff inclusion (1000 x the matrix), and <P> was then still up to 20 % off;
    # 1e-10 is below what the energy resolves in double precision (the trust region collapses)
    energy_0, gradient_0 = energy_and_gradient(x)
    norm_rhs_0 = discretization.communicator.max(np.abs(gradient_0).max())
    if discretization.communicator.rank == 0:
        print(f'Rhs at new load step    {norm_rhs_0:10.2e}')
        print(f'En  at new load step    {energy_0:10.2e}')

    # count the CG iterations: one Hessian product per CG iteration
    nb_hessian_products = [0]

    def hessian_product_counted(x, v):
        nb_hessian_products[0] += 1
        return hessian_product(x, v)

    result = tr_newton_bounded(
        energy_and_gradient, x, hessian_product_counted,
        jac=True,
        precond=preconditioner_Green,
        gtol=1e-8 * norm_rhs_0,         # stop: max |dE/du| <= gtol
        maxiter=100,                    # Newton iterations per increment
        inner_tol=cg_tol,               # CG tolerance (None: adaptive, see cg_tol)
        comm=MPI.COMM_WORLD
    )
    x = result.x
    u, f = split(x)

    sum_CG_its      += nb_hessian_products[0]
    sum_Newton_its  += result.nit
    iteration_total += result.nit

    # constitutive response at the solution (fills the fields for plotting and saving)
    energy, gradient = energy_and_gradient(x)
    material.get_energy_density(total_strain_field, energy_field)

    # homogenized 1st Piola-Kirchhoff stress  <P> = (1 / cell area) * sum over quadrature points of w_q P_q
    # (the quadrature weights include the pixel area; the holes contribute P = 0)
    P_sum = to_host(xp.einsum('ijqxy,q->ij', stress_field.s, quadrature_weights))
    P_mean = MPI.COMM_WORLD.allreduce(P_sum, op=MPI.SUM) / (cell_size[0] * cell_size[1])
    F_mean = np.eye(dim) + macro_gradient_total
    F_mean[l, l] += f
    homogenized_stress_log.append([inc, *F_mean.ravel(), *P_mean.ravel()])

    if discretization.communicator.rank == 0:
        print(f'  {result.message}')
        print(f'  Newton its = {result.nit}  |  CG its = {nb_hessian_products[0]}')
        print(f'  energy                          {energy:10.2e}')
        print(f'  max |rhs| / max |rhs_0|         {result.max_grad / norm_rhs_0:10.2e}')
        print(f'  <P> (units of mu_m)             [[{P_mean[0, 0]:+.4e}, {P_mean[0, 1]:+.4e}], '
              f'[{P_mean[1, 0]:+.4e}, {P_mean[1, 1]:+.4e}]]')
        print(f'  lateral stretch F_ll - 1        {f:+.5f}   (<P_ll> / <P_kk> = {P_mean[l, l] / P_mean[k, k]:.1e})')
    if not result.success:
        raise RuntimeError(f'trust-region Newton did not converge in increment {inc}: {result.message}')

    # save per load increment (one trust-region solve per increment)
    if save_per_it:
        save_npy(
            data_folder_path + f'displacement_fluctuation_field_inc{inc}.npy',
            to_host(displacement_fluctuation_field.s.mean(axis=1)),
            tuple(discretization.subdomain_locations_no_buffers),
            tuple(discretization.nb_of_pixels_global),
            MPI.COMM_WORLD
        )
        save_npy(
            data_folder_path + f'energy_field_inc{inc}.npy',
            to_host(energy_field.s[0, 0].mean(axis=0)),
            tuple(discretization.subdomain_locations_no_buffers),
            tuple(discretization.nb_of_pixels_global),
            MPI.COMM_WORLD
        )
        save_npy(
            data_folder_path + f'stress_field_inc{inc}.npy',
            to_host(stress_field.s.mean(axis=2)),
            tuple(discretization.subdomain_locations_no_buffers),
            tuple(discretization.nb_of_pixels_global),
            MPI.COMM_WORLD
        )

    if plot_load_cases and discretization.communicator.size == 1:
        # Plot the first two components of the solution field
        # Calculate total macroscopic gradient (F - I)
        total_macro_gradient = F_mean - np.eye(dim)            # prescribed F_kk and the computed F_ll
        # visualization_utils works on a unit cell [0,1]^2 -> same deformation in unit-cell coordinates
        L_cell = np.diag(domain_size)
        u_unit_cell.s[0] = displacement_fluctuation_field.s[0] / domain_size[0]
        u_unit_cell.s[1] = displacement_fluctuation_field.s[1] / domain_size[1]
        x_plot_ixyz = visualization_utils.get_deformed_grid_coords_two_dim(discretization,
                                                                           macro_gradient_ij=np.linalg.inv(L_cell) @ total_macro_gradient @ L_cell,
                                                                           displacement_fluctuation=u_unit_cell)
        visualization_utils.plot_field_on_grid(
            coordinates_for_plot=x_plot_ixyz,
            field_to_plot=to_host(phase_field.s[0, 0]),
            name=f'phase (0 matrix, 1 void, 2 stiff), load increment {inc}')
        if not args.show:
            plt.savefig(figure_folder_path + f'deformed_phase_inc{inc}.png', dpi=150)
        plt.close('all')        # free the figure (else memory grows with every increment)
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
    print(f'element_type     : {element_type}')
    print(f'number_of_pixels : {number_of_pixels}')
    print(f'preconditioner   : {preconditioner_type}')
    print(f'Total CG its     : {sum_CG_its}')
    print(f'Total Newton its : {sum_Newton_its}')
    print(f'Elapsed time     : {elapsed_time:.2f} s  ({elapsed_time / 60:.2f} min)')
    np.savez(data_folder_path + 'info_log_final.npz', **_info)
    # stress-strain curve: one row per load increment, <P> in units of mu_m (times MU_M_MPA for MPa)
    np.savetxt(data_folder_path + 'homogenized_stress.csv', np.array(homogenized_stress_log), delimiter=',',
               header='inc,F_00,F_01,F_10,F_11,P_00,P_01,P_10,P_11', comments='', fmt='%.10e')
    print(f'homogenized stress -> {data_folder_path}homogenized_stress.csv')