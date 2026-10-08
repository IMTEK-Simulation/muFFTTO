"""
2D third-medium contact, neo-Hookean finite strain, solved as an ENERGY
MINIMIZATION with NuMPI's trust-region Newton-CG.

    Pi(u) = int_Omega W(F) dx
          + k_r/2 int_Omega ( Hu : Hu - (1/dim) Lu . Lu ) dx

    grad Pi = G^T P   + R u
    hess Pi = G^T C G + R        with  R = k_r (H^T H - L^T L/dim)

The inner Steihaug CG is preconditioned with the Green operator
M = G^T C_ref G + R, built once from a fixed reference material.

The load is applied in equal increments of lam from 0 to 1, and each
increment is one plain call to `tr_newton_bounded`.

SAVING (for muEye)
------------------
One NetCDF file, written with muFFTTO.io_utils.FieldWriter, one frame per
stored increment (`dump_every`).  Fields muEye can render:

  u_total      displacement, dim components   -> Dataset panel "Displacement"
  u_fluc_only  the fluctuation alone, same units
  phase_field  1 = matrix, 0 = third medium
  detF         det F
  F_flat       F, 4 components, c = i*dim + j
  P_flat       P, 4 components, c = i*dim + j

plus the raw fluctuation `u_fluc` (physical units, restart with
io_utils.load_fields) and, as frame variables, the increment history: lam,
applied_deformation_gradient, F10, P10, Pxx, min_det_F, energy, nb_outer,
nb_hessp, nb_precond, grad_inf, converged, elapsed_time.  Run parameters are
global attributes.  Every frame is synced to disk, so a killed run still
leaves a complete file up to its last frame.  Read it with read_tmc.py
(io_utils.read_file).

In muEye: Dataset panel -> Displacement = u_total, then pick a field and a
component.  Warp scale 1.0 is the physical deformation.

Three things muEye's format forces, each of which the raw solver fields break:

1. Geometry is warped as x = C.s + u, with C from the `deformation_gradient`
   global attribute -- written ONCE at file creation, so it cannot follow
   lam.  The macro part is therefore folded into the displacement,

       u_total = lam * H_macro . X + u_fluc

   and the cell is left orthogonal (deformation_gradient = I).  The full
   deformed configuration then animates across frames, which the attribute
   route cannot do.

2. "Fields on a sub-point (quadrature) subdivision are read, but derived
   scalars operate on sub-point 0 only."  F and P live on quadrature points,
   so muEye would show one quadrature point rather than the average.  The
   copies here are averaged over the quadrature axis.  NOTE this smooths:
   a single collapsing quadrature point looks milder than min_det_F says.

3. A (i, j, q, x, y) field gives dimensions tensor_dim__F-0 and -1, which a
   collection registering F as one 4-component field cannot map -- the
   'has 2 entries ... expects 4 entries' error.  One component axis avoids
   it.
"""

import os
import sys
import time
import shlex
import inspect

import numpy as np
from mpi4py import MPI
from matplotlib import pyplot as plt

from NuMPI.Optimization import tr_newton_bounded

# `precond` exists only in a hand-patched BoundedTRNewtonCG.py; NuMPI reports
# version 0.15.1 either way, so fail here instead of with a bare TypeError
# several hundred lines down.
if 'precond' not in inspect.signature(tr_newton_bounded).parameters:
    raise RuntimeError(
        'Installed NuMPI has no `precond` support; this script needs the '
        'patched BoundedTRNewtonCG.py. Stock NuMPI 0.15.1 will not work.')

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from muFFTTO import domain, tensor_operations
from muFFTTO import microstructure_library
from muFFTTO import material_models
from muFFTTO import visualization_utils
from muFFTTO import io_utils

# ============================================================================
# problem setup
# ============================================================================
nnn = 128
ninc = 100

# plotting: deformed mesh every `plot_every` increments (0 = only the final
# response curves).  Serial runs only -- the fields are distributed under MPI.
plot_every = 0

# saving: write a frame every `dump_every` increments.  The final increment
# is always written.  0 = final frame only.  The increment history (frame
# variables) is stored with the frames, so it is complete only for 1.
dump_every = 1
output_name = 'tmc_run.nc'

# solver settings, named so they are recorded in the output file too
SOLVER_GTOL = 1e-4
SOLVER_MAXITER = 500
SOLVER_INNER_TOL = 1e-6

start_time = time.time()

number_of_pixels = (nnn, nnn)
domain_size = [1, 1]
dim = len(domain_size)
problem_type = 'elasticity'
discretization_type = 'finite_element'
element_type = 'bilinear_rectangle'
formulation = 'finite_strain'

my_cell = domain.PeriodicUnitCell(domain_size=domain_size,
                                  problem_type=problem_type)

discretization = domain.Discretization(cell=my_cell,
                                       nb_of_pixels_global=number_of_pixels,
                                       discretization_type=discretization_type,
                                       element_type=element_type)

comm = MPI.COMM_WORLD
rank = discretization.communicator.rank


def root_print(*a, **kw):
    if rank == 0:
        print(*a, **kw)


# ============================================================================
# output folder
# ============================================================================
script_name = os.path.splitext(os.path.basename(__file__))[0]
file_folder_path = os.path.dirname(os.path.realpath(__file__))
data_folder_path = (file_folder_path + '/exp_data/' + script_name + '/'
                    + f'Nx={nnn}Ny={nnn}/')
if rank == 0:
    os.makedirs(data_folder_path, exist_ok=True)
comm.Barrier()

# ============================================================================
# material parameters
# ============================================================================
# matrix: soft neo-Hookean
E_matrix = 100.0
nu_matrix = 0.3
lam_matrix = E_matrix * nu_matrix / ((1 + nu_matrix) * (1 - 2 * nu_matrix))
mu_matrix = E_matrix / (2 * (1 + nu_matrix))
K, G = material_models.get_bulk_and_shear_modulus(E_matrix, nu_matrix)

# third medium ("void"): same neo-Hookean, k_v times softer
k_v = 1e-5
E_void = k_v * E_matrix
nu_void = nu_matrix
lam_void = E_void * nu_void / ((1 + nu_void) * (1 - 2 * nu_void))
mu_void = E_void / (2 * (1 + nu_void))

# HuHu-LuLu regularization
alpha = 1e-6
k_r = alpha * domain_size[0] ** 2 * (K + G * 4 / 3)
inv_tr_I = 1.0 / dim

# reference material for the Green preconditioner
_i = np.eye(dim)
II = np.einsum('ij,kl->ijkl', _i, _i)
I4rt = np.einsum('ik,jl->ijkl', _i, _i)
I4s = 0.5 * (I4rt + np.einsum('il,jk->ijkl', _i, _i))
ref_mat = lam_matrix * II + 2.0 * mu_matrix * I4s

# ============================================================================
# geometry
# ============================================================================
geometry_name = 'contact_test_geometry_2'

phase_field = discretization.get_scalar_field(name='phase_field')
phase_field.s[0, 0] = microstructure_library.get_geometry(
    nb_voxels=discretization.nb_of_pixels,
    microstructure_name=geometry_name,
    coordinates=discretization.fft.coords
)

matrix_mask = phase_field.s[0, 0] > 0
inc_mask = phase_field.s[0, 0] == 0

# ============================================================================
# material fields
# ============================================================================
lam_field = discretization.get_quad_field_scalar(name='lam_field')
mu_field = discretization.get_quad_field_scalar(name='mu_field')

lam_field.s[0, 0, :, matrix_mask] = lam_matrix
lam_field.s[0, 0, :, inc_mask] = lam_void
mu_field.s[0, 0, :, matrix_mask] = mu_matrix
mu_field.s[0, 0, :, inc_mask] = mu_void

material = material_models.NeoHookean(
    discretization=discretization,
    lam_1qxyz=lam_field,
    mu_1qxyz=mu_field,
    name='neo_hookean_two_phase'
)

# ============================================================================
# fields
#
# CONVENTION: macro_gradient_full_field holds H_macro WITHOUT the identity;
# `set_F` adds I.
# ============================================================================
macro_gradient_full_field = discretization.get_gradient_size_field(
    name='macro_gradient_full_field')
grad_u_field = discretization.get_displacement_gradient_sized_field(name='grad_u')
total_strain_field = discretization.get_displacement_gradient_sized_field(name='F')
stress_field = discretization.get_displacement_gradient_sized_field(name='P')
tangent_field = discretization.get_material_data_size_field_mugrid(name='C')
energy_field = discretization.get_quad_field_scalar(name='energy_density')
J_1qxyz = discretization.get_quad_field_scalar(name='J')

x_field = discretization.get_unknown_size_field(name='x_field')
grad_field = discretization.get_unknown_size_field(name='grad_field')
Ru_field = discretization.get_unknown_size_field(name='Ru_field')

v_field = discretization.get_unknown_size_field(name='v_field')
Bv_field = discretization.get_unknown_size_field(name='Bv_field')

# preconditioner scratch, kept separate from the hessp path
r_field = discretization.get_unknown_size_field(name='r_field')
Pv_field = discretization.get_unknown_size_field(name='Pv_field')

hess_u_ijkqxyz = discretization.get_displacement_hessian_size_field(name='hess_u')
HtH_field = discretization.get_unknown_size_field(name='HtH')
lap_u_inxyz = discretization.get_displacement_laplacian_at_quad_field(name='lap_u')
LtL_field = discretization.get_displacement_sized_field(name='LtL')

displacement_fluctuation_field = discretization.get_unknown_size_field(
    name='u_fluc')

u_shape = np.asarray(displacement_fluctuation_field.s).shape
_qw = np.asarray(discretization.quadrature_weights).reshape((-1,) + (1,) * dim)


# ============================================================================
# reductions
# ============================================================================
def global_sum(a):
    return comm.allreduce(float(a), op=MPI.SUM)


def global_min(a):
    return comm.allreduce(float(a), op=MPI.MIN)


def global_max(a):
    return comm.allreduce(float(a), op=MPI.MAX)


def dot_global(a, b):
    return global_sum(np.dot(np.asarray(a).ravel(), np.asarray(b).ravel()))


def global_mean(a):
    a = np.asarray(a)
    return global_sum(np.sum(a)) / max(global_sum(float(a.size)), 1.0)


# ============================================================================
# deformation gradient
# ============================================================================
def min_J(F_in):
    tensor_operations.det2(F_in, J_1qxyz)
    return global_min(np.min(J_1qxyz.s))


def set_F(lam, u_field, out):
    """F = I + lam * H_macro + grad(u~), rebuilt from scratch."""
    discretization.fft.communicate_ghosts(u_field)
    discretization.apply_gradient_operator_mugrid(u_inxyz=u_field,
                                                  grad_u_ijqxyz=grad_u_field)
    out.s[...] = grad_u_field.s + lam * macro_gradient_full_field.s
    for d in range(dim):
        out.s[d, d] += 1.0


# ============================================================================
# regularization operator  R = k_r (H^T H - L^T L / dim)
# ============================================================================
def add_regularization(x, out, scale=1.0):
    """out += scale * R x.  Ghosts of x must already be current."""
    discretization.apply_hessian_operator_to_vector_field_mugrid(
        u_inxyz=x, hess_u_ijkqxyz=hess_u_ijkqxyz)
    discretization.apply_hessian_operator_transposed_to_vector_field_mugrid(
        hess_u_ijkqxyz=hess_u_ijkqxyz, nodal_field_inxyz=HtH_field,
        apply_weights=True)

    discretization.laplacian.apply(nodal_field=x,
                                   quadrature_point_field=lap_u_inxyz)
    discretization.laplacian.transpose(quadrature_point_field=lap_u_inxyz,
                                       nodal_field=LtL_field,
                                       weights=discretization.quadrature_weights)

    out.s[...] += scale * k_r * (HtH_field.s - inv_tr_I * LtL_field.s)


# ============================================================================
# Green preconditioner:  M = G^T C_ref G + R,  applied as M^-1
#
# Built ONCE from the fixed reference material.  With a preconditioner the
# trust region is measured in the M-norm, so M must not change mid-run or the
# meaning of the radius changes with it.
# ============================================================================
def operator_for_preconditioner(input_field_inxyz, output_field_inxyz):
    """The operator whose Fourier symbol the Green blocks invert."""
    discretization.fft.communicate_ghosts(field=input_field_inxyz)
    discretization.apply_system_matrix_mugrid(
        material_data_field=ref_mat,
        input_field_inxyz=input_field_inxyz,
        output_field_inxyz=output_field_inxyz,
        formulation=formulation)
    add_regularization(input_field_inxyz, output_field_inxyz)


preconditioner = discretization.get_preconditioner_Green_mugrid(
    reference_material_data_ijkl=ref_mat,
    operator=operator_for_preconditioner,
)


def precond(v):
    """Return M^-1 v."""
    r_field.s[...] = v
    discretization.fft.communicate_ghosts(r_field)
    discretization.apply_preconditioner_mugrid(
        preconditioner_Fourier_fnfnqks=preconditioner,
        input_nodal_field_fnxyz=r_field,
        output_nodal_field_fnxyz=Pv_field)
    return np.array(Pv_field.s, copy=True)


# ============================================================================
# objective, gradient, Hessian-vector product
#
# The load parameter is read from the module-level `lam_current`, which the
# increment loop sets before each solve.
# ============================================================================
lam_current = 0.0

# Large but FINITE: np.inf would make ared = inf - inf = nan, and a nan rho is
# never rejected *and* never shrinks the radius, so the solver spins to
# maxiter.  A finite penalty gives rho << 0, hence rejection and delta/4.
INADMISSIBLE_ENERGY = 1e30


def _set_F_from(x):
    x_field.s[...] = x
    set_F(lam_current, x_field, total_strain_field)
    return min_J(total_strain_field)


def fun_grad(x):
    """(Pi, grad Pi), globally reduced."""
    J_min = _set_F_from(x)
    if not np.isfinite(J_min) or J_min <= 0.0:
        return INADMISSIBLE_ENERGY, np.zeros(u_shape)

    material.get_energy_density(total_strain_field, energy_field)
    Pi = global_sum(np.sum(_qw * energy_field.s[0, 0]))
    if not np.isfinite(Pi):
        return INADMISSIBLE_ENERGY, np.zeros(u_shape)

    # regularization energy 1/2 <u, R u>, reusing R u for the gradient
    Ru_field.s[...] = 0.0
    add_regularization(x_field, Ru_field)
    Pi += 0.5 * dot_global(x_field.s, Ru_field.s)

    material.get_stress(total_strain_field, stress_field)
    discretization.fft.communicate_ghosts(stress_field)
    discretization.apply_gradient_transposed_operator_mugrid(
        gradient_field_ijqxyz=stress_field, div_u_fnxyz=grad_field,
        apply_weights=True)
    grad_field.s[...] += Ru_field.s

    return Pi, np.array(grad_field.s, copy=True)


def hessp(x, v):
    """B v = (G^T C G + R) v, with C evaluated at x."""
    _set_F_from(x)
    material.get_algorithmic_tangent(total_strain_field, tangent_field)

    v_field.s[...] = v
    discretization.fft.communicate_ghosts(v_field)
    discretization.apply_system_matrix_mugrid(
        material_data_field=tangent_field,
        input_field_inxyz=v_field,
        output_field_inxyz=Bv_field,
        formulation=formulation)
    add_regularization(v_field, Bv_field)
    discretization.fft.communicate_ghosts(Bv_field)

    return np.array(Bv_field.s, copy=True)


# ============================================================================
# macroscopic loading:  F_bar = I + lam * H_macro,  lam: 0 -> 1
# ============================================================================
H_macro = np.zeros((dim, dim))
H_macro[1, 0] = -0.3


discretization.get_macro_gradient_field_mugrid(
    macro_gradient_ij=H_macro,
    macro_gradient_field_ijqxyz=macro_gradient_full_field
)

# ============================================================================
# muEye view fields
#
# Pixel level, no sub-point axis, one component axis -- see the module
# docstring for why the raw F / P / u_fluc cannot be used directly.
# ============================================================================
def _pixel_field(name, ncomp):
    """A pixel-level field with `ncomp` components and no sub-point axis,
    written through `.p` with shape (ncomp, nx, ny)."""
    return discretization.field_collection.real_field(name, [ncomp], 'pixel')


# Undeformed node positions X, for the affine part of the displacement.
# fft.coords is integer grid indices in some muFFT versions and normalized
# [0, 1) coordinates in others, so the scaling is chosen accordingly.
_coords = np.asarray(discretization.fft.coords)
_scale = np.asarray(domain_size, dtype=float)
if np.issubdtype(_coords.dtype, np.integer):
    _scale = _scale / np.asarray(number_of_pixels, dtype=float)
_positions = _coords * _scale.reshape((dim,) + (1,) * dim)

# muEye's displacement is in GRID-POINT units (see its
# scripts/make_test_volume.py), while everything here is in physical length.
# Dividing by the grid spacing makes muEye's "Warp scale" = 1.0 the actual
# physical deformation rather than one 1/nb_pixels of it.
_grid_spacing = (np.asarray(domain_size, dtype=float)
                 / np.asarray(number_of_pixels, dtype=float))
_u_to_grid = (1.0 / _grid_spacing).reshape((dim,) + (1,) * dim)

view_fields = {
    'u_total': _pixel_field('u_total', dim),         # muEye "Displacement"
    # The fluctuation alone, same units.  Selecting this as the Displacement
    # in muEye shows the fluctuation on its own, with no affine part: if the
    # picture is flat, the fluctuation really is negligible; if it wiggles,
    # then u_total carries it too and the affine part is simply larger.
    'u_fluc_only': _pixel_field('u_fluc_only', dim),
    'detF': _pixel_field('detF', 1),
    'F_flat': _pixel_field('F_flat', dim * dim),     # c = i*dim + j
    'P_flat': _pixel_field('P_flat', dim * dim),
}

# Filled by update_view_fields, reported per frame: the affine and
# fluctuation parts of the warp, in grid points.
warp_magnitudes = {'affine': 0.0, 'fluct': 0.0}


def update_view_fields(lam):
    """Refresh the muEye copies from the current F / P / u state.

    `total_strain_field`, `stress_field` and `displacement_fluctuation_field`
    must already hold the state being written.  Pixel fields are written
    through `.p`, which is (ncomp, nx, ny) -- the layout muEye expects.
    """
    # displacement = affine macro part + stored fluctuation
    u_f = np.asarray(displacement_fluctuation_field.s)
    u_f = u_f.reshape((dim, -1) + u_f.shape[-dim:])[:, 0]
    # trim any ghost/buffer padding: the view fields are exactly nb_pixels
    nx_loc = view_fields['u_total'].p.shape[1:]
    u_f = u_f[(slice(None),) + tuple(slice(0, n) for n in nx_loc)]

    affine = np.tensordot(lam * H_macro, _positions, axes=(1, 0))
    affine = affine[(slice(None),) + tuple(slice(0, n) for n in nx_loc)]

    affine_g = affine * _u_to_grid
    fluct_g = u_f * _u_to_grid

    view_fields['u_total'].p[...] = affine_g + fluct_g
    view_fields['u_fluc_only'].p[...] = fluct_g

    warp_magnitudes['affine'] = global_max(np.max(np.abs(affine_g)))
    warp_magnitudes['fluct'] = global_max(np.max(np.abs(fluct_g)))

    # tensors: average over the quadrature axis, fold (i, j) -> i*dim + j
    Fp = np.asarray(total_strain_field.s).mean(axis=2)
    Pp = np.asarray(stress_field.s).mean(axis=2)
    view_fields['F_flat'].p[...] = Fp.reshape((dim * dim,) + Fp.shape[2:])
    view_fields['P_flat'].p[...] = Pp.reshape((dim * dim,) + Pp.shape[2:])

    J = Fp[0, 0] * Fp[1, 1] - Fp[0, 1] * Fp[1, 0]
    view_fields['detF'].p[...] = J.reshape((1,) + J.shape)


# ============================================================================
# output file (muFFTTO.io_utils): the view fields, the raw fluctuation and the
# phase field as frames, the increment history as frame variables, the run
# parameters as global attributes
# ============================================================================
hist = {k: [] for k in ('lam', 'F10', 'P10', 'Pxx', 'min_J', 'energy',
                        'nb_outer', 'nb_hessp', 'nb_precond', 'grad_inf',
                        'converged')}

output_path = data_folder_path + output_name
_vol_matrix = global_sum(float(np.count_nonzero(matrix_mask)))
_vol_total = global_sum(float(matrix_mask.size))

writer = io_utils.FieldWriter(
    output_path,
    [view_fields['u_total'], view_fields['u_fluc_only'], phase_field,
     view_fields['detF'], view_fields['F_flat'], view_fields['P_flat'],
     displacement_fluctuation_field],
    attributes={
        # muEye reads the cell shape from this attribute, and global
        # attributes are defined at file creation -- so it cannot follow lam.
        # Identity keeps the cell orthogonal; the macro deformation rides in
        # u_total instead, which is what lets the deformed shape animate.
        'deformation_gradient': np.eye(dim),
        'displacement_field': 'u_total',
        'displacement_units': 'grid points (physical displacement / grid '
                              'spacing), so Warp scale 1.0 is the physical '
                              'deformation',
        'component_order': 'F_flat / P_flat: component c = i*dim + j '
                           '(row-major), quadrature-averaged to pixel values',
        # what the 0s and 1s of the phase field mean travels with the file
        'phase_matrix_value': 1.0,
        'phase_void_value': 0.0,
        'phase_note': 'phase > 0: matrix (E_matrix); '
                      'phase == 0: third medium (k_v * E_matrix)',
        'volume_fraction_matrix': _vol_matrix / max(_vol_total, 1.0),
        # run parameters
        'command_line': shlex.join(sys.argv),
        'geometry': geometry_name,
        'element_type': element_type,
        'formulation': formulation,
        'nb_grid_pts': number_of_pixels,
        'domain_size': np.asarray(domain_size, dtype=float),
        'H_macro': H_macro,
        'ninc': ninc,
        'dump_every': dump_every,
        'E_matrix': E_matrix,
        'nu_matrix': nu_matrix,
        'k_v': k_v,
        'alpha': alpha,
        'k_r': k_r,
        'gtol': SOLVER_GTOL,
        'inner_tol': SOLVER_INNER_TOL,
        'maxiter': SOLVER_MAXITER},
    # per-increment, grid-less values; `applied_deformation_gradient` is the
    # macro F_bar of the frame, which the file-constant
    # `deformation_gradient` attribute cannot express.  converged: 1 = the
    # increment reached gtol, 0 = it did not and is not an equilibrium.
    frame_variables={'increment': (), 'lam': (),
                     'applied_deformation_gradient': (dim, dim),
                     'F10': (), 'P10': (), 'Pxx': (), 'min_det_F': (),
                     'energy': (), 'nb_outer': (), 'nb_hessp': (),
                     'nb_precond': (), 'grad_inf': (), 'converged': (),
                     'elapsed_time': ()})

frame_increments = []


def write_frame(inc, lam):
    """Refresh the muEye view fields and append them as one new frame."""
    update_view_fields(lam)
    last = {k: v[-1] for k, v in hist.items() if v}
    writer.write(increment=inc, lam=lam,
                 applied_deformation_gradient=np.eye(dim) + lam * H_macro,
                 F10=last.get('F10', np.nan), P10=last.get('P10', np.nan),
                 Pxx=last.get('Pxx', np.nan), min_det_F=last.get('min_J', np.nan),
                 energy=last.get('energy', np.nan),
                 nb_outer=last.get('nb_outer', 0), nb_hessp=last.get('nb_hessp', 0),
                 nb_precond=last.get('nb_precond', 0),
                 grad_inf=last.get('grad_inf', np.nan),
                 converged=last.get('converged', 0),
                 elapsed_time=time.time() - start_time)
    frame_increments.append(int(inc))


# ============================================================================
# incremental loading
# ============================================================================
u = np.zeros(u_shape)

serial = discretization.communicator.size == 1

root_print('=' * 70)
root_print(f'geometry     : {geometry_name}')
root_print(f'element_type : {element_type}')
root_print(f'pixels       : {number_of_pixels}')
root_print(f'increments   : {ninc}')
root_print(f'output       : {output_path}')
root_print(f'dump_every   : {dump_every}'
           + ('   (final frame only)' if dump_every <= 0 else ''))

for inc in range(1, ninc + 1):
    lam_current = inc / float(ninc)

    res = tr_newton_bounded(
        fun=fun_grad,
        x0=u,
        hessp=hessp,
        precond=precond,
        jac=True,
        gtol=SOLVER_GTOL,
        maxiter=SOLVER_MAXITER,
        inner_tol=SOLVER_INNER_TOL,
        inner_maxiter=None,
        comm=comm,
    )

    u = np.asarray(res.x)

    J_min = _set_F_from(u)
    material.get_stress(total_strain_field, stress_field)
    displacement_fluctuation_field.s[...] = u

    root_print('-' * 70)
    root_print(f'increment {inc:4d}   lam = {lam_current:.4f}   '
               f'lam*H_10 = {lam_current * H_macro[1, 0]:+.5f}')
    root_print(f'  {res.message}')
    root_print(f'  outer its {res.nit:4d} | hessp {res.nb_hessp:6d} '
               f'| |grad|_inf {res.max_grad:10.3e} | Pi {res.fun:12.6e}')
    root_print(f'  min(det F) {J_min:10.3e} | '
               f'P_xx {global_mean(stress_field.s[0, 0]):10.3e} | '
               f'P_yx {global_mean(stress_field.s[1, 0]):10.3e}')

    hist['lam'].append(lam_current)
    hist['F10'].append(lam_current * H_macro[1, 0])
    hist['P10'].append(float(global_mean(stress_field.s[1, 0])))
    hist['Pxx'].append(float(global_mean(stress_field.s[0, 0])))
    hist['min_J'].append(J_min)
    hist['energy'].append(float(res.fun))
    hist['nb_outer'].append(int(res.nit))
    hist['nb_hessp'].append(int(res.nb_hessp))
    hist['nb_precond'].append(int(res.get('nb_precond', 0)))
    hist['grad_inf'].append(float(res.max_grad))
    # per-increment, not once at the end: an increment that hit maxiter is not
    # an equilibrium, and the file has to say which ones those were
    hist['converged'].append(int(bool(res.success)))

    if dump_every > 0 and inc % dump_every == 0:
        write_frame(inc, lam_current)
        root_print(f'  warp (grid pts): affine {warp_magnitudes["affine"]:8.3f}'
                   f'   fluctuation {warp_magnitudes["fluct"]:8.3f}'
                   f'   ratio {warp_magnitudes["fluct"] / max(warp_magnitudes["affine"], 1e-30):6.3f}')

    # ---- deformed mesh -----------------------------------------------------
    if serial and plot_every and inc % plot_every == 0:
        x_plot_ixyz = visualization_utils.get_deformed_grid_coords_two_dim(
            discretization,
            macro_gradient_ij=lam_current * H_macro,
            displacement_fluctuation=displacement_fluctuation_field)
        visualization_utils.plot_field_on_grid(
            coordinates_for_plot=x_plot_ixyz,
            field_to_plot=phase_field.s[0, 0],
            name=f'phase, increment {inc}   lam = {lam_current:.4f}')

root_print('=' * 70)
root_print(f'final min J : {min_J(total_strain_field):.6e}')

# ============================================================================
# finish the output file
# ============================================================================
elapsed = time.time() - start_time
nb_not_converged = int(sum(1 for c in hist['converged'] if not c))

# The final increment always gets a frame: it is the state every final plot
# describes, and the restart point.
if not frame_increments or frame_increments[-1] != ninc:
    write_frame(ninc, lam_current)

writer.close()

if rank == 0:
    print(f'elapsed            : {elapsed:.1f} s')
    print(f'increments not converged: {nb_not_converged}'
          + ('   <-- those states are NOT equilibria'
             if nb_not_converged else ''))
    print(f'wrote              : {output_path} '
          f'({len(frame_increments)} frame(s))')
    print('muEye: Dataset panel -> Displacement = u_total, '
          'then pick phase_field / detF / F_flat / P_flat')

# ============================================================================
# response curves
# ============================================================================
if rank == 0 and len(hist['lam']) > 1:
    fig, ax = plt.subplots(1, 4, figsize=(19, 4))

    ax[0].plot(hist['F10'], hist['P10'], '-o', ms=3, color='k')
    ax[0].set_xlabel(r'$\bar{F}_{10}$')
    ax[0].set_ylabel(r'$\bar{P}_{10}$')

    ax[1].plot(hist['F10'], hist['Pxx'], '-o', ms=3, color='C0')
    ax[1].set_xlabel(r'$\bar{F}_{10}$')
    ax[1].set_ylabel(r'$\bar{P}_{xx}$')

    ax[2].semilogy(hist['lam'], hist['min_J'], '-o', ms=3, color='C3')
    ax[2].set_xlabel(r'$\lambda$')
    ax[2].set_ylabel(r'$\min \det F$')

    ax[3].plot(hist['lam'], hist['energy'], '-o', ms=3, color='C2')
    ax[3].set_xlabel(r'$\lambda$')
    ax[3].set_ylabel(r'$\Pi$')

    for a in ax:
        a.grid(alpha=.3)

    fig.tight_layout()
    fig.savefig(data_folder_path + 'response.png', dpi=150)
    plt.show()