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

SAVING
------
One NetCDF file, written frame by frame as the run proceeds:

  <output>   frames of u_fluc, F and P, plus the static phase field.
             Every `dump_every` increments, and the final increment always.
             Global attributes carry the run parameters, the command line
             and the per-increment histories.

The phase field says which material each pixel is: phase > 0 is the matrix,
phase == 0 the third medium.  That mapping is written as global attributes
(`phase_matrix_value`, `phase_void_value`, `phase_note`) alongside the field
itself, so the array is readable without this script.  If muGrid cannot lay
the field out, it falls back to `phase_field_flat` + `phase_field_shape`.

det F is not stored: it is a pure function of F.  Recompute it on read with
J = F[0,0]*F[1,1] - F[0,1]*F[1,0].  (As a quad scalar it also trips muGrid's
NetCDF layout check.)  min(det F) per increment is in `hist_min_J`.

All fields of a frame go out in ONE write() call.  muGrid opens a new frame
per append_frame(), so writing them one at a time would scatter them across
separate frames and leave each frame's other variables at their fill value.

Global attributes are written as placeholders up front and overwritten at the
end, so a run that is killed still leaves a readable file.
"""

import os
import sys
import time
import shlex
import inspect

import muGrid
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

# ============================================================================
# problem setup
# ============================================================================
nnn = 32
ninc = 10

# plotting: deformed mesh every `plot_every` increments (0 = only the final
# response curves).  Serial runs only -- the fields are distributed under MPI.
plot_every = 1

# saving: write a frame every `dump_every` increments.  The final increment
# is always written.  0 = final frame only.
dump_every = 1
no_flush = False           # True = do not sync each frame as it is written
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
H_macro[1, 0] = -0.2

discretization.get_macro_gradient_field_mugrid(
    macro_gradient_ij=H_macro,
    macro_gradient_field_ijqxyz=macro_gradient_full_field
)

# ============================================================================
# output file: fields as frames, scalars as global attributes
#
# Placeholders for everything known only at the end are written now, so a
# killed run still leaves a file that opens and says converged=0.
# ============================================================================
hist = {k: [] for k in ('lam', 'F10', 'P10', 'Pxx', 'min_J', 'energy',
                        'nb_outer', 'nb_hessp', 'nb_precond', 'grad_inf',
                        'converged')}

output_path = data_folder_path + output_name
_MSG_LEN = 256

fio = muGrid.FileIONetCDF(output_path,
                          muGrid.FileIONetCDF.OpenMode.Overwrite,
                          discretization.communicator)

# Every field of a frame must go out in ONE write() call (see module
# docstring), so they are collected into a single list.
#
# det F is deliberately NOT stored: it is a pure function of F, which is
# here, and as a quad scalar its (1, 1, q, x, y) layout trips muGrid's
# NetCDF consistency check (count has 6 entries, imap 5).  Recompute it on
# read with  J = F[0,0]*F[1,1] - F[0,1]*F[1,0].
frame_fields = []
for _name in ['u_fluc', 'F', 'P', 'phase_field']:
    try:
        fio.register_field_collection(discretization.field_collection,
                                      field_names=[_name])
    except RuntimeError as _err:
        # A field muGrid cannot lay out should not cost the whole run, which
        # would otherwise die before computing anything.
        root_print(f'WARNING: field {_name!r} not written '
                   f'({str(_err).strip().splitlines()[-1][:60]})')
    else:
        frame_fields.append(_name)

if not frame_fields:
    raise RuntimeError('no field could be registered for output')
root_print('writing fields: ' + ', '.join(frame_fields))

# The phase field is static and only says which material each pixel is, so
# what it MEANS has to travel with it: without the mapping below a reader
# sees an array of 0s and 1s and cannot tell which is the third medium.
fio.write_global_attribute('phase_matrix_value', [1.0])   # phase > 0
fio.write_global_attribute('phase_void_value', [0.0])     # phase == 0
fio.write_global_attribute('phase_note',
                           'phase > 0: matrix (E_matrix); '
                           'phase == 0: third medium (k_v * E_matrix)')
_vol_matrix = global_sum(float(np.count_nonzero(matrix_mask)))
_vol_total = global_sum(float(matrix_mask.size))
fio.write_global_attribute('volume_fraction_matrix',
                           [_vol_matrix / max(_vol_total, 1.0)])

# Fallback: if the phase field could not be registered as a per-frame field,
# store it as a global attribute instead -- it is static, so one copy is all
# that was ever needed.  Gathering only makes sense on one rank.
if 'phase_field' not in frame_fields:
    if discretization.communicator.size == 1:
        fio.write_global_attribute(
            'phase_field_flat',
            [float(v) for v in np.asarray(phase_field.s[0, 0]).ravel()])
        fio.write_global_attribute(
            'phase_field_shape',
            [int(n) for n in np.asarray(phase_field.s[0, 0]).shape])
        root_print('phase field stored as a global attribute '
                   '(phase_field_flat, C order)')
    else:
        root_print('WARNING: phase field not stored (needs a serial run to '
                   'gather, or a working field registration)')

fio.write_global_attribute('command_line', shlex.join(sys.argv))
fio.write_global_attribute('geometry', geometry_name)
fio.write_global_attribute('element_type', element_type)
fio.write_global_attribute('formulation', formulation)
fio.write_global_attribute('nb_grid_pts', [int(n) for n in number_of_pixels])
fio.write_global_attribute('domain_size', [float(x) for x in domain_size])
fio.write_global_attribute('H_macro', [float(x) for x in H_macro.ravel()])
fio.write_global_attribute('ninc', [int(ninc)])
fio.write_global_attribute('dump_every', [int(dump_every)])
fio.write_global_attribute('E_matrix', [float(E_matrix)])
fio.write_global_attribute('nu_matrix', [float(nu_matrix)])
fio.write_global_attribute('k_v', [float(k_v)])
fio.write_global_attribute('alpha', [float(alpha)])
fio.write_global_attribute('k_r', [float(k_r)])
fio.write_global_attribute('gtol', [float(SOLVER_GTOL)])
fio.write_global_attribute('inner_tol', [float(SOLVER_INNER_TOL)])
fio.write_global_attribute('maxiter', [int(SOLVER_MAXITER)])

_maxlen = ninc
_max_frames = (_maxlen // dump_every + 3) if dump_every > 0 else 2
fio.write_global_attribute('converged', [0])
fio.write_global_attribute('nb_not_converged', [0])
fio.write_global_attribute('lam_reached', [0.0])
fio.write_global_attribute('elapsed_time', [0.0])
fio.write_global_attribute('final_message', ' ' * _MSG_LEN)
for _name in ('lam', 'F10', 'P10', 'Pxx', 'min_J', 'energy', 'grad_inf'):
    fio.write_global_attribute(f'hist_{_name}', [0.0] * _maxlen)
for _name in ('nb_outer', 'nb_hessp', 'nb_precond', 'converged'):
    fio.write_global_attribute(f'hist_{_name}', [0] * _maxlen)
fio.write_global_attribute('frame_increments', [-1] * _max_frames)

flush_frames = (not no_flush) and hasattr(fio, 'sync')
frame_increments = []


def write_frame(inc):
    """Commit the current u / F / P as one new frame.

    The fields must already hold the state of interest; this only writes.
    """
    fio.append_frame().write(frame_fields)
    if flush_frames:
        fio.sync()
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
               f'P_xx {stress_field.s[0, 0].mean():10.3e} | '
               f'P_yx {stress_field.s[1, 0].mean():10.3e}')

    hist['lam'].append(lam_current)
    hist['F10'].append(lam_current * H_macro[1, 0])
    hist['P10'].append(float(stress_field.s[1, 0].mean()))
    hist['Pxx'].append(float(stress_field.s[0, 0].mean()))
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
        write_frame(inc)

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
    write_frame(ninc)


def _upd(name, value):
    fio.update_global_attribute(name, name, value)


_upd('converged', [int(nb_not_converged == 0)])
_upd('nb_not_converged', [nb_not_converged])
_upd('lam_reached', [float(lam_current)])
_upd('elapsed_time', [float(elapsed)])
_upd('final_message', str(res.message)[:_MSG_LEN])
for _name in ('lam', 'F10', 'P10', 'Pxx', 'min_J', 'energy', 'grad_inf',
              'nb_outer', 'nb_hessp', 'nb_precond', 'converged'):
    if hist[_name]:
        _upd(f'hist_{_name}', hist[_name])
_upd('frame_increments', frame_increments)
fio.close()

if rank == 0:
    print(f'elapsed            : {elapsed:.1f} s')
    print(f'increments not converged: {nb_not_converged}'
          + ('   <-- those states are NOT equilibria'
             if nb_not_converged else ''))
    print(f'wrote              : {output_path} '
          f'({len(frame_increments)} frame(s))')

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