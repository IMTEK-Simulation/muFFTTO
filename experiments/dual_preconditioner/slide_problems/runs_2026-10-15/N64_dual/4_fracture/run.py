"""
Phase-field fracture with third-medium contact on an interface-fitted grid (2D, finite strain)
==============================================================================================

The model and the solver are those of
examples/phase_field_fracture/example_2D_phase_field_fracture_contact.py
(third-medium contact, AT2 fracture of the solid, split neo-Hookean law,
staggered scheme, trust-region Newton-CG). Here the grid is adapted to the
geometry: the nodes of the regular pixel grid X are moved, x = phi(X) = X + d_g(X),
so that element edges follow the solid / third-medium interface
(muFFTTO.grid_adaptation, method 'zecevic': Zecevic et al., Mech. Mater. 2026). The goal is a better representation of the geometry (the
contact surfaces), not refinement at the crack.

Pull-back. The unknowns u~(X), d(X), the FFT and the preconditioners stay on
the regular grid. With the grid map F_g = I + grad_X d_g, J_g = det F_g (per
quadrature point, isoparametric bilinear map):

    grad_x v = grad_X v . F_g^-1,      dx = J_g dX

so the deformation gradient is F = I + lam H_macro + grad_X u~ . F_g^-1 and every
integral gets the weight J_g:

    energy     sum_q w_q J_g W(F)
    gradient   B^T W [ J_g P(F) . F_g^-T ]
    Hessian    B^T W [ A' : B v ],   A'_ijkl = J_g A_imkn(F) F_g^-1_jm F_g^-1_ln
    damage     Gc l B^T W [ J_g F_g^-1 F_g^-T ] B  +  N^T W J_g (Gc/l + 2H) N,  rhs N^T W J_g 2H

The mean physical gradient is still I + lam H_macro (the fluctuation part
averages to zero over the deformed periodic cell) and the mean stress is
sum_q w_q J_g P / |cell|. The preconditioners (Green of the undamaged solid on
the regular grid, Green-Jacobi with the diagonal of the pulled-back Hessian)
do not see the grid map except through that diagonal.

Limitation: the HuHu contact regularization is evaluated in reference
coordinates (second derivatives with respect to X, not x). It is a stabilizer
of the collapsing third medium only; with d_g = 0 everything reduces exactly
to the regular-grid example.

Model of the regular-grid example:

Energy, with F = I + lam H_macro + grad u~ (u~ periodic):

    Pi(u~, d) = int  g_s(d) W_iso(F) + g_v(d, F) W_vol(F)  dx            elastic
              + k_r/2 int  Hu~ : Hu~ - (1/dim) Lu~ . Lu~  dx              HuHu-LuLu
              + Gc/2 int  d^2/l + l |grad d|^2  dx                        AT2 crack

    W_iso = mu/2 (J^-2/3 (I1 + 1) - 3)        isochoric (plane strain), I1 = F:F
    W_vol = kappa/2 ln(J)^2                   volumetric
    g(d)  = (1 - d)^2 + k                     degradation
    g_s   = g(d) in the solid, 1 in the third medium
    g_v   = g_s for J > 1 (expansion), 1 for J <= 1 (compression)

The split keeps compression undamaged, so crack faces and the crushed medium
still resist interpenetration. Only the solid can be damaged: the damage problem
is assembled over the solid elements only (natural boundary grad d . n = 0 at the
cavity surface), nodes touching only the medium stay d = 0
(damage_in_medium = True solves it on the whole cell). The third medium has the same law with
mu, kappa scaled by k_v. The second-gradient term (Faltus et al.) stabilizes
the collapsing medium. With Q2 elements ('biquadratic_rectangle') both its
Hessian and its Laplacian part act; with the default Q1 elements the
Laplacian of a bilinear field vanishes and only the mixed second derivative
is penalized.

Algorithm, per load increment lam -> lam + dlam (staggered):

    1. u~: minimize Pi(., d) with NuMPI's trust-region Newton-CG; the inner
       CG is preconditioned with a Green operator of the undamaged solid plus
       the regularization. det F <= 0 is rejected (large finite energy).
    2. H = max(H_{n-1}, psi^+(F)),  psi^+ = W_iso + <J > 1> W_vol  (solid only)
    3. d: [Gc l B^T W B + N^T W (Gc/l + 2H) N] d = N^T W 2H  (linear, CG with a
       Green(-Jacobi) preconditioner, as in example_2D_phase_field_fracture_AT2)
    repeat 1-3 until max|d - d_old| < staggered_tol, then H_{n-1} <- H.

The load step is adaptive. Each increment starts from the linear
extrapolation of the last two converged fluctuations (or from the last one if
that is not admissible); the step is halved when no admissible start exists
(det F <= 0 somewhere), the trust-region solve fails or the staggered loop
does not converge within max_staggered_iterations, and grows back after a
success.

Math and scheme: docs/examples/phase_field_fracture_contact.md.
Output: NetCDF via muFFTTO.io_utils (damage, u~, H, det F and phase, plus the
increment history); plot it with plot_phase_field_fracture_contact_output.py.
Needs the patched NuMPI with `precond` in tr_newton_bounded (see
docs/examples/internal_contact.md).
"""
import sys
import os
import time
import inspect

sys.path.insert(0, '/home/martin/PycharmProjects/muFFTTO')

import numpy as np  # noqa: E402
from mpi4py import MPI  # noqa: E402
from muGrid import Solvers  # noqa: E402
from NuMPI.Optimization import tr_newton_bounded  # noqa: E402

from muFFTTO import domain  # noqa: E402
from muFFTTO import solvers_nonlinear  # noqa: E402
from muFFTTO import microstructure_library  # noqa: E402
from muFFTTO import io_utils  # noqa: E402
from muFFTTO import grid_adaptation  # noqa: E402
from muFFTTO import tensor_operations  # noqa: E402

if 'precond' not in inspect.signature(tr_newton_bounded).parameters:
    raise RuntimeError('this example needs the patched NuMPI with `precond` in tr_newton_bounded')

# ----------------------------------------------------------------------------
# Parameters
# ----------------------------------------------------------------------------
nnn = 64                                   # pixels per direction
# where the fields live: 'cpu' or 'gpu' (GPU needs muGrid built with CUDA + cupy)
device = 'cpu'
number_of_pixels = (nnn, nnn)
domain_size = [1.0, 1.0]
# 'bilinear_rectangle' (Q1) or 'biquadratic_rectangle' (Q2). With Q1 on pixels the
# discrete Laplacian vanishes, so only the HuHu part of the regularization acts.
element_type = 'bilinear_rectangle'
geometry_name = 'bubble_array'      # any name from muFFTTO.geometry.available()
geometry_parameters = {}                   # passed to the geometry, e.g. {'notch_length': 0.1}

# load: F_bar = I + lam H_macro, lam: 0 -> 1; compression F11 = 1 - 0.27 lam with a small shear
# H10 = 0.01 that breaks the symmetry of the cell
H_macro = np.array([[0.0, 0.0],
                    [0.0, -0.4]])
nb_increments = 100                         # nominal increments (largest step 1/nb_increments)
lam_end = 1.0                              # the load runs to lam_end
min_step = 1.0 / nb_increments / 64        # smallest step after halving

# solid (neo-Hookean) and third medium
E_solid, nu_solid = 100.0, 0.3
k_v = 1e-5                                 # third-medium stiffness ratio
# material of the voids: 'neo_hookean' = third medium (the split neo-Hookean law of the solid scaled
# by k_v, undamaged; its energy kappa ln(J)^2 blows up as J -> 0, which transmits contact), or
# 'saint_venant_kirchhoff' = no contact: an isotropic, super-soft St. Venant-Kirchhoff solid
# (Lame constants of the solid scaled by k_v), finite for any F, also J <= 0 (no barrier); the
# admissibility J > 0 is then required in the solid only (SVK softens in strong compression, the
# Hessian may become indefinite), or 'linear' = no contact: super-soft small-strain linear elasticity,
# eps = sym(F - I), convex (constant tangent), finite for any F, J > 0 in the solid only
void_model = 'neo_hookean'
alpha = 1e-6                               # HuHu-LuLu regularization strength
# where the HuHu-LuLu term acts: 'everywhere' (solid and third medium) or 'medium' (the third medium only, as in
# Frederiksen et al., CMAME 436 (2025) 117595: in the solid it is a second-gradient stiffness that is not part of
# the material and penalizes the displacement jump of a crack). The Green preconditioner keeps the constant term.
regularization_region = 'everywhere'

# phase-field fracture (AT2)
Gc = 0.5                                   # fracture toughness
# 'uniform': Gc everywhere; 'random': a correlated random field Gc(x) with values spread uniformly over Gc_range
# (correlation length Gc_correlation_length, fixed seed). The field is defined in physical coordinates, so every
# grid samples the same material; Gc is then set to the mean of Gc_range (run name, damage preconditioner).
Gc_field = 'uniform'
Gc_range = (0.5, 1.0)
Gc_correlation_length = 0.1
Gc_seed = 0
# fixed in physical units (not in pixels), so a finer grid refines the same problem:
# 1 px at 32^2, 2 px at 64^2, 4 px at 128^2
length_scale = domain_size[0] / 32
k_residual = 1e-4                          # residual stiffness in g(d)
solve_damage_problem = True                # False: purely elastic (d = 0, no staggered loop)
staggered_tol = 1e-3                       # max |d - d_old|
max_staggered_iterations = 300            # not converged by then: the increment is retried with half the step
damage_preconditioner = 'Green_Jacobi'     # or 'Green'
# False: the damage problem lives on the solid only (natural boundary grad d . n = 0 at the
# solid / third-medium interface; nodes that touch only the medium are pinned to d = 0).
# True: solved on the whole cell, d diffuses into the medium (old behaviour, for comparison)
damage_in_medium = False
# mechanics (inner CG of the trust region): 'Green' = fixed Green operator of the undamaged
# solid; 'Green_Jacobi' = D^-1/2 G D^-1/2 with D = diag(B^T A B + R) of the current tangent,
# recomputed at the start of every trust-region solve (M stays fixed within a solve);
# 'dual' = G K(C_ref A^+ C_ref) G with the current tangent A (pseudo-inverse through |eigenvalues|, so it stays SPD for
# an indefinite tangent), C_ref = 2 mu_solid I_sym; equals G wherever A = C_ref; recomputed once per trust-region solve
mechanics_preconditioner = 'dual'
DUAL_EIGENVALUE_CUTOFF = 1e-6                # 'dual': relative cutoff of the tangent pseudo-inverse

# trust-region Newton-CG. The solver works on the energy divided by the force scale
# s = E_solid h^(dim-1) (h = pixel size): a nodal force for a unit strain error, so the
# tolerances below do not change with the resolution (a nodal force ~ stress x h^(dim-1)).
# Scaling the energy, gradient and Hessian by a constant leaves all steps unchanged.
SOLVER_GTOL = 3e-6                         # max |nodal force| / s  (~ residual stress / E)
SOLVER_MAXITER = 500
# inner (Steihaug) CG stops on the energy-norm error: Hestenes-Stiefel lower estimate with delay SOLVER_INNER_DELAY,
# |e_k|_K <= SOLVER_INNER_TOL |e_0|_K. The energy norm does not depend on the preconditioner, so CG counts of
# different preconditioners measure the same accuracy. None = inexact Newton (NuMPI forcing
# min(0.5, sqrt(max|g| / s)): loose far from the solution, tight close to it)
SOLVER_INNER_TOL = 1e-6
SOLVER_INNER_DELAY = 10
# cap of one inner CG solve, in multiples of the number of unknowns n. In exact arithmetic CG needs at most n steps,
# but on badly conditioned systems (soft third medium, cracks, adapted grids) round-off delays it far beyond n; with
# the cap at n (NuMPI's default) such solves were cut off before the energy stop. The energy stop is the stopping
# rule, the cap only guards against a hang.
SOLVER_INNER_MAXITER_FACTOR = 20
# 'tr_newton' (trust-region Newton-CG, handles negative curvature, e.g. buckling) or 'newton_cg' (plain Newton
# with full steps and the same inner PCG: fails once the tangent is no longer positive definite)
mechanics_solver = 'tr_newton'
INADMISSIBLE_ENERGY = 1e30                 # finite, so the trust region shrinks (inf - inf = nan)

# grid: True = conformal grid from muFFTTO.grid_adaptation.adapt_grid (cached in exp_data/grids),
# False = regular pixel grid (d_g = 0, the phase as in the regular-grid example)
adapt_grid_to_geometry = False
# HuHu-LuLu regularization of the third medium with second derivatives in physical coordinates x
# (True) or in the reference coordinates X of the regular grid (False, earlier version)
regularization_in_physical_coordinates = True
grid_parameters = {'method': 'zecevic', 'min_jacobian': 0.01}

# Output: every run gets its own, new folder (earlier runs are never overwritten)
#   <output_root>/<geometry>_<load>_<Q1|Q2>_<adapted|regular>_N<nnn>_Gc<Gc>_<G|GJ>_<EW|inner..>[_tags]_<date-time>/
# with contact_fracture.nc (one frame per accepted increment), run.log and, at
# the end, response.png and contact_fracture.mp4. output_root = None: no output.
output_root = '/home/martin/PycharmProjects/muFFTTO/examples/phase_field_fracture_adapted_grid/exp_data/slides/runs'
grid_folder = '/home/martin/PycharmProjects/muFFTTO/examples/phase_field_fracture_adapted_grid/exp_data/slides/../grids'
make_figures = False                        # response figure and MP4 at the end (needs netCDF4)
# Restart: contact_fracture.nc of an earlier run of this script with the same grid, geometry,
# element, material and load. The run continues from its last frame (u~, d, H, lam, dlam) in a
# new folder; the earlier frames are copied into the new file first, so that it holds the
# whole load path. The file is only read. None: start at lam = 0.
restart_file = None

# ----------------------------------------------------------------------------
# Discretizations: vector (u~) and scalar (d) on the same grid and element
# ----------------------------------------------------------------------------
disc_u = domain.Discretization(cell=domain.PeriodicUnitCell(domain_size=domain_size, problem_type='elasticity'),
                               nb_of_pixels_global=number_of_pixels, discretization_type='finite_element',
                               element_type=element_type, device=device)
disc_d = domain.Discretization(cell=domain.PeriodicUnitCell(domain_size=domain_size, problem_type='conductivity'),
                               nb_of_pixels_global=number_of_pixels, discretization_type='finite_element',
                               element_type=element_type, device=device)

# field data (.s) are numpy arrays on the CPU and cupy arrays on the GPU
if device == 'gpu':
    import cupy as xp
else:
    xp = np
to_host = tensor_operations.to_host
comm = MPI.COMM_WORLD
rank = comm.rank
dim = 2
pixel_size = (np.prod(domain_size) / np.prod(number_of_pixels)) ** (1.0 / dim)


log_file = None                            # run.log in the run folder, set below


def root_print(*args):
    if rank == 0:
        print(*args, flush=True)
        if log_file is not None:
            with open(log_file, 'a') as log:
                print(*args, file=log)


def global_sum(a):
    return comm.allreduce(float(a), op=MPI.SUM)


def global_min(a):
    return comm.allreduce(float(a), op=MPI.MIN)


def global_max(a):
    return comm.allreduce(float(a), op=MPI.MAX)


def global_mean(a):
    return global_sum(a.sum()) / max(global_sum(float(a.size)), 1.0)


def dot_global(a, b):
    return global_sum((a.ravel() * b.ravel()).sum())


# ----------------------------------------------------------------------------
# Geometry and material parameters at the quadrature points
# ----------------------------------------------------------------------------
coords = np.asarray(disc_u.fft.coords)                                         # local nodes X / L, [2, x, y]
local_index = tuple(np.rint(coords[k] * number_of_pixels[k]).astype(int) % number_of_pixels[k] for k in range(dim))
if adapt_grid_to_geometry:
    # the same grid on every rank: computed once (rank 0), cached, broadcast
    grid_file = os.path.join(grid_folder, f'grid_{geometry_name}_N{nnn}_' + '_'.join(
        f'{key}{value:g}' if not isinstance(value, str) else f'{value}'
        for key, value in sorted(grid_parameters.items())) + '.npz')
    grid = None
    if rank == 0:
        if os.path.exists(grid_file) and repr(geometry_parameters) == str(np.load(grid_file)['geometry_parameters']):
            grid = dict(np.load(grid_file))
        else:
            d_g_global, phase_global, grid_info = grid_adaptation.adapt_grid(
                nnn, geometry_name, geometry_parameters=geometry_parameters, **grid_parameters)
            grid = {'d_g': d_g_global, 'phase': phase_global, 'geometry_parameters': repr(geometry_parameters),
                    **{key: np.asarray(value) for key, value in grid_info.items()}}
            os.makedirs(grid_folder, exist_ok=True)
            np.savez(grid_file, **grid)
    grid = comm.bcast(grid, root=0)
    grid_displacement_local = grid['d_g'][(slice(None),) + local_index]   # [2, x, y]
    phase_pixel = grid['phase'][local_index]                               # phase at the deformed element centroid
else:
    grid = None
    grid_displacement_local = np.zeros((dim,) + coords.shape[1:])
    phase_pixel = microstructure_library.get_geometry(nb_voxels=disc_u.nb_of_pixels,
                                                      microstructure_name=geometry_name,
                                                      coordinates=disc_u.fft.coords,
                                                      **geometry_parameters)       # 1 solid, 0 medium
nb_q = disc_u.nb_quad_points_per_pixel
solid_q = xp.broadcast_to(xp.asarray(phase_pixel > 0), (nb_q, *phase_pixel.shape))  # [q, x, y]


def correlated_random_field(points, correlation_length, seed):
    """Smooth periodic random field on the unit cell with values in (0, 1), at the points [2, ...].

    A random Fourier series sum_k a_k (xi_k cos 2 pi k.x + eta_k sin 2 pi k.x), xi, eta ~ N(0, 1), with the Gaussian
    spectrum a_k^2 ~ exp(-(2 pi |k| l_c)^2 / 2), mapped by the normal CDF of its standard deviation to a uniform
    marginal. It depends on the points only, not on the grid.
    """
    from scipy.special import ndtr
    rng = np.random.default_rng(seed)
    k_max = int(np.ceil(3.0 / (2 * np.pi * correlation_length)))
    wave_vectors = [(i, j) for i in range(-k_max, k_max + 1) for j in range(0, k_max + 1) if j > 0 or i > 0]
    field = np.zeros(points.shape[1:])
    variance = 0.0
    for i, j in wave_vectors:
        amplitude = np.exp(-(2 * np.pi * np.hypot(i, j) * correlation_length) ** 2 / 4)
        xi, eta = rng.standard_normal(2)
        phase = 2 * np.pi * (i * points[0] + j * points[1])
        field += amplitude * (xi * np.cos(phase) + eta * np.sin(phase))
        variance += amplitude ** 2
    return ndtr(field / np.sqrt(variance))


# fracture toughness per pixel (constant, or the random field at the physical pixel centres)
if Gc_field == 'random':
    # physical pixel centre / L: the regular centre moved by the mean grid displacement of the four corner nodes
    # (d_g is periodic, so averaging rolled copies needs no wrapping; the field is periodic too)
    d_g = grid_displacement_local / np.asarray(domain_size).reshape(-1, 1, 1)
    corner_mean = 0.25 * (d_g + np.roll(d_g, -1, 1) + np.roll(d_g, -1, 2) + np.roll(np.roll(d_g, -1, 1), -1, 2))
    pixel_centre = coords + 0.5 / np.asarray(number_of_pixels).reshape(-1, 1, 1) + corner_mean
    Gc_pixel = Gc_range[0] + (Gc_range[1] - Gc_range[0]) * correlated_random_field(pixel_centre, Gc_correlation_length,
                                                                                   Gc_seed)
    Gc = 0.5 * (Gc_range[0] + Gc_range[1])
else:
    Gc_pixel = np.full(phase_pixel.shape, Gc)
Gc_q = xp.broadcast_to(xp.asarray(Gc_pixel), solid_q.shape)                 # [q, x, y]

mu_solid = E_solid / (2 * (1 + nu_solid))
lam_solid = E_solid * nu_solid / ((1 + nu_solid) * (1 - 2 * nu_solid))
kappa_solid = lam_solid + 2.0 * mu_solid / 3.0
mu_q = xp.where(solid_q, mu_solid, k_v * mu_solid)
kappa_q = xp.where(solid_q, kappa_solid, k_v * kappa_solid)
k_r = alpha * domain_size[0] ** 2 * (kappa_solid + 4.0 * mu_solid / 3.0)
# 1 where the regularization acts, 0 elsewhere, at the quadrature points [q, x, y]
regularization_weight_q = xp.where(solid_q, 0.0, 1.0) if regularization_region == 'medium' else xp.ones(solid_q.shape)
quad_weights = xp.asarray(disc_u.quadrature_weights).reshape((-1,) + (1,) * dim)

# grid map: F_g = I + grad_X d_g, J_g, F_g^-1 at the quadrature points (constant during the run)
grid_displacement = disc_u.get_unknown_size_field(name='grid_displacement')
grid_displacement.s[:, 0] = xp.asarray(grid_displacement_local)
disc_u.fft.communicate_ghosts(grid_displacement)
grid_gradient = disc_u.get_gradient_size_field(name='grid_gradient')
disc_u.apply_gradient_operator_mugrid(u_inxyz=grid_displacement, grad_u_ijqxyz=grid_gradient)
F_g = grid_gradient.s.copy() + xp.eye(dim).reshape(dim, dim, *(1,) * (grid_gradient.s.ndim - 2))
J_g = F_g[0, 0] * F_g[1, 1] - F_g[0, 1] * F_g[1, 0]                            # [q, x, y]
F_g_inv = xp.stack([xp.stack([F_g[1, 1], -F_g[0, 1]]), xp.stack([-F_g[1, 0], F_g[0, 0]])]) / J_g  # [i, j, q, x, y]
weights_J = quad_weights * J_g                                                  # w_q J_g
# damage conductivity of the map: J_g F_g^-1 F_g^-T
K_g = J_g * xp.einsum('kj...,lj...->kl...', F_g_inv, F_g_inv)
if not global_min(np.min(J_g)) > 0.0:
    raise ValueError('the grid map is not invertible (J_g <= 0)')

root_print(f'mechanics preconditioner {mechanics_preconditioner}, damage preconditioner {damage_preconditioner}')
root_print(f'grid {number_of_pixels}, {element_type}, ranks {comm.size}, '
           f'{"adapted" if adapt_grid_to_geometry else "regular"} grid (min J_g {global_min(np.min(J_g)):.3f}, '
           f'max J_g {global_max(np.max(J_g)):.3f}), solid fraction '
           f'{global_sum(np.sum(weights_J * solid_q)) / np.prod(domain_size):.4f}, '
           f'l = {length_scale:.4f}, Gc = {Gc}' + (f' (random {Gc_range[0]:g} ... {Gc_range[1]:g}, correlation length '
                                                   f'{Gc_correlation_length:g}, seed {Gc_seed})'
                                                   if Gc_field == 'random' else '') + f', k_r = {k_r:.2e}')


def split_neo_hookean(F, mu, kappa, g, tangent=False):
    """W, psi^+, P = dW/dF (and A = dP/dF) of the split neo-Hookean law, pointwise.

    F: [2, 2, ...]; mu, kappa, g: [...]. See the module docstring.
    """
    xp_ = tensor_operations.get_array_module(F)   # numpy or cupy, where F lives
    J = F[0, 0] * F[1, 1] - F[0, 1] * F[1, 0]
    FiT = xp_.stack([xp_.stack([F[1, 1], -F[1, 0]]), xp_.stack([-F[0, 1], F[0, 0]])]) / J       # F^-T
    lnJ = xp_.log(J)
    I1 = xp_.einsum('ij...,ij...->...', F, F)
    a = J ** (-2.0 / 3.0)
    b = (I1 + 1.0) / 3.0
    W_iso = 0.5 * mu * (a * (I1 + 1.0) - 3.0)
    W_vol = 0.5 * kappa * lnJ ** 2
    g_vol = xp_.where(lnJ > 0, g, 1.0)
    psi_plus = W_iso + xp_.where(lnJ > 0, W_vol, 0.0)
    W = g * W_iso + g_vol * W_vol
    P = g * mu * a * (F - b * FiT) + g_vol * kappa * lnJ * FiT
    if not tangent:
        return W, psi_plus, P
    identity = xp_.eye(2)
    dd = xp_.einsum('ik,jl->ijkl', identity, identity).reshape((2, 2, 2, 2) + (1,) * (F.ndim - 2))
    FiT_il_FiT_kj = xp_.einsum('il...,kj...->ijkl...', FiT, FiT)
    A_iso = (-2.0 / 3.0 * xp_.einsum('kl...,ij...->ijkl...', FiT, F - b * FiT) + dd
             - 2.0 / 3.0 * xp_.einsum('kl...,ij...->ijkl...', F, FiT) + b * FiT_il_FiT_kj)
    A_vol = xp_.einsum('ij...,kl...->ijkl...', FiT, FiT) - lnJ * FiT_il_FiT_kj
    return W, psi_plus, P, g * mu * a * A_iso + g_vol * kappa * A_vol


def saint_venant_kirchhoff(F, mu, lam, tangent=False):
    """W, P (and A) of W = lam/2 (tr E)^2 + mu E:E, E = (F^T F - I)/2 (plane strain), pointwise."""
    xp_ = tensor_operations.get_array_module(F)
    identity = xp_.eye(2).reshape((2, 2) + (1,) * (F.ndim - 2))
    E = 0.5 * (xp_.einsum('ki...,kj...->ij...', F, F) - identity)
    trE = E[0, 0] + E[1, 1]
    S = lam * trE * identity + 2.0 * mu * E                                      # 2nd Piola-Kirchhoff
    W = 0.5 * lam * trE ** 2 + mu * xp_.einsum('ij...,ij...->...', E, E)
    P = xp_.einsum('ik...,kj...->ij...', F, S)
    if not tangent:
        return W, xp_.zeros_like(W), P
    FFt = xp_.einsum('im...,km...->ik...', F, F)
    A = (xp_.einsum('ik,jl...->ijkl...', xp_.eye(2), S) + lam * xp_.einsum('ij...,kl...->ijkl...', F, F)
         + mu * (xp_.einsum('ik...,jl->ijkl...', FFt, xp_.eye(2)) + xp_.einsum('il...,kj...->ijkl...', F, F)))
    return W, xp_.zeros_like(W), P, A


def linear_elastic(F, mu, lam, tangent=False):
    """W, P (and A) of W = lam/2 (tr eps)^2 + mu eps:eps, eps = sym(F - I) (plane strain), pointwise."""
    xp_ = tensor_operations.get_array_module(F)
    identity = xp_.eye(2).reshape((2, 2) + (1,) * (F.ndim - 2))
    eps = 0.5 * (F + xp_.swapaxes(F, 0, 1)) - identity
    tr_eps = eps[0, 0] + eps[1, 1]
    W = 0.5 * lam * tr_eps ** 2 + mu * xp_.einsum('ij...,ij...->...', eps, eps)
    P = lam * tr_eps * identity + 2.0 * mu * eps
    if not tangent:
        return W, xp_.zeros_like(W), P
    I2 = xp_.eye(2)
    A = (lam * xp_.einsum('ij,kl->ijkl', I2, I2)
         + mu * (xp_.einsum('ik,jl->ijkl', I2, I2) + xp_.einsum('il,jk->ijkl', I2, I2)))
    A = xp_.broadcast_to(A.reshape((2, 2, 2, 2) + (1,) * (F.ndim - 2)), (2, 2, 2, 2) + F.shape[2:])
    return W, xp_.zeros_like(W), P, A


def material(F, g, tangent=False):
    """W, psi^+, P (and A) at the quadrature points: split neo-Hookean in the solid; in the voids
    the third medium (neo-Hookean) or a super-soft St. Venant-Kirchhoff / linear solid (void_model)."""
    xp_ = tensor_operations.get_array_module(F)
    if void_model == 'neo_hookean':
        return split_neo_hookean(F, mu_q, kappa_q, g, tangent)
    identity = xp_.eye(2).reshape((2, 2) + (1,) * (F.ndim - 2))
    solid = split_neo_hookean(xp_.where(solid_q, F, identity), mu_q, kappa_q, g, tangent)   # no log of J <= 0
    void_law = saint_venant_kirchhoff if void_model == 'saint_venant_kirchhoff' else linear_elastic
    void = void_law(F, k_v * mu_solid, k_v * lam_solid, tangent)
    return tuple(xp_.where(solid_q, a, b) for a, b in zip(solid, void))


# ----------------------------------------------------------------------------
# Fields
# ----------------------------------------------------------------------------
macro_gradient = disc_u.get_gradient_size_field(name='macro_gradient')          # H_macro (without I)
disc_u.get_macro_gradient_field_mugrid(macro_gradient_ij=H_macro, macro_gradient_field_ijqxyz=macro_gradient)
grad_u = disc_u.get_displacement_gradient_sized_field(name='grad_u')
F_field = disc_u.get_displacement_gradient_sized_field(name='F')
stress_field = disc_u.get_displacement_gradient_sized_field(name='P')
tangent_field = disc_u.get_material_data_size_field_mugrid(name='A')
u_fluc = disc_u.get_unknown_size_field(name='u_fluc')
x_field = disc_u.get_unknown_size_field(name='x_field')
grad_field = disc_u.get_unknown_size_field(name='grad_field')
Ru_field = disc_u.get_unknown_size_field(name='Ru_field')
v_field = disc_u.get_unknown_size_field(name='v_field')
Bv_field = disc_u.get_unknown_size_field(name='Bv_field')
r_field = disc_u.get_unknown_size_field(name='r_field')
Pv_field = disc_u.get_unknown_size_field(name='Pv_field')
hess_u = disc_u.get_displacement_hessian_size_field(name='hess_u')
HtH_field = disc_u.get_unknown_size_field(name='HtH')
lap_u = disc_u.get_displacement_laplacian_at_quad_field(name='lap_u')
LtL_field = disc_u.get_displacement_sized_field(name='LtL')
u_shape = u_fluc.s.shape

damage = disc_d.get_unknown_size_field(name='damage')                 # d, nodal
damage_old = disc_d.get_unknown_size_field(name='damage_old')
rhs_d = disc_d.get_unknown_size_field(name='rhs_d')
damage_q = disc_d.get_quad_field_scalar(name='damage_q')
history = disc_d.get_quad_field_scalar(name='history')                # H, current iterate
history_n = disc_d.get_quad_field_scalar(name='history_converged')    # H of the last increment
reaction = disc_d.get_quad_field_scalar(name='reaction')              # Gc/l + 2H
grad_d = disc_d.get_gradient_size_field(name='grad_damage')
temp_q = disc_d.get_quad_field_scalar(name='temp_q')
temp_n = disc_d.get_unknown_size_field(name='temp_n')
x_scaled_d = disc_d.get_unknown_size_field(name='x_scaled_d')
# damage domain: weight 1 at solid quadrature points, 0 in the medium; free nodes = nodes of
# at least one solid element (the others get a unit row and stay d = 0)
damage_weight_q = xp.ones(solid_q.shape) if damage_in_medium else solid_q.astype(float)
temp_q.s[0, 0] = damage_weight_q
disc_d.apply_N_transposed_operator_mugrid(quad_field_ijqxyz=temp_q, nodal_field_inxyz=temp_n, apply_weights=True)
damage_free_nodes = temp_n.s > 0                                      # [1, nodes, x, y]

# pixel-level output fields: phase and quadrature-averaged det F
phase_out = disc_u.field_collection.real_field('phase', [1], 'pixel')
phase_out.s[0, 0] = xp.asarray(phase_pixel)
detF_out = disc_u.field_collection.real_field('detF', [1], 'pixel')
Gc_out = disc_u.field_collection.real_field('Gc', [1], 'pixel')
Gc_out.s[0, 0] = xp.asarray(Gc_pixel)

g_q = xp.ones(solid_q.shape)               # g_s(d) at the quadrature points
# tangent A(F(x)) of the last state (x, lam) it was built for: x is fixed during a whole
# Steihaug-CG solve, so it is rebuilt once per trust-region iterate, not per Hessian product
tangent_cache = {'valid': False, 'x': None, 'lam': None}
lam_current = 0.0


def update_degradation():
    disc_d.apply_N_operator_mugrid(damage, damage_q)
    g_q[...] = xp.where(solid_q, (1.0 - damage_q.s[0, 0]) ** 2 + k_residual, 1.0)
    tangent_cache['valid'] = False         # the tangent depends on g(d)


# ----------------------------------------------------------------------------
# Elasticity: F, energy, gradient, Hessian-vector product, preconditioner
# ----------------------------------------------------------------------------
def set_F(lam, u_field):
    """F = I + lam H_macro + grad_X u~ . F_g^-1; returns the global min det F."""
    disc_u.fft.communicate_ghosts(u_field)
    disc_u.apply_gradient_operator_mugrid(u_inxyz=u_field, grad_u_ijqxyz=grad_u)
    F_field.s[...] = xp.einsum('ij...,jk...->ik...', grad_u.s, F_g_inv) + lam * macro_gradient.s
    for k in range(dim):
        F_field.s[k, k] += 1.0
    F = F_field.s
    J = F[0, 0] * F[1, 1] - F[0, 1] * F[1, 0]
    if void_model != 'neo_hookean':                    # the soft void may invert, only the solid must not
        J = xp.where(solid_q, J, xp.inf)
    return global_min(xp.min(J))


def add_regularization(x, out, weight_q=1.0):
    """out += R x = k_r (H^T w H - L^T w L / dim) x with the weight w at the quadrature points (1: everywhere);
    ghosts of x must be current."""
    disc_u.apply_hessian_operator_to_vector_field_mugrid(u_inxyz=x, hess_u_ijkqxyz=hess_u)
    hess_u.s[...] *= weight_q
    disc_u.apply_hessian_operator_transposed_to_vector_field_mugrid(hess_u_ijkqxyz=hess_u,
                                                                    nodal_field_inxyz=HtH_field,
                                                                    apply_weights=True)
    out.s[...] += k_r * HtH_field.s
    if laplacian_acts:
        disc_u.laplacian.apply(nodal_field=x, quadrature_point_field=lap_u)
        lap_u.s[...] *= weight_q
        disc_u.laplacian.transpose(quadrature_point_field=lap_u, nodal_field=LtL_field,
                                   weights=disc_u.quadrature_weights)
        out.s[...] -= (k_r / dim) * LtL_field.s


# the discrete Laplacian of a bilinear (Q1) field on axis-aligned pixels is identically zero,
# so the L^T L part is skipped when it has no effect (it acts for Q2)
LtL_field.s[...] = xp.asarray(np.random.default_rng(rank).random(LtL_field.s.shape))
disc_u.fft.communicate_ghosts(LtL_field)
disc_u.laplacian.apply(nodal_field=LtL_field, quadrature_point_field=lap_u)
laplacian_acts = global_max(np.max(np.abs(lap_u.s))) > 0.0

# Regularization in physical coordinates. The HuHu-LuLu term penalizes the second gradient of the
# fluctuation with respect to x (the physical cell), not X: a fluctuation that is locally affine in x
# must cost nothing, whatever the grid map. For the isoparametric (bilinear) map x(X) = X + d_g(X),
# by the chain rule
#     H_x u~ = F_g^-T [ H_X u~ - (grad_x u~)_c H_X x_c ] F_g^-1,   H_X x = H_X d_g,  grad_x = grad_X . F_g^-1
# (exact for the discrete fields: H_x of the interpolant of an x-affine field is zero), L_x = tr H_x, and
#     Pi_reg = k_r/2 sum_q w_q J_g ( H_x u~ : H_x u~ - (1/dim) L_x u~ . L_x u~ ).
# With d_g = 0 this is the term of the regular grid. regularization_in_physical_coordinates = False
# evaluates it with respect to X (the earlier version).
hess_grid_field = disc_u.get_displacement_hessian_size_field(name='hess_grid')
disc_u.apply_hessian_operator_to_vector_field_mugrid(u_inxyz=grid_displacement, hess_u_ijkqxyz=hess_grid_field)
hess_grid = hess_grid_field.s.copy()                                            # H_X d_g [c, a, b, q, x, y]
grad_reg = disc_u.get_gradient_size_field(name='grad_reg')
reg_temp = disc_u.get_unknown_size_field(name='reg_temp')


def physical_hessian(x):
    """H_x u~ [i, k, l, q, x, y] of the nodal field x (ghosts current)."""
    disc_u.apply_hessian_operator_to_vector_field_mugrid(u_inxyz=x, hess_u_ijkqxyz=hess_u)
    disc_u.apply_gradient_operator_mugrid(u_inxyz=x, grad_u_ijqxyz=grad_reg)
    grad_x = xp.einsum('ia...,ak...->ik...', grad_reg.s, F_g_inv)
    T = hess_u.s - xp.einsum('ic...,cab...->iab...', grad_x, hess_grid)
    return xp.einsum('ak...,iab...,bl...->ikl...', F_g_inv, T, F_g_inv, optimize=True)


def add_regularization_physical(x, out):
    """out += R x with R the Hessian of Pi_reg in physical coordinates; ghosts of x must be current."""
    H_x = physical_hessian(x)
    L_x = xp.einsum('ikk...->i...', H_x)                    # LuLu part: L_x = tr H_x (zero for Q1 on pixels)
    Y = H_x - (1.0 / dim) * xp.einsum('i...,kl->ikl...', L_x, xp.eye(dim))
    Y = k_r * J_g * regularization_weight_q * Y            # quadrature weights w_q: in the transposed operators
    # adjoint of H_x = F_g^-T T F_g^-1, T = H_X u~ - grad_x u~ . H_X d_g, grad_x = grad_X . F_g^-1
    T_hat = xp.einsum('ak...,ikl...,bl...->iab...', F_g_inv, Y, F_g_inv, optimize=True)
    hess_u.s[...] = T_hat
    disc_u.apply_hessian_operator_transposed_to_vector_field_mugrid(hess_u_ijkqxyz=hess_u, nodal_field_inxyz=reg_temp,
                                                                    apply_weights=True)
    out.s[...] += reg_temp.s
    grad_x_hat = -xp.einsum('iab...,cab...->ic...', T_hat, hess_grid)
    grad_reg.s[...] = xp.einsum('ic...,ac...->ia...', grad_x_hat, F_g_inv)
    disc_u.fft.communicate_ghosts(grad_reg)
    disc_u.apply_gradient_transposed_operator_mugrid(gradient_field_ijqxyz=grad_reg, div_u_fnxyz=reg_temp,
                                                     apply_weights=True)
    out.s[...] += reg_temp.s


def add_regularization_of_the_problem(x, out):
    if regularization_in_physical_coordinates:
        add_regularization_physical(x, out)
    else:
        add_regularization(x, out, regularization_weight_q)


def field_to_optimizer(field):
    """Copy of the field data as a numpy array (host), for NuMPI's optimizer."""
    if device == 'gpu':
        return field.s.get()  # .get() already makes a host copy
    return np.array(field.s, copy=True)


def fun_grad(x):
    """(Pi_elastic + Pi_reg, gradient) at the fluctuation x, damage fixed."""
    x_field.s[...] = xp.asarray(x.reshape(u_shape))
    if not set_F(lam_current, x_field) > 0.0:
        return INADMISSIBLE_ENERGY, np.zeros(u_shape)
    W, _, P = material(F_field.s, g_q)
    Pi = global_sum(np.sum(weights_J * W))
    if not np.isfinite(Pi):
        return INADMISSIBLE_ENERGY, np.zeros(u_shape)
    Ru_field.s[...] = 0.0
    add_regularization_of_the_problem(x_field, Ru_field)
    Pi += 0.5 * dot_global(x_field.s, Ru_field.s)
    stress_field.s[...] = J_g * xp.einsum('ij...,kj...->ik...', P, F_g_inv)     # J_g P . F_g^-T
    disc_u.fft.communicate_ghosts(stress_field)
    disc_u.apply_gradient_transposed_operator_mugrid(gradient_field_ijqxyz=stress_field, div_u_fnxyz=grad_field,
                                                     apply_weights=True)
    grad_field.s[...] += Ru_field.s
    return Pi, field_to_optimizer(grad_field)


def update_tangent(lam, x):
    """Tangent A(F) at (lam, x) into tangent_field, unless it is already there (cached)."""
    # x comes from the optimizer (numpy) or from a field (numpy or cupy): compare on the host
    x = to_host(x)
    stale = (not tangent_cache['valid'] or tangent_cache['lam'] != lam
             or not np.array_equal(tangent_cache['x'], np.reshape(x, -1)))
    if global_max(float(stale)) == 0.0:    # the same decision on all ranks (set_F is collective)
        return
    x_field.s[...] = xp.asarray(x.reshape(u_shape))
    set_F(lam, x_field)
    A = material(F_field.s, g_q, tangent=True)[3]
    # pulled back: A'_ijkl = J_g A_imkn F_g^-1_jm F_g^-1_ln
    A = xp.einsum('imkn...,jm...->ijkn...', A, F_g_inv)
    tangent_field.s[...] = J_g * xp.einsum('ijkn...,ln...->ijkl...', A, F_g_inv)
    tangent_cache.update(valid=True, lam=lam, x=np.array(x, copy=True).reshape(-1))


def hessp(x, v):
    """(B^T A B + R) v with the tangent A at x (rebuilt only when x, lam or d changed)."""
    update_tangent(lam_current, x)
    v_field.s[...] = xp.asarray(v.reshape(u_shape))
    disc_u.fft.communicate_ghosts(v_field)
    disc_u.apply_system_matrix_mugrid(material_data_field=tangent_field, input_field_inxyz=v_field,
                                      output_field_inxyz=Bv_field, formulation='finite_strain')
    add_regularization_of_the_problem(v_field, Bv_field)
    disc_u.fft.communicate_ghosts(Bv_field)
    return field_to_optimizer(Bv_field)


# Green preconditioner of the undamaged solid at F = I, plus the regularization (built once)
reference_tangent = split_neo_hookean(np.eye(2)[:, :, None], np.array([mu_solid]), np.array([kappa_solid]),
                                      np.array([1.0]), tangent=True)[3][..., 0]
# Green-Jacobi: the Jacobi diagonal carries the local magnitude of the tangent, so the Green part only needs its
# structure; the identity (delta_ik delta_jl) needs fewer CG iterations than the solid tangent (Ladecky et al.,
# CMAME 461 (2026) 119269, Sec. 4.5; bubble array, Green-Jacobi: -8 % at 32^2, -19 % at 48^2). The factor 2 mu_solid
# keeps the magnitude of the solid, i.e. the trust-region norm.
# Dual: C_ref = 2 mu_solid I_sym (small-strain Navier operator), so that the trial flux q = C_ref B G f of the dual
# bound is symmetric, i.e. in the range of a hyperelastic tangent (which has no stiffness for skew gradients at F = I)
green_reference = {'Green_Jacobi': 'identity', 'dual': 'symmetric'}.get(mechanics_preconditioner, 'solid')
if green_reference == 'identity':
    reference_tangent = 2 * mu_solid * np.einsum('ik,jl->ijkl', np.eye(2), np.eye(2)).reshape(reference_tangent.shape)
elif green_reference == 'symmetric':
    reference_tangent = mu_solid * (np.einsum('ik,jl->ijkl', np.eye(2), np.eye(2))
                                    + np.einsum('il,jk->ijkl', np.eye(2), np.eye(2))).reshape(reference_tangent.shape)


def operator_for_preconditioner(input_field_inxyz, output_field_inxyz):
    disc_u.fft.communicate_ghosts(field=input_field_inxyz)
    disc_u.apply_system_matrix_mugrid(material_data_field=reference_tangent, input_field_inxyz=input_field_inxyz,
                                      output_field_inxyz=output_field_inxyz, formulation='finite_strain')
    add_regularization(input_field_inxyz, output_field_inxyz)


green_u = disc_u.get_preconditioner_Green_mugrid(reference_material_data_ijkl=reference_tangent,
                                                 operator=operator_for_preconditioner)


jacobi_u = None                            # (D / d_ref)^-1/2 of the mechanics Hessian (Green_Jacobi only)
# The trust region is measured in the norm of the preconditioner, |s|_M with M = P^-1. For Green,
# M = K_ref: the energy norm of the undamaged solid. For Green_Jacobi, M = D^1/2 K_ref D^1/2 would
# be ~ d_ref times larger; with D normalized by the (uniform) diagonal d_ref of K_ref, M = K_ref
# wherever the tangent is the reference one, so both preconditioners use the same trust-region
# measure. A constant scaling does not change the CG iterates.
reference_diagonal = global_max(np.max(
    1.0 / disc_u.get_preconditioner_Jacobi_mugrid(operator=operator_for_preconditioner).s ** 2))


def update_mechanics_jacobi(lam, u_field):
    """D^-1/2 of B^T A(F) B + R at the state (lam, u_field), for the next trust-region solve."""
    global jacobi_u
    update_tangent(lam, u_field.s)

    def hessian_operator(input_field_inxyz, output_field_inxyz):
        disc_u.fft.communicate_ghosts(input_field_inxyz)
        disc_u.apply_system_matrix_mugrid(material_data_field=tangent_field, input_field_inxyz=input_field_inxyz,
                                          output_field_inxyz=output_field_inxyz, formulation='finite_strain')
        add_regularization_of_the_problem(input_field_inxyz, output_field_inxyz)

    jacobi_u = float(np.sqrt(reference_diagonal)) * disc_u.get_preconditioner_Jacobi_mugrid(operator=hessian_operator).s


# Dual preconditioner P = G K(C_ref A^-1 C_ref) G. With G = K_ref^-1, K_ref = B^T W C_ref B, the flux q = C_ref B G f
# is admissible (B^T W q = f), so by the complementary energy f.K^-1 f <= q.W A^-1 q = f.P f for an SPD tangent A.
# The pulled-back tangent carries the grid map, so on an adapted grid P sees J_g and F_g (G does not). The
# regularization R is only in G.
tangent_inv_field = disc_u.get_material_data_size_field_mugrid(name='A_dual')
dual_temp_1 = disc_u.get_unknown_size_field(name='dual_temp_1')
dual_temp_2 = disc_u.get_unknown_size_field(name='dual_temp_2')


def update_mechanics_dual(lam, u_field):
    """C_ref |A|^-1 C_ref of the tangent at the state (lam, u_field), for the next trust-region solve."""
    update_tangent(lam, u_field.s)
    A = tangent_field.s
    shape = A.shape
    A = xp.moveaxis(A.reshape((dim * dim, dim * dim) + shape[4:]), (0, 1), (-2, -1))      # [..., (ij), (kl)]
    A = 0.5 * (A + xp.swapaxes(A, -1, -2))
    eigenvalues, eigenvectors = xp.linalg.eigh(A)
    eigenvalues = xp.abs(eigenvalues)
    # pseudo-inverse: modes softer than DUAL_EIGENVALUE_CUTOFF x the stiffest one of the same quadrature point are
    # dropped. At F = I the skew (rotation) mode of a hyperelastic tangent has zero stiffness; a floor would
    # put a huge weight on it. Only a rigid motion has a skew gradient everywhere, so P stays SPD.
    keep = eigenvalues > DUAL_EIGENVALUE_CUTOFF * xp.max(eigenvalues, axis=-1, keepdims=True)
    inverse_eigenvalues = xp.where(keep, 1.0 / xp.where(keep, eigenvalues, 1.0), 0.0)
    inverse = xp.einsum('...ab,...b,...cb->...ac', eigenvectors, inverse_eigenvalues, eigenvectors)
    # C_ref A^+ C_ref with C_ref = 2 mu_solid I_sym
    sym = xp.asarray(0.5 * (np.einsum('ik,jl->ijkl', np.eye(dim), np.eye(dim))
                            + np.einsum('il,jk->ijkl', np.eye(dim), np.eye(dim))).reshape(dim * dim, dim * dim))
    inverse = (2 * mu_solid) ** 2 * xp.einsum('ab,...bc,cd->...ad', sym, inverse, sym)
    tangent_inv_field.s[...] = xp.moveaxis(inverse, (-2, -1), (0, 1)).reshape(shape)


def apply_dual(r, out):
    """out = G K(C_ref A^-1 C_ref) G r (fields)."""
    disc_u.fft.communicate_ghosts(r)
    disc_u.apply_preconditioner_mugrid(preconditioner_Fourier_fnfnqks=green_u, input_nodal_field_fnxyz=r,
                                       output_nodal_field_fnxyz=dual_temp_1)
    disc_u.fft.communicate_ghosts(dual_temp_1)
    disc_u.apply_system_matrix_mugrid(material_data_field=tangent_inv_field, input_field_inxyz=dual_temp_1,
                                      output_field_inxyz=dual_temp_2, formulation='finite_strain')
    disc_u.fft.communicate_ghosts(dual_temp_2)
    disc_u.apply_preconditioner_mugrid(preconditioner_Fourier_fnfnqks=green_u, input_nodal_field_fnxyz=dual_temp_2,
                                       output_nodal_field_fnxyz=out)


def precond(v):
    r_field.s[...] = xp.asarray(v.reshape(u_shape))
    if mechanics_preconditioner == 'dual':
        apply_dual(r_field, Pv_field)
        return field_to_optimizer(Pv_field)
    if mechanics_preconditioner == 'Green_Jacobi':
        r_field.s[...] *= jacobi_u
    disc_u.fft.communicate_ghosts(r_field)
    disc_u.apply_preconditioner_mugrid(preconditioner_Fourier_fnfnqks=green_u, input_nodal_field_fnxyz=r_field,
                                       output_nodal_field_fnxyz=Pv_field)
    if mechanics_preconditioner == 'Green_Jacobi':
        Pv_field.s[...] *= jacobi_u
    return field_to_optimizer(Pv_field)


# ----------------------------------------------------------------------------
# Damage: [Gc l B^T W B + N^T W (Gc/l + 2H) N] d = N^T W 2H
# ----------------------------------------------------------------------------
def apply_damage_operator(input_field_inxyz, output_field_inxyz, reaction_q, weight_q=1.0, pulled_back=True,
                          toughness_q=None):
    """Damage operator on the deformed grid with the toughness Gc(x) (pulled_back=False, constant toughness: on the
    regular grid, for the Green preconditioner, which needs a translation-invariant operator)."""
    disc_d.apply_gradient_operator_mugrid(u_inxyz=input_field_inxyz, grad_u_ijqxyz=grad_d)
    if pulled_back:
        grad_d.s[...] = xp.einsum('ak...,kl...->al...', grad_d.s, K_g)
    grad_d.s[...] *= (Gc if toughness_q is None else toughness_q) * length_scale * weight_q
    disc_d.apply_gradient_transposed_operator_mugrid(gradient_field_ijqxyz=grad_d, div_u_fnxyz=output_field_inxyz,
                                                     apply_weights=True)
    disc_d.apply_N_operator_mugrid(input_field_inxyz, temp_q)
    temp_q.s[...] *= reaction_q * weight_q * (J_g if pulled_back else 1.0)
    disc_d.apply_N_transposed_operator_mugrid(quad_field_ijqxyz=temp_q, nodal_field_inxyz=temp_n,
                                              apply_weights=True)
    output_field_inxyz.s[...] += temp_n.s


def K_d(x, Ax):
    """Damage operator on the damage domain; unit rows (identity) for the pinned nodes."""
    apply_damage_operator(x, Ax, reaction.s, damage_weight_q, toughness_q=Gc_q)
    Ax.s[...] = xp.where(damage_free_nodes, Ax.s, x.s)


green_d = disc_d.get_preconditioner_Green_mugrid(
    reference_material_data_ijkl=None,
    operator=lambda input_field_inxyz, output_field_inxyz: apply_damage_operator(
        input_field_inxyz, output_field_inxyz, Gc / length_scale, pulled_back=False),
    invert_zero_mode=True)
jacobi_d = None


def M_d(x, Px):
    if damage_preconditioner == 'Green_Jacobi':
        x_scaled_d.s[...] = jacobi_d.s * x.s
        disc_d.apply_preconditioner_mugrid(preconditioner_Fourier_fnfnqks=green_d, input_nodal_field_fnxyz=x_scaled_d,
                                           output_nodal_field_fnxyz=Px)
        Px.s[...] *= jacobi_d.s
    else:
        disc_d.apply_preconditioner_mugrid(preconditioner_Fourier_fnfnqks=green_d, input_nodal_field_fnxyz=x,
                                           output_nodal_field_fnxyz=Px)
    Px.s[...] = xp.where(damage_free_nodes, Px.s, x.s)       # identity on the pinned nodes


class IterationCounter:
    def __init__(self):
        self.count = 0

    def __call__(self, iteration, fields):
        self.count = iteration


def solve_damage():
    """History update and one damage solve; returns (max |d - d_old|, CG iterations)."""
    global jacobi_d
    psi_plus = material(F_field.s, g_q)[1]
    history.s[0, 0] = xp.maximum(history_n.s[0, 0], xp.where(solid_q, psi_plus, 0.0))
    reaction.s[...] = Gc_q / length_scale + 2.0 * history.s
    temp_q.s[...] = 2.0 * history.s * J_g
    disc_d.apply_N_transposed_operator_mugrid(quad_field_ijqxyz=temp_q, nodal_field_inxyz=rhs_d, apply_weights=True)
    rhs_d.s[...] *= damage_free_nodes                          # pinned nodes: d = 0
    damage_old.s[...] = damage.s
    if damage_preconditioner == 'Green_Jacobi':
        jacobi_d = disc_d.get_preconditioner_Jacobi_mugrid(
            operator=lambda input_field_inxyz, output_field_inxyz: K_d(input_field_inxyz, output_field_inxyz))
    counter = IterationCounter()
    Solvers.conjugate_gradients(comm=disc_d.communicator, fc=disc_d.field_collection, hessp=K_d, b=rhs_d,
                                x=damage, prec=M_d, rtol=1e-8, maxiter=5000, callback=counter)
    update_degradation()
    return global_max(np.max(np.abs(damage.s - damage_old.s))), counter.count


# ----------------------------------------------------------------------------
# One load increment (staggered); False if it has to be retried with a smaller step
# ----------------------------------------------------------------------------
force_scale = E_solid * pixel_size ** (dim - 1)    # s: the solver sees Pi / s, g / s, H / s


# the inner CG cap: SOLVER_INNER_MAXITER_FACTOR x the global number of unknowns
inner_maxiter = SOLVER_INNER_MAXITER_FACTOR * int(round(global_sum(float(np.prod(u_shape)))))


def scaled_fun_grad(x):
    value, gradient = fun_grad(x)
    if value >= 0.5 * INADMISSIBLE_ENERGY:
        return value, gradient
    return value / force_scale, gradient / force_scale


def scaled_hessp(x, v):
    return hessp(x, v) / force_scale


def solve_increment(lam):
    global lam_current
    lam_current = lam
    if not set_F(lam, u_fluc) > 0.0:
        return False, 'warm start not admissible (det F <= 0)', {}
    stats = {'nb_hessp': 0, 'nb_outer': 0, 'cg_damage': 0, 'staggered_iterations': 0}
    for stagger in range(1, max_staggered_iterations + 1):
        if mechanics_preconditioner == 'Green_Jacobi':
            update_mechanics_jacobi(lam, u_fluc)
        elif mechanics_preconditioner == 'dual':
            update_mechanics_dual(lam, u_fluc)
        res = solvers_nonlinear.minimize(fun=scaled_fun_grad, x0=field_to_optimizer(u_fluc), method=mechanics_solver,
                                         hessp=scaled_hessp, precond=precond,
                                         gtol=SOLVER_GTOL, maxiter=SOLVER_MAXITER, inner_tol=SOLVER_INNER_TOL,
                                         inner_stop='energy', inner_delay=SOLVER_INNER_DELAY,
                                         inner_maxiter=inner_maxiter, comm=comm)
        stats['nb_hessp'] += int(res.nb_hessp)
        stats['nb_outer'] += int(res.nit)
        if not res.success or res.fun >= 0.5 * INADMISSIBLE_ENERGY:
            return False, f'trust region failed in staggered iteration {stagger}: {res.message}', stats
        u_fluc.s[...] = xp.asarray(np.asarray(res.x).reshape(u_shape))
        stats['min_J'] = set_F(lam, u_fluc)
        stats['energy'] = float(res.fun) * force_scale
        change, nb_cg = solve_damage() if solve_damage_problem else (0.0, 0)
        stats['cg_damage'] += nb_cg
        stats['staggered_iterations'] = stagger
        stats['damage_change'] = change
        if change < staggered_tol:
            break
    if change >= staggered_tol:
        return False, (f'staggered loop not converged after {max_staggered_iterations} iterations '
                       f'(max |d - d_old| = {change:.2e})'), stats
    return True, 'converged', stats


# ----------------------------------------------------------------------------
# Output
# ----------------------------------------------------------------------------
run_attributes = {'geometry': geometry_name, 'geometry_parameters': repr(geometry_parameters),
                  'element_type': element_type, 'nb_grid_pts': number_of_pixels,
                  'domain_size': np.asarray(domain_size), 'H_macro': H_macro, 'E_solid': E_solid,
                  'nu_solid': nu_solid, 'k_v': k_v, 'alpha': alpha, 'k_r': k_r, 'Gc': Gc,
                  'regularization_region': regularization_region, 'inner_maxiter': inner_maxiter,
                  'Gc_field': Gc_field, 'Gc_range': np.asarray(Gc_range), 'Gc_correlation_length': Gc_correlation_length,
                  'Gc_seed': Gc_seed,
                  'length_scale': length_scale, 'k_residual': k_residual, 'staggered_tol': staggered_tol,
                  'mechanics_preconditioner': mechanics_preconditioner, 'green_reference': green_reference,
                  'mechanics_solver': mechanics_solver,
                  'inner_tol': 'forcing' if SOLVER_INNER_TOL is None else SOLVER_INNER_TOL,
                  'gtol': SOLVER_GTOL, 'force_scale': force_scale,
                  'damage_in_medium': int(damage_in_medium),
                  'adapted_grid': int(adapt_grid_to_geometry), 'grid_parameters': repr(grid_parameters),
                  'regularization_physical': int(regularization_in_physical_coordinates),
                  'void_model': void_model,
                  'damage_preconditioner': damage_preconditioner,
                  'nb_increments': nb_increments,
                  'restart_from': '' if restart_file is None else os.path.abspath(restart_file),
                  'solve_damage_problem': int(solve_damage_problem), 'lam_end': lam_end,
                  'note': 'u_fluc and damage are nodal (Q2: sub-point 0 = pixel corner); '
                          'detF and phase are pixel values'}
frame_variables = {'increment': (), 'lam': (), 'dlam': (), 'F10': (), 'P10': (), 'Pxx': (),
                   'mean_P': (dim, dim), 'min_det_F': (), 'energy': (), 'damage_max': (),
                   'staggered_iterations': (), 'nb_outer': (), 'nb_hessp': (), 'cg_damage': (),
                   'converged': (), 'elapsed_time': ()}
output_fields = [damage, u_fluc, history, detF_out, phase_out, grid_displacement, Gc_out]
# a restart must continue the same problem: these attributes have to match
restart_keys = ['geometry', 'geometry_parameters', 'element_type', 'nb_grid_pts', 'domain_size', 'H_macro',
                'E_solid', 'nu_solid', 'k_v', 'alpha', 'Gc', 'length_scale', 'k_residual', 'damage_in_medium',
                'adapted_grid', 'grid_parameters']
writer = None
output_file = None
if output_root is not None:
    load_tag = '_'.join(f'F{i}{j}{H_macro[i, j]:+g}' for i, j in zip(*np.nonzero(H_macro)))
    # short names (muGrid's NetCDF writer limits the file path): Q1/Q2, preconditioner G/GJ,
    # inner tolerance EW (forcing) or its value
    element_tag = {'bilinear_rectangle': 'Q1', 'biquadratic_rectangle': 'Q2'}.get(element_type, element_type)
    run_name = (f'{geometry_name}_{load_tag}_{element_tag}'
                f'_{"adapted" if adapt_grid_to_geometry else "regular"}'
                f'_N{nnn}_Gc{Gc:g}' + (f'r{Gc_range[0]:g}-{Gc_range[1]:g}' if Gc_field == 'random' else '')
                + f'_{ {"Green_Jacobi": "GJ", "Green": "G", "dual": "D"}[mechanics_preconditioner]}'
                f'_{"EW" if SOLVER_INNER_TOL is None else f"inner{SOLVER_INNER_TOL:g}"}'
                f'{"_dmg-all" if damage_in_medium else ""}'
                f'{"" if staggered_tol == 1e-3 else f"_stag{staggered_tol:g}"}'
                f'{"" if restart_file is None else "_restart"}'
                f'{"" if solve_damage_problem else "_elastic"}'
                f'{"" if mechanics_solver == "tr_newton" else "_newton-cg"}'
                f'{ {"neo_hookean": "", "saint_venant_kirchhoff": "_svk-void", "linear": "_lin-void"}[void_model]}'
                f'{"_reg-medium" if regularization_region == "medium" else ""}'
                f'{"_gpu" if device == "gpu" else ""}_' + time.strftime('%Y%m%d-%H%M%S'))
    output_folder = None
    if rank == 0:                          # a new folder; never reuse an existing one
        for attempt in range(100):
            candidate = os.path.join(output_root, run_name + (f'_{attempt}' if attempt else ''))
            try:
                os.makedirs(candidate)
                output_folder = candidate
                break
            except FileExistsError:
                continue
    output_folder = comm.bcast(output_folder, root=0)
    output_file = os.path.join(output_folder, 'contact_fracture.nc')
    if len(output_file) > 230:             # muGrid's NetCDF writer overflows a fixed buffer at ~245 characters
        raise ValueError(f'output path too long for muGrid ({len(output_file)} characters): {output_file}')
    log_file = os.path.join(output_folder, 'run.log')
    root_print(f'run folder {output_folder}')
    writer = io_utils.FieldWriter(output_file, output_fields, attributes=run_attributes,
                                  frame_variables=frame_variables)

# ----------------------------------------------------------------------------
# Load stepping
# ----------------------------------------------------------------------------
start_time = time.time()
lam, dlam, increment = 0.0, 1.0 / nb_increments, 0
max_step = dlam
u_previous, dlam_previous = u_fluc.s.copy(), None     # last converged state before the current one
if restart_file is not None:
    previous, mismatch = None, None
    if rank == 0:
        import netCDF4
        with netCDF4.Dataset(restart_file, 'r') as dataset:
            stored = {key: dataset.getncattr(key) for key in dataset.ncattrs()}
            previous = {name: np.asarray(dataset.variables[name][:]) for name in frame_variables}
        mismatch = [key for key in restart_keys
                    if (str(stored.get(key)) != str(run_attributes[key]) if isinstance(run_attributes[key], str)
                        else not np.allclose(np.ravel(stored.get(key, np.nan)), np.ravel(run_attributes[key])))]
    previous, mismatch = comm.bcast((previous, mismatch), root=0)
    if mismatch:
        raise ValueError(f'{restart_file} is a different problem: {mismatch} differ')
    nb_previous = len(previous['lam'])
    for frame in range(nb_previous):         # copy the earlier load path into the new file
        io_utils.load_fields(restart_file, output_fields, frame=frame)
        if frame == nb_previous - 2:
            u_previous = u_fluc.s.copy()
        if writer is not None:
            writer.write(**{name: previous[name][frame] for name in frame_variables})
    # the fields now hold the last frame; H of that frame is the converged history
    history_n.s[...] = history.s
    lam, increment = float(previous['lam'][-1]), int(previous['increment'][-1])
    dlam_previous = float(previous['dlam'][-1])
    dlam = min(2.0 * dlam_previous, max_step)
    start_time -= float(previous['elapsed_time'][-1])   # elapsed_time continues
    root_print(f'restart from {restart_file}: frame {nb_previous - 1}, increment {increment}, lam {lam:.4f}')
update_degradation()
while lam < lam_end - 1e-12:
    dlam = min(dlam, lam_end - lam)
    u_saved, d_saved = u_fluc.s.copy(), damage.s.copy()
    # predictor: linear extrapolation in lam, if admissible (else the warm start u_n)
    if dlam_previous is not None:
        u_fluc.s[...] = u_saved + (dlam / dlam_previous) * (u_saved - u_previous)
        if not set_F(lam + dlam, u_fluc) > 0.0:
            u_fluc.s[...] = u_saved
    ok, message, stats = solve_increment(lam + dlam)
    if not ok:
        u_fluc.s[...], damage.s[...] = u_saved, d_saved
        update_degradation()
        root_print(f'  lam {lam + dlam:.5f}: {message} -> halve the step')
        dlam *= 0.5
        if dlam < min_step:
            root_print('step below min_step: stopping')
            break
        continue
    lam += dlam
    increment += 1
    u_previous, dlam_previous = u_saved, dlam
    history_n.s[...] = history.s
    F = F_field.s
    _, _, P = material(F, g_q)
    mean_P = np.array([[global_sum(np.sum(weights_J * P[i, j])) / np.prod(domain_size) for j in range(dim)]
                       for i in range(dim)])
    J = F[0, 0] * F[1, 1] - F[0, 1] * F[1, 0]
    detF_out.s[0, 0] = J.mean(axis=0)
    d_max = global_max(np.max(damage.s))
    root_print(f'inc {increment:3d}  lam {lam:.4f} (dlam {dlam:.4f})  F10 {lam * H_macro[1, 0]:+.4f}  '
               f'P10 {mean_P[1, 0]:+.4e}  min J {stats["min_J"]:.2e}  max d {d_max:.3f}  '
               f'stagger {stats["staggered_iterations"]:3d}  hessp {stats["nb_hessp"]:6d}  {message}')
    if writer is not None:
        writer.write(increment=increment, lam=lam, dlam=dlam, F10=lam * H_macro[1, 0], P10=mean_P[1, 0],
                     Pxx=mean_P[0, 0], mean_P=mean_P, min_det_F=stats['min_J'], energy=stats['energy'],
                     damage_max=d_max, staggered_iterations=stats['staggered_iterations'],
                     nb_outer=stats['nb_outer'], nb_hessp=stats['nb_hessp'], cg_damage=stats['cg_damage'],
                     converged=int(message == 'converged'), elapsed_time=time.time() - start_time)
    dlam = min(2.0 * dlam, max_step)

if writer is not None:
    writer.close()
root_print(f'elapsed {time.time() - start_time:.1f} s, {increment} increments, final lam {lam:.4f}')
if writer is not None:
    root_print(f'wrote {output_file}')
    if make_figures and rank == 0:
        import subprocess
        plot_script = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                   'plot_contact_fracture_adapted_grid_output.py')
        # a child started under mpirun must not see the MPI launcher variables, otherwise
        # its MPI_Init (muGrid import) tries to join this job and aborts
        plot_env = {k: v for k, v in os.environ.items()
                    if not k.startswith(('OMPI_', 'PMIX_', 'PMI_', 'ORTE_', 'HYDRA_', 'MPI_LOCALRANKID'))}
        for extra in ([], ['--animate']):
            subprocess.run([sys.executable, plot_script, output_file, '--save'] + extra,
                           check=False, env=plot_env)
