"""
PF-CZM (Wu 2017, JMPS 103:72-99) in a periodic concrete meso-cell, 2D (plane strain) and 3D, CPU or GPU
=======================================================================================================

Mesh-size study version of example_2D_pfczm_concrete.py (same model and solvers), with
  * DIM = 2 or 3 (spherical aggregates), DEVICE = cpu or gpu (muGrid CUDA, fields are cupy arrays),
  * a generated, resolution-independent aggregate geometry (random sequential addition of disks / spheres, Fuller
    grading, periodic, minimum gap), evaluated at the pixel centres of every grid,
  * Wu's own concrete parameters and internal length:  three-point bending of Rots (1988), Wu Sec. 4.3:
        E0 = 20 GPa, nu = 0.2, ft = 2.4 MPa, Gf = 0.113 N/mm, b = 1.5 mm (Wu uses b = 2.5 and 1.5 mm there),
    l_ch = E0 Gf / ft^2 = 392 mm, Cornelissen softening (p = 2, a2 = 1.3868, a3 = 0.6567), xi = 2, c0 = pi.
  * aggregates 8-20 mm (5-13 b), E = 70 GPa, nu = 0.2, ft and Gf AGG_STRENGTH (2) x the mortar (crackable),
  * cell CELL_MM = 100 mm, fixed b: b/h = 0.48, 0.96, 1.92, 3.84, 7.68, 15.4 at 32 ... 1024 (coarse grids under-resolve b),
  * time-budget checkpoint / restart (TIME_BUDGET s -> <base>_checkpoint.npz, exit code 3; RESTART_FROM=<npz>).

Equations (per staggered iteration, all linear-symmetric elastic solves):
    sigma = s C_iso : eps,  s = stiffness(x) [omega(d) + k]
    Y = sigma_eq^2 / (2 E),  sigma_eq = (beta_c <sigma1> + sqrt(3 J2)) / (1 + beta_c),  beta_c = 9 (fc/ft = 10),
        effective stress of the phase (2D plane strain: sigma_zz = lam tr eps),  H = max(H_n, Y),  H_n(0) = Y0 = ft^2/(2E)
    min_d  int Gf/(pi b) (2d - d^2 + b^2 |grad d|^2) + omega(d) H,   d_n <= d <= 1   (projected convexified Newton,
        Green-Jacobi PCG for the reduced Newton systems, energy backtracking)
    omega(d) = (1-d)^p / ((1-d)^p + a1 d (1 + a2 d + a2 a3 d^2)),  a1 = 4 l_ch / (pi b)  (per phase)

Outputs (exp_data/): <base>_iterations.csv, <base>_steps.csv (appended across restarts), <base>_frames/ (float16
phase field: 2D every FRAME_EVERY steps, 3D mid-plane z = 1/2 every step and the full field every FULL_EVERY steps),
<base>_phase.npy (aggregate mask, int8), <base>_checkpoint.npz.
"""
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

import numpy as np
import time

from muFFTTO import domain
from muFFTTO import solvers
from muFFTTO import topology_optimization   # energy_stop_kwargs
from muFFTTO import tensor_operations

here = os.path.dirname(os.path.abspath(__file__))
start_wall = time.time()
# ----------------------------------------------------------------------------
# Parameters (env)
# ----------------------------------------------------------------------------
dim = int(os.environ.get('DIM', 2))
N = int(os.environ.get('NPIX', 64))
number_of_pixels = (N,) * dim
domain_size = [1.0] * dim
device = os.environ.get('DEVICE', 'cpu')
if device == 'gpu':
    import cupy as xp
else:
    xp = np
to_host = tensor_operations.to_host
L_unit = float(os.environ.get('CELL_MM', 100.0))   # mm
E_unit = 20e3                                      # MPa (E0 of Wu's Rots beam concrete)
E0, nu = 1.0, 0.2
ft = 2.4 / E_unit
Gf = 0.113 / (E_unit * L_unit)
b_mm = float(os.environ.get('B_MM', 1.5))
b = b_mm / L_unit
beta_c = 10.0 - 1.0
agg_E = float(os.environ.get('AGG_E_GPA', 70.0)) * 1e3 / E_unit
agg_strength = float(os.environ.get('AGG_STRENGTH', 2.0))
agg_d_mm = (float(os.environ.get('AGG_DMIN_MM', 8.0)), float(os.environ.get('AGG_DMAX_MM', 20.0)))
agg_fraction = float(os.environ.get('AGG_FRACTION', 0.45 if dim == 2 else 0.35))
agg_gap_mm = float(os.environ.get('AGG_GAP_MM', 1.5))
seed = int(os.environ.get('SEED', 1))
law = os.environ.get('SOFTENING', 'cornelissen')
p, a2, a3 = {'linear': (2.0, -0.5, 0.0), 'exponential': (2.5, 2 ** (5 / 3) - 3, 0.0),
             'cornelissen': (2.0, 1.3868, 0.6567)}[law]
k_residual = float(os.environ.get('K_RESIDUAL', 1e-3))
load_dir = os.environ.get('LOAD_DIR', 'xx')
macro_strain_direction = np.zeros((dim, dim))
if load_dir == 'xy':
    macro_strain_direction[0, 1] = macro_strain_direction[1, 0] = 0.5        # load = engineering shear gamma
    ic = (0, 1)
else:
    ic = {'xx': (0, 0), 'yy': (1, 1), 'zz': (2, 2)}[load_dir]
    macro_strain_direction[ic] = 1.0                                          # uniaxial strain
load_steps = np.linspace(0.0, float(os.environ.get('LOAD_MAX', 3e-3)), int(os.environ.get('LOAD_STEPS', 300)) + 1)[1:]
staggered_tol = float(os.environ.get('STAGGERED_TOL', 1e-3))
max_staggered_iterations = int(os.environ.get('MAX_STAGGERED', 500))
stop_fraction = float(os.environ.get('STOP_FRACTION', 0.02))
stop_steps = int(os.environ.get('STOP_STEPS', 3))
time_budget = float(os.environ.get('TIME_BUDGET', 1e12))
checkpoint_every = int(os.environ.get('CHECKPOINT_EVERY', 10))
frame_every = int(os.environ.get('FRAME_EVERY', 1 if N <= 256 else 3))
full_every = int(os.environ.get('FULL_EVERY', 20))
newton_tol = 1e-6
max_newton = 50
cg_tol = 1e-6
cg_delay = 10
preconditioner_type = sys.argv[1] if len(sys.argv) > 1 else 'dual'
assert preconditioner_type in ('Green', 'Green_Jacobi', 'dual')
base_name = (f'PFCZM{dim}D_N{N}_{preconditioner_type}_{load_dir}_b{b_mm:g}mm_k{k_residual:g}'
             + os.environ.get('RUN_SUFFIX', ''))
out_dir = os.path.join(here, 'exp_data')
iterations_file = os.path.join(out_dir, base_name + '_iterations.csv')
steps_file = os.path.join(out_dir, base_name + '_steps.csv')
frames_dir = os.path.join(out_dir, base_name + '_frames')
checkpoint_file = os.path.join(out_dir, base_name + '_checkpoint.npz')
restart_from = os.environ.get('RESTART_FROM')

# ----------------------------------------------------------------------------
# Aggregate geometry: periodic RSA of disks / spheres, Fuller grading P(d) = (d / d_max)^0.5 by volume
# ----------------------------------------------------------------------------
def make_aggregates():
    rng = np.random.default_rng(seed)
    dmin, dmax = agg_d_mm[0] / L_unit, agg_d_mm[1] / L_unit
    gap = agg_gap_mm / L_unit
    # number density n(d) ~ d^-(dim + 1.5) (volume fraction density ~ d^-0.5); inverse transform sampling
    e = dim + 0.5
    vol = (lambda d: np.pi * d ** 2 / 4) if dim == 2 else (lambda d: np.pi * d ** 3 / 6)
    sizes, total = [], 0.0
    while total < agg_fraction:
        u = rng.random()
        d = (dmin ** -e + u * (dmax ** -e - dmin ** -e)) ** (-1 / e)
        sizes.append(d)
        total += vol(d)
    sizes = np.sort(sizes)[::-1]
    placed = []
    centres, radii = np.zeros((0, dim)), np.zeros(0)
    for d in sizes:
        r = d / 2
        cand = rng.random((20000, dim))                      # candidate centres, first admissible one is taken
        for i0 in range(0, len(cand), 500):
            c = cand[i0:i0 + 500]
            if len(radii):
                dx = (c[:, None, :] - centres[None, :, :] + 0.5) % 1.0 - 0.5
                ok = np.all(np.sqrt(np.sum(dx ** 2, axis=-1)) >= r + radii[None, :] + gap, axis=1)
            else:
                ok = np.ones(len(c), dtype=bool)
            if ok.any():
                cc = c[np.argmax(ok)]
                placed.append((cc, r))
                centres, radii = np.vstack([centres, cc]), np.append(radii, r)
                break
    return placed


def aggregate_mask(placed):
    """1 at the pixel centres inside an aggregate"""
    h = 1.0 / N
    mask = np.zeros((N,) * dim, dtype=bool)
    for c, r in placed:
        lo = np.floor((c - r) / h - 0.5).astype(int)
        hi = np.ceil((c + r) / h - 0.5).astype(int) + 1
        axes = [np.arange(lo[a], hi[a]) for a in range(dim)]
        grids = np.meshgrid(*[(ax + 0.5) * h - c[a] for a, ax in enumerate(axes)], indexing='ij')
        inside = sum(g ** 2 for g in grids) <= r ** 2
        idx = np.ix_(*[ax % N for ax in axes])
        mask[idx] |= inside
    return mask


placed = make_aggregates()
agg_host = aggregate_mask(placed)
os.makedirs(out_dir, exist_ok=True)
np.save(os.path.join(out_dir, base_name + '_phase.npy'), agg_host.astype(np.int8))
print(f'DIM {dim}, grid {number_of_pixels}, {device}, cell {L_unit:g} mm, b = {b_mm:g} mm (b/h = {b * N:.2f}), '
      f'l_ch = {E0 * Gf / ft ** 2 * L_unit:.0f} mm, {law}, k = {k_residual}, {len(placed)} aggregates '
      f'{agg_d_mm[0]:g}-{agg_d_mm[1]:g} mm, fraction {agg_host.mean():.3f} (target {agg_fraction}), E_agg/E = {agg_E:g}, '
      f'strength x{agg_strength:g}, load {load_dir} to {load_steps[-1]:g} in {len(load_steps)} steps, '
      f'{preconditioner_type}', flush=True)

# ----------------------------------------------------------------------------
# Discretizations
# ----------------------------------------------------------------------------
element_type = 'bilinear_rectangle' if dim == 2 else 'trilinear_hexahedron'
cell_u = domain.PeriodicUnitCell(domain_size=domain_size, problem_type='elasticity')
disc_u = domain.Discretization(cell=cell_u, nb_of_pixels_global=number_of_pixels,
                               discretization_type='finite_element', element_type=element_type, device=device)
cell_d = domain.PeriodicUnitCell(domain_size=domain_size, problem_type='conductivity')
disc_d = domain.Discretization(cell=cell_d, nb_of_pixels_global=number_of_pixels,
                               discretization_type='finite_element', element_type=element_type, device=device)
comm = disc_u.communicator
assert comm.size == 1, 'serial only'
wq = xp.asarray(np.asarray(to_host(disc_d.quadrature_weights))).reshape((-1,) + (1,) * dim)

mortar = xp.asarray(np.where(agg_host, 0.0, 1.0))
stiffness = xp.where(mortar > 0, 1.0, agg_E)
E_x = stiffness * E0
ft_x = xp.where(mortar > 0, ft, agg_strength * ft)
Gf_x = xp.where(mortar > 0, Gf, agg_strength * Gf)
a1 = 4.0 * (E_x * Gf_x / ft_x ** 2) / (np.pi * b)
c_diff = 2.0 * Gf_x * b / np.pi
c_loc = 2.0 * Gf_x / (np.pi * b)
c_diff_ref, c_loc_ref = 2.0 * Gf * b / np.pi, 2.0 * Gf / (np.pi * b)

if dim == 2:     # plane strain
    lam, mu = E0 * nu / ((1 + nu) * (1 - 2 * nu)), E0 / (2 * (1 + nu))
else:
    lam, mu = E0 * nu / ((1 + nu) * (1 - 2 * nu)), E0 / (2 * (1 + nu))
I2 = np.eye(dim)
C_iso = (lam * np.einsum('ij,kl->ijkl', I2, I2)
         + mu * (np.einsum('ik,jl->ijkl', I2, I2) + np.einsum('il,jk->ijkl', I2, I2)))
I_sym = 0.5 * (np.einsum('ik,jl->ijkl', I2, I2) + np.einsum('il,jk->ijkl', I2, I2))
degraded_scale = disc_u.get_quad_field_scalar(name='degraded_scale')
C_d = domain.ScaledMaterial([(degraded_scale, C_iso)])
C_dual = domain.ScaledMaterial([(lambda: 1.0 / degraded_scale.s[0, 0], C_iso)])

# ----------------------------------------------------------------------------
# Fields
# ----------------------------------------------------------------------------
macro_strain = disc_u.get_gradient_size_field(name='macro_strain')
strain = disc_u.get_gradient_size_field(name='strain')
rhs_u = disc_u.get_unknown_size_field(name='rhs_u')
u_fluct = disc_u.get_unknown_size_field(name='u_fluctuation')
dual_temp_1 = disc_u.get_unknown_size_field(name='dual_temp_1')
dual_temp_2 = disc_u.get_unknown_size_field(name='dual_temp_2')
x_scaled_u = disc_u.get_unknown_size_field(name='x_scaled_u')

damage = disc_d.get_unknown_size_field(name='damage')
trial = disc_d.get_unknown_size_field(name='damage_trial')
grad_n = disc_d.get_unknown_size_field(name='energy_gradient')
rhs_d = disc_d.get_unknown_size_field(name='rhs_d')
delta = disc_d.get_unknown_size_field(name='newton_step')
damage_q = disc_d.get_quad_field_scalar(name='damage_q')
grad_d = disc_d.get_gradient_size_field(name='grad_d')
temp_q = disc_d.get_quad_field_scalar(name='temp_q')
temp_n = disc_d.get_unknown_size_field(name='temp_n')
x_scaled_d = disc_d.get_unknown_size_field(name='x_scaled_d')
masked = disc_d.get_unknown_size_field(name='masked')
nq = damage_q.s.shape[1]
H_n = xp.broadcast_to(ft_x ** 2 / (2 * E_x), (nq,) + (N,) * dim).copy()   # history, starts at Y0
H = H_n.copy()
d_n = xp.zeros((N,) * dim)
hess_q = None
active_free = xp.ones((N,) * dim)


# ----------------------------------------------------------------------------
# Constitutive functions
# ----------------------------------------------------------------------------
def omega_and_derivatives(d, derivatives=True):
    d = xp.clip(d, 0.0, 1.0)
    A = (1.0 - d) ** p
    Q = a1 * (d + a2 * d ** 2 + a2 * a3 * d ** 3)
    S = A + Q
    if not derivatives:
        return A / S
    A1 = -p * (1.0 - d) ** (p - 1)
    A2 = p * (p - 1) * (1.0 - d) ** (p - 2) if p != 2 else 2.0
    Q1 = a1 * (1 + 2 * a2 * d + 3 * a2 * a3 * d ** 2)
    Q2 = a1 * (2 * a2 + 6 * a2 * a3 * d)
    num = A1 * Q - A * Q1
    w1 = num / S ** 2
    w2 = ((A2 * Q - A * Q2) * S - 2 * num * (A1 + Q1)) / S ** 3
    return w1, w2


def update_material():
    disc_d.apply_N_operator_mugrid(damage, damage_q)
    degraded_scale.s[0, 0] = (omega_and_derivatives(damage_q.s[0, 0], False) + k_residual) * stiffness


def driving_force_Y():
    disc_u.apply_gradient_operator_symmetrized_mugrid(u_inxyz=u_fluct, grad_u_ijqxyz=strain)
    strain.s[...] += macro_strain.s
    e = strain.s
    tr = sum(e[i, i] for i in range(dim))
    if dim == 2:
        sxx = (lam * tr + 2 * mu * e[0, 0]) * stiffness
        syy = (lam * tr + 2 * mu * e[1, 1]) * stiffness
        sxy = 2 * mu * e[0, 1] * stiffness
        szz = lam * tr * stiffness
        s1 = xp.maximum(0.5 * (sxx + syy) + xp.sqrt(0.25 * (sxx - syy) ** 2 + sxy ** 2), szz)
        mean = (sxx + syy + szz) / 3
        J2 = 0.5 * ((sxx - mean) ** 2 + (syy - mean) ** 2 + (szz - mean) ** 2) + sxy ** 2
    else:
        s = [[(lam * tr * (i == j) + 2 * mu * e[i, j]) * stiffness for j in range(3)] for i in range(3)]
        q = (s[0][0] + s[1][1] + s[2][2]) / 3
        p1 = s[0][1] ** 2 + s[0][2] ** 2 + s[1][2] ** 2
        p2 = (s[0][0] - q) ** 2 + (s[1][1] - q) ** 2 + (s[2][2] - q) ** 2 + 2 * p1       # = dev : dev
        J2 = 0.5 * p2
        pp = xp.sqrt(p2 / 6) + 1e-300
        b00, b11, b22 = (s[0][0] - q) / pp, (s[1][1] - q) / pp, (s[2][2] - q) / pp
        b01, b02, b12 = s[0][1] / pp, s[0][2] / pp, s[1][2] / pp
        detB = b00 * (b11 * b22 - b12 ** 2) - b01 * (b01 * b22 - b12 * b02) + b02 * (b01 * b12 - b11 * b02)
        phi = xp.arccos(xp.clip(detB / 2, -1.0, 1.0)) / 3
        s1 = q + 2 * pp * xp.cos(phi)                                                    # largest principal stress
    seq = (beta_c * xp.maximum(s1, 0.0) + xp.sqrt(3 * J2)) / (1 + beta_c)
    return seq ** 2 / (2 * E_x)


# ----------------------------------------------------------------------------
# Elastic operators and preconditioners
# ----------------------------------------------------------------------------
def K_u(x, Ax):
    disc_u.apply_system_matrix_mugrid(material_data_field=C_d, input_field_inxyz=x, output_field_inxyz=Ax,
                                      formulation='small_strain')


green_u = disc_u.get_preconditioner_Green_mugrid(
    reference_material_data_ijkl=2.0 * mu * I_sym if preconditioner_type == 'Green_Jacobi' else C_iso,
    formulation='small_strain', name='green_u')
jacobi_u = None


def M_u(x, Px):
    if preconditioner_type == 'dual':
        disc_u.fft.communicate_ghosts(x)
        disc_u.apply_preconditioner_mugrid(preconditioner_Fourier_fnfnqks=green_u, input_nodal_field_fnxyz=x,
                                           output_nodal_field_fnxyz=dual_temp_1)
        disc_u.apply_system_matrix_mugrid(material_data_field=C_dual, input_field_inxyz=dual_temp_1,
                                          output_field_inxyz=dual_temp_2, formulation='small_strain')
        disc_u.fft.communicate_ghosts(dual_temp_2)
        disc_u.apply_preconditioner_mugrid(preconditioner_Fourier_fnfnqks=green_u, input_nodal_field_fnxyz=dual_temp_2,
                                           output_nodal_field_fnxyz=Px)
    elif preconditioner_type == 'Green_Jacobi':
        x_scaled_u.s[...] = jacobi_u.s * x.s
        disc_u.apply_preconditioner_mugrid(preconditioner_Fourier_fnfnqks=green_u, input_nodal_field_fnxyz=x_scaled_u,
                                           output_nodal_field_fnxyz=Px)
        Px.s[...] *= jacobi_u.s
    else:
        disc_u.apply_preconditioner_mugrid(preconditioner_Fourier_fnfnqks=green_u, input_nodal_field_fnxyz=x,
                                           output_nodal_field_fnxyz=Px)


# ----------------------------------------------------------------------------
# Damage problem
# ----------------------------------------------------------------------------
def apply_diffusion_mass(x, Ax, local_q, cd):
    """[cd B^T W B + N^T W local N] x"""
    disc_d.apply_gradient_operator_mugrid(u_inxyz=x, grad_u_ijqxyz=grad_d)
    grad_d.s[...] *= cd
    disc_d.apply_gradient_transposed_operator_mugrid(gradient_field_ijqxyz=grad_d, div_u_fnxyz=Ax, apply_weights=True)
    if local_q is None:
        return
    disc_d.apply_N_operator_mugrid(x, temp_q)
    temp_q.s[0, 0] *= local_q
    disc_d.apply_N_transposed_operator_mugrid(quad_field_ijqxyz=temp_q, nodal_field_inxyz=temp_n, apply_weights=True)
    Ax.s[...] += temp_n.s


def energy(d_field):
    disc_d.apply_N_operator_mugrid(d_field, damage_q)
    dq = damage_q.s[0, 0]
    disc_d.apply_gradient_operator_mugrid(u_inxyz=d_field, grad_u_ijqxyz=grad_d)
    g2 = xp.sum(grad_d.s[0] ** 2, axis=0)
    dens = (Gf_x / (np.pi * b)) * (2 * dq - dq ** 2 + b ** 2 * g2) + omega_and_derivatives(dq, False) * H
    return float(xp.sum(wq * dens))


def energy_gradient():
    global hess_q
    disc_d.apply_N_operator_mugrid(damage, damage_q)
    dq = damage_q.s[0, 0]
    w1, w2 = omega_and_derivatives(dq)
    hess_q = xp.maximum(w2 * H - c_loc, 0.1 * c_loc)
    temp_q.s[0, 0] = c_loc * (1 - dq) + w1 * H
    disc_d.apply_N_transposed_operator_mugrid(quad_field_ijqxyz=temp_q, nodal_field_inxyz=temp_n, apply_weights=True)
    apply_diffusion_mass(damage, grad_n, None, c_diff)
    grad_n.s[...] += temp_n.s


def K_newton(x, Ax):
    masked.s[0, 0] = active_free * x.s[0, 0]
    apply_diffusion_mass(masked, Ax, hess_q, c_diff)
    Ax.s[0, 0] = active_free * Ax.s[0, 0] + (1.0 - active_free) * x.s[0, 0]


green_d = disc_d.get_preconditioner_Green_mugrid(
    reference_material_data_ijkl=None, invert_zero_mode=True, name='green_damage',
    operator=lambda input_field_inxyz, output_field_inxyz: apply_diffusion_mass(
        input_field_inxyz, output_field_inxyz, c_loc_ref, c_diff_ref))
jacobi_d = None


def M_d(x, Px):
    x_scaled_d.s[0, 0] = active_free * jacobi_d.s[0, 0] * x.s[0, 0]
    disc_d.apply_preconditioner_mugrid(preconditioner_Fourier_fnfnqks=green_d, input_nodal_field_fnxyz=x_scaled_d,
                                       output_nodal_field_fnxyz=Px)
    Px.s[0, 0] = active_free * jacobi_d.s[0, 0] * Px.s[0, 0] + (1.0 - active_free) * x.s[0, 0]


def solve_damage():
    """projected Newton for min Pi_d on d_n <= d <= 1; returns (Newton its, CG its)"""
    global active_free, jacobi_d
    nb_cg = 0
    E_cur = energy(damage)
    for it in range(max_newton):
        energy_gradient()
        d = damage.s[0, 0]
        g = grad_n.s[0, 0]
        at_bound = ((d <= d_n + 1e-12) & (g > 0)) | ((d >= 1.0 - 1e-12) & (g < 0))
        active_free = xp.where(at_bound, 0.0, 1.0)
        rhs_d.s[0, 0] = -active_free * g
        if float(xp.max(xp.abs(rhs_d.s))) == 0.0:
            return it, nb_cg
        jacobi_d = disc_d.get_preconditioner_Jacobi_mugrid(
            operator=lambda input_field_inxyz, output_field_inxyz: K_newton(input_field_inxyz, output_field_inxyz),
            name='jacobi_d')
        delta.s[...] = 0.0
        norms = solvers.conjugate_gradients_mugrid_experimental(
            comm=comm, fc=disc_d.field_collection, hessp=K_newton, b=rhs_d, x=delta, P=M_d, maxiter=5000,
            stop_crit_norm='energy_lower_estim', d_min=cg_delay,
            **topology_optimization.energy_stop_kwargs(disc_d, K_newton, delta, cg_tol))[1]
        nb_cg += int(norms['stop_iteration']['energy_lower_estim'])
        t = 1.0
        for _ in range(30):
            trial.s[0, 0] = xp.clip(d + t * active_free * delta.s[0, 0], d_n, 1.0)
            E_trial = energy(trial)
            if E_trial <= E_cur + 1e-4 * float(xp.sum(g * (trial.s[0, 0] - d))) or t < 1e-6:
                break
            t *= 0.5
        step = float(xp.max(xp.abs(trial.s[0, 0] - d)))
        damage.s[...] = trial.s
        E_cur = E_trial
        if step < newton_tol:
            return it + 1, nb_cg
    return max_newton, nb_cg


def save_checkpoint(next_step, stresses):
    np.savez(checkpoint_file + '.tmp.npz', next_step=next_step, damage=to_host(damage.s), u_fluct=to_host(u_fluct.s),
             H_n=to_host(H_n), stresses=np.array(stresses))
    os.replace(checkpoint_file + '.tmp.npz', checkpoint_file)


# ----------------------------------------------------------------------------
# Load stepping
# ----------------------------------------------------------------------------
os.makedirs(frames_dir, exist_ok=True)
first_step = 0
macro_stresses = [0.0]
if restart_from:
    st = np.load(restart_from)
    first_step = int(st['next_step'])
    damage.s[...] = xp.asarray(st['damage'])
    u_fluct.s[...] = xp.asarray(st['u_fluct'])
    H_n = xp.asarray(st['H_n'])
    H = H_n.copy()
    d_n = damage.s[0, 0].copy()
    macro_stresses = list(st['stresses'])
    print(f'restart from {restart_from} at load step {first_step}', flush=True)
else:
    with open(iterations_file, 'w') as f:
        f.write('step,load,staggered_iteration,cg_u,time_u,newton_d,cg_d,time_d,damage_change,max_d,min_s_matrix\n')
    with open(steps_file, 'w') as f:
        f.write('step,load,stress,max_d,min_s_matrix,crack_fraction_0.9,staggered_its,converged,cg_u_step,mean_cg_u,'
                'time_u_step,time_step\n')
update_material()
total_cg_u, nb_not_converged, nb_broken = 0, 0, 0
volume_mortar = float(xp.sum(mortar))
for step in range(first_step, len(load_steps)):
    load = load_steps[step]
    disc_u.get_macro_gradient_field_mugrid(macro_gradient_ij=load * macro_strain_direction,
                                           macro_gradient_field_ijqxyz=macro_strain)
    cg_u_step, time_u_step = 0, 0.0
    step_start = time.time()
    for stagger in range(max_staggered_iterations):
        t0 = time.time()
        disc_u.get_rhs_mugrid(material_data_field_ijklqxyz=C_d, macro_gradient_field_ijqxyz=macro_strain, rhs_inxyz=rhs_u)
        if preconditioner_type == 'Green_Jacobi':
            jacobi_u = disc_u.get_preconditioner_Jacobi_mugrid(material_data_field_ijklqxyz=C_d,
                                                               formulation='small_strain', name='jacobi_u')
        norms_u = solvers.conjugate_gradients_mugrid_experimental(
            comm=comm, fc=disc_u.field_collection, hessp=K_u, b=rhs_u, x=u_fluct, P=M_u, maxiter=20000,
            stop_crit_norm='energy_lower_estim', d_min=cg_delay,
            **topology_optimization.energy_stop_kwargs(disc_u, K_u, u_fluct, cg_tol))[1]
        cg_u = int(norms_u['stop_iteration']['energy_lower_estim'])
        time_u = time.time() - t0
        t0 = time.time()
        H = xp.maximum(H_n, driving_force_Y())
        d_old = damage.s[0, 0].copy()
        newton_its, cg_d = solve_damage()
        update_material()
        time_d = time.time() - t0
        change = float(xp.max(xp.abs(damage.s[0, 0] - d_old)))
        s_min = float(xp.min(degraded_scale.s[0, 0]))
        cg_u_step += cg_u
        time_u_step += time_u
        total_cg_u += cg_u
        with open(iterations_file, 'a') as f:
            f.write(f'{step},{load:.6g},{stagger + 1},{cg_u},{time_u:.4g},{newton_its},{cg_d},{time_d:.4g},'
                    f'{change:.4g},{float(xp.max(damage.s)):.6g},{s_min:.4g}\n')
        if change < staggered_tol:
            break
    converged = change < staggered_tol
    nb_not_converged += not converged
    H_n = H.copy()
    d_n = damage.s[0, 0].copy()
    sigma = disc_u.get_homogenized_stress_mugrid(material_data_field_ijklqxyz=C_d, displacement_field_inxyz=u_fluct,
                                                 macro_gradient_field_ijqxyz=macro_strain, formulation='small_strain')
    crack = float(xp.sum((d_n > 0.9) * mortar)) / volume_mortar
    macro_stresses.append(float(sigma[ic]))
    with open(steps_file, 'a') as f:
        f.write(f'{step},{load:.6g},{sigma[ic]:.6g},{float(xp.max(d_n)):.6g},{s_min:.4g},{crack:.4g},{stagger + 1},'
                f'{int(converged)},{cg_u_step},{cg_u_step / (stagger + 1):.2f},{time_u_step:.3f},'
                f'{time.time() - step_start:.3f}\n')
    if dim == 2 and step % frame_every == 0:
        np.save(os.path.join(frames_dir, f'd_{step:04d}.npy'), to_host(d_n).astype(np.float16))
    if dim == 3:
        np.save(os.path.join(frames_dir, f'd_mid_{step:04d}.npy'), to_host(d_n[:, :, N // 2]).astype(np.float16))
        if step % full_every == 0:
            np.save(os.path.join(frames_dir, f'd_full_{step:04d}.npy'), to_host(d_n).astype(np.float16))
    print(f'step {step:3d}  load = {load:.4e}  sigma = {sigma[ic] * E_unit:.4f} MPa  max d = {float(xp.max(d_n)):.4f}  '
          f'min s = {s_min:.2e}  crack(d>0.9) = {crack:.3f}  staggered its = {stagger + 1:3d}  '
          f'step CG u = {cg_u_step}  mean CG u = {cg_u_step / (stagger + 1):.1f}  '
          f'time (u, step) = ({time_u_step:.2f}, {time.time() - step_start:.2f}) s'
          + ('' if converged else f'  ACCEPTED NOT CONVERGED after MAX_STAGGERED = {max_staggered_iterations} '
                                  f'(max |d - d_old| = {change:.2e})'), flush=True)
    nb_broken = nb_broken + 1 if sigma[ic] < stop_fraction * max(macro_stresses) else 0
    if stop_fraction > 0 and nb_broken >= stop_steps:
        print(f'stress < {stop_fraction:g} of the peak for {stop_steps} steps: stop')
        break
    if (step + 1) % checkpoint_every == 0:
        save_checkpoint(step + 1, macro_stresses)
    if time.time() - start_wall > time_budget and step + 1 < len(load_steps):
        save_checkpoint(step + 1, macro_stresses)
        print(f'time budget {time_budget:g} s reached after load step {step}: checkpoint written, exit 3', flush=True)
        sys.exit(3)

if dim == 3:
    np.save(os.path.join(frames_dir, 'd_full_final.npy'), to_host(d_n).astype(np.float16))
print(f'Elapsed time: {time.time() - start_wall:.1f} s')
print(f'{preconditioner_type}: total CG iterations elasticity = {total_cg_u} (this segment)')
if nb_not_converged:
    print(f'WARNING: {nb_not_converged} load steps accepted NOT CONVERGED (MAX_STAGGERED = {max_staggered_iterations})')
print('RUN FINISHED')
