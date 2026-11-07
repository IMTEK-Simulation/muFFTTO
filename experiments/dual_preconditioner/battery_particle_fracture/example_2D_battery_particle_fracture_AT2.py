"""
Chemo-mechanical AT2 fracture of a core-shell cathode particle during lithiation (2D periodic FFT cell)
=======================================================================================================

Adapted from ../damage_AT2_small_strain/schneider2025_notch/example_2D_phase_field_fracture_AT2.py (copied, not
imported). Model and parameters after Tu, Wu & Martinez-Paneda, Applied Energy 403 (2026) 127105
(NMC811 core @ NMC532 shell, Table 1), cast into a periodic 2D cell (deviations: see README.md):

  Li diffusion, normalised filling c = c_Li / c_max,i (continuous across the core-shell interface), implicit Euler:
      (1/dt) M_m c + B^T W m D(d) B c = (1/dt) M_m c_old + N^T W q
      m = c_max,i / c_max,core (capacity weight), D = D_i ((1 - d)^2 + K_DIFF) (damage blocks Li, Tu Sec. 2.4)
      q: constant current (CC, Tu eq. 31) as a volumetric source in a surface band of width BAND inside the shell;
         when the CV current KAPPA int_band (1 - c) has dropped to the CC current, constant voltage (CV): q = KAPPA (1 - c) in the band, until the current drops
         to 10 % of the CC current (Tu Sec. 2.5)
  Elasticity, small strain, plane strain, zero macroscopic strain (E = 0):
      eps = sym(grad u~) - eps_Li,   eps_Li = Omega_i c_max,i / 3 (c - c0) I   (Tu eq. 6)
      B^T W g(d) C_0(x) B u~ = B^T W g(d) C_0(x) eps_Li
      core / shell C_0 from Table 1, matrix (binder / electrolyte) MATRIX_STIFFNESS x core, no Li strain, no damage
  AT2 damage (Tu eq. 15-17, Amor split, history H): [Gc l B^T W B + N^T W (Gc/l + 2H) N] d = N^T W 2H
      Gc per phase, interface Gc_I = GC_INTERFACE x Gc_ave blended with zeta = exp(-|r - R| / L_ZETA) (Tu eq. 20-23)
      initial surface crack in the shell: H = ALPHA0 on one pixel row, length A_BAR x shell thickness (Tu eq. 32)

Nondimensionalisation: length L_cell = 12 um, stress E_core = 230 GPa, time L^2 / D_core = 4417 s, c by c_max,core.

Staggered per time step: diffusion -> elasticity -> history -> damage, until max |d - d_old| < 1e-3 or
MAX_STAGGERED (then ACCEPTED and logged).

Preconditioners (argv[1]) of the ELASTIC solve: 'Green' (C_ref = core tangent), 'Green_Jacobi' (2 mu I_sym + Jacobi),
'dual' (G K(C_ref C_d^-1 C_ref) G). Damage: Green-Jacobi (as the base script). Diffusion: DIFF_PRECOND
(Green / Green_Jacobi / dual, reference D_core, 1/dt); DIFF_SHADOW=1 also runs the other two on the same system from
the same warm start (iterations counted, solution discarded): all three on an identical trajectory.
CG stops on the energy-error estimate (tol 1e-6, warm start, delay d_min 10) everywhere.
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

# ----------------------------------------------------------------------------
# Physical parameters (Tu et al. 2026, Table 1) and scales
# ----------------------------------------------------------------------------
L_cell = float(os.environ.get('L_CELL_UM', 12.0)) * 1e-6     # m
E_scale = 230e9                                              # Pa (core)
D_scale = 3.26e-14                                           # m^2/s (core)
t_scale = L_cell ** 2 / D_scale                              # s
cmax = {'core': 51765.0, 'shell': 49000.0}                   # mol/m^3
Omega = {'core': 7.88e-7, 'shell': 4.86e-7}                  # m^3/mol
D_phys = {'core': 3.26e-14, 'shell': 2.48e-14}
E_phys = {'core': 230e9, 'shell': 201e9}
nu = 0.253
Gc_phys = {'core': 0.299, 'shell': 0.408}                    # N/m
R_core_um = float(os.environ.get('R_CORE_UM', 4.0))
h_bar = float(os.environ.get('H_BAR', 0.3))                  # shell thickness / R_core (paper range 0.1 - 0.3)
c_rate = float(os.environ.get('C_RATE', 1.0))
c_init = 0.1                                                 # SOL 10 % at the start, stress free

# nondimensional
R_core = R_core_um * 1e-6 / L_cell
R_out = R_core * (1.0 + h_bar)
shell = R_out - R_core
m_cap = {p: cmax[p] / cmax['core'] for p in cmax}
beta = {p: Omega[p] * cmax[p] / 3.0 for p in cmax}           # Li strain per unit filling
D_nd = {p: D_phys[p] / D_scale for p in D_phys}
E_nd = {p: E_phys[p] / E_scale for p in E_phys}
Gc_nd = {p: Gc_phys[p] / (E_scale * L_cell) for p in Gc_phys}
gc_interface = float(os.environ.get('GC_INTERFACE', 1.0))    # Gc_I / Gc_ave
alpha0 = 1e12 / E_scale

# ----------------------------------------------------------------------------
# Numerical parameters (env)
# ----------------------------------------------------------------------------
# DIM (env): 2 (default; plane strain disk) or 3 (sphere in a periodic cube); DEVICE (env): cpu or gpu (muGrid CUDA,
# field data are cupy arrays; the geometry is set up on the host and copied)
dim = int(os.environ.get('DIM', 2))
domain_size = [1.0] * dim
N = int(os.environ.get('NPIX', 64))
number_of_pixels = (N,) * dim
h = 1.0 / N
element_type = 'bilinear_rectangle' if dim == 2 else 'trilinear_hexahedron'
device = os.environ.get('DEVICE', 'cpu')
if device == 'gpu':
    import cupy as xp
else:
    xp = np
preconditioner_type = sys.argv[1] if len(sys.argv) > 1 else 'dual'
assert preconditioner_type in ('Green', 'Green_Jacobi', 'dual')
matrix_stiffness = float(os.environ.get('MATRIX_STIFFNESS', 1e-2))
k_residual = float(os.environ.get('K_RESIDUAL', 1e-4))
length_scale = float(os.environ.get('LENGTH_SCALE', 1.0 / 16))   # fixed: l/h = 2, 4, 8 at 32^2, 64^2, 128^2
l_zeta = float(os.environ.get('L_ZETA', 1.0 / 32))
# L_CORE_UM, L_SHELL_UM (env): Tu et al.'s per-phase length scales in um (0.23 / 0.27) instead of LENGTH_SCALE (the
# matrix takes the shell value); L_ZETA_UM (env): the interface zone width in um (Tu: 0.1)
l_phys_um = {'core': float(os.environ['L_CORE_UM']), 'shell': float(os.environ['L_SHELL_UM'])} \
    if 'L_CORE_UM' in os.environ else None
if 'L_ZETA_UM' in os.environ:
    l_zeta = float(os.environ['L_ZETA_UM']) * 1e-6 / L_cell
D_m = float(os.environ.get('D_MATRIX', 1e-3))
k_diff = float(os.environ.get('K_DIFF', 1e-4))
diff_degrade = os.environ.get('DIFF_DEGRADE', '1') == '1'
dt = float(os.environ.get('DT', 0.005))                     # in t_scale (C = 1: 0.82 t_scale)
t_max = float(os.environ.get('T_MAX', 2.0 / c_rate))
band = float(os.environ.get('BAND', 1.0 / 32))
kappa = float(os.environ.get('KAPPA', 1e3))
a_bar = float(os.environ.get('A_BAR', 0.5))                  # initial crack / shell thickness (0: no initial crack)
max_staggered = int(os.environ.get('MAX_STAGGERED', 500))
staggered_tol = float(os.environ.get('STAGGERED_TOL', 1e-3))
diff_precond = os.environ.get('DIFF_PRECOND', 'Green')
diff_shadow = os.environ.get('DIFF_SHADOW', '1') == '1'
snap_every = int(os.environ.get('SNAP_EVERY', 10))
cg_tol = 1e-6
cg_delay = 10

run_tag = f'N{N}_{preconditioner_type}_ms{matrix_stiffness:g}' + ('_3D' if dim == 3 else '') \
    + os.environ.get('RUN_SUFFIX', '')
# CHECKPOINT_SECONDS (env, default 1800): write <run>_checkpoint.npz after a time step at most this often (wall time);
# RESTART=1 (env): continue from that checkpoint (the CSV rows of later steps are dropped, then appended to)
checkpoint_seconds = float(os.environ.get('CHECKPOINT_SECONDS', 1800))
restart = os.environ.get('RESTART', '0') == '1'
here = os.path.dirname(os.path.abspath(__file__))
data_folder = os.path.join(here, 'exp_data')
os.makedirs(data_folder, exist_ok=True)

# ----------------------------------------------------------------------------
# Discretizations: vector (u~) and scalar (d and c)
# ----------------------------------------------------------------------------
cell_u = domain.PeriodicUnitCell(domain_size=domain_size, problem_type='elasticity')
disc_u = domain.Discretization(cell=cell_u, nb_of_pixels_global=number_of_pixels,
                               discretization_type='finite_element', element_type=element_type, device=device)
cell_d = domain.PeriodicUnitCell(domain_size=domain_size, problem_type='conductivity')
disc_d = domain.Discretization(cell=cell_d, nb_of_pixels_global=number_of_pixels,
                               discretization_type='finite_element', element_type=element_type, device=device)
comm = disc_u.communicator
assert comm.size == 1, 'serial only'

# ----------------------------------------------------------------------------
# Geometry: one core-shell particle in the centre; phase per pixel from the pixel centre
# ----------------------------------------------------------------------------
centre = np.full(dim, 0.5)
nodes = np.asarray(tensor_operations.to_host(disc_d.fft.coords))       # [dim, x, y(, z)], lower-left node of the pixel
r_pix = np.sqrt(sum((nodes[a] + 0.5 * h - centre[a]) ** 2 for a in range(dim)))
r_node = np.sqrt(sum((nodes[a] - centre[a]) ** 2 for a in range(dim)))
core = (r_pix < R_core).astype(float)
shell_px = ((r_pix >= R_core) & (r_pix < R_out)).astype(float)
particle = core + shell_px
surface_band = ((r_pix < R_out) & (r_pix >= R_out - band)).astype(float)
node_in_particle = r_node < R_out


def per_phase(values, matrix_value):
    return core * values['core'] + shell_px * values['shell'] + (1.0 - particle) * matrix_value


stiffness = per_phase(E_nd, matrix_stiffness)                   # per pixel, relative to the core
beta_px = per_phase(beta, 0.0)
m_px = per_phase(m_cap, 1.0)
D0_px = per_phase(D_nd, D_m)
zeta = np.exp(-np.abs(r_pix - R_core) / l_zeta)
if l_phys_um is not None:          # per pixel length scale (nondimensional)
    l_nd = {p: l_phys_um[p] * 1e-6 / L_cell for p in l_phys_um}
    length_scale = per_phase(l_nd, l_nd['shell'])
    l_ref = l_nd['core']
else:
    l_ref = length_scale
l_min = float(np.min(length_scale))
Gc_ave = 0.5 * (Gc_nd['core'] + Gc_nd['shell'])
Gc_bulk = per_phase(Gc_nd, Gc_nd['shell'])
Gc_px = (1.0 - zeta) ** 2 * (Gc_bulk - gc_interface * Gc_ave) + gc_interface * Gc_ave   # Tu eq. 23
area = h ** dim                                                         # pixel area / voxel volume
Q_cc = c_rate * t_scale / 3600.0 * np.sum(m_px * particle) * area      # Tu eq. 31: full capacity in 1/C hours
q_cc = Q_cc / (np.sum(surface_band) * area)                             # per unit band area

print(f'grid {number_of_pixels}, cell {L_cell * 1e6:.2f} um, h = {h * L_cell * 1e6:.4f} um, l_min = {l_min:.4f} '
      f'(min l/h = {l_min / h:.2f}, {l_min * L_cell * 1e6:.3f} um), l_zeta = {l_zeta * L_cell * 1e6:.3f} um, '
      f't_scale = {t_scale:.1f} s, R_core = {R_core:.4f}, R_out = {R_out:.4f}, shell = {shell:.4f} ({shell / h:.1f} px), '
      f'matrix stiffness = {matrix_stiffness:g}, dt = {dt} ({dt * t_scale:.1f} s), C = {c_rate}, '
      f'particle fraction = {particle.mean():.3f}, band px = {int(surface_band.sum())}, '
      f'Gc core/shell = {Gc_nd["core"]:.3e}/{Gc_nd["shell"]:.3e}, beta core/shell = {beta["core"]:.4f}/'
      f'{beta["shell"]:.4f}, precond = {preconditioner_type}, diffusion precond = {diff_precond} '
      f'(shadow {diff_shadow})', flush=True)

# ----------------------------------------------------------------------------
# Elastic material (plane strain Lame of the core; per pixel scale)
# ----------------------------------------------------------------------------
lam = nu / ((1 + nu) * (1 - 2 * nu))
mu = 0.5 / (1 + nu)
bulk_qxyz = (lam + 2.0 * mu / dim) * stiffness
mu_qxyz = mu * stiffness
crack_driving = particle

I2 = np.eye(dim)
C_iso = (lam * np.einsum('ij,kl->ijkl', I2, I2)
         + mu * (np.einsum('ik,jl->ijkl', I2, I2) + np.einsum('il,jk->ijkl', I2, I2)))
I_sym = 0.5 * (np.einsum('ik,jl->ijkl', I2, I2) + np.einsum('il,jk->ijkl', I2, I2))
degraded_scale = disc_u.get_quad_field_scalar(name='degraded_scale')   # s = g(d) * stiffness
C_d = domain.ScaledMaterial([(degraded_scale, C_iso)])
C_dual = domain.ScaledMaterial([(lambda: 1.0 / degraded_scale.s[0, 0], C_iso)])

# ----------------------------------------------------------------------------
# Fields
# ----------------------------------------------------------------------------
eigen_rhs = disc_u.get_gradient_size_field(name='macro_strain')     # E - eps_Li (E = 0)
strain = disc_u.get_gradient_size_field(name='strain')
rhs_u = disc_u.get_unknown_size_field(name='rhs_u')
u_fluct = disc_u.get_unknown_size_field(name='u_fluctuation')
dual_temp_1 = disc_u.get_unknown_size_field(name='dual_temp_1')
dual_temp_2 = disc_u.get_unknown_size_field(name='dual_temp_2')
x_scaled_u = disc_u.get_unknown_size_field(name='x_scaled_u')

damage = disc_d.get_unknown_size_field(name='damage')
damage_old = disc_d.get_unknown_size_field(name='damage_old')
rhs_d = disc_d.get_unknown_size_field(name='rhs_d')
damage_q = disc_d.get_quad_field_scalar(name='damage_at_quad')
history = disc_d.get_quad_field_scalar(name='history')
history_n = disc_d.get_quad_field_scalar(name='history_converged')
reaction = disc_d.get_quad_field_scalar(name='reaction')
grad_d = disc_d.get_gradient_size_field(name='grad_damage')
temp_q = disc_d.get_quad_field_scalar(name='temp_q')
temp_n = disc_d.get_unknown_size_field(name='temp_n')
x_scaled_d = disc_d.get_unknown_size_field(name='x_scaled_d')

conc = disc_d.get_unknown_size_field(name='concentration')
conc_old = disc_d.get_unknown_size_field(name='concentration_old')
conc_start = disc_d.get_unknown_size_field(name='concentration_start')
conc_shadow = disc_d.get_unknown_size_field(name='concentration_shadow')
conc_q = disc_d.get_quad_field_scalar(name='concentration_at_quad')
rhs_c = disc_d.get_unknown_size_field(name='rhs_c')
diffusivity = disc_d.get_quad_field_scalar(name='diffusivity')       # m D(d)
mass_c = disc_d.get_quad_field_scalar(name='mass_c')                 # m / dt (+ KAPPA in the band during CV)
c_temp_1 = disc_d.get_unknown_size_field(name='c_temp_1')
c_temp_2 = disc_d.get_unknown_size_field(name='c_temp_2')
x_scaled_c = disc_d.get_unknown_size_field(name='x_scaled_c')
weights_q = np.asarray(tensor_operations.to_host(disc_d.quadrature_weights)).reshape((-1,) + (1,) * dim)

# initial surface crack (Tu eq. 32): one pixel row on the +x axis, from R_out inwards over A_BAR x shell
crack_seed = np.zeros_like(history.s[0, 0])
if a_bar > 0:
    a_len = max(a_bar * shell, h)
    y_row = centre[1] + 0.5 * h                                      # centre of a pixel row
    in_row = (np.abs(nodes[1] + 0.5 * h - y_row) < 0.5 * h) & (nodes[0] + 0.5 * h > centre[0])
    if dim == 3:                                                     # a planar patch of width ~ a_len in z
        in_row &= np.abs(nodes[2] + 0.5 * h - centre[2]) < max(a_len, h)
    seed_px = in_row & (r_pix < R_out) & (r_pix > R_out - a_len)
    crack_seed = np.broadcast_to(alpha0 * seed_px, crack_seed.shape).astype(float)   # over q
history_n.s[0, 0] = xp.asarray(crack_seed)

# host copies for output, device copies for the field arithmetic
particle_h, core_h = particle, core
(stiffness, beta_px, m_px, D0_px, Gc_px, particle, core, shell_px, surface_band, r_pix, bulk_qxyz, mu_qxyz,
 weights_q) = (xp.asarray(a) for a in (stiffness, beta_px, m_px, D0_px, Gc_px, particle, core, shell_px, surface_band,
                                        r_pix, bulk_qxyz, mu_qxyz, weights_q))
crack_driving = particle
if not np.isscalar(length_scale):
    length_scale = xp.asarray(length_scale)


# ----------------------------------------------------------------------------
# Operators
# ----------------------------------------------------------------------------
def K_u(x, Ax):
    disc_u.apply_system_matrix_mugrid(material_data_field=C_d, input_field_inxyz=x,
                                      output_field_inxyz=Ax, formulation='small_strain')


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
        disc_u.apply_preconditioner_mugrid(preconditioner_Fourier_fnfnqks=green_u,
                                           input_nodal_field_fnxyz=x_scaled_u, output_nodal_field_fnxyz=Px)
        Px.s[...] *= jacobi_u.s
    else:
        disc_u.apply_preconditioner_mugrid(preconditioner_Fourier_fnfnqks=green_u,
                                           input_nodal_field_fnxyz=x, output_nodal_field_fnxyz=Px)


def apply_scalar_operator(x, Ax, grad_coef, reaction_coef):
    """[B^T W a B + N^T W r N] x, a and r at the quadrature points (scalars or [q, x, y] arrays)"""
    disc_d.apply_gradient_operator_mugrid(u_inxyz=x, grad_u_ijqxyz=grad_d)
    grad_d.s[...] *= grad_coef
    disc_d.apply_gradient_transposed_operator_mugrid(gradient_field_ijqxyz=grad_d, div_u_fnxyz=Ax,
                                                     apply_weights=True)
    disc_d.apply_N_operator_mugrid(x, temp_q)
    temp_q.s[...] *= reaction_coef
    disc_d.apply_N_transposed_operator_mugrid(quad_field_ijqxyz=temp_q, nodal_field_inxyz=temp_n, apply_weights=True)
    Ax.s[...] += temp_n.s


# damage (Green-Jacobi, homogeneous reference with the core Gc)
def K_d(input_field_inxyz, output_field_inxyz):
    apply_scalar_operator(input_field_inxyz, output_field_inxyz, Gc_px * length_scale, reaction.s)


green_d = disc_d.get_preconditioner_Green_mugrid(
    reference_material_data_ijkl=None, invert_zero_mode=True, name='green_d',
    operator=lambda input_field_inxyz, output_field_inxyz: apply_scalar_operator(
        input_field_inxyz, output_field_inxyz, Gc_nd['core'] * l_ref, Gc_nd['core'] / l_ref))
jacobi_d = None


def M_d(x, Px):
    x_scaled_d.s[...] = jacobi_d.s * x.s
    disc_d.apply_preconditioner_mugrid(preconditioner_Fourier_fnfnqks=green_d,
                                       input_nodal_field_fnxyz=x_scaled_d, output_nodal_field_fnxyz=Px)
    Px.s[...] *= jacobi_d.s


# diffusion: N^T W (m/dt [+ kappa]) N + B^T W m D B; reference D_core, 1/dt (regular: zero mode inverted)
def K_c(input_field_inxyz, output_field_inxyz):
    apply_scalar_operator(input_field_inxyz, output_field_inxyz, diffusivity.s, mass_c.s)


green_c = disc_d.get_preconditioner_Green_mugrid(
    reference_material_data_ijkl=None, invert_zero_mode=True, name='green_c',
    operator=lambda input_field_inxyz, output_field_inxyz: apply_scalar_operator(
        input_field_inxyz, output_field_inxyz, 1.0, 1.0 / dt))
jacobi_c = None


def make_M_c(kind):
    def M_c(x, Px):
        if kind == 'dual':          # G [ N^T W (r_ref^2 / r) N + B^T W (D_ref^2 / D) B ] G
            disc_d.apply_preconditioner_mugrid(preconditioner_Fourier_fnfnqks=green_c, input_nodal_field_fnxyz=x,
                                               output_nodal_field_fnxyz=c_temp_1)
            apply_scalar_operator(c_temp_1, c_temp_2, 1.0 / diffusivity.s, dt ** -2 / mass_c.s)
            disc_d.apply_preconditioner_mugrid(preconditioner_Fourier_fnfnqks=green_c, input_nodal_field_fnxyz=c_temp_2,
                                               output_nodal_field_fnxyz=Px)
        elif kind == 'Green_Jacobi':
            x_scaled_c.s[...] = jacobi_c.s * x.s
            disc_d.apply_preconditioner_mugrid(preconditioner_Fourier_fnfnqks=green_c,
                                               input_nodal_field_fnxyz=x_scaled_c, output_nodal_field_fnxyz=Px)
            Px.s[...] *= jacobi_c.s
        else:
            disc_d.apply_preconditioner_mugrid(preconditioner_Fourier_fnfnqks=green_c,
                                               input_nodal_field_fnxyz=x, output_nodal_field_fnxyz=Px)
    return M_c


def cg(disc, K, b, x, P):
    norms = solvers.conjugate_gradients_mugrid_experimental(
        comm=comm, fc=disc.field_collection, hessp=K, b=b, x=x, P=P, maxiter=20000,
        stop_crit_norm='energy_lower_estim', d_min=cg_delay,
        **topology_optimization.energy_stop_kwargs(disc, K, x, cg_tol))[1]
    return int(norms['stop_iteration']['energy_lower_estim'])


# ----------------------------------------------------------------------------
# Constitutive helpers
# ----------------------------------------------------------------------------
def update_degraded_material():
    disc_d.apply_N_operator_mugrid(damage, damage_q)
    g = (1.0 - damage_q.s[0, 0]) ** 2
    degraded_scale.s[0, 0] = (g + k_residual) * stiffness
    D = xp.where(particle > 0, D0_px * (g + k_diff), D0_px) if diff_degrade else D0_px
    diffusivity.s[0, 0] = m_px * D


def update_eigenstrain():
    """eigen_rhs = E - eps_Li = - beta (c - c0) I in the particle"""
    disc_d.apply_N_operator_mugrid(conc, conc_q)
    eps_li = beta_px * (conc_q.s[0, 0] - c_init)
    eigen_rhs.s[...] = 0.0
    for i in range(dim):
        eigen_rhs.s[i, i] = -eps_li


def tensile_energy(eps):
    trace = xp.einsum('ii...->...', eps)
    eps_dev = eps - trace / dim * xp.eye(dim).reshape(dim, dim, *([1] * trace.ndim))
    return 0.5 * bulk_qxyz * xp.maximum(trace, 0.0) ** 2 + mu_qxyz * xp.einsum('ij...,ij...->...', eps_dev, eps_dev)


def elastic_strain():
    disc_u.apply_gradient_operator_symmetrized_mugrid(u_inxyz=u_fluct, grad_u_ijqxyz=strain)
    strain.s[...] += eigen_rhs.s
    return strain.s


def crack_measures():
    """AT2 crack length Gamma (in the particle), the normalised crack area a_c = |{d > 0.95}| / |particle| (Tu eq. 33,
    pixel = mean of its quad points), and the broken areas of the shell, of the interface zone and of the core"""
    disc_d.apply_N_operator_mugrid(damage, damage_q)
    disc_d.apply_gradient_operator_mugrid(u_inxyz=damage, grad_u_ijqxyz=grad_d)
    density = damage_q.s[0, 0] ** 2 / (2 * length_scale) + 0.5 * length_scale * xp.sum(grad_d.s[0] ** 2, axis=0)
    gamma = float(xp.sum(weights_q * density * particle))
    broken = (damage_q.s[0, 0].mean(axis=0) > 0.95) * particle
    a_c = float(broken.sum() / particle.sum())
    near_interface = xp.abs(r_pix - R_core) < length_scale
    return gamma, a_c, float(broken[near_interface].sum() * area), float((broken * shell_px).sum() * area), \
        float((broken * core).sum() * area)


# ----------------------------------------------------------------------------
# Time stepping
# ----------------------------------------------------------------------------
iter_file = os.path.join(data_folder, f'battery_{run_tag}_iterations.csv')
step_file = os.path.join(data_folder, f'battery_{run_tag}_steps.csv')
snap_file = os.path.join(data_folder, f'battery_{run_tag}_snapshots.npz')
diff_kinds = ['Green', 'Green_Jacobi', 'dual']
checkpoint_file = os.path.join(data_folder, f'battery_{run_tag}_checkpoint.npz')
if not restart:
    with open(iter_file, 'w') as f:
        f.write('step,time,staggered_iteration,cg_u,cg_d,cg_c,' + ','.join(f'cg_c_{k}' for k in diff_kinds)
                + ',time_u,damage_change\n')
    with open(step_file, 'w') as f:
        f.write('step,time,stage,staggered_its,converged,d_max,gamma,a_c,broken_interface,broken_shell,broken_core,'
                'sol,sol_core,sol_shell,c_max_band,current,sigma_xx,sigma_yy,cg_u_mean,cg_c_mean,step_time\n')

conc.s[...] = c_init
update_degraded_material()
snapshots = {'time': [], 'damage': [], 'concentration': []}
start_time = time.time()
total_cg_u, nb_not_converged = 0, 0
stage = 'CC'
# CC -> CV pixel by pixel: a band pixel switches from the CC source q_cc to the CV (Robin) source KAPPA (1 - c) once
# its c reaches 1 - q_cc / KAPPA (continuous local current, c <= ~1); CV_SWITCH=cmax: the old global switch of the whole
# band at max c = 1 (gave a 1.5x - 9.5x current surge)
legacy_switch = os.environ.get('CV_SWITCH') == 'cmax'
cv_px = xp.zeros_like(surface_band)                      # 1: band pixel in CV
step = -1


def to_h(a):
    return np.asarray(tensor_operations.to_host(a))


def mid_slice(a):                                        # 3D: the mid-plane z = 1/2 for snapshots
    a = to_h(a)
    return a[..., N // 2] if dim == 3 else a


if restart:
    ck = np.load(checkpoint_file)
    step = int(ck['step'])
    for field, name in ((conc, 'conc'), (damage, 'damage'), (u_fluct, 'u_fluct'), (history_n, 'history_n')):
        field.s[...] = xp.asarray(ck[name])
    history.s[...] = history_n.s
    cv_px = xp.asarray(ck['cv_px'])
    stage = str(ck['stage'])
    update_degraded_material()
    for fname in (iter_file, step_file):                 # drop the rows after the checkpoint
        lines = open(fname).read().splitlines()
        with open(fname, 'w') as f:
            f.write('\n'.join([lines[0]] + [ln for ln in lines[1:] if int(ln.split(',')[0]) <= step]) + '\n')
    if os.path.exists(snap_file):
        sz = np.load(snap_file)
        keep = sz['time'] <= (step + 1) * dt + 1e-12
        snapshots = {k: list(sz[k][keep]) for k in ('time', 'damage', 'concentration')}
    print(f'restart from {checkpoint_file}: after step {step} ({stage})', flush=True)
last_checkpoint = time.time()
while True:
    step += 1
    t_now = (step + 1) * dt
    conc_old.s[...] = conc.s
    # CC: source q_cc in the band; CV: q = kappa (1 - c) in the band (the kappa part implicit, in mass_c)
    cc_px = surface_band * (1.0 - cv_px)
    mass_c.s[0, 0] = m_px / dt + kappa * cv_px
    disc_d.apply_N_operator_mugrid(conc_old, temp_q)
    temp_q.s[0, 0] = temp_q.s[0, 0] * m_px / dt + q_cc * cc_px + kappa * cv_px
    disc_d.apply_N_transposed_operator_mugrid(quad_field_ijqxyz=temp_q, nodal_field_inxyz=rhs_c, apply_weights=True)
    cg_u_step, cg_c_step = 0, 0
    step_start = time.time()
    for stagger in range(max_staggered):
        # 1. diffusion with the current D(d)
        if diff_shadow or diff_precond == 'Green_Jacobi':
            jacobi_c = disc_d.get_preconditioner_Jacobi_mugrid(operator=K_c, name='jacobi_c')
        conc_start.s[...] = conc.s
        cg_c = cg(disc_d, K_c, rhs_c, conc, make_M_c(diff_precond))
        cg_c_all = {k: '' for k in diff_kinds}
        cg_c_all[diff_precond] = cg_c
        if diff_shadow:
            for kind in diff_kinds:
                if kind != diff_precond:
                    conc_shadow.s[...] = conc_start.s
                    cg_c_all[kind] = cg(disc_d, K_c, rhs_c, conc_shadow, make_M_c(kind))
        # 2. elasticity with the Li strain of the current c
        t0 = time.time()
        update_eigenstrain()
        disc_u.get_rhs_mugrid(material_data_field_ijklqxyz=C_d, macro_gradient_field_ijqxyz=eigen_rhs, rhs_inxyz=rhs_u)
        if preconditioner_type == 'Green_Jacobi':
            jacobi_u = disc_u.get_preconditioner_Jacobi_mugrid(material_data_field_ijklqxyz=C_d,
                                                               formulation='small_strain', name='jacobi_u')
        cg_u = cg(disc_u, K_u, rhs_u, u_fluct, M_u)
        time_u = time.time() - t0
        total_cg_u += cg_u
        # 3. history and damage
        history.s[0, 0] = np.maximum(history_n.s[0, 0], crack_driving * tensile_energy(elastic_strain()))
        reaction.s[...] = Gc_px / length_scale + 2.0 * history.s
        temp_q.s[...] = 2.0 * history.s
        disc_d.apply_N_transposed_operator_mugrid(quad_field_ijqxyz=temp_q, nodal_field_inxyz=rhs_d,
                                                  apply_weights=True)
        damage_old.s[...] = damage.s
        jacobi_d = disc_d.get_preconditioner_Jacobi_mugrid(operator=K_d, name='jacobi_d')
        cg_d = cg(disc_d, K_d, rhs_d, damage, M_d)
        update_degraded_material()
        change = float(xp.max(xp.abs(damage.s - damage_old.s)))
        cg_u_step += cg_u
        cg_c_step += cg_c
        with open(iter_file, 'a') as f:
            f.write(f'{step},{t_now:.6g},{stagger + 1},{cg_u},{cg_d},{cg_c},'
                    + ','.join(str(cg_c_all[k]) for k in diff_kinds) + f',{time_u:.4g},{change:.4g}\n')
        if change < staggered_tol:
            break
    converged = change < staggered_tol
    nb_not_converged += not converged
    history_n.s[...] = history.s
    sigma = disc_u.get_homogenized_stress_mugrid(material_data_field_ijklqxyz=C_d, displacement_field_inxyz=u_fluct,
                                                 macro_gradient_field_ijqxyz=eigen_rhs, formulation='small_strain')
    gamma, a_c, b_int, b_shell, b_core = crack_measures()
    disc_d.apply_N_operator_mugrid(conc, conc_q)
    c_px = conc_q.s[0, 0].mean(axis=0)
    sol = float(xp.sum(m_px * c_px * particle) / xp.sum(m_px * particle))
    sol_core = float(xp.sum(c_px * core) / core.sum())
    sol_shell = float(xp.sum(c_px * shell_px) / shell_px.sum())
    c_max_band = float(c_px[surface_band > 0].max())
    current = float(q_cc * cc_px.sum() * area + kappa * xp.sum((1.0 - c_px) * cv_px) * area)
    with open(step_file, 'a') as f:
        f.write(f'{step},{t_now:.6g},{stage},{stagger + 1},{int(converged)},{float(damage.s.max()):.4g},{gamma:.5g},'
                f'{a_c:.5g},{b_int:.5g},{b_shell:.5g},{b_core:.5g},{sol:.5g},{sol_core:.5g},{sol_shell:.5g},'
                f'{c_max_band:.5g},{current / Q_cc:.4g},{float(sigma[0, 0]):.5g},{float(sigma[1, 1]):.5g},'
                f'{cg_u_step / (stagger + 1):.2f},{cg_c_step / (stagger + 1):.2f},{time.time() - step_start:.3g}\n')
    print(f'step {step:3d} t = {t_now:.4g} ({t_now * t_scale:.0f} s) {stage} SOL = {sol:.3f} (core {sol_core:.3f}, '
          f'shell {sol_shell:.3f}) max d = {float(damage.s.max()):.3f} a_c = {a_c:.4f} Gamma = {gamma:.4f} '
          f'stagger = {stagger + 1:3d} mean CG (u, c) = ({cg_u_step / (stagger + 1):.1f}, {cg_c_step / (stagger + 1):.1f}) '
          f'last CG (u, d, c) = ({cg_u}, {cg_d}, {cg_c}) shadow c = {cg_c_all} time = {time.time() - step_start:.2f} s'
          + ('' if converged else f'  MAX_STAGGERED HIT, step ACCEPTED (max |d - d_old| = {change:.2e})'), flush=True)
    finished = (stage == 'CV' and current < 0.1 * Q_cc) or t_now >= t_max - 1e-12
    if legacy_switch:
        if c_max_band >= 1.0:
            cv_px = surface_band.copy()
    else:
        cv_px = xp.maximum(cv_px, surface_band * (c_px >= 1.0 - q_cc / kappa))
    nb_cv = float(cv_px.sum())
    new_stage = 'CC' if nb_cv == 0 else ('CV' if nb_cv == float(surface_band.sum()) else 'MIX')
    if new_stage != stage:
        print(f'stage {stage} -> {new_stage} at t = {t_now:.4g}', flush=True)
        stage = new_stage
    if step % snap_every == snap_every - 1 or finished:
        snapshots['time'].append(t_now)
        snapshots['damage'].append(mid_slice(damage.s[0, 0]))
        snapshots['concentration'].append(mid_slice(conc.s[0, 0]))
        np.savez(snap_file, time=np.array(snapshots['time']), damage=np.array(snapshots['damage']),
                 concentration=np.array(snapshots['concentration']), particle=mid_slice(particle_h),
                 core=mid_slice(core_h), h=h, length_scale=l_ref, t_scale=t_scale, L_cell=L_cell)
    if checkpoint_seconds > 0 and time.time() - last_checkpoint > checkpoint_seconds and not finished:
        np.savez(checkpoint_file + '.tmp.npz', step=step, conc=to_h(conc.s), damage=to_h(damage.s),
                 u_fluct=to_h(u_fluct.s), history_n=to_h(history_n.s), cv_px=to_h(cv_px), stage=stage)
        os.replace(checkpoint_file + '.tmp.npz', checkpoint_file)
        last_checkpoint = time.time()
    if finished:
        print(f'end of lithiation at t = {t_now:.4g} ({t_now * t_scale:.0f} s): SOL = {sol:.3f}, a_c = {a_c:.4f}')
        break

print(f'Elapsed time: {time.time() - start_time:.1f} s')
print(f'{preconditioner_type}: total CG iterations elasticity = {total_cg_u}')
if nb_not_converged:
    print(f'WARNING: {nb_not_converged} steps hit MAX_STAGGERED = {max_staggered} and were accepted')
