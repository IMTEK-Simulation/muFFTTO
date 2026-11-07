"""
Phase-field regularised cohesive zone model (PF-CZM, Wu 2017 JMPS 103:72-99) in a periodic concrete meso-cell
=============================================================================================================

Same discretizations, elastic preconditioners (Green / Green_Jacobi / dual), CG with the energy-error stop and
staggered scheme as example_2D_gradient_damage_localization.py / the AT2 script, with Wu's PF-CZM:

    Pi(u, d) = int  omega(d) psi0(eps) dx + Gf / (c0 b) int  alpha(d) + b^2 |grad d|^2 dx,        c0 = pi
    alpha(d) = 2 d - d^2                                         (xi = 2)
    omega(d) = (1-d)^p / ((1-d)^p + a1 d (1 + a2 d + a2 a3 d^2)),   a1 = 4 l_ch / (pi b),  l_ch = E0 Gf / ft^2
        linear:      p = 2,   a2 = -1/2,            a3 = 0
        exponential: p = 5/2, a2 = 2^(5/3) - 3,     a3 = 0
        cornelissen: p = 2,   a2 = 1.3868,          a3 = 0.6567     (default)

Elastic problem (linear-symmetric per staggered iteration): s = stiffness(x) [omega(d) + k] in the mortar,
s = stiffness (never degraded) in the aggregates.
Crack driving force (Wu eq. 2.28, modified for tension):  Y = sigma_eq^2 / (2 E0),
    sigma_eq = (beta_c <sigma1> + sqrt(3 J2)) / (1 + beta_c),  beta_c = fc/ft - 1 = 9,  effective stress sigma = C0:eps
    (plane strain, sigma_zz = lam tr eps),   H = max(Y0, H_n, Y),  Y0 = ft^2 / (2 E0)
Damage: minimise, for fixed H and d_n <= d <= 1 (irreversibility + bound),
    Pi_d(d) = int_mortar  Gf/(pi b) (2 d - d^2 + b^2 |grad d|^2) + omega(d) H  dx
with a projected Newton method (active set on the bounds, the reduced Newton system by PCG with the Green-Jacobi
preconditioner of the homogeneous operator 2 Gf b/pi B^T B + 2 Gf/(pi b) N^T N; the indefinite local Hessian
omega'' H - 2 Gf/(pi b) is clipped from below at 0.1 * 2 Gf/(pi b) (convexified Newton), energy-decrease backtracking).
Matrix: Wu's concrete (E0 = 30 GPa, nu = 0.2, ft = 3 MPa, Gf = 0.12 N/mm, b = 5 mm, fc/ft = 10). Aggregates
(AGG_DAMAGE=1): E = 70 GPa, nu = 0.2, ft and Gf AGG_STRENGTH (2) x the matrix values, their own a1, Y0 -- they
can crack, at twice the strength. AGG_DAMAGE=0: tough aggregates (damage functional on the matrix pixels only,
d = 0 at the nodes inside aggregates); tested: the band (width ~ pi b) then bridges on aggregates and the
stress never drops below ~20 % of the peak.
Geometry: ../asr_gel_expansion/geometry/asr_meso_s1.npy (gel pockets -> matrix) scaled to the cell CELL_MM = 80 mm
(aggregate diameters 6.4-16 mm, 44 %), or GEOMETRY=disks (polydisperse_pores, 25 %).
Units: length = cell size, stress = 30 GPa.
"""
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

import numpy as np
import time

from muFFTTO import domain
from muFFTTO import microstructure_library
from muFFTTO import solvers
from muFFTTO import topology_optimization   # energy_stop_kwargs

here = os.path.dirname(os.path.abspath(__file__))
# ----------------------------------------------------------------------------
# Parameters (env)
# ----------------------------------------------------------------------------
N = int(os.environ.get('NPIX', 64))
number_of_pixels = (N, N)
domain_size = [1.0, 1.0]
# Wu 2017 concrete (Sec. 4.2 bar, L-panel): E0 = 30 GPa, nu = 0.2, ft = 3 MPa, Gf = 0.12 N/mm, b = 5 mm.
# Cell CELL_MM (default 80 mm = 16 b, so b/h = 2, 4, 8 at 32^2, 64^2, 128^2)
L_unit = float(os.environ.get('CELL_MM', 80.0))       # mm
E_unit = 30e3                                          # MPa
b_mm = float(os.environ.get('B_MM', 5.0))
E0, nu = 1.0, 0.2                                  # mortar / concrete matrix (Wu)
ft = 3.0 / E_unit
Gf = 0.12 / (E_unit * L_unit)
fc_ft = 10.0
beta_c = fc_ft - 1.0
b = b_mm / L_unit                                  # internal length (fixed across the grids)
# aggregates (AGG_DAMAGE=1, default): E 70 GPa, nu 0.2, ft and Gf AGG_STRENGTH x mortar (default 2: 6 MPa, 0.24 N/mm),
# damageable with their own PF-CZM parameters; AGG_DAMAGE=0: tough (no damage field, d = 0 inside)
agg_E = float(os.environ.get('AGG_E_GPA', 70.0)) * 1e3 / E_unit
agg_strength = float(os.environ.get('AGG_STRENGTH', 2.0))
agg_damage = os.environ.get('AGG_DAMAGE', '1') == '1'
law = os.environ.get('SOFTENING', 'cornelissen')
p, a2, a3 = {'linear': (2.0, -0.5, 0.0), 'exponential': (2.5, 2 ** (5 / 3) - 3, 0.0),
             'cornelissen': (2.0, 1.3868, 0.6567)}[law]
k_residual = float(os.environ.get('K_RESIDUAL', 1e-3))
agg_stiffness = agg_E / E0
geometry = os.environ.get('GEOMETRY', 'asr')       # 'asr' (../asr_gel_expansion/geometry/asr_meso_s1.npy, gel = mortar)
#                                                   or 'disks' (polydisperse_pores, 25 %)
load_dir = os.environ.get('LOAD_DIR', 'xx')
macro_strain_direction = np.zeros((2, 2))
if load_dir == 'xy':
    macro_strain_direction[0, 1] = macro_strain_direction[1, 0] = 0.5        # load = gamma
    ic = (0, 1)
else:
    ic = (0, 0) if load_dir == 'xx' else (1, 1)
    macro_strain_direction[ic] = 1.0
load_steps = np.linspace(0.0, float(os.environ.get('LOAD_MAX', 3e-3)), int(os.environ.get('LOAD_STEPS', 240)) + 1)[1:]
staggered_tol = float(os.environ.get('STAGGERED_TOL', 1e-3))
max_staggered_iterations = int(os.environ.get('MAX_STAGGERED', 500))
stop_fraction = float(os.environ.get('STOP_FRACTION', 0.02))
stop_steps = int(os.environ.get('STOP_STEPS', 3))
newton_tol = 1e-6                                  # max |d_new - d| of a Newton step
max_newton = 50
cg_tol = 1e-6
cg_delay = 10
preconditioner_type = sys.argv[1] if len(sys.argv) > 1 else 'dual'
assert preconditioner_type in ('Green', 'Green_Jacobi', 'dual')
run_tag = f'_{geometry}_{load_dir}_{law}_b{b_mm:g}mm_k{k_residual:g}' + ('' if agg_damage else '_aggtough') + os.environ.get('RUN_SUFFIX', '')
base_name = f'CZM_N{N}_{preconditioner_type}{run_tag}'
iterations_file = os.path.join(here, 'exp_data', base_name + '_iterations.csv')
steps_file = os.path.join(here, 'exp_data', base_name + '_steps.csv')
fields_file = os.path.join(here, 'exp_data', base_name + '_fields.npz')

# ----------------------------------------------------------------------------
# Discretizations
# ----------------------------------------------------------------------------
cell_u = domain.PeriodicUnitCell(domain_size=domain_size, problem_type='elasticity')
disc_u = domain.Discretization(cell=cell_u, nb_of_pixels_global=number_of_pixels,
                               discretization_type='finite_element', element_type='bilinear_rectangle')
cell_d = domain.PeriodicUnitCell(domain_size=domain_size, problem_type='conductivity')
disc_d = domain.Discretization(cell=cell_d, nb_of_pixels_global=number_of_pixels,
                               discretization_type='finite_element', element_type='bilinear_rectangle')
comm = disc_u.communicator
dim = 2
assert comm.size == 1, 'serial only'
wq = np.asarray(disc_d.quadrature_weights)          # physical weights per pixel, shape (q,)
print(f'grid {number_of_pixels}, cell {L_unit:g} mm, b = {b_mm:g} mm (b/h = {b * N:.2f}), l_ch = {E0 * Gf / ft ** 2 * L_unit:.0f} mm, '
      f'{law} (p = {p}, a2 = {a2:.4f}, a3 = {a3}), k = {k_residual}, aggregates {agg_stiffness}x, '
      f'geometry {geometry}, load {load_dir} to {load_steps[-1]:g}, {preconditioner_type}')

# ----------------------------------------------------------------------------
# Microstructure: mortar (1, cracks), aggregates (0, stiff, tough)
# ----------------------------------------------------------------------------
xy = np.asarray(disc_u.fft.coords)
if geometry == 'asr':
    fine = np.load(os.path.join(here, '..', 'asr_gel_expansion', 'geometry', 'asr_meso_s1.npy'))
    idx = [np.clip(np.floor(xy[a] * fine.shape[a] + 1e-9).astype(int), 0, fine.shape[a] - 1) for a in range(2)]
    mortar = np.where(fine[tuple(idx)] == 1, 0.0, 1.0)              # 0 paste, 1 aggregate, 2 gel (-> paste)
else:
    mortar = np.where(np.asarray(microstructure_library.get_geometry(
        nb_voxels=disc_u.nb_of_pixels, microstructure_name='polydisperse_pores', coordinates=disc_u.fft.coords)) > 0,
        1.0, 0.0)
stiffness = np.where(mortar > 0, 1.0, agg_stiffness)
agg = mortar == 0
# nodes inside aggregates (all four adjacent pixels aggregate; a node is the lower-left corner of its pixel)
fixed_node = agg & np.roll(agg, 1, 0) & np.roll(agg, 1, 1) & np.roll(np.roll(agg, 1, 0), 1, 1)
free_node = np.where(fixed_node & (not agg_damage), 0.0, 1.0)
damaged = np.ones((N, N)) if agg_damage else mortar.copy()   # pixels carrying the damage functional
# per-pixel PF-CZM parameters (aggregates: ft, Gf x agg_strength, E = agg_E)
E_x = np.where(mortar > 0, E0, agg_E)
ft_x = np.where(mortar > 0, ft, agg_strength * ft)
Gf_x = np.where(mortar > 0, Gf, agg_strength * Gf)
a1 = 4.0 * (E_x * Gf_x / ft_x ** 2) / (np.pi * b)     # a1 = 4 l_ch / (pi b), per pixel

lam = E0 * nu / ((1 + nu) * (1 - 2 * nu))            # plane strain
mu = E0 / (2 * (1 + nu))
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
H_n = np.broadcast_to(ft_x ** 2 / (2 * E_x), (nq, N, N)).copy()   # history, starts at Y0 = ft^2 / (2 E)
H = H_n.copy()
d_n = np.zeros((N, N))                              # nodal damage of the last load step (lower bound)
hess_q = np.zeros((nq, N, N))                       # convexified local Hessian
active_free = free_node.copy()                      # free (inactive) nodes of the current Newton step

c_diff = 2.0 * Gf_x * b / np.pi                     # per pixel: Hessian of Gf/(pi b) b^2 |grad d|^2
c_loc = 2.0 * Gf_x / (np.pi * b)                    # per pixel: alpha'' = -2 -> local Hessian - c_loc
c_diff_ref, c_loc_ref = 2.0 * Gf * b / np.pi, 2.0 * Gf / (np.pi * b)   # mortar values: Green reference


# ----------------------------------------------------------------------------
# Constitutive functions
# ----------------------------------------------------------------------------
def omega_and_derivatives(d):
    d = np.clip(d, 0.0, 1.0)
    A = (1.0 - d) ** p
    A1 = -p * (1.0 - d) ** (p - 1)
    A2 = p * (p - 1) * (1.0 - d) ** (p - 2) if p != 2 else 2.0 * np.ones_like(d)
    Q = a1 * (d + a2 * d ** 2 + a2 * a3 * d ** 3)
    Q1 = a1 * (1 + 2 * a2 * d + 3 * a2 * a3 * d ** 2)
    Q2 = a1 * (2 * a2 + 6 * a2 * a3 * d)
    S = A + Q
    num = A1 * Q - A * Q1
    w = A / S
    w1 = num / S ** 2
    w2 = ((A2 * Q - A * Q2) * S ** 2 - num * 2 * S * (A1 + Q1)) / S ** 4
    return w, w1, w2


def update_material():
    disc_d.apply_N_operator_mugrid(damage, damage_q)
    w = omega_and_derivatives(damage_q.s[0, 0])[0]
    degraded_scale.s[0, 0] = np.where(damaged > 0, (w + k_residual) * stiffness, stiffness)


def effective_stress_Y():
    disc_u.apply_gradient_operator_symmetrized_mugrid(u_inxyz=u_fluct, grad_u_ijqxyz=strain)
    strain.s[...] += macro_strain.s
    e = strain.s                                     # [2, 2, q, x, y]
    tr = e[0, 0] + e[1, 1]
    sxx = (lam * tr + 2 * mu * e[0, 0]) * stiffness   # effective stress of the phase, C_phase = stiffness C_iso
    syy = (lam * tr + 2 * mu * e[1, 1]) * stiffness
    sxy = 2 * mu * e[0, 1] * stiffness
    szz = lam * tr * stiffness
    s1_plane = 0.5 * (sxx + syy) + np.sqrt(0.25 * (sxx - syy) ** 2 + sxy ** 2)
    s1 = np.maximum(s1_plane, szz)
    mean = (sxx + syy + szz) / 3
    J2 = 0.5 * ((sxx - mean) ** 2 + (syy - mean) ** 2 + (szz - mean) ** 2) + sxy ** 2
    seq = (beta_c * np.maximum(s1, 0.0) + np.sqrt(3 * J2)) / (1 + beta_c)
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
# Damage problem: energy, gradient, (convexified) Hessian on the mortar pixels
# ----------------------------------------------------------------------------
def apply_diffusion_mass(x, Ax, local_q, domain_q, cd=None):
    """[c_diff B^T W m B + N^T W m local N] x"""
    disc_d.apply_gradient_operator_mugrid(u_inxyz=x, grad_u_ijqxyz=grad_d)
    grad_d.s[...] *= (c_diff if cd is None else cd) * (1.0 if domain_q is None else domain_q)
    disc_d.apply_gradient_transposed_operator_mugrid(gradient_field_ijqxyz=grad_d, div_u_fnxyz=Ax, apply_weights=True)
    disc_d.apply_N_operator_mugrid(x, temp_q)
    temp_q.s[0, 0] *= local_q if domain_q is None else local_q * domain_q
    disc_d.apply_N_transposed_operator_mugrid(quad_field_ijqxyz=temp_q, nodal_field_inxyz=temp_n, apply_weights=True)
    Ax.s[...] += temp_n.s


def energy(d_field):
    disc_d.apply_N_operator_mugrid(d_field, damage_q)
    dq = damage_q.s[0, 0]
    w = omega_and_derivatives(dq)[0]
    disc_d.apply_gradient_operator_mugrid(u_inxyz=d_field, grad_u_ijqxyz=grad_d)
    g2 = np.sum(grad_d.s[0] ** 2, axis=0)
    dens = (Gf_x / (np.pi * b)) * (2 * dq - dq ** 2 + b ** 2 * g2) + w * H
    return float(np.sum(wq[:, None, None] * dens * damaged))


def energy_gradient():
    """g = c_diff B^T W m B d + N^T W m [c_loc (1 - d) + omega'(d) H]; sets the Hessian coefficient hess_q"""
    global hess_q
    disc_d.apply_N_operator_mugrid(damage, damage_q)
    dq = damage_q.s[0, 0]
    _, w1, w2 = omega_and_derivatives(dq)
    hess_q = np.maximum(w2 * H - c_loc, 0.1 * c_loc)
    apply_diffusion_mass(damage, grad_n, 0.0 * dq, damaged)       # diffusion part only
    temp_q.s[0, 0] = (c_loc * (1 - dq) + w1 * H) * damaged
    disc_d.apply_N_transposed_operator_mugrid(quad_field_ijqxyz=temp_q, nodal_field_inxyz=temp_n, apply_weights=True)
    grad_n.s[...] += temp_n.s


def K_newton(x, Ax):
    masked.s[0, 0] = active_free * x.s[0, 0]
    apply_diffusion_mass(masked, Ax, hess_q, damaged)
    Ax.s[0, 0] = active_free * Ax.s[0, 0] + (1.0 - active_free) * x.s[0, 0]


green_d = disc_d.get_preconditioner_Green_mugrid(
    reference_material_data_ijkl=None, invert_zero_mode=True, name='green_damage',
    operator=lambda input_field_inxyz, output_field_inxyz: apply_diffusion_mass(
        input_field_inxyz, output_field_inxyz, c_loc_ref, None, c_diff_ref))
jacobi_d = None


def M_d(x, Px):
    x_scaled_d.s[0, 0] = active_free * jacobi_d.s[0, 0] * x.s[0, 0]
    disc_d.apply_preconditioner_mugrid(preconditioner_Fourier_fnfnqks=green_d, input_nodal_field_fnxyz=x_scaled_d,
                                       output_nodal_field_fnxyz=Px)
    Px.s[0, 0] = active_free * jacobi_d.s[0, 0] * Px.s[0, 0] + (1.0 - active_free) * x.s[0, 0]


def solve_damage():
    """projected Newton for min Pi_d on d_n <= d <= 1 (fixed nodes d = 0); returns (Newton its, CG its)"""
    global active_free, jacobi_d
    lb, ub = d_n, 1.0
    nb_cg = 0
    E_cur = energy(damage)
    for it in range(max_newton):
        energy_gradient()
        d = damage.s[0, 0]
        g = grad_n.s[0, 0]
        at_lb = (d <= lb + 1e-12) & (g > 0)
        at_ub = (d >= ub - 1e-12) & (g < 0)
        active_free = free_node * np.where(at_lb | at_ub, 0.0, 1.0)
        rhs_d.s[0, 0] = -active_free * g
        if np.max(np.abs(rhs_d.s)) == 0.0:
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
        for _ in range(30):                          # backtracking on the energy (projected step)
            trial.s[0, 0] = free_node * np.clip(d + t * active_free * delta.s[0, 0], lb, ub)
            E_trial = energy(trial)
            if E_trial <= E_cur + 1e-4 * float(np.sum(g * (trial.s[0, 0] - d))) or t < 1e-6:
                break
            t *= 0.5
        step = float(np.max(np.abs(trial.s[0, 0] - d)))
        damage.s[...] = trial.s
        E_cur = E_trial
        if step < newton_tol:
            return it + 1, nb_cg
    return max_newton, nb_cg


# ----------------------------------------------------------------------------
# Load stepping
# ----------------------------------------------------------------------------
os.makedirs(os.path.join(here, 'exp_data'), exist_ok=True)
with open(iterations_file, 'w') as f:
    f.write('step,load,staggered_iteration,cg_u,time_u,newton_d,cg_d,time_d,damage_change,max_d,min_s_matrix\n')
with open(steps_file, 'w') as f:
    f.write('step,load,stress,max_d,min_s_matrix,crack_fraction_0.9,staggered_its,converged,cg_u_step,mean_cg_u,'
            'time_u_step,time_step\n')
update_material()
start_time = time.time()
total_cg_u, nb_not_converged, nb_broken = 0, 0, 0
macro_stresses = [0.0]
d_frames, frame_loads = [], []
area_mortar = mortar.sum()
for step, load in enumerate(load_steps):
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
        H = np.maximum(H_n, effective_stress_Y()) * damaged + H_n * (1 - damaged)
        d_old = damage.s[0, 0].copy()
        newton_its, cg_d = solve_damage()
        update_material()
        time_d = time.time() - t0
        change = float(np.max(np.abs(damage.s[0, 0] - d_old)))
        s_min = float(np.min(degraded_scale.s[0, 0][:, mortar > 0]))
        cg_u_step += cg_u
        time_u_step += time_u
        total_cg_u += cg_u
        with open(iterations_file, 'a') as f:
            f.write(f'{step},{load:.6g},{stagger + 1},{cg_u},{time_u:.4g},{newton_its},{cg_d},{time_d:.4g},'
                    f'{change:.4g},{float(damage.s.max()):.6g},{s_min:.4g}\n')
        if change < staggered_tol:
            break
    converged = change < staggered_tol
    nb_not_converged += not converged
    H_n = H.copy()
    d_n = damage.s[0, 0].copy()
    sigma = disc_u.get_homogenized_stress_mugrid(material_data_field_ijklqxyz=C_d, displacement_field_inxyz=u_fluct,
                                                 macro_gradient_field_ijqxyz=macro_strain, formulation='small_strain')
    crack = float(((d_n > 0.9) & (mortar > 0)).sum() / area_mortar)
    macro_stresses.append(float(sigma[ic]))
    d_frames.append(d_n.astype(np.float32))
    frame_loads.append(load)
    with open(steps_file, 'a') as f:
        f.write(f'{step},{load:.6g},{sigma[ic]:.6g},{float(d_n.max()):.6g},{s_min:.4g},{crack:.4g},{stagger + 1},'
                f'{int(converged)},{cg_u_step},{cg_u_step / (stagger + 1):.2f},{time_u_step:.3f},'
                f'{time.time() - step_start:.3f}\n')
    print(f'step {step:3d}  load = {load:.4e}  sigma = {sigma[ic] * E_unit:.4f} MPa  max d = {float(d_n.max()):.4f}  '
          f'min s = {s_min:.2e}  crack(d>0.9) = {crack:.3f}  staggered its = {stagger + 1:3d}  '
          f'step CG u = {cg_u_step}  mean CG u = {cg_u_step / (stagger + 1):.1f}  '
          f'time (u, step) = ({time_u_step:.2f}, {time.time() - step_start:.2f}) s'
          + ('' if converged else f'  ACCEPTED NOT CONVERGED after MAX_STAGGERED = {max_staggered_iterations} '
                                  f'(max |d - d_old| = {change:.2e})'), flush=True)
    nb_broken = nb_broken + 1 if sigma[ic] < stop_fraction * max(macro_stresses) else 0
    if stop_fraction > 0 and nb_broken >= stop_steps:
        print(f'stress < {stop_fraction:g} of the peak for {stop_steps} steps: stop')
        break

np.savez_compressed(fields_file, d=np.array(d_frames), load=np.array(frame_loads), phase=mortar,
                    stress=np.array(macro_stresses[1:]), length_scale=b, cell_mm=L_unit, k_residual=k_residual, E_unit=E_unit)
print(f'Elapsed time: {time.time() - start_time:.1f} s')
print(f'{preconditioner_type}: total CG iterations elasticity = {total_cg_u}')
if nb_not_converged:
    print(f'WARNING: {nb_not_converged} load steps accepted NOT CONVERGED (MAX_STAGGERED = {max_staggered_iterations})')
