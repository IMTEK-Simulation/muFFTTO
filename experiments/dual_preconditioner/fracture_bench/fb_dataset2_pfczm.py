"""
Fracture-Bench Dataset 2 (Krishnan et al. 2026, arXiv:2611.03473, Sec. 2.2.2 / Table 1) as a PERIODIC analogue, PF-CZM
=====================================================================================================================
Copy of ../localization_softening_bands/pfczm_concrete.py (Wu's PF-CZM, staggered scheme, projected Newton for the
damage, CG with the energy-error stop, Green / Green_Jacobi / dual, CPU or GPU, checkpoint / restart). Their parameters:
    plate 4 x 8 mm -> periodic cell 4 x 8 mm, GRID = NX,NY (their 128,256: h = 0.03125 mm), plane strain, small strain,
    matrix E = 377 GPa, nu = 0.3, Gf = 10 N/m, l0 = b = 0.04 mm, linear softening (a2 = -1/2, a3 = 0);
    inclusions d = 2 mm at (1.5, 6) mm (upper) and d = 3 mm at (2, 2) mm (lower), each hard (E, Gf x 10) or soft
    (x 0.1): INCL = '<upper>,<lower>' (default 'hard,soft'); all phases can crack;
    horizontal pre-crack from x = 0 to the tip CRACK_TIP = 'xc,yc' mm (default 1.5,4.2): d = 1 (irreversible) on the
    nodes within half a pixel of y = yc, x <= xc;
    load: uniaxial macro strain yy in increments 1e-4 mm / 8 mm = 1.25e-5 (their top displacement increment),
    LOAD_STEPS (default 80, theirs 31), stopped when the cell is cut through (STOP_FRACTION of the peak).
Not given by them: tensile strength FT_MPA (default l_ch = E Gf / ft^2 = 4 l0 -> 153.5 MPa; inclusions scaled with
E and Gf, same l_ch); residual stiffness K_RESIDUAL (default 1e-4).
Crack driving force DRIVER = 'spectral' (default; their Y = psi0+, principal strains, plane strain e_zz = 0) or 'wu'
(Wu's modified Rankine, fc/ft = 10), H = max(H_n, Y), H_n(0) = Y0 = ft^2 / (2E); full degradation omega(d) C.
Differences to theirs: periodic cell under a macroscopic strain (no clamped bottom / free sides), regular grid (no
adaptive refinement). Outputs as pfczm_concrete.py with the base name FB2_<NX>x<NY>_<P>_<incl>_tip..._k...
    python fb_dataset2_pfczm.py [Green|Green_Jacobi|dual]
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
dim = 2
NX, NY = (int(v) for v in os.environ.get('GRID', '128,256').split(','))
number_of_pixels = (NX, NY)
L_unit = 8.0                                       # length unit = plate height 8 mm
domain_size = [4.0 / L_unit, 1.0]
device = os.environ.get('DEVICE', 'cpu')
if device == 'gpu':
    import cupy as xp
else:
    xp = np
to_host = tensor_operations.to_host
E_unit = 377e3                                     # MPa: matrix E
E0, nu = 1.0, 0.3
Gf_phys = 0.01                                     # N/mm (10 N/m)
b_mm = float(os.environ.get('B_MM', 0.04))
ft_mpa = float(os.environ.get('FT_MPA', np.sqrt(E_unit * Gf_phys / (4.0 * b_mm))))
ft = ft_mpa / E_unit
Gf = Gf_phys / (E_unit * L_unit)
b = b_mm / L_unit
beta_c = 10.0 - 1.0                                # DRIVER=wu only
driver = os.environ.get('DRIVER', 'spectral')
incl = os.environ.get('INCL', 'hard,soft').split(',')
FAC = {'hard': 10.0, 'soft': 0.1}
xc_mm, yc_mm = (float(v) for v in os.environ.get('CRACK_TIP', '1.5,4.2').split(','))
law = os.environ.get('SOFTENING', 'linear')
p, a2, a3 = {'linear': (2.0, -0.5, 0.0), 'exponential': (2.5, 2 ** (5 / 3) - 3, 0.0),
             'cornelissen': (2.0, 1.3868, 0.6567)}[law]
k_residual = float(os.environ.get('K_RESIDUAL', 1e-4))
load_dir = 'yy'
macro_strain_direction = np.zeros((dim, dim))
ic = (1, 1)
macro_strain_direction[ic] = 1.0
load_steps = (1e-4 / L_unit) * np.arange(1, int(os.environ.get('LOAD_STEPS', 80)) + 1)
staggered_tol = float(os.environ.get('STAGGERED_TOL', 1e-3))
max_staggered_iterations = int(os.environ.get('MAX_STAGGERED', 500))
stop_fraction = float(os.environ.get('STOP_FRACTION', 0.02))
stop_steps = int(os.environ.get('STOP_STEPS', 3))
time_budget = float(os.environ.get('TIME_BUDGET', 1e12))
checkpoint_every = int(os.environ.get('CHECKPOINT_EVERY', 10))
frame_every = int(os.environ.get('FRAME_EVERY', 1 if NY <= 512 else 2))
full_every = 1
newton_tol = 1e-6
max_newton = 50
cg_tol = 1e-6
cg_delay = 10
preconditioner_type = sys.argv[1] if len(sys.argv) > 1 else 'dual'
assert preconditioner_type in ('Green', 'Green_Jacobi', 'dual')
base_name = (f'FB2_{NX}x{NY}_{preconditioner_type}_{incl[0]}-{incl[1]}_tip{xc_mm:g}-{yc_mm:g}_{law}_{driver}'
             f'_ft{ft_mpa:.0f}_k{k_residual:g}' + os.environ.get('RUN_SUFFIX', ''))
out_dir = os.path.join(here, 'exp_data')
iterations_file = os.path.join(out_dir, base_name + '_iterations.csv')
steps_file = os.path.join(out_dir, base_name + '_steps.csv')
frames_dir = os.path.join(out_dir, base_name + '_frames')
checkpoint_file = os.path.join(out_dir, base_name + '_checkpoint.npz')
restart_from = os.environ.get('RESTART_FROM')

# ----------------------------------------------------------------------------
# Geometry: two inclusions (factor FAC on E, Gf, ft), pixel centres; pre-crack nodes
# ----------------------------------------------------------------------------
h_mm = L_unit / NY                                 # = 4 mm / NX
xn = np.arange(NX)[:, None] * h_mm * np.ones((1, NY))          # node = lower-left pixel corner, mm
yn = np.ones((NX, 1)) * np.arange(NY)[None, :] * h_mm
fac_host = np.ones((NX, NY))
for (cx, cy, r), kind in zip(((1.5, 6.0, 1.0), (2.0, 2.0, 1.5)), incl):
    fac_host[(xn + 0.5 * h_mm - cx) ** 2 + (yn + 0.5 * h_mm - cy) ** 2 <= r ** 2] = FAC[kind]
precrack_host = (np.abs(yn - yc_mm) < 0.5 * h_mm) & (xn <= xc_mm + 1e-9)
os.makedirs(out_dir, exist_ok=True)
np.save(os.path.join(out_dir, base_name + '_phase.npy'), fac_host.astype(np.float32))
print(f'grid {number_of_pixels}, {device}, plate 4 x 8 mm, h = {h_mm * 1e3:.2f} um, b = {b_mm:g} mm (b/h = {b_mm / h_mm:.2f}), '
      f'ft = {ft_mpa:.1f} MPa, l_ch = {E0 * Gf / ft ** 2 * L_unit:.3f} mm, {law}, k = {k_residual}, driver {driver}, '
      f'inclusions upper {incl[0]} / lower {incl[1]}, pre-crack tip ({xc_mm}, {yc_mm}) mm ({int(precrack_host.sum())} nodes), '
      f'load yy to {load_steps[-1]:.4g} in {len(load_steps)} steps, {preconditioner_type}', flush=True)

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

fac = xp.asarray(fac_host)
mortar = xp.asarray(np.where(fac_host == 1.0, 1.0, 0.0))     # 1 matrix, 0 inclusion (statistics only)
stiffness = fac
E_x = stiffness * E0
ft_x = ft * fac
Gf_x = Gf * fac
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
H_n = xp.broadcast_to(ft_x ** 2 / (2 * E_x), (nq, NX, NY)).copy()   # history, starts at Y0
H = H_n.copy()
d_n = xp.asarray(np.where(precrack_host, 1.0, 0.0))
hess_q = None
active_free = xp.ones((NX, NY))


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
    tr = e[0, 0] + e[1, 1]
    if driver == 'spectral':                       # psi0+ = lam/2 <tr e>+^2 + mu sum <e_i>+^2 (plane strain, e_zz = 0)
        r = xp.sqrt(0.25 * (e[0, 0] - e[1, 1]) ** 2 + e[0, 1] ** 2)
        e1, e2 = 0.5 * tr + r, 0.5 * tr - r
        return (0.5 * lam * xp.maximum(tr, 0.0) ** 2 + mu * (xp.maximum(e1, 0.0) ** 2 + xp.maximum(e2, 0.0) ** 2)) * stiffness
    sxx = (lam * tr + 2 * mu * e[0, 0]) * stiffness
    syy = (lam * tr + 2 * mu * e[1, 1]) * stiffness
    sxy = 2 * mu * e[0, 1] * stiffness
    szz = lam * tr * stiffness
    s1 = xp.maximum(0.5 * (sxx + syy) + xp.sqrt(0.25 * (sxx - syy) ** 2 + sxy ** 2), szz)
    mean = (sxx + syy + szz) / 3
    J2 = 0.5 * ((sxx - mean) ** 2 + (syy - mean) ** 2 + (szz - mean) ** 2) + sxy ** 2
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
    damage.s[0, 0] = d_n
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
