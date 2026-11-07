"""
ASR gel expansion with AT2 phase field, 2D and 3D, CPU or GPU: mesh-size study at fixed physical l
==================================================================================================

Generalisation of asr_AT2_gel_expansion.py (same AT2 model: hybrid C_d = g(d) C_0, Amor split on the elastic strain
eps - eps_g, history field, staggered elasticity / damage with max|d - d_old| < 1e-3, damage CG with Green-Jacobi,
CG energy stopping 1e-6 / delay 10; elastic preconditioner Green / Green_Jacobi / dual from argv[1]).

* Cell 0.1 m (length unit), periodic; Falsafi et al. Table 2 materials (paste, aggregate, gel = linear, never
  degraded, no damage driving), plane strain in 2D. Stress unit E_paste = 12 GPa.
* Geometry from GEOMETRY_NPZ: 2D geometry/falsafi_s1.npz (ellipses (cx, cy, a, b, angle)), 3D geometry/asr3D_s1.npz
  (ellipsoids: centre, semi-axes, rotation); evaluated at the pixel lower-left nodes of every grid. Gel pockets: the
  cell of the GEL_CELLS^d grid (default 64: 1.5625 mm) containing each site; on coarser grids the single pixel
  containing the site, eigenstrain x (N / GEL_CELLS)^d (same expansion volume), Voigt gel/aggregate stiffness.
* Load: gel eigenstrain eps_g I, dV/V = dim eps_g (small strain) from 0 to DVV_MAX in LOAD_STEPS steps.
* Zero mean stress, CONTROL (env):
    'exact': superposition, dim(dim+1)/2 unit-strain solves + 1 eigenstrain solve per staggered iteration
    'cheap' (default): one solve per staggered iteration with the current E, then E <- E - C_eff^-1 <sigma> with
             C_eff from the unit solves at the end of the previous load step; a step converges when also
             |<sigma>| < TOL_S |C_eff E|
* Output (exp_data/): <run>_iterations.csv (every staggered iteration), <run>_steps.npz (per step: dV/V, E,
  stiffness ratio ||C_eff||/||C_eff,0||, max/mean d, CG means, times), <run>_damage/step_<k>.npy (float32: 2D full
  field every step; 3D mid-plane z = 1/2 every step + full field every SNAP_FULL steps), <run>_checkpoint.npz
  (every CHECKPOINT_SECONDS at the end of a load step; RESTART_FROM=<checkpoint> continues).
* MAX_STAGGERED (default 500): a step that hits it is accepted and logged.

    python asr_AT2_mesh_study.py <Green|Green_Jacobi|dual>
env: DIM (2), NPIX, DEVICE (cpu|gpu), LENGTH_SCALE (1/64 = 1.5625 mm), GEOMETRY_NPZ, GEL_CELLS (64), DVV_MAX (0.064),
     LOAD_STEPS (64), CONTROL (cheap), TOL_S (1e-4), MAX_STAGGERED (500), K_RESIDUAL (1e-4), SNAP_FULL (16),
     CHECKPOINT_SECONDS (3600), RESTART_FROM, RUN_SUFFIX
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
dim = int(os.environ.get('DIM', 2))
N = int(os.environ.get('NPIX', 64))
device = os.environ.get('DEVICE', 'cpu')
if device == 'gpu':
    import cupy as xp
else:
    xp = np
to_host = tensor_operations.to_host
length_scale = float(os.environ.get('LENGTH_SCALE', 1.0 / 64))
geometry_npz = os.environ.get('GEOMETRY_NPZ', 'geometry/falsafi_s1.npz' if dim == 2 else 'geometry/asr3D_s1.npz')
gel_cells = int(os.environ.get('GEL_CELLS', 64))
k_residual = float(os.environ.get('K_RESIDUAL', 1e-4))
E_unit, L_unit = 12e9, 0.1
PHASES = {0: dict(E=12e9 / E_unit, nu=0.30, Gc=60.0 / (E_unit * L_unit)),     # cement paste
          1: dict(E=59e9 / E_unit, nu=0.30, Gc=160.0 / (E_unit * L_unit)),    # aggregate
          2: dict(E=11e9 / E_unit, nu=0.18, Gc=160.0 / (E_unit * L_unit))}    # ASR gel (never degraded, no driving)
dvv_max = float(os.environ.get('DVV_MAX', 0.064))
load_steps = np.linspace(0.0, dvv_max, int(os.environ.get('LOAD_STEPS', 64)) + 1)[1:]
staggered_tol = 1e-3
max_staggered = int(os.environ.get('MAX_STAGGERED', 500))
control = os.environ.get('CONTROL', 'cheap')
tol_s = float(os.environ.get('TOL_S', 1e-4))
snap_full = int(os.environ.get('SNAP_FULL', 16))
checkpoint_seconds = float(os.environ.get('CHECKPOINT_SECONDS', 3600))
cg_tol, cg_delay = 1e-6, 10
preconditioner_type = sys.argv[1] if len(sys.argv) > 1 else 'dual'
assert preconditioner_type in ('Green', 'Green_Jacobi', 'dual') and control in ('exact', 'cheap')
run_name = (f'ASRmesh_{dim}D_N{N}_{preconditioner_type}_l{length_scale:g}_{control}'
            + os.environ.get('RUN_SUFFIX', ''))
out_dir = os.path.join(here, 'exp_data')
iterations_file = os.path.join(out_dir, run_name + '_iterations.csv')
steps_file = os.path.join(out_dir, run_name + '_steps.npz')
snap_dir = os.path.join(out_dir, run_name + '_damage')
checkpoint_file = os.path.join(out_dir, run_name + '_checkpoint.npz')


def plane_strain_lame(E, nu):
    return E * nu / ((1 + nu) * (1 - 2 * nu)), E / (2 * (1 + nu))


element = 'bilinear_rectangle' if dim == 2 else 'trilinear_hexahedron'
cell_u = domain.PeriodicUnitCell(domain_size=[1.0] * dim, problem_type='elasticity')
disc_u = domain.Discretization(cell=cell_u, nb_of_pixels_global=(N,) * dim, discretization_type='finite_element',
                               element_type=element, device=device)
cell_d = domain.PeriodicUnitCell(domain_size=[1.0] * dim, problem_type='conductivity')
disc_d = domain.Discretization(cell=cell_d, nb_of_pixels_global=(N,) * dim, discretization_type='finite_element',
                               element_type=element, device=device)
comm = disc_u.communicator
assert comm.size == 1, 'serial (one GPU) script'

# ---- geometry at the pixel lower-left nodes
geo = np.load(geometry_npz if os.path.isabs(geometry_npz) else os.path.join(here, geometry_npz))
X = np.asarray(to_host(disc_u.fft.coords))                    # [dim, x, y(, z)]
label = np.zeros(X.shape[1:], dtype=np.int8)
pts = X.reshape(dim, -1).T
inside = np.zeros(pts.shape[0], dtype=bool)
for g in geo['aggs']:
    if dim == 2:
        cx, cy, a, b, ang = g
        dx = (pts - [cx, cy] + 0.5) % 1.0 - 0.5
        xr = np.cos(ang) * dx[:, 0] + np.sin(ang) * dx[:, 1]
        yr = -np.sin(ang) * dx[:, 0] + np.cos(ang) * dx[:, 1]
        inside |= (xr / a) ** 2 + (yr / b) ** 2 < 1.0
    else:
        r = g[3:6].max()
        dx = (pts - g[:3] + 0.5) % 1.0 - 0.5
        near = np.all(np.abs(dx) < r, axis=1)                  # bounding box first (many small aggregates)
        y = dx[near] @ g[6:].reshape(3, 3).T
        inside[np.flatnonzero(near)[np.sum((y / g[3:6]) ** 2, axis=1) < 1.0]] = True
label.reshape(-1)[inside] = 1
gel_eps_factor = 1.0
sites = geo['gel_sites']
if N >= gel_cells:
    assert N % gel_cells == 0
    cell_idx = np.floor(sites * gel_cells).astype(int) % gel_cells
    r = N // gel_cells
    for c in cell_idx:
        label[tuple(slice(ci * r, (ci + 1) * r) for ci in c)] = 2
else:
    pix = np.floor(sites * N).astype(int) % N
    label[tuple(pix.T)] = 2
    gel_eps_factor = (N / gel_cells) ** dim
gel_h = label == 2
print(f'{dim}D grid {N}^{dim} (h = {100 / N:.4g} mm), l = {length_scale * 100:.4g} mm (l/h = {length_scale * N:.3g}), '
      f'paste {np.mean(label == 0):.3f} aggregate {np.mean(label == 1):.3f} gel {np.mean(gel_h):.4f} '
      f'(eigenstrain factor {gel_eps_factor:g}), control {control}, preconditioner {preconditioner_type}, '
      f'device {device}', flush=True)

lam_h, mu_h, Gc_h = (np.zeros(label.shape) for _ in range(3))
for p, prm in PHASES.items():
    lp, mp = plane_strain_lame(prm['E'], prm['nu'])
    lam_h[label == p], mu_h[label == p], Gc_h[label == p] = lp, mp, prm['Gc']
if gel_eps_factor < 1.0:
    lg, mg = plane_strain_lame(PHASES[2]['E'], PHASES[2]['nu'])
    la, ma = plane_strain_lame(PHASES[1]['E'], PHASES[1]['nu'])
    lam_h[gel_h] = gel_eps_factor * lg + (1 - gel_eps_factor) * la
    mu_h[gel_h] = gel_eps_factor * mg + (1 - gel_eps_factor) * ma
gel = xp.asarray(gel_h)
lam_x, mu_x, Gc_x = xp.asarray(lam_h), xp.asarray(mu_h), xp.asarray(Gc_h)
kappa_x = lam_x + 2.0 * mu_x / dim                            # bulk modulus (2D: plane strain lam + mu)
crack_driving = xp.asarray(np.where(gel_h, 0.0, 1.0))
lam_r, mu_r = plane_strain_lame(PHASES[0]['E'], PHASES[0]['nu'])
kappa_r = lam_r + 2.0 * mu_r / dim
Gc_ref = PHASES[0]['Gc']

I = np.eye(dim)
I_sym = 0.5 * (np.einsum('ik,jl->ijkl', I, I) + np.einsum('il,jk->ijkl', I, I))
P_vol = np.einsum('ij,kl->ijkl', I, I) / dim
P_dev = I_sym - P_vol
C_ref = dim * kappa_r * P_vol + 2 * mu_r * P_dev
s_vol = disc_u.get_quad_field_scalar(name='s_vol')
s_dev = disc_u.get_quad_field_scalar(name='s_dev')
C_d = domain.ScaledMaterial([(s_vol, xp.asarray(P_vol)), (s_dev, xp.asarray(P_dev))])
C_dual = domain.ScaledMaterial([(lambda: (dim * kappa_r) ** 2 / s_vol.s[0, 0], xp.asarray(P_vol)),
                                (lambda: (2 * mu_r) ** 2 / s_dev.s[0, 0], xp.asarray(P_dev))])

VOIGT = [(0, 0), (1, 1), (0, 1)] if dim == 2 else [(0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1)]
nv = len(VOIGT)
E_UNIT = []
for (i, j) in VOIGT:
    e = np.zeros((dim, dim))
    e[i, j] = e[j, i] = 1.0
    E_UNIT.append(e)
MANDEL_W = np.array([1.0 if i == j else np.sqrt(2.0) for (i, j) in VOIGT])


def voigt(s):
    s = np.asarray(to_host(s))
    return np.array([s[i, j] for (i, j) in VOIGT])


def from_par(e):
    return sum(e[k] * E_UNIT[k] for k in range(nv))


def mandel_norm(M):
    return np.linalg.norm(np.diag(MANDEL_W) @ M @ np.diag(1.0 / MANDEL_W))


macro = disc_u.get_gradient_size_field(name='macro_field')
strain = disc_u.get_gradient_size_field(name='strain')
rhs_u = disc_u.get_unknown_size_field(name='rhs_u')
u_unit = [disc_u.get_unknown_size_field(name=f'u_unit_{k}') for k in range(nv)]
u_gel = disc_u.get_unknown_size_field(name='u_gel') if control == 'exact' else None
u_tot = disc_u.get_unknown_size_field(name='u_total')
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


def K_u(x, Ax):
    disc_u.apply_system_matrix_mugrid(material_data_field=C_d, input_field_inxyz=x, output_field_inxyz=Ax,
                                      formulation='small_strain')


green_u = disc_u.get_preconditioner_Green_mugrid(
    reference_material_data_ijkl=2.0 * mu_r * I_sym if preconditioner_type == 'Green_Jacobi' else C_ref,
    formulation='small_strain', name='green_u')
dual_temp_1 = disc_u.get_unknown_size_field(name='dual_temp_1')
dual_temp_2 = disc_u.get_unknown_size_field(name='dual_temp_2')
x_scaled_u = disc_u.get_unknown_size_field(name='x_scaled_u')
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
        disc_u.apply_preconditioner_mugrid(preconditioner_Fourier_fnfnqks=green_u, input_nodal_field_fnxyz=x,
                                           output_nodal_field_fnxyz=Px)


def apply_damage_operator(input_field_inxyz, output_field_inxyz, reaction_q, gc=None):
    disc_d.apply_gradient_operator_mugrid(u_inxyz=input_field_inxyz, grad_u_ijqxyz=grad_d)
    grad_d.s[...] *= (Gc_x if gc is None else gc) * length_scale
    disc_d.apply_gradient_transposed_operator_mugrid(gradient_field_ijqxyz=grad_d, div_u_fnxyz=output_field_inxyz,
                                                     apply_weights=True)
    disc_d.apply_N_operator_mugrid(input_field_inxyz, temp_q)
    temp_q.s[...] *= reaction_q
    disc_d.apply_N_transposed_operator_mugrid(quad_field_ijqxyz=temp_q, nodal_field_inxyz=temp_n, apply_weights=True)
    output_field_inxyz.s[...] += temp_n.s


def K_d(input_field_inxyz, output_field_inxyz):
    apply_damage_operator(input_field_inxyz, output_field_inxyz, reaction.s)


green_d = disc_d.get_preconditioner_Green_mugrid(
    reference_material_data_ijkl=None,
    operator=lambda input_field_inxyz, output_field_inxyz: apply_damage_operator(
        input_field_inxyz, output_field_inxyz, Gc_ref / length_scale, gc=Gc_ref),
    invert_zero_mode=True, name='green_d')
x_scaled_d = disc_d.get_unknown_size_field(name='x_scaled_d')
jacobi_d = None


def M_d(x, Px):
    x_scaled_d.s[...] = jacobi_d.s * x.s
    disc_d.apply_preconditioner_mugrid(preconditioner_Fourier_fnfnqks=green_d, input_nodal_field_fnxyz=x_scaled_d,
                                       output_nodal_field_fnxyz=Px)
    Px.s[...] *= jacobi_d.s


def update_degraded_material():
    disc_d.apply_N_operator_mugrid(damage, damage_q)
    g = (1.0 - damage_q.s[0, 0]) ** 2 + k_residual
    s_vol.s[0, 0] = xp.where(gel, dim * kappa_x, g * dim * kappa_x)
    s_dev.s[0, 0] = xp.where(gel, 2 * mu_x, g * 2 * mu_x)


def tensile_energy(eps):
    trace = xp.einsum('ii...->...', eps)
    eps_dev = eps - trace / dim * xp.eye(dim).reshape(dim, dim, *([1] * trace.ndim))
    return 0.5 * kappa_x * xp.maximum(trace, 0.0) ** 2 + mu_x * xp.einsum('ij...,ij...->...', eps_dev, eps_dev)


def solve_u(x):
    norms = solvers.conjugate_gradients_mugrid_experimental(
        comm=comm, fc=disc_u.field_collection, hessp=K_u, b=rhs_u, x=x, P=M_u, maxiter=50000,
        stop_crit_norm='energy_lower_estim', d_min=cg_delay,
        **topology_optimization.energy_stop_kwargs(disc_u, K_u, x, cg_tol))[1]
    return int(norms['stop_iteration']['energy_lower_estim'])


def mean_stress(u):
    return disc_u.get_homogenized_stress_mugrid(material_data_field_ijklqxyz=C_d, displacement_field_inxyz=u,
                                                macro_gradient_field_ijqxyz=macro, formulation='small_strain')


eps_gel_field = None


def set_macro(E_ij, eps_g):
    """macro = E - eps_g (gel only)"""
    disc_u.get_macro_gradient_field_mugrid(macro_gradient_ij=E_ij, macro_gradient_field_ijqxyz=macro)
    for i in range(dim):
        macro.s[i, i] -= xp.where(gel, eps_g * gel_eps_factor, 0.0)


def solve_with_macro(u):
    disc_u.get_rhs_mugrid(material_data_field_ijklqxyz=C_d, macro_gradient_field_ijqxyz=macro, rhs_inxyz=rhs_u)
    return solve_u(u)


def unit_solves():
    M, cgs = np.zeros((nv, nv)), []
    for k in range(nv):
        disc_u.get_macro_gradient_field_mugrid(macro_gradient_ij=E_UNIT[k], macro_gradient_field_ijqxyz=macro)
        cgs.append(solve_with_macro(u_unit[k]))
        M[:, k] = voigt(mean_stress(u_unit[k]))
    return M, cgs


def refresh_jacobi():
    global jacobi_u
    if preconditioner_type == 'Green_Jacobi':
        jacobi_u = disc_u.get_preconditioner_Jacobi_mugrid(material_data_field_ijklqxyz=C_d,
                                                           formulation='small_strain', name='jacobi_u')


def fmax(a):
    return float(to_host(xp.max(a)))


# ---------------------------------------------------------------------------- state, restart
os.makedirs(snap_dir, exist_ok=True)
rec = {k: [] for k in ('dVV', 'E', 'stiffness_ratio', 'max_d', 'mean_d', 'staggered', 'converged', 'cg_main_mean',
                       'cg_all_mean', 'cg_d_mean', 'time_step')}
first_step = 0
E_par = np.zeros(nv)
restart_from = os.environ.get('RESTART_FROM')
if restart_from:
    state = np.load(restart_from, allow_pickle=True)
    damage.s[...] = xp.asarray(state['damage'])
    history_n.s[...] = xp.asarray(state['history_n'])
    u_tot.s[...] = xp.asarray(state['u_tot'])
    for k in range(nv):
        u_unit[k].s[...] = xp.asarray(state['u_unit'][k])
    E_par = state['E_par']
    rec = {k: list(v) for k, v in state['rec'].item().items()}
    first_step = int(state['step']) + 1
    print(f'restart from {restart_from}: load step {first_step}', flush=True)
else:
    with open(iterations_file, 'w') as f:
        f.write('step,dVV,staggered_iteration,cg_main,cg_unit,cg_d,time_u,time_d,damage_change,max_d,mean_stress_rel,'
                + ','.join(f'E{k}' for k in range(nv)) + ',stiffness_ratio\n')
update_degraded_material()
history.s[...] = history_n.s
refresh_jacobi()
t0 = time.time()
M_eff, cg_unit0 = unit_solves()          # C_eff of the current (initial or restarted) damage
if restart_from is None:
    C0_norm = mandel_norm(M_eff)
else:
    C0_norm = float(state['C0_norm'])
print(f'initial unit solves: CG {cg_unit0}, {time.time() - t0:.1f} s', flush=True)
last_checkpoint = time.time()
start = time.time()
nb_not_converged = 0


def snapshot(step):
    d = to_host(damage.s[0, 0]).astype(np.float32)
    if dim == 2:
        np.save(os.path.join(snap_dir, f'step_{step:03d}.npy'), d)
    else:
        np.save(os.path.join(snap_dir, f'step_{step:03d}_midz.npy'), d[:, :, N // 2])
        if (step + 1) % snap_full == 0 or step == len(load_steps) - 1:
            np.save(os.path.join(snap_dir, f'step_{step:03d}_full.npy'), d)


for step in range(first_step, len(load_steps)):
    dvv = load_steps[step]
    eps_g = dvv / dim
    t_step = time.time()
    cg_main_step, cg_all_step, cg_d_step, nb_solves = 0, 0, 0, 0
    for stagger in range(max_staggered):
        t0 = time.time()
        refresh_jacobi()
        cg_unit = 0
        if control == 'exact':
            M_eff, cgs = unit_solves()
            cg_unit = sum(cgs)
            set_macro(np.zeros((dim, dim)), eps_g)
            cg_main = solve_with_macro(u_gel)
            sigma_g = voigt(mean_stress(u_gel))
            E_par = np.linalg.solve(M_eff, -sigma_g)
            u_tot.s[...] = u_gel.s + sum(float(E_par[k]) * u_unit[k].s for k in range(nv))
            set_macro(from_par(E_par), eps_g)
            sig = voigt(mean_stress(u_tot))
            nb = nv + 1
        else:
            set_macro(from_par(E_par), eps_g)
            cg_main = solve_with_macro(u_tot)
            sig = voigt(mean_stress(u_tot))
            nb = 1
        stress_rel = np.abs(sig).max() / max(np.abs(M_eff @ E_par).max(), 1e-300)
        disc_u.fft.communicate_ghosts(u_tot)
        disc_u.apply_gradient_operator_symmetrized_mugrid(u_inxyz=u_tot, grad_u_ijqxyz=strain)
        strain.s[...] += macro.s                               # elastic strain E + grad u - eps_g
        if control == 'cheap':
            E_par = E_par - np.linalg.solve(M_eff, sig)
        time_u = time.time() - t0

        history.s[0, 0] = xp.maximum(history_n.s[0, 0], crack_driving * tensile_energy(strain.s))
        reaction.s[...] = Gc_x / length_scale + 2.0 * history.s
        temp_q.s[...] = 2.0 * history.s
        disc_d.apply_N_transposed_operator_mugrid(quad_field_ijqxyz=temp_q, nodal_field_inxyz=rhs_d, apply_weights=True)
        damage_old.s[...] = damage.s
        t0 = time.time()
        jacobi_d = disc_d.get_preconditioner_Jacobi_mugrid(operator=K_d, name='jacobi_d')
        norms_d = solvers.conjugate_gradients_mugrid_experimental(
            comm=comm, fc=disc_d.field_collection, hessp=K_d, b=rhs_d, x=damage, P=M_d, maxiter=50000,
            stop_crit_norm='energy_lower_estim', d_min=cg_delay,
            **topology_optimization.energy_stop_kwargs(disc_d, K_d, damage, cg_tol))[1]
        cg_d = int(norms_d['stop_iteration']['energy_lower_estim'])
        time_d = time.time() - t0
        update_degraded_material()
        change = fmax(xp.abs(damage.s - damage_old.s))
        d_max = fmax(damage.s)
        cg_main_step += cg_main
        cg_all_step += cg_main + cg_unit
        cg_d_step += cg_d
        nb_solves += nb
        with open(iterations_file, 'a') as f:
            f.write(f'{step},{dvv:.6g},{stagger + 1},{cg_main},{cg_unit},{cg_d},{time_u:.4g},{time_d:.4g},'
                    f'{change:.4g},{d_max:.4g},{stress_rel:.3g},' + ','.join(f'{v:.6g}' for v in E_par)
                    + f',{mandel_norm(M_eff) / C0_norm:.6g}\n')
        converged = change < staggered_tol and (control == 'exact' or stress_rel < tol_s)
        if converged:
            break
    nb_not_converged += not converged
    history_n.s[...] = history.s
    if control == 'cheap':                    # C_eff of the converged damage: next step's E update and the output
        refresh_jacobi()
        M_eff, cgs = unit_solves()
        cg_all_step += sum(cgs)
        nb_solves += nv
    stiff = mandel_norm(M_eff) / C0_norm
    for k, v in (('dVV', dvv), ('E', E_par.copy()), ('stiffness_ratio', stiff), ('max_d', d_max),
                 ('mean_d', float(to_host(xp.mean(damage.s)))), ('staggered', stagger + 1),
                 ('converged', int(converged)), ('cg_main_mean', cg_main_step / (stagger + 1)),
                 ('cg_all_mean', cg_all_step / nb_solves), ('cg_d_mean', cg_d_step / (stagger + 1)),
                 ('time_step', time.time() - t_step)):
        rec[k].append(v)
    print(f'step {step:3d}  dV/V = {dvv:.4f}  |C|/|C0| = {stiff:.4f}  max d = {d_max:.3f}  '
          f'mean d = {rec["mean_d"][-1]:.4f}  staggered its = {stagger + 1:3d}  '
          f'mean CG u (main, all) = ({cg_main_step / (stagger + 1):.1f}, {cg_all_step / nb_solves:.1f})  '
          f'mean CG d = {cg_d_step / (stagger + 1):.1f}  <sigma> rel = {stress_rel:.1e}  '
          f'time = {time.time() - t_step:.1f} s'
          + ('' if converged else f'  NOT CONVERGED: step ACCEPTED at MAX_STAGGERED = {max_staggered} '
                                  f'(max |d - d_old| = {change:.2e}, <sigma> rel = {stress_rel:.1e})'), flush=True)
    snapshot(step)
    np.savez(steps_file, label=label if dim == 2 else label[:, :, N // 2], dim=dim, N=N,
             length_scale=length_scale, preconditioner=preconditioner_type, control=control,
             **{k: np.array(v) for k, v in rec.items()})
    if time.time() - last_checkpoint > checkpoint_seconds and step < len(load_steps) - 1:
        np.savez(checkpoint_file + '.tmp.npz', damage=to_host(damage.s), history_n=to_host(history_n.s),
                 u_tot=to_host(u_tot.s), u_unit=np.array([to_host(u.s) for u in u_unit]), E_par=E_par,
                 rec=np.array(rec, dtype=object), step=step, C0_norm=C0_norm)
        os.replace(checkpoint_file + '.tmp.npz', checkpoint_file)
        last_checkpoint = time.time()
        print(f'checkpoint after step {step}', flush=True)
print(f'Elapsed time: {time.time() - start:.1f} s')
if nb_not_converged:
    print(f'WARNING: {nb_not_converged} load steps accepted unconverged (MAX_STAGGERED = {max_staggered})')
