"""
ASR damage of a 2D concrete meso-structure: AT2 phase-field fracture driven by expanding gel pockets
===================================================================================================

Adapted copy of damage_AT2_small_strain/schneider2025_notch/example_2D_phase_field_fracture_AT2.py (same AT2 model,
hybrid formulation, Amor split, history field, staggered scheme, CG energy stopping, Green / Green-Jacobi / dual
elastic preconditioners), set up after the ASR example of Falsafi, Leute, Ladecky, Junge (arXiv 2207.10657, Sec. 3.3,
Fig. 7, Table 2), with AT2 phase field instead of their crack-band damage.

Three phases (geometry/asr_meso_s<SEED>.npy from make_asr_geometry.py): cement paste (0), aggregates (1), ASR gel
pockets (2). Paste and aggregates crack (AT2, own Gc); the gel is linear elastic, never degraded, its tensile energy
does not drive the damage. The gel carries an isotropic eigenstrain eps_g(t) I, tr = 2 eps_g = dV/V (area expansion),
increased load step by load step up to DVV_MAX.

Free expansion (zero mean stress) by superposition, exact at every staggered iteration: with the current damage
    u~ = u_g + sum_k E_k u_k,      eps = E + sym grad u~,   eps_el = eps - eps_g
    u_k: fluctuation for a unit macro strain e_k (xx, yy, xy),  u_g: for the eigenstrain alone (E = 0)
    <sigma> = C_eff E + <sigma_g> = 0   ->   E = -C_eff^-1 <sigma_g>
so every staggered iteration has 4 elastic solves (3 unit + 1 eigenstrain), all with the same degraded C_d and
preconditioner; C_eff is the effective stiffness of the damaged cell (stiffness reduction ||C_eff|| / ||C_eff,0||).
The tensile driving energy psi^+ uses the elastic strain eps_el.

Units: length 0.1 m (cell [0,1)^2), stress E_paste = 12 GPa, so Gc -> Gc / (12e9 Pa * 0.1 m).

    python asr_AT2_gel_expansion.py <Green|Green_Jacobi|dual>
env: NPIX (64), SEED (1), LENGTH_SCALE (1/16), DVV_MAX (0.064), LOAD_STEPS (64), MAX_STAGGERED (500),
     GEOMETRY_NPZ / GEL_CELLS (pocket geometry of the 'real parameters' runs, see below),
     K_RESIDUAL (1e-4), RUN_SUFFIX ('')
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

xp = np
here = os.path.dirname(os.path.abspath(__file__))

# ----------------------------------------------------------------------------
# Parameters (nondimensional: length 0.1 m, stress 12 GPa)
# ----------------------------------------------------------------------------
dim = 2
domain_size = [1.0, 1.0]
N = int(os.environ.get('NPIX', 64))
number_of_pixels = (N, N)
seed = int(os.environ.get('SEED', 1))
geometry_file = os.environ.get('GEOMETRY_FILE', os.path.join(here, 'geometry', f'asr_meso_s{seed}.npy'))
length_scale = float(os.environ.get('LENGTH_SCALE', 1.0 / 16))
k_residual = float(os.environ.get('K_RESIDUAL', 1e-4))
E_unit, L_unit = 12e9, 0.1
#                 E [-]          nu     Gc [-]                       Table 2 of the paper
PHASES = {0: dict(E=12e9 / E_unit, nu=0.30, Gc=60.0 / (E_unit * L_unit)),     # cement paste
          1: dict(E=59e9 / E_unit, nu=0.30, Gc=160.0 / (E_unit * L_unit)),    # aggregate
          2: dict(E=11e9 / E_unit, nu=0.18, Gc=160.0 / (E_unit * L_unit))}    # ASR gel (never degraded, no driving)
dvv_max = float(os.environ.get('DVV_MAX', 0.064))
load_steps = np.linspace(0.0, dvv_max, int(os.environ.get('LOAD_STEPS', 64)) + 1)[1:]   # dV/V of the gel
staggered_tol = 1e-3               # max |d - d_old| for the staggered loop
max_staggered_iterations = int(os.environ.get('MAX_STAGGERED', 500))   # the step is ACCEPTED when hit (logged)
cg_tol = 1e-6                      # CG: energy-error estimate <= cg_tol * energy of the solution (warm start, delay 10)
cg_delay = 10
preconditioner_type = sys.argv[1] if len(sys.argv) > 1 else 'dual'
assert preconditioner_type in ('Green', 'Green_Jacobi', 'dual')
damage_preconditioner_type = 'Green_Jacobi'
run_name = f'ASR_N{N}_{preconditioner_type}_s{seed}_l{length_scale:g}' + os.environ.get('RUN_SUFFIX', '')
iterations_file = os.path.join(here, 'exp_data', run_name + '_iterations.csv')
steps_file = os.path.join(here, 'exp_data', run_name + '_steps.npz')


def plane_strain_lame(E, nu):
    return E * nu / ((1 + nu) * (1 - 2 * nu)), E / (2 * (1 + nu))


# ----------------------------------------------------------------------------
# Two discretizations of the same grid: vector (u~) and scalar (d)
# ----------------------------------------------------------------------------
cell_u = domain.PeriodicUnitCell(domain_size=domain_size, problem_type='elasticity')
disc_u = domain.Discretization(cell=cell_u, nb_of_pixels_global=number_of_pixels,
                               discretization_type='finite_element', element_type='bilinear_rectangle')
cell_d = domain.PeriodicUnitCell(domain_size=domain_size, problem_type='conductivity')
disc_d = domain.Discretization(cell=cell_d, nb_of_pixels_global=number_of_pixels,
                               discretization_type='finite_element', element_type='bilinear_rectangle')
comm = disc_u.communicator
assert comm.size == 1, 'serial script'

# ----------------------------------------------------------------------------
# Microstructure: a pixel takes the label of its lower-left node on the fine geometry grid
# ----------------------------------------------------------------------------
# GEOMETRY_NPZ (env, e.g. geometry/falsafi_s1.npz from make_asr_geometry_falsafi.py): aggregates on a fine grid +
# gel sites as points; a gel pocket is the square cell of the GEL_CELLS^2 grid (default 64: 1.5625 mm) containing the
# site. On a grid N >= GEL_CELLS the pocket is resolved ((N / GEL_CELLS)^2 pixels); on a coarser grid it is the one
# pixel containing the site, with the eigenstrain scaled by (N / GEL_CELLS)^2 (same expansion volume of the pocket)
geometry_npz = os.environ.get('GEOMETRY_NPZ')
gel_cells = int(os.environ.get('GEL_CELLS', 64))
xy = np.asarray(tensor_operations.to_host(disc_u.fft.coords))
gel_eps_factor = 1.0
if geometry_npz is None:
    fine = np.load(geometry_file)
    idx = [np.clip(np.floor(xy[a] * fine.shape[a] + 1e-9).astype(int), 0, fine.shape[a] - 1) for a in range(2)]
    label = fine[tuple(idx)].astype(int)                     # [x, y]: 0 paste, 1 aggregate, 2 gel
else:
    geo = np.load(geometry_npz if os.path.isabs(geometry_npz) else os.path.join(here, geometry_npz))
    fine = geo['aggregate']
    idx = [np.clip(np.floor(xy[a] * fine.shape[a] + 1e-9).astype(int), 0, fine.shape[a] - 1) for a in range(2)]
    label = fine[tuple(idx)].astype(int)
    if N >= gel_cells:
        assert N % gel_cells == 0
        cell_of_pixel = np.floor(xy * gel_cells / 1.0 + 1e-9).astype(int)          # [2, x, y]
        site_cells = {tuple(c) for c in np.floor(geo['gel_sites'] * gel_cells).astype(int) % gel_cells}
        for (ci, cj) in site_cells:
            label[(cell_of_pixel[0] == ci) & (cell_of_pixel[1] == cj)] = 2
    else:
        pix = np.floor(geo['gel_sites'] * N).astype(int) % N
        label[pix[:, 0], pix[:, 1]] = 2
        gel_eps_factor = (N / gel_cells) ** 2
gel = label == 2
print(f'grid {number_of_pixels}, l = {length_scale:.4f} (l/h = {length_scale * N:.2f}), '
      f'area fractions paste {np.mean(label == 0):.3f} aggregate {np.mean(label == 1):.3f} gel {np.mean(gel):.4f} '
      f'({gel.sum()} pixels, eigenstrain factor {gel_eps_factor:g}), '
      f'preconditioner {preconditioner_type}')

lam_x, mu_x, Gc_x = (np.zeros(label.shape) for _ in range(3))
for p, prm in PHASES.items():
    lam_p, mu_p = plane_strain_lame(prm['E'], prm['nu'])
    lam_x[label == p], mu_x[label == p], Gc_x[label == p] = lam_p, mu_p, prm['Gc']
if gel_eps_factor < 1.0:   # coarse grid: the gel pixel is a mixture of gel (fraction gel_eps_factor) and aggregate (Voigt)
    lam_g, mu_g = plane_strain_lame(PHASES[2]['E'], PHASES[2]['nu'])
    lam_a, mu_a = plane_strain_lame(PHASES[1]['E'], PHASES[1]['nu'])
    lam_x[gel] = gel_eps_factor * lam_g + (1 - gel_eps_factor) * lam_a
    mu_x[gel] = gel_eps_factor * mu_g + (1 - gel_eps_factor) * mu_a
kappa_x = lam_x + mu_x                                       # 2D (plane strain) bulk modulus, Amor split
crack_driving = np.where(gel, 0.0, 1.0)
lam_r, mu_r = plane_strain_lame(PHASES[0]['E'], PHASES[0]['nu'])    # reference: the undamaged paste
kappa_r = lam_r + mu_r
Gc_ref = PHASES[0]['Gc']

I2 = np.eye(2)
I_sym = 0.5 * (np.einsum('ik,jl->ijkl', I2, I2) + np.einsum('il,jk->ijkl', I2, I2))
P_vol = 0.5 * np.einsum('ij,kl->ijkl', I2, I2)
P_dev = I_sym - P_vol
C_ref = 2 * kappa_r * P_vol + 2 * mu_r * P_dev               # = lam_r I x I + 2 mu_r I_sym

# C_d = s_vol P_vol + s_dev P_dev,  s_vol = g(d) 2 kappa(x), s_dev = g(d) 2 mu(x)  (gel: not degraded)
s_vol = disc_u.get_quad_field_scalar(name='s_vol')
s_dev = disc_u.get_quad_field_scalar(name='s_dev')
C_d = domain.ScaledMaterial([(s_vol, P_vol), (s_dev, P_dev)])
# dual: C_ref C_d^-1 C_ref = (2 kappa_r)^2 / s_vol P_vol + (2 mu_r)^2 / s_dev P_dev (evaluated on the fly)
C_dual = domain.ScaledMaterial([(lambda: (2 * kappa_r) ** 2 / s_vol.s[0, 0], P_vol),
                                (lambda: (2 * mu_r) ** 2 / s_dev.s[0, 0], P_dev)])

# ----------------------------------------------------------------------------
# Fields
# ----------------------------------------------------------------------------
macro = disc_u.get_gradient_size_field(name='macro_field')   # E, e_k or -eps_g (spatially varying)
strain = disc_u.get_gradient_size_field(name='strain')
rhs_u = disc_u.get_unknown_size_field(name='rhs_u')
u_unit = [disc_u.get_unknown_size_field(name=f'u_unit_{k}') for k in range(3)]
u_gel = disc_u.get_unknown_size_field(name='u_gel')
u_tot = disc_u.get_unknown_size_field(name='u_total')
E_UNIT = [np.array([[1.0, 0.0], [0.0, 0.0]]), np.array([[0.0, 0.0], [0.0, 1.0]]), np.array([[0.0, 1.0], [1.0, 0.0]])]
# Voigt-like: stress (s_xx, s_yy, s_xy), strain parameters (E_xx, E_yy, E_xy) with E = sum a_k e_k


def voigt(s):
    return np.array([s[0, 0], s[1, 1], s[0, 1]])


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


# ----------------------------------------------------------------------------
# Operators
# ----------------------------------------------------------------------------
def K_u(x, Ax):
    disc_u.apply_system_matrix_mugrid(material_data_field=C_d, input_field_inxyz=x,
                                      output_field_inxyz=Ax, formulation='small_strain')


# Green and dual: C_ref = paste; Green-Jacobi: 2 mu_r I_sym (the Jacobi diagonal carries the local magnitude)
green_u = disc_u.get_preconditioner_Green_mugrid(
    reference_material_data_ijkl=2.0 * mu_r * I_sym if preconditioner_type == 'Green_Jacobi' else C_ref,
    formulation='small_strain', name='green_u')
dual_temp_1 = disc_u.get_unknown_size_field(name='dual_temp_1')
dual_temp_2 = disc_u.get_unknown_size_field(name='dual_temp_2')
jacobi_u = None
x_scaled_u = disc_u.get_unknown_size_field(name='x_scaled_u')


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


def apply_damage_operator(input_field_inxyz, output_field_inxyz, reaction_q, gc=None):
    """[Gc l B^T W B + N^T W c N] d, Gc per pixel (default Gc_x)"""
    disc_d.apply_gradient_operator_mugrid(u_inxyz=input_field_inxyz, grad_u_ijqxyz=grad_d)
    grad_d.s[...] *= (Gc_x if gc is None else gc) * length_scale
    disc_d.apply_gradient_transposed_operator_mugrid(gradient_field_ijqxyz=grad_d,
                                                     div_u_fnxyz=output_field_inxyz, apply_weights=True)
    disc_d.apply_N_operator_mugrid(input_field_inxyz, temp_q)
    temp_q.s[...] *= reaction_q
    disc_d.apply_N_transposed_operator_mugrid(quad_field_ijqxyz=temp_q, nodal_field_inxyz=temp_n,
                                              apply_weights=True)
    output_field_inxyz.s[...] += temp_n.s


def K_d(input_field_inxyz, output_field_inxyz):
    x, Ax = input_field_inxyz, output_field_inxyz
    apply_damage_operator(x, Ax, reaction.s)


green_d = disc_d.get_preconditioner_Green_mugrid(
    reference_material_data_ijkl=None,
    operator=lambda input_field_inxyz, output_field_inxyz: apply_damage_operator(
        input_field_inxyz, output_field_inxyz, Gc_ref / length_scale, gc=Gc_ref),
    invert_zero_mode=True, name='green_d')
jacobi_d = None
x_scaled_d = disc_d.get_unknown_size_field(name='x_scaled_d')


def M_d(x, Px):
    x_scaled_d.s[...] = jacobi_d.s * x.s
    disc_d.apply_preconditioner_mugrid(preconditioner_Fourier_fnfnqks=green_d,
                                       input_nodal_field_fnxyz=x_scaled_d, output_nodal_field_fnxyz=Px)
    Px.s[...] *= jacobi_d.s


def update_degraded_material():
    disc_d.apply_N_operator_mugrid(damage, damage_q)
    g = (1.0 - damage_q.s[0, 0]) ** 2 + k_residual
    s_vol.s[0, 0] = np.where(gel, 2 * kappa_x, g * 2 * kappa_x)
    s_dev.s[0, 0] = np.where(gel, 2 * mu_x, g * 2 * mu_x)


def tensile_energy(eps):
    """psi^+ = kappa/2 <tr eps>_+^2 + mu eps_dev : eps_dev  (Amor split, elastic strain)"""
    trace = np.einsum('ii...->...', eps)
    eps_dev = eps - trace / dim * np.eye(dim).reshape(dim, dim, *([1] * trace.ndim))
    return 0.5 * kappa_x * np.maximum(trace, 0.0) ** 2 + mu_x * np.einsum('ij...,ij...->...', eps_dev, eps_dev)


def solve_u(x):
    """CG for K x = rhs_u (warm start from x), energy-error stopping; returns the iteration count"""
    norms = solvers.conjugate_gradients_mugrid_experimental(
        comm=comm, fc=disc_u.field_collection, hessp=K_u, b=rhs_u, x=x, P=M_u, maxiter=20000,
        stop_crit_norm='energy_lower_estim', d_min=cg_delay,
        **topology_optimization.energy_stop_kwargs(disc_u, K_u, x, cg_tol))[1]
    return int(norms['stop_iteration']['energy_lower_estim'])


def mean_stress(u):
    return disc_u.get_homogenized_stress_mugrid(material_data_field_ijklqxyz=C_d, displacement_field_inxyz=u,
                                                macro_gradient_field_ijqxyz=macro, formulation='small_strain')


def mandel_norm(M):
    """Frobenius norm of the 3x3 map (E_xx, E_yy, E_xy) -> (s_xx, s_yy, s_xy) in Mandel form"""
    D = np.diag([1.0, 1.0, np.sqrt(2.0)])
    Dinv_e = np.diag([1.0, 1.0, 1.0 / np.sqrt(2.0)])        # E_xy parameter -> Mandel sqrt(2) E_xy
    return np.linalg.norm(D @ M @ Dinv_e)


# ----------------------------------------------------------------------------
# Load stepping (gel expansion) with the staggered scheme
# ----------------------------------------------------------------------------
os.makedirs(os.path.dirname(iterations_file), exist_ok=True)
with open(iterations_file, 'w') as f:
    f.write('step,dVV,staggered_iteration,cg_gel,cg_unit_xx,cg_unit_yy,cg_unit_xy,cg_d,time_u,time_d,'
            'damage_change,E_xx,E_yy,E_xy,mean_stress_rel,stiffness_ratio,max_d\n')
start_time = time.time()
update_degraded_material()
gel_q = np.broadcast_to(gel, s_vol.s[0, 0].shape)            # [q, x, y]
rec = dict(dVV=[0.0], E=[np.zeros(3)], stiffness_ratio=[1.0], damage=[], max_d=[0.0], mean_d=[0.0],
           staggered=[0], converged=[1], cg_gel_mean=[], cg_all_mean=[], cg_d_mean=[], time_step=[])
C_eff_0 = None
nb_not_converged = 0
for step, dvv in enumerate(load_steps):
    eps_g = 0.5 * dvv                                         # isotropic eigenstrain, tr = dV/V
    cg_g_step, cg_all_step, cg_d_step, nb_solves = 0, 0, 0, 0
    step_start = time.time()
    for stagger in range(max_staggered_iterations):
        # 1. elasticity with the current damage: 3 unit macro strains + the eigenstrain, then E for <sigma> = 0
        t0 = time.time()
        if preconditioner_type == 'Green_Jacobi':
            jacobi_u = disc_u.get_preconditioner_Jacobi_mugrid(material_data_field_ijklqxyz=C_d,
                                                               formulation='small_strain', name='jacobi_u')
        M_eff = np.zeros((3, 3))
        cg_unit = []
        for k in range(3):
            disc_u.get_macro_gradient_field_mugrid(macro_gradient_ij=E_UNIT[k], macro_gradient_field_ijqxyz=macro)
            disc_u.get_rhs_mugrid(material_data_field_ijklqxyz=C_d, macro_gradient_field_ijqxyz=macro, rhs_inxyz=rhs_u)
            cg_unit.append(solve_u(u_unit[k]))
            M_eff[:, k] = voigt(mean_stress(u_unit[k]))
        macro.s[...] = 0.0
        macro.s[0, 0] = np.where(gel_q, -eps_g * gel_eps_factor, 0.0)
        macro.s[1, 1] = np.where(gel_q, -eps_g * gel_eps_factor, 0.0)
        disc_u.get_rhs_mugrid(material_data_field_ijklqxyz=C_d, macro_gradient_field_ijqxyz=macro, rhs_inxyz=rhs_u)
        cg_gel = solve_u(u_gel)
        sigma_g = voigt(mean_stress(u_gel))
        E_par = np.linalg.solve(M_eff, -sigma_g)              # <sigma> = M_eff E + <sigma_g> = 0
        E_ij = sum(E_par[k] * E_UNIT[k] for k in range(3))
        u_tot.s[...] = u_gel.s + sum(E_par[k] * u_unit[k].s for k in range(3))
        # elastic strain eps_el = E + sym grad u~ - eps_g  (macro: E - eps_g)
        macro.s[...] += E_ij[(...,) + (np.newaxis,) * 3]
        check = mean_stress(u_tot)
        mean_stress_rel = np.abs(check).max() / max(np.abs(sigma_g).max(), 1e-300)
        disc_u.fft.communicate_ghosts(u_tot)
        disc_u.apply_gradient_operator_symmetrized_mugrid(u_inxyz=u_tot, grad_u_ijqxyz=strain)
        strain.s[...] += macro.s
        time_u = time.time() - t0
        if C_eff_0 is None:
            C_eff_0 = mandel_norm(M_eff)

        # 2. history  H = max(H_{n-1}, psi^+(eps_el)), the gel does not drive the damage
        history.s[0, 0] = np.maximum(history_n.s[0, 0], crack_driving * tensile_energy(strain.s))

        # 3. damage:  [Gc l B^T W B + N^T W (Gc/l + 2H) N] d = N^T W 2H
        reaction.s[...] = Gc_x / length_scale + 2.0 * history.s
        temp_q.s[...] = 2.0 * history.s
        disc_d.apply_N_transposed_operator_mugrid(quad_field_ijqxyz=temp_q, nodal_field_inxyz=rhs_d, apply_weights=True)
        damage_old.s[...] = damage.s
        t0 = time.time()
        jacobi_d = disc_d.get_preconditioner_Jacobi_mugrid(operator=K_d, name='jacobi_d')
        norms_d = solvers.conjugate_gradients_mugrid_experimental(
            comm=comm, fc=disc_d.field_collection, hessp=K_d, b=rhs_d, x=damage, P=M_d, maxiter=20000,
            stop_crit_norm='energy_lower_estim', d_min=cg_delay,
            **topology_optimization.energy_stop_kwargs(disc_d, K_d, damage, cg_tol))[1]
        cg_d = int(norms_d['stop_iteration']['energy_lower_estim'])
        time_d = time.time() - t0
        update_degraded_material()
        change = float(np.max(np.abs(damage.s - damage_old.s)))
        d_max = float(np.max(damage.s))
        stiff = mandel_norm(M_eff) / C_eff_0
        cg_g_step += cg_gel
        cg_all_step += cg_gel + sum(cg_unit)
        cg_d_step += cg_d
        nb_solves += 1
        with open(iterations_file, 'a') as f:
            f.write(f'{step},{dvv:.6g},{stagger + 1},{cg_gel},{cg_unit[0]},{cg_unit[1]},{cg_unit[2]},{cg_d},'
                    f'{time_u:.4g},{time_d:.4g},{change:.4g},{E_par[0]:.6g},{E_par[1]:.6g},{E_par[2]:.6g},'
                    f'{mean_stress_rel:.3g},{stiff:.6g},{d_max:.4g}\n')
        if change < staggered_tol:
            break
    converged = change < staggered_tol
    nb_not_converged += not converged
    history_n.s[...] = history.s
    rec['dVV'].append(dvv)
    rec['E'].append(E_par.copy())
    rec['stiffness_ratio'].append(stiff)       # C_eff of the last staggered iteration (damage before its last update)
    rec['damage'].append(np.copy(damage.s[0, 0]).astype(np.float32))
    rec['max_d'].append(d_max)
    rec['mean_d'].append(float(np.mean(damage.s)))
    rec['staggered'].append(stagger + 1)
    rec['converged'].append(int(converged))
    rec['cg_gel_mean'].append(cg_g_step / nb_solves)
    rec['cg_all_mean'].append(cg_all_step / (4 * nb_solves))
    rec['cg_d_mean'].append(cg_d_step / nb_solves)
    rec['time_step'].append(time.time() - step_start)
    print(f'step {step:3d}  dV/V = {dvv:.4f}  E = ({E_par[0]:.3e}, {E_par[1]:.3e}, {E_par[2]:.2e})  '
          f'|C|/|C0| = {stiff:.4f}  max d = {d_max:.3f}  mean d = {rec["mean_d"][-1]:.4f}  '
          f'staggered its = {stagger + 1:3d}  mean CG u (gel, all 4) = ({cg_g_step / nb_solves:.1f}, '
          f'{cg_all_step / (4 * nb_solves):.1f})  mean CG d = {cg_d_step / nb_solves:.1f}  '
          f'<sigma> rel = {mean_stress_rel:.1e}  time = {time.time() - step_start:.1f} s'
          + ('' if converged else f'  NOT CONVERGED: step ACCEPTED at MAX_STAGGERED = {max_staggered_iterations} '
                                  f'(max |d - d_old| = {change:.2e})'), flush=True)
    np.savez(steps_file, label=label, length_scale=length_scale, preconditioner=preconditioner_type,
             **{k: np.array(v) for k, v in rec.items()})

print(f'Elapsed time: {time.time() - start_time:.1f} s')
if nb_not_converged:
    print(f'WARNING: {nb_not_converged} load steps accepted unconverged (MAX_STAGGERED = {max_staggered_iterations})')
