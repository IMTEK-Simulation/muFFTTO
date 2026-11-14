"""
Phase-field solid-state sintering with elasticity (Wang 2006 type) -- elastic CG cost of Green / Green-Jacobi / dual
=====================================================================================================================

Fields (nodal, Q1 'conductivity' discretisation): conserved density rho (1 solid, 0 pore), non-conserved grain order
parameters eta_i (one per particle). Free energy (Wang, Acta Mater 54 (2006) 953; dimensionless A = 16, B = 1,
k_rho = 10, k_eta = 5, mapped to Cu units through calibrate_wang.py):

    F = int f(rho, eta) + k_rho/2 |grad rho|^2 + sum_i k_eta/2 |grad eta_i|^2 + psi_el(rho, eps) dx
    f = A rho^2 (1-rho)^2 + B [rho^2 + 6(1-rho) S2 - 4(2-rho) S3 + 3 S2^2],  S2 = sum eta_i^2, S3 = sum eta_i^3
    psi_el = 1/2 eps : C(rho) : eps,  C(rho) = (h(rho) + (1-h(rho)) k_res) C_Cu,  h = rho^3 (10 - 15 rho + 6 rho^2)

One time step:
 1. mechanics (the compared solve): div sigma = 0 on the periodic cell with prescribed macroscopic stress
    Sigma = -P e_y e_y (die hot pressing; LOAD=hydro: -P I). No eigenstrain, so the problem is linear in the macro
    strain: 3 unit-strain solves (warm started from the previous step), C_eff from their mean stresses,
    E = C_eff^-1 Sigma, u = sum E_k u_k. Every solve: PCG with energy-error stopping 1e-6, adaptive delay d_min 10.
 2. Cahn-Hilliard  d rho/dt = div(M grad mu),  mu = df/drho + k_rho (-Lap) rho + d psi_el/d rho,
    M = M_vol h + M_vap (1-h) + M_surf rho^2 (1-rho)^2 + M_gb sum_{i!=j} eta_i eta_j      (Wang mobility)
 3. Allen-Cahn  d eta_i/dt = -L (df/deta_i + k_eta (-Lap) eta_i)
 Rigid-body motion (Wang's advection) is NOT implemented (see README); the cell is periodic with fixed size, so the
 solid fraction is conserved and densification shows as neck growth, pore rounding/closure and stiffening.

Spatial discretisation: everything uses muFFTTO's own Q1 operators and FFT engine (disc_pf.fft, the same kind of
engine the Green preconditioners use): div(M grad mu) = -(G^T W M G mu) / m (lumped nodal mass m = h^2), the Laplacian
-Lap = (G^T W G) / m, whose Fourier symbol lam(k) is taken from the FFT of its impulse response. Semi-implicit
stabilised update in Fourier space (Zhu et al. 1999 style):
    rho^{n+1}_hat = [rho_hat + dt R_hat + dt M0 (a lam + k_rho lam^2) rho_hat] / [1 + dt M0 (a lam + k_rho lam^2)]
    eta^{n+1}_hat = [eta_hat - dt L g_hat + dt L S eta_hat]                     / [1 + dt L (S + k_eta lam)]
with R = div(M grad mu^n), g = df/deta^n (k_eta Lap implicit).

Units: length nm, energy density e0 = gamma_s / (gamma_s* l0) [J/m^3], time s. Elasticity in units of E_Cu.

    python sintering_phase_field.py <Green|Green_Jacobi|dual>
env: NPIX (64), DEVICE (cpu|gpu), STEPS (400), DT (s, 5e-7), WIDTH (surface 10-90 % width in nm, 4), PRESSURE (Pa,
     50e6), LOAD (uniaxial|hydro), K_RES (1e-4), GEOMETRY (geometry/particles_s2.npz), SNAP (every 10 steps),
     MECH_EVERY (1: elastic solves every MECH_EVERY phase-field steps, d psi_el/d rho lagged in between; SNAP must be a
     multiple of it),
     CHECKPOINT_SECONDS (1800), RESTART_FROM, RUN_SUFFIX
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
dim = 2
N = int(os.environ.get('NPIX', 64))
device = os.environ.get('DEVICE', 'cpu')
if device == 'gpu':
    import cupy as xp
else:
    xp = np
to_host = tensor_operations.to_host
preconditioner_type = sys.argv[1] if len(sys.argv) > 1 else 'dual'
assert preconditioner_type in ('Green', 'Green_Jacobi', 'dual')
n_steps = int(os.environ.get('STEPS', 400))
dt = float(os.environ.get('DT', 5e-7))
width_nm = float(os.environ.get('WIDTH', 4.0))
pressure = float(os.environ.get('PRESSURE', 50e6))
load = os.environ.get('LOAD', 'uniaxial')
k_res = float(os.environ.get('K_RES', 1e-4))
geometry = os.environ.get('GEOMETRY', 'geometry/particles_s2.npz')
snap_every = int(os.environ.get('SNAP', 10))
mech_every = int(os.environ.get('MECH_EVERY', 1))   # elastic problem solved every MECH_EVERY phase-field steps
checkpoint_seconds = float(os.environ.get('CHECKPOINT_SECONDS', 1800))
cg_tol, cg_delay = 1e-6, 10

# ---------------------------------------------------------------- physical parameters (Cu, ~700 K)
# Seiz, Hierl, Nestler, Rheinheimer, arXiv:2308.09682 (2024), Table 1 ("approximating copper at ~700 K")
T = 700.0
kB = 1.380649e-23
gamma_s_phys = 2.0          # J/m^2   surface energy
D_vol = 1e-15               # m^2/s   volume diffusion
D_gb = 5.5e-11              # m^2/s   grain-boundary diffusion
D_surf = 1.69e-10           # m^2/s   surface diffusion
delta_i = 2e-10             # m       physical (diffusion) interface width
Omega = 1.22e-29            # m^3     atomic volume
m_gb = 1.0 / 8e10           # m^4/(J s)  GB mobility := 1 / tau_gv (Seiz surface inverse mobility; assumption)
E_cu, nu_cu = 120e9, 0.34   # Pa      Cu elastic constants (polycrystal average)
vap_ratio = 0.1             # M_vap / M_vol (Wang 2006 uses a small vapour mobility; assumption)

cal = np.load(os.path.join(here, 'geometry', 'wang_calibration.npz'))
A, B, k_rho_d, k_eta_d = (float(cal[k]) for k in ('A', 'B', 'k_rho', 'k_eta'))
l0 = width_nm / float(cal['width'])                          # nm per Wang length unit
e0 = gamma_s_phys / (float(cal['gamma_s']) * l0 * 1e-9)      # J/m^3
gamma_gb_model = float(cal['gamma_gb']) * e0 * l0 * 1e-9      # J/m^2
k_rho = k_rho_d * l0 ** 2                                    # e0 nm^2
k_eta = k_eta_d * l0 ** 2
I_s = float(cal['I_s']) * l0 * 1e-9                          # m, int rho^2 (1-rho)^2 dx across a surface
I_gb = float(cal['I_gb']) * l0 * 1e-9                        # m, int sum_{i!=j} eta_i eta_j dx across a GB
to_code_M = e0 / 1e-18                                       # m^5/(J s) -> nm^2/s (mu in e0, lengths nm)
M_vol = D_vol * Omega / (kB * T) * to_code_M
M_vap = vap_ratio * M_vol
M_surf = D_surf * delta_i * Omega / (kB * T) / I_s * to_code_M
M_gb = D_gb * delta_i * Omega / (kB * T) / I_gb * to_code_M
L_ac = m_gb * gamma_gb_model / (k_eta_d * (l0 * 1e-9) ** 2)        # 1/s (code: e0 units):  v = L k_eta curvature = m_gb gamma_gb K
psi_to_e0 = E_cu / e0                                        # elastic energy density in E_cu units -> e0

geo = np.load(os.path.join(here, geometry) if not os.path.isabs(geometry) else geometry)
cell_nm = float(geo['cell'])
centres, radii = geo['centres'], geo['radii']
n_grains = len(radii)
h = cell_nm / N
run_name = (f'SINT_N{N}_{preconditioner_type}_W{width_nm:g}_P{pressure / 1e6:g}{load[0]}_k{k_res:g}'
            + os.environ.get('RUN_SUFFIX', ''))
out_dir = os.path.join(here, 'exp_data')
steps_file = os.path.join(out_dir, run_name + '_steps.npz')
snap_file = os.path.join(out_dir, run_name + '_snaps')
checkpoint_file = os.path.join(out_dir, run_name + '_checkpoint.npz')
os.makedirs(snap_file, exist_ok=True)

# ---------------------------------------------------------------- discretisations (same grid, cell in nm)
element = 'bilinear_rectangle'
cell_u = domain.PeriodicUnitCell(domain_size=[cell_nm] * dim, problem_type='elasticity')
disc_u = domain.Discretization(cell=cell_u, nb_of_pixels_global=(N,) * dim, discretization_type='finite_element',
                               element_type=element, device=device)
cell_pf = domain.PeriodicUnitCell(domain_size=[cell_nm] * dim, problem_type='conductivity')
disc_pf = domain.Discretization(cell=cell_pf, nb_of_pixels_global=(N,) * dim, discretization_type='finite_element',
                                element_type=element, device=device)
comm = disc_u.communicator
assert comm.size == 1, 'serial (one GPU) script'
fft = disc_pf.fft
print(f'grid {N}^2, h = {h:.4g} nm, cell {cell_nm:g} nm, surface width {width_nm:g} nm (W/h = {width_nm / h:.3g}), '
      f'l0 = {l0:.4g} nm, e0 = {e0:.4g} J/m^3, gamma_gb(model) = {gamma_gb_model:.3g} J/m^2, '
      f'M_vol/vap/surf/gb = {M_vol:.3g}/{M_vap:.3g}/{M_surf:.3g}/{M_gb:.3g} nm^2/s, L = {L_ac:.3g} 1/s, '
      f'{n_grains} grains, P = {pressure / 1e6:g} MPa ({load}), k_res = {k_res:g}, dt = {dt:g} s, '
      f'{n_steps} steps, preconditioner {preconditioner_type}, device {device}', flush=True)

# ---------------------------------------------------------------- phase-field fields and Q1 operators
rho = disc_pf.get_unknown_size_field(name='rho')
etas = [disc_pf.get_unknown_size_field(name=f'eta_{i}') for i in range(n_grains)]
mu = disc_pf.get_unknown_size_field(name='mu')
tmp_n = disc_pf.get_unknown_size_field(name='tmp_n')
tmp_n2 = disc_pf.get_unknown_size_field(name='tmp_n2')
grad_pf = disc_pf.get_gradient_size_field(name='grad_pf')
q_pf = disc_pf.get_quad_field_scalar(name='q_pf')
q_mob = disc_pf.get_quad_field_scalar(name='q_mobility')
f_hat = disc_pf.ffield_collection.complex_field(name='pf_hat', components=(1,), sub_pt='nodal_points')
f_hat2 = disc_pf.ffield_collection.complex_field(name='pf_hat2', components=(1,), sub_pt='nodal_points')
norm_fft = fft.normalisation

tmp_n.s[...] = 1.0
disc_pf.apply_N_operator_mugrid(tmp_n, q_pf)
disc_pf.apply_N_transposed_operator_mugrid(quad_field_ijqxyz=q_pf, nodal_field_inxyz=tmp_n2, apply_weights=True)
m_lumped = float(to_host(tmp_n2.s[0, 0, 0, 0]))               # = h^2


def stiffness(x, out, coef=None):
    """out = G^T W (coef G x)  (coef at quadrature points; None = 1)"""
    fft.communicate_ghosts(x)
    disc_pf.apply_gradient_operator_mugrid(u_inxyz=x, grad_u_ijqxyz=grad_pf)
    if coef is not None:
        grad_pf.s[...] *= coef
    disc_pf.apply_gradient_transposed_operator_mugrid(gradient_field_ijqxyz=grad_pf, div_u_fnxyz=out,
                                                      apply_weights=True)


def to_quad(x, out):
    fft.communicate_ghosts(x)
    disc_pf.apply_N_operator_mugrid(x, out)


# Fourier symbol of the lumped Q1 Laplacian, -Lap ~ lam(k)
tmp_n.s[...] = 0.0
tmp_n.s[0, 0, 0, 0] = 1.0
stiffness(tmp_n, tmp_n2)
fft.fft(tmp_n2, f_hat)
lam = xp.real(f_hat.s[0, 0]).copy() / m_lumped               # unnormalised FFT of the stencil = its symbol
lam[xp.abs(lam) < 1e-14] = 0.0
lam_max = float(to_host(lam.max()))


def neg_laplacian(x, out):
    stiffness(x, out)
    out.s[...] /= m_lumped


# ---------------------------------------------------------------- free energy
def f_derivs(r, es):
    s2 = sum(e ** 2 for e in es)
    s3 = sum(e ** 3 for e in es)
    dr = A * (2 * r * (1 - r) ** 2 - 2 * r ** 2 * (1 - r)) + B * (2 * r - 6 * s2 + 4 * s3)
    de = [B * (12 * (1 - r) * e - 12 * (2 - r) * e ** 2 + 12 * e * s2) for e in es]
    return dr, de


def interp_h(r):
    r = xp.clip(r, 0.0, 1.0)
    return r ** 3 * (10 - 15 * r + 6 * r ** 2)


def interp_dh(r):
    r = xp.clip(r, 0.0, 1.0)
    return 30 * r ** 2 * (1 - r) ** 2


# ---------------------------------------------------------------- initial condition (or restart)
X = np.asarray(to_host(disc_pf.fft.coords)) * cell_nm          # nodal coordinates [2, x, y] in nm
delta = width_nm / (2 * np.arctanh(0.8))                       # tanh scale for a 10-90 % width W
dist = np.empty((n_grains,) + X.shape[1:])
power = np.empty_like(dist)
for i, (c, r) in enumerate(zip(centres, radii)):
    d = np.sqrt(sum(((X[a] - c[a] + cell_nm / 2) % cell_nm - cell_nm / 2) ** 2 for a in range(2)))
    dist[i] = d - r
    power[i] = d ** 2 - r ** 2
owner = np.argmin(power, axis=0)
for i in range(n_grains):
    prof = 0.5 * (1 - np.tanh(dist[i] / delta))
    etas[i].s[0, 0] = xp.asarray(np.where(owner == i, prof, 0.0))
rho.s[0, 0] = sum(e.s[0, 0] for e in etas)

# ---------------------------------------------------------------- mechanics (as asr_AT2_mesh_study.py)
lam_cu = E_cu * nu_cu / ((1 + nu_cu) * (1 - 2 * nu_cu)) / E_cu    # plane strain, units E_cu
mu_cu = E_cu / (2 * (1 + nu_cu)) / E_cu
kappa_cu = lam_cu + 2.0 * mu_cu / dim
I = np.eye(dim)
I_sym = 0.5 * (np.einsum('ik,jl->ijkl', I, I) + np.einsum('il,jk->ijkl', I, I))
P_vol = np.einsum('ij,kl->ijkl', I, I) / dim
P_dev = I_sym - P_vol
C_solid = dim * kappa_cu * P_vol + 2 * mu_cu * P_dev
s_vol = disc_u.get_quad_field_scalar(name='s_vol')
s_dev = disc_u.get_quad_field_scalar(name='s_dev')
C_rho = domain.ScaledMaterial([(s_vol, xp.asarray(P_vol)), (s_dev, xp.asarray(P_dev))])
C_dual = domain.ScaledMaterial([(lambda: (dim * kappa_cu) ** 2 / s_vol.s[0, 0], xp.asarray(P_vol)),
                                (lambda: (2 * mu_cu) ** 2 / s_dev.s[0, 0], xp.asarray(P_dev))])
VOIGT = [(0, 0), (1, 1), (0, 1)]
nv = len(VOIGT)
E_UNIT = []
for (i, j) in VOIGT:
    e = np.zeros((dim, dim))
    e[i, j] = e[j, i] = 1.0
    E_UNIT.append(e)
Sigma = -pressure / E_cu * (np.diag([0.0, 1.0]) if load == 'uniaxial' else np.eye(dim))
Sigma_v = np.array([Sigma[i, j] for (i, j) in VOIGT])


def voigt(s):
    s = np.asarray(to_host(s))
    return np.array([s[i, j] for (i, j) in VOIGT])


macro = disc_u.get_gradient_size_field(name='macro_field')
strain = disc_u.get_gradient_size_field(name='strain')
rhs_u = disc_u.get_unknown_size_field(name='rhs_u')
u_unit = [disc_u.get_unknown_size_field(name=f'u_unit_{k}') for k in range(nv)]
u_tot = disc_u.get_unknown_size_field(name='u_total')
rho_q = disc_pf.get_quad_field_scalar(name='rho_q')
psi_q = disc_pf.get_quad_field_scalar(name='dpsi_drho_q')


def K_u(x, Ax):
    disc_u.apply_system_matrix_mugrid(material_data_field=C_rho, input_field_inxyz=x, output_field_inxyz=Ax,
                                      formulation='small_strain')


green_u = disc_u.get_preconditioner_Green_mugrid(
    reference_material_data_ijkl=2.0 * mu_cu * I_sym if preconditioner_type == 'Green_Jacobi' else C_solid,
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


def update_material():
    to_quad(rho, rho_q)
    g = interp_h(rho_q.s[0, 0]) * (1 - k_res) + k_res
    s_vol.s[0, 0] = g * dim * kappa_cu
    s_dev.s[0, 0] = g * 2 * mu_cu


def solve_u(x):
    norms = solvers.conjugate_gradients_mugrid_experimental(
        comm=comm, fc=disc_u.field_collection, hessp=K_u, b=rhs_u, x=x, P=M_u, maxiter=50000,
        stop_crit_norm='energy_lower_estim', d_min=cg_delay,
        **topology_optimization.energy_stop_kwargs(disc_u, K_u, x, cg_tol))[1]
    return int(norms['stop_iteration']['energy_lower_estim'])


def mechanics():
    """3 warm-started unit solves, C_eff, E = C_eff^-1 Sigma; returns CG counts, C_eff, E"""
    global jacobi_u
    update_material()
    if preconditioner_type == 'Green_Jacobi':
        jacobi_u = disc_u.get_preconditioner_Jacobi_mugrid(material_data_field_ijklqxyz=C_rho,
                                                           formulation='small_strain', name='jacobi_u')
    C_eff, cgs = np.zeros((nv, nv)), []
    for k in range(nv):
        disc_u.get_macro_gradient_field_mugrid(macro_gradient_ij=E_UNIT[k], macro_gradient_field_ijqxyz=macro)
        disc_u.get_rhs_mugrid(material_data_field_ijklqxyz=C_rho, macro_gradient_field_ijqxyz=macro, rhs_inxyz=rhs_u)
        cgs.append(solve_u(u_unit[k]))
        C_eff[:, k] = voigt(disc_u.get_homogenized_stress_mugrid(
            material_data_field_ijklqxyz=C_rho, displacement_field_inxyz=u_unit[k],
            macro_gradient_field_ijqxyz=macro, formulation='small_strain'))
    E_par = np.linalg.solve(C_eff, Sigma_v)
    E_mat = sum(E_par[k] * E_UNIT[k] for k in range(nv))
    u_tot.s[...] = sum(float(E_par[k]) * u_unit[k].s for k in range(nv))
    disc_u.fft.communicate_ghosts(u_tot)
    disc_u.apply_gradient_operator_symmetrized_mugrid(u_inxyz=u_tot, grad_u_ijqxyz=strain)
    disc_u.get_macro_gradient_field_mugrid(macro_gradient_ij=E_mat, macro_gradient_field_ijqxyz=macro)
    strain.s[...] += macro.s
    # d psi_el / d rho = 1/2 h'(rho) (1 - k_res) eps : C_Cu : eps   (quad points, e0 units)
    tr = strain.s[0, 0] + strain.s[1, 1]
    eCe = lam_cu * tr ** 2 + 2 * mu_cu * xp.einsum('ij...,ij...->...', strain.s, strain.s)
    psi_q.s[0, 0] = 0.5 * interp_dh(rho_q.s[0, 0]) * (1 - k_res) * eCe * psi_to_e0
    return cgs, C_eff, E_par


def quad_to_nodal(q, out):
    disc_pf.apply_N_transposed_operator_mugrid(quad_field_ijqxyz=q, nodal_field_inxyz=out, apply_weights=True)
    out.s[...] /= m_lumped


# ---------------------------------------------------------------- phase-field step
M0 = None


def pf_step():
    """one semi-implicit stabilised CH + AC step"""
    global M0
    r = rho.s[0, 0]
    es = [e.s[0, 0] for e in etas]
    dfr, dfe = f_derivs(r, es)
    # chemical potential mu = df/drho + k_rho (-Lap) rho + dpsi/drho
    neg_laplacian(rho, tmp_n)
    quad_to_nodal(psi_q, tmp_n2)
    mu.s[0, 0] = dfr + k_rho * tmp_n.s[0, 0] + tmp_n2.s[0, 0]
    # Wang mobility at quadrature points
    to_quad(rho, q_pf)
    rq = xp.clip(q_pf.s[0, 0], 0.0, 1.0)
    hq = interp_h(rq)
    mob = M_vol * hq + M_vap * (1 - hq) + M_surf * rq ** 2 * (1 - rq) ** 2
    s1 = 0.0
    s2 = 0.0
    for e in etas:
        to_quad(e, q_pf)
        eq = xp.clip(q_pf.s[0, 0], 0.0, 1.0)
        s1 = s1 + eq
        s2 = s2 + eq ** 2
    mob = mob + M_gb * (s1 ** 2 - s2)                             # sum_{i != j} eta_i eta_j
    q_mob.s[0, 0] = mob
    if M0 is None:
        M0 = float(to_host(mob.max()))                            # stabilisation mobility (fixed)
    stiffness(mu, tmp_n, coef=q_mob.s)                             # G^T W M G mu
    tmp_n.s[...] = rho.s - dt / m_lumped * tmp_n.s                 # rho + dt div(M grad mu)
    a_stab = 2.0 * A
    lin = M0 * (a_stab * lam + k_rho * lam ** 2)
    fft.fft(tmp_n, f_hat)
    fft.fft(rho, f_hat2)
    f_hat.s[0, 0] = (f_hat.s[0, 0] + dt * lin * f_hat2.s[0, 0]) / (1 + dt * lin)
    fft.ifft(f_hat, rho)
    rho.s[...] *= norm_fft
    # Allen-Cahn (with the old rho)
    S_ac = 12.0 * B * 2
    for e, g in zip(etas, dfe):
        tmp_n.s[0, 0] = e.s[0, 0] - dt * L_ac * g
        fft.fft(tmp_n, f_hat)
        fft.fft(e, f_hat2)
        f_hat.s[0, 0] = (f_hat.s[0, 0] + dt * L_ac * S_ac * f_hat2.s[0, 0]) / (1 + dt * L_ac * (S_ac + k_eta * lam))
        fft.ifft(f_hat, e)
        e.s[...] *= norm_fft


def diagnostics():
    r = rho.s[0, 0]
    out = dict(mean_rho=float(to_host(r.mean())), solid_fraction=float(to_host(xp.mean(r > 0.5))))
    fft.communicate_ghosts(rho)
    disc_pf.apply_gradient_operator_mugrid(u_inxyz=rho, grad_u_ijqxyz=grad_pf)
    out['surface_length'] = float(to_host(xp.mean(xp.sqrt(xp.sum(grad_pf.s[0] ** 2, axis=0))))) * cell_nm ** 2
    s1 = sum(e.s[0, 0] for e in etas)
    s2 = sum(e.s[0, 0] ** 2 for e in etas)
    out['gb_length'] = float(to_host(xp.mean(s1 ** 2 - s2))) * cell_nm ** 2 / (I_gb * 1e9)
    return out


def pore_count(r):
    from scipy import ndimage
    lab, n = ndimage.label(r < 0.5)
    for a in (0, 1):                                    # merge across the periodic boundary
        first = np.take(lab, 0, axis=a)
        last = np.take(lab, -1, axis=a)
        for p, q in zip(first, last):
            if p and q and p != q:
                lab[lab == q] = p
    return len(np.unique(lab[lab > 0]))


def snapshot(step):
    r = to_host(rho.s[0, 0]).astype(np.float32)
    grains = np.argmax(np.stack([to_host(e.s[0, 0]) for e in etas]), axis=0).astype(np.int8)
    np.savez_compressed(os.path.join(snap_file, f'step_{step:05d}.npz'), rho=r, grains=grains, t=step * dt)


# ---------------------------------------------------------------- time loop
rec = {k: [] for k in ('t', 'cg', 'cg_mean', 'time_mech', 'time_pf', 'C_eff', 'E', 'mean_rho', 'solid_fraction',
                       'surface_length', 'gb_length', 'pores', 'max_rho', 'min_rho')}
first_step = 0
restart_from = os.environ.get('RESTART_FROM')
if restart_from:
    st = np.load(restart_from, allow_pickle=True)
    rho.s[...] = xp.asarray(st['rho'])
    for i, e in enumerate(etas):
        e.s[...] = xp.asarray(st['etas'][i])
    for k in range(nv):
        u_unit[k].s[...] = xp.asarray(st['u_unit'][k])
    rec = {k: list(v) for k, v in st['rec'].item().items()}
    first_step = int(st['step']) + 1
    M0 = float(st['M0'])
    print(f'restart from {restart_from}: step {first_step}', flush=True)
C_s_v = np.array([[C_solid[i, j, k, l] for (k, l) in VOIGT] for (i, j) in VOIGT])
start = time.time()
last_checkpoint = time.time()
for step in range(first_step, n_steps + 1):
    if step % mech_every and step != n_steps:
        pf_step()
        continue
    t0 = time.time()
    cgs, C_eff, E_par = mechanics()
    t_mech = time.time() - t0
    diag = diagnostics()
    if step % snap_every == 0 or step == n_steps:
        snapshot(step)
        pores = pore_count(to_host(rho.s[0, 0]))
    else:
        pores = -1
    for k, v in (('t', step * dt), ('cg', cgs), ('cg_mean', float(np.mean(cgs))), ('time_mech', t_mech),
                 ('C_eff', C_eff / C_s_v[1, 1]), ('E', E_par), ('pores', pores),
                 ('max_rho', float(to_host(rho.s.max()))), ('min_rho', float(to_host(rho.s.min())))):
        rec[k].append(v)
    for k, v in diag.items():
        rec[k].append(v)
    if step == n_steps:
        rec['time_pf'].append(0.0)
        break
    t0 = time.time()
    pf_step()
    rec['time_pf'].append(time.time() - t0)
    if step % snap_every == 0:
        print(f'step {step:5d} t = {step * dt * 1e6:9.3f} us  CG {cgs}  C_yy/C_s = {C_eff[1, 1] / C_s_v[1, 1]:.4f}  '
              f'E_yy = {E_par[1]:.3e}  solid {diag["solid_fraction"]:.4f}  <rho> {diag["mean_rho"]:.5f}  '
              f'surface {diag["surface_length"]:.1f} nm  GB {diag["gb_length"]:.1f} nm  pores {pores}  '
              f'rho [{rec["min_rho"][-1]:.3f}, {rec["max_rho"][-1]:.3f}]  mech {t_mech:.2f} s  '
              f'pf {rec["time_pf"][-1]:.3f} s', flush=True)
    if step % snap_every == 0:
        np.savez(steps_file, N=N, preconditioner=preconditioner_type, width=width_nm, h=h, dt=dt, pressure=pressure,
                 load=load, k_res=k_res, mech_every=mech_every, **{k: np.array(v) for k, v in rec.items()})
    if time.time() - last_checkpoint > checkpoint_seconds:
        np.savez(checkpoint_file + '.tmp.npz', rho=to_host(rho.s), etas=np.array([to_host(e.s) for e in etas]),
                 u_unit=np.array([to_host(u.s) for u in u_unit]), rec=np.array(rec, dtype=object), step=step, M0=M0)
        os.replace(checkpoint_file + '.tmp.npz', checkpoint_file)
        last_checkpoint = time.time()
np.savez(steps_file, N=N, preconditioner=preconditioner_type, width=width_nm, h=h, dt=dt, pressure=pressure,
         load=load, k_res=k_res, mech_every=mech_every, **{k: np.array(v) for k, v in rec.items()})
print(f'Elapsed time: {time.time() - start:.1f} s, mean CG per elastic solve {np.mean(rec["cg_mean"]):.2f}')
