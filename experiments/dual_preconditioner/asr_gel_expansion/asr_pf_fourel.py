"""
ASR gel expansion, 2D phase field in the set-up of Fourel, Gunn, Molinari 2026 (fourel.pdf in damage_AT2_small_strain/)
====================================================================================================================

Copy of asr_AT2_mesh_study.py (2D only, CPU), extended so that every ingredient of the old model can be swapped for
the Fourel one separately (VARIANT presets below, every switch also an env variable):

  switch      old model (asr_AT2_mesh_study.py)              Fourel set-up
  ---------   --------------------------------------------   ------------------------------------------------------
  MATERIALS   falsafi: paste 12 GPa / agg 59 / gel 11 GPa    fourel (Table 5): paste 12, aggregate 60, gel 60 GPa,
              (nu 0.3/0.3/0.18), Gc 60/160 J/m^2, gel        nu 0.2, psi_c 200 / 600 / 600 J/m^3, gel damageable
              never degraded, no driving                     (aggregate properties)
  DAMAGE      AT2 (no threshold)                             AT1 (elastic threshold psi_c = 3 Gc / (16 l))
  SPLIT       hybrid: C_d = g(d) C0, Amor split only in      stress: Amor split in the stress (Fourel eq. 11),
              the driving force                              sigma = g (K <tr e>+ I + 2 mu e_D) + K <tr e>- I
  DRIVING     amor: psi+ = K/2 <tr e>+^2 + mu e_D:e_D        vol: psi+ = K/2 <tr e>+^2 (Fourel eq. 13)
  L_MM        1.5625 mm                                      0.5 mm (Fourel l_c = 0.25 mm; see README)
  SIGMA_MPA   0 (zero mean stress)                           -1 MPa in x and y ("Free", Table 6)

Fracture parameters from Fourel's threshold energy densities psi_c (fourel materials):
  AT1: homogeneous elastic limit psi+ = psi_c = 3 Gc / (16 l)  ->  Gc = 16 l psi_c / 3
  AT2 (no threshold; DAMAGE=AT2 ablation): the same peak (critical) energy 27 Gc / (512 l) = psi_c -> Gc = 512 l psi_c / 27

AT1 with a history field: minimise int g(d) H + 3 Gc / 8 (d / l + l |grad d|^2), g = (1-d)^2 + k, H = max(H_n, psi+).
The bound d >= 0 is handled by the threshold-history substitution H_eff = max(H, psi_c): the damage equation
  2 H_eff d - 3/4 Gc l Lap d = 2 (H_eff - psi_c)
is linear (same CG as AT2) with a zero right-hand side wherever psi+ < psi_c, i.e. no damage below the threshold
(exact for homogeneous states; the profile tails decay like exp(-|x| / (sqrt(2) l)) outside the band instead of the
compact AT1 support). d is clipped to [0, 1] after each damage solve (AT1 and AT2; consistent-mass round-off only).
History H = max(H, psi+) is updated over the staggered iterations as well (not max(H_n, psi+) as in the base
script): d is then non-decreasing within a load step and the loop cannot 2-cycle (it did for the brittle variants).

Elasticity with SPLIT=stress: sigma(e) = C(e) : e with C = s_vol P_vol + s_dev P_dev,
  s_vol = dim K (g if tr e > 0 else 1),  s_dev = 2 mu g      (2D plane strain: K = lam + mu, dim K = 2K)
piecewise linear and positively homogeneous, so the secant C(e) is also the (semismooth) Newton tangent. A Newton
step with the active set (sign of tr e per quadrature point) frozen is the linear problem K(C) u = -B^T C (E - e_g),
solved by the same PCG (warm start = the previous iterate: CG's initial residual is the Newton residual). The active
set is updated after every solve; the dual's C_ref C^-1 C_ref = (dim K_r)^2 / s_vol P_vol + (2 mu_r)^2 / s_dev P_dev
stays exact for the switching tangent.

Mean-stress control (macro strain E unknown, <sigma(E + grad u - e_g)> = sigma_bar): superposition is invalid with
the nonlinear stress, so E is a quasi-Newton unknown of the same fixed-point loop: after every solve
E <- E - C_eff^-1 (<sigma> - sigma_bar), with C_eff = the effective tangent stiffness from 3 unit-strain solves with
the frozen (damage, active set) material at the end of the previous load step ('cheap' control of the base script).
One loop iteration = 1 elastic solve (current E, damage, active set) + E update + active-set update + 1 damage solve;
the load step is converged when max |d - d_old| < 1e-3, |<sigma> - sigma_bar| < TOL_S |C_eff E| (or < TOL_S |sigma_bar|)
and no quadrature point changed its active set.

DAMAGE=PFCZM (VARIANT=czm, 'real concrete'): Wu's PF-CZM exactly as ../localization_softening_bands/pfczm_concrete.py
(alpha = 2d - d^2, c0 = pi, omega = (1-d)^2 / ((1-d)^2 + a1 d (1 + a2 d + a2 a3 d^2)), Cornelissen a2 = 1.3868,
a3 = 0.6567, a1 = 4 l_ch / (pi b), b = L_MM, projected convexified Newton with d_n <= d <= 1, Green-Jacobi PCG,
energy backtracking), but with the driving force of this script (volumetric tension, DRIVING) instead of Wu's
Rankine sigma_eq, and the stress split. Gf paste GF_PASTE (60 J/m^2), aggregate and gel 3x; ft paste FT_PASTE_MPA (3);
onset H > psi0 = ft^2 / (2 E), the aggregate's psi0 = 3 x the paste's (Fourel's psi_c ratio) -> ft_agg = 11.6 MPa.

Load: gel eigenstrain eps_ASR I (in-plane), eps_ASR 0 -> EPS_MAX in LOAD_STEPS steps (Fourel: 0 -> 5 % in steps of
0.025 %). Each gel pocket's eigenstrain is scaled by (true pocket area / its pixel area) so the expansion volume is the
same on every grid (factor 0.7-1.5 at 64^2, ~1 at 256^2).

Geometry geometry/fourel2D_s1.npz (make_fourel_geometry_2D.py): 35 mm periodic cell, 91 circular aggregates, 28 gel
pockets d = 0.75 mm; rasterised at the pixel lower-left nodes. Length unit = cell (35 mm), stress unit 12 GPa.

Output exp_data/ASRpf_<variant>_N<N>_<P>{_suffix}: _iterations.csv (every loop iteration), _steps.npz (per step:
eps_ASR, E, stiffness ratio, max/mean d, CG, times), _damage/step_<k>.npy (float32 nodal damage every step).

    python asr_pf_fourel.py <Green|Green_Jacobi|dual>
env: VARIANT (fourel | czm | old | hybrid | at2 | amor | softgel), NPIX (128), EPS_MAX (0.015), LOAD_STEPS (60),
     MATERIALS, DAMAGE, SPLIT, DRIVING, L_MM, SIGMA_MPA, K_RESIDUAL (1e-4), TOL_S (1e-4), MAX_STAGGERED (500),
     RUN_SUFFIX, GEOMETRY_NPZ
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
sys.path.insert(0, here)

dim = int(os.environ.get('DIM', 2))
device = os.environ.get('DEVICE', 'cpu')
if device == 'gpu':
    import cupy as xp
else:
    xp = np
to_host = tensor_operations.to_host
if dim == 2:
    from make_fourel_geometry_2D import rasterise  # noqa: E402
else:
    from make_fourel_geometry_3D import rasterise  # noqa: E402
N = int(os.environ.get('NPIX', 128))
variant = os.environ.get('VARIANT', 'fourel')
PRESETS = {   # (MATERIALS, DAMAGE, SPLIT, DRIVING, L_MM, SIGMA_MPA)
    'fourel': ('fourel', 'AT1', 'stress', 'vol', 0.5, -1.0),    # full Fourel set-up
    'old': ('falsafi', 'AT2', 'hybrid', 'amor', 1.5625, 0.0),     # asr_AT2_mesh_study.py on the new geometry
    'hybrid': ('fourel', 'AT1', 'hybrid', 'vol', 0.5, -1.0),    # ablation: no split in the stress
    'at2': ('fourel', 'AT2', 'stress', 'vol', 0.5, -1.0),       # ablation: no threshold
    'amor': ('fourel', 'AT1', 'stress', 'amor', 0.5, -1.0),     # ablation: standard Amor psi+ (vol + dev)
    'softgel': ('fourel_softgel', 'AT1', 'stress', 'vol', 0.5, -1.0),  # ablation: gel 11 GPa, never damaged
    'czm': ('fourel', 'PFCZM', 'stress', 'vol', 0.5, -1.0),     # (B) Wu PF-CZM, real concrete Gf and ft
    'czmR': ('fourel', 'PFCZM', 'stress', 'rankine', 0.5, -1.0),  # B with Wu's Rankine-type driving force
    'fourelR': ('fourel', 'AT1', 'stress', 'rankine', 0.5, -1.0),  # A with the Rankine-type driving force
}
p_mat, p_dam, p_split, p_drv, p_l, p_sig = PRESETS[variant]
materials = os.environ.get('MATERIALS', p_mat)
damage_model = os.environ.get('DAMAGE', p_dam)
split = os.environ.get('SPLIT', p_split)
driving = os.environ.get('DRIVING', p_drv)
l_mm = float(os.environ.get('L_MM', p_l))
sigma_mpa = float(os.environ.get('SIGMA_MPA', p_sig))
assert damage_model in ('AT1', 'AT2', 'PFCZM') and split in ('hybrid', 'stress') and driving in ('amor', 'vol', 'rankine')
geometry_npz = os.path.join(here, os.environ.get('GEOMETRY_NPZ', f'geometry/fourel{dim}D_s1.npz'))
cell_mm = float(np.load(geometry_npz)['cell_mm'])
E_unit, L_unit = 12e9, cell_mm * 1e-3
length_scale = l_mm / cell_mm
k_residual = float(os.environ.get('K_RESIDUAL', 1e-4))
eps_max = float(os.environ.get('EPS_MAX', 0.015))
load_steps = np.linspace(0.0, eps_max, int(os.environ.get('LOAD_STEPS', 60)) + 1)[1:]
staggered_tol = 1e-3
max_staggered = int(os.environ.get('MAX_STAGGERED', 500))
tol_s = float(os.environ.get('TOL_S', 1e-4))
cg_tol, cg_delay = 1e-6, 10
snap_full = int(os.environ.get('SNAP_FULL', 10))                 # 3D: full damage field every SNAP_FULL steps
checkpoint_seconds = float(os.environ.get('CHECKPOINT_SECONDS', 3600))
restart_from = os.environ.get('RESTART_FROM')
preconditioner_type = sys.argv[1] if len(sys.argv) > 1 else 'dual'
assert preconditioner_type in ('Green', 'Green_Jacobi', 'dual')


def gc_from_psi_c(psi_c):
    """Gc [J/m^2] with the same threshold / critical energy density psi_c [J/m^3] at length l"""
    l_m = l_mm * 1e-3
    return 16.0 * l_m * psi_c / 3.0 if damage_model == 'AT1' else 512.0 * l_m * psi_c / 27.0


if damage_model == 'PFCZM':      # real concrete fracture energies (ratio 3 as Fourel), energy-consistent ft
    GF_PASTE, FT_PASTE = float(os.environ.get('GF_PASTE', 60.0)), float(os.environ.get('FT_PASTE_MPA', 3.0)) * 1e6
    psi0_paste = FT_PASTE ** 2 / (2 * 12e9)                  # threshold energy density ft^2 / (2 E) of the paste
    FT_AGG = np.sqrt(2 * 60e9 * 3.0 * psi0_paste)            # psi0_agg = 3 psi0_paste -> ft_agg = 11.6 MPa
    PHASES = {0: dict(E=12e9, nu=0.2, Gc=GF_PASTE, ft=FT_PASTE, damageable=True),
              1: dict(E=60e9, nu=0.2, Gc=3 * GF_PASTE, ft=FT_AGG, damageable=True),
              2: dict(E=60e9, nu=0.2, Gc=3 * GF_PASTE, ft=FT_AGG, damageable=True)}
elif materials == 'falsafi':        # Falsafi et al. Table 2 (the old model)
    PHASES = {0: dict(E=12e9, nu=0.30, Gc=60.0, damageable=True),
              1: dict(E=59e9, nu=0.30, Gc=160.0, damageable=True),
              2: dict(E=11e9, nu=0.18, Gc=160.0, damageable=False)}
else:                             # Fourel Table 5
    PHASES = {0: dict(E=12e9, nu=0.2, Gc=gc_from_psi_c(200.0), damageable=True),
              1: dict(E=60e9, nu=0.2, Gc=gc_from_psi_c(600.0), damageable=True),
              2: dict(E=60e9, nu=0.2, Gc=gc_from_psi_c(600.0), damageable=True)}
    if materials == 'fourel_softgel':
        PHASES[2] = dict(E=11e9, nu=0.18, Gc=gc_from_psi_c(600.0), damageable=False)
run_name = (f'ASRpf_{variant}_N{N}_{preconditioner_type}' if dim == 2 else
            f'ASRpf_{variant}_3D_N{N}_{preconditioner_type}') + os.environ.get('RUN_SUFFIX', '')
out_dir = os.environ.get('OUT_DIR', os.path.join(here, 'exp_data'))
iterations_file = os.path.join(out_dir, run_name + '_iterations.csv')
steps_file = os.path.join(out_dir, run_name + '_steps.npz')
snap_dir = os.path.join(out_dir, run_name + '_damage')
checkpoint_file = os.path.join(out_dir, run_name + '_checkpoint.npz')


def plane_strain_lame(E, nu):          # Lame constants (2D: plane strain)
    return E * nu / ((1 + nu) * (1 - 2 * nu)), E / (2 * (1 + nu))


element = 'bilinear_rectangle' if dim == 2 else 'trilinear_hexahedron'
cell_u = domain.PeriodicUnitCell(domain_size=[1.0] * dim, problem_type='elasticity')
disc_u = domain.Discretization(cell=cell_u, nb_of_pixels_global=(N,) * dim, discretization_type='finite_element',
                               element_type=element, device=device)
cell_d = domain.PeriodicUnitCell(domain_size=[1.0] * dim, problem_type='conductivity')
disc_d = domain.Discretization(cell=cell_d, nb_of_pixels_global=(N,) * dim, discretization_type='finite_element',
                               element_type=element, device=device)
comm = disc_u.communicator

# ---- geometry: labels at the pixel lower-left nodes; per-pocket eigenstrain factor (true area / pixel area) so the
# expansion volume is grid-independent; a pocket that hits no node gets its nearest node (coarse 3D grids)
label = rasterise(geometry_npz, N).astype(np.int8)
geo = np.load(geometry_npz)
eps_factor_h = np.zeros((N,) * dim)
nb_point_pockets = 0
for g in geo['gels']:
    c, r = np.asarray(g[:dim]), g[dim]
    m = int(np.ceil(r * N)) + 1
    idx = [np.arange(int(np.floor(c[a] * N)) - m, int(np.floor(c[a] * N)) + m + 2) for a in range(dim)]
    grids = np.meshgrid(*idx, indexing='ij')
    dist2 = sum(((gi / N - c[a] + 0.5) % 1.0 - 0.5) ** 2 for a, gi in enumerate(grids))
    sel = tuple(gi % N for gi in grids)
    pocket = (dist2 < r * r) & (label[sel] == 2)
    vol = (np.pi * r * r if dim == 2 else 4.0 / 3.0 * np.pi * r ** 3) * N ** dim     # true volume in pixels
    if pocket.sum() == 0:
        node = tuple(int(np.round(c[a] * N)) % N for a in range(dim))
        label[node] = 2
        eps_factor_h[node] = vol
        nb_point_pockets += 1
        continue
    eps_factor_h[tuple(s_[pocket] for s_ in sel)] = vol / pocket.sum()
gel_h = label == 2
if nb_point_pockets:
    print(f'{nb_point_pockets} gel pockets hit no node: one pixel each (eigenstrain x true volume)', flush=True)

lam_h, mu_h, Gc_h, drv_h, ft_h, E_h = (np.zeros(label.shape) for _ in range(6))
for p, prm in PHASES.items():
    lp, mp = plane_strain_lame(prm['E'] / E_unit, prm['nu'])
    lam_h[label == p], mu_h[label == p] = lp, mp
    ft_h[label == p], E_h[label == p] = prm.get('ft', 0.0) / E_unit, prm['E'] / E_unit
    Gc_h[label == p] = prm['Gc'] / (E_unit * L_unit)
    drv_h[label == p] = 1.0 if prm['damageable'] else 0.0
undamageable = xp.asarray(drv_h == 0.0)
lam_x, mu_x, Gc_x = xp.asarray(lam_h), xp.asarray(mu_h), xp.asarray(Gc_h)
kappa_x = lam_x + 2.0 * mu_x / dim                            # bulk modulus (2D plane strain: lam + mu)
crack_driving = xp.asarray(drv_h)
eps_factor = xp.asarray(eps_factor_h)
lam_r, mu_r = plane_strain_lame(PHASES[0]['E'] / E_unit, PHASES[0]['nu'])   # reference: undamaged paste
kappa_r = lam_r + 2.0 * mu_r / dim
Gc_ref = PHASES[0]['Gc'] / (E_unit * L_unit)
# AT1: c_w = 8/3, Gc/c_w (d/l + l |grad d|^2); psi_c = 3 Gc / (16 l) per pixel
if damage_model == 'AT1':
    diff_coef = 0.75 * Gc_x * length_scale                    # 3/4 Gc l
    diff_ref = 0.75 * Gc_ref * length_scale
    psi_c_x = 3.0 * Gc_x / (16.0 * length_scale)
elif damage_model == 'PFCZM':
    # Wu (2017) PF-CZM, as ../localization_softening_bands/pfczm_concrete.py: crack band b = l, Cornelissen softening,
    # a1 = 4 l_ch / (pi b), l_ch = E Gf / ft^2; onset where the history H > psi0 = ft^2 / (2 E) (H starts at psi0)
    b_cz = length_scale
    E_x, ft_x = xp.asarray(E_h), xp.asarray(ft_h)
    czm_p, czm_a2, czm_a3 = 2.0, 1.3868, 0.6567
    a1 = 4.0 * (E_x * Gc_x / ft_x ** 2) / (np.pi * b_cz)
    c_diff, c_loc = 2.0 * Gc_x * b_cz / np.pi, 2.0 * Gc_x / (np.pi * b_cz)
    c_diff_ref, c_loc_ref = 2.0 * Gc_ref * b_cz / np.pi, 2.0 * Gc_ref / (np.pi * b_cz)
    psi_c_x = ft_x ** 2 / (2 * E_x)
    diff_coef = diff_ref = None
    wq = xp.asarray(np.asarray(to_host(disc_d.quadrature_weights))).reshape((-1,) + (1,) * dim)
else:
    diff_coef, diff_ref = Gc_x * length_scale, Gc_ref * length_scale
    psi_c_x = None
sigma_bar = sigma_mpa * 1e6 / E_unit
print(f'variant {variant}: materials {materials}, damage {damage_model}, split {split}, driving {driving}, '
      f'l = {l_mm:g} mm (l/h = {l_mm / (cell_mm / N):.3g}), sigma_bar = {sigma_mpa:g} MPa, k = {k_residual:g}; '
      f'{dim}D grid {N}^{dim} (h = {cell_mm / N:.4g} mm), paste {np.mean(label == 0):.3f} aggregate {np.mean(label == 1):.3f} '
      f'gel {np.mean(gel_h):.4f}; Gc [J/m^2] = ' + ', '.join(f'{PHASES[p]["Gc"]:.4g}' for p in PHASES)
      + (('; ft [MPa] = ' + ', '.join(f'{PHASES[p]["ft"] / 1e6:.4g}' for p in PHASES)
          + '; l_ch [mm] = ' + ', '.join(f'{1e3 * PHASES[p]["E"] * PHASES[p]["Gc"] / PHASES[p]["ft"] ** 2:.4g}'
                                         for p in PHASES)) if damage_model == 'PFCZM' else
         ('; psi_c [J/m^3] = ' + ', '.join(f'{3 * PHASES[p]["Gc"] / (16 * l_mm * 1e-3):.4g}' for p in PHASES))
         if damage_model == 'AT1' else '')
      + f'; preconditioner {preconditioner_type}, device {device}', flush=True)

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
sig_target = np.array([sigma_bar if i == j else 0.0 for (i, j) in VOIGT])


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
tension = xp.zeros(s_vol.s[0, 0].shape, dtype=bool)          # active set: tr(e) > 0 per quadrature point


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


def apply_damage_operator(input_field_inxyz, output_field_inxyz, reaction_q, diff=None):
    disc_d.apply_gradient_operator_mugrid(u_inxyz=input_field_inxyz, grad_u_ijqxyz=grad_d)
    grad_d.s[...] *= diff_coef if diff is None else diff
    disc_d.apply_gradient_transposed_operator_mugrid(gradient_field_ijqxyz=grad_d, div_u_fnxyz=output_field_inxyz,
                                                     apply_weights=True)
    disc_d.apply_N_operator_mugrid(input_field_inxyz, temp_q)
    temp_q.s[...] *= reaction_q
    disc_d.apply_N_transposed_operator_mugrid(quad_field_ijqxyz=temp_q, nodal_field_inxyz=temp_n, apply_weights=True)
    output_field_inxyz.s[...] += temp_n.s


def K_d(input_field_inxyz, output_field_inxyz):
    apply_damage_operator(input_field_inxyz, output_field_inxyz, reaction.s)


reaction_ref = {'AT2': Gc_ref / length_scale, 'AT1': 3.0 * Gc_ref / (8.0 * length_scale)}.get(damage_model)
if damage_model == 'PFCZM':
    reaction_ref, diff_ref = c_loc_ref, c_diff_ref
# Green of the homogeneous damage operator with the paste's reaction (AT2: Gc/l, AT1: 2 psi_c = 3 Gc/(8 l),
# PF-CZM: the paste's c_loc, c_diff)
green_d = disc_d.get_preconditioner_Green_mugrid(
    reference_material_data_ijkl=None,
    operator=lambda input_field_inxyz, output_field_inxyz: apply_damage_operator(
        input_field_inxyz, output_field_inxyz,
        reaction_ref, diff=diff_ref),
    invert_zero_mode=True, name='green_d')
x_scaled_d = disc_d.get_unknown_size_field(name='x_scaled_d')
jacobi_d = None


def M_d(x, Px):
    x_scaled_d.s[...] = jacobi_d.s * x.s
    disc_d.apply_preconditioner_mugrid(preconditioner_Fourier_fnfnqks=green_d, input_nodal_field_fnxyz=x_scaled_d,
                                       output_nodal_field_fnxyz=Px)
    Px.s[...] *= jacobi_d.s


def update_material():
    """C = s_vol P_vol + s_dev P_dev from the current damage (and, SPLIT=stress, the active set `tension`)"""
    disc_d.apply_N_operator_mugrid(damage, damage_q)
    if damage_model == 'PFCZM':
        g = xp.where(undamageable, 1.0, omega_and_derivatives(damage_q.s[0, 0], False) + k_residual)
    else:
        g = xp.where(undamageable, 1.0, (1.0 - damage_q.s[0, 0]) ** 2 + k_residual)
    g_vol = g if split == 'hybrid' else xp.where(tension, g, 1.0)
    s_vol.s[0, 0] = g_vol * dim * kappa_x
    s_dev.s[0, 0] = g * 2 * mu_x


# ---------------------------------------------------------------- PF-CZM damage (copied from pfczm_concrete.py)
def omega_and_derivatives(d, derivatives=True):
    d = xp.clip(d, 0.0, 1.0)
    A = (1.0 - d) ** czm_p
    Q = a1 * (d + czm_a2 * d ** 2 + czm_a2 * czm_a3 * d ** 3)
    S = A + Q
    if not derivatives:
        return A / S
    A1 = -czm_p * (1.0 - d) ** (czm_p - 1)
    A2 = 2.0
    Q1 = a1 * (1 + 2 * czm_a2 * d + 3 * czm_a2 * czm_a3 * d ** 2)
    Q2 = a1 * (2 * czm_a2 + 6 * czm_a2 * czm_a3 * d)
    num = A1 * Q - A * Q1
    w1 = num / S ** 2
    w2 = ((A2 * Q - A * Q2) * S - 2 * num * (A1 + Q1)) / S ** 3
    return w1, w2


if damage_model == 'PFCZM':
    trial = disc_d.get_unknown_size_field(name='damage_trial')
    grad_n = disc_d.get_unknown_size_field(name='energy_gradient')
    delta = disc_d.get_unknown_size_field(name='newton_step')
    masked = disc_d.get_unknown_size_field(name='masked')
    czm = dict(hess_q=None, active=xp.ones((N,) * dim), d_n=xp.zeros((N,) * dim))


def czm_diffusion_mass(x, Ax, local_q, cd):
    disc_d.apply_gradient_operator_mugrid(u_inxyz=x, grad_u_ijqxyz=grad_d)
    grad_d.s[...] *= cd
    disc_d.apply_gradient_transposed_operator_mugrid(gradient_field_ijqxyz=grad_d, div_u_fnxyz=Ax, apply_weights=True)
    disc_d.apply_N_operator_mugrid(x, temp_q)
    temp_q.s[0, 0] *= local_q
    disc_d.apply_N_transposed_operator_mugrid(quad_field_ijqxyz=temp_q, nodal_field_inxyz=temp_n, apply_weights=True)
    Ax.s[...] += temp_n.s


def czm_energy(d_field, H):
    disc_d.apply_N_operator_mugrid(d_field, damage_q)
    dq = damage_q.s[0, 0]
    disc_d.apply_gradient_operator_mugrid(u_inxyz=d_field, grad_u_ijqxyz=grad_d)
    g2 = xp.sum(grad_d.s[0] ** 2, axis=0)
    dens = (Gc_x / (np.pi * b_cz)) * (2 * dq - dq ** 2 + b_cz ** 2 * g2) + omega_and_derivatives(dq, False) * H
    return float(xp.sum(wq * dens))


def czm_K_newton(x, Ax):
    act = czm['active']
    masked.s[0, 0] = act * x.s[0, 0]
    czm_diffusion_mass(masked, Ax, czm['hess_q'], c_diff)
    Ax.s[0, 0] = act * Ax.s[0, 0] + (1.0 - act) * x.s[0, 0]


def czm_M(x, Px):
    act = czm['active']
    x_scaled_d.s[0, 0] = act * jacobi_d.s[0, 0] * x.s[0, 0]
    disc_d.apply_preconditioner_mugrid(preconditioner_Fourier_fnfnqks=green_d, input_nodal_field_fnxyz=x_scaled_d,
                                       output_nodal_field_fnxyz=Px)
    Px.s[0, 0] = act * jacobi_d.s[0, 0] * Px.s[0, 0] + (1.0 - act) * x.s[0, 0]


def czm_solve_damage(H, newton_tol=1e-6, max_newton=50):
    """projected (convexified) Newton for min Pi_d on d_n <= d <= 1 (d_n: damage of the last load step);
    returns the CG iterations"""
    global jacobi_d
    nb_cg = 0
    E_cur = czm_energy(damage, H)
    for it in range(max_newton):
        disc_d.apply_N_operator_mugrid(damage, damage_q)
        dq = damage_q.s[0, 0]
        w1, w2 = omega_and_derivatives(dq)
        czm['hess_q'] = xp.maximum(w2 * H - c_loc, 0.1 * c_loc)
        temp_q.s[0, 0] = c_loc * (1 - dq) + w1 * H
        disc_d.apply_N_transposed_operator_mugrid(quad_field_ijqxyz=temp_q, nodal_field_inxyz=temp_n,
                                                  apply_weights=True)
        disc_d.apply_gradient_operator_mugrid(u_inxyz=damage, grad_u_ijqxyz=grad_d)
        grad_d.s[...] *= c_diff
        disc_d.apply_gradient_transposed_operator_mugrid(gradient_field_ijqxyz=grad_d, div_u_fnxyz=grad_n,
                                                         apply_weights=True)
        grad_n.s[...] += temp_n.s
        d, g = damage.s[0, 0], grad_n.s[0, 0]
        at_bound = ((d <= czm['d_n'] + 1e-12) & (g > 0)) | ((d >= 1.0 - 1e-12) & (g < 0))
        czm['active'] = xp.where(at_bound, 0.0, 1.0)
        rhs_d.s[0, 0] = -czm['active'] * g
        if float(xp.max(xp.abs(rhs_d.s))) == 0.0:
            return nb_cg
        jacobi_d = disc_d.get_preconditioner_Jacobi_mugrid(
            operator=lambda input_field_inxyz, output_field_inxyz: czm_K_newton(input_field_inxyz, output_field_inxyz),
            name='jacobi_d')
        delta.s[...] = 0.0
        norms = solvers.conjugate_gradients_mugrid_experimental(
            comm=comm, fc=disc_d.field_collection, hessp=czm_K_newton, b=rhs_d, x=delta, P=czm_M, maxiter=5000,
            stop_crit_norm='energy_lower_estim', d_min=cg_delay,
            **topology_optimization.energy_stop_kwargs(disc_d, czm_K_newton, delta, cg_tol))[1]
        nb_cg += int(norms.get('stop_iteration', {}).get('energy_lower_estim', 0))
        t = 1.0
        for _ in range(30):
            trial.s[0, 0] = xp.clip(d + t * czm['active'] * delta.s[0, 0], czm['d_n'], 1.0)
            E_trial = czm_energy(trial, H)
            if E_trial <= E_cur + 1e-4 * float(xp.sum(g * (trial.s[0, 0] - d))) or t < 1e-6:
                break
            t *= 0.5
        step_size = float(xp.max(xp.abs(trial.s[0, 0] - d)))
        damage.s[...] = trial.s
        E_cur = E_trial
        if step_size < newton_tol:
            return nb_cg
    return nb_cg


def strain_split(eps):
    trace = xp.einsum('ii...->...', eps)
    eps_dev = eps - trace / dim * xp.eye(dim).reshape(dim, dim, *([1] * trace.ndim))
    return trace, eps_dev


def driving_energy(eps):
    if driving == 'rankine' and dim == 3:
        tr = eps[0, 0] + eps[1, 1] + eps[2, 2]
        sg = [[lam_x * tr * (i == j) + 2 * mu_x * eps[i, j] for j in range(3)] for i in range(3)]
        q = (sg[0][0] + sg[1][1] + sg[2][2]) / 3
        p1 = sg[0][1] ** 2 + sg[0][2] ** 2 + sg[1][2] ** 2
        p2 = (sg[0][0] - q) ** 2 + (sg[1][1] - q) ** 2 + (sg[2][2] - q) ** 2 + 2 * p1
        pp = xp.sqrt(p2 / 6) + 1e-300
        b00, b11, b22 = (sg[0][0] - q) / pp, (sg[1][1] - q) / pp, (sg[2][2] - q) / pp
        b01, b02, b12 = sg[0][1] / pp, sg[0][2] / pp, sg[1][2] / pp
        detB = b00 * (b11 * b22 - b12 ** 2) - b01 * (b01 * b22 - b12 * b02) + b02 * (b01 * b12 - b11 * b02)
        s1 = q + 2 * pp * xp.cos(xp.arccos(xp.clip(detB / 2, -1.0, 1.0)) / 3)       # largest principal stress
        seq = (9.0 * xp.maximum(s1, 0.0) + xp.sqrt(1.5 * p2)) / 10.0               # sqrt(3 J2), J2 = p2 / 2
        E_young = mu_x * (3 * lam_x + 2 * mu_x) / (lam_x + mu_x)
        return seq ** 2 / (2 * E_young)
    if driving == 'rankine':
        # Wu (2017) / pfczm_concrete.py: Y = sigma_eq^2 / (2 E) of the undamaged stress (plane strain sigma_zz),
        # sigma_eq = (beta_c <sigma_1> + sqrt(3 J2)) / (1 + beta_c), beta_c = 9; sees the hoop tension around an
        # expanding pocket (where tr e = 0)
        tr = eps[0, 0] + eps[1, 1]
        sxx = lam_x * tr + 2 * mu_x * eps[0, 0]
        syy = lam_x * tr + 2 * mu_x * eps[1, 1]
        sxy = 2 * mu_x * eps[0, 1]
        szz = lam_x * tr
        s1 = xp.maximum(0.5 * (sxx + syy) + xp.sqrt(0.25 * (sxx - syy) ** 2 + sxy ** 2), szz)
        mean = (sxx + syy + szz) / 3
        J2 = 0.5 * ((sxx - mean) ** 2 + (syy - mean) ** 2 + (szz - mean) ** 2) + sxy ** 2
        seq = (9.0 * xp.maximum(s1, 0.0) + xp.sqrt(3 * J2)) / 10.0
        E_young = mu_x * (3 * lam_x + 2 * mu_x) / (lam_x + mu_x)
        return seq ** 2 / (2 * E_young)
    trace, eps_dev = strain_split(eps)
    psi = 0.5 * kappa_x * xp.maximum(trace, 0.0) ** 2
    if driving == 'amor':
        psi = psi + mu_x * xp.einsum('ij...,ij...->...', eps_dev, eps_dev)
    return psi


def solve_u(x):
    norms = solvers.conjugate_gradients_mugrid_experimental(
        comm=comm, fc=disc_u.field_collection, hessp=K_u, b=rhs_u, x=x, P=M_u, maxiter=50000,
        stop_crit_norm='energy_lower_estim', d_min=cg_delay,
        **topology_optimization.energy_stop_kwargs(disc_u, K_u, x, cg_tol))[1]
    return int(norms.get('stop_iteration', {}).get('energy_lower_estim', 0))   # 0: zero right-hand side


def mean_stress(u):
    return disc_u.get_homogenized_stress_mugrid(material_data_field_ijklqxyz=C_d, displacement_field_inxyz=u,
                                                macro_gradient_field_ijqxyz=macro, formulation='small_strain')


def set_macro(E_ij, eps_g):
    """macro = E - eps_g I (gel only, per-pocket area factor)"""
    disc_u.get_macro_gradient_field_mugrid(macro_gradient_ij=E_ij, macro_gradient_field_ijqxyz=macro)
    for i in range(dim):
        macro.s[i, i] -= eps_g * eps_factor


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


# ---------------------------------------------------------------------------- state
os.makedirs(snap_dir, exist_ok=True)
rec = {k: [] for k in ('eps_asr', 'E', 'stiffness_ratio', 'C_eff', 'max_d', 'mean_d', 'staggered', 'converged',
                       'cg_main_mean', 'cg_main_sum', 'cg_all_sum', 'nb_solves', 'cg_d_mean', 'time_step',
                       'time_u', 'time_d')}
first_step = 0
if restart_from:          # state at the end of a load step (checkpoint written by this script)
    state = np.load(restart_from, allow_pickle=True)
    damage.s[...] = xp.asarray(state['damage'])
    history_n.s[...] = xp.asarray(state['history_n'])
    history.s[...] = history_n.s
    u_tot.s[...] = xp.asarray(state['u_tot'])
    for k in range(nv):
        u_unit[k].s[...] = xp.asarray(state['u_unit'][k])
    tension[...] = xp.asarray(state['tension'])
    E_par, M_eff, C0_norm = state['E_par'], state['M_eff'], float(state['C0_norm'])
    rec = {k: list(v) for k, v in state['rec'].item().items()}
    cg_total_all = int(state['cg_total_all'])
    first_step = int(state['step']) + 1
    update_material()
    print(f'restart from {restart_from}: load step {first_step}', flush=True)
else:
    with open(iterations_file, 'w') as f:
        f.write('step,eps_asr,staggered_iteration,cg_main,cg_d,time_u,time_d,damage_change,max_d,mean_stress_rel,'
                'flips,' + ','.join(f'E{k}' for k in range(nv)) + '\n')
    if damage_model == 'PFCZM':
        history.s[0, 0] = psi_c_x                   # H starts at psi0 = ft^2 / (2 E): no damage below it
    update_material()
    refresh_jacobi()
    t0 = time.time()
    M_eff, cg_unit0 = unit_solves()
    C0_norm = mandel_norm(M_eff)
    E_par = np.linalg.solve(M_eff, sig_target)         # undamaged, linear: the mean-stress state at eps_ASR = 0
    print(f'initial unit solves: CG {cg_unit0}, {time.time() - t0:.1f} s; E0 = {E_par}', flush=True)
    cg_total_all = sum(cg_unit0)
start = time.time()
last_checkpoint = time.time()
nb_not_converged = 0


def save_field(name, field):
    np.save(os.path.join(snap_dir, name), to_host(field).astype(np.float32))


for step in range(first_step, len(load_steps)):
    eps_g = load_steps[step]
    t_step = time.time()
    cg_main_step, cg_d_step, time_u_step, time_d_step = 0, 0, 0.0, 0.0
    if damage_model == 'PFCZM':
        czm['d_n'] = damage.s[0, 0].copy()           # irreversibility d >= d_n
    for stagger in range(max_staggered):
        t0 = time.time()
        refresh_jacobi()
        set_macro(from_par(E_par), eps_g)
        cg_main = solve_with_macro(u_tot)
        sig = voigt(mean_stress(u_tot))
        res = sig - sig_target
        stress_rel = np.abs(res).max() / max(np.abs(M_eff @ E_par).max(), np.abs(sig_target).max(), 1e-300)
        disc_u.fft.communicate_ghosts(u_tot)
        disc_u.apply_gradient_operator_symmetrized_mugrid(u_inxyz=u_tot, grad_u_ijqxyz=strain)
        strain.s[...] += macro.s                               # elastic strain E + grad u - eps_g
        E_par = E_par - np.linalg.solve(M_eff, res)
        time_u = time.time() - t0

        trace = xp.einsum('ii...->...', strain.s)              # [q, x, y(, z)]
        new_tension = trace > 0.0
        flips = int(xp.sum(new_tension != tension)) if split == 'stress' else 0
        tension[...] = new_tension

        # history monotone also over the staggered iterations (H starts the step at H_n): d is non-decreasing in
        # the loop, so the loop converges (with H = max(H_n, psi+) the brittle variants 2-cycle)
        history.s[0, 0] = xp.maximum(history.s[0, 0], crack_driving * driving_energy(strain.s))
        if damage_model == 'AT1':
            h_eff = xp.maximum(history.s[0, 0], psi_c_x)
            reaction.s[0, 0] = 2.0 * h_eff
            temp_q.s[0, 0] = 2.0 * (h_eff - psi_c_x)
        elif damage_model == 'AT2':
            reaction.s[...] = Gc_x / length_scale + 2.0 * history.s
            temp_q.s[...] = 2.0 * history.s
        damage_old.s[...] = damage.s
        t0 = time.time()
        if damage_model == 'PFCZM':
            cg_d = czm_solve_damage(history.s[0, 0])
        else:
            disc_d.apply_N_transposed_operator_mugrid(quad_field_ijqxyz=temp_q, nodal_field_inxyz=rhs_d,
                                                      apply_weights=True)
            jacobi_d = disc_d.get_preconditioner_Jacobi_mugrid(operator=K_d, name='jacobi_d')
            norms_d = solvers.conjugate_gradients_mugrid_experimental(
                comm=comm, fc=disc_d.field_collection, hessp=K_d, b=rhs_d, x=damage, P=M_d, maxiter=50000,
                stop_crit_norm='energy_lower_estim', d_min=cg_delay,
                **topology_optimization.energy_stop_kwargs(disc_d, K_d, damage, cg_tol))[1]
            cg_d = int(norms_d.get('stop_iteration', {}).get('energy_lower_estim', 0))
            damage.s[...] = xp.clip(damage.s, 0.0, 1.0)
        time_d = time.time() - t0
        update_material()
        change = fmax(xp.abs(damage.s - damage_old.s))
        d_max = fmax(damage.s)
        cg_main_step += cg_main
        cg_d_step += cg_d
        time_u_step += time_u
        time_d_step += time_d
        with open(iterations_file, 'a') as f:
            f.write(f'{step},{eps_g:.6g},{stagger + 1},{cg_main},{cg_d},{time_u:.4g},{time_d:.4g},'
                    f'{change:.4g},{d_max:.4g},{stress_rel:.3g},{flips},' + ','.join(f'{v:.6g}' for v in E_par) + '\n')
        converged = change < staggered_tol and stress_rel < tol_s and flips == 0
        if converged:
            break
    nb_not_converged += not converged
    history_n.s[...] = history.s
    refresh_jacobi()                           # C_eff of the converged state: next step's E update and the output
    M_eff, cgs = unit_solves()
    cg_unit = sum(cgs)
    stiff = mandel_norm(M_eff) / C0_norm
    cg_total_all += cg_main_step + cg_unit
    for k, v in (('eps_asr', eps_g), ('E', E_par.copy()), ('stiffness_ratio', stiff), ('C_eff', M_eff.copy()),
                 ('max_d', d_max), ('mean_d', float(to_host(xp.mean(damage.s)))), ('staggered', stagger + 1),
                 ('converged', int(converged)), ('cg_main_mean', cg_main_step / (stagger + 1)),
                 ('cg_main_sum', cg_main_step), ('cg_all_sum', cg_main_step + cg_unit),
                 ('nb_solves', stagger + 1 + nv), ('cg_d_mean', cg_d_step / (stagger + 1)),
                 ('time_step', time.time() - t_step), ('time_u', time_u_step), ('time_d', time_d_step)):
        rec[k].append(v)
    print(f'step {step:3d}  eps_ASR = {100 * eps_g:.3f} %  tr E = {100 * sum(E_par[:dim]):.4f} %  '
          f'|C|/|C0| = {stiff:.4f}  max d = {d_max:.3f}  mean d = {rec["mean_d"][-1]:.4f}  '
          f'its = {stagger + 1:3d}  mean CG u = {cg_main_step / (stagger + 1):.1f} (unit {cgs})  '
          f'mean CG d = {cg_d_step / (stagger + 1):.1f}  <sigma> rel = {stress_rel:.1e}  flips {flips}  '
          f'time = {time.time() - t_step:.1f} s'
          + ('' if converged else f'  NOT CONVERGED: step ACCEPTED at MAX_STAGGERED = {max_staggered} '
                                  f'(max |d - d_old| = {change:.2e}, <sigma> rel = {stress_rel:.1e}, flips {flips})'),
          flush=True)
    # damage (and, PF-CZM, 1 - omega(d): d << 1 already means omega ~ 0); 2D every step, 3D the mid-plane z = 1/2
    # every step and the full field every SNAP_FULL steps
    d_field = damage.s[0, 0]
    fields = [('step', d_field)] + ([('loss', 1.0 - omega_and_derivatives(d_field, False))]
                                    if damage_model == 'PFCZM' else [])
    for key, fld in fields:
        if dim == 2:
            save_field(f'{key}_{step:03d}.npy', fld)
        else:
            save_field(f'{key}_{step:03d}_midz.npy', fld[:, :, N // 2])
            if (step + 1) % snap_full == 0 or step == len(load_steps) - 1:
                save_field(f'{key}_{step:03d}_full.npy', fld)
    np.savez(steps_file, label=label if dim == 2 else label[:, :, N // 2], dim=dim, N=N, variant=variant,
             materials=materials, damage_model=damage_model, split=split, driving=driving, l_mm=l_mm,
             sigma_mpa=sigma_mpa, k_residual=k_residual, preconditioner=preconditioner_type, cell_mm=cell_mm,
             cg_total_all=cg_total_all, wall_time=float(np.sum(rec['time_step'])),
             **{k: np.array(v) for k, v in rec.items()})
    if time.time() - last_checkpoint > checkpoint_seconds and step < len(load_steps) - 1:
        np.savez(checkpoint_file + '.tmp.npz', damage=to_host(damage.s), history_n=to_host(history_n.s),
                 u_tot=to_host(u_tot.s), u_unit=np.array([to_host(u.s) for u in u_unit]), tension=to_host(tension),
                 E_par=E_par, M_eff=M_eff, C0_norm=C0_norm, rec=np.array(rec, dtype=object), step=step,
                 cg_total_all=cg_total_all)
        os.replace(checkpoint_file + '.tmp.npz', checkpoint_file)
        last_checkpoint = time.time()
        print(f'checkpoint after step {step}', flush=True)
print(f'Elapsed time: {time.time() - start:.1f} s, total elastic CG iterations {cg_total_all}')
if nb_not_converged:
    print(f'WARNING: {nb_not_converged} load steps accepted unconverged (MAX_STAGGERED = {max_staggered})')
