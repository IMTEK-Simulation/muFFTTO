"""
Fourel, Gunn, Molinari 2026 ASR model (nonlocal continuum damage, one Gaussian crack spot at a time), 2D or 3D, CPU or
GPU, on the muFFTTO bilinear / trilinear FE discretisation: their Moulinec-Suquet fixed-point solver vs our Newton-PCG
=====================================================================================================================

Model (their Sec. 2 / Alg. 1). 3D: their model (K = lambda + 2 mu / 3). 2D: plane-strain Lame constants with the 2D
split (2D bulk kappa = lambda + mu, P_vol = I x I / 2) and an in-plane eigenstrain ("2D world").
* eps = eps_e + eps_c, eps_c = eps_ASR I in the gel only
* sigma = g (kappa tr+ I + 2 mu dev eps_e) + kappa tr- I,  g = 1 - phi (1 - K_RESIDUAL)   (Amor split in the stress)
  tr+ = max(tr eps_e, 0): piecewise linear in eps_e, tangent switches with sign(tr eps_e) at every quadrature point
* psi+ = 1/2 g kappa (tr+)^2 (pixel mean over the quad points), nonlocal psi_nl = alpha * psi+ (normalised periodic
  Gaussian alpha0 = exp(-r^2 / 2 l_c^2), FFT convolution on the pixel grid)
* damage loop of a load step: while some pixel has psi_nl >= psi_c(phase): the pixel with the highest psi_nl among
  them gets a Gaussian spot (peak 1) phi <- min(phi + alpha0(. - x*), 1), re-solve elasticity, repeat
  (BATCH=1: all local maxima above psi_c that are > 4 l_c apart get a spot at once). A spot that would not change phi
  (fully damaged neighbourhood) masks its pixel.
* phi, materials: constant per pixel / voxel (labels at the lower-left nodes, rasterise() of make_fourel_geometry_2D
  / make_fourel_geometry_3D)
* Table 5: paste E 12 GPa, aggregate = gel 60 GPa, nu 0.2; psi_c 200 / 600 / 600 J/m^3, l_c 0.25 mm. Units MPa, mm.
* load: eps_ASR 0 -> EPS_MAX (5 %) in LOAD_STEPS (200) steps after a step 0 with eps_ASR = 0; mean stress prescribed
  LOADING=free: all normal <sigma_ii> = -1 MPa, shear 0;  LOADING=cY: <sigma_yy> = -10 MPa (3D: their c-Y = c-Z);
  LOADING=tension: no eigenstrain, E_xx prescribed 0 -> TENSION_EPS in LOAD_STEPS steps, other <sigma> = 0
  (uniaxial-tension strength check of the psi_c calibration).

Elastic solvers (argv[1]):
* MS: basic Moulinec-Suquet scheme on the same FE discretisation: u <- u - G0 B^T W sigma(E + B u - eps_c) with the
  Green operator G0 = (B^T W C0 B)^-1 of C0 = mean of min / max of the undamaged phase moduli (kappa, mu separately),
  built once as in their Alg. 1 (MS_C0=current: min of the current damaged moduli -> kappa0 -> kappa_max / 2, no
  contraction in the stiff phase, observed oscillation), mean-stress control as in Kabel et al. (Q = I):
  E <- E + C0^-1 (sigma_bar - <sigma>). This is preconditioned Richardson with step 1 = the basic scheme (eq. 3);
  Willot's rotated scheme is the same iteration with a different (hexahedral FE-like) discrete gradient. Stop when
  ||eps_k - eps_{k-1}||_L2 / ||eps_k||_L2 < TOL_MS (their r = strain increment; default 1e-4), MS_MAXITER cap.
  1 iteration = 1 B, 1 B^T, 1 Green application (one FFT pair) = the cost of 1 Green-PCG iteration.
* Green / Green_Jacobi / dual: Newton for the piecewise-linear law = active-set iteration: with the tangent of the
  current sign pattern solve K_t u = -B^T W C_t (E - eps_c) by PCG (warm start, energy-error stopping
  ||e||_K <= 1e-6 ||u||_K, delay 10), exact line search on the convex energy along the step when the pattern
  changed (plain active-set Newton cycles), update the pattern, repeat until no quad point changes sign. Mean stress:
  macro strain update E <- E + C_eff^-1 (sigma_bar - <sigma>) between the solves with the superposition warm start
  u += sum dE_k u_k, C_eff = tangent effective stiffness from dim(dim+1)/2 unit solves u_k at the start of every load
  step (also the stiffness output), Broyden-updated inside a step and recomputed when the stress residual does not
  halve per update. A solve is converged when no sign changed and
  max |<sigma> - sigma_bar| < TOL_S (MPa), or when <= 2 points (on the kink tr eps_e = 0) still flip but the PCG
  stopped at its delay floor (step below the CG tolerance; needed for K_RESIDUAL = 1e-6). At most 60 Newton steps.
  Preconditioners: Green C_ref = undamaged aggregate tangent; Green_Jacobi C_ref = 2 mu_agg I + Jacobi;
  dual G K(C_ref C_t^-1 C_ref) G (cost ~ 2 Green iterations).

Output (OUT_DIR, default exp_data/): FOUREL_<run>_solves.csv (one row per elastic solve), FOUREL_<run>_steps.npz (per
load step), FOUREL_<run>_damage/step_<k>.npy (phi float32; 2D every SNAP_EVERY steps, 3D mid-plane z = N/2 every
SNAP_EVERY steps as step_<k>_midz.npy + full field every SNAP_FULL steps), FOUREL_<run>_checkpoint.npz (every
CHECKPOINT_SECONDS at the end of a load step; RESTART_FROM=<checkpoint> continues; the CSV is appended).
3D runs are tagged FOUREL3D_N<N>_...

    python fourel_nonlocal_damage.py <MS|Green|Green_Jacobi|dual>
env: DIM (2), NPIX (64), DEVICE (cpu|gpu), GEOMETRY_NPZ (geometry/fourel2D_s1.npz | geometry/fourel3D_s1.npz),
     LOADING (free|cY|tension), LOAD_STEPS (200), EPS_MAX (0.05), STEPS_RUN (all), K_RESIDUAL (1e-4), BATCH (0),
     TOL_MS (1e-4), MS_MAXITER (20000), TOL_S (1e-3 MPa), TENSION_EPS (6e-4), CHECK_EVERY (0: every n-th solve
     measure the energy error vs a 1e-10 Green-CG reference), MAX_SPOTS_STEP (100000), SNAP_EVERY (1), SNAP_FULL (10),
     CHECKPOINT_SECONDS (3600), RESTART_FROM, OUT_DIR, MUFFTTO_ROOT (library checkout, default ../../..), RUN_SUFFIX
"""
import sys
import os
import itertools

here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.environ.get('MUFFTTO_ROOT', os.path.join(here, '../../..'))))
sys.path.insert(0, here)

import numpy as np
import time

from muFFTTO import domain
from muFFTTO import solvers
from muFFTTO import topology_optimization   # energy_stop_kwargs
from muFFTTO import tensor_operations

dim = int(os.environ.get('DIM', 2))
if dim == 2:
    from make_fourel_geometry_2D import rasterise
else:
    from make_fourel_geometry_3D import rasterise
device = os.environ.get('DEVICE', 'cpu')
if device == 'gpu':
    import cupy as xp
else:
    xp = np
to_host = tensor_operations.to_host
N = int(os.environ.get('NPIX', 64))
geometry_npz = os.environ.get('GEOMETRY_NPZ', f'geometry/fourel{dim}D_s1.npz')
geometry_npz = geometry_npz if os.path.isabs(geometry_npz) else os.path.join(here, geometry_npz)
loading = os.environ.get('LOADING', 'free')
nb_load_steps = int(os.environ.get('LOAD_STEPS', 200))
eps_max = float(os.environ.get('EPS_MAX', 0.05))
steps_run = int(os.environ.get('STEPS_RUN', nb_load_steps))
k_residual = float(os.environ.get('K_RESIDUAL', 1e-4))
batch = int(os.environ.get('BATCH', 0))
tol_ms = float(os.environ.get('TOL_MS', 1e-4))
ms_maxiter = int(os.environ.get('MS_MAXITER', 20000))
ms_trace = bool(int(os.environ.get('MS_TRACE', 0)))
ms_c0 = os.environ.get('MS_C0', 'init')
tol_s = float(os.environ.get('TOL_S', 1e-3))
tension_eps = float(os.environ.get('TENSION_EPS', 6e-4))
check_every = int(os.environ.get('CHECK_EVERY', 0))
max_spots_step = int(os.environ.get('MAX_SPOTS_STEP', 100000))
snap_every = int(os.environ.get('SNAP_EVERY', 1))
snap_full = int(os.environ.get('SNAP_FULL', 10))
checkpoint_seconds = float(os.environ.get('CHECKPOINT_SECONDS', 3600))
restart_from = os.environ.get('RESTART_FROM')
cg_tol, cg_delay = 1e-6, 10
solver_type = sys.argv[1] if len(sys.argv) > 1 else 'Green'
assert solver_type in ('MS', 'Green', 'Green_Jacobi', 'dual') and loading in ('free', 'cY', 'tension')

geo = np.load(geometry_npz)
L_mm = float(geo['cell_mm'])
h_mm = L_mm / N
l_c = 0.25                                                    # mm
# Table 5 (MPa; psi_c in MPa = 1e6 J/m^3)
PHASES = {0: dict(E=12e3, nu=0.2, psi_c=200e-6),              # cement paste
          1: dict(E=60e3, nu=0.2, psi_c=600e-6),              # aggregate
          2: dict(E=60e3, nu=0.2, psi_c=600e-6)}              # ASR gel (damageable, as aggregate)
VOIGT = [(0, 0), (1, 1), (0, 1)] if dim == 2 else [(0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1)]
nv = len(VOIGT)
normal = np.array([i == j for (i, j) in VOIGT])
sigma_bar = np.where(normal, -1.0, 0.0)                       # Voigt, MPa
stress_ctrl = np.ones(nv, dtype=bool)
if loading == 'cY':
    sigma_bar[1] = -10.0
elif loading == 'tension':
    sigma_bar[:] = 0.0
    stress_ctrl[0] = False
if loading == 'tension':
    eps_asr_steps = np.zeros(nb_load_steps + 1)
    exx_steps = np.linspace(0.0, tension_eps, nb_load_steps + 1)
else:
    eps_asr_steps = np.linspace(0.0, eps_max, nb_load_steps + 1)
    exx_steps = np.zeros(nb_load_steps + 1)

run_name = (f'FOUREL{"3D" if dim == 3 else ""}_N{N}_{solver_type}_{loading}_k{k_residual:g}'
            + ('_batch' if batch else '')
            + (f'_tolms{tol_ms:g}' if solver_type == 'MS' and tol_ms != 1e-4 else '')
            + ('_C0current' if solver_type == 'MS' and ms_c0 == 'current' else '')
            + os.environ.get('RUN_SUFFIX', ''))
out_dir = os.environ.get('OUT_DIR', os.path.join(here, 'exp_data'))
solves_file = os.path.join(out_dir, run_name + '_solves.csv')
steps_file = os.path.join(out_dir, run_name + '_steps.npz')
snap_dir = os.path.join(out_dir, run_name + '_damage')
checkpoint_file = os.path.join(out_dir, run_name + '_checkpoint.npz')
os.makedirs(snap_dir, exist_ok=True)


def lame(E, nu):
    """Lame constants (3D, = plane strain in 2D)"""
    return E * nu / ((1 + nu) * (1 - 2 * nu)), E / (2 * (1 + nu))


element = 'bilinear_rectangle' if dim == 2 else 'trilinear_hexahedron'
cell_u = domain.PeriodicUnitCell(domain_size=[1.0] * dim, problem_type='elasticity')
disc_u = domain.Discretization(cell=cell_u, nb_of_pixels_global=(N,) * dim, discretization_type='finite_element',
                               element_type=element, device=device)
comm = disc_u.communicator
assert comm.size == 1, 'serial (one GPU) script'
AX = tuple(range(dim))                                        # pixel axes of a pixel field

# ---- geometry, pixel-wise materials
label = rasterise(geometry_npz, N).astype(np.int8)
kappa_h, mu_h, psic_h = (np.zeros((N,) * dim) for _ in range(3))
for p, prm in PHASES.items():
    lp, mp = lame(prm['E'], prm['nu'])
    kappa_h[label == p], mu_h[label == p], psic_h[label == p] = lp + 2 * mp / dim, mp, prm['psi_c']
lam_a, mu_a = lame(PHASES[1]['E'], PHASES[1]['nu'])
kappa_a = lam_a + 2 * mu_a / dim
gel = xp.asarray(label == 2)
kappa_x, mu_x, psic_x = xp.asarray(kappa_h), xp.asarray(mu_h), xp.asarray(psic_h)
print(f'{dim}D {N}^{dim}, cell {L_mm:g} mm, h = {h_mm:.4f} mm, l_c/h = {l_c / h_mm:.2f}; paste {np.mean(label == 0):.3f} '
      f'aggregate {np.mean(label == 1):.3f} gel {np.mean(label == 2):.4f}; solver {solver_type}, loading {loading}, '
      f'k = {k_residual:g}, batch {batch}, device {device}', flush=True)

I2 = np.eye(dim)
I_sym = 0.5 * (np.einsum('ik,jl->ijkl', I2, I2) + np.einsum('il,jk->ijkl', I2, I2))
P_vol = np.einsum('ij,kl->ijkl', I2, I2) / dim
P_dev = I_sym - P_vol
C_ref = dim * kappa_a * P_vol + 2 * mu_a * P_dev              # undamaged aggregate (stiffest phase)

s_vol = disc_u.get_quad_field_scalar(name='s_vol')            # dim kappa g_v
s_dev = disc_u.get_quad_field_scalar(name='s_dev')            # 2 mu g
C_t = domain.ScaledMaterial([(s_vol, xp.asarray(P_vol)), (s_dev, xp.asarray(P_dev))])
C_dual = domain.ScaledMaterial([(lambda: (dim * kappa_a) ** 2 / s_vol.s[0, 0], xp.asarray(P_vol)),
                                (lambda: (2 * mu_a) ** 2 / s_dev.s[0, 0], xp.asarray(P_dev))])

E_UNIT = []
for (i, j) in VOIGT:
    e = np.zeros((dim, dim))
    e[i, j] = e[j, i] = 1.0
    E_UNIT.append(e)
COMP = ['xyz'[i] + 'xyz'[j] for (i, j) in VOIGT]


def voigt(s):
    s = np.asarray(to_host(s))
    return np.array([s[i, j] for (i, j) in VOIGT])


def from_par(e):
    return sum(e[k] * E_UNIT[k] for k in range(nv))


def trace(a):
    return sum(a[i, i] for i in range(dim))


macro = disc_u.get_gradient_size_field(name='macro_field')
eps = disc_u.get_gradient_size_field(name='eps_elastic')
sig = disc_u.get_gradient_size_field(name='sigma')
rhs_u = disc_u.get_unknown_size_field(name='rhs_u')
force = disc_u.get_unknown_size_field(name='force')
du = disc_u.get_unknown_size_field(name='du')
u = disc_u.get_unknown_size_field(name='u_total')
u_unit = [disc_u.get_unknown_size_field(name=f'u_unit_{k}') for k in range(nv)] if solver_type != 'MS' else []
u_check = disc_u.get_unknown_size_field(name='u_check') if check_every else None
Kx_check = disc_u.get_unknown_size_field(name='Kx_check') if check_every else None
QAX = tuple(range(2, eps.s.ndim))                            # quad point + pixel axes of a gradient field
I_q = xp.asarray(I2).reshape((dim, dim) + (1,) * (dim + 1))

phi = xp.zeros((N,) * dim)                                    # damage, per pixel
g_x = xp.ones((N,) * dim)
masked = xp.zeros((N,) * dim, dtype=bool)

# ---- Gaussian kernels (periodic, pixel grid)
dxy = (np.arange(N) + N // 2) % N - N // 2
r2 = sum(np.reshape(dxy, [-1 if a == b else 1 for b in range(dim)]) ** 2 for a in range(dim)) * h_mm ** 2
alpha0_h = np.exp(-r2 / (2 * l_c ** 2))                       # peak 1 at pixel (0, ..., 0)
alpha0 = xp.asarray(alpha0_h)
alpha_hat = xp.fft.rfftn(xp.asarray(alpha0_h / alpha0_h.sum()), axes=AX)


def K_t(x, Ax):
    disc_u.apply_system_matrix_mugrid(material_data_field=C_t, input_field_inxyz=x, output_field_inxyz=Ax,
                                      formulation='small_strain')


green_u = None
if solver_type != 'MS':
    green_u = disc_u.get_preconditioner_Green_mugrid(
        reference_material_data_ijkl=2.0 * mu_a * I_sym if solver_type == 'Green_Jacobi' else C_ref,
        formulation='small_strain', name='green_u')
green_check = disc_u.get_preconditioner_Green_mugrid(reference_material_data_ijkl=C_ref, formulation='small_strain',
                                                     name='green_check') if check_every else None
dual_temp_1 = disc_u.get_unknown_size_field(name='dual_temp_1') if solver_type == 'dual' else None
dual_temp_2 = disc_u.get_unknown_size_field(name='dual_temp_2') if solver_type == 'dual' else None
x_scaled_u = disc_u.get_unknown_size_field(name='x_scaled_u') if solver_type == 'Green_Jacobi' else None
jacobi_u = None


def M_u(x, Px):
    if solver_type == 'dual':
        disc_u.fft.communicate_ghosts(x)
        disc_u.apply_preconditioner_mugrid(preconditioner_Fourier_fnfnqks=green_u, input_nodal_field_fnxyz=x,
                                           output_nodal_field_fnxyz=dual_temp_1)
        disc_u.apply_system_matrix_mugrid(material_data_field=C_dual, input_field_inxyz=dual_temp_1,
                                          output_field_inxyz=dual_temp_2, formulation='small_strain')
        disc_u.fft.communicate_ghosts(dual_temp_2)
        disc_u.apply_preconditioner_mugrid(preconditioner_Fourier_fnfnqks=green_u, input_nodal_field_fnxyz=dual_temp_2,
                                           output_nodal_field_fnxyz=Px)
    elif solver_type == 'Green_Jacobi':
        x_scaled_u.s[...] = jacobi_u.s * x.s
        disc_u.apply_preconditioner_mugrid(preconditioner_Fourier_fnfnqks=green_u,
                                           input_nodal_field_fnxyz=x_scaled_u, output_nodal_field_fnxyz=Px)
        Px.s[...] *= jacobi_u.s
    else:
        disc_u.apply_preconditioner_mugrid(preconditioner_Fourier_fnfnqks=green_u, input_nodal_field_fnxyz=x,
                                           output_nodal_field_fnxyz=Px)


def M_check(x, Px):
    disc_u.apply_preconditioner_mugrid(preconditioner_Fourier_fnfnqks=green_check, input_nodal_field_fnxyz=x,
                                       output_nodal_field_fnxyz=Px)


def pcg(x, b, P, tol):
    norms = solvers.conjugate_gradients_mugrid_experimental(
        comm=comm, fc=disc_u.field_collection, hessp=K_t, b=b, x=x, P=P, maxiter=50000,
        stop_crit_norm='energy_lower_estim', d_min=cg_delay,
        **topology_optimization.energy_stop_kwargs(disc_u, K_t, x, tol))[1]
    if 'stop_iteration' not in norms:                          # zero residual at the start: nothing to do
        return 0
    return int(norms['stop_iteration']['energy_lower_estim'])


# ---------------------------------------------------------------- constitutive law
E_par = np.zeros(nv)
eps_asr = 0.0


def set_macro():
    disc_u.get_macro_gradient_field_mugrid(macro_gradient_ij=from_par(E_par), macro_gradient_field_ijqxyz=macro)
    if eps_asr != 0.0:
        for i in range(dim):
            macro.s[i, i] -= xp.where(gel, eps_asr, 0.0)


def elastic_strain(x):
    """eps <- E - eps_c + B x"""
    disc_u.fft.communicate_ghosts(x)
    disc_u.apply_gradient_operator_symmetrized_mugrid(u_inxyz=x, grad_u_ijqxyz=eps)
    eps.s[...] += macro.s


def set_tangent():
    """tangent of the sign pattern of the current eps; returns the pattern"""
    pos = trace(eps.s) > 0.0
    s_vol.s[0, 0] = dim * kappa_x * xp.where(pos, g_x, 1.0)
    s_dev.s[0, 0] = 2 * mu_x * g_x
    return pos


def stress():
    """sig <- C_t : eps (exact stress for the pattern of eps), returns <sigma> Voigt"""
    tr = trace(eps.s)
    for i in range(dim):
        for j in range(dim):
            sig.s[i, j] = s_dev.s[0, 0] * (eps.s[i, j] - (tr / dim if i == j else 0.0))
        sig.s[i, i] += s_vol.s[0, 0] / dim * tr
    return voigt(sig.s.mean(axis=QAX))


def psi_plus_pixel():
    tr = trace(eps.s)
    return (0.5 * g_x * kappa_x * xp.maximum(tr, 0.0) ** 2).mean(axis=0)


def solve_E_update(M, r):
    dE = np.zeros(nv)
    m = stress_ctrl
    dE[m] = np.linalg.solve(M[np.ix_(m, m)], r[m])
    return dE


# ---------------------------------------------------------------- MS basic scheme
green_ms, C0_par, M0 = None, None, None


def ms_reference():
    """C0 = mean of min / max of the pixel moduli (kappa, mu separately). Default MS_C0=init: undamaged phases, built
    once (their Alg. 1 builds Gamma0 in the initialisation). MS_C0=current: min of the current (damaged) moduli; with
    min -> k ~ 0 this puts kappa0 -> kappa_max / 2, |1 - kappa_max / kappa0| -> 1 and the basic scheme stops
    contracting in the stiff phase (observed: oscillation, no convergence)."""
    global green_ms, C0_par, M0
    gmin = float(g_x.min()) if ms_c0 == 'current' else 1.0
    k0 = 0.5 * (gmin * kappa_h.min() + kappa_h.max())
    m0 = 0.5 * (gmin * mu_h.min() + mu_h.max())
    if C0_par is None or abs(k0 / C0_par[0] - 1) > 0.01 or abs(m0 / C0_par[1] - 1) > 0.01:
        C0 = dim * k0 * P_vol + 2 * m0 * P_dev
        green_ms = disc_u.get_preconditioner_Green_mugrid(reference_material_data_ijkl=C0, formulation='small_strain',
                                                          name='green_ms')
        M0 = np.array([voigt(np.einsum('ijkl,kl->ij', C0, E_UNIT[k])) for k in range(nv)]).T
        C0_par = (k0, m0)


def solve_ms():
    """basic scheme, returns (iterations, <sigma>)"""
    global E_par
    eps_prev = None
    rel = np.nan
    for it in range(ms_maxiter + 1):
        set_macro()
        elastic_strain(u)
        set_tangent()
        if it == 0:
            ms_reference()
        sbar = stress()
        eps_tot = eps.s + (xp.where(gel, eps_asr, 0.0) * I_q if eps_asr else 0.0)
        if eps_prev is not None:
            rel = float(xp.linalg.norm(eps_tot - eps_prev)) / max(float(xp.linalg.norm(eps_tot)), 1e-300)
            if rel < tol_ms:
                return it, sbar
        if it == ms_maxiter:
            print(f'  MS NOT CONVERGED after {ms_maxiter} its (rel. increment {rel:.2e})', flush=True)
            return it, sbar
        if ms_trace and it % 500 == 0 and it > 0:
            print(f'    MS it {it} rel. increment {rel:.3e} C0 (kappa, mu) {C0_par}  <s> {sbar}', flush=True)
        eps_prev = eps_tot.copy()
        del eps_tot
        disc_u.apply_gradient_transposed_operator_mugrid(gradient_field_ijqxyz=sig, div_u_fnxyz=force,
                                                         apply_weights=True)
        disc_u.apply_preconditioner_mugrid(preconditioner_Fourier_fnfnqks=green_ms, input_nodal_field_fnxyz=force,
                                           output_nodal_field_fnxyz=du)
        u.s[...] -= du.s
        E_par = E_par + solve_E_update(M0, sigma_bar - sbar)


# ---------------------------------------------------------------- Newton-PCG
M_eff = None
newton_trace = []


def refresh_jacobi():
    global jacobi_u
    if solver_type == 'Green_Jacobi':
        jacobi_u = disc_u.get_preconditioner_Jacobi_mugrid(material_data_field_ijklqxyz=C_t,
                                                           formulation='small_strain', name='jacobi_u')


def unit_solves():
    """tangent C_eff with the current tangent (s fields), returns (M, CG its)"""
    M, its = np.zeros((nv, nv)), 0
    refresh_jacobi()
    for k in range(nv):
        disc_u.get_macro_gradient_field_mugrid(macro_gradient_ij=E_UNIT[k], macro_gradient_field_ijqxyz=macro)
        disc_u.get_rhs_mugrid(material_data_field_ijklqxyz=C_t, macro_gradient_field_ijqxyz=macro, rhs_inxyz=rhs_u)
        its += pcg(u_unit[k], rhs_u, M_u, cg_tol)
        M[:, k] = voigt(disc_u.get_homogenized_stress_mugrid(material_data_field_ijklqxyz=C_t,
                                                             displacement_field_inxyz=u_unit[k],
                                                             macro_gradient_field_ijqxyz=macro,
                                                             formulation='small_strain'))
    return M, its


def directional_force(eps0, deps, a):
    """d/da of the (convex) stored energy at eps0 + a deps: <sigma(eps0 + a deps) : deps>"""
    e = eps0 + a * deps
    tr = trace(e)
    trd = trace(deps)
    kap = kappa_x * xp.where(tr > 0.0, g_x, 1.0)
    dev_dot = xp.einsum('ij...,ij...->...', e, deps) - tr * trd / dim
    return float(xp.mean(kap * tr * trd + 2 * mu_x * g_x * dev_dot))


def line_search(eps0, deps):
    """exact line search on the convex, piecewise quadratic energy along the Newton step (only when the sign
    pattern changed): the root of the monotone directional force in (0, 1] (Illinois regula falsi). Without it the
    active-set Newton can cycle between sign patterns (seen at 64^2 and 128^2: 199 iterations, ~2e4 CG its)."""
    f0, f1 = directional_force(eps0, deps, 0.0), directional_force(eps0, deps, 1.0)
    if f1 <= 0.0 or f0 >= 0.0:
        return 1.0
    a, b, fa, fb, side = 0.0, 1.0, f0, f1, 0
    for _ in range(60):
        c = (a * fb - b * fa) / (fb - fa)
        fc = directional_force(eps0, deps, c)
        if abs(fc) < 1e-10 * abs(f0) or b - a < 1e-12:
            return c
        if fc < 0.0:
            a, fa = c, fc
            if side == -1:
                fb *= 0.5
            side = -1
        else:
            b, fb = c, fc
            if side == 1:
                fa *= 0.5
            side = 1
    return c


def solve_newton():
    """active-set Newton + PCG + macro update; returns (CG its, Newton its, unit CG its, <sigma>)"""
    global E_par, M_eff
    global newton_trace
    cg_its, unit_its, newton = 0, 0, 0
    newton_trace = []
    set_macro()
    elastic_strain(u)
    pattern = set_tangent()
    sbar = stress()
    prev = None                     # (E, <sigma>, |r|) of the last equilibrated iterate with an unchanged pattern
    for newton in range(1, 61):
        r = np.where(stress_ctrl, sigma_bar - sbar, 0.0)
        if np.abs(r).max() > tol_s:
            dE = solve_E_update(M_eff, r)
            E_par = E_par + dE
            for k in range(nv):                                # superposition warm start
                u.s[...] += float(dE[k]) * u_unit[k].s
        set_macro()
        elastic_strain(u)
        u_old, eps_old = u.s.copy(), eps.s.copy()
        refresh_jacobi()
        disc_u.get_rhs_mugrid(material_data_field_ijklqxyz=C_t, macro_gradient_field_ijqxyz=macro, rhs_inxyz=rhs_u)
        its_n = pcg(u, rhs_u, M_u, cg_tol)
        cg_its += its_n
        elastic_strain(u)
        new_pattern = set_tangent()
        flips = int(xp.count_nonzero(new_pattern != pattern))
        alpha = 1.0
        if flips:
            alpha = line_search(eps_old, eps.s - eps_old)
            if alpha < 1.0:
                u.s[...] = u_old + alpha * (u.s - u_old)
                eps.s[...] = eps_old + alpha * (eps.s - eps_old)
                new_pattern = set_tangent()
                flips = max(flips, 1)                          # not converged: the next tangent differs
        del u_old, eps_old
        newton_trace.append(f'{its_n}/{flips}' + (f'/a{alpha:.2f}' if alpha < 1.0 else ''))
        pattern = new_pattern
        sbar = stress()
        r_new = np.abs(np.where(stress_ctrl, sigma_bar - sbar, 0.0)).max()
        if flips == 0 and r_new < tol_s:
            break
        if flips <= 2 and its_n <= cg_delay and r_new < tol_s:
            # the PCG stopped at the delay floor: the start was already within the energy tolerance of this tangent's
            # solution; the remaining 1-2 flips are quad points sitting on the kink tr eps_e = 0 (the minimiser of the
            # convex energy is there, neither branch's tangent reaches it) -> converged to the CG tolerance
            break
        if flips == 0:
            if prev is not None:
                # two equilibrated states of the same tangent: secant = tangent C_eff along dE (good Broyden)
                dE, ds = E_par - prev[0], sbar - prev[1]
                m = stress_ctrl
                if np.linalg.norm(dE[m]) > 0:
                    Mm = M_eff[np.ix_(m, m)]
                    Mm += np.outer(ds[m] - Mm @ dE[m], dE[m]) / (dE[m] @ dE[m])
                    M_eff[np.ix_(m, m)] = Mm
                if r_new > 0.5 * prev[2]:                     # stale C_eff: recompute
                    M_eff, its = unit_solves()
                    unit_its += its
            prev = (E_par.copy(), sbar.copy(), r_new)
        else:
            prev = None
    else:
        print('  NEWTON NOT CONVERGED', flush=True)
    return cg_its, newton, unit_its, sbar


def accuracy_check():
    """energy error of u vs a 1e-10 Green-CG solution of the same linear problem (tangent, E fixed)"""
    disc_u.get_rhs_mugrid(material_data_field_ijklqxyz=C_t, macro_gradient_field_ijqxyz=macro, rhs_inxyz=rhs_u)
    u_check.s[...] = u.s
    solvers.conjugate_gradients_mugrid_experimental(
        comm=comm, fc=disc_u.field_collection, hessp=K_t, b=rhs_u, x=u_check, P=M_check, maxiter=100000,
        stop_crit_norm='energy_lower_estim', d_min=cg_delay,
        **topology_optimization.energy_stop_kwargs(disc_u, K_t, u_check, 1e-10))
    u_check.s[...] -= u.s
    K_t(u_check, Kx_check)
    e2 = float(xp.dot(u_check.s.ravel(), Kx_check.s.ravel()))
    K_t(u, Kx_check)
    n2 = float(xp.dot(u.s.ravel(), Kx_check.s.ravel()))
    return np.sqrt(max(e2, 0.0) / max(n2, 1e-300))


# ---------------------------------------------------------------- damage
def add_spots(psi_nl):
    """their rule (one spot at argmax) or BATCH; returns the number of spots added"""
    global phi, g_x
    over = (psi_nl >= psic_x) & ~masked
    if not bool(over.any()):
        return 0
    if batch:
        is_max = over.copy()
        for sh in itertools.product((-1, 0, 1), repeat=dim):
            if any(sh):
                is_max &= psi_nl >= xp.roll(psi_nl, sh, axis=AX)
        cand = to_host(xp.flatnonzero(is_max.ravel()))
        vals = to_host(psi_nl.ravel()[xp.asarray(cand)]) if cand.size else np.zeros(0)
        cand = cand[np.argsort(-vals)]
        if cand.size == 0:
            cand = np.array([int(xp.argmax(xp.where(over, psi_nl, -xp.inf)))])
        chosen = []
        for c in cand:
            ci = np.array(np.unravel_index(int(c), (N,) * dim))
            if all(np.linalg.norm(((ci - q + N // 2) % N - N // 2) * h_mm) > 4 * l_c for q in chosen):
                chosen.append(ci)
    else:
        chosen = [np.array(np.unravel_index(int(xp.argmax(xp.where(over, psi_nl, -xp.inf))), (N,) * dim))]
    added = 0
    for ci in chosen:
        new = xp.minimum(phi + xp.roll(alpha0, tuple(int(c) for c in ci), axis=AX), 1.0)
        if float((new - phi).max()) < 1e-3:
            masked[tuple(int(c) for c in ci)] = True
            continue
        phi = new
        added += 1
    g_x = 1.0 - phi * (1.0 - k_residual)
    return added if added else (-1)                            # -1: only masking happened, try again


def snapshot(step):
    d = to_host(phi).astype(np.float32)
    if dim == 2:
        if step % snap_every == 0 or step == last_step:
            np.save(os.path.join(snap_dir, f'step_{step:03d}.npy'), d)
    else:
        if step % snap_every == 0 or step == last_step:
            np.save(os.path.join(snap_dir, f'step_{step:03d}_midz.npy'), d[:, :, N // 2])
        if step % snap_full == 0 or step == last_step:
            np.save(os.path.join(snap_dir, f'step_{step:03d}_full.npy'), d)


# ---------------------------------------------------------------- main loop
rec = {k: [] for k in ('eps_asr', 'Exx_load', 'E', 'sigma', 'C_eff', 'spots', 'solves', 'linear_its', 'newton_its',
                       'unit_its', 'time_step', 'time_solve', 'damaged_fraction', 'mean_phi')}
solve_count, spots_total, first_step = 0, 0, 0
if restart_from:
    state = np.load(restart_from, allow_pickle=True)
    phi = xp.asarray(state['phi'])
    masked = xp.asarray(state['masked'])
    g_x = 1.0 - phi * (1.0 - k_residual)
    u.s[...] = xp.asarray(state['u'])
    for k in range(len(u_unit)):
        u_unit[k].s[...] = xp.asarray(state['u_unit'][k])
    E_par = state['E_par'].copy()
    rec = {k: list(v) for k, v in state['rec'].item().items()}
    solve_count, spots_total = int(state['solve_count']), int(state['spots_total'])
    first_step = int(state['step']) + 1
    with open(solves_file) as f:                               # drop rows of the unfinished step(s) after the checkpoint
        lines = f.readlines()
    with open(solves_file, 'w') as f:
        f.writelines([lines[0]] + [ln for ln in lines[1:] if int(ln.split(',')[0]) < first_step])
    print(f'restart from {restart_from}: load step {first_step}', flush=True)
else:
    with open(solves_file, 'w') as f:
        f.write('step,eps_asr,solve,spots_total,linear_its,newton_its,unit_its,time_solve,max_psi_ratio,'
                + ','.join('s' + c for c in COMP) + ',' + ','.join('E' + c for c in COMP)
                + ',energy_error,newton_trace\n')
last_step = min(steps_run, nb_load_steps)
last_checkpoint = time.time()
t_start = time.time()
for step in range(first_step, last_step + 1):
    eps_asr = eps_asr_steps[step]
    if not stress_ctrl[0]:
        E_par[0] = exx_steps[step]
    t_step = time.time()
    lin_step, newton_step, unit_step, solves_step, spots_step, t_solve_step = 0, 0, 0, 0, 0, 0.0
    if solver_type != 'MS':
        set_macro()
        elastic_strain(u)
        set_tangent()
        M_eff, unit_its = unit_solves()
        unit_step += unit_its
    while True:
        t0 = time.time()
        if solver_type == 'MS':
            lin, sbar = solve_ms()
            newton_its = 1
            newton_trace = []
            unit_its = 0
        else:
            lin, newton_its, unit_its, sbar = solve_newton()
        t_solve = time.time() - t0
        solve_count += 1
        err = accuracy_check() if (check_every and solve_count % check_every == 0) else np.nan
        lin_step += lin
        newton_step += newton_its
        unit_step += unit_its
        solves_step += 1
        t_solve_step += t_solve
        psi_nl = xp.fft.irfftn(xp.fft.rfftn(psi_plus_pixel(), axes=AX) * alpha_hat, s=(N,) * dim, axes=AX)
        ratio = float((psi_nl / psic_x).max())
        with open(solves_file, 'a') as f:
            f.write(f'{step},{eps_asr:.6g},{solve_count},{spots_total},{lin},{newton_its},{unit_its},{t_solve:.4g},'
                    f'{ratio:.4g},' + ','.join(f'{v:.6g}' for v in sbar) + ','
                    + ','.join(f'{v:.6g}' for v in E_par) + f',{err:.3g},' + ' '.join(newton_trace) + '\n')
        if spots_step >= max_spots_step:
            print(f'  MAX_SPOTS_STEP reached in step {step}', flush=True)
            break
        added = add_spots(psi_nl)
        while added == -1:
            added = add_spots(psi_nl)
        if added == 0:
            break
        spots_step += added
        spots_total += added
    rec_vals = dict(eps_asr=eps_asr, Exx_load=exx_steps[step], E=E_par.copy(), sigma=sbar,
                    C_eff=M_eff.copy() if M_eff is not None else np.full((nv, nv), np.nan),
                    spots=spots_step, solves=solves_step, linear_its=lin_step, newton_its=newton_step,
                    unit_its=unit_step, time_step=time.time() - t_step, time_solve=t_solve_step,
                    damaged_fraction=float(xp.mean(phi > 0.5)), mean_phi=float(phi.mean()))
    for k, v in rec_vals.items():
        rec[k].append(v)
    snapshot(step)
    np.savez(steps_file, label=label if dim == 2 else label[:, :, N // 2], dim=dim, N=N, solver=solver_type,
             loading=loading, k_residual=k_residual, batch=batch, tol_ms=tol_ms, h_mm=h_mm, l_c=l_c,
             **{k: np.array(v) for k, v in rec.items()})
    print(f'step {step:3d} eps_ASR = {100 * eps_asr:.3f} %  Exx = {E_par[0]:.3e}  tr E = {E_par[:dim].sum():.3e}  '
          f'<s> = (' + ', '.join(f'{v:.3f}' for v in sbar) + f')  spots {spots_step:4d} (total {spots_total})  '
          f'solves {solves_step:4d}  its (lin {lin_step}, newton {newton_step}, unit {unit_step})  '
          f'phi>0.5 {rec_vals["damaged_fraction"]:.4f}  time {rec_vals["time_step"]:.1f} s', flush=True)
    if time.time() - last_checkpoint > checkpoint_seconds and step < last_step:
        np.savez(checkpoint_file + '.tmp.npz', phi=to_host(phi), masked=to_host(masked), u=to_host(u.s),
                 u_unit=np.array([to_host(v.s) for v in u_unit]), E_par=E_par, rec=np.array(rec, dtype=object),
                 step=step, solve_count=solve_count, spots_total=spots_total)
        os.replace(checkpoint_file + '.tmp.npz', checkpoint_file)
        last_checkpoint = time.time()
        print(f'checkpoint after step {step}', flush=True)
print(f'Elapsed time: {time.time() - t_start:.1f} s, solves {solve_count}, spots {spots_total}')
