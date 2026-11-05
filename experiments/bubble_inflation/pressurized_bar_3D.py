"""
3D hollow bar with a pressurised cylindrical cavity, lateral faces x = const clamped (finite strain, regular voxels)
==================================================================================================================

The 2D pressurised-hole plate (pressurized_hole.py) extruded along z and sealed by solid caps; geometry in
bar3d_geometry.py (cross-section 0.875 x 0.875, cavity radius 0.3125, length 1.0, caps 0.125, void t = 1/16).

    energy  Pi(u) = int_solid W(F) dX  -  p int_cavity J dX,     p = lam P_MAX (follower pressure)
    W = mu/2 (J^-2/3 I1 - 3) + kappa/2 ln(J)^2   (3D neo-Hookean, E = 100, nu = 0.3)

Void-only nodes (all 8 adjacent voxels void: frame and cavity interior) are pinned; the gas energy depends only on
the nodes of the cavity surface (the signed hexahedron volumes sum to the enclosed volume; 2 x 2 x 2 Gauss integrates
the trilinear J exactly), so the pin does not change the load. det F > 0 is required in the solid only.
Clamps: the two full node planes x = t and x = Lx - t (through the void), u = 0.

Preconditioners (Green, C_ref = the solid tangent at F = I):
    G       projected Green Q M Q
    G_wall  corrected Green for the two clamp planes, the capacitance system [S B; B^T 0] (S = E^T M E, B = E^T N)
            solved by a 2D FFT over the planes: S is block-circulant in (y, z) with a 6 x 6 symbol (2 planes x 3
            components) per wave-number pair; the 3 translations only act at wave number 0 (9 x 9 there). Setup:
            6 Green applications. Pinned void dofs are projected out (Q M_wall Q).

Per load increment: plain trust-region Newton-CG from the last converged state, inner CG stopped on the energy-error
estimate (rtol 1e-6, adaptive delay, minimum 10), step halved on failure.

Hessian products: MATRIX_FREE=1 (default) applies the exact directional derivative of the stress, dP = A(F) : grad v,
in closed form from F (21 numbers per quadrature point stored per Newton iterate: F, F^-T, J^-2/3, ln J, I1) instead of
the 81-component tangent; MATRIX_FREE=0 stores the tangent. CHECK_MF=1: compare both once at a deformed state.

Environment: NNN (32, voxels per unit length), P_MAX (30), NB_INC (30), PRECOND (G_wall | G), DEVICE (gpu | cpu),
MATRIX_FREE (1), CHECK_MF (0), RUNS_DIR (./runs),
PERIODIC (0: sealed bar; 1: repeating open channel, periodic in z, no caps), P_VAR (0) and P_OFFSET (1): pressure
p(z) = p (P_OFFSET + P_VAR cos(2 pi z / Lz)) (P_OFFSET 0, P_VAR 1: +p at z = 0, suction -p at z = Lz/2), at the gas voxel centres, energy -int p(X) J dX), LOAD (follower: the gas energy above, follows
the deforming wall; dead: the consistent nodal forces of that pressure on the UNDEFORMED wall, f = p dV/du at u = 0,
energy -f.u: only the right-hand side, no load term in the Hessian).
WAVE=1 (follower only): travelling pressure wave p(z, t) = p (P_OFFSET + P_VAR cos 2 pi (z / Lz - t / T)), quasi-static
(no inertia: every time step is an equilibrium, solved from the previous one): the amplitude ramps to P_MAX in NB_INC
steps (t = 0), then t runs over N_PERIODS periods in NB_T steps per period.
BAR_L (1.0): bar / channel length along z; WAVES (1): number of pressure wavelengths along Lz (p uses WAVES z / Lz).
PULSE=1 (with WAVE=1): a single wave instead: p = p sin(2 pi zeta / l) for 0 <= zeta < l (pressure lobe, then suction
lobe), 0 elsewhere, l = PULSE_LEN Lz (1/3), zeta = z - s(t) (periodic), s = (t / T) PULSE_TRAVEL Lz (2/3: the pulse goes
from the first to the last third of the channel over t / T = 0 .. 1).
Output: runs/bar3d_<...>/ with run.log, state.npz (u per increment, p, cavity volume, CG counts) and, MUEYE_NC=1
(default), bar3d_mueye.nc written during the run, one frame per accepted increment, directly readable by muEye: per-voxel
(sub-point 'pixel') fields solid, cavity, det_F (quadrature mean) and displacement in grid-point units (u / h), with the
pinned cavity-interior nodes filled by the harmonic extension of the cavity surface (display only; MUEYE_FILL=0: off,
the gas region then shows the pinned nodes); frame variables pressure, phase (t / T), cavity_volume_ratio, increment.
OUTPUT_EVERY=k (1): store u in state.npz and write the muEye frame only every k-th increment (and the last one).
"""
import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import numpy as np  # noqa: E402
from mpi4py import MPI  # noqa: E402

from muFFTTO import domain  # noqa: E402
from muFFTTO import solvers_nonlinear  # noqa: E402
from bar3d_geometry import bar_geometry, DEFAULTS  # noqa: E402

nnn = int(os.environ.get('NNN', 32))
P_MAX = float(os.environ.get('P_MAX', 30.0))
NB_INC = int(os.environ.get('NB_INC', 30))
PRECOND = os.environ.get('PRECOND', 'G_wall')
DEVICE = os.environ.get('DEVICE', 'gpu')
MATRIX_FREE = os.environ.get('MATRIX_FREE', '1') == '1'
CHECK_MF = os.environ.get('CHECK_MF', '0') == '1'
PERIODIC = os.environ.get('PERIODIC', '0') == '1'
P_VAR = float(os.environ.get('P_VAR', 0.0))
P_OFFSET = float(os.environ.get('P_OFFSET', 1.0))
LOAD = os.environ.get('LOAD', 'follower')
WAVE = os.environ.get('WAVE', '0') == '1'
NB_T = int(os.environ.get('NB_T', 24))
BAR_L = float(os.environ.get('BAR_L', DEFAULTS['BAR_L']))
WAVES = float(os.environ.get('WAVES', 1))
PULSE = os.environ.get('PULSE', '0') == '1'
MUEYE_NC = os.environ.get('MUEYE_NC', '1') == '1'
MUEYE_FILL = os.environ.get('MUEYE_FILL', '1') == '1'
OUTPUT_EVERY = int(os.environ.get('OUTPUT_EVERY', 1))   # store u / write the muEye frame every k-th increment
PULSE_LEN = float(os.environ.get('PULSE_LEN', 1 / 3))
PULSE_TRAVEL = float(os.environ.get('PULSE_TRAVEL', 2 / 3))
N_PERIODS = float(os.environ.get('N_PERIODS', 1))
assert not (WAVE and LOAD == 'dead'), 'WAVE needs the follower pressure'
assert LOAD in ('follower', 'dead')
E_solid, nu_solid = 100.0, 0.3
SOLVER_GTOL = 3e-6
SOLVER_MAXITER = 500
INNER_TOL, INNER_DELAY = 1e-6, 10
MIN_STEP = 1.0 / NB_INC / 64
INADMISSIBLE = 1e30

if DEVICE == 'gpu':
    import cupy as xp
else:
    xp = np


def to_host(a):
    return a.get() if hasattr(a, 'get') else np.asarray(a)


dim = 3
comm = MPI.COMM_WORLD
assert comm.size == 1, 'single rank only (node masks are built on the global grid)'
geo = bar_geometry(nnn, periodic=PERIODIC, BAR_L=BAR_L)
N, L, h, m = geo['N'], geo['L'], geo['h'], geo['margin']
disc = domain.Discretization(cell=domain.PeriodicUnitCell(domain_size=list(L), problem_type='elasticity'),
                             nb_of_pixels_global=N, discretization_type='finite_element',
                             element_type='trilinear_hexahedron', device=DEVICE)
out_dir = os.path.join(os.environ.get('RUNS_DIR', os.path.join(os.path.dirname(os.path.realpath(__file__)), 'runs')),
                       f'bar3d_n{nnn}_R{DEFAULTS["HOLE_R"]:g}_p{P_MAX:g}{"_per" if PERIODIC else ""}{"" if BAR_L == DEFAULTS["BAR_L"] else f"_L{BAR_L:g}"}{"" if WAVES == 1 else f"_k{WAVES:g}"}{f"_pv{P_VAR:g}" if P_VAR else ""}{"" if P_OFFSET == 1 else f"_po{P_OFFSET:g}"}'
                       f'{"_dead" if LOAD == "dead" else ""}{("_pulse" if PULSE else "_wave") if WAVE else ""}{"" if MATRIX_FREE else "_At"}_{PRECOND}_{DEVICE}_' + time.strftime('%Y%m%d-%H%M%S'))
os.makedirs(out_dir)
log_file = os.path.join(out_dir, 'run.log')


def log(*args):
    print(*args, flush=True)
    with open(log_file, 'a') as f:
        print(*args, file=f)


# ----------------------------------------------------------------------------- masks (host, then device)
solid_px, cav_px = geo['solid'], geo['cavity']
void_node = np.ones(N, bool)                       # node (i,j,k) = corner of voxels (i-1..i, j-1..j, k-1..k)
for a in (0, 1):
    for b in (0, 1):
        for c in (0, 1):
            void_node &= np.roll(~solid_px, (a, b, c), (0, 1, 2))
u_shape = disc.get_unknown_size_field(name='shape_probe').s.shape          # [3, 1, x, y, z]
ix = np.arange(N[0]).reshape(-1, 1, 1) * np.ones(N, int)
clamp_node = (ix == geo['clamp_x'][0]) | (ix == geo['clamp_x'][1])
fixed_h = np.zeros(u_shape, bool)
fixed_h[:, 0] = void_node | clamp_node
free = xp.asarray((~fixed_h).astype(float))
nb_q = disc.nb_quad_points_per_pixel
solid_q = xp.asarray(np.broadcast_to(solid_px, (nb_q, *N)))
cav_q = xp.asarray(np.broadcast_to(cav_px, (nb_q, *N)).astype(float))          # gas mask (volume)
p_shape = cav_px * (P_OFFSET + P_VAR * np.cos(2 * np.pi * WAVES * geo['Zc'] / L[2]))          # p(X) / p at the voxel centres
gas_q = xp.asarray(np.broadcast_to(p_shape, (nb_q, *N)).astype(float)) if LOAD == 'follower' else 0.0 * cav_q
phase_current = 0.0                                                             # t / T of the pressure wave


def pressure_profile(z, phase):
    """p(z, t) / p for t / T = phase (WAVE: travelling cosine; PULSE: one sine wavelength moving along z)"""
    if PULSE:
        zeta = np.mod(z - phase * PULSE_TRAVEL * L[2], L[2])
        ell = PULSE_LEN * L[2]
        return np.where(zeta < ell, np.sin(2 * np.pi * zeta / ell), 0.0)
    return P_OFFSET + P_VAR * np.cos(2 * np.pi * (WAVES * z / L[2] - phase))


def set_phase(phase, force=False):
    """WAVE: p(z, t) / p on the gas voxels for t / T = phase"""
    global gas_q, phase_current
    if phase != phase_current or force:
        shape = cav_px * pressure_profile(geo['Zc'], phase)
        gas_q = xp.asarray(np.broadcast_to(shape, (nb_q, *N)).astype(float))
        phase_current = phase
quad_w = xp.asarray(np.asarray(to_host(disc.quadrature_weights))).reshape(-1, 1, 1, 1)
I3 = xp.eye(3).reshape(3, 3, 1, 1, 1, 1)

mu_s = E_solid / (2 * (1 + nu_solid))
lam_s = E_solid * nu_solid / ((1 + nu_solid) * (1 - 2 * nu_solid))
kappa_s = lam_s + 2.0 * mu_s / 3.0
eps3 = np.zeros((3, 3, 3))
for (i, j, k), s in {(0, 1, 2): 1, (1, 2, 0): 1, (2, 0, 1): 1, (0, 2, 1): -1, (2, 1, 0): -1, (1, 0, 2): -1}.items():
    eps3[i, j, k] = s


# ----------------------------------------------------------------------------- material
def det3(F):
    return (F[0, 0] * (F[1, 1] * F[2, 2] - F[1, 2] * F[2, 1]) - F[0, 1] * (F[1, 0] * F[2, 2] - F[1, 2] * F[2, 0])
            + F[0, 2] * (F[1, 0] * F[2, 1] - F[1, 1] * F[2, 0]))


def cross_rows(a, b, mod):
    return mod.stack([a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]])


def cofactor(F):
    """cof F = J F^-T (finite for any F): row i = (row i+1) x (row i+2) of F"""
    mod = xp if not isinstance(F, np.ndarray) else np
    return mod.stack([cross_rows(F[1], F[2], mod), cross_rows(F[2], F[0], mod), cross_rows(F[0], F[1], mod)])


def neo_hookean(F, tangent=False):
    mod = xp if not isinstance(F, np.ndarray) else np
    J = det3(F)
    FiT = cofactor(F) / J
    lnJ = mod.log(J)
    I1 = mod.einsum('ij...,ij...->...', F, F)
    a = J ** (-2.0 / 3.0)
    W = 0.5 * mu_s * (a * I1 - 3.0) + 0.5 * kappa_s * lnJ ** 2
    P = mu_s * a * (F - I1 / 3.0 * FiT) + kappa_s * lnJ * FiT
    if not tangent:
        return W, P
    dd = mod.einsum('ik,jl->ijkl', mod.eye(3), mod.eye(3)).reshape((3, 3, 3, 3) + (1,) * (F.ndim - 2))
    FF = mod.einsum('il...,kj...->ijkl...', FiT, FiT)
    A_iso = (dd - 2.0 / 3.0 * (mod.einsum('ij...,kl...->ijkl...', F, FiT) + mod.einsum('ij...,kl...->ijkl...', FiT, F))
             + 2.0 / 9.0 * I1 * mod.einsum('ij...,kl...->ijkl...', FiT, FiT) + I1 / 3.0 * FF)
    A_vol = mod.einsum('ij...,kl...->ijkl...', FiT, FiT) - lnJ * FF
    return W, P, mu_s * a * A_iso + kappa_s * A_vol


def material(F, p, tangent=False):
    out = list(neo_hookean(xp.where(solid_q, F, I3), tangent))          # F = I in the void (no log of J <= 0)
    out = [xp.where(solid_q, v, 0.0) for v in out]
    if p != 0.0:
        w = p * gas_q
        out[0] = out[0] - w * det3(F)
        out[1] = out[1] - w * cofactor(F)
        if tangent:                                                      # d cof_ij / d F_kl = eps_ikm eps_jln F_mn
            e = xp.asarray(eps3)
            out[2] = out[2] - w * xp.einsum('ikm,jln,mn...->ijkl...', e, e, F)
    return tuple(out)


def stress_increment(st, G, p):
    """dP = A(F) : G (solid law + gas), from the stored state st = (F, F^-T, J^-2/3, ln J, I1); G = grad v"""
    F, FiT, a, lnJ, I1 = st[:5]
    trFiG = xp.einsum('ij...,ij...->...', FiT, G)
    trFG = xp.einsum('ij...,ij...->...', F, G)
    M = xp.einsum('il...,kl...,kj...->ij...', FiT, G, FiT)               # F^-T G^T F^-T
    dP = (mu_s * a * (G - 2.0 / 3.0 * F * trFiG - 2.0 / 3.0 * FiT * trFG + 2.0 / 9.0 * I1 * FiT * trFiG
                      + I1 / 3.0 * M) + kappa_s * (FiT * trFiG - lnJ * M))
    dP = xp.where(solid_q, dP, 0.0)
    if p != 0.0:                                     # d cof(F)[G] = cof(F + G) - cof(F) - cof(G) (cof is quadratic)
        Fg = st[5]
        dP = dP - p * gas_q * (cofactor(Fg + G) - cofactor(Fg) - cofactor(G))
    return dP


def mf_state(F):
    Fs = xp.where(solid_q, F, I3)
    J = det3(Fs)
    FiT = cofactor(Fs) / J
    return (Fs, FiT, J ** (-2.0 / 3.0), xp.log(J), xp.einsum('ij...,ij...->...', Fs, Fs), F)


# finite-difference check of P = dW/dF and A = dP/dF (solid law and gas term), on the host
_rng = np.random.default_rng(1)
_F = np.eye(3).reshape(3, 3, 1) + 0.2 * _rng.standard_normal((3, 3, 4))
_W, _P, _A = neo_hookean(_F, tangent=True)
_dF = 1e-6
_errP = _errA = _errC = 0.0
for k in range(3):
    for l_ in range(3):
        Fp, Fm = _F.copy(), _F.copy()
        Fp[k, l_] += _dF
        Fm[k, l_] -= _dF
        _errP = max(_errP, np.max(np.abs((neo_hookean(Fp)[0] - neo_hookean(Fm)[0]) / (2 * _dF) - _P[k, l_])))
        _errA = max(_errA, np.max(np.abs((neo_hookean(Fp)[1] - neo_hookean(Fm)[1]) / (2 * _dF) - _A[:, :, k, l_])))
        dC = np.einsum('ikm,jln,mn...->ijkl...', eps3, eps3, _F)[:, :, k, l_]
        _errC = max(_errC, np.max(np.abs((cofactor(Fp) - cofactor(Fm)) / (2 * _dF) - dC)))
        _errC = max(_errC, np.max(np.abs((det3(Fp) - det3(Fm)) / (2 * _dF) - cofactor(_F)[k, l_])))
fd_check = (_errP / np.max(np.abs(_P)), _errA / np.max(np.abs(_A)), _errC)

# ----------------------------------------------------------------------------- fields and operators
grad_u = disc.get_displacement_gradient_sized_field(name='grad_u')
stress = disc.get_displacement_gradient_sized_field(name='P')
tangent_field = disc.get_material_data_size_field_mugrid(name='A') if (not MATRIX_FREE or CHECK_MF) else None
mf = {'state': None}
names = ['x', 'g', 'v', 'Kv', 'r', 'Pr', 't1', 't2']
F_ = {k: disc.get_unknown_size_field(name=k) for k in names}
p_current = 0.0
force_scale = E_solid * h ** (dim - 1)


def deformation_gradient(x):
    F_['x'].s[...] = xp.asarray(np.reshape(x, u_shape))
    disc.fft.communicate_ghosts(F_['x'])
    disc.apply_gradient_operator_mugrid(u_inxyz=F_['x'], grad_u_ijqxyz=grad_u)
    return grad_u.s + I3


def fun_grad(x):
    F = deformation_gradient(x)
    if not float(xp.min(xp.where(solid_q, det3(F), xp.inf))) > 0.0:
        return INADMISSIBLE, np.zeros(np.shape(x))
    W, P = material(F, p_current)
    stress.s[...] = P
    disc.fft.communicate_ghosts(stress)
    disc.apply_gradient_transposed_operator_mugrid(gradient_field_ijqxyz=stress, div_u_fnxyz=F_['g'],
                                                   apply_weights=True)
    value, grad = float(xp.sum(quad_w * W)), to_host(F_['g'].s * free).reshape(-1)
    if LOAD == 'dead':
        value -= p_current * float(f_dead @ np.reshape(x, -1))
        grad = grad - p_current * f_dead
    return value / force_scale, grad.reshape(np.shape(x)) / force_scale


# LOAD=dead: f0 = dV_p/du at u = 0 (V_p = int p_shape J dX): B^T W (p_shape cof I), the nodal forces per unit p
if LOAD == 'dead':
    stress.s[...] = xp.asarray(np.broadcast_to(p_shape, (nb_q, *N))) * I3
    disc.fft.communicate_ghosts(stress)
    disc.apply_gradient_transposed_operator_mugrid(gradient_field_ijqxyz=stress, div_u_fnxyz=F_['g'],
                                                   apply_weights=True)
    f_dead = to_host(F_['g'].s * free).reshape(-1)


cache = {'x': None, 'p': None}


def update_tangent(x):
    x = np.asarray(x).reshape(-1)
    if cache['x'] is not None and cache['p'] == (p_current, phase_current) and np.array_equal(cache['x'], x):
        return
    if MATRIX_FREE:
        mf['state'] = mf_state(deformation_gradient(x))
    else:
        tangent_field.s[...] = material(deformation_gradient(x), p_current, tangent=True)[2]
    cache.update(x=x.copy(), p=(p_current, phase_current))


def apply_K(fi, fo, material_data):
    disc.fft.communicate_ghosts(fi)
    disc.apply_system_matrix_mugrid(material_data_field=material_data, input_field_inxyz=fi,
                                    output_field_inxyz=fo, formulation='finite_strain')


def apply_hessian(fi, fo):
    if MATRIX_FREE:
        def constitutive(gradient_field, stress_field):
            stress_field.s[...] = stress_increment(mf['state'], gradient_field.s, p_current)
        disc.apply_system_matrix_mugrid_explicit_stress(constitutive=constitutive, input_field_inxyz=fi,
                                                        output_field_inxyz=fo, formulation='finite_strain')
    else:
        apply_K(fi, fo, tangent_field)


cg_count = {'n': 0}


def hessp(x, v):
    cg_count['n'] += 1
    update_tangent(x)
    v = xp.asarray(np.reshape(v, u_shape))
    F_['v'].s[...] = v * free
    apply_hessian(F_['v'], F_['Kv'])
    return to_host(F_['Kv'].s * free + v * (1.0 - free)).reshape(-1) / force_scale


C_ref = to_host(neo_hookean(np.eye(3).reshape(3, 3, 1), tangent=True)[2][..., 0])
C_ref_dev = np.asarray(C_ref)                   # numpy: the library moves it to the field's device
green = disc.get_preconditioner_Green_mugrid(
    reference_material_data_ijkl=C_ref_dev, name='green_solid',
    operator=lambda input_field_inxyz, output_field_inxyz: apply_K(input_field_inxyz, output_field_inxyz, C_ref_dev))


def apply_green(fi, fo):
    disc.fft.communicate_ghosts(fi)
    disc.apply_preconditioner_mugrid(preconditioner_Fourier_fnfnqks=green, input_nodal_field_fnxyz=fi,
                                     output_nodal_field_fnxyz=fo)


setup_info = ''
if PRECOND == 'G_wall':
    t0 = time.time()
    planes = geo['clamp_x']
    ny, nz = N[1], N[2]
    kernel = xp.zeros((2, 3, 2, 3, ny, nz))                 # [plane p, comp c, plane q, comp e, dy, dz]
    for q, xq in enumerate(planes):
        for e in range(3):
            F_['t1'].s[...] = 0.0
            F_['t1'].s[e, 0, xq, 0, 0] = 1.0
            apply_green(F_['t1'], F_['t2'])
            for p_, xpl in enumerate(planes):
                kernel[p_, :, q, e] = F_['t2'].s[:, 0, xpl]
    symbol = xp.fft.fft2(kernel.reshape(6, 6, ny, nz), axes=(-2, -1)).reshape(6, 6, ny * nz).transpose(2, 0, 1)
    symbol = 0.5 * (symbol + xp.conj(symbol.transpose(0, 2, 1)))
    E3 = np.vstack([np.eye(3), np.eye(3)])                  # row (plane, comp) -> comp
    A0 = np.zeros((9, 9))
    A0[:6, :6] = to_host(symbol[0].real)
    A0[:6, 6:] = ny * nz * E3
    A0[6:, :6] = E3.T
    A0_inv = xp.asarray(np.linalg.inv(A0))
    symbol_inv = xp.linalg.inv(symbol[1:])

    def wall_solve(g, f):
        """[S B; B^T 0][lam; mu] = [g; f], g [plane, comp, y, z]"""
        g_hat = xp.fft.fft2(g.reshape(6, ny, nz), axes=(-2, -1)).reshape(6, ny * nz)
        lam_hat = xp.empty_like(g_hat)
        sol0 = A0_inv @ xp.concatenate([g_hat[:, 0].real, f])
        lam_hat[:, 0] = sol0[:6]
        lam_hat[:, 1:] = xp.einsum('kab,bk->ak', symbol_inv, g_hat[:, 1:])
        lam = xp.real(xp.fft.ifft2(lam_hat.reshape(6, ny, nz), axes=(-2, -1))).reshape(2, 3, ny, nz)
        return lam, sol0[6:]

    def wall_get(field):
        return xp.stack([field.s[:, 0, x] for x in planes])

    rng = np.random.default_rng(0)
    g_t, f_t = xp.asarray(rng.standard_normal((2, 3, ny, nz))), xp.asarray(rng.standard_normal(3))
    lam_t, mu_t = wall_solve(g_t, f_t)
    F_['t1'].s[...] = 0.0
    for p_, x in enumerate(planes):
        F_['t1'].s[:, 0, x] = lam_t[p_]
    apply_green(F_['t1'], F_['t2'])
    res = wall_get(F_['t2']) + mu_t.reshape(1, 3, 1, 1) - g_t
    setup_info = (f'; wall FFT: 2 planes x {ny} x {nz} nodes, setup {time.time() - t0:.2f} s, check '
                  f'|S lam + B mu - g|/|g| {float(xp.linalg.norm(res) / xp.linalg.norm(g_t)):.1e}, |B^T lam - f|/|f| '
                  f'{float(xp.linalg.norm(lam_t.sum(axis=(0, 2, 3)) - f_t) / xp.linalg.norm(f_t)):.1e}')


def precond(v):
    v = xp.asarray(np.reshape(v, u_shape))
    r = F_['r']
    r.s[...] = v * free
    if PRECOND == 'G_wall':
        apply_green(r, F_['t1'])
        lam_w, mu_w = wall_solve(wall_get(F_['t1']), xp.stack([r.s[c].sum() for c in range(3)]))
        F_['t2'].s[...] = r.s
        for p_, x in enumerate(planes):
            F_['t2'].s[:, 0, x] -= lam_w[p_]
        apply_green(F_['t2'], F_['Pr'])
        for c in range(3):
            F_['Pr'].s[c] -= mu_w[c]
    else:
        apply_green(r, F_['Pr'])
    return to_host(F_['Pr'].s * free + v * (1.0 - free)).reshape(-1)


def cavity_volume(x):
    return float(xp.sum(quad_w * cav_q * det3(deformation_gradient(x))))


mf_check = ''
if CHECK_MF:
    rng = np.random.default_rng(3)
    x_c = 0.1 * h * rng.standard_normal(int(np.prod(u_shape))) * to_host(free).reshape(-1)
    p_current = 5.0
    F_c = deformation_gradient(x_c)
    mf['state'] = mf_state(F_c)
    tangent_field.s[...] = material(F_c, p_current, tangent=True)[2]
    F_['v'].s[...] = xp.asarray(rng.standard_normal(u_shape)) * free
    apply_K(F_['v'], F_['t1'], tangent_field)
    _mf = MATRIX_FREE
    MATRIX_FREE = True
    apply_hessian(F_['v'], F_['t2'])
    MATRIX_FREE = _mf
    mf_check = (f'; matrix-free vs stored tangent at p = 5, random state: rel. diff '
                f'{float(xp.linalg.norm(F_["t2"].s - F_["t1"].s) / xp.linalg.norm(F_["t1"].s)):.1e}')
    p_current, mf['state'] = 0.0, None
    cache.update(x=None, p=None)


# ----------------------------------------------------------------------------- load stepping
log(f'3D hollow bar: h 1/{nnn}, cell {N} voxels (L {tuple(round(v, 4) for v in L)}), {DEFAULTS}, P_MAX {P_MAX}, '
    f'precond {PRECOND}, device {DEVICE}; solid voxels {int(solid_px.sum())}, cavity voxels {int(cav_px.sum())}, '
    f'pinned void nodes {int(void_node.sum())}, clamped nodes {int(clamp_node.sum())}, free dofs '
    f'{int(to_host(free).sum())}; FD check rel. err P {fd_check[0]:.1e}, A {fd_check[1]:.1e}, cof/J {fd_check[2]:.1e}'
    + f'; {"periodic open channel" if PERIODIC else "sealed bar"}, load {LOAD}, p(z) = p ({P_OFFSET:g} + {P_VAR:g} cos 2 pi z/Lz)'
    + setup_info + mf_check + f'; Hessian {"matrix-free" if MATRIX_FREE else "stored tangent"}')
inner_maxiter = 20 * int(np.prod(u_shape))
# muEye-ready NetCDF: per-voxel fields (sub-point 'pixel'), displacement in grid points
mueye = None
if MUEYE_NC:
    from muFFTTO import io_utils
    fc = disc.field_collection
    me = {k: fc.real_field(f'mueye_{k}', [n], 'pixel') for k, n in (('solid', 1), ('cavity', 1), ('det_F', 1),
                                                                     ('displacement', 3))}
    me['solid'].s[0, 0] = xp.asarray(solid_px.astype(float))
    me['cavity'].s[0, 0] = xp.asarray(cav_px.astype(float))
    void_node_ = void_node
    cav_node_me = void_node_ & ~clamp_node & np.any([np.roll(cav_px, (a, b, c), (0, 1, 2)) for a in (0, 1)
                                                     for b in (0, 1) for c in (0, 1)], axis=0)
    if MUEYE_FILL:
        from display_fill import HarmonicFill
        mueye_fill = HarmonicFill(cav_node_me)
    mueye = {'writer': io_utils.FieldWriter(os.path.join(out_dir, 'bar3d_mueye.nc'), list(me.values()),
                                            attributes={'nb_of_pixels': list(N), 'domain_lengths': list(L),
                                                        'displacement_units': f'grid points (x {h:g} for physical)',
                                                        'precond': PRECOND},
                                            frame_variables={'pressure': (), 'phase': (), 'cavity_volume_ratio': (),
                                                             'increment': ()}),
             'fields': me, 'prev': None}


def write_mueye(u_flat, F, increment):
    if mueye is None:
        return
    u3 = u_flat.reshape(u_shape)[:, 0]
    if MUEYE_FILL:
        u3 = mueye_fill(u3, start=mueye['prev'])
        mueye['prev'] = u3
    me = mueye['fields']
    me['displacement'].s[:, 0] = xp.asarray(u3 / h)
    me['det_F'].s[0, 0] = xp.where(xp.asarray(solid_px), xp.mean(det3(F), axis=0), 1.0)
    mueye['writer'].write(pressure=p_current, phase=phase_current,
                          cavity_volume_ratio=hist['volume'][-1] / hist['volume'][0], increment=increment)
u = np.zeros(int(np.prod(u_shape)))
lam, dlam, inc = 0.0, 1.0 / NB_INC, 0
hist = {'p': [0.0], 'volume': [cavity_volume(u)], 'u': [u.reshape(u_shape)[:, 0].astype(np.float32)], 'hessp': [0],
        'nit': [0], 'time': [0.0], 'min_J': [1.0], 'phase': [0.0], 'inner_cg': [], 'inner_inc': []}
write_mueye(u, deformation_gradient(u), 0)
if PULSE:
    set_phase(0.0, force=True)
lam_end = 1.0 + (N_PERIODS if WAVE else 0.0)          # load path: [0, 1] amplitude ramp, then t / T = lam - 1


def max_step(lam_):
    return 1.0 / NB_INC if lam_ < 1.0 - 1e-12 else 1.0 / NB_T


t_start = time.time()
while lam < lam_end - 1e-12:
    dlam = min(dlam, (1.0 if lam < 1.0 - 1e-12 else lam_end) - lam)
    p_current = min(lam + dlam, 1.0) * P_MAX
    set_phase(max(lam + dlam - 1.0, 0.0))
    t_inc = time.time()
    cg_count['n'] = 0
    inner = []

    def newton_callback(x_):
        inner.append(cg_count['n'])
        cg_count['n'] = 0

    res = solvers_nonlinear.minimize(fun=fun_grad, x0=u.copy(), method='tr_newton', hessp=hessp, precond=precond,
                                     callback=newton_callback,
                                     gtol=SOLVER_GTOL, maxiter=SOLVER_MAXITER, inner_tol=INNER_TOL,
                                     inner_stop='energy', inner_delay=INNER_DELAY, inner_maxiter=inner_maxiter,
                                     comm=comm)
    if not res.success or res.fun >= 0.5 * INADMISSIBLE:
        log(f'  p {p_current:7.3f}: {res.message} (nit {res.nit}, hessp {res.nb_hessp}) -> halve the step')
        dlam *= 0.5
        if dlam < MIN_STEP:
            log('step below the minimum: stop')
            break
        continue
    lam += dlam
    inc += 1
    u = np.asarray(res.x).reshape(-1)
    F = deformation_gradient(u)
    min_J = float(xp.min(xp.where(solid_q, det3(F), xp.inf)))
    hist['p'].append(p_current)
    hist['volume'].append(cavity_volume(u))
    last = not (min(lam, 1.0) * P_MAX < P_MAX - 1e-12 or lam < lam_end - 1e-12)
    store = (inc % OUTPUT_EVERY == 0) or last
    hist['u'].append(u.reshape(u_shape)[:, 0].astype(np.float32) if store else None)
    hist['hessp'].append(int(res.nb_hessp))
    hist['nit'].append(int(res.nit))
    hist['time'].append(time.time() - t_inc)
    hist['min_J'].append(min_J)
    hist['phase'].append(phase_current)
    hist['inner_cg'].extend(inner)
    hist['inner_inc'].extend([inc] * len(inner))
    if store:
        write_mueye(u, F, inc)
    log(f'inc {inc:3d}  p {p_current:7.3f}  t/T {phase_current:.3f}  cavity volume x{hist["volume"][-1] / hist["volume"][0]:.3f}  '
        f'min J solid {min_J:.3f}  max |u| {np.abs(u).max():.4f}  newton {res.nit:3d}  hessp {res.nb_hessp:6d}  '
        f'per solve {res.nb_hessp / max(res.nit, 1):6.1f}  inner CG {inner}  {time.time() - t_inc:6.1f} s'
        + (f'  GPU pool {xp.get_default_memory_pool().total_bytes() / 2**30:.1f} GB' if DEVICE == 'gpu' else ''))
    stored = [i for i, v in enumerate(hist['u']) if v is not None]
    np.savez(os.path.join(out_dir, 'state.npz'), **{k: np.asarray(v) for k, v in hist.items() if k != 'u'},
             u=np.asarray([hist['u'][i] for i in stored]), u_increments=np.asarray(stored),
             solid_px=solid_px, cav_px=cav_px, fixed=fixed_h, clamp_node=clamp_node, N=np.asarray(N), L=np.asarray(L),
             h=h, nnn=nnn, P_MAX=P_MAX, precond=PRECOND, wave=int(WAVE), P_VAR=P_VAR, P_OFFSET=P_OFFSET, WAVES=WAVES, pulse=int(PULSE), PULSE_LEN=PULSE_LEN, PULSE_TRAVEL=PULSE_TRAVEL,
             **{**DEFAULTS, 'BAR_L': BAR_L})
    dlam = min(2.0 * dlam, max_step(lam))
if mueye is not None:
    mueye['writer'].close()
log(f'elapsed {time.time() - t_start:.1f} s, {inc} increments, final p {p_current:.3f}, t/T {phase_current:.3f}; output {out_dir}')
