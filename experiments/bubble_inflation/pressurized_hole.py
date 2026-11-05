"""
Free plate with one pressurised circular hole (finite strain, regular pixel grid)
=================================================================================

Stripped-down variant of bubble_inflation.py: no adapted grid, no HuHu regularization, no third medium (no contact),
no damage, no load predictor.

    cell [0, 1]^2, periodic, F_bar = I (no macro strain)
    cell    = [0, L]^2, L = PLATE + 2 t, pixel size h = 1 / NNN (NNN = pixels per unit length)
    plate   = pixels with centre in [t, L - t]^2 (side PLATE, fixed), split neo-Hookean solid, E = 100, nu = 0.3
              t = VOID_T is a PHYSICAL thickness; changing it changes only the void, not the plate (void rule: under refinement the void must keep its physical size,
              else Green's lambda_min -> 0 like a void of a few pixels; void_study.py), t / h and PLATE / h must be integers
    frame   = the pixels outside the plate: void, ZERO stiffness (free plate edges)
    hole    = GEOM=hole (default): pixels of the plate with centre within HOLE_R of HOLE_C;
              GEOM=channel: a straight horizontal channel of height CH_H through the middle of the plate, from the left
              to the right face (with CLAMP=1 its end edges lie on the clamped walls); void, zero stiffness, gas pressure p
    energy  Pi(u) = int_solid W(F) dX  -  p int_hole J dX,     p = lam P_MAX (follower pressure)

Void-only nodes (all 4 adjacent pixels void: frame and hole interior) carry no stiffness and are pinned (u = 0); the
pressure energy depends only on the nodes of the hole boundary (the signed element areas sum to the area of the
polygon; 2 x 2 Gauss integrates the Q1 J exactly), so the pin does not change the load. Void elements may invert,
det F > 0 is required in the solid only.

CLAMP=1 (default): the plate's left and right faces are clamped (u = 0). Each clamp is the FULL node column through
the void (zero stiffness there: the same problem), so the fixed set is two straight parallel lines plus the pinned
void nodes. CLAMP=0: free plate; rigid motions are removed by a statically determinate support (no reaction, the
pressure load is self-equilibrated): bottom-middle node fixed (u_x = u_y = 0), top-middle node u_x = 0.

Preconditioners (Green: C_ref = solid tangent at F = I):
    G       projected Green Q M Q (M = K_ref^-1 periodic, Q = projection on the free dofs)
    G_corr  corrected Green (experiments/linear_elastic_mesh_study, PRECOND=green_dirichlet): the exact inverse of the
            reference problem with the REAL clamp dofs only (the clamped solid faces, not the void parts of the clamp
            columns, not the pinned void nodes: correcting void dofs makes the reference a clamped void, far from the
            real free surface), by the capacitance system [S B; B^T 0] with S = E^T M E (dense, m Green applications,
            built once), B = E^T N (translations); every pinned void dof is projected out, Q M_corr Q
    G_wall  corrected Green for the two clamped full columns only, the capacitance system solved by a 1D FFT along the
            wall (no dense matrix): the periodic Green operator is translation invariant, so S = E^T M E on two full
            parallel node columns is block-circulant in y with a 4 x 4 symbol (2 lines x 2 components) per wave number;
            the translation constraints only act at wave number 0 (6 x 6 system there). Setup: 4 Green applications.
            The pinned void nodes (frame, hole interior; zero stiffness) are projected out, Q M_wall Q.
    GJ, D   Green-Jacobi (2 mu I) and dual, free plate tests only

Per load increment: plain trust-region Newton-CG (NuMPI) from the last converged state, inner CG stopped on the
energy-error estimate (rtol 1e-6, adaptive delay, minimum 10); the step is halved on failure.

Environment: NNN (32, pixels per unit length), PLATE (0.875), P_MAX (30), NB_INC (30), GEOM (hole | channel), HOLE_R (0.25), CH_H (0.25), VOID_T (1/16), CLAMP (1), PRECOND (G_wall | G_corr | G | GJ | D).
Output: runs/hole_<...>/ with run.log and state.npz (u per increment, p, hole area, CG counts).
"""
import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import numpy as np  # noqa: E402
from mpi4py import MPI  # noqa: E402

from muFFTTO import domain  # noqa: E402
from muFFTTO import solvers_nonlinear  # noqa: E402

# ----------------------------------------------------------------------------- parameters
nnn = int(os.environ.get('NNN', 32))
P_MAX = float(os.environ.get('P_MAX', 30.0))
NB_INC = int(os.environ.get('NB_INC', 30))
HOLE_R = float(os.environ.get('HOLE_R', 0.25))
GEOM = os.environ.get('GEOM', 'hole')
CH_H = float(os.environ.get('CH_H', 0.25))           # GEOM=channel: channel height
VOID_T = float(os.environ.get('VOID_T', 1 / 16))      # physical void thickness around the plate
PLATE = float(os.environ.get('PLATE', 0.875))         # plate side (physical, fixed)
MARGIN_PX = int(round(VOID_T * nnn))
PLATE_PX = int(round(PLATE * nnn))
assert MARGIN_PX >= 1 and abs(MARGIN_PX - VOID_T * nnn) < 1e-9, 'VOID_T must be a whole number of pixels'
assert abs(PLATE_PX - PLATE * nnn) < 1e-9, 'PLATE must be a whole number of pixels'
ncell = PLATE_PX + 2 * MARGIN_PX                       # pixels per direction of the cell
L = ncell / nnn                                        # cell side
HOLE_C = (0.5 * L, 0.5 * L)
PRECOND = os.environ.get('PRECOND', 'G_corr')        # G_corr = corrected Green, G = projected Green, GJ, D
CLAMP = os.environ.get('CLAMP', '1') == '1'          # clamp the plate's left and right faces
E_solid, nu_solid = 100.0, 0.3
SOLVER_GTOL = 3e-6
SOLVER_MAXITER = 500
INNER_TOL, INNER_DELAY = 1e-6, 10
MIN_STEP = 1.0 / NB_INC / 64
INADMISSIBLE = 1e30
DUAL_CUTOFF = 1e-6

dim = 2
N = (ncell, ncell)
h = 1.0 / nnn
comm = MPI.COMM_WORLD
assert comm.size == 1, 'single rank only (node masks are built on the global grid)'

disc = domain.Discretization(cell=domain.PeriodicUnitCell(domain_size=[L, L], problem_type='elasticity'),
                             nb_of_pixels_global=N, discretization_type='finite_element',
                             element_type='bilinear_rectangle', device='cpu')

out_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'runs',
                       (f'hole_n{nnn}_R{HOLE_R:g}_' if GEOM == 'hole' else f'channel_n{nnn}_H{CH_H:g}_') + f'p{P_MAX:g}_t{VOID_T:g}{"_clampLR" if CLAMP else "_free"}_{PRECOND}_' + time.strftime('%Y%m%d-%H%M%S'))
os.makedirs(out_dir)
log_file = os.path.join(out_dir, 'run.log')


def log(*args):
    print(*args, flush=True)
    with open(log_file, 'a') as f:
        print(*args, file=f)


# ----------------------------------------------------------------------------- geometry (pixels)
coords = np.asarray(disc.fft.coords)                                    # nodes X, [2, x, y]
centre = coords * L + 0.5 * h                                               # pixel centres
in_plate = np.all((centre > MARGIN_PX * h) & (centre < L - MARGIN_PX * h), axis=0)
in_hole = in_plate & ((np.hypot(centre[0] - HOLE_C[0], centre[1] - HOLE_C[1]) < HOLE_R) if GEOM == 'hole'
                      else (np.abs(centre[1] - HOLE_C[1]) < 0.5 * CH_H))
solid_px = in_plate & ~in_hole
nb_q = disc.nb_quad_points_per_pixel
solid_q = np.broadcast_to(solid_px, (nb_q, *N))
hole_q = np.broadcast_to(in_hole, (nb_q, *N)).astype(float)

# node (i, j) = lower-left corner of pixel (i, j); its 4 pixels are (i-1..i, j-1..j)
void_node = np.ones(N, bool)
for a in (0, 1):
    for b in (0, 1):
        void_node &= np.roll(~solid_px, (a, b), (0, 1))
i_mid = ncell // 2
bottom, top = MARGIN_PX, ncell - MARGIN_PX                                # node rows of the plate's bottom / top edge
u_shape = disc.get_unknown_size_field(name='shape_probe').s.shape       # [2, 1, x, y]
fixed = np.zeros(u_shape, bool)
fixed[:, 0] = void_node
ix = np.arange(ncell)[:, None] * np.ones((1, ncell), int)
clamp_node = CLAMP & ((ix == MARGIN_PX) | (ix == ncell - MARGIN_PX))     # full node columns (through the void)
if CLAMP:
    fixed[:, 0] |= clamp_node
else:
    fixed[0, 0, i_mid, bottom] = fixed[1, 0, i_mid, bottom] = True      # support: pin
    fixed[0, 0, i_mid, top] = True                                      # support: roller (u_x = 0)
free = (~fixed).astype(float)

mu_s = E_solid / (2 * (1 + nu_solid))
lam_s = E_solid * nu_solid / ((1 + nu_solid) * (1 - 2 * nu_solid))
kappa_s = lam_s + 2.0 * mu_s / 3.0
quad_w = np.asarray(disc.quadrature_weights).reshape(-1, 1, 1)
identity = np.eye(2).reshape(2, 2, 1, 1, 1)


# ----------------------------------------------------------------------------- material
def neo_hookean(F, tangent=False):
    """W, P (and A) of W = mu/2 (J^-2/3 (I1 + 1) - 3) + kappa/2 ln(J)^2 (plane strain), pointwise."""
    J = F[0, 0] * F[1, 1] - F[0, 1] * F[1, 0]
    FiT = np.stack([np.stack([F[1, 1], -F[1, 0]]), np.stack([-F[0, 1], F[0, 0]])]) / J
    lnJ = np.log(J)
    I1 = np.einsum('ij...,ij...->...', F, F)
    a = J ** (-2.0 / 3.0)
    b = (I1 + 1.0) / 3.0
    W = 0.5 * mu_s * (a * (I1 + 1.0) - 3.0) + 0.5 * kappa_s * lnJ ** 2
    P = mu_s * a * (F - b * FiT) + kappa_s * lnJ * FiT
    if not tangent:
        return W, P
    dd = np.einsum('ik,jl->ijkl', np.eye(2), np.eye(2)).reshape((2, 2, 2, 2) + (1,) * (F.ndim - 2))
    FiT_il_FiT_kj = np.einsum('il...,kj...->ijkl...', FiT, FiT)
    A_iso = (-2.0 / 3.0 * np.einsum('kl...,ij...->ijkl...', FiT, F - b * FiT) + dd
             - 2.0 / 3.0 * np.einsum('kl...,ij...->ijkl...', F, FiT) + b * FiT_il_FiT_kj)
    A_vol = np.einsum('ij...,kl...->ijkl...', FiT, FiT) - lnJ * FiT_il_FiT_kj
    return W, P, mu_s * a * A_iso + kappa_s * A_vol


def material(F, p, tangent=False, with_pressure=True):
    """solid: neo-Hookean; void: nothing; hole: the gas, energy -p J."""
    out = list(neo_hookean(np.where(solid_q, F, identity), tangent))     # F = I in the void: no log of J <= 0
    out = [np.where(solid_q, v, 0.0) for v in out]
    if with_pressure and p != 0.0:
        J = F[0, 0] * F[1, 1] - F[0, 1] * F[1, 0]
        cof = np.stack([np.stack([F[1, 1], -F[1, 0]]), np.stack([-F[0, 1], F[0, 0]])])   # J F^-T (finite for J <= 0)
        w = p * hole_q
        out[0] = out[0] - w * J
        out[1] = out[1] - w * cof
        if tangent:                                                     # d(cof)/dF: d cof_ij / d F_kl
            dcof = np.zeros((2, 2, 2, 2))
            dcof[0, 0, 1, 1] = dcof[1, 1, 0, 0] = 1.0
            dcof[0, 1, 1, 0] = dcof[1, 0, 0, 1] = -1.0
            out[2] = out[2] - w * dcof.reshape(2, 2, 2, 2, 1, 1, 1)
    return tuple(out)


# ----------------------------------------------------------------------------- fields and operators
grad_u = disc.get_displacement_gradient_sized_field(name='grad_u')
stress = disc.get_displacement_gradient_sized_field(name='P')
tangent_field = disc.get_material_data_size_field_mugrid(name='A')
dual_field = disc.get_material_data_size_field_mugrid(name='A_dual')
u_field = disc.get_unknown_size_field(name='u')
x_field = disc.get_unknown_size_field(name='x')
g_field = disc.get_unknown_size_field(name='g')
v_field = disc.get_unknown_size_field(name='v')
Kv_field = disc.get_unknown_size_field(name='Kv')
r_field = disc.get_unknown_size_field(name='r')
Pr_field = disc.get_unknown_size_field(name='Pr')
t1_field = disc.get_unknown_size_field(name='t1')
t2_field = disc.get_unknown_size_field(name='t2')
p_current = 0.0
force_scale = E_solid * h ** (dim - 1)


def deformation_gradient(x):
    x_field.s[...] = np.reshape(x, u_shape)
    disc.fft.communicate_ghosts(x_field)
    disc.apply_gradient_operator_mugrid(u_inxyz=x_field, grad_u_ijqxyz=grad_u)
    return grad_u.s + identity


def fun_grad(x):
    F = deformation_gradient(x)
    J = F[0, 0] * F[1, 1] - F[0, 1] * F[1, 0]
    if not np.min(np.where(solid_q, J, np.inf)) > 0.0:
        return INADMISSIBLE, np.zeros(np.shape(x))
    W, P = material(F, p_current)
    stress.s[...] = P
    disc.fft.communicate_ghosts(stress)
    disc.apply_gradient_transposed_operator_mugrid(gradient_field_ijqxyz=stress, div_u_fnxyz=g_field,
                                                   apply_weights=True)
    return float(np.sum(quad_w * W)) / force_scale, (g_field.s * free).reshape(np.shape(x)) / force_scale


cache = {'x': None, 'p': None}


def update_tangent(x):
    x = np.asarray(x).reshape(-1)
    if cache['x'] is not None and cache['p'] == p_current and np.array_equal(cache['x'], x):
        return
    F = deformation_gradient(x)
    tangent_field.s[...] = material(F, p_current, tangent=True)[2]
    cache.update(x=x.copy(), p=p_current)


def apply_K(field_in, field_out, material_data):
    disc.fft.communicate_ghosts(field_in)
    disc.apply_system_matrix_mugrid(material_data_field=material_data, input_field_inxyz=field_in,
                                    output_field_inxyz=field_out, formulation='finite_strain')


def hessp(x, v):
    """masked Hessian: H on the free dofs, identity on the fixed ones"""
    update_tangent(x)
    v = np.reshape(v, u_shape)
    v_field.s[...] = v * free
    apply_K(v_field, Kv_field, tangent_field)
    return (Kv_field.s * free + v * (1.0 - free)).reshape(-1) / force_scale


# preconditioners: Green of C_ref (GJ: 2 mu I, G and D: the solid tangent at F = I), separate field names
C_solid = neo_hookean(np.eye(2).reshape(2, 2, 1), tangent=True)[2][..., 0]
C_ref = 2 * mu_s * np.einsum('ik,jl->ijkl', np.eye(2), np.eye(2)) if PRECOND == 'GJ' else C_solid
green = disc.get_preconditioner_Green_mugrid(
    reference_material_data_ijkl=C_ref, name=f'green_{PRECOND}',
    operator=lambda input_field_inxyz, output_field_inxyz: apply_K(input_field_inxyz, output_field_inxyz, C_ref))
jacobi = None
d_ref = float(np.max(1.0 / disc.get_preconditioner_Jacobi_mugrid(
    operator=lambda input_field_inxyz, output_field_inxyz: apply_K(input_field_inxyz, output_field_inxyz, C_ref), name='jacobi_ref').s ** 2))


def update_preconditioner(x):
    global jacobi
    update_tangent(x)
    if PRECOND == 'GJ':
        def masked_K(input_field_inxyz, output_field_inxyz):
            fi, fo = input_field_inxyz, output_field_inxyz
            apply_K(fi, fo, tangent_field)
            fo.s[...] = fo.s * free + fi.s * (1.0 - free)
        diag_inv_sqrt = disc.get_preconditioner_Jacobi_mugrid(operator=masked_K, name='jacobi_K').s
        if not np.all(np.isfinite(diag_inv_sqrt)):
            raise RuntimeError('non-positive Jacobi diagonal')
        jacobi = np.sqrt(d_ref) * diag_inv_sqrt
    elif PRECOND == 'D':                     # C_ref A^+ C_ref of the MATERIAL tangent (no gas): zero in the void
        F = deformation_gradient(x)
        A = material(F, 0.0, tangent=True, with_pressure=False)[2]
        shape = A.shape
        A = np.moveaxis(A.reshape((4, 4) + shape[4:]), (0, 1), (-2, -1))
        A = 0.5 * (A + np.swapaxes(A, -1, -2))
        ev, evec = np.linalg.eigh(A)
        ev = np.abs(ev)
        keep = ev > DUAL_CUTOFF * np.maximum(np.max(ev, axis=-1, keepdims=True), 1e-300)
        inv = np.einsum('...ab,...b,...cb->...ac', evec, np.where(keep, 1.0 / np.where(keep, ev, 1.0), 0.0), evec)
        C = C_ref.reshape(4, 4)
        dual_field.s[...] = np.moveaxis(np.einsum('ab,...bc,cd->...ad', C, inv, C), (-2, -1), (0, 1)).reshape(shape)


def apply_green(fi, fo):
    disc.fft.communicate_ghosts(fi)
    disc.apply_preconditioner_mugrid(preconditioner_Fourier_fnfnqks=green, input_nodal_field_fnxyz=fi,
                                     output_nodal_field_fnxyz=fo)


if PRECOND == 'G_corr':
    fixed_idx = tuple(np.argwhere(fixed & ~void_node[None, None]).T)   # real clamp dofs only (component, sub-point, x, y)
    m = len(fixed_idx[0])
    t_setup = time.time()
    A_cap = np.zeros((m + 2, m + 2))
    for j in range(m):                                                  # S = E^T M E, one Green application per dof
        t1_field.s[...] = 0.0
        t1_field.s[tuple(f[j] for f in fixed_idx)] = 1.0
        apply_green(t1_field, t2_field)
        A_cap[:m, j] = t2_field.s[fixed_idx]
    for c in range(2):                                                  # B = E^T N: the fixed dofs a translation moves
        A_cap[:m, m + c] = A_cap[m + c, :m] = (fixed_idx[0] == c)
    A_cap_inv = np.linalg.inv(A_cap)
    corr_setup = (m, time.time() - t_setup)

if PRECOND == 'G_wall':
    assert CLAMP, 'G_wall needs the clamped columns'
    t_setup = time.time()
    lines = (MARGIN_PX, ncell - MARGIN_PX)                                # node columns x of the two walls
    kernel = np.zeros((2, 2, 2, 2, ncell))                                # [line p, comp c, line q, comp e, offset]
    for q, xq in enumerate(lines):                                      # unit impulse at (comp e, line q, y = 0)
        for e in range(2):
            t1_field.s[...] = 0.0
            t1_field.s[e, 0, xq, 0] = 1.0
            apply_green(t1_field, t2_field)
            for p_, xp in enumerate(lines):
                kernel[p_, :, q, e] = t2_field.s[:, 0, xp, :]
    symbol = np.fft.fft(kernel.reshape(4, 4, ncell), axis=-1).transpose(2, 0, 1)     # [k, 4, 4]
    symbol = 0.5 * (symbol + np.conj(symbol.transpose(0, 2, 1)))
    E2 = np.array([[1, 0], [0, 1], [1, 0], [0, 1]], float)              # row (line, comp) -> comp
    A0 = np.zeros((6, 6))
    A0[:4, :4] = symbol[0].real
    A0[:4, 4:] = ncell * E2
    A0[4:, :4] = E2.T
    A0_inv = np.linalg.inv(A0)
    symbol_inv = np.linalg.inv(symbol[1:])
    wall_idx = (slice(None), 0, list(lines), slice(None))               # u[:, 0, lines, :] -> [comp, line, y]

    def wall_solve(g, f):
        """[S B; B^T 0][lam; mu] = [g; f], g [line, comp, y]: by the FFT along y"""
        g_hat = np.fft.fft(g.reshape(4, ncell), axis=-1)
        lam_hat = np.empty_like(g_hat)
        sol0 = A0_inv @ np.concatenate([g_hat[:, 0].real, f])
        lam_hat[:, 0] = sol0[:4]
        lam_hat[:, 1:] = np.einsum('kab,bk->ak', symbol_inv, g_hat[:, 1:])
        return np.real(np.fft.ifft(lam_hat, axis=-1)).reshape(2, 2, ncell), sol0[4:]

    def wall_get(field):
        return np.stack([field.s[:, 0, x, :] for x in lines])           # [line, comp, y]

    def wall_sub(field, lam):
        for p_, x in enumerate(lines):
            field.s[:, 0, x, :] -= lam[p_]

    # check once: residual of the saddle system with S applied by Green (exact up to round-off)
    rng = np.random.default_rng(0)
    g_test, f_test = rng.standard_normal((2, 2, ncell)), rng.standard_normal(2)
    lam_t, mu_t = wall_solve(g_test, f_test)
    t1_field.s[...] = 0.0
    for p_, x in enumerate(lines):
        t1_field.s[:, 0, x, :] = lam_t[p_]
    apply_green(t1_field, t2_field)
    res = wall_get(t2_field) + mu_t.reshape(1, 2, 1) - g_test
    wall_check = (np.linalg.norm(res) / np.linalg.norm(g_test),
                  np.linalg.norm(lam_t.sum(axis=(0, 2)) - f_test) / np.linalg.norm(f_test))
    corr_setup = (4 * ncell, time.time() - t_setup)


def precond(v):
    v = np.reshape(v, u_shape)
    r_field.s[...] = v * free
    if PRECOND == 'G_wall':                 # the same correction for the wall dofs only, by the wall FFT
        apply_green(r_field, t1_field)
        lam_w, mu_w = wall_solve(wall_get(t1_field), np.array([r_field.s[c].sum() for c in range(2)]))
        t2_field.s[...] = r_field.s
        wall_sub(t2_field, lam_w)
        apply_green(t2_field, Pr_field)
        for c in range(2):
            Pr_field.s[c] -= mu_w[c]
    elif PRECOND == 'G_corr':               # z = M (r - E lam) - N mu,  [S B; B^T 0][lam; mu] = [E^T M r; N^T r]
        apply_green(r_field, t1_field)
        sol = A_cap_inv @ np.concatenate([t1_field.s[fixed_idx], [r_field.s[c].sum() for c in range(2)]])
        t2_field.s[...] = r_field.s
        t2_field.s[fixed_idx] -= sol[:m]
        apply_green(t2_field, Pr_field)
        for c in range(2):
            Pr_field.s[c] -= sol[m + c]
    elif PRECOND == 'D':
        apply_green(r_field, t1_field)
        apply_K(t1_field, t2_field, dual_field)
        apply_green(t2_field, Pr_field)
    else:
        if PRECOND == 'GJ':
            r_field.s[...] *= jacobi
        apply_green(r_field, Pr_field)
        if PRECOND == 'GJ':
            Pr_field.s[...] *= jacobi
    return (Pr_field.s * free + v * (1.0 - free)).reshape(-1)


def hole_area(x):
    F = deformation_gradient(x)
    return float(np.sum(quad_w * hole_q * (F[0, 0] * F[1, 1] - F[0, 1] * F[1, 0])))


# ----------------------------------------------------------------------------- load stepping
log(f'pressurised {GEOM}: h 1/{nnn}, cell {ncell}^2 px (L {L:g}), plate {PLATE:g}, R {HOLE_R}, P_MAX {P_MAX}, void t {VOID_T:g} ({MARGIN_PX} px), precond {PRECOND}; '
    f'solid pixels {int(solid_px.sum())}, hole pixels {int(in_hole.sum())}, pinned void nodes {int(void_node.sum())}, '
    f'free dofs {int(free.sum())}, ' + (f'clamped columns x = {MARGIN_PX}, {ncell - MARGIN_PX} h' if CLAMP else 'free plate')
    + (f'; corrected Green: {corr_setup[0]} fixed dofs, setup {corr_setup[1]:.1f} s' if PRECOND == 'G_corr' else '')
    + (f'; wall FFT: {corr_setup[0]} wall dofs, setup {corr_setup[1]:.2f} s, check |S lam + B mu - g|/|g| '
       f'{wall_check[0]:.1e}, |B^T lam - f|/|f| {wall_check[1]:.1e}' if PRECOND == 'G_wall' else ''))
inner_maxiter = 20 * int(np.prod(u_shape))
u = np.zeros(int(np.prod(u_shape)))
lam, dlam, inc = 0.0, 1.0 / NB_INC, 0
history = {'p': [0.0], 'area': [hole_area(u)], 'u': [u.reshape(u_shape)[:, 0].copy()], 'hessp': [0], 'nit': [0],
           'time': [0.0], 'min_J': [1.0]}
t0 = time.time()
while lam < 1.0 - 1e-12:
    dlam = min(dlam, 1.0 - lam)
    p_current = (lam + dlam) * P_MAX
    update_preconditioner(u)
    t_inc = time.time()
    res = solvers_nonlinear.minimize(fun=fun_grad, x0=u.copy(), method='tr_newton', hessp=hessp, precond=precond,
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
    J = F[0, 0] * F[1, 1] - F[0, 1] * F[1, 0]
    min_J = float(np.min(J[solid_q]))
    history['p'].append(p_current)
    history['area'].append(hole_area(u))
    history['u'].append(u.reshape(u_shape)[:, 0].copy())
    history['hessp'].append(int(res.nb_hessp))
    history['nit'].append(int(res.nit))
    history['time'].append(time.time() - t_inc)
    history['min_J'].append(min_J)
    log(f'inc {inc:3d}  p {p_current:7.3f}  hole area {history["area"][-1]:.4f} '
        f'(x{history["area"][-1] / history["area"][0]:.3f})  min J solid {min_J:.3f}  max |u| {np.abs(u).max():.4f}  '
        f'newton {res.nit:3d}  hessp {res.nb_hessp:6d}  {time.time() - t_inc:6.1f} s')
    np.savez(os.path.join(out_dir, 'state.npz'), **{k: np.asarray(v) for k, v in history.items()},
             solid_px=solid_px, hole_px=in_hole, fixed=fixed, clamp_node=clamp_node, nnn=nnn, ncell=ncell, L=L, h=h, void_t=VOID_T, P_MAX=P_MAX, HOLE_R=HOLE_R, precond=PRECOND)
    dlam = min(2.0 * dlam, 1.0 / NB_INC)
log(f'elapsed {time.time() - t0:.1f} s, {inc} increments, final p {lam * P_MAX:.3f}; output {out_dir}')
