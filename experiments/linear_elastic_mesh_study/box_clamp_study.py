"""
Corrected Green when the fixed dofs are several full grid lines: can the capacitance system S = E^T G E still be
solved by FFTs along the lines?

A square (side 1, 2x2 holes of radius HOLE_RADIUS) in a periodic cell with a zero-stiffness void buffer of physical
thickness VOID_THICKNESS, loaded by a uniform body force in -y on the solid. Both displacement components are fixed on
full node lines through the whole periodic cell (void included, same problem as fixing only the solid's faces):

    CLAMP=two   the lines x = x0 and x = x1 (left and right faces): parallel lines, S block-circulant along y
    CLAMP=box   additionally y = y0 and y = y1 (all four faces): perpendicular lines, S is not circulant

INNER=  how S lam = g is solved inside the preconditioner (the outer CG is plain, energy stop CG_TOL, delay 10):
    dense       S assembled (one Green application per fixed dof), dense inverse: the reference
    cg          plain inner CG, relative residual INNER_TOL
    line        inner PCG, block-Jacobi over the single lines: each line's own S block inverted by its 1D FFT symbol
    family      inner PCG, block-Jacobi over the families of parallel lines (vertical / horizontal): each family's
                coupled S block inverted per wave number (a (2 lines x 2 components)^2 symbol)
    familydirect  the family preconditioner applied once, no iteration (exact when there is one family, CLAMP=two)
    kff         inner PCG preconditioned by K_FF = P K_ref P: the homogeneous reference stiffness applied to the
                zero-padded vector, restricted to the fixed dofs (the sub-block of K_ref; S^-1 is the sub-block of its
                Schur complement, K_FF - K_FI K_II^-1 K_IF, so this is only an approximation)
    strip       inner PCG preconditioned by the local Schur complement K_FF - K_FI K_II,w^-1 K_IF: K_II restricted to the
                free dofs within STRIP_W nodes (Chebyshev distance) of a fixed node, clamped at the strip's edge;
                K_ref assembled as a sparse matrix by probing, the strip block factorized once (STRIP_W=0 is K_FF)
PATCH=k (with INNER=family): a second level on the patches around the crossings, the fixed dofs within k nodes (along
the lines) of a crossing; their block S_ZZ is assembled (one Green application per patch dof, the full columns S Z kept)
and PATCH_MODE= combines it with the family preconditioner M^-1 (Q = Z S_ZZ^-1 Z^T):
    add   C^-1 = M^-1 + Q                               (additive)
    bal   C^-1 = Q + (I - Q S) M^-1 (I - S Q)           (multiplicative / balancing; S Q uses the stored S Z, so no
                                                         Green application beyond the PCG's own one per step)
A node where a vertical and a horizontal line cross belongs to both families (overlapping additive Schwarz: the
contributions of both are added); OVERLAP=0 gives it to the vertical family only, the horizontal lines' circulant
inverse then restricted to their remaining nodes (R C^-1 R^T, SPD but a poor approximation).

usage:  N=64 CLAMP=box INNER=family INNER_TOL=1e-6 python box_clamp_study.py   (run_box.sh for the sweep)
writes exp_data/runs_box/N<n>_<clamp>_<inner>[tol].npz; box_clamp_plot.py makes the figure
"""
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import time

import numpy as np

from muFFTTO import constraints
from muFFTTO import domain
from muFFTTO import material_models
from muFFTTO import solvers
from muFFTTO import topology_optimization

HERE = os.path.dirname(os.path.realpath(__file__))
RUNS = os.path.join(HERE, 'exp_data', 'runs_box')
os.makedirs(RUNS, exist_ok=True)

n = int(os.environ.get('N', 32))  # pixels over the square's side
clamp = os.environ.get('CLAMP', 'box')
inner = os.environ.get('INNER', 'dense')
inner_tol = float(os.environ.get('INNER_TOL', 1e-8))
void_thickness = float(os.environ.get('VOID_THICKNESS', 1 / 32))
hole_radius = float(os.environ.get('HOLE_RADIUS', 0.15))
cg_tol, cg_delay = float(os.environ.get('CG_TOL', 1e-6)), 10
body_force = -1e-3
lam, mu = material_models.get_lame_parameters(E=1.0, poisson=0.3)
patch = int(os.environ.get('PATCH', -1))
patch_mode = os.environ.get('PATCH_MODE', 'bal')
strip_w = int(os.environ.get('STRIP_W', 2))
name = f'N{n}_{clamp}_{inner}' + (f'{strip_w}' if inner == 'strip' else '') + \
       (f'{inner_tol:g}' if inner in ('cg', 'line', 'family', 'kff', 'strip') else '') + \
       ('_nooverlap' if os.environ.get('OVERLAP') == '0' else '') + \
       (f'_patch{patch}{patch_mode}' if patch >= 0 and inner == 'family' else '')

h = 1.0 / n
nb_void = int(round(void_thickness / h))
nb_px = n + 2 * nb_void
cell = domain.PeriodicUnitCell(domain_size=[nb_px * h, nb_px * h], problem_type='elasticity')
disc = domain.Discretization(cell=cell, nb_of_pixels_global=(nb_px, nb_px), discretization_type='finite_element',
                             element_type='bilinear_rectangle')
ix, iy = disc.fft.icoords
x0, x1 = nb_void, nb_void + n  # node indices of the square's faces (same in y)
solid = (ix >= x0) & (ix < x1) & (iy >= x0) & (iy < x1)
for xc in (0.25, 0.75):
    for yc in (0.25, 0.75):
        solid &= ((ix - x0 + 0.5) * h - xc) ** 2 + ((iy - x0 + 0.5) * h - yc) ** 2 > hole_radius ** 2

lam_q, mu_q = disc.get_quad_field_scalar(name='lam_q'), disc.get_quad_field_scalar(name='mu_q')
lam_q.s[...], mu_q.s[...] = 0, 0
lam_q.s[..., solid], mu_q.s[..., solid] = lam, mu
material = material_models.LinearElastic(discretization=disc, lam_1qxyz=lam_q, mu_1qxyz=mu_q, name='elasticity')

# fixed lines through the whole periodic cell: vertical lines (fixed ix) and, for the box, horizontal lines (fixed iy)
vertical = [x0, x1]
horizontal = [x0, x1] if clamp == 'box' else []
line_mask = np.zeros((nb_px, nb_px), dtype=bool)
for v in vertical:
    line_mask |= ix == v
for w in horizontal:
    line_mask |= iy == w
constrained = disc.get_displacement_sized_field(name='constrained')
constrained.s[...] = 0
constrained.s[:, 0, line_mask] = 1
prescribed = disc.get_displacement_sized_field(name='prescribed')
prescribed.s[...] = 0
nodal_constraints = constraints.NodalConstraints.from_mask(disc, constrained_mask=constrained,
                                                           prescribed_values=prescribed)

# uniform body force on the solid: each solid pixel gives h^2 / 4 of its force to each of its 4 nodes
force = disc.get_unknown_size_field(name='force')
force.s[...] = 0
share = 0.25 * h ** 2 * body_force * solid
force.s[1, 0] = share + np.roll(share, 1, 0) + np.roll(share, 1, 1) + np.roll(np.roll(share, 1, 0), 1, 1)


def K_fun(x, Ax):
    disc.apply_system_matrix_mugrid_explicit_stress(constitutive=material.get_stress, input_field_inxyz=x,
                                                    output_field_inxyz=Ax, formulation='small_strain')
    disc.fft.communicate_ghosts(Ax)


green = disc.get_preconditioner_Green_mugrid(
    reference_material_data_ijkl=material_models.get_elastic_tensor_from_lame(dim=2, lam=lam, mu=mu))


def M_fun(x, Px):
    disc.fft.communicate_ghosts(x)
    disc.apply_preconditioner_mugrid(preconditioner_Fourier_fnfnqks=green, input_nodal_field_fnxyz=x,
                                     output_nodal_field_fnxyz=Px)


t_setup = time.time()
lam_ref, mu_ref = disc.get_quad_field_scalar(name='lam_ref'), disc.get_quad_field_scalar(name='mu_ref')
lam_ref.s[...], mu_ref.s[...] = lam, mu
reference = material_models.LinearElastic(discretization=disc, lam_1qxyz=lam_ref, mu_1qxyz=mu_ref, name='reference')
Ky_in, Ky_out = disc.get_unknown_size_field(name='Ky_in'), disc.get_unknown_size_field(name='Ky_out')
kff_apps = [0]


def K_FF(y):  # P K_ref P y: reference stiffness of the zero-padded vector, restricted to the fixed dofs
    Ky_in.s[...] = 0
    Ky_in.s[fixed] = y
    disc.fft.communicate_ghosts(Ky_in)
    disc.apply_system_matrix_mugrid_explicit_stress(constitutive=reference.get_stress, input_field_inxyz=Ky_in,
                                                    output_field_inxyz=Ky_out, formulation='small_strain')
    kff_apps[0] += 1
    return Ky_out.s[fixed].copy()

fixed = tuple(np.argwhere(constrained.s != 0).T)  # (component, sub-point, x, y)
m = len(fixed[0])
dof_index = -np.ones(constrained.s.shape, dtype=int)
dof_index[fixed] = np.arange(m)
Ey, MEy = disc.get_unknown_size_field(name='Ey'), disc.get_unknown_size_field(name='MEy')
green_apps = [0]


def assemble_K_ref():
    """sparse K_ref by probing: the bilinear stencil couples nodes within +-1, so sources on a coloured sub-lattice of
    period p >= 3 (dividing nb_px) never share a receiver; 2 p^2 applications"""
    import scipy.sparse
    p = next(q for q in range(3, nb_px + 1) if nb_px % q == 0)
    gid = lambda d, x_, y_: (d * nb_px + x_) * nb_px + y_
    rows, cols, vals = [], [], []
    X, Y = np.meshgrid(np.arange(nb_px), np.arange(nb_px), indexing='ij')
    for cx in range(p):
        for cy in range(p):
            for e in range(2):
                Ky_in.s[...] = 0
                Ky_in.s[e, 0, cx::p, cy::p] = 1.0
                disc.fft.communicate_ghosts(Ky_in)
                disc.apply_system_matrix_mugrid_explicit_stress(constitutive=reference.get_stress,
                                                                input_field_inxyz=Ky_in, output_field_inxyz=Ky_out,
                                                                formulation='small_strain')
                for dx in (-1, 0, 1):
                    for dy in (-1, 0, 1):
                        sx_, sy_ = (X + dx) % nb_px, (Y + dy) % nb_px  # candidate source of each receiver
                        hit = (sx_ % p == cx) & (sy_ % p == cy)
                        for d in range(2):
                            rows.append(gid(d, X[hit], Y[hit]))
                            cols.append(gid(e, sx_[hit], sy_[hit]))
                            vals.append(Ky_out.s[d, 0][hit])
    K = scipy.sparse.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                                shape=(2 * nb_px ** 2,) * 2)
    return 0.5 * (K + K.T), 2 * p ** 2


def make_strip(K, w_):
    """local Schur complement K_FF - K_FIw K_IwIw^-1 K_IwF, I_w = free dofs within w_ nodes of a fixed node"""
    import scipy.sparse.linalg
    gid_fixed = (fixed[0] * nb_px + fixed[2]) * nb_px + fixed[3]
    near = line_mask.copy()
    for _ in range(w_):  # periodic dilation by one node (Chebyshev)
        near = near | np.roll(near, 1, 0) | np.roll(near, -1, 0)
        near = near | np.roll(near, 1, 1) | np.roll(near, -1, 1)
    strip_nodes = np.flatnonzero((near & ~line_mask).ravel())
    gid_strip = np.concatenate([d * nb_px ** 2 + strip_nodes for d in range(2)])
    K_FF_, K_FI_ = K[gid_fixed][:, gid_fixed], K[gid_fixed][:, gid_strip]
    if len(gid_strip) == 0:
        return lambda r: K_FF_ @ r, 0
    lu = scipy.sparse.linalg.splu(K[gid_strip][:, gid_strip].tocsc())
    return lambda r: K_FF_ @ r - K_FI_ @ lu.solve(K_FI_.T @ r), len(gid_strip)


def balance_translations(prec, SB_):
    """the Schur complement of the periodic K_ref annihilates the rigid translations (constants per component on F),
    S does not: balancing with the coarse space B, C^-1 = Q + (I - Q S) prec (I - S Q), Q = B (B^T S B)^-1 B^T"""
    Bf = np.stack([(fixed[0] == d).astype(float) for d in range(2)], axis=1)
    BSB_inv = np.linalg.inv(Bf.T @ SB_)

    def out(r):
        q = Bf @ (BSB_inv @ (Bf.T @ r))
        z = prec(r - SB_ @ (BSB_inv @ (Bf.T @ r)))
        return z - Bf @ (BSB_inv @ (SB_.T @ z)) + q
    return out


def S_fun(y):  # S y = E^T M E y, one Green application
    Ey.s[...] = 0
    Ey.s[fixed] = y
    M_fun(Ey, MEy)
    green_apps[0] += 1
    return MEy.s[fixed].copy()


def unit_response(component, node):  # M applied to a unit force at one node: the full field (no restriction)
    Ey.s[...] = 0
    Ey.s[(component, 0) + node] = 1.0
    M_fun(Ey, MEy)
    return MEy.s[:, 0].copy()


# a family = parallel lines; its nodes along the line direction t (all nb_px of them), lines at positions pos
families = [dict(axis=0, pos=vertical)] + ([dict(axis=1, pos=horizontal)] if horizontal else [])
mid = nb_px // 2
overlap = os.environ.get('OVERLAP', '1') == '1'
for fam in families:
    nl = len(fam['pos'])

    def node(p, t, fam=fam):  # grid node of line p at the coordinate t along it
        return (p, t) if fam['axis'] == 0 else (t, p)

    # kernel[(l, d), (l', e), k]: displacement d on line l at offset k from a unit force e on line l' at mid
    kernel = np.zeros((nl * 2, nl * 2, nb_px))
    for lp, p_src in enumerate(fam['pos']):
        for e in range(2):
            u_resp = unit_response(e, node(p_src, mid))
            for l, p in enumerate(fam['pos']):
                for d in range(2):
                    line_vals = u_resp[d][p, :] if fam['axis'] == 0 else u_resp[d][:, p]
                    kernel[2 * l + d, 2 * lp + e] = np.roll(line_vals, -mid)
    symbol = np.fft.fft(kernel, axis=-1).transpose(2, 0, 1)
    symbol = 0.5 * (symbol + np.conj(symbol.transpose(0, 2, 1)))
    ev, evec = np.linalg.eigh(symbol)
    fam['symbol_inv_family'] = np.einsum('kij,kj,klj->kil', evec, 1.0 / np.maximum(ev, 1e-12 * ev.max()), np.conj(evec))
    fam['symbol_inv_line'] = []  # each line on its own: the diagonal 2x2 blocks of the symbol
    for l in range(nl):
        blk = symbol[:, 2 * l:2 * l + 2, 2 * l:2 * l + 2]
        ev, evec = np.linalg.eigh(blk)
        fam['symbol_inv_line'].append(
            np.einsum('kij,kj,klj->kil', evec, 1.0 / np.maximum(ev, 1e-12 * ev.max()), np.conj(evec)))
    # dof index of (line l, component d, t); -1 where the node belongs to an earlier family (crossing points)
    idx = np.empty((nl, 2, nb_px), dtype=int)
    for l, p in enumerate(fam['pos']):
        for d in range(2):
            idx[l, d] = dof_index[d, 0, p, :] if fam['axis'] == 0 else dof_index[d, 0, :, p]
    if not overlap:
        owned = np.ones(m, dtype=bool)
        for f_prev in families[:families.index(fam)]:
            owned[f_prev['idx'][f_prev['idx'] >= 0]] = False
        idx[(idx >= 0) & ~owned[np.maximum(idx, 0)]] = -1
    fam['idx'] = idx
    green_apps[0] += 2 * nl


def C_inv(res, mode):  # block-Jacobi: each family (mode='family') or each single line (mode='line') by its FFT symbol
    out = np.zeros_like(res)
    for fam in families:
        idx, nl = fam['idx'], len(fam['pos'])
        vals = np.where(idx >= 0, res[np.maximum(idx, 0)], 0.0).reshape(2 * nl, nb_px)
        vals_hat = np.fft.fft(vals, axis=-1)
        if mode == 'family':
            sol_hat = np.einsum('kde,ek->dk', fam['symbol_inv_family'], vals_hat)
        else:
            sol_hat = np.concatenate([np.einsum('kde,ek->dk', fam['symbol_inv_line'][l], vals_hat[2 * l:2 * l + 2])
                                      for l in range(nl)])
        sol = np.real(np.fft.ifft(sol_hat, axis=-1)).reshape(nl, 2, nb_px)
        out[idx[idx >= 0]] += sol[idx >= 0]
    return out


def patch_dofs(k):
    """fixed dofs within k nodes (periodic distance along the line) of a crossing of a vertical and a horizontal line"""
    dist = lambda a, b: np.minimum(np.abs(a - b), nb_px - np.abs(a - b))
    near = np.zeros(m, dtype=bool)
    for v in vertical:
        for w_ in horizontal:
            near |= ((fixed[2] == v) & (dist(fixed[3], w_) <= k)) | ((fixed[3] == w_) & (dist(fixed[2], v) <= k))
    return np.flatnonzero(near)


def make_two_level(k, mode, S_cols):
    """family preconditioner + second level on the crossing patches; S_cols(Z) gives the columns S[:, Z]"""
    Z = patch_dofs(k)
    SZ = S_cols(Z)
    SZZ_inv = np.linalg.inv(0.5 * (SZ[Z] + SZ[Z].T))

    def prec(r):
        if mode == 'add':
            out = C_inv(r, 'family')
            out[Z] += SZZ_inv @ r[Z]
            return out
        q = SZZ_inv @ r[Z]
        w_ = C_inv(r - SZ @ q, 'family')
        w_[Z] -= SZZ_inv @ (SZ.T @ w_)
        w_[Z] += q
        return w_
    return prec, len(Z)


inner_its = []
spectra = {}
if inner == 'dense':
    S = np.stack([S_fun(np.eye(m)[j]) for j in range(m)], axis=1)
    S = 0.5 * (S + S.T)
    S_inv = np.linalg.inv(S)
    rng = np.random.default_rng(0)
    variants = [('none', lambda r: r), ('kff', K_FF), ('line', lambda r: C_inv(r, 'line')),
                ('family', lambda r: C_inv(r, 'family'))]
    K_sp = assemble_K_ref()[0]
    SB = S @ np.stack([(fixed[0] == d).astype(float) for d in range(2)], axis=1)
    for w_ in (0, 1, 2, 4, 8, 16, nb_px):
        st = make_strip(K_sp, w_)[0]
        variants.append((f'strip{w_ if w_ < nb_px else "all"}', st))
        variants.append((f'strip{w_ if w_ < nb_px else "all"}bal', balance_translations(st, SB)))
    if horizontal:
        for k in (0, 2, 8):
            for mode in ('add', 'bal'):
                variants.append((f'family_patch{k}{mode}', make_two_level(k, mode, lambda Z: S[:, Z])[0]))
    for label, prec in variants:  # how good the preconditioners are: spectrum of C^-1 S (small meshes)
        if m <= 3000:
            CS = np.stack([prec(S[:, j]) for j in range(m)], axis=1)
            ev = np.sort(np.linalg.eigvals(CS).real)
            spectra[f'eig_{label}'] = ev
            print(f'{label:20s}: eig(C^-1 S) in [{ev[0]:.4f}, {ev[-1]:.4f}], cond {ev[-1] / ev[0]:.3g}', flush=True)
    print(f'cond(S) = {np.linalg.cond(S):.3g}', flush=True)


def S_solve(g, tol):
    if inner == 'dense':
        return S_inv @ g
    if inner == 'familydirect':
        inner_its.append(0)
        return C_inv(g, 'family')
    prec = (lambda r: r) if inner == 'cg' else K_FF if inner == 'kff' else strip_prec if inner == 'strip' else \
        two_level if two_level else (lambda r: C_inv(r, inner))
    y, res = np.zeros_like(g), g.copy()
    zz = prec(res)
    p, rz, rr0 = zz.copy(), res @ zz, g @ g
    for it in range(1, 10 * m + 1):
        Sp = S_fun(p)
        a = rz / (p @ Sp)
        y += a * p
        res -= a * Sp
        if res @ res <= tol ** 2 * rr0:
            break
        zz = prec(res)
        rz, rz_old = res @ zz, rz
        p = zz + rz / rz_old * p
    inner_its.append(it)
    return y


two_level, nb_patch, nb_strip, probe_apps = None, 0, 0, 0
if inner == 'strip':
    K_sp, probe_apps = assemble_K_ref()
    strip_prec, nb_strip = make_strip(K_sp, strip_w)
    strip_prec = balance_translations(
        strip_prec, np.stack([S_fun((fixed[0] == d).astype(float)) for d in range(2)], axis=1))
    print(f'strip: w = {strip_w}, {nb_strip} strip dofs, K_ref probed with {probe_apps} applications', flush=True)
if patch >= 0 and inner == 'family':
    two_level, nb_patch = make_two_level(
        patch, patch_mode, lambda Z: np.stack([S_fun(np.eye(m)[j]) for j in Z], axis=1))
    print(f'second level: {nb_patch} patch dofs (crossings +-{patch} nodes), {patch_mode}', flush=True)
B = np.stack([(fixed[0] == d).astype(float) for d in range(2)], axis=1)  # E^T N
W = np.stack([S_solve(B[:, d], 1e-12) for d in range(2)], axis=1)
BW_inv = np.linalg.inv(B.T @ W)
inner_its.clear()
setup_apps = green_apps[0] + (m if inner == 'dense' else 0)
green_apps[0] = 0
setup_time = time.time() - t_setup
if inner in ('line', 'family', 'familydirect'):
    g_test = np.random.default_rng(1).standard_normal(m)
    print(f'|S C_family^-1 g - g| / |g| = {np.linalg.norm(S_fun(C_inv(g_test, "family")) - g_test) / np.linalg.norm(g_test):.2e}',
          flush=True)
    green_apps[0] = 0
print(f'{name}: {m} fixed dofs, {nb_px}^2 grid, setup {setup_time:.1f} s', flush=True)
Mr, w = disc.get_unknown_size_field(name='Mr'), disc.get_unknown_size_field(name='w')


def P_fun(r, z):  # [S B; B^T 0][lam; mu] = [E^T M r; N^T r], z = M (r - E lam) - N mu
    M_fun(r, Mr)
    Sg = S_solve(Mr.s[fixed], inner_tol)
    mu_ = BW_inv @ (B.T @ Sg - np.array([r.s[d].sum() for d in range(2)]))
    w.s[...] = r.s
    w.s[fixed] -= Sg - W @ mu_
    M_fun(w, z)
    for d in range(2):
        z.s[d] -= mu_[d]
    z.s[fixed] = 0.0
    green_apps[0] += 2


rhs, u = disc.get_unknown_size_field(name='rhs'), disc.get_unknown_size_field(name='u')
rhs.s[...] = force.s
nodal_constraints.apply_to_rhs(rhs, K_fun)
u.s[...] = 0
K_free = nodal_constraints.project_operator(K_fun)
t0 = time.time()
norms = solvers.conjugate_gradients_mugrid_experimental(
    comm=disc.communicator, fc=disc.field_collection, hessp=K_free, b=rhs, x=u, P=P_fun,
    maxiter=int(os.environ.get('CG_MAXITER', 5000)), stop_crit_norm='energy_lower_estim', d_min=cg_delay,
    **topology_optimization.energy_stop_kwargs(disc, K_free, u, cg_tol))[1]
cg_its = norms['stop_iteration']['energy_lower_estim']
cg_its = int(cg_its) if cg_its is not None else -len(norms['energy_lower_estim'])
wall_time = time.time() - t0
compliance = float(np.sum(force.s * u.s))
print(f'{name}: outer {cg_its}, inner/outer {np.mean(inner_its) if inner_its else 0:.1f}, Green apps {green_apps[0]} '
      f'(+ {setup_apps} setup), compliance {compliance:.10e}, {wall_time:.1f} s', flush=True)
np.savez(os.path.join(RUNS, name + '.npz'), n=n, clamp=clamp, inner=inner, inner_tol=inner_tol, m=m, cg_its=cg_its,
         inner_its=np.array(inner_its, dtype=int), green_apps=green_apps[0], setup_apps=setup_apps,
         compliance=compliance, time=wall_time, setup_time=setup_time, nb_patch=nb_patch,
         kff_apps=kff_apps[0], strip_w=strip_w, nb_strip=nb_strip, probe_apps=probe_apps, **spectra)
