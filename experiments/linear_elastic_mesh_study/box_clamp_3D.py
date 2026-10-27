"""
3D version of box_clamp_study.py: a cube clamped on all six faces, the corrected Green's capacitance system
S = E^T G E solved without (or with) assembling it.

A cube (side 1, 8 spherical holes of radius HOLE_RADIUS) in a periodic cell with a zero-stiffness void buffer of
physical thickness VOID_THICKNESS (at least one voxel), loaded by a uniform body force in -y on the solid. All three
displacement components are fixed on the six face planes x, y, z = x0 and x1, each a full grid plane through the whole
periodic cell (void included: same problem). Plain outer CG, energy stop 1e-6, delay 10.

INNER=
    green       plain (projected) Green on the same fixed dofs: the baseline
    dense       S assembled (one Green application per fixed dof), dense inverse
    cg          plain inner CG, relative residual INNER_TOL
    family      inner PCG, block-Jacobi over the 3 families of parallel planes (normal x, y, z); each family's two
                planes inverted exactly by a 2D FFT over the plane, a 6x6 block (2 planes x 3 components) per wave-number
                pair; nodes on the edges and corners belong to every family that contains them (contributions summed)
    familydirect  the family preconditioner applied once, no iteration
PATCH=k (with INNER=family): a balancing second level on strips of +-k nodes along the 12 edge lines (where two
perpendicular planes cross; full lines through the cell). The 4 parallel edges of one direction form a group whose
block S_ZZ is circulant along the edges: inverted exactly by a 1D FFT along them (setup: one Green application per dof of
a cross-section of the group). The three groups are summed (Q), and
    C^-1 = Q + (I - Q S) M^-1 (I - S Q)
costs two Green applications per inner step on top of the PCG's own one (S Z is too large to store in 3D).

usage:  N=16 INNER=family INNER_TOL=1e-6 python box_clamp_3D.py      (run_box_3D.sh for the sweep)
writes exp_data/runs_box3d/N<n>_<inner>[tol].npz (with the displacement on the cube's nodes for N <= 32)
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
RUNS = os.path.join(HERE, 'exp_data', 'runs_box3d')
os.makedirs(RUNS, exist_ok=True)

n = int(os.environ.get('N', 16))
inner = os.environ.get('INNER', 'family')
inner_tol = float(os.environ.get('INNER_TOL', 1e-6))
void_thickness = float(os.environ.get('VOID_THICKNESS', 1 / 16))
hole_radius = float(os.environ.get('HOLE_RADIUS', 0.15))
cg_tol, cg_delay = 1e-6, 10
body_force = -1e-3
lam, mu = material_models.get_lame_parameters(E=1.0, poisson=0.3)
patch = int(os.environ.get('PATCH', -1))
name = f'N{n}_{inner}' + (f'{inner_tol:g}' if inner in ('cg', 'family') else '') + \
       (f'_patch{patch}bal' if patch >= 0 and inner == 'family' else '')

h = 1.0 / n
nb_void = max(1, int(round(void_thickness / h)))
nb_px = n + 2 * nb_void
cell = domain.PeriodicUnitCell(domain_size=[nb_px * h] * 3, problem_type='elasticity')
disc = domain.Discretization(cell=cell, nb_of_pixels_global=(nb_px,) * 3, discretization_type='finite_element',
                             element_type='trilinear_hexahedron')
coords = disc.fft.icoords
x0, x1 = nb_void, nb_void + n
solid = np.ones(coords[0].shape, dtype=bool)
for c in coords:
    solid &= (c >= x0) & (c < x1)
for xc in (0.25, 0.75):
    for yc in (0.25, 0.75):
        for zc in (0.25, 0.75):
            solid &= sum(((c - x0 + 0.5) * h - cc) ** 2 for c, cc in zip(coords, (xc, yc, zc))) > hole_radius ** 2

lam_q, mu_q = disc.get_quad_field_scalar(name='lam_q'), disc.get_quad_field_scalar(name='mu_q')
lam_q.s[...], mu_q.s[...] = 0, 0
lam_q.s[..., solid], mu_q.s[..., solid] = lam, mu
material = material_models.LinearElastic(discretization=disc, lam_1qxyz=lam_q, mu_1qxyz=mu_q, name='elasticity')

plane_mask = np.zeros(coords[0].shape, dtype=bool)
for c in coords:
    plane_mask |= (c == x0) | (c == x1)
constrained = disc.get_displacement_sized_field(name='constrained')
constrained.s[...] = 0
constrained.s[:, 0, plane_mask] = 1
prescribed = disc.get_displacement_sized_field(name='prescribed')
prescribed.s[...] = 0
nodal_constraints = constraints.NodalConstraints.from_mask(disc, constrained_mask=constrained,
                                                           prescribed_values=prescribed)

# uniform body force on the solid: each solid voxel gives h^3 / 8 of its force to each of its 8 nodes
force = disc.get_unknown_size_field(name='force')
force.s[...] = 0
share = 0.125 * h ** 3 * body_force * solid
for sx in (0, 1):
    for sy in (0, 1):
        for sz in (0, 1):
            force.s[1, 0] += np.roll(share, (sx, sy, sz), axis=(0, 1, 2))


def K_fun(x, Ax):
    disc.apply_system_matrix_mugrid_explicit_stress(constitutive=material.get_stress, input_field_inxyz=x,
                                                    output_field_inxyz=Ax, formulation='small_strain')
    disc.fft.communicate_ghosts(Ax)


green = disc.get_preconditioner_Green_mugrid(
    reference_material_data_ijkl=material_models.get_elastic_tensor_from_lame(dim=3, lam=lam, mu=mu))
green_apps = [0]


def M_fun(x, Px):
    disc.fft.communicate_ghosts(x)
    disc.apply_preconditioner_mugrid(preconditioner_Fourier_fnfnqks=green, input_nodal_field_fnxyz=x,
                                     output_nodal_field_fnxyz=Px)
    green_apps[0] += 1


t_setup = time.time()
fixed = tuple(np.argwhere(constrained.s != 0).T)  # (component, sub-point, x, y, z)
m = len(fixed[0])
dof_index = -np.ones(constrained.s.shape, dtype=int)
dof_index[fixed] = np.arange(m)
P_fun = nodal_constraints.project_preconditioner(M_fun)
inner_its = []
if inner != 'green':
    Ey, MEy = disc.get_unknown_size_field(name='Ey'), disc.get_unknown_size_field(name='MEy')

    def S_fun(y):  # S y = E^T M E y, one Green application
        Ey.s[...] = 0
        Ey.s[fixed] = y
        M_fun(Ey, MEy)
        return MEy.s[fixed].copy()

    # families of parallel planes: normal axis a, planes at x0 and x1; idx[l, d, t1, t2] = dof of plane l, component d
    # at the in-plane node (t1, t2); symbol (nb, nb, 6, 6) from a unit force per (plane, component) at the node (0, 0)
    families = []
    pos = (x0, x1)
    if inner in ('family', 'familydirect'):
        for a in range(3):
            def plane(arr, p, a=a):  # the slice of a (.., nx, ny, nz) array on the plane normal to a at p
                return np.take(arr, p, axis=arr.ndim - 3 + a)

            idx = np.stack([np.stack([plane(dof_index[d, 0], p) for d in range(3)]) for p in pos])  # (2, 3, nb, nb)
            kernel = np.zeros((6, 6, nb_px, nb_px))
            for lp, p_src in enumerate(pos):
                for e in range(3):
                    y = np.zeros(m)
                    y[idx[lp, e, 0, 0]] = 1.0
                    Sy = S_fun(y)
                    kernel[:, 3 * lp + e] = Sy[idx].reshape(6, nb_px, nb_px)
            symbol = np.fft.fft2(kernel).transpose(2, 3, 0, 1)
            symbol = 0.5 * (symbol + np.conj(np.swapaxes(symbol, -1, -2)))
            ev, evec = np.linalg.eigh(symbol)
            symbol_inv = np.einsum('...ij,...j,...kj->...ik', evec, 1.0 / np.maximum(ev, 1e-12 * ev.max()),
                                   np.conj(evec))
            families.append((idx.reshape(6, nb_px, nb_px), symbol_inv))

    def C_inv(res):  # overlapping block-Jacobi over the families, each family exactly by its 2D FFT
        out = np.zeros_like(res)
        for idx, symbol_inv in families:
            sol = np.real(np.fft.ifft2(np.einsum('yzde,eyz->dyz', symbol_inv, np.fft.fft2(res[idx]))))
            out[idx.ravel()] += sol.ravel()
        return out

    # second level on the edge strips: per edge direction a, the 4 edges at (p, q) in {x0, x1}^2 of the other two axes;
    # the strip of edge (p, q): fixed nodes with (coord_b = p, |coord_c - q| <= k) or (coord_c = q, |coord_b - p| <= k)
    edge_groups = []
    if patch >= 0 and inner == 'family':
        def pdist(a_, b_):
            return min(abs(a_ - b_), nb_px - abs(a_ - b_))
        for a in range(3):
            b, c = [ax_ for ax_ in range(3) if ax_ != a]
            section = set()
            for p_ in pos:
                for q_ in pos:
                    for t in range(nb_px):
                        if pdist(t, q_) <= patch:
                            section.add((p_, t))
                        if pdist(t, p_) <= patch:
                            section.add((t, q_))
            section = sorted(section)
            idx = np.empty((3 * len(section), nb_px), dtype=int)  # (component, (b, c)) x coordinate along a
            for d in range(3):
                for i_, (vb, vc) in enumerate(section):
                    node = [0, 0, 0]
                    node[b], node[c] = vb, vc
                    for t in range(nb_px):
                        node[a] = t
                        idx[d * len(section) + i_, t] = dof_index[(d, 0) + tuple(node)]
            assert (idx >= 0).all()
            ns_ = idx.shape[0]
            kernel = np.zeros((ns_, ns_, nb_px))
            for j in range(ns_):  # unit force at the section dof j on the plane coord_a = 0
                y = np.zeros(m)
                y[idx[j, 0]] = 1.0
                kernel[:, j] = S_fun(y)[idx]
            symbol = np.fft.fft(kernel, axis=-1).transpose(2, 0, 1)
            symbol = 0.5 * (symbol + np.conj(np.swapaxes(symbol, -1, -2)))
            ev, evec = np.linalg.eigh(symbol)
            edge_groups.append((idx, np.einsum('kij,kj,klj->kil', evec, 1.0 / np.maximum(ev, 1e-12 * ev.max()),
                                               np.conj(evec))))
        print(f'edge strips +-{patch}: {sum(g[0].size for g in edge_groups)} patch dofs '
              f'({sum(g[0].shape[0] for g in edge_groups)} Green applications of setup)', flush=True)

    # dofs shared by several groups (near the corners) are weighted by 1/sqrt(multiplicity) on both sides
    multiplicity = np.zeros(m)
    for idx, _ in edge_groups:
        multiplicity[idx.ravel()] += 1
    weight = 1.0 / np.sqrt(np.maximum(multiplicity, 1))

    def Q_apply(r):  # weighted sum over the edge groups of their exact (circulant) patch inverse
        out = np.zeros_like(r)
        rw = r * weight
        for idx, symbol_inv in edge_groups:
            out[idx.ravel()] += np.real(np.fft.ifft(np.einsum('kde,ek->dk', symbol_inv,
                                                              np.fft.fft(rw[idx], axis=-1)), axis=-1)).ravel()
        return out * weight

    def two_level(r):  # balancing: Q r + (I - Q S) M^-1 (r - S Q r)
        q = Q_apply(r)
        w_ = C_inv(r - S_fun(q))
        return q + w_ - Q_apply(S_fun(w_))

    if inner == 'dense':
        S = np.empty((m, m))
        for j in range(m):
            e_j = np.zeros(m)
            e_j[j] = 1.0
            S[:, j] = S_fun(e_j)
        S = 0.5 * (S + S.T)
        S_inv = np.linalg.inv(S)
        del S

    def S_solve(g, tol):
        if inner == 'dense':
            return S_inv @ g
        if inner == 'familydirect':
            inner_its.append(0)
            return C_inv(g)
        prec = (lambda r: r) if inner == 'cg' else two_level if edge_groups else C_inv
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

    B = np.stack([(fixed[0] == d).astype(float) for d in range(3)], axis=1)  # E^T N
    W = np.stack([S_solve(B[:, d], 1e-12) for d in range(3)], axis=1)
    BW_inv = np.linalg.inv(B.T @ W)
    inner_its.clear()
    Mr, w = disc.get_unknown_size_field(name='Mr'), disc.get_unknown_size_field(name='w')

    def P_fun(r, z):  # [S B; B^T 0][lam; mu] = [E^T M r; N^T r], z = M (r - E lam) - N mu
        M_fun(r, Mr)
        Sg = S_solve(Mr.s[fixed], inner_tol)
        mu_ = BW_inv @ (B.T @ Sg - np.array([r.s[d].sum() for d in range(3)]))
        w.s[...] = r.s
        w.s[fixed] -= Sg - W @ mu_
        M_fun(w, z)
        for d in range(3):
            z.s[d] -= mu_[d]
        z.s[fixed] = 0.0
setup_time, setup_apps = time.time() - t_setup, green_apps[0]
if inner in ('family', 'familydirect'):
    g_test = np.random.default_rng(1).standard_normal(m)
    print(f'|S C^-1 g - g| / |g| = {np.linalg.norm(S_fun(C_inv(g_test)) - g_test) / np.linalg.norm(g_test):.2e}',
          flush=True)
green_apps[0] = 0
print(f'{name}: grid {nb_px}^3, {m} fixed dofs, setup {setup_time:.1f} s ({setup_apps} Green apps)', flush=True)

rhs, u = disc.get_unknown_size_field(name='rhs'), disc.get_unknown_size_field(name='u')
rhs.s[...] = force.s
nodal_constraints.apply_to_rhs(rhs, K_fun)
u.s[...] = 0
K_free = nodal_constraints.project_operator(K_fun)
t0 = time.time()
norms = solvers.conjugate_gradients_mugrid_experimental(
    comm=disc.communicator, fc=disc.field_collection, hessp=K_free, b=rhs, x=u, P=P_fun, maxiter=5000,
    stop_crit_norm='energy_lower_estim', d_min=cg_delay,
    **topology_optimization.energy_stop_kwargs(disc, K_free, u, cg_tol))[1]
cg_its = norms['stop_iteration']['energy_lower_estim']
cg_its = int(cg_its) if cg_its is not None else -1
wall_time = time.time() - t0
compliance = float(np.sum(force.s * u.s))
print(f'{name}: outer {cg_its}, inner/outer {np.mean(inner_its) if inner_its else 0:.1f}, Green apps {green_apps[0]} '
      f'(+ {setup_apps} setup), compliance {compliance:.10e}, {wall_time:.1f} s', flush=True)
np.savez(os.path.join(RUNS, name + '.npz'), n=n, inner=inner, inner_tol=inner_tol, m=m, cg_its=cg_its,
         inner_its=np.array(inner_its, dtype=int), green_apps=green_apps[0], setup_apps=setup_apps,
         compliance=compliance, time=wall_time, setup_time=setup_time,
         energy_error_sq=np.array(norms['energy_lower_estim']),  # estimate of ||u - u_k||_K^2 per iteration
         **({'u_cube': u.s[:, 0, x0:x1 + 1, x0:x1 + 1, x0:x1 + 1].copy(),  # displacement on the cube's nodes
             'solid_cube': solid[x0:x1, x0:x1, x0:x1].copy()} if n <= 32 else {}))
