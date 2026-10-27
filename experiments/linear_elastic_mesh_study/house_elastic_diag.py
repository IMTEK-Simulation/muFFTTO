"""
Diagnostic: elastic house solve on the GPU (self-weight + wind 1 kPa), CG iterations and time for one preconditioner.
Env: H, PRECOND=green|green_c|gj|gj_c, HOUSE_ROOF, HOUSE_PANELS, VOID_EPS (void stiffness / wall), MAXITER
"""
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
import time

import numpy as np
import cupy as xp

import house_wind_3D as house
from muFFTTO import constraints, domain, material_models, solvers, topology_optimization

h = house.h
precond = os.environ.get('PRECOND', 'green_c')
void_eps = float(os.environ.get('VOID_EPS', 0))
phase, base = house.geometry()
shape = phase.shape
disc = domain.Discretization(cell=domain.PeriodicUnitCell(domain_size=[n * h for n in shape], problem_type='elasticity'),
                             nb_of_pixels_global=shape, discretization_type='finite_element',
                             element_type='trilinear_hexahedron', device='gpu')
lam, mu = material_models.get_lame_parameters(E=house.E_WALL, poisson=house.POISSON)
I3 = np.eye(3)
I_sym = 0.5 * (np.einsum('ik,jl->ijkl', I3, I3) + np.einsum('il,jk->ijkl', I3, I3))
C_iso = lam * np.einsum('ij,kl->ijkl', I3, I3) + 2 * mu * I_sym
stiffness = np.where(phase == 1, 1.0, np.where(phase == 2, house.E_PANEL / house.E_WALL, void_eps))
scale = disc.get_quad_field_scalar(name='scale')
scale.s[0, 0] = xp.asarray(stiffness)
C_d = domain.ScaledMaterial([(scale, C_iso)])
foot = phase[:, base, :] > 0
clamped = np.zeros((shape[0], shape[2]), dtype=bool)
for da in (0, 1):
    for db in (0, 1):
        clamped |= np.roll(np.roll(foot, da, 0), db, 1)
cmask = np.zeros((3, 1) + shape)
cmask[:, 0, :, base, :] = clamped
constrained = disc.get_displacement_sized_field(name='constrained_dofs_field')
constrained.s[...] = xp.asarray(cmask)
prescribed = disc.get_displacement_sized_field(name='prescribed_displacement')
prescribed.s[...] = 0.0
nc = constraints.NodalConstraints.from_mask(disc, constrained_mask=constrained, prescribed_values=prescribed)
fixed_np = tuple(np.argwhere(cmask != 0).T)
fixed = tuple(xp.asarray(f) for f in fixed_np)
wind, _ = house.wind_forces(phase)
weight = np.zeros_like(wind)
idx = np.argwhere(phase > 0)
w = -9.81 * h ** 3 / 8 * np.array([{1: 1800.0, 2: 500.0}[p] for p in phase[tuple(idx.T)]])
for corner in np.ndindex(2, 2, 2):
    np.add.at(weight[1], tuple(((idx + corner) % np.array(shape)).T), w)


def K(x, Ax):
    disc.apply_system_matrix_mugrid(material_data_field=C_d, input_field_inxyz=x, output_field_inxyz=Ax,
                                    formulation='small_strain')
    disc.fft.communicate_ghosts(Ax)


K_free = nc.project_operator(K)
greens = {'C': C_iso, '2mu': 2 * mu / house.E_WALL * I_sym}  # Green-Jacobi: dimensionless reference, D carries the scale
which = '2mu' if precond.startswith('gj') else 'C'
green = disc.get_preconditioner_Green_mugrid(reference_material_data_ijkl=greens[which])


def G(x, Px):
    disc.fft.communicate_ghosts(x)
    disc.apply_preconditioner_mugrid(preconditioner_Fourier_fnfnqks=green, input_nodal_field_fnxyz=x,
                                     output_nodal_field_fnxyz=Px)


t0 = time.time()
if precond.endswith('_c'):
    m = len(fixed[0])
    Ey, MEy, Mr, wk = (disc.get_unknown_size_field(name=f'c_{k}') for k in ('Ey', 'MEy', 'Mr', 'w'))
    S = xp.empty((m, m))
    for j in range(m):
        Ey.s[...] = 0
        Ey.s[tuple(int(f[j]) for f in fixed_np)] = 1.0
        G(Ey, MEy)
        S[:, j] = MEy.s[fixed]
    S_inv = xp.linalg.inv(0.5 * (S + S.T))
    B = xp.asarray(np.stack([(fixed_np[0] == d).astype(float) for d in range(3)], axis=1))
    W = S_inv @ B
    BW_inv = xp.linalg.inv(B.T @ W)

    def M0(r, z):
        G(r, Mr)
        Sg = S_inv @ Mr.s[fixed]
        mu_ = BW_inv @ (B.T @ Sg - xp.stack([r.s[d].sum() for d in range(3)]))
        wk.s[...] = r.s
        wk.s[fixed] -= Sg - W @ mu_
        G(wk, z)
        for d in range(3):
            z.s[d] -= mu_[d]
        z.s[fixed] = 0.0
else:
    M0 = nc.project_preconditioner(G)
setup = time.time() - t0
if precond.startswith('gj'):
    jac = disc.get_preconditioner_Jacobi_mugrid(material_data_field_ijklqxyz=C_d, formulation='small_strain',
                                                name='jacobi')
    xs = disc.get_unknown_size_field(name='xs')
    free = 1.0 - constrained.s
    jac.s[...] = xp.nan_to_num(jac.s, posinf=0.0)

    def M(x, Px):
        xs.s[...] = jac.s * x.s * free
        M0(xs, Px)
        Px.s[...] *= jac.s * free
else:
    M = M0
rhs, u = (disc.get_unknown_size_field(name=n) for n in ('rhs', 'u'))
if os.environ.get('CHECK'):
    tx, ty, ta, tb = (disc.get_unknown_size_field(name=f'chk_{k}') for k in 'xyab')
    rng = np.random.default_rng(1)
    free_np = xp.asarray(1.0 - cmask)
    tx.s[...] = xp.asarray(rng.standard_normal(tx.s.shape)) * free_np
    ty.s[...] = xp.asarray(rng.standard_normal(tx.s.shape)) * free_np
    for name, op in (('G', G), ('M', M)):
        x0 = tx.s.copy()
        op(tx, ta)
        changed = float(xp.abs(tx.s - x0).max())
        op(ty, tb)
        print(f'{name}: input changed by {changed:.1e}; y.Ax {float(xp.sum(ty.s * ta.s)):.6e} x.Ay {float(xp.sum(tx.s * tb.s)):.6e}')
        tx.s[...] = x0
    s1, s2 = float(xp.sum(ty.s * ta.s)), float(xp.sum(tx.s * tb.s))
    print(f'M symmetry: y.Mx {s1:.6e} x.My {s2:.6e} rel {abs(s1 - s2) / abs(s1):.1e}, x.Mx {float(xp.sum(tx.s * ta.s)):.3e}')
    # power iteration for lambda_max of M K on the free dofs
    for it in range(60):
        K_free(tx, ta)
        M(ta, tb)
        lam_ = float(xp.sum(tb.s * ta.s)) / float(xp.sum(tx.s * ta.s))
        tx.s[...] = tb.s / float(xp.sqrt(xp.sum(tb.s * tb.s)))
    print(f'lambda_max (power iteration, Rayleigh in K) ~ {lam_:.4f}')
    sys.exit(0)
rhs.s[...] = 0
rhs.s[:, 0] = xp.asarray(weight + wind)
nc.apply_to_rhs(rhs, K)
u.s[...] = 0
rz, cg_alpha, prev = [], [], [None]


def record(iteration, x_s, r_s, p_s, z_s, rr):
    rz.append(float(xp.sum(r_s * z_s)))
    if prev[0] is not None:
        cg_alpha.append(float(xp.sum((x_s - prev[0]) * p_s) / xp.sum(p_s * p_s)))
    prev[0] = x_s.copy()


t0 = time.time()
norms = solvers.conjugate_gradients_mugrid_experimental(
    comm=disc.communicator, fc=disc.field_collection, hessp=K_free, b=rhs, x=u, P=M,
    maxiter=int(os.environ.get('MAXITER', 20000)), stop_crit_norm='energy_lower_estim', d_min=10,
    callback=record, **topology_optimization.energy_stop_kwargs(disc, K_free, u, 1e-6))[1]
wall = time.time() - t0
its = norms.get('stop_iteration', {}).get('energy_lower_estim')
its = -1 if its is None else int(its)
e = np.asarray(norms['energy_lower_estim'])
print(f'h {h:g} grid {shape} roof {os.environ.get("HOUSE_ROOF", "1")} panels {os.environ.get("HOUSE_PANELS", "soft")} '
      f'void_eps {void_eps:g} {precond:8s}: CG its {its} (ran {len(e)}), {wall:.1f} s, setup {setup:.1f} s, '
      f'energy estimate at 10, 100, 1000, end: {[float(e[k]) for k in (10, 100, 1000, -1) if k < len(e)]}, '
      f'max |u| {float(xp.abs(u.s).max()) * 1e3:.4f} mm, compliance {float(xp.sum(rhs.s * u.s)):.6e}', flush=True)
a = np.array(cg_alpha)
b = (np.array(rz[1:]) / np.array(rz[:-1]))[:len(a)]
k = min(len(a), int(os.environ.get('RITZ_K', 400)))
T = np.diag(1 / a[:k])
T[1:, 1:] += np.diag(b[:k - 1] / a[:k - 1])
off = np.sqrt(np.abs(b[:k - 1])) / a[:k - 1]
T += np.diag(off, 1) + np.diag(off, -1)
ritz = np.linalg.eigvalsh(T)
print(f'Ritz values from {k} CG steps: smallest {ritz[:6]}, largest {ritz[-4:]}', flush=True)
