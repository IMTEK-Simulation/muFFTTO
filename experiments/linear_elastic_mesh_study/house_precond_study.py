"""
Which preconditioner for the elastic solves of the cracking house? (GPU) Frozen-damage comparison on the damage fields
of the reference run (house_damage_3D.py, NetCDF): at a few converged load steps (intact, damage starting, peak wind,
just after the first crack, last step), the elastic problem of that step (C_d = g(d) C_0, self-weight + the wind of
that step, walls clamped at the foundation) is solved by CG from zero with each preconditioner:

    green       Q G Q                                      G: Green of C_ref = C_wall
    green_c     corrected Green  (Q K(C_ref) Q)^-1          S = E^T G E assembled
    gj          Q D^-1/2 G_I D^-1/2 Q                      Green-Jacobi: G_I Green of the dimensionless 2 mu / E I_sym,
                                                           D = diag K (zero on void nodes -> D^-1/2 := 0)
    gj_c        D^-1/2 (Q K(I) Q)^-1 D^-1/2                Green-Jacobi with the corrected Green of 2 mu / E I_sym
    dual        Q G K(C_ref C_d^+ C_ref) G Q               dual; C_d^+ pseudo-inverse (0 in the void)
    dual_c      M_c K(C_ref C_d^+ C_ref) M_c               dual with the corrected Green on both sides
    dual_eps_c  as dual_c, the void taken as DUAL_VOID_EPS C_ref inside the dual only (default 1e-3)

All see exactly the same matrix (zero-stiffness void, as in the damage run).

usage:  H=0.125 python house_precond_study.py              writes exp_data/house/precond_h<h>.npz
        PLOT=1 H=0.125 python house_precond_study.py       writes exp_data/house_precond_h<h>.png
"""
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import time

import numpy as np

import house_wind_3D as house
import house_damage_3D as damage_run

h = house.h
HERE = os.path.dirname(os.path.realpath(__file__))
NC = os.path.join(house.OUT, f'{damage_run.case}.nc')
OUT_FILE = os.path.join(house.OUT, f'precond_h{h:g}{os.environ.get("OUT_TAG", "")}.npz')
PRECONDS = tuple(os.environ.get('PRECONDS', 'green,green_c,gj,gj_c,dual,dual_c,dual_eps_c').split(','))
DUAL_VOID_EPS = float(os.environ.get('DUAL_VOID_EPS', 1e-3))
LABELS = {'green': 'Green', 'green_c': 'Green, corrected', 'gj': 'Green-Jacobi', 'gj_c': 'Green-Jacobi, corrected',
          'dual': 'dual', 'dual_c': 'dual, corrected', 'dual_eps_c': 'dual, corrected, void as soft'}
maxiter = int(os.environ.get('MAXITER', 10000))


def pick_frames():
    """converged frames: intact, damage starting (max d ~ 0.2), peak wind, just after the first big drop, last"""
    import netCDF4
    with netCDF4.Dataset(NC) as ds:
        conv = np.flatnonzero(np.asarray(ds['converged'][:]) > 0)
        load = np.asarray(ds['load'][:])[conv]
        dmax = np.asarray(ds['d_max'][:])[conv]
    peak = int(np.argmax(load))
    after = peak + 1 + int(np.argmax(load[peak + 1:] < 0.9 * load[peak])) if peak + 1 < len(conv) else peak
    starting = int(np.argmin(np.abs(dmax[:peak + 1] - 0.2)))
    picks = sorted({0, starting, peak, after, len(conv) - 1})
    return [int(conv[k]) for k in picks]


def run():
    import cupy as xp
    import netCDF4
    from muFFTTO import constraints, domain, material_models, solvers, topology_optimization

    phase, base = house.geometry()
    shape = phase.shape
    disc = domain.Discretization(cell=domain.PeriodicUnitCell(domain_size=[n * h for n in shape],
                                                              problem_type='elasticity'),
                                 nb_of_pixels_global=shape, discretization_type='finite_element',
                                 element_type='trilinear_hexahedron', device='gpu')
    disc_d = domain.Discretization(cell=domain.PeriodicUnitCell(domain_size=[n * h for n in shape],
                                                                problem_type='conductivity'),
                                   nb_of_pixels_global=shape, discretization_type='finite_element',
                                   element_type='trilinear_hexahedron', device='gpu')
    lam, mu = material_models.get_lame_parameters(E=house.E_WALL, poisson=house.POISSON)
    I3 = np.eye(3)
    I_sym = 0.5 * (np.einsum('ik,jl->ijkl', I3, I3) + np.einsum('il,jk->ijkl', I3, I3))
    C_iso = lam * np.einsum('ij,kl->ijkl', I3, I3) + 2 * mu * I_sym
    stiffness = xp.asarray(np.where(phase == 1, 1.0, np.where(phase == 2, house.E_PANEL / house.E_WALL, 0.0)))
    scale, dual_scale, dual_eps_scale = (disc.get_quad_field_scalar(name=n) for n in
                                         ('scale', 'dual_scale', 'dual_eps_scale'))
    C_d = domain.ScaledMaterial([(scale, C_iso)])
    C_dual = domain.ScaledMaterial([(dual_scale, C_iso)])  # C_iso (s C_iso)^+ C_iso = C_iso / s, 0 where s = 0
    C_dual_eps = domain.ScaledMaterial([(dual_eps_scale, C_iso)])  # void as eps C_iso: 1 / eps there

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
    free = xp.asarray(1.0 - cmask)
    wind_np, _ = house.wind_forces(phase)
    weight_np = np.zeros_like(wind_np)
    idx = np.argwhere(phase > 0)
    w = -9.81 * h ** 3 / 8 * np.array([damage_run.RHO[p] for p in phase[tuple(idx.T)]])
    for corner in np.ndindex(2, 2, 2):
        np.add.at(weight_np[1], tuple(((idx + corner) % np.array(shape)).T), w)
    wind, weight = xp.asarray(wind_np), xp.asarray(weight_np)

    def K(x, Ax):
        disc.apply_system_matrix_mugrid(material_data_field=C_d, input_field_inxyz=x, output_field_inxyz=Ax,
                                        formulation='small_strain')
        disc.fft.communicate_ghosts(Ax)

    K_free = nc.project_operator(K)
    greens = {'C': disc.get_preconditioner_Green_mugrid(reference_material_data_ijkl=C_iso, name='green_C'),
              'I': disc.get_preconditioner_Green_mugrid(reference_material_data_ijkl=2 * mu / house.E_WALL * I_sym,
                                                        name='green_I')}

    def G(which):
        def apply(x, Px):
            disc.fft.communicate_ghosts(x)
            disc.apply_preconditioner_mugrid(preconditioner_Fourier_fnfnqks=greens[which], input_nodal_field_fnxyz=x,
                                             output_nodal_field_fnxyz=Px)
        return apply

    corrected_cache = {}

    def corrected(which):
        if which in corrected_cache:
            return corrected_cache[which]
        t0 = time.time()
        m = len(fixed_np[0])
        Ey, MEy, Mr, wk = (disc.get_unknown_size_field(name=f'c_{which}_{k}') for k in ('Ey', 'MEy', 'Mr', 'w'))
        S = xp.empty((m, m))
        for j in range(m):
            Ey.s[...] = 0
            Ey.s[tuple(int(f[j]) for f in fixed_np)] = 1.0
            G(which)(Ey, MEy)
            S[:, j] = MEy.s[fixed]
        S_inv = xp.linalg.inv(0.5 * (S + S.T))
        B = xp.asarray(np.stack([(fixed_np[0] == d).astype(float) for d in range(3)], axis=1))
        W = S_inv @ B
        BW_inv = xp.linalg.inv(B.T @ W)

        def apply(r, z):
            G(which)(r, Mr)
            Sg = S_inv @ Mr.s[fixed]
            mu_ = BW_inv @ (B.T @ Sg - xp.stack([r.s[d].sum() for d in range(3)]))
            wk.s[...] = r.s
            wk.s[fixed] -= Sg - W @ mu_
            G(which)(wk, z)
            for d in range(3):
                z.s[d] -= mu_[d]
            z.s[fixed] = 0.0
        corrected_cache[which] = (apply, time.time() - t0)
        return corrected_cache[which]

    t1, t2, xs = (disc.get_unknown_size_field(name=n) for n in ('t1', 't2', 'xs'))
    jac = [None]

    def make(p):
        setup = 0.0
        if p in ('green_c', 'dual_c', 'dual_eps_c'):
            Mc, setup = corrected('C')
        elif p == 'gj_c':
            Mc, setup = corrected('I')
        else:
            Mc = nc.project_preconditioner(G('I' if p == 'gj' else 'C'))
        if p in ('green', 'green_c'):
            return Mc, setup
        if p in ('gj', 'gj_c'):
            def M(x, Px):
                xs.s[...] = jac[0] * x.s
                Mc(xs, Px)
                Px.s[...] *= jac[0]
            return M, setup

        C_ = C_dual_eps if p == 'dual_eps_c' else C_dual

        def M(x, Px):
            Mc(x, t1)
            disc.apply_system_matrix_mugrid(material_data_field=C_, input_field_inxyz=t1, output_field_inxyz=t2,
                                            formulation='small_strain')
            disc.fft.communicate_ghosts(t2)
            t2.s[...] *= free
            Mc(t2, Px)
        return M, setup

    rhs, u = (disc.get_unknown_size_field(name=n) for n in ('rhs', 'u'))
    d_nodal = disc_d.get_unknown_size_field(name='d_nodal')
    d_quad = disc_d.get_quad_field_scalar(name='d_quad')
    frames = pick_frames()
    rows = {p: [] for p in PRECONDS}
    setups = {}
    with netCDF4.Dataset(NC) as ds:
        info = [(f, float(ds['load'][f]), float(ds['d_max'][f]), float(ds['cracked_volume'][f]),
                 np.asarray(ds['damage'][f]).reshape(shape)) for f in frames]
    for f, load, dmax, cracked, d in info:
        d_nodal.s[0, 0] = xp.asarray(d)
        disc_d.apply_N_operator_mugrid(d_nodal, d_quad)
        scale.s[0, 0] = ((1 - d_quad.s[0, 0]) ** 2 + damage_run.k_residual) * stiffness
        dual_scale.s[...] = xp.where(scale.s > 0, 1.0 / xp.where(scale.s > 0, scale.s, 1.0), 0.0)
        dual_eps_scale.s[...] = 1.0 / xp.maximum(scale.s, DUAL_VOID_EPS)
        diag = disc.get_preconditioner_Jacobi_mugrid(material_data_field_ijklqxyz=C_d, formulation='small_strain',
                                                     name='jacobi')
        jac[0] = xp.nan_to_num(diag.s, nan=0.0, posinf=0.0) * free  # D^-1/2, 0 on void and fixed nodes
        rhs.s[...] = 0
        rhs.s[:, 0] = weight + load * wind
        nc.apply_to_rhs(rhs, K)
        for p in PRECONDS:
            M, setup = make(p)
            setups[p] = setup
            u.s[...] = 0
            t0 = time.time()
            norms = solvers.conjugate_gradients_mugrid_experimental(
                comm=disc.communicator, fc=disc.field_collection, hessp=K_free, b=rhs, x=u, P=M, maxiter=maxiter,
                stop_crit_norm='energy_lower_estim', d_min=10,
                **topology_optimization.energy_stop_kwargs(disc, K_free, u, 1e-6))[1]
            wall = time.time() - t0
            its = norms['stop_iteration']['energy_lower_estim']
            its = maxiter if its is None else int(its)
            compliance = float(xp.sum(rhs.s * u.s))
            rows[p].append((f, load, dmax, cracked, its, wall, wall / its, compliance))
            print(f'frame {f:4d}  wind {load:6.2f} kPa  max d {dmax:.2f}  cracked {cracked:.2f} m3  {p:8s} CG its '
                  f'{its:6d}  {wall:7.1f} s ({1e3 * wall / its:.0f} ms/it, setup {setup:.0f} s)  compliance '
                  f'{compliance:.6e}', flush=True)
            np.savez(OUT_FILE, **{p: np.array(r) for p, r in rows.items() if r},
                     **{f'setup_{p}': s for p, s in setups.items()})


def plot():
    import matplotlib

    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    r = np.load(OUT_FILE)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.2), layout='constrained')
    first = next(p for p in PRECONDS if p in r.files)
    names = [f'{"intact" if k == 0 else ""}\nq {row[1]:.1f} kPa\nmax d {row[2]:.2f}\ncracked {row[3]:.2f} m$^3$'
             for k, row in enumerate(r[first])]
    x = np.arange(len(names))
    width = 0.8 / len(PRECONDS)
    for k, p in enumerate(PRECONDS):
        if p not in r.files:
            continue
        rows = r[p]
        c = f'C{k // 2}'
        hatch = '' if p.endswith('_c') else '//'
        axes[0].bar(x[:len(rows)] + (k - (len(PRECONDS) - 1) / 2) * width, rows[:, 4], width, color=c, hatch=hatch, edgecolor='w',
                    label=LABELS[p])
        axes[1].bar(x[:len(rows)] + (k - (len(PRECONDS) - 1) / 2) * width, rows[:, 5] + float(r[f'setup_{p}']) / len(rows), width,
                    color=c, hatch=hatch, edgecolor='w', label=LABELS[p])
        axes[2].bar(x[:len(rows)] + (k - (len(PRECONDS) - 1) / 2) * width, 1e3 * rows[:, 6], width, color=c, hatch=hatch, edgecolor='w',
                    label=LABELS[p])
    for ax, t, yl in ((axes[0], 'CG iterations to $10^{-6}$ (energy)', 'iterations'),
                      (axes[1], 'time per solve (setup of S shared out)', 's'),
                      (axes[2], 'time per CG iteration', 'ms')):
        ax.set_title(t)
        ax.set_ylabel(yl)
        ax.set_xticks(x, names, fontsize=8)
        ax.grid(True, axis='y', color='0.9')
    axes[0].set_yscale('log')
    axes[1].set_yscale('log')
    axes[0].legend(fontsize=8, frameon=False, ncol=2)
    fig.suptitle(f'cracking house, voxel {h * 100:g} cm: the elastic solve of five load steps, frozen damage, '
                 f'one preconditioner each (solid: corrected for the clamped foundation, hatched: projected)',
                 fontsize=12)
    out = os.path.join(HERE, 'exp_data', f'house_precond_h{h:g}.png')
    fig.savefig(out, dpi=140)
    print(f'wrote {out}')


if __name__ == '__main__':
    plot() if os.environ.get('PLOT') else run()
