"""
Is the dual preconditioner mesh independent for a cracked structure, with the void and the Dirichlet conditions
handled? 2D cantilever with a frozen phase-field crack, mesh study, four preconditioners.

Beam L x H = 1 x 0.2 (bilinear pixels, E = 1, nu = 0.3) in a void of fixed physical thickness VOID_T (default 0.05),
left face clamped, uniform shear load at the tip. A frozen crack near the clamp: vertical segment at x = CRACK_X
(default 0.1) from the top face down to depth CRACK_A (default H/2), phase-field profile d = exp(-dist / l), material
C = ((1 - d)^2 + k) C_0, k = K_RES (default 1e-4); l = L_FACTOR h (default 2, l shrinks with h as in practice) or
l = L_FIX (fixed physical length). CORE > 0: d = 1 within CORE * l of the crack (a fully broken band, stiffness k).

    green        Q G Q                                        G: Green of C_0
    green_c      corrected Green (Q K(C_0) Q)^-1              S = E^T G E assembled, 2 translations
    dual         Q G K(C_0 C_eps^-1 C_0) G Q                  C_eps = max(s, EPS) C_0 inside the dual only
    dual_c       M_c K(C_0 C_eps^-1 C_0) M_c                  M_c corrected Green

usage:  N_H=16,32,64,128,256 python cantilever_crack_precond.py         (runs -> exp_data/runs_crack/)
        PLOT=1 python cantilever_crack_precond.py                        (exp_data/cantilever_crack_precond.png)
"""
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import time

import numpy as np

HERE = os.path.dirname(os.path.realpath(__file__))
RUNS = os.path.join(HERE, 'exp_data', 'runs_crack')
os.makedirs(RUNS, exist_ok=True)
L, H = 1.0, 0.2
void_t = float(os.environ.get('VOID_T', 0.05))
crack_x = float(os.environ.get('CRACK_X', 0.1))
crack_a = float(os.environ.get('CRACK_A', 0.1))
k_res = float(os.environ.get('K_RES', 1e-4))
l_factor = float(os.environ.get('L_FACTOR', 2))
l_fix = float(os.environ.get('L_FIX', 0))
eps = float(os.environ.get('EPS', 1e-3))
core = float(os.environ.get('CORE', 0))  # d = 1 within CORE * l of the crack (a fully broken band), exp outside
PRECONDS = os.environ.get('PRECONDS', 'green,green_c,dual,dual_c').split(',')
FIELDS = os.environ.get('FIELDS') == '1'  # also store u and d (-> exp_data/runs_crack_fields/)
FIELD_RUNS = os.path.join(HERE, 'exp_data', 'runs_crack_fields')
os.makedirs(FIELD_RUNS, exist_ok=True)
force_y = -1e-4


def tag():
    return (f'l{l_fix:g}' if l_fix > 0 else f'l{l_factor:g}h') + f'_k{k_res:g}_eps{eps:g}' + \
        (f'_core{core:g}' if core > 0 else '') + (f'_a{crack_a:g}' if crack_a != 0.1 else '')


def solve(n_H, precond):
    from muFFTTO import constraints, domain, material_models, solvers, topology_optimization

    h = H / n_H
    nv = max(1, int(round(void_t / h)))
    n_L = int(round(L / h))
    shape = (n_L + 2 * nv, n_H + 2 * nv)
    disc = domain.Discretization(cell=domain.PeriodicUnitCell(domain_size=[n * h for n in shape],
                                                              problem_type='elasticity'),
                                 nb_of_pixels_global=shape, discretization_type='finite_element',
                                 element_type='bilinear_rectangle')
    ell = l_fix if l_fix > 0 else l_factor * h
    lam, mu = material_models.get_lame_parameters(E=1.0, poisson=0.3)
    I2 = np.eye(2)
    C0 = lam * np.einsum('ij,kl->ijkl', I2, I2) + mu * (np.einsum('ik,jl->ijkl', I2, I2) +
                                                       np.einsum('il,jk->ijkl', I2, I2))
    # quadrature point coordinates relative to the beam's lower left corner: pixel corner + Gauss offsets
    q = disc.get_quad_field_scalar(name='probe')
    nq = q.s.shape[2]
    gauss = 0.5 + np.array([-1, 1]) / (2 * np.sqrt(3))
    offsets = [(gauss[a], gauss[b]) for a in range(2) for b in range(2)] if nq == 4 else [(0.5, 0.5)] * nq
    ix, iy = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing='ij')
    beam = (ix >= nv) & (ix < nv + n_L) & (iy >= nv) & (iy < nv + n_H)
    scale = disc.get_quad_field_scalar(name='scale')
    dual_scale = disc.get_quad_field_scalar(name='dual_scale')
    d_max = 0.0
    for k, (ox, oy) in enumerate(offsets):
        x = (ix - nv + ox) * h
        y = (iy - nv + oy) * h
        dx = x - crack_x
        dy = np.maximum(0.0, (H - crack_a) - y)  # distance to the segment x = crack_x, y in [H - a, H]
        d = np.exp(-np.maximum(np.sqrt(dx ** 2 + dy ** 2) - core * ell, 0.0) / ell)
        d_max = max(d_max, float(d[beam].max()))
        s = np.where(beam, (1 - d) ** 2 + k_res, 0.0)
        scale.s[0, 0, k] = s
        dual_scale.s[0, 0, k] = 1.0 / np.maximum(s, eps)
    C_d = domain.ScaledMaterial([(scale, C0)])
    C_dual = domain.ScaledMaterial([(dual_scale, C0)])

    constrained = disc.get_displacement_sized_field(name='constrained')
    constrained.s[...] = 0
    left = (ix == nv) & (iy >= nv) & (iy <= nv + n_H)
    constrained.s[:, 0, left] = 1
    prescribed = disc.get_displacement_sized_field(name='prescribed')
    prescribed.s[...] = 0
    nc = constraints.NodalConstraints.from_mask(disc, constrained_mask=constrained, prescribed_values=prescribed)
    fixed = tuple(np.argwhere(constrained.s != 0).T)
    free = 1.0 - constrained.s
    force = disc.get_unknown_size_field(name='force')
    force.s[...] = 0
    right = (ix == nv + n_L) & (iy >= nv) & (iy <= nv + n_H)
    w = np.where((iy == nv) | (iy == nv + n_H), 0.5, 1.0) / n_H
    force.s[1, 0, right] = force_y * w[right]

    def K(x, Ax):
        disc.apply_system_matrix_mugrid(material_data_field=C_d, input_field_inxyz=x, output_field_inxyz=Ax,
                                        formulation='small_strain')
        disc.fft.communicate_ghosts(Ax)

    K_free = nc.project_operator(K)
    green = disc.get_preconditioner_Green_mugrid(reference_material_data_ijkl=C0, name='green_C0')

    def G(x, Px):
        disc.fft.communicate_ghosts(x)
        disc.apply_preconditioner_mugrid(preconditioner_Fourier_fnfnqks=green, input_nodal_field_fnxyz=x,
                                         output_nodal_field_fnxyz=Px)

    t0 = time.time()
    if precond.endswith('_c'):
        m = len(fixed[0])
        Ey, MEy, Mr, wk = (disc.get_unknown_size_field(name=f'c_{k}') for k in ('Ey', 'MEy', 'Mr', 'w'))
        S = np.empty((m, m))
        for j in range(m):
            Ey.s[...] = 0
            Ey.s[tuple(f[j] for f in fixed)] = 1.0
            G(Ey, MEy)
            S[:, j] = MEy.s[fixed]
        S_inv = np.linalg.inv(0.5 * (S + S.T))
        B = np.stack([(fixed[0] == d).astype(float) for d in range(2)], axis=1)
        W = S_inv @ B
        BW_inv = np.linalg.inv(B.T @ W)

        def Mc(r, z):
            G(r, Mr)
            Sg = S_inv @ Mr.s[fixed]
            mu_ = BW_inv @ (B.T @ Sg - np.array([r.s[d].sum() for d in range(2)]))
            wk.s[...] = r.s
            wk.s[fixed] -= Sg - W @ mu_
            G(wk, z)
            for d in range(2):
                z.s[d] -= mu_[d]
            z.s[fixed] = 0.0
    else:
        Mc = nc.project_preconditioner(G)
    setup = time.time() - t0
    if precond.startswith('dual'):
        t1, t2 = (disc.get_unknown_size_field(name=n) for n in ('t1', 't2'))

        def M(x, Px):
            Mc(x, t1)
            disc.apply_system_matrix_mugrid(material_data_field=C_dual, input_field_inxyz=t1, output_field_inxyz=t2,
                                            formulation='small_strain')
            disc.fft.communicate_ghosts(t2)
            t2.s[...] *= free
            Mc(t2, Px)
    else:
        M = Mc

    rhs, u = (disc.get_unknown_size_field(name=n) for n in ('rhs', 'u'))
    rhs.s[...] = force.s
    nc.apply_to_rhs(rhs, K)
    u.s[...] = 0
    rz, alphas, prev, energy = [], [], [None], []

    def record(iteration, x_s, r_s, p_s, z_s, stop):  # CG coefficients -> Ritz values; exact error from Pi
        rz.append(float(np.sum(r_s * z_s)))
        energy.append(-0.5 * float(np.sum(rhs.s * x_s) + np.sum(r_s * x_s)))
        if prev[0] is not None:
            alphas.append(float(np.sum((x_s - prev[0]) * p_s) / np.sum(p_s * p_s)))
        prev[0] = x_s.copy()

    t0 = time.time()
    norms = solvers.conjugate_gradients_mugrid_experimental(
        comm=disc.communicator, fc=disc.field_collection, hessp=K_free, b=rhs, x=u, P=M, maxiter=20000,
        stop_crit_norm='energy_lower_estim', d_min=10, callback=record,
        **topology_optimization.energy_stop_kwargs(disc, K_free, u, 1e-6))[1]
    wall = time.time() - t0
    its = norms['stop_iteration']['energy_lower_estim']
    a = np.array(alphas)
    b = (np.array(rz[1:]) / np.array(rz[:-1]))[:len(a)]
    T = np.diag(1 / a)
    T[1:, 1:] += np.diag(b[:-1] / a[:-1])
    off = np.sqrt(b[:-1]) / a[:-1]
    ritz = np.linalg.eigvalsh(T + np.diag(off, 1) + np.diag(off, -1))
    r = dict(n_H=n_H, h=h, ell=ell, d_max=d_max, cg_its=-1 if its is None else int(its), time=wall, setup=setup,
             compliance=float(np.sum(force.s * u.s)), lam_min=float(ritz.min()), lam_max=float(ritz.max()),
             energy_functional=np.array(energy), grid=np.array(shape))
    if FIELDS:  # for the deformation video: nodal displacement, damage at the pixel centres, beam mask
        xc, yc = (ix - nv + 0.5) * h - crack_x, np.maximum(0.0, (H - crack_a) - (iy - nv + 0.5) * h)
        r.update(u=np.array(u.s[:, 0]), beam=beam, nv=nv,
                 d=np.exp(-np.maximum(np.sqrt(xc ** 2 + yc ** 2) - core * ell, 0.0) / ell))
    return r


def plot():
    import matplotlib

    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter, NullFormatter

    labels = {'green': 'Green, projected', 'green_c': 'Green, corrected', 'dual': 'dual, projected',
              'dual_c': 'dual, corrected'}
    # Green = blue, dual = orange; corrected = dark, solid, filled marker; projected = light, dashed, open marker
    styles = {'green': dict(color='#86b6ef', ls=(0, (5, 3)), marker='s', mfc='white'),
              'green_c': dict(color='#184f95', ls='-', marker='o'),
              'dual': dict(color='#f4a582', ls=(0, (5, 3)), marker='D', mfc='white'),
              'dual_c': dict(color='#c4441a', ls='-', marker='^')}
    sfx = (f'_core{core:g}' if core > 0 else '') + (f'_a{crack_a:g}' if crack_a != 0.1 else '')
    cases = [(f'l2h_k{k_res:g}_eps{eps:g}' + sfx, r'$\ell = 2h$'), (f'l0.02_k{k_res:g}_eps{eps:g}' + sfx, r'$\ell = 0.02$ fixed')]
    fig, axes = plt.subplots(2, 4, figsize=(20, 9), layout='constrained')
    for row, (t, title) in enumerate(cases):
        ax_its, ax_max, ax_min, ax_time = axes[row]
        for p in labels:
            rows = []
            for f in sorted(os.listdir(RUNS)):
                if f.startswith(f'{t}_{p}_n') and f.endswith('.npz'):
                    rows.append(np.load(os.path.join(RUNS, f)))
            if not rows:
                continue
            rows.sort(key=lambda r: int(r['n_H']))
            h = np.array([int(r['n_H']) for r in rows])  # x: pixels across the beam height
            kw = dict(styles[p], lw=2, ms=7, mew=1.5, label=labels[p] if row == 0 else None)
            ax_its.semilogx(h, [int(r['cg_its']) for r in rows], **kw)
            ax_max.loglog(h, [float(r['lam_max']) for r in rows], **kw)
            ax_min.loglog(h, [float(r['lam_min']) for r in rows], **kw)
            ax_time.loglog(h, [float(r['time']) + float(r['setup']) for r in rows], **kw)
        for ax, yl in ((ax_its, 'CG iterations to $10^{-6}$ (energy)'), (ax_max, r'$\lambda_{max}$ (Ritz)'),
                       (ax_min, r'$\lambda_{min}$ (Ritz)'), (ax_time, 'wall time, setup + CG [s]')):
            ax.set_xlabel('$n_H$ (pixels across the beam height)')
            ax.set_ylabel(yl)
            ax.set_xscale('log', base=2)
            ax.set_xticks([16, 32, 64, 128, 256], ['16', '32', '64', '128', '256'])
            ax.xaxis.set_minor_formatter(NullFormatter())
            ax.grid(True, which='both', color='0.9')
            ax.set_title(f'{title}: {yl.split(" (")[0].split(" to")[0]}', fontsize=10)
        ax_its.set_ylim(bottom=0)
        for ax in (ax_its, ax_max, ax_min, ax_time):  # plain numbers on the y axes, no powers of ten
            ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f'{v:.0f}' if v >= 1 else np.format_float_positional(v, trim='-')))
            ax.yaxis.set_minor_formatter(NullFormatter())
    fig.legend(*axes[0, 0].get_legend_handles_labels(), loc='outside lower center', ncol=4, frameon=False, fontsize=12, handlelength=4)
    fig.suptitle(f'clamped cantilever with a frozen phase-field crack at x = {crack_x:g} (depth {crack_a:g}), '
                 f'k = {k_res:g}, void {void_t:g}, dual with the void as {eps:g} C$_0$'
                 + (f', broken core {core:g} $\\ell$' if core > 0 else ''), fontsize=12)
    out = os.path.join(HERE, 'exp_data', 'cantilever_crack_precond' + sfx + '.png')
    fig.savefig(out, dpi=130)
    print(f'wrote {out}')


if __name__ == '__main__':
    if os.environ.get('PLOT'):
        plot()
        sys.exit(0)
    for n_H in [int(n) for n in os.environ.get('N_H', '16,32,64,128').split(',')]:
        for p in PRECONDS:
            out = os.path.join(FIELD_RUNS if FIELDS else RUNS, f'{tag()}_{p}_n{n_H:04d}.npz')
            if os.path.exists(out):
                continue
            r = solve(n_H, p)
            np.savez(out, **r)
            print(f'{tag():24s} n_H {n_H:4d} grid {tuple(r["grid"])} l {r["ell"]:.4f} max d {r["d_max"]:.3f} '
                  f'{p:8s} CG its {r["cg_its"]:6d}  lambda [{r["lam_min"]:.2e}, {r["lam_max"]:.3f}]  '
                  f'{r["time"]:7.1f} s (setup {r["setup"]:.1f} s)  compliance {r["compliance"]:.8e}', flush=True)
