"""
3D version of the cantilever study: does the theory of report/cantilever_cg_report.pdf hold in 3D?

A beam L x H x W = 1 x 0.2 x 0.2 (trilinear voxels, small strain, linear elastic, E = 1, nu = 0.3) in a periodic cell,
surrounded by void (zero stiffness). CG with the Green preconditioner (the beam as reference), energy-norm stopping
(1e-6, delay 10). Two supports:

    clamp     left face clamped (all three components), uniform shear load F_y on the right face
    traction  nothing constrained: the clamp replaced by its reactions as tractions on the left face (parabolic shear
              over the height, uniform over the width; linear bending stress of the moment F L) and the parabolic
              shear F on the right face; the nodal forces are in equilibrium to round-off

and two kinds of void: VOID pixels on each side (default 2), or VOID_THICKNESS = t in physical units.

Claims to check:
    1. a fixed number of void voxels -> lambda_min falls under refinement; a fixed physical void -> lambda_min constant
    2. clamp -> lambda_max grows like 1/h; traction -> lambda_max = 1 and a mesh-independent CG count

usage:  BC=clamp|traction VOID_THICKNESS=0.2 N_H=4,8,16,32 python cantilever_3D_study.py      (run_3D.sh runs all)
        PLOT_ONLY=1 python cantilever_3D_study.py                                              (the figure)
writes exp_data/runs3d/<case>_n_H<n>.npz per mesh (reused if present) and exp_data/cantilever_3D_study.png
"""
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import time

import numpy as np

HERE = os.path.dirname(os.path.realpath(__file__))
RUNS = os.path.join(HERE, 'exp_data', 'runs3d')
os.makedirs(RUNS, exist_ok=True)

beam_length, beam_height, beam_width = 1.0, 0.2, 0.2
E_young, poisson = 1.0, 0.3
total_force_y = -1e-4
cg_tol, cg_delay = 1e-6, 10
supports = os.environ.get('BC', 'clamp')
void_thickness = float(os.environ.get('VOID_THICKNESS', 0))  # > 0: physical thickness, else VOID pixels
nb_void_pixels = int(os.environ.get('VOID', 2))
mesh_list = [int(n) for n in os.environ.get('N_H', '4,8,16,32').split(',')]


def case_name(bc, t):
    return f'{bc}_' + (f'void{t:g}' if t > 0 else f'void{nb_void_pixels}px')


def ritz_values(r):
    """eigenvalues of the Lanczos matrix built from the CG coefficients"""
    a, b = r['cg_alpha'], r['cg_beta']
    k = len(a)
    T = np.diag(1 / a)
    T[1:, 1:] += np.diag(b[:k - 1] / a[:k - 1])
    off = np.sqrt(b[:k - 1]) / a[:k - 1]
    T += np.diag(off, 1) + np.diag(off, -1)
    return np.linalg.eigvalsh(T)


def solve(n_H):
    from muFFTTO import constraints
    from muFFTTO import domain
    from muFFTTO import material_models
    from muFFTTO import solvers
    from muFFTTO import topology_optimization

    h = beam_height / n_H
    n_L, n_W = int(round(beam_length / h)), int(round(beam_width / h))
    nb_void = int(round(void_thickness / h)) if void_thickness > 0 else nb_void_pixels
    number_of_pixels = (n_L + 2 * nb_void, n_H + 2 * nb_void, n_W + 2 * nb_void)
    cell = domain.PeriodicUnitCell(domain_size=[n * h for n in number_of_pixels], problem_type='elasticity')
    disc = domain.Discretization(cell=cell, nb_of_pixels_global=number_of_pixels,
                                 discretization_type='finite_element', element_type='trilinear_hexahedron')

    ix, iy, iz = disc.fft.icoords
    bx, by, bz = (nb_void, nb_void + n_L), (nb_void, nb_void + n_H), (nb_void, nb_void + n_W)
    beam_pixels = (ix >= bx[0]) & (ix < bx[1]) & (iy >= by[0]) & (iy < by[1]) & (iz >= bz[0]) & (iz < bz[1])

    def face(x_index):  # the nodes of a cross-section
        return (ix == x_index) & (iy >= by[0]) & (iy <= by[1]) & (iz >= bz[0]) & (iz <= bz[1])

    left, right = face(bx[0]), face(bx[1])

    lam, mu = material_models.get_lame_parameters(E=E_young, poisson=poisson)
    lam_q = disc.get_quad_field_scalar(name='lam_first_lame')
    mu_q = disc.get_quad_field_scalar(name='mu_second_lame')
    lam_q.s[...] = 0
    mu_q.s[...] = 0
    lam_q.s[..., beam_pixels] = lam
    mu_q.s[..., beam_pixels] = mu
    material = material_models.LinearElastic(discretization=disc, lam_1qxyz=lam_q, mu_1qxyz=mu_q,
                                             name='linear_isotropic_elasticity')

    constrained = disc.get_displacement_sized_field(name='constrained_dofs_field')
    constrained.s[...] = 0
    if supports == 'clamp':
        constrained.s[:, 0, left] = 1
    prescribed = disc.get_displacement_sized_field(name='prescribed_displacement')
    prescribed.s[...] = 0.0
    nodal_constraints = constraints.NodalConstraints.from_mask(disc, constrained_mask=constrained,
                                                               prescribed_values=prescribed)

    # nodal weights of a uniform traction across the width (trapezoid = exact for linear shape functions)
    w_z = np.full(n_W + 1, h)
    w_z[[0, -1]] = h / 2

    def consistent(traction):  # f_j = int t(eta) N_j(eta) d eta over the height, eta = y - H/2 (2-point Gauss: exact)
        eta = np.arange(n_H + 1) * h - beam_height / 2
        f = np.zeros(n_H + 1)
        for g in (-1 / np.sqrt(3), 1 / np.sqrt(3)):
            e = 0.5 * (eta[:-1] + eta[1:]) + 0.5 * h * g
            f[:-1] += 0.5 * traction(e) * (eta[1:] - e)
            f[1:] += 0.5 * traction(e) * (e - eta[:-1])
        return f

    force = disc.get_unknown_size_field(name='force_field')
    force.s[...] = 0
    area = beam_height * beam_width
    second_moment = beam_width * beam_height ** 3 / 12
    if supports == 'clamp':  # uniform shear on the right face
        uniform = consistent(lambda e: total_force_y / area + 0 * e)
        force.s[1, 0, right] = uniform[iy[right] - by[0]] * w_z[iz[right] - bz[0]]
    else:
        shear = consistent(lambda e: 1.5 * total_force_y / area * (1 - (2 * e / beam_height) ** 2))
        bending = consistent(lambda e: beam_length * total_force_y * e / second_moment)
        force.s[1, 0, right] = shear[iy[right] - by[0]] * w_z[iz[right] - bz[0]]
        force.s[1, 0, left] = -shear[iy[left] - by[0]] * w_z[iz[left] - bz[0]]
        force.s[0, 0, left] = bending[iy[left] - by[0]] * w_z[iz[left] - bz[0]]

    def K_fun(x, Ax):
        disc.apply_system_matrix_mugrid_explicit_stress(constitutive=material.get_stress, input_field_inxyz=x,
                                                        output_field_inxyz=Ax, formulation='small_strain')
        disc.fft.communicate_ghosts(Ax)

    C_ref = material_models.get_elastic_tensor_from_lame(dim=3, lam=lam, mu=mu)
    green = disc.get_preconditioner_Green_mugrid(reference_material_data_ijkl=C_ref)

    def M_fun(x, Px):
        disc.fft.communicate_ghosts(x)
        disc.apply_preconditioner_mugrid(preconditioner_Fourier_fnfnqks=green, input_nodal_field_fnxyz=x,
                                         output_nodal_field_fnxyz=Px)

    rhs = disc.get_unknown_size_field(name='rhs_field')
    u = disc.get_unknown_size_field(name='solution')
    rhs.s[...] = force.s
    nodal_constraints.apply_to_rhs(rhs, K_fun)
    u.s[...] = 0
    K_free = nodal_constraints.project_operator(K_fun)

    # history: energy functional (exact CG error), CG coefficients (Ritz values); the callback comes after the step
    # x_k = x_{k-1} + alpha_{k-1} p_{k-1} and before the direction update, so p_s is still p_{k-1}
    energy_functional, rz, cg_alpha, previous_x = [], [], [], [None]

    def record(iteration, x_s, r_s, p_s, z_s, rr):
        energy_functional.append(-0.5 * float(np.dot(rhs.s.ravel(), x_s.ravel()) + np.dot(r_s.ravel(), x_s.ravel())))
        rz.append(float(np.dot(r_s.ravel(), z_s.ravel())))
        if previous_x[0] is not None:
            cg_alpha.append(float(np.dot((x_s - previous_x[0]).ravel(), p_s.ravel()) / np.dot(p_s.ravel(), p_s.ravel())))
        previous_x[0] = np.copy(x_s)

    t0 = time.time()
    norms = solvers.conjugate_gradients_mugrid_experimental(
        comm=disc.communicator, fc=disc.field_collection, hessp=K_free, b=rhs, x=u,
        P=nodal_constraints.project_preconditioner(M_fun), maxiter=20000, stop_crit_norm='energy_lower_estim',
        d_min=cg_delay, callback=record, **topology_optimization.energy_stop_kwargs(disc, K_free, u, cg_tol))[1]
    wall = time.time() - t0
    nodal_constraints.impose_prescribed_values(u)
    return dict(n_H=n_H, h=h, pixels=np.array(number_of_pixels), nb_void=nb_void,
                nb_constrained_dofs=int(np.sum(constrained.s)), cg_its=int(norms['stop_iteration']['energy_lower_estim']),
                time=wall, compliance=float(np.sum(force.s * u.s)), force_sum=force.s.sum(axis=(1, 2, 3, 4)),
                energy_functional=np.array(energy_functional), cg_alpha=np.array(cg_alpha),
                cg_beta=np.array(rz[1:]) / np.array(rz[:-1]))


if not os.environ.get('PLOT_ONLY'):
    case = case_name(supports, void_thickness)
    for n_H in mesh_list:
        run_file = os.path.join(RUNS, f'{case}_n_H{n_H}.npz')
        if os.path.exists(run_file):
            continue
        r = solve(n_H)
        np.savez(run_file, **r)
        w = ritz_values(r)
        print(f'{case:22s} n_H = {n_H:3d}  grid {tuple(int(p) for p in r["pixels"])}  CG its {r["cg_its"]:4d}  '
              f'lambda [{w.min():.3e}, {w.max():.3f}]  compliance {r["compliance"]:.6e}  '
              f'sum of forces {np.abs(r["force_sum"]).max():.1e}  ({r["time"]:.0f} s)', flush=True)
    sys.exit(0)

# ------------------------------------------------------------------------------------------------------------------
# figure: CG iterations, lambda_max, lambda_min vs h; CG convergence on the finest mesh
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker

CASES = [('clamp', 0, 'clamped, 2 void voxels'), ('clamp', 0.2, 'clamped, void 0.2'),
         ('traction', 0, 'forces only, 2 void voxels'), ('traction', 0.2, 'forces only, void 0.2')]
styles = {('clamp', 0): ('C0', '--'), ('clamp', 0.2): ('C0', '-'), ('traction', 0): ('C3', '--'),
          ('traction', 0.2): ('C3', '-')}
fig, axes = plt.subplots(2, 2, figsize=(14, 10), layout='constrained')
axes = axes.ravel()
print(f'{"case":28s} {"n_H":>4s} {"grid":>16s} {"CG its":>7s} {"lam_min":>10s} {"lam_max":>8s} {"compliance":>12s}')
for bc, t, label in CASES:
    rows = []
    for n_H in (4, 8, 16, 32):
        path = os.path.join(RUNS, f'{case_name(bc, t)}_n_H{n_H}.npz')
        if not os.path.exists(path):
            continue
        r = np.load(path)
        w = ritz_values(r)
        rows.append((n_H, float(r['h']), int(r['cg_its']), w.min(), w.max(), float(r['compliance'])))
        print(f'{label:28s} {n_H:4d} {str(tuple(int(p) for p in r["pixels"])):>16s} {rows[-1][2]:7d} '
              f'{w.min():10.3e} {w.max():8.3f} {rows[-1][5]:12.6e}')
        last = r
    if not rows:
        continue
    n_H, h, its, lam_min, lam_max, _ = map(np.array, zip(*rows))
    colour, ls = styles[(bc, t)]
    slope = np.polyfit(np.log(h[-3:]), np.log(its[-3:]), 1)[0] if len(h) >= 3 else np.nan
    axes[0].loglog(h, its, 'o', color=colour, ls=ls, label=f'{label}  (finest: $\\propto h^{{{slope:.2f}}}$)')
    axes[1].loglog(h, lam_max, 'o', color=colour, ls=ls, label=label)
    axes[2].loglog(h, lam_min, 'o', color=colour, ls=ls, label=label)
    Pi = last['energy_functional']
    e = np.sqrt(np.maximum(Pi[:-1] - Pi[-1], 0) / -Pi[-1])
    e[e < 1e-7] = np.nan
    axes[3].semilogy(e, color=colour, ls=ls, lw=1.5, label=f'{label}, $n_H$ = {int(last["n_H"])}')
axes[1].axhline(1.0, color='0.5', ls=':', lw=1)
axes[0].set_ylabel('CG iterations to $10^{-6}$ (energy)')
axes[1].set_ylabel(r'$\lambda_{max}$ of $M^{-1}K$')
axes[2].set_ylabel(r'$\lambda_{min}$ of $M^{-1}K$')
axes[3].set_ylabel(r'relative energy error $\|u - u_k\|_K / \|u\|_K$')
axes[3].set_xlabel('CG iteration')
for ax, title in zip(axes, ('CG iterations vs h', 'largest eigenvalue vs h', 'smallest eigenvalue vs h',
                            'CG convergence on the finest mesh')):
    ax.set_title(title, fontsize=11)
    ax.grid(True, color='0.9', lw=0.5)
    ax.legend(frameon=False, fontsize=9)
    if ax is not axes[3]:
        ax.set_xlabel('voxel size h')
        ax.set_xticks([0.2 / n for n in (4, 8, 16, 32)], [f'1/{5 * n}' for n in (4, 8, 16, 32)])
        ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
fig.suptitle('3D beam 1 × 0.2 × 0.2 in void, Green preconditioner: clamp vs forces only, fixed voxels vs fixed '
             'physical void', fontsize=12)
out = os.path.join(HERE, 'exp_data', 'cantilever_3D_study.png')
fig.savefig(out, dpi=150)
print(f'wrote {out}')
