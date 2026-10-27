"""
A house under wind: 3D linear elasticity on voxels, the walls clamped at the foundation.

House 6 m deep (x, the wind direction) x 8 m long (z), eaves at 3 m, ridge at 5 m (y up). Masonry walls 0.25 m thick
(E = 10 GPa, nu = 0.2), a gable roof slab 0.25 m thick with 0.25 m overhang (same material). Door and windows closed by
soft panels (E = 0.2 GPa) that pass their wind load to the frames. Around the house a void of 0.5 m (zero stiffness).

Wind from -x, peak velocity pressure q = 1 kPa, Eurocode-style external pressure coefficients on the outer surface
(a voxel face counts if the straight line from it out of the house in its normal direction meets no solid):
    windward wall  (normal -x)  cp = +0.8   pushed in
    leeward wall   (normal +x)  cp = -0.5   sucked out
    side walls     (normal +-z) cp = -0.7   sucked out
    roof, upward faces (+y)     cp = -0.6   uplift
The force on a face is -cp q n A, split equally between its 4 nodes.

CG with energy stopping (1e-6, delay 10), Green preconditioner with the walls as reference, two versions:
    PRECOND=green            projected plain Green
    PRECOND=green_dirichlet  corrected Green, S = E^T G E solved by an inner CG in each outer iteration
                             (INNER_TOL, default 1e-4), never assembled

usage:  H=0.125 PRECOND=green_dirichlet python house_wind_3D.py      (solves, stores exp_data/house/<case>.npz)
        PLOT=1 H=0.125 python house_wind_3D.py                        (figures from the stored run)
"""
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import time

import numpy as np

HERE = os.path.dirname(os.path.realpath(__file__))
OUT = os.path.join(HERE, 'exp_data', 'house')
os.makedirs(OUT, exist_ok=True)

h = float(os.environ.get('H', 0.125))
precond = os.environ.get('PRECOND', 'green_dirichlet')
inner_tol = float(os.environ.get('INNER_TOL', 1e-4))
cg_tol, cg_delay = 1e-6, 10
DEPTH, LENGTH, EAVES, RIDGE = 6.0, 8.0, 3.0, 5.0
WALL, ROOF, OVERHANG, VOID = 0.25, 0.25, 0.25, 0.5
E_WALL, E_PANEL, POISSON = 10e9, 0.2e9, 0.2
q_wind = 1e3
CP = {'windward': 0.8, 'leeward': -0.5, 'side': -0.7, 'roof': -0.6}
case = f'house_h{h:g}_' + ('pgdi' + f'{inner_tol:g}' if precond == 'green_dirichlet' else 'plain')
slope = (RIDGE - EAVES) / (DEPTH / 2)

# openings: (wall, horizontal range along the wall, vertical range); wall 'x0' is the windward facade
OPENINGS = [('x0', (3.5, 4.5), (0.0, 2.1)),  # door
            ('x0', (1.0, 2.2), (1.0, 2.2)), ('x0', (5.8, 7.0), (1.0, 2.2)),
            ('x1', (1.5, 2.7), (1.0, 2.2)), ('x1', (5.3, 6.5), (1.0, 2.2)),
            ('z0', (2.4, 3.6), (1.0, 2.2)), ('z1', (2.4, 3.6), (1.0, 2.2)),
            ('z0', (2.6, 3.4), (3.4, 4.0)), ('z1', (2.6, 3.4), (3.4, 4.0))]


def geometry():
    """voxel grid and the phase of every voxel: 0 void, 1 wall/roof, 2 panel"""
    nx = int(round((DEPTH + 2 * OVERHANG + 2 * VOID) / h))
    ny = int(round((RIDGE + ROOF / np.cos(np.arctan(slope)) + 2 * VOID) / h)) + 1
    nz = int(round((LENGTH + 2 * OVERHANG + 2 * VOID) / h))
    x0, y0, z0 = VOID + OVERHANG, VOID, VOID + OVERHANG  # house corner at the foundation
    X, Y, Z = np.meshgrid((np.arange(nx) + 0.5) * h - x0, (np.arange(ny) + 0.5) * h - y0,
                          (np.arange(nz) + 0.5) * h - z0, indexing='ij')
    underside = EAVES + slope * (DEPTH / 2 - np.abs(X - DEPTH / 2))
    inside = (X > 0) & (X < DEPTH) & (Z > 0) & (Z < LENGTH) & (Y > 0)
    band = (X < WALL) | (X > DEPTH - WALL) | (Z < WALL) | (Z > LENGTH - WALL)
    walls = inside & band & (Y < underside)
    roof = (X > -OVERHANG) & (X < DEPTH + OVERHANG) & (Z > -OVERHANG) & (Z < LENGTH + OVERHANG) & \
           (Y >= underside) & (Y < underside + ROOF / np.cos(np.arctan(slope)))
    panel = np.zeros_like(walls)
    for wall, (a, b), (c, d) in OPENINGS:
        in_wall = {'x0': X < WALL, 'x1': X > DEPTH - WALL, 'z0': Z < WALL, 'z1': Z > LENGTH - WALL}[wall]
        along = Z if wall[0] == 'x' else X
        panel |= walls & in_wall & (along > a) & (along < b) & (Y > c) & (Y < d)
    # variants for diagnostics: HOUSE_ROOF=0 (no roof), HOUSE_PANELS=soft|wall|void (closed openings: soft panels,
    # masonry, or left open)
    if os.environ.get('HOUSE_ROOF', '1') == '0':
        roof[...] = False
    panels = os.environ.get('HOUSE_PANELS', 'soft')
    phase = np.zeros(X.shape, dtype=int)
    phase[walls | roof] = 1
    phase[panel] = {'soft': 2, 'wall': 1, 'void': 0}[panels]
    base = int(round(y0 / h))  # node plane of the foundation
    return phase, base


def wind_forces(phase):
    """nodal forces (3, nx, ny, nz) on the exposed outer faces, and the faces with their cp for the figures"""
    solid = phase > 0
    force = np.zeros((3,) + solid.shape)
    faces = []
    for axis, sign, kind in ((0, -1, 'windward'), (0, 1, 'leeward'), (2, -1, 'side'), (2, 1, 'side'),
                             (1, 1, 'roof')):
        s = solid if sign > 0 else np.flip(solid, axis)
        # exposed: no solid beyond the voxel in direction sign along axis
        beyond = np.flip(np.cumsum(np.flip(s, axis), axis), axis) - s
        exposed = s & (beyond == 0)
        if sign < 0:
            exposed = np.flip(exposed, axis)
        idx = np.argwhere(exposed)
        f = -CP[kind] * q_wind * h ** 2 * sign / 4  # force component along axis per node
        plane = idx[:, axis] + (1 if sign > 0 else 0)  # node plane of the face
        others = [a for a in range(3) if a != axis]
        for da in (0, 1):
            for db in (0, 1):
                node = idx.copy()
                node[:, axis] = plane
                node[:, others[0]] += da
                node[:, others[1]] += db
                node %= np.array(solid.shape)
                np.add.at(force[axis], tuple(node.T), f)
        faces.append((kind, axis, sign, idx))
    return force, faces


def corrected_green(disc, M_fun, fixed, inner_tol, inner_its, assemble=False):
    """corrected Green as PRECOND=green_dirichlet_inner in cantilever_mesh_study.py, 3 translations:
    [S B; B^T 0][lam; mu] = [E^T M r; N^T r], z = M (r - E lam) - N mu, S = E^T M E solved by an inner CG,
    or (assemble=True) assembled once from one Green application per fixed dof and inverted"""
    nb_fixed = len(fixed[0])
    Ey, MEy, Mr, w = (disc.get_unknown_size_field(name=f'gdi_{k}') for k in ('Ey', 'MEy', 'Mr', 'w'))

    def S_fun(y):
        Ey.s[...] = 0
        Ey.s[fixed] = y
        M_fun(Ey, MEy)
        return MEy.s[fixed].copy()

    S_inv = None
    if assemble:
        S = np.empty((nb_fixed, nb_fixed))
        unit = np.zeros(nb_fixed)
        for j in range(nb_fixed):
            unit[j] = 1.0
            S[:, j] = S_fun(unit)
            unit[j] = 0.0
        S_inv = np.linalg.inv(0.5 * (S + S.T))

    def S_solve(g, tol):
        if S_inv is not None:
            return S_inv @ g
        y, res = np.zeros_like(g), g.copy()
        p, rr, rr0 = res.copy(), g @ g, g @ g
        for it in range(1, 10 * nb_fixed + 1):
            Sp = S_fun(p)
            a = rr / (p @ Sp)
            y += a * p
            res -= a * Sp
            rr, rr_old = res @ res, rr
            if rr <= tol ** 2 * rr0:
                break
            p = res + rr / rr_old * p
        inner_its.append(it)
        return y

    B = np.stack([(fixed[0] == d).astype(float) for d in range(3)], axis=1)
    W = np.stack([S_solve(B[:, d], 1e-10) for d in range(3)], axis=1)
    BW_inv = np.linalg.inv(B.T @ W)
    inner_its.clear()

    def P_fun(r, z):
        M_fun(r, Mr)
        Sg = S_solve(Mr.s[fixed], inner_tol)
        mu_ = BW_inv @ (B.T @ Sg - np.array([r.s[d].sum() for d in range(3)]))
        w.s[...] = r.s
        w.s[fixed] -= Sg - W @ mu_
        M_fun(w, z)
        for d in range(3):
            z.s[d] -= mu_[d]
        z.s[fixed] = 0.0
    return P_fun


def solve():
    from muFFTTO import constraints
    from muFFTTO import domain
    from muFFTTO import material_models
    from muFFTTO import solvers
    from muFFTTO import topology_optimization

    phase, base = geometry()
    shape = phase.shape
    cell = domain.PeriodicUnitCell(domain_size=[n * h for n in shape], problem_type='elasticity')
    disc = domain.Discretization(cell=cell, nb_of_pixels_global=shape, discretization_type='finite_element',
                                 element_type='trilinear_hexahedron')
    lam_q = disc.get_quad_field_scalar(name='lam_first_lame')
    mu_q = disc.get_quad_field_scalar(name='mu_second_lame')
    lam_q.s[...] = 0
    mu_q.s[...] = 0
    for ph, E in ((1, E_WALL), (2, E_PANEL)):
        lam, mu = material_models.get_lame_parameters(E=E, poisson=POISSON)
        lam_q.s[..., phase == ph] = lam
        mu_q.s[..., phase == ph] = mu
    material = material_models.LinearElastic(discretization=disc, lam_1qxyz=lam_q, mu_1qxyz=mu_q,
                                             name='linear_isotropic_elasticity')

    # foundation: every node of the bottom face of a voxel standing on the base plane
    foot = phase[:, base, :] > 0
    clamped = np.zeros((shape[0], shape[2]), dtype=bool)
    for da in (0, 1):
        for db in (0, 1):
            clamped |= np.roll(np.roll(foot, da, 0), db, 1)
    constrained = disc.get_displacement_sized_field(name='constrained_dofs_field')
    constrained.s[...] = 0
    constrained.s[:, 0, :, base, :] = clamped
    prescribed = disc.get_displacement_sized_field(name='prescribed_displacement')
    prescribed.s[...] = 0.0
    nodal_constraints = constraints.NodalConstraints.from_mask(disc, constrained_mask=constrained,
                                                               prescribed_values=prescribed)

    nodal_force, _ = wind_forces(phase)
    force = disc.get_unknown_size_field(name='force_field')
    force.s[:, 0] = nodal_force

    def K_fun(x, Ax):
        disc.apply_system_matrix_mugrid_explicit_stress(constitutive=material.get_stress, input_field_inxyz=x,
                                                        output_field_inxyz=Ax, formulation='small_strain')
        disc.fft.communicate_ghosts(Ax)

    t_setup = time.time()
    lam, mu = material_models.get_lame_parameters(E=E_WALL, poisson=POISSON)
    green = disc.get_preconditioner_Green_mugrid(
        reference_material_data_ijkl=material_models.get_elastic_tensor_from_lame(dim=3, lam=lam, mu=mu))

    def M_fun(x, Px):
        disc.fft.communicate_ghosts(x)
        disc.apply_preconditioner_mugrid(preconditioner_Fourier_fnfnqks=green, input_nodal_field_fnxyz=x,
                                         output_nodal_field_fnxyz=Px)

    P_fun = nodal_constraints.project_preconditioner(M_fun)
    fixed = tuple(np.argwhere(constrained.s != 0).T)
    nb_fixed = len(fixed[0])
    inner_its = []
    if precond == 'green_dirichlet':
        P_fun = corrected_green(disc, M_fun, fixed, inner_tol, inner_its)
    setup_time = time.time() - t_setup
    print(f'{case}: grid {shape}, {int((phase == 1).sum())} wall voxels, {int((phase == 2).sum())} panel voxels, '
          f'{nb_fixed} fixed dofs, total wind force {force.s.sum(axis=(1, 2, 3, 4))} N', flush=True)

    rhs = disc.get_unknown_size_field(name='rhs_field')
    u = disc.get_unknown_size_field(name='solution')
    rhs.s[...] = force.s
    nodal_constraints.apply_to_rhs(rhs, K_fun)
    u.s[...] = 0
    K_free = nodal_constraints.project_operator(K_fun)
    energy_functional = []

    def record(iteration, x_s, r_s, p_s, z_s, rr):
        energy_functional.append(-0.5 * float(np.dot(rhs.s.ravel(), x_s.ravel()) + np.dot(r_s.ravel(), x_s.ravel())))

    t0 = time.time()
    norms = solvers.conjugate_gradients_mugrid_experimental(
        comm=disc.communicator, fc=disc.field_collection, hessp=K_free, b=rhs, x=u, P=P_fun, maxiter=20000,
        stop_crit_norm='energy_lower_estim', d_min=cg_delay, callback=record,
        **topology_optimization.energy_stop_kwargs(disc, K_free, u, cg_tol))[1]
    wall = time.time() - t0
    nodal_constraints.impose_prescribed_values(u)
    cg_its = int(norms['stop_iteration']['energy_lower_estim'])

    Ku = disc.get_unknown_size_field(name='Ku_field')
    K_fun(u, Ku)
    reaction = np.array([(Ku.s[d] - force.s[d])[constrained.s[d] != 0].sum() for d in range(3)])
    strain = disc.get_gradient_size_field(name='strain_field')
    stress = disc.get_gradient_size_field(name='stress_field')
    disc.fft.communicate_ghosts(u)
    disc.apply_gradient_operator_symmetrized_mugrid(u_inxyz=u, grad_u_ijqxyz=strain)
    material.get_stress(strain, stress)
    s = stress.s.mean(axis=2)  # voxel average of the quadrature points
    dev = s - np.trace(s) / 3 * np.eye(3)[:, :, None, None, None]
    von_mises = np.sqrt(1.5 * np.einsum('ij...,ij...->...', dev, dev))
    print(f'{case}: CG its {cg_its}, inner its {np.mean(inner_its) if inner_its else 0:.1f} per outer, '
          f'setup {setup_time:.1f} s, CG {wall:.1f} s, max |u| {np.abs(u.s).max() * 1e3:.4f} mm, '
          f'max von Mises (walls) {von_mises[phase == 1].max() / 1e3:.1f} kPa, reaction + load {reaction + force.s.sum(axis=(1, 2, 3, 4))}',
          flush=True)
    np.savez(os.path.join(OUT, f'{case}.npz'), phase=phase, base=base, h=h, u=np.array(u.s[:, 0]),
             von_mises=von_mises, cg_its=cg_its, inner_its=np.array(inner_its), setup_time=setup_time, time=wall,
             energy_functional=np.array(energy_functional), nb_fixed=nb_fixed, reaction=reaction)


def plot():
    import matplotlib

    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import cm, colors

    r = np.load(os.path.join(OUT, f'{case}.npz'))
    phase, u, vm = r['phase'], r['u'], r['von_mises']
    nx, ny, nz = phase.shape
    keep = (slice(0, nx), slice(int(r['base']), ny), slice(0, nz))  # cut the void below the foundation
    phase, vm = phase[keep], vm[keep]
    u = np.pad(u, ((0, 0), (0, 1), (0, 1), (0, 1)), mode='wrap')[(slice(None),) + tuple(
        slice(k.start, (k.stop or 0) + 1) for k in keep)]
    gx, gy, gz = np.meshgrid(*(np.arange(n + 1) * h for n in phase.shape), indexing='ij')
    magnify = 0.4 / np.abs(u).max()
    _, faces = wind_forces(r['phase'])

    def draw(ax, deformed, colour_by, elev, azim, title):
        d = magnify if deformed else 0.0
        X, Y, Z = gx + d * u[0], gy + d * u[1], gz + d * u[2]
        solid = phase > 0
        if colour_by == 'phase':
            fc = np.empty(phase.shape + (4,))
            fc[phase == 1] = colors.to_rgba('#d9c7a3')
            fc[phase == 2] = colors.to_rgba('#7fb3d5', 0.8)
        else:
            norm = colors.LogNorm(vmin=max(np.percentile(vm[phase == 1], 5), 1), vmax=vm[phase == 1].max())
            fc = cm.inferno(norm(np.maximum(vm, 1)))
        # plot axes: x = depth, y = length, z = height
        ax.voxels(X.transpose(0, 2, 1), Z.transpose(0, 2, 1), Y.transpose(0, 2, 1), solid.transpose(0, 2, 1),
                  facecolors=fc.transpose(0, 2, 1, 3), edgecolor=(0, 0, 0, 0.08), linewidth=0.2)
        ax.view_init(elev=elev, azim=azim)
        ax.set_box_aspect((nx, nz, phase.shape[1]))
        ax.set_axis_off()
        ax.set_title(title, fontsize=11)
        return None if colour_by == 'phase' else norm

    fig = plt.figure(figsize=(18, 13), layout='constrained')
    ax = fig.add_subplot(2, 2, 1, projection='3d')
    draw(ax, False, 'phase', 20, -130, 'set-up: masonry walls and roof (sand), closed door and window panels (blue)')
    # wind pressure arrows on the windward facade and the uplift on the roof, a sample of the faces
    z0 = int(r['base'])
    for kind, axis, sign, idx in faces:
        if kind not in ('windward', 'roof', 'leeward'):
            continue
        idx = idx[(idx[:, 0] % 4 == 0) & (idx[:, 2] % 4 == 0) & (idx[:, 1] % 4 == 0)] if kind != 'roof' else \
            idx[(idx[:, 0] % 4 == 0) & (idx[:, 2] % 4 == 0)]
        c = (idx + 0.5) * h
        c[:, 1] -= z0 * h
        c[:, axis] += sign * 0.5 * h
        vec = np.zeros(3)
        vec[axis] = -CP[kind] * sign * 0.9
        start = c - (vec if CP[kind] > 0 else 0 * vec)
        ax.quiver(start[:, 0], start[:, 2], start[:, 1], vec[0], vec[2], vec[1], color={'windward': 'C3',
                  'leeward': 'C0', 'roof': 'C2'}[kind], lw=0.8, arrow_length_ratio=0.3)
    ax.text2D(0.02, 0.02, 'wind from the left: pressure +0.8q (red), suction -0.5q leeward (blue), uplift -0.6q roof '
              '(green), -0.7q sides', transform=ax.transAxes, fontsize=9)
    ax = fig.add_subplot(2, 2, 2, projection='3d')
    norm = draw(ax, True, 'vm', 20, -130, f'deformed × {magnify:.0f}, von Mises stress, windward side')
    ax = fig.add_subplot(2, 2, 3, projection='3d')
    draw(ax, True, 'vm', 25, 50, 'deformed, leeward side')
    ax = fig.add_subplot(2, 2, 4)
    for c, label in (('pgdi', 'corrected Green (inner CG for S)'), ('plain', 'plain Green')):
        f = [p for p in os.listdir(OUT) if p.startswith(f'house_h{h:g}_{c}') and p.endswith('.npz')]
        if not f:
            continue
        rr = np.load(os.path.join(OUT, f[0]))
        Pi = rr['energy_functional']
        e = np.sqrt(np.maximum(Pi[:-1] - Pi[-1], 0) / -Pi[-1])
        e[e < 1e-7] = np.nan
        extra = f', {rr["inner_its"].mean():.0f} inner its each' if rr['inner_its'].size else ''
        ax.semilogy(e, lw=1.5, label=f'{label}: {int(rr["cg_its"])} its{extra}, '
                    f'{float(rr["time"]) + float(rr["setup_time"]):.0f} s')
    ax.set_xlabel('CG iteration')
    ax.set_ylabel(r'relative energy error $\|u - u_k\|_K / \|u\|_K$')
    ax.set_title(f'CG convergence ({int(r["nb_fixed"])} clamped dofs at the foundation)', fontsize=11)
    ax.grid(True, color='0.9')
    ax.legend(frameon=False)
    fig.colorbar(cm.ScalarMappable(norm=norm, cmap='inferno'), ax=fig.axes[1:3], shrink=0.6,
                 label='von Mises stress [Pa]')
    fig.suptitle(f'house under wind, q = {q_wind / 1e3:g} kPa, voxel {h * 100:g} cm, grid {r["phase"].shape}, '
                 f'max |u| = {np.abs(r["u"]).max() * 1e3:.3f} mm', fontsize=13)
    out = os.path.join(HERE, 'exp_data', f'house_wind_h{h:g}.png')
    fig.savefig(out, dpi=130)
    print(f'wrote {out}')


def plot_geometry():
    """the voxel geometry and the wind load, no solve"""
    import matplotlib

    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import colors

    phase, base = geometry()
    _, faces = wind_forces(phase)
    phase = phase[:, base:, :]
    nx, ny, nz = phase.shape
    fc = np.zeros(phase.shape + (4,))
    fc[phase == 1] = colors.to_rgba('#d9c7a3')
    fc[phase == 2] = colors.to_rgba('#5b9bd5', 0.9)
    arrow_colour = {'windward': 'C3', 'leeward': 'C0', 'side': 'C4', 'roof': 'C2'}
    fig = plt.figure(figsize=(18, 8), layout='constrained')
    for k, (elev, azim, title) in enumerate(((18, -125, 'windward side (wind from the left)'),
                                             (28, 55, 'leeward side and the roof'))):
        ax = fig.add_subplot(1, 2, k + 1, projection='3d')
        ax.voxels(phase.transpose(0, 2, 1) > 0, facecolors=fc.transpose(0, 2, 1, 3),
                  edgecolor=(0, 0, 0, 0.12), linewidth=0.2)
        for kind, axis, sign, idx in faces:  # a sample of the loaded faces, arrow length ~ |cp|
            step = 6
            idx = idx[np.all(idx[:, [a for a in range(3) if a != axis]] % step == step // 2, axis=1)]
            c = (idx + 0.5).astype(float)
            c[:, 1] -= base
            c[:, axis] += sign * 0.5
            vec = np.zeros(3)
            vec[axis] = -CP[kind] * sign * 2.5 / h * 0.125
            start = c - vec if CP[kind] > 0 else c
            ax.quiver(start[:, 0], start[:, 2], start[:, 1], vec[0], vec[2], vec[1], color=arrow_colour[kind],
                      lw=1.0, arrow_length_ratio=0.3)
        ax.view_init(elev=elev, azim=azim)
        ax.set_box_aspect((nx, nz, ny))
        ax.set_xlabel('x (wind) [voxels]')
        ax.set_ylabel('z [voxels]')
        ax.set_zlabel('y (up) [voxels]')
        ax.set_title(title, fontsize=11)
    for kind, c in arrow_colour.items():
        fig.axes[0].plot([], [], color=c, lw=2, label=f'{kind}: cp = {CP[kind]:+g}' + (' (both sides)' if kind == 'side' else ''))
    fig.axes[0].plot([], [], 's', color='#d9c7a3', label='masonry walls and roof, E = 10 GPa')
    fig.axes[0].plot([], [], 's', color='#5b9bd5', label='door and window panels, E = 0.2 GPa')
    fig.legend(*fig.axes[0].get_legend_handles_labels(), loc='lower center', ncol=6, frameon=False, fontsize=10)
    fig.suptitle(f'house 6 × 8 m, eaves 3 m, ridge 5 m, walls and roof 0.25 m, voxel {h * 100:g} cm, '
                 f'grid {nx} × {ny + base} × {nz}; walls clamped at the foundation; wind q = 1 kPa', fontsize=12)
    fig.get_layout_engine().set(rect=(0, 0.05, 1, 0.95))
    out = os.path.join(HERE, 'exp_data', f'house_geometry_h{h:g}.png')
    fig.savefig(out, dpi=130)
    print(f'wrote {out}')


if __name__ == '__main__':
    if os.environ.get('GEOMETRY'):
        plot_geometry()
    elif os.environ.get('PLOT'):
        plot()
    else:
        solve()
