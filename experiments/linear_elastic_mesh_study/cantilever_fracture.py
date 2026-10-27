"""
Full phase-field fracture (AT2) of a heterogeneous cantilever with holes: does the dual preconditioner keep the
elastic solves mesh independent through a whole crack propagation? (mesh study, corrected Green vs corrected dual)

Beam L x H = 1 x 0.2 (bilinear pixels, E = 1, nu = 0.3) with NB_HOLES (default 4) holes of radius 0.05 on the centre
line, in a void of thickness VOID_T (default 0.05), left face clamped. Heterogeneous stiffness (HETERO = kappa,
default 10): C_0(x) = kappa^-phi C_iso, phi = (1 + sin(2 pi x / 0.2 + 0.7) sin(2 pi y / 0.1 + 0.3)) / 2, as in
cantilever_mesh_study.py; the same in physical units on every mesh.

Damage as in house_damage_3D.py (Miehe history field, Amor split of the driving force). Elastic problem (MODEL):
'amor' (default) C(d) = g(d) C^+ + C^-, the Amor split also in the stiffness, so cracks closed in compression keep their
bulk stiffness (sign of tr eps frozen per staggered iteration, each solve linear); 'hybrid' C(d) = g(d) C_0(x);
g(d) = (1 - d)^2 + k,
AT2 crack density Gc/2 (d^2/l + l |grad d|^2), staggered per load step until max |d - d_old| < 1e-2. l = L_SCALE fixed
(default 0.0125: 2h at n_H = 32, 4h at 64, 8h at 128), so every mesh solves the same physical problem; Gc from the
tensile strength f_t = F_T (default 0.01 E) through the AT2 peak stress sigma_c = 9/16 sqrt(E Gc / (3 l)).

Load: displacement control by the mean vertical displacement of the right face. For a fixed damage field u = lam u_1
(u_1: unit total tip force, warm started), so one elastic solve per staggered iteration gives the force lam that hits
the target, also on the softening branch. Targets in multiples of the elastic tip deflection at which the largest
principal stress first reaches f_t: 0.25, 0.5, 0.75, then STEP (default 0.05; 0.005-0.02 adapted to the staggered
iterations once max d > 0.2); stop when the force has dropped below STOP_FRACTION (default 0.05) of its peak (with max d > 0.9) or at STOP_DISPLACEMENT (default 6) x.

Elastic preconditioner (PRECOND): green_c (corrected Green, C_ref = C_iso) or dual_c (M_c K(C_iso C_eps^-1 C_iso) M_c,
C_eps = C(d) + EPS C_iso pointwise (hybrid: max(s, EPS) C_iso), EPS default 1e-3, voids and holes as EPS C_iso);
damage: Green-Jacobi.
CG stops on the energy-error estimate, 1e-6, delay 10, warm started.

usage:  N_H=64 PRECOND=dual_c [DEVICE=gpu] python cantilever_fracture.py
writes exp_data/fracture/<case>.nc (fields d, u, load and CG its of every staggered iteration) and <case>_iterations.csv
(every staggered iteration: CG its and times of the elastic and the damage solve)
"""
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import time

import numpy as np

HERE = os.path.dirname(os.path.realpath(__file__))
OUT = os.path.join(HERE, 'exp_data', 'fracture')
os.makedirs(OUT, exist_ok=True)

L, H = 1.0, 0.2
n_H = int(os.environ.get('N_H', 32))
precond = os.environ.get('PRECOND', 'green_c')
device = os.environ.get('DEVICE', 'cpu')
void_t = float(os.environ.get('VOID_T', 0.05))
nb_holes = int(os.environ.get('NB_HOLES', 4))
hole_radius = 0.05
kappa = float(os.environ.get('HETERO', 10))
length_scale = float(os.environ.get('L_SCALE', 0.0125))
f_t = float(os.environ.get('F_T', 0.01))
eps = float(os.environ.get('EPS', 1e-3))
step_coarse = float(os.environ.get('STEP', 0.05))
# MODEL: 'amor' (default) the Amor split also in the elastic problem: C(d) = g(d) C^+ + C^-, C^+ = K a I x I + 2 mu P_dev,
# C^- = K (1 - a) I x I, a = [tr eps > 0] frozen per staggered iteration (from the last u), so a crack closed in
# compression keeps its bulk stiffness (no interpenetration); 'hybrid': C(d) = g(d) C_0 (compression degraded too)
model = os.environ.get('MODEL', 'amor')
k_residual = 1e-4
E_young, poisson = 1.0, 0.3
Gc = 3 * length_scale * (16 / 9 * f_t) ** 2 / E_young
staggered_tol, max_staggered = 1e-2, int(os.environ.get('MAX_STAGGERED', 300))
cg_tol, cg_delay = 1e-6, 10
stop_fraction = float(os.environ.get('STOP_FRACTION', 0.05))  # stop below this fraction of the peak force
stop_displacement = float(os.environ.get('STOP_DISPLACEMENT', 6.0))  # or at this tip deflection (x crit)
max_steps = 1000
case = f'nH{n_H}_{precond}_holes{nb_holes}_het{kappa:g}_l{length_scale:g}_ft{f_t:g}' + ('_amor' if model == 'amor' else '')


def main():
    if device == 'gpu':
        import cupy as xp
    else:
        xp = np
    from muFFTTO import constraints, domain, io_utils, material_models, solvers, topology_optimization

    h = H / n_H
    nv = max(1, int(round(void_t / h)))
    n_L = int(round(L / h))
    shape = (n_L + 2 * nv, n_H + 2 * nv)
    disc_u, disc_d = (domain.Discretization(
        cell=domain.PeriodicUnitCell(domain_size=[n * h for n in shape], problem_type=p), nb_of_pixels_global=shape,
        discretization_type='finite_element', element_type='bilinear_rectangle', device=device)
        for p in ('elasticity', 'conductivity'))

    # beam pixels (holes cut out), heterogeneous stiffness factor s_0 per pixel (0 in voids and holes)
    ix, iy = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing='ij')
    xc, yc = (ix - nv + 0.5) * h, (iy - nv + 0.5) * h  # pixel centres from the beam's lower left corner
    solid = (ix >= nv) & (ix < nv + n_L) & (iy >= nv) & (iy < nv + n_H)
    for i in range(nb_holes):
        solid &= (xc - (i + 0.5) * L / nb_holes) ** 2 + (yc - 0.5 * H) ** 2 > hole_radius ** 2
    phi = 0.5 * (1 + np.sin(2 * np.pi * xc / 0.2 + 0.7) * np.sin(2 * np.pi * yc / 0.1 + 0.3))
    s0_np = np.where(solid, kappa ** -phi, 0.0)
    s0, solid_x = xp.asarray(s0_np), xp.asarray(solid)

    lam, mu = material_models.get_lame_parameters(E=E_young, poisson=poisson)
    bulk = lam + mu  # 2D: lam + 2 mu / dim
    I2 = np.eye(2)
    C_iso = lam * np.einsum('ij,kl->ijkl', I2, I2) + mu * (np.einsum('ik,jl->ijkl', I2, I2) +
                                                          np.einsum('il,jk->ijkl', I2, I2))
    degraded_scale = disc_u.get_quad_field_scalar(name='degraded_scale')
    dual_scale = disc_u.get_quad_field_scalar(name='dual_scale')
    if model == 'amor':  # full C_ijkl per quadrature point
        C_d = disc_u.get_material_data_size_field_mugrid(name='C_amor')
        C_dual = disc_u.get_material_data_size_field_mugrid(name='C_dual_amor')
        II = xp.asarray(np.einsum('ij,kl->ijkl', I2, I2)).reshape(2, 2, 2, 2, 1, 1, 1)
        P_dev = xp.asarray(0.5 * (np.einsum('ik,jl->ijkl', I2, I2) + np.einsum('il,jk->ijkl', I2, I2))
                           - 0.5 * np.einsum('ij,kl->ijkl', I2, I2)).reshape(2, 2, 2, 2, 1, 1, 1)
        # Mandel form (symmetric 2 x 2 strains -> 3-vectors) for the pointwise inverse in the dual
        MANDEL, W_M = [(0, 0), (1, 1), (0, 1)], np.array([1.0, 1.0, np.sqrt(2.0)])

        def to_mandel(C):  # [2, 2, 2, 2, ...] -> [..., 3, 3]
            return xp.moveaxis(xp.stack([xp.stack([W_M[a] * W_M[b] * C[i, j, k, l] for b, (k, l) in enumerate(MANDEL)])
                                         for a, (i, j) in enumerate(MANDEL)]), (0, 1), (-2, -1))

        def from_mandel(M, out):  # [..., 3, 3] -> out [2, 2, 2, 2, ...]
            for a, (i, j) in enumerate(MANDEL):
                for b, (k, l) in enumerate(MANDEL):
                    for ii, jj in {(i, j), (j, i)}:
                        for kk, ll in {(k, l), (l, k)}:
                            out[ii, jj, kk, ll] = M[..., a, b] / (W_M[a] * W_M[b])

        M_iso = to_mandel(xp.asarray(C_iso))
    else:
        C_d = domain.ScaledMaterial([(degraded_scale, C_iso)])
        C_dual = domain.ScaledMaterial([(dual_scale, C_iso)])

    # clamp: all nodes of the left face of the beam; load: unit total vertical force on the right face
    cmask = np.zeros((2, 1) + shape)
    left = (ix == nv) & (iy >= nv) & (iy <= nv + n_H)
    cmask[:, 0, left] = 1
    constrained = disc_u.get_displacement_sized_field(name='constrained')
    constrained.s[...] = xp.asarray(cmask)
    prescribed = disc_u.get_displacement_sized_field(name='prescribed')
    prescribed.s[...] = 0
    nc = constraints.NodalConstraints.from_mask(disc_u, constrained_mask=constrained, prescribed_values=prescribed)
    fixed_np = tuple(np.argwhere(cmask != 0).T)
    fixed = tuple(xp.asarray(f) for f in fixed_np)
    m = len(fixed_np[0])
    free = 1.0 - constrained.s
    right = (ix == nv + n_L) & (iy >= nv) & (iy <= nv + n_H)
    force_np = np.zeros((2, 1) + shape)
    force_np[1, 0, right] = np.where((iy[right] == nv) | (iy[right] == nv + n_H), 0.5, 1.0) / n_H
    force, right_x = xp.asarray(force_np), xp.asarray(right)

    def K_u(x, Ax):
        disc_u.apply_system_matrix_mugrid(material_data_field=C_d, input_field_inxyz=x, output_field_inxyz=Ax,
                                          formulation='small_strain')
        disc_u.fft.communicate_ghosts(Ax)

    K_free = nc.project_operator(K_u)
    green_u = disc_u.get_preconditioner_Green_mugrid(reference_material_data_ijkl=C_iso, name='green_u')

    def G(x, Px):
        disc_u.fft.communicate_ghosts(x)
        disc_u.apply_preconditioner_mugrid(preconditioner_Fourier_fnfnqks=green_u, input_nodal_field_fnxyz=x,
                                           output_nodal_field_fnxyz=Px)

    # corrected Green: [S B; B^T 0][lam; mu] = [E^T G r; N^T r], z = G (r - E lam) - N mu, S = E^T G E assembled once
    t0 = time.time()
    Ey, MEy, Mr, wk = (disc_u.get_unknown_size_field(name=f'cg_{k}') for k in ('Ey', 'MEy', 'Mr', 'w'))
    S = xp.empty((m, m))
    for j in range(m):
        Ey.s[...] = 0
        Ey.s[tuple(int(f[j]) for f in fixed_np)] = 1.0
        G(Ey, MEy)
        S[:, j] = MEy.s[fixed]
    S_inv = xp.linalg.inv(0.5 * (S + S.T))
    B = xp.asarray(np.stack([(fixed_np[0] == d).astype(float) for d in range(2)], axis=1))
    W = S_inv @ B
    BW_inv = xp.linalg.inv(B.T @ W)

    def Mc(r, z):
        G(r, Mr)
        Sg = S_inv @ Mr.s[fixed]
        mu_ = BW_inv @ (B.T @ Sg - xp.stack([r.s[d].sum() for d in range(2)]))
        wk.s[...] = r.s
        wk.s[fixed] -= Sg - W @ mu_
        G(wk, z)
        for d in range(2):
            z.s[d] -= mu_[d]
        z.s[fixed] = 0.0

    t1, t2 = (disc_u.get_unknown_size_field(name=n) for n in ('dual_t1', 'dual_t2'))

    def M_dual(x, Px):
        Mc(x, t1)
        disc_u.apply_system_matrix_mugrid(material_data_field=C_dual, input_field_inxyz=t1, output_field_inxyz=t2,
                                          formulation='small_strain')
        disc_u.fft.communicate_ghosts(t2)
        t2.s[...] *= free
        Mc(t2, Px)

    P_u = {'green_c': Mc, 'dual_c': M_dual}[precond]
    setup_u = time.time() - t0

    # damage fields and operators (as house_damage_3D.py)
    damage, damage_old, rhs_d, temp_n = (disc_d.get_unknown_size_field(name=n) for n in
                                         ('damage', 'damage_old', 'rhs_d', 'temp_n'))
    damage_q, history, history_n, reaction, temp_q = (disc_d.get_quad_field_scalar(name=n) for n in
                                                      ('damage_q', 'history', 'history_n', 'reaction', 'temp_q'))
    grad_d = disc_d.get_gradient_size_field(name='grad_damage')
    for f in (damage, history, history_n):
        f.s[...] = 0
    strain = disc_u.get_gradient_size_field(name='strain')
    rhs_u, u, u_1 = (disc_u.get_unknown_size_field(name=n) for n in ('rhs_u', 'u', 'u_unit'))
    u.s[...] = 0
    u_1.s[...] = 0

    def apply_damage_operator(x, Ax, reaction_q):  # [Gc l B^T W B + N^T W c N] d
        disc_d.apply_gradient_operator_mugrid(u_inxyz=x, grad_u_ijqxyz=grad_d)
        grad_d.s[...] *= Gc * length_scale
        disc_d.apply_gradient_transposed_operator_mugrid(gradient_field_ijqxyz=grad_d, div_u_fnxyz=Ax,
                                                         apply_weights=True)
        disc_d.apply_N_operator_mugrid(x, temp_q)
        temp_q.s[...] *= reaction_q
        disc_d.apply_N_transposed_operator_mugrid(quad_field_ijqxyz=temp_q, nodal_field_inxyz=temp_n,
                                                  apply_weights=True)
        Ax.s[...] += temp_n.s

    def K_d(input_field_inxyz, output_field_inxyz):
        apply_damage_operator(input_field_inxyz, output_field_inxyz, reaction.s)

    green_d = disc_d.get_preconditioner_Green_mugrid(
        reference_material_data_ijkl=None, invert_zero_mode=True, name='green_d',
        operator=lambda input_field_inxyz, output_field_inxyz: apply_damage_operator(
            input_field_inxyz, output_field_inxyz, Gc / length_scale))
    x_scaled = disc_d.get_unknown_size_field(name='x_scaled_d')
    jacobi = [None]

    def M_d(x, Px):
        x_scaled.s[...] = jacobi[0].s * x.s
        disc_d.apply_preconditioner_mugrid(preconditioner_Fourier_fnfnqks=green_d, input_nodal_field_fnxyz=x_scaled,
                                           output_nodal_field_fnxyz=Px)
        Px.s[...] *= jacobi[0].s

    def update_material():
        disc_d.apply_N_operator_mugrid(damage, damage_q)
        g = (1 - damage_q.s[0, 0]) ** 2 + k_residual
        degraded_scale.s[0, 0] = g * s0
        if model != 'amor':
            dual_scale.s[...] = 1.0 / xp.maximum(degraded_scale.s, eps)
            return
        # tension (tr eps > 0, from the last u): g C_0; compression: K I x I + g 2 mu P_dev, all scaled by s_0
        tension = xp.einsum('ii...->...', compute_strain(u)) > 0
        C_d.s[...] = s0 * (xp.where(tension, g, 1.0) * bulk * II + g * 2 * mu * P_dev)
        # dual: C_iso (C_d + EPS C_iso)^-1 C_iso pointwise (voids and holes: C_iso / EPS)
        M_eps = to_mandel(C_d.s) + eps * M_iso
        from_mandel(M_iso @ xp.linalg.inv(M_eps) @ M_iso, C_dual.s)

    def compute_strain(field):
        disc_u.fft.communicate_ghosts(field)
        disc_u.apply_gradient_operator_symmetrized_mugrid(u_inxyz=field, grad_u_ijqxyz=strain)
        return strain.s

    def tensile_energy():  # Amor split, scaled by the local stiffness (zero in voids and holes)
        e = compute_strain(u)
        tr = xp.einsum('ii...->...', e)
        dev = e - tr / 2 * xp.asarray(I2).reshape(2, 2, 1, 1, 1)
        return (0.5 * bulk * xp.maximum(tr, 0) ** 2 + mu * xp.einsum('ij...,ij...->...', dev, dev)) * s0

    def cg(disc, K, b, x, P):
        t_cg = time.time()
        norms = solvers.conjugate_gradients_mugrid_experimental(
            comm=disc.communicator, fc=disc.field_collection, hessp=K, b=b, x=x, P=P, maxiter=20000,
            stop_crit_norm='energy_lower_estim', d_min=cg_delay,
            **topology_optimization.energy_stop_kwargs(disc, K, x, cg_tol))[1]
        its = norms['stop_iteration']['energy_lower_estim']
        return (20000 if its is None else int(its)), time.time() - t_cg

    def tip(field):
        return float(field.s[1, 0][right_x].mean())

    def solve_unit():
        rhs_u.s[...] = force
        nc.apply_to_rhs(rhs_u, K_u)
        its, t = cg(disc_u, K_free, rhs_u, u_1, P_u)
        nc.impose_prescribed_values(u_1)
        return its, t

    # ---- elastic tip deflection at which the largest principal stress first reaches f_t ----
    update_material()
    its0, t_lin = solve_unit()
    e = compute_strain(u_1)
    tr = xp.einsum('ii...->...', e)
    sig = (lam * tr * xp.asarray(I2).reshape(2, 2, 1, 1, 1) + 2 * mu * e) * s0
    p1 = 0.5 * (sig[0, 0] + sig[1, 1]) + xp.sqrt(0.25 * (sig[0, 0] - sig[1, 1]) ** 2 + sig[0, 1] ** 2)
    force_crit = f_t / float(xp.abs(p1).max())  # the beam bends both ways: |p1| (sign of the load irrelevant)
    target_crit = force_crit * tip(u_1)
    print(f'{case}: grid {shape}, h {h:g}, l = {length_scale:g} ({length_scale / h:g} h), Gc = {Gc:.3e}, '
          f'{int(solid.sum())} solid pixels, {m} fixed dofs, {precond} set up in {setup_u:.1f} s, linear solve '
          f'{its0} its ({t_lin:.1f} s); first f_t at tip force {force_crit:.4e}, deflection {target_crit:.4e}',
          flush=True)

    writer = io_utils.FieldWriter(
        os.path.join(OUT, f'{case}.nc'), [damage, u],
        attributes={'h': h, 'grid': list(shape), 'nv': nv, 'n_H': n_H, 'L': L, 'H': H, 'kappa': kappa,
                    'nb_holes': nb_holes, 'hole_radius': hole_radius, 'length_scale': length_scale, 'Gc': Gc,
                    'f_t': f_t, 'k_residual': k_residual, 'eps': eps, 'precond': precond, 'setup_u': setup_u,
                    'force_crit': force_crit, 'target_crit': target_crit},
        frame_variables={k: () for k in ('step', 'target', 'force', 'staggered', 'converged', 'd_max', 'cg_u',
                                          'cg_d', 'time_u', 'time_d', 'wall_time')})
    np.save(os.path.join(OUT, f'{case}_solid.npy'), np.stack([solid, s0_np]))
    log = open(os.path.join(OUT, f'{case}_iterations.csv'), 'w')
    log.write('step,staggered,target,force,cg_u,cg_d,time_u,time_d,damage_change\n')

    t_start = time.time()
    target, increment, peak = 0.0, step_coarse, 0.0
    for step in range(max_steps):
        target += (0.25 if target < 0.74 * target_crit else increment) * target_crit
        cg_u_step = cg_d_step = 0
        time_u_step = time_d_step = 0.0
        for stagger in range(max_staggered):
            its_u, time_u = solve_unit()
            force_ = target / tip(u_1)
            u.s[...] = force_ * u_1.s

            history.s[0, 0] = xp.maximum(history_n.s[0, 0], tensile_energy())
            reaction.s[...] = Gc / length_scale + 2 * history.s
            temp_q.s[...] = 2 * history.s
            disc_d.apply_N_transposed_operator_mugrid(quad_field_ijqxyz=temp_q, nodal_field_inxyz=rhs_d,
                                                      apply_weights=True)
            damage_old.s[...] = damage.s
            jacobi[0] = disc_d.get_preconditioner_Jacobi_mugrid(operator=K_d, name='jacobi_d')
            its_d, time_d = cg(disc_d, K_d, rhs_d, damage, M_d)
            update_material()
            change = float(xp.abs(damage.s - damage_old.s).max())
            cg_u_step, cg_d_step = cg_u_step + its_u, cg_d_step + its_d
            time_u_step, time_d_step = time_u_step + time_u, time_d_step + time_d
            log.write(f'{step},{stagger + 1},{target:.6e},{force_:.6e},{its_u},{its_d},{time_u:.4g},{time_d:.4g},'
                      f'{change:.4g}\n')
            log.flush()
            # one frame per staggered iteration (cg_u, cg_d: of this iteration), so the video shows the crack run
            writer.write(step=step, target=target, force=force_, staggered=stagger + 1,
                         converged=int(change < staggered_tol), d_max=float(damage.s.max()), cg_u=its_u, cg_d=its_d,
                         time_u=time_u, time_d=time_d, wall_time=time.time() - t_start)
            if change < staggered_tol:
                break
        history_n.s[...] = history.s
        peak = max(peak, force_)
        d_max = float(damage.s.max())
        print(f'step {step:3d}  tip {target / target_crit:6.2f} x crit  force {force_ / force_crit:6.3f} x crit '
              f'({force_ / peak:.2f} of peak)  max d {d_max:.3f}  staggered {stagger + 1:3d}  CG u {cg_u_step:6d} '
              f'({cg_u_step / (stagger + 1):.0f} per solve)  CG d {cg_d_step:5d}  {time.time() - t_start:.0f} s'
              + ('' if change < staggered_tol else '  NOT CONVERGED'), flush=True)
        if d_max > 0.2:  # adapt the deflection increment to the number of staggered iterations
            increment = min(increment, 0.02)
            increment = increment / 2 if stagger + 1 > 30 else increment * 1.5 if stagger + 1 <= 3 else increment
            increment = float(np.clip(increment, 0.005, 0.02))
        if (force_ < stop_fraction * peak and d_max > 0.9) or target > stop_displacement * target_crit:
            print(f'stop: force {force_ / peak:.2f} of the peak, tip {target / target_crit:.1f} x crit', flush=True)
            break
    writer.close()
    log.close()
    print(f'done in {time.time() - t_start:.0f} s', flush=True)


if __name__ == '__main__':
    main()
