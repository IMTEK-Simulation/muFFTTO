"""
The house of house_wind_3D.py under growing wind, with phase-field damage (AT2): where does it crack? (GPU)

Same geometry, supports (walls clamped at the foundation) and wind pressure coefficients. Loads: self-weight (masonry
1800 kg/m^3, panels 500 kg/m^3) at full value, the wind q = 1 kPa times a load factor that grows step by step.

Damage as in experiments/dual_preconditioner/damage_AT2_small_strain (Miehe history field, Amor split, hybrid
formulation): C(d) = ((1 - d)^2 + k) C_0, AT2 crack density Gc/2 (d^2/l + l |grad d|^2), staggered per load step.
Only the masonry (walls and roof) can crack; Gc follows from the tensile strength f_t through the AT2 peak stress
sigma_c = 9/16 sqrt(E Gc / (3 l)), with l = 2 voxels.

Load control: before the damage run, the elastic problem is solved for the self-weight and for the wind alone; the
wind q_crit at which the largest principal stress in the masonry first reaches f_t sets the scale. The run is then
controlled by the mean horizontal eaves displacement: for a fixed damage field u = u_weight + lam u_wind is linear, so
in every staggered iteration both are solved (warm started) and the wind factor lam is chosen to hit the target
displacement. This follows the softening branch past the peak wind, so the cracks grow step by step.

Elastic solves: CG with the corrected Green, S assembled once (one Green application per fixed dof): it depends on the
reference material and the fixed dofs only, not on the damage. Damage solves: CG with Green-Jacobi.

usage:  H=0.125 python house_damage_3D.py
        writes exp_data/house/damage_<...>.nc (NetCDF, one frame per staggered iteration: damage, displacement, load,
        iteration counts and times, CG convergence histories (energy-error estimate per iteration, NaN-padded)) and
        damage_<...>_checkpoint.npz (fields of the last converged load step)
"""
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import time

import numpy as np

import house_wind_3D as house

h = house.h
OUT = house.OUT
f_t = float(os.environ.get('F_T', 0.2e6))
length_scale = 2 * h
Gc = 3 * length_scale * (16 / 9 * f_t) ** 2 / house.E_WALL
k_residual = 1e-4
RHO = {1: 1800.0, 2: 500.0}
gravity = os.environ.get('GRAVITY', '1') == '1'
staggered_tol, max_staggered = 1e-2, int(os.environ.get('MAX_STAGGERED', 300))
cg_tol, cg_delay = 1e-6, 10
MAX_HISTORY = 5000
# load path: the mean horizontal eaves displacement T is prescribed (the wind follows from it), in multiples of its
# elastic value at q_crit: 0.25, 0.5, 0.75, then increments STEP_COARSE, adapted to the number of staggered
# iterations once max d > 0.3 (at most STEP_FINE when a step needs > 20)
step_coarse = float(os.environ.get('STEP_COARSE', 0.1))
step_fine = float(os.environ.get('STEP_FINE', 0.02))
stop_fraction = float(os.environ.get('STOP_FRACTION', 0.2))  # stop when the wind is this fraction of its peak
stop_displacement = float(os.environ.get('STOP_DISPLACEMENT', 40.0))  # or the eaves moved this x elastic at q_crit
max_steps = int(os.environ.get('MAX_STEPS', 400))
case = f'damage_h{h:g}_ft{f_t / 1e6:g}' + ('' if gravity else '_nograv')


def main():
    import cupy as xp
    from muFFTTO import constraints
    from muFFTTO import domain
    from muFFTTO import io_utils
    from muFFTTO import material_models
    from muFFTTO import solvers
    from muFFTTO import topology_optimization

    phase, base = house.geometry()
    shape = phase.shape
    disc_u, disc_d = (domain.Discretization(
        cell=domain.PeriodicUnitCell(domain_size=[n * h for n in shape], problem_type=p), nb_of_pixels_global=shape,
        discretization_type='finite_element', element_type='trilinear_hexahedron', device='gpu')
        for p in ('elasticity', 'conductivity'))

    lam, mu = material_models.get_lame_parameters(E=house.E_WALL, poisson=house.POISSON)
    bulk = lam + 2 * mu / 3
    I3 = np.eye(3)
    C_iso = lam * np.einsum('ij,kl->ijkl', I3, I3) + mu * (np.einsum('ik,jl->ijkl', I3, I3) +
                                                          np.einsum('il,jk->ijkl', I3, I3))
    stiffness = xp.asarray(np.where(phase == 1, 1.0, np.where(phase == 2, house.E_PANEL / house.E_WALL, 0.0)))
    masonry = xp.asarray(phase == 1)
    degraded_scale = disc_u.get_quad_field_scalar(name='degraded_scale')
    C_d = domain.ScaledMaterial([(degraded_scale, C_iso)])

    # supports and loads
    foot = phase[:, base, :] > 0
    clamped = np.zeros((shape[0], shape[2]), dtype=bool)
    for da in (0, 1):
        for db in (0, 1):
            clamped |= np.roll(np.roll(foot, da, 0), db, 1)
    cmask = np.zeros((3, 1) + shape)
    cmask[:, 0, :, base, :] = clamped
    constrained = disc_u.get_displacement_sized_field(name='constrained_dofs_field')
    constrained.s[...] = xp.asarray(cmask)
    prescribed = disc_u.get_displacement_sized_field(name='prescribed_displacement')
    prescribed.s[...] = 0.0
    nodal_constraints = constraints.NodalConstraints.from_mask(disc_u, constrained_mask=constrained,
                                                               prescribed_values=prescribed)
    fixed_np = tuple(np.argwhere(cmask != 0).T)
    fixed = tuple(xp.asarray(f) for f in fixed_np)
    m = len(fixed_np[0])
    wind_np, _ = house.wind_forces(phase)
    weight_np = np.zeros_like(wind_np)
    if gravity:  # rho g h^3 per voxel, 1/8 to each of its nodes
        idx = np.argwhere(phase > 0)
        w = -9.81 * h ** 3 / 8 * np.array([RHO[p] for p in phase[tuple(idx.T)]])
        for corner in np.ndindex(2, 2, 2):
            np.add.at(weight_np[1], tuple(((idx + corner) % np.array(shape)).T), w)
    wind, weight = xp.asarray(wind_np), xp.asarray(weight_np)

    def K_u(x, Ax):
        disc_u.apply_system_matrix_mugrid(material_data_field=C_d, input_field_inxyz=x, output_field_inxyz=Ax,
                                          formulation='small_strain')
        disc_u.fft.communicate_ghosts(Ax)

    K_free = nodal_constraints.project_operator(K_u)
    green_u = disc_u.get_preconditioner_Green_mugrid(reference_material_data_ijkl=C_iso, name='green_u')

    def G(x, Px):
        disc_u.fft.communicate_ghosts(x)
        disc_u.apply_preconditioner_mugrid(preconditioner_Fourier_fnfnqks=green_u, input_nodal_field_fnxyz=x,
                                           output_nodal_field_fnxyz=Px)

    # corrected Green: [S B; B^T 0][lam; mu] = [E^T G r; N^T r], z = G (r - E lam) - N mu, S = E^T G E assembled
    t0 = time.time()
    Ey, MEy, Mr, wk = (disc_u.get_unknown_size_field(name=f'cg_{k}') for k in ('Ey', 'MEy', 'Mr', 'w'))
    S = xp.empty((m, m))
    for j in range(m):
        Ey.s[...] = 0
        Ey.s[tuple(int(f[j]) for f in fixed_np)] = 1.0
        G(Ey, MEy)
        S[:, j] = MEy.s[fixed]
    S_inv = xp.linalg.inv(0.5 * (S + S.T))
    del S
    B = xp.asarray(np.stack([(fixed_np[0] == d).astype(float) for d in range(3)], axis=1))
    W = S_inv @ B
    BW_inv = xp.linalg.inv(B.T @ W)

    def P_u(r, z):
        G(r, Mr)
        Sg = S_inv @ Mr.s[fixed]
        mu_ = BW_inv @ (B.T @ Sg - xp.stack([r.s[d].sum() for d in range(3)]))
        wk.s[...] = r.s
        wk.s[fixed] -= Sg - W @ mu_
        G(wk, z)
        for d in range(3):
            z.s[d] -= mu_[d]
        z.s[fixed] = 0.0
    setup_u = time.time() - t0

    # damage fields and operators
    damage, damage_old, rhs_d, temp_n = (disc_d.get_unknown_size_field(name=n) for n in
                                         ('damage', 'damage_old', 'rhs_d', 'temp_n'))
    damage_q, history, history_n, reaction, temp_q = (disc_d.get_quad_field_scalar(name=n) for n in
                                                      ('damage_q', 'history', 'history_n', 'reaction', 'temp_q'))
    grad_d = disc_d.get_gradient_size_field(name='grad_damage')
    for f in (damage, history, history_n):
        f.s[...] = 0
    strain = disc_u.get_gradient_size_field(name='strain')
    rhs_u, u = (disc_u.get_unknown_size_field(name=n) for n in ('rhs_u', 'u'))
    u.s[...] = 0

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
        degraded_scale.s[0, 0] = ((1 - damage_q.s[0, 0]) ** 2 + k_residual) * stiffness

    def compute_strain():
        disc_u.fft.communicate_ghosts(u)
        disc_u.apply_gradient_operator_symmetrized_mugrid(u_inxyz=u, grad_u_ijqxyz=strain)
        return strain.s

    def tensile_energy():  # Amor split, masonry only
        e = compute_strain()
        tr = xp.einsum('ii...->...', e)
        dev = e - tr / 3 * xp.asarray(I3).reshape(3, 3, 1, 1, 1, 1)
        psi = 0.5 * bulk * xp.maximum(tr, 0) ** 2 + mu * xp.einsum('ij...,ij...->...', dev, dev)
        return psi * masonry

    def cg(disc, K, b, x, P, label):  # returns the iterations, the energy-error estimates and the time
        t_cg = time.time()

        def progress(iteration, x_s, r_s, p_s, z_s, stop_crit):
            if iteration and iteration % 1000 == 0:
                print(f'      CG {label} it {iteration:5d}  {time.time() - t_cg:6.0f} s', flush=True)

        norms = solvers.conjugate_gradients_mugrid_experimental(
            comm=disc.communicator, fc=disc.field_collection, hessp=K, b=b, x=x, P=P, maxiter=20000,
            stop_crit_norm='energy_lower_estim', d_min=cg_delay, callback=progress,
            **topology_optimization.energy_stop_kwargs(disc, K, x, cg_tol))[1]
        if 'stop_iteration' not in norms:  # zero right-hand side (no self-weight): x = 0 at once
            return 0, np.zeros(0), time.time() - t_cg
        its = norms['stop_iteration']['energy_lower_estim']
        if its is None:
            print(f'      CG {label} NOT CONVERGED in 20000 iterations', flush=True)
            its = 20000
        return int(its), np.array(norms['energy_lower_estim']), time.time() - t_cg

    def padded(history_):
        out = np.full(MAX_HISTORY, np.nan)
        out[:min(len(history_), MAX_HISTORY)] = history_[:MAX_HISTORY]
        return out

    # ---- load range from the linear solution: q_crit where max principal stress (masonry) reaches f_t ----
    update_material()
    principal = {}
    for name, f in (('weight', weight), ('wind', wind)):
        rhs_u.s[...] = 0
        rhs_u.s[:, 0] = f
        nodal_constraints.apply_to_rhs(rhs_u, K_u)
        u.s[...] = 0
        its, _, t_lin = cg(disc_u, K_free, rhs_u, u, P_u, 'u')
        compute_strain()
        tr = xp.einsum('ii...->...', strain.s)
        stress_ = lam * tr * xp.asarray(I3).reshape(3, 3, 1, 1, 1, 1) + 2 * mu * strain.s
        principal[name] = (stress_ * masonry).mean(axis=2)  # voxel average of the quadrature points, [3, 3, x, y, z]
        print(f'linear {name}: CG its {its} ({t_lin:.0f} s), max |u| {float(xp.abs(u.s).max()) * 1e3:.4f} mm',
              flush=True)

    def max_principal(q, argmax=False):
        s_ = (principal['weight'] + q * principal['wind']).reshape(9, -1).T.reshape(-1, 3, 3)
        top = xp.linalg.eigvalsh(s_)[:, -1]
        return int(xp.argmax(top)) if argmax else float(top.max())

    lo, hi = 0.0, 1.0
    while max_principal(hi) < f_t:
        lo, hi = hi, 2 * hi
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if max_principal(mid) < f_t else (lo, mid)
    q_crit = hi
    where = np.unravel_index(max_principal(q_crit, argmax=True), shape)
    del principal
    print(f'{case}: grid {shape}, {m} fixed dofs, l = {length_scale:g} m, Gc = {Gc:.2f} N/m, weight '
          f'{weight_np.sum() / 1e3:.1f} kN, wind at 1 kPa {wind_np.sum(axis=(1, 2, 3)) / 1e3} kN, corrected Green '
          f'set up in {setup_u:.0f} s; linear first crack (max principal stress = f_t = {f_t / 1e6:g} MPa) at '
          f'q_crit = {q_crit:.3f} kPa, voxel {tuple(int(v) for v in where)} (phase {phase[where]}), height '
          f'{(where[1] - base + 0.5) * h:.2f} m; eaves displacement control, increments {step_coarse:g} (adaptive, '
          f'min 0.01) x elastic at q_crit; stop below {stop_fraction:g} of the peak wind or at '
          f'{stop_displacement:g} x', flush=True)

    # ---- damage run, controlled by the mean horizontal eaves displacement ----
    # for a fixed damage field the problem is linear: u = u_g + lam u_w (self-weight, wind of 1 kPa), so the wind
    # factor lam that gives the target eaves displacement T follows from one solve each (warm started)
    u_g, u_w = (disc_u.get_unknown_size_field(name=n) for n in ('u_weight', 'u_wind'))
    u_g.s[...] = 0
    u_w.s[...] = 0
    eaves = int(round((house.VOID + house.EAVES) / h))  # node plane of the eaves
    eaves_mask = xp.asarray(phase[:, eaves - 1, :] > 0)

    def eaves_ux(field):
        return float(field.s[0, 0][:, eaves, :][eaves_mask].mean())

    def masonry_cracked(d_nodal):  # volume of masonry voxels whose mean nodal damage exceeds 0.9
        d_vox = sum(xp.roll(d_nodal, (-a_, -b_, -c_), (0, 1, 2)) for a_, b_, c_ in np.ndindex(2, 2, 2)) / 8
        return float(((d_vox > 0.9) & masonry).sum()) * h ** 3

    def solve_split():
        its, hists, times = 0, [], 0.0
        for field, f in ((u_g, weight), (u_w, wind)):
            rhs_u.s[...] = 0
            rhs_u.s[:, 0] = f
            nodal_constraints.apply_to_rhs(rhs_u, K_u)
            i, hist, t = cg(disc_u, K_free, rhs_u, field, P_u, 'u')
            nodal_constraints.impose_prescribed_values(field)
            its, times = its + i, times + t
            hists.append(hist)
        return its, hists[1], times

    solve_split()
    target_crit = eaves_ux(u_g) + q_crit * eaves_ux(u_w)  # elastic eaves displacement at q_crit
    nc_file = os.path.join(OUT, f'{case}.nc')
    writer = io_utils.FieldWriter(
        nc_file, [damage, u],
        attributes={'h': h, 'grid': list(shape), 'base_node_plane': base, 'E_wall': house.E_WALL,
                    'E_panel': house.E_PANEL, 'poisson': house.POISSON, 'f_t': f_t, 'Gc': Gc,
                    'length_scale': length_scale, 'k_residual': k_residual, 'gravity': int(gravity),
                    'q_wind': house.q_wind, 'q_crit_linear': q_crit, 'cg_tol': cg_tol,
                    'staggered_tol': staggered_tol, 'preconditioner_u': 'corrected Green, S assembled',
                    'preconditioner_d': 'Green-Jacobi', 'nb_fixed_dofs': m, 'setup_u': setup_u,
                    'control': 'mean horizontal eaves displacement', 'target_crit': target_crit},
        frame_variables={k: () for k in ('step', 'load', 'target', 'staggered_iteration', 'converged',
                                          'damage_change', 'u_eaves', 'd_max', 'cracked_volume', 'cg_u', 'cg_d',
                                          'time_u', 'time_d', 'wall_time')} |
                        {'cg_u_history': (MAX_HISTORY,), 'cg_d_history': (MAX_HISTORY,)})
    print(f'NetCDF: {nc_file}; eaves displacement at q_crit (elastic) {target_crit * 1e3:.4f} mm', flush=True)
    t_start = time.time()
    target, increment, peak = 0.0, step_coarse, 0.0
    for step in range(max_steps):
        target += (0.25 if target < 0.74 * target_crit else increment) * target_crit
        cg_u_step = cg_d_step = 0
        for stagger in range(max_staggered):
            its_u, hist_u, time_u = solve_split()
            factor = (target - eaves_ux(u_g)) / eaves_ux(u_w)
            u.s[...] = u_g.s + factor * u_w.s

            history.s[0, 0] = xp.maximum(history_n.s[0, 0], tensile_energy())
            reaction.s[...] = Gc / length_scale + 2 * history.s
            temp_q.s[...] = 2 * history.s
            disc_d.apply_N_transposed_operator_mugrid(quad_field_ijqxyz=temp_q, nodal_field_inxyz=rhs_d,
                                                      apply_weights=True)
            damage_old.s[...] = damage.s
            jacobi[0] = disc_d.get_preconditioner_Jacobi_mugrid(operator=K_d, name='jacobi_d')
            its_d, hist_d, time_d = cg(disc_d, K_d, rhs_d, damage, M_d, 'd')
            update_material()
            change = float(xp.abs(damage.s - damage_old.s).max())
            cg_u_step += its_u
            cg_d_step += its_d
            d_max = float(damage.s.max())
            cracked = masonry_cracked(damage.s[0, 0])
            print(f'    step {step:3d} staggered {stagger + 1:3d}  wind {factor:7.3f} kPa  CG u {its_u:5d} '
                  f'({time_u:.0f} s)  CG d {its_d:4d} ({time_d:.0f} s)  max |d - d_old| {change:.2e}  '
                  f'max d {d_max:.3f}', flush=True)
            writer.write(step=step, load=factor, target=target, staggered_iteration=stagger + 1,
                         converged=int(change < staggered_tol), damage_change=change, u_eaves=target,
                         d_max=d_max, cracked_volume=cracked, cg_u=its_u, cg_d=its_d, time_u=time_u,
                         time_d=time_d, wall_time=time.time() - t_start, cg_u_history=padded(hist_u),
                         cg_d_history=padded(hist_d))
            if change < staggered_tol:
                break
        history_n.s[...] = history.s
        peak = max(peak, factor)
        np.savez(os.path.join(OUT, f'{case}_checkpoint.npz'), step=step, damage=xp.asnumpy(damage.s),
                 u_weight=xp.asnumpy(u_g.s), u_wind=xp.asnumpy(u_w.s), history=xp.asnumpy(history_n.s),
                 target=target, factor=factor)
        print(f'step {step:3d}  eaves {target * 1e3:8.4f} mm ({target / target_crit:6.2f} x elastic at q_crit)  '
              f'wind {factor:7.3f} kPa ({factor / max(peak, 1e-30):.2f} of peak)  max d {d_max:.3f}  '
              f'cracked masonry {cracked:.3f} m3  staggered {stagger + 1:3d}  CG u {cg_u_step:6d} '
              f'({cg_u_step / (2 * (stagger + 1)):.0f} per solve)  CG d {cg_d_step:6d}  {time.time() - t_start:.0f} s'
              + ('' if change < staggered_tol else '  NOT CONVERGED'), flush=True)
        # adapt the displacement increment to the number of staggered iterations
        if d_max > 0.3:
            increment = min(increment, step_fine) if stagger + 1 > 20 else increment
            increment = increment / 2 if stagger + 1 > 30 else increment * 1.5 if stagger + 1 <= 3 else increment
            increment = float(np.clip(increment, 0.01, 0.5))
        if (peak > 0 and factor < stop_fraction * peak and d_max > 0.9) or target > stop_displacement * target_crit:
            print(f'stop: wind {factor:.3f} kPa = {factor / peak:.2f} of the peak {peak:.3f} kPa, eaves displacement '
                  f'{target / target_crit:.1f} x elastic at q_crit, cracked masonry {cracked:.2f} m3', flush=True)
            break
    writer.close()


if __name__ == '__main__':
    main()
