"""Green preconditioners that see the grid map: 2x2 nodal block Jacobi scalings, on real Newton systems.

On an adapted grid the pulled-back operator has the geometric coefficients J_g F_g^-1 (.) F_g^-T: element areas vary
by up to 1/min_jacobian and the elements in the ligaments are stretched (stiff in one direction, soft in the other).
The Green operator G is FFT-diagonal and cannot contain them; a nodal scaling S can, P = S G S:

    Green                      G_solid                    (C_ref = tangent of the solid at F = I)
    Green-Jacobi               S = D^-1/2,   G_identity   D: diagonal of the tangent K   (what the runs use)
    Green + block Jacobi       S = B^-1/2,   G_identity   B: 2x2 nodal blocks of K       (x-y coupling, direction)
    Green + grid Jacobi        S = Dg^-1/2,  G_solid      Dg: diagonal of the pulled-back reference operator
    Green + grid block Jacobi  S = Bg^-1/2,  G_solid      Bg: its 2x2 nodal blocks       (geometry only, no material)

The pulled-back reference operator is B^T (J_g F_g^-1 C_ref F_g^-T) B + R: C_ref everywhere, so the grid scalings see
the grid map only. The blocks are probed: the operator applied to one displacement component on every 4th node in
each direction (the operators couple nodes of the same element only), 2 * 16 applications.

Each Newton system K s = -g is the first one of an increment of a finished run (predictor state, as in
energy_error_estimate_study.py), and every preconditioner runs PCG with the stopping rule of the runs (delayed
Hestenes-Stiefel energy estimate, tol 1e-6, delay 10). The true relative energy error of the returned step is checked
against a tight reference solve.

usage: python block_jacobi_study.py RUN_PY RUN_NC OUT_PREFIX [--increments 10 50 70 90]
       python block_jacobi_study.py --replot OUT_PREFIX      (figure only, from OUT_PREFIX.npz)
writes OUT_PREFIX.npz, OUT_PREFIX.png and prints a table
"""
import argparse
import os
import sys
import time

import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('paths', nargs='+', help='RUN_PY RUN_NC OUT_PREFIX, or OUT_PREFIX with --replot (RUN_PY: the '
                                             'run.py that produced RUN_NC, its setup is reused)')
parser.add_argument('--replot', action='store_true', help='only redraw the figure from OUT_PREFIX.npz')
parser.add_argument('--increments', type=int, nargs='+', default=None)
parser.add_argument('--tol', type=float, default=1e-6)
parser.add_argument('--delay', type=int, default=10)
parser.add_argument('--maxiter', type=int, default=100000)
args = parser.parse_args()
if args.replot:
    (args.out,) = args.paths
else:
    args.run_py, args.run_nc, args.out = args.paths
VARIANTS = ('Green', 'Green-Jacobi', 'Green + block Jacobi', 'Green + grid Jacobi', 'Green + grid block Jacobi')

PROBE_SPACING = 4                  # > coupling radius (1 node for Q1) and a divisor of the grid size


def load_setup(preconditioner):
    """Namespace of run.py executed up to its load loop, with the given mechanics preconditioner, no output."""
    source = open(args.run_py).read()
    source = source[:source.index('start_time = time.time()')]
    source = source.replace("mechanics_preconditioner = 'Green_Jacobi'", f"mechanics_preconditioner = '{preconditioner}'")
    source = source.replace("make_figures = True", "make_figures = False")
    source = source.replace('\noutput_root = ', '\noutput_root = None  # ', 1)
    namespace = {'__name__': 'study', '__file__': os.path.abspath(args.run_py)}
    exec(compile(source, args.run_py, 'exec'), namespace)
    return namespace


def apply_green(namespace, v):
    """G v with the Green operator of the namespace (no Jacobi scaling)."""
    namespace['r_field'].s[...] = v.reshape(namespace['u_shape'])
    namespace['disc_u'].fft.communicate_ghosts(namespace['r_field'])
    namespace['disc_u'].apply_preconditioner_mugrid(preconditioner_Fourier_fnfnqks=namespace['green_u'],
                                                    input_nodal_field_fnxyz=namespace['r_field'],
                                                    output_nodal_field_fnxyz=namespace['Pv_field'])
    return np.array(namespace['Pv_field'].s).ravel()


class GridReferenceOperator:
    """v -> B^T (J_g F_g^-1 C_ref F_g^-T) B v + R v: the reference material everywhere, pulled back with the grid map
    (as update_tangent pulls back the tangent), so it sees the grid only."""

    def __init__(self, setup, reference_tangent):
        self.setup = setup
        disc_u, J_g, F_g_inv = setup['disc_u'], setup['J_g'], setup['F_g_inv']
        C_ref = np.broadcast_to(np.asarray(reference_tangent).reshape(2, 2, 2, 2, *(1,) * J_g.ndim),
                                (2, 2, 2, 2, *J_g.shape))
        pulled = np.einsum('imkn...,jm...->ijkn...', C_ref, F_g_inv)
        self.material = disc_u.get_material_data_size_field_mugrid(name='grid_reference')
        self.material.s[...] = J_g * np.einsum('ijkn...,ln...->ijkl...', pulled, F_g_inv)
        self.v = disc_u.get_unknown_size_field(name='grid_reference_in')
        self.Kv = disc_u.get_unknown_size_field(name='grid_reference_out')

    def __call__(self, v):
        disc_u = self.setup['disc_u']
        self.v.s[...] = v.reshape(self.setup['u_shape'])
        disc_u.fft.communicate_ghosts(self.v)
        disc_u.apply_system_matrix_mugrid(material_data_field=self.material, input_field_inxyz=self.v,
                                          output_field_inxyz=self.Kv, formulation='finite_strain')
        self.setup['add_regularization_of_the_problem'](self.v, self.Kv)
        return np.array(self.Kv.s).ravel()


def nodal_blocks(operator, u_shape):
    """2x2 diagonal blocks of a symmetric operator per node, [x, y, i, j], by probing."""
    n = u_shape[-1]
    blocks = np.zeros((n, n, 2, 2))
    for ox in range(PROBE_SPACING):
        for oy in range(PROBE_SPACING):
            for j in range(2):
                probe = np.zeros(u_shape)
                probe[j, 0, ox::PROBE_SPACING, oy::PROBE_SPACING] = 1.0
                response = operator(probe.ravel()).reshape(u_shape)
                blocks[ox::PROBE_SPACING, oy::PROBE_SPACING, :, j] = np.moveaxis(
                    response[:, 0, ox::PROBE_SPACING, oy::PROBE_SPACING], 0, -1)
    return 0.5 * (blocks + np.swapaxes(blocks, -1, -2))


def check_blocks(operator, blocks, u_shape, nb_checks=6):
    """Largest relative deviation of probed blocks from exact unit-vector columns."""
    rng = np.random.default_rng(0)
    deviation = 0.0
    for _ in range(nb_checks):
        x, y, j = rng.integers(u_shape[-2]), rng.integers(u_shape[-1]), rng.integers(2)
        unit = np.zeros(u_shape)
        unit[j, 0, x, y] = 1.0
        column = operator(unit.ravel()).reshape(u_shape)[:, 0, x, y]
        deviation = max(deviation, np.abs(column - blocks[x, y, :, j]).max() / np.abs(blocks[x, y]).max())
    return deviation


def inverse_square_root(blocks):
    """B^-1/2 per node (symmetric); eigenvalues clipped to a small positive floor."""
    values, vectors = np.linalg.eigh(blocks)
    values = np.maximum(values, 1e-12 * values.max())
    return np.einsum('xyik,xyk,xyjk->xyij', vectors, values ** -0.5, vectors)


def block_scaling(S, u_shape):
    """v -> S v with the nodal 2x2 blocks S [x, y, i, j]."""
    def apply(v):
        nodal = v.reshape(u_shape)[:, 0]                                       # [i, x, y]
        return np.einsum('xyij,jxy->ixy', S, nodal)[:, np.newaxis].ravel()
    return apply


def diagonal_scaling(blocks):
    """v -> D^-1/2 v with D the diagonal of the blocks."""
    d = np.moveaxis(np.diagonal(blocks, axis1=-2, axis2=-1), -1, 0)[:, np.newaxis]   # [i, 1, x, y]
    scale = (np.maximum(d, 1e-12 * d.max()) ** -0.5).ravel()
    return lambda v: scale * v


def scaled(green, scaling):
    """P = S G S."""
    return lambda v: scaling(green(scaling(v)))


def pcg(b, K, P, tol=None):
    """PCG from 0, stopping with the rule of the runs (tol given) or at a relative residual of 1e-13; returns the
    step and the number of iterations (None on negative curvature)."""
    s, r = np.zeros_like(b), b.copy()
    z = P(r)
    p, rz = z.copy(), r @ z
    steps = []                                                     # Delta_j = alpha_j (r_j, z_j)
    for iteration in range(args.maxiter):
        Kp = K(p)
        curvature = p @ Kp
        if curvature <= 0.0:
            return s, None
        alpha = rz / curvature
        s += alpha * p
        r -= alpha * Kp
        steps.append(alpha * rz)
        if tol is not None:
            k = len(steps) - 1
            if k >= args.delay and sum(steps[k - args.delay:]) <= tol ** 2 * sum(steps):
                return s, iteration + 1
        elif np.linalg.norm(r) <= 1e-13 * np.linalg.norm(b):
            return s, iteration + 1
        z = P(r)
        rz_new = r @ z
        p = z + (rz_new / rz) * p
        rz = rz_new
    return s, args.maxiter


def set_state(setup, u, u_fallback, damage, lam):
    """Put the state into the setup; the fallback fluctuation is used if u is not admissible."""
    setup['damage'].s[...] = damage.reshape(setup['damage'].s.shape)
    setup['update_degradation']()
    setup['lam_current'] = lam
    setup['u_fluc'].s[...] = u.reshape(setup['u_shape'])
    if not setup['set_F'](lam, setup['u_fluc']) > 0.0:
        setup['u_fluc'].s[...] = u_fallback.reshape(setup['u_shape'])
        setup['set_F'](lam, setup['u_fluc'])
    return setup['field_to_optimizer'](setup['u_fluc'])


def study():
    """Run the study for the chosen increments; returns ({frame: entry}, n)."""
    import netCDF4

    # G_solid from the Green setup; G_identity, the tangent and the state from the Green-Jacobi setup
    green_setup = load_setup('Green')
    setup = load_setup('Green_Jacobi')
    u_shape, n = setup['u_shape'], setup['nnn']
    green_solid = lambda v: apply_green(green_setup, v)
    green_identity = lambda v: apply_green(setup, v)

    # the grid scalings do not depend on the state
    grid_reference = GridReferenceOperator(setup, green_setup['reference_tangent'])
    grid_blocks = nodal_blocks(grid_reference, u_shape)
    print(f'grid reference blocks: probing deviation {check_blocks(grid_reference, grid_blocks, u_shape):.1e}')
    static = {'Green': green_solid,
              'Green + grid Jacobi': scaled(green_solid, diagonal_scaling(grid_blocks)),
              'Green + grid block Jacobi': scaled(green_solid,
                                                  block_scaling(inverse_square_root(grid_blocks), u_shape))}

    dataset = netCDF4.Dataset(args.run_nc)
    lam_path = np.asarray(dataset['lam'][:])
    u_path = np.asarray(dataset['u_fluc'][:])
    damage_path = np.asarray(dataset['damage'][:])
    F11_path = 1 + lam_path * np.asarray(dataset.H_macro).reshape(2, 2)[1, 1]
    increments = args.increments or sorted({1, len(lam_path) // 3, (2 * len(lam_path)) // 3, len(lam_path) - 2})

    results = {}
    for k in increments:
        # the first Newton system of increment k + 1, at the predictor as in the runs
        lam_next = lam_path[k + 1]
        u_k, u_previous = u_path[k], u_path[k - 1] if k > 0 else np.zeros_like(u_path[k])
        dlam_k = lam_path[k] - (lam_path[k - 1] if k > 0 else 0.0)
        predictor = u_k + (lam_next - lam_path[k]) / dlam_k * (u_k - u_previous)
        x = set_state(setup, predictor, u_k, damage_path[k], lam_next)
        b = -setup['fun_grad'](x)[1].ravel()
        K = lambda v: setup['hessp'](x, v.reshape(u_shape)).ravel()
        tangent_blocks = nodal_blocks(K, u_shape)
        deviation = check_blocks(K, tangent_blocks, u_shape)
        preconditioners = dict(static)
        preconditioners['Green-Jacobi'] = scaled(green_identity, diagonal_scaling(tangent_blocks))
        preconditioners['Green + block Jacobi'] = scaled(
            green_identity, block_scaling(inverse_square_root(tangent_blocks), u_shape))
        start = time.time()
        # the Green-Jacobi of the runs: the same operator as the probed one up to a constant factor and round-off
        # (~1e-16); on ill-conditioned systems CG amplifies that round-off (loss of orthogonality), so the difference
        # of the two counts measures the round-off noise of the CG counts
        setup['update_mechanics_jacobi'](lam_next, setup['u_fluc'])
        _, count_runs = pcg(b, K, lambda v: setup['precond'](v.reshape(u_shape)).ravel(), tol=args.tol)
        reference, count = pcg(b, K, preconditioners['Green-Jacobi'])
        if count is None:
            print(f'frame {k}: negative curvature, skipped')
            continue
        energy = reference @ K(reference)
        entry = {'F11': F11_path[k + 1], 'Green-Jacobi, other round-off': count_runs}
        for name in VARIANTS:
            s, count = pcg(b, K, preconditioners[name], tol=args.tol)
            error = np.sqrt(max((reference - s) @ K(reference - s), 0.0) / energy) if count else np.nan
            entry[name] = (count, error)
        results[k] = entry
        print(f'frame {k:3d} F11 {entry["F11"]:.3f} (probing deviation {deviation:.0e}, Green-Jacobi of the runs, '
              f'other round-off: {count_runs} CG, {time.time() - start:.0f} s): '
              + ' | '.join(f'{name} {c if c else "neg. curv."} CG, err {e:.1e}' for name, (c, e) in
                           ((name, entry[name]) for name in VARIANTS)))
        sys.stdout.flush()
    return results, n


if args.replot:
    saved = np.load(args.out + '.npz', allow_pickle=True)
    results, n, run_nc = saved['results'].item(), int(saved['n']), str(saved['run_nc'])
else:
    results, n = study()
    run_nc = args.run_nc
    np.savez(args.out + '.npz', results=np.array(results, dtype=object), run_nc=run_nc, n=n)

# ---- figure: CG iterations per variant and state -------------------------------------------------------------------
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

SURFACE, TEXT_1, TEXT_2, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1", "#a3a29c"
COLORS = {'Green': "#2a78d6", 'Green-Jacobi': "#eb6834", 'Green + block Jacobi': "#1baf7a",
          'Green + grid Jacobi': "#8f6bd8", 'Green + grid block Jacobi': "#d6457d"}
adapted = 'adapted' in os.path.basename(os.path.dirname(run_nc))
fig, ax = plt.subplots(figsize=(11, 5.6), facecolor=SURFACE)
fig.subplots_adjust(left=0.08, right=0.98, top=0.8, bottom=0.12)
fig.text(0.08, 0.93, f'Grid-aware Green preconditioners, {n}² {"adapted" if adapted else "regular"} grid',
         fontsize=15, fontweight='bold', color=TEXT_1)
fig.text(0.08, 0.85, f'CG iterations to the energy stop (tol {args.tol:g}, delay {args.delay}) on the first Newton '
                      f'system of an increment;\nlabel: true relative energy error of the returned step',
         fontsize=10, color=TEXT_2)
ax.set_facecolor(SURFACE)
for side in ('top', 'right'):
    ax.spines[side].set_visible(False)
for side in ('left', 'bottom'):
    ax.spines[side].set_color(AXIS)
ax.tick_params(colors=TEXT_2, labelsize=10)
ax.grid(axis='y', color=GRID, linewidth=0.8)
ax.set_axisbelow(True)
states = list(results)
width = 0.8 / len(VARIANTS)
for index, name in enumerate(VARIANTS):
    positions = np.arange(len(states)) + (index - (len(VARIANTS) - 1) / 2) * width
    counts = [results[k][name][0] or 0 for k in states]
    ax.bar(positions, counts, width=width * 0.9, color=COLORS[name], label=name)
    for position, k in zip(positions, states):
        count, error = results[k][name]
        ax.text(position, (count or 0) * 1.08 + 1, f'{error:.0e}' if count else 'n.c.', ha='center', va='bottom',
                fontsize=7, color=TEXT_2, rotation=90)
ax.set_yscale('log')
ax.set_xticks(np.arange(len(states)), [f'F11 = {results[k]["F11"]:.3f}' for k in states])
ax.set_ylabel('CG iterations (log)', color=TEXT_2)
ax.legend(frameon=False, fontsize=9, ncol=3, loc='upper left')
ax.set_ylim(top=ax.get_ylim()[1] * 8)
fig.savefig(args.out + '.png', dpi=150, facecolor=SURFACE)
print(f'wrote {args.out}.npz and {args.out}.png')
