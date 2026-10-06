"""
Re-optimize the result of example_2D_conductivity_TO.py with the phase field
restricted to the discrete set {0, 1/n, ..., 1}.

Starts from the smooth optimum rounded to the nearest level and runs a
sensitivity-guided discrete descent on the same objective: in each step the
pixels with the largest predicted decrease move by one level, the step is
accepted only if the objective decreases, and the number of moved pixels
is adapted (doubled on success, halved on failure).
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from mpi4py import MPI

import example_2D_conductivity_TO as base
from muFFTTO import solvers
from muFFTTO import topology_optimization

assert MPI.COMM_WORLD.size == 1, 'serial only'

args = [a for a in sys.argv[1:] if not a.startswith('--')]
nb_phase_levels = int(args[0]) if args else 20
# --flux-only: minimize only the weighted flux mismatch, without the phase-field term
flux_only = '--flux-only' in sys.argv
max_steps = 2000
nb_moves_init = 64
max_failed_in_a_row = 200  # stop after this many consecutive rejected single-pixel moves

script_name = os.path.splitext(os.path.basename(__file__))[0]
file_folder_path = os.path.dirname(os.path.realpath(__file__))
base_data_path = os.path.join(file_folder_path, 'data', 'example_2D_conductivity_TO') + '/'
data_folder_path = os.path.join(file_folder_path, 'data', script_name) + '/'
os.makedirs(data_folder_path, exist_ok=True)

file_tag = f'{base.preconditioner_type}_eta_{base.eta}_w_{base.weights[0]}'
smooth = np.load(base_data_path + file_tag + '_smooth.npy')
run_tag = '_flux_only' if flux_only else ''

disc = base.discretization


def homogenized_conductivity(phase):
    """Effective conductivity tensor of a phase field (same model as the optimization)."""
    rho = disc.get_scalar_field(name='rho_homogenization')
    rho_q = disc.get_quad_field_scalar(name='rho_q_homogenization')
    mat = disc.get_material_data_size_field_mugrid(name='mat_homogenization')
    grad = disc.get_gradient_size_field(name='grad_homogenization')
    rhs = disc.get_unknown_size_field(name='rhs_homogenization')
    u = disc.get_unknown_size_field(name='u_homogenization')

    rho.s[0, 0] = phase
    disc.fft.communicate_ghosts(rho)
    disc.apply_N_operator_mugrid(rho, rho_q)
    C_diff = base.conductivity_C_0 - base.conductivity_C_void
    mat.s[...] = C_diff[..., np.newaxis, np.newaxis, np.newaxis] * np.power(rho_q.s, base.p)[0, 0] + \
                 base.conductivity_C_void[..., np.newaxis, np.newaxis, np.newaxis]

    def K_fun(x, Ax):
        disc.apply_system_matrix_mugrid(material_data_field=mat, input_field_inxyz=x, output_field_inxyz=Ax)

    C = np.zeros([base.dim, base.dim])
    for i in range(base.dim):
        macro_gradient = np.zeros(base.dim)
        macro_gradient[i] = 1
        disc.get_macro_gradient_field_mugrid(macro_gradient_ij=macro_gradient, macro_gradient_field_ijqxyz=grad)
        rhs.s.fill(0)
        u.s.fill(0)
        disc.get_rhs_mugrid(material_data_field_ijklqxyz=mat, macro_gradient_field_ijqxyz=grad, rhs_inxyz=rhs)
        solvers.conjugate_gradients_mugrid(comm=disc.communicator, fc=disc.field_collection, hessp=K_fun,
                                           b=rhs, x=u, P=base.M_fun_Green, tol=1e-8, maxiter=10000)
        C[i] = disc.get_homogenized_stress_mugrid(material_data_field_ijklqxyz=mat, displacement_field_inxyz=u,
                                                  macro_gradient_field_ijqxyz=grad)
    return C


phase_field_tmp = disc.get_scalar_field(name='phase_field_flux_only')
sensitivity_phase_field_tmp = disc.get_scalar_field(name='sensitivity_phase_field_flux_only')


def full_objective(phase_flat):
    f, g = base.objective_function_multiple_load_cases(phase_flat)
    f, g = float(f), g.reshape(disc.nb_of_pixels).copy()
    if flux_only:
        phase_field_tmp.s[0, 0] = phase_flat.reshape(disc.nb_of_pixels)
        f -= float(topology_optimization.objective_function_phase_field(
            discretization=disc, phase_field_1nxyz=phase_field_tmp, eta=base.eta,
            double_well_depth=base.double_well_depth_test))
        sensitivity_phase_field_tmp.s.fill(0)
        topology_optimization.sensitivity_phase_field_term_FE_NEW(
            discretization=disc, phase_field_1nxyz=phase_field_tmp, p=base.p, eta=base.eta,
            output_array=sensitivity_phase_field_tmp, double_well_depth=1)
        g -= sensitivity_phase_field_tmp.s[0, 0]
    return f, g


def objective(levels_int):
    return full_objective((levels_int / nb_phase_levels).ravel())


f_smooth, _ = full_objective(smooth.ravel())
print('objective:', 'weighted flux mismatch only' if flux_only else 'flux mismatch + phase-field term')

# integer level index of every pixel, rho = k / n
k_field = np.round(smooth * nb_phase_levels).astype(int)
f, g = objective(k_field)
f_rounded = f
nb_moves = nb_moves_init
history = [f]
nb_evaluations = 1

print(f'objective  smooth = {f_smooth:.6e}   rounded = {f_rounded:.6e}')
print(f'{"step":>5} {"objective":>14} {"moved":>6} {"evals":>6}')

# A one-level step (1/n) is far outside the linear range for some pixels (e.g. thin
# necks), so a pixel whose single move fails is blocked and the next candidate is tried.
blocked = np.zeros(k_field.size, dtype=bool)
progress_since_unblock = False
nb_failed_in_a_row = 0
step = 0
while step < max_steps:
    # one-level move against the sensitivity, only where it stays inside [0, n]
    direction = -np.sign(g).astype(int)
    feasible = (direction != 0) & (k_field + direction >= 0) & (k_field + direction <= nb_phase_levels)
    predicted_decrease = np.where(feasible.ravel() & ~blocked, np.abs(g).ravel(), 0.0)
    order = np.argsort(predicted_decrease)[::-1]
    nb_candidates = int(np.count_nonzero(predicted_decrease))
    if nb_candidates == 0:
        if blocked.any() and progress_since_unblock:
            # the field changed since these pixels failed: give them another chance
            blocked[:] = False
            progress_since_unblock = False
            continue
        print(f'no improving one-level move left: discrete local minimum after {step} steps')
        break

    chosen = order[:min(nb_moves, nb_candidates)]
    k_trial = k_field.copy().ravel()
    k_trial[chosen] += direction.ravel()[chosen]
    k_trial = k_trial.reshape(k_field.shape)
    f_trial, g_trial = objective(k_trial)
    nb_evaluations += 1
    if f_trial < f:
        print(f'{step:5d} {f_trial:14.6e} {len(chosen):6d} {nb_evaluations:6d}')
        k_field, f, g = k_trial, f_trial, g_trial
        history.append(f)
        nb_moves = 2 * nb_moves
        progress_since_unblock = True
        nb_failed_in_a_row = 0
        step += 1
    elif nb_moves > 1:
        nb_moves //= 2
    else:
        blocked[chosen] = True
        nb_failed_in_a_row += 1
        if nb_failed_in_a_row >= max_failed_in_a_row:
            print(f'{max_failed_in_a_row} single-pixel moves in a row rejected: stopping after {step} steps')
            break

discrete = k_field / nb_phase_levels
rounded = np.round(smooth * nb_phase_levels) / nb_phase_levels

print(f'\nobjective evaluations: {nb_evaluations}')
print(f'pixels changed vs. rounded start: {np.count_nonzero(discrete != rounded)}')
print(f'values used: {np.unique(discrete).tolist()}')

results = {}
for name, field in [('smooth', smooth), ('rounded', rounded), ('re-optimized', discrete)]:
    results[name] = homogenized_conductivity(field)

f_final = f
target = base.conductivity_C_target
print(f'\n{"field":>13} {"objective":>12} {"C11":>9} {"C22":>9} {"C12":>9} {"err C11":>8} {"err C22":>8}')
for name, f_val in [('smooth', f_smooth), ('rounded', f_rounded), ('re-optimized', f_final)]:
    C = results[name]
    print(f'{name:>13} {f_val:12.6e} {C[0, 0]:9.5f} {C[1, 1]:9.5f} {C[0, 1]:9.5f} '
          f'{100 * (C[0, 0] - target[0, 0]) / target[0, 0]:7.2f}% '
          f'{100 * (C[1, 1] - target[1, 1]) / target[1, 1]:7.2f}%')

np.save(data_folder_path + file_tag + f'_levels_{nb_phase_levels}{run_tag}_discrete.npy', discrete)
np.savez(data_folder_path + file_tag + f'_levels_{nb_phase_levels}{run_tag}_log.npz',
         history=np.array(history), objective_smooth=f_smooth, objective_rounded=f_rounded,
         objective_discrete=f_final, **{f'C_{k}': v for k, v in results.items()}, target=target)
print('saved to', data_folder_path)
