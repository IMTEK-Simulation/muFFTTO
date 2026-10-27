"""
Level-set topology optimization of the auxetic unit cell -- test against the phase-field formulation.

Same problem, discretization, solvers and adjoint sensitivities as example_2D_elasticity_TO.py (imported, so the
setup there -- NPIX, SOFT_EXP, CG_MODE, preconditioner argument ... -- applies unchanged); only the design
representation and update differ:

- design: level-set function phi on the nodes, solid where phi > 0. Material density by a smoothed Heaviside,
  rho = H_eps(phi) (ersatz material, interface ~2 eps wide), fed to the SIMP law of the base script with p = 1
  (LS_P), so the density sensitivity dJ/drho is the shape-derivative integrand also on the void side.
- no phase-field terms in the objective (gradient / double well): J = sum_lc w f_sigma.
- update (Allaire, Jouve, Toader 2004): Hamilton-Jacobi equation phi_t + V |grad phi| = 0 with the velocity
  V = dJ/dphi = dJ/drho H_eps'(phi), H^1-smoothed (Helmholtz filter in Fourier space, length LS_ALPHA pixels) and
  normalized to max|V| = 1. Smoothing dJ/dphi (a Sobolev gradient) keeps V a descent direction where |grad phi| ~ 1;
  smoothing dJ/drho alone does not (dJ/drho jumps across the interface band, the steps stalled at J ~ 1.5);
  Godunov upwind scheme, CFL 0.5. The pseudo-time per iteration (LS_STEP, in pixels the interface may move) is
  increased by 1.2 on an accepted step and halved on a rejected one (J did not decrease).
- reinitialization to a signed distance (Sussman's equation, LS_REINIT pseudo-time steps) only of the start, by
  default. LS_REINIT_EVERY > 0 also reinitializes every LS_REINIT_EVERY accepted steps and at each continuation
  stage, outside the step acceptance -- off: the discrete scheme moves the interface by up to a pixel (20-50
  pixels flipped on the thin slits at 64^2), J jumped 5-10x every time, while |grad phi| at the interface stayed ~1
  without it.
- LS_SHARP=1: two-phase geometry. The mechanics sees rho = 1 where phi > 0 and 0 elsewhere (each pixel solid or
  void, no intermediate stiffness); H_eps'(phi) is kept only to build the velocity, which is then no longer the exact
  gradient of the sharp J -- the step acceptance (J decreased) checks descent. Started from a perforated plate it
  stalls within a few iterations (J changes only when the interface crosses a pixel).
- LS_EPS_CONT (e.g. '2,1,0.5,0'): continuation in the Heaviside half-width, in pixels, 0 = sharp (as LS_SHARP=1,
  the velocity then uses the smallest nonzero eps). Each stage runs until its stop criterion (no decrease at the
  minimal step, or the ftol criterion below) or LS_MAXITER iterations; the next stage starts with
  the initial step and a fresh ftol window. Overrides LS_EPS and LS_SHARP.
- start (LS_INIT): 'holes' -- perforated plate, LS_HOLES^2 holes with slightly randomized centres and radii (the HJ
  update moves boundaries but does not nucleate holes in 2D); 'phase' -- the start of the phase-field runs
  (sin pattern + uniform noise, seed 0, serial, no INIT_NOISE_GRID), phi = rho_0 - 1/2, then reinitialized;
  'pf_final' -- a converged phase-field design (LS_INIT_FILE, a _final.nc of example_2D_elasticity_TO.py on the same
  grid), thresholded at rho = 1/2: phi = rho - 1/2, then reinitialized.

Serial only (the upwind differences use periodic np.roll on the full grid). Example:
    NPIX=64 python example_2D_elasticity_TO_level_set.py Green_Jacobi
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault('SHOW_PLOTS', '0')

import numpy as np
import time
from mpi4py import MPI

import example_2D_elasticity_TO as base
from muFFTTO import io_utils
from muFFTTO import material_models
from muFFTTO import solvers

if MPI.COMM_WORLD.size > 1:
    raise RuntimeError('the level-set driver is serial only')

discretization = base.discretization
xp, to_host = base.xp, base.to_host
n_pix = base.n_pix
h = base.pixel_size  # [hx, hy]

# Level-set parameters (environment overrides)
base.phase_field_regularization = False
base.p = int(os.environ.get('LS_P', 1))  # SIMP exponent of the ersatz material
sharp = os.environ.get('LS_SHARP', '0') == '1'  # two-phase: rho = 1 (phi > 0) / 0, see above
eps = float(os.environ.get('LS_EPS', 2.0)) * min(h)  # half-width of the smoothed Heaviside
eps_continuation = [float(e) for e in os.environ['LS_EPS_CONT'].split(',')] if 'LS_EPS_CONT' in os.environ \
    else [0.0 if sharp else eps / min(h)]  # stages, in pixels; 0 = sharp
eps_band = min(e for e in eps_continuation if e > 0) * min(h) if any(e > 0 for e in eps_continuation) else eps
alpha = float(os.environ.get('LS_ALPHA', 2.0)) * min(h)  # H^1 smoothing length of the velocity
init = os.environ.get('LS_INIT', 'holes')  # 'holes', 'phase' or 'pf_final' (see above)
init_file = os.environ.get('LS_INIT_FILE')  # init = 'pf_final': the phase-field result to start from
if init == 'pf_final' and init_file is None:
    raise ValueError('LS_INIT=pf_final needs LS_INIT_FILE (a _final.nc of example_2D_elasticity_TO.py)')
nb_holes = int(os.environ.get('LS_HOLES', 4))  # start: nb_holes x nb_holes holes
hole_radius = float(os.environ.get('LS_RADIUS', 0.3))  # hole radius / hole spacing
step = float(os.environ.get('LS_STEP', 2.0))  # initial pseudo-time per iteration, in pixels the interface may move
step_min, step_max = 0.05, 8.0
nb_reinit = int(os.environ.get('LS_REINIT', 20))
reinit_every = int(os.environ.get('LS_REINIT_EVERY', 0))  # 0: never (only the start)
maxiter = int(os.environ.get('LS_MAXITER', 200))  # per continuation stage
ftol = 1e-5  # stop: relative decrease of the best J over the last 20 accepted iterations below ftol
cfl = 0.5


def heaviside(phi):
    """Smoothed Heaviside H_eps(phi) (C^1, exactly 0 / 1 for |phi| >= eps)."""
    t = np.clip(phi / eps, -1, 1)
    return 0.5 * (1 + t + np.sin(np.pi * t) / np.pi)


def density(phi):
    """Material density of the design: H_eps(phi), or the sharp characteristic function of phi > 0."""
    return (phi > 0).astype(float) if sharp else heaviside(phi)


def heaviside_derivative(phi):
    """dH_eps/dphi."""
    return np.where(np.abs(phi) < eps, 0.5 / eps * (1 + np.cos(np.pi * phi / eps)), 0.0)


def upwind_gradient_norms(phi):
    """Godunov |grad phi| for a positive (first) and a negative (second) speed, periodic."""
    plus_sq, minus_sq = 0.0, 0.0
    for axis in range(phi.ndim):
        d_minus = (phi - np.roll(phi, 1, axis=axis)) / h[axis]
        d_plus = (np.roll(phi, -1, axis=axis) - phi) / h[axis]
        plus_sq = plus_sq + np.maximum(d_minus, 0) ** 2 + np.minimum(d_plus, 0) ** 2
        minus_sq = minus_sq + np.minimum(d_minus, 0) ** 2 + np.maximum(d_plus, 0) ** 2
    return np.sqrt(plus_sq), np.sqrt(minus_sq)


def hamilton_jacobi(phi, velocity, pseudo_time):
    """Advance phi_t + V |grad phi| = 0 by pseudo_time (|V| <= 1, times in units of min(h))."""
    nb_steps = max(1, int(np.ceil(pseudo_time / cfl)))
    dt = pseudo_time * min(h) / nb_steps
    for _ in range(nb_steps):
        grad_plus, grad_minus = upwind_gradient_norms(phi)
        phi = phi - dt * (np.maximum(velocity, 0) * grad_plus + np.minimum(velocity, 0) * grad_minus)
    return phi


def reinitialize(phi):
    """Towards a signed distance: phi_t + S(phi_0) (|grad phi| - 1) = 0 (Sussman, Smereka, Osher 1994)."""
    sign = phi / np.sqrt(phi ** 2 + min(h) ** 2)
    dt = cfl * min(h)
    for _ in range(nb_reinit):
        grad_plus, grad_minus = upwind_gradient_norms(phi)
        phi = phi - dt * (np.maximum(sign, 0) * (grad_plus - 1) + np.minimum(sign, 0) * (grad_minus - 1))
    return phi


wave_numbers = np.meshgrid(*[2 * np.pi * np.fft.fftfreq(n, d=hi) for n, hi in zip(base.number_of_pixels, h)],
                           indexing='ij')
helmholtz_filter = 1.0 / (1.0 + alpha ** 2 * sum(k ** 2 for k in wave_numbers))


def velocity_from_sensitivity(dJ_dphi):
    """H^1-smoothed, normalized descent velocity: V = (1 - alpha^2 Laplace)^-1 dJ/dphi / max|.|."""
    v = np.real(np.fft.ifftn(helmholtz_filter * np.fft.fftn(dJ_dphi)))
    return v / np.max(np.abs(v))


def initial_level_set():
    """Signed distance of a perforated plate: holes on a nb_holes^2 lattice, perturbed (seed 0); or, for
    init = 'phase', rho_0 - 1/2 of the phase-field start (as in example_2D_elasticity_TO.py)."""
    coords = to_host(discretization.fft.coords)
    if init == 'phase':
        np.random.seed(0)
        rho_0 = (np.sin(coords[0] * 4 * np.pi) + np.sin(coords[1] * 4 * np.pi) + 2) / 4
        rho_0 = rho_0 + 0.5 * np.random.rand(1, 1, *base.number_of_pixels)[0, 0]
        return (rho_0 - 0.5) * min(h)  # scaled to roughly a distance (|grad phi| ~ 1) before reinitialization
    if init == 'pf_final':
        import netCDF4
        with netCDF4.Dataset(init_file) as file:
            rho_final = np.asarray(file['phase_field_solution'][:]).reshape(base.number_of_pixels)
        return (rho_final - 0.5) * min(h)
    rng = np.random.default_rng(0)
    spacing = 1.0 / nb_holes
    phi = np.full(base.number_of_pixels, np.inf)
    for i in range(nb_holes):
        for j in range(nb_holes):
            centre = (np.array([i, j]) + 0.5 + 0.15 * rng.uniform(-1, 1, 2)) * spacing
            radius = hole_radius * spacing * (1 + 0.2 * rng.uniform(-1, 1))
            dist_sq = sum((np.mod(coords[a] - centre[a] + 0.5, 1.0) - 0.5) ** 2 for a in range(2))
            phi = np.minimum(phi, np.sqrt(dist_sq) - radius)
    return phi


if __name__ == '__main__':
    start_time = time.time()
    script_name = os.path.splitext(os.path.basename(__file__))[0]
    file_folder_path = os.path.dirname(os.path.realpath(__file__))
    data_folder_path = os.path.join(file_folder_path, 'data', script_name) + '/'
    figure_folder_path = os.path.join(file_folder_path, 'figures', script_name) + '/'
    os.makedirs(data_folder_path, exist_ok=True)
    os.makedirs(figure_folder_path, exist_ok=True)

    output_name = data_folder_path + (f'N{n_pix}_soft{base.soft_phase:g}_{base.cg_mode}_{base.preconditioner_type}'
                                      f'_w_{base.weight}_p{base.p}_eps{"-".join(f"{e:g}" for e in eps_continuation)}'
                                      f'_alpha{alpha / min(h):g}'
                                      + (f'_holes{nb_holes}' if init == 'holes' else f'_init{init}')
                                      + ('_sharp' if sharp and 'LS_EPS_CONT' not in os.environ else '')) \
        + os.environ.get('RUN_SUFFIX', '')
    parameters = {'nb_of_pixels': base.number_of_pixels, 'domain_size': base.domain_size,
                  'preconditioner_type': base.preconditioner_type, 'weight': base.weight, 'p': base.p,
                  'soft_phase': base.soft_phase, 'cg_mode': base.cg_mode, 'method': 'level_set_hamilton_jacobi',
                  'ls_eps_continuation': eps_continuation, 'ls_alpha': alpha, 'ls_init': init, 'ls_holes': nb_holes,
                  'ls_radius': hole_radius,
                  'target_C_voigt': material_models.compute_Voigt_notation_4order(base.elastic_C_target)}

    density_iterate = discretization.get_scalar_field(name='phase_field_iterate')  # rho = H(phi), as in the PF runs
    level_set_iterate = discretization.get_scalar_field(name='level_set_iterate')
    history_writer = io_utils.FieldWriter(output_name + '_history.nc', [density_iterate, level_set_iterate],
                                          attributes=parameters,
                                          frame_variables={'objective': (), 'volume_fraction': (),
                                                           'nb_evaluations': (), 'step': (), 'eps': ()})

    def set_stage(eps_pixels):
        """Heaviside of the continuation stage: smoothed with half-width eps_pixels, or sharp for 0."""
        global eps, sharp
        sharp = eps_pixels == 0
        eps = eps_band if sharp else eps_pixels * min(h)


    set_stage(eps_continuation[0])
    phi = reinitialize(initial_level_set())
    f, dJ_drho = base.objective_function_multiple_load_cases(density(phi).ravel())
    nb_evaluations = 1
    objective_history = [f]
    stage_starts = [0]  # index into objective_history where each stage starts


    def write_frame():
        density_iterate.s[0, 0] = xp.asarray(density(phi))
        level_set_iterate.s[0, 0] = xp.asarray(phi)
        history_writer.write(objective=f, volume_fraction=float(np.mean(density(phi))),
                             nb_evaluations=nb_evaluations, step=step, eps=0.0 if sharp else eps / min(h))


    write_frame()
    print(f'LS it 0: J = {f:.6e}, volume = {np.mean(density(phi)):.3f}')
    iteration = 0
    for stage, eps_pixels in enumerate(eps_continuation):
        if stage > 0:
            set_stage(eps_pixels)
            if reinit_every:
                phi = reinitialize(phi)
            f, dJ_drho = base.objective_function_multiple_load_cases(density(phi).ravel())
            nb_evaluations += 1
            step = float(os.environ.get('LS_STEP', 2.0))
            objective_history.append(f)
            stage_starts.append(len(objective_history) - 1)
            write_frame()
        print(f'LS stage {stage}: eps = {eps_pixels:g} px{" (sharp)" if sharp else ""}, J = {f:.6e}')
        for stage_iteration in range(1, maxiter + 1):
            iteration += 1
            # V = dJ/dphi: phi decreases (material removed) where adding material would increase J
            velocity = velocity_from_sensitivity(dJ_drho.reshape(base.number_of_pixels) * heaviside_derivative(phi))
            while True:
                phi_trial = hamilton_jacobi(phi, velocity, step)
                f_trial, dJ_drho_trial = base.objective_function_multiple_load_cases(density(phi_trial).ravel())
                nb_evaluations += 1
                if f_trial < f or step <= step_min:
                    break
                step = max(step / 2, step_min)
                print(f'   rejected: J = {f_trial:.6e} >= {f:.6e}, step -> {step:g}')
            if f_trial >= f:
                print(f'LS stop: no decrease at the minimal step {step_min}')
                break
            phi, f, dJ_drho = phi_trial, f_trial, dJ_drho_trial
            step = min(1.2 * step, step_max)
            if reinit_every and stage_iteration % reinit_every == 0:
                phi = reinitialize(phi)
                f, dJ_drho = base.objective_function_multiple_load_cases(density(phi).ravel())
                nb_evaluations += 1
            objective_history.append(f)
            write_frame()
            print(f'LS it {iteration}: J = {f:.6e}, volume = {np.mean(density(phi)):.3f}, step = {step:.3g}, '
                  f'evaluations = {nb_evaluations}')
            stage_history = objective_history[stage_starts[-1]:]
            best_before = min(stage_history[:-20]) if len(stage_history) > 20 else np.inf
            if best_before - min(stage_history) < ftol * abs(best_before):
                print(f'LS stop: relative decrease of the best J over 20 iterations < {ftol}')
                break
    history_writer.close()

    # Homogenized tangent of the final ersatz design (the material of the last accepted evaluation is re-set)
    base.objective_function_multiple_load_cases(density(phi).ravel())
    material = base.material_data_field_C_0_rho_ijklqxyz
    dim = base.dim
    homogenized_C_ijkl = np.zeros(4 * [dim])
    macro_gradient_field_ijqxyz = discretization.get_gradient_size_field(name='macro_gradient_field_ls_final')
    rhs_field = discretization.get_unknown_size_field(name='rhs_field_ls_final')
    displacement_field = discretization.get_unknown_size_field(name='displacement_field_ls_final')


    def K_fun_final(x, Ax):
        discretization.apply_system_matrix_mugrid(material_data_field=material, input_field_inxyz=x,
                                                  output_field_inxyz=Ax, formulation='small_strain')


    for i in range(dim):
        for j in range(dim):
            macro_gradient_ij = np.zeros([dim, dim])
            macro_gradient_ij[i, j] = 1
            discretization.get_macro_gradient_field_mugrid(macro_gradient_ij=macro_gradient_ij,
                                                           macro_gradient_field_ijqxyz=macro_gradient_field_ijqxyz)
            discretization.get_rhs_mugrid(material_data_field_ijklqxyz=material,
                                          macro_gradient_field_ijqxyz=macro_gradient_field_ijqxyz,
                                          rhs_inxyz=rhs_field)
            solvers.conjugate_gradients_mugrid(comm=discretization.communicator, fc=discretization.field_collection,
                                               hessp=K_fun_final, b=rhs_field, x=displacement_field,
                                               P=base.M_fun_Green, tol=1e-5, maxiter=10000)
            homogenized_C_ijkl[:, :, i, j] = discretization.get_homogenized_stress_mugrid(
                material_data_field_ijklqxyz=material, displacement_field_inxyz=displacement_field,
                macro_gradient_field_ijqxyz=macro_gradient_field_ijqxyz, formulation='small_strain')
    C_voigt = material_models.compute_Voigt_notation_4order(homogenized_C_ijkl)
    formatter = {'float_kind': lambda x: f'{x:0.5f}'}
    print('Optimized elastic tangent:\n' + np.array2string(C_voigt, formatter=formatter))
    print('Target elastic tangent:\n' + np.array2string(parameters['target_C_voigt'], formatter=formatter))

    solution_phase = discretization.get_scalar_field(name='phase_field_solution')
    solution_phase.s[0, 0] = xp.asarray(density(phi))
    level_set_solution = discretization.get_scalar_field(name='level_set_solution')
    level_set_solution.s[0, 0] = xp.asarray(phi)
    io_utils.save_fields(output_name + '_final.nc', [solution_phase, level_set_solution], attributes={
        **parameters,
        'nb_iterations': len(objective_history) - 1,
        'stage_starts': stage_starts,
        'nb_evaluations': nb_evaluations,
        'elapsed_time': time.time() - start_time,
        'objective_history': objective_history,
        'homogenized_C_voigt': C_voigt,
        'num_iteration_mech': base.info_mech['num_iteration_adjoint'],
        'num_iteration_adjoint': base.info_adjoint['num_iteration_adjoint']})

    # Figure: start and final design on their exact pixels (2x2 periodic tiling), objective history
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), dpi=150)
    for ax, field, title in [(axes[0], density(reinitialize(initial_level_set())), 'start'),
                             (axes[1], density(phi), f'final ({len(objective_history) - 1} it.)')]:
        ax.imshow(np.tile(field, (2, 2)).T, origin='lower', cmap='Greys', vmin=0, vmax=1,
                  interpolation='nearest', extent=[0, 2, 0, 2])
        ax.set_title(title)
        ax.set_xticks([]), ax.set_yticks([])
    axes[2].semilogy(objective_history, '-o', ms=3)
    for stage_start, eps_pixels in zip(stage_starts[1:], eps_continuation[1:]):
        axes[2].axvline(stage_start, color='0.6', ls=':')
        axes[2].annotate(f'eps {eps_pixels:g}', (stage_start, 1), xycoords=('data', 'axes fraction'),
                         xytext=(2, -10), textcoords='offset points', fontsize=7)
    axes[2].set_xlabel('accepted iteration (stage switches dotted)')
    axes[2].set_ylabel('objective J')
    axes[2].grid(True, alpha=0.3)
    fig.tight_layout()
    figure_name = figure_folder_path + os.path.basename(output_name) + '.png'
    fig.savefig(figure_name, bbox_inches='tight')
    print(f'figure: {figure_name}')
