"""
Reaction-diffusion level-set topology optimization of the auxetic unit cell (Yamada, Izui, Nishiwaki, Takezawa 2010):
sharp two-phase geometry as the level set, nucleation and removal of features as the phase field.

Same problem, discretization, solvers and adjoint sensitivities as example_2D_elasticity_TO.py (imported, so NPIX,
SOFT_EXP, CG_MODE, the preconditioner argument ... apply unchanged):

- design: level set phi on the nodes, bounded to [-1, 1]; solid where phi > 0. RD_GEOMETRY='quad' (default): phi is
  interpolated to the quadrature points (2 per pixel, linear triangles) and thresholded there, so every quadrature point
  is exactly solid or void (C_0 or soft C_0) -- a true two-phase material; 'node': rho = [phi > 0] on the nodes,
  interpolated (values 0, 1/3, 2/3, 1 at the quadrature points along the interface).
- update, everywhere in the cell (not only at the interface as the Hamilton-Jacobi level set):
      d phi / dt = -g + tau Laplace phi,   g = dJ/drho / rms(dJ/drho)   (SIMP p = 1)
  semi-implicit, diagonal in Fourier space: phi <- clip((phi - c g)^ / (1 + c tau k^2), -1, 1); the diffusion keeps
  phi smooth (tau = RD_LENGTH^2 pixels^2 sets the smallest feature), the clip keeps it bounded (no reinitialization).
  Schedules (RD_ACCEPT='all'): RD_ANNEAL=K > 0 lowers the step as c_k = RD_STEP / (1 + k / K); RD_LENGTH_END < RD_LENGTH
  shrinks the diffusion length geometrically from RD_LENGTH to RD_LENGTH_END over RD_MAXITER iterations (coarse
  features first, finer ones later).
  Where g < 0 (material would decrease J) phi rises and solid appears, also inside the void and vice versa.
- step control, RD_ACCEPT: 'all' (default, as Yamada et al.) -- fixed step c = RD_STEP, every step accepted, the
  best design is kept (J of a sharp design jumps when quadrature points change phase; requiring a decrease stalled
  at J ~ 1.5 from the phase-field start, the first-order g mispredicts a void -> solid flip at contrast 1e5);
  'monotone' -- c grows by 1.2 on an accepted step and is halved on a rejected one (J increased). In both,
  a step that does not change the geometry (no quadrature point changes phase) is accepted without solving (J and g
  unchanged), so phi keeps drifting until the next feature changes.
- stop: ('monotone') c < RD_STEP_MIN, best J not decreased by a relative ftol over RD_PATIENCE accepted steps,
  or RD_MAXITER steps.
- start (RD_INIT): 'phase' -- the phase-field start (sin pattern + uniform noise, seed 0), phi = clip(2 rho_0 - 1);
  'solid' -- nearly full material, phi = 0.25 + 0.3 noise (seed 0): ~8 % isolated void pinholes (a uniform solid
  has a zero load right-hand side and a uniform g, nothing to start from), every feature has to be nucleated; 'holes' -- the
  perforated plate of the Hamilton-Jacobi level set, phi = clip(distance / 4 pixels); 'file' -- the level set of a
  _final.nc of this script (RD_INIT_FILE, same grid), e.g. to polish a design with RD_ACCEPT=monotone.

Serial only. Example:
    NPIX=64 CG_MODE=energy python example_2D_elasticity_TO_reaction_diffusion.py Green
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
    raise RuntimeError('the reaction-diffusion driver is serial only')

discretization = base.discretization
xp, to_host = base.xp, base.to_host
n_pix = base.n_pix
h = base.pixel_size

base.phase_field_regularization = False
base.p = 1  # the sensitivity is then independent of rho (and of the quadrature threshold below)
geometry = os.environ.get('RD_GEOMETRY', 'quad')  # 'quad' or 'node' (see above)
rd_length = float(os.environ.get('RD_LENGTH', 2.0))  # diffusion length sqrt(tau), in pixels
rd_length_end = float(os.environ.get('RD_LENGTH_END', rd_length))
anneal = float(os.environ.get('RD_ANNEAL', 0))  # 0: constant step
step = float(os.environ.get('RD_STEP', 0.1))  # initial c
step_min, step_max = float(os.environ.get('RD_STEP_MIN', 1e-3)), float(os.environ.get('RD_STEP_MAX', 1.0))
init = os.environ.get('RD_INIT', 'phase')
accept = os.environ.get('RD_ACCEPT', 'all')  # 'all' or 'monotone' (see above)
init_file = os.environ.get('RD_INIT_FILE')  # init = 'file'
if init == 'file' and init_file is None:
    raise ValueError('RD_INIT=file needs RD_INIT_FILE (a _final.nc of this script)')
maxiter = int(os.environ.get('RD_MAXITER', 300))
patience = int(os.environ.get('RD_PATIENCE', 50))
ftol = 1e-4

wave_numbers = np.meshgrid(*[2 * np.pi * np.fft.fftfreq(n, d=hi) for n, hi in zip(base.number_of_pixels, h)],
                           indexing='ij')
laplace_symbol = sum(k ** 2 for k in wave_numbers)  # -Laplace in Fourier space


def quad_values(phi):
    """phi at the 2 quadrature points (centroids) of each pixel: lower-left triangle (nodes (0,0), (1,0), (0,1)) and
    upper-right triangle ((1,0), (0,1), (1,1)), as discretization.apply_N_operator_mugrid; shape [2, nx, ny]."""
    phi_10, phi_01 = np.roll(phi, -1, 0), np.roll(phi, -1, 1)
    return np.stack([(phi + phi_10 + phi_01) / 3, (phi_10 + phi_01 + np.roll(phi_10, -1, 1)) / 3])


if geometry == 'quad':
    base.quad_point_density = lambda phi_q: (phi_q > 0).astype(phi_q.dtype)


def design_input(phi):
    """The nodal field handed to the objective: phi itself (thresholded at the quadrature points by the base script)
    or the nodal characteristic function."""
    return phi if geometry == 'quad' else (phi > 0).astype(float)


def phase_pattern(phi):
    """What decides the geometry the mechanics sees (to skip solves of unchanged designs)."""
    return quad_values(phi) > 0 if geometry == 'quad' else phi > 0


def solid_fraction(phi):
    return float(np.mean(phase_pattern(phi)))


def reaction_diffusion_step(phi, g, c, length):
    tau = (length * min(h)) ** 2
    phi_hat = (np.fft.fftn(phi) - c * np.fft.fftn(g)) / (1 + c * tau * laplace_symbol)
    return np.clip(np.real(np.fft.ifftn(phi_hat)), -1, 1)


def initial_level_set():
    coords = to_host(discretization.fft.coords)
    if init == 'phase':
        np.random.seed(0)
        rho_0 = (np.sin(coords[0] * 4 * np.pi) + np.sin(coords[1] * 4 * np.pi) + 2) / 4
        rho_0 = rho_0 + 0.5 * np.random.rand(1, 1, *base.number_of_pixels)[0, 0]
        return np.clip(2 * rho_0 - 1, -1, 1)
    if init == 'file':
        import netCDF4
        with netCDF4.Dataset(init_file) as file:
            return np.asarray(file['level_set_solution'][:]).reshape(base.number_of_pixels)
    if init == 'solid':
        return 0.25 + 0.3 * np.random.default_rng(0).uniform(-1, 1, base.number_of_pixels)
    rng = np.random.default_rng(0)  # 'holes', as the Hamilton-Jacobi level set (LS_HOLES=4, LS_RADIUS=0.3)
    spacing = 1.0 / 4
    distance = np.full(base.number_of_pixels, np.inf)
    for i in range(4):
        for j in range(4):
            centre = (np.array([i, j]) + 0.5 + 0.15 * rng.uniform(-1, 1, 2)) * spacing
            radius = 0.3 * spacing * (1 + 0.2 * rng.uniform(-1, 1))
            dist_sq = sum((np.mod(coords[a] - centre[a] + 0.5, 1.0) - 0.5) ** 2 for a in range(2))
            distance = np.minimum(distance, np.sqrt(dist_sq) - radius)
    return np.clip(distance / (4 * min(h)), -1, 1)


def evaluate(phi):
    f, dJ_drho = base.objective_function_multiple_load_cases(design_input(phi).ravel())
    g = dJ_drho.reshape(base.number_of_pixels)
    return f, g / np.sqrt(np.mean(g ** 2))


if __name__ == '__main__':
    start_time = time.time()
    script_name = os.path.splitext(os.path.basename(__file__))[0]
    file_folder_path = os.path.dirname(os.path.realpath(__file__))
    data_folder_path = os.path.join(file_folder_path, 'data', script_name) + '/'
    figure_folder_path = os.path.join(file_folder_path, 'figures', script_name) + '/'
    os.makedirs(data_folder_path, exist_ok=True)
    os.makedirs(figure_folder_path, exist_ok=True)

    output_name = data_folder_path + (f'N{n_pix}_soft{base.soft_phase:g}_{base.cg_mode}_{base.preconditioner_type}'
                                      f'_w_{base.weight}_{geometry}_L{rd_length:g}'
                                      + (f'-{rd_length_end:g}' if rd_length_end != rd_length else '')
                                      + f'_c{step:g}' + (f'_anneal{anneal:g}' if anneal else '') + f'_{accept}_init{init}') \
        + os.environ.get('RUN_SUFFIX', '')
    parameters = {'nb_of_pixels': base.number_of_pixels, 'domain_size': base.domain_size,
                  'preconditioner_type': base.preconditioner_type, 'weight': base.weight, 'p': base.p,
                  'soft_phase': base.soft_phase, 'cg_mode': base.cg_mode, 'method': 'reaction_diffusion_level_set',
                  'rd_geometry': geometry, 'rd_length': rd_length, 'rd_length_end': rd_length_end,
                  'rd_anneal': anneal, 'rd_step': step, 'rd_accept': accept, 'rd_init': init,
                  'target_C_voigt': material_models.compute_Voigt_notation_4order(base.elastic_C_target)}

    density_iterate = discretization.get_scalar_field(name='phase_field_iterate')  # nodal [phi > 0], for plots
    level_set_iterate = discretization.get_scalar_field(name='level_set_iterate')
    history_writer = io_utils.FieldWriter(output_name + '_history.nc', [density_iterate, level_set_iterate],
                                          attributes=parameters,
                                          frame_variables={'objective': (), 'volume_fraction': (),
                                                           'nb_evaluations': (), 'step': (), 'eps': ()})

    phi = initial_level_set()
    f, g = evaluate(phi)
    nb_evaluations, nb_skipped = 1, 0
    objective_history = [f]
    best = (f, phi.copy())


    def write_frame():
        density_iterate.s[0, 0] = xp.asarray((phi > 0).astype(float))
        level_set_iterate.s[0, 0] = xp.asarray(phi)
        history_writer.write(objective=f, volume_fraction=solid_fraction(phi), nb_evaluations=nb_evaluations,
                             step=step, eps=0.0)


    write_frame()
    print(f'RD it 0: J = {f:.6e}, solid = {solid_fraction(phi):.3f}')
    step_0 = step
    for iteration in range(1, maxiter + 1):
        length = rd_length * (rd_length_end / rd_length) ** ((iteration - 1) / max(maxiter - 1, 1))
        if accept == 'all' and anneal:
            step = step_0 / (1 + (iteration - 1) / anneal)
        while True:
            phi_trial = reaction_diffusion_step(phi, g, step, length)
            if np.array_equal(phase_pattern(phi_trial), phase_pattern(phi)):  # same geometry: same J and g
                f_trial, g_trial = f, g
                nb_skipped += 1
                break
            f_trial, g_trial = evaluate(phi_trial)
            nb_evaluations += 1
            if accept == 'all' or f_trial <= f or step <= step_min:
                break
            step = max(step / 2, step_min)
            print(f'   rejected: J = {f_trial:.6e} > {f:.6e}, step -> {step:g}')
        if accept == 'monotone' and f_trial > f:
            print(f'RD stop: no decrease at the minimal step {step_min}')
            break
        phi, f, g = phi_trial, f_trial, g_trial
        if accept == 'monotone':
            step = min(1.2 * step, step_max)
        objective_history.append(f)
        if f < best[0]:
            best = (f, phi.copy())
        write_frame()
        print(f'RD it {iteration}: J = {f:.6e}, solid = {solid_fraction(phi):.3f}, step = {step:.3g}, '
              f'evaluations = {nb_evaluations}, skipped = {nb_skipped}')
        best_before = min(objective_history[:-patience]) if len(objective_history) > patience else np.inf
        if best_before - min(objective_history) < ftol * abs(best_before):
            print(f'RD stop: relative decrease of the best J over {patience} iterations < {ftol}')
            break
    history_writer.close()
    f, phi = best

    # Homogenized tangent of the best design (its material is re-set by one more evaluation)
    evaluate(phi)
    material = base.material_data_field_C_0_rho_ijklqxyz
    dim = base.dim
    homogenized_C_ijkl = np.zeros(4 * [dim])
    macro_gradient_field_ijqxyz = discretization.get_gradient_size_field(name='macro_gradient_field_rd_final')
    rhs_field = discretization.get_unknown_size_field(name='rhs_field_rd_final')
    displacement_field = discretization.get_unknown_size_field(name='displacement_field_rd_final')


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
    solution_phase.s[0, 0] = xp.asarray((phi > 0).astype(float))
    level_set_solution = discretization.get_scalar_field(name='level_set_solution')
    level_set_solution.s[0, 0] = xp.asarray(phi)
    io_utils.save_fields(output_name + '_final.nc', [solution_phase, level_set_solution], attributes={
        **parameters,
        'nb_iterations': len(objective_history) - 1,
        'nb_evaluations': nb_evaluations,
        'nb_skipped_solves': nb_skipped,
        'elapsed_time': time.time() - start_time,
        'objective_history': objective_history,
        'homogenized_C_voigt': C_voigt,
        'num_iteration_mech': base.info_mech['num_iteration_adjoint'],
        'num_iteration_adjoint': base.info_adjoint['num_iteration_adjoint']})
    print(f'elapsed: {time.time() - start_time:.1f} s, evaluations: {nb_evaluations}, skipped: {nb_skipped}')
