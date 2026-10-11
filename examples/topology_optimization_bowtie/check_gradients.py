"""
Finite-difference check of every term of the objective of example_2D_elasticity_TO_bowtie.py and of its gradient.

Each term f with analytic gradient g is compared with central differences
    fd(h) = (f(x + h d) - f(x - h d)) / (2 h)
along a random direction d and along single components (one nodal value). The relative error
|fd - g.d| / max(|g.d|, 1e-6 |g| |d|) should fall like h^2 until the solver noise is reached.

Terms:
  filter            F (Helmholtz): <F a, b> = <a, F b> (symmetry) and the linear map itself
  projection        H_beta,theta (theta = 0.25, 0.5, 0.75; beta = 1, 4, 16), elementwise
  filter+projection <w, H(F rho)> for a fixed weight field w (chain rule F^T [H' w])
  stiffness case k  weighted stress mismatch of one load case (xx, yy, shear) w.r.t. the physical density
  robust design j   mismatch of the eroded / intermediate / dilated design w.r.t. its physical density
  free scale        mismatch with the optimal stiffness factor t* (envelope theorem)
  full objective    plain filter, robust filter (beta 1 and 8), robust-shape (beta 8), Poisson ratio only, length scale
                    constraints (beta 8), phase field
  phase field       gradient term eta int |grad rho|^2 and double well c/eta int rho^2 (1 - rho)^2, separately

    python check_gradients.py [-n 32]
"""
import argparse
import importlib.util
import os
import sys

import numpy as np

here = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(here, '../..')))
from muFFTTO import density_filter, topology_optimization  # noqa: E402

parser = argparse.ArgumentParser(description='finite-difference check of all objective terms')
parser.add_argument('-n', type=int, default=32, help='pixels per direction (default 32)')
arguments = parser.parse_args()
n = arguments.n
STEPS = [1e-2, 1e-3, 1e-4, 1e-5]
results = []


def load_example(*options):
    """A fresh instance of the example module, configured by command-line options."""
    sys.argv = ['example', '-n', str(n), '--simp-p', '3', '--weight', '50', *options]
    spec = importlib.util.spec_from_file_location(f'example_{len(results)}_{"_".join(options)}',
                                                  os.path.join(here, 'example_2D_elasticity_TO_bowtie.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.root_print = lambda *a, **k: None
    module.cg_tol = 1e-11                       # tight solves (relative to |b|): f accurate to ~1e-11
    C_t = module.orthotropic_stiffness(0.024, 0.0072, 0.0008, -0.5)
    for k in range(module.nb_load_cases):
        module.target_stresses[k] = module.tensor_stress(module.strain_amplitude * C_t[:, k])
    return module


def check(name, fun, x, directions):
    """fun(x) -> (f, g). Prints the best relative error over the steps for each direction; records pass/fail."""
    f0, g = fun(x)
    worst = 0.0
    second_order = True
    for label, d in directions:
        derivative = float(np.dot(g, d))
        # error relative to the derivative, but at least 1e-6 |g| |d| (a component with an almost zero
        # derivative is otherwise judged by the noise of f alone)
        reference = max(abs(derivative), 1e-6 * np.linalg.norm(g) * np.linalg.norm(d))
        errors = []
        for h in STEPS:
            fd = (fun(x + h * d)[0] - fun(x - h * d)[0]) / (2 * h)
            errors.append(abs(fd - derivative) / max(reference, 1e-300))
        best = min(errors)
        worst = max(worst, best)
        if label == 'random':        # truncation error ~ h^2: a factor 100 per factor 10 in h (> 30 accepted)
            second_order = errors[1] > 0 and errors[0] / errors[1] > 30
        print(f'  {name:42s} {label:12s} g.d {derivative: .6e}   rel. error per h '
              + ' '.join(f'{e:.1e}' for e in errors), flush=True)
    results.append((name, worst, second_order))


rng = np.random.default_rng(2026)
x_grid = np.arange(n) / n
X, Y = np.meshgrid(x_grid, x_grid, indexing='ij')
design = np.clip(0.5 + 0.3 * np.sin(2 * np.pi * X) * np.cos(4 * np.pi * Y) + 0.1 * rng.standard_normal((n, n)),
                 0.05, 0.95).reshape(-1)


def directions_for(size, picks):
    d_random = rng.standard_normal(size)
    out = [('random', d_random / np.abs(d_random).max())]
    for i in picks:
        e = np.zeros(size)
        e[i] = 1.0
        out.append((f'component {i}', e))
    return out


picks = [int(rng.integers(design.size)) for _ in range(3)]

# ------------------------------------------------------------------------------------------------------------
print('1. filter and projection (library muFFTTO.density_filter)')
robust = load_example('--regularization', 'filter', '--robust', '--filter-radius', '3')
helmholtz = density_filter.HelmholtzFilter(robust.discretization, radius=3 * float(np.max(robust.pixel_size)))
a, b = rng.random(design.size), rng.random(design.size)
symmetry = abs(np.dot(helmholtz.apply(a).reshape(-1), b) - np.dot(a, helmholtz.apply(b).reshape(-1))) / abs(np.dot(a, b))
print(f'  filter symmetry |<Fa,b> - <a,Fb>| / <a,b> = {symmetry:.1e}')
results.append(('filter symmetry', symmetry, True))
w_field = rng.standard_normal(design.size)
check('filter <w, F rho>', lambda r: (np.dot(w_field, helmholtz.apply(r).reshape(-1)),
                                       helmholtz.apply(w_field).reshape(-1)), design, directions_for(design.size, picks))
for theta in (0.25, 0.5, 0.75):
    for beta in (1.0, 4.0, 16.0):
        t = rng.random(200)
        check(f'projection H(beta {beta:g}, theta {theta})',
              lambda s, b=beta, th=theta: (np.sum(density_filter.heaviside_projection(s, b, th)),
                                           density_filter.heaviside_projection_derivative(s, b, th)),
              t, [('ones', np.ones_like(t))])
for beta in (1.0, 8.0):
    plain = density_filter.FilterAndProjection(robust.discretization, radius=3 * float(np.max(robust.pixel_size)), beta=beta)

    def chain(r, p=plain):
        value = np.dot(w_field, p.forward(r).reshape(-1))
        return value, p.backward(w_field).reshape(-1)
    check(f'filter+projection <w, H(F rho)> beta {beta:g}', chain, design, directions_for(design.size, picks))

# ------------------------------------------------------------------------------------------------------------
print('2. stress mismatch per load case and per robust design (w.r.t. the physical density)')
rho_bar = np.clip(design, 0.05, 0.95)
for k, case in enumerate(robust.load_case_names):
    robust.matched_cases = [k]
    check(f'mismatch, load case {case}', lambda r: robust.mismatch_and_gradient(r, 0)[:2], rho_bar,
          directions_for(design.size, picks))
robust.matched_cases = [0, 1, 2]
robust.density.beta = 8.0
designs = robust.density.forward(design)
for j, name in enumerate(robust.design_names):
    check(f'mismatch, robust {name} design (beta 8)', lambda r, j=j: robust.mismatch_and_gradient(r, j)[:2],
          np.asarray(designs[j]).reshape(-1), directions_for(design.size, picks))

# ------------------------------------------------------------------------------------------------------------
print('3. full objectives (w.r.t. the design variable)')
for beta in (1.0, 8.0):
    robust.density.beta = beta
    check(f'robust filter objective, beta {beta:g}', robust.objective, design, directions_for(design.size, picks))
shape = load_example('--regularization', 'filter', '--robust', '--robust-shape', '--filter-radius', '3')
shape.density.beta = 8.0
check('robust-shape objective, beta 8', shape.objective, design, directions_for(design.size, picks))
length_scale = load_example('--regularization', 'filter', '--filter-radius', '3', '--length-scale', '100')
length_scale.density.beta = 8.0
check('plain filter + length scale (100), beta 8', length_scale.objective, design, directions_for(design.size, picks))
poisson = load_example('--regularization', 'filter', '--filter-radius', '3', '--objective', 'poisson', '--nu=-0.5')
for beta in (1.0, 8.0):
    poisson.density.beta = beta
    check(f'Poisson-ratio objective, beta {beta:g}', poisson.objective, design, directions_for(design.size, picks))
poisson_robust = load_example('--regularization', 'filter', '--robust', '--filter-radius', '3', '--objective', 'poisson',
                              '--nu=-0.5', '--length-scale', '1000')
poisson_robust.density.beta = 8.0
poisson_robust.cg_tol = 1e-14     # nu of the very soft eroded design: f noise ~1e-4 at 1e-11 (h^2 clean at 1e-14)
check('Poisson-ratio objective, robust + length scale, beta 8', poisson_robust.objective, design,
      directions_for(design.size, picks))
plain_example = load_example('--regularization', 'filter', '--filter-radius', '3')
for beta in (1.0, 8.0):
    plain_example.density.beta = beta
    check(f'plain filter objective, beta {beta:g}', plain_example.objective, design, directions_for(design.size, picks))
plain_example.arguments.free_scale = True
plain_example.density.beta = 4.0
check('plain filter objective, free scale, beta 4', plain_example.objective, design, directions_for(design.size, picks))
plain_example.arguments.free_scale = False
phase = load_example('--regularization', 'phase-field', '--double-well', '10', '--eta', '1')
check('phase-field objective (mismatch + regularization)', phase.objective, design, directions_for(design.size, picks))

# ------------------------------------------------------------------------------------------------------------
print('4. phase-field regularization terms separately')


def phase_field_term(r, double_well_depth, gradient_term=True):
    """eta int |grad rho|^2 + c/eta int rho^2 (1 - rho)^2 (gradient_term False: double well only)."""
    def value_and_gradient(c):
        phase.rho_nodal.s[...] = r.reshape(phase.rho_nodal.s.shape)
        value = topology_optimization.objective_function_phase_field(
            discretization=phase.discretization, phase_field_1nxyz=phase.rho_nodal, eta=phase.eta,
            double_well_depth=c)
        phase.sensitivity.s.fill(0)
        topology_optimization.sensitivity_phase_field_term_FE_NEW(
            discretization=phase.discretization, phase_field_1nxyz=phase.rho_nodal, p=phase.simp_p, eta=phase.eta,
            output_array=phase.sensitivity, double_well_depth=c)
        return value, phase.sensitivity.s[0, 0].reshape(-1).copy()
    if gradient_term:
        return value_and_gradient(double_well_depth)
    f_c, g_c = value_and_gradient(double_well_depth)
    f_0, g_0 = value_and_gradient(0.0)
    return f_c - f_0, g_c - g_0


check('phase field: gradient term eta int |grad rho|^2', lambda r: phase_field_term(r, 0.0), design,
      directions_for(design.size, picks))
check('phase field: double well c/eta int rho^2(1-rho)^2', lambda r: phase_field_term(r, 10.0, False), design,
      directions_for(design.size, picks))

# ------------------------------------------------------------------------------------------------------------
print('\nsummary (best relative error over h, worst over the directions):')
failed = 0
for name, error, second_order in results:
    # 'noise': above 1e-5 only through the noise floor of f, while the random direction converges like h^2
    status = 'ok' if error < 1e-5 else ('noise' if second_order and error < 1e-3 else 'FAIL')
    failed += status == 'FAIL'
    print(f'  {status:5s} {error:.1e}  {name}')
print(f'{len(results) - failed} of {len(results)} checks passed (tolerance 1e-5; "noise": h^2 convergence down to '
      f'the noise floor of f)')
sys.exit(1 if failed else 0)
