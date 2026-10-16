"""
Phase-field fracture (AT2) of a periodic porous cell under macroscopic tension
==============================================================================

Standard second-order phase-field model of brittle fracture
(Bourdin, Francfort & Marigo 2000; Miehe, Hofacker & Welschinger 2010):

    Pi(u, d) = int_Omega  g(d) psi^+(eps) + psi^-(eps)  dx
             + Gc/2 int_Omega  d^2/l + l |grad d|^2  dx         (AT2 crack density)

    g(d) = (1 - d)^2 + k           degradation function, k << 1 keeps K regular
    eps  = E + sym(grad u~)        E: macroscopic strain, u~: periodic fluctuation

Driving force (Amor, Marigo & Maurini 2009, volumetric-deviatoric split):

    psi^+ = K/2 <tr eps>_+^2 + mu eps_dev : eps_dev       K = lam + 2 mu / dim
    psi_0 = lam/2 (tr eps)^2 + mu eps : eps               (= psi^+ + psi^-)

Irreversibility with the history field of Miehe et al. (2010),

    H(x, t) = max_{s <= t} psi^+(eps(x, s)),

and the 'hybrid' formulation of Ambati, Gerasimov & De Lorenzis (2015): the
elastic problem uses the isotropic degradation C(d) = g(d) C_0, so it stays
linear and symmetric (the usual PCG with the Green preconditioner), while only
the tensile energy H drives the crack.

Staggered (alternate minimisation) scheme, per load step E_n = t_n E_dir:

    1. elasticity:   B^T W g(d) C_0 B u~ = -B^T W g(d) C_0 E
    2. history:      H = max(H_{n-1}, psi^+(eps))
    3. damage:       [Gc l B^T W B + N^T W (Gc/l + 2 H) N] d = N^T W 2H
    repeat 1-3 until max|d - d_old| < tol, then H_{n-1} <- H.

The damage equation is the stationarity condition of Pi w.r.t. d with
psi^+ replaced by H; it is linear in d and SPD, solved with CG and an FFT
(Green) preconditioner built from the homogeneous operator H = 0.
The elastic problem lives on an 'elasticity' discretization, the damage on a
'conductivity' (scalar) discretization of the same grid; both have the same
MPI decomposition, so quadrature/nodal arrays are copied with .s[...].

Output: homogenized stress-strain curve sigma_xx(E_xx) and damage snapshots.
Set ``live_plot`` to 'step' or 'iteration' to watch the damage evolve.
"""
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

import numpy as np
import time
from mpi4py import MPI

from muGrid import Solvers

from muFFTTO import domain
from muFFTTO import microstructure_library
from muFFTTO import material_models
from muFFTTO import io_utils
from muFFTTO import solvers
from muFFTTO import topology_optimization   # energy_stop_kwargs

# ----------------------------------------------------------------------------
# Parameters
# ----------------------------------------------------------------------------
domain_size = [1, 1]
number_of_pixels = (int(os.environ.get('NPIX', 64)),) * 2
element_type = 'bilinear_rectangle'

E_young, poisson = 1.0, 0.2        # matrix
pore_contrast = 1e-3               # stiffness of the pore relative to the matrix
Gc = 5e-4                          # fracture toughness (critical energy release rate)
# LENGTH_SCALE (env): fixed in physical units, the same crack problem on every grid; default l = 2 pixels
length_scale = float(os.environ['LENGTH_SCALE']) if 'LENGTH_SCALE' in os.environ \
    else 2.0 * domain_size[0] / number_of_pixels[0]
run_tag = f'_l{length_scale:g}' if 'LENGTH_SCALE' in os.environ else ''
k_residual = 1e-4                  # residual stiffness in g(d)

macro_strain_direction = np.array([[1.0, 0.0],
                                   [0.0, 0.0]])            # uniaxial strain E_xx
load_steps = np.linspace(0.0, 0.08, 81)[1:]
staggered_tol = 1e-3               # max |d - d_old| for the staggered loop
max_staggered_iterations = 500
cg_rtol = 1e-8
# 'Green': M = G (FFT inverse of the homogeneous reference operator)
# 'Green_Jacobi': M = D^-1/2 G D^-1/2 with D = diag(K), recomputed when K changes
# preconditioner of the ELASTIC problem (first argument): 'Green', 'Green_Jacobi' or 'dual' G K(C_ref C_d^-1 C_ref) G;
# the damage problem uses Green-Jacobi in every run, so only the elastic solves differ
preconditioner_type = sys.argv[1] if len(sys.argv) > 1 else 'Green_Jacobi'
damage_preconditioner_type = 'Green_Jacobi'
cg_tol = 1e-6                      # CG: energy-error estimate <= cg_tol * energy of the solution (warm start, delay 10)
cg_delay = 10

figure_folder = os.path.join(os.path.dirname(__file__), 'figures')
# NetCDF output (None: no output); read it back with io_utils.read_file or
# io_utils.load_fields, replot it with plot_phase_field_fracture_output.py
output_file = os.path.join(os.path.dirname(__file__), 'exp_data', f'AT2_N{number_of_pixels[0]}_{preconditioner_type}{run_tag}.nc')
# one frame per 'step' (converged load step) or per 'iteration' (every staggered iteration)
output_every = 'step'

# live plot of the damage (serial runs): None, 'step' (every load step)
# or 'iteration' (every staggered iteration)
live_plot = None

# ----------------------------------------------------------------------------
# Two discretizations of the same grid: vector (u~) and scalar (d)
# ----------------------------------------------------------------------------
cell_u = domain.PeriodicUnitCell(domain_size=domain_size, problem_type='elasticity')
disc_u = domain.Discretization(cell=cell_u, nb_of_pixels_global=number_of_pixels,
                               discretization_type='finite_element', element_type=element_type)

cell_d = domain.PeriodicUnitCell(domain_size=domain_size, problem_type='conductivity')
disc_d = domain.Discretization(cell=cell_d, nb_of_pixels_global=number_of_pixels,
                               discretization_type='finite_element', element_type=element_type)

comm = disc_u.communicator
dim = disc_u.domain_dimension
assert disc_u.fft.nb_subdomain_grid_pts == disc_d.fft.nb_subdomain_grid_pts, \
    'the two discretizations must have the same MPI decomposition'

if comm.rank == 0:
    print(f'grid {number_of_pixels}, ranks {comm.size}, l = {length_scale:.4f}, Gc = {Gc}')

# ----------------------------------------------------------------------------
# Microstructure and elastic material C_0
# ----------------------------------------------------------------------------
phase = microstructure_library.get_geometry(nb_voxels=disc_u.nb_of_pixels,
                                            microstructure_name='circle_inclusion',
                                            coordinates=disc_u.fft.coords)   # 1 matrix, 0 pore
K_0, G_0 = material_models.get_bulk_and_shear_modulus(E=E_young, poisson=poisson)
lam, mu = material_models.get_lame_parameters_from_bulk_and_shear(K_0, G_0, dim=dim)

lam_11qxyz = disc_u.get_quad_field_scalar(name='lam_first_lame')
mu_11qxyz = disc_u.get_quad_field_scalar(name='mu_second_lame')
stiffness = np.where(phase > 0, 1.0, pore_contrast)          # per pixel
lam_11qxyz.s[0, 0, ...] = lam * stiffness                       # broadcast over q
mu_11qxyz.s[0, 0, ...] = mu * stiffness
bulk_qxyz = lam_11qxyz.s[0, 0] + 2.0 * mu_11qxyz.s[0, 0] / dim

material = material_models.LinearElastic(discretization=disc_u, lam_1qxyz=lam_11qxyz,
                                         mu_1qxyz=mu_11qxyz, name='linear_isotropic_elasticity')
C_0 = disc_u.get_material_data_size_field_mugrid(name='C_0')
material.get_algorithmic_tangent(disc_u.get_strain_sized_field(name='zero_strain'), C_0)
C_d = disc_u.get_material_data_size_field_mugrid(name='C_degraded')   # g(d) C_0

# ----------------------------------------------------------------------------
# Fields
# ----------------------------------------------------------------------------
macro_strain = disc_u.get_gradient_size_field(name='macro_strain')
strain = disc_u.get_gradient_size_field(name='strain')
rhs_u = disc_u.get_unknown_size_field(name='rhs_u')
u_fluct = disc_u.get_unknown_size_field(name='u_fluctuation')

damage = disc_d.get_unknown_size_field(name='damage')              # d, nodal [1, n, x, y]
damage_old = disc_d.get_unknown_size_field(name='damage_old')
rhs_d = disc_d.get_unknown_size_field(name='rhs_d')
damage_q = disc_d.get_quad_field_scalar(name='damage_at_quad')     # N d
history = disc_d.get_quad_field_scalar(name='history')             # H, current iterate
history_n = disc_d.get_quad_field_scalar(name='history_converged')  # H of the last load step
reaction = disc_d.get_quad_field_scalar(name='reaction')           # Gc/l + 2H
grad_d = disc_d.get_gradient_size_field(name='grad_damage')
temp_q = disc_d.get_quad_field_scalar(name='temp_q')
temp_n = disc_d.get_unknown_size_field(name='temp_n')


# ----------------------------------------------------------------------------
# Operators
# ----------------------------------------------------------------------------
def K_u(x, Ax):
    """Elastic stiffness  B^T W g(d) C_0 B x  (small strain)."""
    disc_u.apply_system_matrix_mugrid(material_data_field=C_d, input_field_inxyz=x,
                                      output_field_inxyz=Ax, formulation='small_strain')


C_ref = material_models.get_elastic_material_tensor(dim=dim, K=K_0, mu=G_0, kind='linear')
green_u = disc_u.get_preconditioner_Green_mugrid(reference_material_data_ijkl=C_ref,
                                                 formulation='small_strain')


# dual: C_ref C_d^-1 C_ref, C_d = g(d) C_0(x); the inverse on the symmetric strains (Mandel), pointwise
MANDEL = [(0, 0), (1, 1), (0, 1)]
W_MANDEL = np.array([1.0, 1.0, np.sqrt(2.0)])


def to_mandel(C):
    return np.array([[W_MANDEL[a] * W_MANDEL[b] * C[i, j, k, l] for b, (k, l) in enumerate(MANDEL)]
                     for a, (i, j) in enumerate(MANDEL)])


def from_mandel(M):
    C = np.zeros((2, 2, 2, 2) + M.shape[2:])
    for a, (i, j) in enumerate(MANDEL):
        for b, (k, l) in enumerate(MANDEL):
            for ii, jj in {(i, j), (j, i)}:
                for kk, ll in {(k, l), (l, k)}:
                    C[ii, jj, kk, ll] = M[a, b] / (W_MANDEL[a] * W_MANDEL[b])
    return C


M_ref = to_mandel(C_ref)
C_dual = disc_u.get_material_data_size_field_mugrid(name='C_dual')
dual_temp_1 = disc_u.get_unknown_size_field(name='dual_temp_1')
dual_temp_2 = disc_u.get_unknown_size_field(name='dual_temp_2')


def update_dual_material():
    M_d = np.moveaxis(to_mandel(C_d.s), (0, 1), (-2, -1))
    C_dual.s[...] = from_mandel(np.moveaxis(M_ref @ np.linalg.inv(M_d) @ M_ref, (-2, -1), (0, 1)))


jacobi_u = None                     # D^-1/2 of the elastic K (Green_Jacobi only)
x_scaled_u = disc_u.get_unknown_size_field(name='x_scaled_u')


def M_u(x, Px):
    if preconditioner_type == 'dual':
        disc_u.fft.communicate_ghosts(x)
        disc_u.apply_preconditioner_mugrid(preconditioner_Fourier_fnfnqks=green_u, input_nodal_field_fnxyz=x,
                                           output_nodal_field_fnxyz=dual_temp_1)
        disc_u.apply_system_matrix_mugrid(material_data_field=C_dual, input_field_inxyz=dual_temp_1,
                                          output_field_inxyz=dual_temp_2, formulation='small_strain')
        disc_u.fft.communicate_ghosts(dual_temp_2)
        disc_u.apply_preconditioner_mugrid(preconditioner_Fourier_fnfnqks=green_u, input_nodal_field_fnxyz=dual_temp_2,
                                           output_nodal_field_fnxyz=Px)
        return
    if preconditioner_type == 'Green_Jacobi':
        x_scaled_u.s[...] = jacobi_u.s * x.s
        disc_u.apply_preconditioner_mugrid(preconditioner_Fourier_fnfnqks=green_u,
                                           input_nodal_field_fnxyz=x_scaled_u, output_nodal_field_fnxyz=Px)
        Px.s[...] *= jacobi_u.s
    else:
        disc_u.apply_preconditioner_mugrid(preconditioner_Fourier_fnfnqks=green_u,
                                           input_nodal_field_fnxyz=x, output_nodal_field_fnxyz=Px)


def apply_damage_operator(input_field_inxyz, output_field_inxyz, reaction_q):
    """[Gc l B^T W B + N^T W c N] d  with the reaction coefficient c at quad points."""
    # diffusion: Gc l B^T W B d
    disc_d.apply_gradient_operator_mugrid(u_inxyz=input_field_inxyz, grad_u_ijqxyz=grad_d)
    grad_d.s[...] *= Gc * length_scale
    disc_d.apply_gradient_transposed_operator_mugrid(gradient_field_ijqxyz=grad_d,
                                                     div_u_fnxyz=output_field_inxyz,
                                                     apply_weights=True)
    # reaction: N^T W c N d
    disc_d.apply_N_operator_mugrid(input_field_inxyz, temp_q)
    temp_q.s[...] *= reaction_q
    disc_d.apply_N_transposed_operator_mugrid(quad_field_ijqxyz=temp_q, nodal_field_inxyz=temp_n,
                                              apply_weights=True)
    output_field_inxyz.s[...] += temp_n.s


def K_d(x, Ax):
    apply_damage_operator(x, Ax, reaction.s)


def K_d_as_operator(input_field_inxyz, output_field_inxyz):
    apply_damage_operator(input_field_inxyz, output_field_inxyz, reaction.s)


# Green preconditioner of the homogeneous damage operator (H = 0); the mass term
# makes its zero-frequency block regular, so it is inverted as well
green_d = disc_d.get_preconditioner_Green_mugrid(
    reference_material_data_ijkl=None,
    operator=lambda input_field_inxyz, output_field_inxyz: apply_damage_operator(
        input_field_inxyz, output_field_inxyz, Gc / length_scale),
    invert_zero_mode=True)


jacobi_d = None                     # D^-1/2 of the damage K (Green_Jacobi only)
x_scaled_d = disc_d.get_unknown_size_field(name='x_scaled_d')


def M_d(x, Px):
    if damage_preconditioner_type == 'Green_Jacobi':
        x_scaled_d.s[...] = jacobi_d.s * x.s
        disc_d.apply_preconditioner_mugrid(preconditioner_Fourier_fnfnqks=green_d,
                                           input_nodal_field_fnxyz=x_scaled_d, output_nodal_field_fnxyz=Px)
        Px.s[...] *= jacobi_d.s
    else:
        disc_d.apply_preconditioner_mugrid(preconditioner_Fourier_fnfnqks=green_d,
                                           input_nodal_field_fnxyz=x, output_nodal_field_fnxyz=Px)


class IterationCounter:
    def __init__(self):
        self.count = 0

    def __call__(self, iteration, fields):
        self.count = iteration


# ----------------------------------------------------------------------------
# Constitutive helpers (pointwise, arrays of shape [.., q, x, y])
# ----------------------------------------------------------------------------
def update_degraded_material():
    """C_d = g(d) C_0 with d interpolated to the quadrature points."""
    disc_d.apply_N_operator_mugrid(damage, damage_q)
    g_qxyz = (1.0 - damage_q.s[0, 0]) ** 2 + k_residual
    C_d.s[...] = g_qxyz[np.newaxis, np.newaxis, np.newaxis, np.newaxis] * C_0.s


def tensile_energy(eps_ijqxyz):
    """psi^+ = K/2 <tr eps>_+^2 + mu eps_dev : eps_dev  (Amor split)."""
    trace = np.einsum('ii...->...', eps_ijqxyz)
    eps_dev = eps_ijqxyz - trace / dim * np.eye(dim).reshape(dim, dim, *([1] * trace.ndim))
    return (0.5 * bulk_qxyz * np.maximum(trace, 0.0) ** 2
            + mu_11qxyz.s[0, 0] * np.einsum('ij...,ij...->...', eps_dev, eps_dev))


def total_strain():
    """eps = E + sym(grad u~) at the quadrature points."""
    disc_u.apply_gradient_operator_symmetrized_mugrid(u_inxyz=u_fluct, grad_u_ijqxyz=strain)
    strain.s[...] += macro_strain.s
    return strain.s


# ----------------------------------------------------------------------------
# Live plot (serial only): one figure, redrawn in place
# ----------------------------------------------------------------------------
if live_plot and comm.size == 1:
    import matplotlib
    matplotlib.use('TkAgg')   # a real window; PyCharm's plot panel cannot redraw a figure
    import matplotlib.pyplot as plt

    plt.ion()
    live_fig, (live_ax_d, live_ax_s) = plt.subplots(1, 2, figsize=(10, 4.3))


def plot_live(title):
    """Redraw the damage field and the stress-strain curve."""
    if not live_plot or comm.size > 1:
        return
    live_ax_d.clear()
    live_ax_d.imshow(damage.s[0, 0].T, origin='lower', vmin=0, vmax=1, cmap='inferno',
                     extent=(0, domain_size[0], 0, domain_size[1]))
    live_ax_d.set_title(title, fontsize=10)
    live_ax_s.clear()
    live_ax_s.plot(macro_strains, macro_stresses, 'o-', markersize=3)
    live_ax_s.set_xlabel(r'$\bar{\varepsilon}_{xx}$')
    live_ax_s.set_ylabel(r'$\bar{\sigma}_{xx}$')
    plt.pause(0.01)


# ----------------------------------------------------------------------------
# Load stepping with the staggered scheme
# ----------------------------------------------------------------------------
start_time = time.time()
total_cg_u, total_cg_d = 0, 0

writer = None
if output_file is not None:
    if comm.rank == 0:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
    MPI.COMM_WORLD.Barrier()
    # history (current H) equals history_n at the end of every load step
    writer = io_utils.FieldWriter(
        output_file, [damage, u_fluct, history],
        attributes={'nb_grid_pts': number_of_pixels, 'domain_size': domain_size, 'E': E_young,
                    'poisson': poisson, 'pore_contrast': pore_contrast, 'Gc': Gc,
                    'length_scale': length_scale, 'k_residual': k_residual,
                    'macro_strain_direction': macro_strain_direction,
                    'preconditioner_type': preconditioner_type, 'output_every': output_every},
        frame_variables={'load': (), 'macro_stress': (dim, dim), 'step': (), 'staggered_iteration': (),
                         'damage_change': (), 'converged': ()})
macro_strains, macro_stresses = [0.0], [0.0]
snapshots = {}
update_degraded_material()

for step, load in enumerate(load_steps):
    macro_strain_ij = load * macro_strain_direction
    disc_u.get_macro_gradient_field_mugrid(macro_gradient_ij=macro_strain_ij,
                                           macro_gradient_field_ijqxyz=macro_strain)
    for stagger in range(max_staggered_iterations):
        # 1. elasticity with the current damage (warm start from the last u~)
        disc_u.get_rhs_mugrid(material_data_field_ijklqxyz=C_d,
                              macro_gradient_field_ijqxyz=macro_strain, rhs_inxyz=rhs_u)
        if preconditioner_type == 'Green_Jacobi':
            jacobi_u = disc_u.get_preconditioner_Jacobi_mugrid(material_data_field_ijklqxyz=C_d,
                                                               formulation='small_strain', name='jacobi_u')
        elif preconditioner_type == 'dual':
            update_dual_material()
        counter_u = IterationCounter()
        norms_u = solvers.conjugate_gradients_mugrid_experimental(
            comm=comm, fc=disc_u.field_collection, hessp=K_u, b=rhs_u, x=u_fluct, P=M_u, maxiter=20000,
            stop_crit_norm='energy_lower_estim', d_min=cg_delay,
            **topology_optimization.energy_stop_kwargs(disc_u, K_u, u_fluct, cg_tol))[1]
        counter_u.count = int(norms_u['stop_iteration']['energy_lower_estim'])
        total_cg_u += counter_u.count

        # 2. history field  H = max(H_{n-1}, psi^+)
        history.s[0, 0] = np.maximum(history_n.s[0, 0], tensile_energy(total_strain()))

        # 3. damage:  [Gc l B^T W B + N^T W (Gc/l + 2H) N] d = N^T W 2H
        reaction.s[...] = Gc / length_scale + 2.0 * history.s
        temp_q.s[...] = 2.0 * history.s
        disc_d.apply_N_transposed_operator_mugrid(quad_field_ijqxyz=temp_q, nodal_field_inxyz=rhs_d,
                                                  apply_weights=True)
        damage_old.s[...] = damage.s
        if damage_preconditioner_type == 'Green_Jacobi':
            jacobi_d = disc_d.get_preconditioner_Jacobi_mugrid(operator=K_d_as_operator, name='jacobi_d')
        counter_d = IterationCounter()
        norms_d = solvers.conjugate_gradients_mugrid_experimental(
            comm=comm, fc=disc_d.field_collection, hessp=K_d, b=rhs_d, x=damage, P=M_d, maxiter=20000,
            stop_crit_norm='energy_lower_estim', d_min=cg_delay,
            **topology_optimization.energy_stop_kwargs(disc_d, K_d, damage, cg_tol))[1]
        counter_d.count = int(norms_d['stop_iteration']['energy_lower_estim'])
        total_cg_d += counter_d.count
        update_degraded_material()

        change = float(disc_d.mpi_reduction.max(np.asarray(np.max(np.abs(damage.s - damage_old.s)))))
        if live_plot == 'iteration':
            plot_live(f'step {step}, E_xx = {load:.4f}, iteration {stagger + 1}')
        if writer is not None and output_every == 'iteration' and change >= staggered_tol:
            writer.write(load=load, step=step, staggered_iteration=stagger + 1, damage_change=change,
                         converged=0,
                         macro_stress=disc_u.get_homogenized_stress_mugrid(
                             material_data_field_ijklqxyz=C_d, displacement_field_inxyz=u_fluct,
                             macro_gradient_field_ijqxyz=macro_strain, formulation='small_strain'))
        if change < staggered_tol:
            break

    history_n.s[...] = history.s
    sigma_ij = disc_u.get_homogenized_stress_mugrid(material_data_field_ijklqxyz=C_d,
                                                    displacement_field_inxyz=u_fluct,
                                                    macro_gradient_field_ijqxyz=macro_strain,
                                                    formulation='small_strain')
    d_max = float(disc_d.mpi_reduction.max(np.asarray(np.max(damage.s))))
    macro_strains.append(load)
    macro_stresses.append(sigma_ij[0, 0])
    if comm.rank == 0:
        print(f'step {step:3d}  E_xx = {load:.4f}  sigma_xx = {sigma_ij[0, 0]:.5e}  '
              f'max d = {d_max:.3f}  staggered its = {stagger + 1:3d}  '
              f'CG its (u, d) = ({counter_u.count}, {counter_d.count})')
    plot_live(f'step {step}, E_xx = {load:.4f}')
    if writer is not None:
        # the converged state (in 'iteration' mode the last iteration of the step)
        writer.write(load=load, macro_stress=sigma_ij, step=step, staggered_iteration=stagger + 1,
                     damage_change=change, converged=1)
    if step in (len(load_steps) // 2, len(load_steps) - 1):
        snapshots[load] = np.copy(damage.s[0, 0])

if writer is not None:
    writer.close()
if comm.rank == 0:
    print(f'Elapsed time: {time.time() - start_time:.1f} s')
    print(f'{preconditioner_type}: total CG iterations elasticity = {total_cg_u}, damage = {total_cg_d}')

# ----------------------------------------------------------------------------
# Final figure (serial only: the fields are distributed over the ranks otherwise)
# ----------------------------------------------------------------------------
if comm.size == 1:
    import matplotlib.pyplot as plt

    plt.ioff()

    os.makedirs(figure_folder, exist_ok=True)
    fig, axes = plt.subplots(1, 1 + len(snapshots), figsize=(4.5 * (1 + len(snapshots)), 4))
    axes[0].plot(macro_strains, macro_stresses, 'o-', markersize=3)
    axes[0].set_xlabel(r'macroscopic strain $\bar{\varepsilon}_{xx}$')
    axes[0].set_ylabel(r'macroscopic stress $\bar{\sigma}_{xx}$')
    axes[0].set_title('AT2, uniaxial strain')
    x, y = disc_d.fft.coords
    for ax, (load, d_xy) in zip(axes[1:], snapshots.items()):
        mesh = ax.pcolormesh(x, y, d_xy, vmin=0, vmax=1, cmap='inferno', shading='auto')
        ax.set_aspect('equal')
        ax.set_title(rf'damage $d$ at $\bar{{\varepsilon}}_{{xx}} = {load:.3f}$')
        fig.colorbar(mesh, ax=ax)
    fig.tight_layout()
    file_name = os.path.join(figure_folder, f'phase_field_fracture_AT2_N{number_of_pixels[0]}_{preconditioner_type}{run_tag}.png')
    fig.savefig(file_name, dpi=150)
    print(f'saved {file_name}')
    plt.show()
