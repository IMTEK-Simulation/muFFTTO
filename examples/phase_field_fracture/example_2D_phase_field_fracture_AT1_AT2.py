"""
Phase-field fracture (AT1 or AT2) of a periodic porous cell under macroscopic tension
=====================================================================================

The material models of :mod:`muFFTTO.phase_field_fracture` in the staggered scheme of the 'hybrid' formulation
(Ambati, Gerasimov & De Lorenzis 2015), per load step E_n = t_n E_dir:

    1. elasticity:  B^T W g(d) C_0(x) B u~ = -B^T W g(d) C_0(x) E          (CG, dual preconditioner)
    2. history:     H = max(H_{n-1}, psi^+(eps), psi_c)                     (split: Amor by default)
    3. damage:      [diffusion B^T W B + N^T W (reaction + 2H) N] d = N^T W (2H - offset)
                                                                           (CG, Green of the homogeneous operator)
    repeat 1-3 until max |d - d_old| < tol, then H_{n-1} <- H.

AT2: diffusion Gc l, reaction Gc / l, offset 0. AT1: diffusion 3/4 Gc l, reaction 0, offset 3 Gc / (8 l); the
threshold psi_c = 3 Gc / (16 l) in H keeps the damage system SPD, d is clipped to [0, 1] after each solve.

The elastic material is C(x) = g(d) s(x) C_0 (s = 1 in the matrix, a small contrast in the pore), a
:class:`muFFTTO.domain.ScaledMaterial`, so the dual preconditioner G K(C_0 / (g s)) G needs no stored tangent. All
linear solves stop on the energy-error estimate (relative tolerance 1e-6, adaptive delay with minimum 10).

Run (environment variables, defaults in brackets):
    MODEL [AT2] | AT1,  SPLIT [amor] | spectral | none,  N [64] pixels per direction,  STEPS [40] load steps
    python example_2D_phase_field_fracture_AT1_AT2.py
    mpirun -n 4 python example_2D_phase_field_fracture_AT1_AT2.py
"""
import os
import time

import numpy as np

from muFFTTO import domain, material_models, microstructure_library, phase_field_fracture, preconditioners, solvers

# ----------------------------------------------------------------------------------------------------------------
# Parameters
# ----------------------------------------------------------------------------------------------------------------
model = os.environ.get('MODEL', 'AT2')
split = os.environ.get('SPLIT', 'amor')
N = int(os.environ.get('N', 64))
nb_steps = int(os.environ.get('STEPS', 40))
domain_size = [1.0, 1.0]
number_of_pixels = (N, N)
element_type = 'bilinear_rectangle'

E_young, poisson = 1.0, 0.2           # matrix
pore_contrast = 1e-3                  # stiffness of the pore relative to the matrix
Gc = 5e-4                             # fracture toughness
length_scale = 2.0 / N                # l = 2 pixels
k_residual = 1e-4                     # residual stiffness in g(d)
macro_strain_direction = np.array([[1.0, 0.0], [0.0, 0.0]])  # uniaxial strain E_xx
load_steps = np.linspace(0.0, 0.08, nb_steps + 1)[1:]
staggered_tol = 1e-3                  # max |d - d_old|
max_staggered_iterations = 300
cg_tol = 1e-6                         # relative energy error of every linear solve

# ----------------------------------------------------------------------------------------------------------------
# Two discretizations of the same grid: displacement u~ (vector) and damage d (scalar)
# ----------------------------------------------------------------------------------------------------------------
disc_u = domain.Discretization(cell=domain.PeriodicUnitCell(domain_size=domain_size, problem_type='elasticity'),
                               nb_of_pixels_global=number_of_pixels, element_type=element_type)
disc_d = domain.Discretization(cell=domain.PeriodicUnitCell(domain_size=domain_size, problem_type='conductivity'),
                               nb_of_pixels_global=number_of_pixels, element_type=element_type)
comm = disc_u.communicator

# ----------------------------------------------------------------------------------------------------------------
# Material: C = g(d) s(x) C_0
# ----------------------------------------------------------------------------------------------------------------
K_0, mu_0 = material_models.get_bulk_and_shear_modulus(E=E_young, poisson=poisson)
lam_0, _ = material_models.get_lame_parameters_from_bulk_and_shear(K_0, mu_0, dim=2)
C_0 = material_models.get_elastic_material_tensor(dim=2, K=K_0, mu=mu_0, kind='linear')
phase = microstructure_library.get_geometry(nb_voxels=disc_u.nb_of_pixels, microstructure_name='circle_inclusion',
                                            coordinates=disc_u.fft.coords)          # 1 matrix, 0 pore
stiffness = np.where(phase > 0, 1.0, pore_contrast)                                # s(x) per pixel
scale = disc_u.get_quad_field_scalar(name='material_scale')                        # g(d) s(x) at quad points
material = domain.ScaledMaterial([(scale, C_0)])

density = phase_field_fracture.CrackDensity(model, Gc, length_scale)

# ----------------------------------------------------------------------------------------------------------------
# Fields
# ----------------------------------------------------------------------------------------------------------------
macro_strain = disc_u.get_gradient_size_field(name='macro_strain')
strain = disc_u.get_gradient_size_field(name='strain')
rhs_u = disc_u.get_unknown_size_field(name='rhs_u')
u = disc_u.get_unknown_size_field(name='u_fluctuation')
damage = disc_d.get_unknown_size_field(name='damage')
damage_old = disc_d.get_unknown_size_field(name='damage_old')
rhs_d = disc_d.get_unknown_size_field(name='rhs_d')
damage_q = disc_d.get_quad_field_scalar(name='damage_q')
history = disc_d.get_quad_field_scalar(name='history')
history_n = disc_d.get_quad_field_scalar(name='history_n')
work_d = (disc_d.get_gradient_size_field(name='work_gradient'), disc_d.get_quad_field_scalar(name='work_quad'),
          disc_d.get_unknown_size_field(name='work_nodal'))


def update_material():
    """scale = g(d) s(x), d interpolated to the quadrature points."""
    disc_d.apply_N_operator_mugrid(damage, damage_q)
    scale.s[0, 0] = phase_field_fracture.degradation(damage_q.s[0, 0], k_residual) * stiffness


# ----------------------------------------------------------------------------------------------------------------
# Operators and preconditioners
# ----------------------------------------------------------------------------------------------------------------
def K_u(x, Kx):
    disc_u.apply_system_matrix_mugrid(material_data_field=material, input_field_inxyz=x, output_field_inxyz=Kx,
                                      formulation='small_strain')


M_u = preconditioners.Dual(disc_u, C_0, material, formulation='small_strain')


def K_d(x, Kx):
    density.apply_operator(disc_d, x, Kx, density.reaction_field(history.s), *work_d)


def K_d_reference(input_field_inxyz, output_field_inxyz):       # homogeneous damage operator (H = psi_c)
    density.apply_operator(disc_d, input_field_inxyz, output_field_inxyz, density.preconditioner_reaction, *work_d)


M_d = preconditioners.Green(disc_d, operator=K_d_reference, invert_zero_mode=True, name='green_damage')


def solve(disc, K, M, b, x):
    _, info = solvers.conjugate_gradients(disc.communicator, disc.field_collection, K, b, x, M, tol=cg_tol)
    return info['iterations']


def tensile_energy():
    """psi^+ of eps = E + sym(grad u~), with the Lame parameters of the material s(x) C_0."""
    disc_u.apply_gradient_operator_symmetrized_mugrid(u_inxyz=u, grad_u_ijqxyz=strain)
    strain.s[...] += macro_strain.s
    _, psi_plus = phase_field_fracture.elastic_energy(strain.s, lam_0 * stiffness, mu_0 * stiffness, split=split)
    return psi_plus


# ----------------------------------------------------------------------------------------------------------------
# Load stepping with the staggered scheme
# ----------------------------------------------------------------------------------------------------------------
if comm.rank == 0:
    print(f'{model}, split {split}, grid {number_of_pixels}, ranks {comm.size}, l = {length_scale:.4f}, Gc = {Gc}')
history.s[...] = density.threshold
history_n.s[...] = density.threshold
update_material()
macro_strains, macro_stresses, cg_u, cg_d = [0.0], [0.0], 0, 0
start_time = time.time()
for step, load in enumerate(load_steps):
    disc_u.get_macro_gradient_field_mugrid(macro_gradient_ij=load * macro_strain_direction,
                                           macro_gradient_field_ijqxyz=macro_strain)
    for stagger in range(max_staggered_iterations):
        # 1. elasticity with the current damage (warm start from the last u~)
        disc_u.get_rhs_mugrid(material_data_field_ijklqxyz=material, macro_gradient_field_ijqxyz=macro_strain,
                              rhs_inxyz=rhs_u)
        cg_u += solve(disc_u, K_u, M_u, rhs_u, u)
        # 2. history
        phase_field_fracture.update_history(history.s[0, 0], history_n.s[0, 0], tensile_energy(), density.threshold)
        # 3. damage
        density.rhs(disc_d, history.s, rhs_d, work_d[1])
        damage_old.s[...] = damage.s
        cg_d += solve(disc_d, K_d, M_d, rhs_d, damage)
        if model == 'AT1':
            np.clip(damage.s, 0.0, 1.0, out=damage.s)
        update_material()
        change = float(disc_d.mpi_reduction.max(np.abs(damage.s - damage_old.s)))
        if change < staggered_tol:
            break
    history_n.s[...] = history.s
    sigma = disc_u.get_homogenized_stress_mugrid(material_data_field_ijklqxyz=material, displacement_field_inxyz=u,
                                                 macro_gradient_field_ijqxyz=macro_strain, formulation='small_strain')
    macro_strains.append(load)
    macro_stresses.append(sigma[0, 0])
    if comm.rank == 0:
        print(f'step {step:3d}  E_xx = {load:.4f}  sigma_xx = {sigma[0, 0]:.4e}  '
              f'max d = {float(disc_d.mpi_reduction.max(damage.s)):.3f}  staggered its = {stagger + 1}')
if comm.rank == 0:
    print(f'{time.time() - start_time:.1f} s, CG iterations: elasticity {cg_u}, damage {cg_d}')

# ----------------------------------------------------------------------------------------------------------------
# Figure (serial: the fields are distributed over the ranks otherwise)
# ----------------------------------------------------------------------------------------------------------------
if comm.size == 1:
    import matplotlib.pyplot as plt

    fig, (ax_curve, ax_damage) = plt.subplots(1, 2, figsize=(10, 4.3))
    ax_curve.plot(macro_strains, macro_stresses, 'o-', markersize=3)
    ax_curve.set_xlabel(r'macroscopic strain $\bar{\varepsilon}_{xx}$', fontsize=12)
    ax_curve.set_ylabel(r'macroscopic stress $\bar{\sigma}_{xx}$', fontsize=12)
    ax_curve.set_title(f'{model}, {split} split', fontsize=12)
    image = ax_damage.imshow(damage.s[0, 0].T, origin='lower', vmin=0, vmax=1, cmap='inferno',
                             interpolation='nearest', extent=(0, domain_size[0], 0, domain_size[1]))
    ax_damage.set_title(rf'damage $d$ at $\bar{{\varepsilon}}_{{xx}} = {load_steps[-1]:.3f}$', fontsize=12)
    fig.colorbar(image, ax=ax_damage)
    fig.tight_layout()
    file_name = os.path.join(os.path.dirname(os.path.abspath(__file__)), f'phase_field_fracture_{model}.png')
    fig.savefig(file_name, dpi=150)
    print(f'saved {file_name}')
    plt.show()
