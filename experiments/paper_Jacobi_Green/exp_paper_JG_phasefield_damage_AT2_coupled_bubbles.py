import time
import os
import sys
import argparse

import numpy as np
from mpi4py import MPI
from NuMPI.IO import save_npy
from NuMPI.IO import load_npy

sys.path.append("/home/martin/Programming/muFFTTO_paralellFFT_test/muFFTTO")
sys.path.append('../..')  # Add parent directory to path

from muFFTTO import domain
from muFFTTO import solvers
from muFFTTO import microstructure_library

parser = argparse.ArgumentParser(
    prog="exp_paper_JG_phasefield_damage_AT2_coupled_bubbles.py",
    description="Coupled phase-field damage model (AT-2) for micromechanics with bubbles."
)
parser.add_argument("-n", "--nb_pixel", default="16")
parser.add_argument("-Gc", "--fracture_toughness", default="1.0", type=float)
parser.add_argument("-ell", "--length_scale", default="2.0", type=float)
parser.add_argument(
    "-p", "--preconditioner_type",
    type=str,
    choices=["Green", "Jacobi", "Green_Jacobi"],
    default="Green_Jacobi",
    help="Type of preconditioner to use"
)

script_name = os.path.splitext(os.path.basename(__file__))[0]
args = parser.parse_args()
nnn = int(args.nb_pixel)
Gc_val = args.fracture_toughness
ell_val = args.length_scale
preconditioner_type = args.preconditioner_type

file_folder_path = os.path.dirname(os.path.realpath(__file__))  # script directory
data_folder_path = file_folder_path + '/exp_data/' + script_name + '/'
figure_folder_path = file_folder_path + '/figures/' + script_name + '/'

save_results = True
_info = {}
start_time = time.time()

number_of_pixels = (nnn, nnn, nnn)
domain_size = [1, 1, 1]
Nx = number_of_pixels[0]
Ny = number_of_pixels[1]
Nz = number_of_pixels[2]

problem_type = 'elasticity'
discretization_type = 'finite_element'
element_type = 'trilinear_hexahedron'
formulation = 'small_strain'

_info['problem_type'] = problem_type
_info['discretization_type'] = discretization_type
_info['element_type'] = element_type
_info['formulation'] = formulation
_info['preconditioner_type'] = preconditioner_type

my_cell = domain.PeriodicUnitCell(domain_size=domain_size,
                                  problem_type=problem_type)

discretization = domain.Discretization(cell=my_cell,
                                       nb_of_pixels_global=number_of_pixels,
                                       discretization_type=discretization_type,
                                       element_type=element_type)
if discretization.communicator.rank == 0:
    print(f'preconditioner: {preconditioner_type}')
    print(f'Gc = {Gc_val}, ell = {ell_val}')

file_folder_path = os.path.dirname(os.path.realpath(__file__))
data_folder_path = (
        file_folder_path + '/exp_data/' + script_name + '/' + f'Nx={Nx}' + f'Ny={Ny}' + f'Nz={Nz}'
        + f'_{preconditioner_type}' + '/')
figure_folder_path = (file_folder_path + '/figures/' + script_name + '/' f'Nx={Nx}' + f'Ny={Ny}' + f'Nz={Nz}'
                      + f'_{preconditioner_type}' + '/')
if discretization.communicator.rank == 0:
    if not os.path.exists(file_folder_path):
        os.makedirs(file_folder_path)
    if not os.path.exists(data_folder_path):
        os.makedirs(data_folder_path)
    if not os.path.exists(figure_folder_path):
        os.makedirs(figure_folder_path)

_info['nb_of_pixels'] = discretization.nb_of_pixels_global
_info['domain_size'] = domain_size
_info['Gc'] = Gc_val
_info['ell'] = ell_val

# identity tensor                                               [single tensor]
i = np.eye(discretization.domain_dimension)
I = np.einsum('ij,xyz', i, np.ones(number_of_pixels))

# identity tensors                                            [grid of tensors]
I4 = np.einsum('il,jk', i, i)
I4rt = np.einsum('ik,jl', i, i)
II = np.einsum('ij...  ,kl...  ->ijkl...', i, i)
I4s = (I4 + I4rt) / 2.
I4d = (I4s - II / 3.)

# Material parameters
model_parameters_linear = {'K': 2.0, 'mu': 1.0}
model_parameters_damage = {
    'K': 200.0,
    'mu': 100.0,
    'Gc': Gc_val,
    'ell': ell_val
}

_info['model_parameters_linear'] = model_parameters_linear
_info['model_parameters_damage'] = model_parameters_damage

# Load geometry (bubbles)
results_name = (f'bubbles_' + f'dof={nnn}')
geom_folder_path = file_folder_path + '/exp_data/' + 'exp_paper_JG_nonlinear_elasticity_JZ_bubles_generate_geom/'

inclusions = load_npy(geom_folder_path + results_name + f'.npy',
                      tuple(discretization.subdomain_locations_no_buffers),
                      tuple(discretization.nb_of_pixels), MPI.COMM_WORLD)

matrix_mask = inclusions > 0
inc_mask = inclusions == 0
del inclusions

# Damage penalty constant (AT-2 model)
cw = 8. / 3.  # For AT-2: cw = 8/3

# ============================================================================
# CONSTITUTIVE MODELS
# ============================================================================

def linear_elastic_q_points(strain_ijqxyz,
                            tangent_ijklqxyz,
                            stress_ijqxyz,
                            phase_xyz,
                            **kwargs):
    """Linear elastic constitutive model."""
    K = kwargs['K']
    mu = kwargs['mu']

    tangent_ijklqxyz.s[..., phase_xyz] = (K * II + 2. * mu * I4d)[..., None, None]
    stress_ijqxyz.s[..., phase_xyz] = np.einsum('ijklqx...,lkqx...  ->ijqx...  ',
                                                tangent_ijklqxyz.s[..., phase_xyz],
                                                strain_ijqxyz.s[..., phase_xyz])


def phase_field_damage_q_points(strain_ijqxyz,
                                tangent_ijklqxyz,
                                stress_ijqxyz,
                                damage_xyz,
                                phase_xyz,
                                **kwargs):
    """
    AT-2 phase-field damage model.

    Strain energy density: psi(eps, d) = g(d) * psi_e(eps)
    where g(d) = (1 - d)^2  (AT-2 degradation function)
    w(d) = d^2  (AT-2 crack geometric function)
    """
    K = kwargs['K']
    mu = kwargs['mu']
    Gc = kwargs['Gc']
    ell = kwargs['ell']

    # Volumetric strain
    strain_trace_qx = np.einsum('ii...', strain_ijqxyz.s[..., phase_xyz]) / 3

    strain_vol_ijqxyz_s = np.zeros_like(strain_ijqxyz.s[..., phase_xyz])
    for d_dim in np.arange(discretization.domain_dimension):
        strain_vol_ijqxyz_s[d_dim, d_dim] = strain_trace_qx

    # Deviatoric strain
    strain_dev_ijqxyz_s = strain_ijqxyz.s[..., phase_xyz] - strain_vol_ijqxyz_s

    # Elastic strain energy density (undamaged)
    psi_e_qx = 0.5 * (3.0 * K * strain_trace_qx**2 +
                      2.0 * mu * np.einsum('ij...,ji...->...', strain_dev_ijqxyz_s, strain_dev_ijqxyz_s))

    # Damage variable at quadrature points
    d_qx = damage_xyz.s[..., phase_xyz]

    # Degradation function: g(d) = (1-d)^2
    g_d = (1.0 - d_qx)**2
    dg_dd = -2.0 * (1.0 - d_qx)

    # Stress calculation: sigma = g(d) * C : epsilon
    sig_vol = 3.0 * K * strain_vol_ijqxyz_s
    sig_dev = 2.0 * mu * strain_dev_ijqxyz_s
    stress_undamaged = sig_vol + sig_dev

    stress_ijqxyz.s[..., phase_xyz] = g_d * stress_undamaged

    # Algorithmic tangent: d(sigma)/d(eps)
    # = g(d) * C_linear
    C_lin = (K * II + 2. * mu * I4d)[..., np.newaxis, np.newaxis]
    tangent_ijklqxyz.s[..., phase_xyz] = g_d * C_lin

    # Store elastic strain energy density for damage evolution
    return psi_e_qx, dg_dd


def constitutive_q_points(strain_ijqxyz, tangent_ijklqxyz, stress_ijqxyz, damage_field_xyz):
    """Dispatch to appropriate constitutive model."""
    global matrix_mask, inc_mask

    linear_elastic_q_points(strain_ijqxyz=strain_ijqxyz,
                            tangent_ijklqxyz=tangent_ijklqxyz,
                            stress_ijqxyz=stress_ijqxyz,
                            phase_xyz=matrix_mask,
                            **model_parameters_linear)

    phase_field_damage_q_points(strain_ijqxyz=strain_ijqxyz,
                                tangent_ijklqxyz=tangent_ijklqxyz,
                                stress_ijqxyz=stress_ijqxyz,
                                damage_xyz=damage_field_xyz,
                                phase_xyz=inc_mask,
                                **model_parameters_damage)


def constitutive(strain_ijqxyz, sig_ijqxyz, K4_ijklqxyz, damage_field_xyz):
    constitutive_q_points(strain_ijqxyz=strain_ijqxyz,
                          tangent_ijklqxyz=K4_ijklqxyz,
                          stress_ijqxyz=sig_ijqxyz,
                          damage_field_xyz=damage_field_xyz)


# ============================================================================
# DAMAGE EVOLUTION
# ============================================================================

def compute_elastic_energy_density(strain_ijqxyz, **kwargs):
    """Compute elastic strain energy density for the entire domain: psi_e = 0.5 * sigma : epsilon"""
    K = kwargs['K']
    mu = kwargs['mu']

    # Compute for full domain
    strain_trace_qx = np.einsum('ii...', strain_ijqxyz.s) / 3

    strain_vol_ijqxyz_s = np.zeros_like(strain_ijqxyz.s)
    for d_dim in np.arange(discretization.domain_dimension):
        strain_vol_ijqxyz_s[d_dim, d_dim] = strain_trace_qx

    strain_dev_ijqxyz_s = strain_ijqxyz.s - strain_vol_ijqxyz_s

    psi_e_qx = 0.5 * (3.0 * K * strain_trace_qx**2 +
                      2.0 * mu * np.einsum('ij...,ji...->...', strain_dev_ijqxyz_s, strain_dev_ijqxyz_s))

    return psi_e_qx


def update_damage_field(damage_field, strain_field, discretization, phase_xyz, **kwargs):
    """
    Minimize damage energy at fixed strain:
    E[d] = int [Gc/cw * (d/ell + ell*|grad d|^2) + (1-d)^2 * psi_e] dV

    Euler-Lagrange equation:
    Gc/cw * (-1/ell + 2*ell*Laplacian(d)) - 4*(1-d)*psi_e = 0

    Solve via gradient descent or fixed-point iteration.
    """
    Gc = kwargs['Gc']
    ell = kwargs['ell']
    K = kwargs['K']
    mu = kwargs['mu']

    # Compute elastic energy density at quadrature points (for entire domain)
    psi_e_qx = compute_elastic_energy_density(strain_field, K=K, mu=mu)

    # Average elastic energy from quadrature points to nodes
    # psi_e_qx has shape (n_quad, Nx, Ny, Nz), we need to average over quadrature points
    psi_e_nodal = np.mean(psi_e_qx, axis=0, keepdims=False)

    # Threshold based on elastic energy (damage activates when psi_e exceeds critical value)
    psi_c = Gc / (4.0 * cw * ell)  # Critical energy density
    threshold_damage_xyz = np.maximum(0.0, 1.0 - np.sqrt(np.maximum(0.0, psi_c / (psi_e_nodal + 1e-16))))

    # For simplicity, use max damage value across the inclusion region
    # Flatten phase_xyz and threshold_damage_xyz for proper indexing
    phase_xyz_flat = phase_xyz.ravel() if hasattr(phase_xyz, 'ravel') else phase_xyz
    threshold_damage_flat = threshold_damage_xyz.ravel() if hasattr(threshold_damage_xyz, 'ravel') else threshold_damage_xyz

    if np.any(phase_xyz_flat):
        max_threshold = np.max(threshold_damage_flat[phase_xyz_flat])
        current_max_d = np.max(damage_field.s[...])
        new_damage_level = max(current_max_d, min(max_threshold, 1.0))
        # Update damage field uniformly (proportional update)
        if new_damage_level > current_max_d:
            damage_field.s[...] = new_damage_level

    # Clamp damage to [0, 1]
    damage_field.s[...] = np.clip(damage_field.s[...], 0.0, 1.0)

    return damage_field


# ============================================================================
# PLOTTING FUNCTION
# ============================================================================

def plot_results(iteration, total_strain, stress, damage, K4_ijkl, discretization, figure_folder_path):
    """
    Plot 2D slices of 3D fields: damage, material stiffness, strain, and stress.
    """
    if discretization.communicator.rank != 0:
        return

    import matplotlib.pyplot as plt
    import matplotlib as mpl

    # Get middle slice index
    mid_z = discretization.nb_of_pixels_global[2] // 2

    fig, axs = plt.subplots(2, 2, figsize=(14, 12))

    # 1. Damage field
    damage_slice = damage.s[0, 0, :, :, mid_z]
    im0 = axs[0, 0].imshow(damage_slice, cmap='Reds', vmin=0, vmax=1.0, origin='lower')
    axs[0, 0].set_title(f'Damage Field d (it {iteration})', fontsize=12, fontweight='bold')
    plt.colorbar(im0, ax=axs[0, 0], label='d')

    # 2. Material stiffness (K[0,0,0,0] component, averaged over quadrature points)
    # K4_ijkl.s has shape (i, j, k, l, quad, x, y, z)
    K_full = K4_ijkl.s[0, 0, 0, 0, :, :, :, mid_z]  # (quad, x, y)
    K_component = np.mean(K_full, axis=0)  # Average over quadrature points -> (x, y)
    K_mean = K_component.mean()
    K_std = K_component.std()
    im1 = axs[0, 1].imshow(K_component, cmap='viridis', origin='lower')
    axs[0, 1].set_title(f'Stiffness K₀₀₀₀ (it {iteration})', fontsize=12, fontweight='bold')
    plt.colorbar(im1, ax=axs[0, 1], label='K')

    # 3. Equivalent strain
    # Strain field shape: (i, j, quad, x, y, z)
    strain_trace = np.einsum('ii...', total_strain.s) / 3  # shape: (quad, x, y, z)
    strain_vol = np.zeros_like(total_strain.s)
    for d in range(discretization.domain_dimension):
        strain_vol[d, d] = strain_trace
    strain_dev = total_strain.s - strain_vol
    strain_eq_full = np.sqrt((2.0 / 3.0) * np.einsum('ij...,ji...->...', strain_dev, strain_dev))  # (quad, x, y, z)

    # Average over quadrature points and slice at mid_z
    strain_eq = np.mean(strain_eq_full, axis=0)  # (x, y, z)
    im2 = axs[1, 0].imshow(strain_eq[:, :, mid_z], cmap='plasma', origin='lower')
    axs[1, 0].set_title(f'Equivalent Strain εeq (it {iteration})', fontsize=12, fontweight='bold')
    plt.colorbar(im2, ax=axs[1, 0], label='εeq')

    # 4. Stress component (sigma_xy), averaged over quadrature points
    # Stress field shape: (i, j, quad, x, y, z)
    stress_full = stress.s[0, 1, :, :, :, mid_z]  # (quad, x, y)
    stress_slice = np.mean(stress_full, axis=0)  # (x, y)
    stress_max = np.max(np.abs(stress_slice))
    im3 = axs[1, 1].imshow(stress_slice, cmap='RdBu_r', vmin=-stress_max, vmax=stress_max, origin='lower')
    axs[1, 1].set_title(f'Stress σ₀₁ (it {iteration})', fontsize=12, fontweight='bold')
    plt.colorbar(im3, ax=axs[1, 1], label='σ')

    plt.tight_layout()

    if not os.path.exists(figure_folder_path):
        os.makedirs(figure_folder_path)

    plt.savefig(f"{figure_folder_path}iteration_{iteration:03d}.png", dpi=100, bbox_inches='tight')
    plt.close()

    if discretization.communicator.rank == 0:
        print(f'  → Saved plot: iteration_{iteration:03d}.png')


# ============================================================================
# SETUP FIELDS
# ============================================================================

macro_gradient_inc_field = discretization.get_gradient_size_field(name='macro_gradient_inc_field')
displacement_increment_field = discretization.get_unknown_size_field(name='displacement_increment_field')

strain_fluc_field = discretization.get_displacement_gradient_sized_field(name='strain_fluctuation_field')
total_strain_field = discretization.get_displacement_gradient_sized_field(name='strain_field')
rhs_field = discretization.get_unknown_size_field(name='rhs_field')

stress_field = discretization.get_displacement_gradient_sized_field(name='stress_field')
K4_ijklqyz = discretization.get_material_data_size_field_mugrid(name='K4_ijklqxyz')

# Damage field
damage_field = discretization.get_scalar_field(name='damage_field')
damage_field.s[...] = 0.0  # Initialize undamaged

# For storing previous damage for convergence check
damage_field_old = discretization.get_scalar_field(name='damage_field_old')
damage_field_old.s[...] = 0.0

# evaluate material law
constitutive(total_strain_field, stress_field, K4_ijklqyz, damage_field)

if save_results:
    results_name = (f'init_K')
    save_npy(fn=data_folder_path + results_name + f'.npy',
             data=K4_ijklqyz.s[0, 0, 0, 0].mean(axis=0),
             subdomain_locations=tuple(discretization.subdomain_locations_no_buffers),
             nb_grid_pts=tuple(discretization.nb_of_pixels_global),
             components_are_leading=True,
             comm=MPI.COMM_WORLD)

# set macroscopic loading increment
ninc = 10
_info['ninc'] = ninc

macro_gradient_inc = np.zeros(shape=(3, 3))
macro_gradient_inc[0, 1] += 0.05 / float(ninc)
macro_gradient_inc[1, 0] += 0.05 / float(ninc)
dt = 1. / float(ninc)

# set macroscopic gradient
discretization.get_macro_gradient_field_mugrid(macro_gradient_ij=macro_gradient_inc,
                                               macro_gradient_field_ijqxyz=macro_gradient_inc_field)

# assembly preconditioner
preconditioner = discretization.get_preconditioner_Green_mugrid(
    reference_material_data_ijkl=I4s)


def M_fun_Green(x, Px):
    discretization.fft.communicate_ghosts(x)
    discretization.apply_preconditioner_mugrid(preconditioner_Fourier_fnfnqks=preconditioner,
                                               input_nodal_field_fnxyz=x,
                                               output_nodal_field_fnxyz=Px)


sum_CG_its = 0
sum_Newton_its = 0
sum_damage_its = 0
start_time = time.time()
iteration_total = 0

# ============================================================================
# INCREMENTAL LOADING WITH COUPLED DAMAGE
# ============================================================================

for inc in range(ninc):
    if discretization.communicator.rank == 0:
        print(f'\n{"="*70}')
        print(f'Load Increment {inc+1}/{ninc}')
        print(f'{"="*70}')

    total_strain_field.s[...] += macro_gradient_inc_field.s[...]

    # evaluate material law
    constitutive(total_strain_field, stress_field, K4_ijklqyz, damage_field)

    # assembly rhs
    discretization.fft.communicate_ghosts(stress_field)
    discretization.apply_gradient_transposed_operator_mugrid(gradient_field_ijqxyz=stress_field,
                                                             div_u_fnxyz=rhs_field,
                                                             apply_weights=True)
    rhs_field.s[...] *= -1

    En = np.sqrt(
        discretization.communicator.sum(np.dot(total_strain_field.s.ravel(), total_strain_field.s.ravel())))

    rhs_t_norm = np.sqrt(discretization.communicator.sum(np.dot(rhs_field.s.ravel(), rhs_field.s.ravel())))
    iiter = 0

    norm_rhs = np.sqrt(discretization.communicator.sum(np.dot(rhs_field.s.ravel(), rhs_field.s.ravel())))
    if discretization.communicator.rank == 0:
        print(f'Rhs at new load step: {norm_rhs:.2e}')
        print(f'Strain norm: {En:.2e}')

    # ========================================================================
    # COUPLED DISPLACEMENT-DAMAGE LOOP
    # ========================================================================

    damage_its = 0
    damage_converged = False

    while not damage_converged:

        # ====================================================================
        # DISPLACEMENT UPDATE (Newton iteration via CG)
        # ====================================================================

        if preconditioner_type == 'Green':
            M_fun = M_fun_Green
        elif preconditioner_type == 'Green_Jacobi':
            K_diag_alg = discretization.get_preconditioner_Jacobi_mugrid(
                material_data_field_ijklqxyz=K4_ijklqyz, formulation=formulation)

            def M_fun_Jacobi(x, Px):
                discretization.fft.communicate_ghosts(x)
                x_jacobi_temp = discretization.get_unknown_size_field(name='x_jacobi_temp')

                x_jacobi_temp.s[...] = K_diag_alg.s * x.s
                discretization.fft.communicate_ghosts(x_jacobi_temp)
                discretization.apply_preconditioner_mugrid(preconditioner_Fourier_fnfnqks=preconditioner,
                                                           input_nodal_field_fnxyz=x_jacobi_temp,
                                                           output_nodal_field_fnxyz=Px)

                Px.s[...] = K_diag_alg.s * Px.s
                discretization.fft.communicate_ghosts(Px)

            M_fun = M_fun_Jacobi

        def K_fun(x, Ax):
            discretization.apply_system_matrix_mugrid(material_data_field=K4_ijklqyz,
                                                      input_field_inxyz=x,
                                                      output_field_inxyz=Ax,
                                                      formulation=formulation)
            discretization.fft.communicate_ghosts(Ax)

        norms = dict()
        norms['residual_rr'] = []
        norms['residual_rz'] = []

        def callback(it, x, r, p, z, stop_crit_norm):
            global norms
            norm_of_rr = discretization.communicator.sum(np.dot(r.ravel(), r.ravel()))
            norm_of_rz = discretization.communicator.sum(np.dot(r.ravel(), z.ravel()))
            norms['residual_rr'].append(norm_of_rr)
            norms['residual_rz'].append(norm_of_rz)

            if discretization.communicator.rank == 0 and it % 50 == 0:
                print(f"  CG it {it:5d}: stop_crit = {stop_crit_norm:.5e}")

        displacement_increment_field.s.fill(0)
        solvers.conjugate_gradients_mugrid(
            comm=discretization.communicator,
            fc=discretization.field_collection,
            hessp=K_fun,
            b=rhs_field,
            x=displacement_increment_field,
            P=M_fun,
            tol=1e-4,
            maxiter=20000,
            callback=callback,
            rtol=True
        )

        nb_it_comb = len(norms['residual_rr'])
        if discretization.communicator.rank == 0:
            print(f'  CG iterations: {nb_it_comb}')
        sum_CG_its += nb_it_comb

        iiter += 1
        sum_Newton_its += 1
        iteration_total += 1

        discretization.apply_gradient_operator_symmetrized_mugrid(
            u_inxyz=displacement_increment_field,
            grad_u_ijqxyz=strain_fluc_field)

        total_strain_field.s[...] += strain_fluc_field.s[...]
        constitutive(total_strain_field, stress_field, K4_ijklqyz, damage_field)

        discretization.apply_gradient_transposed_operator_mugrid(gradient_field_ijqxyz=stress_field,
                                                                 div_u_fnxyz=rhs_field,
                                                                 apply_weights=True)
        rhs_field.s[...] *= -1

        norm_rhs = np.sqrt(discretization.communicator.sum(
            np.dot(rhs_field.s.ravel(), rhs_field.s.ravel())))
        norm_strain_fluc = np.sqrt(discretization.communicator.sum(
            np.dot(strain_fluc_field.s.ravel(), strain_fluc_field.s.ravel())))

        if discretization.communicator.rank == 0:
            print(f'  Newton it {iiter}: norm_rhs = {norm_rhs:.2e}, norm_du = {norm_strain_fluc:.2e}')

        # ====================================================================
        # DAMAGE UPDATE (after every Newton iteration)
        # ====================================================================

        damage_field_old.s[...] = damage_field.s[...]

        update_damage_field(damage_field, total_strain_field, discretization, inc_mask,
                           **model_parameters_damage)

        # Check damage field convergence
        damage_diff = np.sqrt(discretization.communicator.sum(
            np.dot((damage_field.s - damage_field_old.s).ravel(),
                   (damage_field.s - damage_field_old.s).ravel())))

        damage_its += 1
        sum_damage_its += 1

        if discretization.communicator.rank == 0:
            print(f'  Damage it {damage_its}: max_d = {np.max(damage_field.s[...]):.4f}, '
                  f'change = {damage_diff:.2e}')

        # Convergence criterion: both displacement and damage must be small
        newton_converged = (norm_rhs < 1.e-5) and (norm_strain_fluc < 1.e-5)
        damage_converged_iter = (damage_diff < 1e-3) or (damage_its >= 3)

        if newton_converged and damage_converged_iter:
            damage_converged = True

        if iiter >= 10:
            damage_converged = True

    if save_results:
        results_name = (f'K4_ijklqyz_it{iteration_total}')
        save_npy(data_folder_path + results_name + f'.npy', K4_ijklqyz.s[0, 0, 0, 0].mean(axis=0),
                 tuple(discretization.subdomain_locations_no_buffers),
                 tuple(discretization.nb_of_pixels_global), MPI.COMM_WORLD)

        results_name = (f'damage_it{iteration_total}')
        save_npy(data_folder_path + results_name + f'.npy', damage_field.s[0, 0].copy(),
                 tuple(discretization.subdomain_locations_no_buffers),
                 tuple(discretization.nb_of_pixels_global), MPI.COMM_WORLD)

        # Plot results
        plot_results(iteration_total, total_strain_field, stress_field, damage_field, K4_ijklqyz,
                     discretization, figure_folder_path)

    end_time = time.time()
    elapsed_time = end_time - start_time

    if discretization.communicator.rank == 0:
        print(f'\nIncrement Summary:')
        print(f'  Total Newton iterations: {sum_Newton_its}')
        print(f'  Total CG iterations: {sum_CG_its}')
        print(f'  Damage iterations: {damage_its}')
        print(f'  Max damage: {np.max(damage_field.s[...]):.4f}')
        print(f'  Elapsed time: {elapsed_time:.2f} s')

end_time = time.time()
elapsed_time = end_time - start_time

if discretization.communicator.rank == 0:
    print(f'\n{"="*70}')
    print(f'SIMULATION COMPLETED')
    print(f'{"="*70}')
    print(f'Total elements: {number_of_pixels}')
    print(f'Total Newton iterations: {sum_Newton_its}')
    print(f'Total CG iterations: {sum_CG_its}')
    print(f'Total damage iterations: {sum_damage_its}')
    print(f'Total elapsed time: {elapsed_time:.2f} s ({elapsed_time/60:.2f} min)')

if save_results:
    _info['sum_Newton_its'] = sum_Newton_its
    _info['iteration_total'] = iteration_total
    _info['sum_CG_its'] = sum_CG_its
    _info['sum_damage_its'] = sum_damage_its
    _info['elapsed_time'] = elapsed_time
    if MPI.COMM_WORLD.rank == 0:
        np.savez(data_folder_path + f'info_log_final.npz', **_info)
        print(f'Results saved to: {data_folder_path}')
