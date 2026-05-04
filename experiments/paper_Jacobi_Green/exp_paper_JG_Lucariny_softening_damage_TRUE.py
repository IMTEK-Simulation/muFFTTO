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
    prog="exp_paper_JG_Lucariny_softening_damage_bubles.py",
    description="Solve non-linear elasticity with exponential softening damage (Lucarini & Segurado style, viscous)."
)
parser.add_argument("-n", "--nb_pixel", default="16")
parser.add_argument("-exp0", "--eps0_damage", default="0.05", type=float)
parser.add_argument("-dmax", "--max_damage", default="0.99", type=float)
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
eps0_val = args.eps0_damage
dmax_val = args.max_damage
preconditioner_type = args.preconditioner_type

# Viscosity parameter (Lucarini & Segurado style)
eta_val = 1e-3

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
_info['eta'] = eta_val

my_cell = domain.PeriodicUnitCell(domain_size=domain_size,
                                  problem_type=problem_type)

discretization = domain.Discretization(cell=my_cell,
                                       nb_of_pixels_global=number_of_pixels,
                                       discretization_type=discretization_type,
                                       element_type=element_type)
if discretization.communicator.rank == 0:
    print(f'preconditioer {preconditioner_type}')
file_folder_path = os.path.dirname(os.path.realpath(__file__))  # script directory
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

start_time = time.time()

# identity tensor                                               [single tensor]
i = np.eye(discretization.domain_dimension)
I = np.einsum('ij,xyz', i, np.ones(number_of_pixels))

# identity tensors                                            [grid of tensors]
I4 = np.einsum('il,jk', i, i)
I4rt = np.einsum('ik,jl', i, i)
II = np.einsum('ij...  ,kl...  ->ijkl...', i, i)
I4s = (I4 + I4rt) / 2.
I4d = (I4s - II / 3.)

model_parameters_non_linear = {'K': 2,
                               'mu': 1.0,
                               'eps0': eps0_val,
                               'dmax': dmax_val,
                               'eta': eta_val}

model_parameters_linear = {'K': 20,
                           'mu': 10}

_info['model_parameters_non_linear'] = model_parameters_non_linear
_info['model_parameters_linear'] = model_parameters_linear

phase_field = discretization.get_scalar_field(name='phase_field')

# History variable: kappa = max equivalent strain reached (at quadrature)
kappa_field = discretization.get_quad_field_scalar(name='kappa_field')
kappa_field.s[...] = 0.0

# Viscous damage variable d (at quadrature)
d_field = discretization.get_quad_field_scalar(name='d_field')
d_field.s[...] = 0.0

# save
results_name = (f'bubbles_' + f'dof={nnn}')
geom_folder_path = file_folder_path + '/exp_data/' + 'exp_paper_JG_nonlinear_elasticity_JZ_bubles_generate_geom/'

inclusions = load_npy(geom_folder_path + results_name + f'.npy',
                      tuple(discretization.subdomain_locations_no_buffers),
                      tuple(discretization.nb_of_pixels), MPI.COMM_WORLD)

matrix_mask = inclusions > 0
inc_mask = inclusions == 0
del inclusions


# linear elasticity
# -----------------
def linear_elastic_q_points(strain_ijqxyz,
                            tangent_ijklqxyz,
                            stress_ijqxyz,
                            phase_xyz,
                            **kwargs):
    K = kwargs['K']
    mu = kwargs['mu']

    tangent_ijklqxyz.s[..., phase_xyz] = (K * II + 2. * mu * I4d)[..., None, None]
    stress_ijqxyz.s[..., phase_xyz] = np.einsum('ijklqx...,lkqx...  ->ijqx...  ',
                                                tangent_ijklqxyz.s[..., phase_xyz],
                                                strain_ijqxyz.s[..., phase_xyz])


def softening_damage_q_points(strain_ijqxyz,
                              tangent_ijklqxyz,
                              stress_ijqxyz,
                              phase_xyz,
                              kappa_field,
                              d_field,
                              dt,
                              **kwargs):
    """
    Exponential softening damage model (Lucarini & Segurado style, viscous):
    - history variable kappa = max_t eps_eq
    - d_eq = dmax * (1 - exp(-kappa/eps0))
    - viscous evolution: d^{n+1} = d^n + dt/eta * (d_eq - d^n)
    - consistent viscous tangent
    """
    K = kwargs['K']
    mu = kwargs['mu']
    eps0 = kwargs['eps0']
    dmax = kwargs.get('dmax', 0.99)
    eta = kwargs['eta']

    # --- 1. Kinematics: volumetric / deviatoric split ---
    strain_phase = strain_ijqxyz.s[..., phase_xyz]

    strain_trace_qx = np.einsum('ii...', strain_phase) / 3.0

    strain_vol_ijqxyz_s = np.zeros_like(strain_phase)
    for d in range(discretization.domain_dimension):
        strain_vol_ijqxyz_s[d, d, ...] = strain_trace_qx

    strain_dev_ijqxyz_s = strain_phase - strain_vol_ijqxyz_s

    eps_small = 1e-15
    # Equivalent strain
    strain_dev_ddot = np.einsum('ij...,ji...->...', strain_dev_ijqxyz_s, strain_dev_ijqxyz_s)
    strain_eq_qx = np.sqrt((2.0 / 3.0) * strain_dev_ddot)
    strain_eq_safe = np.maximum(strain_eq_qx, eps_small)

    # --- 2. History variable kappa = max(kappa_old, strain_eq) ---
    # kappa_field.s has shape (1, 1, q, Nx, Ny, Nz)
    kappa_old = kappa_field.s[0, 0, :, phase_xyz].T  # shape (q, Nx_inc, Ny_inc, Nz_inc) -> but q is leading
    # Here strain_eq_qx has shape (q, Nx_inc, Ny_inc, Nz_inc)
    kappa_new = np.maximum(kappa_old, strain_eq_qx)
    kappa_field.s[0, 0, :, phase_xyz] = kappa_new.T

    # --- 3. Equilibrium damage d_eq(kappa) ---
    d_eq = dmax * (1.0 - np.exp(-kappa_new / eps0))

    # --- 4. Viscous update of damage ---
    # d^{n+1} = d^n + dt/eta * (d_eq - d^n)
    d_old = d_field.s[0, 0, :, phase_xyz].T
    alpha = dt / eta
    d_trial = d_old + alpha * (d_eq - d_old)
    d_trial = np.clip(d_trial, 0.0, dmax)
    d_field.s[0, 0, :, phase_xyz] = d_trial.T

    stiffness_retention = 1.0 - d_trial

    # --- 5. Linear stress ---
    sig_vol = 3.0 * K * strain_vol_ijqxyz_s
    sig_dev = 2.0 * mu * strain_dev_ijqxyz_s
    total_lin_stress = sig_vol + sig_dev

    # Damaged stress
    stress_ijqxyz.s[..., phase_xyz] = stiffness_retention * total_lin_stress

    # --- 6. Consistent viscous tangent ---

    # Term A: (1 - d) * C_linear
    C_lin = (K * II + 2.0 * mu * I4d)[..., np.newaxis, np.newaxis]
    term_A = stiffness_retention * C_lin

    # Active set: where kappa is driven by current strain_eq
    active = (strain_eq_qx > kappa_old) & (strain_eq_qx > eps_small)

    # deq/deps = (2/3) * eps_dev / eps_eq on active set
    dkappa_deps = np.zeros_like(strain_dev_ijqxyz_s)
    dkappa_deps[..., active] = (2.0 / 3.0) * (
        strain_dev_ijqxyz_s[..., active] / strain_eq_safe[..., active]
    )

    # d(d_eq)/d(kappa)
    dd_eq_dkappa = (dmax / eps0) * np.exp(-kappa_new / eps0)

    # Viscous factor: d^{n+1} = d^n + alpha (d_eq - d^n)
    # => ∂d^{n+1}/∂d_eq = alpha
    # => ∂d^{n+1}/∂kappa = alpha * dd_eq_dkappa
    visc_factor = alpha  # dt/eta
    dd_dkappa_visc = visc_factor * dd_eq_dkappa

    # Term B: - (∂σ/∂d) ⊗ (∂d/∂ε)
    # ∂σ/∂d = -C_lin:ε  (since σ = (1-d) C ε)
    # But we already have total_lin_stress = C ε
    # ∂σ/∂d = - total_lin_stress
    # ∂d/∂ε = dd_dkappa_visc * dkappa_deps
    factor = dd_dkappa_visc
    ddeps = factor * dkappa_deps  # same shape as strain_dev_ijqxyz_s

    # term_B = - (∂σ/∂d) ⊗ (∂d/∂ε) = - ( -total_lin_stress ) ⊗ ddeps = total_lin_stress ⊗ ddeps
    # term_B = np.einsum(
    #     'ij...,kl...->ijkl...',
    #     total_lin_stress,
    #     ddeps
    # )
    # Symmetrize

   # term_B = 0.5 * (term_B + np.einsum('ijkl...->klij...', term_B))
    term_B = 0.0
    tangent_ijklqxyz.s[..., phase_xyz] = term_A + term_B
    # inside softening_damage_q_points



def constitutive_q_points(strain_ijqxyz, tangent_ijklqxyz, stress_ijqxyz, dt):
    global matrix_mask, inc_mask
    linear_elastic_q_points(strain_ijqxyz=strain_ijqxyz,
                            tangent_ijklqxyz=tangent_ijklqxyz,
                            stress_ijqxyz=stress_ijqxyz,
                            phase_xyz=matrix_mask,
                            **model_parameters_linear)

    softening_damage_q_points(strain_ijqxyz=strain_ijqxyz,
                              tangent_ijklqxyz=tangent_ijklqxyz,
                              stress_ijqxyz=stress_ijqxyz,
                              phase_xyz=inc_mask,
                              kappa_field=kappa_field,
                              d_field=d_field,
                              dt=dt,
                              **model_parameters_non_linear)


def constitutive(strain_ijqxyz,
                 sig_ijqxyz,
                 K4_ijklqxyz,
                 dt):
    constitutive_q_points(strain_ijqxyz=strain_ijqxyz,
                          tangent_ijklqxyz=K4_ijklqxyz,
                          stress_ijqxyz=sig_ijqxyz,
                          dt=dt)


def plot_results(iteration, total_strain, stress, discretization):
    if discretization.communicator.rank != 0:
        return

    import matplotlib.pyplot as plt

    mid_z = discretization.nb_of_pixels_global[2] // 2

    fig, axs = plt.subplots(1, 3, figsize=(15, 5))

    strain_trace = np.einsum('ii...', total_strain.s) / 3
    strain_vol = np.zeros_like(total_strain.s)
    for d in range(discretization.domain_dimension):
        strain_vol[d, d, ...] = strain_trace
    strain_dev = total_strain.s - strain_vol
    strain_eq = np.sqrt((2. / 3.) * np.einsum('ij...,ji...->...', strain_dev, strain_dev))

    im0 = axs[0].imshow(strain_eq[0, :, :, mid_z], cmap='viridis')
    axs[0].set_title(f'Equivalent Strain (it {iteration})')
    plt.colorbar(im0, ax=axs[0])

    im1 = axs[1].imshow(stress.s[0, 1, 0, :, :, mid_z], cmap='magma')
    axs[1].set_title('Stress sigma_xy')
    plt.colorbar(im1, ax=axs[1])

    damage_vis = d_field.s[0, 0, 0]  # q=0 slice
    im2 = axs[2].imshow(damage_vis[:, :, mid_z], cmap='Reds', vmin=0, vmax=dmax_val)
    axs[2].set_title('Damage Variable (d)')
    plt.colorbar(im2, ax=axs[2])

    plt.tight_layout()
    if not os.path.exists(figure_folder_path):
        os.makedirs(figure_folder_path)
    plt.savefig(f"{figure_folder_path}iteration_{iteration:02d}.png")
    plt.close()


macro_gradient_inc_field = discretization.get_gradient_size_field(name='macro_gradient_inc_field')
displacement_increment_field = discretization.get_unknown_size_field(name='displacement_increment_field')

strain_fluc_field = discretization.get_displacement_gradient_sized_field(name='strain_fluctuation_field')
total_strain_field = discretization.get_displacement_gradient_sized_field(name='strain_field')
rhs_field = discretization.get_unknown_size_field(name='rhs_field')

stress_field = discretization.get_displacement_gradient_sized_field(name='stress_field')
K4_ijklqyz = discretization.get_material_data_size_field_mugrid(name='K4_ijklqxyz')

x = np.linspace(start=0, stop=domain_size[0], num=number_of_pixels[0])
y = np.linspace(start=0, stop=domain_size[1], num=number_of_pixels[1])
X, Y = np.meshgrid(x, y, indexing='ij')

# set macroscopic loading increment
ninc = 10
_info['ninc'] = ninc

macro_gradient_inc = np.zeros(shape=(3, 3))
macro_gradient_inc[0, 1] += 0.05 / float(ninc)
macro_gradient_inc[1, 0] += 0.05 / float(ninc)
dt = 1. / float(ninc)

discretization.get_macro_gradient_field_mugrid(macro_gradient_ij=macro_gradient_inc,
                                               macro_gradient_field_ijqxyz=macro_gradient_inc_field)

# initial constitutive evaluation (dt used but damage=0 so harmless)
constitutive(total_strain_field, stress_field, K4_ijklqyz, dt)

if save_results:
    results_name = (f'init_K')
    save_npy(fn=data_folder_path + results_name + f'.npy',
             data=K4_ijklqyz.s[0, 0, 0, 0].mean(axis=0),
             subdomain_locations=tuple(discretization.subdomain_locations_no_buffers),
             nb_grid_pts=tuple(discretization.nb_of_pixels_global),
             components_are_leading=True,
             comm=MPI.COMM_WORLD)

preconditioner = discretization.get_preconditioner_Green_mugrid(
    reference_material_data_ijkl=I4s)


def M_fun_Green(x, Px):
    discretization.fft.communicate_ghosts(x)
    discretization.apply_preconditioner_mugrid(preconditioner_Fourier_fnfnqks=preconditioner,
                                               input_nodal_field_fnxyz=x,
                                               output_nodal_field_fnxyz=Px)


sum_CG_its = 0
sum_Newton_its = 0
start_time = time.time()
iteration_total = 0

for inc in range(ninc):
    if discretization.communicator.rank == 0:
        print(f'Increment {inc}')
        print(f'==========================================================================')

    total_strain_field.s[...] += macro_gradient_inc_field.s[...]

    constitutive(total_strain_field, stress_field, K4_ijklqyz, dt)

    discretization.fft.communicate_ghosts(stress_field)
    discretization.apply_gradient_transposed_operator_mugrid(gradient_field_ijqxyz=stress_field,
                                                             div_u_fnxyz=rhs_field,
                                                             apply_weights=True)
    rhs_field.s[...] *= -1

    plot_results(iteration_total, total_strain_field, stress_field, discretization)

    if save_results:
        results_name = (f'K4_ijklqyz_it{iteration_total}')
        save_npy(data_folder_path + results_name + f'.npy', K4_ijklqyz.s[0, 0, 0, 0].mean(axis=0),
                 tuple(discretization.subdomain_locations_no_buffers),
                 tuple(discretization.nb_of_pixels_global), MPI.COMM_WORLD)

    En = np.sqrt(
        discretization.communicator.sum(np.dot(total_strain_field.s.ravel(), total_strain_field.s.ravel())))

    iiter = 0

    norm_rhs = np.sqrt(discretization.communicator.sum(np.dot(rhs_field.s.ravel(), rhs_field.s.ravel())))
    if discretization.communicator.rank == 0:
        print('Rhs at new laod step {0:10.2e}'.format(norm_rhs))
        print('En at new laod step {0:10.2e}'.format(En))

    while True:
        if preconditioner_type == 'Green':
            M_fun = M_fun_Green
        elif preconditioner_type == 'Green_Jacobi':
            K_diag_alg = discretization.get_preconditioner_Jacobi_mugrid(
                material_data_field_ijklqxyz=K4_ijklqyz, formulation=formulation)

            def M_fun_Jacobi(x, Px):
                Px.s[...] = K_diag_alg.s * x.s
                discretization.fft.communicate_ghosts(Px)
                discretization.apply_preconditioner_mugrid(preconditioner_Fourier_fnfnqks=preconditioner,
                                                           input_nodal_field_fnxyz=Px,
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

            if discretization.communicator.rank == 0:
                print(f"{it:5} stop_crit_norm = {stop_crit_norm:.5}")

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
            print(f'nb iteration CG = {nb_it_comb}')
        sum_CG_its += nb_it_comb

        iiter += 1
        sum_Newton_its += 1
        iteration_total += 1

        discretization.apply_gradient_operator_symmetrized_mugrid(
            u_inxyz=displacement_increment_field,
            grad_u_ijqxyz=strain_fluc_field)

        total_strain_field.s[...] += strain_fluc_field.s[...]
        constitutive(total_strain_field, stress_field, K4_ijklqyz, dt)

        discretization.apply_gradient_transposed_operator_mugrid(gradient_field_ijqxyz=stress_field,
                                                                 div_u_fnxyz=rhs_field,
                                                                 apply_weights=True)
        rhs_field.s[...] *= -1

        plot_results(iteration_total, total_strain_field, stress_field, discretization)

        strain_trace = np.einsum('ii...', total_strain_field.s) / 3
        strain_dev_s = total_strain_field.s - (np.eye(discretization.domain_dimension)[:, :, None, None, None, None] * strain_trace)
        strain_eq_all = np.sqrt((2. / 3.) * np.einsum('ij...,ji...->...', strain_dev_s, strain_dev_s))
        max_strain_eq = np.max(strain_eq_all)
        max_damage = np.max(d_field.s)

        if save_results:
            results_name = (f'K4_ijklqyz_it{iteration_total}')
            save_npy(data_folder_path + results_name + f'.npy', K4_ijklqyz.s[0, 0, 0, 0].mean(axis=0),
                     tuple(discretization.subdomain_locations_no_buffers),
                     tuple(discretization.nb_of_pixels_global), MPI.COMM_WORLD)

        norm_rhs = np.sqrt(discretization.communicator.sum(
            np.dot(rhs_field.s.ravel(), rhs_field.s.ravel())))
        norm_strain_fluc = np.sqrt(discretization.communicator.sum(
            np.dot(strain_fluc_field.s.ravel(), strain_fluc_field.s.ravel())))

        if discretization.communicator.rank == 0:
            print('=====================')
            print(f'Max Damage Detected: {max_damage:.4f}')
            print('strain_fluc_field {0:10.2e}'.format(norm_strain_fluc))
            print('norm_rhs {0:10.2e}'.format(norm_rhs))
            print('En {0:10.2e}'.format(En))

        if norm_rhs < 1.e-5 and iiter > 0:
            if discretization.communicator.rank == 0:
                print('Newton-Raphson converged.')
            break
        if iiter == 10:
            break

    end_time = time.time()
    elapsed_time = end_time - start_time

    if discretization.communicator.rank == 0:
        print("number_of_pixels: ", number_of_pixels)
        print(f'Total number of CG {sum_CG_its}')
        print(f'Total number of sum_Newton_its {sum_Newton_its}')
        print("Elapsed time : ", elapsed_time)

    if save_results:
        _info['sum_Newton_its'] = sum_Newton_its
        _info['iteration_total'] = iteration_total
        _info['sum_CG_its'] = sum_CG_its
        _info['elapsed_time'] = elapsed_time
        if MPI.COMM_WORLD.rank == 0:
            np.savez(data_folder_path + f'info_log_final.npz', **_info)

if discretization.communicator.rank == 0:
    end_time = time.time()
    elapsed_time = end_time - start_time
    print("  time: ", elapsed_time)
