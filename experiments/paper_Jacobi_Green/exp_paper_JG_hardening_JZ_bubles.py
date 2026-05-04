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
    prog="exp_paper_JG_hardening_JZ_bubles.py",
    description="Solve non-linear elasticity with power-law hardening."
)
parser.add_argument("-n", "--nb_pixel", default="16")
parser.add_argument("-alpha", "--alpha_hardening", default="10.0", type=float)
parser.add_argument("-n_hard", "--exponent_hardening", default="2.0", type=float)
parser.add_argument(
    "-p", "--preconditioner_type",
    type=str,
    choices=["Green", "Jacobi", "Green_Jacobi"],
    default="Green",
    help="Type of preconditioner to use"
)

script_name = os.path.splitext(os.path.basename(__file__))[0]
args = parser.parse_args()
nnn = int(args.nb_pixel)
alpha_val = args.alpha_hardening
n_hard_val = args.exponent_hardening
preconditioner_type = args.preconditioner_type

file_folder_path = os.path.dirname(os.path.realpath(__file__))
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
    print(f'preconditioner {preconditioner_type}')

data_folder_path = (
        file_folder_path + '/exp_data/' + script_name + '/' + f'Nx={Nx}' + f'Ny={Ny}' + f'Nz={Nz}'
        + f'_{preconditioner_type}' + '/')
figure_folder_path = (file_folder_path + '/figures/' + script_name + '/' f'Nx={Nx}' + f'Ny={Ny}' + f'Nz={Nz}'
                      + f'_{preconditioner_type}' + '/')

if discretization.communicator.rank == 0:
    if not os.path.exists(data_folder_path):
        os.makedirs(data_folder_path, exist_ok=True)
    if not os.path.exists(figure_folder_path):
        os.makedirs(figure_folder_path, exist_ok=True)

_info['nb_of_pixels'] = discretization.nb_of_pixels_global
_info['domain_size'] = domain_size

# Identity tensors
i = np.eye(discretization.domain_dimension)
I4s = (np.einsum('il,jk', i, i) + np.einsum('ik,jl', i, i)) / 2.
II = np.einsum('ij,kl->ijkl', i, i)
I4d = I4s - II / 3.

model_parameters_non_linear = {'K': 2.0,
                               'mu0': 1.0,
                               'alpha': alpha_val,
                               'eps0': 0.05,
                               'n': n_hard_val}

model_parameters_linear = {'K': 2.0,
                           'mu': 1.0}

_info['model_parameters_non_linear'] = model_parameters_non_linear
_info['model_parameters_linear'] = model_parameters_linear

# Load geometry
results_name = (f'bubbles_' + f'dof={nnn}')
geom_folder_path = file_folder_path + '/exp_data/' + 'exp_paper_JG_nonlinear_elasticity_JZ_bubles_generate_geom/'

inclusions = load_npy(geom_folder_path + results_name + f'.npy',
                      tuple(discretization.subdomain_locations_no_buffers),
                      tuple(discretization.nb_of_pixels), MPI.COMM_WORLD)

matrix_mask = inclusions > 0
inc_mask = inclusions == 0
del inclusions

def linear_elastic_q_points(strain_ijqxyz, tangent_ijklqxyz, stress_ijqxyz, phase_xyz, **kwargs):
    K = kwargs['K']
    mu = kwargs['mu']
    tangent_ijklqxyz.s[..., phase_xyz] = (K * II + 2. * mu * I4d)[..., None, None]
    stress_ijqxyz.s[..., phase_xyz] = np.einsum('ijklqx...,lkqx...->ijqx...',
                                                tangent_ijklqxyz.s[..., phase_xyz],
                                                strain_ijqxyz.s[..., phase_xyz])

def power_law_hardening_q_points(strain_ijqxyz, tangent_ijklqxyz, stress_ijqxyz, phase_xyz, **kwargs):
    """
    Power-law hardening constitutive model.
    mu(eps_eq) = mu0 * (1 + alpha * (eps_eq/eps0)^n)
    """
    K = kwargs['K']
    mu0 = kwargs['mu0']
    alpha = kwargs['alpha']
    eps0 = kwargs['eps0']
    n = kwargs['n']

    # Volumetric strain
    strain_trace_qx = np.einsum('ii...', strain_ijqxyz.s[..., phase_xyz]) / 3.
    strain_vol_s = np.zeros_like(strain_ijqxyz.s[..., phase_xyz])
    for d in range(discretization.domain_dimension):
        strain_vol_s[d, d] = strain_trace_qx

    # Deviatoric strain
    strain_dev_s = strain_ijqxyz.s[..., phase_xyz] - strain_vol_s

    # Equivalent strain
    strain_dev_ddot = np.einsum('ij...,ji...->...', strain_dev_s, strain_dev_s)
    strain_eq_qx = np.sqrt((2. / 3.) * strain_dev_ddot)
    
    eps_small = 1e-15
    strain_eq_safe = np.maximum(strain_eq_qx, eps_small)

    # Current shear modulus
    ratio = strain_eq_qx / eps0
    mu_curr = mu0 * (1.0 + alpha * ratio**n)

    # Stress
    sig_vol = 3. * K * strain_vol_s
    sig_dev = 2. * mu_curr * strain_dev_s
    stress_ijqxyz.s[..., phase_xyz] = sig_vol + sig_dev

    # Tangent
    # Term 1: 3*K*I_vol + 2*mu_curr*I_dev
    C_linear_part = (K * II + 2. * mu_curr * I4d)[..., np.newaxis, np.newaxis]
    
    # Term 2: 2 * strain_dev \otimes d(mu)/d(eps)
    # d(mu)/d(eps_eq) = mu0 * alpha * n * (eps_eq/eps0)^(n-1) / eps0
    dmu_deq = mu0 * alpha * n * (ratio**(n-1.)) / eps0
    # deq/deps = (2/3) * eps_dev / eps_eq
    deq_deps = (2./3.) * strain_dev_s / strain_eq_safe
    
    term_B = 2.0 * dmu_deq * np.einsum('ij...,kl...->ijkl...', strain_dev_s, deq_deps)
    
    tangent_ijklqxyz.s[..., phase_xyz] = C_linear_part + term_B

def constitutive(strain_ijqxyz, sig_ijqxyz, K4_ijklqxyz):
    global matrix_mask, inc_mask
    linear_elastic_q_points(strain_ijqxyz=strain_ijqxyz,
                            tangent_ijklqxyz=K4_ijklqxyz,
                            stress_ijqxyz=sig_ijqxyz,
                            phase_xyz=matrix_mask,
                            **model_parameters_linear)
    power_law_hardening_q_points(strain_ijqxyz=strain_ijqxyz,
                                 tangent_ijklqxyz=K4_ijklqxyz,
                                 stress_ijqxyz=sig_ijqxyz,
                                 phase_xyz=inc_mask,
                                 **model_parameters_non_linear)

# Fields setup
macro_gradient_inc_field = discretization.get_gradient_size_field(name='macro_gradient_inc_field')
displacement_increment_field = discretization.get_unknown_size_field(name='displacement_increment_field')
strain_fluc_field = discretization.get_displacement_gradient_sized_field(name='strain_fluctuation_field')
total_strain_field = discretization.get_displacement_gradient_sized_field(name='strain_field')
rhs_field = discretization.get_unknown_size_field(name='rhs_field')
stress_field = discretization.get_displacement_gradient_sized_field(name='stress_field')
K4_ijklqyz = discretization.get_material_data_size_field_mugrid(name='K4_ijklqxyz')

# Initial evaluation
constitutive(total_strain_field, stress_field, K4_ijklqyz)

# Macro loading
ninc = 1
macro_gradient_inc = np.zeros((3, 3))
macro_gradient_inc[0, 1] = 0.05
macro_gradient_inc[1, 0] = 0.05
discretization.get_macro_gradient_field_mugrid(macro_gradient_ij=macro_gradient_inc,
                                               macro_gradient_field_ijqxyz=macro_gradient_inc_field)

preconditioner = discretization.get_preconditioner_Green_mugrid(reference_material_data_ijkl=I4s)

def M_fun_Green(x, Px):
    discretization.fft.communicate_ghosts(x)
    discretization.apply_preconditioner_mugrid(preconditioner_Fourier_fnfnqks=preconditioner,
                                               input_nodal_field_fnxyz=x,
                                               output_nodal_field_fnxyz=Px)

sum_CG_its = 0
sum_Newton_its = 0
iteration_total = 0

for inc in range(ninc):
    if discretization.communicator.rank == 0:
        print(f'Increment {inc}')
    
    total_strain_field.s[...] += macro_gradient_inc_field.s[...]
    constitutive(total_strain_field, stress_field, K4_ijklqyz)

    discretization.fft.communicate_ghosts(stress_field)
    discretization.apply_gradient_transposed_operator_mugrid(gradient_field_ijqxyz=stress_field,
                                                             div_u_fnxyz=rhs_field,
                                                             apply_weights=True)
    rhs_field.s[...] *= -1

    En = np.sqrt(discretization.communicator.sum(np.dot(total_strain_field.s.ravel(), total_strain_field.s.ravel())))
    iiter = 0

    while True:
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

        norms = {'residual_rr': []}
        def callback(it, x, r, p, z, stop_crit_norm):
            norms['residual_rr'].append(discretization.communicator.sum(np.dot(r.ravel(), r.ravel())))
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
            tol=1e-5,
            maxiter=1000,
            callback=callback,
            rtol=False
        )

        sum_CG_its += len(norms['residual_rr'])
        iiter += 1
        iteration_total += 1

        discretization.apply_gradient_operator_symmetrized_mugrid(
            u_inxyz=displacement_increment_field,
            grad_u_ijqxyz=strain_fluc_field)

        total_strain_field.s[...] += strain_fluc_field.s[...]
        constitutive(total_strain_field, stress_field, K4_ijklqyz)

        discretization.apply_gradient_transposed_operator_mugrid(gradient_field_ijqxyz=stress_field,
                                                                 div_u_fnxyz=rhs_field,
                                                                 apply_weights=True)
        rhs_field.s[...] *= -1

        norm_rhs = np.sqrt(discretization.communicator.sum(np.dot(rhs_field.s.ravel(), rhs_field.s.ravel())))
        if discretization.communicator.rank == 0:
            print(f'Newton iteration {iiter}, norm_rhs {norm_rhs:.2e}')

        if norm_rhs < 1.e-5 or iiter >= 10:
            break

    if discretization.communicator.rank == 0:
        print(f"Total CG: {sum_CG_its}, Newton its: {iiter}")

if discretization.communicator.rank == 0:
    print(f"Finished. Total time: {time.time() - start_time:.2f}s")
