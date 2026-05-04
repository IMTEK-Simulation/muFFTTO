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
    prog="exp_paper_JG_softening_damage_JZ_bubles_compute.py",
    description="Solve non-linear elasticity with exponential softening damage (compute part)."
)
parser.add_argument("-n", "--nb_pixel", default="32")
parser.add_argument("-exp0", "--eps0_damage", default="0.05", type=float)
parser.add_argument("-dmax", "--max_damage", default="0.99", type=float)
parser.add_argument(
    "-p", "--preconditioner_type",
    type=str,
    choices=["Green", "Jacobi", "Green_Jacobi"],  # example options
    default="Green_Jacobi",
    help="Type of preconditioner to use"
)

script_name = "exp_paper_JG_softening_damage_JZ_bubles" # Keep original script name for data folders if desired
args = parser.parse_args()
nnn = int(args.nb_pixel)
eps0_val = args.eps0_damage
dmax_val = args.max_damage
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
    print(f'preconditioer {preconditioner_type}')
file_folder_path = os.path.dirname(os.path.realpath(__file__))  # script directory
data_folder_path = (
        file_folder_path + '/exp_data/' + script_name + '/' + f'Nx={Nx}' + f'Ny={Ny}' + f'Nz={Nz}'
        + f'_{preconditioner_type}' + '/')

if discretization.communicator.rank == 0:
    if not os.path.exists(file_folder_path):
        os.makedirs(file_folder_path)
    if not os.path.exists(data_folder_path):
        os.makedirs(data_folder_path)

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

model_parameters_non_linear = {'K': 200,
                               'mu': 100.0,
                               'eps0': eps0_val,
                               'dmax': dmax_val}

model_parameters_linear = {'K': 2,
                           'mu': 1}

_info['model_parameters_non_linear'] = model_parameters_non_linear
_info['model_parameters_linear'] = model_parameters_linear

phase_field = discretization.get_scalar_field(name='phase_field')

# load geometry
results_name = (f'bubbles_' + f'dof={nnn}')
geom_folder_path = file_folder_path + '/exp_data/' + 'exp_paper_JG_nonlinear_elasticity_JZ_bubles_generate_geom/'

inclusions = load_npy(geom_folder_path + results_name + f'.npy',
                      tuple(discretization.fft.subdomain_locations),
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
    # parameters
    K = kwargs['K']
    mu = kwargs['mu']

    # elastic stiffness tensor, and stress response
    tangent_ijklqxyz.s[..., phase_xyz] = (K * II + 2. * mu * I4d)[..., None, None]
    stress_ijqxyz.s[..., phase_xyz] = np.einsum('ijklqx...,lkqx...  ->ijqx...  ',
                                                tangent_ijklqxyz.s[..., phase_xyz],
                                                strain_ijqxyz.s[..., phase_xyz])


###
def softening_damage_q_points(strain_ijqxyz,
                               tangent_ijklqxyz,
                               stress_ijqxyz,
                               phase_xyz,
                               **kwargs):
    """
    Exponential softening damage model.
    Stiffness degrades as equivalent strain increases.
    """
    K = kwargs['K']
    mu = kwargs['mu']
    eps0 = kwargs['eps0']
    dmax = kwargs.get('dmax', 0.99)

    # Volumetric strain (trace / 3)
    strain_trace_qx = np.einsum('ii...', strain_ijqxyz.s[..., phase_xyz]) / 3

    # Volumetric strain tensor
    strain_vol_ijqxyz_s = np.zeros_like(strain_ijqxyz.s[..., phase_xyz])
    for d in np.arange(discretization.domain_dimension):
        strain_vol_ijqxyz_s[d, d] = strain_trace_qx

    # Deviatoric strain
    strain_dev_ijqxyz_s = strain_ijqxyz.s[..., phase_xyz] - strain_vol_ijqxyz_s

    # Equivalent strain
    strain_dev_ddot = np.einsum('ij...,ji...->...', strain_dev_ijqxyz_s, strain_dev_ijqxyz_s)
    strain_eq_qx = np.sqrt((2. / 3.) * strain_dev_ddot)

    # Handle zero strain to avoid division by zero
    eps_small = 1e-15
    strain_eq_safe = np.maximum(strain_eq_qx, eps_small)
    
    # Damage calculation
    damage = dmax * (1.0 - np.exp(-strain_eq_qx / eps0))
    stiffness_retention = 1.0 - damage
    
    # Linear Stress calculation
    sig_vol = 3. * K * strain_vol_ijqxyz_s
    sig_dev = 2. * mu * strain_dev_ijqxyz_s
    total_lin_stress = sig_vol + sig_dev

    # Damaged Stress
    stress_ijqxyz.s[..., phase_xyz] = stiffness_retention * total_lin_stress

    # Algorithmic Tangent calculation
    C_lin = (K * II + 2. * mu * I4d)[..., np.newaxis, np.newaxis]
    term_A = stiffness_retention * C_lin
    
    # Picard iteration (use stiffness_retention * C_lin)
    tangent_ijklqxyz.s[..., phase_xyz] = term_A


def constitutive_q_points(strain_ijqxyz, tangent_ijklqxyz, stress_ijqxyz):
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
                               **model_parameters_non_linear)


def constitutive(strain_ijqxyz,
                 sig_ijqxyz,
                 K4_ijklqxyz):
    constitutive_q_points(strain_ijqxyz=strain_ijqxyz,
                          tangent_ijklqxyz=K4_ijklqxyz,
                          stress_ijqxyz=sig_ijqxyz)


macro_gradient_inc_field = discretization.get_gradient_size_field(name='macro_gradient_inc_field')
displacement_increment_field = discretization.get_unknown_size_field(name='displacement_increment_field')

strain_fluc_field = discretization.get_displacement_gradient_sized_field(name='strain_fluctuation_field')
total_strain_field = discretization.get_displacement_gradient_sized_field(name='strain_field')
rhs_field = discretization.get_unknown_size_field(name='rhs_field')

stress_field = discretization.get_displacement_gradient_sized_field(name='stress_field')
K4_ijklqyz = discretization.get_material_data_size_field_mugrid(name='K4_ijklqxyz')

# evaluate material law
constitutive(total_strain_field, stress_field, K4_ijklqyz)

def save_iteration_results(iteration):
    if not save_results:
        return
    # Save Tangent
    save_npy(data_folder_path + f'K4_ijklqyz_it{iteration}.npy', K4_ijklqyz.s[0, 0, 0, 0].mean(axis=0),
             tuple(discretization.fft.subdomain_locations),
             tuple(discretization.nb_of_pixels_global), MPI.COMM_WORLD)
    # Save Total Strain
    save_npy(data_folder_path + f'strain_it{iteration}.npy', np.ascontiguousarray(total_strain_field.s),
             tuple(discretization.fft.subdomain_locations),
             tuple(discretization.nb_of_pixels_global), MPI.COMM_WORLD)
    # Save Stress
    save_npy(data_folder_path + f'stress_it{iteration}.npy', np.ascontiguousarray(stress_field.s),
             tuple(discretization.fft.subdomain_locations),
             tuple(discretization.nb_of_pixels_global), MPI.COMM_WORLD)

# Initial results
save_iteration_results(0)

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
start_time = time.time()
iteration_total = 0

# incremental loading
for inc in range(ninc):
    if discretization.communicator.rank == 0:
        print(f'Increment {inc}')
        print(f'==========================================================================')

    total_strain_field.s[...] += macro_gradient_inc_field.s[...]

    # evaluate material law
    constitutive(total_strain_field, stress_field, K4_ijklqyz)

    # assembly rhs
    discretization.fft.communicate_ghosts(stress_field)
    discretization.apply_gradient_transposed_operator_mugrid(gradient_field_ijqxyz=stress_field,
                                                             div_u_fnxyz=rhs_field,
                                                             apply_weights=True)
    rhs_field.s[...] *= -1
        
    En = np.sqrt(
        discretization.communicator.sum(np.dot(total_strain_field.s.ravel(), total_strain_field.s.ravel())))

    iiter = 0

    norm_rhs = np.sqrt(discretization.communicator.sum(np.dot(rhs_field.s.ravel(), rhs_field.s.ravel())))
    if discretization.communicator.rank == 0:
        print('Rhs at new load step {0:10.2e}'.format(norm_rhs))
        print('En at new load step {0:10.2e}'.format(En))

    # iterate as long as the iterative update does not vanish
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

        norms = dict()
        norms['residual_rr'] = []

        def callback(it, x, r, p, z, stop_crit_norm):
            global norms
            norm_of_rr = discretization.communicator.sum(np.dot(r.ravel(), r.ravel()))
            norms['residual_rr'].append(norm_of_rr)

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
        constitutive(total_strain_field, stress_field, K4_ijklqyz)

        discretization.apply_gradient_transposed_operator_mugrid(gradient_field_ijqxyz=stress_field,
                                                                 div_u_fnxyz=rhs_field,
                                                                 apply_weights=True)
        rhs_field.s[...] *= -1

        # Detect breaking (max damage)
        strain_trace = np.einsum('ii...', total_strain_field.s) / 3
        strain_dev_s = total_strain_field.s - (np.eye(3)[:, :, None, None, None, None] * strain_trace)
        strain_eq_all = np.sqrt((2. / 3.) * np.einsum('ij...,ji...->...', strain_dev_s, strain_dev_s))
        max_damage = dmax_val * (1.0 - np.exp(-np.max(strain_eq_all) / eps0_val))

        save_iteration_results(iteration_total)

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

        if norm_rhs < 1.e-5 and iiter > 0: break
        if iiter == 10: break

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

end_time = time.time()
elapsed_time = end_time - start_time
print("  time: ", elapsed_time)
