import time
import os
import sys
import argparse

import numpy as np
from mpi4py import MPI
from NuMPI.IO import save_npy

# Add the project root to sys.path relatively
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from muGrid import Solvers
from muFFTTO import domain
from muFFTTO import solvers
from muFFTTO import microstructure_library
from muFFTTO import material_models
from muFFTTO import visualization_utils

# Numerical settings ============================
NX = 128                    # Arbitrary number of pixels (default of -n; NY follows from the cell)
N_inc = 10                  # Number of increment for the applied load
cg_tol = 1e-6               # CG tolerance, ABSOLUTE on ||b - Ax|| (used as atol, rtol = 0)
E_LOAD = 1.0                # scales the total macro load below (finite strain: response is NOT linear in E_LOAD)
plot_load_cases = False      # plot the deformed cell after every load increment
plot_fig4 = True            # plot nominal stress vs nominal strain at the end (comparison with paper Fig. 4)
MATERIAL = 'paper'      # 'paper' = Li & Rudykh (2019): contrast 1000, Lambda/mu 1000
                            # 'simplified' = contrast 10, Lambda/mu 100 (Martin's report: -3.5 % before buckling,
                            #                <= 1 % after, at 128 px; much cheaper CG). NOT the paper's material!
# Load case: Li & Rudykh (2019) Fig. 4 = uniaxial COMPRESSION, lateral side free =============
LOAD_MODE = 'fig4'                              # 'fig4' = uniaxial compression below, 'manual' = your own load_pattern
LOAD_DIRECTION = 'Y'                            # 'X' = Fig. 4(a), 'Y' = Fig. 4(b)
EPS_MAX = {'X': 0.020, 'Y': 0.030}              # final nominal strain = critical strains eps_cr (paper text)
Y_MAX_MPA = {'X': 0.014, 'Y': 0.026}            # y-axis top of the Fig. 4 plot [MPa] (same in the small-strain script)
# Free lateral side, SIMPLE version: lateral macro strain = +nu * compressive strain, with the small-strain
# Poisson ratio printed as "nu = ..." by the Fig. 4 block of the small-strain script.
# Use the value for the SAME NX and element_type as this run!
NU_LAT = {'paper':      {'X': 0.2575, 'Y': 0.2797},   # small strain, NX = 64, bilinear_rectangle, paper material
          'simplified': {'X': 0.3269, 'Y': 0.3933}}[MATERIAL]   # small strain, NX = 64, bilinear, simplified
PAPER_SLOPE_MPA = {'X': {'Exp.': 0.32, 'Sim. periodic': 0.35},   # initial slopes digitised from Fig. 4,
                   'Y': {'Exp.': 0.46, 'Sim. periodic': 0.50}}   # approx. +-5 %
# =============================================================================================
# ====================================================================================================================================================

# ============================================================================
# argument parsing
# ============================================================================
parser = argparse.ArgumentParser(
    prog='exp_finite_strain_2D_NeoHookean.py',
    description='Solve finite strain NeoHookean elasticity in 2D'
)
parser.add_argument('-n',   '--nb_pixel',       default=str(NX))
parser.add_argument('-inc', '--nb_increments',  default=str(N_inc))
parser.add_argument(
    '--save_per_it',
    action='store_false',
    help='Enable saving every iteration'
)


script_name = os.path.splitext(os.path.basename(__file__))[0]
args        = parser.parse_args()
nnn         = int(args.nb_pixel)
ninc        = int(args.nb_increments)
save_per_it =  args.save_per_it
# ============================================================================
# problem setup
# ============================================================================

# Unit Cells input: -- PHYSICAL units ===================
R_MM         = 3.0              # void & inclusion radius r                      [mm]
MU_M_MPA     = 0.6              # matrix shear modulus mu^(m) (FLX9860)           [MPa]
# MU_I_MPA     = 600.0            # stiff inclusion shear modulus mu^(i) (VeroWhite) [MPa]
# LAM_OVER_MU  = 1e3              # Lambda / mu, matrix and inclusion (Sec. 3, after Eq. 1) [-]
MU_I_MPA     = {'paper': 600.0, 'simplified': 6.0}[MATERIAL]     # inclusion shear modulus mu^(i) [MPa] (paper: 600)
LAM_OVER_MU  = {'paper': 1e3,   'simplified': 100.0}[MATERIAL]   # Lambda / mu, matrix and inclusion [-] (paper: 1000)
C_MATRIX     = 0.25             # target matrix volume fraction c^(m) ratio            [-]
SPECIMEN_MM  = (92.37, 91.42)   # specimen width x height (finite sample, check only) [mm]
THICKNESS_MM = 10.0             # out-of-plane thickness -> plane strain, not used in 2D [mm]
MU_V_MPA     = 1e-4 * MU_M_MPA  # void stand-in shear modulus (arbitrary) [MPa]

# Reference scales (nondimensionalisation) ====================================================
L_REF = R_MM        # length scale  [mm]  -> lengths in units of r
S_REF = MU_M_MPA    # stress scale  [MPa] -> stresses / moduli in units of mu^(m)

# Nondimensional quantities used by the solver (derived, do not edit) =========================
c_matrix = C_MATRIX
r = R_MM / L_REF                        # = 1
CONTRAST = MU_I_MPA / S_REF             # = 1000
MU_VOID = MU_V_MPA / S_REF              # = 1e-4  (finite strain: 0 is NOT allowed, neo-Hookean needs mu > 0)
incl_vol_frac = 1.0 - c_matrix          # total circle fraction (voids + stiff)

# Domain: TRIANGULAR lattice (paper Fig. 1a), centre spacing a, periodic cell 2a x sqrt(3)a
a = r * np.sqrt(2 * np.pi / (np.sqrt(3) * incl_vol_frac))
ligament = a - 2 * r                                            # thinnest matrix ligament (void-void)

domain_size      = [2 * a, np.sqrt(3) * a]
number_of_pixels = (nnn, int(round(nnn * domain_size[1] / domain_size[0])))   # square pixels
dim              = len(domain_size)
h_x, h_y = domain_size[0] / number_of_pixels[0], domain_size[1] / number_of_pixels[1]   # actual sizes

if MPI.COMM_WORLD.rank == 0:
    print(f'a = {a * L_REF:.3f} mm | ligament = {ligament * L_REF:.3f} mm | '
          f'pixel = {h_x * L_REF:.4f} x {h_y * L_REF:.4f} mm | '
          f'{ligament / max(h_x, h_y):.1f} px across ligament | grid = {number_of_pixels}')


problem_type        = 'elasticity'
discretization_type = 'finite_element'
element_type        = 'bilinear_rectangle' #
formulation         = 'finite_strain'
preconditioner_type = 'Green'

_info = {
    'problem_type':        problem_type,
    'discretization_type': discretization_type,
    'element_type':        element_type,
    'formulation':         formulation,
    'preconditioner_type': preconditioner_type,
    'nb_of_pixels':        number_of_pixels,
    'domain_size':         domain_size,
    'ninc':                ninc,
}

# ============================================================================
# discretization
# ============================================================================
my_cell = domain.PeriodicUnitCell(domain_size=domain_size,
                                  problem_type=problem_type)

discretization = domain.Discretization(cell=my_cell,
                                       nb_of_pixels_global=number_of_pixels,
                                       discretization_type=discretization_type,
                                       element_type=element_type)

# ============================================================================
# output folders
# ============================================================================
file_folder_path   = os.path.dirname(os.path.realpath(__file__))
data_folder_path   = (file_folder_path + '/exp_data/' + script_name + '/'
                      + f'Nx={number_of_pixels[0]}Ny={number_of_pixels[1]}_{preconditioner_type}_{LOAD_DIRECTION}_{MATERIAL}/')
figure_folder_path = (file_folder_path + '/figures/' + script_name + '/'
                      + f'Nx={number_of_pixels[0]}Ny={number_of_pixels[1]}_{preconditioner_type}_{LOAD_DIRECTION}_{MATERIAL}/')

if discretization.communicator.rank == 0:
    os.makedirs(data_folder_path,   exist_ok=True)
    os.makedirs(figure_folder_path, exist_ok=True)

# ============================================================================
# material parameters
# ============================================================================
# matrix: soft neo-Hookean
# nondimensional Lame parameters (plane strain uses the 3D lambda), same as the small-strain script
# matrix: soft neo-Hookean (FLX9860)
mu_matrix  = 1.0
lam_matrix = LAM_OVER_MU * mu_matrix

# inclusion: stiff neo-Hookean (VeroWhite)
mu_inc  = CONTRAST
lam_inc = LAM_OVER_MU * mu_inc

# void stand-in (compressible)
mu_void  = MU_VOID
lam_void = 0.0

# reference material for Green preconditioner
i    = np.eye(dim)
II   = np.einsum('ij,kl->ijkl', i, i)
I4rt = np.einsum('ik,jl->ijkl', i, i)
I4s  = 0.5 * (I4rt + np.einsum('il,jk->ijkl', i, i))
I4d  = I4s - II / 2.0       # 2D: divide by dim=2 not 3
ref_mat = lam_matrix * II + 2.0 * mu_matrix * I4s

_info['lam_matrix'] = lam_matrix
_info['mu_matrix']  = mu_matrix
_info['lam_inc']    = lam_inc
_info['mu_inc']     = mu_inc
_info['mu_void']    = mu_void
_info['E_LOAD']     = E_LOAD
_info['cg_tol']     = cg_tol

# ============================================================================
# geometry — Composite A (0 = matrix, 1 = void, 2 = stiff inclusion)
# ============================================================================
geometry_ID = 'geometry_stefanus'

phase_field = discretization.get_scalar_field(name='phase_field')
phase_field.s[0, 0] = microstructure_library.get_geometry(
    nb_voxels=discretization.nb_of_pixels,
    microstructure_name=geometry_ID,
    coordinates=discretization.fft.coords,
    vol_frac=incl_vol_frac,
    domain_size=domain_size
)

matrix_mask = phase_field.s[0, 0] == 0
void_mask   = phase_field.s[0, 0] == 1
inc_mask    = phase_field.s[0, 0] == 2

# ============================================================================
# material fields
# ============================================================================
lam_field = discretization.get_quad_field_scalar(name='lam_field')
mu_field  = discretization.get_quad_field_scalar(name='mu_field')

lam_field.s[0, 0, :, matrix_mask] = lam_matrix
lam_field.s[0, 0, :, void_mask]   = lam_void
lam_field.s[0, 0, :, inc_mask]    = lam_inc
mu_field.s[0, 0,  :, matrix_mask] = mu_matrix
mu_field.s[0, 0,  :, void_mask]   = mu_void
mu_field.s[0, 0,  :, inc_mask]    = mu_inc

if discretization.communicator.rank == 0:
    print(f'lam min/max : {lam_field.s.min():.4f} / {lam_field.s.max():.4f}')
    print(f'mu  min/max : {mu_field.s.min():.4f}  / {mu_field.s.max():.4f}')
    print(f'material: {MATERIAL} | contrast mu_i/mu_m = {CONTRAST:g} | Lambda/mu = {LAM_OVER_MU:g} | '
          f'NU_LAT = {NU_LAT}')

# ============================================================================
# material model
# ============================================================================
material = material_models.NeoHookean(
    discretization=discretization,
    lam_1qxyz=lam_field,
    mu_1qxyz=mu_field,
    name='neo_hookean_two_phase'
)

# ============================================================================
# fields
# ============================================================================
macro_gradient_inc_field       = discretization.get_gradient_size_field(name='macro_gradient_inc_field')
displacement_fluctuation_field = discretization.get_unknown_size_field(name='displacement_fluctuation_field')
displacement_increment_field   = discretization.get_unknown_size_field(name='displacement_increment_field')
strain_fluc_field              = discretization.get_displacement_gradient_sized_field(name='strain_fluctuation_field')
total_strain_field             = discretization.get_displacement_gradient_sized_field(name='total_strain_field')
stress_field                   = discretization.get_displacement_gradient_sized_field(name='stress_field')
tangent_field                  = discretization.get_material_data_size_field_mugrid(name='tangent_field')
rhs_field                      = discretization.get_unknown_size_field(name='rhs_field')
energy_field                   = discretization.get_quad_field_scalar(name='energy_field')
u_unit_cell                    = discretization.get_unknown_size_field(name='u_unit_cell')

# ============================================================================
# initialize F = I (reference configuration)
# ============================================================================
total_strain_field.s[...] = 0.0
for d in range(dim):
    total_strain_field.s[d, d] = 1.0

# ============================================================================
# preconditioner
# ============================================================================
preconditioner = discretization.get_preconditioner_Green_mugrid(
    reference_material_data_ijkl=ref_mat
)

def M_fun_Green(x, Px):
    discretization.fft.communicate_ghosts(x)
    discretization.apply_preconditioner_mugrid(
        preconditioner_Fourier_fnfnqks=preconditioner,
        input_nodal_field_fnxyz=x,
        output_nodal_field_fnxyz=Px
    )

# ============================================================================
# macroscopic loading
# # ============================================================================
k, l = (0, 1) if LOAD_DIRECTION == 'X' else (1, 0)   # k = loaded axis, l = free lateral axis
load_pattern       = np.zeros((dim, dim))       # total macro gradient F - I
if LOAD_MODE == 'fig4':
    load_pattern[k, k] = -EPS_MAX[LOAD_DIRECTION]                           # uniaxial compression
    load_pattern[l, l] = +NU_LAT[LOAD_DIRECTION] * EPS_MAX[LOAD_DIRECTION]  # lateral expansion (free side, simple)
else:                                           # 'manual': set any components here (index [i, j] = du_i/dx_j)
    load_pattern[0, 1] = 0.01                   # shear
    load_pattern[0, 0] = 0.04                   # stretch in x
    # load_pattern[1, 1] = 0.05                 # stretch in y
macro_gradient_inc = E_LOAD * load_pattern / float(ninc)

if MPI.COMM_WORLD.rank == 0:
    if LOAD_MODE == 'fig4':
        print(f'load: compression in {LOAD_DIRECTION} up to eps = {E_LOAD * EPS_MAX[LOAD_DIRECTION]} in {ninc} increments '
              f'({E_LOAD * EPS_MAX[LOAD_DIRECTION] / ninc:.2e} per increment), lateral nu = {NU_LAT[LOAD_DIRECTION]}')
    else:
        print(f'load (manual): total F - I =\n{E_LOAD * load_pattern}\nin {ninc} increments, '
              f'largest component per increment = {np.abs(macro_gradient_inc).max():.2e}')

discretization.get_macro_gradient_field_mugrid(
    macro_gradient_ij=macro_gradient_inc,
    macro_gradient_field_ijqxyz=macro_gradient_inc_field
)

_info['macro_gradient_inc']     = macro_gradient_inc
_info['norm_strain_fluc_field'] = []

# initial constitutive evaluation
material.get_stress(total_strain_field, stress_field)
material.get_algorithmic_tangent(total_strain_field, tangent_field)

if discretization.communicator.rank == 0:
    print(f'tangent min/max : {tangent_field.s.min():.4f} / {tangent_field.s.max():.4f}')

# ============================================================================
# incremental loading — Newton-CG loop
# ============================================================================
sum_CG_its      = 0
sum_Newton_its  = 0
iteration_total = 0
start_time      = time.time()

# ====================================================================================================================================================
P_weighted = discretization.get_displacement_gradient_sized_field(name='P_weighted')


def average_P():
    """volume average of the 1st Piola-Kirchhoff stress P (same averaging as get_homogenized_stress_mugrid)"""
    P_weighted.s[...] = stress_field.s[...]
    discretization.apply_quadrature_weights_on_gradient_field_mugrid(grad_field=P_weighted)
    return discretization.mpi_reduction.sum(P_weighted.s, axis=tuple(range(-dim - 1, 0))) / \
        discretization.cell.domain_volume


# history per increment (index 0 = undeformed state)
hist = {'eps_nominal': [0.0], 'P_load': [0.0], 'P_lat': [0.0], 'lateral_stretch': [1.0],
        'H': [np.zeros((dim, dim))], 'P': [np.zeros((dim, dim))]}   # full macro F - I and average P# ====================================================================================================================================================

for inc in range(ninc):
    if discretization.communicator.rank == 0:
        print(f'Increment {inc}')
        print('=' * 70)

    # apply macroscopic deformation gradient increment
    total_strain_field.s[...] += macro_gradient_inc_field.s[...]

    # constitutive response at new F
    material.get_stress(total_strain_field, stress_field)
    material.get_algorithmic_tangent(total_strain_field, tangent_field)

    # assemble rhs = -div(P)
    discretization.fft.communicate_ghosts(stress_field)
    discretization.apply_gradient_transposed_operator_mugrid(
        gradient_field_ijqxyz=stress_field,
        div_u_fnxyz=rhs_field,
        apply_weights=True
    )
    rhs_field.s[...] *= -1

    En = np.sqrt(discretization.communicator.sum(
        np.dot(total_strain_field.s.ravel(), total_strain_field.s.ravel())
    ))
    norm_rhs_0 = np.sqrt(discretization.communicator.sum(
        np.dot(rhs_field.s.ravel(), rhs_field.s.ravel())
    ))

    if discretization.communicator.rank == 0:
        print(f'Rhs at new load step    {norm_rhs_0:10.2e}')
        print(f'En  at new load step    {En:10.2e}')

    # ------------------------------------------------------------------
    # Newton loop
    # ------------------------------------------------------------------
    iiter    = 0
    norm_rhs = norm_rhs_0

    while True:

        def K_fun(x, Ax):
            discretization.apply_system_matrix_mugrid(
                material_data_field=tangent_field,
                input_field_inxyz=x,
                output_field_inxyz=Ax,
                formulation=formulation
            )
            discretization.fft.communicate_ghosts(Ax)

        norms = {'residual_rr': [], 'residual_rz': []}

        def callback(it, fields):                   # muGrid callback: fields['rr'] = ||r||^2
            norms['residual_rr'].append(fields['rr'])

        displacement_increment_field.s.fill(0)

        Solvers.conjugate_gradients(
            comm=discretization.communicator,
            fc=discretization.field_collection,
            hessp=K_fun,
            b=rhs_field,
            x=displacement_increment_field,
            prec=M_fun_Green,
            rtol=1e-3, atol=cg_tol,
            maxiter=10000,
            callback=callback,
        )

        nb_it_cg    = len(norms['residual_rr'])
        sum_CG_its += nb_it_cg
        iiter      += 1
        sum_Newton_its  += 1
        iteration_total += 1

        if discretization.communicator.rank == 0:
            print(f'  Newton it {iiter}  |  CG its = {nb_it_cg}')

        # strain from displacement increment
        discretization.apply_gradient_operator_mugrid(
            u_inxyz=displacement_increment_field,
            grad_u_ijqxyz=strain_fluc_field
        )

        norm_strain_fluc = np.sqrt(discretization.communicator.sum(
            np.dot(strain_fluc_field.s.ravel(), strain_fluc_field.s.ravel())
        ))

        # update total strain and displacement
        total_strain_field.s[...]             += strain_fluc_field.s[...]
        displacement_fluctuation_field.s[...] += displacement_increment_field.s[...]

        # re-evaluate constitutive response
        material.get_stress(total_strain_field, stress_field)
        material.get_algorithmic_tangent(total_strain_field, tangent_field)
        material.get_energy_density(total_strain_field, energy_field)

        # recompute rhs
        discretization.fft.communicate_ghosts(stress_field)
        discretization.apply_gradient_transposed_operator_mugrid(
            gradient_field_ijqxyz=stress_field,
            div_u_fnxyz=rhs_field,
            apply_weights=True
        )
        rhs_field.s[...] *= -1

        norm_rhs = np.sqrt(discretization.communicator.sum(
            np.dot(rhs_field.s.ravel(), rhs_field.s.ravel())
        ))

        _info['norm_strain_fluc_field'].append(norm_strain_fluc)

        if discretization.communicator.rank == 0:
            print(f'  norm(strain_fluc) / En          {norm_strain_fluc / En:10.2e}')
            print(f'  norm(rhs) / norm(rhs_0)         {norm_rhs / norm_rhs_0:10.2e}')
            print(f'  norm(rhs)                       {norm_rhs:10.2e}')



        # save per iteration
        if save_per_it:
            save_npy(
                data_folder_path + f'displacement_fluctuation_field_it{iteration_total}.npy',
                displacement_fluctuation_field.s.mean(axis=1),
                tuple(discretization.subdomain_locations_no_buffers),
                tuple(discretization.nb_of_pixels_global),
                MPI.COMM_WORLD
            )
            save_npy(
                data_folder_path + f'energy_field_it{iteration_total}.npy',
                energy_field.s[0, 0].mean(axis=0),
                tuple(discretization.subdomain_locations_no_buffers),
                tuple(discretization.nb_of_pixels_global),
                MPI.COMM_WORLD
            )
            save_npy(
                data_folder_path + f'stress_field_it{iteration_total}.npy',
                stress_field.s.mean(axis=2),
                tuple(discretization.subdomain_locations_no_buffers),
                tuple(discretization.nb_of_pixels_global),
                MPI.COMM_WORLD
            )

    # ====================================================================================================================================================
        # convergence check
        if (norm_strain_fluc / En < 1e-8 or norm_rhs / norm_rhs_0 < 1e-6) and iiter > 0:
            break
        if iiter >= 100:
            if discretization.communicator.rank == 0:
                print(f'  WARNING: Newton not converged in increment {inc} (norm(rhs)/norm(rhs_0) = {norm_rhs / norm_rhs_0:.1e})')
            break

    # homogenized stress and macro strain of this converged increment
    total_macro_gradient = (inc + 1) * macro_gradient_inc        # macro F - I
    P_avg = average_P()
    hist['eps_nominal'].append(-total_macro_gradient[k, k])      # compression > 0 (paper convention)
    hist['P_load'].append(P_avg[k, k])
    hist['P_lat'].append(P_avg[l, l])
    hist['lateral_stretch'].append(1.0 + total_macro_gradient[l, l])
    hist['H'].append(total_macro_gradient.copy())
    hist['P'].append(P_avg.copy())
    if discretization.communicator.rank == 0:
        if LOAD_MODE == 'fig4':
            print(f'  eps = {hist["eps_nominal"][-1]:.5f} | nominal stress = {-P_avg[k, k] * S_REF:.6f} MPa | '
                  f'lateral stress = {P_avg[l, l] * S_REF:+.2e} MPa ({abs(P_avg[l, l] / P_avg[k, k]):.2%} of axial)')
        else:
            print(f'  F - I = {np.array2string(total_macro_gradient.ravel(), precision=5)} | '
                  f'P [MPa] = {np.array2string(P_avg.ravel() * S_REF, precision=6)}  (order 00, 01, 10, 11)')    # ====================================================================================================================================================

    if plot_load_cases and discretization.communicator.size == 1:
        # Plot the first two components of the solution field
        # Calculate total macroscopic gradient (F - I)
        total_macro_gradient = (inc + 1) * macro_gradient_inc
        # visualization_utils works on a unit cell [0,1]^2 -> same deformation in unit-cell coordinates
        L_cell = np.diag(domain_size)
        u_unit_cell.s[0] = displacement_fluctuation_field.s[0] / domain_size[0]
        u_unit_cell.s[1] = displacement_fluctuation_field.s[1] / domain_size[1]
        x_plot_ixyz = visualization_utils.get_deformed_grid_coords_two_dim(discretization,
                                                                           macro_gradient_ij=np.linalg.inv(L_cell) @ total_macro_gradient @ L_cell,
                                                                           displacement_fluctuation=u_unit_cell)
        visualization_utils.plot_field_on_grid(
            coordinates_for_plot=x_plot_ixyz,
            field_to_plot=phase_field.s[0, 0],
            name=(f'compression {LOAD_DIRECTION}: eps = {hist["eps_nominal"][-1]:.4f} ({inc + 1}/{ninc})'
                  if LOAD_MODE == 'fig4' else f'manual load, increment {inc + 1}/{ninc}'))
# ============================================================================
# timing and summary
# ============================================================================
end_time     = time.time()
elapsed_time = end_time - start_time

_info['sum_Newton_its']  = sum_Newton_its
_info['sum_CG_its']      = sum_CG_its
_info['iteration_total'] = iteration_total
_info['elapsed_time']    = elapsed_time

if discretization.communicator.rank == 0:
    print('=' * 70)
    print(f'element_type     : {element_type}')
    print(f'number_of_pixels : {number_of_pixels}')
    print(f'preconditioner   : {preconditioner_type}')
    print(f'Total CG its     : {sum_CG_its}')
    print(f'Total Newton its : {sum_Newton_its}')
    print(f'Elapsed time     : {elapsed_time:.2f} s  ({elapsed_time / 60:.2f} min)')
    np.savez(data_folder_path + 'info_log_final.npz', **_info)

    # ------------------------------------------------------------------------
    # all load modes: full macro F - I and average P per increment -> CSV
    # ------------------------------------------------------------------------
    H_all = np.array(hist['H']).reshape(-1, dim * dim)
    P_all = np.array(hist['P']).reshape(-1, dim * dim) * S_REF
    fname_all = data_folder_path + f'stress_strain_{LOAD_MODE}.csv'
    np.savetxt(fname_all, np.column_stack([np.arange(len(H_all)), H_all, P_all]), delimiter=',', fmt='%.8f',
               comments='', header='increment,H00,H01,H10,H11,P00 [MPa],P01 [MPa],P10 [MPa],P11 [MPa]')
    print(f'saved {fname_all}   (H = F - I macro, P = 1st Piola-Kirchhoff average)')

if discretization.communicator.rank == 0 and LOAD_MODE == 'fig4':
    # ------------------------------------------------------------------------
    # Fig. 4 data: nominal strain / nominal stress (compression > 0) -> CSV
    # ------------------------------------------------------------------------
    eps_nom    = np.array(hist['eps_nominal'])
    stress_MPa = -np.array(hist['P_load']) * S_REF               # nominal stress = -P_kk (1st Piola-Kirchhoff)
    lat_MPa    = np.array(hist['P_lat']) * S_REF                 # should be ~0 (free lateral side)
    stretch    = np.array(hist['lateral_stretch'])
    nu_nom     = np.zeros_like(eps_nom)
    nu_nom[1:] = (stretch[1:] - 1.0) / eps_nom[1:]               # nominal Poisson ratio (here = NU_LAT by construction)
    fname = data_folder_path + f'Fig4_finite_strain_{LOAD_DIRECTION}.csv'
    np.savetxt(fname, np.column_stack([eps_nom, stress_MPa, lat_MPa, stretch, nu_nom]),
               delimiter=',', fmt='%.8f', comments='',
               header=f'nominal strain eps_{LOAD_DIRECTION} [-],nominal stress [MPa],lateral stress [MPa],'
                      f'lateral stretch [-],nominal Poisson ratio [-]')
    print(f'saved {fname}')
    print(f'initial slope = {stress_MPa[1] / eps_nom[1]:.4f} MPa | secant slope at eps = {eps_nom[-1]:.4f}: '
          f'{stress_MPa[-1] / eps_nom[-1]:.4f} MPa | max |lateral / axial stress| = '
          f'{np.max(np.abs(lat_MPa[1:] / stress_MPa[1:])):.2%}')

    if plot_fig4:
        from matplotlib import pyplot as plt
        fig, ax = plt.subplots(figsize=(5, 4.2))
        ax.plot(eps_nom, stress_MPa, 'o-', color='tab:green', ms=3,
                label=f'muFFTTO finite strain ({number_of_pixels[0]}x{number_of_pixels[1]}, {element_type}, {MATERIAL})')
        for (name, slope), style in zip(PAPER_SLOPE_MPA[LOAD_DIRECTION].items(), ['k-', 'r-.']):
            ax.plot(eps_nom, slope * eps_nom, style, lw=1, label=f'paper {name} (initial slope, digitised)')
        ax.set_xlim(0, EPS_MAX[LOAD_DIRECTION])
        ax.xaxis.set_major_locator(plt.MaxNLocator(5))                       # 0, 0.004, ..., 0.020 (as small strain)
        ax.yaxis.set_major_locator(plt.MultipleLocator(0.002))               # y ticks every 0.002 MPa (as small strain)
        ax.set_ylim(0, max(Y_MAX_MPA[LOAD_DIRECTION], 1.05 * stress_MPa.max()))   # fixed top, grows if the curve is higher
        ax.set_xlabel(rf'Nominal strain, $\varepsilon_{LOAD_DIRECTION}$')
        ax.set_ylabel('Nominal stress (MPa)')
        ax.set_title(f'Fig. 4{"a" if LOAD_DIRECTION == "X" else "b"}: uniaxial compression in {LOAD_DIRECTION} '
                     f'(up to eps_cr)', fontsize=10)
        ax.legend(fontsize=7, loc='upper left')
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(figure_folder_path + f'Fig4_finite_strain_{LOAD_DIRECTION}.png', dpi=200)
        plt.show()