import sys
import os
import argparse

from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import numpy as np
import time
from mpi4py import MPI

from muGrid import Solvers

from muFFTTO import domain
from muFFTTO import microstructure_library
from muFFTTO import material_models
from muFFTTO import visualization_utils

problem_type = 'elasticity' #or conductivity
discretization_type = 'finite_element'
element_type = 'bilinear_rectangle'# 'biquadratic_rectangle'#'linear_triangles'
formulation = 'small_strain'

# Numerical settings ============================
NX = 128                # Arbitrary number of pixels
cg_tol = 1e-6           # CG tolerance (relative)
plot_load_cases = False  # example's plots assume a [0,1] domain -> distorted for this cell
# ===============================================

# Unit Cells input: -- PHYSICAL units ===================
R_MM         = 3.0              # void & inclusion radius r                      [mm]
MU_M_MPA     = 0.6              # matrix shear modulus mu^(m) (FLX9860)           [MPa]
MU_I_MPA     = 600.0            # stiff inclusion shear modulus mu^(i) (VeroWhite) [MPa]
LAM_OVER_MU  = 1e3              # Lambda / mu, matrix and inclusion (Sec. 3, after Eq. 1) [-]
C_MATRIX     = 0.25              # target matrix volume fraction c^(m) ratio            [-]
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
MU_VOID = 0. # MU_V_MPA / S_REF         # = 1e-4
incl_vol_frac = 1.0 - c_matrix          # total circle fraction (voids + stiff)

# Domain: TRIANGULAR lattice (paper Fig. 1a), centre spacing a, periodic cell 2a x sqrt(3)a
a = r * np.sqrt(2 * np.pi / (np.sqrt(3) * incl_vol_frac))
ligament = a - 2 * r                                            # thinnest matrix ligament (void-void)

# Mesh RESOLUTION: defined on the LIGAMENT (thinnest feature, paper Fig. 1a), not on the cell.
# "mesh sensitive analysis" (Sec. 3). Command-line options are used by that driver script.
parser = argparse.ArgumentParser()
parser.add_argument('--px_per_ligament', type=int, default=6)   # pixels across ligament a - 2r (numerical choice)
args, _ = parser.parse_known_args()
PX_PER_LIGAMENT = args.px_per_ligament
h = ligament / PX_PER_LIGAMENT                                  # target pixel size, units of r
# =============================================================================================

domain_size = [2 * a, np.sqrt(3) * a]
# number_of_pixels = (int(round(domain_size[0] / h)), int(round(domain_size[1] / h))) # ->the one with the h = minimum true meaningful proportion
number_of_pixels = (NX, int(round(NX * domain_size[1] / domain_size[0]))) #manually
h_x, h_y = domain_size[0] / number_of_pixels[0], domain_size[1] / number_of_pixels[1]   # actual sizes

if MPI.COMM_WORLD.rank == 0:
    print(f'a = {a * L_REF:.3f} mm | ligament = {ligament * L_REF:.3f} mm | '
          f'pixel = {h_x * L_REF:.4f} x {h_y * L_REF:.4f} mm | '
          f'{ligament / max(h_x, h_y):.1f} px across ligament | grid = {number_of_pixels}')

my_cell = domain.PeriodicUnitCell(domain_size=domain_size,
                                  problem_type=problem_type)

discretization = domain.Discretization(cell=my_cell,
                                       nb_of_pixels_global=number_of_pixels,
                                       discretization_type=discretization_type,
                                       element_type=element_type)
start_time = time.time()
print(f'{MPI.COMM_WORLD.rank:6} {MPI.COMM_WORLD.size:6} {str(discretization.fft.nb_domain_grid_pts):>15} '
      f'{str(discretization.fft.nb_subdomain_grid_pts):>15} {str(discretization.fft.subdomain_locations):>15}')

# material distribution
geometry_ID = 'geometry_stefanus'

phase_field = discretization.get_scalar_field(name='phase_field')
phase_field.s[0, 0] = microstructure_library.get_geometry(nb_voxels=discretization.nb_of_pixels,
                                                          microstructure_name=geometry_ID,
                                                          coordinates=discretization.fft.coords,
                                                          vol_frac=incl_vol_frac,
                                                          domain_size=domain_size)

# volume fraction actually resolved on the pixel grid (differs from target on coarse grids)
nb_px_global = np.prod(discretization.nb_of_pixels_global)
vf_soft = discretization.mpi_reduction.sum(phase_field.s[0, 0] == 1) / nb_px_global
vf_stiff = discretization.mpi_reduction.sum(phase_field.s[0, 0] == 2) / nb_px_global

if MPI.COMM_WORLD.rank == 0:
    print(f'target inclusion vol. frac = {incl_vol_frac:.4f} | '
          f'resolved: soft = {vf_soft:.4f}, stiff = {vf_stiff:.4f}, total = {vf_soft + vf_stiff:.4f}')


# PLOTTING Material Structure
# one fixed colour per phase: 0 = matrix, 1 = soft, 2 = stiff
phase_cmap = ListedColormap(['#5aa85a',   # matrix -> green
                             'white',     # soft   -> white
                             '#0040b0'])  # stiff  -> blue
phase_norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5], phase_cmap.N)

# plot ONE unit cell in real units [mm]
Lx_mm, Ly_mm = domain_size[0] * L_REF, domain_size[1] * L_REF
nx, ny = discretization.nb_of_pixels_global
X_plot, Y_plot = np.meshgrid(np.linspace(0, Lx_mm, nx + 1),       # pixel EDGES (pixel i spans [x_i, x_i+h])
                             np.linspace(0, Ly_mm, ny + 1), indexing='ij')

fig, ax = plt.subplots(figsize=(6, 6 * Ly_mm / Lx_mm))
pcm = ax.pcolormesh(X_plot, Y_plot, phase_field.s[0, 0],
                    cmap=phase_cmap, norm=phase_norm, shading='flat')
cbar = fig.colorbar(pcm, ax=ax, ticks=[0, 1, 2])
cbar.ax.set_yticklabels(['matrix', 'void', 'stiff'])

# exact circles (r = R_MM) on top of the pixel map -> visual check of radius and positions
centres_frac = [(0.00, 0.00), (0.25, 0.50), (0.75, 0.50), (0.50, 0.00)]
for fx, fy in centres_frac:
    for sx in (-1, 0, 1):                 # images that cut into the cell
        for sy in (-1, 0, 1):
            ax.add_patch(plt.Circle(((fx + sx) * Lx_mm, (fy + sy) * Ly_mm), R_MM,
                                    fill=False, color='k', lw=0.8, ls='--'))
ax.set_xlim(0, Lx_mm)
ax.set_ylim(0, Ly_mm)
ax.set_aspect('equal')
ax.set_xlabel('x [mm]')
ax.set_ylabel('y [mm]')
ax.set_title(f'Composite A unit cell: {Lx_mm:.2f} x {Ly_mm:.2f} mm\n'
             f'r = {R_MM} mm, a = {a * L_REF:.3f} mm, c(m) = {C_MATRIX}', fontsize=10)
plt.show()

# Plot domain size result
Lx, Ly = domain_size                                   # cell size, units of r
nx, ny = discretization.nb_of_pixels_global
X_plot, Y_plot = np.meshgrid(np.linspace(0, 1, nx + 1),            # pixel EDGES, normalised
                             np.linspace(0, 1, ny + 1), indexing='ij')

fig, ax = plt.subplots(figsize=(6, 6 * Ly / Lx))
pcm = ax.pcolormesh(X_plot, Y_plot, phase_field.s[0, 0],
                    cmap=phase_cmap, norm=phase_norm, shading='flat')
cbar = fig.colorbar(pcm, ax=ax, ticks=[0, 1, 2])
cbar.ax.set_yticklabels(['matrix', 'void', 'stiff'])

# exact circles (radius r) on top of the pixel map. In normalised coordinates a circle has
# # semi-axes r/Lx and r/Ly; with the aspect ratio below it is drawn round again.
# centres_frac = [(0.00, 0.00), (0.25, 0.50), (0.75, 0.50), (0.50, 0.00)]   # as in geometry_stefanus
# for fx, fy in centres_frac:
#     for sx in (-1, 0, 1):                 # periodic images that cut into the cell
#         for sy in (-1, 0, 1):
#             ax.add_patch(Ellipse((fx + sx, fy + sy), width=2 * r / Lx, height=2 * r / Ly,
#                                  fill=False, color='k', lw=0.8, ls='--'))
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_aspect(Ly / Lx)                    # 1 unit in y is drawn Ly/Lx as long as 1 unit in x -> true shape
ax.set_xticks(np.linspace(0, 1, 5))
ax.set_yticks(np.linspace(0, 1, 5))
ax.set_xlabel(r'$x / L_x$')
ax.set_ylabel(r'$y / L_y$')
ax.set_title(f'Composite A unit cell (normalised), true ratio $L_x : L_y$ = 2 : $\\sqrt{{3}}$\n'
             f'$L_x$ = {Lx * L_REF:.2f} mm, $L_y$ = {Ly * L_REF:.2f} mm, '
             f'r = {R_MM} mm, c(m) = {C_MATRIX}', fontsize=10)
plt.show()

matrix_mask = phase_field.s[0, 0] == 0
inc_soft_mask = phase_field.s[0, 0] == 1
inc_stiff_mask = phase_field.s[0, 0] == 2

# initialize material data
# nondimensional Lame parameters, set directly (plane strain uses the 3D lambda)
mu_0, lam_0 = 1.0, LAM_OVER_MU * 1.0                  # matrix
mu_1, lam_1 = MU_VOID, 0.0                            # void stand-in (compressible)
mu_2, lam_2 = CONTRAST, LAM_OVER_MU * CONTRAST        # stiff inclusion
# Sanity check =========================================================
# ---- check against Li & Rudykh (2019), Sec. 2: convert nondimensional values back to physical units ----
R_PHYS_MM = L_REF          # paper: void / inclusion radius r = 3 mm      -> length scale (r = 1 here)
MU_M_PHYS_MPA = S_REF     # paper: matrix shear modulus 0.6 MPa         -> stress scale (mu_0 = 1 here)
SPECIMEN_MM = (92.37, 91.42)   # paper: specimen width x height (finite sample, NOT modelled here)
# Sanity check =========================================================

r_resolved = np.sqrt(vf_stiff * domain_size[0] * domain_size[1] / np.pi)  # 1 stiff circle per cell
if MPI.COMM_WORLD.rank == 0:
    print('--- paper check (physical units) ---')
    print(f'mu_matrix    = {mu_0 * MU_M_PHYS_MPA:8.3f} MPa   (paper 0.6 MPa)')
    print(f'mu_stiff     = {mu_2 * MU_M_PHYS_MPA:8.3f} MPa   (paper 600 MPa, VeroWhite)')
    print(f'c_matrix     = {1 - vf_soft - vf_stiff:8.4f}       (paper 0.25; target here {c_matrix})')
    print(f'radius       = {r * R_PHYS_MM:8.3f} mm    (paper 3 mm); resolved on grid = {r_resolved * R_PHYS_MM:.3f} mm')
    print(f'cell         = {domain_size[0] * R_PHYS_MM:8.3f} mm x {domain_size[1] * R_PHYS_MM:.3f} mm '
          f'(2a x sqrt(3)a, a = {a * R_PHYS_MM:.3f} mm; paper a = 6.598 mm at c_matrix = 0.25)')
    print(f'cells across specimen = {SPECIMEN_MM[0] / (domain_size[0] * R_PHYS_MM):.2f} x '
          f'{SPECIMEN_MM[1] / (domain_size[1] * R_PHYS_MM):.2f}   (paper: 7 x 8 cells = 14 x 16 circles)')

lam_11qxyz = discretization.get_quad_field_scalar(name='lam_first_lame')
mu_11qxyz = discretization.get_quad_field_scalar(name='mu_second_lame')

# apply material distribution
lam_11qxyz.s[..., matrix_mask] = lam_0
lam_11qxyz.s[..., inc_soft_mask] = lam_1
lam_11qxyz.s[..., inc_stiff_mask] = lam_2

mu_11qxyz.s[..., matrix_mask] = mu_0
mu_11qxyz.s[..., inc_soft_mask] = mu_1
mu_11qxyz.s[..., inc_stiff_mask] = mu_2

material = material_models.LinearElastic(discretization=discretization,
                                         lam_1qxyz=lam_11qxyz,
                                         mu_1qxyz=mu_11qxyz,
                                         name='linear_isotropic_elasticity')

material_data_field_C_0 = discretization.get_material_data_size_field_mugrid(name='elastic_tensor')

# populate the field with C_0 material
total_strain_ijqxyz = discretization.get_strain_sized_field(name='total_strain_field')

material.get_algorithmic_tangent(total_strain_ijqxyz, material_data_field_C_0)

def K_fun(x, Ax):

    discretization.apply_system_matrix_mugrid(material_data_field=material_data_field_C_0,
                                              input_field_inxyz=x,
                                              output_field_inxyz=Ax,
                                              formulation='small_strain')
    discretization.fft.communicate_ghosts(Ax)


# reference medium for the preconditioner = matrix
K_0 = lam_0 + 2.0 * mu_0 / 3.0   # bulk modulus as used by get_elastic_material_tensor (3D form)
G_0 = mu_0

# preconditioner
elastic_C_ref = material_models.get_elastic_material_tensor(dim=discretization.domain_dimension,
                                                 K=K_0,
                                                 mu=G_0,
                                                 kind='linear')
preconditioner = discretization.get_preconditioner_Green_mugrid(reference_material_data_ijkl=elastic_C_ref)

def M_fun(x, Px):
    """
    Function to compute the product of the Preconditioner matrix with a vector.
    The Preconditioner is represented by the convolution operator.
    """
    discretization.fft.communicate_ghosts(x)
    discretization.apply_preconditioner_mugrid(preconditioner_Fourier_fnfnqks=preconditioner,
                                               input_nodal_field_fnxyz=x,
                                               output_nodal_field_fnxyz=Px)

# Allocate fields
macro_gradient_field = discretization.get_gradient_size_field(name='macro_gradient_field')
rhs_field = discretization.get_unknown_size_field(name='rhs_field')
solution_field = discretization.get_unknown_size_field(name='solution')


def callback(iteration, fields):
    """
    Callback function to print the current solution, residual, and search direction.
    """
    # norm_of_rr = fields['rr']
    # if discretization.communicator.rank == 0:
    #     print(f"{iteration:5} norm of residual = {norm_of_rr:.5}")

homogenized_stress = {}
dim = discretization.domain_dimension
homogenized_C_ijkl = np.zeros(np.array(4 * [dim, ]))
# compute whole homogenized elastic tangent
for i in range(dim):
    for j in range(dim):
        # set macroscopic gradient (Load)
        macro_gradient_ij = np.zeros([dim, dim])
        macro_gradient_ij[i, j] = 1
        # Set up right hand side
        discretization.get_macro_gradient_field_mugrid(macro_gradient_ij=macro_gradient_ij,
                                                       macro_gradient_field_ijqxyz=macro_gradient_field)
        # Solve mechanical equilibrium constrain
        discretization.get_rhs_mugrid(material_data_field_ijklqxyz=material_data_field_C_0,
                                      macro_gradient_field_ijqxyz=macro_gradient_field,
                                      rhs_inxyz=rhs_field)
        # tol is ABSOLUTE on ||b - Ax||; scale with ||b|| so cg_tol (line 51) acts as a RELATIVE tolerance
        # norm_b = np.sqrt(discretization.communicator.sum(np.dot(rhs_field.s.ravel(), rhs_field.s.ravel())))
        # print(f'Norm of b = {norm_b:.5}')
        Solvers.conjugate_gradients(
            comm=discretization.communicator,
            fc=discretization.field_collection,
            hessp=K_fun,  # linear operator
            b=rhs_field,  # right-hand side
            x=solution_field,
            prec=M_fun,
            tol=cg_tol, #* norm_b,
            maxiter=2000,
            callback=callback)


        if plot_load_cases and discretization.communicator.size == 1:
            # Plot the first two components of the solution field
            try:
                x_plot_ixyz=visualization_utils.get_deformed_grid_coords_two_dim(discretization,
                                                 macro_gradient_ij=macro_gradient_ij,
                                                 displacement_fluctuation=solution_field)

                visualization_utils.plot_field_on_grid(
                    coordinates_for_plot=x_plot_ixyz,
                    field_to_plot=solution_field.s[0, 0],
                    name = fr'$\tilde{{u}}_{{x}}$   ')
                visualization_utils.plot_field_on_grid(
                    coordinates_for_plot=x_plot_ixyz,
                    field_to_plot=solution_field.s[1, 0],
                    name=fr'$\tilde{{u}}_{{y}}$   ')
            except:
                print(f"Plotting failed:  ")
        # ----------------------------------------------------------------------
        # compute homogenized stress field corresponding
        stress_ij = discretization.get_homogenized_stress_mugrid(
            material_data_field_ijklqxyz=material_data_field_C_0,
            displacement_field_inxyz=solution_field,
            macro_gradient_field_ijqxyz=macro_gradient_field,
            formulation='small_strain')
        homogenized_C_ijkl[i, j] = stress_ij             # same as before
        homogenized_stress[(i, j)] = stress_ij           # keep it for printing

fmt = {'float_kind': lambda x: f"{x:0.8f}"}
if MPI.COMM_WORLD.rank == 0:
    print("Homogenized elastic tangent =\n" +
          np.array2string(material_models.compute_Voigt_notation_4order(homogenized_C_ijkl), formatter=fmt))
    for (i, j), stress_ij in homogenized_stress.items():
        print(f"Homogenized stress for unit macro strain E[{i},{j}] = 1  (units of mu_m) =\n" +
              np.array2string(stress_ij, formatter=fmt))

# ============================================================================================
# ONE applied load (shear and/or compression) + deformed plot with muFFTTO's visualization_utils
# ============================================================================================
SHEAR_STRAIN = 0.0      # engineering shear strain gamma            [-]
LOAD_X = 0.0             # normal strain in X (negative = compression) [-]
LOAD_Y = 0.5             # normal strain in Y                          [-]

macro_gradient_ij = np.zeros([dim, dim])
macro_gradient_ij[0, 1] = SHEAR_STRAIN
macro_gradient_ij[0, 0] = LOAD_X
macro_gradient_ij[1, 1] = LOAD_Y

# solve the cell problem for this load (same steps as inside the loop)
discretization.get_macro_gradient_field_mugrid(macro_gradient_ij=macro_gradient_ij,
                                               macro_gradient_field_ijqxyz=macro_gradient_field)
discretization.get_rhs_mugrid(material_data_field_ijklqxyz=material_data_field_C_0,
                              macro_gradient_field_ijqxyz=macro_gradient_field,
                              rhs_inxyz=rhs_field)
solution_field.s.fill(0)
Solvers.conjugate_gradients(comm=discretization.communicator,
                            fc=discretization.field_collection,
                            hessp=K_fun, b=rhs_field, x=solution_field, prec=M_fun,
                            tol=cg_tol, maxiter=5000, callback=callback)

stress_load = discretization.get_homogenized_stress_mugrid(
    material_data_field_ijklqxyz=material_data_field_C_0,
    displacement_field_inxyz=solution_field,
    macro_gradient_field_ijqxyz=macro_gradient_field,
    formulation='small_strain')
if MPI.COMM_WORLD.rank == 0:
    print(f"Applied macro gradient =\n{macro_gradient_ij}\nHomogenized stress (units of mu_m) =\n" +
          np.array2string(stress_load, formatter=fmt) +
          "\n(in MPa) =\n" + np.array2string(stress_load * S_REF, formatter=fmt))

# muFFTTO's visualization_utils works on a UNIT cell [0,1] x [0,1].
# Map the load and the displacement to that unit cell, let muFFTTO deform the grid, map back to mm.
if discretization.communicator.size == 1:
    L_cell = np.diag(domain_size)                                   # cell size, units of r
    E_unit = np.linalg.inv(L_cell) @ macro_gradient_ij @ L_cell     # same deformation in unit-cell coords
    u_unit = discretization.get_unknown_size_field(name='u_unit_cell')
    u_unit.s[0] = solution_field.s[0] / domain_size[0]
    u_unit.s[1] = solution_field.s[1] / domain_size[1]

    x_unit = visualization_utils.get_deformed_grid_coords_two_dim(discretization,
                                                                  macro_gradient_ij=E_unit,
                                                                  displacement_fluctuation=u_unit)
    x_mm = np.einsum('ij,jxy->ixy', L_cell, x_unit) * L_REF        # deformed nodes [mm]

    visualization_utils.plot_field_on_grid(coordinates_for_plot=x_mm,
                                           field_to_plot=phase_field.s[0, 0],   # 0 matrix, 1 void, 2 stiff
                                           name=f'deformed cell [mm], gamma = {SHEAR_STRAIN}, '
                                                f'eps_x = {LOAD_X}, eps_y = {LOAD_Y}')

    # same muFFTTO coordinates, drawn without the black grid lines and with the phase colours
    fig, ax = plt.subplots(figsize=(6, 6 * Ly_mm / Lx_mm))
    ax.pcolormesh(x_mm[0], x_mm[1], phase_field.s[0, 0], cmap=phase_cmap, norm=phase_norm, shading='flat')
    ax.plot([0, Lx_mm, Lx_mm, 0, 0], [0, 0, Ly_mm, Ly_mm, 0], 'k--', lw=0.8)   # undeformed cell
    ax.set_aspect('equal')
    ax.set_xlabel('x [mm]')
    ax.set_ylabel('y [mm]')
    ax.set_title(f'deformed cell: gamma = {SHEAR_STRAIN}, eps_x = {LOAD_X}, eps_y = {LOAD_Y}', fontsize=10)
    plt.show()

end_time = time.time()
elapsed_time = end_time - start_time
print("Elapsed time: ", elapsed_time)