import sys
import os

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

# Unit Cells input: -- PHYSICAL units ===================
R_MM         = 3.0              # void & inclusion radius r                      [mm]
MU_M_MPA     = 0.6              # matrix shear modulus mu^(m) (FLX9860)           [MPa]
MU_I_MPA     = 600.0            # stiff inclusion shear modulus mu^(i) (VeroWhite) [MPa]
LAM_OVER_MU  = 1e3              # Lambda / mu, matrix and inclusion (Sec. 3, after Eq. 1) [-]
C_MATRIX     = 0.25             # matrix volume fraction c^(m)                   [-]
SPECIMEN_MM  = (92.37, 91.42)   # specimen width x height (finite sample, check only) [mm]
THICKNESS_MM = 10.0             # out-of-plane thickness -> plane strain, not used in 2D [mm]
MU_V_MPA     = 1e-4 * MU_M_MPA  # void stand-in shear modulus (numerical, NOT in paper) [MPa]

# Reference scales (nondimensionalisation) ====================================================
L_REF = R_MM        # length scale  [mm]  -> lengths in units of r
S_REF = MU_M_MPA    # stress scale  [MPa] -> stresses / moduli in units of mu^(m)

# Nondimensional quantities used by the solver (derived, do not edit) =========================
c_matrix = C_MATRIX
r = R_MM / L_REF                   # = 1
CONTRAST = MU_I_MPA / S_REF        # = 1000
MU_VOID = MU_V_MPA / S_REF         # = 1e-4
inclusion_vol_frac = 1.0 - c_matrix                   # total circle fraction (voids + stiff)

# Numerical settings ==========================================================================
skip_shear = True        # only the two normal load cases (cell is orthotropic in X/Y)
eps_plot = -0.05         # macro compression in X used for the field plot
cg_tol = 1e-6           # CG tolerance (relative)
# =============================================================================================

# Domain: TRIANGULAR lattice (paper Fig. 1a), centre spacing a, periodic cell 2a x sqrt(3)a
a = r * np.sqrt(2 * np.pi / (np.sqrt(3) * inclusion_vol_frac))   # from c_circles = 2 pi r^2 / (sqrt(3) a^2)
pixels_per_a = 96        # aim for >= 5-8 px across the ligament

domain_size = [2 * a, np.sqrt(3) * a]
number_of_pixels = (int(round(2 * pixels_per_a)), int(round(np.sqrt(3) * pixels_per_a)))

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
                                                          vol_frac=inclusion_vol_frac,
                                                          domain_size=domain_size)

# volume fraction actually resolved on the pixel grid (differs from target on coarse grids)
nb_px_global = np.prod(discretization.nb_of_pixels_global)
vf_soft = discretization.mpi_reduction.sum(phase_field.s[0, 0] == 1) / nb_px_global
vf_stiff = discretization.mpi_reduction.sum(phase_field.s[0, 0] == 2) / nb_px_global

if MPI.COMM_WORLD.rank == 0:
    print(f'target inclusion vol. frac = {inclusion_vol_frac:.4f} | '
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
centres_frac = [(0.00, 0.00), (0.25, 0.50), (0.75, 0.50), (0.50, 0.00)]   # as in geometry_stefanus
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

# preconditioner: matrix as reference, built from the SAME (lambda, mu) as the material
dim = discretization.domain_dimension
I2 = np.eye(dim)
II = np.einsum('ij,kl->ijkl', I2, I2)
I4s = 0.5 * (np.einsum('ik,jl->ijkl', I2, I2) + np.einsum('il,jk->ijkl', I2, I2))
elastic_C_ref = lam_0 * II + 2.0 * mu_0 * I4s
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
displacement_fluctuation_field = discretization.get_unknown_size_field(name='solution')


def callback(iteration, fields):
    """
    Callback function to print the current solution, residual, and search direction.
    """
    norm_of_rr = fields['rr']
    # if discretization.communicator.rank == 0:
    #     print(f"{iteration:5} norm of residual = {norm_of_rr:.5}")


def solve_for_macro_gradient(macro_gradient_ij):
    """Solve for the fluctuation under a prescribed macro strain; returns number of CG iterations."""
    discretization.get_macro_gradient_field_mugrid(macro_gradient_ij=macro_gradient_ij,
                                                   macro_gradient_field_ijqxyz=macro_gradient_field)
    discretization.get_rhs_mugrid(material_data_field_ijklqxyz=material_data_field_C_0,
                                  macro_gradient_field_ijqxyz=macro_gradient_field,
                                  rhs_inxyz=rhs_field)
    displacement_fluctuation_field.s.fill(0)

    # tol is an ABSOLUTE threshold on ||b - Ax||  ->  scale with ||b|| to make cg_tol relative
    norm_b = np.sqrt(discretization.communicator.sum(np.dot(rhs_field.s.ravel(), rhs_field.s.ravel())))
    its = []

    def counting_callback(iteration, fields):
        its.append(iteration)
        callback(iteration, fields)

    Solvers.conjugate_gradients(
        comm=discretization.communicator,
        fc=discretization.field_collection,
        hessp=K_fun,  # linear operator
        b=rhs_field,  # right-hand side
        x=displacement_fluctuation_field,
        prec=M_fun,
        tol=cg_tol * norm_b,
        maxiter=5000,
        callback=counting_callback)
    return len(its)


# homogenized stiffness: unit macro strain per (k, l) -> column C[:, :, k, l]
homogenized_C_ijkl = np.zeros(4 * [dim])
load_cases = [(0, 0), (1, 1)] if skip_shear else [(0, 0), (1, 1), (0, 1)]
cg_its = {}
for k, l in load_cases:
    macro_gradient_ij = np.zeros([dim, dim])
    macro_gradient_ij[k, l] = 1.0
    cg_its[(k, l)] = solve_for_macro_gradient(macro_gradient_ij)
    homogenized_C_ijkl[:, :, k, l] = discretization.get_homogenized_stress_mugrid(
        material_data_field_ijklqxyz=material_data_field_C_0,
        displacement_field_inxyz=displacement_fluctuation_field,
        macro_gradient_field_ijqxyz=macro_gradient_field,
        formulation='small_strain')

C_voigt = material_models.compute_Voigt_notation_4order(homogenized_C_ijkl)  # [xx, yy, xy]
S_voigt = np.zeros((3, 3))
if skip_shear:
    S_voigt[:2, :2] = np.linalg.inv(C_voigt[:2, :2])  # orthotropic: normal block decouples
else:
    S_voigt = np.linalg.inv(C_voigt)

# lateral faces traction-free (plane strain), paper notation
E_X = 1.0 / S_voigt[0, 0]
E_Y = 1.0 / S_voigt[1, 1]
nu_YX = -S_voigt[1, 0] / S_voigt[0, 0]  # load in X
nu_XY = -S_voigt[0, 1] / S_voigt[1, 1]  # load in Y
if not skip_shear:
    G_XY = 1.0 / S_voigt[2, 2]           # effective in-plane shear modulus / mu_m
    coupling = np.abs(C_voigt[:2, 2]).max() / np.abs(C_voigt[:2, :2]).max()   # ~0 for orthotropic cells
    if MPI.COMM_WORLD.rank == 0:
        print(f'G_XY / mu_m = {G_XY:.4f}   normal-shear coupling = {coupling:.2e}')

if MPI.COMM_WORLD.rank == 0:
    print(f'CG iterations per load case: {cg_its}')
    print('Homogenized C (Voigt [xx, yy, xy], units of mu_matrix) =\n' +
          np.array2string(C_voigt, formatter={'float_kind': lambda v: f'{v:12.6f}'}))
    print(f'E_X / mu_m = {E_X:.4f}   nu_YX = {nu_YX:.4f}   (load in X)')
    print(f'E_Y / mu_m = {E_Y:.4f}   nu_XY = {nu_XY:.4f}   (load in Y)')

# state for the field plot: uniaxial compression in X, lateral faces traction-free
macro_gradient_ij = np.zeros([dim, dim])
macro_gradient_ij[0, 0] = eps_plot
macro_gradient_ij[1, 1] = -nu_YX * eps_plot
solve_for_macro_gradient(macro_gradient_ij)

# strain from displacement increment
# Apply small strain (Linearized) Elasticity
strain_fluc_field = discretization.get_strain_sized_field(name='strain_fluc_field')
discretization.apply_gradient_operator_symmetrized_mugrid(
    u_inxyz=displacement_fluctuation_field,
    grad_u_ijqxyz=strain_fluc_field,
)

# update total strain and displacement
total_strain_ijqxyz.s[...] = macro_gradient_field.s[...] + strain_fluc_field.s[...]

# re-evaluate constitutive response
total_stress_field = discretization.get_strain_sized_field(name='total_stress_field')

material.get_stress(total_strain_ijqxyz, total_stress_field)

Von_Mises_1nxyz = discretization.get_scalar_field(name='Von_Mises')

s = total_stress_field.s
sxx = s[0, 0]
syy = s[1, 1]
sxy = s[0, 1]

# plane strain: sigma_zz = lambda * tr(eps), per quadrature point
szz = lam_11qxyz.s[0, 0] * (total_strain_ijqxyz.s[0, 0] + total_strain_ijqxyz.s[1, 1])

vm = np.sqrt(
    0.5 * ((sxx - syy) ** 2 + (syy - szz) ** 2 + (szz - sxx) ** 2)
    + 3.0 * sxy ** 2
)

#  I want to plot the field/ Von Mises stress. I have four quad points , but matplotlib plots only one number per pixel # shape: (4, 16, 32)
Von_Mises_1nxyz.s[...] = np.mean(vm, axis=0)

def deformed_grid_physical(macro_gradient_ij, u_fluc_field):
    """Deformed nodal grid in physical units (serial only).
    visualization_utils.get_deformed_grid_coords_two_dim uses coordinates in [0, 1],
    which is inconsistent with physical displacements when L != 1."""
    nx, ny = discretization.nb_of_pixels_global
    gx, gy = np.meshgrid(np.linspace(0, domain_size[0], nx + 1),
                         np.linspace(0, domain_size[1], ny + 1), indexing='ij')
    x_ixy = np.stack([gx, gy])
    x_ixy = x_ixy + np.einsum('ij,jxy->ixy', macro_gradient_ij, x_ixy)  # x + E x
    u_ixy = u_fluc_field.s[:, 0]                                         # first nodal point per pixel
    u_ixy = np.pad(u_ixy, ((0, 0), (0, 1), (0, 1)), mode='wrap')         # periodic end nodes
    return x_ixy + u_ixy


if discretization.communicator.size == 1:
    x_plot_ixyz = deformed_grid_physical(macro_gradient_ij, displacement_fluctuation_field)
    vm_plot = np.ma.masked_where(inc_soft_mask, Von_Mises_1nxyz.s[0, 0])  # hide the void stand-in
    fig, ax = plt.subplots(figsize=(5, 5))
    pcm = ax.pcolormesh(x_plot_ixyz[0], x_plot_ixyz[1], vm_plot, shading='flat', cmap='jet')
    fig.colorbar(pcm, ax=ax, label=r'$\sigma_{vM} / \mu_m$')
    ax.set_aspect('equal')
    ax.set_xlabel('x / r')
    ax.set_ylabel('y / r')
    ax.set_title(fr'uniaxial compression X, $\varepsilon$ = {eps_plot}')
    plt.show()

end_time = time.time()
elapsed_time = end_time - start_time
print("Elapsed time: ", elapsed_time)
