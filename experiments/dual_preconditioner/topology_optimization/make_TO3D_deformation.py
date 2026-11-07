"""
Deformation of a 3D TO design under its load cases, for viewing in muEye (deformed configuration x = F s + u).

Loads a design (a _final.nc, or the last frame of a _history.nc) of example_3D_elasticity_TO.py, builds the same
material (SIMP C(rho) = rho^p (C_0 - C_void) + C_void, p = 2, K_0 = 1, G_0 = 0.5, C_void = 10^-SOFT_EXP C_0, rho
interpolated to the quadrature points), solves the 6 load cases of the study (xx, yy, zz, yz, xz, xy) with
Green-Jacobi CG (energy-error stop, rtol 1e-6) and writes <design>_deformation.nc with one frame per load case:
    phase_field     the design (1 stiff, 0 soft)
    displacement    the periodic displacement fluctuation u~ (3 components, in grid points: muEye convention)
    applied_deformation_gradient   F = I + E of the load case (muEye adds the macroscopic part F s)
    load_case, homogenized_stress  (3 x 3)
In muEye: field phase_field, isosurface 0.5; Dataset panel -> Displacement = displacement; raise the Warp scale (the
strains are unit, the file has no physical scale).

usage:  [DEVICE=gpu] [SOFT_EXP=4] python make_TO3D_deformation.py <design.nc> [output.nc]
"""
import os
import sys

import netCDF4
import numpy as np
from mpi4py import MPI

from muFFTTO import domain, io_utils, material_models, solvers, topology_optimization

from elastic_properties import summary, voigt_from_stresses

device = os.environ.get('DEVICE', 'cpu')
if device == 'gpu':
    import cupy as xp
else:
    xp = np

design_file = sys.argv[1]
out_file = sys.argv[2] if len(sys.argv) > 2 else design_file.rsplit('_', 1)[0] + '_deformation.nc'
with netCDF4.Dataset(design_file) as d:
    name = 'phase_field_solution' if 'phase_field_solution' in d.variables else 'phase_field_iterate'
    rho = np.array(d[name][-1, 0])
N = rho.shape[0]
dim = 3
soft_phase = 10.0 ** (-int(os.environ.get('SOFT_EXP', 4)))
p = 2

cell = domain.PeriodicUnitCell(domain_size=[1, 1, 1], problem_type='elasticity')
disc = domain.Discretization(cell=cell, nb_of_pixels_global=(N, N, N), discretization_type='finite_element',
                             element_type='trilinear_hexahedron', device=device,
                             low_memory=os.environ.get('LOW_MEMORY', '0') == '1')
K_0, G_0 = 1.0, 0.5
C_0 = material_models.get_elastic_material_tensor(dim=dim, K=K_0, mu=G_0, kind='linear')
C_void = soft_phase * C_0

phase = disc.get_scalar_field(name='phase_field')
phase.s[0, 0] = xp.asarray(rho)          # serial: the whole grid
phase_q = disc.get_quad_field_scalar(name='phase_field_quad')
disc.apply_N_operator_mugrid(phase, phase_q)
material = domain.ScaledMaterial([(xp.power(phase_q.s[0, 0], p), C_0 - C_void), (1.0, C_void)])

# Green-Jacobi preconditioner (reference 2 mu I), as in the TO runs
green = disc.get_preconditioner_Green_mugrid(
    reference_material_data_ijkl=2 * G_0 * np.einsum('ik,jl->ijkl', np.eye(dim), np.eye(dim)))
K_diag = disc.get_preconditioner_Jacobi_mugrid(material_data_field_ijklqxyz=material)
tmp = disc.get_unknown_size_field(name='x_jacobi_temp')


def M_fun(x, Px):
    disc.fft.communicate_ghosts(x)
    tmp.s[...] = K_diag.s * x.s
    disc.apply_preconditioner_mugrid(preconditioner_Fourier_fnfnqks=green, input_nodal_field_fnxyz=tmp,
                                     output_nodal_field_fnxyz=Px)
    Px.s[...] = K_diag.s * Px.s
    disc.fft.communicate_ghosts(Px)


def K_fun(x, Ax):
    disc.apply_system_matrix_mugrid(material_data_field=material, input_field_inxyz=x, output_field_inxyz=Ax,
                                    formulation='small_strain')


u = disc.get_unknown_size_field(name='displacement')
stresses = []
rhs = disc.get_unknown_size_field(name='rhs')
macro = disc.get_gradient_size_field(name='macro_gradient_field')
writer = io_utils.FieldWriter(out_file, [phase, u],
                              attributes={'nb_of_pixels': [N, N, N], 'domain_lengths': [1.0, 1.0, 1.0],
                                          'design_file': os.path.basename(design_file), 'soft_phase': soft_phase,
                                          'load_cases': 'xx, yy, zz, yz, xz, xy (unit macro strains)'},
                              frame_variables={'applied_deformation_gradient': (dim, dim), 'load_case': (),
                                               'homogenized_stress': (dim, dim)})
for lc, (i, j) in enumerate([(0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1)]):
    E = np.zeros((dim, dim))
    E[i, j] = E[j, i] = 1.0 if i == j else 0.5
    disc.get_macro_gradient_field_mugrid(macro_gradient_ij=E, macro_gradient_field_ijqxyz=macro)
    rhs.s.fill(0)
    disc.get_rhs_mugrid(material_data_field_ijklqxyz=material, macro_gradient_field_ijqxyz=macro, rhs_inxyz=rhs)
    u.s.fill(0)
    its = []
    solvers.conjugate_gradients_mugrid_experimental(
        comm=disc.communicator, fc=disc.field_collection, hessp=K_fun, b=rhs, x=u, P=M_fun, maxiter=100000,
        callback=lambda it, *a: its.append(it), stop_crit_norm='energy_lower_estim', d_min=10,
        **topology_optimization.energy_stop_kwargs(disc, K_fun, u, 1e-6))
    sigma = disc.get_homogenized_stress_mugrid(material_data_field_ijklqxyz=material, displacement_field_inxyz=u,
                                               macro_gradient_field_ijqxyz=macro, formulation='small_strain')
    writer.write(applied_deformation_gradient=np.eye(dim) + E, load_case=lc, homogenized_stress=sigma)
    stresses.append(sigma)
    print(f'load case {lc} (E_{i}{j}): {len(its)} CG iterations, max |u~| = {float(abs(u.s).max()):.3e}, '
          f'homogenized stress diagonal {np.round(np.diag(sigma), 4)}', flush=True)
writer.close()
print('effective properties:', summary(voigt_from_stresses(stresses)))
# muEye takes a displacement only from a per-pixel field (sub-point 'pixel'; the one nodal point of a trilinear
# hexahedron is the pixel's corner, the same grid point) and reads it in grid-point units
with netCDF4.Dataset(out_file, 'a') as d:
    d.renameDimension('subpt__nodal_points-1', 'subpt__pixel-1')
    for frame in range(d['displacement'].shape[0]):
        d['displacement'][frame] = d['displacement'][frame] * N
    d.setncattr('displacement_units', f'grid points (x 1/{N} for cell units)')
print('written', out_file)
