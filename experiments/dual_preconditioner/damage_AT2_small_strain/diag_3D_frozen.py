"""
3D elastic CG iterations on FROZEN damage fields of the 64^3, l/h = 2 runs, transferred to finer grids at the same
l/h -- the iteration counts of a 96^3 run without running it.

Damage snapshots (three phases of the solver) of
    incl   AT2_N64_dual_polydisperse_pores_incl10_l0.03125_3D   (stiff tough inclusions 10x)
    pores  AT2_N64_Green_Jacobi_polydisperse_pores_l0.03125_3D
    pores_d1  the same field, but the pores as fully broken matrix (stiffness k C_0 in the pore voxels, as
              DAMAGE_IN_PORES=one in the 2D runs) instead of 1e-3 g(d) C_0: the lowest stiffness is k everywhere
    calm    the load step at half the peak load
    peak    the last load step before the break
    broken  the final state (cell cut through)
are transferred to the N^3 grid (nodes at i / N) by periodic trilinear interpolation of ln d, and the crack band is
narrowed to l_N = 2 / N (fixed l/h = 2) with the AT2 profile d = exp(-r / l):
    d_N = d_64 ** (l_64 / l_N)        (= exp(-r / l_N) for d_64 = exp(-r / l_64); also shrinks the background damage,
                                       which is likewise smaller for smaller l)
The balls are regenerated exactly on the N^3 grid. One cold-start elastic solve per (case, phase, N, preconditioner),
CG with the energy-error stop of the runs (rtol 1e-6, delay 10), and the extreme Ritz values of the preconditioned
operator from a plain PCG of the same length.

usage:  python diag_3D_frozen.py CASE N [PRECONDITIONERS]    e.g.  python diag_3D_frozen.py incl 96 dual,Green_Jacobi
        writes exp_data/diag_3D_frozen_<CASE>_N<N>.csv
"""
import csv
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

import numpy as np
import netCDF4
from scipy import ndimage

from muFFTTO import domain
from muFFTTO import microstructure_library
from muFFTTO import material_models
from muFFTTO import solvers
from muFFTTO import topology_optimization   # energy_stop_kwargs

HERE = os.path.dirname(os.path.realpath(__file__))
SOURCES = {'incl': ('AT2_N64_dual_polydisperse_pores_incl10_l0.03125_3D.nc', 10.0),
           'pores': ('AT2_N64_Green_Jacobi_polydisperse_pores_l0.03125_3D.nc', 1e-3),
           'pores_d1': ('AT2_N64_Green_Jacobi_polydisperse_pores_l0.03125_3D.nc', 1.0)}
case, n = sys.argv[1], int(sys.argv[2])
PRECONDS = tuple((sys.argv[3] if len(sys.argv) > 3 else 'dual,Green_Jacobi').split(','))
source, disk_stiffness = SOURCES[case]
# K_RESIDUAL (env): the residual stiffness k of g(d) = (1 - d)^2 + k (default 1e-4; for pores_d1 also the pore stiffness)
k_residual, cg_tol, cg_delay = float(os.environ.get('K_RESIDUAL', 1e-4)), 1e-6, 10
E_young, poisson = 1.0, 0.2
n_source, lh = 64, 2.0

with netCDF4.Dataset(os.path.join(HERE, 'exp_data', source)) as ds:
    stress = np.asarray(ds['macro_stress'][:, 0, 0])
    loads = np.asarray(ds['load'][:])
    i_peak = int(stress.argmax())
    i_calm = int(np.argmin(np.abs(loads - 0.5 * loads[i_peak])))
    STATES = {name: (np.asarray(ds['damage'][i, 0]), loads[i])
              for name, i in (('calm', i_calm), ('peak', i_peak), ('broken', len(loads) - 1))}


def transfer(d64):
    """nodal d on 64^3 -> nodal d on n^3 at the same l/h"""
    log_d = np.log(np.clip(d64, 1e-12, None))
    if n != n_source:
        idx = np.arange(n) * n_source / n
        grid = np.meshgrid(idx, idx, idx, indexing='ij')
        log_d = ndimage.map_coordinates(log_d, grid, order=1, mode='grid-wrap')
    return np.clip(np.exp(log_d * n / n_source), 0.0, None)       # d^(l_64 / l_n), l_64 / l_n = n / 64


cell = domain.PeriodicUnitCell(domain_size=[1, 1, 1], problem_type='elasticity')
disc = domain.Discretization(cell=cell, nb_of_pixels_global=(n, n, n), discretization_type='finite_element',
                             element_type='trilinear_hexahedron')
cell_d = domain.PeriodicUnitCell(domain_size=[1, 1, 1], problem_type='conductivity')
disc_d = domain.Discretization(cell=cell_d, nb_of_pixels_global=(n, n, n), discretization_type='finite_element',
                               element_type='trilinear_hexahedron')
phase = microstructure_library.get_geometry(nb_voxels=disc.nb_of_pixels, microstructure_name='polydisperse_pores',
                                            coordinates=disc.fft.coords)
stiffness = np.where(phase > 0, 1.0, disk_stiffness)
K_0, G_0 = material_models.get_bulk_and_shear_modulus(E=E_young, poisson=poisson)
lam, mu = material_models.get_lame_parameters_from_bulk_and_shear(K_0, G_0, dim=3)
I3 = np.eye(3)
I_sym = 0.5 * (np.einsum('ik,jl->ijkl', I3, I3) + np.einsum('il,jk->ijkl', I3, I3))
C_iso = lam * np.einsum('ij,kl->ijkl', I3, I3) + 2.0 * mu * I_sym
scale = disc.get_quad_field_scalar(name='scale')
C_d = domain.ScaledMaterial([(scale, C_iso)])
C_dual = domain.ScaledMaterial([(lambda: 1.0 / scale.s[0, 0], C_iso)])
green = {'Green': disc.get_preconditioner_Green_mugrid(reference_material_data_ijkl=C_iso,
                                                       formulation='small_strain', name='green_iso'),
         'Green_Jacobi': disc.get_preconditioner_Green_mugrid(reference_material_data_ijkl=2.0 * mu * I_sym,
                                                              formulation='small_strain', name='green_2mu')}
green['dual'] = green['Green']
damage = disc_d.get_unknown_size_field(name='damage')
damage_q = disc_d.get_quad_field_scalar(name='damage_q')
macro = disc.get_gradient_size_field(name='macro')
rhs = disc.get_unknown_size_field(name='rhs')
u = disc.get_unknown_size_field(name='u')
t1, t2 = disc.get_unknown_size_field(name='t1'), disc.get_unknown_size_field(name='t2')
xs = disc.get_unknown_size_field(name='xs')
r, z, p, q = (disc.get_unknown_size_field(name=f'lanczos_{c}') for c in 'rzpq')
state = {}


def K(x, Ax):
    disc.apply_system_matrix_mugrid(material_data_field=C_d, input_field_inxyz=x, output_field_inxyz=Ax,
                                    formulation='small_strain')


def M(x, Px):
    P = state['P']
    if P == 'dual':
        disc.fft.communicate_ghosts(x)
        disc.apply_preconditioner_mugrid(preconditioner_Fourier_fnfnqks=green[P], input_nodal_field_fnxyz=x,
                                         output_nodal_field_fnxyz=t1)
        disc.apply_system_matrix_mugrid(material_data_field=C_dual, input_field_inxyz=t1, output_field_inxyz=t2,
                                        formulation='small_strain')
        disc.fft.communicate_ghosts(t2)
        disc.apply_preconditioner_mugrid(preconditioner_Fourier_fnfnqks=green[P], input_nodal_field_fnxyz=t2,
                                         output_nodal_field_fnxyz=Px)
    elif P == 'Green_Jacobi':
        xs.s[...] = state['jacobi'].s * x.s
        disc.apply_preconditioner_mugrid(preconditioner_Fourier_fnfnqks=green[P], input_nodal_field_fnxyz=xs,
                                         output_nodal_field_fnxyz=Px)
        Px.s[...] *= state['jacobi'].s
    else:
        disc.apply_preconditioner_mugrid(preconditioner_Fourier_fnfnqks=green[P], input_nodal_field_fnxyz=x,
                                         output_nodal_field_fnxyz=Px)


def ritz(nb_its):
    """plain PCG from u = 0 for nb_its iterations; Ritz values of M K from its Lanczos tridiagonal"""
    r.s[...] = rhs.s
    M(r, z)
    p.s[...] = z.s
    rz = float(np.vdot(r.s, z.s))
    alphas, betas = [], []
    for _ in range(nb_its):
        disc.fft.communicate_ghosts(p)
        K(p, q)
        alpha = rz / float(np.vdot(p.s, q.s))
        r.s[...] -= alpha * q.s
        M(r, z)
        rz_new = float(np.vdot(r.s, z.s))
        alphas.append(alpha), betas.append(rz_new / rz)
        p.s[...] = z.s + betas[-1] * p.s
        rz = rz_new
    m = len(alphas)
    T = np.zeros((m, m))
    for j in range(m):
        T[j, j] = 1.0 / alphas[j] + (betas[j - 1] / alphas[j - 1] if j > 0 else 0.0)
        if j + 1 < m:
            T[j, j + 1] = T[j + 1, j] = np.sqrt(betas[j]) / alphas[j]
    return np.linalg.eigvalsh(T)


rows = []
for name, (d64, load) in STATES.items():
    damage.s[0, 0] = transfer(d64)
    disc_d.apply_N_operator_mugrid(damage, damage_q)
    scale.s[0, 0] = ((1.0 - damage_q.s[0, 0]) ** 2 + k_residual) * stiffness
    if case == 'pores_d1':
        scale.s[0, 0] = np.where(phase > 0, scale.s[0, 0], k_residual)     # pores: g(1) C_0 = k C_0
    macro_ij = np.zeros((3, 3))
    macro_ij[0, 0] = load
    disc.get_macro_gradient_field_mugrid(macro_gradient_ij=macro_ij, macro_gradient_field_ijqxyz=macro)
    disc.get_rhs_mugrid(material_data_field_ijklqxyz=C_d, macro_gradient_field_ijqxyz=macro, rhs_inxyz=rhs)
    for P in PRECONDS:
        state['P'] = P
        if P == 'Green_Jacobi':
            state['jacobi'] = disc.get_preconditioner_Jacobi_mugrid(material_data_field_ijklqxyz=C_d,
                                                                    formulation='small_strain', name='jac')
        u.s[...] = 0.0
        norms = solvers.conjugate_gradients_mugrid_experimental(
            comm=disc.communicator, fc=disc.field_collection, hessp=K, b=rhs, x=u, P=M, maxiter=20000,
            stop_crit_norm='energy_lower_estim', d_min=cg_delay,
            **topology_optimization.energy_stop_kwargs(disc, K, u, cg_tol))[1]
        its = int(norms['stop_iteration']['energy_lower_estim'])
        ev = ritz(its)
        rows.append(dict(case=case, state=name, load=load, N=n, preconditioner=P, cg_its=its, lambda_min=ev[0],
                         lambda_max=ev[-1], max_d=float(damage.s.max()), crack_voxels=int((damage.s >= 0.9).sum())))
        print(f'{case:5s} {name:6s} N {n:3d} {P:13s} CG {its:5d}   Ritz [{ev[0]:.3e}, {ev[-1]:.3e}]   '
              f'max d {damage.s.max():.2f}, voxels d >= 0.9: {rows[-1]["crack_voxels"]}', flush=True)

out = os.path.join(HERE, 'exp_data', f'diag_3D_frozen_{case}_N{n}'
                   + (f'_k{k_residual:g}' if 'K_RESIDUAL' in os.environ else '') + '.csv')
with open(out, 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader(), w.writerows(rows)
print(f'wrote {out}')
