"""
3D corrected Green with a clamped plane: S = E^T G E inverted by a 2D FFT over the plane, never assembled.

The 3D cantilever of cantilever_3D_study.py (L x H x W = 1 x 0.2 x 0.2, trilinear voxels, void of physical thickness
VOID_THICKNESS around it, uniform shear F_y on the right face), clamped at the left face. Plain outer CG, energy stop
1e-6, delay 10.

VARIANT=
    green        plain (projected) Green, solid face clamped: the baseline
    dense_face   corrected Green, dense S, only the solid face clamped (the original problem)
    dense_plane  corrected Green, dense S, the clamp plane x = x0 through the whole periodic cell (void included:
                 zero stiffness, same problem)
    fft_plane    the same plane, S inverted by the 2D FFT over the plane: the kernel from 3 Green applications (a unit
                 force per component at one plane node), its FFT a 3x3 block per wave-number pair (ky, kz)
    cg_plane     the same plane, S by plain inner CG (relative residual 1e-8)

usage:  N_H=8 VARIANT=fft_plane python plane_clamp_3D.py     (run_plane_3D.sh for the sweep)
writes exp_data/runs3d_plane/n_H<n>_<variant>.npz
"""
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import time

import numpy as np

from muFFTTO import constraints
from muFFTTO import domain
from muFFTTO import material_models
from muFFTTO import solvers
from muFFTTO import topology_optimization

HERE = os.path.dirname(os.path.realpath(__file__))
RUNS = os.path.join(HERE, 'exp_data', 'runs3d_plane')
os.makedirs(RUNS, exist_ok=True)

beam_length, beam_height, beam_width = 1.0, 0.2, 0.2
total_force_y = -1e-4
cg_tol, cg_delay = 1e-6, 10
n_H = int(os.environ.get('N_H', 8))
variant = os.environ.get('VARIANT', 'fft_plane')
void_thickness = float(os.environ.get('VOID_THICKNESS', 0.05))
inner_tol = 1e-8

h = beam_height / n_H
n_L, n_W = int(round(beam_length / h)), int(round(beam_width / h))
nb_void = int(round(void_thickness / h))
nb_px = (n_L + 2 * nb_void, n_H + 2 * nb_void, n_W + 2 * nb_void)
cell = domain.PeriodicUnitCell(domain_size=[n * h for n in nb_px], problem_type='elasticity')
disc = domain.Discretization(cell=cell, nb_of_pixels_global=nb_px, discretization_type='finite_element',
                             element_type='trilinear_hexahedron')
ix, iy, iz = disc.fft.icoords
bx, by, bz = (nb_void, nb_void + n_L), (nb_void, nb_void + n_H), (nb_void, nb_void + n_W)
beam = (ix >= bx[0]) & (ix < bx[1]) & (iy >= by[0]) & (iy < by[1]) & (iz >= bz[0]) & (iz < bz[1])


def face(x_index):
    return (ix == x_index) & (iy >= by[0]) & (iy <= by[1]) & (iz >= bz[0]) & (iz <= bz[1])


lam, mu = material_models.get_lame_parameters(E=1.0, poisson=0.3)
lam_q, mu_q = disc.get_quad_field_scalar(name='lam_q'), disc.get_quad_field_scalar(name='mu_q')
lam_q.s[...], mu_q.s[...] = 0, 0
lam_q.s[..., beam], mu_q.s[..., beam] = lam, mu
material = material_models.LinearElastic(discretization=disc, lam_1qxyz=lam_q, mu_1qxyz=mu_q, name='elasticity')

clamped = (ix == bx[0]) if variant.endswith('_plane') else face(bx[0])
constrained = disc.get_displacement_sized_field(name='constrained')
constrained.s[...] = 0
constrained.s[:, 0, clamped] = 1
prescribed = disc.get_displacement_sized_field(name='prescribed')
prescribed.s[...] = 0
nodal_constraints = constraints.NodalConstraints.from_mask(disc, constrained_mask=constrained,
                                                           prescribed_values=prescribed)

# uniform shear on the right face: consistent nodal weights of a constant traction (trapezoid in y and z)
force = disc.get_unknown_size_field(name='force')
force.s[...] = 0
right = face(bx[1])
w_y = np.where((iy == by[0]) | (iy == by[1]), 0.5, 1.0) * h
w_z = np.where((iz == bz[0]) | (iz == bz[1]), 0.5, 1.0) * h
force.s[1, 0, right] = (total_force_y / (beam_height * beam_width) * w_y * w_z)[right]


def K_fun(x, Ax):
    disc.apply_system_matrix_mugrid_explicit_stress(constitutive=material.get_stress, input_field_inxyz=x,
                                                    output_field_inxyz=Ax, formulation='small_strain')
    disc.fft.communicate_ghosts(Ax)


green = disc.get_preconditioner_Green_mugrid(
    reference_material_data_ijkl=material_models.get_elastic_tensor_from_lame(dim=3, lam=lam, mu=mu))
green_apps = [0]


def M_fun(x, Px):
    disc.fft.communicate_ghosts(x)
    disc.apply_preconditioner_mugrid(preconditioner_Fourier_fnfnqks=green, input_nodal_field_fnxyz=x,
                                     output_nodal_field_fnxyz=Px)
    green_apps[0] += 1


t_setup = time.time()
fixed = tuple(np.argwhere(constrained.s != 0).T)  # (component, sub-point, x, y, z)
m = len(fixed[0])
P_fun = nodal_constraints.project_preconditioner(M_fun)
inner_its = []
if variant != 'green':
    Ey, MEy = disc.get_unknown_size_field(name='Ey'), disc.get_unknown_size_field(name='MEy')

    def S_fun(y):  # S y = E^T M E y, one Green application
        Ey.s[...] = 0
        Ey.s[fixed] = y
        M_fun(Ey, MEy)
        return MEy.s[fixed].copy()

    if variant.startswith('dense'):
        S = np.empty((m, m))
        for j in range(m):
            e_j = np.zeros(m)
            e_j[j] = 1.0
            S[:, j] = S_fun(e_j)
        S_inv = np.linalg.inv(0.5 * (S + S.T))
        del S
        S_solve = lambda g: S_inv @ g
    elif variant == 'fft_plane':
        # the fixed dofs as a (3, ny, nz) array over the plane; the kernel from a unit force at the plane node (0, 0)
        ny, nz = nb_px[1], nb_px[2]
        plane_index = np.empty((3, ny, nz), dtype=int)
        plane_index[fixed[0], fixed[3], fixed[4]] = np.arange(m)
        kernel = np.zeros((3, 3, ny, nz))
        for e in range(3):
            y = np.zeros(m)
            y[plane_index[e, 0, 0]] = 1.0
            Sy = S_fun(y)
            kernel[:, e] = Sy[plane_index]
        symbol = np.fft.fft2(kernel).transpose(2, 3, 0, 1)  # (ny, nz, 3, 3)
        symbol = 0.5 * (symbol + np.conj(np.swapaxes(symbol, -1, -2)))
        ev, evec = np.linalg.eigh(symbol)
        symbol_inv = np.einsum('...ij,...j,...kj->...ik', evec, 1.0 / ev, np.conj(evec))

        def S_solve(g):
            g_hat = np.fft.fft2(g[plane_index])  # (3, ny, nz)
            return np.real(np.fft.ifft2(np.einsum('yzde,eyz->dyz', symbol_inv, g_hat)))[
                (fixed[0], fixed[3], fixed[4])]

        g_test = np.random.default_rng(0).standard_normal(m)
        print(f'|S S_fft^-1 g - g| / |g| = {np.linalg.norm(S_fun(S_solve(g_test)) - g_test) / np.linalg.norm(g_test):.2e}',
              flush=True)
        green_apps[0] -= 1  # the check is not part of the setup
    else:  # cg_plane
        def S_solve(g):
            y, res = np.zeros_like(g), g.copy()
            p, rr, rr0 = res.copy(), res @ res, g @ g
            for it in range(1, 10 * m + 1):
                Sp = S_fun(p)
                a = rr / (p @ Sp)
                y += a * p
                res -= a * Sp
                rr, rr_old = res @ res, rr
                if rr <= inner_tol ** 2 * rr0:
                    break
                p = res + rr / rr_old * p
            inner_its.append(it)
            return y

    B = np.stack([(fixed[0] == d).astype(float) for d in range(3)], axis=1)  # E^T N
    W = np.stack([S_solve(B[:, d]) for d in range(3)], axis=1)
    BW_inv = np.linalg.inv(B.T @ W)
    inner_its.clear()
    Mr, w = disc.get_unknown_size_field(name='Mr'), disc.get_unknown_size_field(name='w')

    def P_fun(r, z):  # [S B; B^T 0][lam; mu] = [E^T M r; N^T r], z = M (r - E lam) - N mu
        M_fun(r, Mr)
        Sg = S_solve(Mr.s[fixed])
        mu_ = BW_inv @ (B.T @ Sg - np.array([r.s[d].sum() for d in range(3)]))
        w.s[...] = r.s
        w.s[fixed] -= Sg - W @ mu_
        M_fun(w, z)
        for d in range(3):
            z.s[d] -= mu_[d]
        z.s[fixed] = 0.0
setup_time, setup_apps = time.time() - t_setup, green_apps[0]
green_apps[0] = 0
print(f'n_H{n_H}_{variant}: grid {nb_px}, {m} fixed dofs, setup {setup_time:.1f} s ({setup_apps} Green apps)', flush=True)

rhs, u = disc.get_unknown_size_field(name='rhs'), disc.get_unknown_size_field(name='u')
rhs.s[...] = force.s
nodal_constraints.apply_to_rhs(rhs, K_fun)
u.s[...] = 0
K_free = nodal_constraints.project_operator(K_fun)
t0 = time.time()
norms = solvers.conjugate_gradients_mugrid_experimental(
    comm=disc.communicator, fc=disc.field_collection, hessp=K_free, b=rhs, x=u, P=P_fun, maxiter=20000,
    stop_crit_norm='energy_lower_estim', d_min=cg_delay,
    **topology_optimization.energy_stop_kwargs(disc, K_free, u, cg_tol))[1]
cg_its = norms['stop_iteration']['energy_lower_estim']
cg_its = int(cg_its) if cg_its is not None else -1
wall_time = time.time() - t0
compliance = float(np.sum(force.s * u.s))
print(f'n_H{n_H}_{variant}: outer {cg_its}, inner/outer {np.mean(inner_its) if inner_its else 0:.1f}, '
      f'Green apps {green_apps[0]} (+ {setup_apps} setup), compliance {compliance:.10e}, {wall_time:.1f} s', flush=True)
np.savez(os.path.join(RUNS, f'n_H{n_H}_{variant}.npz'), n_H=n_H, variant=variant, pixels=np.array(nb_px), m=m,
         cg_its=cg_its, inner_its=np.array(inner_its, dtype=int), green_apps=green_apps[0], setup_apps=setup_apps,
         compliance=compliance, time=wall_time, setup_time=setup_time)
