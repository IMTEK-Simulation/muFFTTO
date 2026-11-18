"""
Memory of the elastic preconditioners of the AT2 study: Green, dual, multigrid (and 'none' = plain CG).

One configuration per process (fresh device memory):  python memory_footprint.py <none|Green|dual|MG> <N> [dim]
env DEVICE gpu (default) | cpu.  Prints one JSON line.

Stages, as in example_2D_phase_field_fracture_AT2.py (scaled material s C_iso, bilinear quads / trilinear hexahedra):
  base     the elastic problem: discretization, s per quadrature point, rhs, u, one K application
  prec     + the preconditioner (Green: the Fourier blocks; dual: Green + two nodal temporaries, 1/s on the fly;
           MG: MultigridQ1 V(1,1) with the rigorous lambda_max bound, setup done)
  solve    + a preconditioned CG solve (30 iterations, energy test as in the study): CG vectors and temporaries
  *_res    the same after returning the cupy pool's cached (freed) blocks: resident memory; without _res the stage
           includes the temporaries of construction / application that the pool keeps (= the peak so far)
GPU: device memory in use (cudaMemGetInfo, includes the cupy pool); CPU: resident set size (VmRSS / peak VmHWM).
The per-pixel cost is the slope between two grid sizes (the CUDA context and imports cancel).
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '../..'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

precond, N = sys.argv[1], int(sys.argv[2])
dim = int(sys.argv[3]) if len(sys.argv) > 3 else 2
device = os.environ.get('DEVICE', 'gpu')
if device == 'gpu':
    import cupy as xp
    xp.zeros(1)                                   # CUDA context


def used():
    if device == 'gpu':
        xp.cuda.Device().synchronize()
        free, total = xp.cuda.runtime.memGetInfo()
        return total - free
    status = open('/proc/self/status').read()
    return 1024 * int(status.split('VmRSS:')[1].split()[0])


def release():
    """GPU: return the cached blocks of the cupy pool (freed temporaries) -> what stays allocated"""
    if device == 'gpu':
        xp.get_default_memory_pool().free_all_blocks()


def peak():
    if device == 'gpu':
        return used()
    status = open('/proc/self/status').read()
    return 1024 * int(status.split('VmHWM:')[1].split()[0])


from muFFTTO import domain, solvers, topology_optimization     # noqa: E402

memory = {'context': used()}
element = 'bilinear_rectangle' if dim == 2 else 'trilinear_hexahedron'
cell = domain.PeriodicUnitCell(domain_size=[1.0] * dim, problem_type='elasticity')
disc = domain.Discretization(cell=cell, nb_of_pixels_global=(N,) * dim, discretization_type='finite_element',
                             element_type=element, device=device)
E, nu = 1.0, 0.2
lam, mu = E * nu / ((1 + nu) * (1 - 2 * nu)), E / (2 * (1 + nu))
I = np.eye(dim)
C_iso = lam * np.einsum('ij,kl->ijkl', I, I) + mu * (np.einsum('ik,jl->ijkl', I, I) + np.einsum('il,jk->ijkl', I, I))
xm = xp if device == 'gpu' else np
scale = disc.get_quad_field_scalar(name='degraded_scale')
rng = np.random.default_rng(0)
scale.s[...] = xm.asarray(1e-4 + rng.random(scale.s.shape))
C_d = domain.ScaledMaterial([(scale, C_iso)])
rhs = disc.get_unknown_size_field(name='rhs_u')
u = disc.get_unknown_size_field(name='u_fluctuation')
w = disc.get_unknown_size_field(name='w_random')
w.s[...] = xm.asarray(rng.standard_normal(w.s.shape))


def K(x, Ax):
    disc.apply_system_matrix_mugrid(material_data_field=C_d, input_field_inxyz=x, output_field_inxyz=Ax,
                                    formulation='small_strain')


K(w, rhs)
memory['base'] = used()

if precond in ('Green', 'dual'):
    green = disc.get_preconditioner_Green_mugrid(reference_material_data_ijkl=C_iso, formulation='small_strain')
if precond == 'dual':
    C_dual = domain.ScaledMaterial([(lambda: 1.0 / scale.s[0, 0], C_iso)])
    t1 = disc.get_unknown_size_field(name='dual_temp_1')
    t2 = disc.get_unknown_size_field(name='dual_temp_2')
if precond == 'MG':
    from mg_q1 import MultigridQ1
    mg = MultigridQ1(disc, C_iso, degree=1, n_coarsest=4, lmax_bound='gershgorin')
    mg.scale.s[...] = scale.s
    mg.setup()


def M(x, Px):
    if precond == 'none':
        Px.s[...] = x.s
    elif precond == 'Green':
        disc.apply_preconditioner_mugrid(preconditioner_Fourier_fnfnqks=green, input_nodal_field_fnxyz=x,
                                         output_nodal_field_fnxyz=Px)
    elif precond == 'dual':
        disc.fft.communicate_ghosts(x)
        disc.apply_preconditioner_mugrid(preconditioner_Fourier_fnfnqks=green, input_nodal_field_fnxyz=x,
                                         output_nodal_field_fnxyz=t1)
        disc.apply_system_matrix_mugrid(material_data_field=C_dual, input_field_inxyz=t1, output_field_inxyz=t2,
                                        formulation='small_strain')
        disc.fft.communicate_ghosts(t2)
        disc.apply_preconditioner_mugrid(preconditioner_Fourier_fnfnqks=green, input_nodal_field_fnxyz=t2,
                                         output_nodal_field_fnxyz=Px)
    else:
        mg.apply(x, Px)


r = disc.get_unknown_size_field(name='probe_r')
z = disc.get_unknown_size_field(name='probe_z')
r.s[...] = rhs.s
M(r, z)                                            # first application: lazily allocated FFT buffers etc.
memory['prec'] = used()
release()
memory['prec_res'] = used()
solvers.conjugate_gradients_mugrid_experimental(
    comm=disc.communicator, fc=disc.field_collection, hessp=K, b=rhs, x=u, P=M, maxiter=30,
    stop_crit_norm='energy_lower_estim', d_min=10,
    **topology_optimization.energy_stop_kwargs(disc, K, u, 1e-6))
memory['solve'] = used()
release()
memory['solve_res'] = used()
memory['peak'] = peak()
levels = [lv.disc.nb_of_pixels_global[0] for lv in mg.levels] if precond == 'MG' else None
print(json.dumps(dict(precond=precond, N=N, dim=dim, device=device, levels=levels, **memory)))
