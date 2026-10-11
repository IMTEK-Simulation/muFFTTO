"""
Time the building blocks of a standard muFFTTO small-strain elasticity solve
(bilinear elements, 2D) for several grid sizes:

    ghosts  communicate_ghosts of the nodal field (MPI halo exchange)
    B       apply_gradient_operator_mugrid           (nodal -> quadrature)
    C       apply_material_data_mugrid                (C : grad u at the quad. points)
    BT      apply_gradient_transposed_operator_mugrid (quadrature -> nodal)
    K       apply_system_matrix_mugrid                (B^T C B, as used in CG)
    M       apply_preconditioner_mugrid               (Green operator, FFT based)
    dot     global dot product (one CG reduction)

    python benchmark_kernels.py [N ...]          (serial)
    mpirun -n 4 python benchmark_kernels.py [N ...]

Prints the time per call (max over ranks) and per pixel (ns, x ranks = core time).
"""
import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

import numpy as np  # noqa: E402
from mpi4py import MPI  # noqa: E402

from muFFTTO import domain, material_models  # noqa: E402

comm = MPI.COMM_WORLD
sizes = [int(a) for a in sys.argv[1:]] or [32, 64, 128, 256, 512]


def timeit(fun, min_time=0.5, min_reps=5):
    fun()                                            # warm-up
    comm.Barrier()
    reps, start = 0, time.perf_counter()
    while True:
        fun()
        reps += 1
        elapsed = comm.allreduce(time.perf_counter() - start, op=MPI.MAX)
        if reps >= min_reps and elapsed > min_time:
            return elapsed / reps


if comm.rank == 0:
    print(f'ranks {comm.size}; time per call [ms], in brackets ns per pixel x ranks')
    print(f'{"N":>5} ' + ' '.join(f'{k:>17}' for k in ['ghosts', 'B', 'C', 'BT', 'K', 'M', 'dot']))
for N in sizes:
    disc = domain.Discretization(cell=domain.PeriodicUnitCell(domain_size=[1, 1], problem_type='elasticity'),
                                 nb_of_pixels_global=(N, N), discretization_type='finite_element',
                                 element_type='bilinear_rectangle')
    lam_q = disc.get_quad_field_scalar(name='lam')
    mu_q = disc.get_quad_field_scalar(name='mu')
    lam_q.s[...] = 1.0
    mu_q.s[...] = 1.0
    material = material_models.LinearElastic(discretization=disc, lam_1qxyz=lam_q, mu_1qxyz=mu_q, name='mat')
    C = disc.get_material_data_size_field_mugrid(name='C')
    material.get_algorithmic_tangent(disc.get_strain_sized_field(name='eps'), C)
    green = disc.get_preconditioner_Green_mugrid(
        reference_material_data_ijkl=material_models.get_elastic_material_tensor(dim=2, K=1.0, mu=1.0, kind='linear'))

    x = disc.get_unknown_size_field(name='x')
    y = disc.get_unknown_size_field(name='y')
    grad = disc.get_gradient_size_field(name='grad')
    x.s[...] = np.random.default_rng(comm.rank).random(x.s.shape)

    def k_apply():
        disc.apply_system_matrix_mugrid(material_data_field=C, input_field_inxyz=x, output_field_inxyz=y,
                                        formulation='small_strain')

    timers = {
        'ghosts': lambda: disc.fft.communicate_ghosts(x),
        'B': lambda: disc.apply_gradient_operator_mugrid(u_inxyz=x, grad_u_ijqxyz=grad),
        'C': lambda: disc.apply_material_data_mugrid(material_data=C, gradient_field=grad),
        'BT': lambda: disc.apply_gradient_transposed_operator_mugrid(gradient_field_ijqxyz=grad, div_u_fnxyz=y),
        'K': k_apply,
        'M': lambda: disc.apply_preconditioner_mugrid(preconditioner_Fourier_fnfnqks=green,
                                                      input_nodal_field_fnxyz=x, output_nodal_field_fnxyz=y),
        'dot': lambda: comm.allreduce(float(np.sum(x.s * y.s))),
    }
    results = {k: timeit(f) for k, f in timers.items()}
    if comm.rank == 0:
        print(f'{N:>5} ' + ' '.join(f'{1e3 * t:8.3f} ({1e9 * t * comm.size / N**2:6.0f})' for t in results.values()),
              flush=True)
