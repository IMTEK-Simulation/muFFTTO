"""
Cost of muGrid ghost exchanges on GPU fields under MPI (one rank per GPU):
    srun -n 2 --gpus-per-task=1 python fable_time_ghosts.py [N, default 256]
Times communicate_ghosts for the three fields K exchanges (nodal vector, stress, hourglass strain), device-synchronised,
and for comparison one fine-level K and a raw CUDA-aware MPI Sendrecv of a buffer of the size of one stress face.
"""
import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
import numpy as np
from mpi4py import MPI

from muFFTTO import domain, material_models, multigrid, tensor_operations

N = int(sys.argv[1]) if len(sys.argv) > 1 else 256
mpi = MPI.COMM_WORLD
C_0 = material_models.get_elastic_material_tensor(dim=3, K=1.0, mu=0.5, kind='linear')
cell = domain.PeriodicUnitCell(domain_size=[1, 1, 1], problem_type='elasticity')
disc = domain.Discretization(cell=cell, nb_of_pixels_global=(N, N, N), discretization_type='finite_element',
                             element_type='trilinear_hexahedron_1Q_hg', device='gpu', hourglass_eps=0.3)
mg = multigrid.Multigrid(disc, C_0)
mg.scale.s[...] = 0.5
u, f = disc.get_unknown_size_field(name='u'), disc.get_unknown_size_field(name='f')
xp = tensor_operations.get_array_module(u.s)
u.s[...] = 1.0
fields = {'nodal vector [3]': u, 'stress [3, 3]': mg.levels[0].stress, 'hourglass strain [3, 4]': disc.hourglass_control.strain}


def timed(fun, repeat=10):
    fun()
    xp.cuda.Device().synchronize()
    mpi.Barrier()
    start = time.perf_counter()
    for _ in range(repeat):
        fun()
    xp.cuda.Device().synchronize()
    mpi.Barrier()
    return 1e3 * (time.perf_counter() - start) / repeat


if mpi.rank == 0:
    print(f'N {N}, {mpi.size} rank(s), local grid {tuple(disc.fft.nb_subdomain_grid_pts)}, '
          f'UCX_TLS={os.environ.get("UCX_TLS", "(default)")}', flush=True)
def once(fun):
    xp.cuda.Device().synchronize()
    mpi.Barrier()
    start = time.perf_counter()
    fun()
    xp.cuda.Device().synchronize()
    mpi.Barrier()
    return 1e3 * (time.perf_counter() - start)


for name, field in fields.items():
    first, second = once(lambda: disc.fft.communicate_ghosts(field)), once(lambda: disc.fft.communicate_ghosts(field))
    if mpi.rank == 0:
        print(f'  first / second exchange of {name:<26s} {first:10.1f} / {second:8.1f} ms', flush=True)
for name, field in fields.items():
    ms = timed(lambda: disc.fft.communicate_ghosts(field))
    face = np.prod(field.s.shape[:-1]) * 8 / 2 ** 20
    if mpi.rank == 0:
        print(f'  communicate_ghosts {name:<26s} {ms:8.2f} ms   (z face {face:6.1f} MiB)', flush=True)
ms = timed(lambda: mg.levels[0].apply_K(u, f), repeat=5)
if mpi.rank == 0:
    print(f'  fine-level K (3 exchanges)                    {ms:8.2f} ms', flush=True)
first, second = once(mg.setup), once(mg.setup)
if mpi.rank == 0:
    print(f'  multigrid setup, first / second                 {first / 1e3:10.1f} / {second / 1e3:8.1f} s', flush=True)
first, second = once(lambda: mg.apply(u, f)), once(lambda: mg.apply(u, f))
if mpi.rank == 0:
    print(f'  V-cycle, first / second                         {first:10.1f} / {second:8.1f} ms', flush=True)
fresh = [disc.get_unknown_size_field(name=f'fresh_{i}') for i in range(3)]          # new buffers: first-use cost again?
for i, field in enumerate(fresh):
    first = once(lambda: disc.fft.communicate_ghosts(field))
    if mpi.rank == 0:
        print(f'  first exchange of a new nodal field ({i})         {first:10.1f} ms', flush=True)
if mpi.size > 1:                                           # raw CUDA-aware MPI: one stress face to each neighbour
    send, receive = xp.ones(9 * N * N), xp.empty(9 * N * N)
    up, down = (mpi.rank + 1) % mpi.size, (mpi.rank - 1) % mpi.size
    ms = timed(lambda: mpi.Sendrecv(send, dest=up, recvbuf=receive, source=down))
    if mpi.rank == 0:
        print(f'  raw MPI Sendrecv of a device buffer           {ms:8.2f} ms   ({send.nbytes / 2 ** 20:6.1f} MiB)', flush=True)
