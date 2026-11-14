"""muGrid features of the environment this runs in (NEMO: inside a job; the login node cannot import the GPU stack)"""
import sys

import muGrid
from mpi4py import MPI

print(getattr(muGrid, '__version__', ''), 'mpi', muGrid.has_mpi, 'cuda', muGrid.has_cuda)
print([n for n in ('GridTransfer', 'linalg', 'memory_profile', 'use_cupy_allocator') if hasattr(muGrid, n)])
print('GridTransfer3D.restrict overloads:\n', muGrid._muGrid.GridTransfer3D.restrict.__doc__)
device = sys.argv[1] if len(sys.argv) > 1 else 'gpu'
engines = [muGrid.FFTEngine(nb_domain_grid_pts=(n,) * 3, communicator=muGrid.Communicator(MPI.COMM_WORLD),
                            nb_ghosts_left=[1] * 3, nb_ghosts_right=[1] * 3, device=device) for n in (16, 8)]
fields = []
for engine in engines:
    engine.real_space_collection.set_nb_sub_pts('nodal_points', 1)
    fields.append(engine.real_space_collection.real_field(name='u', components=(3,), sub_pt='nodal_points'))
try:
    muGrid.GridTransfer(3).restrict(fields[0], fields[1])
    muGrid.GridTransfer(3).prolong(fields[1], fields[0])
    print(f'GridTransfer on {device} fields: works')
except TypeError as error:
    print(f'GridTransfer on {device} fields: NOT available ({str(error)[:80]} ...)')
print('linalg:', [n for n in dir(muGrid.linalg) if not n.startswith('_')])
