"""
Projection of a nodal scalar field of a periodic N^3 grid onto the fields with cubic symmetry.

The symmetry group G of the cube about the cell centre has 48 elements: the 8 combinations of the reflections
x_a -> 1 - x_a (on the periodic nodal grid: node i -> (N - i) mod N) times the 6 permutations of the axes. The
projection is the group mean, and because G = (reflections) x (swap x, y) x (cyclic permutations) it factorises:

    P = R_x R_y R_z  *  (I + T) / 2  *  (I + C + C^2) / 3
        R_a = (I + reflection_a) / 2,   T: (x, y, z) -> (y, x, z),   C: (C f)(x, y, z) = f(y, z, x)

Serial: `cubic_mean` on the array (GPU or host).  MPI (slabs along z, one per rank): R_x, R_y and T act inside a
slab; C, C^2 and the z-reflection move data between slabs and are done by `SlabRemap` -- every rank sends each other
rank exactly the block it needs (the whole field crosses the network three times per projection, nothing is
gathered on one rank).
"""
import numpy as np
from mpi4py import MPI


def reflect(xp, a, axis):
    """a[(N - i) mod N] along `axis`"""
    return xp.roll(xp.flip(a, axis), 1, axis)


def cubic_mean(xp, a):
    """group mean of the whole array a[x, y, z] (serial; numpy or cupy)"""
    a = (a + a.transpose(1, 2, 0) + a.transpose(2, 0, 1)) / 3.0             # (I + C + C^2) / 3
    a = 0.5 * (a + a.transpose(1, 0, 2))                                    # (I + T) / 2
    for axis in range(3):
        a = 0.5 * (a + reflect(xp, a, axis))                                # R_x R_y R_z
    return a


class SlabRemap:
    """(g f)(t) = f(S(t)) for a grid symmetry S(t)_m = t[perm[m]] (reflected to (N - .) mod N if flip[m]) of a field
    distributed in slabs along the last axis. Host arrays; every rank holds f[:, :, z0:z1]."""

    def __init__(self, comm, n, z_start, z_size):
        self.comm, self.n = comm, n
        starts = comm.allgather(int(z_start))
        sizes = comm.allgather(int(z_size))
        self.slabs = [np.arange(start, start + size) for start, size in zip(starts, sizes)]
        self.full = np.arange(n)

    def __call__(self, local, perm, flip=(False, False, False)):
        comm, rank = self.comm, self.comm.rank
        source = lambda t, m: (self.n - t) % self.n if flip[m] else t

        def blocks(target_rank, source_rank):
            """target index lists (per target axis) of the nodes of `target_rank` whose source lies on `source_rank`,
            and the matching source index lists (per source axis)"""
            target = [self.full, self.full, self.slabs[target_rank]]
            axis = perm[2]                                   # the target axis that becomes the source z
            owned = np.isin(source(target[axis], 2), self.slabs[source_rank])
            target[axis] = target[axis][owned]
            return target, [source(target[perm[m]], m) for m in range(3)]

        requests = []
        for target_rank in range(comm.size):                # send: this rank is the source
            _, src = blocks(target_rank, rank)
            block = local[np.ix_(src[0], src[1], src[2] - self.slabs[rank][0])]
            block = np.ascontiguousarray(block.transpose([perm.index(k) for k in range(3)]))      # target axis order
            requests.append(comm.isend(block, dest=target_rank, tag=77))
        result = np.empty_like(local)
        for source_rank in range(comm.size):                # receive: this rank is the target
            target, _ = blocks(rank, source_rank)
            block = comm.recv(source=source_rank, tag=77)
            result[np.ix_(target[0], target[1], target[2] - self.slabs[rank][0])] = block
        MPI.Request.waitall(requests)
        return result


def cubic_mean_distributed(remap, local):
    """group mean of a slab-distributed host array (see the module docstring)"""
    a = (local + remap(local, [1, 2, 0]) + remap(local, [2, 0, 1])) / 3.0   # (I + C + C^2) / 3
    a = 0.5 * (a + a.transpose(1, 0, 2))                                    # (I + T) / 2, inside the slab
    for axis in range(2):
        a = 0.5 * (a + reflect(np, a, axis))                                # R_x, R_y inside the slab
    return 0.5 * (a + remap(a, [0, 1, 2], flip=(False, False, True)))       # R_z


if __name__ == '__main__':                                  # mpirun -n 4 python cubic_symmetry.py [N]
    import sys
    comm = MPI.COMM_WORLD
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 16
    field = np.random.default_rng(0).standard_normal((n, n, n))
    bounds = [n * r // comm.size for r in range(comm.size + 1)]
    z0, z1 = bounds[comm.rank], bounds[comm.rank + 1]
    remap = SlabRemap(comm, n, z0, z1 - z0)
    projected = cubic_mean_distributed(remap, field[:, :, z0:z1].copy())
    reference = cubic_mean(np, field)
    error = comm.allreduce(float(np.abs(projected - reference[:, :, z0:z1]).max()), op=MPI.MAX)
    again = cubic_mean(np, reference)
    if comm.rank == 0:
        print(f'N {n}, {comm.size} rank(s): distributed vs serial projection, max diff {error:.1e}; '
              f'idempotent {np.abs(again - reference).max():.1e}; invariant under x<->z '
              f'{np.abs(reference - reference.transpose(2, 1, 0)).max():.1e}, under z reflection '
              f'{np.abs(reference - reflect(np, reference, 2)).max():.1e}')
