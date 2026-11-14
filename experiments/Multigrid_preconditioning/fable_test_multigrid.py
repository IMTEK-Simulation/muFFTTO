"""
muFFTTO.multigrid against mg_lib (the multigrid the study was done with): the same V-cycle and the same CG iterations.
    python fable_test_multigrid.py [N, default 16] [element] [device]          (serial)
    mpirun -n 4 python fable_test_multigrid.py 32                              (MPI: iterations as in serial)
With the power-iteration start vectors of mg_lib (serial only) the two cycles agree to round-off.
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from mpi4py import MPI

from muFFTTO import domain, material_models, multigrid, tensor_operations
from muGrid import linalg

N = int(sys.argv[1]) if len(sys.argv) > 1 else 16
element = sys.argv[2] if len(sys.argv) > 2 else 'trilinear_hexahedron_1Q_hg'
device = sys.argv[3] if len(sys.argv) > 3 else 'cpu'
serial = MPI.COMM_WORLD.size == 1
rank0 = MPI.COMM_WORLD.rank == 0
C_0 = material_models.get_elastic_material_tensor(dim=3, K=1.0, mu=0.5, kind='linear')
nq = 1 if '_1Q' in element else 8


def smooth_scale():
    """s in [1e-3, 1]: a smooth periodic pattern, the same on every MPI decomposition"""
    x = (np.arange(N) + 0.5) / N
    X, Y, Z = np.meshgrid(x, x, x, indexing='ij')
    rho = 0.5 + 0.5 * np.sin(2 * np.pi * X) * np.cos(4 * np.pi * Y) * np.sin(2 * np.pi * (Z + 0.3 * X))
    return np.repeat((rho ** 2 * (1 - 1e-3) + 1e-3)[None], nq, axis=0) * (1 + 0.1 * np.arange(nq).reshape(-1, 1, 1, 1))


cell = domain.PeriodicUnitCell(domain_size=[1, 1, 1], problem_type='elasticity')
disc = domain.Discretization(cell=cell, nb_of_pixels_global=(N, N, N), discretization_type='finite_element',
                             element_type=element, device=device,
                             hourglass_eps=0.3 if element.endswith('_hg') else None)
xp = tensor_operations.get_array_module(disc.get_unknown_size_field(name='probe').s)
loc, size = disc.fft.subdomain_locations, disc.fft.nb_subdomain_grid_pts
window = tuple(slice(int(l), int(l) + int(n)) for l, n in zip(loc, size))
s_global = smooth_scale()


def mg_lib_start(index, shape):                     # mg_lib.Level.lmax
    v = np.random.default_rng(1234 + index).standard_normal(shape)
    return v - v.mean(axis=(-3, -2, -1), keepdims=True)


mg = multigrid.Multigrid(disc, C_0, power_start=mg_lib_start if serial else None)
mg.scale.s[0, 0] = xp.asarray(s_global[(slice(None),) + window])
mg.setup()
rng = np.random.default_rng(7)
b_global = rng.standard_normal((3, 1, N, N, N))
b_global -= b_global.mean(axis=(-3, -2, -1), keepdims=True)
b, z = disc.get_unknown_size_field(name='b'), disc.get_unknown_size_field(name='z')
b.s[...] = xp.asarray(b_global[(Ellipsis,) + window])
mg.apply(b, z)
if rank0:
    print(f'N {N} {element} {device} ranks {MPI.COMM_WORLD.size}: levels {[lv.disc.nb_of_pixels_global[0] for lv in mg.levels]}'
          f' tail {None if mg.tail is None else [lv.disc.nb_of_pixels_global[0] for lv in mg.tail.levels]}'
          f' lmax {[round(lv.lmax, 6) for lv in mg.levels if lv.lmax]}')

if serial:
    import mg_lib
    reference = mg_lib.Hierarchy(s_global, C_0, element_type=element, hg_eps=0.3, n_coarsest=4, degree=2, device=device,
                                 scalar=True)
    reference.setup('mg')
    z_ref = reference.preconditioner('mg')(xp.asarray(b_global))
    print(f'  V-cycle vs mg_lib: rel diff {float(xp.linalg.norm(z.s - z_ref) / xp.linalg.norm(z_ref)):.2e}'
          f'   lmax mg_lib {[round(lv.lmax("jacobi"), 6) for lv in reference.levels[:-1]]}')

if device == 'cpu':                                 # array-based grid transfer (device fallback) == muGrid kernel
    z_kernel = z.s.copy()
    mg.transfer.use_kernel = False
    mg.apply(b, z)
    mg.transfer.use_kernel = True
    difference = disc.communicator.sum(np.float64(float(np.sum((z.s - z_kernel) ** 2))))
    if rank0:
        print(f'  array grid transfer vs muGrid kernel: abs diff {np.sqrt(difference):.2e}')

# CG with the cycle (plain PCG, relative residual 1e-8): iteration count
x, r, p, Ap = (disc.get_unknown_size_field(name=n) for n in ('x', 'r', 'p', 'Ap'))
dot = lambda u, v: disc.communicator.sum(np.float64(float(linalg.vecdot(u, v))))
linalg.copy(b, r)
mg.apply(r, z)
linalg.copy(z, p)
rz, r0 = dot(r, z), np.sqrt(dot(r, r))
for it in range(1, 200):
    mg.levels[0].apply_K(p, Ap)
    a = rz / dot(p, Ap)
    linalg.axpy(a, p, x)
    linalg.axpy(-a, Ap, r)
    if np.sqrt(dot(r, r)) < 1e-8 * r0:
        break
    mg.apply(r, z)
    rz_new = dot(r, z)
    linalg.axpby(1.0, z, rz_new / rz, p)
    rz = rz_new
energy = dot(x, b)
if rank0:
    print(f'  CG iterations {it}, energy x.b = {energy:.12e}')
