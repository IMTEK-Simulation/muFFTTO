"""
Micro-benchmark: cost of the dual preconditioner P = G K(C_iso A^-1 C_iso) G for the AT2 hybrid material
A = s(x) C_iso, s = g(d) s_0(x), g = (1 - d)^2 + k, s_0 = 1 matrix / 1e-3 pores (polydisperse pores; DIM=2 bilinear
quads, DIM=3 trilinear hexahedra),
with three ways to provide the middle material C_iso A^-1 C_iso = C_iso / s(x):

    full    full C_ijkl field per quadrature point (d^4 q N reals), set after every damage update
    stored  ScaledMaterial with the scalar 1 / s(x) stored (q N reals), set after every damage update
    fly     ScaledMaterial with 1 / s(x) computed from the field s of A at every application (0 extra reals)

Also times the reference operations (one Green application, one K application with A) and the per-design setup.
Checks that the three variants give the same P x.

usage:  DIM=2|3 [ELEMENT=trilinear_hexahedron_1Q] NPIX_LIST=32,64,128 DEVICE=cpu|gpu python bench_dual_inverse.py   -> data/bench_dual_inverse_<dim>D_<device>.txt
"""
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

import time
import numpy as np

from muFFTTO import domain
from muFFTTO import material_models
from muFFTTO import microstructure_library
from muFFTTO import tensor_operations

device = os.environ.get('DEVICE', 'cpu')
if device == 'gpu':
    import cupy as xp
else:
    xp = np
dim = int(os.environ.get('DIM', 2))
element_type = os.environ.get('ELEMENT', 'bilinear_rectangle' if dim == 2 else 'trilinear_hexahedron')
npix_list = [int(n) for n in os.environ.get('NPIX_LIST', '32,64,128' if dim == 2 else '16,32,48').split(',')]
min_time = float(os.environ.get('MIN_TIME', 1.0))  # seconds of repeated calls per timing
k_residual, pore_contrast = 1e-4, 1e-3
E_young, poisson = 1.0, 0.2


def sync():
    if device == 'gpu':
        xp.cuda.Device().synchronize()


def timeit(f):
    """Median time of one call over repeated blocks of calls filling about min_time."""
    f(); sync()                                   # warm-up
    t0 = time.perf_counter(); f(); sync(); one = time.perf_counter() - t0
    reps = max(1, int(min_time / 5 / max(one, 1e-7)))
    blocks = []
    for _ in range(5):
        t0 = time.perf_counter()
        for _ in range(reps):
            f()
        sync()
        blocks.append((time.perf_counter() - t0) / reps)
    return float(np.median(blocks))


lines = []
for n_pix in npix_list:
    cell = domain.PeriodicUnitCell(domain_size=[1] * dim, problem_type='elasticity')
    disc = domain.Discretization(cell=cell, nb_of_pixels_global=(n_pix,) * dim, discretization_type='finite_element',
                                 element_type=element_type,
                                 device=device)
    K_0, G_0 = material_models.get_bulk_and_shear_modulus(E=E_young, poisson=poisson)
    C_0 = material_models.get_elastic_material_tensor(dim=dim, K=K_0, mu=G_0, kind='linear')
    C_0_dev = xp.asarray(C_0)
    phase = microstructure_library.get_geometry(nb_voxels=disc.nb_of_pixels, microstructure_name='polydisperse_pores',
                                                coordinates=disc.fft.coords)
    stiffness = xp.asarray(np.where(tensor_operations.to_host(phase) > 0, 1.0, pore_contrast))
    G_hat = disc.get_preconditioner_Green_mugrid(reference_material_data_ijkl=C_0)

    rng = np.random.default_rng(0)
    d_1n = disc.get_scalar_field(name='damage')
    d_1n.s[...] = xp.asarray(rng.random(d_1n.s.shape))
    d_q = disc.get_quad_field_scalar(name='damage_q')
    disc.apply_N_operator_mugrid(d_1n, d_q)
    s_q = disc.get_quad_field_scalar(name='degraded_scale')
    s_q.s[0, 0] = ((1 - d_q.s[0, 0]) ** 2 + k_residual) * stiffness
    A = domain.ScaledMaterial([(s_q, C_0)])

    # the three middle materials
    full = disc.get_material_data_size_field_mugrid(name='dual_full')
    dual_scale = disc.get_quad_field_scalar(name='dual_scale')

    def setup_full():
        full.s[...] = C_0_dev.reshape(C_0_dev.shape + (1,) * (dim + 1)) / s_q.s[0, 0]

    def setup_stored():
        dual_scale.s[...] = 1.0 / s_q.s

    setup_full(); setup_stored()
    middles = {'full': full,
               'stored': domain.ScaledMaterial([(dual_scale, C_0)]),
               'fly': domain.ScaledMaterial([(lambda: 1.0 / s_q.s[0, 0], C_0)])}

    x = disc.get_unknown_size_field(name='x')
    x.s[...] = xp.asarray(rng.standard_normal(x.s.shape))
    y = disc.get_unknown_size_field(name='y')
    t1 = disc.get_unknown_size_field(name='t1')
    t2 = disc.get_unknown_size_field(name='t2')

    def green(inp, out):
        disc.fft.communicate_ghosts(inp)
        disc.apply_preconditioner_mugrid(preconditioner_Fourier_fnfnqks=G_hat, input_nodal_field_fnxyz=inp,
                                         output_nodal_field_fnxyz=out)

    def K(mat, inp, out):
        disc.apply_system_matrix_mugrid(material_data_field=mat, input_field_inxyz=inp, output_field_inxyz=out,
                                        formulation='small_strain')

    def dual(mat):
        green(x, t1)
        K(mat, t1, t2)
        green(t2, y)
        disc.fft.communicate_ghosts(y)

    # correctness: the three variants give the same P x
    results = {}
    for name, mat in middles.items():
        dual(mat)
        results[name] = np.asarray(y.s.get() if device == 'gpu' else y.s).copy()
    ref = results['full']
    diffs = {k: float(np.abs(v - ref).max() / np.abs(ref).max()) for k, v in results.items()}

    t = {'G': timeit(lambda: green(x, y)),
         'K(A)': timeit(lambda: K(A, x, y)),
         'setup full': timeit(setup_full),
         'setup stored': timeit(setup_stored)}
    for name, mat in middles.items():
        t[f'K({name})'] = timeit(lambda: K(mat, x, y))
        t[f'P({name})'] = timeit(lambda: dual(mat))

    N, q = n_pix ** dim, disc.nb_quad_points_per_pixel
    lines.append(f'N = {n_pix}^{dim}  ({device}, q = {q})   max rel diff vs full: '
                 + ', '.join(f'{k} {v:.1e}' for k, v in diffs.items()))
    lines.append(f'  extra stored reals:  full {dim ** 4 * q * N} ({dim ** 4} q N), stored {q * N} (q N), fly 0')
    for k, v in t.items():
        lines.append(f'  {k:14s} {v * 1e3:9.3f} ms   ({v / t["G"]:5.2f} x G)')
    print('\n'.join(lines[-(2 + len(t)):]), flush=True)

os.makedirs(os.path.join(os.path.dirname(__file__), 'data'), exist_ok=True)
out = os.path.join(os.path.dirname(__file__), 'data', f'bench_dual_inverse_{dim}D_{device}' + (f'_{element_type}' if 'ELEMENT' in os.environ else '') + '.txt')
with open(out, 'w') as fh:
    fh.write('\n'.join(lines) + '\n')
print(f'wrote {out}')
