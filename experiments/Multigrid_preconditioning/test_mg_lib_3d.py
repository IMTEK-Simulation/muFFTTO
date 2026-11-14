"""
3D checks of mg_lib (trilinear hexahedra, option (a) re-discretised coarse grids):
  1. R = P^T for the trilinear transfers
  2. homogeneous material: re-discretisation == Galerkin (2x2x2 Gauss integrates Q1 exactly) -> transfers and the
     Gauss-point ordering are right; heterogeneous: the size of the re-discretisation mismatch (expected > 0)
  3. the V-cycle preconditioner is symmetric and positive definite
  4. CG iterations (library CG, energy stop rtol 1e-6, d_min 0) on a smooth 3D phase field, contrast 1e-4:
     MG vs Green-Jacobi vs Green, N = 16, 32

    python test_mg_lib_3d.py
"""
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

from muFFTTO import material_models, solvers

import mg_lib

C_0 = material_models.get_elastic_material_tensor(dim=3, K=1.0, mu=0.5, kind='linear')
rng = np.random.default_rng(0)


def zero_mean(x):
    return x - x.mean(axis=(-3, -2, -1), keepdims=True)


# ---------------------------------------------------------------- 1, 2
N = 8
s_hom = np.ones((8, N, N, N))
hier = mg_lib.Hierarchy(s_hom, C_0, n_coarsest=4, scalar=True)
fine, coarse = hier.levels[0], hier.levels[1]
print('levels', [lv.N for lv in hier.levels], 'field shape', fine.shape)
uH = rng.standard_normal(coarse.shape)
rh = rng.standard_normal(fine.shape)
a, b = np.sum(mg_lib.prolong_tensor(uH) * rh), np.sum(uH * mg_lib.restrict_tensor(rh))
print(f'1. adjointness   rel diff {abs(a - b) / abs(a):.1e}')
g, d = mg_lib.restrict_tensor(fine.K(mg_lib.prolong_tensor(uH))), coarse.K(uH)
print(f'2. homogeneous:  |R K_h P u - K_H u| / |K_H u| = {np.linalg.norm(g - d) / np.linalg.norm(d):.1e}  (exact: ~1e-16)')
s_het = 10.0 ** rng.uniform(-4, 0, size=(8, N, N, N))
hier_h = mg_lib.Hierarchy(s_het, C_0, n_coarsest=4, scalar=True)
g = mg_lib.restrict_tensor(hier_h.levels[0].K(mg_lib.prolong_tensor(uH)))
d = hier_h.levels[1].K(uH)
print(f'   random 1e-4..1: re-discretisation mismatch {np.linalg.norm(g - d) / np.linalg.norm(g):.2f} (option (a))')

# ---------------------------------------------------------------- 3
hier_h.setup('mg')
B = hier_h.preconditioner('mg')
x, y = zero_mean(rng.standard_normal(fine.shape)), zero_mean(rng.standard_normal(fine.shape))
yBx, xBy = np.sum(y * B(x)), np.sum(x * B(y))
xBx = min(np.sum(v * B(v)) for v in [zero_mean(rng.standard_normal(fine.shape)) for _ in range(5)])
print(f'3. V-cycle asym {abs(yBx - xBy) / abs(yBx):.1e}, min (x, Bx) {xBx:.3e}')


# ---------------------------------------------------------------- 4
def smooth_phase(N, eta_h=2.0, seed=0):
    """random smooth level set (Fourier modes |k| <= 3), tanh profile of width eta = eta_h pixels"""
    r = np.random.default_rng(seed)
    k = np.fft.fftfreq(N, 1 / N)
    KX, KY, KZ = np.meshgrid(k, k, k, indexing='ij')
    spec = (np.sqrt(KX ** 2 + KY ** 2 + KZ ** 2) <= 3) * (r.standard_normal((N, N, N)) + 1j * r.standard_normal((N, N, N)))
    f = np.real(np.fft.ifftn(spec))
    f /= np.abs(f).max()
    return 0.5 * (1 + np.tanh(f * N / 8 / (2 * eta_h)))         # interface slope ~ N/8 per unit f


print('4. CG iterations (energy stop 1e-6, d_min 0), smooth 3D phase field, soft 1e-4:')
for N in (16, 32):
    phi = smooth_phase(N)
    hier = mg_lib.Hierarchy(np.ones((8, N, N, N)), C_0, n_coarsest=4, scalar=True)
    fine = hier.levels[0]
    phq = fine.disc.get_quad_field_scalar(name='phase_q')
    ph = fine.disc.get_scalar_field(name='phase')
    ph.s[0, 0] = phi
    fine.disc.apply_N_operator_mugrid(ph, phq)
    s_q = np.asarray(phq.s[0, 0]) ** 2 * (1 - 1e-4) + 1e-4
    hier.update_material(s_q)
    macro = fine.disc.get_gradient_size_field(name='macro')
    E = np.zeros((3, 3)); E[0, 0] = 1.0
    fine.disc.get_macro_gradient_field_mugrid(macro_gradient_ij=E, macro_gradient_field_ijqxyz=macro)
    rhs = fine.disc.get_unknown_size_field(name='rhs')
    fine.disc.get_rhs_mugrid(material_data_field_ijklqxyz=fine.material, macro_gradient_field_ijqxyz=macro,
                             rhs_inxyz=rhs)
    x = fine.disc.get_unknown_size_field(name='sol')
    row = []
    for method in ('mg', 'green_jacobi', 'green'):
        hier.setup(method)
        Bm = hier.preconditioner(method)
        x.s[...] = 0
        it = {'n': 0}
        solvers.conjugate_gradients_mugrid_experimental(
            comm=fine.disc.communicator, fc=fine.disc.field_collection,
            hessp=lambda a, b: fine._apply_K(fine.material, a, b), b=rhs, x=x,
            P=lambda r, z: z.s.__setitem__(Ellipsis, Bm(r.s)), tol=1e-6, maxiter=5000,
            callback=lambda i, *a: it.__setitem__('n', i), stop_crit_norm='energy_lower_estim', energy_rtol=True,
            d_min=0, tau=0.25)
        row.append(f'{method} {it["n"]}')
    print(f'   N {N}: ' + ', '.join(row) + f'   (s in [{s_q.min():.1e}, {s_q.max():.2f}])')
