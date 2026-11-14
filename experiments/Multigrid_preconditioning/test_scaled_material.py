"""mg_lib scalar path now stores ScaledMaterial: check K, Jacobi, dual K and the V-cycle against the full-material
path (same material C = s C_0), 2D and 3D, and time one K application (laptop CPU)."""
import sys, os, time
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from muFFTTO import material_models
import mg_lib

rng = np.random.default_rng(0)
for dim, N, q in [(2, 32, 2), (3, 16, 8), (3, 32, 8)]:
    C_0 = material_models.get_elastic_material_tensor(dim=dim, K=1.0, mu=0.5, kind='linear')
    s_q = 10.0 ** rng.uniform(-4, 0, size=(q,) + (N,) * dim)
    C_full = np.einsum('ijkl,q...->ijklq...', C_0, s_q)
    hs = mg_lib.Hierarchy(s_q, C_0, scalar=True)
    hf = mg_lib.Hierarchy(C_full, C_0, scalar=False)
    x = rng.standard_normal(hs.levels[0].shape)
    ref = hf.levels[0].disc.get_material_data_size_field_mugrid(name='ref_dual')     # full C_0 / s (any dim)
    ref.s[...] = np.einsum('ijkl,q...->ijklq...', C_0, 1.0 / s_q)
    hf.levels[0]._material_dual = ref
    rel = lambda a, b: np.linalg.norm(a - b) / np.linalg.norm(b)
    print(f'{dim}D N {N}: K {rel(hs.levels[0].K(x), hf.levels[0].K(x)):.1e}  '
          f'dual {rel(hs.levels[0].M("dual", x), hf.levels[0].M("dual", x)):.1e}  '
          f'Jacobi {rel(hs.levels[0].M("jacobi", x), hf.levels[0].M("jacobi", x)):.1e}', end='')
    if dim == 2:     # 2D: coarse levels identical too (Galerkin averaging of s == of C)
        hs.setup('mg'); hf.setup('mg')
        print(f'  V-cycle {rel(hs.preconditioner("mg")(x), hf.preconditioner("mg")(x)):.1e}', end='')
    ts, tf = [], []
    for _ in range(5):
        t0 = time.perf_counter(); hs.levels[0].K(x); ts.append(time.perf_counter() - t0)
        t0 = time.perf_counter(); hf.levels[0].K(x); tf.append(time.perf_counter() - t0)
    print(f'   K time scaled {min(ts) * 1e3:.1f} ms vs full {min(tf) * 1e3:.1f} ms', flush=True)
