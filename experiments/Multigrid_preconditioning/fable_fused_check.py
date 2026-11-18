"""
Is muGrid's fused IsotropicStiffnessOperator (element q1) the stiffness of muFFTTO's 'trilinear_hexahedron' with one
material value per voxel?  And what does one application cost?
    python fable_fused_check.py [N, default 16] [cpu | gpu]
Compares K u, diag(K), the macro right-hand side and the average stress of the two implementations on a random
voxel-wise scale s (A = s C_0), then times K u (and the one-point element with hourglass control for reference).
"""
import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
import muGrid
import numpy as np

from muFFTTO import domain, material_models, tensor_operations

N = int(sys.argv[1]) if len(sys.argv) > 1 else 16
device = sys.argv[2] if len(sys.argv) > 2 else 'cpu'
K_0, G_0 = 1.0, 0.5
lam_0 = K_0 - 2.0 * G_0 / 3.0
C_0 = material_models.get_elastic_material_tensor(dim=3, K=K_0, mu=G_0, kind='linear')
cell = domain.PeriodicUnitCell(domain_size=[1, 1, 1], problem_type='elasticity')
disc = domain.Discretization(cell=cell, nb_of_pixels_global=(N,) * 3, discretization_type='finite_element',
                             element_type='trilinear_hexahedron', device=device)
fc = disc.field_collection
scale = disc.get_quad_field_scalar(name='scale')
material = domain.ScaledMaterial([(scale, C_0)])
u, f_lib, f_fused = (disc.get_unknown_size_field(name=n) for n in ('u', 'f_lib', 'f_fused'))
lam, mu = fc.real_field('lambda'), fc.real_field('mu')
xp = tensor_operations.get_array_module(u.s)
rng = np.random.default_rng(0)
s = 10.0 ** rng.uniform(-3, 0, size=(N, N, N))
scale.s[0, 0] = xp.asarray(s)[None]                        # the same value at the 8 Gauss points of a voxel
lam.p[...] = xp.asarray(lam_0 * s).reshape(lam.p.shape)
mu.p[...] = xp.asarray(G_0 * s).reshape(mu.p.shape)
disc.fft.communicate_ghosts(lam)
disc.fft.communicate_ghosts(mu)
u.s[...] = xp.asarray(rng.standard_normal(u.s.shape))
operator = muGrid.IsotropicStiffnessOperator3D([1.0 / N] * 3, muGrid.FEMElement.q1)


def K_library():
    disc.apply_system_matrix_mugrid(material, u, f_lib, formulation='small_strain')


def K_fused():
    disc.fft.communicate_ghosts(u)
    operator.apply(u, lam, mu, f_fused)


K_library()
K_fused()
norm = float(xp.linalg.norm(f_lib.s))
print(f'N {N} {device}: K u, fused vs library 8-point hexahedron: rel diff {float(xp.linalg.norm(f_fused.s - f_lib.s)) / norm:.1e}')

diagonal = disc.get_unknown_size_field(name='diagonal')
operator.assemble_diagonal(lam, mu, diagonal)
jacobi = disc.get_preconditioner_Jacobi_mugrid(material_data_field_ijklqxyz=material, formulation='small_strain')
reference = 1.0 / jacobi.s ** 2
print(f'  diag(K): rel diff {float(xp.abs(diagonal.s - reference).max() / xp.abs(reference).max()):.1e}')

E = np.zeros((3, 3))
E[0, 0], E[0, 1], E[1, 0] = 1.0, 0.3, 0.3
macro = disc.get_gradient_size_field(name='macro')
disc.get_macro_gradient_field_mugrid(E, macro)
disc.get_rhs_mugrid(material, macro, f_lib)                                        # -B^T W C E
operator.apply_macro_rhs(lam, mu, list(E.ravel()), f_fused)                        # +B^T C E
print(f'  macro right-hand side: rel diff {float(xp.linalg.norm(f_fused.s + f_lib.s) / xp.linalg.norm(f_lib.s)):.1e}')
stress_library = disc.get_homogenized_stress_mugrid(material, u, macro, formulation='small_strain')
stress_fused = np.array(operator.average_stress(u, lam, mu, list(E.ravel()))).reshape(3, 3)
print(f'  average stress: rel diff {np.abs(stress_fused - tensor_operations.to_host(stress_library)).max() / np.abs(tensor_operations.to_host(stress_library)).max():.1e}')


def timed(label, fun, repeat=10):
    fun()
    sync = (lambda: xp.cuda.Device().synchronize()) if xp is not np else (lambda: None)
    sync()
    start = time.perf_counter()
    for _ in range(repeat):
        fun()
    sync()
    print(f'  {label:<52s} {1e3 * (time.perf_counter() - start) / repeat:8.2f} ms')


timed('K u, library 8-point hexahedron (stencils)', K_library)
timed('K u, fused q1 kernel', K_fused)
hg = domain.Discretization(cell=cell, nb_of_pixels_global=(N,) * 3, discretization_type='finite_element',
                           element_type='trilinear_hexahedron_1Q_hg', device=device, hourglass_eps=0.3)
hg_scale = hg.get_quad_field_scalar(name='scale')
hg_scale.s[0, 0] = xp.asarray(s)[None]
hg_material = domain.ScaledMaterial([(hg_scale, C_0)])
hg_u, hg_f = hg.get_unknown_size_field(name='u'), hg.get_unknown_size_field(name='f')
hg_u.s[...] = u.s
timed('K u, one-point hexahedron + hourglass (stencils)',
      lambda: hg.apply_system_matrix_mugrid(hg_material, hg_u, hg_f, formulation='small_strain'))
