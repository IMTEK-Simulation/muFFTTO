"""
Where does the time of one stiffness application K u go? (1Q_hg, ScaledMaterial s C_0, GPU)
    python fable_time_K.py [N, default 64] [device, default gpu]
Times (ms, median of 10, device synchronised) of the pieces of Discretization.apply_system_matrix_mugrid.
"""
import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
import numpy as np

from muFFTTO import domain, material_models, tensor_operations
from muGrid import linalg

N = int(sys.argv[1]) if len(sys.argv) > 1 else 64
device = sys.argv[2] if len(sys.argv) > 2 else 'gpu'
C_0 = material_models.get_elastic_material_tensor(dim=3, K=1.0, mu=0.5, kind='linear')
cell = domain.PeriodicUnitCell(domain_size=[1, 1, 1], problem_type='elasticity')
disc = domain.Discretization(cell=cell, nb_of_pixels_global=(N, N, N), discretization_type='finite_element',
                             element_type='trilinear_hexahedron_1Q_hg', device=device, hourglass_eps=0.3)
u, f = disc.get_unknown_size_field(name='u'), disc.get_unknown_size_field(name='f')
grad = disc.get_gradient_size_field(name='grad_field_temporary')
scale = disc.get_quad_field_scalar(name='scale')
xp = tensor_operations.get_array_module(u.s)
u.s[...] = xp.asarray(np.random.default_rng(0).standard_normal(u.s.shape))
scale.s[...] = 0.5
material = domain.ScaledMaterial([(scale, C_0)])
hg = disc.hourglass_control


def sync():
    if xp is not np:
        xp.cuda.Device().synchronize()


def timed(label, fun, repeat=10):
    fun()
    times = []
    for _ in range(repeat):
        sync()
        t = time.perf_counter()
        fun()
        sync()
        times.append(time.perf_counter() - t)
    print(f'  {label:<44s} {1e3 * np.median(times):8.2f} ms', flush=True)


print(f'N {N} {device}')
timed('K u total (apply_system_matrix_mugrid)',
      lambda: disc.apply_system_matrix_mugrid(material, u, f, formulation='small_strain'))
timed('communicate_ghosts(u)', lambda: disc.fft.communicate_ghosts(u))
timed('gradient_op.apply', lambda: disc.gradient_op.apply(nodal_field=u, quadrature_point_field=grad))
timed('ScaledMaterial.apply (C : g)', lambda: disc.apply_material_data_mugrid(material, grad))
timed('communicate_ghosts(grad)', lambda: disc.fft.communicate_ghosts(grad))
timed('gradient_op.transpose', lambda: disc.gradient_op.transpose(
    quadrature_point_field=grad, nodal_field=f, weights=disc.quadrature_weights))
timed('hourglass: add_stiffness total', lambda: hg.add_stiffness(material, u, f))
timed('hourglass: H apply', lambda: hg.operator.apply(nodal_field=u, quadrature_point_field=hg.strain))
timed('hourglass: strain *= eps kappa (in place)', lambda: hg._scale_by_stiffness(material))
timed('hourglass: H^T', lambda: hg.operator.transpose(quadrature_point_field=hg.strain, nodal_field=hg.force,
                                                      weights=[1.0]))
timed('hourglass: output += force', lambda: f.s.__iadd__(hg.force.s))
from muFFTTO import multigrid
mg = multigrid.Multigrid(disc, C_0)
mg.scale.s[...] = 0.5
mg.setup()
timed('multigrid: lean K (fine level)', lambda: mg.levels[0].apply_K(u, f))
timed('multigrid: K of level 1 (8-point, N/2)', lambda: mg.levels[1].apply_K(mg.levels[1].x, mg.levels[1].b))
timed('multigrid: restrict', lambda: mg.transfer.restrict(u, mg.levels[1].b))
timed('multigrid: prolong', lambda: mg.transfer.prolong(mg.levels[1].x, f))
timed('multigrid: Jacobi product', lambda: mg._jacobi(mg.levels[0], u, f))
timed('multigrid: V-cycle', lambda: mg.apply(u, f))
timed('multigrid: setup', lambda: mg.setup(), repeat=2)
timed('linalg.axpy', lambda: linalg.axpy(0.5, u, f))
timed('cupy  f.s += 0.5 * u.s', lambda: f.s.__iadd__(0.5 * u.s))
timed('linalg.vecdot', lambda: linalg.vecdot(u, f))
timed('np.dot(ravel, ravel)', lambda: float(np.dot(u.s.ravel(), f.s.ravel())))
