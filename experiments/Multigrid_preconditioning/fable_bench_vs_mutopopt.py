"""
The state solve of a 3D topology optimisation, muFFTTO (this study) against muTopOpt, on the same design:

    K(rho) u = -B^T C(rho) : E      for the 6 unit macro strains, cold start, CG to |r| <= rtol |b|

    python fable_bench_vs_mutopopt.py --code mufftto  -n 128
    python fable_bench_vs_mutopopt.py --code mutopopt -n 128 [--precision single] [--element p1]

Same in both codes: grid, material (SIMP C = (k + rho^p (1 - k)) C_solid, E 1, nu 0.3, p 2, k 1e-3), one density per
voxel, load cases, stopping rule, timing (wall time around each solve with the device synchronised, after one untimed
warm-up solve). Different, and the point of the comparison:

                    muFFTTO (to3d_large.py)                         muTopOpt 1.5
    element         one-point hexahedron + hourglass control        trilinear hexahedron q1 (or p1 simplices)
    operator        muGrid stencils + CuPy pointwise kernels        muGrid fused IsotropicStiffnessOperator
    preconditioner  geometric multigrid V(2,2)                      Green-Jacobi (J-FFT)

Design: muTopOpt's start design (`filtered_random`, seed 0, correlation length 3 voxels: smooth grey blobs), or
`--design file.npy` with voxel densities [N, N, N] (e.g. a converged design: sharp interfaces, full contrast).
`--sharpen w` maps the random design through 1/2 (1 + tanh((rho - 1/2) / w)): sharper interfaces without a file.

Reported: CG iterations per solve, ms per solve, ms per CG iteration, set-up time for a new design (multigrid
set-up / preconditioner refresh), peak GPU memory of the process, and <sigma_xx> of the xx load case (the two
discretisations must agree on the physics to a few per cent).
"""
import argparse
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--code', choices=('mufftto', 'mutopopt'), required=True)
parser.add_argument('-n', type=int, default=64)
parser.add_argument('--rtol', type=float, default=1e-6)
parser.add_argument('--design', default='random')
parser.add_argument('--sharpen', type=float, default=0.0)
parser.add_argument('--precision', default='double', choices=('double', 'single'))        # muTopOpt only
parser.add_argument('--element', default=None)   # mufftto: trilinear_hexahedron_1Q_hg | trilinear_hexahedron; mutopopt: q1 | p1
parser.add_argument('--preconditioner', default='green-jacobi')                           # muTopOpt only
parser.add_argument('--operator', default='stencil', choices=('stencil', 'fused'))        # muFFTTO only
parser.add_argument('--mg-precision', default='double', choices=('double', 'single'))     # muFFTTO, fused only
parser.add_argument('--lmax', default='bound', choices=('bound', 'power'))             # muFFTTO: smoother interval end
args = parser.parse_args()
N, E_SOLID, NU_SOLID, PENALTY, VOID = args.n, 1.0, 0.3, 2.0, 1e-3


def design():
    """voxel densities [N, N, N] on the host"""
    if args.design != 'random':
        rho = np.load(args.design)
        assert rho.shape == (N, N, N), rho.shape
        return rho
    from muTopOpt.optimize import initial_density
    rho = initial_density((N, N, N), kind='filtered_random', seed=0, length=3.0 / N, grid_spacing=(1.0 / N,) * 3)
    if args.sharpen > 0:
        rho = 0.5 * (1.0 + np.tanh((rho - 0.5) / args.sharpen))
    return rho


def unit_strains():
    strains = []
    for i, j in [(0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2)]:
        E = np.zeros((3, 3))
        E[i, j] = E[j, i] = 1.0 if i == j else 0.5
        strains.append(E)
    return strains


def gpu_memory_MiB():
    out = subprocess.run(['nvidia-smi', '--query-compute-apps=pid,used_memory', '--format=csv,noheader,nounits'],
                         capture_output=True, text=True).stdout
    return max([int(line.split(',')[1]) for line in out.splitlines() if int(line.split(',')[0]) == os.getpid()] + [0])


rho = design()
strains = unit_strains()

if args.code == 'mutopopt':
    import muGrid
    from muTopOpt import Homogenization, SimpMaterial

    element = args.element or 'q1'
    material = SimpMaterial(E_solid=E_SOLID, nu=NU_SOLID, penalty=PENALTY, void_ratio=VOID)
    homog = Homogenization((N, N, N), material, comm=muGrid.Communicator.world(), element=element,
                           preconditioner=args.preconditioner, device='gpu',
                           dtype={'single': np.float32, 'double': np.float64}[args.precision])
    sync = lambda: homog._xp.cuda.runtime.deviceSynchronize()
    label = f'muTopOpt {element} {args.preconditioner} {args.precision}'
    u = homog.vector_field('bench-u')

    def setup():
        homog.set_density(rho)

    def solve(E):
        u.set_zero()
        homog.solve_macro(E, u, rtol=args.rtol)
        return homog.last_cg_iters

    def stress_xx():
        solve(strains[0])
        return homog.homogenized_stress(u, strains[0])[0, 0]
else:
    from muGrid import linalg
    from muFFTTO import domain, multigrid, tensor_operations

    fused = args.operator == 'fused'      # muGrid's fused q1 kernel on every level: the element of muTopOpt's 'q1'
    element = 'trilinear_hexahedron_1Q' if fused else (args.element or 'trilinear_hexahedron_1Q_hg')
    K_solid, G_solid = E_SOLID / (3 * (1 - 2 * NU_SOLID)), E_SOLID / (2 * (1 + NU_SOLID))
    identity = np.eye(3)
    C_0 = (K_solid - 2 * G_solid / 3) * np.einsum('ij,kl->ijkl', identity, identity) + G_solid * (
        np.einsum('ik,jl->ijkl', identity, identity) + np.einsum('il,jk->ijkl', identity, identity))
    cell = domain.PeriodicUnitCell(domain_size=[1, 1, 1], problem_type='elasticity')
    disc = domain.Discretization(cell=cell, nb_of_pixels_global=(N, N, N), discretization_type='finite_element',
                                 element_type=element, device='gpu',
                                 hourglass_eps=0.3 if element.endswith('_hg') else None)
    mg = multigrid.Multigrid(disc, C_0, operator=args.operator, precision=args.mg_precision, lmax=args.lmax)
    xp = tensor_operations.get_array_module(mg.scale.s)
    sync = lambda: xp.cuda.Device().synchronize()
    label = (f'muFFTTO fused q1, multigrid V(2,2) in {args.mg_precision} precision, CG double' if fused
             else f'muFFTTO {element} multigrid V(2,2) double')
    u, b = disc.get_unknown_size_field(name='u'), disc.get_unknown_size_field(name='b')
    r, z, p, Kp = (disc.get_unknown_size_field(name=name) for name in ('r', 'z', 'p', 'Kp'))
    level = mg.fine_operator
    dot = lambda a, c: float(linalg.vecdot(a, c))
    weight = float(tensor_operations.to_host(disc.quadrature_weights)[0])

    def setup():
        mg.scale.s[0, 0] = xp.asarray(VOID + rho ** PENALTY * (1.0 - VOID))[None]      # the same scale at every q
        mg.setup()

    def solve(E):
        if fused:
            level.operator.apply_macro_rhs(level.lam, level.mu, [float(e) for e in E.ravel()], b)
            linalg.scal(-1.0, b)
        else:
            stress = np.einsum('ijkl,kl->ij', C_0, E)
            for i in range(3):
                for j in range(3):
                    xp.multiply(mg.scale.s[0, 0], -stress[i, j], out=level.stress.s[i, j])
            disc.fft.communicate_ghosts(level.stress)
            disc.gradient_op.transpose(quadrature_point_field=level.stress, nodal_field=b,
                                       weights=disc.quadrature_weights)
        linalg.scal(0.0, u)
        linalg.copy(b, r)
        target = args.rtol ** 2 * dot(b, b)
        mg.apply(r, z)
        linalg.copy(z, p)
        rz = dot(r, z)
        for iteration in range(1, 10000):
            mg.apply_K(p, Kp)
            alpha = rz / dot(p, Kp)
            linalg.axpy(alpha, p, u)
            linalg.axpy(-alpha, Kp, r)
            if dot(r, r) <= target:
                return iteration
            mg.apply(r, z)
            rz_new = dot(r, z)
            linalg.axpby(1.0, z, rz_new / rz, p)
            rz = rz_new

    def stress_xx():
        solve(strains[0])
        if fused:
            disc.fft.communicate_ghosts(u)
            return level.operator.average_stress(u, level.lam, level.mu, [float(e) for e in strains[0].ravel()])[0]
        disc.apply_gradient_operator_mugrid(u_inxyz=u, grad_u_ijqxyz=level.stress)
        level.stress.s[...] += xp.asarray(strains[0]).reshape(3, 3, 1, 1, 1, 1)
        mean = np.array([[float(xp.sum(mg.scale.s[0, 0] * level.stress.s[i, j])) for j in range(3)] for i in range(3)])
        return float(np.einsum('kl,kl->', C_0[0, 0], mean)) * weight


def timed(fun, *arguments):
    sync()
    start = time.perf_counter()
    result = fun(*arguments)
    sync()
    return time.perf_counter() - start, result


setup_first, _ = timed(setup)
solve(strains[0])                                           # warm-up: kernel compilation, FFT plans
setup_again, _ = timed(setup)
times, iterations = zip(*[timed(solve, E) for E in strains])
sigma = stress_xx()
print(f'{label}: N {N}^3, design {args.design}' + (f' sharpened {args.sharpen}' if args.sharpen else '')
      + f', rtol {args.rtol:g}')
print(f'  CG iterations per solve {np.mean(iterations):7.1f}  (per load case: {list(iterations)})')
print(f'  ms per solve            {1e3 * np.mean(times):9.1f}')
print(f'  ms per CG iteration     {1e3 * sum(times) / sum(iterations):9.2f}')
print(f'  set-up for a design     {setup_first:9.2f} s first, {setup_again:.2f} s later')
print(f'  peak GPU memory         {gpu_memory_MiB():9d} MiB  = {gpu_memory_MiB() * 2 ** 20 / N ** 3:.0f} B/voxel')
print(f'  <sigma_xx> for E_xx = 1 {sigma:.6f}   (solid volume fraction {rho.mean():.3f})', flush=True)
