"""
Material fields of the multigrid study: problems P1-P3 of Paper 1 (copied from
experiments/dual_preconditioner/paper1_problems/run_experiment_elasticity.py, data/ copied alongside).

    P1  square inclusion [0.25, 0.75)^2: C_0 in the matrix, kappa C_0 in the inclusion
    P2  58 circles, two orthotropic phases kappa T C_0 T, T = diag(1, 2^1/4) and diag(2^1/4, 1)
    P3  topology-optimization design: A = rho C_0 + (1 - rho) kappa C_0   (kappa < 1: void)

C_0 = isotropic, K = 1, mu = 1/2 (plane strain), the reference of Green and dual.
"""
from pathlib import Path

import numpy as np

from muFFTTO import material_models

HERE = Path(__file__).resolve().parent
DIM = 2
C_0 = material_models.get_elastic_material_tensor(dim=DIM, K=1.0, mu=0.5, kind='linear')


def rotate_index(C, T):
    """T_ia T_jb T_kc T_ld C_abcd: congruence, keeps C symmetric positive definite"""
    return np.einsum('ia,jb,kc,ld,abcd->ijkl', T, T, T, T, C)


def pixel_material(problem, kappa, N):
    """stiffness per pixel [2, 2, 2, 2, N, N] (both triangles of a pixel share it)"""
    C = np.zeros((2, 2, 2, 2, N, N))
    if problem == 'P1':
        x_c = (np.arange(N) + 0.5) / N
        inside = np.abs(x_c - 0.5) < 0.25
        phase = 1 - np.outer(inside, inside)                    # 1 matrix, 0 inclusion
        C = np.einsum('ijkl,xy->ijklxy', C_0, phase + kappa * (1 - phase))
    elif problem == 'P2':
        phase = np.load(HERE / 'data' / 'p2_circles_256.npz')['phase']
        if N >= 256:
            phase = np.kron(phase, np.ones((N // 256, N // 256)))
        else:                                                   # pixel-centre sampling
            f = 256 // N
            phase = phase[f // 2::f, f // 2::f]
        t = 2 ** 0.25
        for value, Cv in [(0, C_0), (1, kappa * rotate_index(C_0, np.diag([1., t]))),
                          (2, kappa * rotate_index(C_0, np.diag([t, 1.])))]:
            C[..., phase == value] = Cv[..., None]
    elif problem == 'P3':
        rho = np.load(HERE / 'data' / 'p3_density_256.npz')['density']
        if N >= 256:
            rho = np.kron(rho, np.ones((N // 256, N // 256)))
        else:                                                   # block average of the density
            f = 256 // N
            rho = rho.reshape(N, f, N, f).mean(axis=(1, 3))
        C = np.einsum('ijkl,xy->ijklxy', C_0, rho + (1 - rho) * kappa)
    else:
        raise ValueError(problem)
    return C


def quad_material(problem, kappa, N):
    """stiffness per quadrature point [2, 2, 2, 2, q=2, N, N]"""
    C = pixel_material(problem, kappa, N)
    return np.repeat(C[:, :, :, :, None], 2, axis=4)
