"""
SUPERSEDED (2026-11-14) by muFFTTO.hourglass_control (element trilinear_hexahedron_1Q_hg, exact per-mode stiffness);
kept only as the record of the first prototype (its c0 was calibrated on an 8^3 grid and does not scale with h).


Flanagan-Belytschko-type hourglass control for the one-point hexahedron (trilinear_hexahedron_1Q) on muFFTTO's
periodic voxel grid -- experiment prototype (2026-11-14) before a possible muFFTTO element.

Regular cube voxel, nodes a = (i, j, k) in {0, 1}^3, xi_a = 2i - 1, eta_a = 2j - 1, zeta_a = 2k - 1. Hourglass base
vectors (Flanagan & Belytschko, IJNME 17 (1981) 679):  h_1 = eta zeta, h_2 = zeta xi, h_3 = xi eta, h_4 = xi eta zeta.
On a regular (affine) voxel they are orthogonal to 1, xi, eta, zeta, so gamma_alpha = h_alpha / 8 and the
stabilisation vanishes for every linear displacement (constant strain): macroscopic loads and homogenised stresses
are unaffected.

    hourglass strains   q_{alpha i}(voxel) = sum_a gamma_{alpha a} u_i(a)          (4 modes x d components)
    energy              E_hg = 1/2 sum_voxels kappa(x) sum_{alpha i} q_{alpha i}^2
    stiffness           K_hg u = H^T (kappa H u),     kappa(x) = eps * c0 * s(x) * (lambda_0 + 2 mu_0)

c0 is calibrated (calibrate_c0) so that eps = 1 gives, for a homogeneous material, the average hourglass energy of the
fully integrated (2x2x2 Gauss) element over the 4 x 3 global hourglass modes. All operations are periodic rolls on
the nodal field [3, 1, N, N, N] (numpy or cupy).
"""
import itertools

import numpy as np

import mg_lib

CORNERS = list(itertools.product((0, 1), repeat=3))                    # (i, j, k)


def _h(alpha, i, j, k):
    xi, eta, zeta = 2 * i - 1, 2 * j - 1, 2 * k - 1
    return (eta * zeta, zeta * xi, xi * eta, xi * eta * zeta)[alpha]


def hourglass_strains(u):
    """[3, 1, N, N, N] nodal -> q [4, 3, N, N, N] per voxel (voxel (x, y, z) has nodes (x + i, y + j, z + k))"""
    xp = mg_lib.xp_of(u)
    u0 = u[:, 0]
    q = xp.zeros((4,) + u0.shape, dtype=u.dtype)
    for (i, j, k) in CORNERS:
        shifted = xp.roll(u0, shift=(-i, -j, -k), axis=(-3, -2, -1))
        for alpha in range(4):
            q[alpha] += (_h(alpha, i, j, k) / 8.0) * shifted
    return q


def hourglass_forces(q_weighted):
    """H^T: q [4, 3, N, N, N] -> nodal [3, 1, N, N, N]"""
    xp = mg_lib.xp_of(q_weighted)
    f = xp.zeros(q_weighted.shape[1:], dtype=q_weighted.dtype)
    for (i, j, k) in CORNERS:
        acc = sum((_h(alpha, i, j, k) / 8.0) * q_weighted[alpha] for alpha in range(4))
        f += xp.roll(acc, shift=(i, j, k), axis=(-3, -2, -1))
    return f[:, None]


def apply_hourglass(u, kappa):
    """K_hg u; kappa: scalar or [N, N, N] per voxel"""
    return hourglass_forces(kappa * hourglass_strains(u))


def global_hourglass_modes(N):
    """the 4 x 3 periodic fields whose restriction to every voxel is +- h_alpha in one component"""
    x = np.arange(N)
    X, Y, Z = np.meshgrid(x, x, x, indexing='ij')
    sign = [(-1.0) ** (Y + Z), (-1.0) ** (Z + X), (-1.0) ** (X + Y), (-1.0) ** (X + Y + Z)]
    modes = []
    for alpha in range(4):
        for comp in range(3):
            u = np.zeros((3, 1, N, N, N))
            u[comp, 0] = sign[alpha]
            modes.append((alpha, comp, u))
    return modes


def calibrate_c0(C_0, N=8):
    """c0 such that eps = 1 reproduces the mean hourglass energy of the 2x2x2 element (homogeneous material)"""
    lam_2mu = C_0[0, 0, 0, 0]
    full = mg_lib.Hierarchy(np.ones((8, N, N, N)), C_0, n_coarsest=N, scalar=True, element_type='trilinear_hexahedron')
    one = mg_lib.Hierarchy(np.ones((1, N, N, N)), C_0, n_coarsest=N, scalar=True, element_type='trilinear_hexahedron_1Q')
    ratios = []
    for alpha, comp, u in global_hourglass_modes(N):
        e_full = float(np.sum(u * full.levels[0].K(u, count=False)))
        e_one = float(np.sum(u * one.levels[0].K(u, count=False)))
        e_hg = float(np.sum(u * apply_hourglass(u, lam_2mu)))
        ratios.append((alpha, comp, e_full, e_one, (e_full - e_one) / e_hg))
    return float(np.mean([r[4] for r in ratios])), ratios
