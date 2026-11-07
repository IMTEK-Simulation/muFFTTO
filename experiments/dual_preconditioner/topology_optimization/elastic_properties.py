"""
Effective elastic properties of a TO design from its homogenised stiffness, as in the TO paper
(Paper__Topology_optimization_with_FFT_accelerated_micromechanical_solvers.pdf, Sec. V):

    directional Poisson ratio   nu_ab = -S_aabb / S_aaaa      (load along X_a, transverse X_b), S = C^-1
    shear modulus               mu    = C_1212                 (no inversion)
    Zener ratio                 A_Z   = 2 C_1212 / (C_1111 - C_1122)    (1: isotropic)

Stiffness in Voigt notation with engineering shear strains: 2D [11, 22, 12], 3D [11, 22, 33, 23, 13, 12]; the shear
entries are C_1212 etc. (the study's load cases are unit strains with eps_12 = eps_21 = 1/2). The normal-normal block
of the Voigt inverse equals that of the tensor compliance, so S_aabb = (C_voigt^-1)[a, b] for the Poisson ratios.
In plane strain an isotropic material has nu_12 = nu / (1 - nu): the study's target nu = -0.5 is nu_12 = -1/3.
"""
import numpy as np


def voigt_from_stresses(stresses):
    """Voigt stiffness from the homogenised stresses of the unit load cases (2D: xx, yy, xy; 3D: xx, yy, zz, yz, xz,
    xy; shear cases with eps_ij = eps_ji = 1/2, i.e. engineering shear strain 1): column k = Voigt(stress of case k)."""
    s = np.asarray(stresses)
    dim = s.shape[-1]
    idx = [(0, 0), (1, 1), (0, 1)] if dim == 2 else [(0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1)]
    s = s.reshape(len(idx), dim, dim)
    return np.array([[s[k][i, j] for k in range(len(idx))] for (i, j) in idx])


def properties(C):
    """dict: nu12, nu21 (and in 3D nu13, nu23), mu (= C_1212), A_Z, from a 2D (3 x 3) or 3D (6 x 6) Voigt stiffness."""
    C = np.asarray(C, dtype=float)
    S = np.linalg.inv(C)
    nu = lambda a, b: -S[a, b] / S[a, a]
    if C.shape[0] == 3:
        mu = C[2, 2]
        out = dict(nu12=nu(0, 1), nu21=nu(1, 0))
    else:
        mu = C[5, 5]
        out = dict(nu12=nu(0, 1), nu21=nu(1, 0), nu13=nu(0, 2), nu23=nu(1, 2))
    out.update(mu=mu, A_Z=2 * mu / (C[0, 0] - C[0, 1]))
    return out


def summary(C):
    """One-line text: Poisson ratio(s), shear modulus, Zener ratio."""
    p = properties(C)
    nus = '  '.join(f'{k} {v:+.3f}' for k, v in p.items() if k.startswith('nu'))
    return f'{nus}  mu {p["mu"]:.3f}  A_Z {p["A_Z"]:.2f}'
