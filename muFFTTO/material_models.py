import numpy as np

'''
Constitute models and related utilities
'''


def compute_Voigt_notation_2order(sigma_ij):
    # function return Voigt notation of second order tensor
    if len(sigma_ij) == 2:
        sigma_voigt_k = np.zeros([3])
        ij_ind = [(0, 0), (1, 1), (0, 1)]

        for k in np.arange(len(sigma_voigt_k)):
            sigma_voigt_k[k] = sigma_ij[ij_ind[k]]

    elif len(sigma_ij) == 3:
        sigma_voigt_k = np.zeros([6])
        ij_ind = [(0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1)]
        for k in np.arange(len(sigma_voigt_k)):
            sigma_voigt_k[k] = sigma_ij[ij_ind[k]]

    return sigma_voigt_k


def compute_Voigt_notation_4order(C_ijkl):
    # function return Voigt notation of elastic tensor
    if len(C_ijkl) == 2:
        C_voigt_kl = np.zeros([3, 3])
        ij_ind = [(0, 0), (1, 1), (0, 1)]
        for k in np.arange(len(C_voigt_kl[0])):
            for l in np.arange(len(C_voigt_kl[1])):
                C_voigt_kl[k, l] = C_ijkl[ij_ind[k] + ij_ind[l]]

    elif len(C_ijkl) == 3:
        C_voigt_kl = np.zeros([6, 6])
        ij_ind = [(0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1)]
        for i in np.arange(len(C_voigt_kl[0])):
            for j in np.arange(len(C_voigt_kl[1])):
                C_voigt_kl[i, j] = C_ijkl[ij_ind[i] + ij_ind[j]]
    return C_voigt_kl


def get_bulk_and_shear_modulus(E, poisson):
    if abs(1 - 2 * poisson) < 1e-10:
        raise ValueError("Poisson's ratio too close to 0.5 (incompressible limit); K is undefined/infinite.")
    K = E / (3 * (1 - 2 * poisson))
    G = E / (2 * (1 + poisson))
    return K, G

def get_lame_parameters(E, poisson):
    """
    Convert Young's modulus and Poisson's ratio to the Lame parameters
    (lambda, mu) for an isotropic linear elastic material.

    lambda = E * poisson / ((1 + poisson) * (1 - 2 * poisson))
    mu     = E / (2 * (1 + poisson))          [mu = shear modulus, same as G]

    Parameters
    ----------
    E : float
        Young's modulus.
    poisson : float
        Poisson's ratio.

    Returns
    -------
    lam : float
        First Lame parameter (lambda).
    mu : float
        Second Lame parameter (mu), equivalent to the shear modulus G.
    """
    if abs(1 - 2 * poisson) < 1e-10:
        raise ValueError("Poisson's ratio too close to 0.5 (incompressible limit); "
                          "lambda is undefined/infinite.")

    lam = E * poisson / ((1 + poisson) * (1 - 2 * poisson))
    mu = E / (2 * (1 + poisson))
    return lam, mu



def get_elastic_material_tensor(dim, K=1, mu=0.5, kind='linear'):
    shape = np.array(4 * [dim, ])
    mat = np.zeros(shape)
    kron = lambda a, b: 1 if a == b else 0

    if kind in 'linear':
        for alpha, beta, gamma, delta in np.ndindex(*shape):
            mat[alpha, beta, gamma, delta] = (K * (kron(alpha, beta) * kron(gamma, delta))
                                              + mu * (kron(alpha, gamma) * kron(beta, delta) +
                                                      kron(alpha, delta) * kron(beta, gamma) -
                                                      2 / 3 * kron(alpha, beta) * kron(gamma, delta)))
            # https://en.wikipedia.org/wiki/Linear_elasticity
    return mat

def linear_isotropic_elasticity_stress_from_strain_lame(strain_ijqxyz, lam_1qxyz, mu_1qxyz, output_stress_ijqxyz):
    """
    Linear elastic stress from strain, isotropic material, Lame parameters only.
    sigma = lambda * tr(eps) * I + 2 * mu * eps

    Parameters
    ----------
    strain_ijqxyz : mugrid field, shape (dim, dim, nb_quad_points, *nb_nodes)
        Input strain field.
    lam_1qxyz : mugrid field, shape (1, nb_quad_points, *nb_nodes)
        First Lame parameter per quad point.
    mu_1qxyz : mugrid field, shape (1, nb_quad_points, *nb_nodes)
        Second Lame parameter (shear modulus) per quad point.
    output_stress_ijqxyz : mugrid field, shape (dim, dim, nb_quad_points, *nb_nodes)
       Output stress field. Written in place.
    """
    strain = strain_ijqxyz.s[...]      # (dim, dim, q, *xyz)
    lam = lam_1qxyz.s[...]             # (1, q, *xyz)
    mu = mu_1qxyz.s[...]               # (1, q, *xyz)

    # symmetrize: handles full-gradient input the same way C_ijkl minor symmetry does
    # strain = (strain + np.swapaxes(strain, 0, 1)) / 2

    dim = strain.shape[0]

    # trace over the first two (tensor) axes only, keep q,*xyz as-is
    trace_eps_qxyz = np.einsum('ii...->...', strain)          # shape (q, *xyz)

    I = np.eye(dim).reshape((dim, dim) + (1,) * (strain.ndim - 2))

    output_stress_ijqxyz.s[...] = lam * trace_eps_qxyz * I + 2.0 * mu * strain


# ------------------------------------------------------------
# Elastic stiffness tensors from Lamé parameters
# ------------------------------------------------------------
def get_elastic_tensor_from_lame(dim, lam, mu):
    """
    Construct linear elastic stiffness tensor from Lamé parameters.

    Parameters
    ----------
    dim : int
        Spatial dimension (2 or 3).
    lam : float
        First Lamé parameter (λ).
    mu : float
        Second Lamé parameter (μ), shear modulus.

    Returns
    -------
    C : ndarray (dim, dim, dim, dim)
        Elastic stiffness tensor.

    Notes
    -----
    The stiffness tensor is computed as:
    C_ijkl = λ δ_ij δ_kl + μ (δ_ik δ_jl + δ_il δ_jk)

    For 2D, this assumes plane strain conditions.
    """
    C = np.zeros((dim, dim, dim, dim))
    for i in range(dim):
        for j in range(dim):
            for k in range(dim):
                for l in range(dim):
                    delta_ij = 1 if i == j else 0
                    delta_kl = 1 if k == l else 0
                    delta_ik = 1 if i == k else 0
                    delta_jl = 1 if j == l else 0
                    delta_il = 1 if i == l else 0
                    delta_jk = 1 if j == k else 0

                    C[i, j, k, l] = (lam * delta_ij * delta_kl +
                                     mu * (delta_ik * delta_jl + delta_il * delta_jk))
    return C


def get_orthotropic_stiffness_tensor_plane_strain(E1, E2, G12, nu12):
    """
    Assemble the stiffness matrix for an orthotropic material in 2D plane strain.

    Parameters:
        E1 (float): Young's modulus in the x-direction.
        E2 (float): Young's modulus in the y-direction.
        G12 (float): Shear modulus in the xy-plane.
        nu12 (float): Poisson's ratio (strain in y due to stress in x).

    Returns:
        np.ndarray: 3x3 stiffness matrix.
    """
    # Compute nu21 from symmetry condition: nu21 / E2 = nu12 / E1
    nu21 = (nu12 * E1) / E2

    # Stiffness matrix components
    factor = 1 / (1 - nu12 * nu21)
    C11 = E1 * factor
    C22 = E2 * factor
    C12 = nu12 * E2 * factor
    C66 = G12

    # Assemble stiffness matrix
    # Initialize the 4th-order stiffness tensor
    C = np.zeros((2, 2, 2, 2))

    # Fill tensor components in plane strain
    C[0, 0, 0, 0] = C11  # xx-xx
    C[1, 1, 1, 1] = C22  # yy-yy
    C[0, 0, 1, 1] = C[1, 1, 0, 0] = C12  # xx-yy and yy-xx
    C[0, 1, 0, 1] = C[1, 0, 1, 0] = C[0, 1, 1, 0] = C[1, 0, 0, 1] = C66  # xy-xy

    return C


def get_elastic_tangent(E, nu, mode="3D"):
    """
    Returns the elastic constitutive (tangent) matrix C for:
        mode = "plane_stress"
        mode = "plane_strain"
        mode = "3D"

    Parameters
    ----------
    E : float
        Young's modulus
    nu : float
        Poisson's ratio
    mode : str
        "plane_stress", "plane_strain", or "3D"

    Returns
    -------
    C : ndarray
        Elastic tangent matrix
    """

    # Lame parameters
    lam = (E * nu) / ((1 + nu) * (1 - 2 * nu))
    mu = E / (2 * (1 + nu))

    if mode.lower() == "3d":
        # 6x6 matrix in Voigt notation
        C = np.array([
            [lam + 2 * mu, lam, lam, 0, 0, 0],
            [lam, lam + 2 * mu, lam, 0, 0, 0],
            [lam, lam,          lam + 2 * mu, 0, 0, 0],
            [0, 0, 0, mu, 0, 0],
            [0, 0, 0, 0, mu, 0],
            [0, 0, 0, 0, 0, mu]
        ])
        return C

    elif mode.lower() == "plane_strain":
        # 3x3 matrix (σxx, σyy, σxy)
        C = np.array([
            [lam + 2 * mu, lam, 0],
            [lam, lam + 2 * mu, 0],
            [0, 0, mu]
        ])
        return C

    elif mode.lower() == "plane_stress":
        # Plane stress uses reduced constitutive matrix
        C11 = E / (1 - nu ** 2)
        C12 = nu * C11
        C66 = E / (2 * (1 + nu))

        C = np.array([
            [C11, C12, 0],
            [C12, C11, 0],
            [0, 0, C66]
        ])
        return C

    else:
        raise ValueError("mode must be 'plane_stress', 'plane_strain', or '3D'")
