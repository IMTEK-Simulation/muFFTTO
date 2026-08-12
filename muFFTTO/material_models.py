import muGrid
import numpy as np
from abc import ABC, abstractmethod

'''
Constitute models and related utilities
'''


# ============================================================================
# Abstract base class — defines the interface every material model must obey
# ============================================================================
class MaterialModelElasticity(ABC):
    """
    Abstract base class for material models.                                                 finite-strain hyperelastic

    All material models must implement:
      - get_stress(strain_ijqxyz, stress_ijqxyz)
      - get_algorithmic_tangent(strain_ijqxyz, tangent_ijklqxyz)

    The convention for array indices is:
      i, j     : spatial dimensions (0..dim-1)
      q        : quadrature point index
      x, y, z  : grid cell indices
    """

    def __init__(self,
                 discretization, name: str = 'base_material'):
        self.name = name

    @abstractmethod
    def get_stress(self,
                   strain_ijqxyz,
                   stress_ijqxyz) -> None:
        """
        Compute stress from strain IN-PLACE into stress_ijqxyz.

        Parameters
        ----------
        strain_ijqxyz  : muGrid array [i, j, q, x, y, z]  — input strain field
        stress_ijqxyz  : muGrid array [i, j, q, x, y, z]  — output stress field (written in-place)
        """
        ...

    @abstractmethod
    def get_algorithmic_tangent(self,
                                strain_ijqxyz,
                                tangent_ijklqxyz) -> None:
        """
        Compute algorithmic tangent from strain IN-PLACE into tangent_ijklqxyz.

        Parameters
        ----------
        strain_ijqxyz    : muGrid array [i, j, q, x, y, z]       — input strain field
        tangent_ijklqxyz : muGrid array [i, j, k, l, q, x, y, z] — output tangent (written in-place)
        """
        ...

    def __repr__(self):
        return f"{self.__class__.__name__}(name='{self.name}')"


# ============================================================================
# Concrete implementation 1: Linear elasticity
# ============================================================================

class LinearElastic(MaterialModelElasticity):
    """
    Isotropic linear elasticity for small strain elasticity.
      σ_ij = λ δ_ij ε_kk  +  2μ ε_ij
      C_ijkl = λ δ_ij δ_kl  +  μ (δ_ik δ_jl + δ_il δ_jk)
    """

    def __init__(self, discretization, lam_1qxyz, mu_1qxyz, name: str = 'linear_elastic'):
        super().__init__(name)

        self.lam = lam_1qxyz  # quadrature field of first lamme moduli
        self.mu = mu_1qxyz  # quadrature field of shear modullus ratios
        self.discretization = discretization

    def get_stress(self, strain_ijqxyz, stress_ijqxyz):
        dim = strain_ijqxyz.s.shape[0]
        # ε_kk (trace) summed over first two indices
        eps_trace_11qxyz = self.discretization.get_quad_field_scalar(name='eps_trace')
        eps_trace_11qxyz.s[0, 0] = np.einsum('iiq...->q...', strain_ijqxyz.s)  # shape [q, x, y, z]
        # σ_ij = λ δ_ij ε_kk  +  2μ ε_ij
        stress_ijqxyz.s[...] = 2.0 * self.mu.s * strain_ijqxyz.s
        for i in range(dim):
            stress_ijqxyz.s[i, i] += self.lam.s[0, 0] * eps_trace_11qxyz.s[0, 0]

    def get_algorithmic_tangent(self, strain_ijqxyz, tangent_ijklqxyz):
        dim = strain_ijqxyz.s.shape[0]
        I = np.eye(dim)

        # δ_ij δ_kl,  δ_ik δ_jl,  δ_il δ_jk  — shape [i,j,k,l]
        IxI = np.einsum('ij,kl->ijkl', I, I)
        IsI1 = np.einsum('ik,jl->ijkl', I, I)
        IsI2 = np.einsum('il,jk->ijkl', I, I)

        # lam and mu have shape [1,1, q, x, y, z] — add [i,j,k,l] axes in front
        lam = self.lam.s[0, 0]  # shape [q, x, y, z]
        mu = self.mu.s[0, 0]  # shape [q, x, y, z]

        # C_ijkl = λ δ_ij δ_kl + μ (δ_ik δ_jl + δ_il δ_jk)
        # IxI has shape [i,j,k,l], lam has shape [q,x,y,z]
        # result must be [i,j,k,l,q,x,y,z]
        # number of trailing axes in lam: q + n_spatial_dims
        n_extra = lam.ndim
        index_extender = (...,) + (np.newaxis,) * n_extra

        tangent_ijklqxyz.s[...] = (
                IxI[index_extender] * lam
                + IsI1[index_extender] * mu
                + IsI2[index_extender] * mu
        )


    def apply_algorithmic_tangent(self, strain_ijqxyz, stress_ijqxyz, tangent_ijklqxyz):
        stress_ijqxyz.s[...] = np.einsum('ijkl...,kl...->ij...',
                                         tangent_ijklqxyz.s,
                                         strain_ijqxyz.s)


# ============================================================================
# Concrete implementation 2: Neo-Hookean (Simo-Pister)
# ============================================================================

class NeoHookeanTMC(MaterialModelElasticity):
    """
    Compressible neo-Hookean (Simo-Pister form).
      W = (λ/2) ln(J)²  +  (μ/2)(I₁ - dim)  -  μ ln(J)
      P_iJ = λ ln(J) F_iJ^{-T}  +  μ (F_iJ - F_iJ^{-T})

    Here strain_ijqxyz stores the DEFORMATION GRADIENT F (not ε),
    and stress_ijqxyz stores the 1st Piola-Kirchhoff stress P.
    """

    def __init__(self, E: float, nu: float, name: str = 'neo_hookean'):
        super().__init__(name)
        self.E = E
        self.nu = nu
        self.lam = E * nu / ((1 + nu) * (1 - 2 * nu))
        self.mu = E / (2 * (1 + nu))

    def get_stress(self, strain_ijqxyz, stress_ijqxyz):
        # strain_ijqxyz = F (deformation gradient field)
        # Iterate over all quadrature points and grid cells
        shape = strain_ijqxyz.shape  # [i, j, q, ...]
        dim = shape[0]
        extra = strain_ijqxyz.shape[2:]  # (q, x, y, z, ...)

        for idx in np.ndindex(*extra):
            F = strain_ijqxyz[(slice(None), slice(None)) + idx]  # dim×dim
            J = np.linalg.det(F)
            Finv = np.linalg.inv(F)
            lnJ = np.log(J)
            P = self.lam * lnJ * Finv.T + self.mu * (F - Finv.T)
            stress_ijqxyz[(slice(None), slice(None)) + idx] = P

    def get_algorithmic_tangent(self, strain_ijqxyz, tangent_ijklqxyz):
        # Material tangent A_iJkL = dP_iJ/dF_kL
        shape = strain_ijqxyz.shape
        extra = strain_ijqxyz.shape[2:]
        dim = shape[0]

        for idx in np.ndindex(*extra):
            F = strain_ijqxyz[(slice(None), slice(None)) + idx]
            J = np.linalg.det(F)
            Finv = np.linalg.inv(F)
            lnJ = np.log(J)
            lam, mu = self.lam, self.mu

            A = np.zeros((dim, dim, dim, dim))
            for i in range(dim):
                for J_ in range(dim):
                    for k in range(dim):
                        for L in range(dim):
                            # dP_iJ/dF_kL = lam * Finv_Ji * Finv_Lk
                            #              + (lam*lnJ - mu) * (Finv_Li * Finv_Jk + Finv_Ji * Finv_Lk) -- skipped here
                            # Closed form (see Holzapfel 2006):
                            A[i, J_, k, L] = (
                                    lam * Finv[J_, i] * Finv[L, k]
                                    + (mu - lam * lnJ) * (Finv[L, i] * Finv[J_, k])
                                    + mu * (i == k) * (J_ == L)
                            )
            tangent_ijklqxyz[(slice(None),) * 4 + idx] = A


# ============================================================================
# Concrete implementation 3: Third Medium (TMC)
# ============================================================================

class ThirdMediumContact(NeoHookeanTMC):
    """
    Third Medium Contact material (Bluhm et al. 2021; Frederiksen et al. 2026).
    Inherits NeoHookean; scales λ and μ by k_v << 1.
    """

    def __init__(self, E_solid: float, nu: float,
                 kv: float = 1e-6,
                 name: str = 'third_medium'):
        lam_s = E_solid * nu / ((1 + nu) * (1 - 2 * nu))
        mu_s = E_solid / (2 * (1 + nu))
        E_m = kv * E_solid
        super().__init__(E=E_m, nu=nu, name=name)
        self.kv = kv
        self.E_solid = E_solid


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


def get_lame_parameters_from_bulk_and_shear(K, G, dim):
    """
    Convert bulk modulus K and shear modulus G to the Lame parameters
    (lambda, mu) for an isotropic linear elastic material, valid in
    2D (plane strain) or 3D.

    mu     = G
    lambda = K - (2/dim) * G

    Parameters
    ----------
    K : float
        Bulk modulus.
    G : float
        Shear modulus.
    dim : int
        Spatial dimension: 2 (plane strain) or 3.

    Returns
    -------
    lam : float
        First Lame parameter (lambda).
    mu : float
        Second Lame parameter (mu), equivalent to the shear modulus G.
    """
    if dim not in (2, 3):
        raise ValueError(f"dim must be 2 or 3, got {dim}")

    mu = G
    lam = K - (2.0 / dim) * G
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
    strain = strain_ijqxyz.s[...]  # (dim, dim, q, *xyz)
    lam = lam_1qxyz.s[...]  # (1, q, *xyz)
    mu = mu_1qxyz.s[...]  # (1, q, *xyz)

    # symmetrize: handles full-gradient input the same way C_ijkl minor symmetry does
    # strain = (strain + np.swapaxes(strain, 0, 1)) / 2

    dim = strain.shape[0]

    # trace over the first two (tensor) axes only, keep q,*xyz as-is
    trace_eps_qxyz = np.einsum('ii...->...', strain)  # shape (q, *xyz)

    I = np.eye(dim).reshape((dim, dim) + (1,) * (strain.ndim - 2))

    output_stress_ijqxyz.s[...] = lam * trace_eps_qxyz * I + 2.0 * mu * strain


# ------------------------------------------------------------
# Elastic stiffness tensors from Lamé parameters
# ------------------------------------------------------------
def get_elastic_tensor_from_lame(dim, lam, mu):
    """
    Construct a linear elastic stiffness tensor from Lamé parameters.

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
            [lam, lam, lam + 2 * mu, 0, 0, 0],
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
