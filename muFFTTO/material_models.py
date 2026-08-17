import muGrid
import numpy as np
from abc import ABC, abstractmethod

from .domain import Discretization
from .tensor_operations import *


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

    def apply_algorithmic_tangent(self, strain_ijqxyz, stress_ijqxyz, tangent_ijklqxyz):
        stress_ijqxyz.s[...] = np.einsum('ijkl...,kl...->ij...',
                                         tangent_ijklqxyz.s,
                                         strain_ijqxyz.s)
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
        super().__init__(discretization, name)
        self.lam = lam_1qxyz          # quadrature field of first Lamé modulus
        self.mu  = mu_1qxyz           # quadrature field of shear modulus
        self.discretization = discretization


    def get_stress(self, strain_ijqxyz, stress_ijqxyz):
        """
        σ_ij = λ δ_ij ε_kk  +  2μ ε_ij
             = λ I_ij tr(ε)  +  2μ ε_ij
        """
        dim = strain_ijqxyz.s.shape[0]

        # tr(ε) = ε_kk — shape [q, x, y, z]
        eps_trace_1qxyz = self.discretization.get_quad_field_scalar(name='eps_trace')
        trace2(strain_ijqxyz, eps_trace_1qxyz)

        # σ = 2μ ε  +  λ tr(ε) I
        stress_ijqxyz.s[...] = 2.0 * self.mu.s * strain_ijqxyz.s
        add_scaled_identity(eps_trace_1qxyz, self.lam, stress_ijqxyz, dim)

    def get_algorithmic_tangent(self, strain_ijqxyz, tangent_ijklqxyz):
        """
        C_ijkl = λ δ_ij δ_kl  +  μ (δ_ik δ_jl  +  δ_il δ_jk)
        """
        dim = strain_ijqxyz.s.shape[0]
        I   = np.eye(dim)

        IxI  = np.einsum('ij,kl->ijkl', I, I)   # δ_ij δ_kl
        IsI1 = np.einsum('ik,jl->ijkl', I, I)   # δ_ik δ_jl
        IsI2 = np.einsum('il,jk->ijkl', I, I)   # δ_il δ_jk

        lam = self.lam.s[0, 0]   # shape [q, x, y, z]
        mu  = self.mu.s[0, 0]    # shape [q, x, y, z]

        n_extra        = lam.ndim
        index_extender = (...,) + (np.newaxis,) * n_extra

        tangent_ijklqxyz.s[...] = (  IxI [index_extender] * lam
                                    + IsI1[index_extender] * mu
                                    + IsI2[index_extender] * mu  )


# ============================================================================
# Concrete implementation 2: Neo-Hookean (Simo-Pister)
# ============================================================================

class NeoHookean(MaterialModelElasticity):
    """
    Compressible neo-Hookean (Simo-Pister form) for finite strain elasticity.
      W = (λ/2) ln(J)²  +  (μ/2)(I₁ - dim)  -  μ ln(J)
      P_iJ = λ ln(J) F^{-T}_iJ  +  μ (F_iJ - F^{-T}_iJ)

    strain_ijqxyz stores the deformation gradient F.
    stress_ijqxyz stores the 1st Piola-Kirchhoff stress P.
    """

    def __init__(self, discretization, lam_1qxyz, mu_1qxyz, name: str = 'neo_hookean'):
        super().__init__(discretization, name)
        self.lam = lam_1qxyz
        self.mu  = mu_1qxyz
        self.discretization = discretization

    def get_energy_density(self, strain_ijqxyz, energy_1qxyz):
        """
        Compute strain energy density  into energy_1qxyz.

        W = (λ/2) (ε_kk)²  +  μ ε_ij ε_ij

        Parameters
        ----------
        strain_ijqxyz : muGrid field [i, j, q, x, y, z] — strain field ε
        energy_1qxyz  : muGrid scalar field [1, 1, q, x, y, z] — output energy density
        """
        lam = self.lam.s[0, 0]  # shape [q, x, y, z]
        mu = self.mu.s[0, 0]  # shape [q, x, y, z]

        # tr(ε) = ε_kk — shape [q, x, y, z]
        eps_trace_1qxyz = self.discretization.get_quad_field_scalar(name='eps_trace')
        trace2(strain_ijqxyz, eps_trace_1qxyz)
        eps_trace = eps_trace_1qxyz.s[0, 0]

        # ε_ij ε_ij — shape [q, x, y, z]
        eps_sq = np.einsum('ij...,ij...->...', strain_ijqxyz.s, strain_ijqxyz.s)

        # W = (λ/2) tr(ε)²  +  μ ε:ε
        energy_1qxyz.s[0, 0] = 0.5 * lam * eps_trace ** 2 + mu * eps_sq

    def get_stress(self, strain_ijqxyz, stress_ijqxyz):
        """
        P = λ ln(J) F^{-T}  +  μ (F - F^{-T})
        """
        F    = strain_ijqxyz
        lam  = self.lam.s[0, 0]   # shape [q, x, y, z]
        mu   = self.mu.s[0, 0]    # shape [q, x, y, z]

        # allocate intermediate fields
        Finv_ijqxyz   = self.discretization.get_strain_sized_field(name='Finv')
        FinvT_ijqxyz  = self.discretization.get_strain_sized_field(name='FinvT')
        J_1qxyz       = self.discretization.get_quad_field_scalar(name='J')
        lnJ_1qxyz     = self.discretization.get_quad_field_scalar(name='lnJ')

        inv2(F,    Finv_ijqxyz)
        trans2(Finv_ijqxyz, FinvT_ijqxyz)
        det2(F,    J_1qxyz)
        log_field(J_1qxyz, lnJ_1qxyz)

        lnJ = lnJ_1qxyz.s[0, 0]   # shape [q, x, y, z]

        # P = λ ln(J) F^{-T}  +  μ (F - F^{-T})
        stress_ijqxyz.s[...] = (  lam * lnJ * FinvT_ijqxyz.s
                                + mu  * (F.s - FinvT_ijqxyz.s)  )

    def get_algorithmic_tangent(self, strain_ijqxyz, tangent_ijklqxyz):
        """
        A_iJkL = λ F^{-T}_iJ F^{-T}_kL
               + (μ - λ ln J) ( F^{-T}_iL F^{-T}_kJ  +  δ_ik δ_JL )

        Two contributions:
          term1 : λ        FinvT_ij FinvT_kl          (volumetric)
          term2 : (μ-λlnJ) FinvT_il FinvT_kj          (distortional, part 1)
          term3 : (μ-λlnJ) δ_ik δ_jl                  (distortional, part 2)

        term2 + term3 share coef2 = μ - λ ln(J) and together are minor-symmetric.
        """
        F = strain_ijqxyz
        lam = self.lam.s[0, 0]
        mu = self.mu.s[0, 0]
        dim = F.s.shape[0]

        Finv_ijqxyz = self.discretization.get_strain_sized_field(name='Finv')
        FinvT_ijqxyz = self.discretization.get_strain_sized_field(name='FinvT')
        J_1qxyz = self.discretization.get_quad_field_scalar(name='J')
        lnJ_1qxyz = self.discretization.get_quad_field_scalar(name='lnJ')
        term1_ijklqxyz = self.discretization.get_material_data_size_field_mugrid(name='term1')
        term2_ijklqxyz = self.discretization.get_material_data_size_field_mugrid(name='term2')
        term3_ijklqxyz = self.discretization.get_material_data_size_field_mugrid(name='term3')

        inv2(F, Finv_ijqxyz)
        trans2(Finv_ijqxyz, FinvT_ijqxyz)
        det2(F, J_1qxyz)
        log_field(J_1qxyz, lnJ_1qxyz)

        lnJ = lnJ_1qxyz.s[0, 0]
        coef1 = lam  # λ
        coef2 = mu - lam * lnJ  # μ - λ ln(J)  — shared by term2 and term3

        n_extra = lnJ.ndim
        index_extender = (...,) + (np.newaxis,) * n_extra

        # term1: λ FinvT_ij FinvT_kl
        dyad22(FinvT_ijqxyz, FinvT_ijqxyz, term1_ijklqxyz)
        term1_ijklqxyz.s[...] *= coef1

        # term2: (μ - λ lnJ) FinvT_il FinvT_kj
        term2_ijklqxyz.s[...] = coef2 * np.einsum('il...,kj...->ijkl...',
                                                  FinvT_ijqxyz.s,
                                                  FinvT_ijqxyz.s)

        # term3: (μ - λ lnJ) δ_ik δ_jl  — same coef2, not μ
        I = np.eye(dim)
        IsI = np.einsum('ik,jl->ijkl', I, I)
        term3_ijklqxyz.s[...] = IsI[index_extender] * coef2

        tangent_ijklqxyz.s[...] = (term1_ijklqxyz.s
                                   + term2_ijklqxyz.s
                                   + term3_ijklqxyz.s)

# ============================================================================
# Concrete implementation 3: Third Medium (TMC)
# ============================================================================


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


def compute_Voigt_notation_2order(tensor_ij):
    """
    Convert a 2nd-order tensor (stress/strain) to Voigt vector notation.

    For 2D (tensor shape (2,2)), Voigt notation maps:
      [0] → σ_xx or ε_xx
      [1] → σ_yy or ε_yy
      [2] → σ_xy or ε_xy
    Returns (3,) vector.

    For 3D (tensor shape (3,3)), Voigt notation maps:
      [0] → σ_xx or ε_xx
      [1] → σ_yy or ε_yy
      [2] → σ_zz or ε_zz
      [3] → σ_yz or ε_yz
      [4] → σ_xz or ε_xz
      [5] → σ_xy or ε_xy
    Returns (6,) vector.

    Parameters
    ----------
    tensor_ij : ndarray
        2nd-order tensor in full index notation.
        Shape (2, 2) for 2D or (3, 3) for 3D.

    Returns
    -------
    voigt_vector : ndarray
        Tensor in Voigt vector notation. Shape (3,) for 2D or (6,) for 3D.
    """
    if tensor_ij.shape == (2, 2):
        return np.array([
            tensor_ij[0, 0],  # σ_xx or ε_xx
            tensor_ij[1, 1],  # σ_yy or ε_yy
            tensor_ij[0, 1]   # σ_xy or ε_xy
        ])
    elif tensor_ij.shape == (3, 3):
        return np.array([
            tensor_ij[0, 0],  # σ_xx or ε_xx
            tensor_ij[1, 1],  # σ_yy or ε_yy
            tensor_ij[2, 2],  # σ_zz or ε_zz
            tensor_ij[1, 2],  # σ_yz or ε_yz
            tensor_ij[0, 2],  # σ_xz or ε_xz
            tensor_ij[0, 1]   # σ_xy or ε_xy
        ])
    else:
        raise ValueError(f"Expected (2,2) or (3,3) tensor, got shape {tensor_ij.shape}")


def compute_Voigt_notation(tensor):
    """
    Convert a tensor to Voigt notation. Automatically handles 2nd-order and 4th-order tensors.

    For 2nd-order tensors (stress/strain):
      - 2D (2,2) → (3,) Voigt vector
      - 3D (3,3) → (6,) Voigt vector

    For 4th-order tensors (stiffness/compliance):
      - 2D (2,2,2,2) → (3,3) Voigt matrix
      - 3D (3,3,3,3) → (6,6) Voigt matrix

    Parameters
    ----------
    tensor : ndarray
        Tensor in full index notation. Can be 2nd-order (shape (2,2), (3,3))
        or 4th-order (shape (2,2,2,2), (3,3,3,3)).

    Returns
    -------
    voigt_tensor : ndarray
        Tensor in Voigt notation.
    """
    if tensor.ndim == 2:
        return compute_Voigt_notation_2order(tensor)
    elif tensor.ndim == 4:
        return compute_Voigt_notation_4order(tensor)
    else:
        raise ValueError(f"Expected 2nd or 4th order tensor (ndim 2 or 4), got ndim {tensor.ndim}")


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
