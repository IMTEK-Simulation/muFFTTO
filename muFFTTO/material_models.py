"""
Constitutive (material) models and elasticity-tensor utilities.

Contents
--------
Field-based material models (operate on muGrid fields at quadrature points):

* :class:`MaterialModelElasticity` -- abstract interface (stress + tangent).
* :class:`LinearElastic` -- isotropic small-strain linear elasticity
  (Lamé parameters lambda, mu given per quadrature point).
* :class:`NeoHookean` -- compressible neo-Hookean (Simo-Pister) hyperelasticity
  at finite strain, written in terms of the deformation gradient F and the
  first Piola-Kirchhoff stress P.

Plain-numpy helpers (single material point, no fields):

* Voigt conversions: :func:`compute_Voigt_notation_2order`,
  :func:`compute_Voigt_notation_4order`, :func:`compute_Voigt_notation`.
* Parameter conversions: :func:`get_bulk_and_shear_modulus`,
  :func:`get_lame_parameters`, :func:`get_lame_parameters_from_bulk_and_shear`.
* Stiffness tensors: :func:`get_elastic_material_tensor`,
  :func:`get_elastic_tensor_from_lame`,
  :func:`get_orthotropic_stiffness_tensor_plane_strain`,
  :func:`get_elastic_tangent` (Voigt matrices).
* :func:`linear_isotropic_elasticity_stress_from_strain_lame` -- field-based
  stress evaluation without a material object.

Index / layout conventions
--------------------------
Fields are muGrid fields whose ``.s`` view has the layout

* 2nd-order tensor field : ``[i, j, q, x, y, (z)]``
* 4th-order tensor field : ``[i, j, k, l, q, x, y, (z)]``
* scalar field           : ``[1, 1, q, x, y, (z)]`` (so ``field.s[0, 0]`` is ``[q, x, y, (z)]``)

where ``i, j, k, l`` are spatial indices, ``q`` the quadrature point within a
pixel and ``x, y, z`` the (local, MPI-distributed) pixel indices. Tensor
contractions follow the standard convention of ``tensor_operations.py``:
``sigma_ij = C_ijkl eps_kl``. Material tangents are stored in the same
standard order, ``A_ijkl = dP_ij / dF_kl``.

Full 4th-order tensors (``(dim,)*4`` arrays) are used everywhere except in
:func:`get_elastic_tangent` and the explicit Voigt conversion helpers.
"""
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
    Abstract base class for material models (small-strain or finite-strain hyperelastic).

    All material models must implement:
      - get_stress(strain_ijqxyz, stress_ijqxyz)
      - get_algorithmic_tangent(strain_ijqxyz, tangent_ijklqxyz)

    The convention for array indices is:
      i, j     : spatial dimensions (0..dim-1)
      q        : quadrature point index
      x, y, z  : grid cell indices

    "strain" is the generic kinematic input: the small-strain tensor eps for
    small-strain models, or the deformation gradient F for finite-strain models.
    "stress" is the work-conjugate stress (Cauchy sigma, or 1st Piola-Kirchhoff P).
    """

    def __init__(self,
                 discretization, name: str = 'base_material'):
        """
        Parameters
        ----------
        discretization : muFFTTO.domain.Discretization
            Discretization providing field allocation; not stored by the base
            class (subclasses store it themselves).
        name : str, optional
            Human-readable identifier, used in ``__repr__``.
        """
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
        """C_ij = tangent_ijkl · strain_kl (standard contraction, see tensor_operations.py).

        Applies a (previously computed) tangent to a strain(-increment) field,
        e.g. to evaluate the linearised stress increment in Newton/CG iterations.

        Parameters
        ----------
        strain_ijqxyz : muGrid field [i, j, q, x, y, z]
            Strain (or strain increment / F increment) field.
        stress_ijqxyz : muGrid field [i, j, q, x, y, z]
            Output field, overwritten in place with tangent : strain.
        tangent_ijklqxyz : muGrid field [i, j, k, l, q, x, y, z]
            Material tangent, e.g. from ``get_algorithmic_tangent``.
        """
        ddot42(tangent_ijklqxyz, strain_ijqxyz, stress_ijqxyz)
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

    The material is heterogeneous: λ and μ are scalar fields given per
    quadrature point (e.g. from a phase/density field). The tangent is
    independent of the strain. In 2D this is plane strain.
    """

    def __init__(self, discretization, lam_1qxyz, mu_1qxyz, name: str = 'linear_elastic'):
        """
        Parameters
        ----------
        discretization : muFFTTO.domain.Discretization
            Used to allocate temporary quadrature fields.
        lam_1qxyz : muGrid scalar field [1, 1, q, x, y, z]
            First Lamé parameter λ at each quadrature point (stored by reference).
        mu_1qxyz : muGrid scalar field [1, 1, q, x, y, z]
            Shear modulus μ at each quadrature point (stored by reference).
        name : str, optional
        """
        super().__init__(discretization, name)
        self.lam = lam_1qxyz          # quadrature field of first Lamé modulus
        self.mu  = mu_1qxyz           # quadrature field of shear modulus
        self.discretization = discretization
    #TODO[MARTIN/STEFANUS] implement energy for linear elasti material model

    def get_stress(self, strain_ijqxyz, stress_ijqxyz):
        """
        σ_ij = λ δ_ij ε_kk  +  2μ ε_ij
             = λ I_ij tr(ε)  +  2μ ε_ij

        Parameters
        ----------
        strain_ijqxyz : muGrid field [i, j, q, x, y, z]
            Small-strain tensor ε (assumed symmetric; not symmetrised here).
        stress_ijqxyz : muGrid field [i, j, q, x, y, z]
            Output Cauchy stress σ, overwritten in place.

        Notes
        -----
        Allocates/reuses a temporary scalar quadrature field named 'eps_trace'.
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

        Parameters
        ----------
        strain_ijqxyz : muGrid field [i, j, q, x, y, z]
            Only used to infer the spatial dimension (the tangent of linear
            elasticity does not depend on the strain).
        tangent_ijklqxyz : muGrid field [i, j, k, l, q, x, y, z]
            Output stiffness tensor field, overwritten in place.

        Notes
        -----
        C has major and both minor symmetries (δ_ik δ_jl + δ_il δ_jk is
        symmetric in k<->l and in i<->j).
        """
        dim = strain_ijqxyz.s.shape[0]
        I   = np.eye(dim)

        IxI  = np.einsum('ij,kl->ijkl', I, I)   # δ_ij δ_kl
        IsI1 = np.einsum('ik,jl->ijkl', I, I)   # δ_ik δ_jl
        IsI2 = np.einsum('il,jk->ijkl', I, I)   # δ_il δ_jk

        lam = self.lam.s[0, 0]   # shape [q, x, y, z]
        mu  = self.mu.s[0, 0]    # shape [q, x, y, z]

        # Append one singleton axis per [q, x, y, z] axis to the (dim,)*4 constant tensors,
        # so they broadcast against the spatially varying λ, μ fields.
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

    Here J = det F, I₁ = tr(FᵀF) = F:F. P = ∂W/∂F follows from
    ∂J/∂F = J F^{-T} and ∂I₁/∂F = 2F. For F -> I the model linearises to
    LinearElastic with the same λ, μ. Requires J > 0 (ln J is evaluated).
    """

    def __init__(self, discretization, lam_1qxyz, mu_1qxyz, name: str = 'neo_hookean'):
        """
        Parameters
        ----------
        discretization : muFFTTO.domain.Discretization
            Used to allocate temporary quadrature fields.
        lam_1qxyz : muGrid scalar field [1, 1, q, x, y, z]
            Lamé parameter λ per quadrature point.
        mu_1qxyz : muGrid scalar field [1, 1, q, x, y, z]
            Shear modulus μ per quadrature point.
        name : str, optional
        """
        super().__init__(discretization, name)
        self.lam = lam_1qxyz
        self.mu  = mu_1qxyz
        self.discretization = discretization

    def get_energy_density(self, strain_ijqxyz, energy_1qxyz):
        """
        Strain-energy density W(F) = (λ/2) ln(J)²  +  (μ/2)(F:F - dim)  -  μ ln(J).

        Parameters
        ----------
        strain_ijqxyz : muGrid field [i, j, q, x, y, z]
            Deformation gradient F.
        energy_1qxyz : muGrid scalar field [1, 1, q, x, y, z]
            Output energy density per quadrature point, overwritten in place
            (not yet integrated/weighted by quadrature weights).
        """
        F = strain_ijqxyz
        lam = self.lam.s[0, 0]
        mu = self.mu.s[0, 0]
        dim = F.s.shape[0]

        J_1qxyz = self.discretization.get_quad_field_scalar(name='J')
        lnJ_1qxyz = self.discretization.get_quad_field_scalar(name='lnJ')
        det2(F, J_1qxyz)
        log_field(J_1qxyz, lnJ_1qxyz)

        lnJ = lnJ_1qxyz.s[0, 0]
        # I₁ = F_ij F_ij = tr(FᵀF), evaluated pointwise -> [q, x, y, z]
        FF = np.einsum('ij...,ij...->...', F.s, F.s)

        energy_1qxyz.s[0, 0] = 0.5 * lam * lnJ ** 2 + 0.5 * mu * (FF - dim) - mu * lnJ

    def get_stress(self, strain_ijqxyz, stress_ijqxyz):
        """
        P = λ ln(J) F^{-T}  +  μ (F - F^{-T})

        Parameters
        ----------
        strain_ijqxyz : muGrid field [i, j, q, x, y, z]
            Deformation gradient F (det F must be positive).
        stress_ijqxyz : muGrid field [i, j, q, x, y, z]
            Output 1st Piola-Kirchhoff stress P, overwritten in place.

        Notes
        -----
        Uses temporary fields named 'Finv', 'FinvT', 'J', 'lnJ'.
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
        Algorithmic tangent in the standard order A_ijkl = ∂P_ij/∂F_kl, so that
        dP_ij = A_ijkl dF_kl (``tensor_operations.ddot42``):

          A_ijkl = λ FinvT_ij FinvT_kl
                 + (μ - λ ln J) FinvT_il FinvT_kj
                 + μ δ_ik δ_jl

        Three contributions:
          term1 : λ          FinvT_ij FinvT_kl   (volumetric)
          term2 : (μ-λ ln J)  FinvT_il FinvT_kj   (distortional coupling)
          term3 :     μ       δ_ik δ_jl           (distortional identity)

        Derivation:
          ∂(ln J)/∂F_kl   = F^{-T}_kl
          ∂F^{-T}_ij/∂F_kl = -F^{-T}_il F^{-T}_kj
          ∂F_ij/∂F_kl      = δ_ik δ_jl
        The tangent has major symmetry A_ijkl = A_klij (hyperelastic material).

        Parameters
        ----------
        strain_ijqxyz : muGrid field [i, j, q, x, y, z]
            Deformation gradient F.
        tangent_ijklqxyz : muGrid field [i, j, k, l, q, x, y, z]
            Output tangent A, overwritten in place.

        Notes
        -----
        Uses temporary fields 'Finv', 'FinvT', 'J', 'lnJ', 'term1', 'term2', 'term3'.
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
        #coef1 = lam  # λ
        coef2 = mu - lam * lnJ  # μ - λ ln(J)  — term2

        n_extra = lnJ.ndim
        index_extender = (...,) + (np.newaxis,) * n_extra

        # term1: λ FinvT_ij FinvT_kl
        term1_ijklqxyz.s[...] = lam * np.einsum('ij...,kl...->ijkl...', FinvT_ijqxyz.s, FinvT_ijqxyz.s)

        # term2: (μ - λ lnJ) FinvT_il FinvT_kj
        term2_ijklqxyz.s[...] = coef2 * np.einsum('il...,kj...->ijkl...', FinvT_ijqxyz.s, FinvT_ijqxyz.s)


        # term3:  μ δ_ik δ_jl
        I = np.eye(dim)
        IsI = np.einsum('ik,jl->ijkl', I, I)
        term3_ijklqxyz.s[...] = IsI[index_extender] * mu

        tangent_ijklqxyz.s[...] = (term1_ijklqxyz.s
                                   + term2_ijklqxyz.s
                                   + term3_ijklqxyz.s)

# ============================================================================
# Concrete implementation 3: Third Medium (TMC)
# ============================================================================
# NOTE: no Third-Medium model is implemented yet; everything below is a set of
# stand-alone (single material point) tensor / parameter utility functions.


def compute_Voigt_notation_4order(C_ijkl):
    """
    Convert a 4th-order tensor (e.g. stiffness) to a Voigt matrix.

    Index pairs are mapped as
      2D: 0 -> (0,0), 1 -> (1,1), 2 -> (0,1)
      3D: 0 -> (0,0), 1 -> (1,1), 2 -> (2,2), 3 -> (1,2), 4 -> (0,2), 5 -> (0,1)
    and ``C_voigt[K, L] = C_ijkl[pair(K) + pair(L)]``.

    Parameters
    ----------
    C_ijkl : ndarray, shape (dim, dim, dim, dim), dim in {2, 3}

    Returns
    -------
    C_voigt_kl : ndarray, shape (3, 3) for 2D or (6, 6) for 3D

    Notes
    -----
    Pure index re-mapping: no factors of 2 or sqrt(2) are applied to shear
    entries (i.e. not Mandel notation). For a minor-symmetric stiffness this
    is the standard Voigt stiffness matrix acting on engineering shear strains
    (gamma = 2 eps_ij).

    Raises
    ------
    ValueError
        If ``dim`` is not 2 or 3.
    """
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
    else:
        raise ValueError(f'Voigt notation is implemented for dim 2 and 3, got dim={len(C_ijkl)}')
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

    Raises
    ------
    ValueError
        If the tensor is not (2, 2) or (3, 3).

    Notes
    -----
    Shear components are copied as-is (tensor shear ε_xy, NOT engineering
    shear γ_xy = 2 ε_xy). When multiplying a strain vector from this function
    with a Voigt stiffness matrix, the shear entries must be doubled by the caller.
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

    Raises
    ------
    ValueError
        If ``tensor.ndim`` is neither 2 nor 4.
    """
    if tensor.ndim == 2:
        return compute_Voigt_notation_2order(tensor)
    elif tensor.ndim == 4:
        return compute_Voigt_notation_4order(tensor)
    else:
        raise ValueError(f"Expected 2nd or 4th order tensor (ndim 2 or 4), got ndim {tensor.ndim}")


def get_bulk_and_shear_modulus(E, poisson):
    """
    Convert Young's modulus and Poisson's ratio to (3D) bulk and shear moduli.

    K = E / (3 (1 - 2 nu)),   G = E / (2 (1 + nu))

    Parameters
    ----------
    E : float
        Young's modulus.
    poisson : float
        Poisson's ratio nu.

    Returns
    -------
    K : float
        Bulk modulus (3D definition).
    G : float
        Shear modulus (= Lamé mu).

    Raises
    ------
    ValueError
        If nu is (numerically) 0.5, where K diverges.
    """
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

    Raises
    ------
    ValueError
        If poisson is (numerically) 0.5.
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

    Raises
    ------
    ValueError
        If dim is not 2 or 3.

    Notes
    -----
    For dim = 2, K is interpreted as the 2D (in-plane) bulk modulus
    K_2D = lambda + mu, so lambda = K - G. This is NOT the same as using the 3D
    bulk modulus in plane strain (where lambda = K_3D - 2/3 G); make sure the
    supplied K matches the intended definition.
    """
    if dim not in (2, 3):
        raise ValueError(f"dim must be 2 or 3, got {dim}")

    mu = G
    lam = K - (2.0 / dim) * G
    return lam, mu


def get_elastic_material_tensor(dim, K=1, mu=0.5, kind='linear'):
    """
    Isotropic elastic stiffness tensor in bulk/shear (volumetric-deviatoric) form.

    C_abcd = K δ_ab δ_cd + μ (δ_ac δ_bd + δ_ad δ_bc - 2/3 δ_ab δ_cd)

    i.e. C = 3K I_vol + 2μ I_dev with the 3D projectors; equivalently
    λ = K - 2/3 μ. The factor 2/3 is the 3D one and is used for dim = 2 as
    well, so in 2D K should be the 3D bulk modulus (plane strain).

    Parameters
    ----------
    dim : int
        Spatial dimension (2 or 3).
    K : float, optional
        Bulk modulus (default 1).
    mu : float, optional
        Shear modulus (default 0.5).
    kind : str, optional
        Only 'linear' is implemented (default). Note the test is
        ``kind in 'linear'`` (substring test): substrings such as 'lin' or ''
        also match, any other value returns a zero tensor.

    Returns
    -------
    mat : ndarray, shape (dim, dim, dim, dim)
    """
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
    strain_ijqxyz : mugrid field, shape (dim, dim, nb_quad_points, *nb_pixels)
        Input strain field (assumed symmetric; not symmetrised).
    lam_1qxyz : mugrid scalar field (leading component axes of size 1, then q, *nb_pixels)
        First Lame parameter per quad point.
    mu_1qxyz : mugrid scalar field (leading component axes of size 1, then q, *nb_pixels)
        Second Lame parameter (shear modulus) per quad point.
    output_stress_ijqxyz : mugrid field, shape (dim, dim, nb_quad_points, *nb_pixels)
       Output stress field. Written in place.

    Notes
    -----
    Functional equivalent of ``LinearElastic.get_stress`` that does not
    allocate temporary muGrid fields. Relies on numpy broadcasting of the
    size-1 component axes of ``lam``/``mu`` against the (dim, dim) tensor axes.
    """
    strain = strain_ijqxyz.s[...]  # (dim, dim, q, *xyz)
    lam = lam_1qxyz.s[...]  # (1, 1, q, *xyz)
    mu = mu_1qxyz.s[...]  # (1, 1, q, *xyz)

    # Disabled alternative kept for reference:
    # symmetrize: handles full-gradient input the same way C_ijkl minor symmetry does
    # strain = (strain + np.swapaxes(strain, 0, 1)) / 2

    dim = strain.shape[0]

    # trace over the first two (tensor) axes only, keep q,*xyz as-is
    trace_eps_qxyz = np.einsum('ii...->...', strain)  # shape (q, *xyz)

    # identity with singleton axes appended so it broadcasts over [q, x, y, z]
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
        np.ndarray: 4th-order stiffness tensor of shape (2, 2, 2, 2)
        (not a 3x3 Voigt matrix; use compute_Voigt_notation_4order for that).

    Notes:
        Entries filled: C_1111 = C11, C_2222 = C22, C_1122 = C_2211 = C12 and
        all four shear permutations C_1212 = C_2121 = C_1221 = C_2112 = G12,
        so that sigma_12 = 2 G12 eps_12. The expressions
        C11 = E1/(1 - nu12 nu21), C12 = nu12 E2/(1 - nu12 nu21) are the
        reduced (plane-STRESS) orthotropic stiffnesses, despite the function name.
    """
    # Compute nu21 from the symmetry condition nu21 / E2 = nu12 / E1
    nu21 = (nu12 * E2) / E1

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
        Elastic tangent matrix in Voigt notation:
        (6, 6) for "3D" with ordering (xx, yy, zz, yz, xz, xy), or (3, 3) for
        the 2D modes with ordering (xx, yy, xy).

    Raises
    ------
    ValueError
        If mode is not one of the supported strings (case-insensitive).

    Notes
    -----
    The shear diagonal entries are mu (not 2 mu), i.e. the matrix acts on
    engineering shear strains gamma_ij = 2 eps_ij. Plane stress uses
    C11 = E / (1 - nu^2), C12 = nu C11, C66 = mu.
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
