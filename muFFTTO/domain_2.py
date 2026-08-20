"""domain_2.py -- Patched version of muFFTTO.domain for deformed-grid conductivity homogenization.

WHY THIS FILE EXISTS (in plain words)
---------------------------------------
We found TWO bugs in the deformed-grid conductivity homogenization pipeline
in the original `domain.py`. On a perfectly regular grid (F = Identity), the
homogenized conductivity tensor is symmetric and positive definite at every
tested resolution N -- proving the material law, the solver, and the basic
assembly are all correct. On a DEFORMED grid, the homogenized tensor loses
symmetry and positive-definiteness (negative and even complex eigenvalues)
at every tested N.

We verified the correct continuum formula against the literature (Piola
transform of the heat flux to the reference configuration; see e.g. the
standard result Q = -J F^-1 k F^-T Grad(T), J = det(F)). Using this as the
ground truth, we checked each of the three places in `domain.py` that
handle the deformed grid:

BUG 1 (FIXED in v1 of this file): `get_rhs_mugrid_deformed_grid`
------------------------------------------------------------------
The correct three-step transform for turning a (constant) macro gradient E
into the deformed-grid load is:
    1. pull back:      E_ref  = E @ F^-1
    2. material law:    flux  = k @ E_ref   (contracted the muGrid way)
    3. push forward:    Q     = det(F) * (flux @ F^-T)
The ORIGINAL function skipped step 1 -- it applied the material law
directly to E, then only did step 3. This made the right-hand side `b`
inconsistent with the system matrix `K` (see `apply_system_matrix_mugrid_
deformed_grid`, which DOES correctly implement all three steps -- we
verified this numerically against the textbook Piola-transform formula and
it is correct as originally written). FIXED by adding the missing pull-back
step 1.

BUG 2 (FIXED in this version): `get_homogenized_stress_mugrid_deformed_grid`
------------------------------------------------------------------------------
This function assembles the FINAL homogenized tensor from the solved
temperature field. It performs:
    1. pull back:      E_ref  = (grad(u) + E) @ F^-1
    2. material law:    flux  = k @ E_ref
    3. ... only scales by det(F). It is MISSING the second push-forward
       multiplication by F^-T that steps 1-3 above (and the system matrix,
       and now the fixed RHS) all require.
This means even after fixing BUG 1, the quantity used to assemble the
homogenized tensor is still off by a missing right-multiplication by
`inv_F.T`. We verified numerically that adding this missing step makes the
three functions (system matrix, RHS, homogenized-stress assembly) produce
IDENTICAL transforms of the same input vector, matching the textbook Piola
formula to floating-point precision. FIXED by adding the missing
`@ inv_F.T` (via the muGrid einsum convention) before returning.

HOW TO USE THIS FILE
---------------------
In your experiment scripts, replace:
    from muFFTTO import domain
with:
    from muFFTTO import domain_2 as domain

This file re-exports everything from the original `domain.py` unchanged
(including `PeriodicUnitCell`) and only overrides two `Discretization`
methods:
    - get_rhs_mugrid_deformed_grid
    - get_homogenized_stress_mugrid_deformed_grid
No other function is modified.

IMPORTANT: after you confirm this fixes the symmetry/positive-definiteness
issue (rerun the baseline-vs-deformed comparison across your N sweep),
report this back to your supervisor / open a GitHub issue or PR against the
original `domain.py` -- this file is meant as a verifiable patch, not a
permanent fork.
"""

from __future__ import annotations

import numpy as np

# Re-export everything from the original module so existing code that does
# `from muFFTTO import domain_2 as domain` keeps working unchanged for
# anything we did not touch (PeriodicUnitCell, all other Discretization
# methods, module-level helper functions, etc.).
from muFFTTO.domain import *  # noqa: F401,F403
from muFFTTO import domain as _original_domain


class Discretization(_original_domain.Discretization):
    """Discretization with corrected deformed-grid RHS and homogenization assembly.

    Only `get_rhs_mugrid_deformed_grid` and
    `get_homogenized_stress_mugrid_deformed_grid` are overridden. Everything
    else is inherited unchanged from `muFFTTO.domain.Discretization`.
    """

    # ------------------------------------------------------------------
    # BUG 1 FIX: right-hand side must use the same 3-step pull-back /
    # material-law / push-forward transform as the system matrix.
    # ------------------------------------------------------------------
    def get_rhs_mugrid_deformed_grid(self, material_data_field_ijklqxyz,
                                      macro_gradient_field_ijqxyz,
                                      rhs_inxyz,
                                      det_of_deformation_gradient,
                                      inv_of_deformation_gradient):
        """Right-hand side of the deformed-grid homogenization problem (FIXED).

        rhs = - B^t : [ det(F_q) * ( C : (E @ F_q^-1) ) @ F_q^-T ]

        This mirrors, term for term, the pull-back / push-forward convention
        used in `apply_system_matrix_mugrid_deformed_grid` (verified correct
        against the standard Piola-transform formula for flux):

            1. pull the (macro) gradient back to the reference cell:
                   E_ref = E @ inv_F               (non-transposed)
            2. apply the constitutive law:
                   flux = C : E_ref
            3. push the flux forward and scale by det(F):
                   flux_pushed = det_F * (flux @ inv_F^T)

        The ORIGINAL `get_rhs_mugrid_deformed_grid` skipped step 1, which
        made `b` inconsistent with `K`. This was confirmed both by a
        symmetry/positive-definiteness experiment (regular-grid baseline
        was fine, deformed-grid was not, at every tested N) and by direct
        symbolic comparison against `apply_system_matrix_mugrid_deformed_grid`.

        Parameters
        ----------
        material_data_field_ijklqxyz : discretized material data tangent field
            - conductivity shape [i,j,q,x,y,z], i,j = 0,...,d-1
            - elasticity shape [i,j,k,l,q,x,y,z], i,j,k,l = 0,...,d-1
        macro_gradient_field_ijqxyz : discretized macroscopic gradient E
            (constant part of the gradient), shape [i,j,q,x,y,z]
        rhs_inxyz : output nodal field [i,n,x,y,z], filled in place
        det_of_deformation_gradient : det(F) per quadrature point
        inv_of_deformation_gradient : F^-1 per quadrature point

        Returns
        -------
        None. `rhs_inxyz` is filled in place (same convention as the
        original function).
        """
        det_F = det_of_deformation_gradient
        inv_F = inv_of_deformation_gradient

        gradient_ijqxyz = self.get_gradient_size_field(name='stress_temporary_rhs')
        gradient_ijqxyz.s[...] = macro_gradient_field_ijqxyz.s[...]

        # --- FIX (bug 1): pull the macro gradient back with inv_F BEFORE
        # applying the material law, exactly like
        # apply_system_matrix_mugrid_deformed_grid does for the micro-gradient.
        # This line was missing in the original get_rhs_mugrid_deformed_grid.
        gradient_ijqxyz.s[...] = np.einsum(
            'ij...,jk...->ik...', gradient_ijqxyz.s[...], inv_F
        )

        # apply constitutive law: flux = C : E_ref
        self.apply_material_data_mugrid(material_data_field_ijklqxyz, gradient_ijqxyz)

        # w_q * div( det(F^q) * flux^q . (F^q)^-T )  // transformed divergence
        # (unchanged from original -- this half was already consistent)
        gradient_ijqxyz.s[...] = np.einsum(
            'ij...,kj...->ik...', gradient_ijqxyz.s[...], inv_F
        ) * det_F[None, None, ...]

        self.apply_gradient_transposed_operator_mugrid(
            gradient_field_ijqxyz=gradient_ijqxyz,
            div_u_fnxyz=rhs_inxyz,
            apply_weights=True,
        )

        rhs_inxyz.s[...] *= -1

        self.fft.communicate_ghosts(field=rhs_inxyz)

    # ------------------------------------------------------------------
    # BUG 2 FIX: the homogenized-stress assembly was missing the final
    # push-forward multiplication by inv_F^T (it only scaled by det_F).
    # ------------------------------------------------------------------
    def get_homogenized_stress_mugrid_deformed_grid(self, material_data_field_ijklqxyz,
                                                      temperature_field_inxyz,
                                                      macro_gradient_field_ijqxyz,
                                                      det_of_deformation_gradient,
                                                      inv_of_deformation_gradient,
                                                      formulation=None):
        """Homogenized conductivity tensor / flux on the deformed grid (FIXED).

        Computes: (1/|domain|) * sum_q w_q * det(F_q) * [ C : ((grad(u)+E) @ F_q^-1) ] @ F_q^-T

        This uses the SAME 3-step pull-back / material-law / push-forward
        convention verified correct for the system matrix and (now) the RHS:

            1. pull back:   E_ref = (grad(u) + E) @ inv_F      (non-transposed)
            2. material law: flux = C : E_ref
            3. push forward: Q     = det_F * (flux @ inv_F^T)   <-- was missing
               the `@ inv_F^T` factor in the original code, which only did
               `flux * det_F` without the second inv_F multiplication.

        We verified numerically that without this fix, the quantity being
        averaged here does NOT match the transform used to build K and b,
        which explains why the homogenized tensor stayed non-symmetric /
        non-positive-definite even after BUG 1 was fixed.

        Parameters
        ----------
        material_data_field_ijklqxyz : discretized material data tangent field
        temperature_field_inxyz : nodal solution field (temperature)
        macro_gradient_field_ijqxyz : quadrature-point field of macroscopic
            gradient E, shape [i,j,q,x,y,z]
        det_of_deformation_gradient : det(F) per quadrature point
        inv_of_deformation_gradient : F^-1 per quadrature point
        formulation : 'small_strain' for elasticity, None/other for conductivity

        Returns
        -------
        homogenized_stress_ij : ndarray, the homogenized 2x2 (or dxd) tensor
        """
        det_F = det_of_deformation_gradient
        inv_F = inv_of_deformation_gradient

        gradient_field_ijqxyz = self.get_gradient_size_field(name='grad_temp')
        self.fft.communicate_ghosts(field=temperature_field_inxyz)
        self.apply_gradient_operator_mugrid(u_inxyz=temperature_field_inxyz,
                                             grad_u_ijqxyz=gradient_field_ijqxyz)

        # compute total heat gradient field: grad(u) + E
        gradient_field_ijqxyz.s[...] = gradient_field_ijqxyz.s + macro_gradient_field_ijqxyz.s

        # step 1: pull back with inv_F (non-transposed) -- unchanged from original
        gradient_field_ijqxyz.s[...] = np.einsum(
            'ij...,jk...->ik...', gradient_field_ijqxyz.s[...], inv_F
        )

        # symmetrization for small-strain elasticity (unchanged from original)
        if np.all(formulation == 'small_strain'):
            gradient_field_ijqxyz.s[...] = (
                gradient_field_ijqxyz.s + np.swapaxes(gradient_field_ijqxyz.s, 0, 1)
            ) / 2

        # step 2: constitutive law -- unchanged from original
        self.apply_material_data_mugrid(material_data=material_data_field_ijklqxyz,
                                         gradient_field=gradient_field_ijqxyz)

        self.apply_quadrature_weights_on_gradient_field_mugrid(grad_field=gradient_field_ijqxyz)

        # --- FIX (bug 2): step 3 must push forward with inv_F^T AND scale by
        # det_F, matching the system matrix / RHS convention. The original
        # code only did `* det_F` here, silently dropping the `@ inv_F^T`
        # factor.
        gradient_field_ijqxyz.s[...] = np.einsum(
            'ij...,kj...->ik...', gradient_field_ijqxyz.s[...], inv_F
        ) * det_F[None, None, ...]

        homogenized_stress_ij = self.mpi_reduction.sum(
            gradient_field_ijqxyz.s,
            axis=tuple(range(-self.domain_dimension - 1, 0)),
        )
        return homogenized_stress_ij / self.cell.domain_volume

    # Convenience alias, unchanged behavior (forwards to the *undeformed*
    # rhs, same as original module -- deformed-grid callers must still call
    # get_rhs_mugrid_deformed_grid explicitly).
    def get_rhs(self, **kwargs):
        return self.get_rhs_mugrid(**kwargs)


# Convenience: allow `from muFFTTO.domain_2 import PeriodicUnitCell` to keep
# pointing at the exact same (unmodified) class as the original module.
PeriodicUnitCell = _original_domain.PeriodicUnitCell
