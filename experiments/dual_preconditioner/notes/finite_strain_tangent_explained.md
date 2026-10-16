# The finite-strain tangent, its rotational stiffness, and why the dual preconditioner must not invert it

Notes of 2026-10-16 (dual preconditioner study, `finite_strain_P123/`). Written as an explanation, step by step.

---

## 1. Where the tangent comes from

At finite strain the stress that enters the equilibrium on the reference grid is the first Piola-Kirchhoff stress
`P` (force in the current configuration per area in the reference configuration). For a hyperelastic material

    P = F S,        S = dpsi/dE,        E = 1/2 (F^T F - I)

with `S` the second Piola-Kirchhoff stress ("material" stress, work-conjugate to the Green-Lagrange strain `E`).
Newton linearizes `P(F)`; the tangent is the fourth-order tensor

    A = dP/dF,        A_ijkl = dP_ij / dF_kl,

stored as a symmetric `d^2 x d^2` matrix (major symmetry `A_ijkl = A_klij`, because `A` is the Hessian of the stored
energy `psi(F)`). The product rule splits it into two parts:

    dP = dF S + F (C_S : dE),        dE = sym(F^T dF),        C_S = d^2 psi / dE dE

and the second variation of the energy for a perturbation `dF` is

    dF : A : dF  =  dE : C_S : dE          (material stiffness)
                 +  tr(dF S dF^T)          (geometric = stress stiffness)

* The **material stiffness** comes from stretching / shearing the material. It only sees `dE = sym(F^T dF)`.
* The **geometric stiffness** comes from the stress that is already there, `S`. It sees the whole `dF`.

---

## 2. Split the perturbation into strain and rotation

Look at a point that is only slightly deformed, `F ~ I`, but already carries a stress `sigma ~ S` (for small strain and
moderate stress the stress measures coincide to first order). Split the gradient perturbation into its symmetric
part (strain) and skew part (rotation),

    dF = eps + W,        eps = sym(dF),        W = skw(dF);    in 2D  W = omega [[0, -1], [1, 0]]  (rotation by omega)

Insert into the two terms:

* material part: `dE = sym(dF) = eps`, so it gives `eps : C : eps` - **rotations do not appear**;
* geometric part: `tr((eps + W) sigma (eps + W)^T)` = `tr(eps sigma eps)` + `2 tr(eps sigma W^T)` + `tr(W sigma W^T)`.

The last term, for the 2D rotation `W`:

    W sigma W^T = omega^2 [[ sigma_22, -sigma_12], [-sigma_12, sigma_11]]   ->   tr(W sigma W^T) = omega^2 tr(sigma)

So the tangent has three kinds of blocks (Mandel basis: 3 symmetric directions, 1 skew direction in 2D):

| block | origin | size | sign |
|---|---|---|---|
| `A_ss` strain-strain | material `C` + geometric `(eps sigma) : eps` | ~ `C + O(sigma)` | positive (the material resists strain) |
| `A_ww` rotation-rotation | **geometric only**: `omega^2 tr(sigma)` | ~ `tr(sigma)` | **sign of the stress**: + tension, - compression, 0 unstressed |
| `A_sw` strain-rotation coupling | geometric: `2 tr(eps sigma W^T)` | ~ deviatoric stress | zero for a pure pressure |

**The rotational stiffness is purely geometric - it is the stress itself.** The material part contributes nothing
to a rigid rotation, because a rigid rotation does not strain the material.

(General `F`: the same structure holds in the current configuration; `C_S` is pushed forward by `F`, and the
geometric part is `delta_ik S_jl` in index form. In the code the tangent is `material.get_algorithmic_tangent`.)

---

## 3. What rotational stiffness means physically

Rotate a stressed element rigidly by a small angle `omega`. There is no strain, so the material part does nothing.
But the nominal stress `P = R sigma` turns with the element, and the existing forces do work during the rotation:

* **Tension** (`tr sigma > 0`): like a guitar string or a pulled rope - turning it sideways produces a restoring
  force. Rotating costs energy: positive rotational stiffness. Tension stabilizes.
* **Compression** (`tr sigma < 0`): like a column under an axial load - tilting it lets the load descend and
  releases energy. Negative rotational stiffness. This is the mechanism of buckling.
* **No stress** (`F = I`, `sigma = 0`): a rigid rotation is free. Zero rotational stiffness - the tangent is
  singular exactly at the reference state (and `C_0`, the reference of the Green operator, is singular on rotations
  for the same reason).

---

## 4. Why the structure is stable although single points have negative stiffness

A single quadrature point cannot rotate on its own: the rotations come from a displacement field, so they must be
compatible with the neighbours. You cannot rotate one element without straining the surrounding ones. For periodic
fields this is quantified by the Korn identity (exact for P1 and for Q1 with 2x2 Gauss points):

    ||grad v||^2 = 2 ||eps(v)||^2 - ||div v||^2      =>      ||W(v)||^2 <= ||eps(v)||^2

So the negative rotational energy of compressed points (of size `|sigma| ||W||^2`) is outweighed by the positive
material energy of the strains it forces (of size `mu ||eps||^2`): the global stiffness `K(A) = B^T W A B` stays
positive definite as long as the compressive stresses are small compared to the material stiffness. When they are
not, `K(A)` itself becomes indefinite - that is real (structural) buckling. It happened for P2 with soft inclusions
on 64^2 in the third increment: CG stopped with "Hessian is not positive definite".

Numbers of our runs (shear 0.2 + stretch 0.2): the stresses are ~0.1, the shear modulus of the solid is 0.385. In the
soft inclusions (`kappa = 1e-3`) the material part is a thousand times smaller while the stress is not, so there the
geometric part dominates. Counted per Newton step, the 4 x 4 tangent had a negative eigenvalue at **hundreds of
quadrature points** (up to 564 on 128^2 for P1 with `kappa = 1e-3`, up to 2054 on 64^2 for `kappa = 1e3`), while
Newton converged normally - the structure was stable, single points were not.

---

## 5. What the dual preconditioner needs, and what goes wrong

The dual preconditioner is `P_D = G_0 K(C_0 A^-1 C_0) G_0`. Its guarantee `lambda_min(P_D K) >= 1` comes from the
complementary energy: for any stress `tau` in equilibrium (`B^T W tau = b`),

    b^T K(A)^-1 b  <=  tau^T W A^-1 tau

and the guess `tau_0 = C_0 B G_0 b` is such a stress. This needs `A^-1` to be a **positive** compliance: a negative
compliance turns the upper bound into nonsense.

Because `tau_0` is symmetric, only the strain block of the compliance enters (Schur complement):

    (A^-1)_ss = S^-1,        S = A_ss - A_sw A_ww^-1 A_ws

In words: "how much does this point strain under a stress if it may also rotate freely". It contains the
**inverse rotational stiffness** `A_ww^-1` through the coupling `A_sw`.

* **Small strain:** `A = C` has no rotational part (`A_sw = A_ww = 0`); the inverse on the symmetric strains is
  safe, the bound holds, the dual works (P1-P3 results).
* **Plain inverse at finite strain:** where `A_ww ~ 0` (stress changes sign), `A_ww^-1` explodes; where `A_ww < 0`
  (compression), the correction changes sign and the compliance can become negative. The middle material
  `C_0 A^-1 C_0` is huge or negative at those points, `P_D` is no longer positive definite, and CG has no
  convergence theory: 1,000-3,000 CG iterations per Newton step instead of 50-100 (`dual_inv` runs: 1451 per Newton
  step on 128^2 for P1, `kappa = 1e-3`; 3374 on 64^2 for `kappa = 1e3`).
* **Pseudo-inverse with |eigenvalues| ("full"):** takes the absolute values before inverting - everything stays
  positive, `P_D` is SPD, CG works. But the small rotational stiffness still produces large entries, so the
  condition number grows along the load path (16^2 spectra, P1, `kappa = 1e3`: 274 after the first increment, 1193
  after the last). The cutoff (1e-6 relative) was almost never active (at most 16 points); the absolute value is
  what matters.
* **Symmetric block ("sym"):** uses `A_ss^-1`, the response to strain with the rotation held fixed. `A_ss` stays
  positive (stretching and shearing always cost energy), `A_ww` never enters - robust and the fastest
  (P1: 30-40 % fewer CG iterations than Green, 72-83 % fewer than Green-Jacobi on 256^2). The price: it ignores the
  softening that free rotation would allow (`A_ss^-1 <= S^-1`), so it is slightly too stiff and the guarantee is lost
  (`lambda_min = 0.88` on 16^2). The bound it keeps: `lambda_min >= theta`,
  `theta = min_q lambda_min(S^1/2 A_ss^-1 S^1/2)`, see the paper, Section 3.1.

---

## 6. Summary

    finite-strain tangent  =  material stiffness  (strains only, always positive)
                           +  geometric stiffness (the stress, acting on rotations and on strain-rotation coupling)

* The **rotational stiffness equals the stress**: zero without load, positive in tension, negative in compression.
* It is harmless for the structure (compatibility / Korn), but deadly when inverted **point by point**.
* Hence the dual preconditioner at finite strain should invert only the strain block of the tangent (`sym`), or at
  least use absolute eigenvalues (`full`), never the plain inverse.

Related: paper, Section 3.1 (derivation, eq. Schur complement and the bounds of both variants);
`finite_strain_P123/finite_strain_P123.py` (`--dual_variant full | sym | inv`; `safeguard_counts` in the data files
count the quadrature points with negative / below-cutoff eigenvalues).
