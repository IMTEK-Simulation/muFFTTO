# Friction and adhesion in the third medium: plan and log (2026-10-25)

Goal: find out how to do friction and adhesion properly with third-medium contact, verified on analytic benchmarks
(same approach as patch test / Hertz / flat punch / disc). All prototypes in experiments/contact_patch_test/patch_contact.py
(env options), 32^2 CPU runs (seconds to a minute), smoothing term only in the gap (REG_REGION=gap).

## Plan
1. [done] Disc between plates, errors vs load (disc_compression/plot_disc_sweep.py).
2. [done] Adhesion, Dugdale type (ADHESION = sigma_0, ADH_JC, ADH_W): patch push-pull, exact tensile plateau -sigma_0 W.
3. [done, flawed] Stick v1: medium shear stiffness grows with compression, isotropic. Pushes the faces sideways.
4. Stick v2: penalise only the tangential shear of the layer, gamma = t.F.n (n = gap normal from the geometry),
   W = c mu/2 s(J) gamma^2. Check: frictionless (c = 0) -> sigma_xx = 0; c -> inf -> bonded reference.
5. Coulomb friction: gamma = gamma_e + gamma_p, slip gamma_p a history variable updated after each converged
   increment by return mapping, |tau| <= mu_c p (p = normal pressure in the medium). Limits: mu_c -> 0 frictionless,
   mu_c -> inf stick.
6. Friction benchmark: Cattaneo-Mindlin (Hertz cylinder + tangential load Q < mu P): stick zone c = a sqrt(1 - Q/(mu P)),
   tangential traction q(x) closed form.
7. Adhesion benchmark: adhesive cylinder (Hertz geometry + Dugdale law, push-pull) vs 2D JKR P(a) and pull-off.
8. Summary: what works, what needs care, recommended settings.

## Results
- (1) disc vs load: centre Brazilian stresses +1.3 % (syy) / +1.4..2.1 % (sxx) at every load, so not finite strain;
  contact a / p0 errors jump with the number of nodes across the contact (3..9): discretisation.
- (2) adhesion patch: plateau -0.467 vs exact -0.469 (0.4 %); snap-in on approach at u = 0.0212 (J_c reached),
  release on separation at u = 0.0193 (expected ~0.0219 minus elastic recovery).
- (3) stick v1: sigma_xx transferred (towards the bonded reference) but the faces slide apart MORE than frictionless
  at c = 1: the isotropic distortion penalty also acts on the uniaxial compression of the medium -> lateral push.
- (4) stick v2 (gamma = t.F.n only): zero slip at c = 1, 100, 1e4 and sigma_xx = bonded reference; the stiffness
  grows like c / J^2 (J ~ 0.01 when closed), so even c = 1 is a perfect stick.
- (5) Coulomb friction coded (MU_C, FRICTION_TOL; return mapping of gamma_p in the staggered loop, slip_n accepted per
  increment), NOT yet run. Next: E_RATIO=0.5 STICK=1 MU_C=0.05/0.2/1 -> slip zones from the free edges; then
  Cattaneo-Mindlin on the Hertz geometry.
- PAUSED 2026-10-25 for the Barzilai-Borwein comparison (user request).
