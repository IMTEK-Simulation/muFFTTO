# TODO / plans

What was done so far: see `NOTES.md`.

## Open small items

- [ ] Video artifact: small red "stars" in the y = Ly/2 section of the 1/64 pulse video
      (displacement there < 1 voxel; cause not found).
- [ ] Fetch the 1/128 pulse muEye file from NEMO (4.8 GB) if wanted.
- [ ] Empty folders of the two failed 1/128 attempts in `$WS/runs/bar3d` (delete only with OK).
- [ ] Projected Green at 1/128 for the pulse: decided NOT to run.

## Liquid inside the channel (planned, not implemented)

With a liquid the pressure becomes an unknown, so the internal pressure pulse can no longer be the
driver; something else has to move the liquid (external squeeze, active wall strain / peristalsis, or a
body force). In the periodic channel the liquid volume is conserved.

Staggered coupling (liquid pressure -> elasticity -> liquid ...) with a quasi-static solid and an
incompressible liquid is unconditionally unstable with one pass per step (added mass). It needs
sub-iterations with relaxation (Aitken / interface quasi-Newton) plus the volume constraint below.

Options, cheapest first:

0. **Incompressible liquid at rest**: cavity volume held at V0, uniform p(t) as a Lagrange multiplier;
   rank-1 term in the Hessian (one extra preconditioner application). Base for all staggered schemes.
1. **1D flow model along z, staggered with the 3D solid** (recommended next): A(z,t), Q(z,t), p(z,t),
   dA/dt + dQ/dz = 0, rho(dQ/dt + d(Q^2/A)/dz) + A dp/dz = -8 pi mu Q/A. A(z) from cavity volume per
   slice, p(z) back through the spatial pressure input. Captures Moens-Korteweg waves and peristaltic
   pumping. Needs wavelength >> radius (longer channel).
2. **Liquid as a viscous Lagrangian material on the same voxel grid, monolithic**: incremental potential
   W_solid + dt mu_f |sym grad v|^2 + kappa_f/2 (J-1)^2 (+ rho/(2 dt^2)|u - 2u_n + u_n-1|^2).
   Symmetric, same Newton-CG + Green. Good for small motion; voxels shear without bound under net flow.
   New preconditioner question: liquid phase with large bulk / near-zero shear next to solid and void.
3. **ALE Stokes / Navier-Stokes in the reference cavity**: mesh motion = harmonic extension of the wall
   (as in `display_fill.py`), exact shared wall nodes, net flow allowed. Saddle point (Uzawa /
   augmented Lagrangian, FFT Stokes preconditioner). Substantial work.
4. Fully Eulerian (immersed boundary, Brinkman, reference map): new code; not planned.

Decisions needed: what drives the liquid; Stokes vs inertial regime (solid inertia?); net flow over many
periods or oscillation only; stay periodic or open channel with inflow/outflow.

## Water tank (option 2 test case, planned)

2D first (plane strain, built on `pressurized_hole.py`): U-shaped elastic tank, bottom clamped, water
inside, air void above (free surface = water/void voxel boundary), gravity as potential rho g z.
Ramp gravity, or release from a tilted surface / sideways kick for sloshing. 64^2-128^2.

Checks: bottom pressure = rho g H; wall deflection vs plate estimate; sloshing frequency
w^2 = g k tanh(k H); Hessian products per solve at 1/32, 1/64, 1/128. Then a 3D box tank.

Pitfalls:
- Water has no shear stiffness: quasi-static problem singular -> time-dependent (viscous mu_f/dt and
  inertia rho/dt^2 terms), or a tiny artificial shear modulus for statics.
- Real water bulk modulus far too stiff: artificial compressibility kappa_f ~ 100-1000 rho g H.
- Volumetric locking of trilinear voxels: use F-bar (exists from the Stefanus composite work).
- Large sloshing / breaking waves tangle the voxels: gentle amplitudes only.
