# diffuse_interface: solver tests for third-medium contact (night 29/30 Oct 2026)

`stefanus_diffuse.py` is a test copy of `../stefanus_2D_finite_strain_NeoHookean_TR.py` (main script unchanged) with
the options below. Results and reasoning: `../NIGHT_LOG.md`.

## Winning set-up

```bash
python stefanus_diffuse.py -n 128 --cells 1x2 --element bilinear_rectangle --sri \
    --incl_contrast 10 --lam_over_mu 100 --third_medium 1e-4 \
    --interp log --diffuse 2 --void_grade 1e-3 \
    --precond dual --ref_lam_over_mu 10 --tr_metric green [--device gpu]
```

* `--precond dual --tr_metric green`: dual preconditioner G K_dual G in the Steihaug CG, trust region measured in the
  Green norm (needs NuMPI with `precond_metric`, branch feat/tr_precond, uncommitted).
* `--ref_lam_over_mu 10`: lambda/mu of the reference medium of Green and dual (default: the matrix, 100).
* `--void_grade 1e-3 --diffuse w`: graded third medium; inside the voids ln mu rises from ln k to ln 1e-3 at the wall
  over ~6 w px; matrix and inclusion stay sharp. Use a fixed physical width: w = 1 / 2 / 4 at 64 / 128 / 256 px.
* Caveat (NIGHT_LOG sec. 17-18): CAP 1e-3 is too stiff for contact (+10 % stress at eps 0.2). CAP 3e-4 with a 1-px
  ramp at 64 px is physically fine but only faster than Green before buckling; over the full contact path the CG totals
  are equal.
* Code speed-ups (all preconditioners): explicit 2x2 SRI tangent (`AtBA`), closed-form dual middle operator at F = I,
  regularization without the identically-zero Q1 Laplacian term.

## Other options (tested, see NIGHT_LOG.md)

| option | meaning | verdict |
|---|---|---|
| `--interp log --diffuse w [--ramp_shift c]` | log-mixed tanh ramp of all phases, shifted into the circles by c w h | helps the dual, but changes buckling (stiff skin) or softens ligaments |
| `--interp log_inside` | ramp entirely inside the circles | too stiff |
| `--interp linear_N` | nodal fractions -> Gauss points by N, linear mixing | Green / Green-Jacobi fail, dual worse |
| `--dual_refresh [--dual_refresh_mode J/b/full] [--dual_refresh_newton]` | rebuild the dual from the current state per increment | -37 % CG at eps 0.08-0.11, no gain later; per Newton step worse |
| `--restart_frame inc*.npz` | continue from a saved frame | used for the refresh tests |
| `--time_precond` | time the preconditioner and the Hessian (with a breakdown) | |
| `--dump_material` | save mu, lambda per pixel (plot with `plot_material.py`) | |
