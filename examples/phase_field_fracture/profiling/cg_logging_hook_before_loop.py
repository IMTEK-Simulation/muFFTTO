# ---- convergence logging (test hook, inserted before the load loop) ----
import NuMPI.Optimization.BoundedTRNewtonCG as _trmod  # noqa: E402

cg_log = []                                # one entry per Steihaug-CG solve
_orig_steihaug = _trmod._steihaug_cg


def _logged_steihaug(hessp_free, g_free, delta, tol, maxiter, pnp, precond=None):
    rec = {'lam': lam_current, 'res': [float(np.sqrt(pnp.sum(g_free * g_free)))], 'tol': tol}

    def logged_precond(r):                 # called with the current residual every CG iteration
        rec['res'].append(float(np.sqrt(pnp.sum(r * r))))
        return precond(r)
    out = _orig_steihaug(hessp_free, g_free, delta, tol, maxiter, pnp, logged_precond)
    rec['res'] = rec['res'][:1] + rec['res'][2:]   # the first precond call is r_0 itself
    rec['status'] = out[3]
    rec['nb_hessp'] = out[4]
    cg_log.append(rec)
    return out


_trmod._steihaug_cg = _logged_steihaug

diag_lams = [0.2, 0.46, 0.56, 0.8]         # elastic, near peak, after 1st crack, late
diag_log = []


def pcg_history(b, apply_M, maxiter=3000, rtol=1e-10):
    """Plain PCG on hessp at the cached tangent; returns ||r_k|| / ||r_0||."""
    x = np.zeros_like(b)
    r = b.copy()
    z = apply_M(r)
    d = z.copy()
    rz = dot_global(r, z)
    r0 = np.sqrt(dot_global(r, r))
    hist = [1.0]
    for _ in range(maxiter):
        Ad = hessp(u_state, d)
        dAd = dot_global(d, Ad)
        if dAd <= 0:
            hist.append(np.nan)
            break
        alpha = rz / dAd
        x += alpha * d
        r -= alpha * Ad
        hist.append(np.sqrt(dot_global(r, r)) / r0)
        if hist[-1] < rtol:
            break
        z = apply_M(r)
        rz_new = dot_global(r, z)
        d = z + (rz_new / rz) * d
        rz = rz_new
    return np.array(hist)


_orig_solve_increment = solve_increment


def solve_increment(lam):
    global u_state, lam_current, mechanics_preconditioner
    if diag_lams and lam >= diag_lams[0] - 1e-12 and set_F(lam, u_fluc) > 0.0:
        diag_lams.pop(0)
        lam_current = lam
        u_state = np.array(u_fluc.s)
        value, g = fun_grad(u_state)
        if value < 0.5 * INADMISSIBLE_ENERGY:
            entry = {'lam': lam}
            chosen = mechanics_preconditioner
            update_mechanics_jacobi(lam, u_fluc)            # also builds the tangent at u_state
            for name in ('Green', 'Green_Jacobi'):
                mechanics_preconditioner = name
                entry[name] = pcg_history(-np.asarray(g), precond)
            mechanics_preconditioner = chosen
            diag_log.append(entry)
            root_print(f'  diagnostic PCG at lam {lam:.3f}: iterations to 1e-10 '
                       f'Green {len(entry["Green"]) - 1}, Green_Jacobi {len(entry["Green_Jacobi"]) - 1}')
    return _orig_solve_increment(lam)
# ---- end of hook ----

