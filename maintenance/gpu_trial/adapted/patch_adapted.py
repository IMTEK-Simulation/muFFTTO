"""Add a CPU/GPU device switch to example_2D_phase_field_fracture_contact_adapted_grid.py.

usage: python patch_adapted.py IN.py OUT.py
"""
import re
import sys

s = open(sys.argv[1]).read()


def rep(old, new, count=1):
    global s
    assert s.count(old) == count, (s.count(old), old[:90])
    s = s.replace(old, new)


def body_np_to(name, module, keep=()):
    """In the body of function `name`, np.<x> -> <module>.<x> (except the names in keep)."""
    global s
    m = re.search(rf"^def {name}\(.*?(?=^\S)", s, flags=re.S | re.M)
    assert m, name
    body = m.group(0)
    keep_re = '|'.join(keep)
    pattern = rf"\bnp\.(?!(?:{keep_re})\b)" if keep else r"\bnp\."
    s = s[:m.start()] + re.sub(pattern, f"{module}.", body) + s[m.end():]


# --- device switch -------------------------------------------------------------
rep("""nnn = 32                                   # pixels per direction
""", """nnn = 32                                   # pixels per direction
# where the fields live: 'cpu' or 'gpu' (GPU needs muGrid built with CUDA + cupy)
device = 'cpu'
""")
rep("from muFFTTO import grid_adaptation  # noqa: E402\n",
    "from muFFTTO import grid_adaptation  # noqa: E402\nfrom muFFTTO import tensor_operations  # noqa: E402\n")
rep("""                               element_type=element_type)
disc_d = domain.Discretization(cell=domain.PeriodicUnitCell(domain_size=domain_size, problem_type='conductivity'),
                               nb_of_pixels_global=number_of_pixels, discretization_type='finite_element',
                               element_type=element_type)
""", """                               element_type=element_type, device=device)
disc_d = domain.Discretization(cell=domain.PeriodicUnitCell(domain_size=domain_size, problem_type='conductivity'),
                               nb_of_pixels_global=number_of_pixels, discretization_type='finite_element',
                               element_type=element_type, device=device)

# field data (.s) are numpy arrays on the CPU and cupy arrays on the GPU
if device == 'gpu':
    import cupy as xp
else:
    xp = np
to_host = tensor_operations.to_host
""")

# --- reductions work on the device of the data -------------------------------------
rep("""def global_mean(a):
    a = np.asarray(a)
    return global_sum(np.sum(a)) / max(global_sum(float(a.size)), 1.0)""",
"""def global_mean(a):
    return global_sum(a.sum()) / max(global_sum(float(a.size)), 1.0)""")
rep("""def dot_global(a, b):
    return global_sum(np.dot(np.asarray(a).ravel(), np.asarray(b).ravel()))""",
"""def dot_global(a, b):
    return global_sum((a.ravel() * b.ravel()).sum())""")

# --- material parameters and grid map at the quadrature points live on the device -----
rep("solid_q = np.broadcast_to(phase_pixel > 0, (nb_q, *phase_pixel.shape))          # [q, x, y]",
    "solid_q = xp.broadcast_to(xp.asarray(phase_pixel > 0), (nb_q, *phase_pixel.shape))  # [q, x, y]")
rep("""mu_q = np.where(solid_q, mu_solid, k_v * mu_solid)
kappa_q = np.where(solid_q, kappa_solid, k_v * kappa_solid)""",
"""mu_q = xp.where(solid_q, mu_solid, k_v * mu_solid)
kappa_q = xp.where(solid_q, kappa_solid, k_v * kappa_solid)""")
rep("quad_weights = np.asarray(disc_u.quadrature_weights).reshape((-1,) + (1,) * dim)",
    "quad_weights = xp.asarray(disc_u.quadrature_weights).reshape((-1,) + (1,) * dim)")
rep("grid_displacement.s[:, 0] = grid_displacement_local",
    "grid_displacement.s[:, 0] = xp.asarray(grid_displacement_local)")
rep("F_g = np.array(grid_gradient.s) + np.eye(dim).reshape(dim, dim, *(1,) * (grid_gradient.s.ndim - 2))",
    "F_g = grid_gradient.s.copy() + xp.eye(dim).reshape(dim, dim, *(1,) * (grid_gradient.s.ndim - 2))")
rep("F_g_inv = np.array([[F_g[1, 1], -F_g[0, 1]], [-F_g[1, 0], F_g[0, 0]]]) / J_g  # [i, j, q, x, y]",
    "F_g_inv = xp.stack([xp.stack([F_g[1, 1], -F_g[0, 1]]), xp.stack([-F_g[1, 0], F_g[0, 0]])]) / J_g  # [i, j, q, x, y]")
rep("K_g = J_g * np.einsum('kj...,lj...->kl...', F_g_inv, F_g_inv)",
    "K_g = J_g * xp.einsum('kj...,lj...->kl...', F_g_inv, F_g_inv)")

# --- material laws: computed with the array module of F (numpy or cupy) -----------------
for name in ("split_neo_hookean", "saint_venant_kirchhoff", "linear_elastic", "material"):
    body_np_to(name, "xp_")
rep('''    """W, psi^+, P = dW/dF (and A = dP/dF) of the split neo-Hookean law, pointwise.

    F: [2, 2, ...]; mu, kappa, g: [...]. See the module docstring.
    """
    J = F[0, 0] * F[1, 1] - F[0, 1] * F[1, 0]
    FiT = xp_.array([[F[1, 1], -F[1, 0]], [-F[0, 1], F[0, 0]]]) / J       # F^-T''',
'''    """W, psi^+, P = dW/dF (and A = dP/dF) of the split neo-Hookean law, pointwise.

    F: [2, 2, ...]; mu, kappa, g: [...]. See the module docstring.
    """
    xp_ = tensor_operations.get_array_module(F)   # numpy or cupy, where F lives
    J = F[0, 0] * F[1, 1] - F[0, 1] * F[1, 0]
    FiT = xp_.stack([xp_.stack([F[1, 1], -F[1, 0]]), xp_.stack([-F[0, 1], F[0, 0]])]) / J       # F^-T''')
rep('''    """W, P (and A) of W = lam/2 (tr E)^2 + mu E:E, E = (F^T F - I)/2 (plane strain), pointwise."""
''', '''    """W, P (and A) of W = lam/2 (tr E)^2 + mu E:E, E = (F^T F - I)/2 (plane strain), pointwise."""
    xp_ = tensor_operations.get_array_module(F)
''')
rep('''    """W, P (and A) of W = lam/2 (tr eps)^2 + mu eps:eps, eps = sym(F - I) (plane strain), pointwise."""
''', '''    """W, P (and A) of W = lam/2 (tr eps)^2 + mu eps:eps, eps = sym(F - I) (plane strain), pointwise."""
    xp_ = tensor_operations.get_array_module(F)
''')
rep('''    the third medium (neo-Hookean) or a super-soft St. Venant-Kirchhoff / linear solid (void_model)."""
''', '''    the third medium (neo-Hookean) or a super-soft St. Venant-Kirchhoff / linear solid (void_model)."""
    xp_ = tensor_operations.get_array_module(F)
''')

# --- fields on the device ------------------------------------------------------------------
rep("damage_weight_q = np.ones(solid_q.shape) if damage_in_medium else solid_q.astype(float)",
    "damage_weight_q = xp.ones(solid_q.shape) if damage_in_medium else solid_q.astype(float)")
rep("phase_out.s[0, 0] = phase_pixel", "phase_out.s[0, 0] = xp.asarray(phase_pixel)")
rep("g_q = np.ones(solid_q.shape)               # g_s(d) at the quadrature points",
    "g_q = xp.ones(solid_q.shape)               # g_s(d) at the quadrature points")
body_np_to("update_degradation", "xp")
body_np_to("set_F", "xp")
rep("LtL_field.s[...] = np.random.default_rng(rank).random(LtL_field.s.shape)",
    "LtL_field.s[...] = xp.asarray(np.random.default_rng(rank).random(LtL_field.s.shape))")
rep("hess_grid = np.array(hess_grid_field.s)                                         # H_X d_g [c, a, b, q, x, y]",
    "hess_grid = hess_grid_field.s.copy()                                            # H_X d_g [c, a, b, q, x, y]")
body_np_to("physical_hessian", "xp")
body_np_to("add_regularization_physical", "xp")

# --- NuMPI's optimizer works with numpy vectors (host); the fields can be on the GPU -----
rep('''def fun_grad(x):
    """(Pi_elastic + Pi_reg, gradient) at the fluctuation x, damage fixed."""''',
'''def field_to_optimizer(field):
    """Copy of the field data as a numpy array (host), for NuMPI's optimizer."""
    if device == 'gpu':
        return field.s.get()  # .get() already makes a host copy
    return np.array(field.s, copy=True)


def fun_grad(x):
    """(Pi_elastic + Pi_reg, gradient) at the fluctuation x, damage fixed."""''')
rep("""    x_field.s[...] = x.reshape(u_shape)
    if not set_F(lam_current, x_field) > 0.0:""",
    """    x_field.s[...] = xp.asarray(x.reshape(u_shape))
    if not set_F(lam_current, x_field) > 0.0:""")
rep("    stress_field.s[...] = J_g * np.einsum('ij...,kj...->ik...', P, F_g_inv)     # J_g P . F_g^-T",
    "    stress_field.s[...] = J_g * xp.einsum('ij...,kj...->ik...', P, F_g_inv)     # J_g P . F_g^-T")
rep("""    grad_field.s[...] += Ru_field.s
    return Pi, np.array(grad_field.s, copy=True)""",
    """    grad_field.s[...] += Ru_field.s
    return Pi, field_to_optimizer(grad_field)""")
rep("""    stale = (not tangent_cache['valid'] or tangent_cache['lam'] != lam
             or not np.array_equal(tangent_cache['x'], np.reshape(x, -1)))""",
"""    # x comes from the optimizer (numpy) or from a field (numpy or cupy): compare on the host
    x = to_host(x)
    stale = (not tangent_cache['valid'] or tangent_cache['lam'] != lam
             or not np.array_equal(tangent_cache['x'], np.reshape(x, -1)))""")
rep("""    x_field.s[...] = x.reshape(u_shape)
    set_F(lam, x_field)""", """    x_field.s[...] = xp.asarray(x.reshape(u_shape))
    set_F(lam, x_field)""")
rep("""    A = np.einsum('imkn...,jm...->ijkn...', A, F_g_inv)
    tangent_field.s[...] = J_g * np.einsum('ijkn...,ln...->ijkl...', A, F_g_inv)""",
"""    A = xp.einsum('imkn...,jm...->ijkn...', A, F_g_inv)
    tangent_field.s[...] = J_g * xp.einsum('ijkn...,ln...->ijkl...', A, F_g_inv)""")
rep("    v_field.s[...] = v.reshape(u_shape)", "    v_field.s[...] = xp.asarray(v.reshape(u_shape))")
rep("""    disc_u.fft.communicate_ghosts(Bv_field)
    return np.array(Bv_field.s, copy=True)""",
    """    disc_u.fft.communicate_ghosts(Bv_field)
    return field_to_optimizer(Bv_field)""")
rep("    jacobi_u = np.sqrt(reference_diagonal) * disc_u.get_preconditioner_Jacobi_mugrid(operator=hessian_operator).s",
    "    jacobi_u = float(np.sqrt(reference_diagonal)) * disc_u.get_preconditioner_Jacobi_mugrid(operator=hessian_operator).s")
rep("    r_field.s[...] = v.reshape(u_shape)", "    r_field.s[...] = xp.asarray(v.reshape(u_shape))")
rep("""        Pv_field.s[...] *= jacobi_u
    return np.array(Pv_field.s, copy=True)""",
    """        Pv_field.s[...] *= jacobi_u
    return field_to_optimizer(Pv_field)""")

# --- damage problem on the device ---------------------------------------------------------
body_np_to("apply_damage_operator", "xp")
body_np_to("K_d", "xp")
body_np_to("M_d", "xp")
rep("    history.s[0, 0] = np.maximum(history_n.s[0, 0], np.where(solid_q, psi_plus, 0.0))",
    "    history.s[0, 0] = xp.maximum(history_n.s[0, 0], xp.where(solid_q, psi_plus, 0.0))")

# --- load stepping: the optimizer gets / returns numpy vectors ---------------------------
rep("        res = tr_newton_bounded(fun=scaled_fun_grad, x0=np.array(u_fluc.s), hessp=scaled_hessp, precond=precond,",
    "        res = tr_newton_bounded(fun=scaled_fun_grad, x0=field_to_optimizer(u_fluc), hessp=scaled_hessp, precond=precond,")
rep("        u_fluc.s[...] = np.asarray(res.x).reshape(u_shape)",
    "        u_fluc.s[...] = xp.asarray(np.asarray(res.x).reshape(u_shape))")
rep("u_previous, dlam_previous = np.copy(u_fluc.s), None     # last converged state before the current one",
    "u_previous, dlam_previous = u_fluc.s.copy(), None     # last converged state before the current one")
rep("            u_previous = np.copy(u_fluc.s)", "            u_previous = u_fluc.s.copy()")
rep("    u_saved, d_saved = np.copy(u_fluc.s), np.copy(damage.s)",
    "    u_saved, d_saved = u_fluc.s.copy(), damage.s.copy()")
open(sys.argv[2], "w").write(s)
print("ok")
