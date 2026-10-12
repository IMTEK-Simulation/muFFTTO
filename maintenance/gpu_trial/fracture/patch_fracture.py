"""Add a CPU/GPU device switch to example_2D_phase_field_fracture_contact.py.

usage: python patch_fracture.py IN.py OUT.py
"""
import re
import sys

s = open(sys.argv[1]).read()


def rep(old, new, count=1):
    global s
    assert s.count(old) == count, (s.count(old), old[:80])
    s = s.replace(old, new)


# --- device switch -----------------------------------------------------------
rep("""nnn = 32                                   # pixels per direction
""", """nnn = 32                                   # pixels per direction
# where the fields live: 'cpu' or 'gpu' (GPU needs muGrid built with CUDA + cupy)
device = 'cpu'
""")
rep("from muFFTTO import io_utils  # noqa: E402\n",
    "from muFFTTO import io_utils  # noqa: E402\nfrom muFFTTO import tensor_operations  # noqa: E402\n")
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

# --- reductions work on the device of the data ---------------------------------
rep("""def global_mean(a):
    a = np.asarray(a)
    return global_sum(np.sum(a)) / max(global_sum(float(a.size)), 1.0)""",
"""def global_mean(a):
    return global_sum(a.sum()) / max(global_sum(float(a.size)), 1.0)""")
rep("""def dot_global(a, b):
    return global_sum(np.dot(np.asarray(a).ravel(), np.asarray(b).ravel()))""",
"""def dot_global(a, b):
    return global_sum((a.ravel() * b.ravel()).sum())""")

# --- material parameters at the quadrature points live on the device ------------
rep("""solid_q = np.broadcast_to(phase_pixel > 0, (nb_q, *phase_pixel.shape))          # [q, x, y]""",
    """solid_q = xp.broadcast_to(xp.asarray(phase_pixel > 0), (nb_q, *phase_pixel.shape))  # [q, x, y]""")
rep("""mu_q = np.where(solid_q, mu_solid, k_v * mu_solid)
kappa_q = np.where(solid_q, kappa_solid, k_v * kappa_solid)""",
"""mu_q = xp.where(solid_q, mu_solid, k_v * mu_solid)
kappa_q = xp.where(solid_q, kappa_solid, k_v * kappa_solid)""")
rep("""quad_weights = np.asarray(disc_u.quadrature_weights).reshape((-1,) + (1,) * dim)""",
    """quad_weights = xp.asarray(disc_u.quadrature_weights).reshape((-1,) + (1,) * dim)""")

# --- material law: computed with the array module of F (numpy or cupy) -----------
m = re.search(r"^def split_neo_hookean\(.*?(?=^\S)", s, flags=re.S | re.M)
body = m.group(0)
body = body.replace('''    J = F[0, 0] * F[1, 1] - F[0, 1] * F[1, 0]
    FiT = np.array([[F[1, 1], -F[1, 0]], [-F[0, 1], F[0, 0]]]) / J       # F^-T''',
'''    xp_ = tensor_operations.get_array_module(F)   # numpy or cupy, where F lives
    J = F[0, 0] * F[1, 1] - F[0, 1] * F[1, 0]
    FiT = xp_.stack([xp_.stack([F[1, 1], -F[1, 0]]), xp_.stack([-F[0, 1], F[0, 0]])]) / J       # F^-T''')
body = re.sub(r"\bnp\.", "xp_.", body)
assert "np." not in body
s = s[:m.start()] + body + s[m.end():]

# --- fields on the device ----------------------------------------------------------
rep("""damage_weight_q = np.ones(solid_q.shape) if damage_in_medium else solid_q.astype(float)""",
    """damage_weight_q = xp.ones(solid_q.shape) if damage_in_medium else solid_q.astype(float)""")
rep("""phase_out.s[0, 0] = phase_pixel""", """phase_out.s[0, 0] = xp.asarray(phase_pixel)""")
rep("""g_q = np.ones(solid_q.shape)               # g_s(d) at the quadrature points""",
    """g_q = xp.ones(solid_q.shape)               # g_s(d) at the quadrature points""")
rep("""    g_q[...] = np.where(solid_q, (1.0 - damage_q.s[0, 0]) ** 2 + k_residual, 1.0)""",
    """    g_q[...] = xp.where(solid_q, (1.0 - damage_q.s[0, 0]) ** 2 + k_residual, 1.0)""")
rep("""LtL_field.s[...] = np.random.default_rng(rank).random(LtL_field.s.shape)""",
    """LtL_field.s[...] = xp.asarray(np.random.default_rng(rank).random(LtL_field.s.shape))""")


# --- NuMPI's optimizer works with numpy vectors (host); the fields can be on the GPU --
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
rep("""    v_field.s[...] = v.reshape(u_shape)""", """    v_field.s[...] = xp.asarray(v.reshape(u_shape))""")
rep("""    disc_u.fft.communicate_ghosts(Bv_field)
    return np.array(Bv_field.s, copy=True)""",
    """    disc_u.fft.communicate_ghosts(Bv_field)
    return field_to_optimizer(Bv_field)""")
rep("""    jacobi_u = np.sqrt(reference_diagonal) * disc_u.get_preconditioner_Jacobi_mugrid(operator=hessian_operator).s""",
    """    jacobi_u = float(np.sqrt(reference_diagonal)) * disc_u.get_preconditioner_Jacobi_mugrid(operator=hessian_operator).s""")
rep("""    r_field.s[...] = v.reshape(u_shape)""", """    r_field.s[...] = xp.asarray(v.reshape(u_shape))""")
rep("""        Pv_field.s[...] *= jacobi_u
    return np.array(Pv_field.s, copy=True)""",
    """        Pv_field.s[...] *= jacobi_u
    return field_to_optimizer(Pv_field)""")

# --- damage problem: masks and history on the device ------------------------------
rep("""    Ax.s[...] = np.where(damage_free_nodes, Ax.s, x.s)""", """    Ax.s[...] = xp.where(damage_free_nodes, Ax.s, x.s)""")
rep("""    Px.s[...] = np.where(damage_free_nodes, Px.s, x.s)       # identity on the pinned nodes""",
    """    Px.s[...] = xp.where(damage_free_nodes, Px.s, x.s)       # identity on the pinned nodes""")
rep("""    history.s[0, 0] = np.maximum(history_n.s[0, 0], np.where(solid_q, psi_plus, 0.0))""",
    """    history.s[0, 0] = xp.maximum(history_n.s[0, 0], xp.where(solid_q, psi_plus, 0.0))""")

# --- load stepping: the optimizer gets / returns numpy vectors ----------------------
rep("""        res = tr_newton_bounded(fun=scaled_fun_grad, x0=np.array(u_fluc.s), hessp=scaled_hessp, precond=precond,""",
    """        res = tr_newton_bounded(fun=scaled_fun_grad, x0=field_to_optimizer(u_fluc), hessp=scaled_hessp, precond=precond,""")
rep("""        u_fluc.s[...] = np.asarray(res.x).reshape(u_shape)""",
    """        u_fluc.s[...] = xp.asarray(np.asarray(res.x).reshape(u_shape))""")
rep("""u_previous, dlam_previous = np.copy(u_fluc.s), None     # last converged state before the current one""",
    """u_previous, dlam_previous = u_fluc.s.copy(), None     # last converged state before the current one""")
rep("""    u_saved, d_saved = np.copy(u_fluc.s), np.copy(damage.s)""",
    """    u_saved, d_saved = u_fluc.s.copy(), damage.s.copy()""")
rep("""            u_previous = np.copy(u_fluc.s)""", """            u_previous = u_fluc.s.copy()""")
open(sys.argv[2], "w").write(s)
print("ok")
