"""
Mesh-size study of a linear elastic cantilever with void pixels.

A horizontal beam (length L, height H) sits in a periodic cell, surrounded on all sides by zero-stiffness (void)
pixels, so it does not touch its periodic images. The left face is clamped (all displacements zero, NodalConstraints)
and a total shear force F_y is applied on the right face (consistent nodal forces of a uniform edge load). Small
strain, plane strain, bilinear elements, macro gradient zero.

The beam geometry is fixed; the pixel size h = H / n_H is refined. Per mesh: tip deflection (mean u_y of the loaded
face), compliance (= F u_tip, twice the strain energy), CG iterations (Green preconditioner with the beam as
reference, energy-norm stopping). Errors are relative to the Richardson extrapolation of the two finest meshes;
Timoshenko beam theory is printed as an outside check (it differs from 2D elasticity by the clamp warping).

usage:  python cantilever_mesh_study.py           (env: N_H='2,4,8,16,32', VOID=2, CG_TOL=1e-6,
        BEAM_LENGTH=1, CLAMP_COLUMNS=1, CLAMP_SPAN=1, LOAD_X=L,
        VOID_THICKNESS=t (physical, instead of VOID pixels), BC=clamp|simple|free|traction|stiff|stiffforce (STIFF_FACTOR, STIFF_LAYERS or STIFF_THICKNESS),
        PRECOND=green_dirichlet|green_dirichlet_S|green_dirichlet_reg (MASS_C), HOLES=n (HOLE_RADIUS), HETERO=kappa; non-default values tag the output names)
        meshes in parallel:  run_parallel.sh, i.e. one SOLVE_ONLY=1 N_H=<n> worker per mesh, then PLOT_ONLY=1
writes exp_data/runs/n_H<n>_cgtol<tol>.npz per mesh (reused if present), and
       exp_data/cantilever_mesh_study.npz, .csv, _convergence.png, _deformed.png, _setup.png
"""
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import time

import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker
from matplotlib.collections import PolyCollection

from muFFTTO import constraints
from muFFTTO import domain
from muFFTTO import material_models
from muFFTTO import solvers
from muFFTTO import topology_optimization

HERE = os.path.dirname(os.path.realpath(__file__))
OUT = os.path.join(HERE, 'exp_data')
os.makedirs(OUT, exist_ok=True)

beam_length = float(os.environ.get('BEAM_LENGTH', 1.0))
beam_height = 0.2
clamp_columns = int(os.environ.get('CLAMP_COLUMNS', 1))  # columns of clamped nodes from the left face inwards
clamp_span = float(os.environ.get('CLAMP_SPAN', 1.0))  # clamped part of the face height, centred
load_x = float(os.environ.get('LOAD_X', beam_length))  # x of the loaded node column (default: the right face)
void_thickness = float(os.environ.get('VOID_THICKNESS', 0))  # > 0: void of this physical thickness on each side
# supports: 'clamp' (left face clamped, shear load at x = LOAD_X), 'simple' (pin at the lower left corner, roller at the
# lower right corner: 3 constrained dofs, shear load at mid-span), 'free' (nothing constrained, self-equilibrated
# load: -F at mid-span, +F/2 at both ends; the solution is unique up to rigid motions, the compliance is not),
# 'traction' (the cantilever without Dirichlet: nothing constrained, the clamp replaced by its reactions as tractions
# on the left face -- parabolic shear -F and linear bending stress of the moment F L -- and parabolic shear F at the
# tip; the exact 2D solution (Timoshenko & Goodier) has the compliance F^2 L^3 / (3 E' I) + 6 F^2 L / (5 mu H))
supports = os.environ.get('BC', 'clamp')
# CLAMP_FULL=1: the clamped node column(s) run through the void buffer over the whole cell height, so the fixed dofs
# form a complete periodic grid line and S = E^T G E is exactly block-circulant (INNER_PRECOND=circ then inverts it)
clamp_full = bool(os.environ.get('CLAMP_FULL'))
# BC=stiff: instead of the clamped face, STIFF_LAYERS pixel columns left of the beam, STIFF_FACTOR times stiffer than
# the beam, held only at the bottom and top node of their outer node column (both components); tip load as for clamp.
# BC=stiffforce: the same layer held by nothing, its outer node column loaded by the reactions of the tip load
# (shear -F, linear bending stress of the moment F (L + layer)); the nodal forces are in equilibrium to round-off.
# PRECOND=green_dirichlet: Green corrected to the exact inverse of the reference problem WITH the fixed dofs:
# z = M (r - E lam) + N alpha, E^T z = 0, N^T (r - E lam) = 0 (E: fixed dofs, N: the translations Green leaves out);
# the (#fixed + 2) system is assembled once from one Green application per fixed dof; 2 Green applications per CG step.
# PRECOND=green_dirichlet_reg: the same without the translations: the reference gets a weak mass term,
# K_c = K(C_0) + c N^T W N with c = MASS_C mu / (cell size)^2, which is regular, so its Green operator G_c inverts the
# zero frequency too; then z = G_c (r - lam), (P G_c P) lam = P G_c r (plain Woodbury, no alpha)
stiff_factor = float(os.environ.get('STIFF_FACTOR', 100))
stiff_layers = int(os.environ.get('STIFF_LAYERS', 1))
stiff_thickness = float(os.environ.get('STIFF_THICKNESS', 0))  # > 0: layer of this physical thickness instead
TAG = ''.join(f'_{k}{v:g}' for k, v, default in (('L', beam_length, 1.0), ('cc', clamp_columns, 1),
                                                  ('span', clamp_span, 1.0), ('load', load_x, beam_length))
              if v != default) + (f'_void{void_thickness:g}' if void_thickness > 0 else '') + \
      (f'_bc{supports}' if supports != 'clamp' else '') + \
      (f'{stiff_factor:g}' + (f'_t{stiff_thickness:g}' if stiff_thickness > 0 else
                             f'x{stiff_layers}' if stiff_layers != 1 else '') if supports.startswith('stiff') else '') + \
      ('_pgd' if os.environ.get('PRECOND') == 'green_dirichlet' else '') + \
      ('_pgdS' if os.environ.get('PRECOND') == 'green_dirichlet_S' else '') + \
      (f'_pgdreg{float(os.environ.get("MASS_C", 1e-3)):g}' if os.environ.get('PRECOND') == 'green_dirichlet_reg' else '') + \
      (f'_pgdi{float(os.environ.get("INNER_TOL", 1e-8)):g}' if os.environ.get('PRECOND') == 'green_dirichlet_inner' else '')
# green_dirichlet_inner variants: INNER_MAXIT=k (exactly k inner CG steps, no tolerance), INNER_WARM=1 (inner CG starts
# from the lambda of the previous outer iteration)
inner_maxit = int(os.environ.get('INNER_MAXIT', 0))
inner_warm = bool(os.environ.get('INNER_WARM'))
if os.environ.get('PRECOND') == 'green_dirichlet_inner' and inner_maxit:
    TAG = TAG.replace(f'_pgdi{float(os.environ.get("INNER_TOL", 1e-8)):g}', f'_pgdik{inner_maxit}')
TAG += ('_clampfull' if clamp_full else '') + ('_direct' if os.environ.get('INNER_DIRECT') else '') + ('_warm' if inner_warm else '') + \
       (f'_ip{os.environ["INNER_PRECOND"]}' if os.environ.get('INNER_PRECOND') else '')
TAG += '_dense' if os.environ.get('DENSE_SPECTRUM') else ''  # runs with the dense spectrum stored apart
# HOLES=n: n circular holes of radius HOLE_RADIUS (physical) on the beam's centre line, at x = (i + 1/2) L / n;
# a pixel is a hole when its centre lies inside a circle
nb_holes = int(os.environ.get('HOLES', 0))
hole_radius = float(os.environ.get('HOLE_RADIUS', 0.05))
TAG += f'_holes{nb_holes}r{hole_radius:g}' if nb_holes > 0 else ''
# HETERO=kappa: heterogeneous beam, stiffness factor kappa^-phi, phi = (1 + sin(2 pi x / 0.2 + 0.7) sin(2 pi y / 0.1 + 0.3))
# / 2 at the pixel centres (x, y from the beam's lower left corner): between 1/kappa and 1, fixed in physical units
hetero_contrast = float(os.environ.get('HETERO', 1))
TAG += f'_het{hetero_contrast:g}' if hetero_contrast != 1 else ''
E_young, poisson = 1.0, 0.3
total_force_y = -1e-4  # per unit thickness; tip deflection ~ 0.05 L
nb_void = int(os.environ.get('VOID', 2))  # void pixels on each side of the beam
cg_tol = float(os.environ.get('CG_TOL', 1e-6))
cg_delay = 10
mesh_list = [int(n) for n in os.environ.get('N_H', '2,4,8,16,32').split(',')]  # pixels over the beam height

lam, mu = material_models.get_lame_parameters(E=E_young, poisson=poisson)


def solve(n_H):
    h = beam_height / n_H
    n_L = int(round(beam_length / h))
    nb_beam_pixels = (n_L, n_H)
    nb_void = int(round(void_thickness / h)) if void_thickness > 0 else globals()['nb_void']
    number_of_pixels = (n_L + 2 * nb_void, n_H + 2 * nb_void)
    cell = domain.PeriodicUnitCell(domain_size=[number_of_pixels[0] * h, number_of_pixels[1] * h],
                                   problem_type='elasticity')
    disc = domain.Discretization(cell=cell, nb_of_pixels_global=number_of_pixels,
                                 discretization_type='finite_element', element_type='bilinear_rectangle')

    # beam pixels and nodes; node (ix, iy) is the lower-left corner of pixel (ix, iy)
    ix, iy = disc.fft.icoords
    beam_x = (nb_void, nb_void + nb_beam_pixels[0])
    beam_y = (nb_void, nb_void + nb_beam_pixels[1])
    beam_pixels = (ix >= beam_x[0]) & (ix < beam_x[1]) & (iy >= beam_y[0]) & (iy < beam_y[1])
    for i in range(nb_holes):  # pixel centres relative to the beam's lower left corner
        x_c, y_c = (i + 0.5) * beam_length / nb_holes, 0.5 * beam_height
        beam_pixels &= ((ix - beam_x[0] + 0.5) * h - x_c) ** 2 + ((iy - beam_y[0] + 0.5) * h - y_c) ** 2 > hole_radius ** 2
    left_nodes = (ix == beam_x[0]) & (iy >= beam_y[0]) & (iy <= beam_y[1])
    half_span = 0.5 * clamp_span * n_H + 1e-9  # in pixels from the mid-height
    clamped_nodes = ((ix >= beam_x[0]) & (ix < beam_x[0] + clamp_columns)
                     & (np.abs(iy - 0.5 * (beam_y[0] + beam_y[1])) <= half_span))
    if clamp_full:  # the clamp column over the whole periodic height, void included (zero stiffness: same problem)
        clamped_nodes = (ix >= beam_x[0]) & (ix < beam_x[0] + clamp_columns)
    right_nodes = (ix == beam_x[0] + int(round(load_x / h))) & (iy >= beam_y[0]) & (iy <= beam_y[1])  # loaded

    lam_q = disc.get_quad_field_scalar(name='lam_first_lame')
    mu_q = disc.get_quad_field_scalar(name='mu_second_lame')
    lam_q.s[...] = 0
    mu_q.s[...] = 0
    lam_q.s[..., beam_pixels] = lam
    mu_q.s[..., beam_pixels] = mu
    if hetero_contrast != 1:
        x_c, y_c = (ix - beam_x[0] + 0.5) * h, (iy - beam_y[0] + 0.5) * h
        phi = 0.5 * (1 + np.sin(2 * np.pi * x_c / 0.2 + 0.7) * np.sin(2 * np.pi * y_c / 0.1 + 0.3))
        factor = np.where(beam_pixels, hetero_contrast ** -phi, 1.0)
        lam_q.s[...] *= factor
        mu_q.s[...] *= factor
    layers = int(round(stiff_thickness / h)) if stiff_thickness > 0 else stiff_layers  # pixel columns of the layer
    if supports in ('stiff', 'stiffforce'):  # the stiff layer left of the beam
        layer = (ix >= beam_x[0] - layers) & (ix < beam_x[0]) & (iy >= beam_y[0]) & (iy < beam_y[1])
        lam_q.s[..., layer] = stiff_factor * lam
        mu_q.s[..., layer] = stiff_factor * mu
    material = material_models.LinearElastic(discretization=disc, lam_1qxyz=lam_q, mu_1qxyz=mu_q,
                                             name='linear_isotropic_elasticity')

    constrained = disc.get_displacement_sized_field(name='constrained_dofs_field')
    if supports == 'clamp':  # both components fixed on CLAMP_COLUMNS node columns, over CLAMP_SPAN of the face height
        constrained.s[:, 0, clamped_nodes] = 1
    elif supports == 'simple':  # pin (u_x, u_y) at the lower left corner, roller (u_y) at the lower right corner
        clamped_nodes = ((ix == beam_x[0]) | (ix == beam_x[1])) & (iy == beam_y[0])
        constrained.s[0, 0, (ix == beam_x[0]) & (iy == beam_y[0])] = 1
        constrained.s[1, 0, clamped_nodes] = 1
    elif supports == 'stiff':  # bottom and top node of the outer node column of the stiff layer
        clamped_nodes = (ix == beam_x[0] - layers) & ((iy == beam_y[0]) | (iy == beam_y[1]))
        constrained.s[:, 0, clamped_nodes] = 1
    else:  # free, traction
        clamped_nodes = np.zeros_like(clamped_nodes)
    prescribed = disc.get_displacement_sized_field(name='prescribed_displacement')
    prescribed.s[...] = 0.0
    nodal_constraints = constraints.NodalConstraints.from_mask(disc, constrained_mask=constrained,
                                                               prescribed_values=prescribed)

    # shear force on a node column: weight 1 at inner nodes, 1/2 at the two corners, summing to n_H
    weights = np.where((iy == beam_y[0]) | (iy == beam_y[1]), 0.5, 1.0) / n_H
    force = disc.get_unknown_size_field(name='force_field')
    force.s[...] = 0

    def column(x_pixels):
        return (ix == beam_x[0] + x_pixels) & (iy >= beam_y[0]) & (iy <= beam_y[1])

    def consistent_nodal_forces(traction):
        """f_j = int t(eta) N_j(eta) d eta over the face, eta = y - H/2; 2-point Gauss per pixel is exact for the
        quadratic tractions used here, so the forces are in equilibrium to round-off"""
        eta = np.arange(n_H + 1) * h - beam_height / 2
        f = np.zeros(n_H + 1)
        for g in (-1 / np.sqrt(3), 1 / np.sqrt(3)):
            e = 0.5 * (eta[:-1] + eta[1:]) + 0.5 * h * g
            f[:-1] += 0.5 * h * traction(e) * (eta[1:] - e) / h
            f[1:] += 0.5 * h * traction(e) * (e - eta[:-1]) / h
        return f

    if supports == 'traction':
        second_moment = beam_height ** 3 / 12
        shear = consistent_nodal_forces(lambda e: 1.5 * total_force_y / beam_height * (1 - (2 * e / beam_height) ** 2))
        bending = consistent_nodal_forces(lambda e: beam_length * total_force_y * e / second_moment)
        left = column(0)
        force.s[1, 0, right_nodes] = shear[iy[right_nodes] - beam_y[0]]
        force.s[1, 0, left] = -shear[iy[left] - beam_y[0]]
        force.s[0, 0, left] = bending[iy[left] - beam_y[0]]
    else:
        if supports in ('simple', 'free'):  # load at mid-span (the measured deflection is there)
            right_nodes = column(n_L // 2)
        force.s[1, 0, right_nodes] = total_force_y * weights[right_nodes]
    if supports == 'stiffforce':  # the reactions of the tip load on the outer node column of the layer
        outer = column(-layers)
        arm = (n_L + layers) * h
        bending = consistent_nodal_forces(lambda e: arm * total_force_y * e / (beam_height ** 3 / 12))
        force.s[1, 0, outer] -= total_force_y * weights[outer]
        force.s[0, 0, outer] += bending[iy[outer] - beam_y[0]]
    if supports == 'free':  # balancing forces at both ends
        for end in (column(0), column(n_L)):
            force.s[1, 0, end] -= 0.5 * total_force_y * weights[end]

    def K_fun(x, Ax):
        disc.apply_system_matrix_mugrid_explicit_stress(constitutive=material.get_stress, input_field_inxyz=x,
                                                        output_field_inxyz=Ax, formulation='small_strain')
        disc.fft.communicate_ghosts(Ax)

    t_setup = time.time()  # setup of the preconditioner: Green operator (+ the correction for the fixed dofs)
    C_ref = material_models.get_elastic_tensor_from_lame(dim=2, lam=lam, mu=mu)
    green = disc.get_preconditioner_Green_mugrid(reference_material_data_ijkl=C_ref)

    def M_fun(x, Px):
        disc.fft.communicate_ghosts(x)
        disc.apply_preconditioner_mugrid(preconditioner_Fourier_fnfnqks=green, input_nodal_field_fnxyz=x,
                                         output_nodal_field_fnxyz=Px)

    P_fun = nodal_constraints.project_preconditioner(M_fun)
    if os.environ.get('PRECOND') in ('green_dirichlet', 'green_dirichlet_S'):  # _S: only S, no translations
        only_S = os.environ.get('PRECOND') == 'green_dirichlet_S'
        fixed = tuple(np.argwhere(constrained.s != 0).T)  # (component, sub-point, x, y) of every fixed dof
        nb_fixed = len(fixed[0])
        unit, Me = (disc.get_unknown_size_field(name=f'gd_{k}') for k in ('unit', 'Me'))
        A = np.zeros((nb_fixed + 2, nb_fixed + 2))
        for j in range(nb_fixed):  # E^T M E, one Green application per fixed dof
            unit.s[...] = 0
            unit.s[tuple(f[j] for f in fixed)] = 1.0
            M_fun(unit, Me)
            A[:nb_fixed, j] = Me.s[fixed]
        for d in range(2):  # E^T N and N^T E: which fixed dofs a translation in direction d moves
            A[:nb_fixed, nb_fixed + d] = A[nb_fixed + d, :nb_fixed] = (fixed[0] == d)
        A_inv = np.linalg.inv(A[:nb_fixed, :nb_fixed] if only_S else A)
        print(f'corrected Green: {nb_fixed} fixed dofs', flush=True)
        Mr, w = (disc.get_unknown_size_field(name=f'gd_{k}') for k in ('Mr', 'w'))

        def P_fun(r, z):
            M_fun(r, Mr)
            sol = A_inv @ (Mr.s[fixed] if only_S else
                           np.concatenate([Mr.s[fixed], [r.s[d].sum() for d in range(2)]]))
            w.s[...] = r.s
            w.s[fixed] -= sol[:nb_fixed]
            M_fun(w, z)
            for d in range(0 if only_S else 2):  # the symmetric system solves for -alpha
                z.s[d] -= sol[nb_fixed + d]
            z.s[fixed] = 0.0

    inner_its = []
    if os.environ.get('PRECOND') == 'green_dirichlet_inner':  # the same correction, S = E^T M E never assembled
        fixed = tuple(np.argwhere(constrained.s != 0).T)
        nb_fixed = len(fixed[0])
        inner_tol = float(os.environ.get('INNER_TOL', 1e-8))
        Ey, MEy = (disc.get_unknown_size_field(name=f'gdi_{k}') for k in ('Ey', 'MEy'))

        def S_fun(y):  # S y = E^T M E y, one Green application
            Ey.s[...] = 0
            Ey.s[fixed] = y
            M_fun(Ey, MEy)
            return MEy.s[fixed].copy()

        # INNER_PRECOND=circ: S preconditioned by its block-circulant approximation along the wall. The fixed dofs must
        # form one straight node column per component (a clamped face, CLAMP_COLUMNS=1); one Green application per
        # component at the middle wall node gives the kernel S_dd'(offset), its 1D FFT the 2x2 symbol per wave number
        inner_precond = os.environ.get('INNER_PRECOND', '')
        C_inv = lambda res: res
        if inner_precond == 'circ':
            wall = [np.flatnonzero(fixed[0] == d) for d in range(2)]
            wall = [idx[np.argsort(fixed[-1][idx])] for idx in wall]  # each component's wall dofs, ordered along y
            n_w = len(wall[0])
            assert len(wall[1]) == n_w and len(set(fixed[-2])) == 1, 'circ needs one straight clamped node column'
            mid = n_w // 2
            kernel = np.zeros((2, 2, n_w))  # kernel[d, e, k]: S row (component d, node mid + k) for a unit at (e, mid)
            for e in range(2):
                y = np.zeros(nb_fixed)
                y[wall[e][mid]] = 1.0
                Sy = S_fun(y)
                for d in range(2):
                    kernel[d, e] = np.roll(Sy[wall[d]], -mid)  # offset k = 0, 1, .., then the negative offsets
            symbol = np.fft.fft(kernel, axis=-1).transpose(2, 0, 1)  # (n_w, 2, 2)
            symbol = 0.5 * (symbol + np.conj(symbol.transpose(0, 2, 1)))
            ev, evec = np.linalg.eigh(symbol)
            ev = np.maximum(ev, 1e-12 * ev.max())  # keep it SPD
            symbol_inv = np.einsum('kij,kj,klj->kil', evec, 1.0 / ev, np.conj(evec))

            def C_inv(res):
                res_hat = np.fft.fft(np.stack([res[wall[d]] for d in range(2)]), axis=-1)  # (2, n_w)
                out = np.real(np.fft.ifft(np.einsum('kde,ek->dk', symbol_inv, res_hat), axis=-1))
                y = np.empty_like(res)
                for d in range(2):
                    y[wall[d]] = out[d]
                return y

        # INNER_DIRECT=1 (with INNER_PRECOND=circ and CLAMP_FULL=1): S is exactly block-circulant, lam = C^-1 g directly,
        # no inner iteration and no S product; the relative error of S C^-1 against the identity is printed once
        inner_direct = bool(os.environ.get('INNER_DIRECT'))
        if inner_precond == 'circ':
            g_test = np.random.default_rng(0).standard_normal(nb_fixed)
            circ_error = np.linalg.norm(S_fun(C_inv(g_test)) - g_test) / np.linalg.norm(g_test)
            print(f'circulant wall preconditioner: |S C^-1 g - g| / |g| = {circ_error:.2e}', flush=True)

        def S_solve(g, tol, y0=None, maxit=0):  # (P)CG on the fixed dofs, relative residual tol (or maxit steps)
            if inner_direct:
                inner_its.append(0)
                return C_inv(g)
            if y0 is None:
                y, res = np.zeros_like(g), g.copy()
            else:
                y = y0.copy()
                res = g - S_fun(y)
            zz = C_inv(res)
            p, rz, rr0 = zz.copy(), res @ zz, g @ g
            for it in range(1, (maxit or 10 * nb_fixed) + 1):
                Sp = S_fun(p)
                a = rz / (p @ Sp)
                y += a * p
                res -= a * Sp
                if not maxit and res @ res <= tol ** 2 * rr0:
                    break
                zz = C_inv(res)
                rz, rz_old = res @ zz, rz
                p = zz + rz / rz_old * p
            inner_its.append(it)
            return y

        B = np.stack([(fixed[0] == d).astype(float) for d in range(2)], axis=1)  # E^T N
        W = np.stack([S_solve(B[:, d], 1e-12) for d in range(2)], axis=1)  # S^-1 E^T N, once
        BW_inv = np.linalg.inv(B.T @ W)
        inner_its.clear()
        print(f'corrected Green, inner CG (tol {inner_tol:g}): {nb_fixed} fixed dofs', flush=True)
        Mr, w = (disc.get_unknown_size_field(name=f'gdi_{k}') for k in ('Mr', 'w'))
        previous_Sg = [None]

        def P_fun(r, z):  # [S B; B^T 0][lam; mu] = [E^T M r; N^T r], z = M (r - E lam) - N mu
            M_fun(r, Mr)
            Sg = S_solve(Mr.s[fixed], inner_tol, y0=previous_Sg[0] if inner_warm else None, maxit=inner_maxit)
            previous_Sg[0] = Sg
            mu_ = BW_inv @ (B.T @ Sg - np.array([r.s[d].sum() for d in range(2)]))
            lam_ = Sg - W @ mu_
            w.s[...] = r.s
            w.s[fixed] -= lam_
            M_fun(w, z)
            for d in range(2):
                z.s[d] -= mu_[d]
            z.s[fixed] = 0.0

    setup_time = time.time() - t_setup
    if os.environ.get('PRECOND') == 'green_dirichlet_reg':
        mass_c = float(os.environ.get('MASS_C', 1e-3)) * mu / max(number_of_pixels) ** 2 / h ** 2
        lam_0, mu_0 = (disc.get_quad_field_scalar(name=f'ref_{k}') for k in ('lam', 'mu'))
        lam_0.s[...], mu_0.s[...] = lam, mu
        reference = material_models.LinearElastic(discretization=disc, lam_1qxyz=lam_0, mu_1qxyz=mu_0,
                                                  name='reference_elasticity')
        u_q = disc.get_custom_size_quad_field(name='mass_u_q', shape=(2,))
        mass_n = disc.get_unknown_size_field(name='mass_n')

        def K_c(input_field_inxyz, output_field_inxyz):  # K(C_0) + c N^T W N
            disc.apply_system_matrix_mugrid_explicit_stress(constitutive=reference.get_stress,
                                                            input_field_inxyz=input_field_inxyz,
                                                            output_field_inxyz=output_field_inxyz,
                                                            formulation='small_strain')
            disc.apply_N_operator_mugrid(input_field_inxyz, u_q)
            u_q.s[...] *= mass_c
            disc.apply_N_transposed_operator_mugrid(quad_field_ijqxyz=u_q, nodal_field_inxyz=mass_n)
            output_field_inxyz.s[...] += mass_n.s

        green_c = disc.get_preconditioner_Green_mugrid(reference_material_data_ijkl=None, operator=K_c,
                                                       invert_zero_mode=True, name='green_regular')

        def G_c(x, Gx):
            disc.fft.communicate_ghosts(x)
            disc.apply_preconditioner_mugrid(preconditioner_Fourier_fnfnqks=green_c, input_nodal_field_fnxyz=x,
                                             output_nodal_field_fnxyz=Gx)

        fixed = tuple(np.argwhere(constrained.s != 0).T)
        nb_fixed = len(fixed[0])
        unit, Ge = (disc.get_unknown_size_field(name=f'gdr_{k}') for k in ('unit', 'Ge'))
        S = np.zeros((nb_fixed, nb_fixed))
        for j in range(nb_fixed):  # P G_c P, one Green application per fixed dof
            unit.s[...] = 0
            unit.s[tuple(f[j] for f in fixed)] = 1.0
            G_c(unit, Ge)
            S[:, j] = Ge.s[fixed]
        S_inv = np.linalg.inv(S)
        Gr, w = (disc.get_unknown_size_field(name=f'gdr_{k}') for k in ('Gr', 'w'))
        print(f'corrected Green without translations: {nb_fixed} fixed dofs, c = {mass_c:.3e}, '
              f'cond(S) = {np.linalg.cond(S):.2e}', flush=True)

        def P_fun(r, z):
            G_c(r, Gr)
            w.s[...] = r.s
            w.s[fixed] -= S_inv @ Gr.s[fixed]
            G_c(w, z)
            z.s[fixed] = 0.0

        setup_time = time.time() - t_setup

    rhs = disc.get_unknown_size_field(name='rhs_field')
    u = disc.get_unknown_size_field(name='solution')
    rhs.s[...] = force.s
    nodal_constraints.apply_to_rhs(rhs, K_fun)
    u.s[...] = 0
    K_free = nodal_constraints.project_operator(K_fun)

    dense_eigenvalues = None
    if os.environ.get('DENSE_SPECTRUM'):  # small meshes only: eigenvalues of M^-1 K column by column (checks Ritz values)
        M_free = nodal_constraints.project_preconditioner(M_fun)
        unit, Ke, MKe = (disc.get_unknown_size_field(name=f'dense_{k}') for k in ('unit', 'Ke', 'MKe'))
        nb_dofs = unit.s.size
        MK = np.zeros((nb_dofs, nb_dofs))
        for j in range(nb_dofs):
            unit.s[...] = 0
            unit.s[np.unravel_index(j, unit.s.shape)] = 1.0  # (ravel() of .s may be a copy)
            disc.fft.communicate_ghosts(unit)
            K_free(unit, Ke)
            M_free(Ke, MKe)
            MK[:, j] = MKe.s.ravel()
        dense_eigenvalues = np.sort(np.linalg.eigvals(MK).real)

    residual_rr = []  # (r_k, r_k) of every iteration
    energy_functional = []  # Pi(u_k) = 1/2 u_k.K u_k - b.u_k = -1/2 (b.u_k + r_k.u_k), as r_k = b - K u_k

    # CG coefficients for the Lanczos matrix (Ritz values of M^-1 K): the callback comes after the step
    # x_k = x_{k-1} + alpha_{k-1} p_{k-1} and before the direction update, so p_s is still p_{k-1}
    rz, cg_alpha, previous_x = [], [], [None]

    def record(iteration, x_s, r_s, p_s, z_s, rr):
        residual_rr.append(float(rr))
        energy_functional.append(-0.5 * float(np.dot(rhs.s.ravel(), x_s.ravel()) + np.dot(r_s.ravel(), x_s.ravel())))
        rz.append(float(np.dot(r_s.ravel(), z_s.ravel())))
        if previous_x[0] is not None:
            cg_alpha.append(float(np.dot((x_s - previous_x[0]).ravel(), p_s.ravel()) / np.dot(p_s.ravel(), p_s.ravel())))
        previous_x[0] = np.copy(x_s)

    t0 = time.time()
    norms = solvers.conjugate_gradients_mugrid_experimental(
        comm=disc.communicator, fc=disc.field_collection, hessp=K_free, b=rhs, x=u,
        P=P_fun, maxiter=int(os.environ.get('CG_MAXITER', 20000)), stop_crit_norm='energy_lower_estim',
        d_min=cg_delay, callback=record, **topology_optimization.energy_stop_kwargs(disc, K_free, u, cg_tol))[1]
    cg_its = norms['stop_iteration']['energy_lower_estim']
    cg_its = int(cg_its) if cg_its is not None else -(len(residual_rr) - 1)  # negative: maxiter, not converged
    wall = time.time() - t0
    nodal_constraints.impose_prescribed_values(u)

    compliance = float(np.sum(force.s * u.s))  # F . u = 2 * strain energy
    # tip deflection; without supports the energy-equivalent one, compliance / F (rigid motions are not fixed)
    tip = compliance / total_force_y if supports == 'traction' else float(np.mean(u.s[1, 0][right_nodes]))
    Ku = disc.get_unknown_size_field(name='Ku_field')
    K_fun(u, Ku)
    reaction_y = float(np.sum((Ku.s - force.s)[1, 0][clamped_nodes]))
    beam_nodes = slice(beam_x[0], beam_x[1] + 1), slice(beam_y[0], beam_y[1] + 1)
    return dict(setup_time=setup_time, n_H=n_H, h=h, pixels=number_of_pixels, nb_clamped_nodes=int(np.sum(clamped_nodes)), tip=tip,
                compliance=compliance, cg_its=cg_its, time=wall,
                reaction_y=reaction_y,
                energy_error_sq=np.array(norms['energy_lower_estim']),  # estimate of ||u - u_k||_K^2, entry k = it k
                residual_rr=np.array(residual_rr), energy_functional=np.array(energy_functional),
                cg_alpha=np.array(cg_alpha), cg_beta=np.array(rz[1:]) / np.array(rz[:-1]),
                inner_its=np.array(inner_its, dtype=int),
                **({} if dense_eigenvalues is None else {'dense_eigenvalues': dense_eigenvalues}), u_beam=np.array(u.s[:, 0][(slice(None),) + beam_nodes]))


# one result file per mesh in exp_data/runs/; a mesh is solved only if its file is missing (delete it to recompute)
RUNS = os.path.join(OUT, 'runs')
os.makedirs(RUNS, exist_ok=True)
results = []
for n_H in mesh_list:
    run_file = os.path.join(RUNS, f'n_H{n_H}_cgtol{cg_tol:g}{TAG}.npz')
    if os.path.exists(run_file):
        r = {k: v[()] for k, v in np.load(run_file).items()}
    elif os.environ.get('PLOT_ONLY'):  # figures of the stored meshes only, never solve
        continue
    else:
        r = solve(n_H)
        np.savez(run_file, **r)
    results.append(r)
    print(f'n_H = {n_H:3d}  h = {r["h"]:.5f}  grid {r["pixels"][0]:4d} x {r["pixels"][1]:3d}  '
          f'tip u_y = {r["tip"]:.8e}  compliance = {r["compliance"]:.8e}  R_y + F_y = '
          f'{r["reaction_y"] + total_force_y:+.1e}  CG its {r["cg_its"]:5d}  ({r["time"]:.1f} s)', flush=True)
if os.environ.get('SOLVE_ONLY'):  # worker of a parallel sweep: store the meshes, no figures
    sys.exit(0)
mesh_list = [int(r['n_H']) for r in results]  # the meshes actually available

h = np.array([r['h'] for r in results])
tip = np.array([r['tip'] for r in results])
compliance = np.array([r['compliance'] for r in results])
cg_its = np.array([r['cg_its'] for r in results])

# observed order from the three finest meshes (halving h), Richardson extrapolation from the two finest
order = np.log2((compliance[-3] - compliance[-2]) / (compliance[-2] - compliance[-1])) if len(h) >= 3 else 2.0
extrapolated = {k: v[-1] + (v[-1] - v[-2]) / (2 ** order - 1) for k, v in (('tip', tip), ('compliance', compliance))}
error_tip = np.abs(tip - extrapolated['tip']) / abs(extrapolated['tip'])
error_compliance = np.abs(compliance - extrapolated['compliance']) / abs(extrapolated['compliance'])

E_ps = E_young / (1 - poisson ** 2)
I = beam_height ** 3 / 12
tip_EB = total_force_y * beam_length ** 3 / (3 * E_ps * I)
tip_timoshenko = tip_EB + total_force_y * beam_length / (5 / 6 * mu * beam_height)
print(f'observed order of the compliance error: {order:.3f}')
print(f'extrapolated tip u_y = {extrapolated["tip"]:.8e}   Timoshenko {tip_timoshenko:.8e} '
      f'(ratio {extrapolated["tip"] / tip_timoshenko:.4f})   Euler-Bernoulli {tip_EB:.8e}')

np.savez(os.path.join(OUT, f'cantilever_mesh_study{TAG}.npz'), h=h, n_H=np.array(mesh_list), tip=tip,
         compliance=compliance, cg_its=cg_its, error_tip=error_tip, error_compliance=error_compliance, order=order,
         tip_extrapolated=extrapolated['tip'], tip_timoshenko=tip_timoshenko, tip_EB=tip_EB,
         beam_length=beam_length, beam_height=beam_height, E=E_young, poisson=poisson, force_y=total_force_y)
with open(os.path.join(OUT, f'cantilever_mesh_study{TAG}.csv'), 'w') as f:
    f.write('n_H,h,tip_uy,compliance,rel_error_tip,rel_error_compliance,cg_its\n')
    for k in range(len(h)):
        f.write(f'{mesh_list[k]},{h[k]:.6g},{tip[k]:.10e},{compliance[k]:.10e},{error_tip[k]:.4e},'
                f'{error_compliance[k]:.4e},{cg_its[k]}\n')

# convergence figure
fig, axes = plt.subplots(1, 3, figsize=(14, 4), layout='constrained')
axes[0].plot(mesh_list, tip / tip_timoshenko, 'o-', label='FE (pixels)')
axes[0].axhline(extrapolated['tip'] / tip_timoshenko, color='0.4', ls='--', label='Richardson extrapolation')
axes[0].axhline(1.0, color='C3', ls=':', label='Timoshenko beam')
axes[0].set_xscale('log', base=2)
axes[0].set_xlabel('pixels over the beam height $n_H$')
axes[0].set_ylabel('tip deflection / Timoshenko')
axes[0].legend(frameon=False, fontsize=8)
fine = slice(0, len(h) - 1)  # the finest mesh is (nearly) the reference itself
axes[1].loglog(h[fine], error_tip[fine], 'o-', label='tip deflection')
axes[1].loglog(h[fine], error_compliance[fine], 's-', label='compliance (energy)')
axes[1].loglog(h[fine], error_compliance[0] * (h[fine] / h[0]) ** 2, 'k:', lw=1, label='$\\propto h^2$')
axes[1].set_xscale('log', base=2)
axes[1].xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
axes[1].set_xlabel('pixel size h')
axes[1].set_ylabel('relative error vs extrapolation')
axes[1].set_title(f'observed order {order:.2f}', fontsize=10)
axes[1].legend(frameon=False, fontsize=8)
axes[2].semilogx(mesh_list, cg_its, 'o-', base=2)
axes[2].set_xlabel('pixels over the beam height $n_H$')
axes[2].set_ylabel('CG iterations (Green, energy stop)')
axes[2].set_ylim(bottom=0)
for ax in axes:
    ax.grid(True, color='0.9', lw=0.5)
fig.suptitle(f'cantilever L = {beam_length:g}, H = {beam_height:g} with void, clamped left, shear force right '
             f'(E = {E_young:g}, ν = {poisson:g}, plane strain)', fontsize=11)
fig.savefig(os.path.join(OUT, f'cantilever_mesh_study{TAG}_convergence.png'), dpi=150)

# deformed beams: coarsest, middle, finest, displacement magnified
scale = 0.25 * beam_length / abs(tip[-1])
picks = sorted({0, len(results) // 2, len(results) - 1})
fig, axes = plt.subplots(len(picks), 1, figsize=(8, 2.2 * len(picks)), layout='constrained', sharex=True)
for ax, k in zip(np.atleast_1d(axes), picks):
    r = results[k]
    U = r['u_beam']
    X = np.stack(np.meshgrid(np.arange(U.shape[1]), np.arange(U.shape[2]), indexing='ij')) * r['h'] + scale * U
    cells = [(i, j) for i in range(U.shape[1] - 1) for j in range(U.shape[2] - 1)]
    ax.add_collection(PolyCollection([[X[:, i, j], X[:, i + 1, j], X[:, i + 1, j + 1], X[:, i, j + 1]]
                                      for i, j in cells], facecolors='#efef9f', edgecolors='#7a7a5a', linewidths=0.3))
    ax.plot([0, 0], [0, beam_height], color='k', lw=3)
    ax.autoscale()
    ax.set_aspect('equal')
    ax.set_title(f'$n_H$ = {r["n_H"]}:  tip u_y = {r["tip"]:.5e}   (displacement × {scale:.0f})', fontsize=10)
np.atleast_1d(axes)[-1].set_xlabel('x')
fig.savefig(os.path.join(OUT, f'cantilever_mesh_study{TAG}_deformed.png'), dpi=150, bbox_inches='tight')
print(f'wrote {OUT}/cantilever_mesh_study*')

# set-up: the whole periodic cell of one coarse mesh with void, beam, clamped nodes and nodal forces
n_H = 4 if 4 in mesh_list else mesh_list[0]
hs = beam_height / n_H
n_L = int(round(beam_length / hs))
nx, ny = n_L + 2 * nb_void, n_H + 2 * nb_void
fig, ax = plt.subplots(figsize=(11, 3.4), layout='constrained')
pixels = [[(i * hs, j * hs), ((i + 1) * hs, j * hs), ((i + 1) * hs, (j + 1) * hs), (i * hs, (j + 1) * hs)]
          for i in range(nx) for j in range(ny)]
in_beam = [nb_void <= i < nb_void + n_L and nb_void <= j < nb_void + n_H for i in range(nx) for j in range(ny)]
ax.add_collection(PolyCollection(pixels, facecolors=['#efef9f' if b else 'white' for b in in_beam],
                                 edgecolors='0.6', linewidths=0.5))
x0, y0 = nb_void * hs, nb_void * hs
ys = y0 + np.arange(n_H + 1) * hs
ax.plot(np.full_like(ys, x0), ys, 's', color='k', ms=6, label='clamped nodes: $u_x = u_y = 0$')
w = np.where((np.arange(n_H + 1) == 0) | (np.arange(n_H + 1) == n_H), 0.5, 1.0) / n_H
for y, wk in zip(ys, w):
    ax.annotate('', xy=(x0 + n_L * hs, y - 2.0 * hs * wk * n_H / 2), xytext=(x0 + n_L * hs, y),
                arrowprops=dict(arrowstyle='->', color='C3', lw=1.6))
ax.plot([], [], color='C3', lw=1.6, label=f'nodal forces, total $F_y$ = {total_force_y:g} (½ at the corners)')
ax.add_patch(plt.Rectangle((0, 0), nx * hs, ny * hs, fill=False, ec='k', lw=1.2, ls='--'))
ax.text(0.5 * hs, (ny - 0.5) * hs, f'void (zero stiffness), {nb_void} pixels', va='center', fontsize=9, color='0.3')
ax.text(x0 + 0.5 * n_L * hs, y0 + 0.5 * n_H * hs, f'beam  L = {beam_length:g}, H = {beam_height:g},  '
        f'E = {E_young:g}, ν = {poisson:g}', ha='center', va='center', fontsize=10)
ax.set_aspect('equal')
ax.set_xlim(-hs, (nx + 1) * hs)
ax.set_ylim(-2.5 * hs, (ny + 0.5) * hs)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.legend(loc='lower center', ncol=2, frameon=False, fontsize=9, bbox_to_anchor=(0.5, -0.02))
ax.set_title(f'periodic cell (dashed) at $n_H$ = {n_H}: {nx} × {ny} pixels, h = {hs:g}; '
             f'study $n_H$ = {", ".join(map(str, mesh_list))}', fontsize=10)
fig.savefig(os.path.join(OUT, f'cantilever_mesh_study{TAG}_setup.png'), dpi=150, bbox_inches='tight')

# CG convergence per mesh: relative energy error ||u - u_k||_K / ||u||_K, exact from the energy functional,
# ||u - u_k||_K^2 = 2 (Pi(u_k) - Pi(u)), ||u||_K^2 = -2 Pi(u), with Pi(u) = Pi(u_N) of the last iterate (whose own
# error is below the tolerance) and its estimate used for stopping (dotted), and relative residual; dashed: the discretization error of the same mesh in the energy norm,
# ||u - u_h||_E / ||u||_E = sqrt((C - C_h) / C) for the compliance C (C from the extrapolation)
fig, axes = plt.subplots(1, 3, figsize=(20, 5.5), layout='constrained')
colours = plt.cm.viridis(np.linspace(0, 0.9, len(results)))
for r, c in zip(results, colours):
    Pi = r['energy_functional']
    e = np.sqrt(np.maximum(Pi[:-1] - Pi[-1], 0) / -Pi[-1])
    e[e < 1e-7] = np.nan  # Pi(u_k) - Pi(u) is at round-off below ~1e-14 |Pi|, i.e. a relative error ~1e-7
    axes[0].semilogy(np.arange(len(e)), e, '-', color=c, lw=1.5, label=f'$n_H$ = {r["n_H"]}  ({r["cg_its"]} its)')
    # the stopping estimate: Hestenes-Stiefel lower bound with adaptive delay, entry k = iteration k
    estimate = np.sqrt(r['energy_error_sq'] / r['compliance'])
    axes[0].semilogy(np.arange(len(estimate)), estimate, ':', color=c, lw=1.4)
    k = min(len(estimate), len(e))
    axes[2].semilogy(np.arange(k), estimate[:k] / e[:k], '-', color=c, lw=1.2)
    discretization_error = np.sqrt(max(extrapolated['compliance'] - r['compliance'], 0) / extrapolated['compliance'])
    if r is not results[-1]:  # the finest mesh is the reference itself
        axes[0].axhline(discretization_error, color=c, ls='--', lw=0.9)
    rr = r['residual_rr']
    axes[1].semilogy(np.arange(len(rr)), np.sqrt(rr / rr[0]), '-', color=c, lw=1.2)
axes[0].axhline(cg_tol, color='C3', ls='-.', lw=1.2, label=f'stopping tolerance {cg_tol:g}')
axes[0].plot([], [], '-', color='0.3', lw=1.5, label='exact error, from the energy Π')
axes[0].plot([], [], ':', color='0.3', lw=1.4, label=f'estimate (HS lower bound, delay ≥ {cg_delay})')
axes[0].plot([], [], '--', color='0.3', lw=0.9, label='discretization error of the mesh')
axes[0].set_ylabel(r'relative energy error  $\|u - u_k\|_K / \|u\|_K$')
axes[0].set_title('CG error: exact (solid), stopping estimate (dotted); discretization error of the mesh (dashed)',
                  fontsize=10)
axes[1].set_ylabel(r'relative residual  $\|r_k\| / \|r_0\|$')
axes[1].set_title('residual (not used for stopping)', fontsize=10)
axes[2].axhline(1.0, color='0.3', lw=0.8)
axes[2].set_ylabel('estimate / exact error')
axes[2].set_title('quality of the stopping estimate (1 = exact; it is a lower bound)', fontsize=10)
for ax in axes:
    ax.set_xlabel('CG iteration')
    ax.grid(True, color='0.9', lw=0.5)
axes[1].legend(*axes[0].get_legend_handles_labels(), frameon=False, fontsize=9, loc='lower right')
fig.suptitle('cantilever with void: CG with the Green preconditioner, energy-norm stopping '
             f'(delay {cg_delay})', fontsize=11)
fig.savefig(os.path.join(OUT, f'cantilever_mesh_study{TAG}_cg_convergence.png'), dpi=150)
