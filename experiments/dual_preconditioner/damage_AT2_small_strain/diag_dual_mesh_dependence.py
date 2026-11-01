"""
Why does the dual preconditioner need more CG iterations on finer grids once the crack grows? Single elastic solves on
a FROZEN damage field, so that only the grid changes:

    the nodal damage of the fixed-l run AT2_N512_dual_polydisperse_pores_incl10_l0.03125 (2D, nine stiff inclusions
    10x, l = 1/32) is sampled at the nodes of the coarser grids (512^2 -> 256, 128, 64: every 2nd, 4th, 8th node,
    exact on the shared nodes), in two states:
        peak    the last load step before the break (max d = 0.48, no crack yet)
        broken  the cell cut through (max d = 1)
    and the residual stiffness k of g(d) = (1 - d)^2 + k is varied (1e-4 as in the runs, 1e-3, 1e-2).

If the dual's iterations stop growing with the grid for large k, the contrast of the broken band is the cause; if they
grow for every k, the resolution of the band is.

Each solve: cold start, CG with the energy-error stop of the runs (rtol 1e-6, delay 10) -> iteration count; then a
plain PCG (local, Lanczos coefficients recorded) of the same length -> Ritz values of the preconditioned operator:
extreme eigenvalue estimates and condition number.

Capped dual ('dual_cap', DUAL_CAP env, default 1e-2): the inverse material of the dual is C_iso / max(s, cap) instead of
C_iso / s (an approximate dual; the system solved is unchanged).
Identity reference ('Green_I', 'dual_I'): C_ref = I_sym, no relation to the material; Green_I = G_I, dual_I =
G_I K(C^-1) G_I with C^-1 = C_iso^-1 / s (C_iso^-1 the inverse on the symmetric tensors). Subsets (env, comma separated): DIAG_STATES (peak,broken),
DIAG_GRIDS (64,128,256,512), DIAG_K (1e-4,1e-3,1e-2), DIAG_PRECONDS (dual,Green_Jacobi,Green, also dual_cap, Green_I, dual_I); a subset tags the output files with
DIAG_TAG.

usage:  python diag_dual_mesh_dependence.py        writes exp_data/diag_dual_mesh_dependence{DIAG_TAG}.csv and
                                                   figures/diag_dual_mesh_dependence{DIAG_TAG}.png
        DIAG_STATES=broken DIAG_PRECONDS=dual,dual_cap DIAG_TAG=_cap python diag_dual_mesh_dependence.py
"""
import csv
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

import numpy as np
import netCDF4
import matplotlib
import matplotlib.ticker

matplotlib.use('Agg')
import matplotlib.pyplot as plt

from muFFTTO import domain
from muFFTTO import microstructure_library
from muFFTTO import material_models
from muFFTTO import solvers
from muFFTTO import topology_optimization   # energy_stop_kwargs

HERE = os.path.dirname(os.path.realpath(__file__))
SOURCE = os.path.join(HERE, 'exp_data', 'AT2_N512_dual_polydisperse_pores_incl10_l0.03125.nc')
GRIDS = tuple(int(n) for n in os.environ.get('DIAG_GRIDS', '64,128,256,512').split(','))
K_RES = tuple(float(k) for k in os.environ.get('DIAG_K', '1e-4,1e-3,1e-2').split(','))
PRECONDS = tuple(os.environ.get('DIAG_PRECONDS', 'dual,Green_Jacobi,Green').split(','))
DUAL_CAP = float(os.environ.get('DUAL_CAP', 1e-2))
TAG = os.environ.get('DIAG_TAG', '')
cg_tol, cg_delay = 1e-6, 10
# POISSON (env): Poisson ratio of matrix and inclusions (default 0.2 as in the runs; the damage field stays frozen)
E_young, poisson, inclusion_stiffness = 1.0, float(os.environ.get('POISSON', 0.2)), 10.0

with netCDF4.Dataset(SOURCE) as ds:
    stress = np.asarray(ds['macro_stress'][:, 0, 0])
    loads = np.asarray(ds['load'][:])
    i_peak = int(stress.argmax())
    STATES = {'peak': (np.asarray(ds['damage'][i_peak, 0]), loads[i_peak]),
              'broken': (np.asarray(ds['damage'][-1, 0]), loads[-1])}
STATES = {name: STATES[name] for name in os.environ.get('DIAG_STATES', 'peak,broken').split(',')}
n_source = next(iter(STATES.values()))[0].shape[0]


def solve_all(n):
    """all (state, k, preconditioner) solves on the n^2 grid"""
    cell = domain.PeriodicUnitCell(domain_size=[1, 1], problem_type='elasticity')
    disc = domain.Discretization(cell=cell, nb_of_pixels_global=(n, n), discretization_type='finite_element',
                                 element_type='bilinear_rectangle')
    cell_d = domain.PeriodicUnitCell(domain_size=[1, 1], problem_type='conductivity')
    disc_d = domain.Discretization(cell=cell_d, nb_of_pixels_global=(n, n), discretization_type='finite_element',
                                   element_type='bilinear_rectangle')
    phase = microstructure_library.get_geometry(nb_voxels=disc.nb_of_pixels, microstructure_name='polydisperse_pores',
                                                coordinates=disc.fft.coords)
    stiffness = np.where(phase > 0, 1.0, inclusion_stiffness)
    K_0, G_0 = material_models.get_bulk_and_shear_modulus(E=E_young, poisson=poisson)
    lam, mu = material_models.get_lame_parameters_from_bulk_and_shear(K_0, G_0, dim=2)
    I2 = np.eye(2)
    I_sym = 0.5 * (np.einsum('ik,jl->ijkl', I2, I2) + np.einsum('il,jk->ijkl', I2, I2))
    C_iso = lam * np.einsum('ij,kl->ijkl', I2, I2) + 2.0 * mu * I_sym
    scale = disc.get_quad_field_scalar(name='scale')
    C_d = domain.ScaledMaterial([(scale, C_iso)])
    C_dual = domain.ScaledMaterial([(lambda: 1.0 / scale.s[0, 0], C_iso)])
    C_dual_cap = domain.ScaledMaterial([(lambda: 1.0 / np.maximum(scale.s[0, 0], DUAL_CAP), C_iso)])
    # inverse of C_iso on the symmetric tensors: (C_iso)^-1 = 1/(2 mu) I_sym - lam / (2 mu (2 lam + 2 mu)) I x I (2D)
    C_iso_inv = I_sym / (2.0 * mu) - lam / (2.0 * mu * (2.0 * lam + 2.0 * mu)) * np.einsum('ij,kl->ijkl', I2, I2)
    assert np.allclose(np.einsum('ijmn,mnkl->ijkl', C_iso, C_iso_inv), I_sym)
    C_dual_I = domain.ScaledMaterial([(lambda: 1.0 / scale.s[0, 0], C_iso_inv)])
    green = {'Green': disc.get_preconditioner_Green_mugrid(reference_material_data_ijkl=C_iso,
                                                           formulation='small_strain', name='green_iso'),
             'Green_Jacobi': disc.get_preconditioner_Green_mugrid(reference_material_data_ijkl=2.0 * mu * I_sym,
                                                                  formulation='small_strain', name='green_2mu')}
    green['dual'] = green['dual_cap'] = green['Green']
    green['Green_I'] = green['dual_I'] = disc.get_preconditioner_Green_mugrid(reference_material_data_ijkl=I_sym,
                                                                              formulation='small_strain', name='green_I')
    damage = disc_d.get_unknown_size_field(name='damage')
    damage_q = disc_d.get_quad_field_scalar(name='damage_q')
    macro = disc.get_gradient_size_field(name='macro')
    rhs = disc.get_unknown_size_field(name='rhs')
    u = disc.get_unknown_size_field(name='u')
    t1, t2 = disc.get_unknown_size_field(name='t1'), disc.get_unknown_size_field(name='t2')
    xs = disc.get_unknown_size_field(name='xs')
    r, z, p, q = (disc.get_unknown_size_field(name=f'lanczos_{c}') for c in 'rzpq')

    def K(x, Ax):
        disc.apply_system_matrix_mugrid(material_data_field=C_d, input_field_inxyz=x, output_field_inxyz=Ax,
                                        formulation='small_strain')

    state = {}

    def M(x, Px):
        P = state['P']
        if P in ('dual', 'dual_cap', 'dual_I'):
            disc.fft.communicate_ghosts(x)
            disc.apply_preconditioner_mugrid(preconditioner_Fourier_fnfnqks=green[P], input_nodal_field_fnxyz=x,
                                             output_nodal_field_fnxyz=t1)
            disc.apply_system_matrix_mugrid(material_data_field={'dual': C_dual, 'dual_cap': C_dual_cap, 'dual_I': C_dual_I}[P],
                                            input_field_inxyz=t1, output_field_inxyz=t2,
                                            formulation='small_strain')
            disc.fft.communicate_ghosts(t2)
            disc.apply_preconditioner_mugrid(preconditioner_Fourier_fnfnqks=green[P], input_nodal_field_fnxyz=t2,
                                             output_nodal_field_fnxyz=Px)
        elif P == 'Green_Jacobi':
            xs.s[...] = state['jacobi'].s * x.s
            disc.apply_preconditioner_mugrid(preconditioner_Fourier_fnfnqks=green[P], input_nodal_field_fnxyz=xs,
                                             output_nodal_field_fnxyz=Px)
            Px.s[...] *= state['jacobi'].s
        else:
            disc.apply_preconditioner_mugrid(preconditioner_Fourier_fnfnqks=green[P], input_nodal_field_fnxyz=x,
                                             output_nodal_field_fnxyz=Px)

    def ritz(nb_its):
        """plain PCG from u = 0 for nb_its iterations; Ritz values of M K from its Lanczos tridiagonal"""
        r.s[...] = rhs.s
        M(r, z)
        p.s[...] = z.s
        rz = float(np.vdot(r.s, z.s))
        alphas, betas = [], []
        for _ in range(nb_its):
            disc.fft.communicate_ghosts(p)
            K(p, q)
            alpha = rz / float(np.vdot(p.s, q.s))
            r.s[...] -= alpha * q.s
            M(r, z)
            rz_new = float(np.vdot(r.s, z.s))
            beta = rz_new / rz
            alphas.append(alpha), betas.append(beta)
            p.s[...] = z.s + beta * p.s
            rz = rz_new
        m = len(alphas)
        T = np.zeros((m, m))
        for j in range(m):
            T[j, j] = 1.0 / alphas[j] + (betas[j - 1] / alphas[j - 1] if j > 0 else 0.0)
            if j + 1 < m:
                T[j, j + 1] = T[j + 1, j] = np.sqrt(betas[j]) / alphas[j]
        return np.linalg.eigvalsh(T)

    rows = []
    step = n_source // n
    for name, (d_fine, load) in STATES.items():
        damage.s[0, 0] = d_fine[::step, ::step]
        disc_d.apply_N_operator_mugrid(damage, damage_q)
        disc.get_macro_gradient_field_mugrid(macro_gradient_ij=np.array([[load, 0.0], [0.0, 0.0]]),
                                             macro_gradient_field_ijqxyz=macro)
        for k in K_RES:
            g = (1.0 - damage_q.s[0, 0]) ** 2 + k
            scale.s[0, 0] = g * stiffness
            disc.get_rhs_mugrid(material_data_field_ijklqxyz=C_d, macro_gradient_field_ijqxyz=macro, rhs_inxyz=rhs)
            for P in PRECONDS:
                state['P'] = P
                if P == 'Green_Jacobi':
                    state['jacobi'] = disc.get_preconditioner_Jacobi_mugrid(material_data_field_ijklqxyz=C_d,
                                                                            formulation='small_strain', name='jac')
                u.s[...] = 0.0
                norms = solvers.conjugate_gradients_mugrid_experimental(
                    comm=disc.communicator, fc=disc.field_collection, hessp=K, b=rhs, x=u, P=M, maxiter=20000,
                    stop_crit_norm='energy_lower_estim', d_min=cg_delay,
                    **topology_optimization.energy_stop_kwargs(disc, K, u, cg_tol))[1]
                its = int(norms['stop_iteration']['energy_lower_estim'])
                ev = ritz(its)
                rows.append(dict(state=name, k=k, N=n, preconditioner=P, cg_its=its, lambda_min=ev[0],
                                 lambda_max=ev[-1], kappa=ev[-1] / ev[0], s_min=float(scale.s.min())))
                print(f'{name:6s} k {k:7.0e} N {n:4d} {P:13s} CG {its:5d}   Ritz [{ev[0]:.3e}, {ev[-1]:.3e}]  '
                      f'kappa {ev[-1] / ev[0]:.3e}', flush=True)
    return rows


rows = []
for n in GRIDS:
    rows += solve_all(n)
out_csv = os.path.join(HERE, 'exp_data', f'diag_dual_mesh_dependence{TAG}.csv')
with open(out_csv, 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader(), w.writerows(rows)

# ---------------------------------------------------------------------------------------------------------------------
COLORS = {'dual': 'C3', 'dual_cap': 'C1', 'dual_I': 'C5', 'Green_Jacobi': 'C2', 'Green': 'C0', 'Green_I': 'C9'}
MARKERS = {'dual': '^', 'dual_cap': 'v', 'dual_I': '<', 'Green_Jacobi': 's', 'Green': 'o', 'Green_I': 'D'}
LABELS = {'dual': 'dual', 'dual_cap': f'capped dual (s >= {DUAL_CAP:g})', 'Green_Jacobi': 'Green-Jacobi',
          'Green': 'Green', 'dual_I': 'dual, identity reference', 'Green_I': 'Green, identity reference'}
STYLES = {1e-4: ('-', 1.0), 1e-3: ('--', 0.75), 1e-2: (':', 0.5)}
fig, axes = plt.subplots(1, len(STATES), figsize=(max(7 * len(STATES), 10), 7), layout='constrained', dpi=150, squeeze=False)
axes = axes[0]
handles = []
for ax, name in zip(axes, STATES):
    for P in PRECONDS:
        for k in K_RES:
            sel = [r for r in rows if r['state'] == name and r['preconditioner'] == P and r['k'] == k]
            ls, a = STYLES[k]
            h, = ax.plot([r['N'] for r in sel], [r['cg_its'] for r in sel], ls, marker=MARKERS[P], ms=7, lw=2,
                         color=COLORS[P], alpha=a, mfc='white' if k != 1e-4 else COLORS[P],
                         label=f'{LABELS[P]}, k = {k:g}')
            if name == list(STATES)[0]:
                handles.append(h)
    d, load = STATES[name]
    ax.set_title(f'frozen damage, {"before the break" if name == "peak" else "cell cut through"}, $\\nu$ = {poisson:g} '
                 f'($\\bar\\varepsilon_{{xx}}$ = {load:.3f}, max d = {d.max():.2f})\none elastic solve, cold start',
                 fontsize=13)
    ax.set_xscale('log', base=2), ax.set_xticks(GRIDS), ax.set_xticklabels([f'{n}$^2$' for n in GRIDS])
    ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax.set_ylim(0, None), ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(6, integer=True))
    ax.set_xlabel('grid (fixed l = 1/32)', fontsize=12), ax.set_ylabel('CG iterations', fontsize=12)
    ax.tick_params(labelsize=12), ax.grid(True, color='0.9')
fig.legend(handles=handles, loc='outside lower center', ncol=len(K_RES) if len(STATES) > 1 else 2, fontsize=11)
out = os.path.join(HERE, 'figures', f'diag_dual_mesh_dependence{TAG}.png')
fig.savefig(out)
print(f'wrote {out_csv}\nwrote {out}')
