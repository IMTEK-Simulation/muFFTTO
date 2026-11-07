"""
Strain localisation in softening bands: implicit-gradient damage (Peerlings et al. 1996)
========================================================================================

Adapted from damage_AT2_small_strain/schneider2025_notch/example_2D_phase_field_fracture_AT2.py (same discretizations,
preconditioners, CG with the energy-error stop, staggered scheme, per-iteration CSV), with the AT2 phase field replaced
by the implicit-gradient enhanced damage model with exponential softening:

    sigma   = s(x) C_iso : eps,     s = stiffness(x) [ (1 - D) + k ]     (secant, linear-symmetric per iteration)
    eps     = E + sym(grad u~)      (periodic fluctuation u~, macroscopic strain E)
    eps_eq  = sqrt(2 psi^+(eps) / E_young)         energy equivalent strain, Amor split (tension + deviatoric/shear)
    ebar - c lap ebar = eps_eq,     c = l^2         nonlocal equivalent strain (Helmholtz, periodic)
    kappa   = max(kappa_n, ebar),   kappa >= kappa_0                     (irreversibility)
    D(kappa) = 1 - kappa_0/kappa [1 - alpha + alpha exp(-beta (kappa - kappa_0))]     (exponential softening)

D = 0 in the stiff (tough) inclusions. Staggered scheme per load step:
    1. elasticity  B^T W s C_iso B u~ = -B^T W s C_iso E   (PCG: Green / Green_Jacobi / dual)
    2. eps_eq at the quadrature points
    3. Helmholtz   [c B^T W B + N^T W N] ebar = N^T W eps_eq   (scalar 'conductivity' discretization; the operator is
       exactly the homogeneous reference, so its FFT Green operator (zero mode inverted) is the exact inverse: one FFT
       solve, checked at start-up)
    4. kappa, D, s;  repeat until max|D - D_old| < STAGGERED_TOL, at most MAX_STAGGERED iterations (then the step is
       ACCEPTED and logged as not converged)
"""
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

import numpy as np
import time

from muFFTTO import domain
from muFFTTO import microstructure_library
from muFFTTO import material_models
from muFFTTO import solvers
from muFFTTO import topology_optimization   # energy_stop_kwargs
from muFFTTO import tensor_operations

xp = np
# ----------------------------------------------------------------------------
# Parameters (env)
# ----------------------------------------------------------------------------
dim_problem = 2
domain_size = [1, 1]
N = int(os.environ.get('NPIX', 64))
number_of_pixels = (N, N)
element_type = 'bilinear_rectangle'
E_young, poisson = 1.0, 0.2
microstructure_name = os.environ.get('MICROSTRUCTURE', 'polydisperse_pores')
geometry_file = os.environ.get('GEOMETRY_FILE')          # .npy phase image (1 matrix, 0 second phase)
inclusion_stiffness = float(os.environ.get('INCLUSION_STIFFNESS', 10.0))   # second phase: tough, this stiffness
length_scale = float(os.environ.get('LENGTH_SCALE', 1.0 / 16))           # l (fixed across the grids), c = l^2
kappa_0 = float(os.environ.get('KAPPA0', 1e-3))
alpha = float(os.environ.get('ALPHA', 1.0))
beta = float(os.environ.get('BETA', 100.0))
k_residual = float(os.environ.get('K_RESIDUAL', 1e-3))
load_dir = os.environ.get('LOAD_DIR', 'xx')              # 'xx' uniaxial strain, 'yy', 'xy' simple shear (gamma)
macro_strain_direction = np.zeros((2, 2))
if load_dir == 'xy':
    macro_strain_direction[0, 1] = macro_strain_direction[1, 0] = 0.5   # load = engineering shear gamma
    ic = (0, 1)
else:
    ic = (0, 0) if load_dir == 'xx' else (1, 1)
    macro_strain_direction[ic] = 1.0
load_steps = np.linspace(0.0, float(os.environ.get('LOAD_MAX', 0.012)), int(os.environ.get('LOAD_STEPS', 120)) + 1)[1:]
staggered_tol = float(os.environ.get('STAGGERED_TOL', 1e-3))
max_staggered_iterations = int(os.environ.get('MAX_STAGGERED', 500))
stop_fraction = float(os.environ.get('STOP_FRACTION', 0.05))   # stop when sigma < STOP_FRACTION x peak, STOP_STEPS steps
stop_steps = int(os.environ.get('STOP_STEPS', 3))
cg_tol = 1e-6
cg_delay = 10
preconditioner_type = sys.argv[1] if len(sys.argv) > 1 else 'dual'
assert preconditioner_type in ('Green', 'Green_Jacobi', 'dual')
run_tag = f'_{load_dir}_l{length_scale:g}_b{beta:g}_k{k_residual:g}' + os.environ.get('RUN_SUFFIX', '')
here = os.path.dirname(os.path.abspath(__file__))
base_name = f'GD_N{N}_{preconditioner_type}{run_tag}'
iterations_file = os.path.join(here, 'exp_data', base_name + '_iterations.csv')
steps_file = os.path.join(here, 'exp_data', base_name + '_steps.csv')
fields_file = os.path.join(here, 'exp_data', base_name + '_fields.npz')

# ----------------------------------------------------------------------------
# Discretizations: vector (u~) and scalar (ebar)
# ----------------------------------------------------------------------------
cell_u = domain.PeriodicUnitCell(domain_size=domain_size, problem_type='elasticity')
disc_u = domain.Discretization(cell=cell_u, nb_of_pixels_global=number_of_pixels,
                               discretization_type='finite_element', element_type=element_type)
cell_e = domain.PeriodicUnitCell(domain_size=domain_size, problem_type='conductivity')
disc_e = domain.Discretization(cell=cell_e, nb_of_pixels_global=number_of_pixels,
                               discretization_type='finite_element', element_type=element_type)
comm = disc_u.communicator
dim = disc_u.domain_dimension
assert comm.size == 1, 'serial only'
print(f'grid {number_of_pixels}, l = {length_scale:.4f} (l/h = {length_scale * N:.2f}), kappa0 = {kappa_0}, '
      f'alpha = {alpha}, beta = {beta}, k = {k_residual}, load {load_dir} to {load_steps[-1]:g}, {preconditioner_type}')

# ----------------------------------------------------------------------------
# Microstructure: matrix (1, softens) and stiff tough inclusions (0)
# ----------------------------------------------------------------------------
phase = microstructure_library.get_geometry(nb_voxels=disc_u.nb_of_pixels, microstructure_name=microstructure_name,
                                            coordinates=disc_u.fft.coords)
if geometry_file is not None:
    fine = np.load(geometry_file)
    xy = np.asarray(disc_u.fft.coords)
    idx = [np.clip(np.floor(xy[a] * fine.shape[a] + 1e-9).astype(int), 0, fine.shape[a] - 1) for a in range(2)]
    phase = fine[tuple(idx)].astype(float)
phase = np.asarray(phase)
matrix = np.where(phase > 0, 1.0, 0.0)                         # per pixel
stiffness = np.where(phase > 0, 1.0, inclusion_stiffness)
K_0, G_0 = material_models.get_bulk_and_shear_modulus(E=E_young, poisson=poisson)
lam, mu = material_models.get_lame_parameters_from_bulk_and_shear(K_0, G_0, dim=dim)
bulk = lam + 2.0 * mu / dim
I2 = np.eye(dim)
C_iso = (lam * np.einsum('ij,kl->ijkl', I2, I2)
         + mu * (np.einsum('ik,jl->ijkl', I2, I2) + np.einsum('il,jk->ijkl', I2, I2)))
I_sym = 0.5 * (np.einsum('ik,jl->ijkl', I2, I2) + np.einsum('il,jk->ijkl', I2, I2))

degraded_scale = disc_u.get_quad_field_scalar(name='degraded_scale')   # s
C_d = domain.ScaledMaterial([(degraded_scale, C_iso)])
C_dual = domain.ScaledMaterial([(lambda: 1.0 / degraded_scale.s[0, 0], C_iso)])   # C_iso (s C_iso)^-1 C_iso

# ----------------------------------------------------------------------------
# Fields
# ----------------------------------------------------------------------------
macro_strain = disc_u.get_gradient_size_field(name='macro_strain')
strain = disc_u.get_gradient_size_field(name='strain')
rhs_u = disc_u.get_unknown_size_field(name='rhs_u')
u_fluct = disc_u.get_unknown_size_field(name='u_fluctuation')
dual_temp_1 = disc_u.get_unknown_size_field(name='dual_temp_1')
dual_temp_2 = disc_u.get_unknown_size_field(name='dual_temp_2')
x_scaled_u = disc_u.get_unknown_size_field(name='x_scaled_u')

ebar = disc_e.get_unknown_size_field(name='ebar')               # nonlocal equivalent strain, nodal
rhs_e = disc_e.get_unknown_size_field(name='rhs_e')
ebar_q = disc_e.get_quad_field_scalar(name='ebar_q')
eps_eq_q = disc_e.get_quad_field_scalar(name='eps_eq_q')
grad_e = disc_e.get_gradient_size_field(name='grad_ebar')
temp_q = disc_e.get_quad_field_scalar(name='temp_q')
temp_n = disc_e.get_unknown_size_field(name='temp_n')
check_n = disc_e.get_unknown_size_field(name='check_n')
nq = ebar_q.s.shape[1]
kappa_n = np.full(ebar_q.s[0, 0].shape, kappa_0)               # history of the last load step, per quad point
kappa = kappa_n.copy()
D = np.zeros_like(kappa)


def K_u(x, Ax):
    disc_u.apply_system_matrix_mugrid(material_data_field=C_d, input_field_inxyz=x, output_field_inxyz=Ax,
                                      formulation='small_strain')


green_u = disc_u.get_preconditioner_Green_mugrid(
    reference_material_data_ijkl=2.0 * mu * I_sym if preconditioner_type == 'Green_Jacobi' else C_iso,
    formulation='small_strain', name='green_u')
jacobi_u = None


def M_u(x, Px):
    if preconditioner_type == 'dual':
        disc_u.fft.communicate_ghosts(x)
        disc_u.apply_preconditioner_mugrid(preconditioner_Fourier_fnfnqks=green_u, input_nodal_field_fnxyz=x,
                                           output_nodal_field_fnxyz=dual_temp_1)
        disc_u.apply_system_matrix_mugrid(material_data_field=C_dual, input_field_inxyz=dual_temp_1,
                                          output_field_inxyz=dual_temp_2, formulation='small_strain')
        disc_u.fft.communicate_ghosts(dual_temp_2)
        disc_u.apply_preconditioner_mugrid(preconditioner_Fourier_fnfnqks=green_u, input_nodal_field_fnxyz=dual_temp_2,
                                           output_nodal_field_fnxyz=Px)
    elif preconditioner_type == 'Green_Jacobi':
        x_scaled_u.s[...] = jacobi_u.s * x.s
        disc_u.apply_preconditioner_mugrid(preconditioner_Fourier_fnfnqks=green_u, input_nodal_field_fnxyz=x_scaled_u,
                                           output_nodal_field_fnxyz=Px)
        Px.s[...] *= jacobi_u.s
    else:
        disc_u.apply_preconditioner_mugrid(preconditioner_Fourier_fnfnqks=green_u, input_nodal_field_fnxyz=x,
                                           output_nodal_field_fnxyz=Px)


def apply_helmholtz(x, Ax):
    """[c B^T W B + N^T W N] x,  c = l^2"""
    disc_e.apply_gradient_operator_mugrid(u_inxyz=x, grad_u_ijqxyz=grad_e)
    grad_e.s[...] *= length_scale ** 2
    disc_e.apply_gradient_transposed_operator_mugrid(gradient_field_ijqxyz=grad_e, div_u_fnxyz=Ax, apply_weights=True)
    disc_e.apply_N_operator_mugrid(x, temp_q)
    disc_e.apply_N_transposed_operator_mugrid(quad_field_ijqxyz=temp_q, nodal_field_inxyz=temp_n, apply_weights=True)
    Ax.s[...] += temp_n.s


green_e = disc_e.get_preconditioner_Green_mugrid(
    reference_material_data_ijkl=None, invert_zero_mode=True, name='green_helmholtz',
    operator=lambda input_field_inxyz, output_field_inxyz: apply_helmholtz(input_field_inxyz, output_field_inxyz))


def solve_helmholtz():
    temp_q.s[...] = eps_eq_q.s
    disc_e.apply_N_transposed_operator_mugrid(quad_field_ijqxyz=temp_q, nodal_field_inxyz=rhs_e, apply_weights=True)
    disc_e.apply_preconditioner_mugrid(preconditioner_Fourier_fnfnqks=green_e, input_nodal_field_fnxyz=rhs_e,
                                       output_nodal_field_fnxyz=ebar)


# start-up check: the FFT Green operator is the exact inverse of the Helmholtz operator
eps_eq_q.s[...] = np.random.default_rng(0).random(eps_eq_q.s.shape)
solve_helmholtz()
apply_helmholtz(ebar, check_n)
print(f'Helmholtz FFT solve: relative residual {np.linalg.norm(check_n.s - rhs_e.s) / np.linalg.norm(rhs_e.s):.2e}')
ebar.s[...] = 0.0


def damage_law(kap):
    Dk = 1.0 - kappa_0 / kap * (1.0 - alpha + alpha * np.exp(-beta * (kap - kappa_0)))
    return np.clip(Dk, 0.0, 1.0) * matrix                     # tough inclusions: D = 0


def update_material():
    degraded_scale.s[0, 0] = ((1.0 - D) + k_residual) * stiffness


def equivalent_strain():
    disc_u.apply_gradient_operator_symmetrized_mugrid(u_inxyz=u_fluct, grad_u_ijqxyz=strain)
    strain.s[...] += macro_strain.s
    eps = strain.s
    tr = np.einsum('ii...->...', eps)
    dev = eps - tr / dim * np.eye(dim).reshape(dim, dim, *([1] * tr.ndim))
    psi_plus = 0.5 * bulk * np.maximum(tr, 0.0) ** 2 + mu * np.einsum('ij...,ij...->...', dev, dev)
    return np.sqrt(2.0 * psi_plus / E_young)


# ----------------------------------------------------------------------------
# Load stepping
# ----------------------------------------------------------------------------
os.makedirs(os.path.join(here, 'exp_data'), exist_ok=True)
with open(iterations_file, 'w') as f:
    f.write('step,load,staggered_iteration,cg_u,time_u,time_e,damage_change,max_D,min_s_matrix\n')
with open(steps_file, 'w') as f:
    f.write('step,load,stress,max_D,min_s_matrix,band_fraction_1e-1,band_fraction_1e-2,staggered_its,converged,'
            'cg_u_step,mean_cg_u,time_u_step,time_step\n')
update_material()
start_time = time.time()
total_cg_u, nb_not_converged = 0, 0
macro_strains, macro_stresses = [0.0], [0.0]
D_frames, frame_loads = [], []
nb_broken = 0
area_matrix = matrix.sum()
for step, load in enumerate(load_steps):
    disc_u.get_macro_gradient_field_mugrid(macro_gradient_ij=load * macro_strain_direction,
                                           macro_gradient_field_ijqxyz=macro_strain)
    cg_u_step, time_u_step = 0, 0.0
    step_start = time.time()
    for stagger in range(max_staggered_iterations):
        t0 = time.time()
        disc_u.get_rhs_mugrid(material_data_field_ijklqxyz=C_d, macro_gradient_field_ijqxyz=macro_strain, rhs_inxyz=rhs_u)
        if preconditioner_type == 'Green_Jacobi':
            jacobi_u = disc_u.get_preconditioner_Jacobi_mugrid(material_data_field_ijklqxyz=C_d,
                                                               formulation='small_strain', name='jacobi_u')
        norms_u = solvers.conjugate_gradients_mugrid_experimental(
            comm=comm, fc=disc_u.field_collection, hessp=K_u, b=rhs_u, x=u_fluct, P=M_u, maxiter=20000,
            stop_crit_norm='energy_lower_estim', d_min=cg_delay,
            **topology_optimization.energy_stop_kwargs(disc_u, K_u, u_fluct, cg_tol))[1]
        cg_u = int(norms_u['stop_iteration']['energy_lower_estim'])
        time_u = time.time() - t0
        t0 = time.time()
        eps_eq_q.s[0, 0] = equivalent_strain()
        solve_helmholtz()
        disc_e.apply_N_operator_mugrid(ebar, ebar_q)
        kappa = np.maximum(kappa_n, ebar_q.s[0, 0])
        D_old = D
        D = damage_law(kappa)
        update_material()
        time_e = time.time() - t0
        change = float(np.max(np.abs(D - D_old)))
        s_matrix_min = float(np.min(degraded_scale.s[0, 0][:, matrix > 0]))
        cg_u_step += cg_u
        time_u_step += time_u
        total_cg_u += cg_u
        with open(iterations_file, 'a') as f:
            f.write(f'{step},{load:.6g},{stagger + 1},{cg_u},{time_u:.4g},{time_e:.4g},{change:.4g},'
                    f'{float(D.max()):.6g},{s_matrix_min:.4g}\n')
        if change < staggered_tol:
            break
    converged = change < staggered_tol
    nb_not_converged += not converged
    kappa_n = kappa
    sigma = disc_u.get_homogenized_stress_mugrid(material_data_field_ijklqxyz=C_d, displacement_field_inxyz=u_fluct,
                                                 macro_gradient_field_ijqxyz=macro_strain, formulation='small_strain')
    D_pix = D.mean(axis=0)
    s_pix = (1.0 - D_pix) + k_residual
    band1 = float(((s_pix < 1e-1) & (matrix > 0)).sum() / area_matrix)
    band2 = float(((s_pix < 1e-2) & (matrix > 0)).sum() / area_matrix)
    macro_strains.append(load)
    macro_stresses.append(float(sigma[ic]))
    D_frames.append(D_pix.astype(np.float32))
    frame_loads.append(load)
    with open(steps_file, 'a') as f:
        f.write(f'{step},{load:.6g},{sigma[ic]:.6g},{float(D.max()):.6g},{s_matrix_min:.4g},{band1:.4g},{band2:.4g},'
                f'{stagger + 1},{int(converged)},{cg_u_step},{cg_u_step / (stagger + 1):.2f},{time_u_step:.3f},'
                f'{time.time() - step_start:.3f}\n')
    print(f'step {step:3d}  load = {load:.5f}  sigma = {sigma[ic]:.5e}  max D = {float(D.max()):.4f}  '
          f'min s = {s_matrix_min:.2e}  band(s<0.1) = {band1:.3f}  staggered its = {stagger + 1:3d}  '
          f'step CG u = {cg_u_step}  mean CG u = {cg_u_step / (stagger + 1):.1f}  '
          f'time (u, step) = ({time_u_step:.2f}, {time.time() - step_start:.2f}) s'
          + ('' if converged else f'  ACCEPTED NOT CONVERGED after MAX_STAGGERED = {max_staggered_iterations} '
                                  f'(max |D - D_old| = {change:.2e})'), flush=True)
    nb_broken = nb_broken + 1 if sigma[ic] < stop_fraction * max(macro_stresses) else 0
    if stop_fraction > 0 and nb_broken >= stop_steps:
        print(f'stress < {stop_fraction:g} of the peak for {stop_steps} steps: stop')
        break

np.savez_compressed(fields_file, D=np.array(D_frames), load=np.array(frame_loads), phase=matrix,
                    stress=np.array(macro_stresses[1:]), length_scale=length_scale, k_residual=k_residual)
print(f'Elapsed time: {time.time() - start_time:.1f} s')
print(f'{preconditioner_type}: total CG iterations elasticity = {total_cg_u}')
if nb_not_converged:
    print(f'WARNING: {nb_not_converged} load steps accepted NOT CONVERGED (MAX_STAGGERED = {max_staggered_iterations})')
