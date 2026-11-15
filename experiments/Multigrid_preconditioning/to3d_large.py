"""
3D phase-field topology optimisation of a periodic unit cell on large grids (256^3 ... 1024^3), one MPI rank per GPU.

The problem is that of ``example_3D_elasticity_TO.py`` (see FABLE_BRIEF.md, section 2) and this script reproduces its
objective and gradient; it exists because that script keeps ~130 nodal vectors per voxel on the GPU. Here every field
is allocated once, named after its role, and shared:

    design         rho (nodal), its sensitivity, one nodal and two quadrature scalar scratch fields
    material       A(x) = s(x) C_0,  s = rho_q^p (1 - k) + k  at the quadrature points  (one scalar field)
    mechanics      u and lambda of the load case being solved, one right-hand side, 4 CG vectors,
                   two gradient-sized scratch fields; the multigrid (muFFTTO.multigrid) works on the same
                   discretization and adds 5 vectors on the fine grid
    warm starts    u and lambda of all load cases between evaluations: on the device (float64 or float32) or in
                   host memory (WARM_STORE), streamed to the device one load case at a time

Objective (per evaluation one mechanical and one adjoint solve per load case, warm started):

    f(rho) = sum_load cases  w sum_k c_k (E_k : (Sigma* - Sigma_h))^2 / W*_k^2            (OBJECTIVE=energy)
             + w_pf / (c_W V^(2/3)) [ eta int |grad rho|^2 + 1/eta int rho^2 (1 - rho)^2 ]

Sigma_h: homogenised stress of the load case, E_k: the "left" strains of its energy terms, W*_k = E_k : C* : E the
target energies of the isotropic auxetic target C* (TARGET_NU, TARGET_SCALE).  OBJECTIVE=stress matches the stresses,
|Sigma* - Sigma_h|^2 / |Sigma*|^2 per load case.

Environment (same names as the other TO scripts): NPIX, ETA, SOFT_EXP, ELEMENT, HG_EPS, DEVICE, OBJECTIVE, TARGET_NU,
TARGET_SCALE, PF_WEIGHT, STRESS_WEIGHT, CG_DELAY, MG_DEGREE, HELM_C, MAXITER, GTOL, INIT_FROM, HISTORY_EVERY, RUN_SUFFIX,
and new here:
    WARM_STORE     device (default) | device32 | host | host32   where the warm starts of the load cases live
    OPERATOR       stencil (default: the element ELEMENT through muFFTTO's stencils) | fused: trilinear hexahedra
                   with one material value per voxel (= trilinear_hexahedron with voxel-wise constant material)
                   through muGrid's fused IsotropicStiffnessOperator on every multigrid level; ELEMENT is ignored
    MG_PRECISION   double (default) | single: the multigrid cycle in float32 (OPERATOR=fused only); CG, the
                   solutions and the stopping test stay in double precision
    MG_COARSE_ELEMENT   element of the coarse multigrid levels (default: fully integrated hexahedra;
                   trilinear_hexahedron_1Q_hg: one-point elements on every level, see muFFTTO.multigrid;
                   MG_COARSE_HG_EPS: their hourglass control, default HG_EPS)
    SYMMETRY       none (default) | cubic | cubic_chiral: the design keeps the 48 symmetries of the cube about the
                   cell centre (cubic), or only its 24 rotations (cubic_chiral: no mirror planes); start design
                   and gradient are projected. The homogenised stiffness is then cubic, and two load cases (xx
                   and xy) give the whole objective: 3x less work and state per evaluation
    DESIGN_OPT     lbfgs_helm (default; NuMPI L-BFGS with the Helmholtz H_0, MPI-capable) | lbfgs_helm_local
                   (design_optimizers.py, serial; the optimizer of the baseline runs)
    OUT_DIR        output folder (default data/to3d_large)
    MAX_STEP       > 0: a trial design of the line search whose MEAN change from the last accepted design exceeds
                   this is rejected without solving (the line search halves the step). Guards against a wild
                   L-BFGS direction on the plateau, whose trial design costs a very long CG solve. 0 (default): off.
    STEP_GROWTH    > 0: the same, relative: a trial design is rejected if its mean change exceeds STEP_GROWTH x the
                   largest mean change of the last 10 accepted steps (needs 2 accepted steps). Observed at 512^3 on
                   the plateau after a restart: accepted steps 2e-5, then a trial of 1.3e-1 whose solves took
                   150-300 CG iterations instead of 10 (2026-11-15). 0 (default): off.
                   Every evaluation prints the max and mean change of its trial design ("TRIAL design").
    CHECKPOINT_EVERY   write <output>_checkpoint.nc (the current design) every n iterations. A run that finds its
                   own checkpoint continues from it (iteration count carried on; the L-BFGS memory and the warm
                   starts are rebuilt); <output>_final.nc is written when the optimizer stops (converged / MAXITER)
    TIME_BUDGET    seconds; after this wall time the run writes a checkpoint and ends cleanly (exit 0, no final file)
                   so that the next job of a chain continues it (set it below the job's time limit)
    ENERGY_TOL     stop when the homogenised energies of the objective have settled: every W_k changed by less
                   than ENERGY_TOL |W*_k| over the last FTOL_WINDOW (default 10) iterations; 0 (default) = off.
                   The stiffness is then final although the interface penalty still reshapes the design
                   (2D prototype stop_criterion_2D.py: 1e-3 after ~20 % of a full run with nu within 0.005 of the
                   final value -- enough before handing the design to a finer grid; 3e-4 after ~40-50 % with the
                   design itself within 1-3 % of the final one). OBJECTIVE=stress: the stress components instead
    FTOL           stop when the objective decreased by less than FTOL (relative) over the last FTOL_WINDOW
                   (default 10) iterations; 0 (default) = off. The design then no longer changes in a way that
                   matters for the objective, long before the step test (largest change of rho < 1e-3) fires
    MISFIT_TOL     stop when the stiffness part of the objective (the summed squared relative misfits of the
                   target energies or stresses) is below this value; 0 (default) = off
    HELMHOLTZ      fft | cg: how the Helmholtz initial inverse Hessian of L-BFGS is applied (default fft on one
                   rank, cg under MPI)
    MEM_REPORT=1   print the memory table (fable_memory.py) at the end
    PROFILE=1      print a cProfile table of rank 0 (functions by own time) after the optimisation

    python to3d_large.py            (serial)          mpirun -n 4 python to3d_large.py       (one rank per GPU)
"""
import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from mpi4py import MPI
from muGrid import linalg

from muFFTTO import domain, io_utils, material_models, multigrid, solvers_nonlinear, tensor_operations
import cubic_symmetry
from cg_energy import conjugate_gradients_energy

to_host = tensor_operations.to_host
C_W = 1.0 / 3.0                                   # 2 int_0^1 sqrt(rho^2 (1 - rho)^2) d rho
LOAD_CASE_STRAINS = [(0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1)]          # xx, yy, zz, yz, xz, xy


def unit_strain(i, j):
    E = np.zeros((3, 3))
    E[i, j] = E[j, i] = 1.0 if i == j else 0.5
    return E


class Converged(Exception):
    """raised by the iteration callback when a stopping test on the objective is met (FTOL, MISFIT_TOL)"""


class TimeBudgetReached(Exception):
    """raised by the iteration callback when TIME_BUDGET is used up (after the checkpoint is written)"""


class WarmStarts:
    """The solutions of all load cases between two evaluations, kept outside the solver's fields.

    ``store`` / ``load`` copy a whole field buffer (ghost layers included). Modes: 'device' (float64 on the device:
    exact), 'device32' (float32 on the device: half the memory; a warm start needs no more than single precision),
    'host' / 'host32' (host memory: nothing on the device). Every buffer is allocated once and reused; host buffers
    of a GPU run are page-locked ("pinned") if the system allows it -- transfers at several GB/s instead of the
    page-fault-bound speed of fresh pageable arrays (a 512^3 field is 3.2 GB, and an evaluation moves 48 of them).
    """

    def __init__(self, mode):
        self.on_host = mode.startswith('host')
        self.dtype = np.dtype(np.float32 if mode.endswith('32') else np.float64)
        self.buffers = {}
        self.seconds = 0.0                         # time spent in store / load

    @staticmethod
    def _buffer(field):
        """the whole buffer of the field: on the GPU as a contiguous view (direct device <-> host copies)"""
        if tensor_operations.get_array_module(field.sg) is np:
            return field.sg
        return tensor_operations.point_view(field)

    def _host_buffer(self, key, shape, pinned):
        if key not in self.buffers:
            if pinned:
                try:
                    import cupyx
                    self.buffers[key] = cupyx.empty_pinned(shape, dtype=self.dtype)
                except Exception:                  # no page-locked memory available: pageable, still reused
                    self.buffers[key] = np.empty(shape, dtype=self.dtype)
            else:
                self.buffers[key] = np.empty(shape, dtype=self.dtype)
        return self.buffers[key]

    def store(self, key, field):
        start = time.time()
        data = self._buffer(field)
        on_gpu = data is not field.sg
        if self.on_host and on_gpu:
            source = data if data.dtype == self.dtype else data.astype(self.dtype)
            source.get(out=self._host_buffer(key, data.shape, pinned=True))
        elif self.on_host:
            self._host_buffer(key, data.shape, pinned=False)[...] = data
        elif key in self.buffers:
            self.buffers[key][...] = data
        else:
            self.buffers[key] = data.astype(self.dtype, copy=True)
        self.seconds += time.time() - start

    def load(self, key, field):
        start = time.time()
        data = self._buffer(field)
        if key not in self.buffers:
            data[...] = 0.0
        elif self.on_host and data is not field.sg:
            if data.dtype == self.dtype:
                data.set(self.buffers[key])
            else:
                data[...] = tensor_operations.get_array_module(data).asarray(self.buffers[key])
        else:
            data[...] = self.buffers[key]
        self.seconds += time.time() - start


class TopologyOptimization:
    """Objective and gradient of the 3D stiffness-target phase-field TO; all fields on one discretization."""

    def __init__(self, n_pix, eta, element='trilinear_hexahedron_1Q_hg', hourglass_eps=0.3, device='cpu',
                 soft_phase=1e-3, p=2, objective='energy', target_nu=-0.5, target_scale=1.0, pf_weight=0.2,
                 stress_weight=1.0, cg_tol=1e-6, cg_delay=0, mg_degree=2, warm_store='device', load_cases=None,
                 symmetry='none', mg_coarse_element=None, mg_coarse_hourglass_eps=None, operator='stencil',
                 mg_precision='double', verbose=True):
        self.n_pix, self.eta, self.p, self.soft_phase = n_pix, eta, p, soft_phase
        if operator == 'fused':               # trilinear hexahedra, one material value per voxel, muGrid's fused
            element = 'trilinear_hexahedron_1Q'    # kernel; the discretization provides grid, fields, design operators
        self.objective, self.cg_tol, self.cg_delay, self.verbose = objective, cg_tol, cg_delay, verbose
        cell = domain.PeriodicUnitCell(domain_size=[1, 1, 1], problem_type='elasticity')
        self.disc = disc = domain.Discretization(
            cell=cell, nb_of_pixels_global=(n_pix,) * 3, discretization_type='finite_element', element_type=element,
            hourglass_eps=hourglass_eps if element.endswith('_hg') else None, device=device)
        self.comm = disc.communicator
        self.rank0 = self.comm.rank == 0
        self.volume = float(cell.domain_volume)
        self.nb_nodes = int(np.prod(disc.nb_of_pixels_global))
        self.weights = np.asarray(to_host(disc.quadrature_weights), dtype=float)        # [q]

        # ---- material: solid C_0, void k C_0; target: isotropic, Poisson ratio target_nu, shear target_scale * 0.15 E_0
        K_0, G_0 = 1.0, 0.5
        self.C_0 = material_models.get_elastic_material_tensor(dim=3, K=K_0, mu=G_0, kind='linear')
        E_0 = 9 * K_0 * G_0 / (3 * K_0 + G_0)
        G_target = target_scale * (3 / 20) * E_0
        K_target, G_target = material_models.get_bulk_and_shear_modulus(E=2 * G_target * (1 + target_nu),
                                                                       poisson=target_nu)
        self.C_target = material_models.get_elastic_material_tensor(dim=3, K=K_target, mu=G_target, kind='linear')

        # ---- load cases: (strain E, [(left strain E_k, target energy W*_k, weight c_k), ...])
        self.w = stress_weight
        self.pf_scale = pf_weight / (C_W * self.volume ** (2 / 3))
        strains = [unit_strain(i, j) for i, j in LOAD_CASE_STRAINS]
        self.symmetry = symmetry
        self._slab_remap = None
        assert symmetry in ('none', 'cubic', 'cubic_chiral'), symmetry
        if load_cases is None and symmetry != 'none':
            # cubic C_h: C1111 = C2222 = C3333, C1122 = C1133 = C2233, C2323 = C1313 = C1212, so the 9 energy terms
            # below are 3 x (xx; xx), 3 x (xx; yy or zz) and 3 x (xy; xy): two load cases, with these weights
            load_cases = [(0, [(0, 3.0), (1, 1.5), (2, 1.5)]), (5, [(5, 3.0)])]
            assert objective == 'energy', 'SYMMETRY is implemented for OBJECTIVE=energy'
        if load_cases is None:                    # the 9 energies C1111 C1122 C1133 | C2222 C2233 | C3333 | 3 shears
            load_cases = [(lc, [(k, 1.0) for k in lefts])
                          for lc, lefts in enumerate([[0, 1, 2], [1, 2], [2], [3], [4], [5]])]
        self.load_cases = [
            (lc, strains[lc], [(strains[k], float(np.einsum('ij,ijkl,kl->', strains[k], self.C_target, strains[lc])),
                                c) for k, c in terms])
            for lc, terms in load_cases]
        self.target_stresses = np.array([np.einsum('ijkl,kl->ij', self.C_target, E) for E in strains])
        self.homogenized_stresses = np.zeros((6, 3, 3))

        # ---- fields (see the module docstring)
        self.mg = multigrid.Multigrid(disc, self.C_0, degree=mg_degree, coarse_element=mg_coarse_element,
                                      coarse_hourglass_eps=mg_coarse_hourglass_eps, operator=operator,
                                      precision=mg_precision)
        self.fused = self.mg.fine_operator if operator == 'fused' else None       # its kernel and Lame fields
        self.scale = self.mg.scale                                          # s at the quadrature points
        self.material = self.mg.material                                    # s C_0
        self.rho = disc.get_scalar_field(name='design')
        self.sensitivity = disc.get_scalar_field(name='design_sensitivity')
        self.nodal_scratch = disc.get_scalar_field(name='nodal_scalar_scratch')
        self.quad_sensitivity = disc.get_quad_field_scalar(name='quad_sensitivity')
        self.quad_scratch = disc.get_quad_field_scalar(name='quad_scalar_scratch')
        self.rho_gradient = disc.get_gradient_of_scalar_field(name='design_gradient_scratch')
        self.u = disc.get_unknown_size_field(name='displacement')
        self.adjoint = disc.get_unknown_size_field(name='adjoint')
        self.rhs = disc.get_unknown_size_field(name='right_hand_side')
        self.cg_work = tuple(disc.get_unknown_size_field(name=f'cg_{name}')
                             for name in ('residual', 'preconditioned_residual', 'direction', 'product'))
        if self.fused is None:
            self.grad_u = disc.get_gradient_size_field(name='grad_field_temporary')  # the library's shared scratch
            self.grad_adjoint = disc.get_gradient_size_field(name='grad_field_temporary_2')
        else:                                 # per voxel: int eps(u) : eps(lambda) and int tr eps(u) tr eps(lambda)
            self.g_shear = disc.field_collection.real_field('sensitivity_shear')
            self.g_volumetric = disc.field_collection.real_field('sensitivity_volumetric')
        self.xp = tensor_operations.get_array_module(self.rho.s)
        self.warm = WarmStarts(warm_store)
        self.hourglass = disc.hourglass_control
        if self.hourglass is not None:                                      # eps kappa_{i alpha} of C_0  [3, 4]
            self.hourglass_kappa = self.hourglass.eps * to_host(
                self.hourglass._kappa_of_tensor(np.asarray(self.C_0), np))
            self.hourglass_mode = disc.field_collection.real_field(
                name='hourglass_strain_of_one_component', components=(1, 4), sub_pt='quad_points')
        self.cg_iterations = {'mechanical': [], 'adjoint': []}
        self.timing = {'setup': 0.0, 'solve': 0.0, 'sensitivity': 0.0, 'symmetry': 0.0}
        self.nb_evaluations = 0

    # ------------------------------------------------------------------ small helpers
    def global_sum(self, array):
        return self.comm.sum(np.float64(float(self.xp.sum(array))))

    def K(self, x, Kx):
        """Kx = K x: the fine-level operator of the multigrid, in double precision (ghosts of Kx not set)"""
        self.mg.apply_K(x, Kx)

    def assemble_force(self, strain, force, factor):
        """force = factor * B^T W s C_0 : strain   for a uniform strain tensor [3, 3]"""
        if self.fused is not None:
            self.fused.operator.apply_macro_rhs(self.fused.lam, self.fused.mu, [float(e) for e in strain.ravel()],
                                                force)
            linalg.scal(factor, force)
            return
        stress = np.einsum('ijkl,kl->ij', self.C_0, strain)
        s = self.scale.s[0, 0]
        for i in range(3):
            for j in range(3):
                self.xp.multiply(s, factor * stress[i, j], out=self.grad_u.s[i, j])
        self.disc.apply_gradient_transposed_operator_mugrid(gradient_field_ijqxyz=self.grad_u, div_u_fnxyz=force,
                                                            apply_weights=True)

    def scaled_mean(self, gradient):
        """sum_q,x w_q s (grad)_ij  -> [3, 3] (global)"""
        s = self.scale.s[0, 0]
        w = self.xp.asarray(self.weights).reshape(-1, 1, 1, 1)
        local = [[float(self.xp.sum(w * s * gradient.s[i, j])) for j in range(3)] for i in range(3)]
        return np.array([[self.comm.sum(np.float64(value)) for value in row] for row in local])

    def homogenized_stress(self, u, E):
        """(1 / V) int s C_0 : (E + grad u)   (C_0 is minor symmetric: no symmetrisation needed)"""
        if self.fused is not None:
            self.disc.fft.communicate_ghosts(u)
            local = self.fused.operator.average_stress(u, self.fused.lam, self.fused.mu, [float(e) for e in E.ravel()])
            return np.array([self.comm.sum(np.float64(value)) for value in local]).reshape(3, 3) / self.volume
        self.disc.apply_gradient_operator_mugrid(u_inxyz=u, grad_u_ijqxyz=self.grad_u)
        self.grad_u.s[...] += self.xp.asarray(E).reshape(3, 3, 1, 1, 1, 1)
        return np.einsum('ijkl,kl->ij', self.C_0, self.scaled_mean(self.grad_u)) / self.volume

    def residual_term(self, u, adjoint, E):
        """lambda . (K u + B^T W s C_0 : E): the residual of the mechanical problem tested with the adjoint -- zero
        for an exact solve, part of the objective as in the reference script (keeps f consistent with its gradient)"""
        product = self.cg_work[3]
        self.K(u, product)
        value = self.comm.sum(np.float64(float(linalg.vecdot(adjoint, product))))
        if self.fused is not None:
            self.assemble_force(E, product, 1.0)
            return value + self.comm.sum(np.float64(float(linalg.vecdot(adjoint, product))))
        self.disc.apply_gradient_operator_mugrid(u_inxyz=adjoint, grad_u_ijqxyz=self.grad_adjoint)
        C_E = np.einsum('ijkl,kl->ij', self.C_0, E)
        return value + float(np.sum(C_E * self.scaled_mean(self.grad_adjoint)))

    def add_sensitivity(self, u, adjoint, E, costate_strain):
        """quad_sensitivity += (grad lambda + costate_strain) : C_0 : (E + grad u)   per quadrature point; the
        chain rule through s(rho) follows once for all load cases (evaluate)"""
        if self.fused is not None:            # the kernel integrates over the voxel: divide by its volume
            operator = self.fused.operator
            self.disc.fft.communicate_ghosts(u)
            self.disc.fft.communicate_ghosts(adjoint)
            operator.compute_sensitivity(u, [float(e) for e in E.ravel()], adjoint,
                                         [float(e) for e in costate_strain.ravel()], self.g_shear, self.g_volumetric)
            target = self.quad_sensitivity.s[0, 0, 0]
            target += (2.0 * self.fused.mu_0 / self.weights[0]) * self.g_shear.p.reshape(target.shape)
            target += (self.fused.lam_0 / self.weights[0]) * self.g_volumetric.p.reshape(target.shape)
            return
        self.disc.apply_gradient_operator_mugrid(u_inxyz=u, grad_u_ijqxyz=self.grad_u)
        self.grad_u.s[...] += self.xp.asarray(E).reshape(3, 3, 1, 1, 1, 1)
        self.disc.apply_gradient_operator_mugrid(u_inxyz=adjoint, grad_u_ijqxyz=self.grad_adjoint)
        self.grad_adjoint.s[...] += self.xp.asarray(costate_strain).reshape(3, 3, 1, 1, 1, 1)
        self.add_contraction(self.grad_adjoint, self.grad_u, self.quad_sensitivity)
        if self.hourglass is not None:
            self.add_hourglass_sensitivity(u, adjoint, self.quad_sensitivity)

    # ------------------------------------------------------------------ cubic symmetry
    def symmetrize(self, field):
        """Project a nodal scalar field onto the fields with cubic symmetry (cubic_symmetry.py). Serial: on the
        device. MPI: R_x, R_y and the x-y swap inside the slabs, the three operations that move z between them."""
        chiral = self.symmetry == 'cubic_chiral'
        if self.comm.size == 1:
            mean = cubic_symmetry.chiral_mean if chiral else cubic_symmetry.cubic_mean
            field.s[0, 0] = mean(self.xp, field.s[0, 0])
            return
        if self._slab_remap is None:
            fft = self.disc.fft
            assert tuple(fft.nb_subdomain_grid_pts[:2]) == (self.n_pix, self.n_pix), 'slabs along z expected'
            self._slab_remap = cubic_symmetry.SlabRemap(self.comm.mpi4py_comm, self.n_pix,
                                                        fft.subdomain_locations[2], fft.nb_subdomain_grid_pts[2])
        mean = cubic_symmetry.chiral_mean_distributed if chiral else cubic_symmetry.cubic_mean_distributed
        field.s[0, 0] = self.xp.asarray(mean(self._slab_remap, to_host(field.s[0, 0])))

    def complete_stresses_by_symmetry(self):
        """homogenised stresses of the load cases that are not solved (cubic symmetry): permutations of xx and xy"""
        stress = self.homogenized_stresses
        for lc, permutation in ((1, [1, 0, 2]), (2, [2, 1, 0])):              # yy, zz from xx
            stress[lc] = stress[0][np.ix_(permutation, permutation)]
        for lc, permutation in ((3, [2, 0, 1]), (4, [0, 2, 1])):              # yz, xz from xy
            stress[lc] = stress[5][np.ix_(permutation, permutation)]

    # ------------------------------------------------------------------ objective and gradient
    def evaluate(self, design):
        """f and df/drho for the local nodal values `design` (host array, any shape)."""
        xp, disc = self.xp, self.disc
        t_start = time.time()
        self.rho.s[0, 0] = xp.asarray(design).reshape(self.rho.s.shape[2:])

        # material scale s = rho_q^p (1 - k) + k and the multigrid for it
        disc.apply_N_operator_mugrid(self.rho, self.scale)
        s = self.scale.s
        xp.power(s, self.p, out=s)
        s *= 1.0 - self.soft_phase
        s += self.soft_phase
        t_material = time.time()
        self.mg.setup()
        self.timing['setup'] += time.time() - t_start
        if self.verbose and (self.rank0 or os.environ.get('TIMING_ALL_RANKS', '0') == '1'):
            print(f'setup (rank {self.comm.rank}): material {t_material - t_start:.1f} s, multigrid '
                  + ', '.join(f'{stage} {seconds:.1f} s' for stage, seconds in self.mg.setup_seconds.items()), flush=True)

        f = self.phase_field_term()                       # writes self.sensitivity
        self.interface_term = f
        self.stiffness_measures = []          # homogenised energies (stresses) relative to their targets, this design
        self.quad_sensitivity.s[...] = 0.0
        for load_case in self.load_cases:
            f += self.load_case_term(*load_case)
        self.misfit = f - self.interface_term             # the stiffness part of the objective (all load cases)

        # chain rule through s(rho_q) and the interpolation: N^T W [ ds/drho_q * (accumulated contractions) ]
        t_sens = time.time()
        disc.apply_N_operator_mugrid(self.rho, self.quad_scratch)                     # rho_q
        dscale = self.quad_scratch.s
        xp.power(dscale, self.p - 1, out=dscale)
        dscale *= self.p * (1.0 - self.soft_phase)
        self.quad_sensitivity.s[...] *= dscale
        disc.apply_N_transposed_operator_mugrid(quad_field_ijqxyz=self.quad_sensitivity,
                                                nodal_field_inxyz=self.nodal_scratch, apply_weights=True)
        self.sensitivity.s[...] += self.nodal_scratch.s
        if self.symmetry != 'none':
            t_symmetry = time.time()
            self.symmetrize(self.sensitivity)             # gradient within the symmetric designs
            self.complete_stresses_by_symmetry()
            self.timing['symmetry'] += time.time() - t_symmetry
        self.timing['sensitivity'] += time.time() - t_sens
        self.nb_evaluations += 1
        return f, to_host(self.sensitivity.s[0, 0]).reshape(-1)

    def phase_field_term(self):
        """w_pf / (c_W V^(2/3)) [eta int |grad rho|^2 + 1/eta int rho^2 (1 - rho)^2] and its gradient
        (gradient energy with the element quadrature, double well with nodal quadrature)"""
        xp, disc, rho = self.xp, self.disc, self.rho
        disc.fft.communicate_ghosts(rho)
        disc.gradient_op.apply(nodal_field=rho, quadrature_point_field=self.rho_gradient)
        w = xp.asarray(self.weights).reshape(-1, 1, 1, 1)
        gradient_energy = self.global_sum(w * self.rho_gradient.s[0] ** 2)
        disc.fft.communicate_ghosts(self.rho_gradient)
        disc.gradient_op.transpose(quadrature_point_field=self.rho_gradient, nodal_field=self.sensitivity,
                                   weights=disc.quadrature_weights)                   # D^T W D rho
        node_volume = self.volume / self.nb_nodes
        r = rho.s
        double_well = node_volume * self.global_sum(r ** 2 * (1 - r) ** 2)
        self.sensitivity.s[...] *= 2 * self.eta * self.pf_scale
        self.sensitivity.s[...] += (self.pf_scale * node_volume / self.eta) * (2 * r * (2 * r * r - 3 * r + 1))
        return self.pf_scale * (self.eta * gradient_energy + double_well / self.eta)

    def load_case_term(self, lc, E, energy_terms):
        """Solve the load case, return its part of the objective and add its part of the quadrature sensitivity."""
        u, adjoint = self.u, self.adjoint
        t_solve = time.time()

        # mechanical problem  K u = -B^T W s C_0 : E
        self.assemble_force(E, self.rhs, -1.0)
        self.warm.load(('u', lc), u)
        self.cg_iterations['mechanical'].append(conjugate_gradients_energy(
            self.comm, self.K, self.mg.apply, self.rhs, u, self.cg_work, tol=self.cg_tol, d_min=self.cg_delay))
        self.warm.store(('u', lc), u)

        stress = self.homogenized_stress(u, E)
        self.homogenized_stresses[lc] = stress
        if self.objective == 'energy':        # W_k / |W*_k| of the energy terms of this load case
            self.stiffness_measures += [float(np.sum(E_k * stress)) / abs(W_k) for E_k, W_k, _ in energy_terms]
        else:                                 # stress components relative to the size of the target stress
            self.stiffness_measures += list(stress.ravel() / np.linalg.norm(self.target_stresses[lc]))

        # objective of the load case and D = d f / d Sigma_h * (-V / 2w): the "stress difference" driving the adjoint
        difference = self.target_stresses[lc] - stress
        if self.objective == 'energy':
            misfits = [(E_k, c_k, float(np.sum(E_k * difference)) / W_k, W_k) for E_k, W_k, c_k in energy_terms]
            f_lc = sum(c_k * misfit ** 2 for _, c_k, misfit, _ in misfits)        # misfit = (W*_k - W_k) / W*_k
            D = sum(c_k * misfit / W_k * E_k for E_k, c_k, misfit, W_k in misfits)
        else:
            norm = float(np.sum(self.target_stresses[lc] ** 2))
            f_lc, D = float(np.sum(difference ** 2)) / norm, difference / norm
        factor = 2.0 * self.w / self.volume

        # adjoint problem  K lambda = (2 w / V) B^T W s C_0 : D
        self.assemble_force(D, self.rhs, factor)
        self.warm.load(('adjoint', lc), adjoint)
        if float(np.abs(D).max()) == 0.0:
            adjoint.sg[...] = 0.0
            self.cg_iterations['adjoint'].append(0)
        else:
            self.cg_iterations['adjoint'].append(conjugate_gradients_energy(
                self.comm, self.K, self.mg.apply, self.rhs, adjoint, self.cg_work, tol=self.cg_tol,
                d_min=self.cg_delay))
        self.warm.store(('adjoint', lc), adjoint)
        self.timing['solve'] += time.time() - t_solve
        t_sens = time.time()

        residual_term = self.residual_term(u, adjoint, E)
        # sensitivity at the quadrature points: (grad lambda - (2 w / V) D) : C_0 : (E + grad u)  [* ds/drho later]
        self.add_sensitivity(u, adjoint, E, -factor * D)
        self.timing['sensitivity'] += time.time() - t_sens
        if self.verbose and self.rank0:
            print(f'load case {lc}: CG mechanical {self.cg_iterations["mechanical"][-1]}, adjoint '
                  f'{self.cg_iterations["adjoint"][-1]}, f {self.w * f_lc:.6e}, residual term {residual_term:.2e}')
        return self.w * f_lc + residual_term

    def add_contraction(self, a, b, out):
        """out += a_ij C_0_ijkl b_kl at every quadrature point (term by term over the non-zeros of C_0: only
        scalar-field temporaries)"""
        xp = self.xp
        scratch = self.quad_scratch.s[0, 0]
        for i in range(3):
            for j in range(3):
                terms = [(k, l, self.C_0[i, j, k, l]) for k in range(3) for l in range(3) if self.C_0[i, j, k, l] != 0]
                if not terms:
                    continue
                (k, l, c) = terms[0]
                xp.multiply(b.s[k, l], c, out=scratch)
                for k, l, c in terms[1:]:
                    scratch += c * b.s[k, l]
                scratch *= a.s[i, j]
                out.s[0, 0] += scratch

    def add_hourglass_sensitivity(self, u, adjoint, out):
        """out += eps / w_q sum_{i alpha} kappa_{i alpha}(C_0) (H u)_{i alpha} (H lambda)_{i alpha}: the hourglass
        part of lambda^T dK/ds u (its energy carries no quadrature weight, N^T W applies one later)"""
        xp, hg = self.xp, self.hourglass
        self.disc.fft.communicate_ghosts(u)
        hg.operator.apply(nodal_field=u, quadrature_point_field=hg.strain)             # H u, all components
        self.disc.fft.communicate_ghosts(adjoint)
        kappa = self.hourglass_kappa / self.weights[0]
        scratch = self.quad_scratch.s[0, 0]
        for i in range(3):
            self.nodal_scratch.s[0, 0] = adjoint.s[i, 0]
            self.disc.fft.communicate_ghosts(self.nodal_scratch)
            hg.operator.apply(nodal_field=self.nodal_scratch, quadrature_point_field=self.hourglass_mode)
            for mode in range(4):
                xp.multiply(hg.strain.s[i, mode], self.hourglass_mode.s[0, mode], out=scratch)
                scratch *= kappa[i, mode]
                out.s[0, 0] += scratch


# ======================================================================================================
# driver
# ======================================================================================================
def interpolate_design(path, disc):
    """nodal design of a saved run (any grid, `_final.nc`, `_checkpoint.nc` or the last frame of a `_history.nc`),
    periodic trilinear interpolation onto the local nodes of `disc`, clipped to [0, 1].
    Axis by axis (the interpolation is a tensor product): three passes over arrays that grow to the local grid."""
    fields = io_utils.read_file(path).fields
    name = next(n for n in ('phase_field_solution', 'phase_field_iterate') if n in fields)
    coarse = np.asarray(fields[name][-1]).reshape(-1)
    nc = int(round(coarse.size ** (1 / 3)))
    values = coarse.reshape(nc, nc, nc)
    n = disc.nb_of_pixels_global[0]
    for axis, (location, size) in enumerate(zip(disc.fft.subdomain_locations, disc.fft.nb_subdomain_grid_pts)):
        position = (int(location) + np.arange(int(size))) * (nc / n)          # local nodes in coarse grid units
        base = np.floor(position).astype(int)
        fraction = (position - base).reshape([-1 if a == axis else 1 for a in range(3)])
        values = (np.take(values, base % nc, axis=axis) * (1.0 - fraction)
                  + np.take(values, (base + 1) % nc, axis=axis) * fraction)
    return np.clip(values, 0, 1), nc


def main():
    start_time = time.time()
    env = os.environ.get
    n_pix = int(env('NPIX', 32))
    eta = float(env('ETA', 2 / n_pix))
    soft_exp = int(env('SOFT_EXP', 3))
    to = TopologyOptimization(
        n_pix, eta, element=env('ELEMENT', 'trilinear_hexahedron_1Q_hg'), hourglass_eps=float(env('HG_EPS', 0.3)),
        device=env('DEVICE', 'cpu'), soft_phase=10.0 ** -soft_exp, objective=env('OBJECTIVE', 'energy'),
        target_nu=float(env('TARGET_NU', -0.5)), target_scale=float(env('TARGET_SCALE', 1.0)),
        pf_weight=float(env('PF_WEIGHT', 0.2)), stress_weight=float(env('STRESS_WEIGHT', 1.0)),
        cg_delay=int(env('CG_DELAY', 0)), mg_degree=int(env('MG_DEGREE', 2)), warm_store=env('WARM_STORE', 'device'),
        symmetry=env('SYMMETRY', 'none'), mg_coarse_element=env('MG_COARSE_ELEMENT', '') or None,
        mg_coarse_hourglass_eps=float(env('MG_COARSE_HG_EPS')) if env('MG_COARSE_HG_EPS') else None,
        operator=env('OPERATOR', 'stencil'), mg_precision=env('MG_PRECISION', 'double'))
    disc, xp, rank0 = to.disc, to.xp, to.rank0
    mpi = MPI.COMM_WORLD
    if rank0:
        print(f'3D TO {n_pix}^3, ' + (f'fused q1 operator, multigrid in {to.mg.precision} precision'
                                      if to.fused is not None else disc.element_type)
              + f', {mpi.size} rank(s), device {disc.device}, eta {eta}, '
              f'void {to.soft_phase:g}, objective {to.objective}, warm starts: {env("WARM_STORE", "device")}, '
              f'symmetry {to.symmetry}, load cases {[lc for lc, _, _ in to.load_cases]}')
        print(f'multigrid levels {[level.disc.nb_of_pixels_global[0] for level in to.mg.levels]}'
              + ('' if to.mg.tail is None else ', serial tail on rank 0'))
        print('Target elastic tangent (Voigt):\n', material_models.compute_Voigt_notation_4order(to.C_target))

    out_dir = env('OUT_DIR', os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data', 'to3d_large'))
    if rank0:
        os.makedirs(out_dir, exist_ok=True)
    mpi.Barrier()
    output = os.path.join(out_dir, f'N{n_pix}_soft{to.soft_phase:g}_eta{eta:g}_{to.objective}' + env('RUN_SUFFIX', ''))

    # ---- start design: the run's own checkpoint if there is one, else INIT_FROM
    init_from = env('INIT_FROM', '')
    iterations_done = 0
    if os.path.exists(output + '_final.nc'):
        if rank0:
            print(f'{output}_final.nc exists: this run is finished, nothing to do')
        return
    if os.path.exists(output + '_checkpoint.nc'):
        init_from = output + '_checkpoint.nc'
        iterations_done = int(np.ravel(io_utils.read_file(init_from).attributes['iteration'])[0])
        if rank0:
            print(f'Continuing from the checkpoint of iteration {iterations_done}')
    if not init_from:
        raise SystemExit('INIT_FROM is required: large runs start from a design, never from noise')
    start, n_start = interpolate_design(init_from, disc)
    if rank0:
        print(f'Initial design from {init_from} ({n_start}^3 -> {n_pix}^3), wall {time.time() - start_time:.1f} s',
              flush=True)
    if to.symmetry != 'none':                              # the start must be symmetric too
        to.rho.s[0, 0] = xp.asarray(start)
        to.symmetrize(to.rho)
        start = to_host(to.rho.s[0, 0])
    x0 = np.ascontiguousarray(start, dtype=float).ravel()

    # ---- Helmholtz initial inverse Hessian (I + c eta^2 (-Lap_h))^-1, Lap_h: 7-point Laplacian
    #      HELMHOLTZ=fft: exact, by FFT (default on one rank); HELMHOLTZ=cg: conjugate gradients on the stencil to a
    #      relative residual of 1e-10 (default under MPI: the distributed GPU FFT costs far more than the few dozen
    #      stencil products, and CG needs only ghost exchanges)
    helm_c = float(env('HELM_C', 4.0))
    h = 1.0 / n_pix
    helmholtz_mode = env('HELMHOLTZ', 'fft' if mpi.size == 1 else 'cg')
    seconds = {'helmholtz': 0.0}
    if helmholtz_mode == 'fft':
        helm_hat = disc.ffield_collection.complex_field(name='helmholtz_hat', components=(1,))
        neg_laplace = sum(4 / h ** 2 * np.sin(np.pi * np.asarray(to_host(disc.fft.ifftfreq[a]), dtype=float) / n_pix)
                          ** 2 for a in range(3))
        helm_symbol = xp.asarray(disc.fft.normalisation / (1.0 + helm_c * eta ** 2 * neg_laplace))

        def helmholtz(v):
            start = time.time()
            to.nodal_scratch.s[0, 0] = xp.asarray(v).reshape(to.nodal_scratch.s.shape[2:])
            disc.fft.fft(to.nodal_scratch, helm_hat)
            helm_hat.s[0, 0] *= helm_symbol
            disc.fft.ifft(helm_hat, to.nodal_scratch)
            result = to_host(to.nodal_scratch.s[0, 0]).ravel()
            seconds['helmholtz'] += time.time() - start
            return result
    else:
        from muGrid import GenericLinearOperator
        stencil = np.zeros((3, 3, 3))                      # I + c eta^2 (-Lap_h): 1 + 6 a at the node, -a at 6 neighbours
        a = helm_c * eta ** 2 / h ** 2
        stencil[1, 1, 1] = 1.0 + 6.0 * a
        for offset in ((0, 1, 1), (2, 1, 1), (1, 0, 1), (1, 2, 1), (1, 1, 0), (1, 1, 2)):
            stencil[offset] = -a
        helmholtz_operator = GenericLinearOperator([-1, -1, -1], stencil[np.newaxis, np.newaxis, np.newaxis])
        # between two evaluations the design fields are free: used as the CG vectors
        z, r, p, Ap = to.nodal_scratch, to.rho, to.sensitivity, disc.get_scalar_field(name='phase_field_iterate')
        dot = lambda u, w: to.comm.sum(np.float64(float(linalg.vecdot(u, w))))

        def helmholtz(v):
            start = time.time()
            r.s[0, 0] = xp.asarray(v).reshape(r.s.shape[2:])
            linalg.copy(r, p)
            linalg.scal(0.0, z)
            rr = dot(r, r)
            target = 1e-20 * rr                            # (relative residual 1e-10)^2
            for _ in range(1000):
                if rr <= target:
                    break
                disc.fft.communicate_ghosts(p)
                helmholtz_operator.apply(nodal_field=p, quadrature_point_field=Ap)
                alpha = rr / dot(p, Ap)
                linalg.axpy(alpha, p, z)
                linalg.axpy(-alpha, Ap, r)
                rr_new = dot(r, r)
                linalg.axpby(1.0, r, rr_new / rr, p)
                rr = rr_new
            result = to_host(z.s[0, 0]).ravel()
            seconds['helmholtz'] += time.time() - start
            return result

    # ---- output: history (every HISTORY_EVERY-th iterate), checkpoint, evaluation log
    parameters = {'nb_of_pixels': (n_pix,) * 3, 'domain_size': [1, 1, 1], 'domain_lengths': [1.0, 1.0, 1.0],
                  'element_type': disc.element_type, 'preconditioner_type': 'MG', 'eta': eta, 'p': to.p,
                  'pf_weight': float(env('PF_WEIGHT', 0.2)), 'stress_weight': to.w, 'soft_phase': to.soft_phase,
                  'target_scale': float(env('TARGET_SCALE', 1.0)), 'target_nu': float(env('TARGET_NU', -0.5)),
                  'objective': to.objective,
                  'target_C_voigt': material_models.compute_Voigt_notation_4order(to.C_target)}
    iterate = disc.get_scalar_field(name='phase_field_iterate')
    history_every = int(env('HISTORY_EVERY', 10))
    checkpoint_every = int(env('CHECKPOINT_EVERY', 0))
    history_name = output + ('_history.nc' if iterations_done == 0 else f'_history_from{iterations_done}.nc')
    history = io_utils.FieldWriter(history_name, [iterate], attributes=parameters, frame_variables={
        'objective': (), 'volume_fraction': (), 'nb_evaluations': ()}) if history_every > 0 else None
    state = {'iteration': iterations_done, 'f': np.nan, 'misfit': np.nan, 'x': None, 'history': [], 'measures': None,
             'stiffness_history': []}
    state['accepted'] = None
    state['steps'], state['trial_mean'] = [], None
    max_step, step_growth = float(env('MAX_STEP', 0)), float(env('STEP_GROWTH', 0))
    energy_tol = float(env('ENERGY_TOL', 0))
    time_budget = float(env('TIME_BUDGET', 0))
    ftol, ftol_window, misfit_tol = float(env('FTOL', 0)), int(env('FTOL_WINDOW', 10)), float(env('MISFIT_TOL', 0))

    def objective(x):
        t = time.time()
        if state['accepted'] is not None:                 # how far is this trial design from the last accepted one?
            change = np.abs(np.asarray(x, dtype=float).ravel() - state['accepted'])
            largest = mpi.allreduce(float(change.max()), op=MPI.MAX)
            mean = mpi.allreduce(float(change.sum())) / n_pix ** 3
            recent = state['steps'][-10:]                  # mean changes of the last accepted steps
            limit = min(max_step if max_step > 0 else np.inf,
                        step_growth * max(recent) if step_growth > 0 and len(recent) >= 2 else np.inf)
            rejected = mean > limit
            state['trial_mean'] = mean
            if rank0:
                print(f'TRIAL design: change from the last accepted design max {largest:.3e}, mean {mean:.3e}'
                      + (f' > limit {limit:.3e}: rejected without solving' if rejected else ''), flush=True)
            if rejected:                                  # the line search halves the step
                return np.inf, np.zeros_like(np.asarray(x, dtype=float))
        f, gradient = to.evaluate(x)
        state['f'], state['misfit'], state['measures'] = f, to.misfit, np.array(to.stiffness_measures)
        if rank0:
            print(f'EVAL {to.nb_evaluations} f {float(f):.12e} wall {time.time() - start_time:.2f} '
                  f'eval {time.time() - t:.2f} s', flush=True)
        return f, gradient

    def callback(x):
        state['iteration'] += 1
        state['accepted'] = np.array(x, dtype=float).ravel()
        if state['trial_mean'] is not None and state['trial_mean'] > 0:
            state['steps'].append(state['trial_mean'])    # the trial evaluated last is the accepted one
        it = state['iteration']
        write_history = history is not None and (it - 1) % history_every == 0
        out_of_time = time_budget > 0 and time.time() - start_time > time_budget
        write_checkpoint = (checkpoint_every > 0 and it % checkpoint_every == 0) or out_of_time
        if write_history or write_checkpoint:
            iterate.s[0, 0] = xp.asarray(x).reshape(iterate.s.shape[2:])
        if write_history:
            history.write(objective=state['f'], volume_fraction=mpi.allreduce(float(np.sum(x))) / n_pix ** 3,
                          nb_evaluations=to.nb_evaluations)
        if write_checkpoint:
            io_utils.save_fields(output + '_checkpoint_new.nc', [iterate], attributes={**parameters, 'iteration': it})
            if rank0:
                os.replace(output + '_checkpoint_new.nc', output + '_checkpoint.nc')
        if out_of_time:
            raise TimeBudgetReached
        # stopping tests on the objective (the optimizers evaluate the accepted iterate last, so state holds its values)
        history_f = state['history']
        history_f.append(state['f'])
        decrease = (history_f[-1 - ftol_window] - history_f[-1]) / abs(history_f[-1]) if len(history_f) > ftol_window \
            else np.inf
        stiffness = state['stiffness_history']
        stiffness.append(state['measures'])
        settled = float(np.abs(stiffness[-1] - stiffness[-1 - ftol_window]).max()) if len(stiffness) > ftol_window \
            else np.inf
        if rank0:
            print(f'ITER {it} f {state["f"]:.8e} stiffness misfit {state["misfit"]:.6e}; over {ftol_window} iterations: '
                  f'relative decrease of f {decrease:.2e}, largest change of an energy / target {settled:.2e}',
                  flush=True)
        reason = (f'ENERGY_TOL: energies changed by {settled:.2e} < {energy_tol:g} of their targets over {ftol_window} '
                  f'iterations' if energy_tol > 0 and settled < energy_tol else
                  f'FTOL: relative decrease {decrease:.2e} < {ftol:g} over {ftol_window} iterations'
                  if ftol > 0 and decrease < ftol else
                  f'MISFIT_TOL: stiffness misfit {state["misfit"]:.3e} < {misfit_tol:g}'
                  if misfit_tol > 0 and state['misfit'] < misfit_tol else None)
        if reason is not None:
            state['x'] = np.array(x, dtype=float).ravel()
            raise Converged(reason)

    # ---- optimisation
    profiler = None
    if env('PROFILE', '0') == '1' and rank0:              # where does the time go? (rank 0, printed at the end)
        import cProfile
        profiler = cProfile.Profile()
        profiler.enable()
    options = dict(gtol=float(env('GTOL', 0.0)), xtol=1e-3, maxiter=int(env('MAXITER', 500)) - iterations_done,
                   maxcor=20, c1=1e-4, max_halvings=40)
    try:
        if env('DESIGN_OPT', 'lbfgs_helm') == 'lbfgs_helm_local':
            import design_optimizers
            assert mpi.size == 1, 'design_optimizers.py is serial'
            result = design_optimizers.l_bfgs_bounded_precond(objective, x0, precond=helmholtz, bounds_lo=0.0,
                                                              bounds_hi=1.0, callback=callback, disp=True, **options)
        else:
            result = solvers_nonlinear.minimize(fun=objective, x0=x0, method='l_bfgs', bounds=(0.0, 1.0),
                                                precond=helmholtz, comm=mpi, callback=callback, disp=True, **options)
    except Converged as stop:
        from types import SimpleNamespace
        result = SimpleNamespace(x=state['x'], success=True, nit=state['iteration'] - iterations_done,
                                 message=f'CONVERGENCE: {stop}')
    except TimeBudgetReached:
        if history is not None:
            history.close()
        if rank0:
            print(f'TIME BUDGET reached at iteration {state["iteration"]} (f {state["f"]:.8e}, wall '
                  f'{time.time() - start_time:.0f} s): checkpoint written, continue with the same command')
        return
    if profiler is not None:
        import pstats
        profiler.disable()
        pstats.Stats(profiler).sort_stats('tottime').print_stats(28)
    if history is not None:
        history.close()
    if rank0:
        print(f'OPTIMIZER success={bool(result.success)} nit={result.nit} message: {result.message}')

    # ---- final design with the homogenised stresses of the last evaluation of every load case
    solution = disc.get_scalar_field(name='phase_field_solution')
    solution.s[0, 0] = xp.asarray(result.x).reshape(solution.s.shape[2:])
    objective(result.x)                                    # stresses (and warm-started solves) at the final design
    io_utils.save_fields(output + '_final.nc', [solution], attributes={
        **parameters, 'nb_iterations': state['iteration'], 'optimizer_success': int(bool(result.success)),
        'optimizer_message': str(result.message), 'elapsed_time': time.time() - start_time,
        'homogenized_stresses': to.homogenized_stresses, 'target_stresses': to.target_stresses,
        'num_iteration_mech': to.cg_iterations['mechanical'], 'num_iteration_adjoint': to.cg_iterations['adjoint']})
    if rank0:
        print(f'Data saved to: {output}_final.nc')
        print(f'TOTAL wall {time.time() - start_time:.1f} s, CG mech {sum(to.cg_iterations["mechanical"])}, '
              f'CG adjoint {sum(to.cg_iterations["adjoint"])}, evaluations {to.nb_evaluations}, '
              f'MG setup {to.timing["setup"]:.1f} s, solves {to.timing["solve"]:.1f} s (of which warm-start '
              f'transfers {to.warm.seconds:.1f} s), sensitivity {to.timing["sensitivity"]:.1f} s (of which symmetry '
              f'{to.timing["symmetry"]:.1f} s), Helmholtz ({helmholtz_mode}) {seconds["helmholtz"]:.1f} s')
    if env('MEM_REPORT', '0') == '1' and rank0:
        import fable_memory
        owners = {'TO + MG level 0': disc}
        owners.update({f'MG level {level.index}': level.disc for level in to.mg.levels[1:]})
        fable_memory.report(owners, n_pix ** 3)


if __name__ == '__main__':
    main()
