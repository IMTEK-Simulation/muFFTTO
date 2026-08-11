"""
MPI version of the finite difference checks in
``test_topology_optimization_conductivity.py``.

Run it with

    mpirun -n 2 .venv/bin/python -m pytest muFFTTO/test/test_topology_optimization_conductivity_mpi.py

The file also works with plain ``pytest`` (a one-rank MPI run). In that case the
FD checks are the serial ones and, in addition,
``test_serial_and_mpi_give_the_same_results`` spawns ``mpirun -n 1`` and
``mpirun -n 2`` as subprocesses and compares the two results pixel by pixel.

Differences with respect to the serial test:

*   every field is decomposed, so ``discretization.nb_of_pixels`` and
    ``field.s`` only hold the pixels owned by the current rank. The phase field
    is therefore built from one globally identical random array that is sliced
    with ``discretization.fft.subdomain_locations``. This makes the initial
    guess independent of the number of ranks.
*   the FD loop runs over *global* pixels. All ranks walk the same loop (the
    objective function contains collective calls, so every rank has to enter
    it), but only the rank that owns the pixel perturbs its local entry.
*   all norms/errors are reduced over MPI before they are compared.
"""

import json
import os
import subprocess
import sys

import numpy as np
import pytest
from mpi4py import MPI

from muFFTTO import domain
from muFFTTO import solvers
from muFFTTO import topology_optimization_conductivity as topology_optimization

comm = MPI.COMM_WORLD


# ---------------------------------------------------------------------------
#  helpers
# ---------------------------------------------------------------------------
def get_local_slice(discretization):
    """Slice that maps a global pixel array onto the subdomain of this rank."""
    offsets = np.asarray(discretization.fft.subdomain_locations, dtype=int)
    nb_local = np.asarray(discretization.nb_of_pixels, dtype=int)
    return tuple(slice(o, o + n) for o, n in zip(offsets, nb_local))


def owns_pixel(discretization, global_index):
    """Is the global pixel ``global_index`` stored on this rank?

    Returns the local index of the pixel or None.
    """
    offsets = np.asarray(discretization.fft.subdomain_locations, dtype=int)
    nb_local = np.asarray(discretization.nb_of_pixels, dtype=int)
    local_index = np.asarray(global_index, dtype=int) - offsets
    if np.all(local_index >= 0) and np.all(local_index < nb_local):
        return tuple(local_index)
    return None


def global_norm(local_array):
    """Frobenius norm of a distributed array."""
    local_sum = np.sum(np.asarray(local_array) ** 2)
    return np.sqrt(comm.allreduce(local_sum, op=MPI.SUM))


def global_max(local_array):
    """Maximum of a distributed array (-inf for an empty subdomain)."""
    local_array = np.asarray(local_array)
    local_max = np.max(local_array) if local_array.size else -np.inf
    return comm.allreduce(local_max, op=MPI.MAX)


def gather_global_pixel_field(discretization, local_pixel_array):
    """Collect a distributed [x,y] array into a global array on every rank."""
    offsets = tuple(int(o) for o in discretization.fft.subdomain_locations)
    pieces = comm.allgather((offsets, np.asarray(local_pixel_array)))
    global_array = np.zeros(discretization.nb_of_pixels_global)
    for offsets, piece in pieces:
        piece_slice = tuple(slice(o, o + n) for o, n in zip(offsets, piece.shape))
        global_array[piece_slice] = piece
    return global_array


def print_root(*args):
    if comm.rank == 0:
        print(*args, flush=True)


# ---------------------------------------------------------------------------
#  problem definition, shared by the FD check and the serial/MPI comparison
# ---------------------------------------------------------------------------
MACRO_GRADIENT = np.array([1.0, 0.0])
CONDUCTIVITY_C_0 = np.array([[1.0, 0.0], [0.0, 1.0]])
CONDUCTIVITY_C_VOID = CONDUCTIVITY_C_0 * 0.0
CONDUCTIVITY_C_TARGET = np.array([[0.2, 0.0], [0.0, 1.0]])
P_EXPONENT = 2
WEIGHT = 3
ETA = 0.3
CG_SETUP = {'cg_tol': 1e-9}


def make_objective_function(discretization, preconditioner_type='Green_Jacobi'):
    """Build the objective function of the serial test for a discretization.

    Returns a callable that takes the phase field (a muGrid field holding the
    pixels of this rank) and returns
    ``(objective, f_sigma, f_phase_field, sensitivity)``. The three scalars are
    the same on every rank, the sensitivity is a distributed field.
    """
    target_flux_ij = np.einsum('ij,j->i', CONDUCTIVITY_C_TARGET, MACRO_GRADIENT)

    macro_gradient_field_ijqxyz = discretization.get_gradient_size_field(name='macro_gradient_field')
    discretization.get_macro_gradient_field_mugrid(macro_gradient_ij=MACRO_GRADIENT,
                                                   macro_gradient_field_ijqxyz=macro_gradient_field_ijqxyz)

    preconditioner_Green = discretization.get_preconditioner_Green_mugrid(
        reference_material_data_ijkl=CONDUCTIVITY_C_0)

    def M_fun_Green(x, Px):
        discretization.fft.communicate_ghosts(x)
        discretization.apply_preconditioner_mugrid(preconditioner_Fourier_fnfnqks=preconditioner_Green,
                                                   input_nodal_field_fnxyz=x,
                                                   output_nodal_field_fnxyz=Px)

    def my_objective_function(phase_field_1nxyz):
        # Phase field in quadrature points
        phase_field_at_quad_poits_1qxyz = discretization.get_quad_field_scalar(
            name='phase_field_at_quads_in_objective_function')
        discretization.fft.communicate_ghosts(phase_field_1nxyz)
        discretization.apply_N_operator_mugrid(phase_field_1nxyz, phase_field_at_quad_poits_1qxyz)

        # Material data in quadrature points
        material_data_field_C_0_rho_ijklqxyz = discretization.get_material_data_size_field_mugrid(
            name='material_data_field_C_0_rho_ijklqxyz_in_objective')
        material_data_field_C_0_rho_ijklqxyz.s[...] = \
            CONDUCTIVITY_C_0[..., np.newaxis, np.newaxis, np.newaxis] * \
            np.power(phase_field_at_quad_poits_1qxyz.s, P_EXPONENT)[0, 0, :, ...]

        f_phase_field = topology_optimization.objective_function_phase_field(discretization=discretization,
                                                                             phase_field_1nxyz=phase_field_1nxyz,
                                                                             eta=ETA,
                                                                             double_well_depth=1)
        #  sensitivity phase field terms
        s_phase_field = discretization.get_scalar_field(name='s_phase_field')
        s_phase_field.s.fill(0)

        topology_optimization.sensitivity_phase_field_term_FE_NEW(discretization=discretization,
                                                                  phase_field_1nxyz=phase_field_1nxyz,
                                                                  p=P_EXPONENT,
                                                                  eta=ETA,
                                                                  output_array=s_phase_field,
                                                                  double_well_depth=1)

        M_fun = M_fun_Green
        if preconditioner_type == 'Green_Jacobi':
            K_diag_alg = discretization.get_preconditioner_Jacobi_mugrid(
                material_data_field_ijklqxyz=material_data_field_C_0_rho_ijklqxyz)

            def M_fun_Green_Jacobi(x, Px):
                discretization.fft.communicate_ghosts(x)
                x_jacobi_temp = discretization.get_unknown_size_field(name='x_jacobi_temp')

                x_jacobi_temp.s[...] = K_diag_alg.s * x.s
                discretization.apply_preconditioner_mugrid(
                    preconditioner_Fourier_fnfnqks=preconditioner_Green,
                    input_nodal_field_fnxyz=x_jacobi_temp,
                    output_nodal_field_fnxyz=Px)

                Px.s[...] = K_diag_alg.s * Px.s
                discretization.fft.communicate_ghosts(Px)

            M_fun = M_fun_Green_Jacobi

        # Solve equilibrium constraint
        def K_fun(x, Ax):
            discretization.apply_system_matrix_mugrid(material_data_field=material_data_field_C_0_rho_ijklqxyz,
                                                      input_field_inxyz=x,
                                                      output_field_inxyz=Ax)

        rhs_inxyz = discretization.get_unknown_size_field(name='rhs_field')
        discretization.get_rhs_mugrid(material_data_field_ijklqxyz=material_data_field_C_0_rho_ijklqxyz,
                                      macro_gradient_field_ijqxyz=macro_gradient_field_ijqxyz,
                                      rhs_inxyz=rhs_inxyz)

        temperature_field = discretization.get_unknown_size_field(name='temperature_field_')
        temperature_field.s.fill(0)

        solvers.conjugate_gradients_mugrid(
            comm=discretization.communicator,
            fc=discretization.field_collection,
            hessp=K_fun,  # linear operator
            b=rhs_inxyz,
            x=temperature_field,
            P=M_fun,
            tol=CG_SETUP['cg_tol'],
            maxiter=10000,
        )

        homogenized_flux = discretization.get_homogenized_stress(
            material_data_field_ijklqxyz=material_data_field_C_0_rho_ijklqxyz,
            displacement_field_inxyz=temperature_field,
            macro_gradient_field_ijqxyz=macro_gradient_field_ijqxyz)

        f_sigma = topology_optimization.compute_flux_equivalence_potential(
            actual_flux_ij=homogenized_flux,
            target_flux_ij=target_flux_ij)

        adjoint_field = discretization.get_unknown_size_field(name='adjoint_field')
        adjoint_field.s.fill(0)

        sensitivity_analytical = discretization.get_scalar_field(name='sensitivity_analytical')
        sensitivity_analytical.s.fill(0)

        sensitivity_analytical.s[0, 0], adjoint_field, adjoint_energies, info_adjoint = \
            topology_optimization.sensitivity_flux_and_adjoint(
                discretization=discretization,
                base_material_data_ijkl=CONDUCTIVITY_C_0,
                void_material_data_ijkl=CONDUCTIVITY_C_VOID,
                displacement_field_inxyz=temperature_field,
                adjoint_field_inxyz=adjoint_field,
                macro_gradient_field_ijqxyz=macro_gradient_field_ijqxyz,
                phase_field_1nxyz=phase_field_1nxyz,
                target_flux_ij=target_flux_ij,
                actual_flux_ij=homogenized_flux,
                preconditioner_fun=M_fun,
                system_matrix_fun=K_fun,
                p=P_EXPONENT,
                weight=WEIGHT,
                disp=False,
                **CG_SETUP)

        sensitivity_analytical.s[...] += s_phase_field.s

        objective_function = WEIGHT * f_sigma + f_phase_field
        return objective_function, f_sigma, f_phase_field, sensitivity_analytical

    return my_objective_function


def make_discretization(domain_size, element_type, nb_pixels):
    element_types = ['linear_triangles', 'linear_triangles_tilled',
                     'trilinear_hexahedron', 'trilinear_hexahedron_1Q']
    my_cell = domain.PeriodicUnitCell(domain_size=domain_size,
                                      problem_type='conductivity')
    return domain.Discretization(cell=my_cell,
                                 nb_of_pixels_global=nb_pixels,
                                 discretization_type='finite_element',
                                 element_type=element_types[element_type])


def set_random_phase_field(discretization, name='phase_field_0', seed=1):
    """Random phase field that does not depend on the number of ranks."""
    rng = np.random.RandomState(seed)
    global_phase_field = rng.rand(*discretization.nb_of_pixels_global)
    phase_field = discretization.get_scalar_field(name=name)
    phase_field.s[0, 0] = global_phase_field[get_local_slice(discretization)]
    discretization.fft.communicate_ghosts(phase_field)
    return phase_field, global_phase_field


@pytest.fixture()
def discretization_fixture(domain_size, element_type, nb_pixels):
    return make_discretization(domain_size, element_type, nb_pixels)


# ---------------------------------------------------------------------------
#  tests
# ---------------------------------------------------------------------------
@pytest.mark.parametrize('domain_size , element_type, nb_pixels', [
    ([1, 2], 0, [4, 5]),
    ([1, 2], 1, [4, 5]),
    ([3.1, 6.4], 0, [7, 6])])
def test_domain_decomposition_is_a_partition_of_the_global_grid(discretization_fixture):
    """The subdomains have to tile the global grid without overlap."""
    discretization = discretization_fixture

    marker = discretization.get_scalar_field(name='decomposition_marker')
    marker.s.fill(1.0)
    ownership = gather_global_pixel_field(discretization, marker.s[0, 0])

    nb_owned = comm.allreduce(int(np.prod(discretization.nb_of_pixels)), op=MPI.SUM)
    assert nb_owned == np.prod(discretization.nb_of_pixels_global), (
        f'the subdomains hold {nb_owned} pixels, the global grid has '
        f'{np.prod(discretization.nb_of_pixels_global)}')
    assert np.all(ownership == 1.0), 'some global pixels are owned by no rank or by several ranks'


@pytest.mark.parametrize('domain_size , element_type, nb_pixels', [
    ([1, 2], 0, [4, 5]),
    ([1, 2], 1, [4, 5]),
    ([3.1, 6.4], 0, [7, 6])])
def test_objective_function_is_the_same_on_all_ranks(discretization_fixture):
    """Objective values are global reductions, so they must be rank independent."""
    discretization = discretization_fixture
    objective_function = make_objective_function(discretization)

    phase_field, _ = set_random_phase_field(discretization)
    objective, f_sigma, f_phase_field, _ = objective_function(phase_field)

    for name, value in (('objective', objective), ('f_sigma', f_sigma), ('f_phase_field', f_phase_field)):
        values = comm.allgather(float(value))
        assert np.allclose(values, values[0], rtol=1e-12, atol=0.0), (
            f'{name} differs between ranks: {values}')


@pytest.mark.parametrize('domain_size , element_type, nb_pixels', [
    ([1, 2], 0, [4, 5]),
    ([1, 2], 1, [4, 5]),
    ([3.1, 6.4], 0, [7, 6])])
def test_fd_check_of_whole_objective_function_2D_conductivity_mpi(discretization_fixture, plot=False):
    """
    Finite difference check of the whole objective function gradient
    with respect to the phase field -- MPI version.

    The FD loop runs over the global pixels: all ranks call the objective
    function (it contains collective operations) but only the owner of the
    perturbed pixel changes its local value.
    """
    discretization = discretization_fixture
    my_objective_function = make_objective_function(discretization)

    print_root('macro_gradient = \n {}'.format(MACRO_GRADIENT))
    print_root('target_flux = \n {}'.format(np.einsum('ij,j->i', CONDUCTIVITY_C_TARGET, MACRO_GRADIENT)))

    phase_field, _ = set_random_phase_field(discretization)
    # Save a copy of the original phase field
    phase_field_0_fixed = discretization.get_scalar_field(name='phase_field_0_fixed')
    phase_field_0_fixed.s[...] = np.copy(phase_field.s)

    _, _, _, analytical_sensitivity = my_objective_function(phase_field)
    analytical_sensitivity_fixed = np.copy(analytical_sensitivity.s)

    # Phase field lives in [0,1] -- large epsilon drives it outside the linearization regime.
    # O(h^2) convergence is visible only for epsilon << 1.
    epsilons = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7]

    fd_sensitivity = discretization.get_scalar_field(name='fd_sensitivity')
    fd_sensitivity_drho_dro = discretization.get_scalar_field(name='fd_sensitivity_drho_dro')
    fd_sensitivity_dsigma_dro = discretization.get_scalar_field(name='fd_sensitivity_dsigma_dro')

    error_fd_vs_analytical = []
    error_fd_vs_analytical_max = []
    norm_fd_sensitivity_dsigma_dro = []
    norm_fd_sensitivity_df_dro = []
    norm_fd_sensitivity = []
    fd_scheme = 2.
    for epsilon in epsilons:
        # loop over every single element of the *global* phase field
        for x in np.arange(discretization.nb_of_pixels_global[0]):
            for y in np.arange(discretization.nb_of_pixels_global[1]):
                local_index = owns_pixel(discretization, (x, y))

                # set phase field back to the unperturbed one
                phase_field.s[...] = np.copy(phase_field_0_fixed.s)
                if local_index is not None:
                    phase_field.s[(0, 0) + local_index] += epsilon / fd_scheme
                discretization.fft.communicate_ghosts(phase_field)

                of_plus_eps, f_sigma_plus_eps, f_rho_plus_eps, _ = my_objective_function(phase_field)

                if local_index is not None:
                    phase_field.s[(0, 0) + local_index] -= epsilon
                discretization.fft.communicate_ghosts(phase_field)

                of_minu_eps, f_sigma_minu_eps, f_rho_minu_eps, _ = my_objective_function(phase_field)

                # the objective values are global, only the owner stores the derivative
                if local_index is not None:
                    owned_pixel = (0, 0) + local_index
                    fd_sensitivity.s[owned_pixel] = (of_plus_eps - of_minu_eps) / epsilon
                    fd_sensitivity_drho_dro.s[owned_pixel] = (f_rho_plus_eps - f_rho_minu_eps) / epsilon
                    fd_sensitivity_dsigma_dro.s[owned_pixel] = (f_sigma_plus_eps - f_sigma_minu_eps) / epsilon

        error_fd_vs_analytical.append(
            global_norm((fd_sensitivity.s - analytical_sensitivity_fixed)[0, 0]))
        error_fd_vs_analytical_max.append(
            global_max((fd_sensitivity.s - analytical_sensitivity_fixed)[0, 0]))
        norm_fd_sensitivity.append(
            global_norm(fd_sensitivity.s[0, 0]))
        norm_fd_sensitivity_df_dro.append(
            global_norm(fd_sensitivity_drho_dro.s[0, 0]))
        norm_fd_sensitivity_dsigma_dro.append(
            global_norm(fd_sensitivity_dsigma_dro.s[0, 0]))

    print_root()
    print_root('nb of ranks         = {}'.format(comm.size))
    print_root('error fd vs analyt. = {}'.format(error_fd_vs_analytical))
    print_root('norm fd sens.       = {}'.format(norm_fd_sensitivity))
    print_root('norm fd d f_rho     = {}'.format(norm_fd_sensitivity_df_dro))
    print_root('norm fd d f_sigma   = {}'.format(norm_fd_sensitivity_dsigma_dro))

    errors = np.array(error_fd_vs_analytical)
    epsilons_arr = np.array(epsilons)
    analytical_norm = global_norm(analytical_sensitivity_fixed[0, 0])
    relative_errors = errors / analytical_norm

    if plot and comm.rank == 0:
        import matplotlib.pyplot as plt
        idx_min = np.argmax(errors)
        plt.figure()
        plt.loglog(epsilons, errors, marker='x', label='FD vs analytical error')
        plt.loglog(epsilons, errors[idx_min] * (epsilons_arr / epsilons_arr[idx_min]) ** 2,
                   linestyle='--', label=r'$O(h^2)$ reference')
        plt.loglog(epsilons, errors[idx_min] * (epsilons_arr / epsilons_arr[idx_min]) ** 1,
                   linestyle='--', label=r'$O(h)$ reference')
        plt.legend(loc='best')
        plt.xlabel('epsilon (FD step size)')
        plt.ylabel('Error (Frobenius norm)')
        plt.title('FD check: whole objective function, {} ranks'.format(comm.size))
        plt.show()

    # Always check: minimum relative error should be small
    assert np.min(relative_errors) < 1e-4, (
        f"FD check failed on {comm.size} rank(s): minimum relative error "
        f"{np.min(relative_errors):.2e} exceeds 1e-4. "
        f"Analytical derivative may be wrong.")

    # Convergence rate check
    log_eps = np.log10(epsilons_arr[:3])
    log_err = np.log10(errors[:3])
    convergence_rate = np.polyfit(log_eps, log_err, 1)[0]
    assert convergence_rate > 0.95, (
        f"FD convergence rate {convergence_rate:.2f} too low "
        f"(expected ~2 for central differences). Analytical derivative may be wrong.")


# ---------------------------------------------------------------------------
#  serial versus MPI comparison
# ---------------------------------------------------------------------------
REFERENCE_SETUP = dict(domain_size=[1, 2], element_type=0, nb_pixels=[4, 5])


def compute_reference_values():
    """Objective, flux and sensitivity for REFERENCE_SETUP, as plain python data.

    The sensitivity is assembled into a global array, so the result does not
    depend on the number of ranks the function was evaluated with.
    """
    discretization = make_discretization(**REFERENCE_SETUP)
    objective_function = make_objective_function(discretization)

    phase_field, _ = set_random_phase_field(discretization)
    objective, f_sigma, f_phase_field, sensitivity = objective_function(phase_field)

    return {
        'nb_ranks': comm.size,
        'objective': float(objective),
        'f_sigma': float(f_sigma),
        'f_phase_field': float(f_phase_field),
        'sensitivity': gather_global_pixel_field(discretization, sensitivity.s[0, 0]).tolist(),
    }


@pytest.mark.skipif(comm.size > 1,
                    reason='driver test: it spawns its own mpirun, run it without mpirun')
def test_serial_and_mpi_give_the_same_results(tmp_path):
    """A one-rank and a two-rank run must produce the same numbers.

    This is the check the FD tests cannot do: a finite difference check only
    proves that the sensitivity is consistent with the objective function of
    the *same* run, it does not notice if both are wrong by the same
    rank-dependent factor.
    """
    if not any(os.access(os.path.join(path, 'mpirun'), os.X_OK)
               for path in os.environ.get('PATH', '').split(os.pathsep)):
        pytest.skip('mpirun not found')

    # importing mpi4py has initialised MPI in this process; the MPI environment
    # variables it left behind make a nested mpirun refuse to start.
    environment = {key: value for key, value in os.environ.items()
                   if not key.startswith(('OMPI_', 'PMIX_', 'PMI_'))}

    results = {}
    for nb_ranks in (1, 2):
        output_file = tmp_path / f'reference_{nb_ranks}.json'
        run = subprocess.run(['mpirun', '-n', str(nb_ranks),
                              sys.executable, '-m', 'mpi4py', __file__, str(output_file)],
                             env=environment, capture_output=True, text=True)
        assert run.returncode == 0, (
            f'mpirun -n {nb_ranks} failed:\n{run.stdout}\n{run.stderr}')
        with open(output_file) as f:
            results[nb_ranks] = json.load(f)

    serial, parallel = results[1], results[2]
    for name in ('objective', 'f_sigma', 'f_phase_field'):
        assert parallel[name] == pytest.approx(serial[name], rel=1e-7, abs=1e-12), (
            f'{name} of the 2 rank run ({parallel[name]!r}) differs from the '
            f'serial one ({serial[name]!r})')

    serial_sensitivity = np.array(serial['sensitivity'])
    parallel_sensitivity = np.array(parallel['sensitivity'])
    absolute_error = np.max(np.abs(parallel_sensitivity - serial_sensitivity))
    assert absolute_error < 1e-7 * np.max(np.abs(serial_sensitivity)), (
        f'sensitivity of the 2 rank run differs from the serial one, '
        f'maximum absolute difference {absolute_error:.2e}')


if __name__ == '__main__':
    # entry point used by test_serial_and_mpi_give_the_same_results:
    #   mpirun -n <n> python test_topology_optimization_conductivity_mpi.py <output.json>
    reference_values = compute_reference_values()
    if comm.rank == 0:
        with open(sys.argv[1], 'w') as output:
            json.dump(reference_values, output)
