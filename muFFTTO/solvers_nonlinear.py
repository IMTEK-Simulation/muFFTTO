import numpy as np

from muFFTTO import solvers

def solve_finite_strain_newton_cg(
        discretization,
        material,
        macro_gradient_ij,
        ninc=1,
        newton_tol=1e-8,
        newton_max_iter=50,
        cg_tol=1e-6,
        cg_max_iter=10000,
        preconditioner_type='Green',
        reference_material_data_ijkl=None,
        formulation='finite_strain',
        verbose=True,
):
    """
    Newton-CG solver for finite strain hyperelasticity.

    Solves the nonlinear equilibrium problem:
        find u_fluc such that:  div( P(F) ) = 0
    where F = I + grad(macro_gradient) + grad(u_fluc)

    The outer loop is Newton's method.
    The inner loop is a preconditioned conjugate gradient (CG) solver
    that solves the linearized system at each Newton step:
        K(u) * du = - R(u)

    Parameters
    ----------
    discretization          : muFFTTO discretization object
    material                : MaterialModelElasticity instance (e.g. NeoHookean)
    macro_gradient_ij       : np.ndarray [dim, dim] — prescribed macroscopic gradient
    ninc                    : number of load increments
    newton_tol              : Newton convergence tolerance on ||R|| / ||R_0||
    newton_max_iter         : max Newton iterations per increment
    cg_tol                  : CG solver tolerance
    cg_max_iter             : max CG iterations
    preconditioner_type     : 'Green' or 'Green_Jacobi'
    reference_material_data_ijkl : reference stiffness for Green preconditioner
    formulation             : passed to system matrix assembly
    verbose                 : print convergence info

    Returns
    -------
    results : dict with fields, norms, and iteration counts
    """

    dim = discretization.domain_dimension

    # ------------------------------------------------------------------
    # allocate fields
    # ------------------------------------------------------------------
    macro_gradient_inc_field       = discretization.get_gradient_size_field(
                                         name='macro_gradient_inc_field')
    displacement_fluctuation_field = discretization.get_unknown_size_field(
                                         name='displacement_fluctuation_field')
    displacement_increment_field   = discretization.get_unknown_size_field(
                                         name='displacement_increment_field')
    strain_fluc_field              = discretization.get_displacement_gradient_sized_field(
                                         name='strain_fluctuation_field')
    total_strain_field             = discretization.get_displacement_gradient_sized_field(
                                         name='total_strain_field')
    stress_field                   = discretization.get_displacement_gradient_sized_field(
                                         name='stress_field')
    tangent_field                  = discretization.get_material_data_size_field_mugrid(
                                         name='tangent_field')
    rhs_field                      = discretization.get_unknown_size_field(
                                         name='rhs_field')
    energy_field                   = discretization.get_quad_field_scalar(
                                         name='energy_field')

    # ------------------------------------------------------------------
    # macroscopic gradient increment
    # ------------------------------------------------------------------
    macro_gradient_inc_ij = macro_gradient_ij / float(ninc)
    discretization.get_macro_gradient_field_mugrid(
        macro_gradient_ij=macro_gradient_inc_ij,
        macro_gradient_field_ijqxyz=macro_gradient_inc_field
    )

    # ------------------------------------------------------------------
    # identity tensors for reference material and preconditioner
    # ------------------------------------------------------------------
    if reference_material_data_ijkl is None:
        i = np.eye(dim)
        I4s = 0.5 * (np.einsum('il,jk->ijkl', i, i)
                     + np.einsum('ik,jl->ijkl', i, i))
        reference_material_data_ijkl = I4s

    # ------------------------------------------------------------------
    # build Green preconditioner (displacement-independent, built once)
    # ------------------------------------------------------------------
    preconditioner = discretization.get_preconditioner_Green_mugrid(
        reference_material_data_ijkl=reference_material_data_ijkl
    )

    # ------------------------------------------------------------------
    # helper: apply preconditioner
    # ------------------------------------------------------------------
    def apply_green_preconditioner(x, Px):
        discretization.fft.communicate_ghosts(x)
        discretization.apply_preconditioner_mugrid(
            preconditioner_Fourier_fnfnqks=preconditioner,
            input_nodal_field_fnxyz=x,
            output_nodal_field_fnxyz=Px
        )

    def apply_green_jacobi_preconditioner(x, Px):
        K_diag = discretization.get_preconditioner_Jacobi_mugrid(
            material_data_field_ijklqxyz=tangent_field,
            formulation=formulation
        )
        x_scaled = discretization.get_unknown_size_field(name='x_scaled')
        x_scaled.s[...] = K_diag.s * x.s
        discretization.fft.communicate_ghosts(x_scaled)
        discretization.apply_preconditioner_mugrid(
            preconditioner_Fourier_fnfnqks=preconditioner,
            input_nodal_field_fnxyz=x_scaled,
            output_nodal_field_fnxyz=Px
        )
        Px.s[...] = K_diag.s * Px.s
        discretization.fft.communicate_ghosts(Px)

    if preconditioner_type == 'Green':
        M_fun = apply_green_preconditioner
    elif preconditioner_type == 'Green_Jacobi':
        M_fun = apply_green_jacobi_preconditioner
    else:
        def M_fun(x, Px):
            Px.s[...] = x.s

    # ------------------------------------------------------------------
    # helper: apply system matrix  K * x -> Ax
    # ------------------------------------------------------------------
    def K_fun(x, Ax):
        discretization.apply_system_matrix_mugrid(
            material_data_field=tangent_field,
            input_field_inxyz=x,
            output_field_inxyz=Ax,
            formulation=formulation
        )
        discretization.fft.communicate_ghosts(Ax)

    # ------------------------------------------------------------------
    # helper: compute residual R = -div(P)  from current total strain
    # ------------------------------------------------------------------
    def compute_residual():
        material.get_stress(total_strain_field, stress_field)
        material.get_algorithmic_tangent(total_strain_field, tangent_field)
        discretization.fft.communicate_ghosts(stress_field)
        discretization.apply_gradient_transposed_operator_mugrid(
            gradient_field_ijqxyz=stress_field,
            div_u_fnxyz=rhs_field,
            apply_weights=True
        )
        rhs_field.s[...] *= -1

    # ------------------------------------------------------------------
    # helper: MPI norm
    # ------------------------------------------------------------------
    def mpi_norm(field):
        return np.sqrt(
            discretization.communicator.sum(
                np.dot(field.s.ravel(), field.s.ravel())
            )
        )

    # ------------------------------------------------------------------
    # tracking
    # ------------------------------------------------------------------
    results = {
        'newton_residuals_per_increment' : [],
        'cg_iterations_per_newton_step'  : [],
        'total_newton_iterations'        : 0,
        'total_cg_iterations'            : 0,
        'displacement_fluctuation_field' : displacement_fluctuation_field,
        'total_strain_field'             : total_strain_field,
        'stress_field'                   : stress_field,
        'tangent_field'                  : tangent_field,
    }

    # ------------------------------------------------------------------
    # incremental loading loop
    # ------------------------------------------------------------------
    for inc in range(ninc):
        if verbose and discretization.communicator.rank == 0:
            print(f'\nIncrement {inc + 1} / {ninc}')
            print('=' * 60)

        # apply macroscopic strain increment
        total_strain_field.s[...] += macro_gradient_inc_field.s[...]

        # compute initial residual for this increment
        compute_residual()
        norm_rhs_0 = mpi_norm(rhs_field)

        if verbose and discretization.communicator.rank == 0:
            print(f'  Initial residual: {norm_rhs_0:.4e}')

        newton_residuals = [norm_rhs_0]

        # --------------------------------------------------------------
        # Newton loop
        # --------------------------------------------------------------
        for iiter in range(newton_max_iter):

            # solve linearised system:  K * du = R
            displacement_increment_field.s.fill(0.0)

            cg_iter_count = [0]

            def cg_callback(it, x, r, p, z, stop_crit_norm):
                cg_iter_count[0] = it

            solvers.conjugate_gradients_mugrid(
                comm=discretization.communicator,
                fc=discretization.field_collection,
                hessp=K_fun,
                b=rhs_field,
                x=displacement_increment_field,
                P=M_fun,
                tol=cg_tol,
                maxiter=cg_max_iter,
                callback=cg_callback,
                rtol=True,
            )

            results['cg_iterations_per_newton_step'].append(cg_iter_count[0])
            results['total_cg_iterations'] += cg_iter_count[0]

            # update displacement and strain
            discretization.apply_gradient_operator_mugrid(
                u_inxyz=displacement_increment_field,
                grad_u_ijqxyz=strain_fluc_field
            )
            total_strain_field.s[...]             += strain_fluc_field.s[...]
            displacement_fluctuation_field.s[...] += displacement_increment_field.s[...]

            # recompute residual
            compute_residual()
            norm_rhs = mpi_norm(rhs_field)
            newton_residuals.append(norm_rhs)

            results['total_newton_iterations'] += 1

            if verbose and discretization.communicator.rank == 0:
                print(f'  Newton it {iiter + 1:3d} | '
                      f'||R|| = {norm_rhs:.4e} | '
                      f'||R||/||R_0|| = {norm_rhs / (norm_rhs_0 + 1e-30):.4e} | '
                      f'CG its = {cg_iter_count[0]}')

            # Newton convergence check
            if norm_rhs / (norm_rhs_0 + 1e-30) < newton_tol and iiter > 0:
                if verbose and discretization.communicator.rank == 0:
                    print(f'  Newton converged in {iiter + 1} iterations.')
                break

        else:
            if verbose and discretization.communicator.rank == 0:
                print(f'  WARNING: Newton did not converge in {newton_max_iter} iterations.')

        results['newton_residuals_per_increment'].append(newton_residuals)

    return results