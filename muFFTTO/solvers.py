import numpy as np
import warnings

from NuMPI.Tools import Reduction
from mpi4py import MPI
from muGrid import Communicator, Field, GlobalFieldCollection


def donothing(*args, **kwargs):
    pass


def findS(curve, Delta, l):
    # function to compute safety factor S
    curve = np.array(curve)
    ind = np.where((curve[l] / curve) <= 1e-4)[0]  # , 1, 'last')
    last_index = ind[-1] if ind.size > 0 else None
    if last_index is None:
        last_index = 0
    S = Reduction(MPI.COMM_WORLD).max(curve[last_index:-1] / Delta[last_index: -1])

    return S


def conjugate_gradients_mugrid(
        comm: Communicator,
        fc: GlobalFieldCollection,
        hessp: callable,
        b: Field,
        x: Field,
        P: callable,
        tol: float = 1e-6,
        rtol: bool = False,
        maxiter: int = 1000,
        callback: callable = None,
        norm_metric: callable = None,
        **kwargs
):
    """
    Conjugate gradient method for matrix-free solution of the linear problem
    Ax = b, where A is represented by the function hessp (which computes the
    product of A with a vector). The method iteratively refines the solution x
    until the residual ||Ax - b|| is less than tol or until maxiter iterations
    are reached.

    Parameters
    ----------
    comm : muGrid.Communicator
        Communicator for parallel processing.
    fc : muGrid.FieldCollection
        Collection holding temporary fields of the CG algorithm.
    hessp : callable
        Function that computes the product of the Hessian matrix A with a vector.
    b : muGrid.Field
        Right-hand side vector.
    x : muGrid.Field
        Initial guess for the solution.
    P : callable
        Function that computes the product of the preconditioner matrix P with a vector.
    tol : float, optional
        Tolerance for convergence. The default is 1e-6.
    maxiter : int, optional
        Maximum number of iterations. The default is 1000.
    callback : callable, optional
        Function to call after each iteration with the current solution, residual,
        and search direction.

    Returns
    -------
    x : array_like
        Approximate solution to the system Ax = b. (Same as input field x.)
    """

    tol_sq = tol * tol
    p = fc.real_field(
        name="cg-search-direction",  # name of the field
        components=(*x.components_shape,),  # shape of components
        sub_pt='nodal_points'  # sub-point type
    )
    Ap = fc.real_field(
        name="cg-hessian-product",  # name of the field
        components=(*x.components_shape,),  # shape of components
        sub_pt='nodal_points'  # sub-point type
    )
    r = fc.real_field(
        name="cg-residual",  # name of the field
        components=(*x.components_shape,),  # shape of components
        sub_pt='nodal_points'  # sub-point type
    )
    z = fc.real_field(
        name="cg-preconditioned_residual",  # name of the field
        components=(*x.components_shape,),  # shape of components
        sub_pt='nodal_points'  # sub-point type
    )

    hessp(x, Ap)
    r.s[...] = b.s - Ap.s
    P(r, z)
    p.s[...] = np.copy(z.s)  # residual


    rr = comm.sum(np.dot(r.s.ravel(), r.s.ravel()))  # initial residual dot product
    rz = comm.sum(np.dot(r.s.ravel(), z.s.ravel()))  # initial residual dot product

    if norm_metric is not None:
        Pr = fc.real_field(
            name="cg-custom_metric_residual",  # name of the field
            components=(*x.components_shape,),  # shape of components
            sub_pt='nodal_points'  # sub-point type
        )
        norm_metric(r, Pr)
        stop_crit = comm.sum(np.dot(r.s.ravel(), Pr.s.ravel()))  # initial residual dot product


    elif norm_metric is None:
        stop_crit = rr

    if stop_crit < tol_sq:
        return x

    if rtol:
        tol_sq = tol_sq * stop_crit

    if callback:
        #callback(0, x.s, r.s, p.s, z.s, stop_crit)
        callback(0, x.s, r.s, p.s, z.s, stop_crit)
    for iteration in range(maxiter):
        # Compute Hessian product
        hessp(p, Ap)

        # Update x (and residual)
        pAp = comm.sum(np.dot(p.s.ravel(), Ap.s.ravel()))
        if pAp <= 0:
            raise RuntimeError("Hessian is not positive definite")

        alpha = rz / pAp
        x.s[...] += alpha * p.s
        r.s[...] -= alpha * Ap.s

        P(r, z)

        # Check convergence
        next_rr = comm.sum(np.dot(r.s.ravel(), r.s.ravel()))
        next_rz = comm.sum(np.dot(r.s.ravel(), z.s.ravel()))
        if norm_metric is not None:
            norm_metric(r, Pr)
            stop_crit = comm.sum(np.dot(r.s.ravel(), Pr.s.ravel()))  # initial residual dot product

        elif norm_metric is None:
            stop_crit = next_rr

        if callback:
            callback(iteration + 1, x.s, r.s, p.s, z.s, stop_crit)

        if stop_crit < tol_sq:
            return x

        # Update search direction
        # beta = next_rr / rr
        beta = next_rz / rz

        rz = next_rz

        p.s[...] = z.s + beta * p.s
        # p.s *= beta
        # p.s += z.s

    if comm.rank == 0:
        warnings.warn("Conjugate gradient algorithm did not converge", RuntimeWarning)
    return x

def conjugate_gradients_mugrid_experimental(
        comm: Communicator,
        fc: GlobalFieldCollection,
        hessp: callable,
        b: Field,
        x: Field,
        P: callable,
        tol: float = 1e-6,
        maxiter: int = 1000,
        callback: callable = None,
        rtol: bool = False,

        norm_metric: callable = None,
        **kwargs
):
    """
    Conjugate gradient method for matrix-free solution of the linear problem
    Ax = b, where A is represented by the function hessp (which computes the
    product of A with a vector). The method iteratively refines the solution x
    until the residual ||Ax- b|| is less than tol or until maxiter iterations
    are reached.

    Parameters
    ----------
    comm : muGrid.Communicator
        Communicator for parallel processing.
    fc : muGrid.FieldCollection
        Collection holding temporary fields of the CG algorithm.
    hessp : callable
        Function that computes the product of the Hessian matrix A with a vector.
    b : muGrid.Field
        Right-hand side vector.
    x : muGrid.Field
        Initial guess for the solution.
    P : callable
        Function that computes the product of the preconditioner matrix P with a vector.
    tol : float, optional
        Tolerance for convergence. The default is 1e-6.
    maxiter : int, optional
        Maximum number of iterations. The default is 1000.
    callback : callable, optional
        Function to call after each iteration with the current solution, residual,
        and search direction.

    Returns
    -------
    x : array_like
        Approximate solution to the systems Ax = b. (Same as input field x.)
    """
    tol_sq = tol * tol
    p = fc.real_field(
        name="cg-search-direction",  # name of the field
        components=(*x.components_shape,),  # shape of components
        sub_pt='nodal_points'  # sub-point type
    )
    Ap = fc.real_field(
        name="cg-hessian-product",  # name of the field
        components=(*x.components_shape,),  # shape of components
        sub_pt='nodal_points'  # sub-point type
    )
    r = fc.real_field(
        name="cg-residual",  # name of the field
        components=(*x.components_shape,),  # shape of components
        sub_pt='nodal_points'  # sub-point type
    )
    z = fc.real_field(
        name="cg-preconditioned_residual",  # name of the field
        components=(*x.components_shape,),  # shape of components
        sub_pt='nodal_points'  # sub-point type
    )

    hessp(x, Ap)
    r.s[...] = b.s[...] - Ap.s[...]
    P(r, z)
    p.s[...] = np.copy(z.s[...])  # residual

    norms = dict()

    rr = comm.sum(np.dot(r.s.ravel(), r.s.ravel()))  # initial residual dot product
    rz = comm.sum(np.dot(r.s.ravel(), z.s.ravel()))  # initial residual dot product




    if norm_metric is not None:
        Pr = fc.real_field(
            name="cg-custom_metric_residual",  # name of the field
            components=(*x.components_shape,),  # shape of components
            sub_pt='nodal_points'  # sub-point type
        )
        norm_metric(r, Pr)
        stop_crit = comm.sum(np.dot(r.s.ravel(), Pr.s.ravel()))  # initial residual dot product

    elif norm_metric is None:
        stop_crit = rr

    if stop_crit < tol_sq:
        return x

    if rtol:
        tol_sq = tol_sq * stop_crit

    if callback:
        # callback(0, x.s, r.s, p.s, z.s, stop_crit)
        callback(0, x.s, r.s, p.s, z.s, stop_crit)

    #  % in the paper this is denoted as k
    l = 0
    d = 0
    Delta = []
    curve = []
    estim = []
    norms['energy_lower_bound']= []
    delay = []
    if "tau" in kwargs:
        tau = kwargs['tau']
    else:
        tau = 0.25


    for iteration in range(maxiter):
        # Compute Hessian product
        hessp(p, Ap)

        # Update x (and residual)
        pAp = comm.sum(np.dot(p.s.ravel(), Ap.s.ravel()))
        if pAp <= 0:
            raise RuntimeError("Hessian is not positive definite")

        alpha = rz / pAp
        x.s[...] += alpha * p.s[...]
        r.s[...] -= alpha * Ap.s[...]

        P(r, z)

        # Check convergence
        next_rr = comm.sum(np.dot(r.s.ravel(), r.s.ravel()))
        next_rz = comm.sum(np.dot(r.s.ravel(), z.s.ravel()))
        if norm_metric is not None:
            norm_metric(r, Pr)
            stop_crit = comm.sum(np.dot(r.s.ravel(), Pr.s.ravel()))  # initial residual dot product

        elif norm_metric is None:
            stop_crit = next_rr

        if callback:
            callback(iteration + 1, x.s, r.s, p.s, z.s, stop_crit)




        # Update search direction
        # beta = next_rr / rr
        beta = next_rz / rz
        p.s[...] = z.s + beta * p.s

        # Energy - error estimator
        Delta.append(alpha * rz)
        curve.append(0)
        curve = (np.asarray(curve) + Delta[-1]).tolist()

        if iteration > 1:
            # safety factor
            S = findS(curve, Delta, l)

            num = S * Delta[-1]
            den = Reduction(MPI.COMM_WORLD).sum(Delta[l:-1])
            while (d >= 0) and (num / den <= tau):
                delay.append(d)
                norms['energy_lower_bound'].append(den + Delta[-1])
                l = l + 1
                d = d - 1
                den = Reduction(MPI.COMM_WORLD).sum(Delta[l:-1])

            d = d + 1

        rz = next_rz
        # p.s *= beta
        # p.s += z.s
        if stop_crit < tol_sq:
            return x, norms

    if comm.rank == 0:
        warnings.warn("Conjugate gradient algorithm did not converge", RuntimeWarning)

    return x, norms


def PCG(Afun, B, x0, P, steps=int(500), toler=1e-6, norm_energy_upper_bound=False,
        lambda_min=None, norm_type='rz',
        callback=None, **kwargs):
    # print('I am in PCG')
    """
    Conjugate gradients solver.

    Parameters
    ----------
    Afun : Matrix, LinOper, or numpy.array of shape (n, n)
        it stores the matrix data of linear system and provides a matrix by
        vector multiplication
    B : VecTri or numpy.array of shape (n,)
        it stores a right-hand side of linear system
    x0 : VecTri or numpy.array of shape (n,)
        initial approximation of solution of linear system
    """
    if x0 is None:
        x0 = np.zeros(B.shape)
    if callback is None:
        callback = donothing
    norms = dict()
    norms['residual_rr'] = []
    norms['residual_rz'] = []
    norms['data_scaled_rz'] = []
    norms['data_scaled_rr'] = []

    norms['energy_upper_bound'] = []
    if "exact_solution" in kwargs:
        norms['energy_iter_error'] = []
        error = x0 - kwargs['exact_solution']
        norms['energy_iter_error'].append(scalar_product_mpi(error, Afun(error)))

    ##
    k = 0
    x_k = np.copy(x0)
    callback(x_k)
    ##
    Ax = Afun(x0)

    r_0 = B - Ax
    z_0 = P(r_0)

    # scalar_product = lambda a, b: np.sum(a * b)

    r_0z_0 = scalar_product_mpi(r_0, z_0)

    norms['residual_rr'].append(scalar_product_mpi(r_0, r_0))
    norms['residual_rz'].append(r_0z_0)

    if norm_energy_upper_bound:
        gamma_mu = 1 / lambda_min
        norms['energy_upper_bound'].append(gamma_mu * r_0z_0)
    if norm_type == 'data_scaled_rz':
        r_1_C_z_1 = scalar_product_mpi(r_0, P(kwargs['norm_metric'](r_0)))
        norms['data_scaled_rz'].append(r_1_C_z_1)
    if norm_type == 'data_scaled_rr':
        r_1_C_r_1 = scalar_product_mpi(r_0, kwargs['norm_metric'](r_0))
        norms['data_scaled_rr'].append(r_1_C_r_1)

    p_0 = np.copy(z_0)
    #  % in the paper this is denoted as k
    l = 0
    d = 0
    Delta = []
    curve = []
    estim = []
    delay = []
    if "tau" in kwargs:
        tau = kwargs['tau']
    else:
        tau = 0.25

    for k in np.arange(1, steps):
        Ap_0 = Afun(p_0)
        # print('callback Ap_0 = {}'.format(np.linalg.norm(Ap_0)))
        # print('ID Ap_0 = {}'.format(id(Ap_0)))

        alpha = float(r_0z_0 / scalar_product_mpi(p_0, Ap_0))
        x_k = x_k + alpha * p_0
        # print('callback xo = {}'.format(np.linalg.norm(x_k)))
        callback(x_k, r_0)
        # print('callback xo = {}'.format(np.linalg.norm(x_k)))
        # print('callback Ap_0 = {}'.format(np.linalg.norm(Ap_0)))
        # print('ID Ap_0 = {}'.format(id(Ap_0)))
        # if xCG.val.mean() > 1e-10:
        #     print('iteration left zero-mean space {} \n {}'.format(xCG.name, xCG.val.mean()))

        r_0 = r_0 - alpha * Ap_0

        z_0 = P(r_0)

        r_1z_1 = scalar_product_mpi(r_0, z_0)
        if "exact_solution" in kwargs:
            error = x_k - kwargs['exact_solution']
            norms['energy_iter_error'].append(scalar_product_mpi(error, Afun(error)))

        if norm_energy_upper_bound:
            norms['energy_upper_bound'].append(gamma_mu * r_0z_0)

        norms['residual_rr'].append(scalar_product_mpi(r_0, r_0))
        norms['residual_rz'].append(r_1z_1)

        # if r_1z_1 < toler:  # TODO[Solver] check out stopping criteria
        #     break
        if norm_type == 'rz':
            if norms['residual_rz'][-1] < toler:  # TODO[Solver] check out stopping criteria
                break
        if norm_type == 'rz_rel':
            if norms['residual_rz'][-1] / norms['residual_rz'][0] < toler:  # TODO[Solver] check out stopping criteria
                break
        if norm_type == 'rr':
            if norms['residual_rr'][-1] < toler:  # TODO[Solver] check out stopping criteria
                break
        if norm_type == 'rr_rel':
            if norms['residual_rr'][-1] / norms['residual_rr'][0] < toler:  # TODO[Solver] check out stopping criteria
                break
        if norm_type == 'energy':
            if norms['energy_upper_bound'][-1] < toler:  # TODO[Solver] check out stopping criteria
                break
        if norm_type == 'data_scaled_rz':
            r_1_C_z_1 = scalar_product_mpi(r_0, P(kwargs['norm_metric'](r_0)))
            norms['data_scaled_rz'].append(r_1_C_z_1)
            if norms['data_scaled_rz'][-1] < toler:  # TODO[Solver] check out stopping criteria
                break
        if norm_type == 'data_scaled_rr':
            r_1_C_r_1 = scalar_product_mpi(r_0, kwargs['norm_metric'](r_0))
            norms['data_scaled_rr'].append(r_1_C_r_1)
            if norms['data_scaled_rr'][-1] < toler:  # TODO[Solver] check out stopping criteria
                break

        beta = r_1z_1 / r_0z_0
        p_0 = z_0 + beta * p_0

        # Energy - error estimator
        Delta.append(alpha * r_0z_0)
        curve.append(0)
        curve = (curve + Delta[-1]).tolist()

        if k > 1:
            # safety factor
            S = findS(curve, Delta, l)

            num = S * Delta[-1]
            den = Reduction(MPI.COMM_WORLD).sum(Delta[l:-1])
            while (d >= 0) and (num / den <= tau):
                delay.append(d)
                estim.append(den + Delta[-1])
                l = l + 1
                d = d - 1
                den = Reduction(MPI.COMM_WORLD).sum(Delta[l:-1])

            d = d + 1

        r_0z_0 = r_1z_1
        if norm_energy_upper_bound:
            # updade upper bound on energy error estim parameter
            gamma_mu = (gamma_mu - alpha) / (lambda_min * (gamma_mu - alpha) + beta)

        if "energy_lower_bound" in kwargs:
            norms['energy_lower_bound'] = estim

    return x_k, norms


def Richardson(Afun, B, x0, omega, P=None, steps=int(500), toler=1e-6):
    # print('I am in PCG')
    """
    Richardson iteration
    𝑥𝑘+1=𝑥𝑘−𝜏(𝐴𝑥𝑘−𝑓)
    Parameters
    ----------
    Afun : Matrix, LinOper, or numpy.array of shape (n, n)
        it stores the matrix data of linear system and provides a matrix by
        vector multiplication
    B : VecTri or numpy.array of shape (n,)
        it stores a right-hand side of linear system
    x0 : VecTri or numpy.array of shape (n,)
        initial approximation of solution of linear system

    """
    if x0 is None:
        x0 = np.zeros(B.shape)
    if P is None:
        P = lambda x: 1 * x

    norms = dict()
    norms['residual_rr'] = []
    ##
    k = 0
    x_k = np.copy(x0)
    ##
    for k in np.arange(1, steps):
        Ax = Afun(x_k)

        r_0 = B - Ax
        x_k += omega * P(r_0)

        norms['residual_rr'].append(scalar_product_mpi(r_0, r_0))
        # print(norms['residual_rr'][-1])
        if norms['residual_rr'][-1] < toler:
            break

    return x_k, norms


def dr_pbcg_mugrid(
        comm: Communicator,
        fc: GlobalFieldCollection,
        hessp: callable,
        b_list: list,
        x_list: list,
        P: callable,
        tol: float = 1e-6,
        rtol: bool = False,
        maxiter: int = 1000,
        callback: callable = None,
        **kwargs
):
    """
    DR-PBCG: Preconditioned Block Conjugate Gradient with deflation-based restart.

    Solves A X = B where B is a block of m right-hand sides simultaneously,
    following Algorithm 5 of Meurant & Tichy (2026).

    Parameters
    ----------
    comm : muGrid.Communicator
    fc : muGrid.GlobalFieldCollection
        Collection that holds all working fields (must have 'nodal_points' sub-pt).
    hessp : callable
        hessp(p, Ap): computes Ap = A p, writing the result into Ap.
    b_list : list of muGrid.Field, length m
        Block of m right-hand-side fields.
    x_list : list of muGrid.Field, length m
        Initial guesses, overwritten in-place with the approximate solutions.
    P : callable
        P(r, z): applies preconditioner M^{-1} to r, writing the result into z.
    tol : float
        Convergence tolerance on ||R_k||_F (absolute, or relative when rtol=True).
    rtol : bool
        If True, tolerance is relative to the initial ||R_0||_F.
    maxiter : int
        Maximum number of iterations.
    callback : callable, optional
        callback(iteration, x_list, stop_crit) called after each iteration.

    Returns
    -------
    x_list : list of Field
        Approximate solutions (same objects as input).
    norms : dict
        'residual_frobenius': list of ||R_k||_F values (one per iteration).
    """
    m    = len(b_list)
    comp = (*b_list[0].components_shape,)

    Q     = [fc.real_field(name=f'dr-pbcg-Q-{j}',    components=comp, sub_pt='nodal_points') for j in range(m)]
    S     = [fc.real_field(name=f'dr-pbcg-S-{j}',    components=comp, sub_pt='nodal_points') for j in range(m)]
    Q_new = [fc.real_field(name=f'dr-pbcg-Qnew-{j}', components=comp, sub_pt='nodal_points') for j in range(m)]
    S_new = [fc.real_field(name=f'dr-pbcg-Snew-{j}', components=comp, sub_pt='nodal_points') for j in range(m)]
    MiQ   = [fc.real_field(name=f'dr-pbcg-MiQ-{j}',  components=comp, sub_pt='nodal_points') for j in range(m)]
    R     = [fc.real_field(name=f'dr-pbcg-R-{j}',    components=comp, sub_pt='nodal_points') for j in range(m)]
    AS    = [fc.real_field(name=f'dr-pbcg-AS-{j}',   components=comp, sub_pt='nodal_points') for j in range(m)]

    def dot(u, v):
        return comm.sum(np.dot(u.s.ravel(), v.s.ravel()))

    def gram(u_list, v_list):
        """m × m matrix G[i,j] = <u_list[i], v_list[j]>."""
        return np.array([[dot(u_list[i], v_list[j]) for j in range(m)]
                         for i in range(m)])

    def householder_qr(r_list, q_list):
        """Thin QR via Householder reflections (MPI-aware).
        Overwrites q_list with orthonormal columns. Returns upper-triangular Psi."""
        Psi = np.zeros((m, m))

        for j in range(m):
            q_list[j].s[...] = r_list[j].s

        # Build the m x m Gram matrix G[i,j] = dot(r_list[i], r_list[j])
        # Then QR-factor G's Cholesky-like structure via Householder on the normal eqs.
        # Instead: work with the "tall" matrix implicitly via its m x m inner product matrix.

        # Step 1: collect all pairwise inner products (the m x m matrix A where A[:,j] is
        # the j-th vector expressed in terms of inner products with all others).
        # Householder QR on this m x m matrix gives us the mixing coefficients.
        A = np.zeros((m, m))
        for i in range(m):
            for j in range(i, m):
                A[i, j] = dot(q_list[i], q_list[j])
                A[j, i] = A[i, j]

        # Step 2: Householder QR of A (plain numpy, O(m^3), cheap)
        Q_coeff, Psi = np.linalg.qr(A)  # A = Q_coeff @ Psi

        # Step 3: form new q_list as linear combinations
        old_fields = [q_list[j].s.copy() for j in range(m)]
        for j in range(m):
            q_list[j].s[...] = sum(Q_coeff[k, j] * old_fields[k] for k in range(m))

        return Psi

    def modified_gram_schmidt(r_list, q_list):
        """Thin QR via Modified Gram-Schmidt (MPI-aware).
        Overwrites q_list with orthonormal columns. Returns upper-triangular Psi."""
        Psi = np.zeros((m, m))
        for j in range(m):
            q_list[j].s[...] = r_list[j].s
        for j in range(m):
            for i in range(j):
                Psi[i, j] = dot(q_list[i], q_list[j])
                q_list[j].s[...] -= Psi[i, j] * q_list[i].s
            nrm = np.sqrt(dot(q_list[j], q_list[j]))
            if nrm < 1e-14:
                raise RuntimeError(
                    f"DR-PBCG: column {j} collapsed during QR (linearly dependent RHS?)"
                )
            Psi[j, j] = nrm
            q_list[j].s[...] /= nrm
        return Psi

    def cholesky_qr(r_list, q_list):
        """Tall-and-thin QR via CholeskyQR (MPI-aware).
        Overwrites q_list with orthonormal columns. Returns upper-triangular Psi."""
        for j in range(m):
            q_list[j].s[...] = r_list[j].s

        # Step 1: Gram matrix G = R^T R  (one global reduction)
        G = np.zeros((m, m))
        for i in range(m):
            for j in range(i, m):
                G[i, j] = dot(q_list[i], q_list[j])
                G[j, i] = G[i, j]

        # Step 2: Cholesky G = L L^T  →  Psi = L^T (upper triangular R-factor)
        try:
            Psi = np.linalg.cholesky(G).T
        except np.linalg.LinAlgError:
            raise RuntimeError(
                "DR-PBCG: Gram matrix not positive definite (linearly dependent RHS?)"
            )

        # Step 3: Q = R Psi^{-1}
        Psi_inv = np.linalg.inv(Psi)
        old_fields = [q_list[j].s.copy() for j in range(m)]
        for j in range(m):
            q_list[j].s[...] = sum(Psi_inv[k, j] * old_fields[k] for k in range(m))

        return Psi
    # ------------------------------------------------------------------
    # Initialisation  (Algorithm 5, lines 2-5)
    # ------------------------------------------------------------------
    for j in range(m):
        hessp(x_list[j], R[j])                        # R[j] = A X0[j]
        R[j].s[...] = b_list[j].s - R[j].s            # R0 = B - A X0  (line 2)

    # Check convergence before QR: zero residual means X0 is already the solution.
    tol_sq     = tol * tol
    R0_norm_sq = comm.sum(sum(np.dot(R[j].s.ravel(), R[j].s.ravel()) for j in range(m)))

    if rtol:
        tol_sq = tol_sq * R0_norm_sq

    norms = {'residual_frobenius': [R0_norm_sq]}

    if R0_norm_sq < tol_sq:
        return x_list, norms
    #np.qr(R, mode='reduced')
    Sigma = modified_gram_schmidt(R, Q)                # [Q0, Sigma0] = qr(R0)  (line 3)
    #Sigma = householder_qr(R, Q)                # [Q0, Sigma0] = qr(R0)  (line 3)
    #Sigma = cholesky_qr(R, Q)                # [Q0, Sigma0] = qr(R0)  (line 3)


    for j in range(m):
        P(Q[j], S[j])                                  # S0 = M^{-1} Q0  (line 4)

    Theta = gram(Q, S)                                 # Theta0 = Q0^T M^{-1} Q0  (line 5)

    if callback:
        callback(0, x_list, R0_norm_sq)

    # ------------------------------------------------------------------
    # Main loop  (Algorithm 5, lines 6-13)
    # ------------------------------------------------------------------
    for iteration in range(maxiter):

        # Line 7: Pi = (S^T A S)^{-1} Theta
        for j in range(m):
            hessp(S[j], AS[j])
        StAS = gram(S, AS)                             # S^T A S  (m × m, SPD)
        Pi   = np.linalg.solve(StAS, Theta)            # (S^T A S)^{-1} Theta

        # Line 8: X_k = X_{k-1} + S Pi Sigma
        C = Pi @ Sigma                                 # m × m
        for j in range(m):
            for i in range(m):
                x_list[j].s[...] += C[i, j] * S[i].s

        # Line 9: R = Q - A S Pi,  then [Q_new, Psi] = qr(R)
        for j in range(m):
            R[j].s[...] = Q[j].s
            for i in range(m):
                R[j].s[...] -= Pi[i, j] * AS[i].s

        # Pre-QR convergence check: ||R_k||_F ≤ ||R||_F * ||Sigma||_F (submultiplicativity).
        # If R ≈ 0 the system is solved; avoid QR of near-zero columns.
        R_norm_sq     = comm.sum(sum(np.dot(R[j].s.ravel(), R[j].s.ravel()) for j in range(m)))
        Sigma_norm_sq = np.trace(Sigma.T @ Sigma)
        stop_crit     = R_norm_sq * Sigma_norm_sq
        if stop_crit < tol_sq:
            norms['residual_frobenius'].append( stop_crit)
            if callback:
                callback(iteration + 1, x_list, stop_crit)
            return x_list, norms

        Psi = modified_gram_schmidt(R, Q_new)

        # Line 10: Theta_new = Q_new^T M^{-1} Q_new
        for j in range(m):
            P(Q_new[j], MiQ[j])
        Theta_new = gram(Q_new, MiQ)

        # Line 11: S_new = M^{-1} Q_new + S Theta^{-1} Psi^T Theta_new
        C2 = np.linalg.solve(Theta, Psi.T @ Theta_new)  # m × m
        for j in range(m):
            S_new[j].s[...] = MiQ[j].s
            for i in range(m):
                S_new[j].s[...] += C2[i, j] * S[i].s

        # Line 12: Sigma_k = Psi Sigma_{k-1}
        Sigma = Psi @ Sigma

        # Convergence: ||R_k||_F^2 = tr(Sigma^T Sigma)
        stop_crit = np.trace(Sigma.T @ Sigma)
        norms['residual_frobenius'].append(stop_crit)

        if callback:
            callback(iteration + 1, x_list, stop_crit)

        if stop_crit < tol_sq:
            return x_list, norms

        # Advance: swap buffers instead of copying field data
        Q, Q_new = Q_new, Q
        S, S_new = S_new, S
        Theta = Theta_new

    if comm.rank == 0:
        warnings.warn("DR-PBCG did not converge", RuntimeWarning)
    return x_list, norms


def scalar_product_mpi(a, b):
    return Reduction(MPI.COMM_WORLD).sum(a * b)


# Update parameters using Adam
def update_parameters_with_adam(x, grads, m, v,
                                t, learning_rate,
                                beta1, beta2,
                                epsilon=1e-8):
    m = beta1 * m + (1.0 - beta1) * grads
    v = beta2 * v + (1.0 - beta2) * grads ** 2
    m_hat = m / (1.0 - beta1 ** (t + 1))
    v_hat = v / (1.0 - beta2 ** (t + 1))
    x = x - learning_rate * m_hat / (np.sqrt(v_hat) + epsilon)
    return x, m, v


# Adam optimization algorithm
def adam(f, df, x0,
         n_iter, alpha, beta1, beta2, eps=1e-8, callback=None, gtol=1e-5, ftol=2.2e-9):
    # phi=[]
    # phi_change=[]
    # Generate an initial point
    x = x0
    phi_old = f(x)

    # Initialize Adam moments
    # m, v = initialize_adam()
    m = np.zeros_like(x0)
    v = np.zeros_like(x0)
    # Run the gradient descent updates
    for t in range(n_iter):
        # Calculate gradient g(t)
        grad = df(x)

        # Update parameters using Adam
        x, m, v = update_parameters_with_adam(x=x, grads=grad, m=m,
                                              v=v, t=t, learning_rate=alpha,
                                              beta1=beta1, beta2=beta2,
                                              epsilon=eps)

        # Evaluate candidate point
        phi = f(x)

        phi_change = phi_old - phi

        phi_old = phi

        max_grad = Reduction(MPI.COMM_WORLD).max(np.abs(grad))
        # abs_grad = np.linalg.norm(grad)
        abs_grad = np.sqrt(Reduction(MPI.COMM_WORLD).sum(grad ** 2))
        norms__ = [phi, phi_change, max_grad, abs_grad, x, t]
        callback(norms__)
        # Report progress
        print('-------------- >>%d = %.5f' % (t, phi))

        if (max_grad < gtol):
            print("CONVERGED because gradient tolerance was reached")
            return [x, phi, t]
        if (phi_change <= ftol * max((1, abs(phi), abs(phi_old)))):
            print("CONVERGED because function tolerance was reached")
            return [x, phi, t]

    return [x, phi, t]


