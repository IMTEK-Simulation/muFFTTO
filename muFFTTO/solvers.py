import numpy as np

from NuMPI.Tools import Reduction
from mpi4py import MPI
from _muGrid import Communicator, Field, FieldCollection


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
        fc: FieldCollection,
        hessp: callable,
        b: Field,
        x: Field,
        P: callable,
        tol: float = 1e-6,
        maxiter: int = 1000,
        callback: callable = None,
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
        unique_name="cg-search-direction",  # name of the field
        components_shape=(*x.components_shape,),  # shape of components
        sub_division='nodal_points'  # sub-point type
    )
    Ap = fc.real_field(
        unique_name="cg-hessian-product",  # name of the field
        components_shape=(*x.components_shape,),  # shape of components
        sub_division='nodal_points'  # sub-point type
    )
    r = fc.real_field(
        unique_name="cg-residual",  # name of the field
        components_shape=(*x.components_shape,),  # shape of components
        sub_division='nodal_points'  # sub-point type
    )
    z = fc.real_field(
        unique_name="cg-preconditioner_residual",  # name of the field
        components_shape=(*x.components_shape,),  # shape of components
        sub_division='nodal_points'  # sub-point type
    )

    hessp(x, Ap)
    r.s = b.s - Ap.s
    P(r, z)
    p.s = np.copy(z.s)  # residual

    if callback:
        callback(0, x.s, r.s, p.s,z.s)

    rr = comm.sum(np.dot(r.s.ravel(), r.s.ravel()))  # initial residual dot product
    rz = comm.sum(np.dot(r.s.ravel(), z.s.ravel()))  # initial residual dot product

    if rr < tol_sq:
        return x

    for iteration in range(maxiter):
        # Compute Hessian product
        hessp(p, Ap)

        # Update x (and residual)
        pAp = comm.sum(np.dot(p.s.ravel(), Ap.s.ravel()))
        if pAp <= 0:
            raise RuntimeError("Hessian is not positive definite")

        alpha = rz / pAp
        x.s += alpha * p.s
        r.s -= alpha * Ap.s

        P(r, z)

        if callback:
            callback(iteration + 1, x.s, r.s, p.s,z.s)

        # Check convergence
        next_rr = comm.sum(np.dot(r.s.ravel(), r.s.ravel()))
        next_rz = comm.sum(np.dot(r.s.ravel(), z.s.ravel()))

        if next_rr < tol_sq:
            return x

        # Update search direction
        # beta = next_rr / rr
        beta = next_rz / rz
        p.s = z.s + beta * p.s
        rz = next_rz
        # p.s *= beta
        # p.s += z.s

    raise RuntimeError("Conjugate gradient algorithm did not converge")


def PCG(Afun, B, x0, P, steps=int(500), toler=1e-6, norm_energy_upper_bound=False, lambda_min=None, norm_type='rz',
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


def gradient_descent(Afun, B, x0, omega, P=None, steps=int(500), toler=1e-6):
    # print('I am in PCG')
    """

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
        z_0 = P(r_0)

        r_0z_0 = scalar_product_mpi(r_0, z_0)
        alpha = float(r_0z_0 / scalar_product_mpi(r_0, Afun(r_0)))

        x_k = x_k + alpha * r_0

        norms['residual_rr'].append(scalar_product_mpi(r_0, r_0))

        if norms['residual_rr'][-1] < toler:
            break

    return x_k, norms


def scalar_product_mpi(a, b):
    return Reduction(MPI.COMM_WORLD).sum(a * b)


# Author: Elvis do A. Soares
# Github: @elvissoares
# Date: 2020-06-05
# Updated: 2021-05-31

" ==== FIRE: Fast Inertial Relaxation Engine ===== "

" References: "
"- Bitzek, E., Koskinen, P., Gähler, F., Moseler, M., & Gumbsch, P. (2006). Structural relaxation made simple. Physical Review Letters, 97(17), 1–4. https://doi.org/10.1103/PhysRevLett.97.170201"
"- Guénolé, J., Nöhring, W. G., Vaid, A., Houllé, F., Xie, Z., Prakash, A., & Bitzek, E. (2020). Assessment and optimization of the fast inertial relaxation engine (FIRE) for energy minimization in atomistic simulations and its implementation in LAMMPS. Computational Materials Science, 175. https://doi.org/10.1016/j.commatsci.2020.109584"

" Global variables for the FIRE algorithm"
alpha0 = 0.1
Ndelay = 5
Nmax = 10000
finc = 1.1
fdec = 0.5
fa = 0.99
Nnegmax = 2000


def optimize_fire(x0, f, df, params, atol=1e-4, dt=0.002, logoutput=False):
    error = 10 * atol
    dtmax = 10 * dt
    dtmin = 0.02 * dt
    alpha = alpha0
    Npos = 0

    x = x0.copy()
    V = np.zeros(x.shape)
    F = -df(x)  # , params

    for i in range(Nmax):

        # P = (F * V).sum()  # dissipated power
        P = Reduction(MPI.COMM_WORLD).sum(F * V)
        if (P > 0):
            Npos = Npos + 1
            if Npos > Ndelay:
                dt = min(dt * finc, dtmax)
                alpha = alpha * fa
        else:
            Npos = 0
            dt = max(dt * fdec, dtmin)
            alpha = alpha0
            V = np.zeros(x.shape)

        V = V + 0.5 * dt * F
        norm_of_V = np.sqrt(scalar_product_mpi(V, V))
        norm_of_F = np.sqrt(scalar_product_mpi(F, F))
        V = (1 - alpha) * V + alpha * F * norm_of_V / norm_of_F
        # V = (1 - alpha) * V + alpha * F * np.linalg.norm(V) / np.linalg.norm(F)

        x = x + dt * V
        F = -df(x)  # , params
        V = V + 0.5 * dt * F

        # error = max(abs(F))
        error = Reduction(MPI.COMM_WORLD).max(abs(F))
        if error < atol: break

        if logoutput:
            print('{} - iteration of FIRE-1 \n'
                  'f(x)= {}, error = {}'.format(i, f(x), error))

    del V, F
    return [x, f(x), i]  # , params


def optimize_fire2(x0, f, df, params, atol=1e-4, dt=0.002, logoutput=False):
    error = 10 * atol
    dtmax = 10 * dt
    dtmin = 0.02 * dt
    alpha = alpha0
    Npos = 0
    Nneg = 0

    x = x0.copy()
    V = np.zeros(x.shape)
    F = -df(x, params)

    for i in range(Nmax):

        P = (F * V).sum()  # dissipated power

        if (P > 0):
            Npos = Npos + 1
            Nneg = 0
            if Npos > Ndelay:
                dt = min(dt * finc, dtmax)
                alpha = alpha * fa
        else:
            Npos = 0
            Nneg = Nneg + 1
            if Nneg > Nnegmax: break
            if i > Ndelay:
                dt = max(dt * fdec, dtmin)
                alpha = alpha0
            x = x - 0.5 * dt * V
            V = np.zeros(x.shape)

        V = V + 0.5 * dt * F
        V = (1 - alpha) * V + alpha * F * np.linalg.norm(V) / np.linalg.norm(F)
        x = x + dt * V
        F = -df(x, params)
        V = V + 0.5 * dt * F

        error = max(abs(F))
        if error < atol: break

        if logoutput: print(f(x, params), error)

    del V, F
    return [x, f(x, params), i]


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


############################################
if __name__ == "__main__":
    ###############
    print('========= Optimizing the Rosenbrock function =========')
    print('xmim=', np.array([1.0, 1.0]))
    print('fmim=', 0.0)


    def gradf(x, params):
        [a, b] = params
        return np.array([-2 * (a - x[0]) - 4 * b * (x[1] - x[0] * x[0]) * x[0], 2 * b * (x[1] - x[0] * x[0])])


    def f(x, params):
        [a, b] = params
        return (np.power((a - x[0]), 2) + b * np.power((x[1] - x[0] * x[0]), 2))


    p = np.array([1, 100])
    x0 = np.array([3.0, 4.0])

    print('Fire version 1')
    [xmin, fmin, Niter] = optimize_fire(x0, f, gradf, params=p, atol=1e-6)

    print("xmin = ", xmin)
    print("fmin = ", fmin)
    print("Iterations = ", Niter)

    print('Fire version 2')
    [xmin, fmin, Niter] = optimize_fire2(x0, f, gradf, p, 1e-6)

    print("xmin = ", xmin)
    print("fmin = ", fmin)
    print("Iterations = ", Niter)

    ########################
    print('========= Optimizing the Eggholder function =========')
    print('xmim=', np.array([512.0, 404.0]))
    print('fmim=', -956.6407)


    def gradf(x, params):
        c0 = params
        arg1 = np.sqrt(np.abs(x[1] + 0.5 * x[0] + c0))
        arg2 = np.sqrt(np.abs(x[0] - (x[1] + c0)))
        return np.array([-x[0] * (-c0 + x[0] - x[1]) * np.cos(arg2) / (2 * arg2 ** 3) - (c0 + x[1]) * (
                c0 + 0.5 * x[0] + x[1]) * np.cos(arg1) / (4 * arg1 ** 3) - np.sin(arg2),
                         x[1] * (-c0 + x[0] - x[1]) * np.cos(arg2) / (2 * arg2 ** 3) - (c0 + x[1]) * (
                                 c0 + 0.5 * x[0] + x[1]) * np.cos(arg1) / (2 * arg1 ** 3) - np.sin(arg2)])


    def f(x, params):
        c0 = params
        return -(x[1] + c0) * np.sin(np.sqrt(np.abs(x[1] + 0.5 * x[0] + c0))) - x[0] * np.sin(
            np.sqrt(np.abs(x[0] - (x[1] + c0))))


    p = 47
    x0 = np.array([0.0, 0.0])

    print('Fire version 1')
    [xmin, fmin, Niter] = optimize_fire(x0, f, gradf, p, 1e-6, 0.1)

    print("xmin = ", xmin)
    print("fmin = ", fmin)
    print("Iterations = ", Niter)

    print('Fire version 2')
    [xmin, fmin, Niter] = optimize_fire2(x0, f, gradf, p, 1e-6, 0.1)

    print("xmin = ", xmin)
    print("fmin = ", fmin)
    print("Iterations = ", Niter)
