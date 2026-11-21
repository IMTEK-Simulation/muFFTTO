"""
Sparse direct solvers for symmetric (possibly indefinite) matrices, with the inertia.

A Newton or trust-region method on a nonconvex energy needs, besides solves with the Hessian ``H``, to know whether
``H + sigma D^2`` is positive definite -- and the number of negative eigenvalues of ``H`` tells when a structure
buckles. Every backend here factorises a symmetric sparse matrix once and returns that number (the inertia,
Sylvester's law of inertia applied to the factorisation), then solves with it.

    solver = direct_solvers.symmetric_solver()        # 'auto': MUMPS if installed, else SciPy
    negative = solver.factorise(K)                    # K: scipy.sparse, symmetric; returns #negative eigenvalues
    x = solver.solve(b)                               # with iterative refinement

Backends
    'scipy'  SciPy's SuperLU with a symmetric fill-reducing ordering and no pivoting: ``K = L U`` with the signs of
             diag(U) = the inertia. Always available (pip-installable everywhere). Without pivoting the
             factorisation of an *indefinite* matrix can be inaccurate or break down (then ``factorise`` returns -1:
             "not positive definite, count unknown"); for positive definite matrices it is a stable Cholesky-like
             factorisation. Slowest of the three.
    'mumps'  MUMPS LDL^T with pivoting (``pip`` cannot install it; ``conda install -c conda-forge python-mumps``).
             The symbolic analysis (ordering) is done once per sparsity pattern.
    'cudss'  NVIDIA cuDSS LDL^T on the GPU (``pip install nvmath-python[cu12] nvidia-cudss-cu12 cupy-cuda12x``),
             analysis once per pattern; 3-5 x faster than MUMPS per factorisation on a laptop GPU.

All three are serial (one MPI process).
"""
import numpy as np
import scipy.sparse as sparse
import scipy.sparse.linalg as sparse_linalg


def symmetric_solver(backend='auto'):
    """A symmetric sparse direct solver: 'auto' (MUMPS if installed, else SciPy), 'scipy', 'mumps' or 'cudss'."""
    if backend == 'auto':
        try:
            import mumps  # noqa: F401
            backend = 'mumps'
        except ImportError:
            backend = 'scipy'
    return {'scipy': SciPySolver, 'mumps': MUMPSSolver, 'cudss': CuDSSSolver}[backend]()


class SymmetricSolver:
    """Common part: iterative refinement of the solve. A backend implements ``factorise`` and ``solve_once``."""

    name = None

    def factorise(self, K):
        """Factorise the symmetric sparse matrix K; returns the number of negative eigenvalues of K (-1: unknown,
        K is not positive definite)."""
        raise NotImplementedError

    def solve_once(self, b):
        """K^-1 b with the factors, without refinement."""
        raise NotImplementedError

    def solve(self, b, rtol=1e-12, max_refinements=10):
        """K^-1 b with iterative refinement (strong material contrast costs digits in the factorisation)."""
        y = self.solve_once(b)
        for _ in range(max_refinements):
            residual = b - self.K @ y
            if np.abs(residual).max() <= rtol * np.abs(b).max():
                break
            y = y + self.solve_once(residual)
        return y


class SciPySolver(SymmetricSolver):
    """SuperLU (scipy.sparse.linalg.splu) with a symmetric ordering and no pivoting: the signs of diag(U) give the
    inertia of K (for the unpivoted factorisation K = L D L^T, D = diag(U))."""

    name = 'scipy'

    def factorise(self, K):
        self.K = sparse.csc_matrix(K)
        try:
            self.lu = sparse_linalg.splu(self.K, permc_spec='MMD_AT_PLUS_A', diag_pivot_thresh=0.0,
                                         options=dict(SymmetricMode=True))
        except RuntimeError:                                    # exactly singular pivot: not positive definite
            self.lu = None
            return -1
        if not np.array_equal(self.lu.perm_r, self.lu.perm_c):  # (a row interchange happened: count not valid)
            return -1 if np.any(self.lu.U.diagonal() <= 0) else 0
        return int(np.count_nonzero(self.lu.U.diagonal() < 0))

    def solve_once(self, b):
        if self.lu is None:
            raise RuntimeError('SciPySolver: the last factorisation failed (matrix not positive definite)')
        return self.lu.solve(b)


class MUMPSSolver(SymmetricSolver):
    """MUMPS LDL^T (python-mumps): pivoting, inertia INFOG(12), the symbolic analysis once per sparsity pattern."""

    name = 'mumps'

    def __init__(self, pivot_tol=0.1):
        import mumps
        self.context = mumps.Context()
        self.pivot_tol = pivot_tol
        self.pattern = None

    def factorise(self, K):
        self.K = sparse.csr_matrix(K)
        pattern = (self.K.shape, self.K.nnz, hash(self.K.indices.tobytes()), hash(self.K.indptr.tobytes()))
        self.context.set_matrix(self.K, symmetric=True)
        self.context.factor(ordering='metis', pivot_tol=self.pivot_tol, reuse_analysis=pattern == self.pattern)
        self.pattern = pattern
        return int(self.context.mumps_instance.infog[12])

    def solve_once(self, b):
        return self.context.solve(b)


class CuDSSSolver(SymmetricSolver):
    """NVIDIA cuDSS LDL^T on the GPU (nvmath-python): the upper triangle, the analysis (plan) once per pattern, the
    inertia from the factorisation."""

    name = 'cudss'

    def __init__(self):
        import cupy
        import cupyx.scipy.sparse
        from nvmath.sparse import advanced
        self.cupy, self.cupy_sparse, self.advanced = cupy, cupyx.scipy.sparse, advanced
        self.solver, self.pattern = None, None

    def factorise(self, K):
        self.K = sparse.csr_matrix(K)
        upper = sparse.triu(self.K).tocsr()
        pattern = (upper.shape, upper.nnz, hash(upper.indices.tobytes()), hash(upper.indptr.tobytes()))
        if pattern != self.pattern:                             # new pattern: copy to the GPU and analyse
            if self.solver is not None:
                self.solver.free()
            self.matrix = self.cupy_sparse.csr_matrix(upper)
            self.rhs = self.cupy.zeros(upper.shape[0])
            options = self.advanced.DirectSolverOptions(
                sparse_system_type=self.advanced.DirectSolverMatrixType.SYMMETRIC,
                sparse_system_view=self.advanced.DirectSolverMatrixViewType.UPPER)
            self.solver = self.advanced.DirectSolver(self.matrix, self.rhs, options=options)
            self.solver.plan()
            self.pattern = pattern
        else:                                                   # same pattern: only the values change
            self.matrix.data[...] = self.cupy.asarray(upper.data)
        self.solver.factorize()
        return int(self.solver.factorization_info.inertia[1])  # (positive, negative)

    def solve_once(self, b):
        self.rhs[...] = self.cupy.asarray(b)
        return self.cupy.asnumpy(self.solver.solve())
