"""
Sparse system matrix (Hessian) of a muFFTTO discretization, assembled element by element.

The cell problems in muFFTTO are matrix-free: ``K u = B^T W C B u`` is applied with the gradient operator ``B``,
the quadrature weights ``W`` and the material tangent ``C``. A sparse direct solver needs the entries of ``K``
instead. They come from the same three ingredients, one pixel (element) at a time:

    K_e[b, a] = sum_q  w_q  B_b(q) : tangent(B_a)(q),

with ``B_a`` the gradient, at the quadrature points of a pixel, of the unit displacement of element unknown ``a``
(component c of one node of the pixel: Q1 has 4 nodes per pixel, Q2 9). ``B_a`` is the same for every pixel and is
taken once from the discretization's own gradient operator, so the matrix is exactly the matrix-free operator.
``tangent`` is applied to ``B_a`` on all pixels at once: anything that acts within a pixel (selective reduced
integration with a pixel mean, a pressure projection per pixel) is part of the element matrix automatically.
The element matrices are added into a sparsity pattern built once from the element connectivity.

    assembly = sparse_assembly.SparseAssembly(discretization)
    K = assembly.assemble(tangent)                    # scipy CSR, unknowns in the order of field.s.ravel()

``tangent(dF)`` maps a gradient-shaped array ``[*gradient_shape, q, x, y(, z)]`` to the linearised stress of the
same shape, e.g. ``lambda dF: material.dP(dF)`` or ``lambda dF: np.einsum('ijkl...,kl...->ij...', C, dF)``.

Serial only (one MPI process).
"""
import itertools

import numpy as np
import scipy.sparse as sparse


class SparseAssembly:
    """Element-by-element assembly of the sparse system matrix of ``discretization`` (fixed sparsity pattern)."""

    _count = itertools.count()

    def __init__(self, discretization):
        if discretization.communicator.size > 1:
            raise NotImplementedError('SparseAssembly: serial only (one MPI process)')
        self.discretization = discretization
        dim = discretization.domain_dimension
        tag = f'_sparse_assembly_{next(self._count)}'
        self.unknown_field = discretization.get_unknown_size_field(name='u' + tag)
        self.gradient_field = discretization.get_gradient_size_field(name='grad' + tag)
        u_shape = self.unknown_field.s.shape                    # [component, node, x, y(, z)]
        nb_pixels = u_shape[2:]
        self.nb_unknowns = int(np.prod(u_shape))
        self.weights = np.asarray(discretization.quadrature_weights)
        unknown_index = np.arange(self.nb_unknowns).reshape(u_shape)

        # nodes of a pixel: node type n of the pixel shifted by o in {0, 1}^dim, if it lies in the pixel
        coordinates = np.asarray(discretization.get_nodal_points_coordinates().s)          # [dim, node, x, y(, z)]
        first_pixel = coordinates[(slice(None), slice(None)) + (0,) * dim]                  # [dim, node]
        node_position = np.round((first_pixel - first_pixel[:, :1])
                                 / np.asarray(discretization.pixel_size)[:, None], 6)       # in pixel units
        self.element_unknowns = [(c, node, offset)
                                 for node in range(u_shape[1])
                                 for offset in itertools.product((0, 1), repeat=dim)
                                 if np.all(node_position[:, node] + offset <= 1.0)
                                 for c in range(u_shape[0])]

        # B_a: gradient at the quadrature points of pixel 0 of the unit displacement of element unknown a
        B, unknown_of_pixel = [], []
        for c, node, offset in self.element_unknowns:
            unit = np.zeros(u_shape)
            unit[(c, node) + offset] = 1.0
            B.append(self.gradient(unit)[(Ellipsis,) + (0,) * dim])
            # global number of this element unknown for every pixel (the node sits in pixel p + offset)
            unknown_of_pixel.append(np.roll(unknown_index[c, node], [-o for o in offset], axis=tuple(range(dim))))
        self.B = np.array(B)                                     # [a, *gradient_shape, q]
        unknown_of_pixel = np.array(unknown_of_pixel).reshape(len(B), -1)              # [a, pixels]

        # sparsity pattern: every (b, a) pair of every pixel; position of each element entry in the CSR values
        n = self.nb_unknowns
        keys = (unknown_of_pixel[:, None, :].astype(np.int64) * n + unknown_of_pixel[None, :, :]).ravel()
        unique_keys, self.position = np.unique(keys, return_inverse=True)
        rows, cols = unique_keys // n, unique_keys % n
        indptr = np.concatenate([[0], np.cumsum(np.bincount(rows, minlength=n))])
        self.pattern = sparse.csr_matrix((np.zeros(unique_keys.size), cols, indptr), shape=(n, n))
        self.nb_pixels = nb_pixels

    def gradient(self, u):
        """Gradient at the quadrature points of the unknown array u (as ``field.s``), as an array."""
        self.unknown_field.s[...] = u.reshape(self.unknown_field.s.shape)
        self.discretization.fft.communicate_ghosts(self.unknown_field)
        self.discretization.apply_gradient_operator_mugrid(u_inxyz=self.unknown_field,
                                                           grad_u_ijqxyz=self.gradient_field)
        return self.gradient_field.s.copy()

    def assemble(self, tangent):
        """The sparse matrix B^T W tangent B (CSR, nb_unknowns x nb_unknowns)."""
        nb_a = len(self.element_unknowns)
        gradient_axes = 'ijkl'[:self.B.ndim - 2]
        contraction = f'b{gradient_axes}q,{gradient_axes}q...,q->b...'
        K_e = np.empty((nb_a, nb_a) + tuple(self.nb_pixels))
        for a in range(nb_a):
            dF = np.ascontiguousarray(np.broadcast_to(self.B[a].reshape(self.B[a].shape + (1,) * len(self.nb_pixels)),
                                                      self.gradient_field.s.shape))
            K_e[:, a] = np.einsum(contraction, self.B, np.asarray(tangent(dF)), self.weights)
        K = self.pattern.copy()
        K.data = np.bincount(self.position, weights=K_e.ravel(), minlength=self.pattern.nnz)
        return K


def bordered(K, column):
    """The symmetric matrix [[K, c], [c^T, c_last]] for one extra (e.g. macroscopic) unknown coupled with all
    others: ``column`` = the Hessian applied to the extra unit vector (length n + 1, last entry = its diagonal)."""
    column = np.asarray(column, dtype=float)
    c = sparse.csr_matrix(column[:-1, None])
    return sparse.bmat([[K, c], [c.T, sparse.csr_matrix([[column[-1]]])]], format='csr')
