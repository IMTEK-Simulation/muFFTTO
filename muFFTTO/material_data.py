"""
Material data that is a sum of scalar fields times constant tensors, ``C(x) = sum_k s_k(x) C_k``.

:class:`ScaledMaterial` is accepted by :class:`muFFTTO.domain.Discretization` wherever material data is applied; it
lives in its own module (no muFFTTO imports apart from :mod:`tensor_operations`) so that both
:mod:`muFFTTO.domain` and :mod:`muFFTTO.material_models` can import it without an import cycle. It is still
available as ``muFFTTO.domain.ScaledMaterial``.
"""
import numpy as np
from muGrid import Field

from muFFTTO import tensor_operations
from muFFTTO.tensor_operations import quad_point_blocks


def _fused_tensor_kernel(xp, tensors, transposed):
    """CuPy elementwise kernel ``out_ij = sum_t s_t sum_kl C_t[i, j, k, l] g_kl`` (``g_kl C_t[k, l, i, j]`` if
    transposed) for constant tensors ``C_t``: arguments ``g_00 .. g_dd, s_0 .. s_T, out_00 .. out_dd``. All inputs
    of a point are read before its outputs are written, so ``out`` may be the gradient itself."""
    d = tensors[0].shape[0]
    pairs = [(i, j) for i in range(d) for j in range(d)]
    inputs = ', '.join([f'T g{i}{j}' for i, j in pairs] + [f'T s{t}' for t in range(len(tensors))])
    outputs = ', '.join(f'T o{i}{j}' for i, j in pairs)
    code = [f'T a{k}{l} = g{k}{l};' for k, l in pairs]
    for i, j in pairs:
        terms = []
        for t, tensor in enumerate(tensors):
            entries = [(tensor[k, l, i, j] if transposed else tensor[i, j, k, l], k, l) for k, l in pairs]
            contraction = ' + '.join(f'({float(c)!r}) * a{k}{l}' for c, k, l in entries if c != 0)
            if contraction:
                terms.append(f's{t} * ({contraction})')
        code.append(f'T r{i}{j} = ' + (' + '.join(terms) if terms else '0') + ';')
    code += [f'o{i}{j} = r{i}{j};' for i, j in pairs]
    return xp.ElementwiseKernel(inputs, outputs, ' '.join(code), 'scaled_material_apply')


class ScaledMaterial:
    """Material data ``C(x) = sum_k s_k(x) C_k`` with scalar scale fields ``s_k`` and constant tensors ``C_k``.

    Avoids storing the full tangent ``C_ijkl`` (elasticity, ``d^4`` values) or ``A_ij`` (conductivity, ``d^2``) per
    quadrature point when the material is a scalar multiple of fixed tensors, e.g. SIMP with a void material
    ``C_void = kappa C_0``: ``C(rho) = s(rho) C_0`` -- or, for any void, ``rho^p (C_1 - C_0) + 1 * C_0`` (two terms).
    Accepted wherever material data is applied (:meth:`Discretization.apply_material_data_mugrid` and the methods
    built on it: system matrix, right-hand side, homogenized stress, stress field, Jacobi preconditioner).

    Parameters
    ----------
    terms : list of (scale, tensor)
        ``scale``: float, ndarray broadcastable to ``[q, x, y(, z)]``, a scalar quadrature muGrid Field
        (``[1, 1, q, x, y(, z)]``, read at use), or a callable returning such an array, evaluated at every application
        (no stored field, e.g. the dual's ``C_0 A^-1 C_0 = C_0 / s(x)`` computed from the field of ``A = s(x) C_0``);
        ``tensor``: ``[d, d, d, d]`` (elasticity) or ``[d, d]`` (conductivity).
    """

    def __init__(self, terms):
        self.terms = list(terms)
        # C_ijkl = C_ijlk for every (elasticity) tensor: then C : grad u = C : sym(grad u), and the small-strain
        # system matrix can skip the symmetrisation of the gradient
        self.minor_symmetric = all(
            np.ndim(tensor) == 4
            and np.allclose(tensor_operations.to_host(tensor), tensor_operations.to_host(tensor).transpose(0, 1, 3, 2))
            for _, tensor in self.terms)

    @staticmethod
    def _scale_array(scale):
        if callable(scale):
            scale = scale()
        if isinstance(scale, Field):
            return scale.s[0, 0]
        return scale

    def apply(self, gradient, transposed=False, out=None, low_memory=False):
        """``sum_k s_k (C_k : g)``; ``g``: ``[d, d, q, x, y(, z)]`` (elasticity) or ``[1, d, q, ...]`` (conductivity).
        ``transposed``: contract the first index pair instead (``g : C_k``).
        ``out``: array of the shape of ``g`` to write the result to (may be ``g`` itself); a new array if None.
        ``low_memory``: evaluate one quadrature point at a time instead of all at once (same result, smaller
        temporaries, slower); the library passes :attr:`Discretization.low_memory`."""
        xp = tensor_operations.get_array_module(gradient)
        shape = gradient.shape
        nc = 2                                    # component axes: (i, j) elasticity, (u, i) conductivity
        ncomp = shape[0] * shape[1]
        if out is None:
            out = xp.empty_like(gradient)
        if xp is not np and all(np.ndim(tensor) == 4 for _, tensor in self.terms):
            return self._apply_fused(xp, gradient, transposed, out)       # GPU: one kernel, no temporaries
        # constant tensors as (ncomp x ncomp) matrices acting on the components of a point, stored in muGrid's
        # memory order: the first index fastest, i.e. as [.., j, i]
        matrices = []
        for scale, tensor in self.terms:
            tensor = xp.asarray(tensor)
            if tensor.ndim == 4:                  # [d,d,d,d]: C_(ij)(kl), rows and columns in the (j, i) order of g
                matrix = tensor.transpose(1, 0, 3, 2).reshape(ncomp, ncomp)
            else:                                 # [d,d] acting on [u=1, d]: A_ij
                matrix = tensor
            matrices.append(matrix if transposed else matrix.T)
        scales = [self._scale_array(scale) for scale, _ in self.terms]
        scales = [scale if np.isscalar(scale) else xp.broadcast_to(xp.asarray(scale), shape[nc:]) for scale in scales]
        for q in quad_point_blocks(shape[nc], low_memory):
            # points first, components last: (j, i) merged into one axis (a view in muGrid's memory order), so the
            # constant tensor acts as a (ncomp x ncomp) matrix on every point (batched matvec)
            block_shape = gradient[:, :, q].shape[nc:]
            g = xp.moveaxis(gradient[:, :, q], (1, 0), (-2, -1)).reshape(block_shape + (ncomp,))
            result = None
            for scale, matrix in zip(scales, matrices):
                term = g @ matrix
                if not np.isscalar(scale):
                    term *= scale[q][..., None]
                elif scale != 1.0:
                    term *= scale
                if result is None:
                    result = term
                else:
                    result += term
            # back to the component-first layout [i, j, q, x, y(, z)]; g is fully read before out is written
            out[:, :, q] = xp.moveaxis(result.reshape(block_shape + (shape[1], shape[0])), (-2, -1), (1, 0))
        return out

    def apply_to_field(self, gradient_field, transposed=False):
        """:meth:`apply` in place on a whole gradient-sized muGrid field (ghost points included; their values are
        meaningless until the next ghost exchange). On the GPU, with scales that are muGrid fields or numbers, the
        kernel runs over the contiguous component vectors of the buffers (:func:`tensor_operations.point_view`) --
        one pass at memory bandwidth. Otherwise :meth:`apply` on the interior."""
        xp = tensor_operations.get_array_module(gradient_field.s)
        scales = [scale() if callable(scale) else scale for scale, _ in self.terms]
        if (xp is np or not all(np.ndim(tensor) == 4 for _, tensor in self.terms)
                or not all(isinstance(scale, Field) or np.isscalar(scale) for scale in scales)):
            return self.apply(gradient_field.s, transposed=transposed, out=gradient_field.s)
        components = tensor_operations.point_view(gradient_field)             # [d, d, q, points]
        d = components.shape[0]
        components = components.reshape(d, d, -1)
        scales = [components.dtype.type(scale) if np.isscalar(scale)
                  else tensor_operations.point_view(scale).reshape(-1) for scale in scales]
        kernel = self._fused_kernel_for(xp, transposed, components.dtype)
        vectors = [components[i, j] for i in range(d) for j in range(d)]
        kernel(*vectors, *scales, *vectors)
        return gradient_field.s

    def _fused_kernel_for(self, xp, transposed, dtype):
        tensors = [np.asarray(tensor_operations.to_host(tensor), dtype=float) for _, tensor in self.terms]
        key = (transposed, str(dtype)) + tuple(tensor.tobytes() for tensor in tensors)
        if getattr(self, '_fused_key', None) != key:
            self._fused_kernel, self._fused_key = _fused_tensor_kernel(xp, tensors, transposed), key
        return self._fused_kernel

    def _apply_fused(self, xp, gradient, transposed, out):
        """:meth:`apply` on the GPU as one elementwise kernel over the points (elasticity): each point reads its
        d x d gradient components and writes the d x d result, with the constant tensors compiled into the kernel --
        no full-size temporaries (the matrix-product path needs two) and one pass over the field."""
        d = gradient.shape[0]
        scales = [self._scale_array(scale) for scale, _ in self.terms]
        scales = [gradient.dtype.type(scale) if np.isscalar(scale) else xp.asarray(scale) for scale in scales]
        kernel = self._fused_kernel_for(xp, transposed, gradient.dtype)
        kernel(*[gradient[i, j] for i in range(d) for j in range(d)], *scales,
               *[out[i, j] for i in range(d) for j in range(d)])
        return out
