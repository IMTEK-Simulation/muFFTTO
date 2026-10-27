"""
constraints.py
==============
Prescribed values on individual degrees of freedom (dofs) of a nodal field,
e.g. fixed or prescribed displacements of selected nodes.

Notation (François)
-------------------
P     : projection onto the constrained dofs (keeps constrained, zeroes free)
Q=I-P : projection onto the free dofs (keeps free, zeroes constrained)

With prescribed values u_D on the constrained dofs, the linear system
K u = b is solved on the free dofs only,

    Q K Q u = Q b - Q K P(u_D),

and the full solution is u = Q(u) + P(u_D).

Efficiency
----------
The constrained dofs are converted once to integer indices into the local
subdomain (``field.s``), on the device of the field (numpy / cupy). Q only
touches the constrained dofs, so its cost is O(nb constrained dofs), not
O(grid). On the GPU a single scatter kernel is used (CuPy fancy-index setitem
launches several kernels, ~10x slower).
"""

import numpy as np

try:
    import cupy
except ImportError:  # CPU-only installation
    cupy = None

from muFFTTO.tensor_operations import to_host


class NodalConstraints:
    """
    Constrained dofs of a nodal (unknown-sized) field with prescribed values.

    Create from global dof indices (``NodalConstraints(...)``) or from a
    boolean mask on the local subdomain (``NodalConstraints.from_mask(...)``).

    Typical use with a linear solver::

        nodal_constraints = NodalConstraints(discretization, constrained_dofs_global, u_D)
        nodal_constraints.apply_to_rhs(rhs_field, K_fun)                 # Q b - Q K P(u_D)
        conjugate_gradients(hessp=nodal_constraints.project_operator(K_fun),        # Q K Q
                            prec=nodal_constraints.project_preconditioner(M_fun),   # Q M Q
                            b=rhs_field, x=solution_field, ...)
        nodal_constraints.impose_prescribed_values(solution_field)       # u = Q(u) + P(u_D)
    """

    def __init__(self, discretization, constrained_dofs_global, prescribed_values_global=0.0,
                 name='nodal_constraints'):
        """
        Parameters
        ----------
        discretization : muFFTTO.domain.Discretization
        constrained_dofs_global : array_like of int, shape (nb_dofs, 2 + dim)
            One row per constrained dof: (component f, nodal point n, global pixel index x, y[, z]),
            i.e. a global index into a field of shape [f, n, x, y, z]. Every MPI rank passes all
            rows; each rank keeps the dofs of its own subdomain.
        prescribed_values_global : float or array_like, shape (nb_dofs,), optional
            Prescribed values u_D, in the order of ``constrained_dofs_global`` (default 0).
        name : str, optional
            Prefix of the fields allocated by this object (use distinct names for several objects).
        """
        dofs = np.atleast_2d(np.asarray(constrained_dofs_global, dtype=np.intp))
        # global pixel index -> local index of the subdomain of this rank
        dofs_local = dofs.copy()
        dofs_local[:, 2:] -= np.asarray(discretization.fft.subdomain_locations)
        self._is_owned = np.all((dofs_local[:, 2:] >= 0)
                                & (dofs_local[:, 2:] < np.asarray(discretization.fft.nb_subdomain_grid_pts)),
                                axis=1)
        self._nb_dofs_global_input = dofs.shape[0]
        self._setup(discretization, dofs_local[self._is_owned], name)
        self.set_prescribed_values(prescribed_values_global)

    @classmethod
    def from_mask(cls, discretization, constrained_mask, prescribed_values=0.0, name='nodal_constraints'):
        """
        Constrained dofs given by a boolean mask on the local subdomain.

        Parameters
        ----------
        discretization : muFFTTO.domain.Discretization
        constrained_mask : muGrid field or array_like (numpy or cupy), broadcastable to [f, n, x, y, z]
            Nonzero (True) on constrained dofs of the local subdomain (``field.s`` layout), e.g. a
            displacement-sized field with 1 on constrained and 0 on free dofs. A mask of shape
            [x, y, z] constrains all components of all nodal points of the selected pixels.
            Read once here; the projections use the extracted indices, not the mask.
        prescribed_values : float, array_like or muGrid field, broadcastable to [f, n, x, y, z], optional
            Prescribed values u_D; only the entries at constrained dofs are used (default 0).
        name : str, optional
        """
        self = cls.__new__(cls)
        self._is_owned = None
        if hasattr(constrained_mask, 's'):  # muGrid field
            constrained_mask = constrained_mask.s
        shape = discretization.get_unknown_size_field(name=name + '_prescribed_values').s.shape
        mask = np.broadcast_to(to_host(constrained_mask).astype(bool), shape)
        self._setup(discretization, np.argwhere(mask), name)
        self.set_prescribed_values(prescribed_values)
        return self

    def _setup(self, discretization, dofs_local, name):
        self.discretization = discretization
        self.name = name
        self.device = discretization.device
        # P(u_D): prescribed values on constrained dofs, zero on free dofs
        self.prescribed_displacement_field = discretization.get_unknown_size_field(
            name=name + '_prescribed_values')
        self.prescribed_displacement_field.s[...] = 0
        self._K_u_D_field = discretization.get_unknown_size_field(name=name + '_K_prescribed_values')
        self._shape = self.prescribed_displacement_field.s.shape

        self.xp = cupy if self.device == 'gpu' else np
        self._dofs_local = dofs_local  # host copy [dof, (f, n, x, y)]
        self._dof_coordinates = None  # nodal coordinates of the constrained dofs, built on first use
        self.nb_local_constrained_dofs = dofs_local.shape[0]
        self.nb_constrained_dofs = int(discretization.mpi_reduction.sum(np.asarray(self.nb_local_constrained_dofs)))
        # one integer index array per axis of .s, on the device
        self._indices = tuple(self.xp.asarray(axis_indices) for axis_indices in dofs_local.T)
        if self.device == 'gpu':
            # raw indexing maps the logical flat index through the strides of the (ghosted) .s view
            self._flat_indices = cupy.asarray(np.ravel_multi_index(tuple(dofs_local.T), self._shape))
            self._zero_dofs_kernel = cupy.ElementwiseKernel('int64 dof', 'raw float64 x', 'x[dof] = 0',
                                                            'zero_constrained_dofs', no_return=True)

    # ------------------------------------------------------------------
    # prescribed values
    # ------------------------------------------------------------------
    def set_prescribed_values(self, prescribed_values):
        """
        Set new prescribed values u_D (e.g. in a load step); the constrained dofs stay the same.

        Parameters
        ----------
        prescribed_values : float, array_like or muGrid field
            Scalar, or values in the format of the constructor: shape (nb_dofs,) in the order of
            ``constrained_dofs_global`` for ``NodalConstraints(...)``, broadcastable to
            [f, n, x, y, z] for ``from_mask``.
        """
        self._store_prescribed_values(self._get_local_values(prescribed_values))

    def set_prescribed_total_displacement(self, total_displacement, macro_gradient_ij):
        """
        Prescribe the TOTAL displacement E.x + u_fluctuation on the constrained dofs, instead of the fluctuation.

        The unknown of the solver is the fluctuation, so u_D = u_total - E.x is stored. With
        total_displacement = 0 the constrained nodes do not move at all (the macroscopic deformation
        is cancelled there). The problem is then no longer a pure periodic homogenization problem.

        Parameters
        ----------
        total_displacement : float, array_like or muGrid field
            Total displacement on the constrained dofs, in the same format as ``set_prescribed_values``.
        macro_gradient_ij : array_like, shape (dim, dim)
            Macroscopic displacement gradient E; call again when E changes.
        """
        self._store_prescribed_values(self._get_local_values(total_displacement)
                                      - self._get_macro_displacement(macro_gradient_ij))

    def _get_local_values(self, prescribed_values):
        """Values at the local constrained dofs, on the device, from a scalar, array or field."""
        if hasattr(prescribed_values, 's'):  # muGrid field
            prescribed_values = prescribed_values.s
        if np.ndim(prescribed_values) == 0:
            return self.xp.full(self.nb_local_constrained_dofs, float(prescribed_values))
        if self._is_owned is not None:  # global dof list
            values = np.broadcast_to(to_host(prescribed_values).astype(float), (self._nb_dofs_global_input,))
            return self.xp.asarray(values[self._is_owned])
        # field-shaped values, gather at the constrained dofs
        values = self.xp.broadcast_to(self.xp.asarray(prescribed_values, dtype=float), self._shape)
        return values[self._indices]

    def _get_macro_displacement(self, macro_gradient_ij):
        """(E.x)_f at the local constrained dofs (f, n, x, y), x = physical coordinates of the nodal point."""
        if self._dof_coordinates is None:  # coordinates of the constrained nodal points, gathered once
            coordinates_inxyz = to_host(self.discretization.get_nodal_points_coordinates().s)
            self._dof_coordinates = self.xp.asarray(
                coordinates_inxyz[(slice(None),) + tuple(self._dofs_local[:, 1:].T)].T)  # [dof, j]
        macro_gradient_ij = self.xp.asarray(macro_gradient_ij, dtype=float)
        return self.xp.einsum('dj,dj->d', macro_gradient_ij[self._indices[0]], self._dof_coordinates)

    def _store_prescribed_values(self, values):
        self.prescribed_values = values
        self.prescribed_displacement_field.s[...] = 0
        self.impose_prescribed_values(self.prescribed_displacement_field)

    # ------------------------------------------------------------------
    # projections (in place)
    # ------------------------------------------------------------------
    def apply_free_node_projection(self, field_inxyz):
        """Q(u): zero the constrained dofs in place, free dofs are untouched. Cost O(nb constrained dofs).

        Ghost values are not updated (the operators communicate ghosts of their input)."""
        if not self.nb_local_constrained_dofs:
            return
        if self.device == 'gpu':
            self._zero_dofs_kernel(self._flat_indices, field_inxyz.s)
        else:
            field_inxyz.s[self._indices] = 0

    def apply_constrained_node_projection(self, field_inxyz):
        """P(u): keep the constrained dofs, zero the free dofs in place. Cost O(grid)."""
        constrained_values = field_inxyz.s[self._indices]
        field_inxyz.s[...] = 0
        field_inxyz.s[self._indices] = constrained_values

    def impose_prescribed_values(self, field_inxyz):
        """u = Q(u) + P(u_D): overwrite the constrained dofs with the prescribed values, update ghosts."""
        if self.nb_local_constrained_dofs:
            field_inxyz.s[self._indices] = self.prescribed_values
        self.discretization.fft.communicate_ghosts(field_inxyz)

    # ------------------------------------------------------------------
    # constrained linear system
    # ------------------------------------------------------------------
    def project_operator(self, K_fun):
        """
        Return ``hessp(x, Ax)`` applying Q K Q.

        K_fun(x, Ax) : unconstrained operator, writes K x into Ax.
        The search directions of CG already lie in range(Q), so Q(x) in place is a
        cheap safeguard and no copy of x is made.
        """

        def hessp(x, Ax):
            self.apply_free_node_projection(x)
            K_fun(x, Ax)
            self.apply_free_node_projection(Ax)

        return hessp

    def project_preconditioner(self, M_fun):
        """
        Return ``prec(r, z)`` applying Q M Q.

        M_fun(r, z) : unconstrained preconditioner, writes M r into z.
        The residual already lies in range(Q), so only the output is projected.
        """

        def prec(r, z):
            M_fun(r, z)
            self.apply_free_node_projection(z)

        return prec

    def apply_to_rhs(self, rhs_inxyz, K_fun):
        """
        rhs <- Q(rhs) - Q K P(u_D), in place.

        K_fun(x, Ax) : unconstrained operator, writes K x into Ax.
        """
        self.apply_free_node_projection(rhs_inxyz)
        K_fun(self.prescribed_displacement_field, self._K_u_D_field)
        self.apply_free_node_projection(self._K_u_D_field)
        rhs_inxyz.s[...] -= self._K_u_D_field.s

    def __repr__(self):
        return (f"{self.__class__.__name__}(name='{self.name}', "
                f"nb_constrained_dofs={self.nb_constrained_dofs}, device='{self.device}')")
