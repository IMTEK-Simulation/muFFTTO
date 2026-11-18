"""
Density filter and Heaviside projection for density-based topology optimization.

An alternative to the phase-field regularization of :mod:`muFFTTO.topology_optimization`. The optimizer works on a
nodal design field ``rho`` in [0, 1]; the material sees the *physical* density

    rho_bar = H_beta( F(rho) ),

where ``F`` is the Helmholtz (PDE) filter (Lazarov & Sigmund 2011) and ``H_beta`` the smoothed Heaviside projection
(Wang, Lazarov & Sigmund 2011). The filter imposes a minimum length scale (features thinner than about the filter
radius ``R`` are smeared out), the projection pushes the filtered field back to 0 / 1 and is sharpened during the
optimization (``beta`` continuation). The gradient of an objective ``f(rho_bar)`` follows from the chain rule,

    df/drho = F^T [ H_beta'(F rho) * df/drho_bar ],

and ``F`` is symmetric, so ``F^T = F``.

Helmholtz filter
----------------
``rho_tilde`` solves ``-r^2 Laplace(rho_tilde) + rho_tilde = rho`` on the periodic cell, with ``r = R / (2 sqrt 3)``
(the filter radius ``R`` gives the same length scale as a linear "hat" filter of radius ``R``). Discretized on the
nodal grid with the periodic second-order finite-difference Laplacian ``L`` (in 2D the 5-point stencil, equal to
the stiffness matrix of linear triangles with lumped mass, divided by the nodal volume),

    (I + r^2 L) rho_tilde = rho,

and solved exactly by FFT: ``L`` is diagonal in Fourier space with eigenvalues
``lambda(k) = sum_d 4 sin^2(pi k_d) / h_d^2`` (``k_d`` in cycles per grid point), so

    rho_tilde_hat(k) = rho_hat(k) / (1 + r^2 lambda(k)).

Properties: the mean is preserved (``lambda(0) = 0``), constants are reproduced, ``(I + r^2 L)^-1`` is symmetric
and an M-matrix inverse (non-negative entries, rows summing to 1), so ``0 <= rho <= 1`` gives
``0 <= rho_tilde <= 1``.

Heaviside projection
--------------------
    H_beta(x) = (tanh(beta eta) + tanh(beta (x - eta))) / (tanh(beta eta) + tanh(beta (1 - eta))),
    H_beta'(x) = beta (1 - tanh^2(beta (x - eta))) / (tanh(beta eta) + tanh(beta (1 - eta))),

with threshold ``eta`` in (0, 1). ``H_beta(0) = 0``, ``H_beta(1) = 1``, ``H_beta`` is monotone, tends to the identity
for ``beta -> 0`` (``beta = 0`` is treated as the identity) and to a step at ``eta`` for ``beta -> infinity``.

Usage
-----
    density = density_filter.FilterAndProjection(discretization, radius=3 * pixel_size, beta=1.0)

    def objective(rho_flat):
        rho_bar = density.forward(rho_flat)            # physical density for the material model
        f, df_drho_bar = ...                            # objective and gradient w.r.t. rho_bar
        return f, density.backward(df_drho_bar)         # gradient w.r.t. the design rho

    for beta in [1, 2, 4, 8, 16, 32]:                   # continuation (restart the optimizer each stage)
        density.beta = beta
        ...

Robust formulation
------------------
:class:`RobustFilterAndProjection` projects the same filtered field with several thresholds (eroded ``0.75``,
intermediate ``0.5``, dilated ``0.25``; Wang, Lazarov & Sigmund 2011) and returns the gradient of a weighted sum of
objectives evaluated on all of them. Requiring the eroded and the dilated design to perform as well enforces a
minimum size of solid and void features of the intermediate (blueprint) design, which the plain
:class:`FilterAndProjection` does not guarantee at large ``beta`` (thin hinges can reappear).

All functions work on nodal scalar fields of the discretization (one node per pixel, e.g. ``linear_triangles``,
``bilinear_rectangle``), are MPI-parallel through the distributed FFT of the discretization, and accept either
muGrid fields or arrays of the local shape ``[1, 1, *nb_subdomain_grid_pts]`` (flattened arrays of that size too).
"""
import numpy as np


def helmholtz_filter_radius_to_length(radius):
    """Length parameter ``r`` of the Helmholtz equation for a filter radius ``R``: ``r = R / (2 sqrt 3)``."""
    return radius / (2.0 * np.sqrt(3.0))


def laplacian_eigenvalues(discretization):
    """Eigenvalues ``lambda(k) = sum_d 4 sin^2(pi k_d) / h_d^2`` of the periodic FD Laplacian, local Fourier grid."""
    frequencies = np.asarray(discretization.fft.fftfreq)          # [d, *nb_fourier_subdomain], cycles per grid point
    pixel_size = np.asarray(discretization.pixel_size, dtype=float)
    return sum(4.0 * np.sin(np.pi * frequencies[d]) ** 2 / pixel_size[d] ** 2 for d in range(frequencies.shape[0]))


def heaviside_projection(x, beta, threshold=0.5):
    """Smoothed Heaviside projection ``H_beta(x)`` (Wang, Lazarov & Sigmund 2011); ``beta = 0`` is the identity."""
    x = np.asarray(x, dtype=float)
    if beta == 0:
        return x.copy()
    denominator = np.tanh(beta * threshold) + np.tanh(beta * (1.0 - threshold))
    return (np.tanh(beta * threshold) + np.tanh(beta * (x - threshold))) / denominator


def heaviside_projection_derivative(x, beta, threshold=0.5):
    """Derivative ``dH_beta/dx`` of :func:`heaviside_projection`."""
    x = np.asarray(x, dtype=float)
    if beta == 0:
        return np.ones_like(x)
    denominator = np.tanh(beta * threshold) + np.tanh(beta * (1.0 - threshold))
    return beta * (1.0 - np.tanh(beta * (x - threshold)) ** 2) / denominator


def measure_of_non_discreteness(rho, communicator=None):
    """Sigmund's grey measure ``M_nd = 4 / N sum rho (1 - rho)``: 0 for a 0/1 design, 1 for rho = 0.5 everywhere."""
    rho = np.asarray(rho, dtype=float)
    local = np.array([np.sum(4.0 * rho * (1.0 - rho)), rho.size], dtype=float)
    total = local if communicator is None else np.array([communicator.sum(local[0]), communicator.sum(local[1])])
    return total[0] / total[1]


def length_scale_constraints(rho_tilde, radius_pixels, beta, eroded=0.75, dilated=0.25, threshold=0.5, c=None):
    """Geometric minimum length scale constraints (Zhou, Lazarov, Wang & Sigmund 2015) on the filtered field.

    With the filtered field ``x = F rho``, the blueprint ``rho_bar = H_beta,threshold(x)`` and the indicator of the
    skeleton of solid / void members ``E = exp(-c |grad x|^2)`` (``grad`` in pixel units; ``|grad x|`` vanishes on
    the medial axis of a member), a solid member is thick enough if ``x >= eroded`` on its skeleton, a void if
    ``x <= dilated`` there:

        g_solid = 1/N sum rho_bar E min(x - eroded, 0)^2,
        g_void  = 1/N sum (1 - rho_bar) E min(dilated - x, 0)^2.

    ``g_solid = g_void = 0`` gives the minimum sizes of the eroded / dilated robust designs (about the filter
    radius) for the blueprint, without evaluating the eroded and dilated designs. ``c = R^4`` (R in pixels) by
    default, as proposed by Zhou et al.

    ``rho_tilde`` is an array of the nodal shape ``[1, 1, *nb_grid_pts]`` or ``nb_grid_pts``. Serial only: the
    spatial gradient uses periodic central differences (``np.roll``) on the whole grid.

    Returns ``g_solid, g_void, dg_solid/dx, dg_void/dx`` (gradients w.r.t. ``x`` of the shape of ``rho_tilde``;
    chain to the design with ``F^T``).
    """
    x = np.asarray(rho_tilde, dtype=float)
    c = float(radius_pixels) ** 4 if c is None else float(c)
    axes = tuple(range(2, x.ndim)) if x.ndim > 2 and x.shape[:2] == (1, 1) else tuple(range(x.ndim))

    def derivative(field, axis):                  # periodic central difference (pixel units); D^T = -D
        return 0.5 * (np.roll(field, -1, axis=axis) - np.roll(field, 1, axis=axis))

    gradients = [derivative(x, axis) for axis in axes]
    E = np.exp(-c * sum(g ** 2 for g in gradients))
    rho_bar = heaviside_projection(x, beta, threshold)
    H_prime = heaviside_projection_derivative(x, beta, threshold)
    m_solid = np.minimum(x - eroded, 0.0)
    m_void = np.minimum(dilated - x, 0.0)
    N = x.size
    g_solid = float(np.sum(rho_bar * E * m_solid ** 2)) / N
    g_void = float(np.sum((1.0 - rho_bar) * E * m_void ** 2)) / N

    def gradient(weight_field, pointwise):
        """d/dx of 1/N sum weight_field E (pointwise part given), including the dependence of E on grad x."""
        w = -c * weight_field * E                  # d(weight E)/d(|grad x|^2)
        through_E = sum(-derivative(2.0 * w * g, axis) for g, axis in zip(gradients, axes))
        return (pointwise + through_E) / N

    dg_solid = gradient(rho_bar * m_solid ** 2, H_prime * E * m_solid ** 2 + rho_bar * E * 2.0 * m_solid)
    dg_void = gradient((1.0 - rho_bar) * m_void ** 2, -H_prime * E * m_void ** 2 - (1.0 - rho_bar) * E * 2.0 * m_void)
    return g_solid, g_void, dg_solid, dg_void


class HelmholtzFilter:
    """Periodic Helmholtz (PDE) density filter ``rho_tilde = (I + r^2 L)^-1 rho``, applied exactly by FFT.

    Parameters
    ----------
    discretization : muFFTTO.domain.Discretization
        Discretization with one node per pixel (the design lives on the nodes).
    radius : float
        Filter radius ``R`` (length units of the cell); ``r = R / (2 sqrt 3)``. ``radius = 0`` is the identity.
    name : str, optional
        Prefix of the work fields allocated in the field collections (distinct filters need distinct names).
    """

    def __init__(self, discretization, radius, name='helmholtz_filter'):
        if discretization.nb_nodes_per_pixel != 1:
            raise ValueError('HelmholtzFilter needs one node per pixel (nodal design field)')
        self.discretization = discretization
        self.radius = float(radius)
        r = helmholtz_filter_radius_to_length(self.radius)
        # Fourier multiplier of (I + r^2 L)^-1, including the 1/N of the unnormalised FFT pair
        self.multiplier = discretization.fft.normalisation / (1.0 + r ** 2 * laplacian_eigenvalues(discretization))
        self._real = discretization.field_collection.real_field(name=name + '_real', components=(1,))
        self._fourier = discretization.ffield_collection.complex_field(name=name + '_fourier', components=(1,))

    def apply(self, field):
        """Filtered field; ``field`` is a muGrid field or an array of the local nodal shape (or its flat version).

        Returns an array of the shape of the input (a new array; the input is not modified). The filter is
        symmetric, so ``apply`` is also its transpose (use it for the chain rule of gradients).
        """
        values = field.s if hasattr(field, 's') else np.asarray(field, dtype=float)
        shape = values.shape
        self._real.s[...] = values.reshape(self._real.s.shape)
        self.discretization.fft.fft(self._real, self._fourier)
        self._fourier.s[...] *= self.multiplier
        self.discretization.fft.ifft(self._fourier, self._real)
        self.discretization.fft.communicate_ghosts(self._real)
        return self._real.s.copy().reshape(shape)

    apply_transpose = apply


class FilterAndProjection:
    """Design ``rho`` -> physical density ``rho_bar = H_beta(F rho)`` and the chain rule for gradients.

    Parameters
    ----------
    discretization : muFFTTO.domain.Discretization
        Discretization with one node per pixel.
    radius : float
        Filter radius ``R`` (length units); see :class:`HelmholtzFilter`. 0 switches the filter off.
    beta : float, optional
        Projection sharpness (default 1); 0 switches the projection off. Can be changed between optimizer stages
        (continuation); :meth:`forward` must be called again after a change.
    threshold : float, optional
        Projection threshold ``eta`` in (0, 1) (default 0.5).
    name : str, optional
        Prefix of the work fields.
    """

    def __init__(self, discretization, radius, beta=1.0, threshold=0.5, name='density_filter'):
        if not 0.0 < threshold < 1.0:
            raise ValueError('threshold must lie in (0, 1)')
        self.filter = HelmholtzFilter(discretization, radius, name=name + '_helmholtz') if radius > 0 else None
        self.beta = float(beta)
        self.threshold = float(threshold)
        self._rho_tilde = None

    def filtered(self, rho):
        """``F rho`` (the identity without filter)."""
        rho = np.asarray(rho.s if hasattr(rho, 's') else rho, dtype=float)
        return rho.copy() if self.filter is None else self.filter.apply(rho)

    def forward(self, rho):
        """Physical density ``H_beta(F rho)`` (same shape as ``rho``); stores ``F rho`` for :meth:`backward`."""
        self._rho_tilde = self.filtered(rho)
        return heaviside_projection(self._rho_tilde, self.beta, self.threshold)

    def backward(self, gradient_physical):
        """Gradient w.r.t. the design: ``F^T [H_beta'(F rho) * gradient_physical]``, at the last :meth:`forward`."""
        if self._rho_tilde is None:
            raise RuntimeError('call forward(rho) before backward(...)')
        gradient_physical = np.asarray(gradient_physical.s if hasattr(gradient_physical, 's') else gradient_physical,
                                       dtype=float)
        gradient_filtered = heaviside_projection_derivative(self._rho_tilde, self.beta, self.threshold) * \
            gradient_physical.reshape(self._rho_tilde.shape)
        result = gradient_filtered if self.filter is None else self.filter.apply_transpose(gradient_filtered)
        return result.reshape(gradient_physical.shape)


class RobustFilterAndProjection:
    """Robust formulation: eroded, intermediate and dilated designs ``H_beta,eta_i(F rho)`` of one filtered field.

    Wang, Lazarov & Sigmund (2011): the same filtered field ``F rho`` is projected with several thresholds, by
    default ``eta = 0.75`` (eroded: thinner solid), ``0.5`` (intermediate, the blueprint) and ``0.25`` (dilated:
    thicker solid), and the objective is evaluated for all of them. A solid member or a void that is thinner than
    the length scale set by the filter radius and the threshold gap disappears (or closes) in the eroded (dilated)
    design, so requiring all designs to perform enforces a minimum size of both solid and void features of the
    intermediate design. One-pixel hinges, which the plain filter + projection allows at large ``beta``, are
    therefore excluded. The pointwise order eroded <= intermediate <= dilated holds because ``H`` decreases with
    the threshold.

    Parameters
    ----------
    discretization : muFFTTO.domain.Discretization
        Discretization with one node per pixel.
    radius : float
        Filter radius ``R`` (length units); see :class:`HelmholtzFilter`.
    beta : float, optional
        Projection sharpness (default 1), shared by all designs; can be changed between optimizer stages.
    thresholds : sequence of float, optional
        Projection thresholds in (0, 1), default ``(0.75, 0.5, 0.25)`` (eroded, intermediate, dilated).
    name : str, optional
        Prefix of the work fields.

    Notes
    -----
    :meth:`forward` returns one physical density per threshold. With the objectives ``f_i(rho_bar_i)`` and their
    gradients ``g_i = df_i/drho_bar_i``, :meth:`backward` returns the gradient of ``sum_i w_i f_i`` w.r.t. the design,
    ``F^T sum_i w_i H'_i(F rho) g_i`` (one filter application). For a worst-case formulation (``max_i f_i``) pass
    weights from a smooth maximum (e.g. the softmax of ``f_i``).
    """

    def __init__(self, discretization, radius, beta=1.0, thresholds=(0.75, 0.5, 0.25), name='robust_filter'):
        thresholds = tuple(float(t) for t in thresholds)
        if not all(0.0 < t < 1.0 for t in thresholds):
            raise ValueError('thresholds must lie in (0, 1)')
        self.filter = HelmholtzFilter(discretization, radius, name=name + '_helmholtz') if radius > 0 else None
        self.beta = float(beta)
        self.thresholds = thresholds
        self._rho_tilde = None

    def filtered(self, rho):
        """``F rho`` (the identity without filter)."""
        rho = np.asarray(rho.s if hasattr(rho, 's') else rho, dtype=float)
        return rho.copy() if self.filter is None else self.filter.apply(rho)

    def forward(self, rho):
        """List of physical densities ``H_beta,eta_i(F rho)``, one per threshold (shape of ``rho``)."""
        self._rho_tilde = self.filtered(rho)
        return [heaviside_projection(self._rho_tilde, self.beta, t) for t in self.thresholds]

    def backward(self, gradients_physical, weights=None):
        """Gradient of ``sum_i w_i f_i`` w.r.t. the design from ``g_i = df_i/drho_bar_i`` (default ``w_i = 1``)."""
        if self._rho_tilde is None:
            raise RuntimeError('call forward(rho) before backward(...)')
        if len(gradients_physical) != len(self.thresholds):
            raise ValueError('one gradient per threshold is needed')
        weights = np.ones(len(self.thresholds)) if weights is None else np.asarray(weights, dtype=float)
        shape = np.shape(gradients_physical[0])
        gradient_filtered = np.zeros_like(self._rho_tilde)
        for w, t, g in zip(weights, self.thresholds, gradients_physical):
            g = np.asarray(g.s if hasattr(g, 's') else g, dtype=float).reshape(self._rho_tilde.shape)
            gradient_filtered += w * heaviside_projection_derivative(self._rho_tilde, self.beta, t) * g
        result = gradient_filtered if self.filter is None else self.filter.apply_transpose(gradient_filtered)
        return result.reshape(shape)
