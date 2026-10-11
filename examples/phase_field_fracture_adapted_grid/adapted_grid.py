"""
Interface-fitted (adapted) periodic grid for a geometry from muFFTTO.geometry.

    python adapted_grid.py [nnn] [geometry_name] [output.png] [min_jacobian]

plots the regular and the adapted grid and prints the quality of both.

The grid keeps its topology (nnn x nnn pixels = bilinear elements, periodic);
only the nodes move, x = phi(X) = X + d_g(X), so that element edges follow the
interface between the phases. The unknowns, the FFT and the preconditioners of
the solver stay on the regular grid X; the solver pulls its operators back with
F_g = I + grad_X d_g and J_g = det F_g at the quadrature points (see the
contact-fracture example in this folder). The pull-back only needs J_g > 0 at
the 2 x 2 Gauss points, so elements may degenerate towards triangles (three
nodes on one line, a corner Jacobian of 0): that is what lets element edges
follow the sharp corners of the interface.

Algorithm (all on the full grid, the same on every MPI rank):

1. Interface: signed distance s(x) (> 0 in phase 1) from a fine periodic image
   (Euclidean distance transform) and its zero contour as polylines (contourpy).
2. Corners: points of the contour where it turns by more than `corner_angle`
   over a length of ~h/3 (the cusps where two ligaments cross). Each corner gets
   the nearest grid node, which is fixed there.
3. Edges: for every grid edge crossed by the interface (its end nodes in
   different phases) the end node closer to the crossing moves along the edge
   onto the crossing point (found by bisection with the geometry indicator
   itself). Nodes stay on their grid lines, so neighbouring elements can
   degenerate towards triangles but do not fold.
4. Relax: the free nodes are smoothed by periodic Laplace (Jacobi) iterations of
   the node displacement, the snapped nodes stay fixed.
5. Check: J at the Gauss points >= `min_jacobian` h^2 and at the element corners
   >= 0 (no fold; 0 = three nodes on a line is allowed). For every element that
   fails, the snapped edge node with the largest move is released (corner nodes
   only if no edge node is left in that element); repeat 4-5.
6. Phase per element: majority of the element area (samples of the bilinear map).

Array layout as in muFFTTO: nodal fields [xy, x, y], node (i, j) at X = (i, j) h,
pixel (i, j) has the corners (i, j), (i+1, j), (i, j+1), (i+1, j+1) (periodic).
"""
import os
import sys

import numpy as np
from scipy import ndimage

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from muFFTTO import geometry  # noqa: E402

GAUSS = 0.5 - 0.5 / np.sqrt(3.0)          # 2-point Gauss abscissae in [0, 1]: GAUSS, 1 - GAUSS


def indicator(geometry_name, points, geometry_parameters):
    """Phase (0 / 1) of the geometry at points [2, ...] (any position, periodic in [0, 1))."""
    return geometry.get(geometry_name, np.mod(points, 1.0), **geometry_parameters)


def signed_distance_image(geometry_name, fine, geometry_parameters):
    """Signed distance to the interface (> 0 in phase 1) on a fine x fine image of pixel centres."""
    centres = (np.arange(fine) + 0.5) / fine
    phase = indicator(geometry_name, np.array(np.meshgrid(centres, centres, indexing='ij')),
                      geometry_parameters) > 0.5
    tiled = np.tile(phase, (3, 3))
    inside = ndimage.distance_transform_edt(tiled)[fine:2 * fine, fine:2 * fine]
    outside = ndimage.distance_transform_edt(~tiled)[fine:2 * fine, fine:2 * fine]
    # distance from a pixel centre to the interface ~ distance to the nearest other-phase centre - 1/2
    return np.where(phase, inside - 0.5, -(outside - 0.5)) / fine


def periodic_interpolation(image, points):
    """Bilinear interpolation of a periodic image of pixel-centre values at points [2, ...]."""
    fine = image.shape[0]
    u = np.mod(points, 1.0) * fine - 0.5
    i0 = np.floor(u).astype(int)
    t = u - i0
    value = 0.0
    for di in (0, 1):
        for dj in (0, 1):
            weight = (t[0] if di else 1 - t[0]) * (t[1] if dj else 1 - t[1])
            value = value + weight * image[(i0[0] + di) % fine, (i0[1] + dj) % fine]
    return value


def interface_corners(sdf, h, corner_angle=35.0):
    """Sharp corners of the zero contour of the periodic image sdf, as points [k, 2] in [0, 1)."""
    import contourpy
    fine = sdf.shape[0]
    pad = fine // 4
    padded = np.pad(sdf, pad, mode='wrap')
    coordinate = (np.arange(-pad, fine + pad) + 0.5) / fine
    lines = contourpy.contour_generator(coordinate, coordinate, padded.T).lines(0.0)   # [x, y] points
    window = h / 3.0
    corners = []
    for line in lines:
        segment = np.linalg.norm(np.diff(line, axis=0), axis=1)
        length = np.r_[0.0, np.cumsum(segment)]
        if length[-1] < 4 * window:
            continue
        step = 1.0 / fine
        s = np.arange(0.0, length[-1], step)
        points = np.c_[np.interp(s, length, line[:, 0]), np.interp(s, length, line[:, 1])]
        k = max(3, int(round(window / step)))
        if np.linalg.norm(line[0] - line[-1]) < 2 * step:          # closed loop: continue over the start
            points = np.r_[points[-k:], points, points[:k]]
        before = points[k:-k] - points[:-2 * k]
        after = points[2 * k:] - points[k:-k]
        cosine = np.sum(before * after, axis=1) / (np.linalg.norm(before, axis=1) * np.linalg.norm(after, axis=1))
        turn = np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))
        for m in np.nonzero(turn > corner_angle)[0]:
            if turn[m] >= turn[max(0, m - k):m + k + 1].max():                 # local maximum
                corners.append(points[m + k])
    corners = np.mod(np.array(corners).reshape(-1, 2), 1.0)
    unique = []                                                                 # the padding repeats corners
    for c in corners:
        if all(np.linalg.norm(np.mod(c - u + 0.5, 1.0) - 0.5) > h / 4 for u in unique):
            unique.append(c)
    return np.array(unique).reshape(-1, 2)


def element_corners(nodes):
    """Corner positions p00, p10, p01, p11 [2, nx, ny] of all elements (periodic images added)."""
    nx, ny = nodes.shape[1:]
    shift_x = np.zeros((2, nx, ny))
    shift_x[0, -1, :] = 1.0
    shift_y = np.zeros((2, nx, ny))
    shift_y[1, :, -1] = 1.0
    p10 = np.roll(nodes, -1, axis=1) + shift_x
    p01 = np.roll(nodes, -1, axis=2) + shift_y
    p11 = np.roll(np.roll(nodes, -1, axis=1), -1, axis=2) + shift_x + shift_y
    return nodes, p10, p01, p11


def bilinear(corners, a, b):
    """Position and Jacobian of the bilinear map at the local coordinates (a, b) in [0, 1]^2."""
    p00, p10, p01, p11 = corners
    x = (1 - a) * (1 - b) * p00 + a * (1 - b) * p10 + (1 - a) * b * p01 + a * b * p11
    dx_da = (1 - b) * (p10 - p00) + b * (p11 - p01)
    dx_db = (1 - a) * (p01 - p00) + a * (p11 - p10)
    return x, dx_da[0] * dx_db[1] - dx_da[1] * dx_db[0]


def jacobians(nodes, h):
    """(J / h^2 at the 4 Gauss points, J / h^2 at the 4 corners) of all elements."""
    corners = element_corners(nodes)
    gauss = np.array([bilinear(corners, a, b)[1] for a in (GAUSS, 1 - GAUSS) for b in (GAUSS, 1 - GAUSS)])
    corner = np.array([bilinear(corners, a, b)[1] for a in (0.0, 1.0) for b in (0.0, 1.0)])
    return gauss / h ** 2, corner / h ** 2


def phase_fractions(nodes, geometry_name, geometry_parameters, samples=8):
    """Area of every element and the fraction of it in phase 1 (samples^2 points of the bilinear map)."""
    corners = element_corners(nodes)
    t = (np.arange(samples) + 0.5) / samples
    area, phase_area = 0.0, 0.0
    for a in t:
        for b in t:
            x, jac = bilinear(corners, a, b)
            area = area + jac
            phase_area = phase_area + jac * indicator(geometry_name, x, geometry_parameters)
    return area / samples ** 2, phase_area / np.where(area != 0, area, 1.0)


def adapt_grid(nnn, geometry_name, geometry_parameters=None, smoothing_iterations=300,
               min_jacobian=0.05, corner_angle=35.0, fine=None, verbose=True):
    """Node displacement d_g [2, nnn, nnn] of the interface-fitted grid and the element phases [nnn, nnn].

    Returns (d_g, phase, info), info: dict with quality measures.
    """
    geometry_parameters = geometry_parameters or {}
    h = 1.0 / nnn
    fine = fine or max(512, 8 * nnn)
    sdf = signed_distance_image(geometry_name, fine, geometry_parameters)

    X = np.array(np.meshgrid(np.arange(nnn) * h, np.arange(nnn) * h, indexing='ij'))
    target = np.array(X, copy=True)

    # 2. corners: the nearest node to each corner is fixed on it
    corners = interface_corners(sdf, h, corner_angle)
    is_corner = np.zeros((nnn, nnn), dtype=bool)
    for c in sorted(corners, key=lambda c: np.min(np.linalg.norm(np.mod(c[:, None, None] - X + 0.5, 1.0) - 0.5,
                                                                    axis=0))):
        offset = np.mod(c[:, None, None] - X + 0.5, 1.0) - 0.5          # minimum image
        distance = np.where(is_corner, np.inf, np.linalg.norm(offset, axis=0))
        i, j = np.unravel_index(np.argmin(distance), distance.shape)
        if distance[i, j] < 0.75 * h:
            is_corner[i, j] = True
            target[:, i, j] = X[:, i, j] + offset[:, i, j]

    # 3. edges: every grid edge crossed by the interface (end nodes in different phases) moves its
    # end node closer to the crossing along the edge onto the crossing point (bisection with the
    # exact indicator); a node on several crossed edges takes the nearest crossing
    phase_node = indicator(geometry_name, X, geometry_parameters) > 0.5
    best = np.full((nnn, nnn), np.inf)
    for axis in (0, 1):
        unit = np.zeros((2, 1, 1))
        unit[axis] = h
        crossed = phase_node != np.roll(phase_node, -1, axis=axis)
        start = X[:, crossed]
        a, b = np.zeros(start.shape[1]), np.ones(start.shape[1])            # start + t unit, t in [0, 1]
        phase_start = phase_node[crossed]
        for _ in range(30):
            mid = 0.5 * (a + b)
            same = (indicator(geometry_name, start + mid * unit[:, 0], geometry_parameters) > 0.5) == phase_start
            a, b = np.where(same, mid, a), np.where(same, b, mid)
        t = 0.5 * (a + b)
        ii, jj = np.nonzero(crossed)
        for i, j, tk in zip(ii, jj, t):
            # the closer end: the start node (t <= 1/2) or its neighbour along the axis
            node = (i, j) if tk <= 0.5 else (((i + 1) % nnn, j) if axis == 0 else (i, (j + 1) % nnn))
            distance = tk * h if tk <= 0.5 else (1 - tk) * h
            if not is_corner[node] and distance < best[node]:
                best[node] = distance
                target[:, node[0], node[1]] = X[:, i, j] + tk * unit[:, 0, 0]
                if node != (i, j):                                          # wrapped neighbour: minimum image
                    target[:, node[0], node[1]] = X[:, node[0], node[1]] - (1 - tk) * unit[:, 0, 0]
    is_edge = np.isfinite(best)

    # 4.-5. relax, check, release
    released = 0
    released_mask = np.zeros((nnn, nnn), dtype=bool)
    while True:
        fixed = is_corner | is_edge
        d_g = np.where(fixed, target - X, 0.0)
        for _ in range(smoothing_iterations):
            mean = 0.25 * (np.roll(d_g, 1, 1) + np.roll(d_g, -1, 1) + np.roll(d_g, 1, 2) + np.roll(d_g, -1, 2))
            d_g = np.where(fixed, d_g, mean)
        gauss, corner = jacobians(X + d_g, h)
        bad = (gauss.min(axis=0) < min_jacobian) | (corner.min(axis=0) < -1e-12)
        if not bad.any():
            break
        move = np.linalg.norm(target - X, axis=0)
        released_before = released
        for i, j in zip(*np.nonzero(bad)):
            nodes = [((i + p) % nnn, (j + q) % nnn) for p in (0, 1) for q in (0, 1)]
            candidates = [c for c in nodes if is_edge[c]] or [c for c in nodes if is_corner[c]]
            if candidates:
                worst = max(candidates, key=lambda c: move[c])
                is_edge[worst] = is_corner[worst] = False
                released_mask[worst] = True
                released += 1
        if released == released_before:
            print(f'warning: {int(bad.sum())} invalid elements without snapped nodes remain')
            break

    # 6. phase by area majority; quality
    area, fraction = phase_fractions(X + d_g, geometry_name, geometry_parameters)
    phase = (fraction > 0.5).astype(float)
    area_regular, fraction_regular = phase_fractions(X, geometry_name, geometry_parameters)
    gauss, corner = jacobians(X + d_g, h)
    info = {'nb_corners_found': len(corners), 'nb_corner_nodes': int(is_corner.sum()),
            'nb_edge_nodes': int(is_edge.sum()), 'nb_released': released,
            'min_gauss_J': float(gauss.min()), 'min_corner_J': float(corner.min()),
            'max_move': float(np.abs(d_g).max() / h),
            'exact_fraction': float(np.mean(sdf > 0)),
            'adapted_fraction': float(np.sum(area * phase)),
            'regular_fraction': float(np.sum(area_regular * (fraction_regular > 0.5))),
            'misassigned_adapted': float(np.sum(area * np.minimum(fraction, 1 - fraction))),
            'misassigned_regular': float(np.sum(area_regular * np.minimum(fraction_regular, 1 - fraction_regular))),
            'nb_cut_adapted': int(np.sum((fraction > 0.02) & (fraction < 0.98))),
            'nb_cut_regular': int(np.sum((fraction_regular > 0.02) & (fraction_regular < 0.98)))}
    info['corners'] = corners
    info['snapped_nodes'] = X + np.where(is_corner | is_edge, d_g, np.nan)
    info['released_nodes'] = X + np.where(released_mask, target - X, np.nan)   # where they were meant to go
    if verbose:
        print(f'adapted grid {nnn}^2, {geometry_name}: {info["nb_corners_found"]} corners found, '
              f'{info["nb_corner_nodes"]} corner + {info["nb_edge_nodes"]} edge nodes snapped ({released} released); '
              f'min J/h^2 Gauss {info["min_gauss_J"]:.3f}, corner {info["min_corner_J"]:.3f}; '
              f'max |d_g|/h {info["max_move"]:.3f}')
        print(f'  solid fraction: exact {info["exact_fraction"]:.4f}, adapted {info["adapted_fraction"]:.4f}, '
              f'regular {info["regular_fraction"]:.4f}; misassigned area: adapted {info["misassigned_adapted"]:.4f}, '
              f'regular {info["misassigned_regular"]:.4f}; cut elements: adapted {info["nb_cut_adapted"]}, '
              f'regular {info["nb_cut_regular"]}')
    return d_g, phase, info


def plot_grid(ax, d_g, phase, color_solid=(0.35, 0.35, 0.35), color_medium=(0.75, 0.87, 1.0)):
    """Elements of the grid X + d_g coloured by phase (periodic corners added)."""
    from matplotlib.colors import ListedColormap
    nnn = d_g.shape[1]
    h = 1.0 / nnn
    i, j = np.meshgrid(np.arange(nnn + 1), np.arange(nnn + 1), indexing='ij')
    x = i * h + d_g[0][i % nnn, j % nnn]
    y = j * h + d_g[1][i % nnn, j % nnn]
    ax.pcolormesh(x, y, phase, cmap=ListedColormap([color_medium, color_solid]), vmin=0, vmax=1,
                  edgecolors='k', linewidth=0.2, shading='flat')
    ax.set_aspect('equal')


if __name__ == '__main__':
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    nnn = int(sys.argv[1]) if len(sys.argv) > 1 else 32
    name = sys.argv[2] if len(sys.argv) > 2 else 'sinusoidal_ligaments'
    output = sys.argv[3] if len(sys.argv) > 3 else f'adapted_grid_{name}_N{nnn}.png'
    options = {}
    if len(sys.argv) > 4:
        options['min_jacobian'] = float(sys.argv[4])
    d_g, phase, info = adapt_grid(nnn, name, **options)
    X = np.array(np.meshgrid(np.arange(nnn) / nnn, np.arange(nnn) / nnn, indexing='ij'))
    regular = (phase_fractions(X, name, {})[1] > 0.5).astype(float)
    fine = 1024
    centres = (np.arange(fine) + 0.5) / fine
    exact = indicator(name, np.array(np.meshgrid(centres, centres, indexing='ij')), {})
    fig, axes = plt.subplots(1, 2, figsize=(13, 6.6))
    plot_grid(axes[0], np.zeros_like(d_g), regular)
    plot_grid(axes[1], d_g, phase)
    labels = [f'regular {nnn}$^2$: solid {info["regular_fraction"]:.4f}, misassigned {info["misassigned_regular"]:.4f}',
              f'adapted {nnn}$^2$: solid {info["adapted_fraction"]:.4f}, misassigned {info["misassigned_adapted"]:.4f}']
    for ax, title in zip(axes, labels):
        ax.contour(centres, centres, exact.T, levels=[0.5], colors='r', linewidths=0.8)
        ax.set_title(title + f'\nexact solid {info["exact_fraction"]:.4f}; red: exact interface', fontsize=10)
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
    axes[1].plot(*info['corners'].T, 'o', mfc='none', mec='C2', ms=6, label='detected corners')
    axes[1].plot(*info['snapped_nodes'].reshape(2, -1), '.', color='C0', ms=3, label='snapped nodes')
    axes[1].plot(*info['released_nodes'].reshape(2, -1), 'x', color='m', ms=5, label='released (target)')
    axes[1].legend(loc='upper right', fontsize=8)
    fig.tight_layout()
    fig.savefig(output, dpi=110)
    print('wrote', output)
