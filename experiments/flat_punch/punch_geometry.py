"""
Flat rigid punch (2D, plane strain): a rigid flat punch of half-width B above an elastic block, one pixel row of third
medium between them (under the punch only), pressed down. All faces lie on pixel edges: exact on the regular grid.

Labels: 0 third medium / void, 1 block, 2 punch. Third medium only in the gap below the punch; true void (zero
stiffness, frame()) everywhere else outside the solids: the top pixel row, one pixel column at each side and above the
block beside the punch. Loading: the block's bottom row on rollers (u_y = 0, u_x free, one node pinned in x); every punch
node prescribed (u_x = 0, u_y = -lam U_MAX): a rigid punch. Exact (frictionless rigid punch on a half-space, Sadowsky):
p(x) = P / (pi sqrt(B^2 - x^2)) for |x| < B, infinite at the corners; per pixel the average
P / pi (asin(x2 / B) - asin(x1 / B)) / (x2 - x1).
"""
import os

import numpy as np

_N = int(os.environ.get('NNN', 32))
_H = 1.0 / _N
B_PX = int(os.environ.get('B_PX', 6))            # punch half-width in pixels
GAP_PX = int(os.environ.get('GAP_PX', 1))        # gap in whole pixel rows
CENTRE_X = 0.5
B = B_PX * _H                                    # punch half-width
SIDE = _H                                        # true void: one pixel column at each side
BLOCK_TOP = 0.625                                # block: y in [0, BLOCK_TOP] (a deeper block: B / depth smaller)
GAP = GAP_PX * _H
PUNCH_TOP = 1.0 - _H                             # punch: y in [BLOCK_TOP + GAP, PUNCH_TOP], top row void


def labels(points):
    x, y = np.asarray(points, dtype=float)
    lab = np.zeros(x.shape, dtype=int)
    lab[(x >= SIDE) & (x <= 1.0 - SIDE) & (y <= BLOCK_TOP)] = 1
    lab[(np.abs(x - CENTRE_X) <= B) & (y >= BLOCK_TOP + GAP) & (y <= PUNCH_TOP)] = 2
    return lab


def frame(points):
    """true void (where not solid): the top row, one column at each side and everything above the block outside the
    punch's width; only the gap below the punch keeps the third medium"""
    x, y = np.asarray(points, dtype=float)
    return (y > PUNCH_TOP) | (x < SIDE) | (x > 1.0 - SIDE) | ((np.abs(x - CENTRE_X) > B) & (y > BLOCK_TOP))


def pixel_average(x1, x2, P):
    """exact Sadowsky pressure averaged over [x1, x2] (relative to the punch axis), 0 outside the punch"""
    a, b = np.clip(x1, -B, B), np.clip(x2, -B, B)
    return P / np.pi * (np.arcsin(b / B) - np.arcsin(a / B)) / (x2 - x1)


def sigma_yy_at_depth(x, z, P, nb_theta=40001):
    """exact half-space sigma_yy at depth z below the surface, at x (relative to the punch axis): the Flamant integral
    of the Sadowsky pressure, -(2 z^3 / pi) int p(s) / ((x - s)^2 + z^2)^2 ds, with s = B sin(theta), p ds = P / pi
    dtheta (no singularity left)"""
    theta = np.linspace(-0.5 * np.pi, 0.5 * np.pi, nb_theta)
    x, z = np.asarray(x, dtype=float)[..., None], np.asarray(z, dtype=float)[..., None]
    kernel = 1.0 / ((x - B * np.sin(theta)) ** 2 + z ** 2) ** 2
    return -(2 * z[..., 0] ** 3 / np.pi) * (P / np.pi) * np.trapezoid(kernel, theta, axis=-1)


def top_row_average(edges, P):
    """exact -sigma_yy averaged over the 2 x 2 Gauss points of the block's top pixel row (depths h/2 -+ h/(2 sqrt 3)):
    what p_surface measures"""
    g = 0.5 - 0.5 / np.sqrt(3.0)
    h = edges[1] - edges[0]
    out = np.zeros(len(edges) - 1)
    for a in (g, 1 - g):
        for b in (g, 1 - g):
            out -= 0.25 * sigma_yy_at_depth(edges[:-1] + a * h, b * h, P)
    return out
