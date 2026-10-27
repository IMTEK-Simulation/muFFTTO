"""
Solid elastic disc squeezed between two rigid flat plates (2D, plane strain), third medium in the two gaps.

Labels: 0 third medium / void, 1 disc, 2 bottom plate, 3 top plate. The plate faces lie on grid lines (exact on any
grid of this family); the disc is fitted with the zecevic grid. Third medium only in the two gaps near the contacts
(|x - CENTRE_X| < MEDIUM_HALF_WIDTH, between a plate face and the disc); true void (zero stiffness, frame()) everywhere
else outside the solids, including the bottom and top pixel rows (they cut the periodic link between the plates).
Loading: both plates rigid (every plate node prescribed): bottom u = (0, +lam U_MAX / 2), top u = (0, -lam U_MAX / 2);
the disc centre node pinned (u = 0; the exact solution is symmetric).
Exact: at each plate, Hertz with one rigid body: a = sqrt(4 P R / (pi E*)), E* = E / (1 - nu^2),
p(x) = 2 P / (pi a^2) sqrt(a^2 - x^2); inside the disc the Brazilian (diametral compression) solution, on the horizontal
diameter (D = 2 R): sigma_xx = 2P/(pi D) ((D^2 - 4x^2)/(D^2 + 4x^2))^2, sigma_yy = -2P/(pi D) (4 D^4/(D^2 + 4x^2)^2 - 1).

GRADE = beta > 0: graded grid, nodes clustered towards x = CENTRE_X and towards the plates (y = 0 / 1, where the
contacts are), by the periodic separable map g(s) = s - beta / (2 pi) sin(2 pi (s - c)), c = 0.5 in x and 0 in y.
"""
import os

import numpy as np

_N = int(os.environ.get('NNN', 32))
_H = 1.0 / _N
CENTRE_X = 0.5
GRADE = float(os.environ.get('GRADE', 0.0))
GAP = float(os.environ.get('GAP', 0.02))         # gap between each plate face and the disc
PLATE_PX = 3                                     # plate thickness in pixel rows
MEDIUM_HALF_WIDTH = 0.2                          # third medium only within this distance of the axis


def _g(s, c):
    return s - GRADE / (2 * np.pi) * np.sin(2 * np.pi * (s - c))


def grade(points):
    """reference grid points [2, ...] -> physical points (identity for GRADE = 0)"""
    x, y = np.asarray(points, dtype=float)
    return np.array([_g(x, CENTRE_X), _g(y, 0.0)])


# one void row at the bottom and one at the top (they cut the periodic link between the plates); symmetric about
# y = 0.5, so the disc centre is a grid node
BOTTOM_VOID = float(_g(_H, 0.0))                 # bottom plate: y in [BOTTOM_VOID, BOTTOM_FACE]
BOTTOM_FACE = float(_g((PLATE_PX + 1) * _H, 0.0))
TOP_FACE = float(_g(1.0 - (PLATE_PX + 1) * _H, 0.0))   # top plate: y in [TOP_FACE, TOP_VOID]
TOP_VOID = float(_g(1.0 - _H, 0.0))
CENTRE_Y = 0.5 * (BOTTOM_FACE + TOP_FACE)
R = 0.5 * (TOP_FACE - BOTTOM_FACE) - GAP         # disc radius
SIDE = float(_g(_H, CENTRE_X))                   # one void column at each side (plates end there)


def labels(points):
    x, y = np.asarray(points, dtype=float)
    lab = np.zeros(x.shape, dtype=int)
    plates = (x >= SIDE) & (x <= 1.0 - SIDE)
    lab[plates & (y >= BOTTOM_VOID) & (y <= BOTTOM_FACE)] = 2
    lab[plates & (y >= TOP_FACE) & (y <= TOP_VOID)] = 3
    lab[(x - CENTRE_X) ** 2 + (y - CENTRE_Y) ** 2 <= R ** 2] = 1
    return lab


def frame(points):
    """true void (where not solid): everything except the two gaps near the contacts"""
    x, y = np.asarray(points, dtype=float)
    gap_zone = (np.abs(x - CENTRE_X) < MEDIUM_HALF_WIDTH) & ((y < CENTRE_Y - 0.5 * R) | (y > CENTRE_Y + 0.5 * R)) \
        & (y > BOTTOM_FACE) & (y < TOP_FACE)
    return ~gap_zone


def brazilian_horizontal(x, P):
    """exact sigma_xx, sigma_yy on the horizontal diameter (x relative to the centre), point loads P"""
    D = 2 * R
    s = 2 * P / (np.pi * D)
    return s * ((D ** 2 - 4 * x ** 2) / (D ** 2 + 4 * x ** 2)) ** 2, -s * (4 * D ** 4 / (D ** 2 + 4 * x ** 2) ** 2 - 1)


def hertz(P):
    """contact half-width and peak pressure at each plate (rigid plate, deformable disc)"""
    E_star = 100.0 / (1 - 0.3 ** 2)
    a = np.sqrt(4 * P * R / (np.pi * E_star))
    return a, 2 * P / (np.pi * a)
