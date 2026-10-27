"""
2D Hertz test (plane strain): a cylinder cap (radius R) above an elastic block, the gap filled with third medium.

Labels: 0 third medium / void, 1 block, 2 cylinder cap (whole width). Third medium in the gap below the cap; true
void (zero stiffness, frame()) in the top pixel row and one pixel column at each side. The block's bottom row is clamped; the cap's flat top is moved
down rigidly (u_x = 0, u_y = -lam U_MAX). Both bodies have the same material, so Hertz contact is frictionless
(Dundurs beta = 0) and the contact half-width a and the pressure p(x) follow from the force per unit length P:
a = sqrt(4 P R / (pi E*)), p(x) = 2 P / (pi a^2) sqrt(a^2 - x^2), E* = E / (2 (1 - nu^2)).
"""
import os

import numpy as np

BLOCK_TOP = 0.5                           # block: y in [0, BLOCK_TOP], x in [SIDE, 1 - SIDE]
CENTRE_X = 0.5
# GRADE = beta > 0: graded grid, the nodes clustered towards the contact (x = CENTRE_X, y = BLOCK_TOP) by the periodic,
# separable map g(s) = s - beta / (2 pi) sin(2 pi (s - c)) per axis: spacing (1 - beta) h at the contact, (1 + beta) h
# at the cell boundary
GRADE = float(os.environ.get('GRADE', 0.0))


def _g(s, c):
    return s - GRADE / (2 * np.pi) * np.sin(2 * np.pi * (s - c))


def grade(points):
    """reference grid points [2, ...] -> physical points (identity for GRADE = 0)"""
    x, y = np.asarray(points, dtype=float)
    return np.array([_g(x, CENTRE_X), _g(y, BLOCK_TOP)])


# one pixel of true void at the top and at both sides of the cell (zero stiffness): it cuts the periodic links (cap top
# row / block bottom row, both prescribed; block left / right side); with grading: the graded edge pixel
_H = 1.0 / int(os.environ.get('NNN', 128))
FRAME = 1.0 - float(_g(1.0 - _H, BLOCK_TOP))
# SIDE_VOID=0: no side columns, block and cap periodic in x (a row of caps 1 apart on an infinite layer)
SIDE = float(_g(_H, CENTRE_X)) if os.environ.get('SIDE_VOID', '1') == '1' else 0.0
R = float(os.environ.get('R', 1.0))        # cylinder radius
CAP_HALF_WIDTH = 0.5 - SIDE               # the cap spans the whole width (up to the side void columns)
GAP = 0.02                                # initial gap between the cap's lowest point and the block
CAP_TOP = 1.0 - FRAME                     # flat top of the cap (moved down)


def labels(points):
    x, y = np.asarray(points, dtype=float)
    lab = np.zeros(x.shape, dtype=int)
    lab[(y <= BLOCK_TOP) & (x >= SIDE) & (x <= 1.0 - SIDE)] = 1
    bottom = BLOCK_TOP + GAP + R - np.sqrt(np.maximum(R ** 2 - (x - CENTRE_X) ** 2, 0.0))   # the cap's curved bottom
    lab[(np.abs(x - CENTRE_X) <= CAP_HALF_WIDTH) & (y >= bottom) & (y <= CAP_TOP)] = 2
    return lab


def frame(points):
    """true void (where not solid): the top pixel row, one pixel column at each side and beside the cap (above the block,
    outside the cap's width): only the gap below the cap keeps the third medium"""
    x, y = np.asarray(points, dtype=float)
    return (y > CAP_TOP) | (x < SIDE) | (x > 1.0 - SIDE) | ((np.abs(x - CENTRE_X) > CAP_HALF_WIDTH) & (y > BLOCK_TOP))
