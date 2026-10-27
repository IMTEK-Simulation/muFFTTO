"""
Contact patch test (2D, plane strain): two flat blocks, one pixel row of third medium between them, pressed together.

Labels: 0 third medium / void, 1 lower block, 2 upper block. True void (zero stiffness, frame()) in the top pixel row and
one pixel column at each side, so the blocks are free at the sides. Loading: the lower block's bottom row on rollers
(u_y = 0, u_x free, one node pinned in x), the upper block's top row moved down (u_y = -lam U_MAX, u_x free, one node pinned in x:
else the upper block is held sideways only by the soft medium). After the
gap closes the exact solution is uniform: sigma_yy = -p, sigma_xx = sigma_xy = 0 in both blocks, p = P / width.
"""
import os

import numpy as np

_H = 1.0 / int(os.environ.get('NNN', 32))
GAP_PX = int(os.environ.get('GAP_PX', 1))        # gap = whole pixel rows: exact on the regular grid
FRAME = _H                                       # true void: the top pixel row
SIDE = _H                                        # true void: one pixel column at each side
LOWER_TOP = 0.5                                  # lower block: y in [0, LOWER_TOP]
GAP = GAP_PX * _H
UPPER_TOP = 1.0 - FRAME                          # upper block: y in [LOWER_TOP + GAP, UPPER_TOP]
E_RATIO = float(os.environ.get('E_RATIO', 1.0))  # E_upper / E_lower (1: identical blocks; != 1: they slide sideways)


def labels(points):
    x, y = np.asarray(points, dtype=float)
    lab = np.zeros(x.shape, dtype=int)
    inside = (x >= SIDE) & (x <= 1.0 - SIDE)
    lab[inside & (y <= LOWER_TOP)] = 1
    lab[inside & (y >= LOWER_TOP + GAP) & (y <= UPPER_TOP)] = 2
    return lab


def frame(points):
    """true void (where not solid): the top pixel row and one pixel column at each side; the gap keeps the medium"""
    x, y = np.asarray(points, dtype=float)
    return (y > UPPER_TOP) | (x < SIDE) | (x > 1.0 - SIDE)
