"""
2D cross-section of a pole (tree) next to a house, on the unit cell (periodic; a void frame keeps the images apart).

Labels: 0 third medium (air), 1 ground, 2 walls and ceiling, 3 roof, 4 pole. The walls reach up to the roof's centre
line and a ceiling slab ties their tops, so walls, ceiling and roof form one closed frame. The pole leans TILT towards the house and has a
notch on its tension side (away from the house) above the ground, so it bends, breaks there and falls onto the roof.
Load (in the simulation): the horizontal displacement of the pole top (crown) is prescribed, the ground's bottom row
is clamped.
"""
import numpy as np

GROUND = (0.03, 0.08)                     # ground slab y range (x: whole cell minus the frame)
FRAME = 0.03                              # void frame at the cell sides and top
WALLS = ((0.43, 0.51), (0.83, 0.91))      # x ranges of the two walls (thickness 0.08)
EAVES_Y = 0.40                            # eaves height
CEILING_T = 0.05                          # ceiling slab below EAVES_Y, between the walls
ROOF = ((0.39, 0.42), (0.67, 0.60), (0.95, 0.42))   # left eave, apex, right eave (centre line)
ROOF_T = 0.06                             # roof slab thickness
POLE_BASE = (0.22, GROUND[1])             # pole axis at the ground
POLE_L, POLE_W = 0.64, 0.05               # pole length and width
TILT = np.deg2rad(5.0)                    # lean towards the house
NOTCH = (0.05, 0.02, 0.45)                # notch: height of its centre above the ground, its width, depth / POLE_W
CROWN = 0.04                              # top part of the pole whose displacement is prescribed


def _segment_distance(points, a, b):
    a, b = np.asarray(a).reshape(2, *([1] * (points.ndim - 1))), np.asarray(b).reshape(2, *([1] * (points.ndim - 1)))
    ab = b - a
    t = np.clip(np.sum((points - a) * ab, axis=0) / np.sum(ab * ab), 0.0, 1.0)
    return np.sqrt(np.sum((points - a - t * ab) ** 2, axis=0))


def pole_coordinates(points):
    """along-axis s (0 at the ground) and across-axis n (> 0 towards the house) of points [2, ...]"""
    axis = np.array([np.sin(TILT), np.cos(TILT)])
    normal = np.array([np.cos(TILT), -np.sin(TILT)])
    rel = points - np.asarray(POLE_BASE).reshape(2, *([1] * (points.ndim - 1)))
    return np.tensordot(axis, rel, 1), np.tensordot(normal, rel, 1)


def labels(points):
    points = np.asarray(points, dtype=float)
    x, y = points
    lab = np.zeros(x.shape, dtype=int)
    inner = (x >= FRAME) & (x <= 1.0 - FRAME) & (y <= 1.0 - FRAME)
    lab[inner & (y >= GROUND[0]) & (y < GROUND[1])] = 1
    roof_line = np.interp(x, [p[0] for p in ROOF], [p[1] for p in ROOF])       # roof centre line height at x
    for x0, x1 in WALLS:
        lab[(x >= x0) & (x <= x1) & (y >= GROUND[1]) & (y <= roof_line)] = 2
    lab[(x >= WALLS[0][0]) & (x <= WALLS[1][1]) & (y >= EAVES_Y - CEILING_T) & (y <= EAVES_Y)] = 2
    roof = (_segment_distance(points, ROOF[0], ROOF[1]) <= ROOF_T / 2) | \
           (_segment_distance(points, ROOF[1], ROOF[2]) <= ROOF_T / 2)
    lab[roof] = 3
    s, n = pole_coordinates(points)
    pole = (s >= -0.02) & (s <= POLE_L) & (np.abs(n) <= POLE_W / 2) & (y >= GROUND[1])
    notch = (np.abs(s - NOTCH[0]) <= NOTCH[1] / 2) & (n < -POLE_W / 2 + NOTCH[2] * POLE_W)
    lab[pole & ~notch] = 4
    return np.where(inner, lab, 0)


def crown(points):
    """the pole's top part, whose horizontal displacement is prescribed"""
    s, n = pole_coordinates(np.asarray(points, dtype=float))
    return (labels(points) == 4) & (s >= POLE_L - CROWN)
