"""
3D concrete meso-structure for the ASR mesh study: periodic cube 0.1 m, ellipsoidal aggregates with Fuller grading,
ASR gel pockets = cubes of the 64^3 grid (1.5625 mm) inside the aggregates.

* aggregates: cumulative passing by volume P(d) = (d / d_max)^0.5 -> number density n(d) ~ d^-3.5 in 3D,
  d = D_MIN ... D_MAX (default 5 ... 26 mm as in the 2D cell), volume fraction AGG_FRACTION (default 0.35, random
  sequential addition largest first; bounding spheres of the major semi-axis + 1 mm gap, periodic)
* gel pockets: the same pocket / aggregate ratio as the 2D cell (75 pockets of 1.5625 mm in 50 % aggregate area:
  3.7 % of the aggregate), i.e. N_GEL = 0.037 * AGG_FRACTION * 64^3 cells of the 64^3 grid fully inside an
  aggregate (corners inside the ellipsoid shrunk by half a cell), >= 2 cells apart (Chebyshev)

Output geometry/asr3D_s<SEED>.npz: 'aggs' (n x 15: centre 3, semi-axes 3, rotation 9 (rows: body axes)),
'gel_sites' (cell centres of the 64^3 grid, in [0, 1)).  env: SEED (1), AGG_FRACTION, D_MIN, D_MAX, GEL_RATIO (0.037)
"""
import os
import numpy as np

seed = int(os.environ.get('SEED', 1))
agg_fraction = float(os.environ.get('AGG_FRACTION', 0.35))
d_min, d_max = float(os.environ.get('D_MIN', 0.05)), float(os.environ.get('D_MAX', 0.26))
gel_ratio = float(os.environ.get('GEL_RATIO', 0.037))
gap, cells = 0.01, 64
rng = np.random.default_rng(seed)

sizes, vol = [], 0.0
a, b = d_min ** -2.5, d_max ** -2.5
while vol < agg_fraction:
    d = (a + rng.random() * (b - a)) ** (-1 / 2.5)
    sizes.append(d)
    vol += np.pi / 6 * d ** 3
sizes = np.sort(sizes)[::-1]


def random_rotation():
    q, r = np.linalg.qr(rng.normal(size=(3, 3)))
    return q * np.sign(np.diag(r))


aggs, centres, radii = [], [], []
for d in sizes:
    r = d / 2
    ar = rng.uniform(1.0, 1.3, size=3)
    axes = r * ar / np.cbrt(np.prod(ar))                # same volume as the ball of diameter d
    for trial in range(4000):
        c = rng.random(3)
        if centres:
            dx = (np.array(centres) - c + 0.5) % 1.0 - 0.5
            if np.any(np.linalg.norm(dx, axis=1) < np.array(radii) + axes.max() + gap):
                continue
        aggs.append(np.concatenate([c, axes, random_rotation().ravel()]))
        centres.append(c)
        radii.append(axes.max())
        break
aggs = np.array(aggs)
real_vol = sum(4 / 3 * np.pi * np.prod(g[3:6]) for g in aggs)


def level(points, g, shrink=0.0):
    """points [n, 3]; < 1 inside the ellipsoid g (periodic), semi-axes reduced by shrink"""
    dx = (points - g[:3] + 0.5) % 1.0 - 0.5
    y = dx @ g[6:].reshape(3, 3).T
    return np.sum((y / (g[3:6] - shrink)) ** 2, axis=1)


n_gel = int(round(gel_ratio * real_vol * cells ** 3))
h = 1.0 / cells
corners = np.array([[i, j, k] for i in (0, 1) for j in (0, 1) for k in (0, 1)]) * h
taken = np.zeros((cells,) * 3, dtype=bool)
sites = []
weights = np.array([np.prod(g[3:6]) for g in aggs])
weights /= weights.sum()
for trial in range(50 * n_gel):
    if len(sites) == n_gel:
        break
    g = aggs[rng.choice(len(aggs), p=weights)]
    u = rng.normal(size=3)
    u *= rng.random() ** (1 / 3) / np.linalg.norm(u)                      # uniform in the unit ball
    p = (g[:3] + g[6:].reshape(3, 3).T @ (u * g[3:6])) % 1.0
    cell = np.floor(p * cells).astype(int) % cells
    if np.any(level(cell * h + corners, g, shrink=0.5 * h) >= 1.0):
        continue
    sl = tuple(np.arange(c - 1, c + 2) % cells for c in cell)
    if taken[np.ix_(*sl)].any():
        continue
    taken[tuple(cell)] = True
    sites.append((cell + 0.5) * h)
sites = np.array(sites)
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'geometry', f'asr3D_s{seed}.npz')
np.savez(out, aggs=aggs, gel_sites=sites)
print(f'{len(aggs)} of {len(sizes)} aggregates (d = {sizes.min() * 100:.1f} ... {sizes.max() * 100:.1f} mm), '
      f'aggregate volume fraction {real_vol:.3f}, {len(sites)} of {n_gel} gel pockets -> {out}')
