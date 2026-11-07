"""
Concrete meso-structure in the setting of Falsafi et al. (arXiv 2207.10657, Fig. 7a): periodic 0.1 m x 0.1 m cell,
ellipsoidal aggregates with Fuller grading, ASR gel sites = single pixels inside the aggregates.

* aggregates: cumulative passing by volume P(d) = (d / d_max)^0.5 (number density n(d) ~ d^-2.5 in 2D),
  d = D_MIN ... D_MAX (default 5 ... 30 mm, read off Fig. 7a), area fraction AGG_FRACTION (default 0.50),
  ellipses (aspect 1 ... 1.5, random angle), random sequential addition largest first, gap 1 mm, periodic
* gel sites: N_GEL points (default 75, counted in Fig. 7a) inside the aggregates, at least 1.5 mm from the
  aggregate boundary and 2.5 mm from each other; on an N^2 grid the gel is the single pixel containing the point
  (pixel [i, j] covers [i h, (i+1) h) x [j h, (j+1) h)), so the site positions are grid-independent while the gel
  area is one pixel (h^2) as in the paper.

Output geometry/falsafi_s<SEED>.npz: 'aggregate' (FINE^2 bool, sampled at the pixel lower-left nodes like the
solver) and 'gel_sites' (N_GEL x 2, in [0, 1)).   env: SEED (1), AGG_FRACTION, D_MIN, D_MAX, N_GEL, FINE (512)
"""
import os
import numpy as np

seed = int(os.environ.get('SEED', 1))
agg_fraction = float(os.environ.get('AGG_FRACTION', 0.50))
d_min, d_max = float(os.environ.get('D_MIN', 0.05)), float(os.environ.get('D_MAX', 0.30))   # x 0.1 m
n_gel = int(os.environ.get('N_GEL', 75))
fine = int(os.environ.get('FINE', 512))
gap, gel_margin, gel_dist = 0.01, 0.015, 0.025
rng = np.random.default_rng(seed)

sizes, area = [], 0.0
a, b = d_min ** -1.5, d_max ** -1.5
while area < agg_fraction:
    d = (a + rng.random() * (b - a)) ** (-1 / 1.5)
    sizes.append(d)
    area += np.pi * d * d / 4
sizes = np.sort(sizes)[::-1]


def ellipse_level(x, y, agg, shrink=0.0):
    cx, cy, ea, eb, ang = agg
    dx = (x - cx + 0.5) % 1.0 - 0.5
    dy = (y - cy + 0.5) % 1.0 - 0.5
    xr = np.cos(ang) * dx + np.sin(ang) * dy
    yr = -np.sin(ang) * dx + np.cos(ang) * dy
    return (xr / (ea - shrink)) ** 2 + (yr / (eb - shrink)) ** 2


# placement test on a 256^2 occupancy grid of the ellipses grown by the gap (exact enough, allows dense packing)
occ_n = 256
ox, oy = np.meshgrid((np.arange(occ_n) + 0.5) / occ_n, (np.arange(occ_n) + 0.5) / occ_n, indexing='ij')
occupied = np.zeros((occ_n, occ_n), dtype=bool)
aggs = []
for d in sizes:
    r = d / 2
    ar = rng.uniform(1.0, 1.5)
    ea, eb = r * np.sqrt(ar), r / np.sqrt(ar)
    for trial in range(3000):
        agg = (rng.random(), rng.random(), ea, eb, rng.uniform(0, np.pi))
        grown = ellipse_level(ox, oy, agg, shrink=-gap) < 1.0
        if not np.any(occupied & grown):
            occupied |= ellipse_level(ox, oy, agg) < 1.0
            aggs.append(agg)
            break

x, y = np.meshgrid(np.arange(fine) / fine, np.arange(fine) / fine, indexing='ij')
aggregate = np.zeros((fine, fine), dtype=bool)
for agg in aggs:
    aggregate |= ellipse_level(x, y, agg) < 1.0

sites = []
for trial in range(200000):
    if len(sites) == n_gel:
        break
    p = rng.random(2)
    if not any(np.all([ellipse_level(p[0] + sx, p[1] + sy, agg, shrink=gel_margin) < 1.0
                       for sx, sy in ((0, 0), (1 / 64, 0), (0, 1 / 64), (1 / 64, 1 / 64))]) for agg in aggs):
        continue                     # the pixel of the 64^2 grid containing p (and its corners) inside an aggregate
    if all(np.hypot(*((p - q + 0.5) % 1.0 - 0.5)) >= gel_dist for q in sites):
        sites.append(p)
sites = np.array(sites)

out = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'geometry', f'falsafi_s{seed}.npz')
np.savez(out, aggregate=aggregate, gel_sites=sites, aggs=np.array(aggs))   # aggs: (cx, cy, a, b, angle)
print(f'{len(aggs)} of {len(sizes)} aggregates placed (d = {sizes.min() * 100:.1f} ... {sizes.max() * 100:.1f} mm), '
      f'aggregate area fraction {aggregate.mean():.3f}, {len(sites)} gel sites -> {out}')
