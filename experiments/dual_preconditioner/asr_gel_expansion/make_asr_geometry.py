"""
Periodic 2D concrete meso-structure for the ASR study: cement paste (0), aggregates (1), ASR gel pockets (2)
=========================================================================================================

Unit cell [0, 1)^2 = 0.1 m x 0.1 m (length unit 0.1 m). Elliptical aggregates with Fuller-distributed sizes
(cumulative passing by volume P(d) = (d / d_max)^0.5, hence a number density n(d) ~ d^-2.5 in 2D), placed largest
first by random sequential addition with periodic images and a minimum gap (bounding circles of the major
semi-axis + gap). Gel pockets: square cells of the 1/32 grid (side 3.125 mm; 1 pixel at 32^2, 2x2 at 64^2, 4x4 at
128^2) fully inside an aggregate with a margin of half a cell, at least 3 cells apart.

The label image is written on a fine FINE x FINE grid (default 512) at the pixel lower-left nodes, the same
sampling the solver uses, so a coarser grid N | FINE sees the same geometry at its own pixels.

    python make_asr_geometry.py            # geometry/asr_meso_s<SEED>.npy  (+ a png)
env: SEED (default 1), AGG_FRACTION (0.50), D_MIN (0.08), D_MAX (0.20), N_GEL (12), FINE (512)
"""
import os
import numpy as np

seed = int(os.environ.get('SEED', 1))
agg_fraction = float(os.environ.get('AGG_FRACTION', 0.50))
d_min, d_max = float(os.environ.get('D_MIN', 0.08)), float(os.environ.get('D_MAX', 0.20))
n_gel = int(os.environ.get('N_GEL', 12))
fine = int(os.environ.get('FINE', 512))
gap = 0.015
gel_grid = 32
rng = np.random.default_rng(seed)

# --- aggregate sizes: inverse transform of n(d) ~ d^-2.5 on [d_min, d_max], until the target area is reached
sizes = []
area = 0.0
while area < agg_fraction:
    u = rng.random()
    a, b = d_min ** -1.5, d_max ** -1.5
    d = (a + u * (b - a)) ** (-1 / 1.5)
    sizes.append(d)
    area += np.pi * d * d / 4
sizes = np.sort(sizes)[::-1]

# --- placement: ellipse semi-axes r sqrt(ar), r / sqrt(ar) (area of the disc of diameter d), random angle
aggs = []                                                   # (cx, cy, a, b, angle)
for d in sizes:
    r = d / 2
    ar = rng.uniform(1.0, 1.5)
    ea, eb = r * np.sqrt(ar), r / np.sqrt(ar)
    for trial in range(5000):
        c = rng.random(2)
        ok = True
        for (cx, cy, a2, b2, _) in aggs:
            dx = (c - np.array([cx, cy]) + 0.5) % 1.0 - 0.5
            if np.hypot(*dx) < ea + a2 + gap:
                ok = False
                break
        if ok:
            aggs.append((c[0], c[1], ea, eb, rng.uniform(0, np.pi)))
            break


def inside(x, y, agg, shrink=0.0):
    """points (x, y) inside the (periodic) ellipse, its semi-axes reduced by shrink"""
    cx, cy, a, b, ang = agg
    dx = (x - cx + 0.5) % 1.0 - 0.5
    dy = (y - cy + 0.5) % 1.0 - 0.5
    xr = np.cos(ang) * dx + np.sin(ang) * dy
    yr = -np.sin(ang) * dx + np.cos(ang) * dy
    return (xr / (a - shrink)) ** 2 + (yr / (b - shrink)) ** 2 < 1.0


x, y = np.meshgrid(np.arange(fine) / fine, np.arange(fine) / fine, indexing='ij')   # lower-left nodes
label = np.zeros((fine, fine), dtype=np.int8)
for agg in aggs:
    label[inside(x, y, agg)] = 1

# --- gel pockets: cells of the 1/32 grid inside an aggregate (all 4 corners + centre inside the shrunk ellipse)
hc = 1.0 / gel_grid
candidates = []
for i in range(gel_grid):
    for j in range(gel_grid):
        pts = np.array([[i, j], [i + 1, j], [i, j + 1], [i + 1, j + 1], [i + .5, j + .5]]) * hc
        for agg in aggs:
            if np.all(inside(pts[:, 0], pts[:, 1], agg, shrink=0.5 * hc)):
                candidates.append((i, j))
                break
rng.shuffle(candidates)
gels = []
for (i, j) in candidates:
    if len(gels) == n_gel:
        break
    if all(max(min(abs(i - gi), gel_grid - abs(i - gi)), min(abs(j - gj), gel_grid - abs(j - gj))) >= 3
           for gi, gj in gels):
        gels.append((i, j))
cell = fine // gel_grid
for (i, j) in gels:
    label[i * cell:(i + 1) * cell, j * cell:(j + 1) * cell] = 2

out = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'geometry', f'asr_meso_s{seed}.npy')
os.makedirs(os.path.dirname(out), exist_ok=True)
np.save(out, label)
print(f'{len(aggs)} aggregates (d = {sizes.min():.3f} ... {sizes.max():.3f}), area fractions: '
      f'paste {np.mean(label == 0):.3f}, aggregate {np.mean(label == 1):.3f}, gel {np.mean(label == 2):.4f} '
      f'({len(gels)} pockets of {len(candidates)} candidates) -> {out}')
