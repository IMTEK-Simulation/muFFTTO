"""
2D concrete meso-structure in the setting of Fourel, Gunn, Molinari 2026 (hal-05696277, fourel.pdf in
damage_AT2_small_strain/): circular aggregates with Fuller grading (their 2D comparison, Sec. 4), ASR gel as small
circular pockets inside the aggregates, 1 % of the area.

Scaled to a 2D cell resolvable on a laptop: cell L = CELL_MM (35 mm) so that 128^2 gives their voxel size
(0.273 mm) and l_c = 0.25 mm spans ~1 pixel, 256^2 ~2 pixels.
* aggregates: cumulative passing by area P(d) = (d / d_max)^0.5 (Fuller, n = 0.5), d = D_MIN ... D_MAX mm
  (default 1.5 ... 8 mm, their 0.5 ... 15 mm does not fit a 35 mm 2D cell / 128^2 grid), area fraction AGG_FRACTION
  (0.55 target, random sequential addition largest first, gap 0.3 mm, periodic)
* gel: circular pockets of diameter GEL_D_MM (0.75 mm, as theirs), randomly inside aggregates (>= 0.3 mm from the
  aggregate boundary, >= 1.5 mm apart), number such that the gel area is GEL_FRACTION (0.01) of the cell

Output geometry/fourel2D_s<SEED>.npz (analytic, rasterise on any grid):
  'aggs' (n x 3: cx, cy, r in units of the cell), 'gels' (m x 3), 'cell_mm' (scalar)
Helper rasterise(npz, N) -> label (N x N int: 0 paste, 1 aggregate, 2 gel), sampled at pixel lower-left nodes
(x_i = i / N), the same convention as the muFFTTO coordinates.

    python make_fourel_geometry_2D.py            # env SEED, CELL_MM, D_MIN, D_MAX, AGG_FRACTION, GEL_D_MM, GEL_FRACTION
"""
import os
import numpy as np


def rasterise(path, N):
    g = np.load(path)
    x, y = np.meshgrid(np.arange(N) / N, np.arange(N) / N, indexing='ij')
    label = np.zeros((N, N), dtype=int)
    for key, val in (('aggs', 1), ('gels', 2)):
        for cx, cy, r in g[key]:
            dx = (x - cx + 0.5) % 1.0 - 0.5
            dy = (y - cy + 0.5) % 1.0 - 0.5
            label[dx * dx + dy * dy < r * r] = val
    return label


if __name__ == '__main__':
    seed = int(os.environ.get('SEED', 1))
    L = float(os.environ.get('CELL_MM', 35.0))
    d_min, d_max = float(os.environ.get('D_MIN', 1.5)) / L, float(os.environ.get('D_MAX', 8.0)) / L
    agg_fraction = float(os.environ.get('AGG_FRACTION', 0.55))
    gel_r = float(os.environ.get('GEL_D_MM', 0.75)) / 2 / L
    gel_fraction = float(os.environ.get('GEL_FRACTION', 0.01))
    gap, gel_margin, gel_dist = 0.3 / L, 0.3 / L, 1.5 / L
    rng = np.random.default_rng(seed)

    # Fuller by area: dP/dd ~ d^-0.5, number density ~ d^-0.5 / d^2 = d^-2.5 -> inverse-CDF sample of d^-2.5
    sizes, area = [], 0.0
    a, b = d_min ** -1.5, d_max ** -1.5
    while area < agg_fraction * 1.05:
        d = (a + rng.random() * (b - a)) ** (-1 / 1.5)
        sizes.append(d)
        area += np.pi * d * d / 4
    sizes = np.sort(sizes)[::-1]

    aggs = []
    for d in sizes:
        r = d / 2
        for trial in range(4000):
            c = rng.random(2)
            if all(np.hypot(*((c - q[:2] + 0.5) % 1.0 - 0.5)) >= r + q[2] + gap for q in aggs):
                aggs.append(np.array([c[0], c[1], r]))
                break
    aggs = np.array(aggs)

    n_gel = int(round(gel_fraction / (np.pi * gel_r ** 2)))
    gels = []
    for trial in range(500000):
        if len(gels) == n_gel:
            break
        p = rng.random(2)
        dist = np.hypot(*((p[:, None] - aggs[:, :2].T + 0.5) % 1.0 - 0.5))
        if not np.any(dist <= aggs[:, 2] - gel_r - gel_margin):
            continue
        if all(np.hypot(*((p - q[:2] + 0.5) % 1.0 - 0.5)) >= gel_dist for q in gels):
            gels.append(np.array([p[0], p[1], gel_r]))
    gels = np.array(gels)

    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'geometry', f'fourel2D_s{seed}.npz')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    np.savez(out, aggs=aggs, gels=gels, cell_mm=L)
    lab = rasterise(out, 512)
    print(f'{len(aggs)} of {len(sizes)} aggregates (d = {2 * aggs[:, 2].min() * L:.2f} ... {2 * aggs[:, 2].max() * L:.2f} mm), '
          f'{len(gels)} of {n_gel} gel pockets (d = {2 * gel_r * L:.2f} mm); area fractions at 512^2: '
          f'paste {np.mean(lab == 0):.3f}, aggregate {np.mean(lab == 1):.3f}, gel {np.mean(lab == 2):.4f} -> {out}')
    for N in (64, 128, 256):
        lab = rasterise(out, N)
        print(f'  {N}^2: h = {L / N:.3f} mm, l_c/h = {0.25 / (L / N):.2f}, gel pixels {np.sum(lab == 2)}, '
              f'gel fraction {np.mean(lab == 2):.4f}')
