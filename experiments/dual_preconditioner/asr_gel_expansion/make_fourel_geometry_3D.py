"""
3D concrete meso-structure in the setting of Fourel, Gunn, Molinari 2026 (fourel.pdf in damage_AT2_small_strain/):
spherical aggregates (theirs: 20-face polyhedra) with Fuller grading, ASR gel as 0.75 mm spherical pockets inside the
aggregates, 1 % of the volume. Periodic cube of edge CELL_MM.

* aggregates: cumulative passing by volume P(d) = (d / d_max)^0.5 (Fuller, n = 0.5), d = D_MIN ... D_MAX mm, number
  density ~ d^-3.5; random sequential addition, largest first, periodic, gap 0.3 mm, target AGG_FRACTION (reached
  fraction printed; RSA of spheres jams at ~0.5-0.55 for this grading)
* gel: pockets of diameter GEL_D_MM (0.75) randomly inside aggregates (>= 0.3 mm from the aggregate surface, >= GEL_DIST
  mm apart), number = GEL_FRACTION * volume / pocket volume
  Their set-up: CELL_MM=70, D_MIN=0.5, D_MAX=15, 64 % aggregates at 256^3 (h = 0.273 mm). D_MIN = 0.5 mm aggregates
  are < 2 voxels wide there; default here: 35 mm cube, d = 1.5 ... 8 mm (128^3 gives their voxel size, as the 2D cell).

Output geometry/fourel3D_s<SEED>[_<CELL_MM>mm].npz: 'aggs' (n x 4: cx, cy, cz, r in cell units), 'gels' (m x 4),
'cell_mm'. rasterise(npz, N) -> label (N^3 int8: 0 paste, 1 aggregate, 2 gel) at the voxel lower-left nodes.

    python make_fourel_geometry_3D.py     # env SEED, CELL_MM (35), D_MIN (1.5), D_MAX (8), AGG_FRACTION (0.55),
                                          #     GEL_D_MM (0.75), GEL_FRACTION (0.01), GEL_DIST (1.0), MAX_TRIALS (400), OUT
"""
import os
import numpy as np


def rasterise(path, N):
    g = np.load(path)
    label = np.zeros((N, N, N), dtype=np.int8)
    for key, val in (('aggs', 1), ('gels', 2)):
        for cx, cy, cz, r in g[key]:
            idx, d2 = [], 0.0
            for a, c in enumerate((cx, cy, cz)):
                lo, hi = int(np.floor((c - r) * N)), int(np.ceil((c + r) * N))
                ii = np.arange(lo, hi + 1)
                dx = ii / N - c
                idx.append(ii % N)
                d2 = d2 + (dx ** 2).reshape([-1 if b == a else 1 for b in range(3)])
            inside = d2 < r * r
            sub = label[np.ix_(*idx)]
            sub[inside] = val
            label[np.ix_(*idx)] = sub
    return label


class Placed:
    """placed spheres (centres in [0, 1)^3), vectorised periodic overlap test"""

    def __init__(self, n_max):
        self.c = np.zeros((n_max, 3))
        self.r = np.zeros(n_max)
        self.n = 0

    def free(self, p, r, gap):
        if self.n == 0:
            return True
        d = (self.c[:self.n] - p + 0.5) % 1.0 - 0.5
        return bool(np.all(np.einsum('ij,ij->i', d, d) >= (self.r[:self.n] + r + gap) ** 2))

    def add(self, p, r):
        self.c[self.n], self.r[self.n] = p, r
        self.n += 1


if __name__ == '__main__':
    seed = int(os.environ.get('SEED', 1))
    L = float(os.environ.get('CELL_MM', 35.0))
    d_min, d_max = float(os.environ.get('D_MIN', 1.5)) / L, float(os.environ.get('D_MAX', 8.0)) / L
    agg_fraction = float(os.environ.get('AGG_FRACTION', 0.55))
    gel_r = float(os.environ.get('GEL_D_MM', 0.75)) / 2 / L
    gel_fraction = float(os.environ.get('GEL_FRACTION', 0.01))
    gel_dist = float(os.environ.get('GEL_DIST', 1.0)) / L
    gap, gel_margin = 0.3 / L, 0.3 / L
    MAX_TRIALS = int(os.environ.get('MAX_TRIALS', 400))
    rng = np.random.default_rng(seed)

    # Fuller by volume: dP/dd ~ d^-0.5, number ~ d^-0.5 / d^3 = d^-3.5 -> inverse-CDF sample of d^-3.5
    sizes, vol = [], 0.0
    a, b = d_min ** -2.5, d_max ** -2.5
    while vol < agg_fraction * 1.1:
        d = (a + rng.random() * (b - a)) ** (-1 / 2.5)
        sizes.append(d)
        vol += np.pi * d ** 3 / 6
    sizes = np.sort(sizes)[::-1]

    cl = Placed(len(sizes))
    aggs, vol = [], 0.0
    for d in sizes:
        r = d / 2
        for trial in range(MAX_TRIALS):
            c = rng.random(3)
            if cl.free(c, r, gap):
                cl.add(c, r)
                aggs.append(np.r_[c, r])
                vol += np.pi * d ** 3 / 6
                break
        if vol >= agg_fraction:
            break
    aggs = np.array(aggs)

    n_gel = int(round(gel_fraction / (4 / 3 * np.pi * gel_r ** 3)))
    big = aggs[aggs[:, 3] > gel_r + gel_margin]
    w = big[:, 3] ** 3 / np.sum(big[:, 3] ** 3)                 # uniform in the aggregate volume
    gcl = Placed(n_gel)
    gels = []
    for trial in range(50 * n_gel):
        if len(gels) == n_gel:
            break
        q = big[rng.choice(len(big), p=w)]
        rr = (q[3] - gel_r - gel_margin) * rng.random() ** (1 / 3)
        v = rng.normal(size=3)
        p = (q[:3] + rr * v / np.linalg.norm(v)) % 1.0
        if gcl.free(p, 0.0, gel_dist - gel_r):                 # centre distance >= gel_dist
            gcl.add(p, gel_r)
            gels.append(np.r_[p, gel_r])
    gels = np.array(gels)

    tag = '' if L == 35.0 else f'_{L:g}mm'
    out = os.environ.get('OUT', os.path.join(os.path.dirname(os.path.abspath(__file__)), 'geometry',
                                             f'fourel3D_s{seed}{tag}.npz'))
    os.makedirs(os.path.dirname(out), exist_ok=True)
    np.savez(out, aggs=aggs, gels=gels, cell_mm=L)
    print(f'{len(aggs)} aggregates (d = {2 * aggs[:, 3].min() * L:.2f} ... {2 * aggs[:, 3].max() * L:.2f} mm, '
          f'volume {vol:.3f}), {len(gels)} of {n_gel} gel pockets (d = {2 * gel_r * L:.2f} mm) -> {out}')
    for N in (64, 128):
        lab = rasterise(out, N)
        print(f'  {N}^3: h = {L / N:.3f} mm, l_c/h = {0.25 / (L / N):.2f}, paste {np.mean(lab == 0):.3f} '
              f'aggregate {np.mean(lab == 1):.3f} gel {np.mean(lab == 2):.4f}')
