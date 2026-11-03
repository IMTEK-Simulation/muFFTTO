"""
Random UD-fibre cross sections after Ernesti, Schneider & Boehlke (2020), CMAME 363, 112793, Sec. 4.4: 16 equal
circular fibres, volume fraction 45 %, periodic, non-overlapping (they use Torquato-Jiao; here random sequential
addition with a minimum gap, and random shaking of the disks if the addition jams).

Units: the cell edge is 1 (= 512 um in the paper: 256 x 256 pixels of h = 2 um).
Phase per pixel (sampled at the pixel centres): 1 matrix (polyamide, brittle), 0 fibre (E-glass, elastic).

usage:  python fiber_geometry.py [N] [seed]        writes figures/fibers_N<N>_s<seed>.png
"""
import os
import sys

import numpy as np

NB_FIBRES = 16
VOLUME_FRACTION = 0.45
RADIUS = np.sqrt(VOLUME_FRACTION / (NB_FIBRES * np.pi))   # 0.0946 (48.4 um)
# minimum surface distance between fibres (FIBER_GAP_UM env, default 2 um = 1 pixel of the paper)
GAP = float(os.environ.get('FIBER_GAP_UM', 2.0)) / 512
CELL_UM = 512.0


def _periodic_dist(a, b):
    d = np.abs(a - b)
    d = np.minimum(d, 1 - d)
    return np.sqrt((d ** 2).sum(-1))


def centres(seed=0):
    rng = np.random.default_rng(seed)
    dmin = 2 * RADIUS + GAP
    for _ in range(2000):                                   # random sequential addition, restart when jammed
        pts = []
        for _ in range(20000):
            p = rng.random(2)
            if all(_periodic_dist(p, q) >= dmin for q in pts):
                pts.append(p)
                if len(pts) == NB_FIBRES:
                    return np.array(pts)
    # fallback: random positions, then push overlapping pairs apart until no overlap is left
    pts = rng.random((NB_FIBRES, 2))
    for _ in range(100000):
        moved = False
        for i in range(NB_FIBRES):
            for j in range(i + 1, NB_FIBRES):
                d = pts[i] - pts[j]
                d -= np.round(d)
                dist = np.linalg.norm(d)
                if dist < dmin:
                    shift = 0.5 * (dmin - dist + 1e-6) * d / max(dist, 1e-12)
                    pts[i] = (pts[i] + shift) % 1
                    pts[j] = (pts[j] - shift) % 1
                    moved = True
        if not moved:
            return pts
    raise RuntimeError('could not place the fibres')


def phase(nb_pixels, seed=0):
    """1 matrix, 0 fibre, per pixel (centres), shape (N, N), [x, y] indexing."""
    c = (np.arange(nb_pixels) + 0.5) / nb_pixels
    X = np.stack(np.meshgrid(c, c, indexing='ij'), axis=-1)
    fibre = np.zeros((nb_pixels, nb_pixels), dtype=bool)
    for p in centres(seed):
        fibre |= _periodic_dist(X, p) < RADIUS
    return np.where(fibre, 0, 1)


if __name__ == '__main__':
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    n = int(sys.argv[1]) if len(sys.argv) > 1 else 256
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    ph = phase(n, seed)
    fig, ax = plt.subplots(figsize=(6, 6), layout='constrained')
    ax.imshow(ph.T, origin='lower', extent=(0, CELL_UM, 0, CELL_UM), interpolation='nearest', cmap='Greys_r')
    ax.set_title(f'{NB_FIBRES} fibres, seed {seed}: fibre fraction {1 - ph.mean():.3f}, N = {n}', fontsize=12)
    ax.set_xlabel('x [um]', fontsize=12)
    ax.set_ylabel('y [um]', fontsize=12)
    out = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'figures', f'fibers_N{n}_s{seed}.png')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=150)
    print(f'wrote {out}; fibre fraction {1 - ph.mean():.4f}')
