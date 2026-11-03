"""
2D polycrystal with an edge crack after Nkoumbou Kaptchouang & Gelebart (2022), CMAME 394, 114921, Sec. 5.3, Fig. 18:
a periodic Voronoi tessellation of the unit square (about 15 x 15 grains), isotropic grains with a Young modulus drawn
uniformly in [E_MIN, E_MAX] = [1, 3] (GPa in the paper, units of 1 GPa here) and one Poisson ratio, a vertical crack
of one pixel width from the top of the cell down to its centre, and a layer of 'void' pixels along the top, which
makes the top / bottom of the periodic cell stress-free (uniaxial tension in x).

The geometry is defined in the continuum (seeds in the unit square) and sampled at the pixel centres, so every grid
sees the same grains; the crack (1 pixel) and the void layer (NB_VOID_ROWS pixels) are defined in pixels, as in the
paper, and get thinner with h.

usage:  python polycrystal_geometry.py [N]     writes figures/polycrystal_geometry_N<N>.png
"""
import os
import sys

import numpy as np

NB_GRAINS = 15 * 15          # Voronoi seeds in the unit cell
E_MIN, E_MAX = 1.0, 3.0      # grain Young modulus (GPa in the paper)
POISSON = 0.3
NB_VOID_ROWS = 2             # void layer at the top (pixels)
SEED = 2022


def grain_ids(nb_pixels, seed=SEED):
    """Periodic Voronoi grain index per pixel (pixel centres), shape (N, N), [x, y] indexing."""
    rng = np.random.default_rng(seed)
    seeds = rng.random((NB_GRAINS, 2))
    c = (np.arange(nb_pixels) + 0.5) / nb_pixels
    X, Y = np.meshgrid(c, c, indexing='ij')
    best = np.full(X.shape, np.inf)
    ids = np.zeros(X.shape, dtype=int)
    for g, (sx, sy) in enumerate(seeds):
        dx = np.abs(X - sx)
        dy = np.abs(Y - sy)
        dist = np.minimum(dx, 1 - dx) ** 2 + np.minimum(dy, 1 - dy) ** 2      # periodic distance
        closer = dist < best
        best[closer] = dist[closer]
        ids[closer] = g
    return ids


def grain_moduli(seed=SEED):
    rng = np.random.default_rng(seed + 1)
    return E_MIN + (E_MAX - E_MIN) * rng.random(NB_GRAINS)


def crack_and_void(nb_pixels):
    """Masks (N, N), [x, y] indexing: crack = one pixel column at x = 1/2 from the centre to the top;
    void = the top NB_VOID_ROWS rows."""
    crack = np.zeros((nb_pixels, nb_pixels), dtype=bool)
    void = np.zeros_like(crack)
    void[:, nb_pixels - NB_VOID_ROWS:] = True
    crack[nb_pixels // 2, nb_pixels // 2:] = True
    return crack, void


def young_modulus(nb_pixels, void_stiffness):
    """Young modulus per pixel: grain modulus, void_stiffness * E_MIN in the crack and the void layer."""
    E = grain_moduli()[grain_ids(nb_pixels)]
    crack, void = crack_and_void(nb_pixels)
    E[crack | void] = void_stiffness * E_MIN
    return E


if __name__ == '__main__':
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    n = int(sys.argv[1]) if len(sys.argv) > 1 else 128
    E = young_modulus(n, 0.0)
    fig, ax = plt.subplots(figsize=(7, 6), layout='constrained')
    im = ax.imshow(E.T, origin='lower', extent=(0, 1, 0, 1), interpolation='nearest', cmap='RdYlBu_r',
                   vmin=0, vmax=E_MAX)
    fig.colorbar(im, ax=ax, label='Young modulus [GPa] (crack, void layer: 0)')
    ax.annotate('', xy=(1.08, 0.5), xytext=(0.92, 0.5), xycoords='axes fraction',
                arrowprops=dict(arrowstyle='->', lw=2))
    ax.annotate('', xy=(-0.08, 0.5), xytext=(0.08, 0.5), xycoords='axes fraction',
                arrowprops=dict(arrowstyle='->', lw=2))
    ax.set_title(f'polycrystal {NB_GRAINS} grains, edge crack, void layer; N = {n}', fontsize=13)
    ax.set_xlabel('x', fontsize=12)
    ax.set_ylabel('y', fontsize=12)
    os.makedirs(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'figures'), exist_ok=True)
    out = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'figures', f'polycrystal_geometry_N{n}.png')
    fig.savefig(out, dpi=200)
    print(f'wrote {out}')
