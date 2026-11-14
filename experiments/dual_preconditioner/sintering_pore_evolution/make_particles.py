"""
Periodic packing of 8 discs of three sizes (radii 14 / 20 / 26 nm scaled by a common factor <= 1.2, 128 nm cell), every disc touching its neighbours
with a small initial overlap (contact flattening, initial neck). Collective rearrangement: random centres, radii
inflated step by step while overlaps beyond the allowed one (d >= (1 - OVERLAP)(r_i + r_j)) are pushed apart,
until the solid area fraction (union of discs, sampled on a 1024^2 grid) reaches TARGET.
Writes geometry/particles_s<seed>.npz: centres [nm], radii [nm], cell [nm].
"""
import os
import sys
import numpy as np

here = os.path.dirname(os.path.abspath(__file__))
seed = int(sys.argv[1]) if len(sys.argv) > 1 else 1
cell = 128.0
radii0 = np.array([26.0, 26.0, 20.0, 20.0, 20.0, 14.0, 14.0, 14.0])
target, overlap = 0.65, 0.06
rng = np.random.default_rng(seed)
c = rng.uniform(0, cell, (len(radii0), 2))
g = (np.arange(1024) + 0.5) / 1024 * cell
X, Y = np.meshgrid(g, g, indexing='ij')


def fraction(c, r):
    inside = np.zeros(X.shape, bool)
    for ci, ri in zip(c, r):
        dx = (X - ci[0] + cell / 2) % cell - cell / 2
        dy = (Y - ci[1] + cell / 2) % cell - cell / 2
        inside |= dx ** 2 + dy ** 2 < ri ** 2
    return inside.mean()


def relax(c, r, it=3000):
    for _ in range(it):
        moved = False
        for i in range(len(r)):
            for j in range(i + 1, len(r)):
                d = (c[j] - c[i] + cell / 2) % cell - cell / 2
                dist = np.linalg.norm(d)
                dmin = (1 - overlap) * (r[i] + r[j])
                if dist < dmin:
                    push = 0.5 * (dmin - dist) * d / max(dist, 1e-9)
                    c[i] -= push
                    c[j] += push
                    moved = True
                elif dist > (1 - overlap / 2) * (r[i] + r[j]) and dist < 1.3 * (r[i] + r[j]):
                    pull = 0.05 * (dist - (1 - overlap / 2) * (r[i] + r[j])) * d / dist   # bring near neighbours into contact
                    c[i] += pull
                    c[j] -= pull
        c %= cell
        if not moved:
            break
    return c


s = 0.5
while True:
    c = relax(c, s * radii0)
    phi = fraction(c, s * radii0)
    if s >= 1.2 or phi >= target:
        break
    s = min(1.2, s * 1.01)
r = s * radii0
c = relax(c, r, 20000)
phi = fraction(c, r)
contacts = []
for i in range(len(r)):
    for j in range(i + 1, len(r)):
        d = np.linalg.norm((c[j] - c[i] + cell / 2) % cell - cell / 2)
        if d < (r[i] + r[j]) * 1.001:
            contacts.append((i, j))
print(f'seed {seed}: scale {s:.3f}, solid fraction {phi:.3f}, radii {np.round(r, 2)}, contacts {len(contacts)}: '
      f'{contacts}')
np.savez(os.path.join(here, 'geometry', f'particles_s{seed}.npz'), centres=c, radii=r, cell=cell, fraction=phi)
