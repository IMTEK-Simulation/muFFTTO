"""Periodic glass-bead / epoxy-like microstructure for the Carrara-parameter fatigue study (see README.md).

Random non-overlapping disks (DIM=2) or spheres (DIM=3), periodic, radii uniform in [R_MIN, R_MAX] (cell units), volume
fraction FRACTION, a gap GAP between beads, and no bead closer than GAP to the notch strip x = 0.375 ... 0.625, y = 0.5
(all z in 3D: a through pre-crack), so the pre-crack lies in the matrix. Writes geometry/<NAME>.npy (uint8, 1 matrix,
0 bead) on an N_FINE^DIM grid and figures/<NAME>_geometry.png.

    R_MIN=0.06 R_MAX=0.1 NAME=glass_beads_r0.06-0.1_512 python make_glass_bead_geometry.py       (local runs, cell 16 l)
    R_MIN=0.08 R_MAX=0.125 N_FINE=2048 NAME=beads2D_2048 python make_glass_bead_geometry.py       (NEMO, cell 64 l:
                                                                                                  radii 5 - 8 l)
    DIM=3 R_MIN=0.08 R_MAX=0.125 N_FINE=256 NAME=beads3D_256 python make_glass_bead_geometry.py
Default (glass_beads_512): radii 0.035 - 0.06, the first local runs (cell 64 l at l/h 0.5 - 2).
"""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

here = os.path.dirname(os.path.abspath(__file__))
DIM = int(os.environ.get('DIM', 2))
N_FINE = int(os.environ.get('N_FINE', 512))
FRACTION, GAP, SEED = float(os.environ.get('FRACTION', 0.20)), 0.015, 1
R_MIN, R_MAX = float(os.environ.get('R_MIN', 0.035)), float(os.environ.get('R_MAX', 0.06))
NAME = os.environ.get('NAME', 'glass_beads_512')
NOTCH = (0.375, 0.625, 0.5)
rng = np.random.default_rng(SEED)
beads, vol = [], 0.0
unit_vol = (lambda r: np.pi * r ** 2) if DIM == 2 else (lambda r: 4.0 / 3.0 * np.pi * r ** 3)
for trial in range(400000):
    if vol >= FRACTION:
        break
    c, r = rng.random(DIM), rng.uniform(R_MIN, R_MAX)
    dy = abs((c[1] - NOTCH[2] + 0.5) % 1.0 - 0.5)          # distance to the notch strip (periodic)
    dx = max(NOTCH[0] - c[0], 0.0, c[0] - NOTCH[1])
    if np.hypot(dx, dy) < r + GAP:
        continue
    if all(np.linalg.norm((c - c2 + 0.5) % 1.0 - 0.5) >= r + r2 + GAP for c2, r2 in beads):
        beads.append((c, r))
        vol += unit_vol(r)
x = (np.arange(N_FINE) + 0.5) / N_FINE
phase = np.ones((N_FINE,) * DIM, dtype=np.uint8)
for c, r in beads:                                          # bounding box only (large 3D grids)
    sl, dist2 = [], 0.0
    for a in range(DIM):
        lo, hi = int(np.floor((c[a] - r) * N_FINE)) - 1, int(np.ceil((c[a] + r) * N_FINE)) + 1
        idx = np.arange(lo, hi + 1)
        d = ((idx + 0.5) / N_FINE - c[a] + 0.5) % 1.0 - 0.5
        shape = [1] * DIM
        shape[a] = len(idx)
        dist2 = dist2 + (d ** 2).reshape(shape)
        sl.append(idx % N_FINE)
    block = phase[np.ix_(*sl)]
    block[dist2 < r ** 2] = 0
    phase[np.ix_(*sl)] = block
os.makedirs(os.path.join(here, 'geometry'), exist_ok=True)
np.save(os.path.join(here, 'geometry', NAME + '.npy'), phase)
print(f'{len(beads)} beads, bead volume fraction {1 - phase.mean():.3f}')
fig, ax = plt.subplots(figsize=(5, 5))
img = phase if DIM == 2 else phase[:, :, N_FINE // 2]
ax.imshow(img.T, origin='lower', extent=(0, 1, 0, 1), cmap='gray', interpolation='nearest')
ax.plot(NOTCH[:2], [NOTCH[2]] * 2, 'r-', lw=2, label='pre-crack')
ax.set_xlabel('x / L')
ax.set_ylabel('y / L')
ax.set_title(f'glass beads, fraction {1 - phase.mean():.2f}' + (' (z = L/2)' if DIM == 3 else ''), fontsize=13)
ax.legend(loc='upper right')
fig.tight_layout()
os.makedirs(os.path.join(here, 'figures'), exist_ok=True)
fig.savefig(os.path.join(here, 'figures', NAME + '_geometry.png'), dpi=150)
