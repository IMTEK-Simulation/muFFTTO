"""
Two equal discs with a small initial overlap (neck), line of centres along y, centred in a periodic cell and isolated
from their periodic images (an isolated pair can approach; a periodic chain could not, its centre distances must sum to
the fixed cell height).  Writes geometry/two_particles.npz: centres [nm], radii [nm], cell [nm] (format of
make_particles.py).

    python make_two_particles.py [radius_nm=24] [overlap=0.06] [cell_nm=128]
"""
import os
import sys
import numpy as np

here = os.path.dirname(os.path.abspath(__file__))
r = float(sys.argv[1]) if len(sys.argv) > 1 else 24.0
overlap = float(sys.argv[2]) if len(sys.argv) > 2 else 0.06    # d = 2 r (1 - overlap), as in make_particles.py
cell = float(sys.argv[3]) if len(sys.argv) > 3 else 128.0
d = 2 * r * (1 - overlap)
centres = np.array([[cell / 2, cell / 2 - d / 2], [cell / 2, cell / 2 + d / 2]])
radii = np.array([r, r])
neck = 2 * np.sqrt(r ** 2 - (d / 2) ** 2)
gap_y = cell - d - 2 * r
gap_x = cell - 2 * r
print(f'r = {r:g} nm, centre distance {d:.3f} nm, sharp initial neck width {neck:.3f} nm, '
      f'gap to periodic images: y {gap_y:.2f} nm, x {gap_x:.2f} nm')
np.savez(os.path.join(here, 'geometry', 'two_particles.npz'), centres=centres, radii=radii, cell=cell)
