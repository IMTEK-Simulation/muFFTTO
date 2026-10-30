"""
Contact indicator of a third-medium run (stefanus_2D_finite_strain_NeoHookean_TR.py --third_medium --frames): per saved
increment the smallest J = det F in the voids (the third medium) and the share of void pixels with J < J_CONTACT. The
faces of a void touch when the medium between them is squeezed flat, i.e. J -> 0.

F per pixel from the nodal displacement u (bilinear pixel, F at the pixel centre: mean of the two edge differences in
each direction) plus F_macro.

usage:  python contact_monitor.py <frame folder> [J_CONTACT]     prints one line per increment;
        exit code 0 once some increment has min J < J_CONTACT (default 0.1), else 1
"""
import glob
import os
import sys

import numpy as np

folder = sys.argv[1]
J_CONTACT = float(sys.argv[2]) if len(sys.argv) > 2 else 0.1
in_contact = False
for f in sorted(glob.glob(os.path.join(folder, 'inc*.npz'))):
    d = np.load(f)
    u, Fm, phase = d['u'], d['F_macro'], d['phase']
    nx, ny = phase.shape
    hx, hy = d['domain_size'][0] / nx, d['domain_size'][1] / ny
    # nodal u[i, x, y] -> pixel-centre derivatives (periodic): mean of the differences along the two pixel edges
    du_dx = 0.5 * ((np.roll(u, -1, 1) - u) + (np.roll(np.roll(u, -1, 1), -1, 2) - np.roll(u, -1, 2))) / hx
    du_dy = 0.5 * ((np.roll(u, -1, 2) - u) + (np.roll(np.roll(u, -1, 1), -1, 2) - np.roll(u, -1, 1))) / hy
    F = Fm[:, :, None, None] + np.stack([du_dx, du_dy], axis=1)          # F[i, j, x, y]
    J = F[0, 0] * F[1, 1] - F[0, 1] * F[1, 0]
    J_void = J[phase == 1]
    hit = J_void.min() < J_CONTACT
    in_contact |= hit
    print(f'{os.path.basename(f)}  eps {float(d["eps"]):.4f}  stress {float(d["stress_MPa"]):.5f} MPa  '
          f'min J in voids {J_void.min():.3f}  J < {J_CONTACT:g}: {100 * np.mean(J_void < J_CONTACT):.2f} % of void'
          + ('  <- contact' if hit else ''))
sys.exit(0 if in_contact else 1)
