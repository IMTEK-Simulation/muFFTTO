"""Voxel geometry of the 3D hollow bar (the 2D pressurised-hole cross-section extruded along z, sealed by solid caps).

    cell     [0, Lx] x [0, Ly] x [0, Lz],  Lx = Ly = PLATE + 2 t,  Lz = BAR_L + 2 t,  voxel size h = 1 / nnn
    bar      [t, Lx - t] x [t, Ly - t] x [t, Lz - t]   (cross-section PLATE x PLATE, length BAR_L), solid
    cavity   circle of radius HOLE_R around the bar axis, z in [t + CAP_T, Lz - t - CAP_T]: gas (zero stiffness, p)
    caps     the solid between the cavity ends and the bar ends (thickness CAP_T): the tube is sealed
    void     everything outside the bar (thickness t on every side): zero stiffness
    clamps   the two full node planes x = t and x = Lx - t (through the void): u = 0
periodic=True: a repeating open channel: the bar and the cavity fill the whole cell in z (Lz = BAR_L, no caps, no void
in z), the gas surface is closed through the periodic boundary.
"""
import numpy as np

DEFAULTS = dict(PLATE=0.875, HOLE_R=0.3125, BAR_L=1.0, CAP_T=0.125, VOID_T=1 / 16)


def bar_geometry(nnn, PLATE=0.875, HOLE_R=0.3125, BAR_L=1.0, CAP_T=0.125, VOID_T=1 / 16, periodic=False):
    h = 1.0 / nnn
    px = {k: v * nnn for k, v in dict(PLATE=PLATE, BAR_L=BAR_L, CAP_T=CAP_T, VOID_T=VOID_T).items()}
    for k, v in px.items():
        assert abs(v - round(v)) < 1e-9 and v >= 1, f'{k} must be a whole number of voxels'
    m, P, B = int(round(px['VOID_T'])), int(round(px['PLATE'])), int(round(px['BAR_L']))
    N = (P + 2 * m, P + 2 * m, B if periodic else B + 2 * m)
    L = tuple(n * h for n in N)
    c = [(np.arange(n) + 0.5) * h for n in N]                          # voxel centres
    X, Y, Z = np.meshgrid(*c, indexing='ij')
    t = VOID_T
    in_bar = (X > t) & (X < L[0] - t) & (Y > t) & (Y < L[1] - t) & (periodic | ((Z > t) & (Z < L[2] - t)))
    in_cav = (in_bar & (np.hypot(X - 0.5 * L[0], Y - 0.5 * L[1]) < HOLE_R)
              & (periodic | ((Z > t + CAP_T) & (Z < L[2] - t - CAP_T))))
    solid = in_bar & ~in_cav
    clamp_x = (m, N[0] - m)                                            # node-plane indices of the clamps
    return dict(N=N, L=L, h=h, solid=solid, cavity=in_cav, margin=m, clamp_x=clamp_x, Zc=Z)
