import sys, numpy as np, netCDF4
d = netCDF4.Dataset(sys.argv[1]); dmg = np.asarray(d["damage"][:]); n = dmg.shape[-1]
# damage is nodal [frame, 1, nodes, x, y]; take nodal sub-point 0 = pixel corner
dm = dmg.reshape(dmg.shape[0], -1, n, n)[:, 0]
regions = {"left beam root (x~0.15, y 0.35-0.45)": (0.12, 0.25, 0.30, 0.50),
           "right beam root (x~0.85, y 0.55-0.65)": (0.75, 0.88, 0.50, 0.70),
           "frame corner bottom-left (0.15,0.15)": (0.10, 0.28, 0.10, 0.28),
           "frame corner top-right (0.85,0.85)": (0.72, 0.90, 0.72, 0.90),
           "frame corner top-left (0.15,0.85)": (0.10, 0.28, 0.72, 0.90),
           "frame corner bottom-right (0.85,0.15)": (0.72, 0.90, 0.10, 0.28)}
x = np.arange(n) / n
for k in [int(a) for a in sys.argv[2:]] or [dm.shape[0] - 1]:
    k = min(k, dm.shape[0] - 1)
    print(f"frame {k + 1}: max d {dm[k].max():.3f}")
    for name, (x0, x1, y0, y1) in regions.items():
        m = dm[k][np.ix_((x >= x0) & (x < x1), (x >= y0) & (x < y1))]
        print(f"   {name:42s} max d {m.max():.3f}")
