import sys, numpy as np, netCDF4
d = netCDF4.Dataset(sys.argv[1]); dmg = np.asarray(d["damage"][:]); n = dmg.shape[-1]
dm = dmg.reshape(dmg.shape[0], -1, n, n)[:, 0]
F10 = np.asarray(d["F10"][:]); P10 = np.asarray(d["P10"][:])
regions = {"right beam root": (0.75, 0.88, 0.50, 0.70), "left beam root": (0.12, 0.25, 0.30, 0.50),
           "frame corner bottom-left": (0.10, 0.28, 0.10, 0.28), "frame corner top-right": (0.72, 0.90, 0.72, 0.90),
           "frame corner top-left": (0.10, 0.28, 0.72, 0.90), "frame corner bottom-right": (0.72, 0.90, 0.10, 0.28),
           "frame walls elsewhere": None}
x = np.arange(n) / n
masks = {}
for name, r in regions.items():
    if r is None:
        continue
    m = np.zeros((n, n), bool); m[np.ix_((x >= r[0]) & (x < r[1]), (x >= r[2]) & (x < r[3]))] = True; masks[name] = m
masks["frame walls elsewhere"] = ~np.any(np.stack(list(masks.values())), axis=0)
print(f"{len(F10)} frames, F10 down to {F10[-1]:.3f}, final P10 {P10[-1]:.4f}, peak |P10| {np.abs(P10).max():.3f} at F10 {F10[np.argmax(np.abs(P10))]:.3f}")
for name, m in masks.items():
    series = dm[:, m].max(axis=1)
    hit = np.nonzero(series > 0.95)[0]
    when = f"cracked (d > 0.95) at F10 = {F10[hit[0]]:.3f}" if len(hit) else "not cracked"
    print(f"  {name:28s} {when:38s} final max d {series[-1]:.2f}")
