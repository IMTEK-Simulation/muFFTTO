import sys, numpy as np, netCDF4
d = netCDF4.Dataset(sys.argv[1]); F = np.asarray(d["F10"][:]); P = -np.asarray(d["P10"][:]); J = np.asarray(d["min_det_F"][:])
dmg = np.asarray(d["damage"][:]); n = dmg.shape[-1]; dm = dmg.reshape(dmg.shape[0], -1, n, n)[:, 0]
x = np.arange(n) / n
def reg(x0, x1, y0, y1):
    m = np.zeros((n, n), bool); m[np.ix_((x >= x0) & (x < x1), (x >= y0) & (x < y1))] = True; return m
regions = {"frame corner bottom-left": reg(0.08, 0.25, 0.08, 0.25), "frame corner top-right": reg(0.75, 0.92, 0.75, 0.92),
           "frame corner top-left": reg(0.08, 0.25, 0.75, 0.92), "frame corner bottom-right": reg(0.75, 0.92, 0.08, 0.25),
           "left support root": reg(0.08, 0.25, 0.28, 0.45), "right support root": reg(0.75, 0.92, 0.55, 0.72)}
rest = ~np.any(np.stack(list(regions.values())), axis=0); regions["elsewhere"] = rest
closed = np.nonzero(J < 0.05)[0]
print(f"{len(F)} frames up to F10 = {F[-1]:.3f};  supports in contact (min J < 0.05) from F10 = {F[closed[0]]:.3f}" if len(closed) else f"{len(F)} frames, gap not closed")
for name, m in regions.items():
    s = dm[:, m].max(axis=1); hit = np.nonzero(s > 0.95)[0]
    print(f"  {name:26s} {'cracks at F10 = %.3f' % F[hit[0]] if len(hit) else 'intact':22s} max d {s[-1]:.2f}")
# load history: local maxima of |P| (peaks) and minima after them
print("  load: " + "  ".join(f"F10 {F[k]:.3f} |P| {P[k]:.3f}" for k in range(0, len(F), max(len(F) // 12, 1))))
peaks = [k for k in range(1, len(P) - 1) if P[k] >= P[k - 1] and P[k] > P[k + 1] and P[k] > 0.05]
print("  peaks of |P10|: " + ", ".join(f"{P[k]:.3f} at F10 {F[k]:.3f}" for k in peaks))
