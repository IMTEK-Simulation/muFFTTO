import sys, netCDF4, numpy as np
d = netCDF4.Dataset(sys.argv[1])
g = -np.asarray(d["F10"][:]); t = -np.asarray(d["P10"][:]); J = np.asarray(d["min_det_F"][:]); no = np.asarray(d["nb_outer"][:])
K = np.gradient(t, g)
for i in range(0, 100, 5): print(f"inc {i+1:3d} shear {g[i]:.3f} tau {t[i]:.4f} K {K[i]:7.3f} minJ {J[i]:.4f} outer {int(no[i])}")
print("K0", K[:5].mean(), "first K>2K0 at inc", int(np.argmax(K > 2 * K[:5].mean())) + 1)
ph = np.asarray(d["phase_field"][-1, 0]); dJ = np.asarray(d["detF"][-1, 0])
print("void detF last frame: min", dJ[ph == 0].min(), "frac < 0.1:", (dJ[ph == 0] < 0.1).mean(), "matrix detF range", dJ[ph > 0].min(), dJ[ph > 0].max())
