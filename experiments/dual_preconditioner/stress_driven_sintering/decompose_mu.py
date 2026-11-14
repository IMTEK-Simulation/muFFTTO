"""Split mean(mu - mu_edge) on the GB into the parts set by the rigid-body law (50 MPa profile runs).
Law limit: <mu>_gb (weight 4 eta0 eta1) = mu_s (weight rho^2 (1-rho)^2, all free surface) + sigma_gb.
So  mean_in - edge = sigma + (mu_s - edge) + (mean_in - <mu>_gb) - residual dmu*e0.   All in units of sigma."""
import glob, os, sys
import numpy as np
L0 = 90.24
def cr(l):
    s = l - .5; i = np.where(s[:-1] * s[1:] < 0)[0]; return i + s[i] / (s[i] - s[i + 1])
runs = [(f'P={P} MPa, W=4 nm, N=96', 96, f'SINT_N96_Green_W4_P{P}u_k0.0001_profile') for P in (0, 25, 50, 100)]
runs.append(('P=50 MPa, W=3 nm, N=128', 128, 'SINT_N128_Green_W3_P50u_k0.0001_W3profile'))
runs.append(('P=50 MPa, W=2 nm, N=192', 192, 'SINT_N192_Green_W2_P50u_k0.0001_W2profile'))
runs.append(('P=50 MPa, W=4 nm, N=96, 4x D_surf', 96, 'SINT_N96_Green_W4_P50u_k0.0001_surf4profile'))
for lab, N, name in runs:
    h = L0 / N
    print(lab, '   [all / sigma]  mean_in-edge = 1 + (mu_s-edge) + (mean_in-<mu>gb) - dmu')
    for f in sorted(glob.glob(f'exp_data/{name}_snaps/step_*.npz'), key=lambda f: int(os.path.basename(f)[5:-4]))[1:]:
        d = np.load(f); r = d['rho'].astype(float); es = d['etas'].astype(float); m = d['mu'].astype(float)
        k = cr(r[:, N // 2]) * h; x0 = .5 * (k.min() + k.max()); a = .5 * (k.max() - k.min())
        x = np.arange(N) * h - x0; row = m[:, N // 2]; edge = np.interp(a, x, row); ins = abs(x) <= a
        S = float(d['sigma_gb']); Sd = max(S, 1.0)
        w = 4 * es[0] * es[1]; sw = r ** 2 * (1 - r) ** 2
        mu_gb = (w * m).sum() / w.sum(); mu_s = (sw * m).sum() / sw.sum()
        mean_in = row[ins].mean()
        dmu = mean_in - edge
        res = (mu_s + S - mu_gb)                      # law residual (nonzero at finite t_r)
        # surface mu near the neck vs far: split surface weight by distance from GB row (y)
        print(f'  t {float(d["t"])*1e6:4.0f} us  a {a:5.2f}  sigma {S/1e6:6.1f} MPa | mean_in-edge {dmu/Sd:.3f} = 1 {(mu_s-edge)/Sd:+.3f} {(mean_in-mu_gb)/Sd:+.3f} {-res/Sd:+.3f}'
              f' | MPa: edge-mu_s {(edge-mu_s)/1e6:6.1f}  <mu>gb-mean_in {(mu_gb-mean_in)/1e6:6.1f}  mean_in-edge {dmu/1e6:6.1f}'
              f' | mu_s {mu_s/1e6:7.1f}  edge {edge/1e6:7.1f}  <mu>gb {mu_gb/1e6:7.1f}  mean_in {mean_in/1e6:7.1f} MPa')
