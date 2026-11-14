"""Implied GB transport B_eff / B at 50 MPa from the GB profile, for several interface widths.
B_eff = V a^2 / (3 mean(mu - mu_edge)) (centre version: V a^2 / (2 (mu_c - mu_edge)))."""
import glob, os
import numpy as np
L0 = 90.24
B = 5.5e-11 * 2e-10 * 1.22e-29 / (1.380649e-23 * 700.0)
def cr(l):
    s = l - .5; i = np.where(s[:-1] * s[1:] < 0)[0]; return i + s[i] / (s[i] - s[i + 1])
for lab, N, name in (('W=4 nm, N=96', 96, 'SINT_N96_Green_W4_P50u_k0.0001_profile'), ('W=3 nm, N=128', 128, 'SINT_N128_Green_W3_P50u_k0.0001_W3profile'), ('W=2 nm, N=192', 192, 'SINT_N192_Green_W2_P50u_k0.0001_W2profile'), ('W=4 nm, N=96, 4x D_surf', 96, 'SINT_N96_Green_W4_P50u_k0.0001_surf4profile')):
    h = L0 / N
    print(lab)
    for f in sorted(glob.glob(f'exp_data/{name}_snaps/step_*.npz'), key=lambda f: int(os.path.basename(f)[5:-4]))[1:]:
        d = np.load(f); r = d['rho'].astype(float); k = cr(r[:, N // 2]) * h; x0 = .5 * (k.min() + k.max()); a = .5 * (k.max() - k.min())
        x = np.arange(N) * h - x0; row = d['mu'][:, N // 2]; V = float(d['V']) * 1e-9; edge = np.interp(a, x, row); ins = abs(x) <= a
        am = a * 1e-9; mean = (row[ins] - edge).mean(); ctr = np.interp(0, x, row) - edge
        S = float(d['sigma_gb']); Vc = 3 * B * S / am ** 2
        print(f'  t {float(d["t"]) * 1e6:4.0f} us  a {a:.2f}  V {V * 1e9:7.0f} nm/s  sigma {S / 1e6:6.1f}  mean(mu-edge)/sigma {mean / S:.3f}  V/Vc {V / Vc:.3f}  B_eff/B mean {V * am ** 2 / (3 * mean) / B:.3f}  centre {V * am ** 2 / (2 * ctr) / B:.3f}  rho_min on GB {r[:, N // 2][ins].min():.3f}')
