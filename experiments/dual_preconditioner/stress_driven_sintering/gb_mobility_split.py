"""Transport along the GB: integral across the GB (over y, mid-neck columns |x| < a/2) of each mobility term, nodal
values, 50 MPa profile runs, t = 200-400 us. Shows how much of the GB transport comes from M_surf rho^2 (1-rho)^2 where
rho dips below 1 on the GB.  Mobilities are taken from the run logs (code units nm^2/s), GB term M_gb (s1^2 - s2)."""
import glob, os, re
import numpy as np
L0 = 90.24
def cr(l):
    s = l - .5; i = np.where(s[:-1] * s[1:] < 0)[0]; return i + s[i] / (s[i] - s[i + 1])
for lab, N, name, log in (('W=4 nm', 96, 'SINT_N96_Green_W4_P50u_k0.0001_profile', 'chain_profile'),
                          ('W=3 nm', 128, 'SINT_N128_Green_W3_P50u_k0.0001_W3profile', 'chain_W3_profile'),
                          ('W=2 nm', 192, 'SINT_N192_Green_W2_P50u_k0.0001_W2profile', 'chain_W2_profile'),
                          ('W=4 nm, 4x D_surf', 96, 'SINT_N96_Green_W4_P50u_k0.0001_surf4profile', 'chain_surf4_profile')):
    txt = open(f'logs/{log}.log').read()
    Mv, Mva, Ms, Mg = map(float, re.search(r'M_vol/vap/surf/gb = ([\d.e+-]+)/([\d.e+-]+)/([\d.e+-]+)/([\d.e+-]+)', txt).groups())
    h = L0 / N
    for f in sorted(glob.glob(f'exp_data/{name}_snaps/step_*.npz'), key=lambda f: int(os.path.basename(f)[5:-4]))[2:]:
        d = np.load(f); r = np.clip(d['rho'].astype(float), 0, 1); es = np.clip(d['etas'].astype(float), 0, 1); s = float(d['s'])
        k = cr(r[:, N // 2]) * h; x0 = .5 * (k.min() + k.max()); a = .5 * (k.max() - k.min()); x = np.arange(N) * h - x0
        cols = abs(x) < a / 2
        gb = Mg * 2 * es[0] * es[1]; sf = Ms * r ** 2 * (1 - r) ** 2
        dy = h * s
        I_gb = (gb[cols].sum(1) * dy).mean(); I_sf = (sf[cols].sum(1) * dy).mean()
        print(f'{lab:18s} t {float(d["t"])*1e6:3.0f} us  int M_gb-term {I_gb:9.3g}  int M_surf-term {I_sf:9.3g} nm^3/s'
              f'  (1 + surf/gb) = {1 + I_sf / I_gb:.3f}  rho min on GB row (mid) {r[cols, N // 2].min():.3f}')
