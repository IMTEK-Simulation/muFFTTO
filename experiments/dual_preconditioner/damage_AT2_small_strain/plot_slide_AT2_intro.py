"""
Panels for the slide 'Phase-field fracture (AT2)' (fracture counterpart of the TO slide 'Phase-field regularization',
plot_phase_field_panels.py): final state of the 1024^2 polydisperse-pores run (l = 1/32 = 32 h, dual), pores white.
  frac_intro_field.pdf    damage field, box = zoom window
  frac_intro_zoom.pdf     zoom across the crack (exact pixels), dashed = profile cut
  frac_intro_field_zoom.pdf  both, side by side, with zoom connector lines (used on the slide)
  frac_intro_profile.pdf  d = exp(-|x|/l) (exact 1D AT2 profile), stiffness g(d) = (1-d)^2 (1-k) + k, width d > 0.1
                          = 2 ln10 l ~ 4.6 l, (the run's damage along the cut is computed but not drawn)
    python plot_slide_AT2_intro.py   [OUT=<dir>: write there instead of figures/ + the talk]
"""
import os
import shutil
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import netCDF4
import numpy as np
from matplotlib.patches import Rectangle

from muFFTTO import microstructure_library

HERE = os.path.dirname(os.path.realpath(__file__))
TAG = 'AT2_N1024_dual_polydisperse_pores_l0.03125'
TALK = os.path.expanduser('~/Projects/Talks/Slides-Green-Jacobi-topology-optimization-NANTES/figures')
OUT = os.environ.get('OUT')
plt.rcParams.update({'text.usetex': True, 'font.family': 'serif', 'font.size': 13})
k = 1e-4

with netCDF4.Dataset(os.path.join(HERE, 'exp_data', f'{TAG}_restart.nc')) as ds:
    d = np.asarray(ds['damage'][-1, 0], float)
    micro = getattr(ds, 'microstructure', None)
with netCDF4.Dataset(os.path.join(HERE, 'exp_data', f'{TAG}.nc')) as ds:
    micro = micro or getattr(ds, 'microstructure', 'circle_inclusion')
N = d.shape[0]
L_PX = N / 32                                          # l = 1/32 of the cell = 32 pixels
X = np.stack(np.meshgrid(np.arange(N + 1) / N, np.arange(N + 1) / N, indexing='ij'))
pore = microstructure_library.get_geometry(nb_voxels=(N, N), microstructure_name=micro, coordinates=X[:, :-1, :-1]) == 0

# profile cut PERPENDICULAR to the crack: normal = eigenvector of the most negative curvature of the (smoothed) damage
# ridge; candidates = fully cracked pixels away from the boundary, the cut (+- 3.5 l) must stay clear of pores and see
# a single clean band (low damage at both ends)
from scipy import ndimage
Z = 192
HALF = 3.5 * L_PX
SIG = L_PX / 2                                         # Hessian scale ~ the band width (3 px gave a tilted cut)
Hxx = ndimage.gaussian_filter(d, SIG, order=(2, 0), mode='wrap')
Hyy = ndimage.gaussian_filter(d, SIG, order=(0, 2), mode='wrap')
Hxy = ndimage.gaussian_filter(d, SIG, order=(1, 1), mode='wrap')
t = np.linspace(-HALF, HALF, int(2 * HALF) + 1)


def normal(x, y):
    w, v = np.linalg.eigh(np.array([[Hxx[x, y], Hxy[x, y]], [Hxy[x, y], Hyy[x, y]]]))
    return v[:, 0]                                       # direction of the strongest (negative) curvature


best = None
for x, y in zip(*np.nonzero(d > 0.98)):
    if not (Z // 2 <= x < N - Z // 2 and Z // 2 <= y < N - Z // 2) or (x + y) % 3:
        continue
    n = normal(x, y)
    px, py = x + t * n[0], y + t * n[1]
    if pore[np.round(px).astype(int) % N, np.round(py).astype(int) % N].any():
        continue
    prof = ndimage.map_coordinates(d, [px, py], order=1, mode='grid-wrap')
    score = prof[:10].max() + prof[-10:].max()
    if best is None or score < best[0]:
        best = (score, x, y, n, prof)
_, cx, cy, nrm, cut = best
# cut direction from the crack core itself: principal axis of the fully cracked pixels (d > 0.97) within 1.5 l of the
# centre (the smoothed Hessian is pulled by the pores beside the band), cut = perpendicular to it
ii, jj = np.nonzero(d > 0.97)
sel = (ii - cx) ** 2 + (jj - cy) ** 2 <= (1.5 * L_PX) ** 2
P = np.stack([ii[sel] - ii[sel].mean(), jj[sel] - jj[sel].mean()])
tang = np.linalg.eigh(P @ P.T)[1][:, -1]                # crack direction
nrm = np.array([-tang[1], tang[0]])
nrm = nrm if nrm[1] > 0 else -nrm
_r = np.deg2rad(8)                                       # user: rotate the cut a bit further (by eye, slide 16)
nrm = np.array([np.cos(_r) * nrm[0] - np.sin(_r) * nrm[1], np.sin(_r) * nrm[0] + np.cos(_r) * nrm[1]])
cut = ndimage.map_coordinates(d, [cx + t * nrm[0], cy + t * nrm[1]], order=1, mode='grid-wrap')
x0, y0 = cx - Z // 2, cy - Z // 2
print('zoom centre', cx, cy, 'normal', np.round(nrm, 3))

cmap = plt.get_cmap('inferno_r')


def rgb(field, mask):
    img = cmap(np.clip(field, 0, 1))
    img[mask] = (1, 1, 1, 1)
    return np.transpose(img, (1, 0, 2))                # x horizontal, y vertical


def save(fig, name):
    for ext in ('pdf', 'png'):
        p = os.path.join(OUT or os.path.join(HERE, 'figures'), f'{name}.{ext}')
        fig.savefig(p, dpi=300, bbox_inches='tight', pad_inches=0.02)
        if not OUT and os.path.isdir(TALK):
            shutil.copy(p, TALK)
    plt.close(fig)


fig, a = plt.subplots(figsize=(4, 4))
a.imshow(rgb(d, pore), origin='lower', interpolation='nearest')
a.add_patch(Rectangle((x0 - 0.5, y0 - 0.5), Z, Z, fill=False, ec='tab:blue', lw=2.5))
a.set_xticks([]), a.set_yticks([])
save(fig, 'frac_intro_field')

fig, a = plt.subplots(figsize=(4, 4))
a.imshow(rgb(d[x0:x0 + Z, y0:y0 + Z], pore[x0:x0 + Z, y0:y0 + Z]), origin='lower', interpolation='nearest')
a.plot(Z // 2 + t * nrm[0], Z // 2 + t * nrm[1], color='tab:blue', lw=2, ls=(0, (4, 3)))   # the cut (x right, y up)
a.set_xlim(-0.5, Z - 0.5), a.set_ylim(-0.5, Z - 0.5)          # keep the image extent (the cut may reach past it)
a.set_xticks([]), a.set_yticks([])
for s in a.spines.values():
    s.set_color('tab:blue'), s.set_linewidth(3)
save(fig, 'frac_intro_zoom')

# both panels in one figure with zoom connector lines: box corners (right side) -> left corners of the zoom panel
from matplotlib.patches import ConnectionPatch
fig = plt.figure(figsize=(8.8, 4))
af = fig.add_axes([0, 0, 4 / 8.8, 1])
az = fig.add_axes([4.8 / 8.8, 0, 4 / 8.8, 1])
af.imshow(rgb(d, pore), origin='lower', interpolation='nearest')
af.add_patch(Rectangle((x0 - 0.5, y0 - 0.5), Z, Z, fill=False, ec='tab:blue', lw=2.5))
az.imshow(rgb(d[x0:x0 + Z, y0:y0 + Z], pore[x0:x0 + Z, y0:y0 + Z]), origin='lower', interpolation='nearest')
az.plot(Z // 2 + t * nrm[0], Z // 2 + t * nrm[1], color='tab:blue', lw=2, ls=(0, (4, 3)))
az.set_xlim(-0.5, Z - 0.5), az.set_ylim(-0.5, Z - 0.5)
for a_ in (af, az):
    a_.set_xticks([]), a_.set_yticks([])
for sp in az.spines.values():
    sp.set_color('tab:blue'), sp.set_linewidth(3)
for yb, yz in ((y0 - 0.5 + Z, 1), (y0 - 0.5, 0)):
    fig.add_artist(ConnectionPatch(xyA=(x0 - 0.5 + Z, yb), coordsA='data', axesA=af, xyB=(0, yz), coordsB='axes fraction',
                                   axesB=az, color='tab:blue', lw=1.5))
save(fig, 'frac_intro_field_zoom')

fig, b = plt.subplots(figsize=(3.3, 2.0))               # small canvas: labels ~8 pt at 0.78 x the slide column
s = np.linspace(-6, 6, 2401)
ex = np.exp(-np.abs(s))
b.plot(s, (1 - ex) ** 2 * (1 - k) + k, color='tab:blue', lw=2.5, label='stiffness $g(d)$')
b.plot(s, ex, color='k', lw=3, label='damage $d$')
xc = t - t[cut >= 0.97 * cut.max()].mean()          # centre of the crack (may lie between pixels)
# (the run's damage along the cut, xc / L_PX vs cut, follows the exact curve; not drawn on the slide)
w = np.log(10)
b.annotate('', xy=(-w, 0.1), xytext=(w, 0.1), arrowprops=dict(arrowstyle='<->', color='tab:red', lw=2))
b.text(-w - 0.2, 0.2, r'$\propto l$', color='tab:red', ha='right', va='center', fontsize=12,
       bbox=dict(facecolor='white', edgecolor='none', pad=1))
b.set_xlim(-6, 6), b.set_ylim(0, 1.08), b.set_xticks(range(-6, 7, 2))
b.set_yticks([0, 0.5, 1], ['$0$', '$0.5$', '$1$'])   # 0 (slide: d in [0, 1]); g(d) bottoms out at k = 1e-4, invisible here
b.set(xlabel='distance across the crack  $x/l$')
b.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), fontsize=10, frameon=False, handlelength=1.3, borderpad=0.3, labelspacing=0.4)   # right of the plot
b.grid(alpha=0.3)
save(fig, 'frac_intro_profile')
print('wrote frac_intro_field, frac_intro_zoom, frac_intro_profile to', OUT or 'figures/ and the talk')
