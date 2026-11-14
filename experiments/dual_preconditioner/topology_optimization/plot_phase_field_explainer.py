"""
Slide figure 'what is a phase field' (rho = phase-field variable, 1 solid / 0 void):
  A  one inclusion of the 1024^2 TO design (contrast 1e4, eta = 4h, dual; NEMO eta / contrast study):
     sharp (rho > 0.5) | phase field rho | zoom on the band with the pixel grid
  B  1D profile across the interface: sharp step vs the phase-field profile rho(s) = 1 / (1 + exp(-s / eta))
     (equipartition profile of eta |grad rho|^2 + rho^2 (1 - rho)^2 / eta), pixel values for eta / h = 1 and 4
  equations: stiffness C(x) = [k + (1 - k) rho^p] C_0 (p = 2, k = 1e-4) and the perimeter energy
    python plot_phase_field_explainer.py  ->  figures/phase_field_explainer.{png,pdf} (+ copy to the talk figures/)
"""
import os
import shutil

import matplotlib.pyplot as plt
import netCDF4
import numpy as np
from matplotlib.patches import Rectangle
from scipy import ndimage

HERE = os.path.dirname(os.path.realpath(__file__))
FINAL = os.path.join(HERE, 'data', 'example_2D_elasticity_TO', 'nemo',
                     'N1024_soft0.0001_energy-warm_dual_eta_0.00390625_w_5.0_l_bfgs_final.nc')
TALK = os.path.expanduser('~/Projects/Talks/Slides-Green-Jacobi-topology-optimization-NANTES/figures')
plt.rcParams.update({'font.size': 15, 'mathtext.fontset': 'cm'})

rho = np.asarray(netCDF4.Dataset(FINAL)['phase_field_solution'][:]).squeeze()   # [x, y]
n = rho.shape[0]
eta_h = 0.00390625 * n                                                           # 4

# one inclusion (void, rho < 0.5) away from the cell boundary: window around its bounding box
lab, nl = ndimage.label(rho < 0.5)
best = None
for i, sl in enumerate(ndimage.find_objects(lab)):
    x0, x1, y0, y1 = sl[0].start, sl[0].stop, sl[1].start, sl[1].stop
    if x0 > 20 and y0 > 20 and x1 < n - 20 and y1 < n - 20:
        size = max(x1 - x0, y1 - y0)
        if best is None or size > best[0]:
            best = (size, x0, x1, y0, y1)
size, x0, x1, y0, y1 = best
cx, cy, half = (x0 + x1) // 2, (y0 + y1) // 2, size // 2 + 24
win = rho[cx - half:cx + half, cy - half:cy + half].T           # [y, x] for imshow
m = win.shape[0]
# zoom: 36 x 36 pixels on the interface at the right-hand tip region (the interface pixel closest to the centre-right)
zs = 36
col = m // 2                                                    # upper edge of the inclusion, straight part
rows = np.where((win[:, col] > 0.5)[1:] != (win[:, col] > 0.5)[:-1])[0]
zr = rows[rows > m // 2][0] if np.any(rows > m // 2) else rows[-1]
zx0, zy0 = int(np.clip(col - zs // 2, 0, m - zs)), int(np.clip(zr - zs // 2, 0, m - zs))
zoom = win[zy0:zy0 + zs, zx0:zx0 + zs]

fig = plt.figure(figsize=(17, 9.5))
gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 0.92], hspace=0.38, wspace=0.12,
                      left=0.05, right=0.93, top=0.93, bottom=0.08)
kw = dict(cmap='gray_r', vmin=0, vmax=1, interpolation='nearest', origin='lower')
a1, a2, a3 = (fig.add_subplot(gs[0, j]) for j in range(3))
a1.imshow((win > 0.5).astype(float), **kw)
a1.set_title('sharp interface:  $\\rho \\in \\{0, 1\\}$')
im = a2.imshow(win, **kw)
a2.set_title('phase field:  $0 \\leq \\rho \\leq 1$')
a2.add_patch(Rectangle((zx0 - 0.5, zy0 - 0.5), zs, zs, fill=False, ec='tab:orange', lw=2.5))
a3.imshow(zoom, **kw)
for k in np.arange(-0.5, zs, 1):
    a3.axhline(k, color='tab:blue', lw=0.4, alpha=0.6)
    a3.axvline(k, color='tab:blue', lw=0.4, alpha=0.6)
a3.set_title(f'zoom with pixels ($\\eta$ = {eta_h:g} h)')
for s in a3.spines.values():
    s.set_color('tab:orange'), s.set_linewidth(2.5)
for a in (a1, a2, a3):
    a.set_xticks([]), a.set_yticks([])
cax = fig.add_axes([0.945, 0.56, 0.012, 0.36])
cb = fig.colorbar(im, cax=cax)
cb.set_label('$\\rho$  (1 solid, 0 void)')
fig.text(0.01, 0.955, 'A', fontsize=22, weight='bold')

# B: 1D profile
b = fig.add_subplot(gs[1, 0:2])
s = np.linspace(-6, 6, 1201)                       # in units of eta
prof = lambda x: 1.0 / (1.0 + np.exp(-x))
b.plot(s, (s > 0).astype(float), color='0.55', lw=2.5, ls='--', label='sharp interface')
b.plot(s, prof(s), color='k', lw=3, label='phase field $\\rho(s) = 1 / (1 + e^{-s/\\eta})$')
for eh, col, mk, ms, lab in [(1, 'tab:purple', 'o', 11, '$\\eta$ / h = 1: 2-3 pixels in the band (under-resolved)'),
                             (4, 'tab:orange', 's', 7, '$\\eta$ / h = 4: resolved')]:
    xs = np.arange(-6, 6 + 1e-9, 1.0 / eh) + (0.5 / eh if eh == 1 else 0.0)
    xs = xs[(xs >= -6) & (xs <= 6)]
    b.plot(xs, prof(xs), mk, color=col, ms=ms, mfc='white' if eh == 1 else col, mew=2.2, label=lab, zorder=5)
w = np.log(9)
b.annotate('', xy=(-w, 0.5), xytext=(w, 0.5), arrowprops=dict(arrowstyle='<->', color='tab:red', lw=2))
b.text(-w - 0.25, 0.5, 'band 0.1 < $\\rho$ < 0.9:  4.4 $\\eta$', color='tab:red', ha='right', va='center', fontsize=14)
b.set_xlim(-6, 6), b.set_ylim(-0.05, 1.08)
b.set_xticks(range(-6, 7, 2))
b.set(xlabel='distance across the interface  s / $\\eta$', ylabel='$\\rho$')
b.legend(loc='upper left', fontsize=12.5, framealpha=0.95)
b.grid(alpha=0.3)
fig.text(0.01, 0.47, 'B', fontsize=22, weight='bold')

# equations
e = fig.add_subplot(gs[1, 2])
e.set_axis_off()
e.text(0.0, 0.92, 'stiffness of the mixture:', fontsize=16)
e.text(0.04, 0.76, '$\\mathbb{C}(\\mathbf{x}) = \\left[\\,k + (1-k)\\,\\rho(\\mathbf{x})^{p}\\,\\right]\\,\\mathbb{C}_0$',
       fontsize=21)
e.text(0.04, 0.63, '$p = 2$,   $k = 10^{-4}$  (contrast $10^4$)', fontsize=15, color='0.3')
e.text(0.0, 0.43, 'interface energy (perimeter):', fontsize=16)
e.text(0.04, 0.25, '$\\int_\\Omega \\eta\\,|\\nabla\\rho|^2 + \\frac{1}{\\eta}\\,\\rho^2(1-\\rho)^2\\;'
       'd\\mathbf{x}\\;\\to\\;\\frac{1}{3}\\,|\\Gamma|$', fontsize=21)
e.text(0.04, 0.10, 'as $\\eta \\to 0$;  $|\\Gamma|$ = length of the interface', fontsize=15, color='0.3')

out = os.path.join(HERE, 'figures', 'phase_field_explainer')
for ext in ('png', 'pdf'):
    fig.savefig(f'{out}.{ext}', dpi=150)
    if os.path.isdir(TALK):
        shutil.copy(f'{out}.{ext}', os.path.join(TALK, f'phase_field_explainer.{ext}'))
print('wrote', out + '.png/.pdf', '(copied to', TALK + ')')
