"""2048^2 TO design (contrast 1e3, l = 4h = 1/512, dual, NEMO): 2 x 2 periodic cells and a zoom on an inclusion tip
with the pixel grid visible.   python plot_TO_2048_design.py -> figures/TO_2048_design.{png,pdf} (+ talk figures/)"""
import os, shutil
import matplotlib.pyplot as plt, netCDF4, numpy as np
from matplotlib.patches import Rectangle
from scipy import ndimage
HERE = os.path.dirname(os.path.realpath(__file__))
TALK = os.path.expanduser('~/Projects/Talks/Slides-Green-Jacobi-topology-optimization-NANTES/figures')
r = np.asarray(netCDF4.Dataset(os.path.join(HERE, 'data/example_2D_elasticity_TO/nemo/'
     'N2048_soft0.001_energy-warm_dual_eta_0.001953125_w_5.0_l_bfgs_final.nc'))['phase_field_solution'][:]).squeeze().T
n = r.shape[0]
lab, _ = ndimage.label(r < 0.5)
best = max((sl for sl in ndimage.find_objects(lab) if sl[0].start > 50 and sl[1].start > 50 and sl[0].stop < n - 50
            and sl[1].stop < n - 50), key=lambda sl: sl[1].stop - sl[1].start)
ys, xs = best
row = (ys.start + ys.stop) // 2
x_tip = xs.stop                                   # right tip of the widest inclusion
z = 96
y0, x0 = row - z // 2, x_tip - z // 2 - 10
plt.rcParams.update({'text.usetex': True, 'font.family': 'serif', 'font.size': 13})
fig, (a, b) = plt.subplots(1, 2, figsize=(9, 4.6), layout='constrained')
a.imshow(np.tile(r, (2, 2)), cmap='gray_r', vmin=0, vmax=1, origin='lower', interpolation='nearest')
a.add_patch(Rectangle((x0, y0), z, z, fill=False, ec='tab:orange', lw=2))
a.set_title(r'$2048^2$, contrast $10^3$, $l = 4h$'), a.axis('off')
b.imshow(r[y0:y0 + z, x0:x0 + z], cmap='gray_r', vmin=0, vmax=1, origin='lower', interpolation='nearest')
b.set_title(f'zoom: ${z}\\times{z}$ pixels'), b.set_xticks([]), b.set_yticks([])
for s in b.spines.values():
    s.set_color('tab:orange'), s.set_linewidth(2.5)
out = os.path.join(HERE, 'figures', 'TO_2048_design')
for ext in ('png', 'pdf'):
    fig.savefig(f'{out}.{ext}', dpi=200)
    shutil.copy(f'{out}.{ext}', TALK)
print('wrote', out)
