"""
Size of the capacitance system S of the corrected Green for a cube of N^3 voxels with 1..6 clamped faces
(all 3 displacement components fixed on every node of a clamped face; faces added in the order x-, x+, y-, y+, z-, z+,
so 2 faces = opposite ends of a beam, 6 = the whole surface). Node counts are exact (shared edges / corners counted
once).

    m          fixed dofs = size of S
    dense S    8 m^2 bytes (float64; its Cholesky factor or inverse is the same size)
    setup      m Green applications to assemble S column by column, (1/3) m^3 flops to factorize it
    reference  one displacement field on the grid, 3 (N+1)^3 * 8 bytes

usage:  python dirichlet_size.py      -> exp_data/dirichlet_size.png, exp_data/dirichlet_size.md
"""
import os

import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker

HERE = os.path.dirname(os.path.realpath(__file__))
N_LIST = np.array([8, 16, 32, 64, 128, 256, 512, 1024])
FACES = ['x-', 'x+', 'y-', 'y+', 'z-', 'z+']


def fixed_nodes(N, k):
    """exact number of nodes on the first k faces of an (N+1)^3 node grid (inclusion-exclusion over the faces)"""
    n = N + 1
    # faces as (axis, side); a node set on face f is {i_axis = side}; intersections of j distinct faces are
    # n^(3 - #distinct axes) if no axis appears with both sides, else empty
    faces = [(a, s) for a in range(3) for s in (0, 1)][:k]
    total = 0
    for mask in range(1, 2 ** k):
        sel = [faces[i] for i in range(k) if mask >> i & 1]
        axes = [a for a, _ in sel]
        if len(set(axes)) < len(axes):  # opposite faces of the same axis do not meet
            continue
        total += (-1) ** (len(sel) + 1) * n ** (3 - len(sel))
    return total


rows = []
m = {k: np.array([3 * fixed_nodes(N, k) for N in N_LIST], dtype=float) for k in range(1, 7)}
field_bytes = 3 * (N_LIST + 1.0) ** 3 * 8
for k in range(1, 7):
    for N, mk, fb in zip(N_LIST, m[k], field_bytes):
        rows.append((k, N, int(mk), 8 * mk ** 2, mk ** 3 / 3, fb))

fig, axes = plt.subplots(1, 3, figsize=(18, 5.6), layout='constrained')
colours = plt.get_cmap('viridis')(np.linspace(0.05, 0.85, 6))
for k in range(1, 7):
    lab = f'{k} clamped face{"s" if k > 1 else ""}' + (' (opposite)' if k == 2 else '') + (' (all)' if k == 6 else '')
    axes[0].plot(N_LIST, m[k], 'o-', color=colours[k - 1], label=lab)
    axes[1].plot(N_LIST, 8 * m[k] ** 2, 'o-', color=colours[k - 1])
    axes[2].plot(N_LIST, m[k] ** 3 / 3, 'o-', color=colours[k - 1])
axes[0].plot(N_LIST, 3 * (N_LIST + 1.0) ** 3, 'k--', lw=1, label='all dofs of the grid, 3 (N+1)$^3$')
axes[1].plot(N_LIST, field_bytes, 'k--', lw=1, label='one displacement field on the grid')
for level, name in ((16e9, '16 GB laptop'), (256e9, '256 GB node'), (4e12, '4 TB')):
    axes[1].axhline(level, color='0.5', lw=0.8, ls=':')
    axes[1].text(N_LIST[0], level * 1.3, name, fontsize=8, color='0.3')
for flops, name in ((1e12 * 60, '1 min at 1 TFLOP/s'), (1e12 * 86400, '1 day at 1 TFLOP/s')):
    axes[2].axhline(flops, color='0.5', lw=0.8, ls=':')
    axes[2].text(N_LIST[0], flops * 1.5, name, fontsize=8, color='0.3')
titles = ('size of S: fixed dofs m', 'memory of dense S: 8 m$^2$ bytes', 'factorizing S: m$^3$/3 flops')
ylabels = ('m', 'bytes', 'flops')
for ax, t, yl in zip(axes, titles, ylabels):
    ax.set(xscale='log', yscale='log', xlabel='N (cube of N$^3$ voxels)', ylabel=yl, title=t)
    ax.set_xticks(N_LIST, [str(n) for n in N_LIST])
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.grid(True, color='0.9', which='both')
axes[0].legend(frameon=False, fontsize=8)
axes[1].legend(frameon=False, fontsize=8, loc='lower right')
fig.suptitle('Corrected Green on an N$^3$ cube: the capacitance system S grows like N$^2$, its dense storage like N$^4$, '
             'its factorization like N$^6$', fontsize=12)
fig.savefig(os.path.join(HERE, 'exp_data', 'dirichlet_size.png'), dpi=110)


def human(b):
    for unit in ('B', 'KB', 'MB', 'GB', 'TB', 'PB'):
        if b < 1000:
            return f'{b:.3g} {unit}'
        b /= 1000
    return f'{b:.3g} EB'


with open(os.path.join(HERE, 'exp_data', 'dirichlet_size.md'), 'w') as out:
    out.write('| clamped faces | N | fixed dofs m | dense S | factorization flops | one grid field |\n|' + '---|' * 6 + '\n')
    for k, N, mk, sb, fl, fb in rows:
        out.write(f'| {k} | {N} | {mk:,} | {human(sb)} | {fl:.2e} | {human(fb)} |\n'.replace(',', ' '))
print('wrote exp_data/dirichlet_size.png, exp_data/dirichlet_size.md')
