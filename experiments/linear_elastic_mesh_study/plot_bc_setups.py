"""
The three supports of the supports study (bc_study.py): left the set-up (beam pixels, constrained dofs, nodal loads,
void of thickness 0.4 dashed), right the computed deformation (displacement magnified, colour = |u|). For the free
beam the rigid motion (not loaded, not fixed by CG) is removed before drawing.

usage:  python plot_bc_setups.py [--n_H 16]
writes exp_data/bc_setups.png
"""
import argparse
import os

import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection

parser = argparse.ArgumentParser()
parser.add_argument('--n_H', type=int, default=16)
args = parser.parse_args()

HERE = os.path.dirname(os.path.realpath(__file__))
RUNS = os.path.join(HERE, 'exp_data', 'runs')
L, H, VOID = 1.0, 0.2, 0.4
SUPPORTS = {'clamp': 'clamped left face, shear load at the tip',
            'simple': 'pin + roller (3 dofs), shear load at mid-span',
            'free': 'free, -F at mid-span and +F/2 at both ends'}


def load(bc):
    tag = f'_void{VOID:g}' + ('' if bc == 'clamp' else f'_bc{bc}')
    return np.load(os.path.join(RUNS, f'n_H{args.n_H}_cgtol1e-06{tag}.npz'))


def quads(X):
    nx, ny = X.shape[1] - 1, X.shape[2] - 1
    return [[X[:, i, j], X[:, i + 1, j], X[:, i + 1, j + 1], X[:, i, j + 1]] for i in range(nx) for j in range(ny)]


def remove_rigid_motion(X0, U):
    """least-squares translation + small rotation about the centroid, subtracted"""
    x, y = X0[0].ravel() - X0[0].mean(), X0[1].ravel() - X0[1].mean()
    A = np.zeros((2 * x.size, 3))
    A[:x.size, 0], A[x.size:, 1] = 1, 1
    A[:x.size, 2], A[x.size:, 2] = -y, x
    c = np.linalg.lstsq(A, np.concatenate([U[0].ravel(), U[1].ravel()]), rcond=None)[0]
    return U - np.stack([c[0] - c[2] * y, c[1] + c[2] * x]).reshape(U.shape)


def arrow(ax, x, y, dy, **kw):
    ax.annotate('', xy=(x, y + dy), xytext=(x, y), arrowprops=dict(arrowstyle='->', lw=1.5, **kw))


fig, axes = plt.subplots(3, 2, figsize=(15, 9.5), layout='constrained')
for row, (bc, title) in enumerate(SUPPORTS.items()):
    r = load(bc)
    h = float(r['h'])
    U = r['u_beam']
    X0 = np.stack(np.meshgrid(np.arange(U.shape[1]), np.arange(U.shape[2]), indexing='ij')) * h
    n_L, n_H = U.shape[1] - 1, U.shape[2] - 1
    ys = np.arange(n_H + 1) * h

    # set-up
    ax = axes[row, 0]
    ax.add_patch(plt.Rectangle((-VOID, -VOID), L + 2 * VOID, H + 2 * VOID, fill=False, ec='0.5', ls='--', lw=1))
    ax.text(-VOID + 0.02, H + VOID - 0.06, f'void {VOID:g} (zero stiffness), periodic cell', fontsize=8, color='0.4')
    ax.add_collection(PolyCollection(quads(X0), facecolors='#efef9f', edgecolors='#9a9a7a', linewidths=0.3))
    scale = 0.25  # arrow length of the full load F; the loaded node column is drawn as a thick line

    def column_load(x, fraction, colour):
        ax.plot([x, x], [0, H], color=colour, lw=3, solid_capstyle='butt')
        length = scale * abs(fraction)  # downward loads point at the top face, upward ones at the bottom face
        tail = H + 0.02 + length if fraction < 0 else -0.02 - length
        arrow(ax, x, tail, -length if fraction < 0 else length, color=colour)

    if bc == 'clamp':
        ax.plot(np.zeros_like(ys), ys, 's', color='k', ms=4, label='$u_x = u_y = 0$')
        column_load(L, -1, 'C3')
    else:
        column_load((n_L // 2) * h, -1, 'C3')
        if bc == 'simple':
            ax.plot([0], [0], '^', color='k', ms=10, label='pin: $u_x = u_y = 0$')
            ax.plot([L], [0], 'o', color='k', mfc='w', ms=9, mew=1.5, label='roller: $u_y = 0$')
        else:
            for x_end in (0.0, L):
                column_load(x_end, 0.5, 'C0')
    ax.plot([], [], color='C3', lw=3, label='load F, spread over the node column')
    if bc == 'free':
        ax.plot([], [], color='C0', lw=3, label='balancing F/2 at each end')
    ax.set_xlim(-VOID - 0.05, L + VOID + 0.05)
    ax.set_ylim(-VOID - 0.05, H + VOID + 0.05)
    ax.set_aspect('equal')
    ax.legend(loc='lower center', ncol=1 if bc == 'free' else 2, frameon=False, fontsize=8, bbox_to_anchor=(0.5, 0.03))
    ax.set_title(f'{title}   ({int(r["nb_clamped_nodes"]) if "nb_clamped_nodes" in r.files else "?"} '
                 f'constrained nodes, CG its {int(r["cg_its"])})', fontsize=10)

    # deformation
    ax = axes[row, 1]
    if bc == 'free':
        U = remove_rigid_motion(X0, U)
    magnify = 0.06 / np.abs(U).max()  # largest drawn displacement 0.06
    X = X0 + magnify * U
    pc = PolyCollection(quads(X), array=np.linalg.norm(U, axis=0)[:-1, :-1].ravel(), cmap='viridis',
                        edgecolors='0.3', linewidths=0.2)
    ax.add_collection(pc)
    ax.add_collection(PolyCollection(quads(X0), facecolors='none', edgecolors='0.75', linewidths=0.3, ls=':'))
    ax.autoscale()
    ax.set_aspect('equal')
    fig.colorbar(pc, ax=ax, shrink=0.8, pad=0.01).set_label('|u|')
    ax.set_title(f'deformation × {magnify:.0f}' + ('  (rigid motion removed)' if bc == 'free' else '')
                 + f',  compliance F·u = {float(r["compliance"]):.4e}', fontsize=10)
for ax in axes.ravel():
    ax.set_xlabel('x')
    ax.set_ylabel('y')
fig.suptitle(f'supports study: beam L = 1, H = 0.2 in a void of {VOID:g}, $n_H$ = {args.n_H} pixels over the height',
             fontsize=12)
out = os.path.join(HERE, 'exp_data', 'bc_setups.png')
fig.savefig(out, dpi=150)
print(f'wrote {out}')
