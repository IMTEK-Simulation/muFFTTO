"""
Memory per pixel of the elastic solve with none / Green / dual / MG (run_memory_footprint.sh ->
exp_data/memory_footprint.jsonl): table + figures/memory_footprint.png.

Bytes per pixel (voxel) = slope between the two largest grids of each (device, dim): the CUDA context, the imports
and the cupy/allocator granularity cancel.
  problem    discretization, s per quadrature point, rhs, u, a random field, K's scratch (base - context)
  precond    preconditioner object + its first application, minus the same stage of 'none' (two probe vectors)
  cg         the CG solve on top (its vectors; the same for every preconditioner up to temporaries)
"""
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
COLOR = {'none': '#7f7f7f', 'Green': '#2ca02c', 'dual': '#d62728', 'MG': '#6a3d9a'}
LABEL = {'none': 'CG only', 'Green': 'Green', 'dual': 'dual', 'MG': 'MG V(1,1)'}
ORDER = ['none', 'Green', 'dual', 'MG']

rows = [json.loads(line) for line in open(os.path.join(HERE, 'exp_data', 'memory_footprint.jsonl')) if line.strip()]
failed = [r for r in rows if r.get('failed')]
rows = [r for r in rows if not r.get('failed')]
data = {}
for r in rows:
    data.setdefault((r['device'], r['dim'], r['precond']), {})[r['N']] = r


def slope(device, dim, precond, a, b=None):
    runs = data[(device, dim, precond)]
    n1, n2 = sorted(runs)[-2:]
    f = (lambda r: r[a] - r[b]) if b else (lambda r: r[a])
    return (f(runs[n2]) - f(runs[n1])) / (n2 ** dim - n1 ** dim)


summary = {}
print('bytes per pixel (2D) / voxel (3D); slope between the two largest grids')
print('stored = resident after returning the cupy pool cache (GPU) / RSS (CPU); peak = incl. freed temporaries')
print(f'{"device":6} {"dim":3} {"grids":>14} {"config":9} {"problem":>8} {"prec":>6} {"cg":>5} {"stored":>7}'
      f' {"peak":>6} {"stored/none":>11} {"peak/none":>9}  MG levels')
for device in ('gpu', 'cpu'):
    for dim in (2, 3):
        if (device, dim, 'none') not in data:
            continue
        res = (lambda st: st + '_res') if 'prec_res' in next(iter(data[(device, dim, 'none')].values())) \
            else (lambda st: st)
        none_prec = slope(device, dim, 'none', res('prec'), 'base')
        for p in ORDER:
            if (device, dim, p) not in data or len(data[(device, dim, p)]) < 2:
                continue
            problem = slope(device, dim, p, 'base', 'context')
            prec = slope(device, dim, p, res('prec'), 'base') - none_prec
            cg = slope(device, dim, p, res('solve'), res('prec')) + none_prec
            stored = slope(device, dim, p, res('solve'), 'context')
            for r in data[(device, dim, p)].values():
                r['max'] = max(r['prec'], r['solve'], r['peak'])
            peak = slope(device, dim, p, 'max', 'context')
            summary[(device, dim, p)] = dict(problem=problem, prec=prec, cg=cg, stored=stored, peak=peak)
            grids = '/'.join(str(n) for n in sorted(data[(device, dim, p)]))
            levels = data[(device, dim, p)][max(data[(device, dim, p)])].get('levels')
            ref = summary[(device, dim, 'none')]
            print(f'{device:6} {dim:3} {grids:>14} {p:9} {problem:8.0f} {prec:6.0f} {cg:5.0f} {stored:7.0f}'
                  f' {peak:6.0f} {stored / ref["stored"]:11.2f} {peak / ref["peak"]:9.2f}  {levels or ""}')
if failed:
    print('failed:', [(r['device'], r['dim'], r['precond'], r['N']) for r in failed])

print('\nmeasured device (GPU) / process (CPU) memory [MB] at the largest grids, incl. context: peak, stored')
for (device, dim, p), runs in sorted(data.items()):
    n = max(runs)
    r = runs[n]
    print(f'{device:4} {dim}D {n:5d}^{dim}  {p:6} peak {r["max"] / 2**20:8.0f}  stored '
          f'{r.get("solve_res", r["solve"]) / 2**20:8.0f}  (context {r["context"] / 2**20:.0f})')

# figure: GPU bytes per pixel, stacked problem | CG | preconditioner, one panel per dimension
fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
for ax, dim in zip(axes, (2, 3)):
    configs = [p for p in ORDER if ('gpu', dim, p) in summary]
    for y, p in enumerate(configs):
        s = summary[('gpu', dim, p)]
        left = 0.0
        for part, color, hatch in (('problem', '#d9d9d9', None), ('cg', '#a6a6a6', None), ('prec', COLOR[p], None)):
            width = max(s[part], 0.0)
            ax.barh(y, width, left=left, color=color, edgecolor='white', linewidth=2, height=0.6)
            left += width
        if s['peak'] > left * 1.02:
            ax.plot([s['peak']] * 2, [y - 0.32, y + 0.32], color='black', linewidth=2)
            ax.plot([left, s['peak']], [y, y], color='black', linewidth=1, linestyle=':')
        ax.text(max(left, s['peak']) + 25 * dim, y, f'{s["stored"]:.0f} B' +
                (f' (peak {s["peak"]:.0f})' if s['peak'] > left * 1.02 else ''), va='center', fontsize=12)
    ax.set_yticks(range(len(configs)), [LABEL[p] for p in configs], fontsize=13)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.45 * max(summary[('gpu', dim, p)]['peak'] for p in configs))
    ax.set_xlabel(f'bytes per {"pixel" if dim == 2 else "voxel"}', fontsize=13)
    ax.tick_params(axis='x', labelsize=12)
    element = 'bilinear quad' if dim == 2 else 'trilinear hexahedron'
    ax.set_title(f'{dim}D, {element}, double precision', fontsize=13)
    ax.grid(axis='x', color='#e6e6e6')
    ax.set_axisbelow(True)
handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in ('#d9d9d9', '#a6a6a6')] + \
          [plt.Rectangle((0, 0), 1, 1, color=COLOR[p]) for p in ('Green', 'dual', 'MG')] + \
          [plt.Line2D([0], [0], color='black', linewidth=2)]
fig.legend(handles, ['elastic problem (K, s, u, f)', 'CG vectors', 'Green operator', 'dual (Green + 2 vectors)',
                     'multigrid hierarchy', 'peak (freed temporaries)'], loc='lower center', ncol=3, fontsize=12,
           frameon=False)
fig.suptitle('GPU memory of the elastic solve, stored and peak (measured, slope between the two largest grids)',
             fontsize=14)
fig.tight_layout(rect=(0, 0.13, 1, 0.93))
out = os.path.join(HERE, 'figures', 'memory_footprint.png')
fig.savefig(out, dpi=150)
print('saved', out)
