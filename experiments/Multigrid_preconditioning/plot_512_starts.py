"""
512^3 auxetic TO (target scale 0.3, 4 L40S) from three starts: objective f against iteration and against wall time.

    python plot_512_starts.py   (reads logs/fable/starts512/lines.txt, the ITER / EVAL / job lines of the three logs)
                                -> figures/TO512_starts_convergence.png

Wall time is summed over all Slurm segments of a run, including segments lost to node faults (real cost).
"""
import re
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
RUNS = {  # log -> (label, colour, style)
    'TO3D_N512_ch512evo_s0.3.log': ('start: Cahn-Hilliard evolved at 512³ (TO interface)', 'tab:red', '-'),
    'TO3D_N512_fromCH_s0.3_fused.log': ('start: Cahn-Hilliard 128³ (bi_s11) upsampled', 'tab:blue', '-'),
    'TO3D_N512_video_s0.3.log': ('chain 64³→512³: 512³ stage from the 256³ result', 'tab:green', '-'),
}


def read(lines):
    runs, name = {}, None
    for line in lines:
        if line.startswith('#FILE'):
            name = line.split()[1]
            runs[name] = dict(iters={}, offset=0.0, last_wall=0.0)
        elif line.startswith('=== job') and ' done ' not in line:
            run = runs[name]
            run['offset'] += run['last_wall']  # new segment: its wall clock starts again at 0
            run['last_wall'] = 0.0
        elif line.startswith('EVAL'):
            runs[name]['last_wall'] = float(re.search(r'wall ([\d.]+)', line).group(1))
        elif line.startswith('ITER'):
            run = runs[name]
            it, f = int(line.split()[1]), float(line.split()[3])
            run['iters'][it] = (f, (run['offset'] + run['last_wall']) / 3600.0)  # repeated after a restart: keep latest
    return runs


runs = read((HERE / 'logs' / 'fable' / 'starts512' / 'lines.txt').read_text().splitlines())
fig, axes = plt.subplots(1, 2, figsize=(14, 5.6))
for name, (label, colour, style) in RUNS.items():
    its = sorted(runs[name]['iters'])
    f = [runs[name]['iters'][i][0] for i in its]
    hours = [runs[name]['iters'][i][1] for i in its]
    final = f'{label}\n{its[-1]} its, {hours[-1]:.1f} h, final f {f[-1]:.4f}'
    axes[0].plot(its, f, style, color=colour, lw=2, label=final)
    axes[1].plot(hours, f, style, color=colour, lw=2)
for ax, xlabel in zip(axes, ['L-BFGS iteration at 512³', 'wall time at 512³ on 4 L40S [h]']):
    ax.set_ylim(0, 4)
    ax.set_xlim(left=0)
    ax.set_xlabel(xlabel, fontsize=14)
    ax.set_ylabel('objective f (misfit + interface)', fontsize=14)
    ax.tick_params(labelsize=13)
    ax.grid(alpha=0.3)
fig.legend(*axes[0].get_legend_handles_labels(), loc='lower center', ncol=3, fontsize=12, frameon=False)
fig.suptitle('512³ auxetic cell (target ν = -0.5, scale 0.3): which start is cheapest', fontsize=15)
fig.tight_layout(rect=(0, 0.17, 1, 1))
out = HERE / 'figures' / 'TO512_starts_convergence.png'
fig.savefig(out, dpi=110)
print('saved', out)
for name, (label, *_) in RUNS.items():
    its = sorted(runs[name]['iters'])
    print(f'{name}: {its[-1]} its, {runs[name]["iters"][its[-1]][1]:.2f} h, f {runs[name]["iters"][its[-1]][0]:.5f}')
