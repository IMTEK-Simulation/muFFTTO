"""
512^3 auxetic TO (target scale 0.3, 4 L40S) from three starts: objective f against iteration and against wall time.

    python plot_512_starts.py   (reads logs/fable/starts512/lines.txt, the ITER / EVAL / job lines of the three logs)
                                -> figures/TO512_starts_convergence.png

Wall time is summed over all Slurm segments of a run, including segments lost to node faults (real cost), and starts
with the time spent before the 512^3 stage (Cahn-Hilliard start job; the chain's 128^3 and 256^3 stages, 1 L40S each;
the chain's 256^3 stage includes 2 h 25 min of one wild trial step, before LBFGS_FIRST_STEP existed).
"""
import re
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
RUNS = {  # log -> (label, colour, style, hours spent before the 512^3 stage: making the start / coarser grids)
    'TO3D_N512_ch512evo_s0.3.log': ('start: Cahn-Hilliard evolved at 512³ (TO interface)', 'tab:red', '-',
                                    9 / 60 + 22 / 3600),                          # CH job 8826536, 1 L40S
    'TO3D_N512_fromCH_s0.3_fused.log': ('start: Cahn-Hilliard 128³ (bi_s11) upsampled', 'tab:blue', '-', 0.0),
    'TO3D_N512_video_s0.3.log': ('chain 128³ → 256³ → 512³ (curve: 512³ stage)', 'tab:green', '-',
                                 (23 * 60 + 49 + 2 * 3600 + 47 * 60 + 56) / 3600),  # 128³ 8820129 + 256³ 8820130, 1 L40S
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
for name, (label, colour, style, before) in RUNS.items():
    its = sorted(runs[name]['iters'])
    f = [runs[name]['iters'][i][0] for i in its]
    hours = [before + runs[name]['iters'][i][1] for i in its]
    final = f'{label}\n{its[-1]} its at 512³, {hours[-1]:.1f} h in total, final f {f[-1]:.4f}'
    axes[0].plot(its, f, style, color=colour, lw=2, label=final)
    axes[1].plot(hours, f, style, color=colour, lw=2)
    if before > 0.5:
        axes[1].plot([0, before], [f[0], f[0]], ':', color=colour, lw=2)  # time on the coarser grids before
        axes[1].annotate('128³ + 256³ stages', (before / 2, f[0]), xytext=(0, 8), textcoords='offset points',
                         ha='center', fontsize=12, color=colour)
for ax, xlabel in zip(axes, ['L-BFGS iteration at 512³', 'total wall time incl. start / coarser grids [h]']):
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
for name, (label, colour, style, before) in RUNS.items():
    its = sorted(runs[name]['iters'])
    print(f'{name}: {its[-1]} its, 512^3 stage {runs[name]["iters"][its[-1]][1]:.2f} h, total '
          f'{before + runs[name]["iters"][its[-1]][1]:.2f} h, f {runs[name]["iters"][its[-1]][0]:.5f}')
