"""
Mid slices (z = 1/2) of the 512^3 designs: one row per run, start | start zoom | final | final zoom, exact pixels.

    python plot_512_designs.py      (designs copied from $WS/runs/FABLE/data to data/fable512, see ROWS)
                                    -> figures/TO512_designs_slices.png
Zoom: the first quarter of the cell in x and y. Black = stiff (rho = 1), white = soft.
"""
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import netCDF4
import numpy as np

HERE = Path(__file__).resolve().parent
D = HERE / 'data' / 'fable512'
ROWS = [  # (run, start file, start label, final file, final label)
    ('A: start = Cahn-Hilliard evolved at 512³ with the TO interface',
     D / 'CH512_start_s11_eta2h.nc', 'start (CH 512³, 9 min)',
     D / 'N512_soft0.001_eta0.00390625_energy_ch512evo_s0.3_final.nc', 'final: 404 its, 6.3 h, f 1.2576'),
    ('B: start = Cahn-Hilliard 128³ bi_s11, upsampled to 512³',
     HERE / 'data' / 'CH3D' / 'CH3D_start_bi_s11.nc', 'start (CH 128³)',
     D / 'N512_soft0.001_eta0.00390625_energy_fromCH_s0.3_fused_final.nc', 'final: 446 its, 17.1 h, f 1.2645'),
    ('C: chain 128³ → 256³ → 512³',
     D / 'N256_soft0.001_eta0.0078125_energy_video_s0.3_final.nc', 'start (256³ result, after 3.2 h)',
     D / 'N512_soft0.001_eta0.00390625_energy_video_s0.3_final.nc', 'final: 56 its at 512³, 5.5 h, f 1.1706'),
    ('backup start, not run',
     D / 'CH512_start_s11_eta2h_coarse.nc', 'coarser CH 512³ (T 12000 ε²)', None, None),
]


def mid_slice(path):
    with netCDF4.Dataset(path) as data:
        field = data['phase_field_iterate'] if 'phase_field_iterate' in data.variables else data['phase_field_solution']
        n = field.shape[-1]
        k = field.shape[0] - 1
        return np.asarray(field[k, ..., n // 2]).reshape(n, n)


fig, axes = plt.subplots(len(ROWS), 4, figsize=(22, 6 * len(ROWS)))
for row, (run, start, start_label, final, final_label) in zip(axes, ROWS):
    for ax in row:
        ax.set_axis_off()
    for (path, label), (full, zoom) in zip([(start, start_label), (final, final_label)], [row[:2], row[2:]]):
        if path is None:
            continue
        rho = mid_slice(path)
        n = rho.shape[0]
        solid, grey = float(np.mean(rho >= 0.5)), float(np.mean((rho > 0.1) & (rho < 0.9)))
        full.imshow(rho.T, origin='lower', cmap='gray_r', vmin=0, vmax=1, interpolation='nearest')
        zoom.imshow(rho[:n // 4, :n // 4].T, origin='lower', cmap='gray_r', vmin=0, vmax=1, interpolation='nearest')
        full.set_title(f'{label}\n{n}³, slice: solid {solid:.2f}, grey {grey:.3f}', fontsize=14)
        zoom.set_title('zoom: first quarter', fontsize=14)
        print(run, label, n, f'solid {solid:.3f} grey {grey:.3f}', flush=True)
    row[0].text(-0.06, 0.5, run, transform=row[0].transAxes, rotation=90, ha='right', va='center', fontsize=15)
fig.suptitle('512³ auxetic cell (target ν = -0.5, scale 0.3): mid slices z = 1/2, black = stiff', fontsize=17)
fig.tight_layout(rect=(0.02, 0, 1, 0.98))
out = HERE / 'figures' / 'TO512_designs_slices.png'
fig.savefig(out, dpi=120)
print('saved', out)
