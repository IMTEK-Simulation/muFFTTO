"""PCG / solver-effort report of a contact-fracture run.

usage: python plot_solver_effort.py RUN/contact_fracture.nc [OUT_PREFIX]
writes OUT_PREFIX_pcg.png and prints a summary table (markdown).
Written for the shear-loaded frame / beam runs (x axis F10; crack events of the beam-root regions
and of the frame, `regions` below); other geometries need their own regions.
"""
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import netCDF4
import numpy as np

# colors: reference palette, light surface
SURFACE, TEXT_1, TEXT_2, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1", "#a3a29c"
SERIES = "#2a78d6"
CRACK = 0.95                        # damage above this counts as cracked

path = sys.argv[1]
out_prefix = sys.argv[2] if len(sys.argv) > 2 else os.path.join(os.path.dirname(path), "solver_effort")
d = netCDF4.Dataset(path)
F10 = np.asarray(d["F10"][:])
hessp = np.asarray(d["nb_hessp"][:]).astype(int)          # mechanics PCG iterations (Steihaug CG)
outer = np.asarray(d["nb_outer"][:]).astype(int)          # trust-region Newton steps
stagger = np.asarray(d["staggered_iterations"][:]).astype(int)
cg_d = np.asarray(d["cg_damage"][:]).astype(int)          # damage PCG iterations
elapsed = np.asarray(d["elapsed_time"][:])
d_max = np.asarray(d["damage_max"][:])
n = int(d.nb_grid_pts[0])
wall = np.diff(np.concatenate([[0.0], elapsed]))           # wall time per increment

# crack events: increments where the damage of a region first exceeds CRACK
dmg = np.asarray(d["damage"][:])
dm = dmg.reshape(dmg.shape[0], -1, n, n)[:, 0]
x = np.arange(n) / n
regions = {"right beam root": (0.75, 0.88, 0.50, 0.70), "left beam root": (0.12, 0.25, 0.30, 0.50)}
events = []
for name, (x0, x1, y0, y1) in regions.items():
    series = dm[:, (x >= x0)[:, None] & (x < x1)[:, None] & ((x >= y0) & (x < y1))[None, :]].max(axis=1)
    hit = np.nonzero(series > CRACK)[0]
    if len(hit):
        events.append((F10[hit[0]], name))
frame = ~np.zeros((n, n), bool)
for (x0, x1, y0, y1) in regions.values():
    frame[np.ix_((x >= x0) & (x < x1), (x >= y0) & (x < y1))] = False
hit = np.nonzero(dm[:, frame].max(axis=1) > CRACK)[0]
if len(hit):
    events.append((F10[hit[0]], "frame"))

per_newton = hessp / np.maximum(outer, 1)
per_stagger_damage = cg_d / np.maximum(stagger, 1)

# ---- figure: small multiples, one measure per axis, shared x ------------------
panels = [(hessp, "Mechanics PCG iterations per increment (= Hessian products)"),
          (per_newton, "Mechanics PCG iterations per Newton step"),
          (stagger, "Staggered iterations per increment"),
          (per_stagger_damage, "Damage PCG iterations per staggered iteration")]
fig, axes = plt.subplots(len(panels), 1, figsize=(10, 10), sharex=True, facecolor=SURFACE)
fig.subplots_adjust(left=0.09, right=0.97, top=0.88, bottom=0.07, hspace=0.45)
fig.text(0.09, 0.962, f"Solver effort per load increment — {n} × {n}", fontsize=15, fontweight="bold",
         color=TEXT_1)
fig.text(0.09, 0.935, "dashed lines: a region cracks (max d > 0.95)", fontsize=10.5, color=TEXT_2)
for ax, (y, title) in zip(axes, panels):
    ax.set_facecolor(SURFACE)
    ax.plot(F10, y, color=SERIES, linewidth=2, marker="o", markersize=3.5, markeredgewidth=0)
    ax.set_title(title, loc="left", fontsize=11.5, color=TEXT_1, pad=5)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.tick_params(colors=TEXT_2, labelsize=9.5)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_ylim(bottom=0)
    for f, name in events:
        ax.axvline(f, color=AXIS, linewidth=1.0, linestyle=(0, (4, 3)), zorder=1)
for f, name in events:
    near_end = abs(f) > 0.8 * abs(F10).max()          # label left of the line near the end of the load
    axes[0].text(f, 0.97, f"{name} cracks  " if near_end else f"  {name} cracks",
                 transform=axes[0].get_xaxis_transform(), color=TEXT_2, fontsize=9.5, va="top",
                 ha="right" if near_end else "left")
axes[-1].set_xlabel("applied shear F₁₀", color=TEXT_2, fontsize=10.5)
axes[0].invert_xaxis()                       # load grows to the right
fig.savefig(out_prefix + "_pcg.png", dpi=100, facecolor=SURFACE)

# ---- summary table --------------------------------------------------------------
print(f"### Solver effort — {n}², {len(F10)} increments, total wall time {elapsed[-1]:.0f} s\n")
print("| | total | per increment (median / max) |")
print("|---|---|---|")
print(f"| Mechanics PCG iterations (Hessian products) | {hessp.sum():,} | {np.median(hessp):.0f} / {hessp.max():,} |")
print(f"| Newton (trust-region) steps | {outer.sum():,} | {np.median(outer):.0f} / {outer.max()} |")
print(f"| Staggered iterations | {stagger.sum():,} | {np.median(stagger):.0f} / {stagger.max()} |")
print(f"| Damage PCG iterations | {cg_d.sum():,} | {np.median(cg_d):.0f} / {cg_d.max():,} |")
print(f"| PCG iterations per Newton step | {hessp.sum() / outer.sum():.0f} (mean) | {np.median(per_newton):.0f} / {per_newton.max():.0f} |")
print(f"| Wall time per increment [s] | {elapsed[-1]:.0f} | {np.median(wall):.2f} / {wall.max():.1f} |")
print()
i_max = int(np.argmax(hessp))
print(f"Most expensive increment: F10 = {F10[i_max]:.3f}: {hessp[i_max]:,} PCG iterations, "
      f"{stagger[i_max]} staggered iterations, {outer[i_max]} Newton steps, {wall[i_max]:.1f} s")
for f, name in sorted(events, reverse=True):
    k = int(np.argmin(np.abs(F10 - f)))
    window = slice(max(k - 2, 0), k + 3)
    print(f"- {name} cracks at F10 = {f:.3f}: around it {hessp[window].sum():,} PCG iterations in "
          f"{window.stop - window.start} increments ({100 * hessp[window].sum() / hessp.sum():.0f}% of the total)")
print(f"\nfigure: {out_prefix}_pcg.png")
