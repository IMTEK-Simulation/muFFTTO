"""Mechanics CG iterations (Hessian products) per load step for several contact runs.

usage: python plot_cg_per_step.py OUT.png "TITLE" label=RUN.nc [label=RUN.nc ...]

example:
    python plot_cg_per_step.py cg.png "Green vs Green-Jacobi, bubbles 32²" \
        "regular, Green-Jacobi=exp_data/.../contact_fracture.nc" "adapted, Green-Jacobi=exp_data/.../contact_fracture.nc"
"""
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import netCDF4
import numpy as np

# reference palette, light surface; categorical slots in fixed order
SURFACE, TEXT_1, TEXT_2, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1", "#a3a29c"
SLOTS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]

out, title = sys.argv[1], sys.argv[2]
runs = []
for arg in sys.argv[3:]:
    label, path = arg.split("=", 1)
    d = netCDF4.Dataset(path)
    H = np.asarray(d.H_macro).reshape(2, 2)
    F11 = 1.0 + np.asarray(d["lam"][:]) * H[1, 1]
    hessp = np.asarray(d["nb_hessp"][:]).astype(float)
    outer = np.maximum(np.asarray(d["nb_outer"][:]).astype(float), 1.0)
    runs.append((label, F11, hessp, hessp / outer))

fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True, facecolor=SURFACE)
fig.subplots_adjust(left=0.09, right=0.97, top=0.88, bottom=0.08, hspace=0.25)
fig.text(0.09, 0.955, f"Mechanics CG iterations per load step — {title}",
         fontsize=15, fontweight="bold", color=TEXT_1)
fig.text(0.09, 0.925, "CG iterations = Hessian products of the inner (Steihaug) CG; log scale",
         fontsize=10.5, color=TEXT_2)
titles = ["CG iterations per load step", "CG iterations per Newton step"]
for ax, panel_title, k in zip(axes, titles, (2, 3)):
    ax.set_facecolor(SURFACE)
    for (label, F11, *series), color in zip(runs, SLOTS):
        y = series[k - 2]
        ax.plot(F11, y, color=color, linewidth=2, marker="o", markersize=3, markeredgewidth=0, label=label)
    ax.set_yscale("log")
    ax.set_title(panel_title, loc="left", fontsize=12, color=TEXT_1, pad=6)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.tick_params(colors=TEXT_2, labelsize=10)
    ax.grid(axis="y", which="major", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
axes[0].legend(frameon=False, fontsize=10, loc="center", bbox_to_anchor=(0.4, 0.3))
axes[-1].set_xlabel("F₁₁ (compression →)", color=TEXT_2, fontsize=10.5)
axes[-1].invert_xaxis()
fig.savefig(out, dpi=150, facecolor=SURFACE)
for label, F11, hessp, per_newton in runs:
    print(f"{label:24s} {len(F11):4d} steps  total {int(hessp.sum()):8,d}  median/step {np.median(hessp):6.0f}  "
          f"max/step {int(hessp.max()):6,d}  median per Newton step {np.median(per_newton):5.0f}")
