"""Convergence histories and final designs of topology optimizations run with different optimizers.

Reads the <output_name>_history.nc files written by example_2D_elasticity_TO.py (frame variables objective,
volume_fraction, nb_evaluations; field phase_field_iterate).

usage: python plot_optimizer_comparison.py OUT.png "TITLE" label=RUN_history.nc [label=RUN_history.nc ...]
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
    runs.append({"label": label,
                 "objective": np.asarray(d["objective"][:]),
                 "volume": np.asarray(d["volume_fraction"][:]),
                 "evaluations": np.asarray(d["nb_evaluations"][:]),
                 "design": np.asarray(d["phase_field_iterate"][-1, 0])})

nb_runs = len(runs)
fig = plt.figure(figsize=(3.2 * max(nb_runs, 3), 9.5), facecolor=SURFACE)
gs = fig.add_gridspec(3, nb_runs, height_ratios=[1.0, 0.7, 1.15], left=0.08, right=0.97, top=0.89, bottom=0.04,
                      hspace=0.42, wspace=0.12)
fig.text(0.08, 0.955, title, fontsize=15, fontweight="bold", color=TEXT_1)
fig.text(0.08, 0.925, "x axis: objective evaluations (each = FE solves of all load cases + adjoints)",
         fontsize=10.5, color=TEXT_2)


def style(ax, panel_title, ylabel):
    ax.set_facecolor(SURFACE)
    ax.set_title(panel_title, loc="left", fontsize=12, color=TEXT_1, pad=6)
    ax.set_ylabel(ylabel, color=TEXT_2, fontsize=10.5)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.tick_params(colors=TEXT_2, labelsize=10)
    ax.grid(axis="y", which="major", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


ax_f = fig.add_subplot(gs[0, :])
ax_v = fig.add_subplot(gs[1, :], sharex=ax_f)
style(ax_f, "Objective", "f (log)")
style(ax_v, "Volume fraction", "mean phase")
ax_f.set_yscale("log")
ax_v.set_xlabel("objective evaluations", color=TEXT_2, fontsize=10.5)
for run, color in zip(runs, SLOTS):
    for ax, y in ((ax_f, run["objective"]), (ax_v, run["volume"])):
        ax.plot(run["evaluations"], y, color=color, linewidth=2, label=run["label"])
    ax_f.plot(run["evaluations"][-1], run["objective"][-1], "o", color=color, markersize=8,
              markeredgecolor=SURFACE, markeredgewidth=2)
    ax_f.annotate(f"{run['objective'][-1]:.3f}", (run["evaluations"][-1], run["objective"][-1]),
                  textcoords="offset points", xytext=(6, 4), color=TEXT_1, fontsize=10)
ax_f.legend(frameon=False, fontsize=10, loc="upper right")

# final designs on their exact pixels
for k, (run, color) in enumerate(zip(runs, SLOTS)):
    ax = fig.add_subplot(gs[2, k])
    nx, ny = run["design"].shape
    ax.pcolormesh(np.arange(nx + 1), np.arange(ny + 1), run["design"].T, shading="flat", cmap="Greys",
                  vmin=0, vmax=1, rasterized=True)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for side in ax.spines.values():
        side.set_color(color)
        side.set_linewidth(2)
    ax.set_title(f"{run['label']}\nf = {run['objective'][-1]:.3f}, vol = {run['volume'][-1]:.3f}, "
                 f"{int(run['evaluations'][-1])} evals", fontsize=10, color=TEXT_1, pad=5)
fig.savefig(out, dpi=200, facecolor=SURFACE)
for run in runs:
    print(f"{run['label']:28s} iterations {len(run['objective']):4d}  evaluations {int(run['evaluations'][-1]):4d}  "
          f"f {run['objective'][-1]:.4f}  volume {run['volume'][-1]:.4f}")
