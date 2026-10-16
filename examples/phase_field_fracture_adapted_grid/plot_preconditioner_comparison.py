"""Mechanics preconditioners (Green vs Green-Jacobi vs dual) on the same problem at several grid sizes.

Reads contact_fracture.nc files of example_2D_phase_field_fracture_contact_adapted_grid.py; grid size and
preconditioner are taken from the file attributes (nb_grid_pts, mechanics_preconditioner). One column per grid size:
    1. macroscopic stress |P_ij| / mu over the applied F_ij (the loaded component): same physics for both,
    2. mechanics CG iterations (Hessian products) per load step,
    3. cumulative wall time.

usage: python plot_preconditioner_comparison.py OUT.png "TITLE" "SUBTITLE" RUN.nc [RUN.nc ...]
"""
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import netCDF4
import numpy as np

# reference palette, light surface; categorical slots in fixed order, colour follows the preconditioner
SURFACE, TEXT_1, TEXT_2, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1", "#a3a29c"
COLORS = {"Green": "#2a78d6", "Green_Jacobi": "#eb6834", "dual": "#1baf7a"}
LABELS = {"Green": "Green", "Green_Jacobi": "Green-Jacobi", "dual": "dual GK(A⁻¹)G"}

out, title, subtitle = sys.argv[1:4]
runs = {}
for path in sys.argv[4:]:
    d = netCDF4.Dataset(path)
    n = int(np.atleast_1d(d.nb_grid_pts)[0])
    preconditioner = str(d.mechanics_preconditioner)
    H = np.asarray(d.H_macro).reshape(2, 2)
    i, j = np.unravel_index(np.argmax(np.abs(H)), H.shape)      # the loaded component
    mu = float(d.E_solid) / (2 * (1 + float(d.nu_solid)))
    lam = np.asarray(d["lam"][:])
    runs[(n, preconditioner)] = {
        "F": (i == j) + lam * H[i, j],
        "F_end": (i == j) + float(d.lam_end) * H[i, j],
        "P": np.abs(np.asarray(d["mean_P"][:, i, j])) / mu,
        "cg": np.asarray(d["nb_hessp"][:]).astype(float),
        "wall": np.asarray(d["elapsed_time"][:]),
        "component": f"{i}{j}"}  # 0-based, as in H_macro[i, j]
sizes = sorted({n for n, _ in runs})
component = next(iter(runs.values()))["component"]

width = max(4.6 * len(sizes), 9.0)
fig, axes = plt.subplots(3, len(sizes), figsize=(width, 10), sharex=True, squeeze=False, facecolor=SURFACE)
left = 0.9 / width                      # 0.9 inch for the tick labels
fig.subplots_adjust(left=left, right=1 - 0.6 / width, top=0.86, bottom=0.07, hspace=0.32, wspace=0.28)
fig.text(left, 0.955, title, fontsize=15, fontweight="bold", color=TEXT_1)
fig.text(left, 0.925, subtitle, fontsize=10.5, color=TEXT_2)
row_titles = [f"|P{component}| / μ", "CG iterations per load step (log)", "cumulative wall time [s]"]
for column, n in enumerate(sizes):
    for row in range(3):
        ax = axes[row, column]
        ax.set_facecolor(SURFACE)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(AXIS)
        ax.tick_params(colors=TEXT_2, labelsize=9.5)
        ax.grid(axis="y", color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        ax.set_title(f"{n}²   {row_titles[row]}" if row == 0 else row_titles[row], loc="left", fontsize=11,
                     color=TEXT_1, pad=5)
        for preconditioner in ("Green", "Green_Jacobi", "dual"):
            run = runs.get((n, preconditioner))
            if run is None:
                continue
            y = (run["P"], run["cg"], run["wall"])[row]
            ax.plot(run["F"], y, color=COLORS[preconditioner], linewidth=2, label=LABELS[preconditioner],
                    marker="o" if row == 1 else None, markersize=2.5, markeredgewidth=0)
            if row == 2:  # end of the run: total time
                ax.annotate(f"{run['wall'][-1]:.0f} s", (run["F"][-1], run["wall"][-1]), textcoords="offset points",
                            xytext=(4, 2), color=TEXT_1, fontsize=9.5)
        if row == 1:
            ax.set_yscale("log")
    axes[-1, column].set_xlabel(f"F{component} (compression →)", color=TEXT_2, fontsize=10)
axes[0, 0].legend(frameon=False, fontsize=10, loc="upper left")
axes[0, 0].invert_xaxis()
fig.savefig(out, dpi=200, facecolor=SURFACE)

print(f"| grid | preconditioner | reached F{component} | load steps | CG total | CG / step (median) | wall time | "
      f"speed-up vs Green |")
print("|---|---|---|---|---|---|---|---|")
for n in sizes:
    for preconditioner in ("Green", "Green_Jacobi", "dual"):
        run = runs.get((n, preconditioner))
        if run is None:
            continue
        other = runs.get((n, "Green"))
        done = "" if abs(run["F"][-1] - run["F_end"]) < 1e-9 else " (stopped)"
        speedup = (f"{other['wall'][-1] / run['wall'][-1]:.2f}x" if preconditioner != "Green" and other and not done
                   else "")
        print(f"| {n}² | {LABELS[preconditioner]} | {run['F'][-1]:.3f}{done} | {len(run['F'])} | "
              f"{int(run['cg'].sum()):,} | {np.median(run['cg']):.0f} | {run['wall'][-1]:.0f} s | {speedup} |")
