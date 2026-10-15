"""Green vs Green-Jacobi for the finite-strain bubble array: grid size and void stiffness.

Reads exp_data/example_2D_homogenization_finite_strain_bubbles_zero_stiffness/*/info_log_final.npz, written by
example_2D_homogenization_finite_strain_bubbles_zero_stiffness.py. Panels:
    1. total CG iterations vs grid size, for void stiffness 0 and for a soft void (--ratio),
    2. total CG iterations vs void stiffness / solid stiffness at one grid size (--n); 0 drawn at the left end,
    3. CG iterations of every Newton step at that grid size, void 0 and the soft void.

usage: python plot_bubbles_preconditioner_study.py OUT.png [--n 64] [--ratio 1e-5]
"""
import argparse
import glob
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# reference palette, light surface; colour follows the preconditioner, line style the void stiffness
SURFACE, TEXT_1, TEXT_2, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1", "#a3a29c"
COLORS = {"Green": "#2a78d6", "Green_Jacobi": "#eb6834"}
LABELS = {"Green": "Green", "Green_Jacobi": "Green-Jacobi"}

parser = argparse.ArgumentParser()
parser.add_argument('out')
parser.add_argument('--n', type=int, default=64, help='grid size of panels 2 and 3')
parser.add_argument('--ratio', type=float, default=1e-5, help='soft-void stiffness ratio of panels 1 and 3')
args = parser.parse_args()

folder = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'exp_data',
                      'example_2D_homogenization_finite_strain_bubbles_zero_stiffness')
runs = []
for path in glob.glob(os.path.join(folder, '*', 'info_log_final.npz')):
    info = np.load(path, allow_pickle=True)
    runs.append({"n": int(info['nb_of_pixels'][0]), "p": str(info['preconditioner_type']),
                 "ratio": float(info['void_ratio']) if 'void_ratio' in info else 0.0,
                 "cg": int(info['sum_CG_its']), "per_newton": np.asarray(info['cg_per_newton_flat']),
                 "time": float(info['elapsed_time']), "P": np.asarray(info['mean_P'])})


def select(**conditions):
    return sorted((r for r in runs if all(np.isclose(r[k], v) if isinstance(v, float) else r[k] == v
                                          for k, v in conditions.items())), key=lambda r: (r["n"], r["ratio"]))


def style(ax, title, xlabel, ylabel):
    ax.set_facecolor(SURFACE)
    ax.set_title(title, loc="left", fontsize=11.5, color=TEXT_1, pad=6)
    ax.set_xlabel(xlabel, color=TEXT_2, fontsize=10)
    ax.set_ylabel(ylabel, color=TEXT_2, fontsize=10)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.tick_params(colors=TEXT_2, labelsize=9.5)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


fig, axes = plt.subplots(1, 3, figsize=(16, 5.6), facecolor=SURFACE)
fig.subplots_adjust(left=0.055, right=0.985, top=0.80, bottom=0.12, wspace=0.27)
fig.text(0.055, 0.945, "Green vs Green-Jacobi — finite-strain homogenization, bubble array (porosity 0.5)",
         fontsize=15, fontweight="bold", color=TEXT_1)
fig.text(0.055, 0.905, "neo-Hookean solid, bubbles with stiffness 0 or ratio x solid; Newton-CG, full steps, "
                       "10 increments; CG stops at energy-norm error 1e-6 (estimate, delay 10)",
         fontsize=10.5, color=TEXT_2)
dashes = {0.0: "-", args.ratio: (0, (4, 2))}
void_label = {0.0: "void 0", args.ratio: f"void {args.ratio:g}"}

# 1. grid size
ax = axes[0]
style(ax, "Total CG iterations vs grid size", "pixels per direction", "CG iterations (10 increments)")
for ratio in (0.0, args.ratio):
    for p in ("Green", "Green_Jacobi"):
        sel = select(p=p, ratio=ratio)
        if sel:
            ax.plot([r["n"] for r in sel], [r["cg"] for r in sel], color=COLORS[p], linestyle=dashes[ratio],
                    linewidth=2, marker="o", markersize=6, label=f"{LABELS[p]}, {void_label[ratio]}")
ax.set_xscale("log", base=2)
sizes = sorted({r["n"] for r in runs})
ax.set_xticks(sizes, [str(n) for n in sizes])
ax.set_ylim(bottom=0)
ax.legend(frameon=False, fontsize=9.5, loc="upper left")

# 2. void stiffness at one grid size
ax = axes[1]
style(ax, f"Total CG iterations vs void stiffness, {args.n}²", "void stiffness / solid stiffness", "")
positive = sorted({r["ratio"] for r in runs if r["n"] == args.n and r["ratio"] > 0})
zero_position = min(positive) / 10 if positive else 1e-7          # where "0" is drawn
for p in ("Green", "Green_Jacobi"):
    sel = select(p=p, n=args.n)
    soft = [r for r in sel if r["ratio"] > 0]
    ax.plot([r["ratio"] for r in soft], [r["cg"] for r in soft], color=COLORS[p], linewidth=2, marker="o",
            markersize=6, label=LABELS[p])
    zero = [r for r in sel if r["ratio"] == 0]
    if zero:
        ax.plot([zero_position], [zero[0]["cg"]], color=COLORS[p], marker="D", markersize=8,
                markeredgecolor=SURFACE, markeredgewidth=1.5, linestyle="none")
ax.set_xscale("log")
ticks = [zero_position] + positive
ax.set_xticks(ticks, ["0"] + [f"$10^{{{int(round(np.log10(t)))}}}$" for t in positive])
ax.axvline(np.sqrt(zero_position * min(positive)), color=AXIS, linewidth=1, linestyle=(0, (2, 2)))
ax.invert_xaxis()                                                  # contrast grows to the right
ax.set_ylim(bottom=0)
ax.legend(frameon=False, fontsize=9.5, loc="lower right")

# 3. CG per Newton step
ax = axes[2]
style(ax, f"CG iterations per Newton step, {args.n}²", "Newton step (all increments)", "")
for ratio in (0.0, args.ratio):
    for p in ("Green", "Green_Jacobi"):
        sel = select(p=p, n=args.n, ratio=ratio)
        if sel:
            y = sel[0]["per_newton"]
            ax.plot(np.arange(1, len(y) + 1), y, color=COLORS[p], linestyle=dashes[ratio], linewidth=1.6,
                    marker="o", markersize=3, label=f"{LABELS[p]}, {void_label[ratio]}")
ax.set_ylim(bottom=0)
ax.legend(frameon=False, fontsize=9.5, loc="upper left", ncol=2)
fig.savefig(args.out, dpi=200, facecolor=SURFACE)

print("| grid | void / solid | Green CG | Green-Jacobi CG | Green time | Green-Jacobi time |")
print("|---|---|---|---|---|---|")
for n, ratio in sorted({(r["n"], r["ratio"]) for r in runs}):
    g, gj = select(p="Green", n=n, ratio=ratio), select(p="Green_Jacobi", n=n, ratio=ratio)
    if g and gj:
        print(f"| {n}² | {ratio:g} | {g[0]['cg']:,} | {gj[0]['cg']:,} | {g[0]['time']:.1f} s | {gj[0]['time']:.1f} s |")
