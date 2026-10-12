"""Video of the third-medium contact run: deformed microstructure + response analysis.

usage: python make_contact_video.py RUN.nc OUT_PREFIX
writes OUT_PREFIX.mp4 (one frame per increment) and OUT_PREFIX_final.png
"""
import sys

import imageio_ffmpeg
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import netCDF4
import numpy as np
from matplotlib import animation
from matplotlib.colors import LinearSegmentedColormap, ListedColormap, Normalize

plt.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()

# ---- colors (reference palette, light surface) ------------------------------
SURFACE = "#fcfcfb"
TEXT_1 = "#0b0b0b"
TEXT_2 = "#52514e"
GRID = "#e6e5e1"
AXIS = "#a3a29c"
FUTURE = "#cfcec8"      # not-yet-reached part of the load path
SERIES = "#2a78d6"      # categorical slot 1
MATRIX = "#8f8e89"      # solid phase: neutral, so blue only means "third medium"
BLUE_RAMP = ["#0d366b", "#184f95", "#256abf", "#3987e5", "#6da7ec", "#9ec5f4", "#cde2fb"]  # 700 -> 100
DETF_CMAP = LinearSegmentedColormap.from_list("detF", BLUE_RAMP)  # dark = compressed, light = undeformed
GAP_CLOSED = 0.05       # min det F below this: the third medium is compressed to 5% -> contact

run_path, out_prefix = sys.argv[1], sys.argv[2]
d = netCDF4.Dataset(run_path)

u = np.asarray(d["u_total"][:])[:, :, 0]          # [frame, 2, nx, ny], grid points
phase = np.asarray(d["phase_field"][0, 0])          # 1 = matrix, 0 = third medium
detF = np.asarray(d["detF"][:])[:, 0]               # [frame, nx, ny]
gamma = -np.asarray(d["F10"][:])                    # applied shear
tau = -np.asarray(d["P10"][:])                      # macroscopic shear stress
min_J = np.asarray(d["min_det_F"][:])
nb_outer = np.asarray(d["nb_outer"][:]).astype(int)
nb_hessp = np.asarray(d["nb_hessp"][:]).astype(int)
nb_frames = len(gamma)
nx, ny = phase.shape
E_matrix = float(d.E_matrix)

stiffness = np.gradient(tau, gamma)                 # tangent  d tau / d gamma
i_closed = int(np.argmax(min_J < GAP_CLOSED))       # first increment with a closed gap
gamma_closed = gamma[i_closed]

# reference grid (pixel positions) and a fixed view over all frames
X0, Y0 = np.meshgrid(np.arange(nx, dtype=float), np.arange(ny, dtype=float), indexing="ij")
Xall = X0[None] + u[:, 0]
Yall = Y0[None] + u[:, 1]
pad = 3.0
xlim = (Xall.min() - pad, Xall.max() + pad)
ylim = (Yall.min() - pad, Yall.max() + pad)

void = phase == 0
matrix = ~void

# ---- figure layout ------------------------------------------------------------
fig = plt.figure(figsize=(16, 9), dpi=100, facecolor=SURFACE)
gs = fig.add_gridspec(3, 2, width_ratios=[1.25, 1.0], left=0.04, right=0.97, top=0.86, bottom=0.08,
                      wspace=0.16, hspace=0.42)
ax_map = fig.add_subplot(gs[:, 0])
ax_tau = fig.add_subplot(gs[0, 1])
ax_k = fig.add_subplot(gs[1, 1], sharex=ax_tau)
ax_j = fig.add_subplot(gs[2, 1], sharex=ax_tau)

fig.text(0.04, 0.955, "Third-medium contact — neo-Hookean, finite strain, 128 × 128 pixels",
         fontsize=17, fontweight="bold", color=TEXT_1)
header = fig.text(0.04, 0.912, "", fontsize=12.5, color=TEXT_2)

# deformed microstructure: matrix grey, third medium coloured by det F
ax_map.set_facecolor(SURFACE)
ax_map.set_aspect("equal")
ax_map.set_xlim(*xlim)
ax_map.set_ylim(*ylim)
ax_map.axis("off")
norm = Normalize(vmin=0.0, vmax=1.0)


def draw_microstructure(k):
    """Deformed configuration of increment k: third medium by det F, matrix grey."""
    for mesh in list(ax_map.collections):
        mesh.remove()
    ax_map.pcolormesh(Xall[k], Yall[k], np.ma.masked_where(matrix, detF[k]),
                      shading="nearest", cmap=DETF_CMAP, norm=norm, rasterized=True)
    ax_map.pcolormesh(Xall[k], Yall[k], np.ma.masked_where(void, np.ones_like(detF[k])),
                      shading="nearest", cmap=ListedColormap([MATRIX]), rasterized=True)


scalar_map = matplotlib.cm.ScalarMappable(norm=norm, cmap=DETF_CMAP)
cbar = fig.colorbar(scalar_map, ax=ax_map, orientation="horizontal", fraction=0.035, pad=0.02, aspect=40)
cbar.set_label("det F in the third medium   (1 = undeformed,  → 0 = gap closed)", color=TEXT_2, fontsize=11)
cbar.outline.set_visible(False)
cbar.ax.tick_params(colors=TEXT_2, labelsize=10, length=0)
ax_map.text(0.0, 1.0, "grey = solid matrix", transform=ax_map.transAxes, color=TEXT_2, fontsize=11,
            va="bottom")


def style(ax, title, ylabel):
    ax.set_facecolor(SURFACE)
    ax.set_title(title, loc="left", fontsize=12.5, color=TEXT_1, pad=6)
    ax.set_ylabel(ylabel, color=TEXT_2, fontsize=11)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.tick_params(colors=TEXT_2, labelsize=10)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    # gap-closure reference line (recessive)
    ax.axvline(gamma_closed, color=AXIS, linewidth=1.0, linestyle=(0, (4, 3)), zorder=1)


style(ax_tau, "Macroscopic shear stress", "−⟨P_yx⟩  [E = %g]" % E_matrix)
style(ax_k, "Tangent stiffness  d⟨P_yx⟩ / dγ   (stiffening = contact)", "stiffness")
style(ax_j, "Smallest det F in the cell", "min det F  (log)")
ax_j.set_yscale("log")
ax_j.set_xlabel("applied shear  γ = −H₁₀", color=TEXT_2, fontsize=11)
plt.setp(ax_tau.get_xticklabels(), visible=False)
plt.setp(ax_k.get_xticklabels(), visible=False)
ax_tau.text(gamma_closed, 0.93, f"  gap closed (min det F < {GAP_CLOSED})", transform=ax_tau.get_xaxis_transform(),
            color=TEXT_2, fontsize=10, va="top", ha="left")
ax_j.axhline(GAP_CLOSED, color=AXIS, linewidth=1.0, linestyle=(0, (4, 3)), zorder=1)

curves = []
for ax, y in ((ax_tau, tau), (ax_k, stiffness), (ax_j, min_J)):
    ax.plot(gamma, y, color=FUTURE, linewidth=2, solid_capstyle="round", zorder=2)  # full path, faint
    done, = ax.plot(gamma[:1], y[:1], color=SERIES, linewidth=2, solid_capstyle="round", zorder=3)
    dot, = ax.plot(gamma[:1], y[:1], "o", color=SERIES, markersize=9, markeredgecolor=SURFACE,
                   markeredgewidth=2, zorder=4)
    value = ax.text(0, 0, "", color=TEXT_1, fontsize=10.5, ha="right", va="bottom", zorder=5)
    curves.append((y, done, dot, value))
ax_tau.set_xlim(0, gamma[-1] * 1.04)


def draw(k):
    draw_microstructure(k)
    state = "gap closed — solids in contact" if min_J[k] < GAP_CLOSED else "gap open"
    header.set_text(f"increment {k + 1:3d} / {nb_frames}    γ = {gamma[k]:.3f}    {state}    "
                    f"solver: {nb_outer[k]} Newton step{'s' if nb_outer[k] != 1 else ''}, "
                    f"{nb_hessp[k]} Hessian products")
    for y, done, dot, value in curves:
        done.set_data(gamma[:k + 1], y[:k + 1])
        dot.set_data([gamma[k]], [y[k]])
        value.set_position((gamma[k], y[k]))
        value.set_text(f"{y[k]:.3g}    ")
    return []




# order of frames: every increment, then hold the final state for 2 s
fps = 12
frames = list(range(nb_frames)) + [nb_frames - 1] * (2 * fps)
writer = animation.FFMpegWriter(fps=fps, bitrate=6000, codec="libx264",
                                extra_args=["-pix_fmt", "yuv420p"])
with writer.saving(fig, out_prefix + ".mp4", dpi=100):
    for k in frames:
        draw(k)
        writer.grab_frame(facecolor=SURFACE)
draw(nb_frames - 1)
fig.savefig(out_prefix + "_final.png", dpi=100, facecolor=SURFACE)
for k in (0, i_closed, nb_frames // 2):  # a few stills to check the layout
    draw(k)
    fig.savefig(f"{out_prefix}_frame{k + 1:03d}.png", dpi=100, facecolor=SURFACE)
print(f"gap closed at increment {i_closed + 1} (gamma = {gamma_closed:.3f}); "
      f"stiffness {stiffness[0]:.2f} -> {stiffness[-1]:.2f}")
