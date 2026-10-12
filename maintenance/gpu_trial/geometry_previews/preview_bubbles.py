import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from muFFTTO import geometry
fig, ax = plt.subplots(1, 2, figsize=(9, 4.6), facecolor="#fcfcfb")
for a, n in zip(ax, (32, 64)):
    x = (np.arange(n) + 0.5) / n
    c = np.stack(np.meshgrid(x, x, indexing="ij"))
    ph = geometry.get("bubble_array", c)
    a.imshow(ph.T, origin="lower", cmap=matplotlib.colors.ListedColormap(["#cde2fb", "#8f8e89"]), extent=(0, 1, 0, 1))
    a.set_title(f"bubble_array ({n}²), solid fraction {ph.mean():.2f}", fontsize=11, color="#0b0b0b"); a.set_xticks([]); a.set_yticks([])
fig.text(0.5, 0.02, "grey = solid, light blue = third medium (bubbles)", ha="center", color="#52514e")
fig.savefig("preview_bubbles.png", dpi=90, facecolor="#fcfcfb")
