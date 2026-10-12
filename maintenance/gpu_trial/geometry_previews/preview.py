import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from muFFTTO import geometry
fig, ax = plt.subplots(1, 3, figsize=(13, 4.6), facecolor="#fcfcfb")
for a, (name, n) in zip(ax, [("contact_test_geometry_2", 128), ("contact_beam_root_notch", 128), ("contact_beam_root_notch", 32)]):
    x = (np.arange(n) + 0.5) / n
    c = np.stack(np.meshgrid(x, x, indexing="ij"))
    ph = geometry.get(name, c)
    a.imshow(ph.T, origin="lower", cmap=matplotlib.colors.ListedColormap(["#cde2fb", "#8f8e89"]), extent=(0, 1, 0, 1))
    a.set_title(f"{name}  ({n}²)", fontsize=11, color="#0b0b0b"); a.set_xticks([]); a.set_yticks([])
    print(name, n, "solid fraction", ph.mean())
fig.text(0.5, 0.02, "grey = solid, light blue = third medium", ha="center", color="#52514e")
fig.savefig("preview.png", dpi=90, facecolor="#fcfcfb")
