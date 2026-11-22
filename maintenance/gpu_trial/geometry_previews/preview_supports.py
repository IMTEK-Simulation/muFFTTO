import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from muFFTTO import geometry
cases = [("contact_test_geometry_2", {}, "original contact geometry"),
         ("frame_contact_supports", {"gap": 0.18}, "frame + supports, gap 0.18"),
         ("frame_contact_supports", {"gap": 0.24}, "frame + supports, gap 0.24")]
fig, ax = plt.subplots(1, 3, figsize=(13, 4.6), facecolor="#fcfcfb")
n = 64
x = (np.arange(n) + 0.5) / n
c = np.stack(np.meshgrid(x, x, indexing="ij"))
for a, (name, kw, title) in zip(ax, cases):
    ph = geometry.get(name, c, **kw)
    a.imshow(ph.T, origin="lower", cmap=matplotlib.colors.ListedColormap(["#cde2fb", "#8f8e89"]), extent=(0, 1, 0, 1))
    a.set_title(f"{title}  (64²)", fontsize=11, color="#0b0b0b"); a.set_xticks([]); a.set_yticks([])
    print(title, "solid fraction", round(ph.mean(), 3))
fig.text(0.5, 0.02, "grey = solid, light blue = third medium", ha="center", color="#52514e")
fig.savefig("preview_supports.png", dpi=90, facecolor="#fcfcfb")
