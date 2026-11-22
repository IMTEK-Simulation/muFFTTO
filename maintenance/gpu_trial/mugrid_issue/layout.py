import muGrid, numpy as np, cupy as cp
for dev in ("cpu", "gpu"):
    fft = muGrid.FFTEngine(nb_domain_grid_pts=(8, 6), nb_ghosts_left=[1, 1], nb_ghosts_right=[1, 1], device=dev)
    fc = fft.real_space_collection
    fc.set_nb_sub_pts("nodal_points", 4); fc.set_nb_sub_pts("quad_points", 9)
    x = fc.real_field("x", (3,), "nodal_points"); g = fc.real_field("g", (3, 2), "quad_points")
    for f in (x, g):
        a = f.sg  # with ghosts
        print(dev, f.name, "shape", a.shape, "strides/8", tuple(s // 8 for s in a.strides))
