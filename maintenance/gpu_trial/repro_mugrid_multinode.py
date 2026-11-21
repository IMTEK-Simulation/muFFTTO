"""Pure-muGrid check: GenericLinearOperator.apply on CPU vs GPU for several
nodal points per pixel (nb_nodes = 1 vs 4).  No muFFTTO involved."""
import numpy as np
import cupy as cp
import muGrid
from muGrid import GenericLinearOperator

nb_grid = (8, 8)
rng = np.random.default_rng(1)

for nb_nodes, nb_quad in [(1, 4), (4, 9), (2, 3)]:
    # stencil [out comps d, quad q, in nodes n, i, j]  (same layout as muFFTTO's B_grad_at_pixel_dqnijk)
    stencil = rng.standard_normal((2, nb_quad, nb_nodes, 2, 2))
    x_host = rng.standard_normal((1, nb_nodes, *nb_grid))
    out = {}
    for dev in ("cpu", "gpu"):
        fft = muGrid.FFTEngine(nb_domain_grid_pts=nb_grid, nb_ghosts_left=[1, 1],
                               nb_ghosts_right=[1, 1], device=dev)
        fc = fft.real_space_collection
        fc.set_nb_sub_pts("quad_points", nb_quad)
        fc.set_nb_sub_pts("nodal_points", nb_nodes)
        op = GenericLinearOperator([0, 0], stencil)
        x = fc.real_field("x", (1,), "nodal_points")
        g = fc.real_field("g", (1, 2), "quad_points")
        x.s[...] = cp.asarray(x_host) if dev == "gpu" else x_host
        fft.communicate_ghosts(x)
        op.apply(nodal_field=x, quadrature_point_field=g)
        out[dev] = cp.asnumpy(g.s) if dev == "gpu" else np.array(g.s)
    d = np.abs(out["cpu"] - out["gpu"]).max()
    print(f"nb_nodes={nb_nodes} nb_quad={nb_quad}: max|cpu-gpu| = {d:.3e}  (scale {np.abs(out['cpu']).max():.2f})"
          f"  nan_gpu={np.isnan(out['gpu']).sum()}")


# reference: g[0,d,q,x,y] = sum_{n,i,j} S[d,q,n,i,j] * x[0,n,x+i,y+j]  (periodic)
print("\nagainst a plain-numpy reference:")
for nb_nodes, nb_quad in [(1, 4), (4, 9), (2, 3)]:
    rng = np.random.default_rng(7)
    S = rng.standard_normal((2, nb_quad, nb_nodes, 2, 2))
    xh = rng.standard_normal((1, nb_nodes, *nb_grid))
    ref = np.zeros((1, 2, nb_quad, *nb_grid))
    for i in range(2):
        for j in range(2):
            ref[0] += np.einsum("dqn,nxy->dqxy", S[..., i, j], np.roll(xh[0], (-i, -j), axis=(1, 2)))
    for dev in ("cpu", "gpu"):
        fft = muGrid.FFTEngine(nb_domain_grid_pts=nb_grid, nb_ghosts_left=[1, 1],
                               nb_ghosts_right=[1, 1], device=dev)
        fc = fft.real_space_collection
        fc.set_nb_sub_pts("quad_points", nb_quad)
        fc.set_nb_sub_pts("nodal_points", nb_nodes)
        op = GenericLinearOperator([0, 0], S)
        x = fc.real_field("x", (1,), "nodal_points")
        g = fc.real_field("g", (1, 2), "quad_points")
        x.s[...] = cp.asarray(xh) if dev == "gpu" else xh
        fft.communicate_ghosts(x)
        op.apply(nodal_field=x, quadrature_point_field=g)
        r = cp.asnumpy(g.s) if dev == "gpu" else np.array(g.s)
        print(f"  nb_nodes={nb_nodes} {dev}: max|muGrid-ref| = {np.abs(r - ref).max():.3e}")
