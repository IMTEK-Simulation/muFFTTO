"""Emulate muGrid's SoA (GPU) sparse-operator kernel in NumPy with the current
and the corrected nodal index formula, and compare with the real GPU result
and a plain NumPy reference.  2D, offset [0, 0], stencil 2x2."""
import numpy as np
import cupy as cp
import muGrid
from muGrid import GenericLinearOperator

nb_grid = (8, 6)


def flat_soa(a_host, shape_comp, nn, nxb, nyb):
    """SoA flat buffer: index = comp_flat*(nn*npix) + sub*npix + x + y*nxb"""
    npix = nxb * nyb
    ncomp = int(np.prod(shape_comp))
    a = a_host.reshape(ncomp, nn, nxb, nyb)
    flat = np.zeros(ncomp * nn * npix)
    for c in range(ncomp):
        for n in range(nn):
            for y in range(nyb):
                flat[c * nn * npix + n * npix + y * nxb: c * nn * npix + n * npix + (y + 1) * nxb] = a[c, n, :, y]
    return flat


def unflat_soa(flat, ncomp, nn, nxb, nyb):
    npix = nxb * nyb
    a = np.zeros((ncomp, nn, nxb, nyb))
    for c in range(ncomp):
        for n in range(nn):
            for y in range(nyb):
                a[c, n, :, y] = flat[c * nn * npix + n * npix + y * nxb: c * nn * npix + n * npix + (y + 1) * nxb]
    return a


for nb_nodes, nb_quad, nb_comp in [(1, 4, 1), (2, 3, 1), (4, 9, 1), (4, 9, 3)]:
    rng = np.random.default_rng(3)
    S = rng.standard_normal((2, nb_quad, nb_nodes, 2, 2))   # [d, q, n, i, j]
    fft = muGrid.FFTEngine(nb_domain_grid_pts=nb_grid, nb_ghosts_left=[1, 1],
                           nb_ghosts_right=[1, 1], device="gpu")
    fc = fft.real_space_collection
    fc.set_nb_sub_pts("quad_points", nb_quad)
    fc.set_nb_sub_pts("nodal_points", nb_nodes)
    x = fc.real_field("x", (nb_comp,), "nodal_points")
    g = fc.real_field("g", (nb_comp, 2), "quad_points")
    x.sg[...] = cp.asarray(rng.standard_normal(x.sg.shape))   # random incl. ghosts
    op = GenericLinearOperator([0, 0], S)
    op.apply(nodal_field=x, quadrature_point_field=g)
    gpu = cp.asnumpy(g.sg)
    xh = cp.asnumpy(x.sg)
    nxb, nyb = xh.shape[-2:]
    npix = nxb * nyb
    xflat = flat_soa(xh, (nb_comp,), nb_nodes, nxb, nyb)
    total_nodal = npix * nb_nodes
    total_quad = npix * nb_quad * 2

    results = {}
    for variant in ("current", "fixed"):
        qflat = np.zeros(nb_comp * total_quad)
        # computable region for offset [0,0], stencil 2x2: x in [0, nxb-1), y in [0, nyb-1)
        for (d, q, n, i, j), v in np.ndenumerate(S):
            pixel_count = i + j * nxb
            for c in range(nb_comp):
                if variant == "current":   # generic.cc:479-480
                    nidx = c * total_nodal + pixel_count * nb_nodes + n
                else:                      # proposed
                    nidx = c * total_nodal + n * npix + pixel_count
                qidx = c * total_quad + d * (npix * nb_quad) + q * npix
                for yy in range(nyb - 1):
                    p = np.arange(nxb - 1) + yy * nxb
                    qflat[qidx + p] += v * xflat[nidx + p]
        results[variant] = unflat_soa(qflat, nb_comp * 2, nb_quad, nxb, nyb).reshape(gpu.shape)

    inner = (..., slice(0, nxb - 1), slice(0, nyb - 1))
    # direct reference on the computable region: g[c,d,q,x,y] = sum S[d,q,n,i,j] x[c,n,x+i,y+j]
    ref = np.zeros_like(gpu)
    for (d, q, n, i, j), v in np.ndenumerate(S):
        ref[:, d, q, :nxb - 1, :nyb - 1] += v * xh[:, n, i:nxb - 1 + i, j:nyb - 1 + j]
    print(f"   |emulated(fixed) - direct reference| = {np.abs(results['fixed'][inner] - ref[inner]).max():.1e}"
          f"   |GPU - direct reference| = {np.abs(gpu[inner] - ref[inner]).max():.1e}")
    print(f"nb_nodes={nb_nodes} nb_quad={nb_quad} nb_comp={nb_comp}: "
          f"|GPU - emulated(current)| = {np.abs(gpu[inner] - results['current'][inner]).max():.1e}   "
          f"|GPU - emulated(fixed)| = {np.abs(gpu[inner] - results['fixed'][inner]).max():.1e}")
