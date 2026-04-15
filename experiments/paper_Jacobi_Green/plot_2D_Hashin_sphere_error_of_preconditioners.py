"""
Plotting script for exp_2D_Hashin_sphere_error_of_preconditioners.py results.

Reads *_log.npz  -> convergence curves (norm of residual vs. CG iteration)
Reads *.npy      -> strain fields (xx-component) for each CG tolerance exponent

Usage:
    python plot_2D_Hashin_sphere_error_of_preconditioners.py -n 128
    python plot_2D_Hashin_sphere_error_of_preconditioners.py -n 64 128
"""

import os
import glob
import re
import argparse

import numpy as np
import matplotlib.pyplot as plt

from muFFTTO import domain


def solution_hashin(coordinates, mu1, lam1, mu2, lam2, R1, R2, center):
    """
    Compute the Hashin-Shtrikman strain field for a coated sphere/cylinder (elasticity).

    Parameters
    ----------
    coordinates : ndarray, shape (..., d)
        Spatial coordinates at each grid point.
    Ci : tuple of 3 stiffness objects (C1, C2, C3)
        Inner sphere, shell, outer matrix. Each must support shear(C) and lam(C).
    R : tuple of 2 floats (R1, R2)
        Inner and outer radii.
    center : array-like, length d
        Center of the inclusion.

    Returns
    -------
    gu : ndarray, shape (..., d, d)
        Strain field (symmetric 2nd-order tensor at each point).
    """
    center = np.asarray(center)
    d = len(center)

    kap1 = lam1 + 2 * mu1 / d

    kap2 = lam2 + 2 * mu2 / d

    phi = (R1 / R2) ** d
    alpha = d * (kap2 - kap1) / ((d - 1) * 2 * mu2 + d * kap1)

    a2 = 1.0 / (1.0 + alpha * phi)
    a1 = (1.0 + alpha) * a2
    b2 = alpha * (R1 ** d) * a2

    a_coef = np.array([a1, a2, 1.0])  # per-region scalar a
    b_coef = np.array([0.0, b2, 0.0])  # per-region scalar b

    Id = np.eye(d)  # (d, d)

    # --- geometry ---
    x = coordinates - center  # (..., d)
    r = np.linalg.norm(x, axis=-1)  # (...)

    # Region: 0 = inner sphere, 1 = shell, 2 = outer matrix
    region = np.where(r <= R1, 0, np.where(r <= R2, 1, 2))  # (...)

    # Safe 1/r (avoid division by zero at origin)
    r_safe = np.where(r > 0, r, 1.0)
    r_inv = 1.0 / r_safe  # (...)

    # Radial unit vector and outer product  e_r ⊗ e_r
    e_r = x * r_inv[..., np.newaxis]  # (..., d)
    e_rr = e_r[..., :, np.newaxis] * e_r[..., np.newaxis, :]  # (..., d, d)

    # Per-point coefficients
    a_reg = a_coef[region]  # (...)
    brd = b_coef[region] * r_inv ** d  # (...)   b[i] / r^d

    # gu = (a + brd) * I  -  d * brd * (e_r ⊗ e_r)
    gu = (
            (a_reg + brd)[..., np.newaxis, np.newaxis] * Id
            - (d * brd)[..., np.newaxis, np.newaxis] * e_rr
    )

    # Fix origin: gu = a1 * I
    gu[r == 0.0] = a_coef[0] * Id

    return gu

if __name__ == '__main__':
    # ---------------------------------------------------------------------------
    # CLI
    # ---------------------------------------------------------------------------
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--nb_pixel', nargs='+', default=1024,
                        help='Grid sizes to plot (default: all found in data dir)')
    args = parser.parse_args()
    n = int(args.nb_pixel)
    # ---------------------------------------------------------------------------
    # Paths
    # ---------------------------------------------------------------------------
    script_dir  = os.path.dirname(os.path.realpath(__file__))
    script_name = 'exp_2D_Hashin_sphere_error_of_preconditioners'
    data_dir    = os.path.join(script_dir, 'exp_data',    script_name)
    fig_dir     = os.path.join(script_dir, 'figures',     script_name)
    os.makedirs(fig_dir, exist_ok=True)


    problem_type = 'elasticity'
    discretization_type = 'finite_element'
    element_type = 'linear_triangles'
    formulation = 'small_strain'

    domain_size = [1, 1]
    dim = len(domain_size)
    number_of_pixels = (n,n)

    my_cell = domain.PeriodicUnitCell(domain_size=domain_size,
                                      problem_type=problem_type)

    discretization = domain.Discretization(cell=my_cell,
                                           nb_of_pixels_global=number_of_pixels,
                                           discretization_type=discretization_type,
                                           element_type=element_type)
    qpoints_coords=discretization.get_quad_points_coordinates()


    # ---------------------------------------------------------------------------
    # Colour / style maps per preconditioner
    # ---------------------------------------------------------------------------
    PREC_STYLES = {
        'Green':        dict(color='tab:green',   linestyle='-',  label='Green'),
        'Jacobi':       dict(color='tab:blue',    linestyle='--', label='Jacobi'),
        'Green_Jacobi': dict(color='tab:red',     linestyle=':',  label='Green-Jacobi'),
    }

    # ---------------------------------------------------------------------------
    # Geometry and analytic solution (computed once)
    # ---------------------------------------------------------------------------
    R1, R2 = 0.2, 0.4
    center = np.array([0.5, 0.5])
    dim = 2

    lambda_1, mu_1 = 0.001, 0.005
    lambda_2, mu_2 = 1.0,   0.5

    first_quads = np.moveaxis(qpoints_coords.s[:, 0, ...], 0, -1)
    eps = solution_hashin(first_quads, mu1=mu_1, lam1=lambda_1,
                          mu2=mu_2, lam2=lambda_2, R1=R1, R2=R2, center=center)

    import matplotlib.colors as mcolors
    import matplotlib.animation as animation
    xlin = np.linspace(0, 1, n + 1)
    X, Y = np.meshgrid(xlin, xlin, indexing='ij')
    circ_offset = 0.5 / np.array(number_of_pixels)

    # ---------------------------------------------------------------------------
    # Movie: evolution wrt CG iteration
    # ---------------------------------------------------------------------------
    cmap = 'RdBu_r'#'Reds'
    norm = mcolors.LogNorm(vmin=1e-8, vmax=1)
    idx_pairs = [(0, 0), (0, 1), (1, 0), (1, 1)]
    comp_labels = [r'$\varepsilon_{xx}$', r'$\varepsilon_{xy}$',
                   r'$\varepsilon_{yx}$', r'$\varepsilon_{yy}$']

    for prec in ['Green', 'Green_Jacobi']:
        for tol in [12]:
            fname_log = os.path.join(data_dir, f'{prec}_n_{n}_cgtol_{tol}_log.npz')
            info = np.load(fname_log, allow_pickle=True)
            nb_it_total = len(info.f.norm_rr)

            # collect iterations that have saved .npy files
            valid_iterations = []
            for iteration in np.arange(1, nb_it_total):
                fname_npy = os.path.join(data_dir,
                    f'{prec}_n_{n}_cgtol_{tol}_it_{iteration}.npy')
                if os.path.exists(fname_npy):
                    valid_iterations.append(iteration)
                else:
                    print(f'Missing {os.path.basename(fname_npy)}, skipping')

            if not valid_iterations:
                print(f'No frames for {prec}, tol={tol} — skipping movie')
                continue

            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            dummy = np.ones((n, n))
            meshes = [ax.pcolormesh(X, Y, dummy, cmap=cmap, norm=norm, shading='flat')
                      for ax in axes.flat]
            for m, ax in zip(meshes, axes.flat):
                plt.colorbar(m, ax=ax)
            for ax, label in zip(axes.flat, comp_labels):
                ax.set_xlabel('x')
                ax.set_ylabel('y')
                ax.set_title(label)
            suptitle = fig.suptitle('', fontsize=13)
            plt.tight_layout()

            def update(iteration, prec=prec, tol=tol):
                fname_npy = os.path.join(data_dir,
                    f'{prec}_n_{n}_cgtol_{tol}_it_{iteration}.npy')
                strain_G = np.load(fname_npy, allow_pickle=True)
                for mesh, (i, j) in zip(meshes, idx_pairs):
                    component = eps[..., i, j]
                    fem_sol = strain_G[i, j, 0]
                    err = np.abs(component - fem_sol)
                    err_pos = np.where(err > 0, err, np.min(err[err > 0]))
                    mesh.set_array(err_pos.ravel())
                suptitle.set_text(
                    f'Preconditioner: {prec},  $n={n}$,  '
                    f'tol=$10^{{-{tol}}}$,  it={iteration}')
                return meshes + [suptitle]

            ani = animation.FuncAnimation(
                fig, update, frames=valid_iterations, blit=False)

            fname_movie = os.path.join(
                fig_dir, f'error_{prec}_n{n}_cgtol{tol}_cmap{cmap}.mp4')
            ani.save(fname_movie, writer=animation.FFMpegWriter(fps=5), dpi=150)
            plt.close(fig)
            print(f'Saved {fname_movie}')
    quit()
    # ---------------------------------------------------------------------------
    # Plot eolution wrt precitioner
    # ---------------------------------------------------------------------------
    for prec in ['Green',   'Green_Jacobi']:
        for tol in [0, 1, 2, 3, 4, 5, 6,7,8 ]:
            fname_npy = os.path.join(data_dir, f'{prec}_n_{n}_cgtol_{tol}.npy')
            if not os.path.exists(fname_npy):
                print(f'Missing {os.path.basename(fname_npy)}, skipping')
                continue

            strain_G = np.load(fname_npy, allow_pickle=True)

            components = [
                (eps[..., 0, 0], r'$\varepsilon_{xx}$', strain_G[0, 0, 0]),
                (eps[..., 0, 1], r'$\varepsilon_{xy}$', strain_G[0, 1, 0]),
                (eps[..., 1, 0], r'$\varepsilon_{yx}$', strain_G[1, 0, 0]),
                (eps[..., 1, 1], r'$\varepsilon_{yy}$', strain_G[1, 1, 0]),
            ]

            fig, axes = plt.subplots(2, 2, figsize=(12, 10))

            for ax, (component, label, fem_sol) in zip(axes.flat, components):
                err = np.abs(component - fem_sol)
                err_pos = np.where(err > 0, err, np.min(err[err > 0]))
                # norm = mcolors.LogNorm(vmin=err_pos.min(), vmax=err_pos.max())
                norm = mcolors.LogNorm(vmin=1e-3, vmax=1)
                cmap='Reds'#'RdBu_r'
                im = ax.pcolormesh(X, Y, err_pos, cmap=cmap, norm=norm, shading='flat')
                plt.colorbar(im, ax=ax)
                ax.set_xlabel('x')
                ax.set_ylabel('y')
                ax.set_title(label)

                # for radius, ls in [(R1, '--'), (R2, '-')]:
                #     ax.add_patch(plt.Circle(center + circ_offset, radius,
                #                             fill=False, edgecolor='black',
                #                             linestyle=ls, linewidth=1.5))

            fig.suptitle(f'Preconditioner: {prec},  $n={n}$,  tol=$10^{{-{tol}}}$', fontsize=13)
            plt.tight_layout()
            fname_fig = os.path.join(fig_dir, f'error_{prec}_n{n}_cgtol{tol}_cmap{cmap}.png')
            plt.savefig(fname_fig, dpi=150, bbox_inches='tight')
            print(f'Saved {fname_fig}')
           # plt.show()