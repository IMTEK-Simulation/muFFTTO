import numpy as np
import matplotlib.pyplot as plt

#
# # ------------------------------------------------------------
# # 1. Equivalent bulk modulus (2D, plane strain)
# # ------------------------------------------------------------
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


if __name__ == "__main__":

    dim = 2
    # Material parameters
    lambda_1 = 0.001  # first Lamé
    mu_1 = 0.005  # second Lamé
    kappa_1 = lambda_1 + 2 * mu_1 / dim  # bulk

    lambda_2 = 1.
    mu_2 = 0.5
    kappa_2 = lambda_2 + 2 * mu_2 / dim

    R1, R2 = 0.2, 0.4

    # Compute coefficients
    # Generate a grid
    N = 512
    xlin = np.linspace(0, 1, N, endpoint=False)
    X, Y = np.meshgrid(xlin, xlin, indexing='ij')
    pts = np.stack([X, Y], axis=-1)

    # Compute strain field with inclusion centered at (0.5, 0.5)
    center = np.array([0.5, 0.5])
    # eps_old = hashin_field_2d(pts, R1, R2, ab1, ab2, center=center)

    eps=solution_hashin(pts, mu1=mu_1, lam1=lambda_1, mu2=mu_2, lam2=lambda_2, R1=R1, R2=R2, center=center)
    # eps has shape (N, N, 2, 2)
    print("Strain at center:", eps[N // 2, N // 2])

    # Plot the strain field components
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Plot each component of the strain tensor
    components = [
        (eps[..., 0, 0], r'$\varepsilon_{xx}$'),
        (eps[..., 0, 1], r'$\varepsilon_{xy}$'),
        (eps[..., 1, 0], r'$\varepsilon_{yx}$'),
        (eps[..., 1, 1], r'$\varepsilon_{yy}$')
    ]

    for ax, (component, label) in zip(axes.flat, components):
        vmin_comp = np.min(component)
        vmax_comp = np.max(component)
        im = ax.imshow(component, extent=[0, 1, 0, 1], origin='lower', cmap='RdBu_r',
                       vmin=0.2, vmax=1.3)
        print(vmin_comp, vmax_comp)

        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title(label)
        plt.colorbar(im, ax=ax)

        # Draw circles for core and shell boundaries
        number_of_pixels=(N,N)
        circle1 = plt.Circle(center+0.5/np.array( number_of_pixels), R1, fill=False, edgecolor='black', linestyle='--', linewidth=1.5)
        circle2 = plt.Circle(center+0.5/np.array( number_of_pixels), R2, fill=False, edgecolor='black', linestyle='-', linewidth=1.5)
        ax.add_patch(circle1)
        ax.add_patch(circle2)

    plt.tight_layout()
    plt.savefig('hashin_strain_field_2d_analytical_v2.png', dpi=150, bbox_inches='tight')
    print("Plot saved as 'hashin_strain_field_2d_analytical_v2.png'")
    plt.show()
