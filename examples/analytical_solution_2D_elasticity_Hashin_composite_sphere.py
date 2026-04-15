import numpy as np
import matplotlib.pyplot as plt

#
# # ------------------------------------------------------------
# # 1. Equivalent bulk modulus (2D, plane strain)
# # ------------------------------------------------------------
# def hashin_kappa_hom_2d(kappa1, kappa2, mu2, R1, R2):
#     """
#     Equivalent bulk modulus for Hashin's composite inclusion in 2D (plane strain).
#     """
#     d = 2
#     phi = (R1 / R2) ** d
#     alpha = dim * (kappa_2 - kappa_1) / ((dim - 1) * 2.0 * mu_2 + dim * kappa_1)  #
#     beta = 1 + 2 * (dim - 1) * mu_2 / (dim * kappa_2)
#     # kappa_3 = kappa_2 * (1.0 - beta * (alpha * phi) / (1.0 + alpha * phi))
#     return kappa_2 * (1.0 - beta * (alpha * phi) / (1.0 + alpha * phi))
#
#
# # ------------------------------------------------------------
# # 2. Coefficients a_i, b_i for phases 1 (core) and 2 (shell)
# # ------------------------------------------------------------
# def hashin_ab_2d(kappa1, kappa2, mu2, R1, R2):
#     """
#     Compute (a1,b1), (a2,b2) for 2D Hashin inclusion.
#     """
#     d = 2
#     phi = (R1 / R2) ** d
#     alpha = d * (kappa2 - kappa1) / ((d - 1) * 2.0 * mu2 + d * kappa2)
#
#     a2 = 1.0 / (1.0 + alpha * phi)
#     a1 = (1.0 + alpha) * a2
#
#     b2 = alpha * (R1 ** d) * a2
#     b1 = 0.0
#
#     return (a1, b1), (a2, b2)
#
#
# # ------------------------------------------------------------
# # 3. Strain field for a given phase (2D)
# # ------------------------------------------------------------
# def hashin_strain_field_2d(x, a_i, b_i):
#     """
#     Strain tensor field for Hashin inclusion in 2D, assuming E = I.
#
#     Parameters
#     ----------
#     x : array (..., 2)
#         Spatial points.
#     a_i, b_i : float
#         Coefficients for the phase.
#     """
#     x = np.asarray(x)
#     r = np.linalg.norm(x, axis=-1)
#     r_safe = np.where(r == 0.0, 1.0, r)
#
#     e_r = x / r_safe[..., None]
#     I = np.eye(2)
#
#     r2 = r_safe ** 2
#     s1 = a_i + b_i / r2
#     s2 = 2.0 * b_i / r2  # because d = 2
#
#     I_b = I.reshape((1,) * (x.ndim - 1) + I.shape)
#     e_r_er = e_r[..., :, None] * e_r[..., None, :]
#
#     eps = s1[..., None, None] * I_b - s2[..., None, None] * e_r_er
#     return eps
#
#
# # ------------------------------------------------------------
# # 4. Full 2D strain field with phase selection
# # ------------------------------------------------------------
# def hashin_field_2d(x, R1, R2, ab1, ab2, center=None):
#     """
#     Full 2D strain field for core, shell, and matrix (matrix = I).
#
#     Parameters
#     ----------
#     x : array (..., 2)
#         Spatial points.
#     R1 : float
#         Core radius.
#     R2 : float
#         Shell outer radius.
#     ab1 : tuple
#         Coefficients (a1, b1) for core.
#     ab2 : tuple
#         Coefficients (a2, b2) for shell.
#     center : array (2,), optional
#         Center of inclusion. Default is (0, 0).
#     """
#     x = np.asarray(x)
#     if center is not None:
#         center = np.asarray(center)
#         x = x - center
#
#     r = np.linalg.norm(x, axis=-1)
#
#     eps = np.zeros(x.shape[:-1] + (2, 2))
#
#     # Core
#     mask1 = r <= R1
#     eps[mask1] = hashin_strain_field_2d(x[mask1], *ab1)
#
#     # Shell
#     mask2 = (r > R1) & (r <= R2)
#     eps[mask2] = hashin_strain_field_2d(x[mask2], *ab2)
#
#     # Matrix: eps = I
#     eps[~(mask1 | mask2)] = np.eye(2)
#
#     return eps
#
#
# def compute_displacement_matti(r, r1, r2,
#                                K1, mu1,
#                                K2, mu2,
#                                K_eff, mu_eff):
#     """
#     Compute radial displacement u(r) for a coated inclusion
#     under macroscopic strain <eps> = Id.
#
#     Parameters
#     ----------
#     r : array_like
#         Radial coordinates.
#     r1, r2 : float
#         Core radius and outer coating radius.
#     K1, mu1 : float
#         Bulk and shear modulus of the core.
#     K2, mu2 : float
#         Bulk and shear modulus of the coating.
#     K_eff, mu_eff : float
#         Bulk and shear modulus of the matrix.
#
#     Returns
#     -------
#     u : ndarray
#         Radial displacement field.
#     """
#
#     r = np.asarray(r)
#
#     # Coefficients from the analytical solution
#     denom = 3 * K2 + 4 * mu2
#     a2 = (3 * K_eff + 4 * mu2) / denom
#     b2 = 3 * r2 ** 3 * (K2 - K_eff) / denom
#     a1 = a2 + b2 / r1 ** 3
#     a_eff = 1.0  # macroscopic strain = 1
#
#     # Allocate displacement array
#     u = np.zeros_like(r)
#
#     # Region 1: core
#     mask1 = r < r1
#     u[mask1] = a1 * r[mask1]
#
#     # Region 2: coating
#     mask2 = (r >= r1) & (r < r2)
#     u[mask2] = a2 * r[mask2] + b2 / r[mask2] ** 2
#
#     # Region 3: matrix
#     mask3 = r >= r2
#     u[mask3] = a_eff * r[mask3]
#
#     return u


def solution_hashin(coordinates, mu1, lam1, mu2, lam2,   R1,R2, center):
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


# ------------------------------------------------------------
# Example usage (remove or adapt for your workflow)
# ------------------------------------------------------------
if __name__ == "__main__":
    # Example parameters
    # r1 = 0.3
    # r2 = 0.4
    #
    # K1, mu1 = 0.006, 0.005
    # K2, mu2 = 1.5, 0.5
    # K_eff = 0.2  # your computed homogenized bulk modulus
    # mu_eff = 0.5  # choose or prescribe
    #
    # # Radial grid
    # r = np.linspace(0, 1.0, 400)
    #
    # u = compute_displacement_matti(r, r1, r2, K1, mu1, K2, mu2, K_eff, mu_eff)

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
    # ab1, ab2 = hashin_ab_2d(kappa_1, kappa_2, mu_2, R1, R2)
    # k_hom = hashin_kappa_hom_2d(kappa_1, kappa_2, mu_2, R1, R2)
    # Generate a grid
    N = 64
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
    plt.savefig('hashin_strain_field_2d_v2.png', dpi=150, bbox_inches='tight')
    print("Plot saved as 'hashin_strain_field_2d.png'")
    plt.show()
