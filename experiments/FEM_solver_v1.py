import numpy as np
import matplotlib.pyplot as plt
import muGrid
from muGrid.Solvers import conjugate_gradients

try:
    from mpi4py import MPI

    comm = muGrid.Communicator(MPI.COMM_WORLD)
except ImportError:
    comm = muGrid.Communicator()


def basisFunctions(knotVector, u, i, p):
    if p == 0:
        if (knotVector[i] <= u < knotVector[i + 1]) or (
                u == knotVector[-1] and knotVector[i] <= u <= knotVector[i + 1]):
            return 1.0
        return 0.0
    leftDen = knotVector[i + p] - knotVector[i]
    rightDen = knotVector[i + p + 1] - knotVector[i + 1]
    val = 0.0
    if leftDen > 0:
        val += ((u - knotVector[i]) / leftDen) * basisFunctions(knotVector, u, i, p - 1)
    if rightDen > 0:
        val += ((knotVector[i + p + 1] - u) / rightDen) * basisFunctions(knotVector, u, i + 1, p - 1)
    return val


def basisFunctionsDerivative(knotVector, u, i, p):
    if p == 0:
        return 0.0
    d1 = knotVector[i + p] - knotVector[i]
    d2 = knotVector[i + p + 1] - knotVector[i + 1]
    N1 = basisFunctions(knotVector, u, i, p - 1)
    N2 = basisFunctions(knotVector, u, i + 1, p - 1)
    term1 = (p / d1) if d1 > 0 else 0.0
    term2 = (p / d2) if d2 > 0 else 0.0
    return term1 * N1 - term2 * N2


def evaluate_basis_and_derivatives(u_q, v_q, knot_u, knot_v, degree, n_u, n_v):
    x_idx = []
    y_idx = []
    dB_du = []
    dB_dv = []
    for i in range(n_u):
        Ni, dNi = basisFunctions(knot_u, u_q, i, degree), basisFunctionsDerivative(knot_u, u_q, i, degree)
        # if Ni == 0 and dNi == 0:
        #     continue
        for j in range(n_v):
            Nj, dNj = basisFunctions(knot_v, v_q, j, degree), basisFunctionsDerivative(knot_v, v_q, j, degree)
            val = Ni * Nj
            du_val = dNi * Nj
            dv_val = Ni * dNj
            # if val != 0 or du_val != 0 or dv_val != 0:
            x_idx.append(i)
            y_idx.append(j)
            dB_du.append(du_val)
            dB_dv.append(dv_val)

    return (np.array(x_idx, dtype=int),
            np.array(y_idx, dtype=int),
            np.array(dB_du, dtype=float),
            np.array(dB_dv, dtype=float))


############################################################################################3
domain_size = (1, 2)
nb_grid_points = (50, 50)
del_x = domain_size[0] / nb_grid_points[0]
del_y = domain_size[1] / nb_grid_points[1]

############################################################################################3
degree = 2
knot_u = np.array([0.0, 0.0, 0.0, del_x, del_x, 2 * del_x, 2 * del_x, 2 * del_x])
knot_v = np.array([0.0, 0.0, 0.0, del_y, del_y, 2 * del_y, 2 * del_y, 2 * del_y])

n_control_u = len(knot_u) - degree - 1
n_control_v = len(knot_v) - degree - 1

nb_derivatives = 2
nb_nodal_points = 4

# nb_interact = degree + 1  # nb_of_nonzero_basis_at_xq
nb_nodes_i = 2
nb_nodes_j = 2

# plot basis 1D and derivative
plot_basis = True
if plot_basis:

    fig = plt.figure(figsize=(8.3, 5.0))
    gs = fig.add_gridspec(2, 1, hspace=0.2, wspace=0.1, width_ratios=[1],
                          height_ratios=[1, 1])
    ax_N = fig.add_subplot(gs[0, 0])
    ax_dN = fig.add_subplot(gs[1, 0])

    for basis_idx in np.arange(n_control_u):
        p_idx = 0
        N_at_x = np.zeros(100)
        dN_at_x = np.zeros(100)
        for x_vals in np.linspace(0, knot_u[-1], 100):
            N_at_x[p_idx] = basisFunctions(knotVector=knot_u,
                                           u=x_vals,
                                           i=basis_idx,
                                           p=degree)
            dN_at_x[p_idx] = basisFunctionsDerivative(knotVector=knot_u,
                                                      u=x_vals,
                                                      i=basis_idx,
                                                      p=degree)
            p_idx += 1
        ax_N.plot(np.linspace(0, knot_u[-1], 100), N_at_x)
        ax_dN.plot(np.linspace(0, knot_u[-1], 100), dN_at_x)

    plt.show()

nb_quad_points = 9  #
q_pts_x_rel = np.array([0.5 - np.sqrt(3 / 5) / 2, 0.5, 0.5 + np.sqrt(3 / 5) / 2]) * del_x  #
q_pts_y_rel = np.array([0.5 - np.sqrt(3 / 5) / 2, 0.5, 0.5 + np.sqrt(3 / 5) / 2]) * del_y  #

# q_pts_x_rel = np.array([0.5])
# q_pts_x_rel = np.array([0.25, 0.5, 0.75])
# q_pts_x_rel = np.linspace(0, 1, 9)[1:-1]

q_pts_x = q_pts_x_rel  # + 3.0

quad_points = np.meshgrid(q_pts_x_rel, q_pts_y_rel)
q_weights_x = np.array([(5 / 9) / 2, (8 / 9) / 2, (5 / 9) / 2]) * del_x
q_weights_y = np.array([(5 / 9) / 2, (8 / 9) / 2, (5 / 9) / 2]) * del_y

weights = np.outer(q_weights_x, q_weights_y).flatten()

# shape functions gradients

B_dqnijk = np.zeros([nb_derivatives, nb_quad_points, nb_nodal_points, nb_nodes_i, nb_nodes_j])

for q in range(nb_quad_points):
    u_q = quad_points[0].flatten()[q]
    v_q = quad_points[1].flatten()[q]

    # dNi = np.zeros([n_control_u - 2])  # -2 because we want only elements in the middle
    #
    # for i in range(n_control_u - 2):  #
    #     print(i)
    #     dNi[i] = basisFunctionsDerivative(knot_u, u_q, i + 1, degree)
    x_idx, y_idx, dB_du, dB_dv = evaluate_basis_and_derivatives(
        u_q, v_q, knot_u, knot_v, degree, n_control_u, n_control_v
    )
    # dB_du = dNi
    #
    # print(dB_du.reshape([n_control_u, n_control_v]))
    # print(dB_dv.reshape([n_control_u, n_control_v]))
    # print(dB_dv)
    # now put this into a grid 5x5
    dB_du_ij = dB_du.reshape([n_control_u, n_control_v])  #
    dB_dv_ij = dB_dv.reshape([n_control_u, n_control_v])  # n
    # dB_du_ij = dB_du.reshape([nb_nodes_i, nb_nodes_j])  # nb_nodes_i, nb_nodes_j
    # dB_dv_ij = dB_dv.reshape([nb_nodes_i, nb_nodes_j])  # nb_nodes_i, nb_nodes_j

    # now separate nodal points
    # 0------1--------0 ------1--------0
    # |               |                |
    # |               |                |
    # 2      3        2       3        2
    # |               |                |
    # |               |                |
    # 0------1--------0 ------1--------0
    # |               |                |
    # |               |                |
    # 2      3        2       3        2
    # |               |                |
    # |               |                |
    # 0------1--------0 ------1--------0

    # B_dqnijk[0, q, 0, ...] = np.copy(dB_du_ij)
    # B_dqnijk[1, q, 0, ...] = np.copy(dB_dv_ij)

    # x derivative
    dB_du_nij = np.zeros([nb_nodal_points, nb_nodes_i, nb_nodes_j])
    #                                     x  y
    dB_du_nij[0, ...] = np.array([[dB_du_ij[0, 0], dB_du_ij[2, 0]],
                                  [dB_du_ij[0, 2], dB_du_ij[2, 2]]])
    dB_du_nij[1, ...] = np.array([[dB_du_ij[1, 0], dB_du_ij[3, 0]],
                                  [dB_du_ij[1, 2], dB_du_ij[3, 2]]])
    dB_du_nij[2, ...] = np.array([[dB_du_ij[0, 1], dB_du_ij[2, 1]],
                                  [dB_du_ij[0, 3], dB_du_ij[2, 3]]])
    dB_du_nij[3, ...] = np.array([[dB_du_ij[1, 1], dB_du_ij[3, 1]],
                                  [dB_du_ij[1, 3], dB_du_ij[3, 3]]])

    B_dqnijk[0, q, ...] = np.copy(dB_du_nij)

    # yderivative
    dB_dv_nij = np.zeros([nb_nodal_points, nb_nodes_i, nb_nodes_j])
    #                                     x  y
    dB_du_nij[0, ...] = np.array([[dB_dv_ij[0, 0], dB_dv_ij[2, 0]],
                                  [dB_dv_ij[0, 2], dB_dv_ij[2, 2]]])
    dB_du_nij[1, ...] = np.array([[dB_dv_ij[1, 0], dB_dv_ij[3, 0]],
                                  [dB_dv_ij[1, 2], dB_dv_ij[3, 2]]])
    dB_du_nij[2, ...] = np.array([[dB_dv_ij[0, 1], dB_dv_ij[2, 1]],
                                  [dB_dv_ij[0, 3], dB_dv_ij[2, 3]]])
    dB_du_nij[3, ...] = np.array([[dB_dv_ij[1, 1], dB_dv_ij[3, 1]],
                                  [dB_dv_ij[1, 3], dB_dv_ij[3, 3]]])

    B_dqnijk[1, q, ...] = np.copy(dB_du_nij)
    print(q)
###
# Here we just prepared the stencil for elements of size 1 in 1D 3 quad points

# We have B,

fc = muGrid.GlobalFieldCollection(nb_domain_grid_pts=nb_grid_points,
                                  sub_pts={"quad_points": nb_quad_points,
                                           "nodal_points": nb_nodal_points})
# I need  less nodal points than pixel. Exactly one less for degree = 2

gradiant_field_ijqxyz = fc.real_field("gradient", components_shape=(1, 2), sub_division="quad_points")

temp_field_inxyz = fc.real_field("temperature", components_shape=(1,), sub_division="nodal_points")
temp_coef = fc.real_field("temperature_coef", components_shape=(1,), sub_division="nodal_points")

print('----------------------------------------------')  # for i in range(2):

point_of_origin = [0, 0]
grad_op = muGrid.ConvolutionOperator(point_of_origin, B_dqnijk)

# grad_op.apply(nodal_field=temp_coef, quadrature_point_field=gradiant_field_ijqxyz)

# grad_op = muGrid.ConvolutionOperator(point_of_origin, B_dqnijk)
# weights = np.ones([nb_quad_points])
# create a field elements with size 1 -- coordinates
# x_coords, y_coords = np.meshgrid(np.arange(nb_grid_points), np.arange(nb_grid_points), indexing="ij")

# xq_coords = np.zeros([nb_quad_points, nb_grid_points])

rhs = fc.real_field("rhs", components_shape=(1,), sub_division="nodal_points")
solution = fc.real_field("solution", components_shape=(1,), sub_division="nodal_points")

x_coords = np.zeros([2, nb_nodal_points, nb_grid_points[0], nb_grid_points[1]])
for n_p in range(nb_nodal_points):
    for dim in range(2):
        x_coords[dim, n_p, ...] = np.copy(rhs.coords[dim]) * domain_size[dim]

# del_x = (rhs.coords[0, 1, 0] - rhs.coords[0, 0, 0])
# del_y = (rhs.coords[1, 0, 1] - rhs.coords[1, 0, 0])

x_coords[0, 1, ...] += del_x / 2

x_coords[1, 2, ...] += del_y / 2

x_coords[0, 3, ...] += del_x / 2
x_coords[1, 3, ...] += del_y / 2

rhs.s[0] = ( np.cos(2 * np.pi * x_coords[0]  ) * np.cos(
    2 * np.pi * x_coords[1] ))

#rhs.p = (x_coords[0] ) ** 2#+ x_coords[1]
rhs.p -= np.mean(rhs.p)


def hessp(x, Ax):
    """
    Function to compute the product of the Hessian matrix with a vector.
    The Hessian is represented by the convolution operator.
    """
    # decomposition.communicate_ghosts(x)
    # x = laplace_2(x)
    grad_op.apply(nodal_field=x, quadrature_point_field=gradiant_field_ijqxyz)
    grad_op.transpose(quadrature_point_field=gradiant_field_ijqxyz, nodal_field=Ax, weights=weights)

    # We need the minus sign because the Laplace operator is negative
    # definite, but the conjugate-gradients solver assumes a
    # positive-definite operator.
    # Ax.s /= -np.mean(grid_spacing) ** 2  # Scale by grid spacing

    return Ax


def callback(it, x, r, p):
    """
    Callback function to print the current solution, residual, and search direction.
    """
    print(f"{it:5} {np.dot(r.ravel(), r.ravel()):.5}")


conjugate_gradients(
    comm=comm,
    fc=fc,
    hessp=hessp,  # linear operator
    b=rhs,
    x=solution,
    tol=1e-6,
    callback=callback,
    maxiter=1000,
)

if plt is not None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    pcm = ax1.pcolormesh(x_coords[0, 0], x_coords[1, 0], rhs.s[0, 0],
                         # cmap=mpl.cm.Greys, vmin=contrast, vmax=1, linewidth=0,
                         rasterized=True)
    pcm = ax2.pcolormesh(x_coords[0, 0], x_coords[1, 0], solution.s[0, 0],
                         # cmap=mpl.cm.Greys, vmin=contrast, vmax=1, linewidth=0,
                         rasterized=True)
    ax1.set_aspect('equal')
    ax2.set_aspect('equal')
    # ax1.imshow(rhs.p[0])
    # ax2.imshow(solution.p[0])
    plt.show()
