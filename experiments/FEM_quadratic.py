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
        if Ni == 0 and dNi == 0:
            continue
        for j in range(n_v):
            Nj, dNj = basisFunctions(knot_v, v_q, j, degree), basisFunctionsDerivative(knot_v, v_q, j, degree)
            val = Ni * Nj
            du_val = dNi * Nj
            dv_val = Ni * dNj
            if val != 0 or du_val != 0 or dv_val != 0:
                x_idx.append(i)
                y_idx.append(j)
                dB_du.append(du_val)
                dB_dv.append(dv_val)
    return (np.array(x_idx, dtype=int),
            np.array(y_idx, dtype=int),
            np.array(dB_du, dtype=float),
            np.array(dB_dv, dtype=float))


degree = 2
knot_u = np.array([0.0, 0.0, 0.0, 3.0, 3.0, 3.0])
knot_v = np.array([0.0, 0.0, 0.0, 3.0, 3.0, 3.0])

n_control_u = 3
n_control_v = 3

nb_derivatives = 2
nb_quad_points = 9
nb_nodes = 1

nb_nodes_i = 3
nb_nodes_j = 3

B_dqnijk = np.zeros([nb_derivatives, nb_quad_points, nb_nodes, nb_nodes_i, nb_nodes_j])
# guassian quad points
# quad_points_par_q = np.array([
#     [0.5 - 1 / (2 * np.sqrt(3)), 0.5 - 1 / (2 * np.sqrt(3))],
#     [0.5 + 1 / (2 * np.sqrt(3)), 0.5 - 1 / (2 * np.sqrt(3))],
#     [0.5 + 1 / (2 * np.sqrt(3)), 0.5 + 1 / (2 * np.sqrt(3))],
#     [0.5 - 1 / (2 * np.sqrt(3)), 0.5 + 1 / (2 * np.sqrt(3))]
# ])
# q_pts_x = np.array([0.5 - np.sqrt(3 / 5) / 2, .5, 0.5 + np.sqrt(3 / 5) / 2])
# quad_points_par_q = np.meshgrid(q_pts_x, q_pts_x)
# q_weights_x = np.array([(5 / 9) / 2, (8 / 9) / 2, (5 / 9) / 2])
# # q_weights_x = np.array([1/3, 1/3, 1/3])
# weights = np.outer(q_weights_x, q_weights_x).flatten()


q_pts_x = np.array([1.5 - np.sqrt(3 / 5) / 2, 1.5, 1.5 + np.sqrt(3 / 5) / 2])
quad_points_par_q = np.meshgrid(q_pts_x, q_pts_x)
q_weights_x = np.array([(5 / 9) / 2, (8 / 9) / 2, (5 / 9) / 2])
# q_weights_x = np.array([1/3, 1/3, 1/3])
weights = np.outer(q_weights_x, q_weights_x).flatten()

for q in range(nb_quad_points):
    u_q = quad_points_par_q[0].flatten()[q]
    v_q = quad_points_par_q[1].flatten()[q]
    # u_q, v_q = quad_points_par_q[q]
    x_idx, y_idx, dB_du, dB_dv = evaluate_basis_and_derivatives(
        u_q, v_q, knot_u, knot_v, degree, n_control_u, n_control_v
    )
    #
    print(dB_du)
    print(dB_dv)
    # dB_du_ij = dB_du.reshape([3, 3])  # nb_nodes_i, nb_nodes_j
    # dB_dv_ij = dB_dv.reshape([3, 3])  # nb_nodes_i, nb_nodes_j
    # B_dqnijk[0, q, 0, ...] = np.copy(dB_du_ij)
    # B_dqnijk[1, q, 0, ...] = np.copy(dB_dv_ij)
    print(q)
# We have B,

nb_grid_points = 10
fc = muGrid.GlobalFieldCollection(nb_domain_grid_pts=(nb_grid_points, nb_grid_points),
                                  sub_pts={"quad_points": nb_quad_points, "nodal_points": 1})

gradiant_field_ijqxyz = fc.real_field("gradient", components_shape=(1, 2), sub_division="quad_points")

temp_field_inxyz = fc.real_field("temperature", components_shape=(1,), sub_division="nodal_points")
x, y = np.meshgrid(np.arange(nb_grid_points), np.arange(nb_grid_points), indexing="ij")
# u_fun= lambda x, y: x**2 + y**2 #+ 2*x*y
# u_fun= lambda x, y: x**1 #+ y**2 + 2*x*y

# u_fun = lambda x, y: 2 * np.pi * x + y ** 2  # + 2*x*y
#
# temp_field_inxyz.s[0, 0, :, :] = u_fun(x, y)
point_of_origin = [-1, -1]
# op = muGrid.ConvolutionOperator(point_of_origin, B_dqnijk)
# op.apply(nodal_field=temp_field_inxyz, quadrature_point_field=gradiant_field_ijqxyz)
print()

print("dB_du length per quad point:", len(dB_du))
print("dB_dv length per quad point:", len(dB_dv))

print("Shape of B_dqnijk:", B_dqnijk.shape)
print("B_dqnijk array:\n", B_dqnijk)

print('----------------------------------------------')  # for i in range(2):

grad_op = muGrid.ConvolutionOperator(point_of_origin, B_dqnijk)
# weights = np.ones([nb_quad_points])

rhs = fc.real_field("rhs")
solution = fc.real_field("solution")

rhs.p = (1 + np.cos(2 * np.pi * x / nb_grid_points) * np.cos(2 * np.pi * y / nb_grid_points)) ** 10
rhs.p -= np.mean(rhs.p)

grid_spacing = 1 / np.array(nb_grid_points)


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
    comm,
    fc,
    hessp,  # linear operator
    rhs,
    solution,
    tol=1e-6,
    callback=callback,
    maxiter=1000,
)

if plt is not None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    ax1.imshow(rhs.p)
    ax2.imshow(solution.p)
    plt.show()
