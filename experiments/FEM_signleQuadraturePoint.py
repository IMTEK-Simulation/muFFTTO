import numpy as np
import matplotlib.pyplot as plt
import muGrid


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


if __name__ == "__main__":
    # properties of our stencil
    # Try to compute B for degree 2. It will need 3x3 nodal point I think
    # WE HAVE FIXED SHAPE OF THE ELEMENT TO 1x1 in this code.
    degree = 1
    knot_u = np.array([0.0, 0.0, 1.0, 1.0])
    knot_v = np.array([0.0, 0.0, 1.0, 1.0])
    # knot_u = np.array([0.0, 0.0, 0.5, 0.5])  # TODO: this is the size of your element
    # knot_v = np.array([0.0, 0.0, 0.5, 0.5])

    n_control_u = 2
    n_control_v = 2

    nb_derivatives = 2  # der in x and y direction
    nb_quad_points = 4  # bilinear elements/basis
    nb_nodes = 1  # just for muGrid framework
    nb_nodes_i = 2  # of the stencil in x direction
    nb_nodes_j = 2  # of the stencil in y direction

    B_dqnijk = np.zeros([nb_derivatives, nb_quad_points, nb_nodes, nb_nodes_i, nb_nodes_j])

    # quad points
    quad_point_helper_0 = 0.5 + 1 / (2 * np.sqrt(3))
    quad_points_par_q = np.array([[0.5 - 1 / (2 * np.sqrt(3)), 0.5 - 1 / (2 * np.sqrt(3))],
                                  [0.5 + 1 / (2 * np.sqrt(3)), 0.5 - 1 / (2 * np.sqrt(3))],
                                  [0.5 + 1 / (2 * np.sqrt(3)), 0.5 + 1 / (2 * np.sqrt(3))],
                                  [0.5 - 1 / (2 * np.sqrt(3)), 0.5 + 1 / (2 * np.sqrt(
                                      3))]])  # TODO: Rescale the positions of the quadrature points  based on the shape of your element
    for q in range(nb_quad_points):
        # quadrature points
        u_q, v_q = quad_points_par_q[q]

        x_idx, y_idx, dB_du, dB_dv = evaluate_basis_and_derivatives(
            u_q, v_q, knot_u, knot_v, degree, n_control_u, n_control_v
        )
        # shape function gradient at the quad point X^q
        # TODO[Pri] Check the derivative direction. It must be clear
        dB_du_ij = dB_du.reshape([nb_nodes_i, nb_nodes_j])
        dB_dv_ij = dB_dv.reshape([nb_nodes_i, nb_nodes_j])

        B_dqnijk[0, q, 0, ...] = np.copy(dB_du_ij)
        B_dqnijk[1, q, 0, ...] = np.copy(dB_dv_ij)

    fc = muGrid.GlobalFieldCollection(nb_domain_grid_pts=(3, 3), sub_pts={"quad_points": 4, "nodal_points": 1})

    # max_basis = 4
    gradiant_field_ijqxyz = fc.real_field("gradient", components_shape=(1, 2), sub_division="quad_points")
    temp_field_inxyz = fc.real_field("temperature", components_shape=(1,), sub_division="nodal_points")

    #### Test field

    # quad_coordinates = discretization.get_quad_points_coordinates()
    x, y = np.meshgrid(np.arange(3), np.arange(3))
    u_fun_4x3y = lambda x, y: 4 * x + 3 * y  # np.sin(x)
    temp_field_inxyz.s[0, 0, :, :] = u_fun_4x3y(x, y)

    # TODO This has to be a discretization stencil dependant # for degree 2 this may be [-1, -1]
    point_of_origin = [0, 0]

    op = muGrid.ConvolutionOperator(point_of_origin, B_dqnijk)

    op.apply(nodal_field=temp_field_inxyz, quadrature_point_field=gradiant_field_ijqxyz)

    print()  # for i in range(2):
    #     for j in range(2):
    #         gradiant_feild.s[0, :N, 0, i, j] = dB_du
    #         gradiant_feild.s[1, :N, 0, i, j] = dB_dv
    #         gradiant_feild.s[2, :N, 0, i, j] = x_idx
    #         gradiant_feild.s[3, :N, 0, i, j] = y_idx
    #         gradiant_feild.s[4, 0, 0, i, j] = u_q
    #         gradiant_feild.s[4, 1, 0, i, j] = v_q
    #
    # print("derivative", gradiant_feild.s[0, :N, 0, i, j])
    # print("derivative", gradiant_feild.s[1, :N, 0, i, j])
    #
    # spans_u = [(knot_u[i], knot_u[i + 1]) for i in range(len(knot_u) - 1) if knot_u[i] != knot_u[i + 1]]
    # spans_v = [(knot_v[i], knot_v[i + 1]) for i in range(len(knot_v) - 1) if knot_v[i] != knot_v[i + 1]]
    #
    # fig, ax = plt.subplots(figsize=(5, 5))
    # for (u0, u1) in spans_u:
    #     ax.plot([u0, u0], [0, 1], 'k--', alpha=0.5)
    #     ax.plot([u1, u1], [0, 1], 'k--', alpha=0.5)
    # for (v0, v1) in spans_v:
    #     ax.plot([0, 1], [v0, v0], 'k--', alpha=0.5)
    #     ax.plot([0, 1], [v1, v1], 'k--', alpha=0.5)
    #
    # ax.plot([u_q], [v_q], 'ro', markersize=8)
    # ax.set_xlim(-0.1, 1.1)
    # ax.set_ylim(-0.1, 1.1)
    # ax.set_xlabel('u')
    # ax.set_ylabel('v')
    # ax.set_aspect("equal")
    # ax.grid(True, alpha=0.3)
    # plt.tight_layout()
    # plt.show()
