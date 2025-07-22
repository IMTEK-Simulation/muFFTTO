import warnings

import numpy as np
from scipy.ndimage import label


def get_shape_function_gradient_matrix(my_domain, element_type):
    """Example function with types documented in the docstring.

    `PEP 484`_ type annotations are supported. If attribute, parameter, and
    return types are annotated according to `PEP 484`_, they do not need to be
    included in the docstring:

    Parameters
    ----------
    param1 : int
        The first parameter.
    param2 : str
        The second parameter.

    Returns
    -------
    bool
        True if successful, False otherwise.

    .. _PEP 484:
        https://www.python.org/dev/peps/pep-0484/

    """
    if not element_type in ['linear_1D', 'linear_triangles', 'linear_triangles_tilled', 'bilinear_rectangle',
                            'trilinear_hexahedron']:
        raise ValueError('Unrecognised element_type {}'.format(element_type))
    # TODO [martin] get rid of match, use IF
    match element_type:
        case 'linear_1D':
            if my_domain.domain_dimension != 1:
                raise ValueError('Element_type {} is implemented only in 1D'.format(element_type))
            """ Geometry for 2 linear triangular elements in pixel
           
                        q = 1        
                x_1_______________x_2______________x_3
        
            """

            my_domain.nb_quad_points_per_pixel = 1
            my_domain.nb_nodes_per_pixel = 1  # left bottom corner belong to pixel.
            my_domain.nb_unique_nodes_per_pixel = 1
            #

            # nodal points offsets
            my_domain.offsets = np.array([[0], [1]])

            """  Structure of B matrix: 
            #     B(:,:,q,e) --> is B matrix evaluate gradient at point q in  element e
            #     B(:,:,q,e) has size [dim,nb_of_nodes/basis_functions] 
            #                           (usually 4 in 2D and 8 in 3D)
            #     B(:,:,q,e) = [ ∂φ_1/∂x_1  ∂φ_2/∂x_1  ∂φ_3/∂x_1 ∂φ_4/∂x_1 ;
            #                    ∂φ_1/∂x_2  ∂φ_2/∂x_2  ∂φ_3/∂x_2 ∂φ_4/∂x_2]   at (q)
            """
            my_domain.B_gradient = np.zeros([my_domain.domain_dimension, 2,
                                             my_domain.nb_quad_points_per_pixel])
            h_x = my_domain.pixel_size[0]

            # @formatter:off
            def basis_interpolator(xi):
                N_at_xi_eta=np.zeros([2, 1])

                N_at_xi_eta[0,0]=   1 - xi
                N_at_xi_eta[1,0]=   xi
                return N_at_xi_eta

            my_domain.N_basis_interpolator_array = np.array([
                lambda xi: 1 - xi    ,
                lambda xi: xi
            ])
            #[f(0.2,0.3) for f in basis_interpolator]
            #test = basis_interpolator(xi=0.0, eta=0.3)
            # @formatter:on
            # quad_points_coord stores spatial coordinates of quad points [dim,q]
            # my_domain.quad_points_coord[:,q] = [x_q,y_q,x_q]
            my_domain.quad_points_coord = np.zeros(
                [my_domain.domain_dimension, my_domain.nb_quad_points_per_pixel])
            my_domain.quad_points_coord[:, 0] = [h_x / 2]

            my_domain.quad_points_coord_parametric = np.zeros(
                [my_domain.domain_dimension, my_domain.nb_quad_points_per_pixel])
            my_domain.quad_points_coord_parametric[:, 0] = [1 / 2]

            # @formatter:off   B(dim,number of nodal values,quad point ,element)
            my_domain.B_gradient[:, :,  0] = [[-1 / h_x,   1 / h_x  ]]
            # @formatter:on
            my_domain.quadrature_weights = np.zeros([my_domain.nb_quad_points_per_pixel])
            my_domain.quadrature_weights[0] = h_x / 2

            B_at_pixel_dnijkq = my_domain.B_gradient.reshape(my_domain.domain_dimension,
                                                             my_domain.nb_nodes_per_pixel,
                                                             *my_domain.domain_dimension * (2,),
                                                             my_domain.nb_quad_points_per_pixel)

            if my_domain.domain_dimension == 2:
                B_at_pixel_dnijkq = np.swapaxes(B_at_pixel_dnijkq, 2, 3)
            elif my_domain.domain_dimension == 3:
                B_at_pixel_dnijkq = np.swapaxes(B_at_pixel_dnijkq, 2, 4)
                warnings.warn('Swapaxes for 3D is not tested.')

            my_domain.B_grad_at_pixel_dqnijk = np.moveaxis(B_at_pixel_dnijkq, [-1], [1])

            return

        case 'linear_triangles':
            if my_domain.domain_dimension != 2:
                raise ValueError('Element_type {} is implemented only in 2D'.format(element_type))
            """ Geometry for 2 linear triangular elements in pixel
                x_3_______________x_4
                  |  \            |
                  |     \  q = 2  |
                  |        \      |
                  | q = 1    \    |
                x_1_______________x_2
      
            """

            my_domain.nb_quad_points_per_pixel = 2
            my_domain.nb_nodes_per_pixel = 1  # left bottom corner belong to pixel.
            my_domain.nb_unique_nodes_per_pixel = 1
            #

            # nodal points offsets
            my_domain.offsets = np.array([[0, 0], [1, 0],
                                          [0, 1], [1, 1]])

            """  Structure of B matrix: 
            #     B(:,:,q,e) --> is B matrix evaluate gradient at point q in  element e
            #     B(:,:,q,e) has size [dim,nb_of_nodes/basis_functions] 
            #                           (usually 4 in 2D and 8 in 3D)
            #     B(:,:,q,e) = [ ∂φ_1/∂x_1  ∂φ_2/∂x_1  ∂φ_3/∂x_1 ∂φ_4/∂x_1 ;
            #                    ∂φ_1/∂x_2  ∂φ_2/∂x_2  ∂φ_3/∂x_2 ∂φ_4/∂x_2]   at (q)
            """
            my_domain.B_gradient = np.zeros([my_domain.domain_dimension, 4,
                                             my_domain.nb_quad_points_per_pixel])
            h_x = my_domain.pixel_size[0]
            h_y = my_domain.pixel_size[1]

            # @formatter:off
            def basis_interpolator(xi, eta):
                N_at_xi_eta=np.zeros([2, 2])

                N_at_xi_eta[0,0]=   1 - xi  - eta  if eta < 1 - xi else 0 * xi
                N_at_xi_eta[0,1]= xi               if eta < 1 - xi else 1 - eta
                N_at_xi_eta[1,0]=             eta  if eta < 1 - xi else 1 - xi
                N_at_xi_eta[1,1]= 0 * xi           if eta < 1 - xi else xi +eta -1
                return N_at_xi_eta

            my_domain.N_basis_interpolator_array = np.array([
                [lambda xi, eta: 1 - xi  - eta  if eta < 1 - xi else 0 * xi,
                 lambda xi, eta:     xi             if eta < 1 - xi else 1 - eta ],[
                 lambda xi, eta:                eta  if eta < 1 - xi else 1 - xi  ,
                 lambda xi, eta:                   0 * xi if eta < 1 - xi else     xi +eta-1]
                  ])
            #[f(0.2,0.3) for f in basis_interpolator]
            #test = basis_interpolator(xi=0.0, eta=0.3)
            # @formatter:on
            # quad_points_coord stores spatial coordinates of quad points [dim,q]
            # my_domain.quad_points_coord[:,q] = [x_q,y_q,x_q]
            my_domain.quad_points_coord = np.zeros(
                [my_domain.domain_dimension, my_domain.nb_quad_points_per_pixel])
            my_domain.quad_points_coord[:, 0] = [h_x / 3, h_y / 3]
            my_domain.quad_points_coord[:, 1] = [h_x * 2 / 3, h_y * 2 / 3]

            my_domain.quad_points_coord_parametric = np.zeros(
                [my_domain.domain_dimension, my_domain.nb_quad_points_per_pixel])
            my_domain.quad_points_coord_parametric[:, 0] = [1 / 3, 1 / 3]
            my_domain.quad_points_coord_parametric[:, 1] = [2 / 3, 2 / 3]

            # @formatter:off   B(dim,number of nodal values,quad point ,element)
            my_domain.B_gradient[:, :,  0] = [[-1 / h_x,   1 / h_x,    0,          0],
                                              [-1 / h_y,         0,    1 / h_y,    0]]
            my_domain.B_gradient[:, :,  1] = [[0,         0, - 1 / h_x, 1 / h_x],
                                              [0, - 1 / h_y,         0, 1 / h_y]]
            # @formatter:on
            my_domain.quadrature_weights = np.zeros([my_domain.nb_quad_points_per_pixel])
            my_domain.quadrature_weights[0] = h_x * h_y / 2
            my_domain.quadrature_weights[1] = h_x * h_y / 2

            B_at_pixel_dnijkq = my_domain.B_gradient.reshape(my_domain.domain_dimension,
                                                             my_domain.nb_nodes_per_pixel,
                                                             *my_domain.domain_dimension * (2,),
                                                             my_domain.nb_quad_points_per_pixel)

            if my_domain.domain_dimension == 2:
                B_at_pixel_dnijkq = np.swapaxes(B_at_pixel_dnijkq, 2, 3)
            elif my_domain.domain_dimension == 3:
                B_at_pixel_dnijkq = np.swapaxes(B_at_pixel_dnijkq, 2, 4)
                warnings.warn('Swapaxes for 3D is not tested.')

            my_domain.B_grad_at_pixel_dqnijk = np.moveaxis(B_at_pixel_dnijkq, [-1], [1])

            return
        case 'linear_triangles_tilled':
            if my_domain.domain_dimension != 2:
                raise ValueError('Element_type {} is implemented only in 2D'.format(element_type))
            """ Geometry for 2 linear triangular elements in pixel
                  |offset_x_dir |
                                x_2_____________x_3
                               |  \            |
                              |     \  q = 2  |
                             |        \      |
                           | q = 1    \    |
                        x_0_______________x_1

            """

            my_domain.nb_quad_points_per_pixel = 2
            my_domain.nb_nodes_per_pixel = 1  # left bottom corner belong to pixel.
            my_domain.nb_unique_nodes_per_pixel = 1
            #

            # nodal points offsets
            my_domain.offsets = np.array([[0, 0], [1, 0],
                                          [0, 1], [1, 1]])

            """  Structure of B matrix: 
            #     B(:,:,q,e) --> is B matrix evaluate gradient at point q in  element e
            #     B(:,:,q,e) has size [dim,nb_of_nodes/basis_functions] 
            #                           (usually 4 in 2D and 8 in 3D)
            #     B(:,:,q,e) = [ ∂φ_1/∂x_1  ∂φ_2/∂x_1  ∂φ_3/∂x_1 ∂φ_4/∂x_1 ;
            #                    ∂φ_1/∂x_2  ∂φ_2/∂x_2  ∂φ_3/∂x_2 ∂φ_4/∂x_2]   at (q)
            """

            h_x = my_domain.pixel_size[0]
            h_y = my_domain.pixel_size[1]
            element_area = h_x * h_y / 2

            my_domain.nodal_points_coord = np.zeros(
                [my_domain.domain_dimension, my_domain.nb_unique_nodes_per_pixel * 2 ** my_domain.domain_dimension])

            offset_x_dir =  1 / 2 * h_x
            my_domain.nodal_points_coord[:, 0] = [0., 0]  #
            my_domain.nodal_points_coord[:, 1] = [h_x, 0]
            my_domain.nodal_points_coord[:, 2] = [offset_x_dir, h_y]
            my_domain.nodal_points_coord[:, 3] = [offset_x_dir + h_x, h_y]

            x_shift = lambda x, y: x + offset_x_dir * y
            x_0, y_0 = my_domain.nodal_points_coord[:, 0]  # 0 1
            x_1, y_1 = my_domain.nodal_points_coord[:, 1]  # 0 1
            x_2, y_2 = my_domain.nodal_points_coord[:, 2]  # 1 0
            x_3, y_3 = my_domain.nodal_points_coord[:, 3]  # 1 1
            # we must use eta and xi from un deformed  pixel
            # @formatter:off
            my_domain.N_basis_interpolator_array = np.array([
                [lambda xi, eta:  ((x_1*y_2-y_1*x_2)+ (y_1-y_2)*x_shift(xi*h_x,eta*h_y) +(x_2-x_1)*eta*h_y)/(2*element_area) if eta < 1 - xi else 0 * xi,
                 lambda xi, eta:  ((x_2*y_0-y_2*x_0)+ (y_2-y_0)*x_shift(xi*h_x,eta*h_y) +(x_0-x_2)*eta*h_y)/(2*element_area) if eta < 1 - xi else ((x_3*y_2-y_3*x_2)+ (y_3-y_2)*x_shift(xi*h_x,eta*h_y) +(x_2-x_3)*eta*h_y)/(2*element_area)    ],
                [lambda xi, eta:  ((x_0*y_1-y_0*x_1)+ (y_0-y_1)*x_shift(xi*h_x,eta*h_y) +(x_1-x_0)*eta*h_y)/(2*element_area)if eta < 1 - xi else ((x_1*y_3-y_1*x_3)+ (y_1-y_3)*x_shift(xi*h_x,eta*h_y) +(x_3-x_1)*eta*h_y)/(2*element_area),
                 lambda xi, eta:                                                       0 * xi if eta < 1 - xi else ((x_2*y_1-y_2*x_1)+ (y_2-y_1)*x_shift(xi*h_x,eta*h_y) +(x_1-x_2)*eta*h_y)/(2*element_area)]
            ])

            # @formatter:on
            # quad_points_coord stores spatial coordinates of quad points [dim,q]
            # in parametric space
            my_domain.quad_points_coord_parametric = np.zeros(
                [my_domain.domain_dimension, my_domain.nb_quad_points_per_pixel])
            my_domain.quad_points_coord_parametric[:, 0] = [1 / 3, 1 / 3]
            my_domain.quad_points_coord_parametric[:, 1] = [2 / 3, 2 / 3]
            # my_domain.quad_points_coord[:,q] = [x_q,y_q,x_q]
            my_domain.quad_points_coord = np.zeros(
                [my_domain.domain_dimension, my_domain.nb_quad_points_per_pixel])

            my_domain.quad_points_coord[:, 0] = [x_shift(h_x / 3, h_y / 3), h_y / 3]
            my_domain.quad_points_coord[:, 1] = [x_shift(h_x * 2 / 3, h_y * 2 / 3), h_y * 2 / 3]

            my_domain.B_gradient = np.zeros([my_domain.domain_dimension, 4,
                                             my_domain.nb_quad_points_per_pixel])

            # det_jacobian = h_x * h_y / 4
            jacobian = np.array([[h_x, h_x / 2], [0, h_y]])
            # inv_jacobian = np.linalg.inv(jacobian)

            # @formatter:off   B(dim,number of nodal values,quad point ,element)
            # construction of B matrix
            for qp in range(0, my_domain.nb_quad_points_per_pixel):
                x_q = my_domain.quad_points_coord_parametric[:, qp]
                xi = x_q[0]
                eta = x_q[1]
                # @formatter:off
                if qp == 0:
                    x_0, y_0= my_domain.nodal_points_coord[:, 0]
                    x_1, y_1= my_domain.nodal_points_coord[:, 1]
                    x_2, y_2= my_domain.nodal_points_coord[:, 2]
                    my_domain.B_gradient[:,0, qp]=np.array( [   y_1-y_2,x_2-x_1]) # point 0 0 --- 0
                    my_domain.B_gradient[:,1, qp]=np.array( [   y_2-y_0,x_0-x_2]) # point 0 1  --- 1
                    my_domain.B_gradient[:,2, qp]=np.array( [   y_0-y_1,x_1-x_0]) # point  1 1  --- 3
                    my_domain.B_gradient[:,3, qp]=np.array( [   0, 0]) # point  1 1  --- 2
                elif qp == 1:
                    x_1, y_1= my_domain.nodal_points_coord[:, 1] #  0 1
                    x_2, y_2= my_domain.nodal_points_coord[:, 2] #  1 0
                    x_3, y_3= my_domain.nodal_points_coord[:, 3] #  1 1
                    my_domain.B_gradient[:,0, qp]=np.array( [  0, 0]) # point 0 0
                    my_domain.B_gradient[:,1, qp]=np.array( [  y_3-y_2, x_2-x_3]) # point 0 1
                    my_domain.B_gradient[:,2, qp]=np.array( [  y_1-y_3, x_3-x_1 ]) # point  1 1
                    my_domain.B_gradient[:,3, qp]=np.array( [  y_2-y_1, x_1-x_2]) # point  1 1
                #my_domain.B_gradient[:,:, qp] = np.matmul(inv_jacobian,my_domain.B_gradient[:,:, qp])
                my_domain.B_gradient[:,:, qp]/=(2*element_area)
                # @formatter:on

            # @formatter:on
            my_domain.quadrature_weights = np.zeros([my_domain.nb_quad_points_per_pixel])
            my_domain.quadrature_weights[0] = element_area
            my_domain.quadrature_weights[1] = element_area

            B_at_pixel_dnijkq = my_domain.B_gradient.reshape(my_domain.domain_dimension,
                                                             my_domain.nb_nodes_per_pixel,
                                                             *my_domain.domain_dimension * (2,),
                                                             my_domain.nb_quad_points_per_pixel)

            if my_domain.domain_dimension == 2:
                B_at_pixel_dnijkq = np.swapaxes(B_at_pixel_dnijkq, 2, 3)
            elif my_domain.domain_dimension == 3:
                B_at_pixel_dnijkq = np.swapaxes(B_at_pixel_dnijkq, 2, 4)
                warnings.warn('Swapaxes for 3D is not tested.')

            my_domain.B_grad_at_pixel_dqnijk = np.moveaxis(B_at_pixel_dnijkq, [-1], [1])

            return
        case 'bilinear_rectangle':
            if my_domain.domain_dimension != 2:
                raise ValueError('Element_type {} is implemented only in 2D'.format(element_type))
            """ 
                %   x_3____e=1__x_4
                %   |            |
                %   |  q=3   q=4 |
                %   |            |
                %   |  q=1   q=2 |
                %   x_1_________x_2 - --------x
                *(-1, 1) | (1, 1)
                *x       
                * | | |
                * | | |
                *-- | --------- | ----->  ξ
                * | | |
                * | | |
                *x - --------x
                *(-1, -1) | (1, -1)
                *
                *N₁ = (1 - ξ)(1 - η) / 4
                *N₂ = (1 + ξ)(1 - η) / 4
                *N₃ = (1 + ξ)(1 + η) / 4
                *N₄ = (1 - ξ)(1 + η) / 4
                * ∂N₁ / ∂ξ = - (1 - η) / 4, ∂N₁ / ∂η = - (1 - ξ) / 4
                * ∂N₂ / ∂ξ = + (1 - η) / 4, ∂N₂ / ∂η = - (1 + ξ) / 4
                * ∂N₃ / ∂ξ = + (1 + η) / 4, ∂N₃ / ∂η = + (1 + ξ) / 4
                * ∂N₄ / ∂ξ = - (1 + η) / 4, ∂N₄ / ∂η = + (1 - ξ) / 4
            """
            my_domain.nb_quad_points_per_pixel = 4
            # my_domain.nb_elements_per_pixel = 1
            my_domain.nb_nodes_per_pixel = 1  # x1 is the only pixel assigned node, x2 belongs to pixel +1
            my_domain.nb_unique_nodes_per_pixel = 1

            #  pixel sizes for better readability
            h_x = my_domain.pixel_size[0]
            h_y = my_domain.pixel_size[1]
            # nodal points offsets
            my_domain.offsets = np.array([[0, 0], [1, 0],
                                          [0, 1], [1, 1]])
            coord_helper = np.zeros(2)
            coord_helper[0] = -1. / (np.sqrt(3))
            coord_helper[1] = +1. / (np.sqrt(3))

            my_domain.quad_points_coord = np.zeros(
                [my_domain.domain_dimension, my_domain.nb_quad_points_per_pixel])

            my_domain.quad_points_coord[:, 0] = [coord_helper[0], coord_helper[0]]
            my_domain.quad_points_coord[:, 1] = [coord_helper[1], coord_helper[0]]
            my_domain.quad_points_coord[:, 2] = [coord_helper[0], coord_helper[1]]
            my_domain.quad_points_coord[:, 3] = [coord_helper[1], coord_helper[1]]

            my_domain.quad_points_coord_parametric = np.zeros(
                [my_domain.domain_dimension, my_domain.nb_quad_points_per_pixel])
            my_domain.quad_points_coord_parametric[:, 0] = [coord_helper[0], coord_helper[0]]
            my_domain.quad_points_coord_parametric[:, 1] = [coord_helper[1], coord_helper[0]]
            my_domain.quad_points_coord_parametric[:, 2] = [coord_helper[0], coord_helper[1]]
            my_domain.quad_points_coord_parametric[:, 3] = [coord_helper[1], coord_helper[1]]

            my_domain.B_gradient = np.zeros([my_domain.domain_dimension, 4,
                                             my_domain.nb_quad_points_per_pixel])
            # Jacobian matrix of transformation from iso-element to current size

            det_jacobian = h_x * h_y / 4

            inv_jacobian = np.array([[h_y / 2, 0], [0, h_x / 2]]) / det_jacobian

            # construction of B matrix
            for qp in range(0, my_domain.nb_quad_points_per_pixel):
                x_q = my_domain.quad_points_coord_parametric[:, qp]
                xi = x_q[0]
                eta = x_q[1]
                # @formatter:off
                my_domain.B_gradient[:,:, qp]=np.array( [[(eta - 1) / 4, (-eta + 1) / 4, (-eta - 1) / 4, (eta + 1) / 4],
                                                         [(xi  - 1) / 4, (-xi  - 1) / 4, (-xi  + 1) / 4, (xi  + 1) / 4]])

                my_domain.B_gradient[:,:, qp] = np.matmul(inv_jacobian,my_domain.B_gradient[:,:, qp])
                # @formatter:on

            my_domain.quadrature_weights = np.zeros([my_domain.nb_quad_points_per_pixel])
            my_domain.quadrature_weights[0] = h_x * h_y / 4
            my_domain.quadrature_weights[1] = h_x * h_y / 4
            my_domain.quadrature_weights[2] = h_x * h_y / 4
            my_domain.quadrature_weights[3] = h_x * h_y / 4

            # we have to recompute positions base on the size of current pixel !!!
            my_domain.quad_points_coord = np.zeros(
                [my_domain.domain_dimension, my_domain.nb_quad_points_per_pixel])

            my_domain.quad_points_coord[:, 0] = [h_x / 2 + h_x * coord_helper[0] / 2,
                                                 h_y / 2 + h_y * coord_helper[0] / 2]
            my_domain.quad_points_coord[:, 1] = [h_x / 2 + h_x * coord_helper[1] / 2,
                                                 h_y / 2 + h_y * coord_helper[0] / 2]
            my_domain.quad_points_coord[:, 2] = [h_x / 2 + h_x * coord_helper[0] / 2,
                                                 h_y / 2 + h_y * coord_helper[1] / 2]
            my_domain.quad_points_coord[:, 3] = [h_x / 2 + h_x * coord_helper[1] / 2,
                                                 h_y / 2 + h_y * coord_helper[1] / 2]
            # TODO find proper way of creating B matrices
            B_at_pixel_dnijkq = my_domain.B_gradient.reshape(my_domain.domain_dimension,
                                                             my_domain.nb_nodes_per_pixel,
                                                             *my_domain.domain_dimension * (2,),
                                                             my_domain.nb_quad_points_per_pixel)

            if my_domain.domain_dimension == 2:
                B_at_pixel_dnijkq = np.swapaxes(B_at_pixel_dnijkq, 2, 3)
            elif my_domain.domain_dimension == 3:
                B_at_pixel_dnijkq = np.swapaxes(B_at_pixel_dnijkq, 2, 4)
                warnings.warn('Swapaxes for 3D is not tested.')

            my_domain.B_grad_at_pixel_dqnijk = np.moveaxis(B_at_pixel_dnijkq, [-1], [1])

            return
        case 'trilinear_hexahedron':
            if my_domain.domain_dimension != 3:
                raise ValueError('Element_type {} is implemented only in 3D'.format(element_type))
            #
            #                    ζ
            #                    ^
            #         (-1,1,1)   |     (1,1,1)
            #                7---|------8
            #               /|   |     /|
            #              / |   |    / |
            #   (-1,-1,1) 5----------6  | (1,-1,1)
            #             |  |   |   |  |
            #             |  |   |   |  |
            #             |  |   +---|-------> ξ
            #             |  |  /    |  |
            #   (-1,1,-1) |  3-/-----|--4 (1,1,-1)
            #             | / /      | /
            #             |/ /       |/
            #             1-/--------2
            #   (-1,-1,-1) /        (1,-1,-1)
            #             /
            #            η
            # N(n,i,j,k)
            # N₁ = (1 - ξ) (1 - η) (1 - ζ) / 8  --   N [0,0,0,0]
            # N₂ = (1 + ξ) (1 - η) (1 - ζ) / 8  --   N [0,1,0,0]
            # N₃ = (1 - ξ) (1 + η) (1 - ζ) / 8  --   N [0,0,1,0]
            # N₄ = (1 + ξ) (1 + η) (1 - ζ) / 8  --   N [0,1,1,0]
            # N₅ = (1 - ξ) (1 - η) (1 + ζ) / 8  --   N [0,0,0,1]
            # N₆ = (1 + ξ) (1 - η) (1 + ζ) / 8  --   N [0,1,0,1]
            # N₇ = (1 - ξ) (1 + η) (1 + ζ) / 8  --   N [0,0,1,1]
            # N₈ = (1 + ξ) (1 + η) (1 + ζ) / 8  --   N [0,1,1,1]

            #
            # ∂N₁/∂ξ = - (1 - η) (1 - ζ) / 8  -- B [0,q,0,0,0,0]
            # ∂N₁/∂η = - (1 - ξ) (1 - ζ) / 8  -- B [1,q,0,0,0,0]
            # ∂N₁/∂ζ = - (1 - ξ) (1 - η) / 8  -- B [2,q,0,0,0,0]
            #
            # ∂N₂/∂ξ = + (1 - η) (1 - ζ) / 8  -- B [0,q,0,1,0,0]
            # ∂N₂/∂η = - (1 + ξ) (1 - ζ) / 8  -- B [1,q,0,1,0,0]
            # ∂N₂/∂ζ = - (1 + ξ) (1 - η) / 8  -- B [2,q,0,1,0,0]
            #
            # ∂N₃/∂ξ = - (1 + η) (1 - ζ) / 8  -- B [0,q,0,0,1,0]
            # ∂N₃/∂η = + (1 - ξ) (1 - ζ) / 8  -- B [1,q,0,0,1,0]
            # ∂N₃/∂ζ = - (1 - ξ) (1 + η) / 8  -- B [2,q,0,0,1,0]
            #
            # ∂N₄/∂ξ = + (1 + η) (1 - ζ) / 8  -- B [0,q,0,1,1,0]
            # ∂N₄/∂η = + (1 + ξ) (1 - ζ) / 8  -- B [1,q,0,1,1,0]
            # ∂N₄/∂ζ = - (1 + ξ) (1 + η) / 8  -- B [2,q,0,1,1,0]
            #
            # ∂N₅/∂ξ = - (1 - η) (1 + ζ) / 8  -- B [0,q,0,0,0,1]
            # ∂N₅/∂η = - (1 - ξ) (1 + ζ) / 8  -- B [1,q,0,0,0,1]
            # ∂N₅/∂ζ = + (1 - ξ) (1 - η) / 8  -- B [2,q,0,0,0,1]
            #
            # ∂N₆/∂ξ = + (1 - η) (1 + ζ) / 8  -- B [0,q,0,1,0,1]
            # ∂N₆/∂η = - (1 + ξ) (1 + ζ) / 8  -- B [1,q,0,1,0,1]
            # ∂N₆/∂ζ = + (1 + ξ) (1 - η) / 8  -- B [2,q,0,1,0,1]
            #
            # ∂N₇/∂ξ = - (1 + η) (1 + ζ) / 8  -- B [0,q,0,0,1,1]
            # ∂N₇/∂η = + (1 - ξ) (1 + ζ) / 8  -- B [1,q,0,0,1,1]
            # ∂N₇/∂ζ = + (1 - ξ) (1 + η) / 8  -- B [2,q,0,0,1,1]
            #
            # ∂N₈/∂ξ = + (1 + η) (1 + ζ) / 8  -- B [0,q,0,1,1,1]
            # ∂N₈/∂η = + (1 + ξ) (1 + ζ) / 8  -- B [1,q,0,1,1,1]
            # ∂N₈/∂ζ = + (1 + ξ) (1 + η) / 8  -- B [2,q,0,1,1,1]
            #
            # quad points:
            # ξ1  = -1/√3, η0 = -1/√3, ζ0 = -1/√3
            # ξ2  =  1/√3, η1 = -1/√3, ζ1 = -1/√3
            # ξ3  = -1/√3, η2 =  1/√3, ζ2 = -1/√3
            # ξ4, =  1/√3, η3 =  1/√3, ζ3 = -1/√3
            # ξ5  = -1/√3, η4 = -1/√3, ζ4 =  1/√3
            # ξ6  =  1/√3, η5 = -1/√3, ζ5 =  1/√3
            # ξ7  = -1/√3, η6 =  1/√3, ζ6 =  1/√3
            # Voxel discretization setting
            my_domain.nb_nodes_per_pixel = 1  # left bottom corner belong to voxel.
            my_domain.nb_unique_nodes_per_pixel = 1
            my_domain.nb_quad_points_per_pixel = 8

            #  pixel sizes for better readability
            del_x = my_domain.pixel_size[0]
            del_y = my_domain.pixel_size[1]
            del_z = my_domain.pixel_size[2]

            """  Structure of B matrix: 
                B(d,q,n,i,j,k) --> is B matrix evaluate gradient at point q in 
                 B(d,q,n,i,j,k)  is ∂φ_nijk/∂x_d   at (q)
                
                
                B(:,:,q) --> is B matrix evaluate gradient at point q in  element e
                B(:,:,q) has size [dim,nb_of_nodes/basis_functions] 
                                      (usually 4 in 2D and 8 in 3D)
                 % B(:,:,q) = [ ∂φ_1/∂x_1  ∂φ_2/∂x_1  ∂φ_3/∂x_1 ... .... ∂φ_8/∂x_1 ;
            %                   ∂φ_1/∂x_2  ∂φ_2/∂x_2  ∂φ_3/∂x_2 ... ...  ∂φ_8/∂x_2 ;
            %                   ∂φ_1/∂x_3  ∂φ_2/∂x_3  ∂φ_3/∂x_3 ... ...  ∂φ_8/∂x_3 ]   at (q)
            """

            # quadrature points : coordinates
            my_domain.quad_points_coord = np.zeros(
                [my_domain.domain_dimension, my_domain.nb_quad_points_per_pixel])

            coord_helper = np.zeros(2)
            coord_helper[0] = -1. / (np.sqrt(3))
            coord_helper[1] = +1. / (np.sqrt(3))

            # quadrature points    # TODO This hold for prototypical element      !!!
            # TODO MAKE clear how to generate B matrices
            my_domain.quad_points_coord[:, 0] = [coord_helper[0], coord_helper[0], coord_helper[0]]
            my_domain.quad_points_coord[:, 1] = [coord_helper[1], coord_helper[0], coord_helper[0]]
            my_domain.quad_points_coord[:, 2] = [coord_helper[0], coord_helper[1], coord_helper[0]]
            my_domain.quad_points_coord[:, 3] = [coord_helper[1], coord_helper[1], coord_helper[0]]
            my_domain.quad_points_coord[:, 4] = [coord_helper[0], coord_helper[0], coord_helper[1]]
            my_domain.quad_points_coord[:, 5] = [coord_helper[1], coord_helper[0], coord_helper[1]]
            my_domain.quad_points_coord[:, 6] = [coord_helper[0], coord_helper[1], coord_helper[1]]
            my_domain.quad_points_coord[:, 7] = [coord_helper[1], coord_helper[1], coord_helper[1]]

            # quadrature points : weights

            my_domain.quadrature_weights = np.zeros([my_domain.nb_quad_points_per_pixel])
            my_domain.quadrature_weights[:] = del_x * del_y * del_z / 8

            # Jabobian
            jacoby_matrix = np.array([[(del_x / 2), 0, 0],
                                      [0, (del_y / 2), 0],
                                      [0, 0, (del_z / 2)]])

            det_jacobian = np.linalg.det(jacoby_matrix)
            inv_jacobian = np.linalg.inv(jacoby_matrix)

            # construction of B matrix
            dim = my_domain.domain_dimension
            B_dqnijk = np.zeros(
                [dim, my_domain.nb_quad_points_per_pixel, my_domain.nb_unique_nodes_per_pixel, 2, 2, 2])
            for quad_point in range(0, my_domain.nb_quad_points_per_pixel):
                x_q = my_domain.quad_points_coord[:, quad_point]
                xi = x_q[0]
                eta = x_q[1]
                zeta = x_q[2]
                # this part have to be hard coded
                # @formatter:off

                B_dqnijk[:, quad_point, 0, 0, 0, 0] = np.array([- (1 - eta) * (1 - zeta) / 8,
                                                                - (1 -  xi) * (1 - zeta) / 8,
                                                                - (1 -  xi) * (1 -  eta) / 8])

                B_dqnijk[:, quad_point, 0, 1, 0, 0] = np.array([+ (1 - eta) * (1 - zeta) / 8,
                                                                - (1 +  xi) * (1 - zeta) / 8,
                                                                - (1 +  xi) * (1 -  eta) / 8])

                B_dqnijk[:, quad_point, 0, 0, 1, 0] = np.array([- (1 + eta) * (1 - zeta) / 8,
                                                                + (1 -  xi) * (1 - zeta) / 8,
                                                                - (1 -  xi) * (1 +  eta) / 8])

                B_dqnijk[:, quad_point, 0, 1, 1, 0] = np.array([+ (1 + eta) * (1 - zeta) / 8,
                                                                + (1 +  xi) * (1 - zeta) / 8,
                                                                - (1 +  xi) * (1 +  eta) / 8])

                B_dqnijk[:, quad_point, 0, 0, 0, 1] = np.array([- (1 - eta) * (1 + zeta) / 8,
                                                                - (1 -  xi) * (1 + zeta) / 8,
                                                                + (1 -  xi) * (1 -  eta) / 8])

                B_dqnijk[:, quad_point, 0, 1, 0, 1] = np.array([+ (1 - eta) * (1 + zeta) / 8,
                                                                - (1 +  xi) * (1 + zeta) / 8,
                                                                + (1 +  xi) * (1 -  eta) / 8])

                B_dqnijk[:, quad_point, 0, 0, 1, 1] = np.array([- (1 + eta) * (1 + zeta) / 8,
                                                                + (1 -  xi) * (1 + zeta) / 8,
                                                                + (1 -  xi) * (1 +  eta) / 8])

                B_dqnijk[:, quad_point, 0, 1, 1, 1] = np.array([+ (1 + eta) * (1 + zeta) / 8,
                                                                + (1 +  xi) * (1 + zeta) / 8,
                                                                + (1 +  xi) * (1 +  eta) / 8])


                # @formatter:on
            # multiplication with inverse of jacobian
            B_dqnijk = np.einsum('dt,tqnijk->dqnijk', inv_jacobian, B_dqnijk)

            my_domain.B_grad_at_pixel_dqnijk = B_dqnijk

            # recompute quad points coordinates
            # @formatter:off
            my_domain.quad_points_coord[:, 0] = [del_x/2 + del_x*coord_helper[0]/2,
                                                 del_y/2 + del_y*coord_helper[0]/2,
                                                 del_z/2 + del_z*coord_helper[0]/2]

            my_domain.quad_points_coord[:, 1] = [del_x/2 + del_x*coord_helper[1]/2,
                                                 del_y/2 + del_y*coord_helper[0]/2,
                                                 del_z/2 + del_z*coord_helper[0]/2]

            my_domain.quad_points_coord[:, 2] = [del_x/2 + del_x*coord_helper[0]/2,
                                                 del_y/2 + del_y*coord_helper[1]/2,
                                                 del_z/2 + del_z*coord_helper[0]/2]

            my_domain.quad_points_coord[:, 3] = [del_x/2 + del_x*coord_helper[1]/2,
                                                 del_y/2 + del_y*coord_helper[1]/2,
                                                 del_z/2 + del_z*coord_helper[0]/2]

            my_domain.quad_points_coord[:, 4] = [del_x/2 + del_x*coord_helper[0]/2,
                                                 del_y/2 + del_y*coord_helper[0]/2,
                                                 del_z/2 + del_z*coord_helper[1]/2]

            my_domain.quad_points_coord[:, 5] = [del_x/2 + del_x*coord_helper[1]/2,
                                                 del_y/2 + del_y*coord_helper[0]/2,
                                                 del_z/2 + del_z*coord_helper[1]/2]

            my_domain.quad_points_coord[:, 6] = [del_x/2 + del_x*coord_helper[0]/2,
                                                 del_y/2 + del_y*coord_helper[1]/2,
                                                 del_z/2 + del_z*coord_helper[1]/2]

            my_domain.quad_points_coord[:, 7] = [del_x/2 + del_x*coord_helper[1]/2,
                                                 del_y/2 + del_y*coord_helper[1]/2,
                                                 del_z/2 + del_z*coord_helper[1]/2]



        case _:

            raise ValueError('Element type {} is not implemented yet'.format(element_type))
