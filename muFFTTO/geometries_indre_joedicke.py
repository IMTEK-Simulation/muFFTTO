"""
@file   Geometries.py

@author Indre  Joedicke <indre.joedicke@imtek.uni-freiburg.de>

@date   18 Jul 2023

@brief  Functions for defining different geometries

Copyright © 2018 Till Junge

µSpectre is free software; you can redistribute it and/or
modify it under the terms of the GNU Lesser General Public License as
published by the Free Software Foundation, either version 3, or (at
your option) any later version.

µSpectre is distributed in the hope that it will be useful, but
WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public License
along with µSpectre; see the file COPYING. If not, write to the
Free Software Foundation, Inc., 59 Temple Place - Suite 330,
Boston, MA 02111-1307, USA.

Additional permission under GNU GPL version 3 section 7

If you modify this Program, or any covered work, by linking or combining it
with proprietary FFT implementations or numerical libraries, containing parts
covered by the terms of those libraries' licenses, the licensors of this
Program grant you additional permission to convey the resulting work.
"""
import numpy as np

### ----- Minimal structure ----- ###
def minimal_structure(nb_grid_pts = [3, 3, 2]):
    """
    Define the easiest possible structure while breaking the
    periodic boundary condition in x- and y-direction.

    Arguments
    ---------
    nb_grid_pts: list of 3 ints
                 Number of grid pts in each direction.
                 Default is [3, 3, 2].
    Returns
    -------
    mask: np.ndarray of floats
          Representation of the geometry with 0 corresponding
          to void and 1 corresponding to material.
    """
    mask = np.ones(nb_grid_pts)
    mask[0, :, :] = 0
    mask[-1, :, :] = 0
    mask[:, 0, :] = 0
    mask[:, -1, :] = 0
    return mask

def minimal_structure_much_void(nb_grid_pts = [3, 3, 2]):
    """
    Define the easiest possible structure while breaking the
    periodic boundary condition in x- and y-direction. Only
    a single line of voxels is material, the rest is void.

    Arguments
    ---------
    nb_grid_pts: list of 3 ints
                 Number of grid pts in each direction.
                 Default is [3, 3, 2].
    Returns
    -------
    mask: np.ndarray of floats
          Representation of the geometry with 0 corresponding
          to void and 1 corresponding to material.
    """
    mask = np.zeros(nb_grid_pts)
    mask[nb_grid_pts[0]//2, nb_grid_pts[1]//2, :] = 1
    return mask

### ----- Define cylinder ----- ###
def cylinder(nb_grid_pts, lengths, radius):
    """
    Define a cylinder. The axis of the cylinder is in
    the middle of the unit cell.

    Arguments
    ---------
    nb_grid_pts: list of 3 ints
                 Number of grid pts in each direction.
                 Default is [3, 3, 2].
    lengths: list of 3 floats
             Lengths of unit cell in each direction.
    radius: float
            Radius of the cylinder.
    Returns
    -------
    mask: np.ndarray of floats
          Representation of the geometry with 0 corresponding
          to void and 1 corresponding to material.
    """
    # Parameters
    hx = lengths[0] / nb_grid_pts[0]
    hy = lengths[1] / nb_grid_pts[1]
    hz = lengths[2] / nb_grid_pts[2]

    x = np.arange(nb_grid_pts[0]) * hx # x-coordinate of voxel
    y = np.arange(nb_grid_pts[1]) * hy # y-coordinate of voxel

    x_axis = lengths[0] / 2
    y_axis = lengths[1] / 2

    # Check the radius
    if (radius > lengths[0]/2 - hx) or (radius > lengths[1]/2 - hy):
        message = 'Attention: The diameter of the cylinder is larger '
        message += 'then the unit cell. THE PERIODIC BOUNDARIES ARE '
        message += 'NOT BROKEN.'
        print(message)

    # x-y-coordinates of the centers of the 5 tetradras
    X, Y = np.meshgrid(x, y)
    X += 0.5 * hx
    Y += 0.5 * hy
    X = X.T
    Y = Y.T

    # Material
    dist = (X - x_axis) ** 2 + (Y - y_axis) ** 2
    dist = dist ** 0.5
    mask = np.ones(nb_grid_pts)
    mask[dist > radius] = 0
    return mask


### ----- Define hollow cylinder ----- ###
def hollow_cylinder(nb_grid_pts, lengths, radius, thickness):
    """
    Define a hollow cylinder. The axis of the cylinder is in
    the middle of the unit cell.

    Arguments
    ---------
    nb_grid_pts: list of 3 ints
                 Number of grid pts in each direction.
                 Default is [3, 3, 2].
    lengths: list of 3 floats
             Lengths of unit cell in each direction.
    radius: float
            Outer radius of the cylinder.
    thickness: float
               Thickness of the cylinder.
    Returns
    -------
    mask: np.ndarray of floats
          Representation of the geometry with 0 corresponding
          to void and 1 corresponding to material.
    """
    # Parameters
    hx = lengths[0] / nb_grid_pts[0]
    hy = lengths[1] / nb_grid_pts[1]
    hz = lengths[2] / nb_grid_pts[2]
    x = np.arange(nb_grid_pts[0]) * hx # x-coordinate of voxel
    y = np.arange(nb_grid_pts[1]) * hy # y-coordinate of voxel

    x_axis = lengths[0] / 2
    y_axis = lengths[1] / 2

    # Check the parameters
    if (radius > lengths[0]/2 - hx) or (radius > lengths[1]/2 - hy):
        message = 'Attention: The diameter of the cylinder is larger '
        message += 'then the unit cell. THE PERIODIC BOUNDARIES ARE '
        message += 'NOT BROKEN.'
        print(message)
    if (hx > thickness) or (hy > thickness):
        message = 'Error: The pixels are larger than the thickness '
        message += 'of the hollow cylinder. Please refine the '
        message += 'discretization.'
        assert hx > thickness, message
        assert hy > thickness, message
    if (3*hx > thickness) or (3*hy > thickness):
        message = 'Error: The thickness is represented by less then 3 pixels.'
        message += ' Please consider refining the discretization.'

    # x-y-coordinates of the centers of the 5 tetradras
    X, Y = np.meshgrid(x, y)
    X += 0.5 * hx
    Y += 0.5 * hy
    X = X.T
    Y = Y.T

    # Material
    dist = (X - x_axis) ** 2 + (Y - y_axis) ** 2
    dist = dist ** 0.5
    mask = np.zeros(nb_grid_pts)
    material = np.logical_and(dist < radius, dist > radius - thickness)
    mask[material] = 1

    return mask

### ----- Beam with rectangular section ----- ###
def rectangular_beam(nb_grid_pts, lengths, Lx, Ly):
    """
    Define a beam with rectangular cross-section.

    Arguments
    ---------
    nb_grid_pts: list of 3 ints
                 Number of grid pts in each direction.
                 Default is [3, 3, 2].
    lengths: list of 3 floats
             Lengths of unit cell in each direction.
    Lx: float
        Length of section in x-direction.
    Ly: float
        Length of section in y-direction.
    Returns
    -------
    mask: np.ndarray of floats
          Representation of the geometry with 0 corresponding
          to void and 1 corresponding to material.
    """
    # Helper definitions
    hx = lengths[0] / nb_grid_pts[0]
    hy = lengths[1] / nb_grid_pts[1]
    hz = lengths[2] / nb_grid_pts[2]

    L_void_x = (lengths[0] - Lx) / 2
    L_void_y = (lengths[0] - Ly) / 2

    # Material
    mask = np.ones(nb_grid_pts)
    helper = round(L_void_x / hx)
    mask[0:helper, :, :] = 0
    mask[-helper:, :, :] = 0
    helper = round(L_void_y / hy)
    mask[:, 0:helper, :] = 0
    mask[:, -helper:, :] = 0

    return mask

### ----- Define chiral metamaterial ----- ###
def chiral_metamaterial(nb_grid_pts, lengths, radius, thickness, alpha=0):
    """
    Define a (relatively simple) chiral metamaterial. It consists of two
    rings connected by four beams. Each beam is inclined by an angle alpha.

    Arguments
    ---------
    nb_grid_pts: list of 3 ints
                 Number of grid pts in each direction.
    lengths: list of 3 floats
             Lengths of unit cell in each direction.
    radius: float
            Radius of the cirlces
    thickness: float
               Thickness of the cirles and beams.
    alpha: float
           Angle at wich the connecting beams are inclined.
           Default is 0, meaning the beams are vertical.
    Returns
    -------
    mask: np.ndarray of floats
          Representation of the geometry with 0 corresponding
          to void and 1 corresponding to material.
    """
    # Parameters
    hx = lengths[0] / nb_grid_pts[0]
    hy = lengths[1] / nb_grid_pts[1]
    hz = lengths[2] / nb_grid_pts[2]
    x_axis = nb_grid_pts[0] // 2 * hx # + 0.5 * hx
    y_axis = nb_grid_pts[1] // 2 * hy # + 0.5 * hy
    thickness_x = round(thickness / hx)
    thickness_y = round(thickness / hy)
    thickness_z = round(thickness / hz)

    # Check wether the parameters are meaningful
    if (radius > lengths[0]/2 - hx) or (radius > lengths[1]/2 - hy):
        message = 'Attention: The diameter of the cylinder is larger '
        message += 'then the unit cell. THE PERIODIC BOUNDARIES ARE '
        message += 'NOT BROKEN.'
        print(message)
    if (radius < thickness + hx) or (radius < thickness + hy):
        message = 'Error: The radius is too small.'
        assert radius < thicknss + hx, message
        assert radius < thickness + hy, message
    if (hx > thickness) or (hy > thickness) or (hy > thickness):
        message = 'Error: The pixels are larger than the thickness.'
        message += ' Please refine the discretization.'
        assert hx > thickness, message
        assert hy > thickness, message
        assert hz > thickness, message
    if (3*hx > thickness) or (3*hy > thickness):
        message = 'Attention: The thickness is represented by less then 3 pixels.'
        message += ' Please consider refining the discretization.'
        print(message)
    helper = np.arctan((radius - thickness / 2) / (lengths[2] - 2 * thickness))
    if (alpha > helper) or (alpha < - helper):
        message = f'Error: The angle must lie between {-helper} and {helper}'
        message += f'but it is {alpha}.'
        assert alpha > helper, message
        assert alpha < - helper, message

    # Circles at top and bottom
    mask = np.zeros(nb_grid_pts)
    for ind_x in range(nb_grid_pts[0]):
        for ind_y in range(nb_grid_pts[1]):
            x = ind_x * hx + 0.5 * hx
            y = ind_y * hy + 0.5 * hy
            dist = np.sqrt((x - x_axis)**2 + (y - y_axis)**2)
            if (dist < radius) and (dist >= radius - thickness):
                mask[ind_x, ind_y, 0:thickness_z] = 1
                mask[ind_x, ind_y, nb_grid_pts[2]-thickness_z:] = 1

    # Step in x- and y-direction of connecting beams
    step_1 = hz * np.tan(alpha)
    helper = (lengths[2] - 2 * thickness) * np.tan(alpha)
    helper = radius - thickness / 2 -\
        ((radius - thickness / 2)**2 - helper**2) ** 0.5
    step_2 = helper / (lengths[2] - 2 * thickness) * hz

    # Starting points for connecting beams
    start1_x = nb_grid_pts[0] // 2 - thickness_x // 2
    start2_x = round((lengths[0] / 2 + radius - thickness) / hx)
    start3_x = nb_grid_pts[0] // 2 - thickness_x // 2
    start4_x = round((lengths[0] / 2 - radius) / hx)
    start1_y = round((lengths[1] / 2 - radius) / hy)
    start2_y = nb_grid_pts[1] // 2 - thickness_y // 2
    start3_y = round((lengths[1] / 2 + radius - thickness) / hy)
    start4_y = nb_grid_pts[1] // 2 - thickness_y // 2

    # Connecting beams
    for ind_z in range(thickness_z, nb_grid_pts[2]-thickness_z):
        help_x1 = round((ind_z - thickness_z) * step_1 / hx)
        help_y1 = round((ind_z - thickness_z) * step_1 / hy)
        help_x2 = round((ind_z - thickness_z) * step_2 / hx)
        help_y2 = round((ind_z - thickness_z) * step_2 / hy)

        # 1. beam
        start_x = start1_x + help_x1
        start_y = start1_y + help_y2
        mask[start_x:start_x+thickness_x, start_y:start_y+thickness_y, ind_z] = 1
        # 2. beam
        start_x = start2_x - help_x2
        start_y = start2_y + help_y1
        mask[start_x:start_x+thickness_x, start_y:start_y+thickness_y, ind_z] = 1
        # 3. beam
        start_x = start3_x - help_x1
        start_y = start3_y - help_y2
        mask[start_x:start_x+thickness_x, start_y:start_y+thickness_y, ind_z] = 1
        # 4. beam
        start_x = start4_x + help_x2
        start_y = start4_y - help_y1
        mask[start_x:start_x+thickness_x, start_y:start_y+thickness_y, ind_z] = 1

    return mask


### ----- Define chiral metamaterial 2 ----- ###
def chiral_metamaterial_2(nb_grid_pts, lengths, radius_out, radius_inn,
                          thickness, alpha=0):
    """
    Define a (more complex) chiral metamaterial. It consists of a beam on each
    face of the RVE connected to the edges by four beams. The beams are
    inclined with an angle alpha.

    Arguments
    ---------
    nb_grid_pts: list of 3 ints
                 Number of grid pts in each direction.
    lengths: list of 3 floats
             Lengths of unit cell in each direction. Note that the size of
             the RVE corresponds to lengths[2], so that lengths[0] and
             lengths[1] must be larger than lengths[2] to break the periodicity.
    radius_out: float
                Outer radius of the circles.
    radius_inn: float
                Inner radius of the circles.
    thickness: float
               Thickness of the connecting beams.
    alpha: float
           Angle at wich the connecting beams are inclined.
           Default is 0.
    Returns
    -------
    mask: np.ndarray of floats
          Representation of the geometry with 0 corresponding
          to void and 1 corresponding to material.
    """
    ### ----- Parameters ----- ###
    # Definition of helper paramaters
    hx = lengths[0] / nb_grid_pts[0]
    hy = lengths[1] / nb_grid_pts[1]
    hz = lengths[2] / nb_grid_pts[2]
    thickness_x = round(thickness / hx)
    thickness_y = round(thickness / hy)
    thickness_z = round(thickness / hz)
    a = lengths[2]
    b = 1.5 * thickness
    boundary = (lengths[0] - a) / 2

    # Check wether the parameters are meaningful
    if (radius_out > a/2 - hx) or (radius_out > a/2 - hy):
        message = 'ATTENTION: The diameter of the outer circle is larger '
        message += 'then the unit cell.'
        print(message)
    if (radius_inn < thickness + hx) or (radius_inn < thickness + hy):
        message = 'ERROR: The inner radius is too small.'
        assert radius_inn < thickness + hx, message
        assert radius_inn < thickness + hy, message
    if (hx > thickness) or (hy > thickness) or (hy > thickness):
        message = 'ERROR: The pixels are larger than the thickness.'
        message += ' Please refine the discretization.'
        assert hx > thickness, message
        assert hy > thickness, message
        assert hz > thickness, message
    if (3*hx > thickness) or (3*hy > thickness):
        message = 'ATTENTION: The thickness is represented by less then 3 pixels.'
        message += ' Please consider refining the discretization.'
        print(message)
    # TODO: angle too large?
    message = 'lengths[0] is not large enough to break the periodicity.'
    assert lengths[0] > a, message
    message = 'lengths[1] is not large enough to break the periodicity.'
    assert lengths[1] > a, message

    ### ----- Define the four corners ----- ###
    mask = np.zeros(nb_grid_pts)
    bx = round(b / hx)
    by = round(b / hy)
    bz = round(b / hz)
    boundary_x = round(boundary / hx)
    boundary_y = round(boundary / hy)
    mask[boundary_x:boundary_x+bx, boundary_y:boundary_y+by, 0:bz] = 1
    mask[-bx-boundary_x:-boundary_x, boundary_y:boundary_y+by, 0:bz] = 1
    mask[boundary_x:boundary_x+bx, -by-boundary_y:-boundary_y, 0:bz] = 1
    mask[-bx-boundary_x:-boundary_x, -by-boundary_y:-boundary_y, 0:bz] = 1
    mask[boundary_x:boundary_x+bx, boundary_y:boundary_y+by, -bz:] = 1
    mask[-bx-boundary_x:-boundary_x, boundary_y:boundary_y+by, -bz:] = 1
    mask[boundary_x:boundary_x+bx, -by-boundary_y:-boundary_y, -bz:] = 1
    mask[-bx-boundary_x:-boundary_x, -by-boundary_y:-boundary_y, -bz:] = 1

    ### ----- Define the circle at each face ----- ###
    # Coordinates of the centers of the 5 tetradras
    # Format X[i, j, k] with i: x-coord, j: y-coord, k: z-coord
    x = np.arange(nb_grid_pts[0]) * hx # x-coordinate of voxel
    y = np.arange(nb_grid_pts[1]) * hy # y-coordinate of voxel
    z = np.arange(nb_grid_pts[2]) * hz # y-coordinate of voxel
    X, Y, Z = np.meshgrid(x, y, z)
    X = X.transpose((1, 0, 2)) + 0.5 * hx
    Y = Y.transpose((1, 0, 2)) + 0.5 * hy
    Z = Z.transpose((1, 0, 2)) + 0.5 * hz

    # Circles in xz-planes
    x0 = lengths[0] / 2
    z0 = lengths[2] / 2
    dist = (X[:, 0, :] - x0) ** 2 + (Z[:, 0, :] - z0) ** 2
    dist = dist ** 0.5
    material = np.logical_and(dist < radius_out, dist > radius_inn)
    material = np.expand_dims(material, axis=1)
    material = np.broadcast_to(material, nb_grid_pts).copy()
    material[:, 0:boundary_y, :] = False
    material[:, -boundary_y:, :] = False
    material[:, boundary_y+thickness_y:-thickness_y-boundary_y, :] = False
    mask[material] = 1

    # Circles in yz-planes
    y0 = lengths[1] / 2
    z0 = lengths[2] / 2
    dist = (Y[0, :, :] - y0) ** 2 + (Z[0, :, :] - z0) ** 2
    dist = dist ** 0.5
    material = np.logical_and(dist < radius_out, dist > radius_inn)
    material = np.expand_dims(material, axis=0)
    material = np.broadcast_to(material, nb_grid_pts).copy()
    material[0:boundary_x, :, :] = False
    material[-boundary_x:, :, :] = False
    material[boundary_x+thickness_x:-thickness_x-boundary_x, :, :] = False
    mask[material] = 1

    # Circles in xy-planes
    x0 = lengths[0] / 2
    y0 = lengths[1] / 2
    dist = (X[:, :, 0] - x0) ** 2 + (Y[:, :, 0] - y0) ** 2
    dist = dist ** 0.5
    material = np.logical_and(dist < radius_out, dist > radius_inn)
    material = np.expand_dims(material, axis=2)
    material = np.broadcast_to(material, nb_grid_pts).copy()
    material[:, :, thickness_z//2:-thickness_z//2] = False
    mask[material] = 1

    ### ----- Define the connecting beams ----- ###
    # Angle between beam and coordinate system
    beta = np.pi / 4 - alpha
    step_exact = hz * np.tan(beta)

    # End of beam is determined by quadratic equation
    helper_a = 1 + np.tan(beta) ** 2
    helper_b = - 2 * (np.tan(beta) + 1) * (a/2 - b/2)
    helper_c = 2 * (a/2 - b/2) ** 2 - (radius_out/2 + radius_inn/2) ** 2
    helper = helper_b ** 2 - 4 * helper_a * helper_c
    # Test wether the angle is possible
    message = 'ERROR: The angle of the material is too large.'
    assert helper > 0, message
    stop = (- helper_b - helper ** 0.5 ) / 2 / helper_a

    # Beams in xz-planes
    start_x = round((boundary + b/2) / hx)
    stop_x = round((boundary + stop + b/2) / hx)
    start_y = boundary_y
    stop_y = boundary_y + thickness_y
    start_z = round(b / 2 / hz)
    stop_z = round((b/2 + stop) / hz)
    t_half_x = round(thickness / 2 / hx)
    t_half_y = round(thickness / 2 / hy)
    t_half_z = round(thickness / 2 / hz)
    for ind_x in range(start_x, stop_x):
        step = round((ind_x - start_x) * step_exact / hz)
        mask[ind_x - t_half_x + 1 : ind_x + t_half_x + 1,
             start_y : stop_y,
             start_z + step - t_half_x : start_z + step + t_half_x] = 1
        mask[ind_x - t_half_x + 1 : ind_x + t_half_x + 1,
             -stop_y : -start_y,
             start_z + step - t_half_z : start_z + step + t_half_z] = 1
        helper = -start_z - step + t_half_z
        if helper > -1:
            mask[-ind_x - t_half_x - 1: -ind_x + t_half_x - 1,
                 start_y : stop_y,
                 -start_z - step - t_half_z : ] = 1
            mask[-ind_x - t_half_x - 1: -ind_x + t_half_x - 1,
                 -stop_y : -start_y,
                 -start_z - step - t_half_z : ] = 1
        else:
            mask[-ind_x -t_half_x - 1: -ind_x + t_half_x - 1,
                 start_y : stop_y,
                 -start_z - step - t_half_z : helper] = 1
            mask[-ind_x - t_half_x - 1: -ind_x + t_half_x - 1,
                 -stop_y : -start_y,
                 -start_z - step - t_half_z : helper] = 1
    for ind_z in range(start_z, stop_z):
        step = round((ind_z - start_z) * step_exact / hx)
        mask[-start_x - step - t_half_x : -start_x - step + t_half_x,
             start_y : stop_y,
             ind_z - t_half_z : ind_z + t_half_z] = 1
        mask[start_x + step - t_half_x : start_x + step + t_half_x,
             start_y : stop_y,
             -ind_z - t_half_z : -ind_z + t_half_z] = 1
        mask[-start_x - step - t_half_x : -start_x - step + t_half_x,
             -stop_y : -start_y,
             ind_z - t_half_z : ind_z + t_half_z] = 1
        mask[start_x + step - t_half_x : start_x + step + t_half_x,
             -stop_y : -start_y,
             -ind_z - t_half_z : -ind_z + t_half_z] = 1

    # Beams in yz-planes
    start_x = boundary_x
    stop_x = boundary_x + thickness_x
    start_y = round((boundary + b/2) / hy)
    stop_y = round((boundary + stop + b/2) / hy)
    for ind_y in range(start_y, stop_y):
        step = round((ind_y - start_y) * step_exact / hz)
        mask[start_x : stop_x,
             ind_y - t_half_y + 1 : ind_y + t_half_y + 1,
             start_z + step - t_half_z : start_z + step + t_half_z] = 1
        mask[-stop_x : -start_x,
             ind_y - t_half_y + 1 : ind_y + t_half_y + 1,
             start_z + step - t_half_z : start_z + step + t_half_z] = 1
        helper = -start_z - step + t_half_z
        if helper > -1:
            mask[start_x : stop_x,
                 -ind_y - t_half_y - 1: -ind_y + t_half_y - 1,
                 -start_z - step - t_half_z : ] = 1
            mask[-stop_x : -start_x,
                 -ind_y - t_half_y - 1: -ind_y + t_half_y - 1,
                 -start_z - step - t_half_z : ] = 1
        else:
            mask[start_x : stop_x,
                 -ind_y - t_half_y - 1: -ind_y + t_half_y - 1,
                 -start_z - step - t_half_z : helper] = 1
            mask[-stop_x : -start_x,
                 -ind_y - t_half_y - 1: -ind_y + t_half_y - 1,
                 -start_z - step - t_half_z : helper] = 1
    for ind_z in range(start_z, stop_z):
        step = round((ind_z - start_z) * step_exact / hy)
        mask[start_x : stop_x,
             -start_y - step - t_half_y : -start_y - step + t_half_y,
             ind_z - t_half_z : ind_z + t_half_z] = 1
        mask[start_x : stop_x,
             start_y + step - t_half_y : start_y + step + t_half_y,
             -ind_z - t_half_z : -ind_z + t_half_z] = 1
        mask[-stop_x : -start_x,
             -start_y - step - t_half_y : -start_y - step + t_half_y,
             ind_z - t_half_z : ind_z + t_half_z] = 1
        mask[-stop_x : -start_x,
             start_y + step - t_half_y : start_y + step + t_half_y,
             -ind_z - t_half_z : -ind_z + t_half_z] = 1

    # Beams in xz-planes
    start_x = round((boundary + b/2) / hx)
    stop_x = round((boundary + stop + b/2) / hx)
    start_y = round((boundary + b/2) / hy)
    stop_y = round((boundary + stop + b/2) / hy)
    stop_z = round(thickness / 2 / hz)
    for ind_x in range(start_x, stop_x):
        step = round((ind_x - start_x) * step_exact / hy)
        mask[ind_x - t_half_x + 1 : ind_x + t_half_x + 1,
             start_y + step - t_half_y : start_y + step + t_half_y,
             : stop_z] = 1
        mask[ind_x - t_half_x + 1 : ind_x + t_half_x + 1,
             start_y + step - t_half_y : start_y + step + t_half_y,
             -stop_z :] = 1
        helper = -start_y - step + t_half_y
        if helper > -1:
            mask[-ind_x - t_half_x - 1: -ind_x + t_half_x - 1,
                 -start_y - step - t_half_y : ,
                 : stop_z] = 1
            mask[-ind_x - t_half_x - 1: -ind_x + t_half_x - 1,
                 -start_y - step - t_half_y : ,
                 -stop_z :] = 1
        else:
            mask[-ind_x - t_half_x - 1: -ind_x + t_half_x - 1,
                 -start_y- step - t_half_y : helper,
                 : stop_z] = 1
            mask[-ind_x - t_half_x - 1: -ind_x + t_half_x - 1,
                 -start_y- step - t_half_y : helper,
                 -stop_z :] = 1
    for ind_y in range(start_y, stop_y):
        step = round((ind_y - start_y) * step_exact / hx)
        mask[-start_x - step - t_half_x : -start_x - step + t_half_x,
             ind_y - t_half_y : ind_y + t_half_y,
             : stop_z] = 1
        mask[start_x + step - t_half_x : start_x + step + t_half_x,
             -ind_y - t_half_y : -ind_y + t_half_y,
             : stop_z] = 1
        mask[-start_x - step - t_half_x : -start_x - step + t_half_x,
             ind_y - t_half_y : ind_y + t_half_y,
             -stop_z :] = 1
        mask[start_x + step - t_half_x : start_x + step + t_half_x,
             -ind_y - t_half_y : -ind_y + t_half_y,
             -stop_z :] = 1

    return mask
