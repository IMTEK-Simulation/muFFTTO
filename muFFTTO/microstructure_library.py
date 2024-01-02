import warnings
import numpy as np


def get_geometry(nb_voxels,
                 microstructure_name='random_distribution',
                 parameter=None):
    if not microstructure_name in ['random_distribution', 'geometry_1_3D', 'geometry_2_3D', 'geometry_3_3D',
                                   'geometry_4_3D', 'geometry_5_3D', 'geometry_6_3D', 'geometry_7_3D', 'geometry_8_3D']:
        raise ValueError('Unrecognised microstructure_name {}'.format(microstructure_name))
    # if not nb_voxels[0] > 19 and nb_voxels[1] > 19 and nb_voxels[2] > 19 and nb_voxels[0]//5!=0 and nb_voxels[1]//5!=0 and nb_voxels[2]//5!=0:
    #     raise ValueError('Microstructure_name {} is implemented only when Size of any dimension is more than 10 and it is multiple of 5'.format(microstructure_name))
    if not (nb_voxels[0] > 19 and nb_voxels[1] > 19 and nb_voxels[2] > 19):
        # and nb_voxels[0] % 5 == 0 and nb_voxels[1] % 5 == 0 and nb_voxels[2] % 5 == 0
        raise ValueError('Microstructure_name {} is implemented only when Size '
                         'of any dimension is more than 19 and it is a multiple of 5'.format(
            microstructure_name))

    match microstructure_name:
        case 'random_distribution':

            phase_field = np.random.rand(*nb_voxels)

        case 'geometry_1_3D':  # TODO[Bharat] : this is an template for you
            # I will create a Cube with the Circle removed from each face
            if nb_voxels.size != 3:
                raise ValueError('Microstructure_name {} is implemented only in 3D'.format(microstructure_name))
            # here should come your code
            if nb_voxels[0] != nb_voxels[1] != nb_voxels[2]:
                raise ValueError(
                    'Microstructure_name {} is implemented only in Nx=Ny=Nz grids'.format(microstructure_name))
            phase_field = Circle_Frame(*nb_voxels)

        case 'geometry_2_3D':  # TODO[Bharat] : this is an template for you
            # I will create a Cube with the Body Diagonals
            if nb_voxels.size != 3:
                raise ValueError('Microstructure_name {} is implemented only in 3D'.format(microstructure_name))
            # here should come your code
            if nb_voxels[0] != nb_voxels[1] != nb_voxels[2]:
                raise ValueError(
                    'Microstructure_name {} is implemented only in Nx=Ny=Nz grids'.format(microstructure_name))
            phase_field = HBCC(*nb_voxels)

        case 'geometry_3_3D':  # TODO[Bharat] : this is an template for you
            # I will create a Cube with the Face Diagonals
            if nb_voxels.size != 3:
                raise ValueError('Microstructure_name {} is implemented only in 3D'.format(microstructure_name))
            # here should come your code
            if nb_voxels[0] != nb_voxels[1] != nb_voxels[2]:
                raise ValueError(
                    'Microstructure_name {} is implemented only in Nx=Ny=Nz grids'.format(microstructure_name))
            phase_field = HFCC(*nb_voxels)

        case 'geometry_4_3D':  # TODO[Bharat] : this is an template for you
            # I will create a Cube with the Single Diagonal on Faces
            if nb_voxels.size != 3:
                raise ValueError('Microstructure_name {} is implemented only in 3D'.format(microstructure_name))
            # here should come your code
            if nb_voxels[0] != nb_voxels[1] != nb_voxels[2]:
                raise ValueError(
                    'Microstructure_name {} is implemented only in Nx=Ny=Nz grids'.format(microstructure_name))
            phase_field = HSCC(*nb_voxels)

        case 'geometry_4_3D':  # TODO[Bharat] : this is an template for you
            # I will just create a Completely filled Cube
            if nb_voxels.size != 3:
                raise ValueError('Microstructure_name {} is implemented only in 3D'.format(microstructure_name))
            # here should come your code
            phase_field = Normalcube(*nb_voxels)

        case 'geometry_5_3D':  # TODO[Bharat] : this is an template for you
            # I will create a Cube with a Sphere removed from it
            if nb_voxels.size != 3:
                raise ValueError('Microstructure_name {} is implemented only in 3D'.format(microstructure_name))
            # here should come your code
            phase_field = SphereinCube(*nb_voxels)

        case 'geometry_6_3D':  # TODO[Bharat] : this is an template for you
            # I will create a Cube with a another isocentric connected with the diagonals of Both Cubes Subtracted
            if nb_voxels.size != 3:
                raise ValueError('Microstructure_name {} is implemented only in 3D'.format(microstructure_name))
            # here should come your code
            if nb_voxels[0] != nb_voxels[1] != nb_voxels[2]:
                raise ValueError(
                    'Microstructure_name {} is implemented only in Nx=Ny=Nz grids'.format(microstructure_name))
            phase_field = Metamaterial_1(*nb_voxels)

        case 'geometry_7_3D':  # TODO[Bharat] : this is an template for you
            # I will create a ligtweight strong metamaterial
            if nb_voxels.size != 3:
                raise ValueError('Microstructure_name {} is implemented only in 3D'.format(microstructure_name))
            # here should come your code
            if nb_voxels[0] != nb_voxels[1] != nb_voxels[2]:
                raise ValueError(
                    'Microstructure_name {} is implemented only in Nx=Ny=Nz grids'.format(microstructure_name))
            phase_field = Metamaterial_2(*nb_voxels)

        case 'geometry_8_3D':  # TODO[Bharat] : this is an template for you
            # I will create a ligtweight strong metamaterial
            if nb_voxels.size != 3:
                raise ValueError('Microstructure_name {} is implemented only in 3D'.format(microstructure_name))
            # here should come your code
            if nb_voxels[0] != nb_voxels[1] != nb_voxels[2]:
                raise ValueError(
                    'Microstructure_name {} is implemented only in Nx=Ny=Nz grids'.format(microstructure_name))
            phase_field = Metamaterial_3(*nb_voxels)

    return phase_field  # size is


def Circle_A(Nx, r):
    if Nx % 2 != 0:
        geom = np.zeros((Nx, Nx))
        for i in range(Nx):
            for j in range(Nx):
                if (i - (Nx) // 2) ** 2 + (j - (Nx) // 2) ** 2 > r ** 2:
                    geom[i, j] = 1
        geom = geom[0:Nx, 0:Nx]
    else:
        geom = np.zeros((Nx, Nx))
        for i in range(Nx):
            for j in range(Nx):
                if (i - (Nx) // 2 + 0.5) ** 2 + (j - (Nx) // 2 + 0.5) ** 2 > r ** 2:
                    geom[i, j] = 1
        geom = geom[0:Nx, 0:Nx]
    return geom


def Circle_Frame(*nb_voxels):
    (Nx, Ny, Nz) = nb_voxels
    # Create cube
    Cube = np.zeros((Nx, Ny, Nz))
    k = int(0.05 * Nx)
    r = int(0.4 * Nx)

    # Create meshgrid
    I, J, K = np.meshgrid(np.arange(1, Nx + 1), np.arange(1, Ny + 1), np.arange(1, Nz + 1))

    G = np.zeros_like(Cube)
    Frame = square_frame2D(Nx, k)
    Circle = Circle_A(Nx, r)
    Overall = np.logical_or(Circle, Frame)

    for i in range(k):
        G[i, :, :] = Overall
        G[-i - 1, :, :] = Overall
        G[:, i, :] = Overall
        G[:, -i - 1, :] = Overall
        G[:, :, i] = Overall
        G[:, :, -i - 1] = Overall

    # Restrict D to Nx x Nx x Nx
    G = G[0:Nx, 0:Nx, 0:Nx]
    return G


def Diagonal2D_A(Nx, k):
    geom = np.zeros((Nx, Nx))
    for i in range(Nx):
        for j in range(Nx):
            if i == j or (i - k <= j <= i + k):
                geom[i, j] = 1
    geom = geom[0:Nx, 0:Nx]
    return geom


def Diagonal2D_B(Nx, k):
    geom = np.zeros((Nx, Nx))
    for i in range(Nx):
        for j in range(Nx):
            if (Nx) - i == j or (Nx) - (i + k + 1) <= j <= (Nx) - (i - k + 1):
                geom[i, j] = 1
    geom = geom[0:Nx, 0:Nx];
    return geom


def Diagonal2D_FACE(Nx, k):
    A = Diagonal2D_A(Nx, k)
    B = Diagonal2D_B(Nx, k)
    C = np.logical_or(A, B).astype(int)
    return C


def HBCC(*nb_voxels):
    (Nx, Ny, Nz) = nb_voxels
    # Create cube
    Cube = np.zeros((Nx, Ny, Nz))
    k = int(0.05 * Nx)
    # Create meshgrid
    I, J, K = np.meshgrid(np.arange(1, Nx + 1), np.arange(1, Ny + 1), np.arange(1, Nz + 1))

    H = np.zeros_like(Cube)
    Frame = square_frame2D(Nx, k)

    # TopandBottom=np.logical_or(Frame,Diagonal1)
    # Sides=np.logical_or(Frame,Diagonal2)
    l = k * np.sqrt(3)
    for i in range(k):
        H[i, :, :] = Frame
        H[-i - 1, :, :] = Frame
        H[:, i, :] = Frame
        H[:, -i - 1, :] = Frame
        H[:, :, i] = Frame
        H[:, :, -i - 1] = Frame

    for i in range(Nx):
        for j in range(Nx):
            for k in range(Nx):
                if i <= j + l and i >= j - l and k <= j + l and k >= j - l:
                    H[i, j, k] = 1
                if i <= Nx - j - 1 + l and i >= Nx - j - 1 - l and k <= j + l and k >= j - l:
                    H[i, j, k] = 1
                if i <= j + l and i >= j - l and k + 1 <= Nx - j + l and k + 1 >= Nx - j - l:
                    H[i, j, k] = 1
                if Nx - i - 1 <= j + l and Nx - i - 1 >= j - l and Nx - k - 1 <= j + l and Nx - k - 1 >= j - l:
                    H[i, j, k] = 1

    # Restrict D to Nx x Nx x Nx
    H = H[0:Nx, 0:Ny, 0:Nz]
    return H


def HFCC(*nb_voxels):
    (Nx, Ny, Nz) = nb_voxels
    # Create cube
    Cube = np.zeros((Nx, Ny, Nz))
    k = int(0.05 * Nx)

    # Create meshgrid
    I, J, K = np.meshgrid(np.arange(1, Nx + 1), np.arange(1, Ny + 1), np.arange(1, Nz + 1))

    # Get Diagonal2D_FACE matrix
    Face = Diagonal2D_FACE(Nx, k)
    Frame = square_frame2D(Nx, k)
    Overall = np.logical_or(Face, Frame)
    # Assign values to D
    D = np.zeros_like(Cube)
    for i in range(k):
        D[i, :, :] = Overall
        D[-i - 1, :, :] = Overall
        D[:, i, :] = Overall
        D[:, -i - 1, :] = Overall
        D[:, :, i] = Overall
        D[:, :, -i - 1] = Overall

    # Restrict D to Nx x Nx x Nx
    D = D[0:Nx, 0:Ny, 0:Nz]
    return D


def HSCC(*nb_voxels):
    (Nx, Ny, Nz) = nb_voxels
    # Create cube
    Cube = np.zeros((Nx, Ny, Nz))
    k = int(0.05 * Nx)
    Frame = square_frame2D(Nx, k)
    E = np.zeros_like(Cube)
    for i in range(k):
        E[i, :, :] = Frame
        E[-i - 1, :, :] = Frame
        E[:, i, :] = Frame
        E[:, -i - 1, :] = Frame
        E[:, :, i] = Frame
        E[:, :, -i - 1] = Frame

    # Restrict D to Nx x Nx x Nx
    E = E[0:Nx, 0:Ny, 0:Nz]
    return E


def Kite(Nx, k):
    # Nx=100
    t = Nx // 2
    # k=5
    geom = np.zeros((Nx + k, Nx + k))

    for k in range(k):
        for i in range(0, Nx):
            for j in range(0, Nx):
                if j == -i + t - 1:
                    geom[i, j] = 1
                    geom[i + k, j] = 1
                    geom[i - k, j] = 1

        for i in range(0, Nx):
            for j in range(0, Nx):
                if i == j + t:
                    geom[i, j] = 1
                    geom[i, j + k] = 1
                    geom[i, j - k] = 1

        for i in range(0, Nx):
            for j in range(0, Nx):
                if j == i + t:
                    geom[i, j] = 1
                    geom[i + k, j] = 1
                    geom[i - k, j] = 1

        for i in range(0, Nx):
            for j in range(0, Nx):
                if j == -i + Nx + t - 1:
                    geom[i, j] = 1
                    geom[i + k, j] = 1
                    geom[i - k, j] = 1

    geom = geom[0:Nx, 0:Nx]
    return geom


def Normalcube(*nb_voxels):
    (Nx, Ny, Nz) = nb_voxels
    # Create cube
    geom = np.ones((Nx, Ny, Nz))
    geom = geom[0:Nx, 0:Ny, 0:Nz]
    return geom


def Sphere(Nx, r):
    geom = np.zeros((Nx, Nx, Nx))
    for i in range(Nx):
        for j in range(Nx):
            for k in range(Nx):
                if Nx % 2 != 0:
                    if np.sqrt((i - Nx // 2) ** 2 + (j - Nx // 2) ** 2 + (k - Nx // 2) ** 2) < r:
                        geom[i, j, k] = 1
                else:
                    if np.sqrt((i - Nx // 2 + 0.5) ** 2 + (j - Nx // 2 + 0.5) ** 2 + (k - Nx // 2 + 0.5) ** 2) < r:
                        geom[i, j, k] = 1
    geom = geom[0:Nx, 0:Nx]
    return geom


def SphereinCube(*nb_voxels):
    (Nx, Ny, Nz) = nb_voxels
    # Create cube
    Cube = np.zeros((Nx, Ny, Nz))
    r = int(0.6 * Nx)

    # Create meshgrid
    I, J, K = np.meshgrid(np.arange(1, Nx + 1), np.arange(1, Nx + 1), np.arange(1, Nx + 1))

    N = np.zeros_like(Cube)
    N[:, :, :] = Normalcube(*nb_voxels) - Sphere(Nx, r)
    # Restrict D to Nx x Nx x Nx
    N = N[0:Nx, 0:Nx, 0:Nx]
    return N


def square_frame2D(Nx, k):
    geom = np.zeros((Nx, Nx))
    for i in range(Nx):
        for j in range(Nx):
            if i < k or Nx - i <= k or j < k or Nx - j <= k:
                geom[i, j] = 1
    geom = geom[0:Nx, 0:Nx]
    return geom


def square_frame2D_flexible(Nx, k1, k2):
    geom1 = square_frame2D(Nx, k1)
    geom2 = square_frame2D(Nx, k2)
    geom = np.logical_xor(geom1, geom2).astype(int)
    geom = geom[0:Nx, 0:Nx]
    return geom


def square_frame2D(Nx, k):
    geom = np.zeros((Nx, Nx))
    for i in range(Nx):
        for j in range(Nx):
            if i < k or Nx - i <= k or j < k or Nx - j <= k:
                geom[i, j] = 1
    geom = geom[0:Nx, 0:Nx]
    return geom


def Metamaterial_1(*nb_voxels):
    (Nx, Ny, Nz) = nb_voxels
    # Create cube
    Cube = np.zeros((Nx, Ny, Nz))
    k = int(0.05 * Nx)
    k2 = int(0.35 * Nx)
    k1 = int(0.30 * Nx)
    l = k * np.sqrt(3)
    # Create cube
    Cube = np.zeros((Nx, Ny, Nz), dtype=int)

    # Create meshgrid
    I, J, K = np.meshgrid(np.arange(1, Nx + 1), np.arange(1, Ny + 1), np.arange(1, Nz + 1))
    Frame1 = square_frame2D_flexible(Nx, k1, k2)
    Frame2 = square_frame2D(Nx, k)

    # Assign values to D
    M = np.zeros_like(Cube)
    for i in range(k):
        M[k1 + i, :, :] = Frame1
        M[-k1 - i - 1, :, :] = Frame1
        M[:, k1 + i, :] = Frame1
        M[:, -k1 - i - 1, :] = Frame1
        M[:, :, k1 + i] = Frame1
        M[:, :, -k1 - i - 1] = Frame1

    for i in range(k):
        M[i, :, :] = Frame2
        M[-i - 1, :, :] = Frame2
        M[:, i, :] = Frame2
        M[:, -i - 1, :] = Frame2
        M[:, :, i] = Frame2
        M[:, :, -i - 1] = Frame2

    for i in range(Nx):
        for j in range(Nx):
            for k in range(Nx):
                if i <= j + l and i >= j - l and k <= j + l and k >= j - l:
                    M[i, j, k] = 1
                if i <= Nx - j - 1 + l and i >= Nx - j - 1 - l and k <= j + l and k >= j - l:
                    M[i, j, k] = 1
                if i <= j + l and i >= j - l and k + 1 <= Nx - j + l and k + 1 >= Nx - j - l:
                    M[i, j, k] = 1
                if Nx - i - 1 <= j + l and Nx - i - 1 >= j - l and Nx - k - 1 <= j + l and Nx - k - 1 >= j - l:
                    M[i, j, k] = 1

    for i in range(k2, Nx - k2):
        for j in range(k2, Nx - k2):
            for k in range(k2, Nx - k2):
                M[i, j, k] = 0

    # Restrict D to Nx x Nx x Nx
    M = M[0:Nx, 0:Nx, 0:Nx]
    return M


def Metamaterial_2(*nb_voxels):
    (Nx, Ny, Nz) = nb_voxels
    # Create cube
    Cube = np.zeros((Nx, Ny, Nz))
    k = int(0.05 * Nx)
    # Create meshgrid
    I, J, K = np.meshgrid(np.arange(1, Nx + 1), np.arange(1, Nx + 1), np.arange(1, Nx + 1))
    h = int(Nx / 2)
    l = int((k / 2) + 1)
    # Get Diagonal2D_FACE matrix
    Face = Diagonal2D_FACE(Nx, k)
    kite = Kite(Nx, l)

    # Assign values to D
    D = np.zeros_like(Cube)
    for i in range(k):
        D[i, :, :] = Face
        D[-i - 1, :, :] = Face
        D[:, i, :] = Face
        D[:, -i - 1, :] = Face
        D[:, :, i] = Face
        D[:, :, -i - 1] = Face

    for i in range(l):
        D[h + i, :, :] = kite
        D[h - i, :, :] = kite
        D[:, h + i, :] = kite
        D[:, h - i, :] = kite

    # Restrict D to Nx x Nx x Nx
    D = D[0:Nx, 0:Nx, 0:Nx]
    return D


def specialface(Nx):
    k = int(0.05 * Nx)
    t = int(Nx // 2)
    geom = np.zeros((Nx + k, Nx + k))
    for i in range(Nx):
        for j in range(Nx):
            if i < 2 * k or Nx - i <= 2 * k or t - k - 1 < i < t + k:
                geom[i, j] = 1

            if j == -i + t - 1:
                geom[i, j] = 1
                geom[i + k, j] = 1
                geom[i - k, j] = 1
            if i == j + t:
                geom[i, j] = 1
                geom[i, j + k] = 1
                geom[i, j - k] = 1
            if j == i + t:
                geom[i, j] = 1
                geom[i + k, j] = 1
                geom[i - k, j] = 1
            if j == -i + Nx + t - 1:
                geom[i, j] = 1
                geom[i + k, j] = 1
                geom[i - k, j] = 1

    geom[0:2 * k, int(0.45 * Nx):int(0.55 * Nx)] = 0
    geom[Nx - 2 * k - 1:Nx, int(0.45 * Nx):int(0.55 * Nx)] = 0

    geom = geom[0:Nx, 0:Nx]
    return geom


# %%
def specialface1(Nx):
    k = int(0.05 * Nx)
    t = int(Nx // 2)
    geom = np.zeros((Nx, Nx))
    for i in range(Nx):
        for j in range(Nx):
            if i == j or (i - k <= j <= i + k):
                geom[i, j] = 1
    for i in range(Nx):
        for j in range(Nx):
            if (Nx) - i == j or (Nx) - (i + k + 1) <= j <= (Nx) - (i - k + 1):
                geom[i, j] = 1

    for i in range(k):
        geom[t - k:t + k, :] = 1

    geom = geom[0:Nx, 0:Nx]
    return geom


def Metamaterial_3(*nb_voxels):
    (Nx, Ny, Nz) = nb_voxels
    # Create cube
    Cube = np.zeros((Nx, Nx, Nx))
    k = int(0.05 * Nx)
    t = Nx // 2
    # Create meshgrid
    I, J, K = np.meshgrid(np.arange(1, Nx + 1), np.arange(1, Nx + 1), np.arange(1, Nx + 1))

    # Get Diagonal2D_FACE matrix
    Face = specialface(Nx)
    Lonee = specialface1(Nx)

    # Assign values to D
    D = np.zeros_like(Cube)
    for i in range(2 * k):
        D[i, :, :] = Face
        D[-i - 1, :, :] = Face
        D[:, i, :] = Face
        D[:, -i - 1, :] = Face
        # D[:, :, i] = Face
        # D[:, :, -i-1] = Face
    D[t, :, :] = Lonee
    D[:, t, :] = Lonee

    for i in range(k):
        D[t - k, :, :] = Lonee
        D[t + k, :, :] = Lonee
        D[:, t - k, :] = Lonee
        D[:, t + k, :] = Lonee

    # Restrict D to Nx x Nx x Nx
    D = D[0:Nx, 0:Nx, 0:Nx]
    return D
