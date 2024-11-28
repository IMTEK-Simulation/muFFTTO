import numpy as np
import scipy.sparse.linalg as sp
import itertools

from sympy.stats.sampling.sample_scipy import scipy

from muFFTTO import domain
from muFFTTO import solvers
from muFFTTO import microstructure_library

# ----------------------------------- GRID ------------------------------------
ndim = 2  # number of dimensions
N = 63  # 31  # number of voxels (assumed equal for all directions)
offset = 3  # 9
ndof = ndim ** 2 * N ** 2  # number of degrees-of-freedom
nb_grid_pts = [N, N]
lengths = [1., 1.]
# ---------------------- PROJECTION, TENSORS, OPERATIONS ----------------------

# tensor operations/products: np.einsum enables index notation, avoiding loops
# e.g. ddot42 performs $C_ij = A_ijkl B_lk$ for the entire grid
trans2 = lambda A2: np.einsum('ij...          ->ji...  ', A2)
ddot42 = lambda A4, B2: np.einsum('ijkl...,lk...  ->ij...  ', A4, B2)
ddot44 = lambda A4, B4: np.einsum('ijkl...,lkmn...->ijmn...', A4, B4)
dot22 = lambda A2, B2: np.einsum('ij...  ,jk...  ->ik...  ', A2, B2)
dot24 = lambda A2, B4: np.einsum('ij...  ,jkmn...->ikmn...', A2, B4)
dot42 = lambda A4, B2: np.einsum('ijkl...,lm...  ->ijkm...', A4, B2)
dyad22 = lambda A2, B2: np.einsum('ij...  ,kl...  ->ijkl...', A2, B2)

shape = tuple((N for _ in range(ndim)))
# identity tensor                                               [single tensor]
i = np.eye(ndim)


def expand(arr):
    new_shape = (np.prod(arr.shape), np.prod(shape))
    ret_arr = np.zeros(new_shape)
    ret_arr[:] = arr.reshape(-1)[:, np.newaxis]
    return ret_arr.reshape((*arr.shape, *shape))


# identity tensors                                            [grid of tensors]
I = expand(i)
I4 = expand(np.einsum('il,jk', i, i))
I4rt = expand(np.einsum('ik,jl', i, i))
I4s = (I4 + I4rt) / 2.
II = dyad22(I, I)

# projection operator                                         [grid of tensors]
# NB can be vectorized (faster, less readable), see: "elasto-plasticity.py"
# - support function / look-up list / zero initialize
delta = lambda i, j: np.array(i == j, dtype=float)  # Dirac delta function
freq = np.arange(-(N - 1) / 2., +(N + 1) / 2.)  # coordinate axis -> freq. axis
Ghat4 = np.zeros([ndim, ndim, ndim, ndim, N, N])  # zero initialize
# - compute
for i, j, l, m in itertools.product(range(ndim), repeat=4):
    for x, y in itertools.product(range(N), repeat=ndim):
        q = np.array([freq[x], freq[y]])  # frequency vector
        if not q.dot(q) == 0:  # zero freq. -> mean
            Ghat4[i, j, l, m, x, y] = -(q[i] * q[j] * q[l] * q[m]) / (q.dot(q)) ** 2 + \
                                      (delta(j, l) * q[i] * q[m] + delta(j, m) * q[i] * q[l] + \
                                       delta(i, l) * q[j] * q[m] + delta(i, m) * q[j] * q[l]) / (2. * q.dot(q))

# (inverse) Fourier transform (for each tensor component in each direction)
fft = lambda x: np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(x), shape))
ifft = lambda x: np.fft.fftshift(np.fft.ifftn(np.fft.ifftshift(x), shape))

# functions for the projection 'G', and the product 'G : K : eps'
G = lambda A2: np.real(ifft(ddot42(Ghat4, fft(A2))))
K_deps = lambda depsm: ddot42(K4, depsm.reshape(ndim, ndim, N, N))
G_K_deps = lambda depsm: G(K_deps(depsm))
G_deps = lambda depsm: G(depsm.reshape(ndim, ndim, N, N))

# ------------------- PROBLEM DEFINITION / CONSTITIVE MODEL -------------------
problem_type = 'elasticity'
discretization_type = 'finite_element'
element_type = 'linear_triangles'
formulation = 'small_strain'
# phase indicator: cubical inclusion of volume fraction (9**3)/(31**3)
my_cell = domain.PeriodicUnitCell(domain_size=lengths,
                                  problem_type=problem_type)

discretization = domain.Discretization(cell=my_cell,
                                       nb_of_pixels_global=nb_grid_pts,
                                       discretization_type=discretization_type,
                                       element_type=element_type)
phase_field_smooth = microstructure_library.get_geometry(nb_voxels=discretization.nb_of_pixels,
                                                         microstructure_name='circle_inclusion',
                                                         coordinates=discretization.fft.coords)

# phase  = np.zeros([N,N,N]); phase[-9:,:9,-9:] = 1.

phase = np.copy(phase_field_smooth)
# material parameters + function to convert to grid of scalars
param = lambda M0, M1: M0 * np.ones(shape) * (1. - phase) + M1 * np.ones(shape) * phase
# K      = param(0.833,8.33)  # bulk  modulus                   [grid of scalars]
# mu     = param(0.386,3.86)  # shear modulus                   [grid of scalars]
# E2, E1 = 1.2, 0
# poisson = .3
# K2, K1 = (E / (3 * (1 - 2 * poisson)) for E in (E2, E1))
# m2, m1 = (E / (2 * (1 + poisson)) for E in (E2, E1))
K = param(1, 1)
mu = param(0.5, 0.5)


# phase_field = phase_field_smooth + 1e-4
def apply_smoother(phase):
    # Define a 2D smoothing kernel
    kernel = np.array([[0.0625, 0.125, 0.0625],
                       [0.125, 0.25, 0.125],
                       [0.0625, 0.125, 0.0625]])

    # Apply convolution for smoothing
    smoothed_arr = scipy.signal.convolve2d(phase, kernel, mode='same', boundary='wrap')
    return smoothed_arr


phase_field = phase_field_smooth

ratios = np.arange(100)
nb_it = np.zeros((1, ratios.size), )
nb_it_ML = np.zeros((1, ratios.size), )
pf_max = np.zeros((1, ratios.size), )
pf_min = np.zeros((1, ratios.size), )

for i in np.arange(ratios.size):
    ratio = ratios[i]
    # phase_field =  ratio*phase_field_smooth + (1-ratio)*phase_field_pwconst
    # phase_field =phase_field_pwconst + 1e-5*np.random.random(phase_field_pwconst.shape)
    #        phase_field =phase_field_pwconst  + 1e-4*phase_field_smooth
    # stiffness tensor                                            [grid of tensors]
    K4 = K * II + 2. * mu * (I4s - 1. / 3. * II)

    if i == 0:
        phase_field = phase_field_smooth + 1e-4
    if i > 0:
        phase_field = apply_smoother(phase_field)
    pf_min[0, i] = np.min(phase_field)
    pf_max[0, i] = np.max(phase_field)

    K4[..., :, :, :] = K4[..., :, :, :] * np.power(phase_field, 1)
    # ----------------------------- NEWTON ITERATIONS -----------------------------

    # set macroscopic loading
    DE = np.zeros([ndim, ndim, N, N])
    DE[0, 0] += 1.0
    DE[1, 1] += 1.0

    # initial residual: distribute "DE" over grid using "K4"
    b = -K_deps(DE)

    iiter = 0


    #strain_field, norms = solvers.PCG(K_deps, b, x0=None, P=G_deps, steps=int(1000), toler=1e-6)
    strain_field, norms = solvers.Richardson(G_K_deps, b, x0=None, omega=0.0001, P=None, steps=int(1000), toler=1e-6)

    nb_it_ML[0, i] = (len(norms['residual_rz']))
    norm_rz = norms['residual_rz'][-1]
    norm_rr = norms['residual_rr'][-1]
    print(norm_rz)
    print(norm_rr)


    print(nb_it_ML)
    print(pf_min)
    print(pf_max)
