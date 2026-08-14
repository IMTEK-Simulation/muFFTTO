import pytest
import numpy as np
import muGrid
from mpi4py import MPI
from muFFTTO import tensor_operations

def _make_fc(nb_pixels=(4, 5), nb_nodal_pts=1):
    """Create a fresh FFTEngine and field collection for a test."""
    fft = muGrid.FFTEngine(
        nb_domain_grid_pts=nb_pixels,
        communicator=muGrid.Communicator(MPI.COMM_WORLD),
    )
    fc = fft.real_space_collection
    fc.set_nb_sub_pts('nodal_points', nb_nodal_pts)
    return fft, fc

@pytest.fixture(params=[(4, 5), (3, 3, 3)])
def fc_setup(request):
    nb_pixels = request.param
    fft, fc = _make_fc(nb_pixels)
    rng = np.random.default_rng(42)
    return fc, rng, len(nb_pixels)

def test_trans2(fc_setup):
    fc, rng, dim = fc_setup
    A2 = fc.real_field(name='A2', components=(dim, dim), sub_pt='nodal_points')
    C2 = fc.real_field(name='C2', components=(dim, dim), sub_pt='nodal_points')
    A2.s[...] = rng.standard_normal(A2.s.shape)
    
    tensor_operations.trans2(A2, C2)
    
    expected = np.einsum('ij...->ji...', A2.s)
    np.testing.assert_allclose(C2.s, expected)

def test_ddot42(fc_setup):
    fc, rng, dim = fc_setup
    A4 = fc.real_field(name='A4', components=(dim, dim, dim, dim), sub_pt='nodal_points')
    B2 = fc.real_field(name='B2', components=(dim, dim), sub_pt='nodal_points')
    C2 = fc.real_field(name='C2', components=(dim, dim), sub_pt='nodal_points')
    A4.s[...] = rng.standard_normal(A4.s.shape)
    B2.s[...] = rng.standard_normal(B2.s.shape)
    
    tensor_operations.ddot42(A4, B2, C2)
    
    expected = np.einsum('ijkl...,lk...->ij...', A4.s, B2.s)
    np.testing.assert_allclose(C2.s, expected)

def test_ddot44(fc_setup):
    fc, rng, dim = fc_setup
    A4 = fc.real_field(name='A4', components=(dim, dim, dim, dim), sub_pt='nodal_points')
    B4 = fc.real_field(name='B4', components=(dim, dim, dim, dim), sub_pt='nodal_points')
    C4 = fc.real_field(name='C4', components=(dim, dim, dim, dim), sub_pt='nodal_points')
    A4.s[...] = rng.standard_normal(A4.s.shape)
    B4.s[...] = rng.standard_normal(B4.s.shape)
    
    tensor_operations.ddot44(A4, B4, C4)
    
    expected = np.einsum('ijkl...,lkmn...->ijmn...', A4.s, B4.s)
    np.testing.assert_allclose(C4.s, expected)

def test_dot22(fc_setup):
    fc, rng, dim = fc_setup
    A2 = fc.real_field(name='A2', components=(dim, dim), sub_pt='nodal_points')
    B2 = fc.real_field(name='B2', components=(dim, dim), sub_pt='nodal_points')
    C2 = fc.real_field(name='C2', components=(dim, dim), sub_pt='nodal_points')
    A2.s[...] = rng.standard_normal(A2.s.shape)
    B2.s[...] = rng.standard_normal(B2.s.shape)
    
    tensor_operations.dot22(A2, B2, C2)
    
    expected = np.einsum('ij...,jk...->ik...', A2.s, B2.s)
    np.testing.assert_allclose(C2.s, expected)

def test_dot24(fc_setup):
    fc, rng, dim = fc_setup
    A2 = fc.real_field(name='A2', components=(dim, dim), sub_pt='nodal_points')
    B4 = fc.real_field(name='B4', components=(dim, dim, dim, dim), sub_pt='nodal_points')
    C4 = fc.real_field(name='C4', components=(dim, dim, dim, dim), sub_pt='nodal_points')
    A2.s[...] = rng.standard_normal(A2.s.shape)
    B4.s[...] = rng.standard_normal(B4.s.shape)
    
    tensor_operations.dot24(A2, B4, C4)
    
    expected = np.einsum('ij...,jkmn...->ikmn...', A2.s, B4.s)
    np.testing.assert_allclose(C4.s, expected)

def test_dot42(fc_setup):
    fc, rng, dim = fc_setup
    A4 = fc.real_field(name='A4', components=(dim, dim, dim, dim), sub_pt='nodal_points')
    B2 = fc.real_field(name='B2', components=(dim, dim), sub_pt='nodal_points')
    C4 = fc.real_field(name='C4', components=(dim, dim, dim, dim), sub_pt='nodal_points')
    A4.s[...] = rng.standard_normal(A4.s.shape)
    B2.s[...] = rng.standard_normal(B2.s.shape)
    
    tensor_operations.dot42(A4, B2, C4)
    
    expected = np.einsum('ijkl...,lm...->ijkm...', A4.s, B2.s)
    np.testing.assert_allclose(C4.s, expected)

def test_dyad22(fc_setup):
    fc, rng, dim = fc_setup
    A2 = fc.real_field(name='A2', components=(dim, dim), sub_pt='nodal_points')
    B2 = fc.real_field(name='B2', components=(dim, dim), sub_pt='nodal_points')
    C4 = fc.real_field(name='C4', components=(dim, dim, dim, dim), sub_pt='nodal_points')
    A2.s[...] = rng.standard_normal(A2.s.shape)
    B2.s[...] = rng.standard_normal(B2.s.shape)
    
    tensor_operations.dyad22(A2, B2, C4)
    
    expected = np.einsum('ij...,kl...->ijkl...', A2.s, B2.s)
    np.testing.assert_allclose(C4.s, expected)

def test_inv2(fc_setup):
    fc, rng, dim = fc_setup
    A2 = fc.real_field(name='A2', components=(dim, dim), sub_pt='nodal_points')
    C2 = fc.real_field(name='C2', components=(dim, dim), sub_pt='nodal_points')
    
    # Fill with identity + noise to ensure invertibility
    A2.s[...] = rng.standard_normal(A2.s.shape) * 0.1
    for i in range(dim):
        A2.s[i, i] += 1.0
        
    tensor_operations.inv2(A2, C2)
    
    # Reshape for matrix multiplication check
    A_mat = np.moveaxis(A2.s, [0, 1], [-2, -1])
    C_mat = np.moveaxis(C2.s, [0, 1], [-2, -1])
    
    res = np.matmul(C_mat, A_mat)
    expected = np.eye(dim)
    
    # Broadcast expected identity to match grid shape
    grid_shape = A_mat.shape[:-2]
    expected = np.broadcast_to(expected, grid_shape + (dim, dim))
    
    np.testing.assert_allclose(res, expected, atol=1e-10)

def test_det2(fc_setup):
    fc, rng, dim = fc_setup
    A2 = fc.real_field(name='A2', components=(dim, dim), sub_pt='nodal_points')
    c = fc.real_field(name='c', components=(1, 1), sub_pt='nodal_points')
    A2.s[...] = rng.standard_normal(A2.s.shape)
    
    tensor_operations.det2(A2, c)
    
    A_mat = np.moveaxis(A2.s, [0, 1], [-2, -1])
    expected = np.linalg.det(A_mat)
    np.testing.assert_allclose(c.s[0, 0], expected)

def test_log_field(fc_setup):
    fc, rng, dim = fc_setup
    a = fc.real_field(name='a', components=(1, 1), sub_pt='nodal_points')
    c = fc.real_field(name='c', components=(1, 1), sub_pt='nodal_points')
    a.s[...] = rng.uniform(0.1, 10.0, a.s.shape)
    
    tensor_operations.log_field(a, c)
    
    np.testing.assert_allclose(c.s, np.log(a.s))

def test_trace2(fc_setup):
    fc, rng, dim = fc_setup
    A2 = fc.real_field(name='A2', components=(dim, dim), sub_pt='nodal_points')
    c = fc.real_field(name='c', components=(1, 1), sub_pt='nodal_points')
    A2.s[...] = rng.standard_normal(A2.s.shape)
    
    tensor_operations.trace2(A2, c)
    
    expected = np.einsum('ii...->...', A2.s)
    np.testing.assert_allclose(c.s[0, 0], expected)

def test_add_scaled_identity(fc_setup):
    fc, rng, dim = fc_setup
    A2 = fc.real_field(name='A2', components=(dim, dim), sub_pt='nodal_points')
    scalar = fc.real_field(name='scalar', components=(1, 1), sub_pt='nodal_points')
    lam = fc.real_field(name='lam', components=(1, 1), sub_pt='nodal_points')
    
    A2.s[...] = rng.standard_normal(A2.s.shape)
    scalar.s[...] = rng.standard_normal(scalar.s.shape)
    lam.s[...] = rng.standard_normal(lam.s.shape)
    
    expected = A2.s.copy()
    for i in range(dim):
        expected[i, i] += lam.s[0, 0] * scalar.s[0, 0]
        
    tensor_operations.add_scaled_identity(scalar, lam, A2, dim)
    np.testing.assert_allclose(A2.s, expected)
