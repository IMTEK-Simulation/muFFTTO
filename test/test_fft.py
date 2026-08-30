import sys
import os
import numpy as np

# Add parent directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from muFFTTO import domain
from muFFTTO import material_models

 ### TODO FIRST OF ALL TEST FFT ON MULTIPLE NODES / QUAD POINTS
def test_fft_per_quadrature_point():
    """Test FFT applied separately to each quadrature point."""

    # Setup discretization
    domain_size = [2, 3]
    problem_type = 'conductivity'

    my_cell = domain.PeriodicUnitCell(domain_size=domain_size,
                                      problem_type=problem_type)
    discretization_type = 'finite_element'
    discretization = domain.Discretization(
        cell=my_cell,
        nb_of_pixels_global=[2, 3],
        discretization_type=discretization_type,
        element_type='bilinear_rectangle'
    )

    print(f"\n{'=' * 70}")
    print(f"FFT Test: domain={domain_size}, element=bilinear_rectangle")
    print(f"{'=' * 70}")

    # Create unit impulse field
    unit_impulse_inxyz = discretization.get_unknown_size_field(name='unit_impulse')
    unit_impulse_response_inxyz = discretization.get_unknown_size_field(name='unit_impulse_response')
    unit_impulse_response_inqks = discretization.ffield_collection.complex_field(
        name='unit_impulse_response_inqks',
        components=(discretization.unknown_size[0],),
        sub_pt='nodal_points'
    )

    # Material setup
    K_1, G_1 = material_models.get_bulk_and_shear_modulus(E=3, poisson=0.2)
    mat_1 = material_models.get_elastic_material_tensor(
        dim=discretization.domain_dimension,
        K=K_1,
        mu=G_1,
        kind='linear'
    )
    material_data_field = discretization.get_material_data_size_field_mugrid(name='material_dat')
    material_data_field.s[...] = mat_1[:, :, :, :, np.newaxis, np.newaxis, np.newaxis]

    print(f"Unknown shape: {discretization.unknown_size}")
    print(f"Number of quadrature points: {discretization.unknown_size[1]}")
    print(f"Spatial shape: {unit_impulse_inxyz.s.shape}")

    # Test for each impulse position
    for impulse_position in np.ndindex(unit_impulse_inxyz.s.shape[0:2]):
        if np.any(np.all(discretization.fft.icoords == 0, axis=0)):
            print(f"\n--- Impulse position {impulse_position} ---")

            # Set unit impulse
            unit_impulse_inxyz.sg.fill(0)
            unit_impulse_inxyz.s[impulse_position + (0,) * (unit_impulse_inxyz.s.ndim - 2)] = 1

            # Apply system matrix
            unit_impulse_response_inxyz.sg.fill(0)
            discretization.apply_system_matrix_mugrid(
                material_data_field=material_data_field,
                input_field_inxyz=unit_impulse_inxyz,
                output_field_inxyz=unit_impulse_response_inxyz
            )

            # Communicate ghosts
            discretization.fft.communicate_ghosts(unit_impulse_response_inxyz)

            # Get spatial data
            spatial_data = unit_impulse_response_inxyz.s.copy()
            print(f"Spatial data shape: {spatial_data.shape}")
            print(f"Spatial data sample [0,0,:3,:3]:\n{spatial_data[0, 0, :3, :3]}")

            # Apply FFT
            unit_impulse_response_inqks.sg.fill(0)
            discretization.fft.fft(unit_impulse_response_inxyz, unit_impulse_response_inqks)

            # Get FFT data
            fft_data = unit_impulse_response_inqks.s.copy()
            print(f"FFT data shape: {fft_data.shape}")
            print(f"FFT data sample [0,0,:3,0]:\n{fft_data[0, 0, :3, 0]}")

            # Check conjugate symmetry for each quadrature point
            print(f"\nChecking conjugate symmetry per quadrature point:")
            num_qpts = discretization.unknown_size[1]

            for q in range(num_qpts):
                # Extract this q's FFT data
                fft_q = fft_data[0, q, :, :]  # Shape: [x, y]

                # Check conjugate symmetry: FFT[k] = conj(FFT[-k])
                fft_conj = np.conj(fft_q)
                fft_flipped = np.flip(np.flip(fft_conj, axis=0), axis=1)

                max_diff = np.max(np.abs(fft_q - fft_flipped))

                if max_diff < 1e-10:
                    print(f"  q={q}: ✓ PASS (conjugate symmetry, max_diff={max_diff:.2e})")
                else:
                    print(f"  q={q}: ✗ FAIL (conjugate symmetry, max_diff={max_diff:.2e})")
                    print(f"       FFT[0,0] = {fft_q[0, 0]}")
                    print(f"       conj(FFT[-1,-1]) = {fft_flipped[0, 0]}")

            # Inverse FFT roundtrip test
            print(f"\nRoundtrip test (FFT -> IFFT):")
            discretization.fft.ifft(unit_impulse_response_inqks, unit_impulse_response_inxyz)
            unit_impulse_response_inxyz.s[:] *= discretization.fft.normalisation

            recovered = unit_impulse_response_inxyz.s.copy()
            roundtrip_error = np.max(np.abs(recovered - spatial_data))

            if roundtrip_error < 1e-12:
                print(f"  ✓ PASS (roundtrip error={roundtrip_error:.2e})")
            else:
                print(f"  ✗ FAIL (roundtrip error={roundtrip_error:.2e})")

    print(f"\n{'=' * 70}")
    print("FFT Test Complete!")
    print(f"{'=' * 70}\n")


if __name__ == "__main__":
    test_fft_per_quadrature_point()