"""
Write a 3D hollow-bar run (pressurized_bar_3D.py, state.npz) as muGrid NetCDF for muEye: one frame per load increment.

    solid          1 in the solid voxels, 0 elsewhere            (muEye: isosurface 0.5 -> the bar)
    cavity         1 in the gas voxels, 0 elsewhere              (isosurface 0.5 -> the gas)
    det_F          det F at the voxel centre (solid voxels; 1 elsewhere)
    displacement   u (3 components, grid-point units: muEye convention); the pinned cavity-interior nodes get the
                   harmonic extension of the cavity-surface displacement (display only), void nodes stay 0
    frame variables: pressure, cavity_volume_ratio, increment
In muEye: Dataset panel -> Displacement = displacement, Warp scale 1 (true deformation); colour by det_F.
The voxel fields are stored at the voxel index (i, j, k) = its lower corner node, as muEye's per-pixel sub-point.

usage:  python bar3d_to_mueye.py <run folder> [output.nc]     (default <run folder>/bar3d_mueye.nc)
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import netCDF4  # noqa: E402
import numpy as np  # noqa: E402

from muFFTTO import domain, io_utils  # noqa: E402

run = sys.argv[1]
out_file = sys.argv[2] if len(sys.argv) > 2 else os.path.join(run, 'bar3d_mueye.nc')
d = dict(np.load(os.path.join(run, 'state.npz')))
if 'u_increments' in d:                              # OUTPUT_EVERY > 1: u only for the stored increments
    for _key in ('p', 'volume', 'phase', 'hessp', 'nit', 'min_J', 'time'):
        if _key in d:
            d[_key] = d[_key][d['u_increments']]
N, L, h = tuple(int(v) for v in d['N']), tuple(float(v) for v in d['L']), float(d['h'])
solid, cav, clamp = d['solid_px'], d['cav_px'], d['clamp_node']
void_node = np.all([np.roll(~solid, (a, b, c), (0, 1, 2)) for a in (0, 1) for b in (0, 1) for c in (0, 1)], axis=0)
cav_node = void_node & ~clamp & np.any([np.roll(cav, (a, b, c), (0, 1, 2)) for a in (0, 1) for b in (0, 1)
                                        for c in (0, 1)], axis=0)


def voxel_detF(u):
    """det F at the voxel centres from the 8 corner nodes (trilinear element, centre gradient)"""
    G = np.zeros((3, 3) + N)
    for ax in range(3):
        acc = 0.0
        for a in (0, 1):
            for b in (0, 1):                     # average the 4 edges of the voxel along `ax`
                shift = [0, 0, 0]
                others = [o for o in range(3) if o != ax]
                shift[others[0]], shift[others[1]] = -a, -b
                base = np.roll(u, shift, (1, 2, 3))
                acc = acc + (np.roll(base, -1, ax + 1) - base) / h
        G[:, ax] = 0.25 * acc
    F = G + np.eye(3).reshape(3, 3, 1, 1, 1)
    return (F[0, 0] * (F[1, 1] * F[2, 2] - F[1, 2] * F[2, 1]) - F[0, 1] * (F[1, 0] * F[2, 2] - F[1, 2] * F[2, 0])
            + F[0, 2] * (F[1, 0] * F[2, 1] - F[1, 1] * F[2, 0]))


disc = domain.Discretization(cell=domain.PeriodicUnitCell(domain_size=list(L), problem_type='elasticity'),
                             nb_of_pixels_global=N, discretization_type='finite_element',
                             element_type='trilinear_hexahedron', device='cpu')
f_solid = disc.get_scalar_field(name='solid')
f_cav = disc.get_scalar_field(name='cavity')
f_J = disc.get_scalar_field(name='det_F')
f_u = disc.get_unknown_size_field(name='displacement')
f_solid.s[0, 0] = solid.astype(float)
f_cav.s[0, 0] = cav.astype(float)
writer = io_utils.FieldWriter(out_file, [f_solid, f_cav, f_J, f_u],
                              attributes={'nb_of_pixels': list(N), 'domain_lengths': list(L),
                                          'source_run': os.path.basename(os.path.normpath(run)),
                                          'precond': str(d['precond'])},
                              frame_variables={'pressure': (), 'cavity_volume_ratio': (), 'increment': ()})
from display_fill import HarmonicFill  # noqa: E402
fill = HarmonicFill(cav_node)                    # display only: harmonic fill of the pinned cavity nodes
u_prev = None
for k in range(len(d['p'])):
    u = fill(d['u'][k].astype(float), start=u_prev)
    u_prev = u
    f_J.s[0, 0] = np.where(solid, voxel_detF(u), 1.0)
    f_u.s[:, 0] = u
    writer.write(pressure=float(d['p'][k]), cavity_volume_ratio=float(d['volume'][k] / d['volume'][0]), increment=k)
writer.close()
# muEye takes a displacement only from a per-pixel field and reads it in grid-point units
with netCDF4.Dataset(out_file, 'a') as nc:
    for name in list(nc.dimensions):
        if name.startswith('subpt__nodal_points'):
            nc.renameDimension(name, name.replace('nodal_points', 'pixel'))
    for frame in range(nc['displacement'].shape[0]):
        nc['displacement'][frame] = nc['displacement'][frame] / h
    nc.setncattr('displacement_units', f'grid points (x {h:g} for physical units)')
print('written', out_file)
