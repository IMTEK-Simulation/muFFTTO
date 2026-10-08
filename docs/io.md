# Saving and loading fields

[`muFFTTO/io_utils.py`](../muFFTTO/io_utils.py) stores muGrid fields in NetCDF
files through muGrid's `FileIONetCDF`, the mechanism muTopOpt uses as well.
It works the same in serial and under MPI: every rank writes its own
subdomain, and the file always holds the global fields.

## The four functions

```python
from muFFTTO import io_utils

# 1. snapshot: the current values of some fields
io_utils.save_fields('result.nc', [damage, u_fluct], attributes={'Gc': Gc, 'l': length_scale})

# 2. load back into existing fields (in place); returns the attributes
attributes = io_utils.load_fields('result.nc', [damage, u_fluct])

# 3. time series: one frame per write(), plus per-frame values
writer = io_utils.FieldWriter('run.nc', [damage, u_fluct, history],
                              attributes={'Gc': Gc},
                              frame_variables={'load': (), 'stress': (2, 2)})
for load in loads:
    ...                                   # solve
    writer.write(load=load, stress=sigma_ij)
writer.close()                            # or: with io_utils.FieldWriter(...) as writer:

io_utils.load_fields('run.nc', [damage, u_fluct, history], frame=10)   # restart from frame 10

# 4. post-processing in numpy, no discretization needed (needs netCDF4)
run = io_utils.read_file('run.nc')
run.fields['damage']           # array [frame, *components, sub_pt, x, y], global grid
run.frame_variables['load']    # array [frame]
run.attributes['Gc']
```

- **Fields** are identified by their muGrid name, so the names in one file
  must be unique. They must not contain `__`, which muGrid uses internally;
  the functions check this.
- **Several discretizations** can share one file, e.g. the vector
  elasticity and the scalar damage discretization of the fracture example.
- **`load_fields`** needs fields with the same components, sub-points and
  global grid as the stored ones (normally: the same script). The number of
  MPI ranks may differ from the run that wrote the file. It fills `.s`; the
  library operators refresh the ghost layers themselves.
- **`read_file`** returns each field with the layout of `field.s` on the
  global grid, so `run.fields['damage'][-1, 0, 0]` is the last damage field as
  an `[x, y]` array.
- **Attributes** (strings, numbers, arrays) are written when the file is
  created. Arrays are stored flattened.

## Workaround for a muGrid limitation

muGrid's NetCDF I/O cannot register fields with a single component and more
than one sub-point per pixel: quadrature-point scalars such as the fields
from `get_quad_field_scalar` (e.g. the history field H), and scalar nodal
fields of elements with several nodes per pixel (Q2, `biquadratic_rectangle`,
e.g. the damage). `io_utils` writes these through a per-pixel helper field
`<name>_qp` with one component per sub-point and converts back on loading. `read_file`
undoes the conversion too, so you never see the helper.

## Examples

- **Phase-field fracture.**
  `examples/phase_field_fracture/example_2D_phase_field_fracture_AT2.py` writes
  `exp_data/phase_field_fracture_AT2.nc`, with one frame per load step or, with
  `output_every = 'iteration'`, per staggered iteration. A frame holds the damage,
  the displacement fluctuation and the history field, plus the load, the
  homogenized stress, the step and iteration numbers and a `converged` flag.
  `plot_phase_field_fracture_output.py` in the same folder replots it:

  ```
  python plot_phase_field_fracture_output.py [file.nc] [frame ...]
  python plot_phase_field_fracture_output.py [file.nc] --animate   # saves file.mp4 (gif without ffmpeg)
  ```

- **Topology optimization.** The scripts in `examples/topology_optimization/`
  write `data/<script>/<prec>_eta_<eta>_w_<w>_history.nc` (the phase field of
  every L-BFGS iteration, switched by `save_history`) and `..._final.nc` (the
  optimum, with the parameters, the homogenized tensor and the optimization log
  as attributes). `example_2D_conductivity_TO_discrete.py` reads the smooth
  optimum back with `load_fields`. See
  [Topology optimization](examples/topology_optimization.md#outputs).

- **Internal contact.** The four scripts in `examples/internal_contact/` write
  `exp_data/<script>/Nx=..Ny=../tmc_run.nc`: one frame per load increment, with the
  increment history (load, mean stresses, min det F, energy, solver counters,
  convergence flag) as frame variables. The `_net_cdf` variant also writes the
  pixel-level view fields for muEye. `read_tmc.py` reads all of them with
  `read_file` (summary and response curves). See
  [Internal contact](examples/internal_contact.md#variant-_net_cdfpy-netcdf-output-for-mueye).
