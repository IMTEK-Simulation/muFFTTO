# GPU trial helpers (October 2026)

Helper scripts from the GPU port of muFFTTO and the contact / fracture / optimizer studies.
The full scratch tree (run copies, logs, NetCDF outputs) is archived, git-ignored, in
`exp_data/gpu_trial_scratch/`; paths in the shell scripts point there.

| file | purpose |
|---|---|
| `sweep_run.py`, `sweep.sh`, `run_example.py` | run an example on CPU or forced GPU, log every implicit host/device transfer |
| `compare.py`, `peek.py` | compare / inspect NetCDF outputs of CPU and GPU runs |
| `iterate.sh` | rerun loop used while porting |
| `domain_gpu.patch`, `example_gpu.patch` | early GPU patches (since merged into muFFTTO) |
| `repro_mugrid_multinode.py`, `mugrid_issue/` | muGrid GenericLinearOperator multi-node GPU bug: reproducer, tests, fix patch, issue text (fix on muGrid branch `bug/gpu-generic-operator-multinode`) |
| `fracture/` | crack-event detection (`break_points.py`, `where_damage.py`), load sequences, patches that set up the fracture studies |
| `adapted/patch_adapted.py` | adds the device switch to the adapted-grid contact example |
| `optimizers/` | run scripts of the L-BFGS / MMA / TR-Newton topology-optimization comparison (`examples/topology_optimization/figures/optimizer_comparison_*.png`) |
| `geometry_previews/` | quick previews of the `frame_contact_supports` and `bubble_array` geometries |
