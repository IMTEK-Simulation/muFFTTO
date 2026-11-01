# Composite A (Li & Rudykh 2019): homogenization scripts

These scripts model the periodic cell of Composite A: a soft matrix with circular voids and stiff circular
inclusions (Li & Rudykh 2019, Fig. 4). Compressive loading leads to buckling.

| script | what it does |
|---|---|
| `Stefanus_2D_homogenization_elasticity.py` | small-strain linear elasticity, homogenized stiffness (4 load cases) |
| `stefanus_2D_homogenization_finite_strain_elasticity_NeoHookean.py` | finite strain, neo-Hookean, plain Newton + CG (the original) |
| `stefanus_2D_homogenization_finite_strain_elasticity_NeoHookean_trust_region.py` | same problem with a **trust-region Newton** solver (NuMPI `tr_newton_bounded`): more robust, recommended |
| `plot_homogenized_stress.py` | stress-strain curve of trust-region runs against the paper's Fig. 4 |

## Installation (once)

Prerequisites:
- Python >= 3.10 and git
- **an MPI library** (Open MPI or MPICH). `mpi4py`, NuMPI and muFFTTO need it even for runs on one process. Check
  with `mpirun --version`. If it is missing, install it, e.g. `sudo apt install openmpi-bin libopenmpi-dev`
  (Ubuntu), or without admin rights `pip install mpich` inside the venv below.

```bash
git clone git@github.com:IMTEK-Simulation/muFFTTO.git
cd muFFTTO
git checkout StefanusMichael_Dev_rebased          # the branch with these scripts

python3 -m venv .venv
source .venv/bin/activate

pip install numpy scipy matplotlib mpi4py muGrid jax
pip install "git+https://github.com/muSpectre/NuMPI@feat/tr_precond"   # trust-region solver with preconditioner
pip install --no-deps -e .                        # muFFTTO itself, editable (your changes apply directly)
```

Check:
```bash
python -c "import muGrid, NuMPI, muFFTTO; print('ok', muGrid.__version__)"
```

Notes:
- Without an MPI library, `import mpi4py` fails with `RuntimeError: cannot load MPI library` (see Prerequisites).
- NuMPI must come from the `feat/tr_precond` branch. The PyPI version has no `precond=` argument in
  `tr_newton_bounded`.
- You can ignore the warning "An NVIDIA GPU may be present ... Falling back to cpu" (it comes from jax).

## Running

Run the scripts from this folder:

```bash
cd "Stefanus/original/Testing file"

# trust-region, 64 px, 30 load increments, bilinear elements
python stefanus_2D_homogenization_finite_strain_elasticity_NeoHookean_trust_region.py -n 64

# recommended element (no volumetric locking), 128 px
python stefanus_2D_homogenization_finite_strain_elasticity_NeoHookean_trust_region.py -n 128 --element biquadratic_rectangle

# stress-strain plot of all runs found in exp_data/ (or pass CSV paths)
python plot_homogenized_stress.py
```

Options of the trust-region script (`-h` lists them all):

| option | meaning |
|---|---|
| `-n 64` | pixels in x (y follows from the cell shape) |
| `-inc 30` | number of load increments |
| `--element biquadratic_rectangle` | Q2 elements instead of Q1 (`bilinear_rectangle`). Q1 locks for nearly incompressible material (too stiff) |
| `--void_layer 16` | adds 16 void pixel columns on the +x side: a strip with free lateral sides |
| `--no-save_per_it` | do not save the `.npy` fields after every increment |
| `--show` | show the deformed cell in a window after every increment (close the window to continue). Default: PNG only |
| `--device gpu` | run on the GPU (needs muGrid built with CUDA plus `cupy`, see below) |

The load, the material parameters and `E_LOAD` are set at the top of each script.

## Output

- `exp_data/<script>/<run>/`: `homogenized_stress.csv` (one row per increment: F and the homogenized 1st Piola-Kirchhoff
  stress <P> in units of the matrix shear modulus mu_m = 0.6 MPa), `info_log_final.npz`, fields per increment
  (`*.npy`), `stress_strain.png` (from the plot script).
- `figures/<script>/<run>/deformed_phase_inc*.png`: deformed cell after every increment.

## Things to know

- **Use at least 64 px.** At 32 px the ligament between the holes is about 1 pixel wide and the original Newton
  script fails. 128 px is better.
- **Do not use too few increments with the original script.** It fails with 2 increments (CG: "operator is not
  positive definite") and works with its default of 50. The trust-region script is more forgiving.
- **The trust-region script is the robust one.** It rejects steps that do not lower the energy and picks the CG
  tolerance adaptively.
- `Stefanus_2D_homogenization_elasticity.py` uses the grid size `NX` set at the top of the script (default 256). `--px_per_ligament` is currently **not** used for the grid.
- Expected run times on a laptop CPU (pip-installed muGrid, tested 2026-11-01):
  trust-region 64 px, 30 increments: ~2 min; trust-region 64 px biquadratic, 10 increments: ~6 min;
  original Newton 64 px, 50 increments: ~8 min; linear small strain (256 x 222): ~8 min.
  Twice the pixels per side costs at least 4x.
- The two original scripts call `plt.show()`: on a desktop a plot window opens and the script **waits until you
  close it**. To run without windows: `MPLBACKEND=Agg python <script>.py`.
- Known modelling limits: with the lateral stretch F fixed, the response is about 20 % too stiff compared with the
  paper. With `--void_layer` (free sides, 1 cell wide) it is too soft. The paper's stress-free lateral side
  (<P_xx> = 0) is not implemented yet in this script.

## GPU (optional)

The PyPI `muGrid` runs on the CPU only. For `--device gpu` you need muGrid compiled with CUDA
(https://github.com/muSpectre/muGrid) plus `cupy`. GPU pays off from about 128 px (2x faster). Ask Martin for help.
