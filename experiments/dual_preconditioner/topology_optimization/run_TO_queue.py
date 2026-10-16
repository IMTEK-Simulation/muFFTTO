"""Overnight queue: example_2D_elasticity_TO.py with CG_MODE=energy (cold starts, energy stop 1e-6) for the three
preconditioners on several grids and soft phases, filter width fixed at 1/64. LANES jobs at once, in the order below;
jobs still running at DEADLINE are stopped (their logs keep the per-evaluation CG counts).

    python run_TO_queue.py            -> logs/TO_N<N>_soft<e>_<preconditioner>.log, queue.log
"""
import datetime
import os
import subprocess
import time

HERE = os.path.dirname(os.path.realpath(__file__))
# the local copy of example_2D_elasticity_TO.py and its data (this folder)
TO = os.path.dirname(os.path.realpath(__file__))
PYTHON = os.path.expanduser('~/micromamba/envs/muffto-gpu/bin/python')
LANES = 6
DEADLINE = datetime.datetime.now().replace(hour=5, minute=45, second=0, microsecond=0)
if DEADLINE < datetime.datetime.now():
    DEADLINE += datetime.timedelta(days=1)

# default: the overnight 2026-10-15 queue (64^2 jobs ran in a first pass). N >= 128 with GTOL=0 (stop on the step
# test or maxiter), see example_2D_elasticity_TO.py.  Other sets:  python run_TO_queue.py SOFT_EXP [SOFT_EXP ...]
# -> all three preconditioners on 64^2, 128^2, 256^2 for each soft phase 1e-SOFT_EXP (Green last on every grid)
import sys
if len(sys.argv) > 1:
    QUEUE = [(n, int(e), p) for n in (64, 128, 256) for p in ('Green_Jacobi', 'dual', 'Green') for e in sys.argv[1:]]
else:
    QUEUE = ([(128, 3, p) for p in ('Green_Jacobi', 'dual', 'Green')]
             + [(128, 5, p) for p in ('Green_Jacobi', 'dual')]
             + [(256, 3, p) for p in ('Green_Jacobi', 'dual', 'Green')]
             + [(128, 5, 'Green')])


def log(message):
    with open(os.path.join(HERE, 'logs', 'queue.log'), 'a') as f:
        f.write(f'{datetime.datetime.now():%H:%M:%S} {message}\n')


running = []
queue = list(QUEUE)
while queue or running:
    for job in list(running):
        (n, e, p), proc, t0 = job
        if proc.poll() is not None:
            running.remove(job)
            log(f'done  N{n} soft1e-{e} {p} rc={proc.returncode} {time.time() - t0:.0f} s')
    if datetime.datetime.now() >= DEADLINE:
        for (n, e, p), proc, t0 in running:
            proc.terminate()
            log(f'STOP  N{n} soft1e-{e} {p} at the deadline after {time.time() - t0:.0f} s')
        for n, e, p in queue:
            log(f'skip  N{n} soft1e-{e} {p}')
        break
    while queue and len(running) < LANES:
        n, e, p = queue.pop(0)
        env = dict(os.environ, NPIX=str(n), SOFT_EXP=str(e), ETA=str(1 / 64), CG_MODE='energy', SHOW_PLOTS='0', GTOL='0',
                   MPLBACKEND='Agg', JAX_PLATFORMS='cpu', OMP_NUM_THREADS='2')
        out = open(os.path.join(HERE, 'logs', f'TO_N{n}_soft1e-{e}_{p}.log'), 'w')
        proc = subprocess.Popen([PYTHON, '-u', 'example_2D_elasticity_TO.py', p], cwd=TO, env=env, stdout=out,
                                stderr=subprocess.STDOUT)
        running.append(((n, e, p), proc, time.time()))
        log(f'start N{n} soft1e-{e} {p}')
    time.sleep(20)
log('queue finished')
