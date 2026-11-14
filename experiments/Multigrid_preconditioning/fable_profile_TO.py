"""
WP1 memory breakdown of the 3D TO as it is (FABLE_BRIEF.md): runs example_3D_elasticity_TO.py unchanged (settings from
the environment, see fable_run_TO.sh), and when the design optimizer returns prints the memory table of fable_memory.py
for the TO discretization and every multigrid level, then exits (the post-processing solves are skipped).
    MAXITER=2 ... python fable_profile_TO.py MG
"""
import os
import runpy
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import design_optimizers
import fable_memory

optimize = design_optimizers.l_bfgs_bounded_precond


def optimize_then_report(fun, x0, **kwargs):
    result = optimize(fun, x0, **kwargs)
    script = fun.__globals__                                    # the namespace of the TO script
    owners = {'TO': script['discretization']}
    for level in script['mg_hierarchy'].levels:
        owners[f'MG level {level.index}'] = level.disc
    fable_memory.report(owners, script['n_pix'] ** 3)
    sys.stdout.flush()
    os._exit(0)


design_optimizers.l_bfgs_bounded_precond = optimize_then_report
runpy.run_path(os.path.join(HERE, os.environ.get('TO_SCRIPT', 'example_3D_elasticity_TO.py')), run_name='__main__')
