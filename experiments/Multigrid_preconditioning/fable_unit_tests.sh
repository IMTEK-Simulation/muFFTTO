#!/usr/bin/env bash
# The unit tests the brief requires (section 6) + the new multigrid tests + the 3D gradient check of the reference script
cd "$(dirname "$0")"; source ./fable_env.sh
( cd ../.. && PYTHONPATH=$PWD $PY -m pytest -q test/test_hourglass_control.py test/test_discretization.py \
    test/test_scaled_material.py test/test_topology_optimization_elasticity.py test/test_adjoint_warm_start.py \
    test/test_material_data_evaluation.py test/test_multigrid.py ) > logs/fable/unit_tests_after.log 2>&1
for E in trilinear_hexahedron trilinear_hexahedron_1Q_hg; do
  ELEMENT=$E OBJECTIVE=energy TIGHT=1 $PY check_objective_gradient_3d.py 2>&1 | grep -E "ELEMENT|direction" >> logs/fable/unit_tests_after.log
done
