#!/usr/bin/env bash
# WP0: unit tests + baseline 32^3 / 64^3 TO histories (reference for every later change)
cd "$(dirname "$0")"; source ./fable_env.sh
( cd ../.. && PYTHONPATH=$PWD $PY -m pytest -q -x test/test_hourglass_control.py test/test_discretization.py \
    test/test_scaled_material.py test/test_topology_optimization_elasticity.py test/test_adjoint_warm_start.py \
    test/test_material_data_evaluation.py ) > logs/fable/unit_tests_baseline.log 2>&1
./fable_run_TO.sh baseline 32 20
./fable_run_TO.sh baseline 64 10
