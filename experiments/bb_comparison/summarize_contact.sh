#!/bin/bash
# one compare_runs.py table per contact problem: latest run per (solver tag, preconditioner), reference = tr_newton GJ
cd "$(dirname "$0")"
PY=~/micromamba/envs/muffto-gpu/bin/python
rm -f contact_table.csv
for prob in patch hertz punch disc; do
    runs=$(ls -d runs/${prob}_* | sed -E 's/_[0-9]{8}-[0-9]{6}$//' | sort -u | while read t; do ls -d ${t}_* | tail -1; done)
    ref=$(echo "$runs" | grep -vE '_(newton_cg|bb[0-9])' | grep '_GJ_')
    others=$(echo "$runs" | grep -v "^$ref$")
    echo "### $prob (reference: tr_newton, Green-Jacobi)"
    $PY compare_runs.py $ref $others --csv contact_table.csv 2>/dev/null
    echo
done
