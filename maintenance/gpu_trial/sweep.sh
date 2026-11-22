#!/bin/bash
# usage: sweep.sh {cpu|gpu}   -- runs every homogenization example, one at a time
DEV=$1
T=$HOME/PycharmProjects/muFFTTO/exp_data/gpu_trial_scratch
EX=~/PycharmProjects/muFFTTO/examples
export MAMBA_ROOT_PREFIX=~/micromamba
eval "$(~/.local/bin/micromamba shell hook -s bash)"
micromamba activate muffto-gpu
export MPLBACKEND=Agg
mkdir -p $T/sweep
for rel in $(cd $EX && find . -path "*homogenization*" -name "*.py" | sort); do
    name=$(basename $rel .py)
    d=$T/sweep/$name/$DEV
    rm -rf $d && mkdir -p $d
    # run a copy so __file__-relative outputs (figures, exp_data) stay in scratch
    cp $EX/$rel $d/$name.py
    cd $d
    start=$(date +%s.%N)
    timeout 1800 python -u $T/sweep_run.py $DEV $d/$name.py < /dev/null > stdout.log 2> stderr.log
    rc=$?
    end=$(date +%s.%N)
    printf "%-75s %s rc=%-4s %8.1f s\n" $name $DEV $rc $(echo "$end - $start" | bc) | tee -a $T/sweep/summary_$DEV.txt
done
echo "SWEEP $DEV DONE"
