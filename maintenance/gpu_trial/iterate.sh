export MAMBA_ROOT_PREFIX=~/micromamba; eval "$(~/.local/bin/micromamba shell hook -s bash)"; micromamba activate muffto-gpu
cd $HOME/PycharmProjects/muFFTTO/exp_data/gpu_trial_scratch
timeout 900 python run_example.py gpu ex3d_gpu.py < /dev/null > gpu.log 2>&1; echo rc=$?
awk '/Traceback/,0' gpu.log | grep -vE "frozen runpy|run_path\(|~~~|\^\^\^|run_example|^\s*(\[|elastic tangent|0 +1 )" | grep -v "^ \[\|^\[\[" | head -25
grep -c "norm of residual" gpu.log
