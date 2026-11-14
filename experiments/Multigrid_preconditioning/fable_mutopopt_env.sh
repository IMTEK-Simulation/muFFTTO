# muTopOpt for the comparison with to3d_large.py (source inside a NEMO job, after fable_env.sh):
# my own clone of https://github.com/muSpectre/muTopOpt in ~/software/muTopOpt_fable (the user's ~/software/muTopOpt is
# not touched), its missing pure-Python dependencies in ~/software/muTopOpt_fable/_deps
export MUTOPOPT=$HOME/software/muTopOpt_fable
export PYTHONPATH=$MUTOPOPT:$MUTOPOPT/_deps${PYTHONPATH:+:$PYTHONPATH}
