"""Preview: ../plot_slide_fracture_whole_run.py (frac_summary_total[_short]) with the AT1 B2 nine-pores series added.
The original script is run unchanged in memory with three patches: two AT1 series appended to SERIES, outputs written
here (figures/frac_summary_total[_short]_AT1.{png,pdf}), nothing copied to the talk folder.
AT1 points are plotted at the AT2-equivalent crack width, sqrt(2) l / h (the convention the script uses for PF-CZM): the
threshold AT1 profile is exp(-s / (sqrt2 l)) (check_crack_profile_notch.py), AT2 exp(-s / l).
    [SHORT=1] python preview_frac_summary_AT1.py"""
import os

H = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.realpath(os.path.join(H, '../plot_slide_fracture_whole_run.py'))
code = open(SRC).read()


def rep(old, new):
    assert code.count(old) == 1, old
    return code.replace(old, new)


AT1_SERIES = '''
_AT1 = os.path.join(HERE, 'damage_AT1_small_strain', 'exp_data', 'nemo', 'study')
SERIES['porous, AT1'] = (lambda N, P: from_iterations(f'{_AT1}/AT1_N{N}_{P}_polydisperse_pores_l0.03125'),
                         [(N, 2 ** 0.5 * N / 32) for N in (32, 64, 128, 256, 512, 1024)], 'tab:blue', 'P', '--')
SERIES['porous, AT1, $1024^2$, $l$ varied'] = (
    lambda x, P: from_iterations(f'{_AT1}/AT1_N1024_{P}_polydisperse_pores_l{x}'),
    [(f'{a / 1024:g}', 2 ** 0.5 * a) for a in (0.25, 0.5, 1, 2, 4, 8, 32)], 'tab:green', 'o', '-')
'''
code = rep("SHORT = os.environ.get('SHORT') == '1'\n", AT1_SERIES + "SHORT = os.environ.get('SHORT') == '1'\n")
code = rep("shutil.copy(p, TALK)", "pass")                                     # no copy into the talk folder
code = rep("shutil.copy(t, TALK)", "pass")
code = rep("p = os.path.join(HERE, 'figures', f'frac_summary_total{\"_short\" if SHORT else \"\"}.{ext}')",
           "p = os.path.join(" + repr(os.path.join(H, 'figures')) +
           ", f'frac_summary_total{\"_short\" if SHORT else \"\"}_AT1.{ext}')")
code = rep("t = os.path.join(HERE, 'figures', 'frac_summary_total_short_ticks.tex')",
           "t = os.path.join(" + repr(os.path.join(H, 'figures')) + ", 'frac_summary_total_short_AT1_ticks.tex')")
code = rep("ax.set_xlim(0.2, 45)", "ax.set_xlim(0.2, 52)")                     # AT1 l = 1/32 at 1024^2: sqrt2 x 32
exec(compile(code, SRC, 'exec'), {'__file__': SRC, '__name__': '__main__'})
