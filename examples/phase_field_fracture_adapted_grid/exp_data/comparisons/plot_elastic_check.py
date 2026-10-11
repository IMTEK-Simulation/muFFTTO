"""Elastic check (damage off): P11(F11) of regular and adapted grids, and the deviation from a reference.

    python plot_elastic_check.py out.png reference.nc run.nc [run.nc ...]
"""
import sys
sys.path.insert(0, '/home/martin/Programming/microTopOpt/muFFTTO')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from muFFTTO import io_utils

out, reference, files = sys.argv[1], sys.argv[2], sys.argv[3:]


def curve(f):
    r = io_utils.read_file(f)
    H = np.asarray(r.attributes['H_macro']).reshape(2, 2)
    F = 1.0 + r.frame_variables['lam'] * H[1, 1]
    kind = 'adapted' if int(r.attributes.get('adapted_grid', 0)) else 'regular'
    reg = ', reg. in x' if int(r.attributes.get('regularization_physical', 0)) and kind == 'adapted' else \
        (', reg. in X' if kind == 'adapted' else '')
    if float(r.attributes.get('alpha', 1.0)) == 0.0:
        reg = ', no reg.'
    return f"{kind} {r.fields['phase'].shape[-2]}$^2${reg}", np.r_[1.0, F], np.r_[0.0, r.frame_variables['mean_P'][:, 1, 1]]


ref_label, F_ref, P_ref = curve(reference)
fig, (ax, ax_err) = plt.subplots(1, 2, figsize=(13, 5))
ax.plot(F_ref, P_ref, 'k-', lw=2.5, label=ref_label + ' (reference)')
for f in files:
    label, F, P = curve(f)
    style = ('-' if 'in x' in label else ':') if label.startswith('adapted') else '--'
    if 'no reg' in label:
        style = '-.'
    color = {'32': 'C0', '64': 'C3', '128': 'C2'}[label.split()[1].split('$')[0]]
    if 'no reg' in label:
        color = 'C1' if label.startswith('adapted') else 'C4'
    ax.plot(F, P, style, color=color, label=label)
    P_on_ref = np.interp(-F_ref, -F, P)
    ax_err.plot(F_ref[1:], 100 * (P_on_ref[1:] - P_ref[1:]) / np.abs(P_ref[1:]), style, color=color, label=label)
    i = np.argmin(np.abs(F - 0.92))
    print(f'{label:16s} P11(F11={F[i]:.3f}) {P[i]:+.4f}  ref {np.interp(-F[i], -F_ref, P_ref):+.4f}')
ax.set_xlabel(r'$\bar F_{11}$')
ax.set_ylabel(r'$\bar P_{11}$')
ax.set_title('elastic (no damage)')
ax.legend(fontsize=9)
ax_err.axhline(0, color='k', lw=0.8)
ax_err.set_xlabel(r'$\bar F_{11}$')
ax_err.set_ylabel(r'($\bar P_{11}$ - reference) / |reference|  [%]')
ax_err.set_title(f'deviation from {ref_label}\nsolid: adapted (reg. in x), dotted: adapted (reg. in X), dashed: regular, dash-dot: no regularization (orange adapted, purple regular)', fontsize=10)
ax_err.legend(fontsize=9)
fig.tight_layout()
fig.savefig(out, dpi=110)
print('wrote', out)
