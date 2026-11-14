"""Summary of the NEMO AT1 study (B2 nine pores): iterations and results only, no timings.
Reads logs/nemo_study/AT1_N*_<P>_polydisperse_pores_l<l>.log + exp_data/nemo/study/*_iterations.csv
(copied from $WS/runs/AT1). Per run: elastic solves, mean CG_u and CG_d per solve, load steps hitting the staggered cap,
peak sigma_xx and its strain, cut-through strain (sigma < 0.2 peak), status.
    python summarize_AT1_study.py            ->  table in the terminal + exp_data/nemo/study/AT1_study_summary.csv"""
import os, re, glob, csv
import numpy as np

H = os.path.dirname(os.path.abspath(__file__))
LOGS, DATA = os.path.join(H, 'logs/nemo_study'), os.path.join(H, 'exp_data/nemo/study')
name_re = re.compile(r'AT1_N(\d+)_(dual|Green_Jacobi|Green)_polydisperse_pores_l([0-9.e-]+)\.log$')
step_re = re.compile(r'step\s+(\d+)\s+E_\w+ = ([0-9.e+-]+)\s+sigma_\w+ = ([0-9.e+-]+).*?staggered its =\s*(\d+)')

rows = []
for f in sorted(glob.glob(os.path.join(LOGS, 'AT1_N*.log'))):
    m = name_re.search(os.path.basename(f))
    if not m:
        continue                                            # smoke tests, notch runs
    N, P, l = int(m[1]), m[2], float(m[3])
    txt = open(f).read()
    st = np.array([[float(a), float(b), int(c)] for _, a, b, c in step_re.findall(txt)])
    if len(st) == 0:
        continue
    E, s, stag = st[:, 0], st[:, 1], st[:, 2]
    ip = int(np.argmax(s))
    cut = np.nonzero((np.arange(len(s)) > ip) & (s < 0.2 * s[ip]))[0]
    with open(os.path.join(DATA, os.path.basename(f)[:-4] + '_iterations.csv')) as fc:
        it = list(csv.DictReader(fc))
    cg_u = np.array([int(r['cg_u']) for r in it]); cg_d = np.array([int(r['cg_d']) for r in it])
    status = 'cut' if 'cell cut through' in txt else ('running' if 'Elapsed time' not in txt else 'end of load')
    rows.append(dict(N=N, lh=round(l * N, 4), l=l, P=P, solves=len(cg_u), cg_u=cg_u.mean(), cg_d=cg_d.mean(),
                     capped=int((stag >= 500).sum()), peak=s[ip], E_peak=E[ip],
                     E_cut=E[cut[0]] if len(cut) else np.nan, steps=len(s), status=status))

rows.sort(key=lambda r: (r['lh'] if r['l'] != 0.03125 else 99, r['N'], r['P']))
hdr = f"{'N':>5} {'l/h':>6} {'P':<13}{'solves':>7} {'CG_u/solve':>11} {'CG_d/solve':>11} {'capped':>7} " \
      f"{'peak sig':>10} {'E_peak':>7} {'E_cut':>7}  status"
print(hdr); print('-' * len(hdr))
for r in rows:
    w = 2 if r['P'] == 'dual' else 1
    print(f"{r['N']:>5} {r['lh']:>6g} {r['P']:<13}{r['solves']:>7} {r['cg_u']:>7.1f}{'(x2)' if w == 2 else '    '} "
          f"{r['cg_d']:>11.1f} {r['capped']:>7} {r['peak']:>10.4e} {r['E_peak']:>7.3f} {r['E_cut']:>7.3f}  {r['status']}")
with open(os.path.join(DATA, 'AT1_study_summary.csv'), 'w', newline='') as fo:
    wr = csv.DictWriter(fo, fieldnames=list(rows[0])); wr.writeheader(); wr.writerows(rows)
