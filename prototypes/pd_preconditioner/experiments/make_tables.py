"""Markdown tables for REPORT.md from the CSV files in results/ (prints to stdout)."""
import csv
import os

from common import RESULTS


def read(name):
    path = os.path.join(RESULTS, name)
    return list(csv.DictReader(open(path))) if os.path.exists(path) else []


def grid_table(rows, contrast):
    """Rows: p, columns: n; cell 'G / GJ / PD'."""
    rows = [r for r in rows if float(r['chi']) == contrast]
    ns = sorted({int(r['n']) for r in rows})
    ps = sorted({int(r['p']) for r in rows})
    cell = {(int(r['n']), int(r['p']), r['method']): r['iterations'] for r in rows}
    lines = ['| p \\ n | ' + ' | '.join(str(n) for n in ns) + ' |', '|---|' + '---|' * len(ns)]
    for p in reversed(ps):
        values = []
        for n in ns:
            if (n, p, 'G') in cell:
                values.append(' / '.join(cell[(n, p, m)] for m in ('G', 'GJ', 'PD')))
            else:
                values.append('')
        lines.append(f'| {p} | ' + ' | '.join(values) + ' |')
    return '\n'.join(lines)


for name, contrasts in (('t1_laminate', (10.0, 1e4)), ('t1b_laminate_random_rhs', (10.0, 1e4)),
                        ('t2_cosine', (1e4, float('inf')))):
    for kind in ('scalar', 'elastic'):
        rows = read(f'{name}_{kind}.csv')
        if not rows:
            continue
        for chi in contrasts:
            print(f'\n#### {name}, {kind}, chi = {chi:g} (cells: G / GJ / PD)\n')
            print(grid_table(rows, chi))

for kind in ('scalar', 'elastic'):
    rows = read(f't3_disk_{kind}.csv')
    if rows:
        steps = sorted({int(r['smoothing']) for r in rows})
        cell = {(int(r['smoothing']), r['method']): r for r in rows}
        print(f'\n#### t3_disk, {kind} (iterations; FFTs in brackets)\n')
        print('| method | ' + ' | '.join(f'i={i}' for i in steps) + ' |')
        print('|---|' + '---|' * len(steps))
        for m in ('G', 'GJ', 'PD'):
            print(f'| {m} | ' + ' | '.join(f"{cell[(i, m)]['iterations']} ({cell[(i, m)]['ffts']})" for i in steps)
                  + ' |')

rows = read('t4_spectra_scalar.csv')
if rows:
    print('\n#### t4 spectra, scalar, chi = 1e4\n')
    print('| n | case | method | lambda_min | lambda_max | cond | eigenvalues = 1 | != 1 | max imag |')
    print('|---|---|---|---|---|---|---|---|---|')
    for r in rows:
        print(f"| {r['n']} | {r['case']} | {r['method']} | {r['lambda_min']} | {r['lambda_max']} | {r['cond']} | "
              f"{r['eigenvalues_equal_1']} | {r['not_equal_1']} | {r['max_imag']} |")

for kind in ('scalar', 'elastic'):
    rows = read(f't5_mean_drift_{kind}.csv')
    if rows:
        print(f'\n#### t5 mean drift, {kind} (relative to Green at the same tolerance)\n')
        print('| tol | method | displacement | displacement, mean removed | strain | |mean of u| |')
        print('|---|---|---|---|---|---|')
        for r in rows:
            print(f"| {float(r['tol']):g} | {r['method']} | {r['displacement_error']} | "
                  f"{r['displacement_error_mean_removed']} | {r['strain_error']} | {r['mean_of_solution']} |")

for kind in ('scalar', 'elastic'):
    rows = read(f't6_rho_clip_{kind}.csv')
    if rows:
        print(f'\n#### t6 rho_clip, {kind}, cosine chi = inf (iterations)\n')
        cases = sorted({(int(r['n']), int(r['p'])) for r in rows})
        print('| n, p | PD 1e-4 | PD 1e-6 | PD 1e-8 | PD 1e-10 | G | GJ |')
        print('|---|---|---|---|---|---|---|')
        for n, p in cases:
            sel = [r for r in rows if int(r['n']) == n and int(r['p']) == p]
            pd = {float(r['rho_clip']): r['iterations'] for r in sel if r['method'] == 'PD'}
            other = {r['method']: r['iterations'] for r in sel if r['method'] != 'PD'}
            print(f"| {n}, {p} | {pd[1e-4]} | {pd[1e-6]} | {pd[1e-8]} | {pd[1e-10]} | {other['G']} | {other['GJ']} |")
