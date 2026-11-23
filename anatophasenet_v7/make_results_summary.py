"""Writes RESULTS_SUMMARY.md from the v7 runs, with the v6 runs beside them."""
import json
import os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PARENT = os.path.dirname(HERE)
RUNS = [('GAIT-IT (5 classes)', 'results_gait_it_r2plus1d_full', 'results_gait_it_r2plus1d_full'),
        ('INIT (6 classes)', 'results_init6_r2plus1d_full', 'results_init6_r2plus1d_full')]


def load(p):
    return json.load(open(os.path.join(p, 'benchmark_summary.json'), encoding='utf-8'))


def stat(s, key):
    v = np.array([r[key] for r in s['fold_results']], dtype=float)
    return v.mean(), v.std()


def row(name, s, t, key, scale=1):
    a, b = stat(s, key)
    c, d = stat(t, key)
    fmt = '{:.2f} ± {:.2f} %' if scale == 1 else '{:.3f} ± {:.3f}'
    return f'| {name} | {fmt.format(c, d)} | {fmt.format(a, b)} | {c - a:+.3f} |' if scale != 1 else \
        f'| {name} | {fmt.format(c, d)} | {fmt.format(a, b)} | {c - a:+.2f} |'


out = ['# AnatoPhaseNet v7 results', '',
       '10-fold subject-exclusive cross-validation, split seed 42, training seed 42. Mean ± population std over folds.',
       'The v6 column is the earlier run. The key-frame and walk-identity errors affected GAIT-IT only; INIT changed only through the crop and the per-fold geometry noise.',
       'Run-to-run noise: an earlier stopped v7 GAIT-IT run with the same code and seed gave fold 1 at 97.20 % cycle / 100 % walk and fold 2 at 98.46 % / 100 %, against 96.98 % / 99.09 % and 96.91 % / 98.15 % here. Changes of this size are not evidence of a real difference.', '']
for title, new, old in RUNS:
    s = load(os.path.join(HERE, new))
    o = load(os.path.join(PARENT, 'anatophasenet_v6', old))
    names = None
    out += [f'## {title}', '',
            f"Parameters {s['params']:,}; {s['num_cycles']} cycles; {s['num_walks']} walks; geometry signals: {len(s['geometry_signals'])}.", '',
            '| Metric | v7 | v6 | Change |', '|---|---|---|---|',
            row('Cycle accuracy', o, s, 'test_cycle_acc'), row('Cycle macro-F1', o, s, 'test_cycle_f1', 0),
            row('Walk accuracy (B8)', o, s, 'test_sess_acc'), row('Walk macro-F1 (B8)', o, s, 'test_sess_f1', 0),
            row('Walk accuracy (soft vote)', o, s, 'test_sess_vote_acc'), row('Severity accuracy', o, s, 'test_severity_acc'), '',
            '| Fold | Cycle acc | Cycle F1 | Walk acc | Vote acc | Severity acc |', '|---|---|---|---|---|---|']
    for r in s['fold_results']:
        out.append(f"| {r['fold']} | {r['test_cycle_acc']:.2f} | {r['test_cycle_f1']:.3f} | {r['test_sess_acc']:.2f} | "
                   f"{r['test_sess_vote_acc']:.2f} | {r['test_severity_acc']:.2f} |")
    cm = np.array(s['aggregated_confusion'])
    rec = cm.diagonal() / cm.sum(1) * 100
    out += ['', 'Per-class recall from the summed cycle confusion matrix: ' + ', '.join(f'{x:.1f} %' for x in rec) + '.', '',
            'Confusion matrix (rows true, columns predicted):', '', '```', str(cm), '```', '']
open(os.path.join(HERE, 'RESULTS_SUMMARY.md'), 'w', encoding='utf-8').write('\n'.join(out))
print('\n'.join(out))
