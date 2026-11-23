"""Batch-size sweep for AnatoPhaseNet v7: runs every dataset x batch size one after another.

    python run_all_experiments.py                 # all 9 runs (3 datasets x batch 4, 8, 16)
    python run_all_experiments.py --summary_only  # rebuild the summary from finished runs
    python run_all_experiments.py --datasets gait_it --batch_sizes 8 16

Each run writes the usual v7 outputs (per-fold metrics CSV, training curves, confusion matrices,
test predictions, aggregated confusion matrix, benchmark_summary.json) to
sweep_results/<dataset>_bs<B>/, plus a full console log (run_log.txt) and a val-vs-test plot.

A run whose benchmark_summary.json already exists is skipped, so after a crash or a closed window
you can just start the same command again and it carries on with the next unfinished run.
A run that fails is recorded and the sweep moves on to the next one.

At the end (and after every run) it writes sweep_results/SWEEP_SUMMARY.md, sweep_summary.csv and
batch_size_comparison.png.
"""
import argparse
import csv
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timedelta

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
MAIN = os.path.join(HERE, 'anato_phasenet_v7.py')

DATASETS = {
    'gait_it': {'title': 'GAIT-IT (5 classes)', 'args': ['--dataset', 'gait_it']},
    'init6': {'title': 'INIT (6 classes)', 'args': ['--dataset', 'init', '--init_classes', '6']},
    'init8': {'title': 'INIT (8 classes)', 'args': ['--dataset', 'init', '--init_classes', '8']},
}


def fmt_td(sec):
    return str(timedelta(seconds=int(sec)))


def run_one(ds, bs, out_root, extra):
    out_dir = os.path.join(out_root, f'{ds}_bs{bs}')
    summary = os.path.join(out_dir, 'benchmark_summary.json')
    if os.path.exists(summary):
        print(f'[SKIP] {ds} batch {bs}: already finished ({out_dir})', flush=True)
        return 'done (earlier)', None
    os.makedirs(out_dir, exist_ok=True)
    cmd = [sys.executable, '-u', '-W', 'ignore', MAIN, *DATASETS[ds]['args'],
           '--batch_size', str(bs), '--out_dir', out_dir, *extra]
    print('\n' + '=' * 80, flush=True)
    print(f'[START] {datetime.now():%Y-%m-%d %H:%M:%S}  {DATASETS[ds]["title"]}  batch size {bs}', flush=True)
    print('        ' + ' '.join(cmd), flush=True)
    print('=' * 80, flush=True)
    env = dict(os.environ, PYTHONIOENCODING='utf-8')
    t0 = time.time()
    with open(os.path.join(out_dir, 'run_log.txt'), 'w', encoding='utf-8') as log:
        p = subprocess.Popen(cmd, cwd=HERE, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                             text=True, encoding='utf-8', errors='replace', bufsize=1)
        for line in p.stdout:
            sys.stdout.write(line)
            log.write(line)
            log.flush()
        rc = p.wait()
    dt = time.time() - t0
    with open(os.path.join(out_dir, 'run_time.txt'), 'w', encoding='utf-8') as f:
        f.write(f'{dt:.0f} seconds ({fmt_td(dt)}), exit code {rc}\n')
    if rc != 0 or not os.path.exists(summary):
        print(f'[FAIL] {ds} batch {bs} after {fmt_td(dt)} (exit code {rc}); see {out_dir}\\run_log.txt', flush=True)
        return f'FAILED (exit {rc})', dt
    print(f'[DONE] {ds} batch {bs} in {fmt_td(dt)}', flush=True)
    return 'done', dt


def plot_val_vs_test(s, path, title):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fr = s['fold_results']
    folds = [r['fold'] for r in fr]
    x = np.arange(len(folds))
    w = 0.27
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.bar(x - w, [r['val_cycle_acc'] for r in fr], w, label='Val cycle acc', color='#ff7f0e')
    ax.bar(x, [r['test_cycle_acc'] for r in fr], w, label='Test cycle acc', color='#1f77b4')
    ax.bar(x + w, [r['test_sess_acc'] for r in fr], w, label='Test walk acc', color='#2ca02c')
    ax.set_xticks(x)
    ax.set_xticklabels([f'F{f}' for f in folds])
    ax.set_ylim(max(0, min(min(r['val_cycle_acc'], r['test_cycle_acc'], r['test_sess_acc']) for r in fr) - 10), 101)
    ax.set_ylabel('Accuracy (%)')
    ax.set_title(title)
    ax.grid(True, axis='y', linestyle='--', alpha=0.5)
    ax.legend(loc='lower right')
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_comparison(rows, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    dss = [d for d in DATASETS if any(r['ds'] == d for r in rows)]
    if not dss:
        return
    fig, axes = plt.subplots(1, len(dss), figsize=(5.2 * len(dss), 4.5), squeeze=False)
    for ax, d in zip(axes[0], dss):
        rr = sorted([r for r in rows if r['ds'] == d], key=lambda r: r['bs'])
        x = np.arange(len(rr))
        w = 0.38
        ax.bar(x - w / 2, [r['cyc'] for r in rr], w, yerr=[r['cyc_sd'] for r in rr], capsize=3,
               label='Cycle acc', color='#1f77b4')
        ax.bar(x + w / 2, [r['walk'] for r in rr], w, yerr=[r['walk_sd'] for r in rr], capsize=3,
               label='Walk acc', color='#2ca02c')
        ax.set_xticks(x)
        ax.set_xticklabels([f'bs {r["bs"]}' for r in rr])
        ax.set_ylim(max(0, min(min(r['cyc'], r['walk']) for r in rr) - 15), 102)
        ax.set_title(DATASETS[d]['title'])
        ax.set_ylabel('Test accuracy (%), mean ± std over folds')
        ax.grid(True, axis='y', linestyle='--', alpha=0.5)
        ax.legend(loc='lower right')
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def build_summary(out_root, datasets, batch_sizes, status):
    rows = []
    md = ['# AnatoPhaseNet v7 batch-size sweep', '',
          f'Updated {datetime.now():%Y-%m-%d %H:%M}. 10-fold subject-exclusive cross-validation, split seed 42, '
          'training seed 42, all other settings at v7 defaults (lr 1e-3, 20 epochs). Mean ± std over folds.',
          'Note: the learning rate is the same for every batch size, so differences mix the effect of batch size '
          'and of the number of optimiser steps. Same-seed reruns differ by up to ~1.5 points per fold.', '',
          '## Overview', '',
          '| Dataset | Batch | Status | Time | Cycle acc | Cycle F1 | Walk acc | Walk F1 | Vote acc | Severity acc | Mean val acc |',
          '|---|---|---|---|---|---|---|---|---|---|---|']
    details = []
    for ds in datasets:
        for bs in batch_sizes:
            out_dir = os.path.join(out_root, f'{ds}_bs{bs}')
            sp = os.path.join(out_dir, 'benchmark_summary.json')
            tp = os.path.join(out_dir, 'run_time.txt')
            t = open(tp, encoding='utf-8').read().split(' (')[1].split(')')[0] if os.path.exists(tp) else '-'
            st = status.get((ds, bs), 'done' if os.path.exists(sp) else 'not run')
            if not os.path.exists(sp):
                md.append(f'| {DATASETS[ds]["title"]} | {bs} | {st} | {t} | - | - | - | - | - | - | - |')
                continue
            s = json.load(open(sp, encoding='utf-8'))
            fr = s['fold_results']

            def ms(k):
                v = np.array([r[k] for r in fr], dtype=float)
                return v.mean(), v.std()
            c, cs = ms('test_cycle_acc'); cf, cfs = ms('test_cycle_f1')
            w, ws = ms('test_sess_acc'); wf, wfs = ms('test_sess_f1')
            v, vs = ms('test_sess_vote_acc'); sv, svs = ms('test_severity_acc')
            va, vas = ms('val_cycle_acc')
            rows.append(dict(ds=ds, bs=bs, time=t, cyc=c, cyc_sd=cs, cf1=cf, walk=w, walk_sd=ws, wf1=wf,
                             vote=v, sev=sv, val=va))
            md.append(f'| {DATASETS[ds]["title"]} | {bs} | {st} | {t} | {c:.2f} ± {cs:.2f} % | {cf:.3f} | '
                      f'{w:.2f} ± {ws:.2f} % | {wf:.3f} | {v:.2f} % | {sv:.2f} % | {va:.2f} % |')
            try:
                plot_val_vs_test(s, os.path.join(out_dir, 'val_vs_test_accuracy.png'),
                                 f'{DATASETS[ds]["title"]}, batch {bs}: validation vs test per fold')
            except Exception as e:
                print(f'[WARN] val-vs-test plot failed for {out_dir}: {e}')
            cm = np.array(s['aggregated_confusion'])
            rec = cm.diagonal() / np.maximum(cm.sum(1), 1) * 100
            details += [f'## {DATASETS[ds]["title"]}, batch size {bs}', '',
                        f'Folder `{os.path.relpath(out_dir, HERE)}`. Parameters {s["params"]:,}; {s["num_cycles"]} cycles; '
                        f'{s["num_walks"]} walks; geometry signals: {len(s["geometry_signals"])}.', '',
                        '| Fold | Selected epoch | Val cycle acc | Test cycle acc | Test cycle F1 | Test walk acc | Vote acc | Severity acc |',
                        '|---|---|---|---|---|---|---|---|']
            for r in fr:
                details.append(f"| {r['fold']} | {r['selected_epoch']} | {r['val_cycle_acc']:.2f} | {r['test_cycle_acc']:.2f} | "
                               f"{r['test_cycle_f1']:.3f} | {r['test_sess_acc']:.2f} | {r['test_sess_vote_acc']:.2f} | "
                               f"{r['test_severity_acc']:.2f} |")
            details += ['', 'Per-class recall from the summed cycle confusion matrix: '
                        + ', '.join(f'{x:.1f} %' for x in rec) + '.', '',
                        'Confusion matrix (rows true, columns predicted):', '', '```', str(cm), '```', '']
    md += ['', 'Plot: `batch_size_comparison.png`. Each run folder also has `aggregated_confusion_matrix.png`, '
           '`val_vs_test_accuracy.png`, and per fold `fold_N_training_curves.png`, `fold_N_confusion_matrix.png`, '
           '`fold_N_metrics.csv`.', ''] + details
    with open(os.path.join(out_root, 'SWEEP_SUMMARY.md'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(md))
    with open(os.path.join(out_root, 'sweep_summary.csv'), 'w', newline='', encoding='utf-8') as f:
        wr = csv.writer(f)
        wr.writerow(['dataset', 'batch_size', 'time', 'cycle_acc_mean', 'cycle_acc_std', 'cycle_f1', 'walk_acc_mean',
                     'walk_acc_std', 'walk_f1', 'vote_acc', 'severity_acc', 'val_cycle_acc'])
        for r in rows:
            wr.writerow([r['ds'], r['bs'], r['time'], f"{r['cyc']:.2f}", f"{r['cyc_sd']:.2f}", f"{r['cf1']:.4f}",
                         f"{r['walk']:.2f}", f"{r['walk_sd']:.2f}", f"{r['wf1']:.4f}", f"{r['vote']:.2f}",
                         f"{r['sev']:.2f}", f"{r['val']:.2f}"])
    try:
        plot_comparison(rows, os.path.join(out_root, 'batch_size_comparison.png'))
    except Exception as e:
        print(f'[WARN] comparison plot failed: {e}')
    return md


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--datasets', nargs='+', default=list(DATASETS), choices=list(DATASETS))
    ap.add_argument('--batch_sizes', nargs='+', type=int, default=[4, 8, 16])
    ap.add_argument('--out_root', default=os.path.join(HERE, 'sweep_results'))
    ap.add_argument('--summary_only', action='store_true')
    ap.add_argument('extra', nargs=argparse.REMAINDER,
                    help='anything after "--" is passed to anato_phasenet_v7.py, e.g. -- --epochs 2 --fold_ids 1')
    a = ap.parse_args()
    extra = a.extra[1:] if a.extra[:1] == ['--'] else a.extra
    os.makedirs(a.out_root, exist_ok=True)

    status = {}
    if not a.summary_only:
        try:
            import torch
            gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO GPU (CPU only, very slow)'
        except ImportError:
            sys.exit(f'PyTorch is not installed for {sys.executable}. Use the Python313 interpreter.')
        runs = [(d, b) for d in a.datasets for b in a.batch_sizes]
        print(f'Python {sys.executable}\nGPU    {gpu}\nRuns   {len(runs)}: ' +
              ', '.join(f'{d}/bs{b}' for d, b in runs) + f'\nOutput {a.out_root}', flush=True)
        t_all = time.time()
        for i, (d, b) in enumerate(runs, 1):
            print(f'\n### Run {i}/{len(runs)}   (sweep time so far {fmt_td(time.time() - t_all)})', flush=True)
            status[(d, b)] = run_one(d, b, a.out_root, extra)[0]
            build_summary(a.out_root, a.datasets, a.batch_sizes, status)
        print(f'\nSweep finished in {fmt_td(time.time() - t_all)}.')
        for (d, b), st in status.items():
            print(f'  {d:8s} bs {b:3d}: {st}')

    md = build_summary(a.out_root, a.datasets, a.batch_sizes, status)
    print('\n' + '\n'.join(md[:12 + len(a.datasets) * len(a.batch_sizes)]))
    print(f'\nFull summary: {os.path.join(a.out_root, "SWEEP_SUMMARY.md")}')


if __name__ == '__main__':
    main()
