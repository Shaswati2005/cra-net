"""Training vs validation accuracy and loss of AnatoPhaseNet v6 (mean over the 10 folds,
shaded band = +/- 1 std across folds), for GAIT-IT and INIT-6. Writes figures/curves_v6.png.

Uses the --ema_warmup runs (results_*_emaw) when they exist. Those runs also log
Train_Cycle_CE, the same cycle cross-entropy as Val_Loss, which is then plotted
instead of the full multi-task Train_Loss so the two loss curves are like-for-like."""
import csv
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
V6 = os.path.join(os.path.dirname(HERE), 'anatophasenet_v6')
RUNS = [('GAIT-IT', 'results_gait_it_r2plus1d_full'), ('INIT-6', 'results_init6_r2plus1d_full')]
TRAIN, VAL = '#2a78d6', '#eb6834'          # categorical slots 1 and 2 of the reference palette
INK, MUTED, GRID = '#0b0b0b', '#52514e', '#e4e3df'


def load(folder):
    if os.path.isdir(os.path.join(V6, folder + '_emaw')):
        folder += '_emaw'
    cols = {'Train_Acc': [], 'Val_Cycle_Acc': [], 'Train_Loss': [], 'Val_Loss': []}
    for k in range(1, 11):
        with open(os.path.join(V6, folder, f'fold_{k}_metrics.csv')) as f:
            rows = list(csv.DictReader(f))
        train_loss = 'Train_Cycle_CE' if 'Train_Cycle_CE' in rows[0] else 'Train_Loss'
        for c in cols:
            src = train_loss if c == 'Train_Loss' else c
            cols[c].append([float(r[src]) for r in rows])
    print(f'{folder}: training loss column = {train_loss}')
    return {c: np.array(v) for c, v in cols.items()}


plt.rcParams.update({'font.family': 'serif', 'font.size': 7, 'axes.edgecolor': MUTED,
                     'axes.labelcolor': INK, 'xtick.color': MUTED, 'ytick.color': MUTED})
fig, axes = plt.subplots(2, 2, figsize=(3.5, 2.75), constrained_layout=True)
for row, (name, folder) in enumerate(RUNS):
    d = load(folder)
    ep = np.arange(1, d['Train_Acc'].shape[1] + 1)
    for col, (tr, va, ylab) in enumerate([('Train_Acc', 'Val_Cycle_Acc', 'Accuracy (%)'),
                                          ('Train_Loss', 'Val_Loss', 'Loss')]):
        ax = axes[row, col]
        for key, colour, label in ((tr, TRAIN, 'Training'), (va, VAL, 'Validation')):
            m, s = d[key].mean(0), d[key].std(0)
            ax.fill_between(ep, m - s, m + s, color=colour, alpha=0.15, linewidth=0)
            ax.plot(ep, m, color=colour, linewidth=1.3, label=label)
        ax.set_title(f'({"abcd"[2 * row + col]}) {name}: {"accuracy" if col == 0 else "loss"}',
                     fontsize=7, color=INK, pad=3)
        ax.set_xlabel('Epoch', labelpad=1)
        ax.set_ylabel(ylab, labelpad=1)
        ax.set_xlim(1, ep[-1])
        ax.set_xticks([1, 5, 10, 15, 20])
        if col == 0:
            ax.set_ylim(0, 100)
        else:
            ax.set_ylim(bottom=0)
        ax.grid(True, color=GRID, linewidth=0.5)
        ax.spines[['top', 'right']].set_visible(False)
        ax.tick_params(length=2, pad=1)
axes[0, 0].legend(loc='lower right', frameon=False, fontsize=6.5, handlelength=1.5)
out = os.path.join(HERE, 'figures', 'curves_v6.png')
fig.savefig(out, dpi=300)
print('saved', out)
