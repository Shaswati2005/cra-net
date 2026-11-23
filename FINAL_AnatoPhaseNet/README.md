# AnatoPhaseNet — final code and results

This is the final model reported in the project report:
- version: v6,
- biomechanical geometry branch on,
- subject-adversary off,
- (2+1)D backbone.

It was copied from `../anatophasenet_v6/` (code), `results_gait_it_r2plus1d_full` and `results_init6_r2plus1d_full`.

## Layout
```
FINAL_AnatoPhaseNet/
├── code/
│   ├── anato_phasenet_final.py   # final model + training/eval script (= v6, paths adjusted)
│   ├── anato_phasenet_v5.py      # imported: severity head, multi-cycle module
│   └── anato_phasenet_v4.py      # imported: data loading, backbone, folds, plotting
└── results/
    ├── RESULTS_SUMMARY.md        # tables for both datasets
    ├── GAIT_IT/                  # 10-fold results, GAIT-IT (5 classes)
    └── INIT/                     # 10-fold results, INIT (6 classes)
```
Each results folder contains:
- `benchmark_summary.json` (all metrics, args, aggregated confusion matrix),
- per-fold `fold_k_metrics.csv`, `fold_k_test_predictions.npz`, `fold_k_training_curves.png`, `fold_k_confusion_matrix.png`,
- `aggregated_confusion_matrix.png`.

## Headline numbers (10-fold, subject-exclusive)
| Dataset | Cycle acc | Cycle macro-F1 | Walk acc |
|---|---|---|---|
| GAIT-IT | 95.47 ± 4.11 % | 0.962 | 96.28 ± 5.72 % |
| INIT-6  | 92.39 ± 5.15 % | 0.906 | 90.63 ± 6.99 % |

## Re-running
Run from `code/` (Command Prompt):
```
set PY=C:\Users\Asus\AppData\Local\Programs\Python\Python313\python.exe
%PY% anato_phasenet_final.py --dataset gait_it
%PY% anato_phasenet_final.py --dataset init --init_classes 6
```
Where things come from:
- **Dataset and `resnet18_pretrained.pth`:** read from the project root (`..\..`).
- **Caches:** the existing ones in `..\anatophasenet_v4\cache` and `..\anatophasenet_v6\cache` are reused if they're there. If not, they're built in `FINAL_AnatoPhaseNet\cache`.
- **New runs:** written to `results\rerun_*`, so the saved final results are never overwritten.
