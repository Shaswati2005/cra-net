# AnatoPhaseNet v7 batch-size sweep

Updated 2026-11-23 10:59. 10-fold subject-exclusive cross-validation, split seed 42, training seed 42, all other settings at v7 defaults (lr 1e-3, 20 epochs). Mean ± std over folds.
Note: the learning rate is the same for every batch size, so differences mix the effect of batch size and of the number of optimiser steps. Same-seed reruns differ by up to ~1.5 points per fold.

## Overview

| Dataset | Batch | Status | Time | Cycle acc | Cycle F1 | Walk acc | Walk F1 | Vote acc | Severity acc | Mean val acc |
|---|---|---|---|---|---|---|---|---|---|---|
| GAIT-IT (5 classes) | 4 | done | 2:58:48 | 93.01 ± 4.24 % | 0.940 | 96.06 ± 3.73 % | 0.964 | 95.36 % | 74.14 % | 92.87 % |
| GAIT-IT (5 classes) | 8 | done | 1:48:04 | 94.23 ± 4.02 % | 0.950 | 96.65 ± 4.11 % | 0.969 | 95.37 % | 79.20 % | 95.05 % |
| GAIT-IT (5 classes) | 16 | done | 1:21:01 | 95.10 ± 5.93 % | 0.956 | 97.20 ± 5.29 % | 0.973 | 96.92 % | 79.94 % | 95.89 % |
| INIT (6 classes) | 4 | done | 0:28:05 | 93.51 ± 5.00 % | 0.922 | 91.25 ± 6.96 % | 0.917 | 93.75 % | 79.55 % | 91.85 % |
| INIT (6 classes) | 8 | done | 0:17:02 | 94.83 ± 4.23 % | 0.935 | 91.88 ± 6.88 % | 0.922 | 93.75 % | 82.52 % | 93.52 % |
| INIT (6 classes) | 16 | done | 0:12:44 | 92.18 ± 4.56 % | 0.908 | 90.62 ± 6.40 % | 0.909 | 92.50 % | 76.81 % | 92.74 % |
| INIT (8 classes) | 4 | done | 0:27:39 | 86.98 ± 4.12 % | 0.814 | 83.75 ± 8.00 % | 0.815 | 85.00 % | nan % | 85.36 % |
| INIT (8 classes) | 8 | done | 0:16:47 | 86.81 ± 4.54 % | 0.814 | 86.25 ± 7.81 % | 0.839 | 86.25 % | nan % | 86.13 % |
| INIT (8 classes) | 16 | done | 0:12:35 | 85.61 ± 5.43 % | 0.799 | 85.00 ± 7.50 % | 0.821 | 82.50 % | nan % | 85.79 % |

Plot: `batch_size_comparison.png`. Each run folder also has `aggregated_confusion_matrix.png`, `val_vs_test_accuracy.png`, and per fold `fold_N_training_curves.png`, `fold_N_confusion_matrix.png`, `fold_N_metrics.csv`.

## GAIT-IT (5 classes), batch size 4

Folder `sweep_results\gait_it_bs4`. Parameters 4,995,134; 3783 cycles; 821 walks; geometry signals: 7.

| Fold | Selected epoch | Val cycle acc | Test cycle acc | Test cycle F1 | Test walk acc | Vote acc | Severity acc |
|---|---|---|---|---|---|---|---|
| 1 | 20 | 97.10 | 95.47 | 0.958 | 95.45 | 99.09 | 71.77 |
| 2 | 20 | 76.72 | 89.00 | 0.910 | 95.37 | 90.74 | 78.76 |
| 3 | 20 | 99.45 | 88.77 | 0.908 | 90.57 | 91.51 | 74.01 |
| 4 | 20 | 96.45 | 97.78 | 0.973 | 100.00 | 98.65 | 74.52 |
| 5 | 20 | 87.57 | 98.94 | 0.989 | 100.00 | 100.00 | 63.12 |
| 6 | 20 | 97.79 | 91.80 | 0.925 | 98.61 | 93.06 | 76.46 |
| 7 | 20 | 95.65 | 94.32 | 0.950 | 97.22 | 98.61 | 67.51 |
| 8 | 20 | 86.28 | 97.70 | 0.980 | 100.00 | 100.00 | 76.73 |
| 9 | 20 | 99.68 | 85.92 | 0.886 | 88.89 | 88.89 | 80.51 |
| 10 | 20 | 92.03 | 90.45 | 0.926 | 94.44 | 93.06 | 78.03 |

Per-class recall from the summed cycle confusion matrix: 99.5 %, 76.6 %, 96.3 %, 99.6 %, 96.8 %.

Confusion matrix (rows true, columns predicted):

```
[[ 201    0    1    0    0]
 [   0  648   66    0  132]
 [   0   26  751    3    0]
 [   0    0    3  726    0]
 [   0   39    0    0 1187]]
```

## GAIT-IT (5 classes), batch size 8

Folder `sweep_results\gait_it_bs8`. Parameters 4,995,134; 3783 cycles; 821 walks; geometry signals: 7.

| Fold | Selected epoch | Val cycle acc | Test cycle acc | Test cycle F1 | Test walk acc | Vote acc | Severity acc |
|---|---|---|---|---|---|---|---|
| 1 | 20 | 98.07 | 92.67 | 0.930 | 98.18 | 93.64 | 75.43 |
| 2 | 20 | 83.78 | 94.02 | 0.953 | 97.22 | 96.30 | 79.73 |
| 3 | 20 | 97.78 | 88.98 | 0.910 | 90.57 | 90.57 | 80.25 |
| 4 | 20 | 96.81 | 99.45 | 0.990 | 100.00 | 100.00 | 79.50 |
| 5 | 20 | 96.83 | 97.16 | 0.975 | 100.00 | 96.83 | 75.89 |
| 6 | 20 | 98.11 | 94.18 | 0.948 | 100.00 | 95.83 | 74.07 |
| 7 | 20 | 97.19 | 97.48 | 0.977 | 100.00 | 100.00 | 79.50 |
| 8 | 20 | 88.09 | 96.68 | 0.962 | 97.22 | 95.83 | 79.54 |
| 9 | 20 | 99.68 | 85.56 | 0.883 | 87.50 | 87.50 | 83.39 |
| 10 | 20 | 94.18 | 96.18 | 0.970 | 95.83 | 97.22 | 84.71 |

Per-class recall from the summed cycle confusion matrix: 99.5 %, 83.5 %, 95.3 %, 99.3 %, 96.7 %.

Confusion matrix (rows true, columns predicted):

```
[[ 201    0    1    0    0]
 [   0  706   50    0   90]
 [   1   32  743    3    1]
 [   0    0    5  724    0]
 [   0   39    1    0 1186]]
```

## GAIT-IT (5 classes), batch size 16

Folder `sweep_results\gait_it_bs16`. Parameters 4,995,134; 3783 cycles; 821 walks; geometry signals: 7.

| Fold | Selected epoch | Val cycle acc | Test cycle acc | Test cycle F1 | Test walk acc | Vote acc | Severity acc |
|---|---|---|---|---|---|---|---|
| 1 | 20 | 97.68 | 98.28 | 0.984 | 100.00 | 99.09 | 78.45 |
| 2 | 20 | 87.94 | 99.03 | 0.993 | 100.00 | 100.00 | 82.05 |
| 3 | 20 | 98.61 | 81.29 | 0.842 | 85.85 | 86.79 | 79.00 |
| 4 | 20 | 95.74 | 99.17 | 0.993 | 100.00 | 100.00 | 73.96 |
| 5 | 20 | 98.94 | 98.58 | 0.987 | 100.00 | 100.00 | 82.62 |
| 6 | 20 | 99.37 | 99.21 | 0.992 | 100.00 | 100.00 | 79.63 |
| 7 | 20 | 98.21 | 96.53 | 0.967 | 100.00 | 100.00 | 74.76 |
| 8 | 20 | 87.36 | 96.16 | 0.945 | 98.61 | 97.22 | 80.31 |
| 9 | 20 | 100.00 | 85.92 | 0.886 | 87.50 | 87.50 | 81.95 |
| 10 | 20 | 95.04 | 96.82 | 0.975 | 100.00 | 98.61 | 86.62 |

Per-class recall from the summed cycle confusion matrix: 100.0 %, 87.7 %, 94.4 %, 99.3 %, 97.1 %.

Confusion matrix (rows true, columns predicted):

```
[[ 202    0    0    0    0]
 [   0  742   19    0   85]
 [   4   37  736    3    0]
 [   0    1    4  724    0]
 [   0   34    1    0 1191]]
```

## INIT (6 classes), batch size 4

Folder `sweep_results\init6_bs4`. Parameters 4,998,539; 589 cycles; 160 walks; geometry signals: 9.

| Fold | Selected epoch | Val cycle acc | Test cycle acc | Test cycle F1 | Test walk acc | Vote acc | Severity acc |
|---|---|---|---|---|---|---|---|
| 1 | 20 | 98.21 | 87.27 | 0.859 | 87.50 | 87.50 | 66.67 |
| 2 | 20 | 89.39 | 98.21 | 0.983 | 100.00 | 100.00 | 85.29 |
| 3 | 20 | 88.64 | 84.85 | 0.846 | 81.25 | 81.25 | 75.00 |
| 4 | 20 | 93.24 | 97.73 | 0.974 | 93.75 | 100.00 | 77.27 |
| 5 | 20 | 90.70 | 97.30 | 0.959 | 100.00 | 100.00 | 96.67 |
| 6 | 20 | 98.15 | 90.70 | 0.832 | 81.25 | 81.25 | 100.00 |
| 7 | 20 | 83.33 | 98.15 | 0.973 | 87.50 | 100.00 | 60.71 |
| 8 | 20 | 95.00 | 93.94 | 0.940 | 93.75 | 100.00 | 62.50 |
| 9 | 20 | 100.00 | 88.33 | 0.877 | 87.50 | 87.50 | 82.86 |
| 10 | 20 | 81.82 | 98.59 | 0.983 | 100.00 | 100.00 | 88.57 |

Per-class recall from the summed cycle confusion matrix: 93.3 %, 89.3 %, 88.9 %, 93.3 %, 94.2 %, 100.0 %.

Confusion matrix (rows true, columns predicted):

```
[[ 56   0   4   0   0   0]
 [ 10 109   3   0   0   0]
 [  9   4 112   1   0   0]
 [  2   0   0  56   2   0]
 [  1   2   0   0  49   0]
 [  0   0   0   0   0 169]]
```

## INIT (6 classes), batch size 8

Folder `sweep_results\init6_bs8`. Parameters 4,998,539; 589 cycles; 160 walks; geometry signals: 9.

| Fold | Selected epoch | Val cycle acc | Test cycle acc | Test cycle F1 | Test walk acc | Vote acc | Severity acc |
|---|---|---|---|---|---|---|---|
| 1 | 20 | 98.21 | 87.27 | 0.859 | 81.25 | 81.25 | 76.67 |
| 2 | 20 | 92.42 | 100.00 | 1.000 | 100.00 | 100.00 | 88.24 |
| 3 | 20 | 88.64 | 92.42 | 0.915 | 87.50 | 87.50 | 83.33 |
| 4 | 20 | 95.95 | 95.45 | 0.946 | 93.75 | 100.00 | 77.27 |
| 5 | 20 | 90.70 | 97.30 | 0.959 | 100.00 | 100.00 | 96.67 |
| 6 | 20 | 96.30 | 90.70 | 0.832 | 81.25 | 81.25 | 96.15 |
| 7 | 20 | 98.48 | 98.15 | 0.973 | 87.50 | 100.00 | 60.71 |
| 8 | 20 | 90.00 | 96.97 | 0.968 | 93.75 | 100.00 | 71.88 |
| 9 | 20 | 97.18 | 90.00 | 0.901 | 93.75 | 87.50 | 85.71 |
| 10 | 20 | 87.27 | 100.00 | 1.000 | 100.00 | 100.00 | 88.57 |

Per-class recall from the summed cycle confusion matrix: 95.0 %, 92.6 %, 91.3 %, 93.3 %, 96.2 %, 100.0 %.

Confusion matrix (rows true, columns predicted):

```
[[ 57   0   3   0   0   0]
 [  6 113   2   1   0   0]
 [  7   3 115   1   0   0]
 [  2   0   0  56   2   0]
 [  1   1   0   0  50   0]
 [  0   0   0   0   0 169]]
```

## INIT (6 classes), batch size 16

Folder `sweep_results\init6_bs16`. Parameters 4,998,539; 589 cycles; 160 walks; geometry signals: 9.

| Fold | Selected epoch | Val cycle acc | Test cycle acc | Test cycle F1 | Test walk acc | Vote acc | Severity acc |
|---|---|---|---|---|---|---|---|
| 1 | 20 | 98.21 | 87.27 | 0.859 | 87.50 | 81.25 | 63.33 |
| 2 | 20 | 92.42 | 98.21 | 0.983 | 100.00 | 100.00 | 64.71 |
| 3 | 20 | 86.36 | 84.85 | 0.851 | 81.25 | 87.50 | 77.78 |
| 4 | 20 | 95.95 | 93.18 | 0.928 | 93.75 | 100.00 | 68.18 |
| 5 | 20 | 90.70 | 91.89 | 0.892 | 93.75 | 93.75 | 100.00 |
| 6 | 20 | 100.00 | 90.70 | 0.832 | 81.25 | 81.25 | 100.00 |
| 7 | 20 | 84.85 | 96.30 | 0.948 | 87.50 | 93.75 | 60.71 |
| 8 | 20 | 91.67 | 89.39 | 0.889 | 93.75 | 100.00 | 56.25 |
| 9 | 20 | 100.00 | 90.00 | 0.901 | 87.50 | 87.50 | 82.86 |
| 10 | 20 | 87.27 | 100.00 | 1.000 | 100.00 | 100.00 | 94.29 |

Per-class recall from the summed cycle confusion matrix: 93.3 %, 89.3 %, 84.9 %, 86.7 %, 96.2 %, 100.0 %.

Confusion matrix (rows true, columns predicted):

```
[[ 56   0   4   0   0   0]
 [ 10 109   2   1   0   0]
 [ 13   6 107   0   0   0]
 [  2   0   0  52   6   0]
 [  1   1   0   0  50   0]
 [  0   0   0   0   0 169]]
```

## INIT (8 classes), batch size 4

Folder `sweep_results\init8_bs4`. Parameters 5,004,573; 589 cycles; 160 walks; geometry signals: 9.

| Fold | Selected epoch | Val cycle acc | Test cycle acc | Test cycle F1 | Test walk acc | Vote acc | Severity acc |
|---|---|---|---|---|---|---|---|
| 1 | 20 | 75.00 | 85.45 | 0.790 | 75.00 | 87.50 | nan |
| 2 | 20 | 86.36 | 87.50 | 0.870 | 81.25 | 81.25 | nan |
| 3 | 20 | 81.82 | 87.88 | 0.808 | 87.50 | 87.50 | nan |
| 4 | 20 | 94.59 | 86.36 | 0.802 | 81.25 | 81.25 | nan |
| 5 | 20 | 90.70 | 93.24 | 0.917 | 100.00 | 100.00 | nan |
| 6 | 20 | 81.48 | 88.37 | 0.743 | 75.00 | 75.00 | nan |
| 7 | 20 | 77.27 | 81.48 | 0.700 | 75.00 | 75.00 | nan |
| 8 | 20 | 88.33 | 81.82 | 0.754 | 81.25 | 87.50 | nan |
| 9 | 20 | 94.37 | 83.33 | 0.830 | 87.50 | 81.25 | nan |
| 10 | 20 | 83.64 | 94.37 | 0.922 | 93.75 | 93.75 | nan |

Per-class recall from the summed cycle confusion matrix: 88.3 %, 95.4 %, 61.4 %, 90.9 %, 51.7 %, 91.7 %, 94.2 %, 100.0 %.

Confusion matrix (rows true, columns predicted):

```
[[ 53   0   0   3   4   0   0   0]
 [  0  62   0   2   0   1   0   0]
 [  8  14  35   0   0   0   0   0]
 [  1   0   0  60   5   0   0   0]
 [ 14   0   3  11  31   1   0   0]
 [  2   0   0   0   0  55   3   0]
 [  1   0   1   1   0   0  49   0]
 [  0   0   0   0   0   0   0 169]]
```

## INIT (8 classes), batch size 8

Folder `sweep_results\init8_bs8`. Parameters 5,004,573; 589 cycles; 160 walks; geometry signals: 9.

| Fold | Selected epoch | Val cycle acc | Test cycle acc | Test cycle F1 | Test walk acc | Vote acc | Severity acc |
|---|---|---|---|---|---|---|---|
| 1 | 20 | 71.43 | 83.64 | 0.779 | 75.00 | 81.25 | nan |
| 2 | 20 | 87.88 | 85.71 | 0.835 | 87.50 | 87.50 | nan |
| 3 | 20 | 84.09 | 90.91 | 0.833 | 87.50 | 87.50 | nan |
| 4 | 20 | 98.65 | 88.64 | 0.839 | 81.25 | 81.25 | nan |
| 5 | 20 | 90.70 | 89.19 | 0.853 | 100.00 | 93.75 | nan |
| 6 | 20 | 79.63 | 90.70 | 0.836 | 81.25 | 81.25 | nan |
| 7 | 20 | 81.82 | 79.63 | 0.669 | 75.00 | 75.00 | nan |
| 8 | 20 | 88.33 | 80.30 | 0.727 | 87.50 | 87.50 | nan |
| 9 | 20 | 91.55 | 85.00 | 0.842 | 93.75 | 93.75 | nan |
| 10 | 20 | 87.27 | 94.37 | 0.922 | 93.75 | 93.75 | nan |

Per-class recall from the summed cycle confusion matrix: 93.3 %, 89.2 %, 61.4 %, 89.4 %, 51.7 %, 90.0 %, 96.2 %, 100.0 %.

Confusion matrix (rows true, columns predicted):

```
[[ 56   0   0   2   2   0   0   0]
 [  0  58   4   2   0   1   0   0]
 [  8  12  35   0   1   1   0   0]
 [  0   0   0  59   7   0   0   0]
 [ 10   0   3  14  31   1   1   0]
 [  2   0   0   0   0  54   4   0]
 [  1   0   1   0   0   0  50   0]
 [  0   0   0   0   0   0   0 169]]
```

## INIT (8 classes), batch size 16

Folder `sweep_results\init8_bs16`. Parameters 5,004,573; 589 cycles; 160 walks; geometry signals: 9.

| Fold | Selected epoch | Val cycle acc | Test cycle acc | Test cycle F1 | Test walk acc | Vote acc | Severity acc |
|---|---|---|---|---|---|---|---|
| 1 | 20 | 69.64 | 80.00 | 0.717 | 87.50 | 75.00 | nan |
| 2 | 20 | 89.39 | 76.79 | 0.742 | 93.75 | 75.00 | nan |
| 3 | 20 | 84.09 | 89.39 | 0.821 | 87.50 | 87.50 | nan |
| 4 | 20 | 93.24 | 88.64 | 0.839 | 87.50 | 81.25 | nan |
| 5 | 20 | 90.70 | 86.49 | 0.805 | 93.75 | 93.75 | nan |
| 6 | 20 | 79.63 | 90.70 | 0.836 | 81.25 | 81.25 | nan |
| 7 | 20 | 86.36 | 81.48 | 0.700 | 75.00 | 75.00 | nan |
| 8 | 20 | 85.00 | 81.82 | 0.742 | 75.00 | 81.25 | nan |
| 9 | 20 | 94.37 | 85.00 | 0.849 | 75.00 | 81.25 | nan |
| 10 | 20 | 85.45 | 95.77 | 0.944 | 93.75 | 93.75 | nan |

Per-class recall from the summed cycle confusion matrix: 93.3 %, 78.5 %, 64.9 %, 83.3 %, 55.0 %, 90.0 %, 96.2 %, 100.0 %.

Confusion matrix (rows true, columns predicted):

```
[[ 56   0   0   2   2   0   0   0]
 [  0  51  10   3   0   1   0   0]
 [  8  11  37   0   0   1   0   0]
 [  0   0   0  55  11   0   0   0]
 [ 12   0   4  10  33   1   0   0]
 [  2   0   0   0   0  54   3   1]
 [  1   0   1   0   0   0  50   0]
 [  0   0   0   0   0   0   0 169]]
```
