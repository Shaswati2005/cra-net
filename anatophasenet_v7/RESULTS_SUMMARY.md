# AnatoPhaseNet v7 results

10-fold subject-exclusive cross-validation, split seed 42, training seed 42. Mean ± population std over folds.
The v6 column is the earlier run. The key-frame and walk-identity errors affected GAIT-IT only; INIT changed only through the crop and the per-fold geometry noise.
Run-to-run noise: an earlier stopped v7 GAIT-IT run with the same code and seed gave fold 1 at 97.20 % cycle / 100 % walk and fold 2 at 98.46 % / 100 %, against 96.98 % / 99.09 % and 96.91 % / 98.15 % here. Changes of this size are not evidence of a real difference.

## GAIT-IT (5 classes)

Parameters 4,995,134; 3783 cycles; 821 walks; geometry signals: 7.

| Metric | v7 | v6 | Change |
|---|---|---|---|
| Cycle accuracy | 95.62 ± 4.30 % | 95.47 ± 4.11 % | +0.15 |
| Cycle macro-F1 | 0.962 ± 0.034 | 0.962 ± 0.033 | -0.000 |
| Walk accuracy (B8) | 97.48 ± 4.39 % | 96.28 ± 5.72 % | +1.20 |
| Walk macro-F1 (B8) | 0.976 ± 0.041 | 0.965 ± 0.053 | +0.011 |
| Walk accuracy (soft vote) | 97.37 ± 4.12 % | 95.53 ± 6.33 % | +1.84 |
| Severity accuracy | 79.46 ± 3.18 % | 80.68 ± 3.06 % | -1.22 |

| Fold | Cycle acc | Cycle F1 | Walk acc | Vote acc | Severity acc |
|---|---|---|---|---|---|
| 1 | 96.98 | 0.972 | 99.09 | 98.18 | 75.00 |
| 2 | 96.91 | 0.974 | 98.15 | 100.00 | 83.01 |
| 3 | 87.73 | 0.901 | 88.68 | 89.62 | 80.04 |
| 4 | 99.45 | 0.995 | 100.00 | 100.00 | 74.24 |
| 5 | 98.23 | 0.984 | 100.00 | 98.41 | 77.66 |
| 6 | 97.09 | 0.974 | 100.00 | 98.61 | 76.98 |
| 7 | 97.48 | 0.976 | 100.00 | 100.00 | 83.28 |
| 8 | 96.93 | 0.957 | 100.00 | 100.00 | 80.05 |
| 9 | 86.64 | 0.893 | 88.89 | 88.89 | 81.23 |
| 10 | 98.73 | 0.990 | 100.00 | 100.00 | 83.12 |

Per-class recall from the summed cycle confusion matrix: 100.0 %, 89.2 %, 95.9 %, 99.6 %, 96.6 %.

Confusion matrix (rows true, columns predicted):

```
[[ 202    0    0    0    0]
 [   0  755   23    0   68]
 [   4   26  748    2    0]
 [   0    0    3  726    0]
 [   0   42    0    0 1184]]
```

## INIT (6 classes)

Parameters 4,998,539; 589 cycles; 160 walks; geometry signals: 9.

| Metric | v7 | v6 | Change |
|---|---|---|---|
| Cycle accuracy | 92.45 ± 4.72 % | 92.39 ± 5.15 % | +0.06 |
| Cycle macro-F1 | 0.911 ± 0.058 | 0.906 ± 0.068 | +0.005 |
| Walk accuracy (B8) | 89.38 ± 7.42 % | 90.62 ± 6.99 % | -1.25 |
| Walk macro-F1 (B8) | 0.899 ± 0.073 | 0.906 ± 0.071 | -0.007 |
| Walk accuracy (soft vote) | 91.88 ± 6.88 % | 93.12 ± 9.04 % | -1.25 |
| Severity accuracy | 79.49 ± 13.24 % | 77.87 ± 14.82 % | +1.62 |

| Fold | Cycle acc | Cycle F1 | Walk acc | Vote acc | Severity acc |
|---|---|---|---|---|---|
| 1 | 85.45 | 0.822 | 81.25 | 81.25 | 80.00 |
| 2 | 98.21 | 0.983 | 100.00 | 100.00 | 61.76 |
| 3 | 87.88 | 0.877 | 87.50 | 87.50 | 77.78 |
| 4 | 93.18 | 0.928 | 87.50 | 87.50 | 68.18 |
| 5 | 93.24 | 0.913 | 100.00 | 100.00 | 100.00 |
| 6 | 90.70 | 0.832 | 81.25 | 81.25 | 100.00 |
| 7 | 98.15 | 0.973 | 87.50 | 93.75 | 60.71 |
| 8 | 89.39 | 0.904 | 81.25 | 93.75 | 75.00 |
| 9 | 88.33 | 0.877 | 87.50 | 93.75 | 82.86 |
| 10 | 100.00 | 1.000 | 100.00 | 100.00 | 88.57 |

Per-class recall from the summed cycle confusion matrix: 95.0 %, 90.2 %, 85.7 %, 86.7 %, 94.2 %, 100.0 %.

Confusion matrix (rows true, columns predicted):

```
[[ 57   0   3   0   0   0]
 [  9 110   2   1   0   0]
 [ 10   8 108   0   0   0]
 [  3   0   0  52   4   1]
 [  1   2   0   0  49   0]
 [  0   0   0   0   0 169]]
```
