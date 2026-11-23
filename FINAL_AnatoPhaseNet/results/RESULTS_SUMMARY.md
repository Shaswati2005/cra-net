# Final results — AnatoPhaseNet (v6, geometry branch, no adversary)

10-fold subject-exclusive cross-validation (split seed 42). Mean ± std over folds.

## GAIT-IT (5 classes)

| Metric | Value |
|---|---|
| Parameters | 5.00 M |
| Cycle accuracy | 95.47 ± 4.11 % |
| Cycle macro-F1 | 0.962 ± 0.033 |
| Walk (session) accuracy | 96.28 ± 5.72 % |
| Walk macro-F1 | 0.965 ± 0.053 |
| Walk majority-vote accuracy | 95.53 % |
| Severity accuracy | 80.68 % |

| Fold | Cycle acc | Cycle F1 | Walk acc | Severity acc |
|---|---|---|---|---|
| 1 | 97.42 | 0.975 | 99.09 | 78.54 |
| 2 | 96.01 | 0.969 | 99.07 | 84.98 |
| 3 | 87.55 | 0.897 | 87.74 | 80.08 |
| 4 | 99.18 | 0.993 | 100.00 | 77.53 |
| 5 | 96.14 | 0.966 | 96.83 | 78.60 |
| 6 | 98.43 | 0.985 | 100.00 | 77.75 |
| 7 | 97.20 | 0.973 | 97.22 | 79.13 |
| 8 | 98.23 | 0.982 | 100.00 | 80.25 |
| 9 | 87.36 | 0.899 | 82.81 | 83.03 |
| 10 | 97.13 | 0.978 | 100.00 | 86.94 |

## INIT (6 classes)

| Metric | Value |
|---|---|
| Parameters | 5.00 M |
| Cycle accuracy | 92.39 ± 5.15 % |
| Cycle macro-F1 | 0.906 ± 0.068 |
| Walk (session) accuracy | 90.62 ± 6.99 % |
| Walk macro-F1 | 0.906 ± 0.071 |
| Walk majority-vote accuracy | 93.12 % |
| Severity accuracy | 77.87 % |

| Fold | Cycle acc | Cycle F1 | Walk acc | Severity acc |
|---|---|---|---|---|
| 1 | 83.64 | 0.773 | 81.25 | 66.67 |
| 2 | 98.21 | 0.983 | 100.00 | 52.94 |
| 3 | 87.88 | 0.877 | 87.50 | 77.78 |
| 4 | 93.18 | 0.928 | 87.50 | 77.27 |
| 5 | 94.59 | 0.930 | 100.00 | 100.00 |
| 6 | 90.70 | 0.832 | 81.25 | 100.00 |
| 7 | 98.15 | 0.973 | 87.50 | 60.71 |
| 8 | 90.91 | 0.902 | 93.75 | 71.88 |
| 9 | 86.67 | 0.864 | 87.50 | 82.86 |
| 10 | 100.00 | 1.000 | 100.00 | 88.57 |
