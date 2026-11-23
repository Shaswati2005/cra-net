# AnatoPhaseNet v7 session notes (2026-11-05)

The full conversation is in `session_transcript_2026-11-05.jsonl` in this folder (raw Claude Code log, one JSON record per line).

## What was asked and done
1. **Check `methodology.pdf` against the model.** It did not match. Its claim that key frames follow gait events was wrong: they sit at a fixed template of the stride (0, 12, 24, 36, 48, 60, 73.3, 86.7, 100 %).
2. **Write v7, its diagram and a detailed methodology.** See `README.md` for the file list and the six corrections.
3. **Train and compare.** The first background run was killed for low memory after 2 folds. The user reran both datasets from Command Prompt.

## Defects found in v6
- Key frames were read in blocks of 8, not 9, so only 851 of 3813 cycles started at phase 0.
- s9's hemiplegic walks are named `s9pat1*`, like the diplegic ones, giving 813 walks instead of 821 and 8 mixed-class walks.
- GAIT-IT frames are 224x224 and the silhouette fills the height, so height and head bob are exactly 0.

## Results (10-fold subject-exclusive, fold mean ± std)
| Dataset | Metric | v6 | v7 |
|---|---|---|---|
| GAIT-IT | Cycle accuracy | 95.47 ± 4.11 % | 95.62 ± 4.30 % |
| GAIT-IT | Walk accuracy | 96.28 ± 5.72 % | 97.48 ± 4.39 % |
| INIT | Cycle accuracy | 92.39 ± 5.15 % | 92.45 ± 4.72 % |
| INIT | Walk accuracy | 90.62 ± 6.99 % | 89.38 ± 7.42 % |

Fold-by-fold results and confusion matrices are in `RESULTS_SUMMARY.md`.

## Conclusions
- v7 matches v6 within noise. Do not claim that v7 is more accurate.
- A same-seed rerun moved a fold by up to about 1.5 points. Wilcoxon p > 0.05 on every metric.
- On GAIT-IT, 20 of the 23 wrong walks come from s7, s16 and s6, and 19 of the 23 are diplegic/Parkinson's confusions.
- Against Albuquerque et al. (94.0 %), v7 is comparable or slightly better, not clearly better: the splits differ and the fold std is 4.30.

## Next steps
- Add the v7 results to `methodology_v7.tex` and `methodology_v7.pdf`.
- Run 3 seeds per dataset (roughly 9-10 hours).
- Work on the diplegic/Parkinson's confusion.

## Running training
From Command Prompt, with heavy apps closed and sleep turned off:
```bat
cd /d C:\Users\Asus\Desktop\cra-net\cra-net\anatophasenet_v7
set PY=C:\Users\Asus\AppData\Local\Programs\Python\Python313\python.exe
set PYTHONIOENCODING=utf-8
%PY% -W ignore anato_phasenet_v7.py --dataset gait_it
%PY% -W ignore anato_phasenet_v7.py --dataset init --init_classes 6
%PY% make_results_summary.py
```
A rerun overwrites the existing results folders, so copy them elsewhere first to keep them.
