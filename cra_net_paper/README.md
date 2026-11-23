# CRA-Net (Bodepu et al. 2025), re-implemented

The frame-level CRA-Net from `../cra_net_paper.pdf`, built from Eqs. 1–13 and Fig. 1:
- **Size:** 0.88 M parameters (paper: 0.934 M).
- **Training:** the paper's setup — Adam 1e-3, 50 epochs, batch 32, checkpoint at best validation accuracy, flip and rotation augmentation.

**Purpose:** measure how much of the paper's 99 % comes from the split. The same model is run three ways:

| `--split` | Train / val / test | Same person in train and test? |
|---|---|---|
| `frame` | random frames, 70/15/15 (probably what the paper did) | yes, even the same walk |
| `sequence` | random walks, 70/15/15 | yes, never the same walk |
| `subject` | our 10 folds: test on group k, validate on group k+1 | **no**: same test people as AnatoPhaseNet v6 |

Only `subject` is comparable with our v6 numbers.

The first run on a dataset builds a frame cache in `cra_net_paper/cache/` (every 4th frame, 64×64):
- GAIT-IT: ~5–10 min
- INIT: ~1 min

## Run (from this folder, Command Prompt)
```
set PY=C:\Users\Asus\AppData\Local\Programs\Python\Python313\python.exe
cd C:\Users\Asus\Desktop\cra-net\cra-net\cra_net_paper

:: INIT, 8 classes (fast, ~1 h each)
%PY% cra_net_paper.py --dataset init --init_classes 8 --split frame       :: -> results_init8_frame
%PY% cra_net_paper.py --dataset init --init_classes 8 --split subject     :: -> results_init8_subject
%PY% cra_net_paper.py --dataset init --init_classes 6 --split subject     :: -> results_init6_subject

:: GAIT-IT (frame ~1 h, subject ~8 h for 10 folds)
%PY% cra_net_paper.py --dataset gait_it --split frame                     :: -> results_gait_it_frame
%PY% cra_net_paper.py --dataset gait_it --split sequence                  :: -> results_gait_it_sequence
%PY% cra_net_paper.py --dataset gait_it --split subject                   :: -> results_gait_it_subject
```
Quick look first: `--split subject --fold_ids 1 2` runs only folds 1 and 2 (compare with v6: 97.4 %, 96.0 %).

## Where the results are
Each `results_*` folder holds:
- **`benchmark_summary.json`:** the headline numbers — frame accuracy, walk accuracy, macro-F1, QWK, per-class precision / recall / F1, confusion matrix, parameters, inference time.
- **`<fold>_result.json`:** one per fold or split, with the epoch-by-epoch history.

The final console lines print the same summary:
```
==== CRA-Net | gait_it | split=subject | 10 run(s) ====
Frame accuracy .. +/- ..% | walk accuracy .. +/- ..% | macro-F1 .. | QWK ..
```

## What to compare
| CRA-Net (`subject`) | AnatoPhaseNet v6 |
|---|---|
| frame accuracy | cycle accuracy: GAIT-IT 95.47 %, INIT-6 92.39 % (v5: 93.27 %) |
| walk accuracy | walk accuracy: GAIT-IT 96.28 %, INIT-6 90.63 % |
| INIT-8 frame accuracy | v4 INIT-8: 85.56 % |

**Settings the paper doesn't give, chosen here:**
- 16/32/64/128 channels and a 64×64 input;
- silhouette bounding-box crop (`--no_crop` turns it off);
- ±10° rotation (`--rotation`);
- the three CAB outputs pooled and concatenated.

Use `--frame_stride 1` to train on every frame (4× slower).
