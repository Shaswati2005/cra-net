# AnatoPhaseNet v5

v4 plus two new components (each can be switched off for ablations):

1. **Severity-aware multi-task learning.** A second head predicts {normal, mild, severe}.
   - GAIT-IT: from `lvl1` / `lvl2` in the sequence names.
   - INIT 6-class: `a-*0.5` = mild, `a-*0` = severe, `nm` = normal.
   - INIT 8-class: disabled, because severity is already part of the class.
2. **Multi-cycle stride-to-stride variability.** Training samples 2–6 consecutive cycles from one walk. The walk is classified from the mean, standard deviation and successive differences of its cycle embeddings.

It reuses v4's code and cache (`../anatophasenet_v4/`), so `anatophasenet_v4` must stay next to this folder.

## Run (from this folder, Command Prompt)
```
set PY=C:\Users\Asus\AppData\Local\Programs\Python\Python313\python.exe

%PY% anato_phasenet_v5.py --dataset gait_it                         :: full model
%PY% anato_phasenet_v5.py --dataset gait_it --no_severity           :: ablation: no severity task
%PY% anato_phasenet_v5.py --dataset gait_it --no_multicycle         :: ablation: no variability module
%PY% anato_phasenet_v5.py --dataset gait_it --no_sym                :: ablation: no half-cycle symmetry
%PY% anato_phasenet_v5.py --dataset gait_it --backbone tsm          :: 2D backbone
%PY% anato_phasenet_v5.py --dataset init --init_classes 6
%PY% anato_phasenet_v5.py --dataset init --init_classes 8
%PY% anato_phasenet_v5.py --dataset gait_it --seed 43               :: repeat with another training seed (same folds)
```
Each run writes to `results_<dataset>_<backbone>_<variant>[_seedN]/`, which contains:
- `benchmark_summary.json`: cycle, session (vote and multi-cycle) and severity accuracy
- per-fold CSVs, plots and `fold_N_test_predictions.npz`, which holds the logits for offline ensembling

The folds are fixed by `--split_seed 42`, identical to v3 and v4.
