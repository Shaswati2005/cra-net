# AnatoPhaseNet v6

v5 plus two new components (each can be switched on or off for ablations):

1. **Biomechanical geometry branch (on by default, `--no_geom` to switch off).**
   - **Why it's needed:** v4 and v5 crop and rescale every frame separately, so every silhouette ends up the same height. That removes:
     - crouch and stoop (the body's height changing),
     - vertical bob,
     - stride width relative to body height.
   - **What v6 does:** it measures 9 numbers per frame on the raw silhouette, before cropping:
     - height,
     - width,
     - lower- and upper-body width,
     - head bob,
     - centroid height,
     - trunk lean,
     - forward mass offset,
     - fill ratio.
   - **Normalisation:** each value is divided by the walk's median height.
   - **Into the model:** a small cyclic temporal CNN encodes the 9 numbers. Its output gets its own classifier, averaged with the body-part classifiers, and is joined to the descriptor used by the severity and session heads.
   - **Target:** diplegic ↔ parkinson confusion.
2. **Subject-adversarial training (off by default, `--adv_subject` to switch on).**
   - **What it does:** a subject classifier sits behind a gradient-reversal layer, so the backbone learns features that don't identify the person.
   - **Target:** subject-specific errors (e.g. GAIT-IT fold 9).

`--no_geom` without `--adv_subject` is the v5 model, so the v5 results serve as that ablation.

It reuses the v4 image cache (`../anatophasenet_v4/cache/`) and v5's heads, so `anatophasenet_v4` and `anatophasenet_v5` must stay next to this folder.
The first run on a dataset builds a small geometry cache in `anatophasenet_v6/cache/`:
- INIT: ~5 min
- GAIT-IT: ~15–25 min

## Run (from this folder, Command Prompt)
```
set PY=C:\Users\Asus\AppData\Local\Programs\Python\Python313\python.exe
cd C:\Users\Asus\Desktop\cra-net\cra-net\anatophasenet_v6

%PY% anato_phasenet_v6.py --dataset gait_it                      :: geometry branch      -> results_gait_it_r2plus1d_full
%PY% anato_phasenet_v6.py --dataset gait_it --adv_subject        :: geometry + adversary -> results_gait_it_r2plus1d_adv
%PY% anato_phasenet_v6.py --dataset init --init_classes 6
%PY% anato_phasenet_v6.py --dataset init --init_classes 8
```
Compare with v5 `results_gait_it_r2plus1d_full` (94.81 % cycle, 95.71 % walk).
