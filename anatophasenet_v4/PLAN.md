# AnatoPhaseNet v4 — plan to improve on v3

## Baseline (v3, GAIT-IT, 10-fold subject-exclusive)
- Test cycle accuracy: **93.0 ± 5.4 %**; session accuracy: 94.5 ± 6.6 %
- Weak folds: fold 3 (79.9 %) and fold 9 (86.6 %). Everything else is 92–98 %.
- Train accuracy reaches 99.96 %, so the gap is generalisation to unseen subjects, not capacity.
- v3 has **no INIT loader**: it only reads GAIT-IT's `key_frames.json` layout.

## What is limiting v3 (found by reading the code + data)

| # | Issue | Evidence | Fix in v4 |
|---|-------|----------|-----------|
| 1 | GAIT-IT `_back` sequences are horizontal mirrors of `_front`. The model has to learn both walking directions, and the "leg asymmetry" `abs(h - flip_W(h))` mostly encodes direction. | Mean-silhouette correlation front vs back: 0.65 as-is, **0.996 after a horizontal flip** | Flip `_back` (GAIT-IT) and left-walking (INIT) sequences at cache time so every cycle walks the same way |
| 2 | Pretrained weights barely used: only `conv1` of the **first** block per stage is copied, BN stats never loaded, second block of each stage missing, and channel 2 (frame diff) gets ImageNet G-channel filters | `_inflate_pretrained_weights` | 3D (2+1)D backbone (main model): 2 blocks per stage, every spatial conv + BN inflated from ResNet-18, zero-initialised cyclic temporal branch (verified: at initialisation it reproduces the 2D ResNet exactly). Inputs = [silhouette, motion, GEI deviation]. A 2D ResNet + temporal-shift backbone is kept as an ablation (`--backbone tsm`) |
| 3 | The final feature map is **3×3** at 96 px, so the "upper/lower" split is 1 row vs 2 rows and the flip-asymmetry runs over 3 columns | 96 → /32 | Drop layer4 → 6×6 map; 6 horizontal body strips + 1 global strip (GaitSet / PCB-style part pooling) |
| 4 | Temporal modelling is one 3-tap conv per stage, zero-padded at both ends of a **periodic** cycle | `FactorizedSpatiotemporalBlock` | Cyclic temporal shift (TSM, no extra params) in every block + circular temporal conv in the head |
| 5 | Head: 5 × 512 = 2560-d features of which 2 are linear combinations of the others (`upper - lower`, velocity), plus a 5120 → 256 FC, about 3M params on 23 subjects | `Anatomical_PTRAP_Head_v3` | Per-part 128-d projections; [attention pool, max pool, half-cycle symmetry] per part; per-part classifiers averaged (an ensemble that also regularises) |
| 6 | Bilateral symmetry in a sagittal view lives in **time**, not in left-right image flips: the stance of one leg (phases 0–4) vs the other (4–8) | — | Symmetry feature = mean `abs(f[t] - f[t + half cycle])` per body part |
| 7 | Only spatial augmentation; the 9 key frames are always identical | `gpu_augment_phases` | Phase jitter (±2 frames around each key frame), silhouette dilation/erosion noise, vectorised erasing |
| 8 | Class imbalance: normal 202 cycles vs parkinson 1256 | cache label counts | sqrt-inverse-frequency sampling (milder than the hard class weights v3 dropped) |
| 9 | Evaluation uses a single view | — | Temporal TTA: average over the 3 jitter views; session vote uses mean log-probability |

## INIT support (new)
- Reads `INIT_8/silhouettes/<class>/<seq>`, 10 subjects, 160 sequences, no key frames.
- Gait cycles are detected from the bounding-box aspect-ratio signal: the full-cycle period comes from autocorrelation (40–110 frames), cycles are cut between every other step peak, and limping gaits are anchored on their dominant peak. That gives ~660 cycles, mostly 60–75 frames each. Full-body (`fb`) cycles are noisier.
- `--init_classes 8` (all classes) or `6` (a-l0 + a-l0.5 merged, a-r0 + a-r0.5 merged, matching the dataset's `silhouettes_6` protocol).
- 10 folds = leave-one-subject-out, with the next subject used for validation.

## Fair comparison
- The subject-exclusive splitter and seed are unchanged from v3, so GAIT-IT folds hold the same subjects.
- The validation fold still picks the checkpoint. Validation accuracy is reported, but **test** accuracy on unseen subjects is the number to compare.

## Ablations (one flag each)
| Question | Flag |
|---|---|
| 3D (2+1)D vs 2D + temporal shift | `--backbone tsm` |
| Does direction canonicalisation help? | `--no_canon` |
| Phase jitter on/off | `--jitter 0` |
| Higher-resolution parts (12×12 map) | `--last_stride 1` |
| Number of body strips | `--num_parts 4/6/8` |
| Class-balanced sampling | `--balance none` |
| Part auxiliary loss | `--part_loss_weight 0` |
| TTA | `--no_tta` |
| Checkpoint criterion | `--select f1/loss/last` |

## How to run
```
cd anatophasenet_v4
python anato_phasenet_v4.py --dataset gait_it                     # results_gait_it_r2plus1d/
python anato_phasenet_v4.py --dataset gait_it --backbone tsm      # results_gait_it_tsm/ (ablation)
python anato_phasenet_v4.py --dataset init --init_classes 8       # results_init8_r2plus1d/
python anato_phasenet_v4.py --dataset init --init_classes 6       # results_init6_r2plus1d/
python anato_phasenet_v4.py --dataset gait_it --fold_ids 3 9      # quick check on v3's weakest folds
```
Caches are built once into `anatophasenet_v4/cache/` (GAIT-IT is about 1 GB because it stores 3 jitter views).
