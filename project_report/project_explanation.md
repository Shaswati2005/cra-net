# Gait Pathology Project: From CRA-Net to AnatoPhaseNet v6

Oct 30, 2026 · @Bidyendu

## Overview

We went from a CRA-Net extension that scored 77.24 % even with leaked subjects to AnatoPhaseNet v6, which scores 95.47 % per cycle and 96.28 % per walk on people it has never seen.

**The task.** Classify a side-view silhouette walk into a gait pathology. The camera only gives binary silhouettes, so no face or clothing is kept.

| Dataset | Subjects | Raw data | Classes | What we classify |
| --- | --- | --- | --- | --- |
| GAIT-IT | 21 | 828 sequences, with key-frame annotations | 5: normal, diplegic, hemiplegic, neuropathic, Parkinsonian; each at 2 severity levels (lvl1, lvl2) | 3813 gait cycles |
| INIT | 10 | 160 sequences, no key frames | 8 styles (6 when the 0 and 0.5 arm levels are merged) | 589 detected cycles |

**The story in one paragraph.** CRA-Net reports 99.16 % on GAIT-IT, but its splits put the same person in training and test. We first extended it (transformer and spatio-temporal attention), then found that it memorised people and that our own "subject split" still leaked. We fixed the protocol to 10 subject-exclusive folds and kept the folds fixed (split seed 42) for every later model. From there, each version fixed one observed failure:

1. Uniform frame sampling → 9 clinical gait phases (PhaseNet).
2. Small, low-detail bodies → body regions and tight cropping (AnatoPhaseNet v1–v3).
3. Six code and data defects → the v4 redesign.
4. Walk-level decisions → severity and stride variability (v5).
5. Lost body height → the geometry branch (v6).

**Why the numbers drop before they rise.** 77 % → 89 % is not a small gain. The 77 % was measured with leakage, and the 89 % is measured on unseen people, which is much harder.

## Stage 1: CRA-Net reproduction and the Swin transformer (up to Sep 2026)

The first step was to rebuild CRA-Net, then swap its convolutional encoders for Swin transformer blocks. Both still classified frames one by one, and neither saved a test result we can report.

**CRA-Net (the paper).** CRA-Net is a frame classifier with 3 encoder and 3 decoder stages. Its key idea is reverse cross-attention fusion (RCAF), r = f × (1 − sigmoid(·)), which looks at what the main attention ignores. It has only 0.93 M parameters, and it reports 99.16 % on GAIT-IT and 99.66 % on INIT, without testing on unseen subjects.

**What each script did:**

| Script | What it is | Change from the one before |
| --- | --- | --- |
| `testing.py` | CRA-Net rebuilt: conv encoders, decoders, 3 RCAF modules, global pooling, linear classifier | Baseline |
| `swin.py` | Transformer version: the 3 conv encoders replaced by Swin transformer blocks (window attention, window 7, 2/4/8 heads) | Transformer encoder, rest of CRA-Net unchanged |
| `drop.py`, `drop2.py` | Swin version with stochastic depth (drop-path 0 / 0.1 / 0.2 per stage) and AdamW, lr 1e-4, weight decay 0.05 | Regularisation for a small dataset |
| `quick_train.py` | Trains on 40 sequences only | Smoke test that the pipeline runs |

**Common setup:** 16 frames sampled evenly from each sequence, resized to 224 × 224, random sequence split with `train_test_split`, 50 epochs.

**Why we moved on.** A transformer needs far more data than 828 sequences, and every frame was still treated as a separate image, so walking as a *movement* was never modelled. That led to adding temporal modelling in ST-CRA-Net.

## Stage 2: ST-CRA-Net, adding time (Sep 2026)

ST-CRA-Net reached 100 % training accuracy but only 77.24 % validation accuracy (macro-F1 0.788), even though the same people appeared in training and validation.

**Idea.** Keep what CRA-Net does well (RCAF) and teach it that a walk is a sequence of frames.

**What we built: ST-CRA-Net** (`st_cra_net.py`). We kept the CRA-Net encoders, decoders and RCAF modules and added three time-aware parts:

- **Foreground-gated axial attention** at the bottleneck: spatial attention that only looks at body pixels.
- **Harmonic temporal attention**: an attention bias made of cos and sin terms, so the model knows walking repeats.
- **TRAP (temporal reverse attention pooling)**: pools the frames the attention focuses on (dominant) and the frames it ignores (reverse).

Input: 12 frames sampled evenly over a whole sequence, at 128 × 128. The model had 6.59 M parameters. It was trained on a random 70/15/15 split of sequences, with the command in `instruction.txt` (`python st_cra_net.py --dataset_dir .../GAIT_IT`), and plotted with `evaluate_and_plot.py`. The log is in `results/training_metrics.csv`.

**Why it failed (two causes):**

1. **Uniform sampling destroys phase alignment.** Frame *i* of one walk shows a different point in the gait cycle than frame *i* of another walk, because walk length and walking speed differ. The temporal attention has to re-learn the alignment for every sample.
2. **Too few samples for too many parameters.** 828 sequences against 6.59 M parameters, with a 2D encoder that sees body shape, let the model memorise individual people.

The second point is the key presentation message: CRA-Net's 99 % largely reflects recognising the *person*, not the *disease*.

## Stage 3: The 3D pipeline, ST-CRA-Net v2 (Oct 10–14, 2026)

We replaced the frame-by-frame 2D model with a small 3D network. On a subject split it reached 87.96 % validation accuracy, but that split later turned out to leak people.

**What changed** (`st_cra_net_v2.py`):

- **(2+1)D convolutions.** A 3D kernel is split into a spatial part (1 × 3 × 3, "what the body looks like") and a temporal part (3 × 1 × 1, "how it moves"). It sees space and time together but needs far fewer weights than full 3D.
- **About 1 M parameters** instead of a per-frame U-Net decoder and 3 RCAF modules. The TRAP head (dominant + reverse attention over time) was kept.
- **A motion channel**: the silhouette plus the difference between consecutive frames.
- **Real augmentation.** In v1 the augmentation transform was written but never applied, so the model saw identical images every epoch. v2 adds affine, flip, erasing, speed and offset jitter, and mixup.
- **Training tricks**: GroupNorm (stable with small batches), EMA of the weights, OneCycle learning rate, mixed precision, label smoothing.
- **First subject-wise split option** (`--split subject`).

**Runs at this point:**

| Run folder | Split | Best result |
| --- | --- | --- |
| `results_v2` | random sequences | 75.00 % val (train 85.97 %) |
| `results_v2_run2` | repeat | 69.44 % val best |
| `results_v2_subj` | subject split | 87.96 % val, macro-F1 0.891 |
| `results_v2_rand`, `results_v2_test` | quick checks | stopped early |

**Takeaway.** The training/validation gap shrank from 100 vs 77 to 95 vs 88, so the smaller 3D model memorised less. The next problem was the evaluation itself (Stage 4).

## Stage 4: Fixing the evaluation protocol (Oct 14, 2026)

Our "subject split" still leaked people, so we switched to 10 folds that hold out whole people. Every number from here on is on unseen people, with the same folds for every model.

**Step 1: cross-validation harness** (`st_cra_net_v3.py`). It added subject-grouped K-fold cross-validation, mean ± std over folds, and a pooling ablation (`--pool trap | mean | max | attn`). Quick 5-fold runs, fold 1 only:

| Run folder | Pooling | Fold-1 test accuracy |
| --- | --- | --- |
| `cv_trap` | TRAP (dominant + reverse) | 65.56 % (F1 0.684) |
| `cv_mean` | plain mean | 45.56 % (F1 0.502) |
| `cv_test` | 2-epoch smoke test | 22.22 % (predicts one class) |

This supported keeping reverse-attention pooling, and it showed how much harder an honest test set is.

**Step 2: the leak** (`st_cra_net_v4.py`). In GAIT-IT the **same person (e.g. `s1`) performs all five gaits**. v2 and v3 grouped samples by `class_subject`, which gave 84 "subjects" instead of 21. One person could be in training as *normal* and in test as *diplegic*.

**The fix ("Change 0").** Group by the person ID only and follow Albuquerque et al. (2021):

- 10 folds, each holding out 2–3 whole people.
- Fold *k* tests on group *k*, validates on group *k+1* (checkpoint selection), and trains on the rest.
- Split seed 42 is fixed, so every later version is tested on exactly the same people. Accuracy differences then come from the model alone.

**Why this matters for the talk.** It explains why our numbers can't be compared directly with CRA-Net's 99 %. Their figure answers "can you recognise this person's gait again?", ours answers "can you diagnose a new patient?"

## Stage 5: ST-PhaseNet, aligning to gait phases (Oct 15, 2026)

Cutting each walk into gait cycles of 9 clinical phases took us to 89.10 % per cycle and 92.72 % per walk, on unseen people. This was the biggest single jump.

**What changed** (`st_phasenet.py`):

- **9 clinical key frames per cycle**, read from GAIT-IT's `key_frames.json`: initial contact, mid-stance, terminal stance, pre-swing, initial swing, mid-swing, terminal swing, and cycle completion.
- **One sample = one cycle**, not one whole walk. That turns 828 sequences into 3813 cycles (4.6× more samples).
- **P-TRAP head**: phase-indexed reverse attention. Each phase gets a learnable embedding and a score, and the head pools a dominant view (the most abnormal phases) and a reverse view (the compensating phases).
- **Walk-level decision**: a walk is labelled by combining all of its cycles.

**Why it works (speed invariance).** If someone walks faster, every phase event just happens earlier. Taking the frame *at each phase*, rather than at fixed times, gives the same 9 inputs at any speed. So input 3 always means "terminal stance", which fixes cause 1 from Stage 2.

**Pretrained variant** (`st_phasenet_pretrained.py`). The spatial (1 × 3 × 3) filters are initialised from ImageNet ResNet-18 (`resnet18_pretrained.pth`). The result was 89.51 % per cycle, but with a larger spread across folds (std 6.60 vs 4.56).

**Runs:** `results_phasenet_benchmark` (main 10-fold), `results_phasenet_pretrained`, and the debugging runs `test_phasenet_quick`, `test_logging_fix`, `results_phasenet_full`.

## Stage 6: AnatoPhaseNet v1–v3, looking at body parts (Oct 15–16, 2026)

Splitting the body into regions and cropping tightly around it raised accuracy to 93.34 % (v2). After that the gains stopped: v3 scored 93.01 %, two folds stayed weak, and training accuracy hit 99.96 %.

| Version | File | What was added | Why | Cycle acc. |
| --- | --- | --- | --- | --- |
| v1 | `anato_phasenet.py` | Upper region (trunk + arms, top 40 %) and lower region (legs, bottom 60 %); flip asymmetry \|h − flip(h)\| | Diseases show up in different body parts; hemiplegia affects one side | 91.08 % |
| v2 | `anato_phasenet_v2.py` | Tight bounding-box crop, resized to 96 × 96; upper−lower contrast; class-balanced loss | Limbs grow from ~35 to ~85 pixels, so the model can see joints | **93.34 %** |
| v3 | `anato_phasenet_v3.py` | Leg-only asymmetry, trunk-mobility feature, focal loss (γ = 1.5) with label smoothing 0.05 | Separate diplegic vs hemiplegic and diplegic vs Parkinsonian; focus on hard cycles | 93.01 % |

**What v3 told us.** Folds 3 (79.9 %) and 9 (86.6 %) stayed weak while training was almost perfect. The problem was generalising to new people, not model capacity. That led to a careful code review of v3 instead of adding more features.

## Stage 7: AnatoPhaseNet v4, v5 and v6 (Oct 16–29, 2026)

v4 fixed six defects found in v3, v5 added walk-level reasoning, and v6 added body-height geometry. The result was 95.47 % per cycle and 96.28 % per walk with 5.0 M parameters, fewer than the 6.59 M we started with.

### v4: the redesign (`anatophasenet_v4/`)

| # | Defect found in v3 | Fix in v4 |
| --- | --- | --- |
| 1 | GAIT-IT `_back` walks are mirror images of `_front` (correlation 0.65 as recorded, **0.996 after a flip**). The "asymmetry" feature mostly encoded walking direction. | Flip every right-to-left walk once, so all walks go the same way (`--no_canon` turns this off) |
| 2 | Only part of ResNet-18 was copied, BN statistics were not loaded, and the motion channel went through ImageNet's green-channel filters | Copy every conv and BN layer of ResNet-18 into a (2+1)D network. The temporal branch starts at zero, so at step 0 the network is exactly the pretrained 2D ResNet. |
| 3 | Zero padding in time, though a gait cycle is periodic | Circular temporal convolution: the phase after "cycle completion" is "initial contact" again |
| 4 | The final map was 3 × 3, so the upper/lower split was 1 row vs 2 | Drop ResNet layer 4 → 6 × 6 map → 6 horizontal body strips + 1 global part |
| 5 | Left-right flip doesn't measure leg symmetry in a side view | Symmetry in *time*: compare each phase with the phase half a cycle later (the other leg) |
| 6 | Head with 2560-d input and about 3 M parameters for about 17 training people | One small classifier per body part, averaged (an ensemble that reduces variance) |

Also added: 3 input channels (silhouette, motion, deviation from the cycle average), ±2-frame phase jitter, erosion/dilation noise, square-root class balancing, and test-time averaging over 3 jitter views. **INIT support** was new too: INIT has no key frames, so cycles are detected from the autocorrelation of the bounding-box aspect ratio.

**Effect:** 94.05 % (std 5.38 → 4.04); fold 3 went from 79.9 to 90.9 %. The TSM ablation (2D ResNet + temporal shift) scored 94.04 %, so most of the gain came from the data and head fixes, not from the 3D backbone. On INIT, a cycle-detector bug was fixed (88.15 → 93.59 % on 6 classes).

### v5: severity and stride variability (`anatophasenet_v5/`)

- **Severity head**: also predict normal / mild (lvl1) / severe (lvl2). The labels come free from the file names. Severity accuracy is 79.59 % with the head and 35.96 % (chance) without it, and cycle accuracy doesn't change.
- **Multi-cycle variability**: feed 2–6 consecutive cycles of one walk, and classify the walk from the mean, standard deviation and successive differences of their embeddings. Stride-to-stride variability is a clinical marker, especially in Parkinson's.

**Effect:** walk accuracy 94.57 → 95.71 %, cycle accuracy 94.05 → 94.81 %. INIT 6-class: 93.27 %.

### v6: geometry branch (`anatophasenet_v6/`)

- **The problem.** Cropping every frame to the same height deletes body height over time: crouch, stoop and vertical bob. Those are exactly what separates diplegic from Parkinsonian gait.
- **The fix.** Measure 9 numbers per frame on the raw silhouette, before cropping: height, width, lower- and upper-body width, head bob, centroid height, trunk lean, forward mass offset and fill ratio. Each is divided by the walk's median height, so camera distance and body size cancel out. A small circular temporal CNN encodes them, with its own classifier.
- **Effect:** 94.81 → **95.47 %** cycle, 95.71 → **96.28 %** walk, severity 80.68 %. Diplegic recall 84.0 → 87.9 %, and diplegic→Parkinsonian errors fell from 91 to 71.
- **Subject-adversarial training (tested, not kept).** A subject classifier with gradient reversal, meant to remove identity cues. It scored 95.16 %, and fold 9 went down, so it stays off.

## What we ran, and where it is

We ran different things at different points because each stage asked a different question: does it train, does it generalise, which part helps, and does it hold on a second dataset.

| When (2026) | Script / command | Dataset | Question it answered | Output folder |
| --- | --- | --- | --- | --- |
| up to Sep | `testing.py`, `swin.py`, `drop.py`, `drop2.py`, `quick_train.py` | GAIT-IT | Can we rebuild CRA-Net and swap in a transformer? | none saved |
| Sep 11 | `python st_cra_net.py --dataset_dir .../GAIT_IT`, then `evaluate_and_plot.py` | GAIT-IT | Does adding time help? (random split) | `results/` |
| Oct 10–14 | `st_cra_net_v2.py` (random, subject split) | GAIT-IT | Does a small 3D model memorise less? | `results_v2*` |
| Oct 14 | `st_cra_net_v3.py --pool trap / mean` | GAIT-IT | Is TRAP pooling better than mean pooling? | `cv_trap`, `cv_mean`, `cv_test` |
| Oct 14 | `st_cra_net_v4.py` | GAIT-IT | Protocol fix: hold out whole people | (protocol only) |
| Oct 15 | `st_phasenet.py`, `st_phasenet_pretrained.py` | GAIT-IT | Do 9 gait phases help? Does ImageNet help? | `results_phasenet_benchmark`, `results_phasenet_pretrained` |
| Oct 15–16 | `anato_phasenet.py`, `_v2.py`, `_v3.py` | GAIT-IT | Do body regions, cropping and focal loss help? | `results_anatophasenet`, `_v2`, `_v3` |
| Oct 16–28 | `anato_phasenet_v4.py --dataset gait_it` (+ `--backbone tsm`) | GAIT-IT | Do the six fixes help? Is the 3D backbone needed? | `results_gait_it_r2plus1d`, `results_gait_it_tsm` |
| Oct 16–28 | `anato_phasenet_v4.py --dataset init --init_classes 6 / 8` | INIT | Does it work on a second dataset? | `results_init6_r2plus1d` (93.59 %), `results_init8_r2plus1d` (85.56 %) |
| Oct 28 | `anato_phasenet_v5.py --dataset gait_it` (+ `--no_severity`) | GAIT-IT | Do severity and multi-cycle help? | `results_gait_it_r2plus1d_full`, `_nosev` |
| Oct 28 | `anato_phasenet_v5.py --dataset init --init_classes 6` | INIT | v5 on INIT | `results_init6_r2plus1d_full` (93.27 %) |
| Oct 29 | `anato_phasenet_v6.py --dataset gait_it` | GAIT-IT | Does body geometry help? | `results_gait_it_r2plus1d_full` |
| Oct 29 | `anato_phasenet_v6.py --dataset gait_it --adv_subject` | GAIT-IT | Does removing identity help? | `results_gait_it_r2plus1d_adv` |
| **still to run** | `anato_phasenet_v6.py --dataset init --init_classes 6` (+ `--adv_subject`) | INIT | v6 on INIT, compared with v5's 93.27 % | `results_init6_r2plus1d_full`, `_adv` |

**How to compare runs fairly.** Every GAIT-IT run from Stage 5 on uses the same 10 folds (split seed 42), and INIT uses leave-one-subject-out folds. Each results folder holds `benchmark_summary.json` (the headline numbers), per-fold CSVs, confusion matrices, training curves and `fold_N_test_predictions.npz`.

**Results across versions (GAIT-IT, unseen people)**

(The online doc shows this as a chart; the export can't carry charts, so here are the same numbers as a table.)

| Model | Params | Cycle acc. (%) | Walk acc. (%) |
| --- | --- | --- | --- |
| ST-PhaseNet | – | 89.10 ± 4.56 | 92.72 ± 4.49 |
| + ImageNet init | – | 89.51 ± 6.60 | 92.64 ± 7.25 |
| v1 regions | – | 91.08 ± 6.35 | 93.26 ± 6.25 |
| v2 + crop | – | 93.34 ± 5.31 | 94.72 ± 6.99 |
| v3 + focal | – | 93.01 ± 5.38 | 94.54 ± 6.63 |
| v4 TSM | 3.62 M | 94.04 ± 3.30 | 94.13 ± 5.35 |
| v4 (2+1)D | 4.13 M | 94.05 ± 4.04 | 94.57 ± 5.78 |
| v5 | 4.93 M | 94.81 ± 4.63 | 95.71 ± 6.37 |
| v6 + adversary | 5.37 M | 95.16 ± 4.74 | 96.27 ± 5.68 |
| **v6** | **5.00 M** | **95.47 ± 4.11** | **96.28 ± 5.72** |

The big steps were phase alignment (PhaseNet) and cropping (v2). v4 mostly made results steadier across people, and v6 is best on every summary metric. ST-CRA-Net's 77.24 % is left off because it was measured with leaked subjects.

## Presentation cheat sheet

The one line to remember: *every change was a fix for a failure we observed, and every number is measured on people the model has never seen.*

**Talking points, in order:**

1. Problem: diagnose gait pathology from cheap side-view silhouettes.
2. CRA-Net's 99 % shares people between training and test. Our extension (ST-CRA-Net, then Swin and (2+1)D) memorised people.
3. We fixed the protocol: 10 folds of held-out people, the same folds for every model.
4. Phase alignment (9 clinical phases) gave the biggest jump: 89 % on unseen people.
5. Body parts and tight cropping: 93 %. Then the gains stopped, so we audited the code.
6. The v4 audit found six defects (mirrored walks, half-loaded pretraining, zero padding on a cycle, and more): 94 %, steadier across people.
7. v5 added severity (free labels, 80 %) and stride variability: walk accuracy 95.7 %.
8. v6 added posture geometry that cropping had removed: 95.47 % cycle, 96.28 % walk, 5.0 M parameters.

**Likely questions:**

- *Why is your accuracy lower than CRA-Net's?* Different question. Theirs re-identifies people already seen; ours diagnoses new ones. ST-CRA-Net got only 77 % even with leakage.
- *Why (2+1)D and not a transformer?* 828 sequences is too little for attention to learn from scratch. (2+1)D reuses ImageNet weights and starts as the exact 2D ResNet. The TSM ablation (94.04 %) shows the backbone matters less than the data fixes.
- *Why 9 key frames?* They make the input invariant to walking speed, and each input slot means one biomechanical event.
- *Why does hemiplegia score so well?* The half-cycle symmetry term compares each leg with the other. Hemiplegia is one-sided, so it breaks that symmetry.
- *What is still wrong?* Diplegic vs Parkinsonian (71 and 43 cycles confused: both are bilateral and flexed), and folds 3 and 9 (87.6 % and 87.4 %), which are tied to specific people. Subject-adversarial training didn't fix them.
- *Limitations?* Both datasets use actors simulating the pathologies, and all results come from one training seed.

**Before the presentation:**

- [ ] Run v6 on INIT 6-class (`--dataset init --init_classes 6`), and compare with v5's 93.27 %
- [ ] Optional: v6 INIT 6-class with `--adv_subject`
- [ ] Optional: multi-seed runs (`--seed 43`) for error bars
- [ ] Next steps for the slide: weak folds 3/9, diplegic vs Parkinsonian, saliency maps, cross-dataset test
