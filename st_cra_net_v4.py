#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ST-CRANet v4 (step 1): corrected, person-independent evaluation protocol.

Problem found in v2/v3: in GAIT-IT the SAME person (e.g. 's1') performs all five gait
types, but v2/v3 grouped by "<class>_<subject>", so one physical person could be in
train (as 'normal') and in test (as 'diplegic').

v4 groups by the person ID only (the 'sN' folder name, shared across classes) and
follows the protocol of Albuquerque et al. (Sensors 2021): K-fold (default 10) cross-validation
where whole people are held out; ~2-3 people per test fold. The next fold's people are
used as validation (checkpoint selection), so the test fold is never used for selection.

Everything else (model, augmentation, training, pooling ablation flags) is v3's.

Quick corrected baseline (2 folds only):
  python -u st_cra_net_v4.py --run_folds 2 --epochs 60 --out_dir v4_baseline_quick
Full protocol:
  python -u st_cra_net_v4.py --epochs 60 --out_dir v4_baseline
"""

import sys
import random
import numpy as np

import st_cra_net_v3 as v3


def make_person_folds(labels, groups, k, seed):
    """Assign each PERSON (id shared across classes) to a fold. Returns fold id per sequence."""
    persons = [g.split('_', 1)[1] for g in groups]  # 'label_sN' -> 'sN'
    uniq = sorted(set(persons), key=lambda s: (len(s), s))
    rng = random.Random(seed)
    rng.shuffle(uniq)
    fold_of = {p: i % k for i, p in enumerate(uniq)}
    fold_id = np.array([fold_of[p] for p in persons])

    n_per_fold = np.bincount([fold_of[p] for p in uniq], minlength=k)
    print(f"[v4] {len(uniq)} distinct people -> {k} folds, people per fold: {n_per_fold.tolist()}")
    # sanity: no person appears in two folds (by construction), and every class appears in each fold
    lab = np.asarray(labels)
    for f in range(k):
        present = sorted(set(lab[fold_id == f].tolist()))
        if len(present) < 5:
            print(f"[v4][WARN] fold {f+1} is missing classes: has {present}")
    return fold_id


if __name__ == '__main__':
    v3.make_folds = make_person_folds  # v3.main() looks this up at call time
    if '--folds' not in sys.argv:
        sys.argv += ['--folds', '10']
    v3.main()
