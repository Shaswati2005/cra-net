"""
===============================================================================
AnatoPhaseNet v7: corrected, self-contained version of the final model
===============================================================================

The architecture, losses and training settings are those of v6 (geometry branch on,
subject adversary off, (2+1)D backbone). This file no longer imports v4/v5.

Corrections against v6:
1. GAIT-IT cycles. key_frames.json stores nine entries per cycle, and the last entry of a
   cycle equals the first entry of the next one. v6 advanced by 8 entries, so only the
   first cycle of a walk started at phase 0 and every later cycle was shifted by one more
   phase and held a repeated frame. v7 advances by 9.
2. Walk identity. seq_id now contains the gait class. s9's hemiplegic recordings carry
   the name s9pat1*, like s9's diplegic ones, so v6 merged 8 pairs of recordings into
   8 mixed-class walks.
3. Geometry signals. GAIT-IT frames arrive normalised (224 x 224, silhouette filling the
   height), so height and head bob are constant there. Signals that are constant over
   a dataset are dropped: 7 signals on GAIT-IT, 9 on INIT.
4. Geometry noise scale. The per-signal standard deviation used by the augmentation comes
   from the training cycles of the fold, not from all cycles of the dataset.
5. Crop. The crop box now includes the last row and column, so padding is equal on both
   sides, and the longer side is exactly the target size.
6. Guards. The run stops if a seq_id repeats, a walk mixes labels or subjects, a key-frame
   list does not follow the nine-entry layout, or a subject has two roles in a fold.

Ablation switches: --no_geom, --no_severity, --no_multicycle, --no_sym, --adv_subject,
--backbone tsm.
===============================================================================
"""

import os
import re
import sys
import math
import json
import random
import argparse
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F


HERE = os.path.dirname(os.path.abspath(__file__))           # .../anatophasenet_v7
PARENT = os.path.dirname(HERE)                              # project root (dataset, resnet18_pretrained.pth)


# =======================================================================
# 0. CONFIGURATION & CLASS DEFINITIONS
# =======================================================================
GAIT_IT_CLASSES = {
    'normal_silhouettes': 0,
    'diplegic_silhouettes': 1,
    'hemiplegic_silhouettes': 2,
    'neuropathic_silhouettes': 3,
    'parkinson_silhouettes': 4,
}
GAIT_IT_NAMES = ['normal', 'diplegic', 'hemiplegic', 'neuropathic', 'parkinson']

# INIT folder names -> label, for the 8-class and the merged 6-class protocol
INIT_8 = ['nm', 'a_l0', 'a_l0.5', 'a_r0', 'a_r0.5', 'l_l0.5', 'l_r0.5', 'fb']
INIT_6_MERGE = {'a_l0.5': 'a_l0', 'a_r0.5': 'a_r0'}

NUM_PHASES = 9
PHASE_NAMES = ['IC', 'LR', 'MST', 'TST', 'PS', 'ISW', 'MSW', 'TSW', 'COMP']
IMG_EXT = ('.jpg', '.png', '.jpeg')

# =======================================================================
# 1. METRICS & VISUALIZATIONS (Pure-NumPy, No sklearn/scipy)
# =======================================================================
def compute_macro_f1(targets, preds, num_classes):
    f1_list = []
    for c in range(num_classes):
        tp = np.sum((preds == c) & (targets == c))
        fp = np.sum((preds == c) & (targets != c))
        fn = np.sum((preds != c) & (targets == c))
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        if precision + recall > 0:
            f1 = 2 * precision * recall / (precision + recall)
        else:
            f1 = 0.0
        f1_list.append(f1)
    return float(np.mean(f1_list))

def confusion_matrix(targets, preds, num_classes):
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(targets, preds):
        cm[t, p] += 1
    return cm

def plot_fold_curves(history, output_path, fold_idx):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    epochs = range(1, len(history['train_loss']) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), dpi=200)

    ax1.plot(epochs, history['train_loss'], label='Train Loss', color='#1f77b4', lw=2)
    ax1.plot(epochs, history['val_loss'], label='Val Loss', color='#d62728', lw=2)
    ax1.set_title(f'Fold {fold_idx} - Loss Progression', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Epoch', fontsize=10)
    ax1.set_ylabel('Loss', fontsize=10)
    ax1.grid(True, linestyle='--', alpha=0.5)
    ax1.legend(fontsize=10)

    ax2.plot(epochs, history['train_acc'], label='Train Acc', color='#1f77b4', lw=2)
    ax2.plot(epochs, history['val_cycle_acc'], label='Val Cycle Acc', color='#ff7f0e', lw=2)
    ax2.plot(epochs, history['val_sess_acc'], label='Val Session Acc', color='#2ca02c', lw=2, linestyle='--')
    ax2.set_title(f'Fold {fold_idx} - Accuracy Progression', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Epoch', fontsize=10)
    ax2.set_ylabel('Accuracy (%)', fontsize=10)
    ax2.grid(True, linestyle='--', alpha=0.5)
    ax2.legend(fontsize=10)

    fig.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()

def plot_confusion_matrix_chart(cm, class_names, output_path, title="Confusion Matrix"):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 6), dpi=200)
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=class_names,
        yticklabels=class_names,
        title=title,
        ylabel='True Label',
        xlabel='Predicted Label'
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, format(cm[i, j], 'd'),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontweight='bold'
            )

    fig.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()

# =======================================================================
# 2. DATASET BUILDERS & CACHE
# =======================================================================
# Every cycle is described by a [9][J] grid of frame indices (9 phases x J jitter
# offsets) into one sequence. The cache stores uint8 tensors [N, J, 9, H, W].

def crop_and_resize_silhouette(im, target_size=96, pad_ratio=0.08):
    arr = np.array(im)
    y_indices, x_indices = np.where(arr > 30)

    if len(y_indices) == 0 or len(x_indices) == 0:
        return im.resize((target_size, target_size), Image.BILINEAR)

    y_min, y_max = y_indices.min(), y_indices.max()
    x_min, x_max = x_indices.min(), x_indices.max()

    h = y_max - y_min + 1
    w = x_max - x_min + 1

    pad_y = int(h * pad_ratio)
    pad_x = int(w * pad_ratio)

    # PIL crop boxes exclude the right and bottom edge, hence the + 1
    y_min = max(0, y_min - pad_y)
    y_max = min(arr.shape[0], y_max + pad_y + 1)
    x_min = max(0, x_min - pad_x)
    x_max = min(arr.shape[1], x_max + pad_x + 1)

    cropped = im.crop((x_min, y_min, x_max, y_max))

    cw, ch = cropped.size
    scale = target_size / max(cw, ch)
    nw, nh = max(1, int(round(cw * scale))), max(1, int(round(ch * scale)))
    resized = cropped.resize((nw, nh), Image.BILINEAR)

    canvas = Image.new('L', (target_size, target_size), 0)
    paste_x = (target_size - nw) // 2
    paste_y = (target_size - nh) // 2
    canvas.paste(resized, (paste_x, paste_y))
    return canvas

def list_frames(seq_dir):
    return sorted(
        [f for f in os.listdir(seq_dir) if f.lower().endswith(IMG_EXT)],
        key=lambda x: int(re.findall(r'\d+', x)[-1]) if re.findall(r'\d+', x) else x
    )

def jitter_grid(phase_idx, offsets, n_frames):
    """phase_idx: 9 frame indices -> [9][J] grid clamped to the sequence."""
    return [[int(min(max(p + o, 0), n_frames - 1)) for o in offsets] for p in phase_idx]

def collect_gait_it(root_dir, offsets):
    """Returns sequences: list of dicts(dir, files, flip, label, person, seq_id, cycles)."""
    seqs = []
    for gait_type, label in GAIT_IT_CLASSES.items():
        gait_p = os.path.join(root_dir, gait_type)
        if not os.path.isdir(gait_p):
            continue
        for subj in sorted(os.listdir(gait_p)):
            side_view_p = os.path.join(gait_p, subj, 'silhouettes', 'side_view')
            meta_p = os.path.join(side_view_p, 'metadata', 'key_frames.json')
            if not os.path.isfile(meta_p):
                continue
            try:
                with open(meta_p, 'r') as f:
                    meta = json.load(f)
            except Exception:
                continue

            for seq_name, kf in meta.items():
                seq_dir = os.path.join(side_view_p, seq_name)
                if not os.path.isdir(seq_dir):
                    continue
                files = list_frames(seq_dir)
                if len(files) == 0:
                    continue

                # Nine entries per annotated cycle; entry 8 of a cycle equals entry 0 of the next
                if len(kf) % NUM_PHASES or any(kf[i] != kf[i + 1] for i in range(NUM_PHASES - 1, len(kf) - 1, NUM_PHASES)):
                    raise ValueError(f"Unexpected key-frame layout in {seq_dir}")
                cycles = []
                for start_i in range(0, len(kf), NUM_PHASES):
                    cycle_idx = kf[start_i:start_i + NUM_PHASES]
                    if max(cycle_idx) < len(files):
                        cycles.append(jitter_grid(cycle_idx, offsets, len(files)))
                if cycles:
                    seqs.append(dict(
                        dir=seq_dir, files=files, label=label, person=subj,
                        # The class is part of the id: s9's hemiplegic walks are named s9pat1*,
                        # like s9's diplegic ones
                        seq_id=f"{gait_type.split('_')[0]}_{subj}_{seq_name}",
                        # '_back' walks the opposite way to '_front' (verified mirror images)
                        flip=seq_name.endswith('_back'),
                        cycles=cycles,
                    ))
    return seqs

def _moving_average(x, k):
    pad = k // 2
    return np.convolve(np.pad(x, pad, mode='edge'), np.ones(k) / k, mode='valid')

def detect_init_cycles(widths, heights, min_period=40, max_period=110):
    """
    Gait cycles from the silhouette aspect ratio (bbox width / height), which peaks at
    double support twice per cycle. Limping gaits alternate a big and a small peak, so the
    step-level period is unreliable; instead the FULL-cycle period is taken from the
    autocorrelation in [min_period, max_period] frames (INIT steps are ~30-37 frames, so
    the half period falls outside this window; the smallest strong lag is taken so that
    3 half-periods of a symmetric gait are not mistaken for one cycle).
    A cycle spans step peak i -> step peak i+2. For asymmetric (limping) gaits only cycles
    that start on the dominant peak are kept, so phase 0 means the same event in every cycle.
    Returns a list of (start, end) frame indices.
    """
    n = len(widths)
    r = _moving_average(widths / np.maximum(heights, 1.0), 5)
    w = r - _moving_average(r, 61)
    w = w / (w.std() + 1e-8)

    hi = min(max_period, n // 2)
    if hi <= min_period:
        period = max(8, n // 2)
    else:
        ac = np.array([np.dot(w[:-lag], w[lag:]) / (n - lag) for lag in range(min_period, hi + 1)])
        # Interior local maxima only: the window edge (e.g. lag 40, on the shoulder of the
        # half-period peak) must not be mistaken for the cycle period.
        peaks_ac = [i for i in range(1, len(ac) - 1) if ac[i] >= ac[i - 1] and ac[i] >= ac[i + 1]]
        if peaks_ac:
            top = max(ac[i] for i in peaks_ac)
            period = min_period + next(i for i in peaks_ac if ac[i] >= 0.8 * top)
        else:
            period = min_period + int(np.argmax(ac))

    # Step peaks (one per double support), non-maximum suppression at ~0.35 cycle
    local_max = [t for t in range(1, n - 1) if w[t] >= w[t - 1] and w[t] >= w[t + 1]]
    peaks = []
    for t in sorted(local_max, key=lambda t: -w[t]):
        if all(abs(t - u) >= 0.35 * period for u in peaks):
            peaks.append(t)
    peaks.sort()

    cycles = [(peaks[i], peaks[i + 2]) for i in range(len(peaks) - 2)
              if 0.7 * period <= peaks[i + 2] - peaks[i] <= 1.3 * period
              and w[peaks[i]] >= w[peaks[i + 1]] - 0.3]
    if not cycles:
        # Fallback: fixed windows of one estimated cycle with half-cycle stride
        cycles = [(s, s + period) for s in range(0, n - period, max(1, period // 2))]
    return cycles

def collect_init(root_dir, offsets, n_classes):
    names = INIT_8 if n_classes == 8 else [c for c in INIT_8 if c not in INIT_6_MERGE]
    seqs = []
    for folder in INIT_8:
        cls_p = os.path.join(root_dir, folder)
        if not os.path.isdir(cls_p):
            continue
        label = names.index(INIT_6_MERGE.get(folder, folder) if n_classes == 6 else folder)
        for seq_name in sorted(os.listdir(cls_p)):
            seq_dir = os.path.join(cls_p, seq_name)
            if not os.path.isdir(seq_dir):
                continue
            files = list_frames(seq_dir)
            if len(files) < 20:
                continue

            widths, heights, cxs = [], [], []
            for f in files:
                a = np.array(Image.open(os.path.join(seq_dir, f)).convert('L')) > 127
                xs = np.where(a.any(axis=0))[0]
                ys = np.where(a.any(axis=1))[0]
                widths.append(xs.max() - xs.min() if len(xs) else 0)
                heights.append(ys.max() - ys.min() if len(ys) else 0)
                cxs.append(xs.mean() if len(xs) else np.nan)
            widths = np.array(widths, dtype=np.float64)
            heights = np.array(heights, dtype=np.float64)
            cxs = np.array(cxs)

            cycles = []
            for s, e in detect_init_cycles(widths, heights):
                phase_idx = [int(round(s + k * (e - s) / (NUM_PHASES - 1))) for k in range(NUM_PHASES)]
                cycles.append(jitter_grid(phase_idx, offsets, len(files)))

            valid = cxs[~np.isnan(cxs)]
            seqs.append(dict(
                dir=seq_dir, files=files, label=label, person=seq_name[:3],
                seq_id=seq_name,
                # canonical direction = walking left-to-right (same as GAIT-IT front after flip rule)
                flip=bool(len(valid) > 1 and valid[-1] < valid[0]),
                cycles=cycles,
            ))
    return seqs, [n.replace('_', '-') for n in names]

def check_sequences(seqs):
    """A walk is one recording of one subject with one label."""
    ids = [s['seq_id'] for s in seqs]
    if len(ids) != len(set(ids)):
        raise ValueError("Repeated seq_id: two recordings would be merged into one walk.")

def load_cache(args, cache_file):
    if os.path.isfile(cache_file):
        d = torch.load(cache_file, weights_only=False)
        print(f"[INFO] Loaded phase cache {cache_file}: {tuple(d['data'].shape)}")
        return d

    offsets = [-args.jitter, 0, args.jitter] if args.jitter > 0 else [0]
    if args.dataset == 'gait_it':
        seqs = collect_gait_it(args.dataset_dir, offsets)
        class_names = GAIT_IT_NAMES
    else:
        seqs, class_names = collect_init(args.dataset_dir, offsets, args.init_classes)

    if args.no_canon:
        for s in seqs:
            s['flip'] = False
    check_sequences(seqs)

    n_cyc = sum(len(s['cycles']) for s in seqs)
    J = len(offsets)
    print(f"[INFO] {len(seqs)} sequences -> {n_cyc} 9-phase cycles over "
          f"{len(set(s['person'] for s in seqs))} subjects. Building cache (J={J})...")

    data = torch.zeros(n_cyc, J, NUM_PHASES, args.img_size, args.img_size, dtype=torch.uint8)
    labels, persons, seq_ids = [], [], []
    i = 0
    for si, s in enumerate(seqs):
        needed = sorted({fi for cyc in s['cycles'] for row in cyc for fi in row})
        frames = {}
        for fi in needed:
            try:
                im = Image.open(os.path.join(s['dir'], s['files'][fi])).convert('L')
                im = crop_and_resize_silhouette(im, target_size=args.img_size)
                if s['flip']:
                    im = im.transpose(Image.FLIP_LEFT_RIGHT)
                frames[fi] = torch.from_numpy(np.array(im, dtype=np.uint8))
            except Exception:
                frames[fi] = torch.zeros(args.img_size, args.img_size, dtype=torch.uint8)
        for cyc in s['cycles']:
            for p, row in enumerate(cyc):
                for j, fi in enumerate(row):
                    data[i, j, p] = frames[fi]
            labels.append(s['label'])
            persons.append(s['person'])
            seq_ids.append(s['seq_id'])
            i += 1
        if (si + 1) % 50 == 0:
            print(f"  cached {si + 1}/{len(seqs)} sequences")

    d = {'data': data, 'labels': torch.tensor(labels, dtype=torch.long), 'persons': persons,
         'seq_ids': seq_ids, 'class_names': class_names, 'offsets': offsets}
    os.makedirs(os.path.dirname(cache_file) or '.', exist_ok=True)
    torch.save(d, cache_file)
    return d

# =======================================================================
# 3. BATCH SAMPLING & GPU AUGMENTATION
# =======================================================================
def gather_phases(data, bi, train):
    """data: [N, J, 9, H, W] uint8 -> [B, 9, H, W] float. Train: random jitter per phase."""
    xb = data[bi]
    B, J = xb.shape[0], xb.shape[1]
    if train and J > 1:
        sel = torch.randint(0, J, (B, NUM_PHASES))
    else:
        sel = torch.full((B, NUM_PHASES), J // 2, dtype=torch.long)
    return xb[torch.arange(B)[:, None], sel, torch.arange(NUM_PHASES)[None, :]]

@torch.no_grad()
def gpu_augment_phases(x):
    B, T, H, W = x.shape
    dev = x.device

    ang = (torch.rand(B, device=dev) * 2 - 1) * math.radians(5)
    sc = 1.0 + (torch.rand(B, device=dev) * 2 - 1) * 0.08
    tx = (torch.rand(B, device=dev) * 2 - 1) * 0.05
    ty = (torch.rand(B, device=dev) * 2 - 1) * 0.05

    cos, sin = torch.cos(ang) / sc, torch.sin(ang) / sc
    theta = torch.zeros(B, 2, 3, device=dev)
    theta[:, 0, 0] = cos
    theta[:, 0, 1] = -sin
    theta[:, 0, 2] = tx
    theta[:, 1, 0] = sin
    theta[:, 1, 1] = cos
    theta[:, 1, 2] = ty

    grid = F.affine_grid(theta, size=(B, T, H, W), align_corners=False)
    x = F.grid_sample(x, grid, mode='bilinear', padding_mode='zeros', align_corners=False)

    # Segmentation-quality noise: random 3x3 dilation or erosion of the whole cycle
    morph = torch.rand(B, device=dev)
    dil = F.max_pool2d(x, 3, 1, 1)
    ero = -F.max_pool2d(-x, 3, 1, 1)
    x = torch.where((morph < 0.15)[:, None, None, None], dil, x)
    x = torch.where(((morph >= 0.15) & (morph < 0.30))[:, None, None, None], ero, x)

    # Random erasing (same box across all phases of a cycle)
    erase = torch.rand(B, device=dev) < 0.20
    eh = torch.randint(H // 14, H // 6 + 1, (B,), device=dev)
    ew = torch.randint(W // 14, W // 6 + 1, (B,), device=dev)
    y0 = (torch.rand(B, device=dev) * (H - eh)).long()
    x0 = (torch.rand(B, device=dev) * (W - ew)).long()
    yy = torch.arange(H, device=dev)[None, :, None]
    xx = torch.arange(W, device=dev)[None, None, :]
    box = ((yy >= y0[:, None, None]) & (yy < (y0 + eh)[:, None, None]) &
           (xx >= x0[:, None, None]) & (xx < (x0 + ew)[:, None, None]) & erase[:, None, None])
    x = x.masked_fill(box[:, None], 0.0)
    return x

def make_sampler_weights(labels_np, tr_idx, mode):
    if mode == 'none':
        return None
    counts = np.bincount(labels_np[tr_idx], minlength=labels_np.max() + 1).astype(np.float64)
    w = 1.0 / np.sqrt(np.maximum(counts[labels_np[tr_idx]], 1.0))
    return w / w.sum()

# =======================================================================
# 4. FOCAL LOSS WITH LABEL SMOOTHING (unchanged from v3)
# =======================================================================
class FocalCrossEntropyLoss(nn.Module):
    def __init__(self, gamma=1.5, label_smoothing=0.05):
        super().__init__()
        self.gamma = gamma
        self.label_smoothing = label_smoothing

    def forward(self, logits, targets):
        num_classes = logits.size(-1)
        with torch.no_grad():
            smooth_targets = torch.full_like(logits, self.label_smoothing / (num_classes - 1))
            smooth_targets.scatter_(-1, targets.unsqueeze(-1), 1.0 - self.label_smoothing)

        log_probs = F.log_softmax(logits.float(), dim=-1)
        probs = torch.exp(log_probs)
        focal_weight = torch.pow(1.0 - probs, self.gamma)
        loss = -torch.sum(focal_weight * smooth_targets * log_probs, dim=-1)
        return loss.mean()

# =======================================================================
# 5. ANATOPHASENET V4 ARCHITECTURE
# =======================================================================
class CyclicTemporalShift(nn.Module):
    """TSM over the 9 phases. Cyclic (roll) because a gait cycle is periodic: COMP ~ next IC."""
    def __init__(self, num_phases, fold_div=8):
        super().__init__()
        self.T = num_phases
        self.fold_div = fold_div

    def forward(self, x):
        nt, c, h, w = x.shape
        x = x.view(nt // self.T, self.T, c, h, w)
        f = c // self.fold_div
        out = torch.cat([
            x[:, :, :f].roll(1, dims=1),
            x[:, :, f:2 * f].roll(-1, dims=1),
            x[:, :, 2 * f:]
        ], dim=2)
        return out.view(nt, c, h, w)


class BasicBlockTSM(nn.Module):
    """torchvision ResNet BasicBlock (same parameter names) with a residual temporal shift."""
    def __init__(self, in_planes, planes, stride, num_phases):
        super().__init__()
        self.shift = CyclicTemporalShift(num_phases)
        self.conv1 = nn.Conv2d(in_planes, planes, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        if stride != 1 or in_planes != planes:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes, planes, 1, stride, bias=False),
                nn.BatchNorm2d(planes)
            )
        else:
            self.downsample = None

    def forward(self, x):
        identity = x if self.downsample is None else self.downsample(x)
        out = self.relu(self.bn1(self.conv1(self.shift(x))))
        out = self.bn2(self.conv2(out))
        return self.relu(out + identity)


class TSMBackbone(nn.Module):
    """2D ResNet-18 (stem + layer1..3) per phase with cyclic temporal shift. In: [B, 3, T, H, W]."""
    def __init__(self, num_phases, last_stride=2):
        super().__init__()
        self.T = num_phases
        self.conv1 = nn.Conv2d(3, 64, 7, 2, 3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(3, 2, 1)
        self.layer1 = nn.Sequential(BasicBlockTSM(64, 64, 1, num_phases), BasicBlockTSM(64, 64, 1, num_phases))
        self.layer2 = nn.Sequential(BasicBlockTSM(64, 128, 2, num_phases), BasicBlockTSM(128, 128, 1, num_phases))
        self.layer3 = nn.Sequential(BasicBlockTSM(128, 256, last_stride, num_phases), BasicBlockTSM(256, 256, 1, num_phases))

    def forward(self, x):
        B, C, T, H, W = x.shape
        h = x.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)
        h = self.maxpool(self.relu(self.bn1(self.conv1(h))))
        h = self.layer3(self.layer2(self.layer1(h)))
        return h.view(B, T, *h.shape[1:])                   # [B, T, 256, Hf, Wf]


class FactorizedSpatiotemporalBlock(nn.Module):
    """
    (2+1)D residual block over the 9 gait phases (successor of the v3 block):
      spatial (1x3x3) -> cyclic temporal (3x1x1) residual -> spatial (1x3x3) -> + shortcut
    - Two spatial convs + BN carry the ImageNet ResNet-18 BasicBlock weights (inflated).
    - The temporal conv uses CIRCULAR padding over phases (the gait cycle is periodic,
      COMP ~ next IC) instead of zero padding, and sits in a zero-initialised residual
      branch, so training starts exactly from the pretrained 2D filters and learns
      inter-phase dynamics on top.
    """
    def __init__(self, in_planes, planes, stride):
        super().__init__()
        self.conv1 = nn.Conv3d(in_planes, planes, (1, 3, 3), (1, stride, stride), (0, 1, 1), bias=False)
        self.bn1 = nn.BatchNorm3d(planes)
        self.tconv = nn.Conv3d(planes, planes, (3, 1, 1), bias=False)
        self.bn_t = nn.BatchNorm3d(planes)
        nn.init.zeros_(self.bn_t.weight)
        self.conv2 = nn.Conv3d(planes, planes, (1, 3, 3), 1, (0, 1, 1), bias=False)
        self.bn2 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        if stride != 1 or in_planes != planes:
            self.downsample = nn.Sequential(
                nn.Conv3d(in_planes, planes, 1, (1, stride, stride), bias=False),
                nn.BatchNorm3d(planes)
            )
        else:
            self.downsample = None

    def forward(self, x):
        identity = x if self.downsample is None else self.downsample(x)
        out = self.relu(self.bn1(self.conv1(x)))
        t = F.pad(out, (0, 0, 0, 0, 1, 1), mode='circular')
        out = self.relu(out + self.bn_t(self.tconv(t)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + identity)


class R2Plus1DBackbone(nn.Module):
    """3D (2+1)D ResNet-18-style backbone (stem + 3 stages x 2 blocks). In: [B, 3, T, H, W]."""
    def __init__(self, num_phases, last_stride=2):
        super().__init__()
        self.conv1 = nn.Conv3d(3, 64, (1, 7, 7), (1, 2, 2), (0, 3, 3), bias=False)
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d((1, 3, 3), (1, 2, 2), (0, 1, 1))
        self.layer1 = nn.Sequential(FactorizedSpatiotemporalBlock(64, 64, 1), FactorizedSpatiotemporalBlock(64, 64, 1))
        self.layer2 = nn.Sequential(FactorizedSpatiotemporalBlock(64, 128, 2), FactorizedSpatiotemporalBlock(128, 128, 1))
        self.layer3 = nn.Sequential(FactorizedSpatiotemporalBlock(128, 256, last_stride), FactorizedSpatiotemporalBlock(256, 256, 1))

    def forward(self, x):
        h = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        h = self.layer3(self.layer2(self.layer1(h)))        # [B, 256, T, Hf, Wf]
        return h.permute(0, 2, 1, 3, 4)                     # [B, T, 256, Hf, Wf]

def load_pretrained_backbone(backbone, path):
    """Copies ImageNet ResNet-18 conv + BN tensors; 2D kernels are inflated to (1, k, k) for 3D."""
    ckpt = torch.load(path, weights_only=True)
    own = backbone.state_dict()
    keep = {}
    for k, v in ckpt.items():
        if k.split('.')[0] not in ('conv1', 'bn1', 'layer1', 'layer2', 'layer3') or k not in own:
            continue
        if own[k].dim() == 5 and v.dim() == 4:
            v = v.unsqueeze(2)
        if own[k].shape == v.shape:
            keep[k] = v
    missing, _ = backbone.load_state_dict(keep, strict=False)
    new_keys = sorted({k.split('.')[-2] for k in missing})
    print(f"[INFO] Loaded {len(keep)} ImageNet ResNet-18 tensors (conv + BN) from {path}. "
          f"Newly initialised backbone tensors: {len(missing)} {new_keys}")


# =======================================================================
# 6. EMA & EVALUATION
# =======================================================================
class EMA:
    def __init__(self, model, decay=0.98):
        self.decay = decay
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model):
        for k, v in model.state_dict().items():
            if v.dtype.is_floating_point:
                self.shadow[k].mul_(self.decay).add_(v.detach(), alpha=1 - self.decay)
            else:
                self.shadow[k].copy_(v)

    def copy_to(self, model):
        model.load_state_dict(self.shadow)

# =======================================================================
# 7. SUBJECT-EXCLUSIVE K-FOLD CV SPLITTER (identical to v3)
# =======================================================================
def make_strict_person_folds(persons, num_folds=10, seed=42):
    unique_persons = sorted(list(set(persons)))
    rng = random.Random(seed)
    shuffled = list(unique_persons)
    rng.shuffle(shuffled)

    person_to_fold = {}
    for i, p in enumerate(shuffled):
        person_to_fold[p] = i % num_folds

    cycle_folds = np.array([person_to_fold[p] for p in persons], dtype=int)
    print(f"[INFO] Created strict subject-exclusive {num_folds}-fold split for {len(unique_persons)} subjects.")
    for f in range(num_folds):
        f_subjs = [p for p in shuffled if person_to_fold[p] == f]
        n_cyc = np.sum(cycle_folds == f)
        print(f"  Fold {f + 1:2d}: {len(f_subjs)} subjects {f_subjs} -> {n_cyc} cycles")
    return cycle_folds

def is_better(select, v_acc, v_f1, v_loss, best):
    if best is None:
        return True
    if select == 'last':
        return True
    if select == 'loss':
        return v_loss < best['loss']
    key = v_f1 if select == 'f1' else v_acc
    best_key = best['f1'] if select == 'f1' else best['acc']
    return key > best_key or (key == best_key and v_loss < best['loss'])

SEVERITY_NAMES = ['normal', 'mild', 'severe']

# =======================================================================
# 1. SEVERITY LABELS (parsed from sequence names, -1 = no label)
# =======================================================================
INIT6_SEVERITY = {'nm': 0, 'a-l0.5': 1, 'a-r0.5': 1, 'a-l0': 2, 'a-r0': 2}

def severity_labels(seq_ids, dataset, init_classes):
    out = []
    for s in seq_ids:
        if dataset == 'gait_it':
            m = re.search(r'lvl(\d)', s)
            out.append(int(m.group(1)) if m else (0 if 'normal' in s else -1))
        elif init_classes == 6:
            out.append(INIT6_SEVERITY.get(s.split('_')[2], -1))
        else:
            out.append(-1)
    return torch.tensor(out, dtype=torch.long)

# =======================================================================
# 2. ARCHITECTURE
# =======================================================================
class PartPhaseHead_v5(nn.Module):
    """
    v4 part-based temporal head with an optional symmetry stream, returning both
    per-part logits [B, P, K] and the normalised cycle descriptor [B, P * F * D].
    """
    def __init__(self, in_ch, num_classes, num_parts=6, num_phases=9, dim=128, dropout=0.3, use_sym=True):
        super().__init__()
        self.num_parts = num_parts
        self.use_sym = use_sym
        P = num_parts + 1
        nf = 3 if use_sym else 2
        self.feat_dim = P * nf * dim

        self.proj = nn.Parameter(torch.empty(P, 2 * in_ch, dim))
        nn.init.xavier_uniform_(self.proj)
        self.phase_embed = nn.Parameter(torch.randn(1, num_phases, P, dim) * 0.02)

        self.tconv = nn.Sequential(
            nn.Conv1d(P * dim, P * dim, 3, padding=1, padding_mode='circular', groups=P, bias=False),
            nn.BatchNorm1d(P * dim),
            nn.ReLU(inplace=True),
        )
        self.attn = nn.Parameter(torch.zeros(P, dim))

        self.bn = nn.BatchNorm1d(self.feat_dim)
        self.drop = nn.Dropout(dropout)
        self.cls_w = nn.Parameter(torch.empty(P, nf * dim, num_classes))
        nn.init.xavier_uniform_(self.cls_w)
        self.cls_b = nn.Parameter(torch.zeros(P, num_classes))

    def forward(self, h):
        B, T, C, Hf, Wf = h.shape
        P = self.num_parts + 1
        h2 = h.reshape(B * T, C, Hf, Wf)

        avg_part = torch.cat([F.adaptive_avg_pool2d(h2, (self.num_parts, 1)),
                              F.adaptive_avg_pool2d(h2, 1)], dim=2).squeeze(-1)            # [BT, C, P]
        max_part = torch.cat([F.adaptive_max_pool2d(h2, (self.num_parts, 1)),
                              F.adaptive_max_pool2d(h2, 1)], dim=2).squeeze(-1)            # [BT, C, P]
        part = torch.cat([avg_part, max_part], dim=1).permute(0, 2, 1)                     # [BT, P, 2C]

        f = torch.einsum('npc,pcd->npd', part, self.proj)
        D = f.shape[-1]
        f = f.view(B, T, P, D) + self.phase_embed

        z = f.permute(0, 2, 3, 1).reshape(B, P * D, T)
        z = self.tconv(z).view(B, P, D, T).permute(0, 3, 1, 2)
        f = f + z

        score = torch.einsum('btpd,pd->btp', f, self.attn)
        a = F.softmax(score, dim=1).unsqueeze(-1)
        feats = [(f * a).sum(dim=1), f.max(dim=1).values]
        if self.use_sym:
            half = (T - 1) // 2
            feats.append((f[:, :half] - f[:, half:2 * half]).abs().mean(dim=1))

        g = torch.cat(feats, dim=-1)                                                        # [B, P, nf*D]
        g = self.bn(g.reshape(B, -1))                                                       # [B, feat_dim]
        logits = torch.einsum('bpd,pdk->bpk', self.drop(g).view(B, P, -1), self.cls_w) + self.cls_b
        return logits, g


def session_statistics(e, m):
    """
    Stride-to-stride variability of a walk.
    e: [B, K, E] cycle embeddings (chronological), m: [B, K] validity mask.
    Returns [B, 3E] = [mean, std, mean successive |difference|] over valid cycles.
    """
    m3 = m.unsqueeze(-1)
    n = m.sum(dim=1, keepdim=True).clamp(min=1.0)
    mean = (e * m3).sum(dim=1) / n
    var = (((e - mean.unsqueeze(1)) ** 2) * m3).sum(dim=1) / (n - 1).clamp(min=1.0)
    std = torch.sqrt(var + 1e-6)
    pair = m[:, 1:] * m[:, :-1]
    diff = ((e[:, 1:] - e[:, :-1]).abs() * pair.unsqueeze(-1)).sum(dim=1) / pair.sum(dim=1, keepdim=True).clamp(min=1.0)
    return torch.cat([mean, std, diff], dim=-1)

GEOM_NAMES = ['height', 'width', 'lower_width', 'upper_width', 'head_bob',
              'centroid_height', 'trunk_lean', 'mass_offset', 'fill']
NUM_GEOM = len(GEOM_NAMES)

# =======================================================================
# 1. GEOMETRY CACHE (measured on the raw frames, before cropping)
# =======================================================================
def frame_stats(path):
    """Raw bounding-box statistics of one silhouette frame, or None if it is empty."""
    a = np.array(Image.open(path).convert('L')) > 127
    rows = np.where(a.any(axis=1))[0]
    if len(rows) == 0:
        return None
    y0, y1 = rows[0], rows[-1]
    cols = np.where(a.any(axis=0))[0]
    x0, x1 = cols[0], cols[-1]
    h, w = y1 - y0 + 1, x1 - x0 + 1
    ys, xs = np.nonzero(a)

    def band(lo, hi):
        sub = a[y0 + int(lo * h): y0 + max(int(hi * h), int(lo * h) + 1)]
        c = np.where(sub.any(axis=0))[0]
        if len(c) == 0:
            return 0.0, float(xs.mean())
        _, bx = np.nonzero(sub)
        return float(c[-1] - c[0] + 1), float(bx.mean())

    low_w, cx_bot = band(0.70, 1.00)
    up_w, _ = band(0.15, 0.55)
    _, cx_top = band(0.00, 0.30)
    return dict(y0=float(y0), h=float(h), w=float(w), cx_box=(x0 + x1) / 2.0,
                cy=float(ys.mean()), cx=float(xs.mean()), area=float(len(ys)),
                low_w=low_w, up_w=up_w, cx_top=cx_top, cx_bot=cx_bot)


def geometry_features(st, h_ref, y0_ref, sign):
    if st is None:
        return np.zeros(NUM_GEOM, dtype=np.float32)
    return np.array([
        st['h'] / h_ref - 1.0,
        st['w'] / h_ref,
        st['low_w'] / h_ref,
        st['up_w'] / h_ref,
        (st['y0'] - y0_ref) / h_ref,
        (st['cy'] - st['y0']) / st['h'],
        sign * (st['cx_top'] - st['cx_bot']) / h_ref,
        sign * (st['cx'] - st['cx_box']) / h_ref,
        st['area'] / (st['h'] * st['w']),
    ], dtype=np.float32)


def load_geometry_cache(args, geom_file, ref):
    """[N, J, 9, NUM_GEOM] float32, aligned cycle-by-cycle with the v4 image cache `ref`."""
    if os.path.isfile(geom_file):
        g = torch.load(geom_file, weights_only=False)
        if g['seq_ids'] == ref['seq_ids']:
            print(f"[INFO] Loaded geometry cache {geom_file}: {tuple(g['geom'].shape)}")
            return g['geom']
        print("[WARN] Geometry cache does not match the image cache, rebuilding it.")

    offsets = ref['offsets']
    if args.dataset == 'gait_it':
        seqs = collect_gait_it(args.dataset_dir, offsets)
    else:
        seqs, _ = collect_init(args.dataset_dir, offsets, args.init_classes)
    if args.no_canon:
        for s in seqs:
            s['flip'] = False
    check_sequences(seqs)

    n_cyc = sum(len(s['cycles']) for s in seqs)
    geom = torch.zeros(n_cyc, len(offsets), NUM_PHASES, NUM_GEOM, dtype=torch.float32)
    seq_ids = []
    print(f"[INFO] Building geometry cache for {len(seqs)} sequences ({n_cyc} cycles)...")
    i = 0
    for si, s in enumerate(seqs):
        needed = sorted({fi for cyc in s['cycles'] for row in cyc for fi in row})
        stats = {}
        for fi in needed:
            try:
                stats[fi] = frame_stats(os.path.join(s['dir'], s['files'][fi]))
            except Exception:
                stats[fi] = None
        valid = [st for st in stats.values() if st is not None]
        h_ref = float(np.median([st['h'] for st in valid])) if valid else 1.0
        y0_ref = float(np.median([st['y0'] for st in valid])) if valid else 0.0
        sign = -1.0 if s['flip'] else 1.0
        feats = {fi: geometry_features(st, max(h_ref, 1.0), y0_ref, sign) for fi, st in stats.items()}
        for cyc in s['cycles']:
            for p, row in enumerate(cyc):
                for j, fi in enumerate(row):
                    geom[i, j, p] = torch.from_numpy(feats[fi])
            seq_ids.append(s['seq_id'])
            i += 1
        if (si + 1) % 50 == 0:
            print(f"  geometry {si + 1}/{len(seqs)} sequences")

    if seq_ids != ref['seq_ids']:
        raise RuntimeError("Geometry cycles are not aligned with the image cache; delete both caches and rerun.")
    geom = torch.nan_to_num(geom, nan=0.0, posinf=0.0, neginf=0.0)
    os.makedirs(os.path.dirname(geom_file) or '.', exist_ok=True)
    torch.save({'geom': geom, 'seq_ids': seq_ids, 'names': GEOM_NAMES}, geom_file)
    return geom


def active_signals(geom):
    """Indices of the signals that vary in this dataset. GAIT-IT frames are already normalised
    to the silhouette height, so height and head bob are constant zero there."""
    return torch.where(geom.reshape(-1, geom.shape[-1]).std(dim=0) > 1e-6)[0]


def check_fold(persons, tr_idx, va_idx, te_idx):
    tr, va, te = ({persons[i] for i in ix} for ix in (tr_idx, va_idx, te_idx))
    if (tr & va) or (tr & te) or (va & te):
        raise ValueError("A subject has two roles in one fold.")


def gather_cycles(data, geom, bi, train):
    """Same jitter view per phase for images [B, 9, H, W] and geometry [B, 9, G]."""
    B, J = len(bi), data.shape[1]
    if train and J > 1:
        sel = torch.randint(0, J, (B, NUM_PHASES))
    else:
        sel = torch.full((B, NUM_PHASES), J // 2, dtype=torch.long)
    rows = torch.as_tensor(np.asarray(bi), dtype=torch.long)[:, None]
    ph = torch.arange(NUM_PHASES)[None, :]
    return data[rows, sel, ph], geom[rows, sel, ph]

# =======================================================================
# 2. ARCHITECTURE
# =======================================================================
class _GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lam):
        ctx.lam = lam
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad):
        return -ctx.lam * grad, None


class WarmupEMA(EMA):
    """EMA whose decay ramps up as min(decay, (1+n)/(10+n)) (timm-style). Plain EMA starts from the
    untrained weights and, at 0.98, keeps them for ~100 steps: on INIT (~29 steps/epoch) the
    validated model then predicts uniformly for the first ~6 epochs."""
    def __init__(self, model, decay=0.98):
        super().__init__(model, decay)
        self.max_decay = decay
        self.n_updates = 0

    def update(self, model):
        self.decay = min(self.max_decay, (1 + self.n_updates) / (10 + self.n_updates))
        self.n_updates += 1
        super().update(model)


class GeometryBranch(nn.Module):
    """Cyclic temporal CNN over the 9-phase biomechanical geometry sequence."""
    def __init__(self, in_dim, num_classes, dim=64, dropout=0.3):
        super().__init__()
        self.norm = nn.BatchNorm1d(in_dim)
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_dim, dim, 3, padding=1, padding_mode='circular', bias=False),
            nn.BatchNorm1d(dim), nn.ReLU(inplace=True))
        self.conv2 = nn.Sequential(
            nn.Conv1d(dim, dim, 3, padding=1, padding_mode='circular', bias=False),
            nn.BatchNorm1d(dim), nn.ReLU(inplace=True))
        self.feat_dim = 3 * dim
        self.bn = nn.BatchNorm1d(self.feat_dim)
        self.drop = nn.Dropout(dropout)
        self.cls = nn.Linear(self.feat_dim, num_classes)

    def forward(self, geo):
        # geo: [B, T, G]
        z = self.norm(geo.float().transpose(1, 2))
        z = self.conv1(z)
        z = z + self.conv2(z)                                                   # [B, D, T]
        half = (z.shape[-1] - 1) // 2
        g = torch.cat([z.mean(-1), z.amax(-1),
                       (z[:, :, :half] - z[:, :, half:2 * half]).abs().mean(-1)], dim=-1)
        g = self.bn(g)
        return self.cls(self.drop(g)), g


class AnatoPhaseNet_v7(nn.Module):
    def __init__(self, num_classes, pretrained_path=None, backbone='r2plus1d', num_phases=9, num_parts=6,
                 last_stride=2, head_dim=128, dropout=0.3, use_sym=True, num_severity=3, emb_dim=256,
                 use_geom=True, geom_dim=64, num_subjects=0, num_geom=NUM_GEOM):
        super().__init__()
        if backbone == 'r2plus1d':
            self.backbone = R2Plus1DBackbone(num_phases, last_stride)
        else:
            self.backbone = TSMBackbone(num_phases, last_stride)

        self.head = PartPhaseHead_v5(256, num_classes, num_parts=num_parts, num_phases=num_phases,
                                     dim=head_dim, dropout=dropout, use_sym=use_sym)
        fd = self.head.feat_dim

        self.geom = GeometryBranch(num_geom, num_classes, dim=geom_dim, dropout=dropout) if use_geom else None
        if self.geom is not None:
            fd += self.geom.feat_dim

        self.severity_head = nn.Sequential(nn.Dropout(dropout), nn.Linear(fd, num_severity))
        self.embed = nn.Sequential(nn.Dropout(dropout), nn.Linear(fd, emb_dim), nn.ReLU(inplace=True))
        self.session_head = nn.Sequential(
            nn.LayerNorm(3 * emb_dim),
            nn.Dropout(dropout),
            nn.Linear(3 * emb_dim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_classes),
        )

        # Subject adversary behind a gradient-reversal layer
        self.adv_head = nn.Sequential(nn.Linear(fd, 128), nn.ReLU(inplace=True),
                                      nn.Linear(128, num_subjects)) if num_subjects > 0 else None
        self.adv_lambda = 0.0

        if pretrained_path and os.path.isfile(pretrained_path):
            load_pretrained_backbone(self.backbone, pretrained_path)

    def backbone_parameters(self):
        return list(self.backbone.parameters())

    def head_parameters(self):
        return [p for n, p in self.named_parameters() if not n.startswith('backbone.')]

    def encode(self, x, geo):
        # x: [B, 9, H, W] in [0, 1], geo: [B, 9, G]
        sil = (x - 0.5) / 0.5
        motion = x - x.roll(1, dims=1)
        dev = x - x.mean(dim=1, keepdim=True)
        inp = torch.stack([sil, motion, dev], dim=1)        # [B, 3, T, H, W]

        part_logits, g = self.head(self.backbone(inp))
        if self.geom is not None:
            g_logits, g_geo = self.geom(geo)
            part_logits = torch.cat([part_logits, g_logits.unsqueeze(1).to(part_logits.dtype)], dim=1)
            g = torch.cat([g, g_geo.to(g.dtype)], dim=-1)

        out = {
            'logits': part_logits.mean(dim=1),
            'part_logits': part_logits,
            'severity': self.severity_head(g),
            'emb': self.embed(g),
        }
        if self.adv_head is not None and self.training:
            out['subject'] = self.adv_head(_GradReverse.apply(g, self.adv_lambda))
        return out

    def session_logits(self, emb, mask):
        return self.session_head(session_statistics(emb.float(), mask.float()))

# =======================================================================
# 3. EVALUATION
# =======================================================================
@torch.no_grad()
def evaluate_v7(model, data, geom, labels, sev_t, seq_ids, idx, device, num_classes, use_amp=True, tta=True,
                use_session_head=True):
    model.eval()
    J = data.shape[1]
    views = list(range(J)) if (tta and J > 1) else [J // 2]

    logits_l, sev_l, emb_l = [], [], []
    bs = 32
    for b in range(int(math.ceil(len(idx) / bs))):
        b_idx = idx[b * bs:(b + 1) * bs]
        acc = None
        for j in views:
            x = data[b_idx, j].to(device).float() / 255.0
            geo = geom[b_idx, j].to(device)
            with torch.autocast('cuda', dtype=torch.float16, enabled=use_amp):
                o = model.encode(x, geo)
            o = {k: o[k].float() for k in ('logits', 'severity', 'emb')}
            acc = o if acc is None else {k: acc[k] + o[k] for k in acc}
        logits_l.append((acc['logits'] / len(views)).cpu())
        sev_l.append((acc['severity'] / len(views)).cpu())
        emb_l.append(acc['emb'] / len(views))

    logits = torch.cat(logits_l)
    sev_logits = torch.cat(sev_l)
    emb = torch.cat(emb_l)
    targets = labels[idx]
    sev_targets = sev_t[idx]

    cycle_loss = float(F.cross_entropy(logits, targets))
    log_probs = F.log_softmax(logits, dim=-1)
    preds = log_probs.argmax(-1).numpy()
    t_np = targets.numpy()
    res = {
        'loss': cycle_loss,
        'cycle_acc': float(np.mean(preds == t_np) * 100.0),
        'cycle_f1': compute_macro_f1(t_np, preds, num_classes),
        'preds': preds, 'targets': t_np,
        'cycle_logits': logits.numpy(), 'sev_logits': sev_logits.numpy(), 'sev_targets': sev_targets.numpy(),
    }

    sv = sev_targets >= 0
    res['sev_acc'] = float((sev_logits[sv].argmax(-1) == sev_targets[sv]).float().mean() * 100.0) if sv.any() else float('nan')

    groups = {}
    for pos, orig_i in enumerate(idx):
        groups.setdefault(seq_ids[orig_i], []).append(pos)
    s_ids = sorted(groups)
    s_targets = np.array([t_np[groups[s][0]] for s in s_ids])

    vote_logp = np.stack([log_probs[groups[s]].mean(0).numpy() for s in s_ids])
    vote_preds = vote_logp.argmax(-1)
    res['sess_vote_acc'] = float(np.mean(vote_preds == s_targets) * 100.0)
    res['sess_vote_f1'] = compute_macro_f1(s_targets, vote_preds, num_classes)

    if use_session_head:
        Lmax = max(len(groups[s]) for s in s_ids)
        E = emb.shape[-1]
        e_pad = torch.zeros(len(s_ids), Lmax, E, device=emb.device)
        m_pad = torch.zeros(len(s_ids), Lmax, device=emb.device)
        for i, s in enumerate(s_ids):
            pos = groups[s]
            e_pad[i, :len(pos)] = emb[pos]
            m_pad[i, :len(pos)] = 1.0
        s_logits = model.session_logits(e_pad, m_pad).float().cpu()
        head_preds = s_logits.argmax(-1).numpy()
        res['sess_head_acc'] = float(np.mean(head_preds == s_targets) * 100.0)
        res['sess_head_f1'] = compute_macro_f1(s_targets, head_preds, num_classes)
        res['session_logits'] = s_logits.numpy()
        res['sess_acc'], res['sess_f1'] = res['sess_head_acc'], res['sess_head_f1']
    else:
        res['sess_head_acc'] = res['sess_head_f1'] = float('nan')
        res['session_logits'] = vote_logp
        res['sess_acc'], res['sess_f1'] = res['sess_vote_acc'], res['sess_vote_f1']

    res['session_ids'] = np.array(s_ids)
    res['session_targets'] = s_targets
    return res

# =======================================================================
# 4. MAIN EXPERIMENT RUNNER
# =======================================================================
def main():
    data_root = os.path.join(PARENT, 'Gait_Dataset_silhoutte_based')
    default_dirs = {
        'gait_it': os.path.join(data_root, 'GAIT_IT'),
        'init': os.path.join(data_root, 'INIT', 'INIT_GaitDB', 'INIT_8', 'silhouettes'),
    }

    parser = argparse.ArgumentParser(description="AnatoPhaseNet v7")
    parser.add_argument('--dataset', type=str, default='gait_it', choices=['gait_it', 'init'])
    parser.add_argument('--dataset_dir', type=str, default=None)
    parser.add_argument('--init_classes', type=int, default=6, choices=[8, 6])
    parser.add_argument('--pretrained_path', type=str, default=os.path.join(PARENT, 'resnet18_pretrained.pth'))
    parser.add_argument('--out_dir', type=str, default=None)
    parser.add_argument('--img_size', type=int, default=96)
    parser.add_argument('--jitter', type=int, default=2)
    parser.add_argument('--no_canon', action='store_true')
    parser.add_argument('--backbone', type=str, default='r2plus1d', choices=['r2plus1d', 'tsm'])
    parser.add_argument('--num_parts', type=int, default=6)
    parser.add_argument('--last_stride', type=int, default=2, choices=[1, 2])
    parser.add_argument('--head_dim', type=int, default=128)
    parser.add_argument('--dropout', type=float, default=0.3)
    # v5 components
    parser.add_argument('--no_severity', action='store_true', help='Disable the severity multi-task head')
    parser.add_argument('--severity_weight', type=float, default=0.3)
    parser.add_argument('--no_multicycle', action='store_true', help='Disable the multi-cycle variability module')
    parser.add_argument('--session_weight', type=float, default=1.0)
    parser.add_argument('--k_max', type=int, default=6, help='Max consecutive cycles per training walk sample')
    parser.add_argument('--no_sym', action='store_true', help='Disable the half-cycle symmetry feature')
    # geometry and adversary components
    parser.add_argument('--no_geom', action='store_true', help='Disable the biomechanical geometry branch')
    parser.add_argument('--geom_dim', type=int, default=64)
    parser.add_argument('--adv_subject', action='store_true', help='Enable subject-adversarial training')
    parser.add_argument('--adv_weight', type=float, default=0.1, help='Max gradient-reversal strength')
    # optimisation
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=16, help='Cycles per step')
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--backbone_lr_mult', type=float, default=0.3)
    parser.add_argument('--weight_decay', type=float, default=5e-2)
    parser.add_argument('--focal_gamma', type=float, default=1.5)
    parser.add_argument('--label_smoothing', type=float, default=0.05)
    parser.add_argument('--part_loss_weight', type=float, default=0.5)
    parser.add_argument('--balance', type=str, default='sqrt', choices=['sqrt', 'none'])
    parser.add_argument('--ema', type=float, default=0.98)
    parser.add_argument('--ema_warmup', action='store_true',
                        help='Ramp the EMA decay up from 0 (results go to *_emaw)')
    parser.add_argument('--select', type=str, default='last', choices=['acc', 'f1', 'loss', 'last'])
    parser.add_argument('--no_tta', action='store_true')
    parser.add_argument('--folds', type=int, default=10)
    parser.add_argument('--run_folds', type=int, default=10)
    parser.add_argument('--fold_ids', type=int, nargs='*', default=None)
    parser.add_argument('--seed', type=int, default=42, help='Training seed')
    parser.add_argument('--split_seed', type=int, default=42, help='Fold split seed (keep 42 to match earlier versions)')
    parser.add_argument('--save_models', action='store_true')
    args = parser.parse_args()

    args.dataset_dir = args.dataset_dir or default_dirs[args.dataset]
    tag_ds = args.dataset if args.dataset == 'gait_it' else f"init{args.init_classes}"

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    use_amp = (device.type == 'cuda')

    # v7 builds its own caches: the cycles of earlier versions are not valid here
    canon_tag = '_nocanon' if args.no_canon else ''
    cache_file = os.path.join(HERE, 'cache', f"{tag_ds}_{args.img_size}_j{args.jitter}{canon_tag}.pt")
    d = load_cache(args, cache_file)
    data, labels, persons, seq_ids = d['data'], d['labels'], d['persons'], d['seq_ids']
    class_names = d['class_names']
    num_classes = len(class_names)
    labels_np = labels.numpy()
    walk_keys = {}
    for s, l, p in zip(seq_ids, labels_np.tolist(), persons):
        if walk_keys.setdefault(s, (l, p)) != (l, p):
            raise ValueError(f"Walk {s} mixes labels or subjects.")

    geom = load_geometry_cache(args, os.path.join(HERE, 'cache', f"{tag_ds}_geom_j{args.jitter}{canon_tag}.pt"), d)
    active = active_signals(geom)
    geom = geom[..., active].contiguous()
    active_names = [GEOM_NAMES[i] for i in active.tolist()]
    num_geom = len(active_names)

    sev_t = severity_labels(seq_ids, args.dataset, args.init_classes)
    use_severity = (not args.no_severity) and bool((sev_t >= 0).any())
    use_mc = not args.no_multicycle
    use_sym = not args.no_sym
    use_geom = not args.no_geom
    use_adv = args.adv_subject

    subj_names = sorted(set(persons))
    subj_t = torch.tensor([subj_names.index(p) for p in persons], dtype=torch.long)

    variant = [n for n, off in (('nosev', not use_severity and not (args.dataset == 'init' and args.init_classes == 8)),
                                ('nomc', not use_mc), ('nosym', not use_sym), ('nogeom', not use_geom),
                                ('adv', use_adv)) if off]
    variant = '_'.join(variant) or 'full'
    seed_tag = '' if args.seed == 42 else f"_seed{args.seed}"
    ema_tag = '_emaw' if args.ema_warmup else ''
    args.out_dir = args.out_dir or os.path.join(HERE, f"results_{tag_ds}_{args.backbone}_{variant}{ema_tag}{seed_tag}")
    os.makedirs(args.out_dir, exist_ok=True)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    print("=" * 75)
    print(f"  AnatoPhaseNet v7 | dataset: {tag_ds} | backbone: {args.backbone} | variant: {variant}")
    print(f"  Geometry branch: {use_geom} | Subject-adversarial: {use_adv} | Severity task: {use_severity} | "
          f"Multi-cycle: {use_mc} | Symmetry: {use_sym} | Device: {device}")
    print("=" * 75)
    print(f"[INFO] Geometry signals in use ({num_geom}): {active_names}")
    print(f"[INFO] Classes: {class_names} | cycles per class: {np.bincount(labels_np, minlength=num_classes).tolist()}")
    if use_severity:
        print(f"[INFO] Severity labels {SEVERITY_NAMES}: {[int((sev_t == s).sum()) for s in range(3)]} "
              f"(unlabelled: {int((sev_t < 0).sum())})")

    model_kw = dict(backbone=args.backbone, num_parts=args.num_parts, last_stride=args.last_stride,
                    head_dim=args.head_dim, dropout=args.dropout, use_sym=use_sym, use_geom=use_geom,
                    geom_dim=args.geom_dim, num_subjects=len(subj_names) if use_adv else 0, num_geom=num_geom)
    model_tmp = AnatoPhaseNet_v7(num_classes, None, **model_kw)
    param_count = sum(p.numel() for p in model_tmp.parameters() if p.requires_grad)
    print(f"[INFO] Architecture: AnatoPhaseNet v7 ({args.backbone}) | Parameters: {param_count / 1e6:.3f}M")
    del model_tmp

    n_folds = min(args.folds, len(set(persons)))
    cycle_folds = make_strict_person_folds(persons, num_folds=n_folds, seed=args.split_seed)
    if args.fold_ids:
        fold_list = [f - 1 for f in args.fold_ids]
    else:
        fold_list = list(range(args.run_folds if args.run_folds > 0 else n_folds))[:n_folds]

    fold_results = []
    agg_conf = np.zeros((num_classes, num_classes), dtype=int)

    for f_idx in fold_list:
        te_idx = np.where(cycle_folds == f_idx)[0]
        va_idx = np.where(cycle_folds == (f_idx + 1) % n_folds)[0]
        tr_idx = np.where((cycle_folds != f_idx) & (cycle_folds != (f_idx + 1) % n_folds))[0]
        check_fold(persons, tr_idx, va_idx, te_idx)
        geom_std = geom[tr_idx].reshape(-1, num_geom).std(dim=0).clamp(min=1e-4)

        print("\n" + "-" * 75)
        print(f"[Fold {f_idx + 1}/{n_folds}] Train: {len(tr_idx)} cycles | Val: {len(va_idx)} cycles | Test: {len(te_idx)} cycles")
        print("-" * 75)

        model = AnatoPhaseNet_v7(num_classes, args.pretrained_path, **model_kw).to(device)
        eval_model = AnatoPhaseNet_v7(num_classes, None, **model_kw).to(device)
        ema = (WarmupEMA if args.ema_warmup else EMA)(model, decay=args.ema)

        steps = max(1, len(tr_idx) // args.batch_size)
        total_steps = args.epochs * steps
        opt = torch.optim.AdamW([
            {'params': model.backbone_parameters(), 'lr': args.lr * args.backbone_lr_mult},
            {'params': model.head_parameters(), 'lr': args.lr},
        ], weight_decay=args.weight_decay)
        sched = torch.optim.lr_scheduler.OneCycleLR(
            opt, max_lr=[args.lr * args.backbone_lr_mult, args.lr],
            total_steps=total_steps, pct_start=0.15
        )
        scaler = torch.amp.GradScaler('cuda', enabled=use_amp)
        crit = FocalCrossEntropyLoss(gamma=args.focal_gamma, label_smoothing=args.label_smoothing)

        samp_w = make_sampler_weights(labels_np, tr_idx, args.balance)
        walks = {}
        for i in tr_idx:
            walks.setdefault(seq_ids[i], []).append(int(i))
        walk_ids = sorted(walks)
        walk_labels = np.array([labels_np[walks[w][0]] for w in walk_ids])
        if args.balance == 'sqrt':
            wc = np.bincount(walk_labels, minlength=num_classes).astype(np.float64)
            walk_p = 1.0 / np.sqrt(np.maximum(wc[walk_labels], 1.0))
        else:
            walk_p = np.ones(len(walk_ids))
        walk_p = walk_p / walk_p.sum()

        best = None
        best_state = None
        history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_cycle_acc': [], 'val_sess_acc': [], 'val_f1': []}
        csv_path = os.path.join(args.out_dir, f'fold_{f_idx+1}_metrics.csv')
        with open(csv_path, 'w', encoding='utf-8') as f:
            # Train_Loss = full multi-task objective. The last three columns make the curves like-for-like:
            # Train_Cycle_CE = cycle cross-entropy on the training batches (same function as Val_Loss);
            # Val_*_Raw = validation of the trained weights before EMA averaging.
            f.write("Epoch,Train_Loss,Train_Acc,Val_Loss,Val_Cycle_Acc,Val_Session_Acc,Val_Session_Vote_Acc,Val_Macro_F1,"
                    "Val_Severity_Acc,Train_Cycle_CE,Val_Loss_Raw,Val_Cycle_Acc_Raw\n")

        g_step = 0
        for ep in range(args.epochs):
            model.train()
            perm = None
            if not use_mc:
                if samp_w is None:
                    perm = np.random.permutation(tr_idx)
                else:
                    perm = np.random.choice(tr_idx, size=steps * args.batch_size, replace=True, p=samp_w)
            ep_loss, ep_ce, n_seen, t_correct = 0.0, 0.0, 0, 0

            for s in range(steps):
                if use_mc:
                    # Cap K so every batch holds >= 2 walks: a single-walk batch is one subject/class, and
                    # train-mode BatchNorm then normalises away the class signal (broke bs 4 / bs 8).
                    # For batch_size >= 2 * k_max this is identical to the original K range.
                    K = random.randint(2, max(2, min(args.k_max, args.batch_size // 2)))
                    nb = max(1, args.batch_size // K)
                    chosen = np.random.choice(len(walk_ids), size=nb, replace=True, p=walk_p)
                    slot_idx = np.zeros((nb, K), dtype=np.int64)
                    slot_mask = np.zeros((nb, K), dtype=np.float32)
                    for b, wi in enumerate(chosen):
                        cyc = walks[walk_ids[wi]]
                        take = min(K, len(cyc))
                        st = random.randint(0, len(cyc) - take)
                        sel = cyc[st:st + take]
                        slot_idx[b, :take] = sel
                        slot_idx[b, take:] = sel[0]
                        slot_mask[b, :take] = 1.0
                    flat = slot_idx.reshape(-1)
                else:
                    flat = perm[s * args.batch_size:(s + 1) * args.batch_size]
                    nb, K = len(flat), 1
                    slot_mask = np.ones((nb, 1), dtype=np.float32)

                x, geo = gather_cycles(data, geom, flat, train=True)
                x = gpu_augment_phases(x.to(device).float() / 255.0)
                geo = geo.to(device)
                # Geometry augmentation: global scale + small feature-wise noise
                geo = geo * (1.0 + (torch.rand(geo.shape[0], 1, 1, device=device) * 2 - 1) * 0.05) \
                    + torch.randn_like(geo) * 0.05 * geom_std.to(device)
                y = labels[flat].to(device)
                sv = sev_t[flat].to(device)
                mask = torch.from_numpy(slot_mask).to(device)
                valid = mask.reshape(-1) > 0

                if use_adv:
                    p = g_step / max(1, total_steps)
                    model.adv_lambda = args.adv_weight * (2.0 / (1.0 + math.exp(-10.0 * p)) - 1.0)

                with torch.autocast('cuda', dtype=torch.float16, enabled=use_amp):
                    o = model.encode(x, geo)

                out = o['logits'][valid]
                yv = y[valid]
                loss = crit(out, yv)
                if args.part_loss_weight > 0:
                    pl = o['part_logits'][valid]
                    loss = loss + args.part_loss_weight * crit(pl.reshape(-1, num_classes), yv.repeat_interleave(pl.shape[1]))
                if use_severity:
                    svv = sv[valid]
                    if (svv >= 0).any():
                        loss = loss + args.severity_weight * F.cross_entropy(
                            o['severity'][valid].float(), svv, ignore_index=-1)
                if use_mc:
                    s_logits = model.session_logits(o['emb'].view(nb, K, -1), mask)
                    loss = loss + args.session_weight * crit(s_logits, y.view(nb, K)[:, 0])
                if use_adv:
                    loss = loss + F.cross_entropy(o['subject'][valid].float(), subj_t[flat].to(device)[valid])

                opt.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(model.parameters(), 2.0)
                scaler.step(opt)
                scaler.update()
                sched.step()
                g_step += 1

                ema.update(model)
                ep_loss += loss.item() * len(yv)
                ep_ce += F.cross_entropy(out.detach().float(), yv, reduction='sum').item()
                n_seen += len(yv)
                t_correct += (out.argmax(dim=-1) == yv).sum().item()

            train_loss = ep_loss / max(1, n_seen)
            train_acc = 100.0 * t_correct / max(1, n_seen)
            train_ce = ep_ce / max(1, n_seen)

            v_raw = evaluate_v7(model, data, geom, labels, sev_t, seq_ids, va_idx, device, num_classes,
                                use_amp, tta=not args.no_tta, use_session_head=use_mc)
            ema.copy_to(eval_model)
            v = evaluate_v7(eval_model, data, geom, labels, sev_t, seq_ids, va_idx, device, num_classes,
                            use_amp, tta=not args.no_tta, use_session_head=use_mc)

            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['val_loss'].append(v['loss'])
            history['val_cycle_acc'].append(v['cycle_acc'])
            history['val_sess_acc'].append(v['sess_acc'])
            history['val_f1'].append(v['cycle_f1'])

            with open(csv_path, 'a', encoding='utf-8') as f:
                f.write(f"{ep+1},{train_loss:.4f},{train_acc:.2f},{v['loss']:.4f},{v['cycle_acc']:.2f},"
                        f"{v['sess_acc']:.2f},{v['sess_vote_acc']:.2f},{v['cycle_f1']:.4f},{v['sev_acc']:.2f},"
                        f"{train_ce:.4f},{v_raw['loss']:.4f},{v_raw['cycle_acc']:.2f}\n")

            if is_better(args.select, v['cycle_acc'], v['cycle_f1'], v['loss'], best):
                best = {'acc': v['cycle_acc'], 'f1': v['cycle_f1'], 'loss': v['loss'], 'sess_acc': v['sess_acc'], 'ep': ep + 1}
                best_state = {k: t.cpu().clone() for k, t in eval_model.state_dict().items()}

            sev_txt = f" | Val Sev Acc: {v['sev_acc']:5.2f}%" if use_severity else ""
            print(f"  Ep [{ep+1:2d}/{args.epochs}] | Train Acc: {train_acc:5.2f}% (Loss: {train_loss:.4f}) | "
                  f"Val Cycle Acc: {v['cycle_acc']:5.2f}% | Val Session Acc: {v['sess_acc']:5.2f}%{sev_txt}")

        # Held-out test evaluation
        eval_model.load_state_dict(best_state)
        if args.save_models:
            torch.save(best_state, os.path.join(args.out_dir, f'fold_{f_idx+1}_model.pth'))
        t = evaluate_v7(eval_model, data, geom, labels, sev_t, seq_ids, te_idx, device, num_classes,
                        use_amp, tta=not args.no_tta, use_session_head=use_mc)
        conf = confusion_matrix(t['targets'], t['preds'], num_classes)
        agg_conf += conf

        np.savez_compressed(
            os.path.join(args.out_dir, f'fold_{f_idx+1}_test_predictions.npz'),
            cycle_logits=t['cycle_logits'], cycle_targets=t['targets'],
            cycle_seq_ids=np.array([seq_ids[i] for i in te_idx]), cycle_index=te_idx,
            severity_logits=t['sev_logits'], severity_targets=t['sev_targets'],
            session_ids=t['session_ids'], session_logits=t['session_logits'], session_targets=t['session_targets'],
            class_names=np.array(class_names),
        )

        try:
            plot_fold_curves(history, os.path.join(args.out_dir, f'fold_{f_idx+1}_training_curves.png'), f_idx + 1)
            plot_confusion_matrix_chart(conf, class_names, os.path.join(args.out_dir, f'fold_{f_idx+1}_confusion_matrix.png'),
                                        f"Fold {f_idx+1} Test Confusion Matrix")
        except Exception as e:
            print(f"  [WARN] Plot error: {e}")

        fold_results.append({
            'fold': f_idx + 1,
            'selected_epoch': best['ep'],
            'val_cycle_acc': best['acc'],
            'test_cycle_acc': t['cycle_acc'],
            'test_cycle_f1': t['cycle_f1'],
            'test_sess_acc': t['sess_acc'],
            'test_sess_f1': t['sess_f1'],
            'test_sess_vote_acc': t['sess_vote_acc'],
            'test_sess_head_acc': t['sess_head_acc'],
            'test_severity_acc': t['sev_acc'],
        })

        print("\n" + "=" * 75)
        print(f">> [Fold {f_idx+1}/{n_folds} FINAL TEST RESULT on Unseen Subjects]")
        print(f"   Test Cycle Acc:   {t['cycle_acc']:5.2f}% | Macro-F1: {t['cycle_f1']:.4f}")
        print(f"   Test Session Acc: {t['sess_acc']:5.2f}% | Macro-F1: {t['sess_f1']:.4f} "
              f"({'multi-cycle head' if use_mc else 'vote'}; vote = {t['sess_vote_acc']:.2f}%)")
        if use_severity:
            print(f"   Test Severity Acc (cycle): {t['sev_acc']:5.2f}%")
        print("   Per-Class Test Metrics:")
        for c_idx, c_name in enumerate(class_names):
            c_tp = conf[c_idx, c_idx]
            c_tot_pred = conf[:, c_idx].sum()
            c_tot_true = conf[c_idx, :].sum()
            c_prec = (c_tp / c_tot_pred) if c_tot_pred > 0 else 0.0
            c_rec = (c_tp / c_tot_true) if c_tot_true > 0 else 0.0
            c_f1 = (2 * c_prec * c_rec / (c_prec + c_rec)) if (c_prec + c_rec) > 0 else 0.0
            print(f"     {c_name:<12s}: Precision: {c_prec:.3f} | Recall: {c_rec:.3f} | F1: {c_f1:.3f} (support={c_tot_true})")
        print("=" * 75)

    def ms(key):
        vals = [r[key] for r in fold_results if not (isinstance(r[key], float) and math.isnan(r[key]))]
        return (float(np.mean(vals)), float(np.std(vals))) if vals else (float('nan'), float('nan'))

    print("\n" + "=" * 75)
    print(f"        ANATOPHASENET V7 FINAL BENCHMARK SUMMARY ({tag_ds}, {args.backbone}, {variant})")
    print("=" * 75)
    print(f"Evaluated Folds: {len(fold_results)} / {n_folds}")
    rows = [('test_cycle_acc', 'TEST Cycle-Level Accuracy         '),
            ('test_sess_acc', 'TEST Session-Level Accuracy       '),
            ('test_sess_vote_acc', 'TEST Session Accuracy (vote)      '),
            ('test_sess_head_acc', 'TEST Session Accuracy (multi-cycle)'),
            ('test_severity_acc', 'TEST Severity Accuracy (cycle)    ')]
    for key, label in rows:
        m, s = ms(key)
        if not math.isnan(m):
            print(f"{label}: {m:5.2f}% +/- {s:4.2f}%")
    for key, label in [('test_cycle_f1', 'TEST Cycle-Level Macro-F1         '), ('test_sess_f1', 'TEST Session-Level Macro-F1       ')]:
        m, s = ms(key)
        print(f"{label}: {m:.4f} +/- {s:.4f}")

    print("\nAggregated Confusion Matrix across evaluated test folds:")
    print(f"Classes: {class_names}")
    print(agg_conf)

    try:
        plot_confusion_matrix_chart(agg_conf, class_names, os.path.join(args.out_dir, 'aggregated_confusion_matrix.png'),
                                    "Aggregated Test Confusion Matrix")
    except Exception as e:
        print(f"  [WARN] Aggregated plot error: {e}")

    summary = {
        'dataset': tag_ds,
        'variant': variant,
        'args': vars(args),
        'params': param_count,
        'geometry_signals': active_names,
        'num_cycles': len(labels_np),
        'num_walks': len(set(seq_ids)),
        'cycle_f1_mean': ms('test_cycle_f1')[0],
        'cycle_f1_std': ms('test_cycle_f1')[1],
        'session_f1_mean': ms('test_sess_f1')[0],
        'session_f1_std': ms('test_sess_f1')[1],
        'cycle_accuracy_mean': ms('test_cycle_acc')[0],
        'cycle_accuracy_std': ms('test_cycle_acc')[1],
        'session_accuracy_mean': ms('test_sess_acc')[0],
        'session_accuracy_std': ms('test_sess_acc')[1],
        'session_vote_accuracy_mean': ms('test_sess_vote_acc')[0],
        'session_multicycle_accuracy_mean': ms('test_sess_head_acc')[0],
        'severity_accuracy_mean': ms('test_severity_acc')[0],
        'aggregated_confusion': agg_conf.tolist(),
        'fold_results': fold_results,
    }
    with open(os.path.join(args.out_dir, 'benchmark_summary.json'), 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=4)
    print(f"[SUCCESS] All experiment artifacts saved to: {args.out_dir}")


if __name__ == '__main__':
    main()
