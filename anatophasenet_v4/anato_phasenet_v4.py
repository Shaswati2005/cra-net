"""
===============================================================================
AnatoPhaseNet v4: Direction-Canonical, Part-Based Phase Network
for GAIT-IT (5 classes) and INIT (8 or 6 classes)
===============================================================================

What changed vs. v3 (see PLAN.md for the reasoning behind each item):
1. Walking-direction canonicalisation: GAIT-IT '_back' sequences are mirror images
   of '_front' ones (corr 0.65 raw -> 0.996 after h-flip). v4 flips them so every
   cycle walks the same way. INIT sequences are flipped from their centroid motion.
2. Backbone (--backbone):
   r2plus1d (default, main model): 3D (2+1)D ResNet-18-style network over the 9 phases.
     Every block = spatial (1x3x3) -> CYCLIC temporal (3x1x1) residual -> spatial (1x3x3).
     All spatial conv + BN weights inflated from ImageNet ResNet-18 (both blocks per stage);
     the temporal branch is zero-initialised so training starts from the pretrained filters.
     v3 had one block per stage, copied only conv1 of each stage, no BN stats, zero padding.
   tsm (ablation): the same 2D ResNet per phase with zero-parameter cyclic Temporal Shift.
3. Feature map kept at 6x6 (layer4 dropped) instead of 3x3, so anatomy is resolved
   by 6 horizontal strips + 1 global strip (GaitSet/PCB-style) instead of a
   1-row / 2-row upper/lower split.
4. Gait-symmetry feature replaced: |phase t - phase t+half-cycle| compares the
   left-leg stance half with the right-leg stance half (true bilateral symmetry in a
   sagittal view). v3's |h - flip_W(h)| measured front/back mirroring, i.e. direction.
5. Per-part classifiers averaged into the final logit (ensemble-like regulariser).
6. Phase-jitter augmentation (+/- a few frames around each key frame), silhouette
   morphology noise, sqrt class-balanced sampling, temporal TTA at evaluation.
7. INIT loader: gait cycles detected from the bounding-box width signal
   (double-support peaks), 9 phases sampled uniformly within each cycle.

Same subject-exclusive fold splitter and seed as v3, so GAIT-IT folds are identical.
===============================================================================
"""

import os
import re
import math
import json
import random
import argparse
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F

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

    y_min = max(0, y_min - pad_y)
    y_max = min(arr.shape[0], y_max + pad_y)
    x_min = max(0, x_min - pad_x)
    x_max = min(arr.shape[1], x_max + pad_x)

    cropped = im.crop((x_min, y_min, x_max, y_max))

    cw, ch = cropped.size
    scale = target_size / max(cw, ch)
    nw, nh = max(1, int(cw * scale)), max(1, int(ch * scale))
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

                cycles = []
                step = 8 if (len(kf) > 9 and kf[8] == kf[9]) else 9
                for start_i in range(0, len(kf) - 8, step):
                    cycle_idx = kf[start_i:start_i + 9]
                    if len(cycle_idx) == 9 and max(cycle_idx) < len(files):
                        cycles.append(jitter_grid(cycle_idx, offsets, len(files)))
                if cycles:
                    seqs.append(dict(
                        dir=seq_dir, files=files, label=label, person=subj,
                        seq_id=f"{subj}_{seq_name}",
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


class PartPhaseHead(nn.Module):
    """
    Part-based temporal head.
    Input  h: [B, T, C, Hf, Wf]
    Parts: `num_parts` horizontal strips (head -> feet) + 1 global strip.
    Per part: avg+max pooling -> separate projection (HPM) -> circular temporal conv
    -> [attention pool | max pool | half-cycle symmetry] -> own BN + classifier.
    Returns per-part logits [B, P+1, K].
    """
    def __init__(self, in_ch, num_classes, num_parts=6, num_phases=9, dim=128, dropout=0.3):
        super().__init__()
        self.num_parts = num_parts
        self.T = num_phases
        P = num_parts + 1

        self.proj = nn.Parameter(torch.empty(P, 2 * in_ch, dim))
        nn.init.xavier_uniform_(self.proj)
        self.phase_embed = nn.Parameter(torch.randn(1, num_phases, P, dim) * 0.02)

        self.tconv = nn.Sequential(
            nn.Conv1d(P * dim, P * dim, 3, padding=1, padding_mode='circular', groups=P, bias=False),
            nn.BatchNorm1d(P * dim),
            nn.ReLU(inplace=True),
        )
        self.attn = nn.Parameter(torch.zeros(P, dim))

        self.bn = nn.BatchNorm1d(P * 3 * dim)
        self.drop = nn.Dropout(dropout)
        self.cls_w = nn.Parameter(torch.empty(P, 3 * dim, num_classes))
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

        f = torch.einsum('npc,pcd->npd', part, self.proj)                                  # [BT, P, D]
        D = f.shape[-1]
        f = f.view(B, T, P, D) + self.phase_embed

        # Circular temporal conv per part (grouped), residual
        z = f.permute(0, 2, 3, 1).reshape(B, P * D, T)
        z = self.tconv(z).view(B, P, D, T).permute(0, 3, 1, 2)                              # [B, T, P, D]
        f = f + z

        # Temporal attention pooling per part
        score = torch.einsum('btpd,pd->btp', f, self.attn)
        a = F.softmax(score, dim=1).unsqueeze(-1)
        att_pool = (f * a).sum(dim=1)                                                       # [B, P, D]
        max_pool = f.max(dim=1).values                                                      # [B, P, D]

        # Half-cycle bilateral symmetry: stance of one leg vs. stance of the other
        half = (T - 1) // 2
        sym = (f[:, :half] - f[:, half:2 * half]).abs().mean(dim=1)                         # [B, P, D]

        g = torch.cat([att_pool, max_pool, sym], dim=-1)                                    # [B, P, 3D]
        g = self.bn(g.reshape(B, -1)).view(B, P, -1)
        g = self.drop(g)
        return torch.einsum('bpd,pdk->bpk', g, self.cls_w) + self.cls_b                     # [B, P, K]


class AnatoPhaseNet_v4(nn.Module):
    def __init__(self, num_classes, pretrained_path=None, num_phases=9, num_parts=6,
                 last_stride=2, head_dim=128, dropout=0.3, backbone='r2plus1d'):
        super().__init__()
        self.T = num_phases

        # ResNet-18 stem + layer1..layer3 (layer4 dropped to keep a 6x6 map at 96px)
        if backbone == 'r2plus1d':
            self.backbone = R2Plus1DBackbone(num_phases, last_stride)
        else:
            self.backbone = TSMBackbone(num_phases, last_stride)

        self.head = PartPhaseHead(256, num_classes, num_parts=num_parts, num_phases=num_phases,
                                  dim=head_dim, dropout=dropout)

        if pretrained_path and os.path.isfile(pretrained_path):
            self._load_pretrained(pretrained_path)

    def _load_pretrained(self, path):
        """Copies ImageNet ResNet-18 conv + BN tensors; 2D kernels are inflated to (1, k, k) for 3D."""
        ckpt = torch.load(path, weights_only=True)
        own = self.backbone.state_dict()
        keep = {}
        for k, v in ckpt.items():
            if k.split('.')[0] not in ('conv1', 'bn1', 'layer1', 'layer2', 'layer3') or k not in own:
                continue
            if own[k].dim() == 5 and v.dim() == 4:
                v = v.unsqueeze(2)
            if own[k].shape == v.shape:
                keep[k] = v
        missing, _ = self.backbone.load_state_dict(keep, strict=False)
        new_keys = sorted({k.split('.')[-2] for k in missing})
        print(f"[INFO] Loaded {len(keep)} ImageNet ResNet-18 tensors (conv + BN) from {path}. "
              f"Newly initialised backbone tensors: {len(missing)} {new_keys}")

    def backbone_parameters(self):
        return list(self.backbone.parameters())

    def head_parameters(self):
        return list(self.head.parameters())

    def forward(self, x, return_parts=False):
        # x: [B, 9, H, W] in [0, 1]
        B, T, H, W = x.shape
        sil = (x - 0.5) / 0.5
        motion = x - x.roll(1, dims=1)                      # cyclic inter-phase difference
        dev = x - x.mean(dim=1, keepdim=True)               # deviation from the cycle's GEI
        inp = torch.stack([sil, motion, dev], dim=1)        # [B, 3, T, H, W]

        h = self.backbone(inp)                              # [B, T, 256, Hf, Wf]

        part_logits = self.head(h)
        logits = part_logits.mean(dim=1)
        if return_parts:
            return logits, part_logits
        return logits

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

@torch.no_grad()
def evaluate_model(model, data, labels, seq_ids, idx, device, num_classes, use_amp=True, tta=True):
    model.eval()
    crit = nn.CrossEntropyLoss()
    J = data.shape[1]
    views = list(range(J)) if (tta and J > 1) else [J // 2]

    all_logits = []
    targets = []
    tot_loss = 0.0
    bs = 32
    for b in range(int(math.ceil(len(idx) / bs))):
        b_idx = idx[b * bs:(b + 1) * bs]
        y = labels[b_idx].to(device)
        logit_sum = 0.0
        for j in views:
            x = data[b_idx, j].to(device).float() / 255.0
            with torch.autocast('cuda', dtype=torch.float16, enabled=use_amp):
                logit_sum = logit_sum + model(x).float()
        out = logit_sum / len(views)
        tot_loss += crit(out, y).item() * len(b_idx)
        all_logits.append(out.cpu())
        targets.extend(y.cpu().tolist())

    all_logits = torch.cat(all_logits, dim=0)
    log_probs = F.log_softmax(all_logits, dim=-1).numpy()
    targets = np.array(targets)
    preds = np.argmax(log_probs, axis=-1)

    cycle_loss = tot_loss / max(1, len(idx))
    cycle_acc = float(np.mean(preds == targets) * 100.0)
    cycle_f1 = compute_macro_f1(targets, preds, num_classes)

    # Multi-cycle session consensus (mean log-probability per sequence)
    groups = {}
    for pos, orig_i in enumerate(idx):
        groups.setdefault(seq_ids[orig_i], []).append(pos)
    session_preds, session_targets = [], []
    for s_id in sorted(groups):
        s_pos = groups[s_id]
        session_preds.append(int(np.argmax(log_probs[s_pos].mean(axis=0))))
        session_targets.append(int(targets[s_pos[0]]))
    session_preds = np.array(session_preds)
    session_targets = np.array(session_targets)
    session_acc = float(np.mean(session_preds == session_targets) * 100.0)
    session_f1 = compute_macro_f1(session_targets, session_preds, num_classes)

    return cycle_loss, cycle_acc, cycle_f1, session_acc, session_f1, preds, targets

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

# =======================================================================
# 8. MAIN EXPERIMENT RUNNER
# =======================================================================
def main():
    here = os.path.dirname(os.path.abspath(__file__))
    parent = os.path.dirname(here)
    data_root = os.path.join(parent, 'Gait_Dataset_silhoutte_based')
    default_dirs = {
        'gait_it': os.path.join(data_root, 'GAIT_IT'),
        'init': os.path.join(data_root, 'INIT', 'INIT_GaitDB', 'INIT_8', 'silhouettes'),
    }

    parser = argparse.ArgumentParser(description="AnatoPhaseNet v4 Pathological Gait Classifier")
    parser.add_argument('--dataset', type=str, default='gait_it', choices=['gait_it', 'init'])
    parser.add_argument('--dataset_dir', type=str, default=None, help='Override dataset path')
    parser.add_argument('--init_classes', type=int, default=8, choices=[8, 6],
                        help='INIT protocol: 8 classes, or 6 with a-*0 / a-*0.5 merged')
    parser.add_argument('--pretrained_path', type=str, default=os.path.join(parent, 'resnet18_pretrained.pth'))
    parser.add_argument('--out_dir', type=str, default=None, help='Output folder (default: results_<dataset>)')
    parser.add_argument('--img_size', type=int, default=96)
    parser.add_argument('--jitter', type=int, default=2, help='Phase jitter in frames (0 disables)')
    parser.add_argument('--no_canon', action='store_true', help='Disable walking-direction canonicalisation')
    parser.add_argument('--backbone', type=str, default='r2plus1d', choices=['r2plus1d', 'tsm'],
                        help='r2plus1d: 3D (2+1)D phase convolutions (main model); tsm: 2D ResNet + temporal shift (ablation)')
    parser.add_argument('--num_parts', type=int, default=6, help='Horizontal body strips')
    parser.add_argument('--last_stride', type=int, default=2, choices=[1, 2], help='layer3 stride (1 -> 12x12 map)')
    parser.add_argument('--head_dim', type=int, default=128)
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-3, help='Head peak LR')
    parser.add_argument('--backbone_lr_mult', type=float, default=0.3, help='Backbone LR = lr * mult')
    parser.add_argument('--weight_decay', type=float, default=5e-2)
    parser.add_argument('--focal_gamma', type=float, default=1.5)
    parser.add_argument('--label_smoothing', type=float, default=0.05)
    parser.add_argument('--part_loss_weight', type=float, default=0.5, help='Weight of per-part auxiliary losses')
    parser.add_argument('--balance', type=str, default='sqrt', choices=['sqrt', 'none'], help='Class-balanced sampling')
    parser.add_argument('--ema', type=float, default=0.98)
    parser.add_argument('--select', type=str, default='last', choices=['acc', 'f1', 'loss', 'last'],
                        help='Validation criterion for picking the checkpoint')
    parser.add_argument('--no_tta', action='store_true', help='Disable temporal-jitter test-time augmentation')
    parser.add_argument('--folds', type=int, default=10)
    parser.add_argument('--run_folds', type=int, default=10, help='Run the first N folds')
    parser.add_argument('--fold_ids', type=int, nargs='*', default=None, help='Run only these folds (1-based)')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--save_models', action='store_true')
    args = parser.parse_args()

    args.dataset_dir = args.dataset_dir or default_dirs[args.dataset]
    tag_ds = args.dataset if args.dataset == 'gait_it' else f"init{args.init_classes}"
    args.out_dir = args.out_dir or os.path.join(here, f"results_{tag_ds}_{args.backbone}")
    os.makedirs(args.out_dir, exist_ok=True)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    use_amp = (device.type == 'cuda')
    print("=" * 75)
    print(f"  AnatoPhaseNet v4 | dataset: {tag_ds} | backbone: {args.backbone} | Device: {device} | AMP: {use_amp}")
    print(f"  Resolution: {args.img_size} | Parts: {args.num_parts} | Jitter: +/-{args.jitter} | "
          f"Canonical direction: {not args.no_canon}")
    print("=" * 75)

    cache_dir = os.path.join(here, 'cache')
    cache_file = os.path.join(
        cache_dir, f"{tag_ds}_{args.img_size}_j{args.jitter}{'_nocanon' if args.no_canon else ''}.pt")
    d = load_cache(args, cache_file)
    data, labels, persons, seq_ids = d['data'], d['labels'], d['persons'], d['seq_ids']
    class_names = d['class_names']
    num_classes = len(class_names)
    labels_np = labels.numpy()
    print(f"[INFO] Classes: {class_names} | cycles per class: {np.bincount(labels_np, minlength=num_classes).tolist()}")

    model_kw = dict(num_parts=args.num_parts, last_stride=args.last_stride,
                    head_dim=args.head_dim, dropout=args.dropout, backbone=args.backbone)
    model_tmp = AnatoPhaseNet_v4(num_classes, None, **model_kw)
    param_count = sum(p.numel() for p in model_tmp.parameters() if p.requires_grad)
    print(f"[INFO] Architecture: AnatoPhaseNet v4 ({args.backbone}) | Parameters: {param_count / 1e6:.3f}M")
    del model_tmp

    n_folds = min(args.folds, len(set(persons)))
    cycle_folds = make_strict_person_folds(persons, num_folds=n_folds, seed=args.seed)

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

        print("\n" + "-" * 75)
        print(f"[Fold {f_idx + 1}/{n_folds}] Train: {len(tr_idx)} cycles | Val: {len(va_idx)} cycles | Test: {len(te_idx)} cycles")
        print("-" * 75)

        model =AnatoPhaseNet_v4(num_classes, args.pretrained_path, **model_kw).to(device)
        eval_model = AnatoPhaseNet_v4(num_classes, None, **model_kw).to(device)
        ema = EMA(model, decay=args.ema)

        steps = max(1, len(tr_idx) // args.batch_size)
        opt = torch.optim.AdamW([
            {'params': model.backbone_parameters(), 'lr': args.lr * args.backbone_lr_mult},
            {'params': model.head_parameters(), 'lr': args.lr},
        ], weight_decay=args.weight_decay)
        sched = torch.optim.lr_scheduler.OneCycleLR(
            opt, max_lr=[args.lr * args.backbone_lr_mult, args.lr],
            total_steps=args.epochs * steps, pct_start=0.15
        )
        scaler = torch.amp.GradScaler('cuda', enabled=use_amp)
        crit = FocalCrossEntropyLoss(gamma=args.focal_gamma, label_smoothing=args.label_smoothing)
        samp_w = make_sampler_weights(labels_np, tr_idx, args.balance)

        best = None
        best_state = None
        history = {
            'train_loss': [], 'train_acc': [],
            'val_loss': [], 'val_cycle_acc': [], 'val_sess_acc': [], 'val_f1': []
        }
        csv_path = os.path.join(args.out_dir, f'fold_{f_idx+1}_metrics.csv')
        with open(csv_path, 'w', encoding='utf-8') as f:
            f.write("Epoch,Train_Loss,Train_Acc,Val_Loss,Val_Cycle_Acc,Val_Session_Acc,Val_Macro_F1\n")

        for ep in range(args.epochs):
            model.train()
            if samp_w is None:
                perm = np.random.permutation(tr_idx)
            else:
                perm = np.random.choice(tr_idx, size=steps * args.batch_size, replace=True, p=samp_w)
            ep_loss = 0.0
            t_correct = 0

            for s in range(steps):
                bi = perm[s * args.batch_size:(s + 1) * args.batch_size]
                x = gather_phases(data, bi, train=True).to(device).float() / 255.0
                y = labels[bi].to(device)
                x = gpu_augment_phases(x)

                with torch.autocast('cuda', dtype=torch.float16, enabled=use_amp):
                    out, part_out = model(x, return_parts=True)
                loss = crit(out, y)
                if args.part_loss_weight > 0:
                    P = part_out.shape[1]
                    loss = loss + args.part_loss_weight * crit(
                        part_out.reshape(-1, num_classes), y.repeat_interleave(P))

                opt.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(model.parameters(), 2.0)
                scaler.step(opt)
                scaler.update()
                sched.step()

                ema.update(model)
                ep_loss += loss.item() * len(bi)
                t_correct += (out.argmax(dim=-1) == y).sum().item()

            train_loss = ep_loss / (steps * args.batch_size)
            train_acc = 100.0 * t_correct / (steps * args.batch_size)

            ema.copy_to(eval_model)
            v_loss, v_acc, v_f1, v_sess_acc, v_sess_f1, _, _ = evaluate_model(
                eval_model, data, labels, seq_ids, va_idx, device, num_classes, use_amp, tta=not args.no_tta
            )

            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['val_loss'].append(v_loss)
            history['val_cycle_acc'].append(v_acc)
            history['val_sess_acc'].append(v_sess_acc)
            history['val_f1'].append(v_f1)

            with open(csv_path, 'a', encoding='utf-8') as f:
                f.write(f"{ep+1},{train_loss:.4f},{train_acc:.2f},{v_loss:.4f},{v_acc:.2f},{v_sess_acc:.2f},{v_f1:.4f}\n")

            if is_better(args.select, v_acc, v_f1, v_loss, best):
                best = {'acc': v_acc, 'f1': v_f1, 'loss': v_loss, 'sess_acc': v_sess_acc, 'ep': ep + 1}
                best_state = {k: v.cpu().clone() for k, v in eval_model.state_dict().items()}

            print(f"  Ep [{ep+1:2d}/{args.epochs}] | Train Acc: {train_acc:5.2f}% (Loss: {train_loss:.4f}) | "
                  f"Val Cycle Acc: {v_acc:5.2f}% | Val Session Acc: {v_sess_acc:5.2f}% "
                  f"(Best: {best['acc']:.2f}% @ep{best['ep']})")

        # Held-out Test Evaluation
        eval_model.load_state_dict(best_state)
        if args.save_models:
            torch.save(best_state, os.path.join(args.out_dir, f'fold_{f_idx+1}_best.pth'))
        t_loss, t_acc, t_f1, t_sess_acc, t_sess_f1, t_preds, t_targets = evaluate_model(
            eval_model, data, labels, seq_ids, te_idx, device, num_classes, use_amp, tta=not args.no_tta
        )
        conf = confusion_matrix(t_targets, t_preds, num_classes)
        agg_conf += conf

        try:
            plot_fold_curves(history, os.path.join(args.out_dir, f'fold_{f_idx+1}_training_curves.png'), f_idx + 1)
            plot_confusion_matrix_chart(conf, class_names, os.path.join(args.out_dir, f'fold_{f_idx+1}_confusion_matrix.png'),
                                        f"Fold {f_idx+1} Test Confusion Matrix")
        except Exception as e:
            print(f"  [WARN] Plot error: {e}")

        fold_results.append({
            'fold': f_idx + 1,
            'best_epoch': best['ep'],
            'val_cycle_acc': best['acc'],
            'val_sess_acc': best['sess_acc'],
            'test_cycle_acc': t_acc,
            'test_cycle_f1': t_f1,
            'test_sess_acc': t_sess_acc,
            'test_sess_f1': t_sess_f1
        })

        print("\n" + "=" * 75)
        print(f">> [Fold {f_idx+1}/{n_folds} FINAL TEST RESULT on Unseen Subjects]")
        print(f"   Val Cycle Acc (selected ep{best['ep']}): {best['acc']:5.2f}%")
        print(f"   Test Cycle Acc:   {t_acc:5.2f}% | Macro-F1: {t_f1:.4f}")
        print(f"   Test Session Acc: {t_sess_acc:5.2f}% | Macro-F1: {t_sess_f1:.4f}")
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

    # Aggregated Summary across all folds
    def ms(key):
        vals = [r[key] for r in fold_results]
        return float(np.mean(vals)), float(np.std(vals))

    print("\n" + "=" * 75)
    print(f"                 ANATOPHASENET V4 FINAL BENCHMARK SUMMARY ({tag_ds})")
    print("=" * 75)
    print(f"Evaluated Folds: {len(fold_results)} / {n_folds}")
    for key, label in [('val_cycle_acc', 'VAL  Cycle-Level Accuracy  '), ('test_cycle_acc', 'TEST Cycle-Level Accuracy  '),
                       ('test_sess_acc', 'TEST Session-Level Accuracy')]:
        m, s = ms(key)
        print(f"{label}: {m:5.2f}% +/- {s:4.2f}%")
    for key, label in [('test_cycle_f1', 'TEST Cycle-Level Macro-F1  '), ('test_sess_f1', 'TEST Session-Level Macro-F1')]:
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
        'args': vars(args),
        'params': param_count,
        'val_cycle_accuracy_mean': ms('val_cycle_acc')[0],
        'cycle_accuracy_mean': ms('test_cycle_acc')[0],
        'cycle_accuracy_std': ms('test_cycle_acc')[1],
        'session_accuracy_mean': ms('test_sess_acc')[0],
        'session_accuracy_std': ms('test_sess_acc')[1],
        'aggregated_confusion': agg_conf.tolist(),
        'fold_results': fold_results,
    }
    with open(os.path.join(args.out_dir, 'benchmark_summary.json'), 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=4)
    print(f"[SUCCESS] All experiment artifacts saved to: {args.out_dir}")


if __name__ == '__main__':
    main()
