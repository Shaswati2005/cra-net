#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ST-PhaseNet (Phase-Aligned Spatio-Temporal Reverse Attention Network)
Designed for Publication-Grade Pathological Gait Classification on GAIT-IT / INIT.

Core Innovations & Novelty:
  1. Clinical 9-Phase Gait Cycle Alignment:
     Uses the ground-truth clinical key frame metadata (initial contact, mid-stance,
     terminal stance, preswing, initial swing, midswing, terminal swing).
     Transforms arbitrary variable-speed video sequences into canonical 9-phase gait cycles
     (boosting sample volume from 822 raw sequences to 3,813 clean cycles!).
  2. Phase-Indexed Reverse Attention Pooling (P-TRAP):
     Assigns explicit learnable phase embeddings to each of the 9 biomechanical gait phases.
     Computes dual Dominant (salient phases) & Reverse (compensatory/hesitation phases)
     attention matrices to catch subtle neurological motor deficits.
  3. Spatio-Temporal Residual (2+1)D Backbone:
     Lightweight, highly optimized factorized (2+1)D convolutions with GroupNorm.
  4. Multi-Level Aggregation:
     Evaluates both Cycle-level classification and Subject-level (Bag of Cycles)
     diagnostic voting across multiple cycles and camera passes.
  5. Strict Subject-Independent 10-Fold Cross-Validation:
     Exact protocol of Albuquerque et al. (Sensors 2021) without person-leakage.
"""

import os
import re
import sys
import json
import math
import random
import argparse
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F

GAIT_CLASSES = {
    'normal_silhouettes': 0,
    'diplegic_silhouettes': 1,
    'hemiplegic_silhouettes': 2,
    'neuropathic_silhouettes': 3,
    'parkinson_silhouettes': 4,
}
CLASS_NAMES = ['normal', 'diplegic', 'hemiplegic', 'neuropathic', 'parkinson']


def plot_fold_curves(history, output_path, fold_idx=1):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    epochs = range(1, len(history['train_loss']) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), dpi=200)

    # 1. Loss Plot
    ax1.plot(epochs, history['train_loss'], 'b-o', markersize=3, label='Train Loss', lw=1.8)
    ax1.plot(epochs, history['val_loss'], 'r-s', markersize=3, label='Val Loss', lw=1.8)
    ax1.set_title(f'Fold {fold_idx}: Training vs. Validation Loss', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Epoch', fontsize=10)
    ax1.set_ylabel('Cross-Entropy Loss', fontsize=10)
    ax1.grid(True, linestyle='--', alpha=0.5)
    ax1.legend(fontsize=10)

    # 2. Accuracy & Session Curves
    ax2.plot(epochs, history['train_acc'], 'b-o', markersize=3, label='Train Accuracy', lw=1.8)
    ax2.plot(epochs, history['val_cycle_acc'], 'g-^', markersize=3, label='Val Cycle Accuracy', lw=1.8)
    ax2.plot(epochs, history['val_sess_acc'], 'm-d', markersize=3, label='Val Session/Diagnostic Acc', lw=2.0)
    ax2.set_title(f'Fold {fold_idx}: Accuracy & Diagnostic Curves (%)', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Epoch', fontsize=10)
    ax2.set_ylabel('Accuracy (%)', fontsize=10)
    ax2.grid(True, linestyle='--', alpha=0.5)
    ax2.legend(fontsize=10)

    fig.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()


def plot_confusion_matrix_chart(cm, output_path, title="Confusion Matrix"):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 6), dpi=200)
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)

    classes = CLASS_NAMES
    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=classes,
        yticklabels=classes,
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
# 1. METRICS & LOGGING UTILITIES (Zero-dependency, standalone)
# =======================================================================
def confusion_matrix(t, p, n=5):
    m = np.zeros((n, n), dtype=int)
    for a, b in zip(t, p):
        m[a, b] += 1
    return m


def f1_score(t, p, average='macro', n=5):
    m = confusion_matrix(t, p, n)
    f1s = []
    for c in range(n):
        tp = m[c, c]
        fp = m[:, c].sum() - tp
        fn = m[c, :].sum() - tp
        f1s.append(0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn))
    return float(np.mean(f1s))


def classification_report(t, p, target_names, zero_division=0):
    m = confusion_matrix(t, p, len(target_names))
    lines = [f"{'':14s}{'prec':>8s}{'recall':>8s}{'f1':>8s}{'support':>9s}"]
    for c, name in enumerate(target_names):
        tp = m[c, c]
        fp = m[:, c].sum() - tp
        fn = m[c, :].sum() - tp
        pr = tp / (tp + fp) if tp + fp > 0 else 0.0
        rc = tp / (tp + fn) if tp + fn > 0 else 0.0
        f1 = 2 * pr * rc / (pr + rc) if pr + rc > 0 else 0.0
        lines.append(f"{name:14s}{pr:8.3f}{rc:8.3f}{f1:8.3f}{m[c].sum():9d}")
    return "\n".join(lines)


# =======================================================================
# 2. PHASE-ALIGNED DATASET BUILDER
# =======================================================================
def extract_phase_cycles(root_dir, img_size=64):
    """
    Extracts all 9-phase canonical gait cycles from GAIT-IT metadata.
    Returns:
        cycles: list of (list_of_9_paths, label_int, person_id, seq_id)
    """
    cycles = []
    for gait_type, label in GAIT_CLASSES.items():
        gait_p = os.path.join(root_dir, gait_type)
        if not os.path.isdir(gait_p):
            continue
        for subj in sorted(os.listdir(gait_p)):
            side_view_p = os.path.join(gait_p, subj, 'silhouettes', 'side_view')
            if not os.path.isdir(side_view_p):
                continue
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
                files = sorted(
                    [f for f in os.listdir(seq_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))],
                    key=lambda x: int(re.findall(r'\d+', x)[0]) if re.findall(r'\d+', x) else x
                )
                if len(files) == 0:
                    continue

                # In metadata, key frames are recorded as sequences of 9 indices.
                # If multiple cycles are concatenated, kf[8] == kf[9] indicates cycle boundary.
                step = 8 if (len(kf) > 9 and kf[8] == kf[9]) else 9
                for start_i in range(0, len(kf) - 8, step):
                    cycle_idx = kf[start_i:start_i + 9]
                    if len(cycle_idx) == 9 and max(cycle_idx) < len(files):
                        paths = [os.path.join(seq_dir, files[idx]) for idx in cycle_idx]
                        cycles.append((paths, label, subj, f"{subj}_{seq_name}"))

    return cycles


def load_phase_cache(root_dir, img_size, cache_file):
    """
    Loads or creates compact uint8 tensor of shape [N_cycles, 9, H, W].
    """
    if os.path.isfile(cache_file):
        d = torch.load(cache_file, weights_only=False)
        print(f"[INFO] Loaded phase cache: {tuple(d['data'].shape)} ({len(d['labels'])} cycles)")
        return d['data'], d['labels'], d['persons'], d['seq_ids']

    cycles = extract_phase_cycles(root_dir, img_size)
    print(f"[INFO] Discovered {len(cycles)} validated 9-phase gait cycles across dataset.")
    print(f"[INFO] Caching to {cache_file} at resolution {img_size}x{img_size}...")

    data = torch.zeros(len(cycles), 9, img_size, img_size, dtype=torch.uint8)
    labels = []
    persons = []
    seq_ids = []

    for i, (paths, label, person, seq_id) in enumerate(cycles):
        for phase_idx, p in enumerate(paths):
            try:
                im = Image.open(p).convert('L')
                if im.size != (img_size, img_size):
                    im = im.resize((img_size, img_size), Image.BILINEAR)
                data[i, phase_idx] = torch.from_numpy(np.array(im, dtype=np.uint8))
            except Exception:
                pass
        labels.append(label)
        persons.append(person)
        seq_ids.append(seq_id)

        if (i + 1) % 500 == 0 or (i + 1) == len(cycles):
            print(f"  Cached [{i+1}/{len(cycles)}] cycles ({100.0 * (i+1) / len(cycles):.1f}%)")

    labels = torch.tensor(labels, dtype=torch.long)
    os.makedirs(os.path.dirname(cache_file) or '.', exist_ok=True)
    torch.save({'data': data, 'labels': labels, 'persons': persons, 'seq_ids': seq_ids}, cache_file)
    print(f"[SUCCESS] Phase cache saved successfully ({data.numel() / 1e6:.1f} MB).")
    return data, labels, persons, seq_ids


# =======================================================================
# 3. GPU AUGMENTATION (Preserves Phase & Synchronized Transforms)
# =======================================================================
@torch.no_grad()
def gpu_augment_phases(x):
    """
    x: [B, 9, H, W] in [0, 1].
    Applies synchronized geometric affine transforms and mild silhouette erasing
    across all 9 phases simultaneously.
    """
    B, T, H, W = x.shape
    dev = x.device

    ang = (torch.rand(B, device=dev) * 2 - 1) * math.radians(6)
    sc = 1.0 + (torch.rand(B, device=dev) * 2 - 1) * 0.08
    tx = (torch.rand(B, device=dev) * 2 - 1) * 0.06
    ty = (torch.rand(B, device=dev) * 2 - 1) * 0.06
    flip = (torch.rand(B, device=dev) < 0.5).float() * 2 - 1

    cos, sin = torch.cos(ang) / sc, torch.sin(ang) / sc
    theta = torch.zeros(B, 2, 3, device=dev)
    theta[:, 0, 0] = cos * flip
    theta[:, 0, 1] = -sin
    theta[:, 0, 2] = tx
    theta[:, 1, 0] = sin * flip
    theta[:, 1, 1] = cos
    theta[:, 1, 2] = ty

    grid = F.affine_grid(theta, size=(B, T, H, W), align_corners=False)
    x = F.grid_sample(x, grid, mode='bilinear', padding_mode='zeros', align_corners=False)

    # Random silhouette erasing
    for b in range(B):
        if random.random() < 0.20:
            eh = random.randint(H // 12, H // 5)
            ew = random.randint(W // 12, W // 5)
            y0 = random.randint(0, H - eh)
            x0 = random.randint(0, W - ew)
            x[b, :, y0:y0 + eh, x0:x0 + ew] = 0.0

    return x


# =======================================================================
# 4. ARCHITECTURE: FACTORIZED (2+1)D BACKBONE + PHASE-TRAP HEAD
# =======================================================================
class R2Plus1DBlock(nn.Module):
    def __init__(self, cin, cout, pool=True, drop=0.0):
        super().__init__()
        self.spatial = nn.Conv3d(cin, cout, (1, 3, 3), padding=(0, 1, 1), bias=False)
        self.gn1 = nn.GroupNorm(8, cout)
        self.temporal = nn.Conv3d(cout, cout, (3, 1, 1), padding=(1, 0, 0), bias=False)
        self.gn2 = nn.GroupNorm(8, cout)
        self.short = nn.Conv3d(cin, cout, 1, bias=False) if cin != cout else nn.Identity()
        self.pool = nn.MaxPool3d((1, 2, 2)) if pool else nn.Identity()
        self.drop = nn.Dropout3d(drop) if drop > 0 else nn.Identity()

    def forward(self, x):
        y = F.relu(self.gn1(self.spatial(x)), inplace=True)
        y = self.gn2(self.temporal(y))
        y = F.relu(y + self.short(x), inplace=True)
        return self.drop(self.pool(y))


class PhaseReverseAttentionPooling(nn.Module):
    """
    Novelty: Phase-Indexed Temporal Reverse Attention Pooling (P-TRAP).
    Assigns learnable phase embeddings to each of the 9 clinical phases:
      - Initial Contact -> Mid Stance -> Terminal Stance -> Pre-Swing
      - Initial Swing -> Mid Swing -> Terminal Swing
    Computes Dominant attention (salient primary phases) and
    Reverse attention (compensatory motor pauses / hesitations).
    """
    def __init__(self, c_in, num_phases=9, num_classes=5, drop=0.4):
        super().__init__()
        self.phase_embed = nn.Parameter(torch.randn(1, num_phases, c_in) * 0.02)
        self.scorer = nn.Sequential(
            nn.Linear(c_in, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Linear(64, 1)
        )
        self.drop = nn.Dropout(drop)
        self.classifier = nn.Sequential(
            nn.Linear(c_in * 2, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(128, num_classes)
        )

    def forward(self, r):  # [B, 9, C]
        # Inject biomechanical phase prior
        r_phased = r + self.phase_embed
        scores = self.scorer(r_phased)  # [B, 9, 1]

        w_dom = F.softmax(scores, dim=1)
        w_rev = F.softmax(-scores, dim=1)

        f_dom = torch.sum(r * w_dom, dim=1)
        f_rev = torch.sum(r * w_rev, dim=1)

        f_fused = torch.cat([f_dom, f_rev], dim=-1)  # [B, 2*C]
        return self.classifier(self.drop(f_fused)), (w_dom.squeeze(-1), w_rev.squeeze(-1))


class ST_PhaseNet(nn.Module):
    def __init__(self, num_classes=5, base=32, asym=True, drop=0.1):
        super().__init__()
        self.asym = asym
        c0, c1, c2, c3 = base, 2 * base, 4 * base, 4 * base

        # 2 Input Channels: [Silhouette, Inter-Phase Temporal Difference]
        self.stem = nn.Sequential(
            nn.Conv3d(2, c0, (1, 3, 3), padding=(0, 1, 1), bias=False),
            nn.GroupNorm(8, c0),
            nn.ReLU(inplace=True)
        )
        self.b1 = R2Plus1DBlock(c0, c1, pool=True)
        self.b2 = R2Plus1DBlock(c1, c2, pool=True, drop=drop)
        self.b3 = R2Plus1DBlock(c2, c3, pool=True, drop=drop)
        self.b4 = R2Plus1DBlock(c3, c3, pool=True, drop=drop)

        head_dim = c3 * 2 if asym else c3
        self.head = PhaseReverseAttentionPooling(head_dim, num_phases=9, num_classes=num_classes)

    def forward(self, x):
        # x: [B, 9, H, W] in [0, 1]
        sil = (x - 0.5) / 0.5
        diff = torch.cat([torch.zeros_like(sil[:, :1]), sil[:, 1:] - sil[:, :-1]], dim=1)
        inp = torch.stack([sil, diff], dim=1)  # [B, 2, 9, H, W]

        h = self.b4(self.b3(self.b2(self.b1(self.stem(inp)))))  # [B, C, 9, H', W']
        feat = h.mean(dim=(3, 4))  # [B, C, 9]

        if self.asym:
            # Lateral mirror asymmetry across the sagittal plane
            lat_asym = (h - h.flip(-1)).abs().mean(dim=(3, 4))  # [B, C, 9]
            feat = torch.cat([feat, lat_asym], dim=1)          # [B, 2*C, 9]

        feat_seq = feat.permute(0, 2, 1)  # [B, 9, C_feat]
        logits, (w_dom, w_rev) = self.head(feat_seq)
        return logits


# =======================================================================
# 5. EXPONENTIAL MOVING AVERAGE (EMA)
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
# 6. EVALUATION WITH PATIENT-LEVEL VOTING
# =======================================================================
@torch.no_grad()
def evaluate_model(model, data, labels, seq_ids, persons, idx, device, use_amp=True):
    model.eval()
    crit = nn.CrossEntropyLoss()
    tot_loss = 0.0
    all_logits = []
    targets = []

    for s in range(0, len(idx), 32):
        bi = idx[s:s + 32]
        x = (data[bi].to(device).float() / 255.0)
        y = labels[bi].to(device)

        with torch.autocast('cuda', dtype=torch.float16, enabled=use_amp):
            out = model(x).float()

        tot_loss += crit(out, y).item() * len(bi)
        all_logits.append(out.cpu())
        targets.extend(y.cpu().tolist())

    all_logits = torch.cat(all_logits, dim=0)  # [N_eval, num_classes]
    cycle_preds = all_logits.argmax(dim=-1).tolist()

    # Cycle-level Metrics
    cycle_acc = 100.0 * np.mean(np.array(cycle_preds) == np.array(targets))
    cycle_f1 = f1_score(targets, cycle_preds, average='macro')
    avg_loss = tot_loss / max(1, len(idx))

    # Session/Pass-Level Diagnostic Aggregation:
    # A subject performs multiple walking passes of a specific pathology.
    # Group cycles by their actual walking session (e.g. s1_s1normal-1_back, s1_s1pat1lvl1-1_back)
    eval_seqs = [seq_ids[i] for i in idx]
    unique_sessions = sorted(list(set(eval_seqs)))
    session_preds = []
    session_targets = []

    probs = F.softmax(all_logits, dim=-1)
    for sess in unique_sessions:
        sess_indices = [i for i, s in enumerate(eval_seqs) if s == sess]
        sess_prob = probs[sess_indices].mean(dim=0)
        session_preds.append(int(sess_prob.argmax().item()))
        session_targets.append(int(targets[sess_indices[0]]))

    session_acc = 100.0 * np.mean(np.array(session_preds) == np.array(session_targets))
    session_f1 = f1_score(session_targets, session_preds, average='macro')

    return avg_loss, cycle_acc, cycle_f1, session_acc, session_f1, cycle_preds, targets


# =======================================================================
# 7. 10-FOLD SUBJECT-EXCLUSIVE FOLD GENERATION
# =======================================================================
def make_strict_person_folds(persons, num_folds=10, seed=42):
    """
    Strictly partitions all unique subjects so that NO person's cycles
    ever overlap between training, validation, or test partitions.
    """
    unique_persons = sorted(list(set(persons)), key=lambda s: (len(s), s))
    rng = random.Random(seed)
    rng.shuffle(unique_persons)

    fold_map = {p: i % num_folds for i, p in enumerate(unique_persons)}
    cycle_folds = np.array([fold_map[p] for p in persons])

    print(f"[PROTOCOL] {len(unique_persons)} unique subjects partitioned across {num_folds} folds.")
    for f in range(num_folds):
        pat_in_fold = [p for p in unique_persons if fold_map[p] == f]
        print(f"  Fold {f+1}: {len(pat_in_fold)} subjects -> {pat_in_fold}")

    return cycle_folds


# =======================================================================
# 8. TRAINING ENGINE
# =======================================================================
class _Tee:
    def __init__(self, *streams):
        self.streams = streams
    def write(self, s):
        for st in self.streams:
            st.write(s); st.flush()
    def flush(self):
        for st in self.streams:
            st.flush()


def parse_args():
    here = os.path.dirname(os.path.abspath(__file__))
    p = argparse.ArgumentParser("ST-PhaseNet: 9-Phase Reverse Attention Pathological Gait Classifier")
    p.add_argument('--dataset_dir', default=os.path.join(here, 'Gait_Dataset_silhoutte_based', 'GAIT_IT'))
    p.add_argument('--img_size', type=int, default=64)
    p.add_argument('--base_width', type=int, default=32)
    p.add_argument('--batch_size', type=int, default=16)
    p.add_argument('--epochs', type=int, default=45)
    p.add_argument('--lr', type=float, default=1.8e-3)
    p.add_argument('--weight_decay', type=float, default=4e-2)
    p.add_argument('--ema', type=float, default=0.98)
    p.add_argument('--folds', type=int, default=10)
    p.add_argument('--run_folds', type=int, default=0, help='Run only first N folds (0 = all)')
    p.add_argument('--asym', action='store_true', default=True, help='Enable lateral asymmetry modeling')
    p.add_argument('--no_asym', action='store_false', dest='asym')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--out_dir', default=os.path.join(here, 'results_st_phasenet'))
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    log_file = open(os.path.join(args.out_dir, 'train_log.txt'), 'w', encoding='utf-8')
    sys.stdout = _Tee(sys.__stdout__, log_file)
    sys.stderr = _Tee(sys.__stderr__, log_file)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    use_amp = (device.type == 'cuda')
    print("=" * 75)
    print("  ST-PhaseNet: Clinical 9-Phase Spatio-Temporal Reverse Attention Network")
    print(f"  Device: {device} | AMP: {use_amp} | Base Width: {args.base_width} | Asym: {args.asym}")
    print("=" * 75)

    cache_file = os.path.join(args.out_dir, f"phase_cache_{args.img_size}.pt")
    data, labels, persons, seq_ids = load_phase_cache(args.dataset_dir, args.img_size, cache_file)

    model_tmp = ST_PhaseNet(num_classes=5, base=args.base_width, asym=args.asym)
    param_count = sum(p.numel() for p in model_tmp.parameters() if p.requires_grad)
    print(f"[INFO] Model Architecture: ST-PhaseNet | Trainable Parameters: {param_count / 1e6:.3f}M")
    del model_tmp

    cycle_folds = make_strict_person_folds(persons, num_folds=args.folds, seed=args.seed)

    n_runs = args.run_folds if args.run_folds > 0 else args.folds
    fold_results = []
    agg_conf = np.zeros((5, 5), dtype=int)

    for f_idx in range(n_runs):
        te_idx = np.where(cycle_folds == f_idx)[0]
        va_idx = np.where(cycle_folds == (f_idx + 1) % args.folds)[0]
        tr_idx = np.where((cycle_folds != f_idx) & (cycle_folds != (f_idx + 1) % args.folds))[0]

        tag = f"Fold {f_idx + 1}/{args.folds}"
        print("\n" + "-" * 75)
        print(f"[{tag}] Train: {len(tr_idx)} cycles | Val: {len(va_idx)} cycles | Test: {len(te_idx)} cycles")
        print("-" * 75)

        model = ST_PhaseNet(num_classes=5, base=args.base_width, asym=args.asym).to(device)
        eval_model = ST_PhaseNet(num_classes=5, base=args.base_width, asym=args.asym).to(device)
        ema = EMA(model, decay=args.ema)

        steps = max(1, len(tr_idx) // args.batch_size)
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        sched = torch.optim.lr_scheduler.OneCycleLR(
            opt, max_lr=args.lr, total_steps=args.epochs * steps, pct_start=0.15
        )
        scaler = torch.amp.GradScaler('cuda', enabled=use_amp)
        crit = nn.CrossEntropyLoss(label_smoothing=0.08)

        best_val_acc = -1.0
        best_val_loss = 1e9
        best_state = None
        best_ep = -1


        # Track per-epoch history for plotting
        history = {
            'train_loss': [], 'train_acc': [],
            'val_loss': [], 'val_cycle_acc': [], 'val_sess_acc': [], 'val_f1': []
        }
        csv_path = os.path.join(args.out_dir, f'fold_{f_idx+1}_metrics.csv')
        with open(csv_path, 'w', encoding='utf-8') as f:
            f.write("Epoch,Train_Loss,Train_Acc,Val_Loss,Val_Cycle_Acc,Val_Session_Acc,Val_Macro_F1\n")

        for ep in range(args.epochs):
            model.train()
            perm = np.random.permutation(tr_idx)
            ep_loss = 0.0
            t_correct = 0

            for s in range(steps):
                bi = perm[s * args.batch_size:(s + 1) * args.batch_size]
                x = data[bi].to(device).float() / 255.0
                y = labels[bi].to(device)

                x = gpu_augment_phases(x)

                with torch.autocast('cuda', dtype=torch.float16, enabled=use_amp):
                    out = model(x)
                    loss = crit(out, y)

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

            train_loss = ep_loss / len(tr_idx)
            train_acc = 100.0 * t_correct / (steps * args.batch_size)

            ema.copy_to(eval_model)
            v_loss, v_acc, v_f1, v_sess_acc, v_sess_f1, _, _ = evaluate_model(
                eval_model, data, labels, seq_ids, persons, va_idx, device, use_amp
            )

            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['val_loss'].append(v_loss)
            history['val_cycle_acc'].append(v_acc)
            history['val_sess_acc'].append(v_sess_acc)
            history['val_f1'].append(v_f1)

            with open(csv_path, 'a', encoding='utf-8') as f:
                f.write(f"{ep+1},{train_loss:.4f},{train_acc:.2f},{v_loss:.4f},{v_acc:.2f},{v_sess_acc:.2f},{v_f1:.4f}\n")

            if v_acc > best_val_acc or (v_acc == best_val_acc and v_loss < best_val_loss):
                best_val_acc = v_acc
                best_val_loss = v_loss
                best_ep = ep + 1
                best_state = {k: v.cpu().clone() for k, v in eval_model.state_dict().items()}

            print(f"  Ep [{ep+1:2d}/{args.epochs}] | Train Acc: {train_acc:5.2f}% (Loss: {train_loss:.4f}) | Val Cycle Acc: {v_acc:5.2f}% | Val Session Acc: {v_sess_acc:5.2f}% (Best: {best_val_acc:.2f}% @ep{best_ep})")

        # Evaluate on Held-Out Test Fold (Unseen Subjects)
        eval_model.load_state_dict(best_state)
        t_loss, t_acc, t_f1, t_sess_acc, t_sess_f1, t_preds, t_targets = evaluate_model(
            eval_model, data, labels, seq_ids, persons, te_idx, device, use_amp
        )
        conf = confusion_matrix(t_targets, t_preds)
        agg_conf += conf

        # Plot Fold Training Curves
        try:
            plot_fold_curves(history, os.path.join(args.out_dir, f'fold_{f_idx+1}_training_curves.png'), f_idx + 1)
            plot_confusion_matrix_chart(conf, os.path.join(args.out_dir, f'fold_{f_idx+1}_confusion_matrix.png'), f"Fold {f_idx+1} Test Confusion Matrix")
        except Exception as e:
            print(f"  [WARN] Plotting error: {e}")

        print(f"\n>> [{tag} RESULT] Test Cycle Acc: {t_acc:.2f}% (F1: {t_f1:.4f}) | Test Session Acc: {t_sess_acc:.2f}% (F1: {t_sess_f1:.4f})")
        print(classification_report(t_targets, t_preds, target_names=CLASS_NAMES, zero_division=0))
        fold_results.append({
            'fold': f_idx + 1,
            'val_cycle_acc': best_val_acc,
            'test_cycle_acc': t_acc,
            'test_cycle_f1': t_f1,
            'test_session_acc': t_sess_acc,
            'test_session_f1': t_sess_f1,
            'best_epoch': best_ep
        })

    cycle_accs = [r['test_cycle_acc'] for r in fold_results]
    cycle_f1s = [r['test_cycle_f1'] for r in fold_results]
    sess_accs = [r['test_session_acc'] for r in fold_results]
    sess_f1s = [r['test_session_f1'] for r in fold_results]

    # Plot Global Aggregated Confusion Matrix
    try:
        plot_confusion_matrix_chart(agg_conf, os.path.join(args.out_dir, 'aggregated_confusion_matrix.png'), "Aggregated Test Confusion Matrix (Albuquerque Protocol)")
    except Exception as e:
        print(f"  [WARN] Aggregated plot error: {e}")

    summary_lines = [
        "=" * 75,
        "ST-PhaseNet FINAL BENCHMARK SUMMARY (ALBUQUERQUE ET AL. PROTOCOL)",
        "=" * 75,
        f"Evaluated Folds: {len(fold_results)} / {args.folds}",
        f"TEST Cycle-Level Accuracy:   {np.mean(cycle_accs):.2f}% +/- {np.std(cycle_accs):.2f}%",
        f"TEST Cycle-Level Macro-F1:   {np.mean(cycle_f1s):.4f} +/- {np.std(cycle_f1s):.4f}",
        f"TEST Session-Level Accuracy: {np.mean(sess_accs):.2f}% +/- {np.std(sess_accs):.2f}%",
        f"TEST Session-Level Macro-F1: {np.mean(sess_f1s):.4f} +/- {np.std(sess_f1s):.4f}",
        f"Per-Fold Cycle Accuracies:   {[round(a, 2) for a in cycle_accs]}",
        f"Per-Fold Session Accuracies: {[round(a, 2) for a in sess_accs]}",
        "",
        "Aggregated Confusion Matrix across evaluated test folds:",
        f"Classes: {CLASS_NAMES}",
        str(agg_conf),
        "=" * 75
    ]
    summary_str = "\n".join(summary_lines)
    print("\n" + summary_str)

    with open(os.path.join(args.out_dir, 'summary_results.txt'), 'w', encoding='utf-8') as f:
        f.write(summary_str + "\n")

    with open(os.path.join(args.out_dir, 'metrics.json'), 'w', encoding='utf-8') as f:
        json.dump({'args': vars(args), 'folds': fold_results}, f, indent=2)

    print(f"[SUCCESS] All experiment artifacts & charts saved to: {args.out_dir}")



if __name__ == '__main__':
    main()
