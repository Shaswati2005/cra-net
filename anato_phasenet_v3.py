"""
===============================================================================
AnatoPhaseNet v3: Biomechanically Decoupled Anatomical Phase Network
with Leg-Specific Asymmetry, Trunk Dynamics & Focal Loss
===============================================================================

Key Enhancements in v3 to target Diplegic & Hemiplegic accuracy:
1. Leg-Specific Lateral Asymmetry (Decouples Hemiplegic vs. Diplegic):
   - Computes sagittal lateral asymmetry specifically on H_lower (|H_lower - H_lower_flip|).
   - In Hemiplegia, unilateral leg circumduction creates a massive asymmetry spike.
   - In Diplegia, bilateral spastic scissoring is symmetric across both legs (near zero asymmetry).
2. Dynamic Cephalo-Thoracic Trunk Mobility Metric (Decouples Diplegic vs. Parkinsonian):
   - Tracks the temporal variance and trajectory of the Upper Region (Trunk/Arms) across all 9 phases.
   - Parkinsonian gait exhibits rigid, frozen trunk stoop across all phases (low variance).
   - Diplegic gait exhibits dynamic compensatory lateral torso sway (high variance).
3. Soft Focal Loss (gamma=1.5) with Label Smoothing (0.05):
   - Replaces hard class weights (which caused the Fold 3 diplegic drop) with adaptive Focal Loss.
   - Dynamically focuses backpropagation onto hard, ambiguous cycles (mild impairment lvl1)
     while down-weighting well-classified easy samples (Normal, Parkinsonian).
4. Reuses the 96x96 Bounding-Box Cache from v2 (Starts training immediately without re-processing).
5. Comprehensive Fold-by-Fold Plotting and Aggregated Benchmark Logging.
===============================================================================
"""

import os
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

# =======================================================================
# 0. CONFIGURATION & GAIT CLASSES
# =======================================================================
GAIT_CLASSES = {
    'normal_silhouettes': 0,
    'diplegic_silhouettes': 1,
    'hemiplegic_silhouettes': 2,
    'neuropathic_silhouettes': 3,
    'parkinson_silhouettes': 4,
}
CLASS_NAMES = ['normal', 'diplegic', 'hemiplegic', 'neuropathic', 'parkinson']
PHASE_NAMES = ['IC', 'LR', 'MST', 'TST', 'PS', 'ISW', 'MSW', 'TSW', 'COMP']

# =======================================================================
# 1. METRICS & VISUALIZATIONS (Pure-NumPy, No sklearn/scipy)
# =======================================================================
def compute_macro_f1(targets, preds, num_classes=5):
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

def confusion_matrix(targets, preds, num_classes=5):
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

    # Loss subplot
    ax1.plot(epochs, history['train_loss'], label='Train Loss', color='#1f77b4', lw=2)
    ax1.plot(epochs, history['val_loss'], label='Val Loss', color='#d62728', lw=2)
    ax1.set_title(f'Fold {fold_idx} - Loss Progression', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Epoch', fontsize=10)
    ax1.set_ylabel('Loss', fontsize=10)
    ax1.grid(True, linestyle='--', alpha=0.5)
    ax1.legend(fontsize=10)

    # Accuracy subplot
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
# 2. DATASET BUILDER & CACHE LOADER (Reuses 96x96 Bounding Box Cache)
# =======================================================================
import re

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

def extract_phase_cycles(root_dir):
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

                step = 8 if (len(kf) > 9 and kf[8] == kf[9]) else 9
                for start_i in range(0, len(kf) - 8, step):
                    cycle_idx = kf[start_i:start_i + 9]
                    if len(cycle_idx) == 9 and max(cycle_idx) < len(files):
                        paths = [os.path.join(seq_dir, files[idx]) for idx in cycle_idx]
                        cycles.append((paths, label, subj, f"{subj}_{seq_name}"))

    print(f"[INFO] Harvested {len(cycles)} clinical 9-phase gait cycles across {len(set(c[2] for c in cycles))} subjects.")
    return cycles

def load_phase_cache(root_dir, img_size, cache_file):
    if os.path.isfile(cache_file):
        d = torch.load(cache_file, weights_only=False)
        print(f"[INFO] Loaded phase cache: {tuple(d['data'].shape)} ({len(d['labels'])} cycles)")
        return d['data'], d['labels'], d['persons'], d['seq_ids']

    # Check if existing normalized cache from results_anatophasenet_v2 exists
    alt_cache = os.path.join('results_anatophasenet_v2', f"phase_cache_norm_{img_size}.pt")
    if os.path.isfile(alt_cache):
        d = torch.load(alt_cache, weights_only=False)
        print(f"[INFO] Reusing existing normalized cache from {alt_cache}: ({len(d['labels'])} cycles)")
        os.makedirs(os.path.dirname(cache_file) or '.', exist_ok=True)
        torch.save(d, cache_file)
        return d['data'], d['labels'], d['persons'], d['seq_ids']

    cycles = extract_phase_cycles(root_dir)
    print(f"[INFO] Normalizing silhouettes and caching {len(cycles)} cycles to {cache_file}...")
    data = torch.zeros(len(cycles), 9, img_size, img_size, dtype=torch.uint8)
    labels, persons, seq_ids = [], [], []

    for i, (paths, label, person, seq_id) in enumerate(cycles):
        for phase_idx, p in enumerate(paths):
            try:
                im = Image.open(p).convert('L')
                norm_im = crop_and_resize_silhouette(im, target_size=img_size)
                data[i, phase_idx] = torch.from_numpy(np.array(norm_im, dtype=np.uint8))
            except Exception:
                pass
        labels.append(label)
        persons.append(person)
        seq_ids.append(seq_id)

    labels = torch.tensor(labels, dtype=torch.long)
    os.makedirs(os.path.dirname(cache_file) or '.', exist_ok=True)
    torch.save({'data': data, 'labels': labels, 'persons': persons, 'seq_ids': seq_ids}, cache_file)
    return data, labels, persons, seq_ids

# =======================================================================
# 3. GPU AUGMENTATION
# =======================================================================
@torch.no_grad()
def gpu_augment_phases(x):
    B, T, H, W = x.shape
    dev = x.device

    ang = (torch.rand(B, device=dev) * 2 - 1) * math.radians(4)
    sc = 1.0 + (torch.rand(B, device=dev) * 2 - 1) * 0.05
    tx = (torch.rand(B, device=dev) * 2 - 1) * 0.04
    ty = (torch.rand(B, device=dev) * 2 - 1) * 0.04

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

    for b in range(B):
        if random.random() < 0.20:
            eh = random.randint(H // 14, H // 6)
            ew = random.randint(W // 14, W // 6)
            y0 = random.randint(0, H - eh)
            x0 = random.randint(0, W - ew)
            x[b, :, y0:y0 + eh, x0:x0 + ew] = 0.0

    return x

# =======================================================================
# 4. FOCAL LOSS WITH LABEL SMOOTHING
# =======================================================================
class FocalCrossEntropyLoss(nn.Module):
    """
    Focal Cross-Entropy with Label Smoothing:
    Loss = - (1 - p_t)^gamma * log(p_t)
    Adaptively focuses learning onto hard/confused samples (mild Diplegic & Hemiplegic)
    without distorting decision thresholds with static weights.
    """
    def __init__(self, gamma=1.5, label_smoothing=0.05):
        super().__init__()
        self.gamma = gamma
        self.label_smoothing = label_smoothing

    def forward(self, logits, targets):
        num_classes = logits.size(-1)
        # Compute smooth one-hot targets
        with torch.no_grad():
            smooth_targets = torch.full_like(logits, self.label_smoothing / (num_classes - 1))
            smooth_targets.scatter_(-1, targets.unsqueeze(-1), 1.0 - self.label_smoothing)

        log_probs = F.log_softmax(logits, dim=-1)
        probs = torch.exp(log_probs)

        # Focal modulating factor: (1 - p)^gamma
        focal_weight = torch.pow(1.0 - probs, self.gamma)
        loss = -torch.sum(focal_weight * smooth_targets * log_probs, dim=-1)
        return loss.mean()

# =======================================================================
# 5. ANATOPHASENET V3 ARCHITECTURE
# =======================================================================
class FactorizedSpatiotemporalBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride=1):
        super().__init__()
        mid_planes = out_planes

        self.spatial = nn.Conv3d(
            in_planes, mid_planes,
            kernel_size=(1, 3, 3),
            stride=(1, stride, stride),
            padding=(0, 1, 1),
            bias=False
        )
        self.bn_s = nn.BatchNorm3d(mid_planes)

        self.temporal = nn.Conv3d(
            mid_planes, out_planes,
            kernel_size=(3, 1, 1),
            stride=(1, 1, 1),
            padding=(1, 0, 0),
            bias=False
        )
        self.bn_t = nn.BatchNorm3d(out_planes)

        self.act = nn.ReLU(inplace=True)

        if stride != 1 or in_planes != out_planes:
            self.downsample = nn.Sequential(
                nn.Conv3d(in_planes, out_planes, kernel_size=1, stride=(1, stride, stride), bias=False),
                nn.BatchNorm3d(out_planes)
            )
        else:
            self.downsample = None

    def forward(self, x):
        res = x
        out = self.act(self.bn_s(self.spatial(x)))
        out = self.bn_t(self.temporal(out))
        if self.downsample is not None:
            res = self.downsample(res)
        return self.act(out + res)


class Anatomical_PTRAP_Head_v3(nn.Module):
    """
    P-TRAP Head v3 with:
    1. Upper Region Dynamics (Trunk mobility vs. frozen stoop)
    2. Lower Region Leg-Specific Asymmetry
    3. Upper-Lower Posture Contrast
    Total Input Dimension:
    - feat_upper (512)
    - feat_lower (512)
    - leg_asym (512)
    - upper_lower_contrast (512)
    - trunk_dynamics (512, broadcast across time)
    Total Dim = 512 * 5 = 2560
    """
    def __init__(self, in_dim=2560, num_classes=5, num_phases=9):
        super().__init__()
        self.num_phases = num_phases
        self.phase_embed = nn.Parameter(torch.randn(1, num_phases, in_dim) * 0.02)

        self.attn_net = nn.Sequential(
            nn.Linear(in_dim, in_dim // 4),
            nn.Tanh(),
            nn.Linear(in_dim // 4, 1)
        )

        self.fc = nn.Sequential(
            nn.Linear(in_dim * 2, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.35),
            nn.Linear(256, num_classes)
        )

    def forward(self, feat_seq):
        # feat_seq: [B, 9, in_dim]
        B = feat_seq.shape[0]

        x = feat_seq + self.phase_embed

        raw_scores = self.attn_net(x).squeeze(-1)  # [B, 9]
        attn_primary = F.softmax(raw_scores, dim=-1).unsqueeze(-1)  # [B, 9, 1]
        pool_primary = torch.sum(x * attn_primary, dim=1)  # [B, in_dim]

        rev_scores = 1.0 - attn_primary
        rev_norm = rev_scores / (torch.sum(rev_scores, dim=1, keepdim=True) + 1e-8)
        pool_reverse = torch.sum(x * rev_norm, dim=1)  # [B, in_dim]

        merged = torch.cat([pool_primary, pool_reverse], dim=-1)  # [B, in_dim * 2]
        return self.fc(merged)


class AnatoPhaseNet_v3(nn.Module):
    def __init__(self, num_classes=5, pretrained_path='resnet18_pretrained.pth'):
        super().__init__()

        # Stem
        self.stem = nn.Sequential(
            nn.Conv3d(2, 64, kernel_size=(1, 7, 7), stride=(1, 2, 2), padding=(0, 3, 3), bias=False),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))
        )

        # 4 Factorized Stages
        self.b1 = FactorizedSpatiotemporalBlock(64, 64, stride=1)
        self.b2 = FactorizedSpatiotemporalBlock(64, 128, stride=2)
        self.b3 = FactorizedSpatiotemporalBlock(128, 256, stride=2)
        self.b4 = FactorizedSpatiotemporalBlock(256, 512, stride=2)

        # Head with in_dim = 2560
        self.head = Anatomical_PTRAP_Head_v3(in_dim=2560, num_classes=num_classes)

        if pretrained_path and os.path.isfile(pretrained_path):
            self._inflate_pretrained_weights(pretrained_path)

    def _inflate_pretrained_weights(self, path):
        try:
            print(f"[INFO] Inflating spatial convolutions with ImageNet ResNet-18 weights from {path}...")
            ckpt = torch.load(path, weights_only=True)

            w_stem = ckpt['conv1.weight']
            w_stem_2ch = w_stem[:, :2, :, :].unsqueeze(2)
            self.stem[0].weight.data.copy_(w_stem_2ch)

            def inflate_conv(layer, w2d):
                layer.weight.data.copy_(w2d.unsqueeze(2))

            inflate_conv(self.b1.spatial, ckpt['layer1.0.conv1.weight'])
            inflate_conv(self.b2.spatial, ckpt['layer2.0.conv1.weight'])
            inflate_conv(self.b3.spatial, ckpt['layer3.0.conv1.weight'])
            inflate_conv(self.b4.spatial, ckpt['layer4.0.conv1.weight'])

            print("[SUCCESS] Spatial layers successfully initialized with generalizable ImageNet visual filters!")
        except Exception as e:
            print(f"[WARN] Error during weight inflation: {e}")

    def forward(self, x):
        # x: [B, 9, H, W]
        sil = (x - 0.5) / 0.5
        diff = torch.cat([torch.zeros_like(sil[:, :1]), sil[:, 1:] - sil[:, :-1]], dim=1)
        inp = torch.stack([sil, diff], dim=1)  # [B, 2, 9, H, W]

        h = self.b4(self.b3(self.b2(self.b1(self.stem(inp)))))  # [B, 512, 9, Hf, Wf]
        B, C, T, Hf, Wf = h.shape

        # ===================================================================
        # TARGETED BIOMECHANICAL DECOUPLING
        # ===================================================================
        split_h = max(1, int(Hf * 0.40))

        # 1. Upper Region: Cephalo-Thoracic (Trunk & Arms)
        h_upper = h[:, :, :, :split_h, :]
        feat_upper = h_upper.mean(dim=(3, 4))  # [B, 512, 9]

        # 2. Lower Region: Pelvo-Pedal (Hips, Knees, Feet)
        h_lower = h[:, :, :, split_h:, :]
        feat_lower = h_lower.mean(dim=(3, 4))  # [B, 512, 9]

        # 3. Leg-Specific Lateral Asymmetry (Decouples Hemiplegic vs. Diplegic!)
        # Unilateral circumduction spikes asymmetry in legs, whereas bilateral scissoring is symmetric
        leg_asym = (h_lower - h_lower.flip(-1)).abs().mean(dim=(3, 4))  # [B, 512, 9]

        # 4. Upper-Lower Contrast Feature (Decouples Stooped Posture vs. Leg Spasticity)
        upper_lower_contrast = feat_upper - feat_lower  # [B, 512, 9]

        # 5. Trunk Mobility & Dynamics (Decouples Parkinsonian Frozen Trunk vs. Diplegic Torso Sway)
        # Compute inter-phase velocity of the trunk across time: |U(t) - U(t-1)|
        trunk_velocity = torch.cat([torch.zeros_like(feat_upper[:, :, :1]), feat_upper[:, :, 1:] - feat_upper[:, :, :-1]], dim=2)  # [B, 512, 9]

        # Concatenate all 5 biomechanically targeted feature streams: [B, 2560, 9]
        feat_combined = torch.cat([
            feat_upper,
            feat_lower,
            leg_asym,
            upper_lower_contrast,
            trunk_velocity
        ], dim=1)

        feat_seq = feat_combined.permute(0, 2, 1)  # [B, 9, 2560]
        return self.head(feat_seq)

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
def evaluate_model(model, data, labels, seq_ids, persons, idx, device, use_amp=True):
    model.eval()
    crit = nn.CrossEntropyLoss()
    tot_loss = 0.0
    all_logits = []
    targets = []

    bs = 32
    n_batches = int(math.ceil(len(idx) / bs))

    for b in range(n_batches):
        b_idx = idx[b * bs:(b + 1) * bs]
        x = data[b_idx].to(device).float() / 255.0
        y = labels[b_idx].to(device)

        with torch.autocast('cuda', dtype=torch.float16, enabled=use_amp):
            out = model(x)
            loss = crit(out, y)

        tot_loss += loss.item() * len(b_idx)
        all_logits.append(out.cpu().float())
        targets.extend(y.cpu().tolist())

    all_logits = torch.cat(all_logits, dim=0).numpy()
    targets = np.array(targets)
    preds = np.argmax(all_logits, axis=-1)

    cycle_loss = tot_loss / max(1, len(idx))
    cycle_acc = float(np.mean(preds == targets) * 100.0)
    cycle_f1 = compute_macro_f1(targets, preds, num_classes=5)

    # Multi-Cycle Session Consensus Voting
    session_preds = []
    session_targets = []
    unique_seqs = sorted(list(set(seq_ids[i] for i in idx)))

    idx_map = {orig_i: pos for pos, orig_i in enumerate(idx)}

    for s_id in unique_seqs:
        s_indices = [idx_map[i] for i in idx if seq_ids[i] == s_id]
        if not s_indices:
            continue
        s_logits = all_logits[s_indices].mean(axis=0)
        session_preds.append(np.argmax(s_logits))
        session_targets.append(targets[s_indices[0]])

    session_preds = np.array(session_preds)
    session_targets = np.array(session_targets)
    session_acc = float(np.mean(session_preds == session_targets) * 100.0)
    session_f1 = compute_macro_f1(session_targets, session_preds, num_classes=5)

    return cycle_loss, cycle_acc, cycle_f1, session_acc, session_f1, preds, targets

# =======================================================================
# 7. SUBJECT-EXCLUSIVE 10-FOLD CV SPLITTER
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

# =======================================================================
# 8. MAIN EXPERIMENT RUNNER
# =======================================================================
def main():
    here = os.path.dirname(os.path.abspath(__file__))
    default_dataset = os.path.join(here, 'Gait_Dataset_silhoutte_based', 'GAIT_IT')

    parser = argparse.ArgumentParser(description="AnatoPhaseNet v3 Pathological Gait Classifier")
    parser.add_argument('--dataset_dir', type=str, default=default_dataset, help='Path to GAIT-IT dataset directory')
    parser.add_argument('--pretrained_path', type=str, default='resnet18_pretrained.pth', help='Path to local ResNet-18 weights')
    parser.add_argument('--out_dir', type=str, default='results_anatophasenet_v3', help='Output folder')
    parser.add_argument('--img_size', type=int, default=96, help='Frame resolution (96 for high-res limb detail)')
    parser.add_argument('--epochs', type=int, default=40, help='Number of epochs per fold')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--lr', type=float, default=3e-4, help='Peak learning rate')
    parser.add_argument('--weight_decay', type=float, default=5e-2, help='Weight decay')
    parser.add_argument('--focal_gamma', type=float, default=1.5, help='Focal loss gamma factor')
    parser.add_argument('--label_smoothing', type=float, default=0.05, help='Label smoothing factor')
    parser.add_argument('--ema', type=float, default=0.98, help='EMA decay factor')
    parser.add_argument('--folds', type=int, default=10, help='Total cross-validation folds')
    parser.add_argument('--run_folds', type=int, default=10, help='Number of folds to run')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    use_amp = (device.type == 'cuda')
    print("=" * 75)
    print("  AnatoPhaseNet v3: Decoupled Anatomical Pathological Gait Classifier")
    print(f"  Device: {device} | AMP: {use_amp} | Resolution: {args.img_size}x{args.img_size}")
    print(f"  Pretrained Backbone: {args.pretrained_path} | Loss: Focal (gamma={args.focal_gamma})")
    print("=" * 75)

    cache_file = os.path.join(args.out_dir, f"phase_cache_norm_{args.img_size}.pt")
    data, labels, persons, seq_ids = load_phase_cache(args.dataset_dir, args.img_size, cache_file)

    model_tmp = AnatoPhaseNet_v3(num_classes=5, pretrained_path=None)
    param_count = sum(p.numel() for p in model_tmp.parameters() if p.requires_grad)
    print(f"[INFO] Architecture: AnatoPhaseNet v3 | Parameters: {param_count / 1e6:.3f}M")
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

        model = AnatoPhaseNet_v3(num_classes=5, pretrained_path=args.pretrained_path).to(device)
        eval_model = AnatoPhaseNet_v3(num_classes=5, pretrained_path=None).to(device)
        ema = EMA(model, decay=args.ema)

        steps = max(1, len(tr_idx) // args.batch_size)
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        sched = torch.optim.lr_scheduler.OneCycleLR(
            opt, max_lr=args.lr, total_steps=args.epochs * steps, pct_start=0.15
        )
        scaler = torch.amp.GradScaler('cuda', enabled=use_amp)
        crit = FocalCrossEntropyLoss(gamma=args.focal_gamma, label_smoothing=args.label_smoothing)

        best_val_acc = -1.0
        best_val_loss = 1e9
        best_state = None
        best_ep = -1

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

        # Held-out Test Evaluation
        eval_model.load_state_dict(best_state)
        t_loss, t_acc, t_f1, t_sess_acc, t_sess_f1, t_preds, t_targets = evaluate_model(
            eval_model, data, labels, seq_ids, persons, te_idx, device, use_amp
        )
        conf = confusion_matrix(t_targets, t_preds)
        agg_conf += conf

        try:
            plot_fold_curves(history, os.path.join(args.out_dir, f'fold_{f_idx+1}_training_curves.png'), f_idx + 1)
            plot_confusion_matrix_chart(conf, os.path.join(args.out_dir, f'fold_{f_idx+1}_confusion_matrix.png'), f"Fold {f_idx+1} Test Confusion Matrix")
        except Exception as e:
            print(f"  [WARN] Plot error: {e}")

        fold_results.append({
            'fold': f_idx + 1,
            'test_cycle_acc': t_acc,
            'test_cycle_f1': t_f1,
            'test_sess_acc': t_sess_acc,
            'test_sess_f1': t_sess_f1
        })

        print("\n" + "=" * 75)
        print(f">> [Fold {f_idx+1}/{args.folds} FINAL TEST RESULT on Unseen Subjects]")
        print(f"   Test Cycle Acc:   {t_acc:5.2f}% | Macro-F1: {t_f1:.4f}")
        print(f"   Test Session Acc: {t_sess_acc:5.2f}% | Macro-F1: {t_sess_f1:.4f}")
        print("   Per-Class Test Metrics:")
        for c_idx, c_name in enumerate(CLASS_NAMES):
            c_tp = conf[c_idx, c_idx]
            c_tot_pred = conf[:, c_idx].sum()
            c_tot_true = conf[c_idx, :].sum()
            c_prec = (c_tp / c_tot_pred) if c_tot_pred > 0 else 0.0
            c_rec = (c_tp / c_tot_true) if c_tot_true > 0 else 0.0
            c_f1 = (2 * c_prec * c_rec / (c_prec + c_rec)) if (c_prec + c_rec) > 0 else 0.0
            print(f"     {c_name:<12s}: Precision: {c_prec:.3f} | Recall: {c_rec:.3f} | F1: {c_f1:.3f} (support={c_tot_true})")
        print("=" * 75)

    # Aggregated Summary across all folds
    print("\n" + "=" * 75)
    print("                 ANATOPHASENET V3 FINAL BENCHMARK SUMMARY")
    print("=" * 75)
    mean_cyc_acc = np.mean([r['test_cycle_acc'] for r in fold_results])
    std_cyc_acc = np.std([r['test_cycle_acc'] for r in fold_results])
    mean_cyc_f1 = np.mean([r['test_cycle_f1'] for r in fold_results])
    std_cyc_f1 = np.std([r['test_cycle_f1'] for r in fold_results])

    mean_sess_acc = np.mean([r['test_sess_acc'] for r in fold_results])
    std_sess_acc = np.std([r['test_sess_acc'] for r in fold_results])
    mean_sess_f1 = np.mean([r['test_sess_f1'] for r in fold_results])
    std_sess_f1 = np.std([r['test_sess_f1'] for r in fold_results])

    print(f"Evaluated Folds: {len(fold_results)} / {args.folds}")
    print(f"TEST Cycle-Level Accuracy:   {mean_cyc_acc:5.2f}% +/- {std_cyc_acc:4.2f}%")
    print(f"TEST Cycle-Level Macro-F1:   {mean_cyc_f1:.4f} +/- {std_cyc_f1:.4f}")
    print(f"TEST Session-Level Accuracy: {mean_sess_acc:5.2f}% +/- {std_sess_acc:4.2f}%")
    print(f"TEST Session-Level Macro-F1: {mean_sess_f1:.4f} +/- {std_sess_f1:.4f}")

    print("\nAggregated Confusion Matrix across evaluated test folds:")
    print(f"Classes: {CLASS_NAMES}")
    print(agg_conf)

    try:
        plot_confusion_matrix_chart(agg_conf, os.path.join(args.out_dir, 'aggregated_confusion_matrix.png'), "Aggregated Test Confusion Matrix")
    except Exception as e:
        print(f"  [WARN] Aggregated plot error: {e}")

    summary_file = os.path.join(args.out_dir, 'benchmark_summary.json')
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump({
            'cycle_accuracy_mean': mean_cyc_acc,
            'cycle_accuracy_std': std_cyc_acc,
            'session_accuracy_mean': mean_sess_acc,
            'session_accuracy_std': std_sess_acc,
            'fold_results': fold_results
        }, f, indent=4)
    print(f"[SUCCESS] All experiment artifacts and high-res charts saved to: {args.out_dir}")


if __name__ == '__main__':
    main()
