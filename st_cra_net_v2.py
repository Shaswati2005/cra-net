#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ST-CRANet v2: a light (2+1)D network for pathological gait classification.

Why v2 (see st_cra_net.py for v1):
  v1 had ~20M params, a per-frame U-Net decoder + 3 RCAF modules, and NO
  augmentation (the transform was never applied), so it memorised the
  ~575 training sequences (100% train / ~77% val).

v2 changes:
  * ~1M-parameter factorised (2+1)D residual network (spatial 1x3x3 + temporal 3x1x1)
  * Input = silhouette + frame-difference (motion) channel
  * GroupNorm (stable for small batches)
  * Temporal Reverse Attention Pooling head kept (dominant + reverse softmax over time)
  * Real GPU augmentation (affine, flip, erasing, temporal speed/offset jitter) + mixup
  * EMA weights, OneCycle LR, AMP, label smoothing
  * Subject-wise split option (no person appears in both train and val/test)
"""

import os
import re
import math
import argparse
import random
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F


def confusion_matrix(t, p, n=5):
    m = np.zeros((n, n), dtype=int)
    for a, b in zip(t, p):
        m[a, b] += 1
    return m


def f1_score(t, p, average='macro', n=5):
    m = confusion_matrix(t, p, n)
    f1s = []
    for c in range(n):
        tp = m[c, c]; fp = m[:, c].sum() - tp; fn = m[c, :].sum() - tp
        f1s.append(0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn))
    return float(np.mean(f1s))


def classification_report(t, p, target_names, zero_division=0):
    m = confusion_matrix(t, p, len(target_names))
    lines = [f"{'':14s}{'prec':>8s}{'recall':>8s}{'f1':>8s}{'n':>6s}"]
    for c, name in enumerate(target_names):
        tp = m[c, c]; fp = m[:, c].sum() - tp; fn = m[c, :].sum() - tp
        pr = tp / (tp + fp) if tp + fp else 0.0
        rc = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * pr * rc / (pr + rc) if pr + rc else 0.0
        lines.append(f"{name:14s}{pr:8.3f}{rc:8.3f}{f1:8.3f}{m[c].sum():6d}")
    return "\n".join(lines)


def _stratified_split(idx, y, test_frac, rng):
    a, b = [], []
    for c in np.unique(y[idx]):
        ci = idx[y[idx] == c].copy()
        rng.shuffle(ci)
        k = max(1, int(round(test_frac * len(ci))))
        b.extend(ci[:k]); a.extend(ci[k:])
    return np.array(a), np.array(b)

GAIT_CLASSES = {
    'normal_silhouettes': 0,
    'diplegic_silhouettes': 1,
    'hemiplegic_silhouettes': 2,
    'neuropathic_silhouettes': 3,
    'parkinson_silhouettes': 4,
}
CLASS_NAMES = ['normal', 'diplegic', 'hemiplegic', 'neuropathic', 'parkinson']
IMG_EXT = ('.png', '.jpg', '.jpeg')


# =======================================================================
# 1. DATA
# =======================================================================
def _natural_key(x):
    return [int(c) if c.isdigit() else c for c in re.split(r'(\d+)', x)]


def scan_dataset(root_dir, min_frames=8):
    """Returns list of (frame_paths, label, subject_group)."""
    samples = []
    for gait_type, label in GAIT_CLASSES.items():
        gait_path = os.path.join(root_dir, gait_type)
        if not os.path.isdir(gait_path):
            continue
        for subject in sorted(os.listdir(gait_path)):
            subj_path = os.path.join(gait_path, subject, 'silhouettes', 'side_view')
            if not os.path.isdir(subj_path):
                continue
            group = f"{label}_{subject}"
            for session in sorted(os.listdir(subj_path)):
                session_path = os.path.join(subj_path, session)
                if not os.path.isdir(session_path):
                    continue
                items = os.listdir(session_path)
                seq_dirs = []
                if any(f.lower().endswith(IMG_EXT) for f in items):
                    seq_dirs.append(session_path)
                else:
                    for s in items:
                        p = os.path.join(session_path, s)
                        if s.startswith('Seq_') and os.path.isdir(p):
                            seq_dirs.append(p)
                for sd in seq_dirs:
                    files = [f for f in os.listdir(sd) if f.lower().endswith(IMG_EXT)]
                    if len(files) < min_frames:
                        continue
                    files.sort(key=_natural_key)
                    samples.append(([os.path.join(sd, f) for f in files], label, group))
    return samples


def load_cache(root_dir, cache_len, img_size, cache_file):
    """uint8 tensor [N, cache_len, H, W] + labels + groups. Cached to disk."""
    if os.path.isfile(cache_file):
        d = torch.load(cache_file, weights_only=False)
        print(f"[INFO] Loaded cache {cache_file}: {tuple(d['data'].shape)}")
        return d['data'], d['labels'], d['groups']

    samples = scan_dataset(root_dir)
    print(f"[INFO] Found {len(samples)} sequences. Caching at {img_size}x{img_size}, {cache_len} frames each...")
    data = torch.zeros(len(samples), cache_len, img_size, img_size, dtype=torch.uint8)
    labels, groups = [], []
    for i, (paths, label, group) in enumerate(samples):
        idx = np.linspace(0, len(paths) - 1, cache_len).round().astype(int)
        for j, k in enumerate(idx):
            try:
                img = Image.open(paths[k]).convert('L').resize((img_size, img_size), Image.BILINEAR)
                data[i, j] = torch.from_numpy(np.array(img, dtype=np.uint8))
            except Exception:
                pass
        labels.append(label)
        groups.append(group)
        if (i + 1) % 100 == 0 or i + 1 == len(samples):
            print(f"  cached {i+1}/{len(samples)}")
    labels = torch.tensor(labels)
    os.makedirs(os.path.dirname(cache_file) or '.', exist_ok=True)
    torch.save({'data': data, 'labels': labels, 'groups': groups}, cache_file)
    return data, labels, groups


def make_splits(labels, groups, mode, seed):
    labels_np = np.asarray(labels)
    n = len(labels_np)
    if mode == 'random':
        idx = np.arange(n)
        rng = np.random.RandomState(seed)
        tr, tmp = _stratified_split(idx, labels_np, 0.30, rng)
        va, te = _stratified_split(tmp, labels_np, 0.50, rng)
        return tr, va, te

    # subject-wise: split people (not sequences) within every class
    rng = random.Random(seed)
    by_class = {}
    for g in sorted(set(groups)):
        by_class.setdefault(int(g.split('_')[0]), []).append(g)
    split_of = {}
    for c, gs in by_class.items():
        rng.shuffle(gs)
        n_g = len(gs)
        n_val = max(1, round(0.15 * n_g))
        n_test = max(1, round(0.15 * n_g))
        for k, g in enumerate(gs):
            split_of[g] = 'val' if k < n_val else ('test' if k < n_val + n_test else 'train')
    tr = np.array([i for i, g in enumerate(groups) if split_of[g] == 'train'])
    va = np.array([i for i, g in enumerate(groups) if split_of[g] == 'val'])
    te = np.array([i for i, g in enumerate(groups) if split_of[g] == 'test'])
    return tr, va, te


def sample_indices(batch_idx, cache_len, seq_len, train, rng):
    """Frame indices per sequence: uniform at eval, random window/speed jitter at train."""
    out = np.zeros((len(batch_idx), seq_len), dtype=np.int64)
    for b in range(len(batch_idx)):
        if train:
            span = rng.uniform(0.65, 1.0) * (cache_len - 1)
            start = rng.uniform(0, (cache_len - 1) - span)
            pts = start + np.linspace(0, span, seq_len)
            pts = pts + rng.uniform(-0.5, 0.5, size=seq_len) * 0.5  # small jitter
            out[b] = np.clip(np.round(pts), 0, cache_len - 1).astype(int)
            out[b].sort()
        else:
            out[b] = np.linspace(0, cache_len - 1, seq_len).round().astype(int)
    return out


@torch.no_grad()
def gpu_augment(x):
    """x: [B,T,H,W] in [0,1]. Same random transform for all frames of a sequence."""
    B, T, H, W = x.shape
    dev = x.device
    ang = (torch.rand(B, device=dev) * 2 - 1) * math.radians(8)
    sc = 1.0 + (torch.rand(B, device=dev) * 2 - 1) * 0.10
    tx = (torch.rand(B, device=dev) * 2 - 1) * 0.08
    ty = (torch.rand(B, device=dev) * 2 - 1) * 0.08
    flip = (torch.rand(B, device=dev) < 0.5).float() * 2 - 1  # +1 or -1
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

    # random erasing (small block, all frames), p=0.25
    for b in range(B):
        if random.random() < 0.25:
            eh, ew = random.randint(H // 10, H // 4), random.randint(W // 10, W // 4)
            y0, x0 = random.randint(0, H - eh), random.randint(0, W - ew)
            x[b, :, y0:y0 + eh, x0:x0 + ew] = 0.0
    return x


# =======================================================================
# 2. MODEL
# =======================================================================
class R2Plus1DBlock(nn.Module):
    """Spatial 1x3x3 conv -> temporal 3x1x1 conv, residual, GroupNorm."""
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


class TemporalReverseAttentionPooling(nn.Module):
    """Dominant (softmax(s)) + reverse (softmax(-s)) attention over time -> classifier."""
    def __init__(self, c, num_classes, drop=0.5):
        super().__init__()
        self.scorer = nn.Sequential(nn.Linear(c, 32), nn.GELU(), nn.Linear(32, 1))
        self.drop = nn.Dropout(drop)
        self.fc = nn.Linear(2 * c, num_classes)

    def forward(self, r):  # r: [B,T,C]
        s = self.scorer(r)
        f_dom = (r * F.softmax(s, 1)).sum(1)
        f_rev = (r * F.softmax(-s, 1)).sum(1)
        return self.fc(self.drop(torch.cat([f_dom, f_rev], -1)))


class ST_CRANetV2(nn.Module):
    def __init__(self, num_classes=5, width=(24, 48, 96, 128), drop=0.1):
        super().__init__()
        c0, c1, c2, c3 = width
        self.stem = nn.Sequential(
            nn.Conv3d(2, c0, (1, 3, 3), padding=(0, 1, 1), bias=False),
            nn.GroupNorm(8, c0), nn.ReLU(inplace=True))
        self.b1 = R2Plus1DBlock(c0, c1, pool=True)
        self.b2 = R2Plus1DBlock(c1, c2, pool=True, drop=drop)
        self.b3 = R2Plus1DBlock(c2, c3, pool=True, drop=drop)
        self.b4 = R2Plus1DBlock(c3, c3, pool=True, drop=drop)
        self.head = TemporalReverseAttentionPooling(c3, num_classes)

    def forward(self, x):
        """x: [B,T,H,W] in [0,1] silhouette frames."""
        sil = (x - 0.5) / 0.5
        diff = torch.cat([torch.zeros_like(sil[:, :1]), sil[:, 1:] - sil[:, :-1]], 1)
        inp = torch.stack([sil, diff], 1)  # [B,2,T,H,W]
        h = self.b4(self.b3(self.b2(self.b1(self.stem(inp)))))
        h = h.mean(dim=(3, 4)).permute(0, 2, 1)  # [B,T,C]
        return self.head(h)


# =======================================================================
# 3. TRAINING UTILITIES
# =======================================================================
class EMA:
    def __init__(self, model, decay):
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
def evaluate(model, data, labels, idx, args, device, use_amp, tta=False):
    model.eval()
    crit = nn.CrossEntropyLoss()
    rng = np.random.RandomState(0)
    tot_loss, preds, targets = 0.0, [], []
    for s in range(0, len(idx), 32):
        bi = idx[s:s + 32]
        fi = torch.from_numpy(sample_indices(bi, data.shape[1], args.seq_len, False, rng)).to(device)
        x = data[torch.as_tensor(bi, device=device)[:, None], fi].float() / 255.0
        y = labels[torch.as_tensor(bi)].to(device)
        with torch.autocast('cuda', dtype=torch.float16, enabled=use_amp):
            out = model(x).float()
            if tta:
                out = (F.softmax(out, 1) + F.softmax(model(x.flip(-1)).float(), 1)).log()
        tot_loss += crit(out, y).item() * len(bi)
        preds.extend(out.argmax(1).cpu().tolist())
        targets.extend(y.cpu().tolist())
    acc = 100.0 * np.mean(np.array(preds) == np.array(targets))
    f1 = f1_score(targets, preds, average='macro')
    return tot_loss / len(idx), acc, f1, preds, targets


def plot_curves(h, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    ep = range(1, len(h['train_loss']) + 1)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 5))
    a1.plot(ep, h['train_loss'], 'b-', label='Train'); a1.plot(ep, h['val_loss'], 'r-', label='Val')
    a1.set_title('Loss'); a1.legend(); a1.grid(alpha=.4)
    a2.plot(ep, h['train_acc'], 'b-', label='Train'); a2.plot(ep, h['val_acc'], 'g-', label='Val (EMA)')
    a2.set_title('Accuracy (%)'); a2.legend(); a2.grid(alpha=.4)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def parse_args():
    here = os.path.dirname(os.path.abspath(__file__))
    default_ds = os.path.join(here, 'Gait_Dataset_silhoutte_based', 'GAIT_IT')
    p = argparse.ArgumentParser("Train ST-CRANet v2")
    p.add_argument('--dataset_dir', default=default_ds)
    p.add_argument('--split', choices=['subject', 'random'], default='subject')
    p.add_argument('--seq_len', type=int, default=16)
    p.add_argument('--cache_len', type=int, default=32)
    p.add_argument('--img_size', type=int, default=64)
    p.add_argument('--batch_size', type=int, default=16)
    p.add_argument('--lr', type=float, default=2e-3)
    p.add_argument('--weight_decay', type=float, default=5e-2)
    p.add_argument('--epochs', type=int, default=80)
    p.add_argument('--mixup', type=float, default=0.2, help='mixup alpha (0 disables)')
    p.add_argument('--no_aug', action='store_true', help='ablation: disable augmentation')
    p.add_argument('--ema', type=float, default=0.98)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--out_dir', default=os.path.join(here, 'results_v2'))
    p.add_argument('--no_amp', action='store_true')
    p.add_argument('--base_width', type=int, default=24)
    return p.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    # Mirror everything printed (stdout + stderr) into out_dir/train_log.txt automatically
    import sys
    class _Tee:
        def __init__(self, *streams):
            self.streams = streams
        def write(self, s):
            for st in self.streams:
                st.write(s); st.flush()
        def flush(self):
            for st in self.streams:
                st.flush()
    _log = open(os.path.join(args.out_dir, 'train_log.txt'), 'w', encoding='utf-8')
    sys.stdout = _Tee(sys.__stdout__, _log)
    sys.stderr = _Tee(sys.__stderr__, _log)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    use_amp = device.type == 'cuda' and not args.no_amp
    print(f"[INFO] device={device} amp={use_amp} split={args.split}")

    cache_file = os.path.join(args.out_dir, f'cache_{args.img_size}_{args.cache_len}.pt')
    data, labels, groups = load_cache(args.dataset_dir, args.cache_len, args.img_size, cache_file)
    if len(labels) == 0:
        print("[ERROR] no sequences found at", args.dataset_dir); return
    tr, va, te = make_splits(labels.numpy(), groups, args.split, args.seed)
    print(f"[INFO] train={len(tr)} val={len(va)} test={len(te)} | "
          f"subjects train={len({groups[i] for i in tr})} val={len({groups[i] for i in va})} test={len({groups[i] for i in te})}")
    data = data.to(device)

    W = args.base_width; widths = (W, 2 * W, 4 * W, 5 * W if W % 8 == 0 else 4 * W)
    model = ST_CRANetV2(num_classes=5, width=widths).to(device)
    print(f"[INFO] Parameters: {sum(p.numel() for p in model.parameters())/1e6:.3f} M")
    ema = EMA(model, args.ema)
    eval_model = ST_CRANetV2(num_classes=5, width=widths).to(device)

    steps_per_epoch = len(tr) // args.batch_size
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=args.lr, total_steps=args.epochs * steps_per_epoch, pct_start=0.15)
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)
    crit = nn.CrossEntropyLoss(label_smoothing=0.1)

    hist = {k: [] for k in ['train_loss', 'train_acc', 'val_loss', 'val_acc', 'val_f1']}
    csv_path = os.path.join(args.out_dir, 'training_metrics.csv')
    with open(csv_path, 'w') as f:
        f.write("Epoch,Train_Loss,Train_Acc,Val_Loss,Val_Acc,Val_Macro_F1\n")
    ckpt = os.path.join(args.out_dir, 'st_cranet_v2_best.pth')
    best_acc, best_loss = -1.0, 1e9
    rng = np.random.RandomState(args.seed)

    for epoch in range(args.epochs):
        model.train()
        perm = np.random.permutation(tr)
        t_loss, t_correct, t_total = 0.0, 0, 0
        for s in range(steps_per_epoch):
            bi = perm[s * args.batch_size:(s + 1) * args.batch_size]
            fi = torch.from_numpy(sample_indices(bi, data.shape[1], args.seq_len, True, rng)).to(device)
            x = data[torch.as_tensor(bi, device=device)[:, None], fi].float() / 255.0
            y = labels[torch.as_tensor(bi)].to(device)
            if not args.no_aug:
                x = gpu_augment(x)

            y2, lam = y, 1.0
            if args.mixup > 0 and not args.no_aug and random.random() < 0.5:
                lam = float(np.random.beta(args.mixup, args.mixup))
                p2 = torch.randperm(x.size(0), device=device)
                x = lam * x + (1 - lam) * x[p2]
                y2 = y[p2]

            with torch.autocast('cuda', dtype=torch.float16, enabled=use_amp):
                out = model(x)
                loss = lam * crit(out, y) + (1 - lam) * crit(out, y2)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            scaler.step(opt); scaler.update(); sched.step()
            ema.update(model)

            t_loss += loss.item() * len(bi)
            if lam == 1.0:  # accuracy only on un-mixed batches
                t_correct += (out.argmax(1) == y).sum().item()
                t_total += len(bi)

        # evaluate EMA weights
        ema.copy_to(eval_model)
        v_loss, v_acc, v_f1, _, _ = evaluate(eval_model, data, labels, va, args, device, use_amp)
        tr_loss = t_loss / (steps_per_epoch * args.batch_size)
        # clean train accuracy (no aug, EMA model) for a fair gap comparison
        _, tr_acc, _, _, _ = evaluate(eval_model, data, labels, tr, args, device, use_amp)

        for k, v in zip(hist.keys(), [tr_loss, tr_acc, v_loss, v_acc, v_f1]):
            hist[k].append(v)
        with open(csv_path, 'a') as f:
            f.write(f"{epoch+1},{tr_loss:.4f},{tr_acc:.2f},{v_loss:.4f},{v_acc:.2f},{v_f1:.4f}\n")
        print(f"Epoch [{epoch+1}/{args.epochs}] loss {tr_loss:.4f} | train acc {tr_acc:.2f}% | "
              f"val loss {v_loss:.4f} acc {v_acc:.2f}% f1 {v_f1:.4f}")

        if v_acc > best_acc or (v_acc == best_acc and v_loss < best_loss):
            best_acc, best_loss = v_acc, v_loss
            torch.save({'epoch': epoch, 'model_state_dict': eval_model.state_dict(),
                        'val_acc': v_acc, 'args': vars(args)}, ckpt)
            print(f"  [SAVED] best val acc {v_acc:.2f}%")
        if (epoch + 1) % 5 == 0 or epoch + 1 == args.epochs:
            plot_curves(hist, os.path.join(args.out_dir, 'training_curves.png'))

    # Final test with best checkpoint (+ flip TTA)
    eval_model.load_state_dict(torch.load(ckpt, weights_only=False)['model_state_dict'])
    report_lines = []
    for name, tta in [('plain', False), ('flip-TTA', True)]:
        t_loss, t_acc, t_f1, preds, targets = evaluate(eval_model, data, labels, te, args, device, use_amp, tta=tta)
        line = f"[TEST/{name}] acc {t_acc:.2f}% | macro-F1 {t_f1:.4f}"
        print("\n" + line)
        report_lines.append(line)
        report_lines.append(classification_report(targets, preds, target_names=CLASS_NAMES, zero_division=0))
        report_lines.append("Confusion matrix:\n" + str(confusion_matrix(targets, preds)) + "\n")
    print("\n".join(report_lines))
    report_lines.append(f"[DONE] best val acc {best_acc:.2f}% | split={args.split} | base_width={args.base_width} | epochs={args.epochs}")
    print(report_lines[-1])
    with open(os.path.join(args.out_dir, 'test_results.txt'), 'w') as f:
        f.write("\n".join(report_lines) + "\n")
    print(f"[SAVED] test report -> {os.path.join(args.out_dir, 'test_results.txt')}")


if __name__ == '__main__':
    main()
