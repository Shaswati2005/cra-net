#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ST-CRANet v3: experiment harness for the paper.

Builds on st_cra_net_v2.py (imported, not modified) and adds:
  * Subject-grouped, class-stratified K-fold cross-validation (no person is ever in two splits)
  * Multiple seeds, results reported as mean +/- std
  * Pooling ablation:  --pool trap | mean | max | attn
        trap = dominant + reverse temporal attention (your TRAP)
        attn = ordinary single-softmax attention pooling (control for TRAP)
  * Optional left/right asymmetry branch:  --asym
  * Optional per-sequence bounding-box crop:  --crop
  * Checkpoint selection on a validation fold, test fold only used once per fold

Example:
  python -u st_cra_net_v3.py --pool trap --folds 5 --out_dir cv_trap
  python -u st_cra_net_v3.py --pool mean --folds 5 --out_dir cv_mean
  python -u st_cra_net_v3.py --pool trap --asym --crop --folds 5 --out_dir cv_trap_asym_crop
Quick check:  add  --folds 5 --run_folds 1 --epochs 2
"""

import os
import sys
import json
import random
import argparse
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F

import st_cra_net_v2 as v2

CLASS_NAMES = v2.CLASS_NAMES


# =======================================================================
# 1. DATA (optional bounding-box crop, view tag)
# =======================================================================
def _view_of(path):
    p = path.lower()
    return 'front' if 'front' in p else ('back' if 'back' in p else 'side')


def _load_sequence(paths, idx, img_size, crop):
    frames = [np.array(Image.open(paths[k]).convert('L'), dtype=np.uint8) for k in idx]
    # make foreground bright, background dark (heuristic: silhouette is the minority)
    if np.mean([f.mean() for f in frames]) > 127:
        frames = [255 - f for f in frames]
    if crop:
        stack = np.stack(frames)  # [T,H,W]
        fg = stack > 127
        rows, cols = np.where(fg.any(0).any(1))[0], np.where(fg.any(0).any(0))[0]
        if len(rows) > 0 and len(cols) > 0:
            r0, r1, c0, c1 = rows.min(), rows.max() + 1, cols.min(), cols.max() + 1
            mh, mw = int(0.05 * (r1 - r0)) + 1, int(0.05 * (c1 - c0)) + 1
            r0, r1 = max(0, r0 - mh), min(stack.shape[1], r1 + mh)
            c0, c1 = max(0, c0 - mw), min(stack.shape[2], c1 + mw)
            frames = [f[r0:r1, c0:c1] for f in frames]
        side = max(frames[0].shape)
        padded = []
        for f in frames:  # same canvas for all frames -> motion preserved
            canvas = np.zeros((side, side), dtype=np.uint8)
            y, x = (side - f.shape[0]) // 2, (side - f.shape[1]) // 2
            canvas[y:y + f.shape[0], x:x + f.shape[1]] = f
            padded.append(canvas)
        frames = padded
    out = [np.array(Image.fromarray(f).resize((img_size, img_size), Image.BILINEAR), dtype=np.uint8) for f in frames]
    return np.stack(out)


def load_cache(root_dir, cache_len, img_size, crop, cache_file):
    if os.path.isfile(cache_file):
        d = torch.load(cache_file, weights_only=False)
        print(f"[INFO] Loaded cache {cache_file}: {tuple(d['data'].shape)}")
        return d['data'], d['labels'], d['groups'], d['views']
    samples = v2.scan_dataset(root_dir)
    print(f"[INFO] {len(samples)} sequences. Caching {img_size}px, {cache_len} frames, crop={crop} ...")
    data = torch.zeros(len(samples), cache_len, img_size, img_size, dtype=torch.uint8)
    labels, groups, views = [], [], []
    for i, (paths, label, group) in enumerate(samples):
        idx = np.linspace(0, len(paths) - 1, cache_len).round().astype(int)
        try:
            data[i] = torch.from_numpy(_load_sequence(paths, idx, img_size, crop))
        except Exception as e:
            print(f"  [WARN] sequence {i} failed: {e}")
        labels.append(label); groups.append(group); views.append(_view_of(paths[0]))
        if (i + 1) % 100 == 0 or i + 1 == len(samples):
            print(f"  cached {i+1}/{len(samples)}")
    labels = torch.tensor(labels)
    os.makedirs(os.path.dirname(cache_file) or '.', exist_ok=True)
    torch.save({'data': data, 'labels': labels, 'groups': groups, 'views': views}, cache_file)
    return data, labels, groups, views


def make_folds(labels, groups, k, seed):
    """Assign each *person* (group) to a fold, balanced per class. Returns fold id per sequence."""
    rng = random.Random(seed)
    fold_of = {}
    by_class = {}
    for g in sorted(set(groups)):
        by_class.setdefault(int(g.split('_')[0]), []).append(g)
    for c, gs in by_class.items():
        rng.shuffle(gs)
        for i, g in enumerate(gs):
            fold_of[g] = i % k
    return np.array([fold_of[g] for g in groups])


# =======================================================================
# 2. MODEL
# =======================================================================
class PoolHead(nn.Module):
    """Temporal pooling head: trap | mean | max | attn."""
    def __init__(self, c_in, num_classes, mode, drop=0.5):
        super().__init__()
        self.mode = mode
        if mode == 'trap':
            self.head = v2.TemporalReverseAttentionPooling(c_in, num_classes, drop)
        else:
            self.scorer = nn.Sequential(nn.Linear(c_in, 32), nn.GELU(), nn.Linear(32, 1)) if mode == 'attn' else None
            self.drop = nn.Dropout(drop)
            self.fc = nn.Linear(c_in, num_classes)

    def forward(self, r):  # [B,T,C]
        if self.mode == 'trap':
            return self.head(r)
        if self.mode == 'mean':
            f = r.mean(1)
        elif self.mode == 'max':
            f = r.max(1).values
        else:  # attn
            f = (r * F.softmax(self.scorer(r), 1)).sum(1)
        return self.fc(self.drop(f))


class ST_CRANetV3(nn.Module):
    def __init__(self, num_classes=5, base=24, pool='trap', asym=False, drop=0.1):
        super().__init__()
        c0, c1, c2, c3 = base, 2 * base, 4 * base, 4 * base
        self.asym = asym
        self.stem = nn.Sequential(
            nn.Conv3d(2, c0, (1, 3, 3), padding=(0, 1, 1), bias=False),
            nn.GroupNorm(8, c0), nn.ReLU(inplace=True))
        self.b1 = v2.R2Plus1DBlock(c0, c1, pool=True)
        self.b2 = v2.R2Plus1DBlock(c1, c2, pool=True, drop=drop)
        self.b3 = v2.R2Plus1DBlock(c2, c3, pool=True, drop=drop)
        self.b4 = v2.R2Plus1DBlock(c3, c3, pool=True, drop=drop)
        self.head = PoolHead(c3 * (2 if asym else 1), num_classes, pool)

    def forward(self, x):
        sil = (x - 0.5) / 0.5
        diff = torch.cat([torch.zeros_like(sil[:, :1]), sil[:, 1:] - sil[:, :-1]], 1)
        h = self.b4(self.b3(self.b2(self.b1(self.stem(torch.stack([sil, diff], 1))))))  # [B,C,T,H,W]
        feat = h.mean(dim=(3, 4))  # [B,C,T]
        if self.asym:
            # left/right asymmetry of the feature map (mirror difference), pooled over space
            a = (h - h.flip(-1)).abs().mean(dim=(3, 4))
            feat = torch.cat([feat, a], 1)
        return self.head(feat.permute(0, 2, 1))


# =======================================================================
# 3. TRAINING OF ONE FOLD
# =======================================================================
def run_fold(args, data, labels, tr, va, te, seed, device, use_amp, tag):
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    rng = np.random.RandomState(seed)
    model = ST_CRANetV3(5, args.base_width, args.pool, args.asym).to(device)
    eval_model = ST_CRANetV3(5, args.base_width, args.pool, args.asym).to(device)
    ema = v2.EMA(model, args.ema)

    steps = max(1, len(tr) // args.batch_size)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.epochs * steps, pct_start=0.15)
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)
    crit = nn.CrossEntropyLoss(label_smoothing=0.1)

    best_val, best_loss, best_state, best_ep = -1.0, 1e9, None, -1
    for epoch in range(args.epochs):
        model.train()
        perm = np.random.permutation(tr)
        for s in range(steps):
            bi = perm[s * args.batch_size:(s + 1) * args.batch_size]
            fi = torch.from_numpy(v2.sample_indices(bi, data.shape[1], args.seq_len, True, rng)).to(device)
            x = data[torch.as_tensor(bi, device=device)[:, None], fi].float() / 255.0
            y = labels[torch.as_tensor(bi)].to(device)
            x = v2.gpu_augment(x)
            y2, lam = y, 1.0
            if args.mixup > 0 and random.random() < 0.5:
                lam = float(np.random.beta(args.mixup, args.mixup))
                p2 = torch.randperm(x.size(0), device=device)
                x, y2 = lam * x + (1 - lam) * x[p2], y[p2]
            with torch.autocast('cuda', dtype=torch.float16, enabled=use_amp):
                out = model(x)
                loss = lam * crit(out, y) + (1 - lam) * crit(out, y2)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward(); scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            scaler.step(opt); scaler.update(); sched.step()
            ema.update(model)

        ema.copy_to(eval_model)
        v_loss, v_acc, v_f1, _, _ = v2.evaluate(eval_model, data, labels, va, args, device, use_amp)
        if v_acc > best_val or (v_acc == best_val and v_loss < best_loss):
            best_val, best_loss, best_ep = v_acc, v_loss, epoch
            best_state = {k: t.clone() for k, t in eval_model.state_dict().items()}
        if (epoch + 1) % 10 == 0 or epoch == 0 or epoch + 1 == args.epochs:
            print(f"  [{tag}] epoch {epoch+1}/{args.epochs} | val acc {v_acc:.2f}% f1 {v_f1:.3f} | best {best_val:.2f}% @ {best_ep+1}")

    # test fold is used exactly once, with the checkpoint chosen on the validation fold
    eval_model.load_state_dict(best_state)
    _, t_acc, t_f1, preds, targets = v2.evaluate(eval_model, data, labels, te, args, device, use_amp)
    return {'val_acc': best_val, 'best_epoch': best_ep + 1, 'test_acc': t_acc, 'test_f1': t_f1,
            'preds': preds, 'targets': targets}


# =======================================================================
# 4. MAIN
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
    p = argparse.ArgumentParser("ST-CRANet v3 cross-validation harness")
    p.add_argument('--dataset_dir', default=os.path.join(here, 'Gait_Dataset_silhoutte_based', 'GAIT_IT'))
    p.add_argument('--pool', choices=['trap', 'mean', 'max', 'attn'], default='trap')
    p.add_argument('--asym', action='store_true', help='add left/right asymmetry branch')
    p.add_argument('--crop', action='store_true', help='crop each sequence to its silhouette bounding box')
    p.add_argument('--folds', type=int, default=5)
    p.add_argument('--run_folds', type=int, default=0, help='only run the first N folds (0 = all); for quick checks')
    p.add_argument('--seeds', type=int, nargs='+', default=[42])
    p.add_argument('--seq_len', type=int, default=16)
    p.add_argument('--cache_len', type=int, default=32)
    p.add_argument('--img_size', type=int, default=64)
    p.add_argument('--base_width', type=int, default=24)
    p.add_argument('--batch_size', type=int, default=16)
    p.add_argument('--epochs', type=int, default=60)
    p.add_argument('--lr', type=float, default=2e-3)
    p.add_argument('--weight_decay', type=float, default=5e-2)
    p.add_argument('--mixup', type=float, default=0.0)
    p.add_argument('--ema', type=float, default=0.98)
    p.add_argument('--out_dir', default=os.path.join(here, 'cv_results'))
    p.add_argument('--no_amp', action='store_true')
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    log = open(os.path.join(args.out_dir, 'train_log.txt'), 'w', encoding='utf-8')
    sys.stdout, sys.stderr = _Tee(sys.__stdout__, log), _Tee(sys.__stderr__, log)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    use_amp = device.type == 'cuda' and not args.no_amp
    print(f"[INFO] device={device} amp={use_amp} | pool={args.pool} asym={args.asym} crop={args.crop} "
          f"folds={args.folds} seeds={args.seeds} epochs={args.epochs}")
    print("[INFO] args:", json.dumps(vars(args)))

    cache_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'cache_v3',
                              f"c_{args.img_size}_{args.cache_len}_{'crop' if args.crop else 'full'}.pt")
    data, labels, groups, views = load_cache(args.dataset_dir, args.cache_len, args.img_size, args.crop, cache_file)
    if len(labels) == 0:
        print("[ERROR] no sequences found:", args.dataset_dir); return
    data = data.to(device)
    n_params = sum(p.numel() for p in ST_CRANetV3(5, args.base_width, args.pool, args.asym).parameters())
    print(f"[INFO] sequences={len(labels)} people={len(set(groups))} params={n_params/1e6:.3f}M "
          f"views={ {v: views.count(v) for v in set(views)} }")

    results = []
    n_run = args.run_folds if args.run_folds > 0 else args.folds
    agg_conf = np.zeros((5, 5), dtype=int)
    for seed in args.seeds:
        fold_id = make_folds(labels.numpy(), groups, args.folds, seed)
        for k in range(n_run):
            te = np.where(fold_id == k)[0]
            va = np.where(fold_id == (k + 1) % args.folds)[0]
            tr = np.where((fold_id != k) & (fold_id != (k + 1) % args.folds))[0]
            tag = f"seed{seed}-fold{k+1}/{args.folds}"
            print(f"\n=== {tag}: train={len(tr)} val={len(va)} test={len(te)} "
                  f"(people {len({groups[i] for i in tr})}/{len({groups[i] for i in va})}/{len({groups[i] for i in te})}) ===")
            r = run_fold(args, data, labels, tr, va, te, seed, device, use_amp, tag)
            print(f"  >> {tag}: best val {r['val_acc']:.2f}% @ep{r['best_epoch']} | TEST acc {r['test_acc']:.2f}% f1 {r['test_f1']:.4f}")
            agg_conf += v2.confusion_matrix(r['targets'], r['preds'])
            results.append({'seed': seed, 'fold': k + 1, 'val_acc': r['val_acc'], 'best_epoch': r['best_epoch'],
                            'test_acc': r['test_acc'], 'test_f1': r['test_f1']})
            with open(os.path.join(args.out_dir, 'cv_results.json'), 'w') as f:
                json.dump({'args': vars(args), 'params': n_params, 'folds': results}, f, indent=2)

    accs = np.array([r['test_acc'] for r in results]); f1s = np.array([r['test_f1'] for r in results])
    lines = [f"CONFIG pool={args.pool} asym={args.asym} crop={args.crop} params={n_params/1e6:.3f}M "
             f"folds={args.folds} seeds={args.seeds} epochs={args.epochs}",
             f"TEST accuracy: {accs.mean():.2f} +/- {accs.std():.2f} %   (over {len(accs)} fold runs)",
             f"TEST macro-F1: {f1s.mean():.4f} +/- {f1s.std():.4f}",
             "Per-fold test acc: " + ", ".join(f"{a:.1f}" for a in accs),
             "",
             "Pooled confusion matrix (rows=true, cols=pred): " + str(CLASS_NAMES),
             str(agg_conf)]
    pooled = agg_conf.diagonal().sum() / max(1, agg_conf.sum()) * 100
    lines.append(f"Pooled accuracy over all test folds: {pooled:.2f}%")
    print("\n" + "\n".join(lines))
    with open(os.path.join(args.out_dir, 'summary.txt'), 'w') as f:
        f.write("\n".join(lines) + "\n")
    print(f"[SAVED] {os.path.join(args.out_dir, 'summary.txt')}")


if __name__ == '__main__':
    main()
