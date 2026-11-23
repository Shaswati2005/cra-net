"""
===============================================================================
CRA-Net (Bodepu et al., Biomed. Signal Process. Control 108 (2025) 107886)
re-implemented in PyTorch, evaluated under three splits
===============================================================================

Architecture, following Sec. 3 and Fig. 1 of the paper (single silhouette frame in,
class out; the paper is frame-level, "this model only considers spatial data"):

  Block I   : 4 double-conv blocks (Eq. 1), 2x2 max-pool between them.
  Block II  : 4 single-conv blocks (Eq. 2). After each of the first three, a CRAB
              (Eqs. 3-9) fuses it with the Block-I output of the same level; the
              CRAB output is max-pooled and fed to the next single-conv block.
  Block III : 3 x (2x2 max-pool -> CAB (Eqs. 10-12)). CAB k takes the pooled
              Block-I and Block-II (CRAB) features of level k.
  Head      : concat(D_f, S_f, C) -> single conv -> flatten -> dense -> softmax (Eq. 13).

The paper does not give channel widths, input size, how the three CAB outputs are
combined, the rotation range, or the train/val/test split. Choices made here
(all switchable):
  * widths 16/32/64/128 and a 64x64 input -> ~0.9 M parameters (paper: 0.934 M);
  * the three CAB outputs are average-pooled to the last level and concatenated
    into C ("integrates features from various local and global layers");
  * each frame is cropped to the silhouette bounding box before resizing
    (--no_crop resizes the whole frame instead);
  * rotation augmentation +/-10 degrees.

Note on Eq. 12: C = (Cs2 * Cs1) + (Cs1 * Cs2). Element-wise multiplication commutes,
so this equals 2 * sigmoid(conv(Cf1)) * sigmoid(conv(Cf2)). Implemented literally.

Training as in Sec. 4.2: Adam, lr 1e-3, (sparse) cross-entropy, 50 epochs, batch 32
(best batch size in Fig. 3), checkpoint with best validation accuracy, horizontal
flip / vertical flip / random rotation augmentation.

Splits (--split):
  frame    : random stratified 70/15/15 split of individual frames. Frames of the
             same walk land in train and test (what a frame-level random split does).
  sequence : random stratified 70/15/15 split of whole sequences. Same people in
             train and test, but never the same walk.
  subject  : the 10 subject-exclusive folds of AnatoPhaseNet v3-v6 (same seed, same
             test people): fold k tests on group k, validates on group k+1.

Metrics: frame accuracy, macro-F1, per-class precision/recall/F1, QWK (the paper's
second metric) and walk accuracy (all frames of a sequence, mean log-probability).
===============================================================================
"""

import os
import re
import json
import time
import random
import argparse
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F

GAIT_IT_CLASSES = ['normal', 'diplegic', 'hemiplegic', 'neuropathic', 'parkinson']
INIT_8 = ['nm', 'a_l0', 'a_l0.5', 'a_r0', 'a_r0.5', 'l_l0.5', 'l_r0.5', 'fb']
INIT_6_MERGE = {'a_l0.5': 'a_l0', 'a_r0.5': 'a_r0'}
IMG_EXT = ('.jpg', '.png', '.jpeg')


# =======================================================================
# 1. DATA
# =======================================================================
def list_frames(seq_dir):
    return sorted(
        [f for f in os.listdir(seq_dir) if f.lower().endswith(IMG_EXT)],
        key=lambda x: int(re.findall(r'\d+', x)[-1]) if re.findall(r'\d+', x) else x
    )


def collect_gait_it(root_dir):
    """Every side-view sequence with frames (no key frames needed: CRA-Net is frame-level)."""
    seqs = []
    for label, name in enumerate(GAIT_IT_CLASSES):
        gait_p = os.path.join(root_dir, f'{name}_silhouettes')
        if not os.path.isdir(gait_p):
            continue
        for subj in sorted(os.listdir(gait_p)):
            side_p = os.path.join(gait_p, subj, 'silhouettes', 'side_view')
            if not os.path.isdir(side_p):
                continue
            for seq_name in sorted(os.listdir(side_p)):
                seq_dir = os.path.join(side_p, seq_name)
                if seq_name == 'metadata' or not os.path.isdir(seq_dir):
                    continue
                files = list_frames(seq_dir)
                if files:
                    seqs.append(dict(dir=seq_dir, files=files, label=label, person=subj,
                                     seq_id=f'{subj}_{name}_{seq_name}'))
    return seqs, list(GAIT_IT_CLASSES)


def collect_init(root_dir, n_classes):
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
            if files:
                seqs.append(dict(dir=seq_dir, files=files, label=label, person=seq_name[:3],
                                 seq_id=seq_name))
    return seqs, [n.replace('_', '-') for n in names]


def load_frame(path, size, crop):
    im = Image.open(path).convert('L')
    if crop:
        a = np.array(im)
        ys, xs = np.where(a > 30)
        if len(ys):
            y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
            py, px = int((y1 - y0) * 0.08), int((x1 - x0) * 0.08)
            im = im.crop((max(0, x0 - px), max(0, y0 - py), min(a.shape[1], x1 + px), min(a.shape[0], y1 + py)))
    w, h = im.size
    s = size / max(w, h)
    nw, nh = max(1, int(w * s)), max(1, int(h * s))
    canvas = Image.new('L', (size, size), 0)
    canvas.paste(im.resize((nw, nh), Image.BILINEAR), ((size - nw) // 2, (size - nh) // 2))
    return np.array(canvas, dtype=np.uint8)


def load_cache(args, cache_file):
    if os.path.isfile(cache_file):
        d = torch.load(cache_file, weights_only=False)
        print(f"[INFO] Loaded frame cache {cache_file}: {tuple(d['frames'].shape)}")
        return d

    if args.dataset == 'gait_it':
        seqs, class_names = collect_gait_it(args.dataset_dir)
    else:
        seqs, class_names = collect_init(args.dataset_dir, args.init_classes)

    jobs, frame_seq = [], []
    for si, s in enumerate(seqs):
        for f in s['files'][::args.frame_stride]:
            jobs.append(os.path.join(s['dir'], f))
            frame_seq.append(si)
    print(f"[INFO] {len(seqs)} sequences, {len(set(s['person'] for s in seqs))} subjects -> "
          f"{len(jobs)} frames (every {args.frame_stride}th). Building cache...")

    frames = np.zeros((len(jobs), args.img_size, args.img_size), dtype=np.uint8)
    with ThreadPoolExecutor(max_workers=8) as ex:
        for i, arr in enumerate(ex.map(lambda p: load_frame(p, args.img_size, not args.no_crop), jobs, chunksize=64)):
            frames[i] = arr
            if (i + 1) % 20000 == 0:
                print(f"  cached {i + 1}/{len(jobs)} frames")

    d = {'frames': torch.from_numpy(frames),
         'frame_seq': torch.tensor(frame_seq, dtype=torch.long),
         'seq_labels': torch.tensor([s['label'] for s in seqs], dtype=torch.long),
         'seq_persons': [s['person'] for s in seqs],
         'seq_ids': [s['seq_id'] for s in seqs],
         'class_names': class_names}
    os.makedirs(os.path.dirname(cache_file), exist_ok=True)
    torch.save(d, cache_file)
    return d


# =======================================================================
# 2. SPLITS
# =======================================================================
def stratified_split(labels, seed, fracs=(0.70, 0.15, 0.15)):
    """Index arrays (train, val, test), stratified by label."""
    rng = np.random.RandomState(seed)
    tr, va, te = [], [], []
    for c in np.unique(labels):
        idx = np.where(labels == c)[0]
        rng.shuffle(idx)
        n_tr = int(round(len(idx) * fracs[0]))
        n_va = int(round(len(idx) * fracs[1]))
        tr += list(idx[:n_tr]); va += list(idx[n_tr:n_tr + n_va]); te += list(idx[n_tr + n_va:])
    return np.array(tr), np.array(va), np.array(te)


def make_strict_person_folds(persons, num_folds=10, seed=42):
    """Identical to AnatoPhaseNet v3-v6, so the test people per fold are the same."""
    unique_persons = sorted(list(set(persons)))
    rng = random.Random(seed)
    shuffled = list(unique_persons)
    rng.shuffle(shuffled)
    person_to_fold = {p: i % num_folds for i, p in enumerate(shuffled)}
    for f in range(num_folds):
        print(f"  Fold {f + 1:2d}: {[p for p in shuffled if person_to_fold[p] == f]}")
    return np.array([person_to_fold[p] for p in persons], dtype=int)


def build_splits(d, args):
    """List of (name, train_frames, val_frames, test_frames)."""
    frame_seq = d['frame_seq'].numpy()
    seq_labels = d['seq_labels'].numpy()
    frame_labels = seq_labels[frame_seq]

    if args.split == 'frame':
        tr, va, te = stratified_split(frame_labels, args.split_seed)
        return [('split', tr, va, te)]

    if args.split == 'sequence':
        s_tr, s_va, s_te = stratified_split(seq_labels, args.split_seed)
        m = lambda s: np.where(np.isin(frame_seq, s))[0]
        return [('split', m(s_tr), m(s_va), m(s_te))]

    print(f"[INFO] Subject-exclusive {args.folds}-fold split (seed {args.split_seed}):")
    seq_folds = make_strict_person_folds(d['seq_persons'], args.folds, args.split_seed)
    frame_folds = seq_folds[frame_seq]
    out = []
    for k in range(args.folds):
        if args.fold_ids and (k + 1) not in args.fold_ids:
            continue
        v = (k + 1) % args.folds
        out.append((f'fold_{k + 1}', np.where((frame_folds != k) & (frame_folds != v))[0],
                    np.where(frame_folds == v)[0], np.where(frame_folds == k)[0]))
    return out


# =======================================================================
# 3. MODEL (Sec. 3, Fig. 1)
# =======================================================================
class SingleConv(nn.Sequential):
    """Eq. 2: S_f = ReLU(BN(conv3x3(X)))."""
    def __init__(self, cin, cout):
        super().__init__(nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True))


class DoubleConv(nn.Sequential):
    """Eq. 1: D_f = ReLU(BN(conv3x3(ReLU(BN(conv3x3(X))))))."""
    def __init__(self, cin, cout):
        super().__init__(SingleConv(cin, cout), SingleConv(cout, cout))


class CRAB(nn.Module):
    """Cross reverse attention block, Eqs. 3-9. Rf1: double-conv features, Rf2: single-conv features."""
    def __init__(self, c):
        super().__init__()
        self.att1 = nn.Conv2d(c, c, 1)
        self.att2 = nn.Conv2d(c, c, 1)
        self.proj1 = nn.Conv2d(c, c, 1)

    def forward(self, rf1, rf2):
        rs1 = torch.sigmoid(self.att1(rf1))      # Eq. 3
        rs2 = torch.sigmoid(self.att2(rf2))      # Eq. 4
        r1 = 1.0 - rs1                            # Eq. 5, b2 = -1, b3 = 1
        r2 = 1.0 - rs2                            # Eq. 6, b0 = -1, b1 = 1
        rc1 = self.proj1(rf1) * r2                # Eq. 7
        rc2 = rc1 + rf2                           # Eq. 8
        return r1 * r2 + rc2                      # Eq. 9


class CAB(nn.Module):
    """Cross attention block, Eqs. 10-12 (literal: the two products in Eq. 12 are equal)."""
    def __init__(self, c):
        super().__init__()
        self.att1 = nn.Conv2d(c, c, 1)
        self.att2 = nn.Conv2d(c, c, 1)

    def forward(self, cf1, cf2):
        cs1 = torch.sigmoid(self.att1(cf1))      # Eq. 10
        cs2 = torch.sigmoid(self.att2(cf2))      # Eq. 11
        return cs2 * cs1 + cs1 * cs2              # Eq. 12


class CRANet(nn.Module):
    def __init__(self, num_classes, img_size=64, widths=(16, 32, 64, 128), head_ch=128, in_ch=1):
        super().__init__()
        w = widths
        self.block1 = nn.ModuleList([DoubleConv(in_ch if i == 0 else w[i - 1], w[i]) for i in range(4)])
        self.block2 = nn.ModuleList([SingleConv(in_ch if i == 0 else w[i - 1], w[i]) for i in range(4)])
        self.crabs = nn.ModuleList([CRAB(w[i]) for i in range(3)])
        self.cabs = nn.ModuleList([CAB(w[i]) for i in range(3)])
        self.pool = nn.MaxPool2d(2)
        self.fuse = SingleConv(w[3] + w[3] + sum(w[:3]), head_ch)
        s = img_size // 8
        self.fc = nn.Linear(head_ch * s * s, num_classes)

    def forward(self, x):
        d = x
        s = x
        cabs = []
        for i in range(4):
            d = self.block1[i](d if i == 0 else self.pool(d))
            s = self.block2[i](s if i == 0 else self.pool(s))
            if i < 3:
                s = self.crabs[i](d, s)                                     # level-i CRAB
                cabs.append(self.cabs[i](self.pool(d), self.pool(s)))       # Block III: pool -> CAB
        size = d.shape[-2:]
        c = torch.cat([F.adaptive_avg_pool2d(t, size) for t in cabs], 1)
        y = self.fuse(torch.cat([d, s, c], 1))                              # Eq. 13
        return self.fc(torch.flatten(y, 1))                                 # softmax inside the loss


# =======================================================================
# 4. AUGMENTATION, METRICS
# =======================================================================
def augment(x, max_deg):
    """Horizontal flip, vertical flip, random rotation (Sec. 4.2). x: [B,1,H,W] in [0,1]."""
    B = x.shape[0]
    hf = torch.rand(B, device=x.device) < 0.5
    vf = torch.rand(B, device=x.device) < 0.5
    x = torch.where(hf[:, None, None, None], x.flip(3), x)
    x = torch.where(vf[:, None, None, None], x.flip(2), x)
    if max_deg > 0:
        a = (torch.rand(B, device=x.device) * 2 - 1) * max_deg * np.pi / 180
        cos, sin = torch.cos(a), torch.sin(a)
        theta = torch.stack([torch.stack([cos, -sin, torch.zeros_like(a)], 1),
                             torch.stack([sin, cos, torch.zeros_like(a)], 1)], 1)
        grid = F.affine_grid(theta, x.shape, align_corners=False)
        x = F.grid_sample(x, grid, align_corners=False)
    return x


def confusion(t, p, k):
    cm = np.zeros((k, k), dtype=np.int64)
    np.add.at(cm, (t, p), 1)
    return cm


def class_report(cm):
    tp = np.diag(cm).astype(float)
    prec = tp / np.maximum(cm.sum(0), 1)
    rec = tp / np.maximum(cm.sum(1), 1)
    f1 = 2 * prec * rec / np.maximum(prec + rec, 1e-12)
    return prec, rec, f1


def qwk(cm):
    """Quadratic weighted kappa, as in the paper (treats class indices as ordered)."""
    k = cm.shape[0]
    w = np.array([[(i - j) ** 2 for j in range(k)] for i in range(k)], dtype=float) / max((k - 1) ** 2, 1)
    n = cm.sum()
    expected = np.outer(cm.sum(1), cm.sum(0)) / max(n, 1)
    denom = (w * expected).sum()
    return float(1 - (w * cm).sum() / denom) if denom > 0 else 0.0


@torch.no_grad()
def predict(model, frames, idx, device, bs=512):
    model.eval()
    out = []
    for i in range(0, len(idx), bs):
        xb = frames[idx[i:i + bs]].to(device, non_blocking=True).float().unsqueeze(1) / 255.0
        with torch.autocast(device_type=device.type, enabled=device.type == 'cuda'):
            out.append(F.log_softmax(model(xb).float(), 1).cpu())
    return torch.cat(out).numpy()


def evaluate(model, d, idx, device, k):
    logp = predict(model, d['frames_dev'], idx, device)
    t = d['frame_labels'][idx]
    p = logp.argmax(1)
    cm = confusion(t, p, k)
    prec, rec, f1 = class_report(cm)
    # walk level: mean log-probability over every test frame of the sequence
    seqs = d['frame_seq_np'][idx]
    ws = np.unique(seqs)
    wl = np.array([logp[seqs == s].mean(0).argmax() for s in ws])
    wt = d['seq_labels_np'][ws]
    return dict(acc=100 * float((p == t).mean()), f1=float(f1.mean()), cm=cm, prec=prec, rec=rec, f1c=f1,
                qwk=qwk(cm), walk_acc=100 * float((wl == wt).mean()), n=len(idx), n_walks=len(ws))


# =======================================================================
# 5. MAIN
# =======================================================================
def main():
    here = os.path.dirname(os.path.abspath(__file__))
    data_root = os.path.join(os.path.dirname(here), 'Gait_Dataset_silhoutte_based')
    default_dirs = {'gait_it': os.path.join(data_root, 'GAIT_IT'),
                    'init': os.path.join(data_root, 'INIT', 'INIT_GaitDB', 'INIT_8', 'silhouettes')}

    ap = argparse.ArgumentParser(description='CRA-Net (Bodepu et al. 2025) under frame / sequence / subject splits')
    ap.add_argument('--dataset', default='gait_it', choices=['gait_it', 'init'])
    ap.add_argument('--dataset_dir', default=None)
    ap.add_argument('--init_classes', type=int, default=8, choices=[8, 6])
    ap.add_argument('--split', default='subject', choices=['frame', 'sequence', 'subject'])
    ap.add_argument('--folds', type=int, default=10)
    ap.add_argument('--fold_ids', type=int, nargs='*', default=None, help='Run only these folds (1-based)')
    ap.add_argument('--split_seed', type=int, default=42)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--img_size', type=int, default=64)
    ap.add_argument('--widths', type=int, nargs=4, default=[16, 32, 64, 128])
    ap.add_argument('--frame_stride', type=int, default=4, help='Keep every Nth frame (1 = all frames)')
    ap.add_argument('--no_crop', action='store_true', help='Resize the whole frame instead of the silhouette box')
    ap.add_argument('--epochs', type=int, default=50)
    ap.add_argument('--batch_size', type=int, default=32)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--rotation', type=float, default=10.0, help='Max rotation (degrees) for augmentation')
    ap.add_argument('--out_dir', default=None)
    args = ap.parse_args()

    args.dataset_dir = args.dataset_dir or default_dirs[args.dataset]
    tag = args.dataset if args.dataset == 'gait_it' else f'init{args.init_classes}'
    args.out_dir = args.out_dir or os.path.join(here, f'results_{tag}_{args.split}')
    os.makedirs(args.out_dir, exist_ok=True)

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    cache_file = os.path.join(here, 'cache',
                              f"{tag}_s{args.img_size}_stride{args.frame_stride}{'_nocrop' if args.no_crop else ''}.pt")
    d = load_cache(args, cache_file)
    class_names = d['class_names']
    k = len(class_names)
    d['frame_seq_np'] = d['frame_seq'].numpy()
    d['seq_labels_np'] = d['seq_labels'].numpy()
    d['frame_labels'] = d['seq_labels_np'][d['frame_seq_np']]
    # keep the uint8 frames on the GPU when they fit (GAIT-IT, stride 4, 64 px: ~0.45 GB)
    on_gpu = device.type == 'cuda' and d['frames'].numel() < 1.5e9
    d['frames_dev'] = d['frames'].to(device) if on_gpu else d['frames'].pin_memory() if device.type == 'cuda' else d['frames']
    labels_dev = torch.from_numpy(d['frame_labels']).to(device)

    probe = CRANet(k, args.img_size, args.widths)
    n_params = sum(p.numel() for p in probe.parameters())
    print(f"[INFO] CRA-Net: {n_params / 1e6:.3f} M parameters (paper: 0.934 M), input {args.img_size}x{args.img_size}")
    gflops = None
    try:
        from torch.utils.flop_counter import FlopCounterMode
        with FlopCounterMode(display=False) as fc:
            probe(torch.zeros(1, 1, args.img_size, args.img_size))
        gflops = fc.get_total_flops() / 2e9   # multiply-adds, the usual "FLOPs" convention
        print(f"[INFO] {gflops:.3f} GMACs per frame (paper reports 1.928 GFLOPs)")
    except Exception:
        pass

    results = []
    for name, tr, va, te in build_splits(d, args):
        print(f"\n[{name}] train {len(tr)} | val {len(va)} | test {len(te)} frames")
        torch.manual_seed(args.seed)
        model = CRANet(k, args.img_size, args.widths).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=args.lr)
        scaler = torch.amp.GradScaler('cuda', enabled=device.type == 'cuda')
        tr_t = torch.from_numpy(tr).to(device)
        best, best_state, hist = -1.0, None, []

        for ep in range(1, args.epochs + 1):
            model.train()
            t0 = time.time()
            perm = tr_t[torch.randperm(len(tr_t), device=device)]
            tot, correct, loss_sum = 0, 0, 0.0
            for i in range(0, len(perm), args.batch_size):
                bi = perm[i:i + args.batch_size]
                xb = d['frames_dev'][bi if on_gpu else bi.cpu()].to(device, non_blocking=True).float().unsqueeze(1) / 255.0
                xb = augment(xb, args.rotation)
                yb = labels_dev[bi]
                with torch.autocast(device_type=device.type, enabled=device.type == 'cuda'):
                    logits = model(xb)
                    loss = F.cross_entropy(logits.float(), yb)
                opt.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
                tot += len(bi); correct += (logits.argmax(1) == yb).sum().item(); loss_sum += loss.item() * len(bi)
            v = evaluate(model, d, va, device, k)
            hist.append(dict(epoch=ep, train_loss=loss_sum / tot, train_acc=100 * correct / tot,
                             val_acc=v['acc'], val_f1=v['f1']))
            mark = ''
            if v['acc'] > best:            # best validation accuracy checkpoint (Sec. 4.2)
                best = v['acc']
                best_state = {kk: vv.detach().clone() for kk, vv in model.state_dict().items()}
                mark = ' *'
            print(f"  ep {ep:2d} | loss {loss_sum / tot:.4f} | train {100 * correct / tot:5.2f}% | "
                  f"val {v['acc']:5.2f}% (F1 {v['f1']:.3f}) | {time.time() - t0:4.1f}s{mark}")

        model.load_state_dict(best_state)
        r = evaluate(model, d, te, device, k)
        # inference time per frame, batch 1 (the paper's Table 6 column)
        x1 = torch.zeros(1, 1, args.img_size, args.img_size, device=device)
        model.eval()
        with torch.no_grad():
            for _ in range(20):
                model(x1)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            t0 = time.time()
            for _ in range(200):
                model(x1)
            if device.type == 'cuda':
                torch.cuda.synchronize()
        r['ms_per_frame'] = (time.time() - t0) / 200 * 1000

        print(f">> [{name}] TEST frame acc {r['acc']:.2f}% | macro-F1 {r['f1']:.3f} | QWK {r['qwk']:.4f} | "
              f"walk acc {r['walk_acc']:.2f}% ({r['n_walks']} walks)")
        for c in range(k):
            print(f"     {class_names[c]:>12s}  P {r['prec'][c]:.3f}  R {r['rec'][c]:.3f}  F1 {r['f1c'][c]:.3f}")
        results.append(dict(name=name, val_best=best, test_frame_acc=r['acc'], test_frame_f1=r['f1'],
                            test_qwk=r['qwk'], test_walk_acc=r['walk_acc'], n_test_frames=r['n'],
                            n_test_walks=r['n_walks'], precision=r['prec'].tolist(), recall=r['rec'].tolist(),
                            f1_per_class=r['f1c'].tolist(), confusion=r['cm'].tolist(),
                            ms_per_frame=r['ms_per_frame'], history=hist))
        with open(os.path.join(args.out_dir, f'{name}_result.json'), 'w') as f:
            json.dump(results[-1], f, indent=1)

    accs = np.array([r['test_frame_acc'] for r in results])
    walks = np.array([r['test_walk_acc'] for r in results])
    agg = np.sum([np.array(r['confusion']) for r in results], axis=0)
    prec, rec, f1 = class_report(agg)
    summary = dict(dataset=tag, split=args.split, args=vars(args), params=n_params, gmacs=gflops,
                   frame_accuracy_mean=float(accs.mean()), frame_accuracy_std=float(accs.std()),
                   walk_accuracy_mean=float(walks.mean()), walk_accuracy_std=float(walks.std()),
                   frame_f1_mean=float(np.mean([r['test_frame_f1'] for r in results])),
                   qwk_mean=float(np.mean([r['test_qwk'] for r in results])),
                   ms_per_frame=float(np.mean([r['ms_per_frame'] for r in results])),
                   class_names=class_names, aggregated_confusion=agg.tolist(),
                   pooled_precision=prec.tolist(), pooled_recall=rec.tolist(), pooled_f1=f1.tolist(),
                   folds=[{kk: vv for kk, vv in r.items() if kk != 'history'} for r in results])
    with open(os.path.join(args.out_dir, 'benchmark_summary.json'), 'w') as f:
        json.dump(summary, f, indent=1)
    print(f"\n==== CRA-Net | {tag} | split={args.split} | {len(results)} run(s) ====")
    print(f"Frame accuracy {accs.mean():.2f} +/- {accs.std():.2f}% | walk accuracy {walks.mean():.2f} +/- "
          f"{walks.std():.2f}% | macro-F1 {summary['frame_f1_mean']:.3f} | QWK {summary['qwk_mean']:.4f}")
    print(f"Saved to {args.out_dir}")


if __name__ == '__main__':
    main()
