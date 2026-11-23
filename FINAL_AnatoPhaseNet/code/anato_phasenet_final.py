"""
===============================================================================
AnatoPhaseNet v6: Geometry-Aware, Subject-Invariant Phase Network
===============================================================================

Builds on AnatoPhaseNet v5 (severity multi-task + multi-cycle stride variability) and
v4 (direction-canonical cycles, cyclic (2+1)D backbone, part head, half-cycle symmetry).
The v4 image cache is reused; v6 adds a small per-frame geometry cache.

New in v6:
1. Biomechanical geometry branch
   v4 crops and rescales every frame on its own, so every silhouette ends up the same
   height. That removes body-height changes (crouch, stoop), vertical bob and
   stride width relative to body height, which separate e.g. spastic diplegic from
   Parkinsonian gait. v6 measures these on the raw frames before cropping:
     0 height / walk height - 1     (crouch / stoop over the cycle)
     1 bbox width / walk height     (stride width)
     2 lower-body width / height    (feet, bottom 30 % of the bbox)
     3 upper-body width / height    (arms + torso, 15-55 % of the bbox)
     4 head bob (top y - walk median) / height
     5 relative centroid height (cy - top) / bbox height
     6 trunk lean (x of top 30 % - x of bottom 30 %) / height, signed in walking direction
     7 forward mass offset (cx - bbox centre) / height, signed in walking direction
     8 fill ratio area / (h * w)
   Every value is normalised by the walk's median height, so camera distance cancels.
   A cyclic temporal CNN turns the 9-phase sequence of these values into features
   [mean, max, half-cycle symmetry]. They feed an extra "part" classifier (averaged with
   the body-part classifiers) and are joined with the image descriptor for the severity,
   embedding and session heads.
2. Subject-adversarial training (--adv_subject)
   A subject classifier sits on the cycle descriptor behind a gradient-reversal layer,
   so the backbone is pushed to drop each person's own walking style
   (DANN-style schedule for the reversal strength).
3. Ablation switches: --no_geom, --adv_subject, plus the v5 ones
   (--no_severity, --no_multicycle, --no_sym).
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

# FINAL folder layout: <project>/FINAL_AnatoPhaseNet/code/this_file.py
HERE = os.path.dirname(os.path.abspath(__file__))           # .../FINAL_AnatoPhaseNet/code
FINAL_DIR = os.path.dirname(HERE)                           # .../FINAL_AnatoPhaseNet
PARENT = os.path.dirname(FINAL_DIR)                         # project root (dataset, resnet18_pretrained.pth)
sys.path.insert(0, HERE)                                    # v4/v5 modules are copied next to this file


def _cache_path(old_dir, name):
    """Reuse an existing cache from the original v4/v6 folders if present, else build it in FINAL_AnatoPhaseNet/cache."""
    old = os.path.join(PARENT, old_dir, 'cache', name)
    return old if os.path.isfile(old) else os.path.join(FINAL_DIR, 'cache', name)

from anato_phasenet_v4 import (  # noqa: E402
    NUM_PHASES, GAIT_IT_NAMES, compute_macro_f1, confusion_matrix, plot_fold_curves, plot_confusion_matrix_chart,
    collect_gait_it, collect_init, load_cache, gpu_augment_phases, make_sampler_weights, FocalCrossEntropyLoss,
    R2Plus1DBackbone, TSMBackbone, AnatoPhaseNet_v4, EMA, make_strict_person_folds, is_better,
)
from anato_phasenet_v5 import SEVERITY_NAMES, severity_labels, PartPhaseHead_v5, session_statistics  # noqa: E402

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


class AnatoPhaseNet_v6(nn.Module):
    def __init__(self, num_classes, pretrained_path=None, backbone='r2plus1d', num_phases=9, num_parts=6,
                 last_stride=2, head_dim=128, dropout=0.3, use_sym=True, num_severity=3, emb_dim=256,
                 use_geom=True, geom_dim=64, num_subjects=0):
        super().__init__()
        if backbone == 'r2plus1d':
            self.backbone = R2Plus1DBackbone(num_phases, last_stride)
        else:
            self.backbone = TSMBackbone(num_phases, last_stride)

        self.head = PartPhaseHead_v5(256, num_classes, num_parts=num_parts, num_phases=num_phases,
                                     dim=head_dim, dropout=dropout, use_sym=use_sym)
        fd = self.head.feat_dim

        self.geom = GeometryBranch(NUM_GEOM, num_classes, dim=geom_dim, dropout=dropout) if use_geom else None
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
            AnatoPhaseNet_v4._load_pretrained(self, pretrained_path)

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
def evaluate_v6(model, data, geom, labels, sev_t, seq_ids, idx, device, num_classes, use_amp=True, tta=True,
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

    parser = argparse.ArgumentParser(description="AnatoPhaseNet v6: geometry-aware, subject-invariant gait classifier")
    parser.add_argument('--dataset', type=str, default='gait_it', choices=['gait_it', 'init'])
    parser.add_argument('--dataset_dir', type=str, default=None)
    parser.add_argument('--init_classes', type=int, default=8, choices=[8, 6])
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
    # v6 components
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
    parser.add_argument('--split_seed', type=int, default=42, help='Fold split seed (keep 42 to match v3/v4/v5)')
    parser.add_argument('--save_models', action='store_true')
    args = parser.parse_args()

    args.dataset_dir = args.dataset_dir or default_dirs[args.dataset]
    tag_ds = args.dataset if args.dataset == 'gait_it' else f"init{args.init_classes}"

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    use_amp = (device.type == 'cuda')

    # Image cache shared with v4/v5; geometry cache is v6's own
    canon_tag = '_nocanon' if args.no_canon else ''
    cache_file = _cache_path('anatophasenet_v4', f"{tag_ds}_{args.img_size}_j{args.jitter}{canon_tag}.pt")
    d = load_cache(args, cache_file)
    data, labels, persons, seq_ids = d['data'], d['labels'], d['persons'], d['seq_ids']
    class_names = d['class_names']
    num_classes = len(class_names)
    labels_np = labels.numpy()

    geom = load_geometry_cache(args, _cache_path('anatophasenet_v6', f"{tag_ds}_geom_j{args.jitter}{canon_tag}.pt"), d)
    geom_std = geom.reshape(-1, NUM_GEOM).std(dim=0).clamp(min=1e-4)

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
    args.out_dir = args.out_dir or os.path.join(FINAL_DIR, 'results', f"rerun_{tag_ds}_{args.backbone}_{variant}{ema_tag}{seed_tag}")
    os.makedirs(args.out_dir, exist_ok=True)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    print("=" * 75)
    print(f"  AnatoPhaseNet v6 | dataset: {tag_ds} | backbone: {args.backbone} | variant: {variant}")
    print(f"  Geometry branch: {use_geom} | Subject-adversarial: {use_adv} | Severity task: {use_severity} | "
          f"Multi-cycle: {use_mc} | Symmetry: {use_sym} | Device: {device}")
    print("=" * 75)
    print(f"[INFO] Classes: {class_names} | cycles per class: {np.bincount(labels_np, minlength=num_classes).tolist()}")
    if use_severity:
        print(f"[INFO] Severity labels {SEVERITY_NAMES}: {[int((sev_t == s).sum()) for s in range(3)]} "
              f"(unlabelled: {int((sev_t < 0).sum())})")

    model_kw = dict(backbone=args.backbone, num_parts=args.num_parts, last_stride=args.last_stride,
                    head_dim=args.head_dim, dropout=args.dropout, use_sym=use_sym, use_geom=use_geom,
                    geom_dim=args.geom_dim, num_subjects=len(subj_names) if use_adv else 0)
    model_tmp = AnatoPhaseNet_v6(num_classes, None, **model_kw)
    param_count = sum(p.numel() for p in model_tmp.parameters() if p.requires_grad)
    print(f"[INFO] Architecture: AnatoPhaseNet v6 ({args.backbone}) | Parameters: {param_count / 1e6:.3f}M")
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

        print("\n" + "-" * 75)
        print(f"[Fold {f_idx + 1}/{n_folds}] Train: {len(tr_idx)} cycles | Val: {len(va_idx)} cycles | Test: {len(te_idx)} cycles")
        print("-" * 75)

        model = AnatoPhaseNet_v6(num_classes, args.pretrained_path, **model_kw).to(device)
        eval_model = AnatoPhaseNet_v6(num_classes, None, **model_kw).to(device)
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
                    K = random.randint(2, max(2, args.k_max))
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

            v_raw = evaluate_v6(model, data, geom, labels, sev_t, seq_ids, va_idx, device, num_classes,
                                use_amp, tta=not args.no_tta, use_session_head=use_mc)
            ema.copy_to(eval_model)
            v = evaluate_v6(eval_model, data, geom, labels, sev_t, seq_ids, va_idx, device, num_classes,
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
        t = evaluate_v6(eval_model, data, geom, labels, sev_t, seq_ids, te_idx, device, num_classes,
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
    print(f"        ANATOPHASENET V6 FINAL BENCHMARK SUMMARY ({tag_ds}, {args.backbone}, {variant})")
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
