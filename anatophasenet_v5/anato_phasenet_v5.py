"""
===============================================================================
AnatoPhaseNet v5: Severity-Aware, Multi-Cycle Stride-Variability Network
===============================================================================

Builds on AnatoPhaseNet v4 (direction-canonical cycles, cyclic (2+1)D phase backbone,
part-based head, half-cycle symmetry). Data loading, backbones and utilities are
imported from ../anatophasenet_v4/anato_phasenet_v4.py and its cache is reused.

New in v5:
1. Severity-aware multi-task learning
   GAIT-IT sequence names carry the impairment level (lvl1 = mild, lvl2 = severe).
   A second head predicts severity {normal, mild, severe} next to the pathology, so
   the network learns "mildly abnormal" separately from "a different abnormality".
   INIT 6-class: a-*0.5 = mild, a-*0 = severe, nm = normal (other classes ignored).
   INIT 8-class: severity is already part of the class, so the task is disabled.
2. Multi-cycle stride-to-stride variability module
   Training samples K consecutive cycles (K random in 2..k_max) from the same walk.
   Each cycle is embedded, and the walk is described by the
     mean, standard deviation, and mean successive difference |e_k+1 - e_k|
   of its cycle embeddings: clinical stride-to-stride variability
   (e.g. irregular Parkinsonian strides vs. stereotyped spastic diplegic strides).
   A session head classifies the walk from these statistics; at test time it sees
   all cycles of a walk.
3. Ablation switches: --no_severity, --no_multicycle, --no_sym (half-cycle symmetry)
4. Per-fold test predictions saved to .npz (for ensembling 3D + 2D or seeds offline)
5. --split_seed fixes the folds independently of --seed (training seed)
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

import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
PARENT = os.path.dirname(HERE)
V4_DIR = os.path.join(PARENT, 'anatophasenet_v4')
sys.path.insert(0, V4_DIR)

from anato_phasenet_v4 import (  # noqa: E402
    NUM_PHASES, compute_macro_f1, confusion_matrix, plot_fold_curves, plot_confusion_matrix_chart,
    load_cache, gather_phases, gpu_augment_phases, make_sampler_weights, FocalCrossEntropyLoss,
    R2Plus1DBackbone, TSMBackbone, AnatoPhaseNet_v4, EMA, make_strict_person_folds, is_better,
)

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


class AnatoPhaseNet_v5(nn.Module):
    def __init__(self, num_classes, pretrained_path=None, backbone='r2plus1d', num_phases=9, num_parts=6,
                 last_stride=2, head_dim=128, dropout=0.3, use_sym=True, num_severity=3, emb_dim=256):
        super().__init__()
        if backbone == 'r2plus1d':
            self.backbone = R2Plus1DBackbone(num_phases, last_stride)
        else:
            self.backbone = TSMBackbone(num_phases, last_stride)

        self.head = PartPhaseHead_v5(256, num_classes, num_parts=num_parts, num_phases=num_phases,
                                     dim=head_dim, dropout=dropout, use_sym=use_sym)
        fd = self.head.feat_dim

        # Severity head (multi-task)
        self.severity_head = nn.Sequential(nn.Dropout(dropout), nn.Linear(fd, num_severity))

        # Cycle embedding + walk-level (multi-cycle) classifier on variability statistics
        self.embed = nn.Sequential(nn.Dropout(dropout), nn.Linear(fd, emb_dim), nn.ReLU(inplace=True))
        self.session_head = nn.Sequential(
            nn.LayerNorm(3 * emb_dim),
            nn.Dropout(dropout),
            nn.Linear(3 * emb_dim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_classes),
        )

        if pretrained_path and os.path.isfile(pretrained_path):
            AnatoPhaseNet_v4._load_pretrained(self, pretrained_path)

    def backbone_parameters(self):
        return list(self.backbone.parameters())

    def head_parameters(self):
        return [p for n, p in self.named_parameters() if not n.startswith('backbone.')]

    def encode(self, x):
        # x: [B, 9, H, W] in [0, 1]
        sil = (x - 0.5) / 0.5
        motion = x - x.roll(1, dims=1)
        dev = x - x.mean(dim=1, keepdim=True)
        inp = torch.stack([sil, motion, dev], dim=1)        # [B, 3, T, H, W]

        part_logits, g = self.head(self.backbone(inp))
        return {
            'logits': part_logits.mean(dim=1),
            'part_logits': part_logits,
            'severity': self.severity_head(g),
            'emb': self.embed(g),
        }

    def session_logits(self, emb, mask):
        return self.session_head(session_statistics(emb.float(), mask.float()))

    def forward(self, x):
        return self.encode(x)['logits']

# =======================================================================
# 3. EVALUATION
# =======================================================================
@torch.no_grad()
def evaluate_v5(model, data, labels, sev_t, seq_ids, idx, device, num_classes, use_amp=True, tta=True,
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
            with torch.autocast('cuda', dtype=torch.float16, enabled=use_amp):
                o = model.encode(x)
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

    # Severity (cycles with a label only)
    sv = sev_targets >= 0
    res['sev_acc'] = float((sev_logits[sv].argmax(-1) == sev_targets[sv]).float().mean() * 100.0) if sv.any() else float('nan')

    # Sessions (walks): positions in chronological order (idx is ascending)
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

    parser = argparse.ArgumentParser(description="AnatoPhaseNet v5: severity-aware multi-cycle gait classifier")
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
    parser.add_argument('--select', type=str, default='last', choices=['acc', 'f1', 'loss', 'last'])
    parser.add_argument('--no_tta', action='store_true')
    parser.add_argument('--folds', type=int, default=10)
    parser.add_argument('--run_folds', type=int, default=10)
    parser.add_argument('--fold_ids', type=int, nargs='*', default=None)
    parser.add_argument('--seed', type=int, default=42, help='Training seed')
    parser.add_argument('--split_seed', type=int, default=42, help='Fold split seed (keep 42 to match v3/v4)')
    parser.add_argument('--save_models', action='store_true')
    args = parser.parse_args()

    args.dataset_dir = args.dataset_dir or default_dirs[args.dataset]
    tag_ds = args.dataset if args.dataset == 'gait_it' else f"init{args.init_classes}"

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    use_amp = (device.type == 'cuda')

    # Cache shared with v4
    cache_file = os.path.join(
        V4_DIR, 'cache', f"{tag_ds}_{args.img_size}_j{args.jitter}{'_nocanon' if args.no_canon else ''}.pt")
    d = load_cache(args, cache_file)
    data, labels, persons, seq_ids = d['data'], d['labels'], d['persons'], d['seq_ids']
    class_names = d['class_names']
    num_classes = len(class_names)
    labels_np = labels.numpy()

    sev_t = severity_labels(seq_ids, args.dataset, args.init_classes)
    use_severity = (not args.no_severity) and bool((sev_t >= 0).any())
    use_mc = not args.no_multicycle
    use_sym = not args.no_sym

    variant = [n for n, off in (('nosev', not use_severity and not (args.dataset == 'init' and args.init_classes == 8)),
                                ('nomc', not use_mc), ('nosym', not use_sym)) if off]
    variant = '_'.join(variant) or 'full'
    seed_tag = '' if args.seed == 42 else f"_seed{args.seed}"
    args.out_dir = args.out_dir or os.path.join(HERE, f"results_{tag_ds}_{args.backbone}_{variant}{seed_tag}")
    os.makedirs(args.out_dir, exist_ok=True)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    print("=" * 75)
    print(f"  AnatoPhaseNet v5 | dataset: {tag_ds} | backbone: {args.backbone} | variant: {variant}")
    print(f"  Severity task: {use_severity} | Multi-cycle variability: {use_mc} (k_max={args.k_max}) | "
          f"Symmetry: {use_sym} | Device: {device}")
    print("=" * 75)
    print(f"[INFO] Classes: {class_names} | cycles per class: {np.bincount(labels_np, minlength=num_classes).tolist()}")
    if use_severity:
        print(f"[INFO] Severity labels {SEVERITY_NAMES}: {[int((sev_t == s).sum()) for s in range(3)]} "
              f"(unlabelled: {int((sev_t < 0).sum())})")

    model_kw = dict(backbone=args.backbone, num_parts=args.num_parts, last_stride=args.last_stride,
                    head_dim=args.head_dim, dropout=args.dropout, use_sym=use_sym)
    model_tmp = AnatoPhaseNet_v5(num_classes, None, **model_kw)
    param_count = sum(p.numel() for p in model_tmp.parameters() if p.requires_grad)
    print(f"[INFO] Architecture: AnatoPhaseNet v5 ({args.backbone}) | Parameters: {param_count / 1e6:.3f}M")
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

        model = AnatoPhaseNet_v5(num_classes, args.pretrained_path, **model_kw).to(device)
        eval_model = AnatoPhaseNet_v5(num_classes, None, **model_kw).to(device)
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

        # Cycle sampler (single-cycle mode) and walk sampler (multi-cycle mode)
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
            f.write("Epoch,Train_Loss,Train_Acc,Val_Loss,Val_Cycle_Acc,Val_Session_Acc,Val_Session_Vote_Acc,Val_Macro_F1,Val_Severity_Acc\n")

        for ep in range(args.epochs):
            model.train()
            perm = None
            if not use_mc:
                if samp_w is None:
                    perm = np.random.permutation(tr_idx)
                else:
                    perm = np.random.choice(tr_idx, size=steps * args.batch_size, replace=True, p=samp_w)
            ep_loss, n_seen, t_correct = 0.0, 0, 0

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

                x = gather_phases(data, flat, train=True).to(device).float() / 255.0
                x = gpu_augment_phases(x)
                y = labels[flat].to(device)
                sv = sev_t[flat].to(device)
                mask = torch.from_numpy(slot_mask).to(device)
                valid = mask.reshape(-1) > 0

                with torch.autocast('cuda', dtype=torch.float16, enabled=use_amp):
                    o = model.encode(x)

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

                opt.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(model.parameters(), 2.0)
                scaler.step(opt)
                scaler.update()
                sched.step()

                ema.update(model)
                ep_loss += loss.item() * len(yv)
                n_seen += len(yv)
                t_correct += (out.argmax(dim=-1) == yv).sum().item()

            train_loss = ep_loss / max(1, n_seen)
            train_acc = 100.0 * t_correct / max(1, n_seen)

            ema.copy_to(eval_model)
            v = evaluate_v5(eval_model, data, labels, sev_t, seq_ids, va_idx, device, num_classes,
                            use_amp, tta=not args.no_tta, use_session_head=use_mc)

            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['val_loss'].append(v['loss'])
            history['val_cycle_acc'].append(v['cycle_acc'])
            history['val_sess_acc'].append(v['sess_acc'])
            history['val_f1'].append(v['cycle_f1'])

            with open(csv_path, 'a', encoding='utf-8') as f:
                f.write(f"{ep+1},{train_loss:.4f},{train_acc:.2f},{v['loss']:.4f},{v['cycle_acc']:.2f},"
                        f"{v['sess_acc']:.2f},{v['sess_vote_acc']:.2f},{v['cycle_f1']:.4f},{v['sev_acc']:.2f}\n")

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
        t = evaluate_v5(eval_model, data, labels, sev_t, seq_ids, te_idx, device, num_classes,
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
    print(f"        ANATOPHASENET V5 FINAL BENCHMARK SUMMARY ({tag_ds}, {args.backbone}, {variant})")
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
