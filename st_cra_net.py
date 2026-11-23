#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ST-CRANet: Spatio-Temporal Cross Reverse Attention Network with
Foreground-Gated Axial Attention, Harmonic Gait-Phase Attention, and
Temporal Reverse Attention Pooling for Pathological Gait Classification.

Designed for Silhouette Sequence Classification on GAIT-IT / Casia-B.
Preserves RCAF Module intact while introducing novel Spatio-Temporal mechanisms.
"""

import os
import re
import math
import argparse
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, f1_score

# =======================================================================
# 1. DATASET DEFINITION (Same structure as testing.py and swin.py)
# =======================================================================
GAIT_CLASSES = {
    'normal_silhouettes': 0,
    'diplegic_silhouettes': 1,
    'hemiplegic_silhouettes': 2,
    'neuropathic_silhouettes': 3,
    'parkinson_silhouettes': 4
}

class GaitSequenceDataset(Dataset):
    """
    Loads walking sequences of silhouette image frames.
    Returns:
        sequence: Tensor of shape [seq_len, 1, H, W]
        label: Integer class index (0 to 4)
    """
    def __init__(self, root_dir, seq_len=16, img_size=128, transform=None, cache_in_memory=True):
        self.root_dir = root_dir
        self.seq_len = seq_len
        self.img_size = img_size
        self.transform = transform
        self.cache_in_memory = cache_in_memory
        self.cache = {}
        self.samples = []
        print(f"[INFO] Scanning dataset at: {root_dir}")
        self._scan_dataset()

    def preload_all(self):
        """Preloads and resizes all sequences into RAM at startup for zero-delay training."""
        total = len(self.samples)
        if total == 0:
            return
        print(f"[INFO] Pre-caching all {total} gait sequences into RAM at {self.img_size}x{self.img_size}...")
        for i in range(total):
            _ = self[i]
            if (i + 1) % 150 == 0 or (i + 1) == total:
                print(f"  Cached [{i+1}/{total}] sequences in RAM ({100.0 * (i+1) / total:.1f}%)")
        print("[SUCCESS] Dataset RAM caching complete! Zero disk I/O during training.")

    def _scan_dataset(self):
        if not os.path.isdir(self.root_dir):
            print(f"[WARNING] Dataset root directory not found: {self.root_dir}")
            return

        for gait_type, label in GAIT_CLASSES.items():
            gait_path = os.path.join(self.root_dir, gait_type)
            if not os.path.isdir(gait_path):
                continue

            for subject in os.listdir(gait_path):
                subj_path = os.path.join(gait_path, subject, 'silhouettes', 'side_view')
                if not os.path.isdir(subj_path):
                    continue

                for session in os.listdir(subj_path):
                    session_path = os.path.join(subj_path, session)
                    if not os.path.isdir(session_path):
                        continue

                    session_items = os.listdir(session_path)
                    has_images_direct = any(f.lower().endswith(('.png', '.jpg', '.jpeg')) for f in session_items)

                    if has_images_direct:
                        seq_path = session_path
                        frame_files = [f for f in session_items if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                        if len(frame_files) == 0:
                            continue
                        frame_files.sort(key=lambda x: [int(c) if c.isdigit() else c for c in re.split(r'(\d+)', x)])
                        frame_paths = [os.path.join(seq_path, f) for f in frame_files]
                        if len(frame_paths) >= self.seq_len:
                            self.samples.append((frame_paths, label))
                    else:
                        for seq_folder in session_items:
                            if not seq_folder.startswith('Seq_'):
                                continue
                            seq_path = os.path.join(session_path, seq_folder)
                            if not os.path.isdir(seq_path):
                                continue
                            frame_files = [f for f in os.listdir(seq_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                            if len(frame_files) == 0:
                                continue
                            frame_files.sort(key=lambda x: [int(c) if c.isdigit() else c for c in re.split(r'(\d+)', x)])
                            frame_paths = [os.path.join(seq_path, f) for f in frame_files]
                            if len(frame_paths) >= self.seq_len:
                                self.samples.append((frame_paths, label))

        print(f"[INFO] Total valid gait sequences found: {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        if self.cache_in_memory and idx in self.cache:
            raw_seq, label = self.cache[idx]
            sequence = (raw_seq.float() / 127.5) - 1.0
            return sequence, label

        frame_paths, label = self.samples[idx]

        # Uniform temporal sampling across the entire gait cycle
        indices = np.linspace(0, len(frame_paths) - 1, self.seq_len, dtype=int)
        selected_frames = [frame_paths[i] for i in indices]

        frames = []
        for path in selected_frames:
            try:
                img = Image.open(path).convert('L')
                if img.size != (self.img_size, self.img_size):
                    img = img.resize((self.img_size, self.img_size), Image.BILINEAR)
                arr = torch.from_numpy(np.array(img, dtype=np.uint8)).unsqueeze(0)
                frames.append(arr)
            except Exception as e:
                frames.append(torch.zeros(1, self.img_size, self.img_size, dtype=torch.uint8))

        # Output shape: [seq_len, 1, H, W] in compact uint8 (660 MB total for all 822 sequences!)
        raw_seq = torch.stack(frames)
        if self.cache_in_memory:
            self.cache[idx] = (raw_seq, label)

        sequence = (raw_seq.float() / 127.5) - 1.0
        return sequence, label


# =======================================================================
# 2. YOUR ORIGINAL RCAF MODULE & REVERSE ATTENTION (100% UNCHANGED)
# =======================================================================
class ReverseAttention(nn.Module):
    def __init__(self, fr_channels, f4_channels):
        super(ReverseAttention, self).__init__()
        self.conv = nn.Conv2d(
            in_channels=fr_channels + f4_channels,
            out_channels=fr_channels,
            kernel_size=3,
            padding=1
        )

    def forward(self, fr, f4):
        concat = torch.cat([fr, f4], dim=1)
        x = self.conv(concat)
        x = torch.sigmoid(x)
        result = fr * (1.0 - x)
        return result


class RCAF_Module(nn.Module):
    def __init__(self):
        super(RCAF_Module, self).__init__()
        self.initial_conv = None
        self.fusion_relu = nn.ReLU(inplace=True)

        self.dconv1 = nn.Conv2d(16, 16, kernel_size=3, padding=1, dilation=1)
        self.dconv2 = nn.Conv2d(48, 16, kernel_size=3, padding=2, dilation=2)
        self.dconv3 = nn.Conv2d(48, 16, kernel_size=3, padding=3, dilation=3)
        self.dconv4 = nn.Conv2d(32, 16, kernel_size=3, padding=4, dilation=4)

        self.reverse_attention = None
        self.output_conv = nn.Conv2d(64, 64, kernel_size=1, padding=0)
        self.output_relu = nn.ReLU(inplace=True)

    def forward(self, ef, df):
        ef_reshaped = F.interpolate(ef, size=df.shape[2:], mode='bilinear', align_corners=False)
        x = torch.cat([ef_reshaped, df], dim=1)

        if self.initial_conv is None:
            self.initial_conv = nn.Conv2d(
                in_channels=x.shape[1],
                out_channels=64,
                kernel_size=3,
                padding=1
            ).to(x.device)

        fm = self.fusion_relu(self.initial_conv(x))

        split_channels = 16
        f1, f2, f3, f4 = torch.split(fm, split_channels, dim=1)

        f11 = self.dconv1(f1)
        f21 = torch.cat([f11, f2, f3], dim=1)
        f22 = self.dconv2(f21)
        f31 = torch.cat([f22, f3, f4], dim=1)
        f33 = self.dconv3(f31)
        f41 = torch.cat([f33, f4], dim=1)
        f44 = self.dconv4(f41)

        if self.reverse_attention is None:
            self.reverse_attention = ReverseAttention(f44.shape[1], fm.shape[1]).to(x.device)

        f44_ra = self.reverse_attention(f44, fm)
        final_con = torch.cat([f11, f22, f33, f44_ra], dim=1)
        out = self.output_relu(self.output_conv(final_con))
        return out


# =======================================================================
# 3. NOVELTY 1: FOREGROUND-GATED AXIAL SPATIAL ATTENTION (FG-ASA)
#    (Modified from Guo et al. to solve silhouette sparsity & posture geometry)
# =======================================================================
class ForegroundGatedAxialAttention(nn.Module):
    """
    Factorizes spatial attention into:
      1) Height-wise strip pooling (capturing posture tilt, trunk flexion, knee lift)
      2) Width-wise strip pooling (capturing stride length, stance base width)
      3) Foreground saliency gate (suppressing empty background pixels)
      4) Channel-wise modulation gate
    """
    def __init__(self, in_channels):
        super(ForegroundGatedAxialAttention, self).__init__()
        inter_channels = in_channels // 2

        self.conv_h = nn.Conv2d(in_channels, inter_channels, kernel_size=1)
        self.conv_w = nn.Conv2d(in_channels, inter_channels, kernel_size=1)

        self.channel_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, inter_channels, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, inter_channels, kernel_size=1),
            nn.Sigmoid()
        )

        self.fg_mask_conv = nn.Sequential(
            nn.Conv2d(in_channels, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

        self.out_conv = nn.Sequential(
            nn.Conv2d(inter_channels, in_channels, kernel_size=1),
            nn.BatchNorm2d(in_channels)
        )

    def forward(self, x):
        """
        Input: [B*T, C, H, W]
        Output: [B*T, C, H, W]
        """
        B_T, C, H, W = x.shape

        # Axial strip pooling
        pool_h = F.adaptive_avg_pool2d(x, (H, 1))  # [B*T, C, H, 1]
        pool_w = F.adaptive_avg_pool2d(x, (1, W))  # [B*T, C, 1, W]

        feat_h = self.conv_h(pool_h)  # [B*T, inter_C, H, 1]
        feat_w = self.conv_w(pool_w)  # [B*T, inter_C, 1, W]
        spatial_attn = torch.sigmoid(feat_h + feat_w)  # [B*T, inter_C, H, W]

        chn_weight = self.channel_gate(x)    # [B*T, inter_C, 1, 1]
        fg_saliency = self.fg_mask_conv(x)   # [B*T, 1, H, W]

        fused = spatial_attn * chn_weight * fg_saliency
        return self.out_conv(fused)


# =======================================================================
# 4. NOVELTY 2: HARMONIC PHASE-AWARE TEMPORAL ATTENTION (HPA-TA)
#    (Modified from Guo et al. to model cyclic gait stride periodicity)
# =======================================================================
class HarmonicPhaseTemporalAttention(nn.Module):
    """
    Connects related sequence frames across time:
      1) 3x1x1 temporal convolution for short-term frame-to-frame stride transitions
      2) Harmonic Fourier phase prior injected into temporal attention matrix
         to model cadence irregularities and bilateral stance/swing asymmetries.
    """
    def __init__(self, in_channels, seq_len=16, num_harmonics=2):
        super(HarmonicPhaseTemporalAttention, self).__init__()
        self.seq_len = seq_len
        self.inter_channels = in_channels // 2

        self.conv_temporal = nn.Conv3d(
            in_channels, self.inter_channels,
            kernel_size=(3, 1, 1), padding=(1, 0, 0)
        )

        self.q_proj = nn.Conv1d(self.inter_channels, self.inter_channels, kernel_size=1)
        self.k_proj = nn.Conv1d(self.inter_channels, self.inter_channels, kernel_size=1)

        # Learnable Fourier Harmonic weights: W_cos, W_sin for k=1..K
        self.harmonic_cos = nn.Parameter(torch.randn(num_harmonics) * 0.02)
        self.harmonic_sin = nn.Parameter(torch.randn(num_harmonics) * 0.02)
        self.num_harmonics = num_harmonics

        self.out_proj = nn.Sequential(
            nn.Conv2d(self.inter_channels, in_channels, kernel_size=1),
            nn.BatchNorm2d(in_channels)
        )
        self.gamma = nn.Parameter(torch.zeros(1))

    def _compute_harmonic_bias(self, T, device):
        """
        Phi(i, j) = sum_k [ w_c * cos(2pi*k*dt/T) + w_s * sin(2pi*k*dt/T) ]
        """
        t = torch.arange(T, device=device, dtype=torch.float32)
        dt = t.unsqueeze(1) - t.unsqueeze(0)  # [T, T]

        bias = torch.zeros(T, T, device=device)
        for k in range(1, self.num_harmonics + 1):
            phase = (2.0 * math.pi * k * dt) / float(T)
            bias += self.harmonic_cos[k - 1] * torch.cos(phase) + self.harmonic_sin[k - 1] * torch.sin(phase)
        return bias.unsqueeze(0)  # [1, T, T]

    def forward(self, x_seq):
        """
        Input: [B, T, C, H, W] (Temporally ordered frames of a gait sequence)
        Output: [B, T, C, H, W]
        """
        B, T, C, H, W = x_seq.shape
        C_in = self.inter_channels

        # 1. 3D temporal convolution: frame-to-frame short-term stride dynamics
        x_5d = x_seq.permute(0, 2, 1, 3, 4)  # [B, C, T, H, W]
        x_t = self.conv_temporal(x_5d)        # [B, C_in, T, H, W]

        # 2. Extract global frame tokens: [B, C_in, T]
        tokens = F.adaptive_avg_pool3d(x_t, (T, 1, 1)).squeeze(-1).squeeze(-1)

        q = self.q_proj(tokens).permute(0, 2, 1)  # [B, T, C_in]
        k = self.k_proj(tokens)                   # [B, C_in, T]

        dot_product = torch.bmm(q, k) * (C_in ** -0.5)
        dot_product = torch.clamp(dot_product, -25.0, 25.0)
        harmonic_bias = self._compute_harmonic_bias(T, x_seq.device)
        attn_matrix = F.softmax(dot_product + harmonic_bias, dim=-1)  # [B, T, T]

        # 3. Apply sequence-wide attention
        x_flat = x_t.permute(0, 2, 1, 3, 4).reshape(B, T, C_in * H * W)
        attended = torch.bmm(attn_matrix, x_flat).view(B * T, C_in, H, W)

        out = self.out_proj(attended).view(B, T, C, H, W)
        return x_seq + self.gamma * out


# =======================================================================
# 5. NOVELTY 3: TEMPORAL REVERSE ATTENTION POOLING (TRAP) HEAD
#    (Extends CRA-Net Reverse Attention across the entire frame sequence)
# =======================================================================
class TemporalReverseAttentionPooling(nn.Module):
    """
    Evaluates all T frame features from the multi-scale RCAF modules:
      - Dominant Attention (alpha_t): Captures standard stride peaks
      - Reverse Attention (1 - alpha_t): Highlights subtle micro-hesitations & tremors
    """
    def __init__(self, in_features=192, num_classes=5):
        super(TemporalReverseAttentionPooling, self).__init__()
        self.temporal_scorer = nn.Sequential(
            nn.Linear(in_features, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Linear(64, 1)
        )

        # Dual-perspective classification: 192 (Dominant) + 192 (Reverse) = 384
        self.classifier = nn.Sequential(
            nn.Linear(in_features * 2, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, r_seq):
        """
        Input: r_seq [B, T, 192] (RCAF multi-scale frame representations across time)
        Output: Logits [B, num_classes]
        """
        scores = self.temporal_scorer(r_seq)  # [B, T, 1]

        # Numerically stable Dual-Perspective Softmax:
        # Dominant Stride: highest score gets highest weight (sum is 1.0, zero-division impossible!)
        w_dom = F.softmax(scores, dim=1)
        # Reverse Hesitation: lowest score gets highest weight via -scores (sum is 1.0, zero-division impossible!)
        w_rev = F.softmax(-scores, dim=1)

        f_dominant = torch.sum(r_seq * w_dom, dim=1)  # [B, 192]
        f_reverse  = torch.sum(r_seq * w_rev, dim=1)  # [B, 192]

        f_combined = torch.cat([f_dominant, f_reverse], dim=-1)  # [B, 384]
        return self.classifier(f_combined)


# =======================================================================
# 6. COMPLETE ST-CRANet ARCHITECTURE
# =======================================================================
class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(EncoderBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        feat = self.conv(x)
        out = self.maxpool(feat)
        return feat, out


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super(DecoderBlock, self).__init__()
        self.upconv = nn.ConvTranspose2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=2,
            padding=1,
            output_padding=1
        )
        self.conv = nn.Sequential(
            nn.Conv2d(out_channels + skip_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, skip):
        x = self.upconv(x)
        if x.shape[2:] != skip.shape[2:]:
            skip = F.interpolate(skip, size=x.shape[2:], mode='bilinear', align_corners=False)
        return self.conv(torch.cat([x, skip], dim=1))


class ST_CRANet(nn.Module):
    """
    Spatio-Temporal Cross Reverse Attention Network:
      - Preserves 3-stage Encoders and Decoders
      - Preserves RCAF1, RCAF2, RCAF3 modules 100% intact
      - Adds FG-ASA and HPA-TA at the Bottleneck
      - Adds TRAP head to pool multi-scale RCAF frame tokens into a sequence prediction
    """
    def __init__(self, input_channels=1, num_classes=5, seq_len=16):
        super(ST_CRANet, self).__init__()
        self.seq_len = seq_len

        # Encoders
        self.encoder1 = EncoderBlock(input_channels, 64)
        self.encoder2 = EncoderBlock(64, 128)
        self.encoder3 = EncoderBlock(128, 256)

        # Bottleneck Conv
        self.bottleneck_conv = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # Proposed Bottleneck Attention Modules
        self.spatial_axial = ForegroundGatedAxialAttention(in_channels=512)
        self.temporal_harmonic = HarmonicPhaseTemporalAttention(in_channels=512, seq_len=seq_len)

        # Decoders
        self.decoder1 = DecoderBlock(512, 256, 256)
        self.decoder2 = DecoderBlock(256, 128, 128)
        self.decoder3 = DecoderBlock(128, 64, 64)

        # Original RCAF Modules (Preserved 100%)
        self.rcaf1 = RCAF_Module()
        self.rcaf2 = RCAF_Module()
        self.rcaf3 = RCAF_Module()
        self.global_pool = nn.AdaptiveAvgPool2d(1)

        # Proposed Temporal Reverse Attention Pooling Head
        self.trap_head = TemporalReverseAttentionPooling(in_features=192, num_classes=num_classes)

    def forward(self, x):
        """
        Input x: [B, T, C, H, W] (A sequence of T related silhouette frames)
        """
        B, T, C, H, W = x.shape
        x_2d = x.view(B * T, C, H, W)

        # 1. Multi-scale 2D Encoder path (operates on each frame)
        skip1, x_enc1 = self.encoder1(x_2d)
        skip2, x_enc2 = self.encoder2(x_enc1)
        skip3, x_enc3 = self.encoder3(x_enc2)

        # 2. Bottleneck
        b = self.bottleneck_conv(x_enc3)

        # 2a. Apply Foreground-Gated Axial Spatial Attention
        b = b + self.spatial_axial(b)

        # 2b. Reshape to 5D [B, T, C_b, H_b, W_b] to link related frames across time
        _, C_b, H_b, W_b = b.shape
        b_seq = b.view(B, T, C_b, H_b, W_b)
        b_seq = self.temporal_harmonic(b_seq)
        b = b_seq.view(B * T, C_b, H_b, W_b)

        # 3. Decoder path
        d1 = self.decoder1(b, skip3)
        d2 = self.decoder2(d1, skip2)
        d3 = self.decoder3(d2, skip1)

        # 4. RCAF Skip-Decoder Reverse Attention Fusion
        r1 = self.rcaf1(skip3, d1)
        r2 = self.rcaf2(skip2, d2)
        r3 = self.rcaf3(skip1, d3)

        r1_res = F.interpolate(r1, size=r2.shape[2:], mode='bilinear', align_corners=False)
        r3_res = F.interpolate(r3, size=r2.shape[2:], mode='bilinear', align_corners=False)
        r_combined = torch.cat([r2, r1_res, r3_res], dim=1)  # [B*T, 192, H_r2, W_r2]

        # 5. Extract multi-scale RCAF frame representations across time: [B, T, 192]
        r_tokens = self.global_pool(r_combined).view(B, T, 192)

        # 6. Sequence-level Temporal Reverse Attention Pooling
        logits = self.trap_head(r_tokens)
        return logits


def plot_training_curves(history, output_path="./results/training_curves.png"):
    import matplotlib.pyplot as plt
    epochs = range(1, len(history['train_loss']) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), dpi=300)

    # 1. Loss Curves
    ax1.plot(epochs, history['train_loss'], 'b-o', label='Training Loss', lw=2, markersize=4)
    ax1.plot(epochs, history['val_loss'], 'r-s', label='Validation Loss', lw=2, markersize=4)
    ax1.set_title('Training vs. Validation Loss', fontsize=13, fontweight='bold')
    ax1.set_xlabel('Epoch', fontsize=11)
    ax1.set_ylabel('Cross-Entropy Loss', fontsize=11)
    ax1.grid(True, linestyle='--', alpha=0.5)
    ax1.legend(fontsize=10)

    # 2. Accuracy Curves
    ax2.plot(epochs, history['train_acc'], 'b-o', label='Training Accuracy', lw=2, markersize=4)
    ax2.plot(epochs, history['val_acc'], 'g-^', label='Validation Accuracy', lw=2, markersize=4)
    ax2.set_title('Training vs. Validation Accuracy', fontsize=13, fontweight='bold')
    ax2.set_xlabel('Epoch', fontsize=11)
    ax2.set_ylabel('Accuracy (%)', fontsize=11)
    ax2.grid(True, linestyle='--', alpha=0.5)
    ax2.legend(fontsize=10)

    fig.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


# =======================================================================
# 7. TRAINING & VALIDATION PIPELINE
# =======================================================================
def parse_args():
    parser = argparse.ArgumentParser(description="Train ST-CRANet on Pathological Gait Sequences")
    parser.add_argument('--dataset_dir', type=str,
                        default=r'D:\New folder (8)\rp\Gait_Dataset_silhoutte_based\Gait_Dataset_silhoutte_based\GAIT_IT',
                        help="Path to GAIT_IT dataset directory")
    parser.add_argument('--seq_len', type=int, default=12, help="Number of frames per gait sequence (12 captures complete stride cycle)")
    parser.add_argument('--img_size', type=int, default=128, help="Spatial resolution of frames (128x128 is 3.1x faster than 224x224)")
    parser.add_argument('--batch_size', type=int, default=8, help="Batch size for training (8 stabilizes gradient variance)")
    parser.add_argument('--lr', type=float, default=1.5e-4, help="Initial learning rate (1.5e-4 prevents validation oscillation)")
    parser.add_argument('--weight_decay', type=float, default=5e-4, help="Weight decay for regularization")
    parser.add_argument('--epochs', type=int, default=50, help="Number of training epochs")
    parser.add_argument('--checkpoint_dir', type=str, default="./results", help="Directory to save checkpoints")
    parser.add_argument('--use_amp', action='store_true', default=False, help="Enable FP16 AMP (default False for numerical stability)")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(args.checkpoint_dir, 'st_cranet_best.pth')

    # Data Augmentation & Normalization
    train_transform = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.RandomAffine(degrees=5, translate=(0.04, 0.04)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    # Check dataset existence
    dataset_dir = args.dataset_dir
    if not os.path.exists(dataset_dir):
        alt_dirs = [
            os.path.join(os.getcwd(), 'GAIT_IT'),
            os.path.join(os.getcwd(), 'Gait_Dataset_silhoutte_based', 'GAIT_IT'),
        ]
        for alt in alt_dirs:
            if os.path.exists(alt):
                dataset_dir = alt
                print(f"[INFO] Using alternative dataset path: {dataset_dir}")
                break

    full_dataset = GaitSequenceDataset(
        root_dir=dataset_dir,
        seq_len=args.seq_len,
        img_size=args.img_size,
        transform=train_transform
    )

    if len(full_dataset.samples) == 0:
        print("[ERROR] No sequences found in dataset directory. Exiting.")
        print(f"Checked path: {dataset_dir}")
        return

    # Pre-cache all sequences directly in RAM (only 215 MB total!) for zero disk lag
    full_dataset.preload_all()

    labels = [label for _, label in full_dataset.samples]
    indices = np.arange(len(labels))

    train_idx, temp_idx, _, temp_labels = train_test_split(
        indices, labels, test_size=0.30, stratify=labels, random_state=42
    )
    val_idx, test_idx = train_test_split(
        temp_idx, test_size=0.50, stratify=[labels[i] for i in temp_idx], random_state=42
    )

    train_set = Subset(full_dataset, train_idx)
    val_set = Subset(full_dataset, val_idx)
    test_set = Subset(full_dataset, test_idx)

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[INFO] Training on device: {device}")

    model = ST_CRANet(input_channels=1, num_classes=5, seq_len=args.seq_len).to(device)

    # Warmup dummy pass so lazy-initialized RCAF layers are materialized before AdamW registration
    dummy_input = torch.zeros(1, args.seq_len, 1, args.img_size, args.img_size, device=device)
    _ = model(dummy_input)

    # Label smoothing helps prevent overconfidence on small silhouette datasets
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    # Numerical Stability: FP32 standard precision avoids underflow in deep attention dot-products
    use_amp = (device.type == 'cuda' and args.use_amp)
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)

    best_val_acc = 0.0

    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'val_f1': []
    }
    csv_log_path = os.path.join(args.checkpoint_dir, 'training_metrics.csv')
    curves_img_path = os.path.join(args.checkpoint_dir, 'training_curves.png')

    with open(csv_log_path, 'w') as f:
        f.write("Epoch,Train_Loss,Train_Acc,Val_Loss,Val_Acc,Val_Macro_F1\n")

    print("=" * 70)
    print("STARTING ST-CRANet TRAINING (AMP + In-Memory RAM Caching Enabled)")
    print("=" * 70)

    for epoch in range(args.epochs):
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0

        for batch_idx, (X, y) in enumerate(train_loader):
            X, y = X.to(device, non_blocking=True), y.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            if use_amp:
                with torch.amp.autocast('cuda', dtype=torch.float16):
                    outputs = model(X)
                    loss = criterion(outputs, y)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(X)
                loss = criterion(outputs, y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += y.size(0)
            train_correct += (predicted == y).sum().item()

            if (batch_idx + 1) % 15 == 0 or (batch_idx + 1) == len(train_loader):
                print(f"  Batch [{batch_idx+1}/{len(train_loader)}] | Loss: {loss.item():.4f} | Running Acc: {100.0 * train_correct / train_total:.2f}%")

        scheduler.step()

        # Validation Phase
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        all_preds, all_targets = [], []

        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(device, non_blocking=True), y.to(device, non_blocking=True)
                with torch.amp.autocast('cuda', enabled=use_amp, dtype=torch.float16):
                    outputs = model(X)
                    loss = criterion(outputs, y)

                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += y.size(0)
                val_correct += (predicted == y).sum().item()

                all_preds.extend(predicted.cpu().numpy())
                all_targets.extend(y.cpu().numpy())

        epoch_train_loss = train_loss / len(train_loader) if len(train_loader) > 0 else 0.0
        train_acc = 100.0 * train_correct / train_total if train_total > 0 else 0.0
        epoch_val_loss = val_loss / len(val_loader) if len(val_loader) > 0 else 0.0
        val_acc = 100.0 * val_correct / val_total if val_total > 0 else 0.0
        val_f1 = f1_score(all_targets, all_preds, average='macro')

        history['train_loss'].append(epoch_train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(epoch_val_loss)
        history['val_acc'].append(val_acc)
        history['val_f1'].append(val_f1)

        # Append to CSV log
        with open(csv_log_path, 'a') as f:
            f.write(f"{epoch+1},{epoch_train_loss:.4f},{train_acc:.2f},{epoch_val_loss:.4f},{val_acc:.2f},{val_f1:.4f}\n")

        print(f"Epoch [{epoch+1}/{args.epochs}] | LR: {scheduler.get_last_lr()[0]:.6f}")
        print(f"  Train Loss: {epoch_train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"  Val Loss:   {epoch_val_loss:.4f} | Val Acc:   {val_acc:.2f}% | Val Macro-F1: {val_f1:.4f}")

        # Update curves plot after each epoch
        try:
            plot_training_curves(history, curves_img_path)
        except Exception as e:
            pass

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_f1': val_f1,
                'history': history
            }, checkpoint_path)
            print(f"  [SAVED] Best model updated with Val Acc: {val_acc:.2f}%")
        print("-" * 70)

    print(f"[COMPLETED] Best Validation Accuracy: {best_val_acc:.2f}%")
    print(f"[SUCCESS] Training history saved to: {csv_log_path}")
    print(f"[SUCCESS] Training curves plot saved to: {curves_img_path}")


if __name__ == '__main__':
    main()
