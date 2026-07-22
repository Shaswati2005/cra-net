#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 26 17:49:40 2025

@author: amit
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 26 17:22:41 2025

@author: amit
"""

import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import numpy as np
import re
import torch
import torch.nn as nn
import torch.nn.functional as F

GAIT_CLASSES = {
    'normal_silhouettes': 0,
    'diplegic_silhouettes': 1,
    'hemiplegic_silhouettes': 2,
    'neuropathic_silhouettes': 3,
    'parkinson_silhouettes': 4
}

class GaitSequenceDataset(Dataset):
    def __init__(self, root_dir, seq_len=20, transform=None):
        self.root_dir = root_dir
        self.seq_len = seq_len
        self.transform = transform
        self.samples = []
        print("[INFO] Scanning dataset...")
        self._scan_dataset()
    
    def _scan_dataset(self):
        for gait_type, label in GAIT_CLASSES.items():
            gait_path = os.path.join(self.root_dir, gait_type)
            if not os.path.isdir(gait_path):
                print(f"[WARNING] Directory not found: {gait_path}")
                continue
            
            print(f"[INFO] Scanning {gait_type}...")
            
            for subject in os.listdir(gait_path):
                subj_path = os.path.join(gait_path, subject, 'silhouettes', 'side_view')
                if not os.path.isdir(subj_path):
                    print(f"[WARNING] Side view directory not found: {subj_path}")
                    continue
                
                for session in os.listdir(subj_path):
                    session_path = os.path.join(subj_path, session)
                    if not os.path.isdir(session_path):
                        continue

                    # Check session_path contents. If it directly contains image files, treat it as one sequence.
                    session_items = os.listdir(session_path)
                    has_images_direct = any(f.lower().endswith(('.png', '.jpg', '.jpeg')) for f in session_items)

                    if has_images_direct:
                        seq_path = session_path
                        frame_files = [f for f in session_items if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                        if len(frame_files) == 0:
                            print(f"[WARNING] No image files found in: {seq_path}")
                            continue
                        frame_files.sort(key=lambda x: [int(c) if c.isdigit() else c for c in re.split(r'(\d+)', x)])
                        frame_paths = [os.path.join(seq_path, f) for f in frame_files]
                        if len(frame_paths) >= self.seq_len:
                            self.samples.append((frame_paths, label))
                            print(f"[INFO] Added sequence: {seq_path} with {len(frame_paths)} frames")
                        else:
                            print(f"[WARNING] Sequence {seq_path} has only {len(frame_paths)} frames (need {self.seq_len})")
                    else:
                        # Otherwise, look for Seq_ folders inside session_path
                        for seq_folder in session_items:
                            if not seq_folder.startswith('Seq_'):
                                continue
                            seq_path = os.path.join(session_path, seq_folder)
                            if not os.path.isdir(seq_path):
                                continue
                            frame_files = [f for f in os.listdir(seq_path) 
                                         if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                            if len(frame_files) == 0:
                                print(f"[WARNING] No image files found in: {seq_path}")
                                continue
                            frame_files.sort(key=lambda x: [int(c) if c.isdigit() else c for c in re.split(r'(\d+)', x)])
                            frame_paths = [os.path.join(seq_path, f) for f in frame_files]
                            if len(frame_paths) >= self.seq_len:
                                self.samples.append((frame_paths, label))
                                print(f"[INFO] Added sequence: {seq_path} with {len(frame_paths)} frames")
                            else:
                                print(f"[WARNING] Sequence {seq_path} has only {len(frame_paths)} frames (need {self.seq_len})")
        
        print(f"[INFO] Total valid sequences found: {len(self.samples)}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        frame_paths, label = self.samples[idx]
        
        indices = np.linspace(0, len(frame_paths) - 1, self.seq_len, dtype=int)
        selected_frames = [frame_paths[i] for i in indices]
        
        frames = []
        for path in selected_frames:
            try:
                img = Image.open(path).convert('L')
                if self.transform:
                    img = self.transform(img)
                else:
                    img = transforms.ToTensor()(img)
                frames.append(img)
            except Exception as e:
                print(f"[ERROR] Could not load image {path}: {e}")
                if self.transform:
                    img = Image.new('L', (112, 112), 0)
                    img = self.transform(img)
                else:
                    img = torch.zeros(1, 112, 112)
                frames.append(img)
        
        sequence = torch.stack(frames)
        return sequence, label

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
        result = fr * (1 - x)
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


from timm.models.swin_transformer import SwinTransformerBlock

class SwinEncoderBlock(nn.Module):
    def __init__(self, dim, input_resolution, num_heads, window_size=7):
        super(SwinEncoderBlock, self).__init__()
        # The core Swin Block for spatial feature extraction
        self.swin_block = SwinTransformerBlock(
            dim=dim, 
            input_resolution=input_resolution,
            num_heads=num_heads, 
            window_size=window_size,
            shift_size=0, 
            mlp_ratio=4.0, # Keep at 4.0 for better accuracy on CUDA
            qkv_bias=True
        )
        # Downsampling to halving H,W and doubling Channels
        self.downsample = nn.Conv2d(dim, dim * 2, kernel_size=2, stride=2)

    def forward(self, x):
        # DEBUG: Track entry into Swin block
        print(f"[DEBUG SwinBlock] Input shape: {x.shape}")
        B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1).contiguous()
        x = self.swin_block(x) 
        skip = x.permute(0, 3, 1, 2).contiguous()
        out = self.downsample(skip)
        return skip, out
    
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
        
        self.conv = nn.Conv2d(
            in_channels=out_channels + skip_channels,
            out_channels=out_channels,
            kernel_size=3,
            padding=1
        )
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x, skip):
        x = self.upconv(x)
        skip_resized = F.interpolate(skip, size=x.shape[2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, skip_resized], dim=1)
        x = self.conv(x)
        x = self.relu(x)
        return x

class Decoder_testing(nn.Module):
    def __init__(self, input_channels=1, num_classes=5):
        super(Decoder_testing, self).__init__()
        
        # Transformer needs an initial embedding (proj 1-channel to 64)
        self.init_proj = nn.Conv2d(input_channels, 64, kernel_size=3, padding=1)
        
        # Replacing Encoders with Swin Blocks
        self.encoder1 = SwinEncoderBlock(dim=64,  input_resolution=(224, 224), num_heads=2)
        self.encoder2 = SwinEncoderBlock(dim=128, input_resolution=(112, 112), num_heads=4)
        self.encoder3 = SwinEncoderBlock(dim=256, input_resolution=(56, 56),   num_heads=8)
        
        # The rest of your architecture remains UNCHANGED
        self.bottleneck = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.decoder1 = DecoderBlock(512, 256, 256)
        self.decoder2 = DecoderBlock(256, 128, 128)
        self.decoder3 = DecoderBlock(128, 64, 64)
        
        self.rcaf1 = RCAF_Module()
        self.rcaf2 = RCAF_Module()
        self.rcaf3 = RCAF_Module()
        
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(192, num_classes)
    
    def forward(self, x):
        print(f"[DEBUG Model] Input batch shape: {x.shape}") # Track batch entry
        batch_size, temporal_dim, channels, height, width = x.shape
        x = x.view(batch_size * temporal_dim, channels, height, width)
        
        print(f"[DEBUG Model] Reshaped for CNN: {x.shape}")
        x = self.init_proj(x)
        
        print("[DEBUG Model] Starting Encoder Path...")
        skip1, x = self.encoder1(x)
        print(f"  - After Encoder1: {x.shape}")
        skip2, x = self.encoder2(x)
        print(f"  - After Encoder2: {x.shape}")
        skip3, x = self.encoder3(x)
        print(f"  - After Encoder3: {x.shape}")
        
        bottleneck = self.bottleneck(x)
        print(f"[DEBUG Model] Bottleneck out: {bottleneck.shape}")
        
        d1 = self.decoder1(bottleneck, skip3)
        d2 = self.decoder2(d1, skip2)
        d3 = self.decoder3(d2, skip1)
        print(f"[DEBUG Model] Decoder path finished: {d3.shape}")
        
        r1 = self.rcaf1(skip3, d1)
        r2 = self.rcaf2(skip2, d2)
        r3 = self.rcaf3(skip1, d3)
        print("[DEBUG Model] RCAF modules finished")
        
        r1_reshaped = F.interpolate(r1, size=r2.shape[2:], mode='bilinear', align_corners=False)
        r3_reshaped = F.interpolate(r3, size=r2.shape[2:], mode='bilinear', align_corners=False)
        r_combined = torch.cat([r2, r1_reshaped, r3_reshaped], dim=1)
        
        features = self.global_pool(r_combined)
        features = features.view(features.size(0), -1)
        output = self.classifier(features)
        print(f"[DEBUG Model] Forward pass COMPLETE. Output shape: {output.shape}")
        return output

class DataAugmentation:
    def __init__(self):
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomRotation(20),
            transforms.RandomResizedCrop(224, scale=(0.9, 1.1)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5), 
            transforms.ColorJitter(brightness=(0.8, 1.2)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
    
    def __call__(self, x):
        batch_size, temporal_dim, channels, height, width = x.shape
        augmented = torch.zeros_like(x)
        
        for b in range(batch_size):
            for t in range(temporal_dim):
                frame = x[b, t]
                if channels == 1:
                    frame = frame.squeeze(0)
                augmented[b, t] = self.transform(frame).unsqueeze(0) if channels == 1 else self.transform(frame)
        
        return augmented

# Data Loading
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

if __name__ == '__main__':
    image_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    dataset = GaitSequenceDataset(
        root_dir=r'D:\New folder (8)\rp\Gait_Dataset_silhoutte_based\Gait_Dataset_silhoutte_based\GAIT_IT',
        seq_len=16,
        transform=image_transform
    )

    labels = [label for _, label in dataset.samples]
    indices = np.arange(len(labels))

    if len(indices) == 0:
        print('[ERROR] No sequences found in dataset. Exiting.')
    else:
        train_idx, temp_idx, _, temp_labels = train_test_split(
            indices, labels, test_size=0.3, stratify=labels, random_state=42
        )

        val_idx, test_idx = train_test_split(
            temp_idx, test_size=0.5, stratify=[labels[i] for i in temp_idx], random_state=42
        )

        from torch.utils.data import Subset

        train_set = Subset(dataset, train_idx)
        val_set = Subset(dataset, val_idx)
        test_set = Subset(dataset, test_idx)

        train_loader = DataLoader(train_set, batch_size=1, shuffle=True, num_workers=2, pin_memory=True)
        val_loader = DataLoader(val_set, batch_size=1, shuffle=False, num_workers=2, pin_memory=True)
        test_loader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=2, pin_memory=True)

        for X, y in train_loader:
            print(X.shape)
            print(y.shape)
            break


        # Model Setup
        # Clear GPU cache first
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")

        # Try CPU first if CUDA fails
        try:
            model = Decoder_testing(input_channels=1, num_classes=5)
            model = model.to(device)
            print("Model successfully moved to device")
        except RuntimeError as e:
            if "CUDA" in str(e):
                print("CUDA error detected, falling back to CPU")
                device = torch.device('cpu')
                model = Decoder_testing(input_channels=1, num_classes=5)
                model = model.to(device)
            else:
                raise e

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        # Checkpoint Setup
        checkpoint_path = os.path.join(os.getcwd(), 'results', 'best_model.pth')
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)

        # Training
        epochs = 50
        best_val_acc = 0.0

        for epoch in range(epochs):
            model.train()
            # Training Phase
            model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0


    
            for batch_idx, (X, y) in enumerate(train_loader):
                print(f"\n--- Processing Batch {batch_idx+1} ---")
                X, y = X.to(device), y.to(device)
                print(f"[DEBUG Train] Data moved to {device}")

                optimizer.zero_grad()
                print("[DEBUG Train] Starting Forward Pass...")
                outputs = model(X)
                
                # Logic for loss calculation...
                bs, td = X.shape[0], X.shape[1]
                outputs = outputs.view(bs, td, -1).mean(dim=1)
                
                print("[DEBUG Train] Calculating Loss...")
                loss = criterion(outputs, y)
                
                print("[DEBUG Train] Starting Backward Pass...")
                loss.backward()
                
                print("[DEBUG Train] Optimizer Step...")
                optimizer.step()

                train_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                train_total += y.size(0)
                train_correct += (predicted == y).sum().item()

                if batch_idx % 10 == 0:
                    print(f'Epoch [{epoch+1}/{epochs}], Batch [{batch_idx+1}/{len(train_loader)}], Loss: {loss.item():.4f}')

            # Validation Phase
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0

            with torch.no_grad():
                for X, y in val_loader:
                    X, y = X.to(device), y.to(device)
                    outputs = model(X)
                    bs, td = X.shape[0], X.shape[1]
                    outputs = outputs.view(bs, td, -1).mean(dim=1)
                    loss = criterion(outputs, y)

                    val_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    val_total += y.size(0)
                    val_correct += (predicted == y).sum().item()

            train_acc = 100 * train_correct / train_total
            val_acc = 100 * val_correct / val_total

            print(f'Epoch [{epoch+1}/{epochs}]')
            print(f'Train Loss: {train_loss/len(train_loader):.4f}, Train Acc: {train_acc:.2f}%')
            print(f'Val Loss: {val_loss/len(val_loader):.4f}, Val Acc: {val_acc:.2f}%')
            print('-' * 60)

            # Save Best Model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_acc': val_acc
                }, checkpoint_path)
                print(f'Best model saved with validation accuracy: {val_acc:.2f}%')

        print(f'Training completed. Best validation accuracy: {best_val_acc:.2f}%')