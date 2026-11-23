import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, f1_score
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import transforms

from st_cra_net import GaitSequenceDataset, ST_CRANet, GAIT_CLASSES

def plot_confusion_matrix(cm, class_names, output_path="confusion_matrix.png"):
    fig, ax = plt.subplots(figsize=(8, 7), dpi=300)
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=class_names, yticklabels=class_names,
           title="ST-CRANet Confusion Matrix (Pathological Gait)",
           ylabel="Ground Truth Label",
           xlabel="Predicted Label")

    plt.setp(ax.get_xticklabels(), rotation=35, ha="right", rotation_mode="anchor")

    # Format numbers inside cells
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontweight="bold")

    fig.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[SUCCESS] Saved Confusion Matrix to: {output_path}")


def evaluate(checkpoint_path="./results/st_cranet_best.pth", dataset_dir="Gait_Dataset_silhoutte_based/GAIT_IT"):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[INFO] Evaluating on device: {device}")

    if not os.path.exists(checkpoint_path):
        print(f"[ERROR] Checkpoint not found at: {checkpoint_path}")
        print("Wait for training to complete at least one epoch and save best_model.pth!")
        return

    # Transform
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    full_dataset = GaitSequenceDataset(root_dir=dataset_dir, seq_len=16, transform=val_transform)
    labels = [label for _, label in full_dataset.samples]
    indices = np.arange(len(labels))

    train_idx, temp_idx, _, temp_labels = train_test_split(
        indices, labels, test_size=0.30, stratify=labels, random_state=42
    )
    val_idx, test_idx = train_test_split(
        temp_idx, test_size=0.50, stratify=[labels[i] for i in temp_idx], random_state=42
    )

    test_set = Subset(full_dataset, test_idx)
    test_loader = DataLoader(test_set, batch_size=4, shuffle=False, num_workers=2)

    # Load Model
    model = ST_CRANet(input_channels=1, num_classes=5, seq_len=16).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    all_preds, all_targets = [], []

    print("[INFO] Running evaluation on Test Set...")
    with torch.no_grad():
        for X, y in test_loader:
            X, y = X.to(device), y.to(device)
            outputs = model(X)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(y.cpu().numpy())

    class_names = ['Normal', 'Diplegic', 'Hemiplegic', 'Neuropathic', 'Parkinson']
    cm = confusion_matrix(all_targets, all_preds)
    plot_confusion_matrix(cm, class_names, "confusion_matrix.png")

    report = classification_report(all_targets, all_preds, target_names=class_names, digits=4)
    print("\n" + "=" * 60)
    print("FINAL TEST CLASSIFICATION REPORT (FOR PAPER TABLE)")
    print("=" * 60)
    print(report)

    # Save text report
    with open("classification_report.txt", "w") as f:
        f.write("ST-CRANet Pathological Gait Classification Report\n")
        f.write("=" * 60 + "\n")
        f.write(report)
    print("[SUCCESS] Saved classification report table to: classification_report.txt")

    # Plot training and validation curves
    plot_training_and_validation_curves(checkpoint, csv_path="./results/training_metrics.csv", output_path="training_curves.png")


def plot_training_and_validation_curves(checkpoint=None, csv_path="./results/training_metrics.csv", output_path="training_curves.png"):
    train_loss, val_loss = [], []
    train_acc, val_acc = [], []

    # Priority 1: Read from CSV if exists
    if os.path.exists(csv_path):
        import csv
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                train_loss.append(float(row['Train_Loss']))
                train_acc.append(float(row['Train_Acc']))
                val_loss.append(float(row['Val_Loss']))
                val_acc.append(float(row['Val_Acc']))
    # Priority 2: Read from checkpoint history dict
    elif checkpoint and 'history' in checkpoint:
        h = checkpoint['history']
        train_loss = h.get('train_loss', [])
        train_acc = h.get('train_acc', [])
        val_loss = h.get('val_loss', [])
        val_acc = h.get('val_acc', [])

    if len(train_loss) == 0:
        print("[INFO] No epoch loss/accuracy history found yet to plot curves.")
        return

    epochs = range(1, len(train_loss) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), dpi=300)

    # 1. Training vs Validation Loss Curve
    ax1.plot(epochs, train_loss, 'b-o', label='Training Loss', lw=2, markersize=4)
    ax1.plot(epochs, val_loss, 'r-s', label='Validation Loss', lw=2, markersize=4)
    ax1.set_title('Training vs. Validation Loss', fontsize=13, fontweight='bold')
    ax1.set_xlabel('Epoch', fontsize=11)
    ax1.set_ylabel('Cross-Entropy Loss', fontsize=11)
    ax1.grid(True, linestyle='--', alpha=0.5)
    ax1.legend(fontsize=10)

    # 2. Training vs Validation Accuracy Curve
    ax2.plot(epochs, train_acc, 'b-o', label='Training Accuracy', lw=2, markersize=4)
    ax2.plot(epochs, val_acc, 'g-^', label='Validation Accuracy', lw=2, markersize=4)
    ax2.set_title('Training vs. Validation Accuracy', fontsize=13, fontweight='bold')
    ax2.set_xlabel('Epoch', fontsize=11)
    ax2.set_ylabel('Accuracy (%)', fontsize=11)
    ax2.grid(True, linestyle='--', alpha=0.5)
    ax2.legend(fontsize=10)

    fig.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[SUCCESS] Saved Training & Validation Curves to: {output_path}")


if __name__ == '__main__':
    evaluate()

