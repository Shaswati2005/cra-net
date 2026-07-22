import os
import numpy as np
import torch
from torchvision import transforms
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split

from testing import GaitSequenceDataset, Decoder_testing

# Small transform
image_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

DATA_ROOT = os.path.join(os.getcwd(), 'Gait_Dataset_silhoutte_based', 'Gait_Dataset_silhoutte_based', 'GAIT_IT')

print('Dataset root:', DATA_ROOT)

dataset = GaitSequenceDataset(root_dir=DATA_ROOT, seq_len=16, transform=image_transform)

n = len(dataset)
print('Total sequences found:', n)
if n == 0:
    raise SystemExit('No sequences found; aborting quick test.')

# limit to small subset
max_samples = min(40, n)
indices = np.arange(n)[:max_samples]
labels = [label for _, label in dataset.samples]
labels_small = [labels[i] for i in indices]

train_idx, temp_idx = train_test_split(indices, test_size=0.4, stratify=labels_small, random_state=42)
val_idx, test_idx = train_test_split(temp_idx, test_size=0.5, stratify=[labels[i] for i in temp_idx], random_state=42)

train_set = Subset(dataset, train_idx)
val_set = Subset(dataset, val_idx)

train_loader = DataLoader(train_set, batch_size=2, shuffle=True, num_workers=0)
val_loader = DataLoader(val_set, batch_size=2, shuffle=False, num_workers=0)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)

model = Decoder_testing(input_channels=1, num_classes=5).to(device)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# One epoch quick train
model.train()
for batch_idx, (X, y) in enumerate(train_loader):
    X, y = X.to(device), y.to(device)
    optimizer.zero_grad()
    outputs = model(X)
    # model returns per-frame logits (batch_size * temporal_dim, num_classes)
    # reshape to (batch_size, temporal_dim, num_classes) and average over temporal dim
    bs, td = X.shape[0], X.shape[1]
    outputs = outputs.view(bs, td, -1).mean(dim=1)
    loss = criterion(outputs, y)
    loss.backward()
    optimizer.step()
    if batch_idx % 5 == 0:
        print(f'Batch {batch_idx}, loss {loss.item():.4f}')

# quick validation
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for X, y in val_loader:
        X, y = X.to(device), y.to(device)
        outputs = model(X)
        bs, td = X.shape[0], X.shape[1]
        outputs = outputs.view(bs, td, -1).mean(dim=1)
        _, predicted = torch.max(outputs.data, 1)
        total += y.size(0)
        correct += (predicted == y).sum().item()

print(f'Quick validation accuracy: {100 * correct / total:.2f}% on {total} samples')
