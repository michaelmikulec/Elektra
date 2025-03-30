import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

class SpecDataset(Dataset):
  def __init__(self, file_paths, transform=None):
    self.file_paths = file_paths
    self.transform  = transform

  def __len__(self):
    return len(self.file_paths)

  def __getitem__(self, idx):
    path   = self.file_paths[idx]
    label  = int(os.path.basename(path).split('_')[0])
    df     = pd.read_parquet(path)
    data   = torch.tensor(df.values, dtype=torch.float32)
    if self.transform:
      data = self.transform(data)
    data   = data.unsqueeze(0)
    return data, label

def print_model_stats(model):
  total_params   = sum(p.numel() for p in model.parameters())
  train_params   = sum(p.numel() for p in model.parameters() if p.requires_grad)
  print(model)
  print(f"Total parameters: {total_params}")
  print(f"Trainable parameters: {train_params}")

class SpecCNN(nn.Module):
  def __init__(self, base_filters=32, num_classes=6, dropout=0.1):
    super().__init__()
    self.features = nn.Sequential(
      nn.Conv2d(1, base_filters, kernel_size=3, stride=1, padding=1),
      nn.BatchNorm2d(base_filters),
      nn.ReLU(),
      nn.MaxPool2d(kernel_size=2, stride=2),
      nn.Conv2d(base_filters, base_filters * 2, kernel_size=3, stride=1, padding=1),
      nn.BatchNorm2d(base_filters * 2),
      nn.ReLU(),
      nn.MaxPool2d(kernel_size=2, stride=2)
    )
    self.classifier = nn.Sequential(
      nn.Dropout(dropout),
      nn.Linear(base_filters * 2 * (5 // 4) * (400 // 4), base_filters * 2),
      nn.ReLU(),
      nn.Dropout(dropout),
      nn.Linear(base_filters * 2, num_classes)
    )

  def forward(self, x):
    x = self.features(x)
    x = x.view(x.size(0), -1)
    x = self.classifier(x)
    return x

def train_spec_cnn(data_dir="./data/training_data/spectrograms/",
                   model_path="./spec_cnn.pth",
                   report_path="./spec_cnn_report.txt",
                   epochs=5, batch_size=64, lr=1e-3):
  files = [
    os.path.join(data_dir, f)
    for f in os.listdir(data_dir)
    if f.endswith(".parquet")
  ]
  dataset = SpecDataset(files)
  loader  = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)
  model   = SpecCNN(base_filters=32, num_classes=6, dropout=0.1)
  device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  model   = model.to(device)

  print_model_stats(model)

  if os.path.exists(model_path):
    print(f"Loading checkpoint from {model_path}")
    model.load_state_dict(torch.load(model_path, map_location=device))
  else:
    print("No existing model found - initializing new model.")

  opt = optim.Adam(model.parameters(), lr=lr)
  crit = nn.CrossEntropyLoss()

  with open(report_path, "w") as rep:
    rep.write("SpecCNN Training Report\n")

  model.train()
  for ep in tqdm(range(epochs), desc="Training"):
    total_loss = 0.0
    correct = 0
    total = 0

    for data, label in tqdm(loader, desc=f"Epoch {ep+1}", leave=False):
      data, label = data.to(device), label.to(device)
      opt.zero_grad()
      output = model(data)
      loss = crit(output, label)
      loss.backward()
      opt.step()
      total_loss += loss.item()
      preds = output.argmax(dim=1)
      correct += (preds == label).sum().item()
      total += label.size(0)
      print(f"Batch Loss: {loss.item():.4f}", end="\r")

    avg_loss = total_loss / len(loader) if len(loader) > 0 else 0
    acc = correct / total if total > 0 else 0
    epoch_msg = f"Epoch {ep+1}/{epochs} | Loss: {avg_loss:.4f} | Acc: {acc:.4f}"
    print(epoch_msg)
    with open(report_path, "a") as rep:
      rep.write(epoch_msg + "\n")

    torch.save(model.state_dict(), model_path)

if __name__ == "__main__":
  train_spec_cnn()
