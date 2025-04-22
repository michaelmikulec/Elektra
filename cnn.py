import os, glob
from tqdm.auto import tqdm
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

class SpectrogramCNN(nn.Module):
  def __init__(self, in_channels=19, num_classes=6):
    super().__init__()
    self.features = nn.Sequential(
      nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
      nn.ReLU(),
      nn.MaxPool2d(2),
      nn.Conv2d(32, 64, kernel_size=3, padding=1),
      nn.ReLU(),
      nn.MaxPool2d(2),
      nn.Conv2d(64, 128, kernel_size=3, padding=1),
      nn.ReLU(),
      nn.MaxPool2d(2)
    )
    self.classifier = nn.Sequential(
      nn.AdaptiveAvgPool2d((1, 1)),
      nn.Flatten(),
      nn.Linear(128, num_classes)
    )

  def forward(self, x):
    x = self.features(x)
    x = self.classifier(x)
    return x

class SpectrogramDataset(Dataset):
  def __init__(self, folder):
    self.files = sorted(glob.glob(os.path.join(folder, 'SPEC_*.pt')))

  def __len__(self):
    return len(self.files)

  def __getitem__(self, idx):
    data = torch.load(self.files[idx])
    return data['spectrogram'], data['label']

def train(model, train_dl, val_dl, optimizer, criterion, device, epochs, ckptPath='checkpoint.pth'):
  print("Training...")
  model.to(device)
  best_val = float('inf')
  train_losses, val_losses, val_accs = [], [], []
  for epoch in range(1, epochs+1):
    print(f'Starting epoch {epoch}/{epochs}')
    model.train()
    tl = 0.0
    for xb, yb in tqdm(train_dl, desc=f'Train Epoch {epoch}/{epochs}', leave=False):
      xb, yb = xb.to(device), yb.to(device)
      optimizer.zero_grad(set_to_none=True)
      out  = model(xb)
      loss = criterion(out, yb)
      loss.backward()
      optimizer.step()
      tl += loss.item() * xb.size(0)
    tl /= len(train_dl.dataset)

    print("Evaluating on validation set...")
    model.eval()
    vl, correct = 0.0, 0
    for xb, yb in tqdm(val_dl, desc=f'Val   Epoch {epoch}/{epochs}', leave=False):
      xb, yb  = xb.to(device), yb.to(device)
      out     = model(xb)
      loss    = criterion(out, yb)
      vl      += loss.item() * xb.size(0)
      correct += (out.argmax(1) == yb).sum().item()
    vl   /= len(val_dl.dataset)
    acc  = correct / len(val_dl.dataset)
    train_losses.append(tl)
    val_losses.append(vl)
    val_accs.append(acc)

    print(f'Epoch {epoch}/{epochs} train_loss={tl:.4f} val_loss={vl:.4f} val_acc={acc:.4f}')
    if vl < best_val:
      best_val = vl
      torch.save(model.state_dict(), ckptPath)

  print("Training complete.")
  return train_losses, val_losses, val_accs

if __name__ == '__main__':
  print("Loading data...")
  dataset = SpectrogramDataset("data/prep/specs")
  n       = len(dataset)
  trainn  = int(0.6 * n)
  valn    = int(0.2 * n)
  testn   = n - trainn - valn
  torch.manual_seed(42)

  print("Splitting data into train, val, and test sets...")
  train_ds, val_ds, test_ds = random_split(dataset, [trainn, valn, testn])
  train_dl = DataLoader(train_ds, batch_size=256, shuffle=True,  num_workers=16, pin_memory=True)
  val_dl   = DataLoader(val_ds,   batch_size=256, shuffle=False, num_workers=16, pin_memory=True)
  test_dl  = DataLoader(test_ds,  batch_size=256, shuffle=False, num_workers=16, pin_memory=True)
  device   = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  model    = SpectrogramCNN()
  opt      = torch.optim.AdamW(model.parameters(), lr=1e-4)
  crit     = torch.nn.CrossEntropyLoss()
  epochs   = 50
  ckptPath = 'checkpoint.pth'
  train_losses, val_losses, val_accs = train(model, train_dl, val_dl, opt, crit, device, epochs=epochs, ckptPath=ckptPath)

