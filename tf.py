import os, math
import pandas as pd
from tqdm.auto import tqdm
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

class EEGTransformer(nn.Module):
  def __init__(
    self,
    n_channels   = 19,
    d_model      = 128,
    n_heads      = 8,
    n_layers     = 6,
    d_ff         = 512,
    dropout      = 0.1,
    n_classes    = 6,
    max_seq_len  = 2000
  ):
    super().__init__()
    self.input_proj = nn.Linear(n_channels, d_model)
    pe               = torch.zeros(max_seq_len, d_model)
    pos              = torch.arange(max_seq_len).unsqueeze(1)
    div_term         = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
    pe[:, 0::2]      = torch.sin(pos * div_term)
    pe[:, 1::2]      = torch.cos(pos * div_term)
    self.register_buffer('pe', pe.unsqueeze(0))

    encoder_layer = nn.TransformerEncoderLayer(
      d_model         = d_model,
      nhead           = n_heads,
      dim_feedforward = d_ff,
      dropout         = dropout,
      batch_first     = True
    )
    self.encoder    = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
    self.norm       = nn.LayerNorm(d_model)
    self.classifier = nn.Sequential(
      nn.Linear(d_model, d_model//2),
      nn.ReLU(),
      nn.Dropout(dropout),
      nn.Linear(d_model//2, n_classes)
    )

  def forward(self, x):
    x = self.input_proj(x)
    x = x + self.pe[:, :x.size(1)]
    x = self.encoder(x)
    x = x.mean(dim=1)
    x = self.norm(x)
    return self.classifier(x)

class EEGDataset(Dataset):
  def __init__(self, folder, classes):
    self.files   = [f for f in os.listdir(folder) if f.endswith('.parquet')]
    self.folder  = folder
    self.cls2idx = {c:i for i,c in enumerate(classes)}

  def __len__(self):
    return len(self.files)

  def __getitem__(self, idx):
    fn        = self.files[idx]
    label_str = fn.split('_')[-1].split('.')[0]
    label_idx = self.cls2idx[label_str]
    df        = pd.read_parquet(os.path.join(self.folder, fn))
    data      = torch.from_numpy(df.values).float()
    return data, torch.tensor(label_idx).long()

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
  data   = "data/prep/eegs"
  labels = ['seizure', 'lpd', 'gpd', 'lrda', 'grda', 'other']

  print("Loading data...")
  dataset = EEGDataset(data, labels)
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
  model    = EEGTransformer()
  opt      = torch.optim.AdamW(model.parameters(), lr=1e-4)
  crit     = torch.nn.CrossEntropyLoss()
  epochs   = 50
  ckptPath = 'checkpoint.pth'
  train_losses, val_losses, val_accs = train(model, train_dl, val_dl, opt, crit, device, epochs=epochs, ckptPath=ckptPath)
