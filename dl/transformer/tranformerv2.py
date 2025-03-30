import os
import math
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

class SinusoidalPositionalEncoding(nn.Module):
  def __init__(self, d_model, max_len=10001):
    super().__init__()
    pos_data          = torch.zeros(max_len, d_model)
    pos_range         = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
    div_term          = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
    pos_data[:, 0::2] = torch.sin(pos_range * div_term)
    pos_data[:, 1::2] = torch.cos(pos_range * div_term)
    pos_data          = pos_data.unsqueeze(0)
    self.register_buffer("pos_data", pos_data)
  
  def forward(self, x):
    seq_len = x.size(1)
    if seq_len > self.pos_data.size(1):
      raise ValueError(str(seq_len))
    x = x + self.pos_data[:, :seq_len, :]
    return x

class LearnablePositionalEncoding(nn.Module):
  def __init__(self, d_model, max_len=10001):
    super().__init__()
    self.pos_data = nn.Parameter(torch.zeros(1, max_len, d_model))
  
  def forward(self, x):
    seq_len = x.size(1)
    if seq_len > self.pos_data.size(1):
      raise ValueError(str(seq_len))
    x = x + self.pos_data[:, :seq_len, :]
    return x

class EEGTransformer(nn.Module):
  def __init__(
    self,
    input_dim=20,
    model_dim=128,
    num_heads=8,
    num_layers=8,
    dim_feedforward=256,
    dropout=0.1,
    num_classes=6,
    max_len=10001,
    use_learnable_pos_emb=True,
    use_cls_token=True,
    pooling="cls"
  ):
    super().__init__()
    self.use_cls_token  = use_cls_token
    self.pooling        = pooling
    if use_cls_token:
      self.cls_token    = nn.Parameter(torch.zeros(1, 1, model_dim))
    self.input_embed    = nn.Linear(input_dim, model_dim)
    if use_learnable_pos_emb:
      self.pos_encoding = LearnablePositionalEncoding(model_dim, max_len=max_len)
    else:
      self.pos_encoding = SinusoidalPositionalEncoding(model_dim, max_len=max_len)
    encoder_layer = nn.TransformerEncoderLayer(
      d_model=model_dim,
      nhead=num_heads,
      dim_feedforward=dim_feedforward,
      dropout=dropout,
      activation="relu",
      batch_first=True
    )
    self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
    self.classifier = nn.Sequential(
      nn.LayerNorm(model_dim),
      nn.Linear(model_dim, model_dim // 2),
      nn.ReLU(),
      nn.Dropout(dropout),
      nn.Linear(model_dim // 2, num_classes)
    )
    self.init_weights()
  
  def init_weights(self):
    if self.use_cls_token:
      nn.init.normal_(self.cls_token, mean=0.0, std=0.02)
  
  def forward(self, x):
    x = self.input_embed(x)
    if self.use_cls_token:
      bsz = x.size(0)
      cls_tokens = self.cls_token.repeat(bsz, 1, 1)
      x = torch.cat([cls_tokens, x], dim=1)
    x = self.pos_encoding(x)
    x = self.encoder(x)
    if self.use_cls_token and self.pooling == "cls":
      x = x[:, 0, :]
    else:
      x = x.mean(dim=1)
    x = self.classifier(x)
    return x

class EEGDataset(Dataset):
  def __init__(self, file_paths, transform=None):
    self.file_paths = file_paths
    self.transform  = transform

  def __len__(self):
    return len(self.file_paths)

  def __getitem__(self, idx):
    path  = self.file_paths[idx]
    label = int(os.path.basename(path).split('_')[0])
    df    = pd.read_parquet(path)
    data  = torch.tensor(df.values, dtype=torch.float32)
    if self.transform:
      data = self.transform(data)
    return data, label

def print_model_stats(model):
  print(model)
  total_params = sum(p.numel() for p in model.parameters())
  train_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
  print(f"Total parameters: {total_params}")
  print(f"Trainable parameters: {train_params}")

def main():
  dataDir   = "./data/training_data/eegs/"
  dataFiles = [
    os.path.join(dataDir, f)
    for f in os.listdir(dataDir)
    if f.endswith(".parquet")
  ]
  dataset = EEGDataset(dataFiles)
  dataLoader = DataLoader(dataset, batch_size=128, shuffle=True, drop_last=False)

  modelName = "./transformer.pth"
  model     = EEGTransformer()
  device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  model     = model.to(device)

  print_model_stats(model)

  if os.path.exists(modelName):
    print("Loading checkpoint from", modelName)
    model.load_state_dict(torch.load(modelName, map_location=device))
  else:
    print("No existing model found - initializing new model.")

  opt    = optim.Adam(model.parameters(), lr=1e-3)
  crit   = nn.CrossEntropyLoss()
  epochs = 5

  model.train()
  for ep in tqdm(range(epochs), desc="Training"):
    total_loss = 0.0
    correct    = 0
    total      = 0

    for data, label in tqdm(dataLoader, desc=f"Epoch {ep+1}", leave=False):
      data, label = data.to(device), label.to(device)
      opt.zero_grad()
      output = model(data)
      loss   = crit(output, label)
      loss.backward()
      opt.step()
      total_loss += loss.item()
      preds   = output.argmax(dim=1)
      correct += (preds == label).sum().item()
      total   += label.size(0)

      # Print the instantaneous loss
      print(f"Batch Loss: {loss.item():.4f}", end="\r")

    avg_loss = total_loss / len(dataLoader) if len(dataLoader) > 0 else 0
    acc      = correct / total if total > 0 else 0
    print(f"Epoch {ep+1}/{epochs} | Loss: {avg_loss:.4f} | Acc: {acc:.4f}")

if __name__ == "__main__":
  main()
