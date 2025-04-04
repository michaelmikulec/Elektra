import os
import random
import math
import torch
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

class SinusoidalPositionalEncoding(nn.Module):
  def __init__(self, d_model, max_len=2001):
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
  def __init__(self, d_model, max_len=2001):
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
    input_dim             = 20,
    model_dim             = 256,
    num_heads             = 8,
    num_layers            = 6,
    dim_feedforward       = 1024,
    dropout               = 0.1,
    num_classes           = 6,
    max_len               = 10000,
    use_learnable_pos_emb = True,
    use_cls_token         = True,
    pooling               = "cls"
  ):
    super().__init__()
    self.use_cls_token = use_cls_token
    self.pooling       = pooling
    if use_cls_token:
      self.cls_token = nn.Parameter(torch.zeros(1, 1, model_dim))
    self.input_embed = nn.Linear(input_dim, model_dim)
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
      bsz        = x.size(0)
      cls_tokens = self.cls_token.repeat(bsz, 1, 1)
      x          = torch.cat([cls_tokens, x], dim=1)

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
    path   = self.file_paths[idx]
    label  = int(os.path.basename(path).split('_')[0])
    df     = pd.read_parquet(path)
    data   = torch.tensor(df.values, dtype=torch.float32)
    if self.transform:
      data = self.transform(data)
    return data, label


def print_model_stats(model):
  print(model)
  total_params = sum(p.numel() for p in model.parameters())
  train_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
  print(f"Total parameters: {total_params}")
  print(f"Trainable parameters: {train_params}")


def select_files_per_class(dataFiles, x):
  groups = {}
  for f in dataFiles:
    base = os.path.basename(f)
    label = base.split('_')[0]
    groups.setdefault(label, []).append(f)
  result = []
  for files in groups.values():
    if len(files) >= x:
      result.extend(random.sample(files, x))
    else:
      result.extend(files)
  return result


def load_model(model_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EEGTransformer()
    if torch.cuda.device_count() > 1:
        model = DataParallel(model)
    model.to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model, device


def train(modelName, modelDir,  dataDir, numPerClass, batchSize, lr, numEpochs):
  dataFiles = [
    os.path.join(dataDir, f)
    for f in os.listdir(dataDir)
    if f.endswith(".parquet")
  ]
  subset     = select_files_per_class(dataFiles, numPerClass)
  dataset    = EEGDataset(subset)
  dataLoader = DataLoader(dataset, batch_size=batchSize, shuffle=True, drop_last=False)
  model      = EEGTransformer()
  device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  model      = model.to(device)

  print_model_stats(model)

  if os.path.exists(modelName):
    print("Loading checkpoint from", modelName)
    model.load_state_dict(torch.load(modelName, map_location=device))
  else:
    print("No existing model found - initializing new model.")

  opt    = optim.Adam(model.parameters(), lr=lr)
  crit   = nn.CrossEntropyLoss()
  epochs = numEpochs

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
      preds      = output.argmax(dim=1)
      correct    += (preds == label).sum().item()
      total      += label.size(0)
      print(f"\nBatch Loss: {loss.item():.4f}", end="\r")

    avg_loss = total_loss / len(dataLoader) if len(dataLoader) > 0 else 0
    acc      = correct / total if total > 0 else 0
    print(f"Epoch {ep+1}/{epochs} | Loss: {avg_loss:.4f} | Acc: {acc:.4f}")

    checkpoint_name = os.path.join(modelDir, f"checkpoints/E{ep+1}A{acc:.3f}L{avg_loss:.3f}.pth")
    torch.save(model.state_dict(), checkpoint_name)
    print(f"Saved model checkpoint to {checkpoint_name}")

  latestModel = ("./models/dl/transformer/transformer.pth")
  torch.save(model.state_dict(), latestModel)
  print(f"Saved model to {latestModel}")


def infer(input_data_or_path, model_path):
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  model = EEGTransformer()
  if torch.cuda.device_count() > 1:
      model = DataParallel(model)
  model.load_state_dict(torch.load(model_path, map_location=device))
  model.to(device)
  model.eval()
  if isinstance(input_data_or_path, str):
    df = pd.read_parquet(input_data_or_path)
    inputs = torch.tensor(df.values, dtype=torch.float32)
  else:
    inputs = input_data_or_path
  if inputs.ndim == 2:
    inputs = inputs.unsqueeze(0)
  inputs = inputs.to(device)
  with torch.no_grad():
    with autocast():
      logits = model(inputs)
  probs = F.softmax(logits, dim=1)
  confidence, pred_idx = torch.max(probs, dim=1)
  label_map = {
      0: "Seizure",
      1: "LRDA",
      2: "GRDA",
      3: "LPD",
      4: "GPD",
      5: "Other"
  }
  print(f"Prediction: {label_map[pred_idx.item()]} with confidence {confidence.item():.2%}")


if __name__ == "__main__":
  dataDir     = "./data/training_data/eegs/"
  modelDir    = "./models/dl/transformer/"
  modelName   = "transformer.pth"
  modelPath   = os.path.join(modelDir, modelName)
  numPerClass = 600
  batchSize   = 100
  lr          = 1e-4
  numEpochs   = 10
  numTraining = 10

  # for i in range(numTraining + 1):
  #   train(modelPath, modelDir,  dataDir, numPerClass, batchSize, lr, numEpochs)



















  print("these two should be Seizure:")
  infer("./0_1132121900_EEG-4010993438-7.parquet", modelPath)
  infer("./0_12031182_EEG-4266393632-3.parquet", modelPath)

  print("these two should be GRDA:")
  infer("./2_644232451_EEG-852583442-67.parquet", modelPath)
  infer("./2_644446039_EEG-852583442-10.parquet", modelPath)

  print("these two should be LPD:")
  infer("./3_207331724_EEG-3556046250-4.parquet", modelPath)
  infer("./3_708361572_EEG-1353134161-2.parquet", modelPath)

  print("these two should be GPD:")
  infer("./4_3221007399_EEG-3232076969-4.parquet", modelPath)
  infer("./4_609700825_EEG-1641054670-64.parquet", modelPath)

  print("these two should be Other:")
  infer("./5_979752000_EEG-1595550186-6.parquet", modelPath)
  infer("./5_999516971_EEG-3580650910-9.parquet", modelPath)
  