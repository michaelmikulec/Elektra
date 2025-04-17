import os
import time
from datetime import datetime
import math
import torch
import pandas as pd
import pyarrow.parquet as pq
import torch.nn as nn
from torch.optim import Adam
from tqdm import trange, tqdm
from sklearn.metrics import accuracy_score, roc_auc_score
from dl.dh.dh import getDataloader
from dl.prep.prep import get_eeg_event_window


class EEGTransformer(nn.Module):
  def __init__(
    self,
    input_dim   = 20,
    model_dim   = 256,
    num_heads   = 8,
    num_layers  = 6,
    ff_dim      = 1024,
    dropout     = 0.1,
    num_classes = 6,
    max_len     = 2000
  ):
    super().__init__()
    self.projector = nn.Linear(input_dim, model_dim)
    pe             = torch.zeros(max_len, model_dim)
    position       = torch.arange(0, max_len).unsqueeze(1)
    div_term       = torch.exp(torch.arange(0, model_dim, 2) * (-math.log(10000.0) / model_dim))
    pe[:, 0::2]    = torch.sin(position * div_term)
    pe[:, 1::2]    = torch.cos(position * div_term)
    self.register_buffer("pe", pe.unsqueeze(0))

    encoder_layer = nn.TransformerEncoderLayer(
      d_model         = model_dim,
      nhead           = num_heads,
      dim_feedforward = ff_dim,
      dropout         = dropout,
      batch_first     = True
    )
    self.encoder    = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
    self.classifier = nn.Sequential(
      nn.LayerNorm(model_dim),
      nn.Linear(model_dim, 128),
      nn.ReLU(),
      nn.Dropout(dropout),
      nn.Linear(128, num_classes)
    )

  def forward(self, x):
    x = self.projector(x)
    x = x + self.pe[:, :x.size(1)]
    x = self.encoder(x)
    x = x.mean(dim=1)
    return self.classifier(x)


def train(
  index_csv       = "data/pidx.csv",
  data_dir        = "data/prep",
  batch_size      = 1,
  num_samples     = 600,
  num_epochs      = 10,
  lr              = 1e-4,
  val_split       = 0.2,
  save_dir        = "dl/tf/models",
  log_file        = "dl/tf/logs/training.log",
  metrics_out     = "dl/tf/logs/training_metrics.parquet",
  resume_training = False
):
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

  os.makedirs(save_dir, exist_ok=True)
  os.makedirs(os.path.dirname(log_file), exist_ok=True)

  log_entries  = []
  metrics      = []
  start_time   = time.time()
  full_loader  = getDataloader(index_csv, data_dir, num_samples=num_samples, batch_size=batch_size, num_workers=16)
  dataset      = full_loader.dataset
  val_size     = int(len(dataset) * val_split)
  train_set    = torch.utils.data.Subset(dataset, list(range(0, len(dataset) - val_size)))
  val_set      = torch.utils.data.Subset(dataset, list(range(len(dataset) - val_size, len(dataset))))
  train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
  val_loader   = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=False)
  model        = EEGTransformer().to(device)
  criterion    = nn.BCEWithLogitsLoss()
  optimizer    = Adam(model.parameters(), lr=lr)

  latest_model_path = None
  latest_epoch      = 0

  if resume_training:
    for fname in sorted(os.listdir(save_dir)):
      if fname.startswith("model_epoch") and fname.endswith(".pt"):
        epoch_num     = int(fname.split("epoch")[-1].split(".pt")[0])
        if epoch_num > latest_epoch:
          latest_epoch      = epoch_num
          latest_model_path = os.path.join(save_dir, fname)
    if latest_model_path:
      print(f"Resuming from {latest_model_path}")
      model.load_state_dict(torch.load(latest_model_path))

  model.train()
  for epoch in trange(latest_epoch, latest_epoch + num_epochs, desc="Epochs"):
    total_loss = 0

    for batch_idx, (eeg, _, labels) in enumerate(tqdm(train_loader, desc=f"Train {epoch+1}", leave=False)):
      eeg, labels = eeg.to(device), labels.to(device)
      optimizer.zero_grad()
      outputs = model(eeg)
      loss    = criterion(outputs, labels)

      if torch.isnan(loss) or torch.isinf(loss):
        raise RuntimeError(f"Invalid loss at epoch {epoch+1}, batch {batch_idx+1}: {loss.item()}")

      loss.backward()
      optimizer.step()
      total_loss += loss.item()

    avg_loss   = total_loss / len(train_loader)

    model.eval()
    all_preds  = []
    all_labels = []
    with torch.no_grad():
      for eeg, _, labels in val_loader:
        eeg    = eeg.to(device)
        preds  = torch.sigmoid(model(eeg)).cpu()
        all_preds.append(preds)
        all_labels.append(labels)

    model.train()
    preds       = torch.cat(all_preds)
    labels      = torch.cat(all_labels)
    pred_labels = preds.argmax(dim = 1)
    true_labels = labels.argmax(dim = 1)
    acc         = accuracy_score(true_labels, pred_labels)

    try:    
      auc = roc_auc_score(labels, preds, average="macro", multi_class="ovr")
    except:
      auc = float("nan")

    runtime     = time.time() - start_time
    runtime_str = time.strftime("%H:%M:%S", time.gmtime(runtime))
    model_path  = os.path.join(save_dir, f"model_epoch{epoch+1}.pt")

    torch.save(model.state_dict(), model_path)

    log_entries.append(f"Epoch {epoch+1}/{latest_epoch + num_epochs} | Loss: {avg_loss:.4f} | Acc: {acc:.4f} | AUC: {auc:.4f} | Time: {runtime_str}")
    metrics.append({
      "epoch"       : epoch + 1,
      "loss"        : avg_loss,
      "accuracy"    : acc,
      "auc"         : auc,
      "runtime_hms" : runtime_str,
      "model_path"  : model_path
    })

  with open(log_file, "a") as f:
    f.write("\n".join(log_entries) + "\n")

  pd.DataFrame(metrics).to_parquet(metrics_out, index=False)

  print("\nTraining complete.\nSummary:")
  print("\n".join(log_entries))


def infer(
  parquet_path,
  offset_sec,
  model_dir = "dl/tf/models",
  log_file  = "dl/tf/logs/inference.log"
):
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

  df         = pq.read_table(parquet_path).to_pandas()
  eeg_tensor = get_eeg_event_window(df, offset_sec)
  eeg_tensor = eeg_tensor.unsqueeze(0).to(device)  # shape: (1, 2000, 20)

  model_files = [
    f for f in os.listdir(model_dir)
    if f.startswith("model_epoch") and f.endswith(".pt")
  ]
  if not model_files:
    raise FileNotFoundError("No model checkpoint found in directory.")

  latest_model = sorted(
    model_files,
    key=lambda x: int(x.split("epoch")[-1].split(".pt")[0])
  )[-1]
  model_path = os.path.join(model_dir, latest_model)

  model = EEGTransformer().to(device)
  model.load_state_dict(torch.load(model_path, map_location=device))
  model.eval()

  with torch.no_grad():
    output = model(eeg_tensor)
    probs  = torch.softmax(output, dim=1).cpu().squeeze().tolist()

  labels          = ["seizure", "lpd", "gpd", "lrda", "grda", "other"]
  predicted_index = int(torch.tensor(probs).argmax())
  predicted_label = labels[predicted_index]

  # Print results
  print("Predicted probabilities:")
  for label, prob in zip(labels, probs):
    print(f"{label:>8}: {prob:.4f}")
  print(f"\nPredicted class: {predicted_label}")

  # Log results
  timestamp = datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")
  log_lines = [f"{timestamp} Input: {parquet_path}", "Probabilities:"]
  log_lines += [f"  {label:>8}: {prob:.4f}" for label, prob in zip(labels, probs)]
  log_lines.append(f"Predicted: {predicted_label}\n")

  os.makedirs(os.path.dirname(log_file), exist_ok=True)
  with open(log_file, "a") as f:
    f.write("\n".join(log_lines) + "\n")