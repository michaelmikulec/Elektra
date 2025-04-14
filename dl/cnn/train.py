import os
import time
import torch
import pandas as pd
import torch.nn as nn
from torch.optim import Adam
from tqdm import trange, tqdm
from sklearn.metrics import accuracy_score, roc_auc_score

from dl.dataHandler.dataHandler import getDataloader
from dl.cnn.cnn import SpectrogramCNN

def train(
  index_csv   = "data/preprocessing_index.csv",
  data_dir    = "data/preproc",
  batch_size  = 32,
  num_samples = 1000,
  num_epochs  = 10,
  lr          = 1e-4,
  val_split   = 0.2,
  save_dir    = "models/dl/cnn",
  log_file    = "logs/dl/cnn/training.log",
  metrics_out = "logs/dl/cnn/training_metrics.parquet"
):
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  os.makedirs(save_dir, exist_ok=True)
  os.makedirs(os.path.dirname(log_file), exist_ok=True)

  log_entries = []
  metrics     = []
  start_time  = time.time()

  full_loader = getDataloader(index_csv, data_dir, num_samples=num_samples, batch_size=batch_size)
  dataset     = full_loader.dataset
  val_size    = int(len(dataset) * val_split)
  train_set, val_set = torch.utils.data.random_split(dataset, [len(dataset) - val_size, val_size])
  train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
  val_loader   = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=False)

  model     = SpectrogramCNN().to(device)
  criterion = nn.BCEWithLogitsLoss()
  optimizer = Adam(model.parameters(), lr=lr)

  latest_model_path = None
  latest_epoch = 0
  for fname in sorted(os.listdir(save_dir)):
    if fname.startswith("model_epoch") and fname.endswith(".pt"):
      epoch_num = int(fname.split("epoch")[-1].split(".pt")[0])
      if epoch_num > latest_epoch:
        latest_epoch = epoch_num
        latest_model_path = os.path.join(save_dir, fname)

  if latest_model_path:
    print(f"Resuming from {latest_model_path}")
    model.load_state_dict(torch.load(latest_model_path))

  model.train()
  for epoch in trange(latest_epoch, latest_epoch + num_epochs, desc="Epochs"):
    total_loss = 0
    for batch_idx, (_, spec, labels) in enumerate(tqdm(train_loader, desc=f"Train {epoch+1}", leave=False)):
      spec, labels = spec.to(device), labels.to(device)
      optimizer.zero_grad()
      outputs = model(spec)
      loss = criterion(outputs, labels)

      if torch.isnan(loss) or torch.isinf(loss):
        raise RuntimeError(f"Invalid loss at epoch {epoch+1}, batch {batch_idx+1}: {loss.item()}")

      loss.backward()
      optimizer.step()
      total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)

    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
      for _, spec, labels in val_loader:
        spec = spec.to(device)
        preds = torch.sigmoid(model(spec)).cpu()
        all_preds.append(preds)
        all_labels.append(labels)

    model.train()
    preds = torch.cat(all_preds)
    labels = torch.cat(all_labels)
    pred_labels = preds.argmax(dim=1)
    true_labels = labels.argmax(dim=1)
    acc = accuracy_score(true_labels, pred_labels)

    try:
      auc = roc_auc_score(labels, preds, average="macro", multi_class="ovr")
    except:
      auc = float("nan")

    runtime = time.time() - start_time
    runtime_str = time.strftime("%H:%M:%S", time.gmtime(runtime))
    model_path = os.path.join(save_dir, f"model_epoch{epoch+1}.pt")
    torch.save(model.state_dict(), model_path)

    log_entries.append(f"Epoch {epoch+1}/{latest_epoch + num_epochs} | Loss: {avg_loss:.4f} | Acc: {acc:.4f} | AUC: {auc:.4f} | Time: {runtime_str}")
    metrics.append({
      "epoch": epoch + 1,
      "loss": avg_loss,
      "accuracy": acc,
      "auc": auc,
      "runtime_hms": runtime_str,
      "model_path": model_path
    })

  with open(log_file, "a") as f:
    f.write("\n".join(log_entries) + "\n")

  pd.DataFrame(metrics).to_parquet(metrics_out, index=False)

  print("\nTraining complete.\nSummary:")
  print("\n".join(log_entries))

if __name__ == "__main__":
  train()
