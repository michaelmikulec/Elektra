import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

from tqdm import tqdm
import time
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn import DataParallel
from torch.cuda.amp import autocast, GradScaler

from model import EEGTransformer
from dataset import EEGDataset, normalize_sample
from config import dataset_config, training_config, model_config

def train(dataset):
  start = time.time()
  loader = DataLoader(
    dataset,
    batch_size=training_config["batch_size"],
    shuffle=True,
    num_workers=training_config["num_workers"],
    pin_memory=True,
    drop_last=True
  )
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  model = EEGTransformer(**model_config)
  if torch.cuda.device_count() > 1:
    model = DataParallel(model)
  model.to(device)

  model_path = training_config["model_path"]
  if os.path.exists(model_path):
    print("Loading checkpoint from", model_path)
    model.load_state_dict(torch.load(model_path, map_location=device))
  else:
    print("No existing model found - initializing new model.")

  optimizer = optim.Adam(model.parameters(), lr=training_config["learning_rate"])
  criterion = nn.CrossEntropyLoss()
  scaler = GradScaler()

  model.train()

  epochs = training_config["num_epochs"]
  training_bar = tqdm(range(epochs), desc="Training", position=0, leave=True)
  log_file = os.path.join(os.path.dirname(model_path), "training_log.txt")
  for epoch in training_bar:
    epoch_loss = 0.0
    correct = 0
    total = 0
    epoch_bar = tqdm(loader, desc=f"Epoch {epoch+1}/{epochs}", position=1, leave=False)
    for inputs, targets in epoch_bar:
      inputs = inputs.to(device)
      targets = targets.to(device)
      optimizer.zero_grad()
      with autocast():
        outputs = model(inputs)
        if torch.isnan(outputs).any() or torch.isinf(outputs).any():
          continue
        loss_value = criterion(outputs, targets)
        if torch.isnan(loss_value) or torch.isinf(loss_value):
          continue

      scaler.scale(loss_value).backward()
      scaler.unscale_(optimizer)
      nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
      scaler.step(optimizer)
      scaler.update()
      epoch_loss += loss_value.item()
      preds = outputs.argmax(dim=1)
      correct += (preds == targets).sum().item()
      total += targets.size(0)
      epoch_bar.set_postfix(loss=f"{loss_value.item():.4f}")

    avg_loss = epoch_loss / len(loader)
    accuracy = correct / total if total else 0
    training_bar.set_postfix(loss=f"{avg_loss:.4f}", accuracy=f"{accuracy:.4f}")
    with open(log_file, "a") as f:
      f.write(f"Epoch {epoch+1}/{epochs} Loss: {avg_loss:.4f} Accuracy: {accuracy:.4f}\n")

  torch.save(model.state_dict(), model_path)
  runtime = time.time() - start
  h = int(runtime // 3600)
  m = int((runtime % 3600) // 60)
  s = int(runtime % 60)
  print(f"Training time: {h:02d}:{m:02d}:{s:02d}")

if __name__ == "__main__":
  train(dataset = EEGDataset(dataset_config["training_data"], 1000))
  
