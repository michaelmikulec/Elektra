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
from dataset import EEGDataset
from config import dataset_config, training_config, model_config

def train():
  start = time.time()

  dataset = EEGDataset(dataset_config["training_data"])
  loader = DataLoader(
    dataset,
    batch_size=training_config["batch_size"],
    shuffle=True,
    num_workers=24,
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
    print(f"Loading checkpoint from {model_path}")
    model.load_state_dict(torch.load(model_path, map_location=device))
  else:
    print("No existing model found — initializing new model.")

  optimizer = optim.Adam(model.parameters(), lr=training_config["learning_rate"])
  criterion = nn.CrossEntropyLoss()
  scaler = GradScaler()

  model.train()
  training_bar = tqdm(
    range(training_config["num_epochs"]),
    desc="  Training Progress",
    position=0,
    leave=True
  )

  for epoch in training_bar:
    epoch_loss = 0.0
    epoch_bar = tqdm(
      loader,
      desc="  Epoch Progress",
      position=1,
      leave=False
    )

    for inputs, targets in epoch_bar:
      inputs = inputs.to(device)
      targets = targets.to(device)

      optimizer.zero_grad()

      with autocast():
        outputs = model(inputs)
        loss_value = criterion(outputs, targets)

      scaler.scale(loss_value).backward()
      scaler.step(optimizer)
      scaler.update()

      epoch_loss += loss_value.item()
      epoch_bar.set_postfix(loss=f"{loss_value.item():.4f}")

    training_bar.set_postfix(avg_loss=f"{epoch_loss / len(loader):.4f}")

  torch.save(model.state_dict(), model_path)

  runtime = time.time() - start
  h = runtime // 3600
  m = runtime // 60 % 60
  s = runtime % 60
  print(f"Training time: {h:02.0f}:{m:02.0f}:{s:02.0f}")

if __name__ == "__main__":
  train()
