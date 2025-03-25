import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import numpy as np
from tqdm import tqdm
from eeg_dataset import EEGDataset
from eeg_transformer import EEGTransformer

def train(model, dataloader, criterion, optimizer, device):
  model.train()
  running_loss = 0.0
  correct = 0
  total = 0
  for batch in tqdm(dataloader, desc="train"):
    data, labels = batch
    data = data.to(device)
    labels = labels.to(device)

    outputs = model(data)
    loss = criterion(outputs, labels)

    optimizer.zero_grad()
    loss.backward()
    # Gradient clipping
    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

    running_loss += loss.item() * data.size(0)
    _, pred = torch.max(outputs, 1)
    correct += (pred == labels).sum().item()
    total += labels.size(0)

  return running_loss / total, correct / total

def validate(model, dataloader, criterion, device):
  model.eval()
  running_loss = 0.0
  correct = 0
  total = 0
  with torch.no_grad():
    for batch in tqdm(dataloader, desc="val"):
      data, labels = batch
      data = data.to(device)
      labels = labels.to(device)

      outputs = model(data)
      loss = criterion(outputs, labels)

      running_loss += loss.item() * data.size(0)
      _, pred = torch.max(outputs, 1)
      correct += (pred == labels).sum().item()
      total += labels.size(0)

  return running_loss / total, correct / total

def main():
  torch.set_num_threads(22)
  torch.set_num_interop_threads(1)

  MODEL = "transfm.pth"
  MODELS_FOLDER = "G:/My Drive/fau/egn4952c_spring_2025/deep_learning/dev/models/"
  DATA_FOLDER = "G:/My Drive/fau/egn4952c_spring_2025/data/600eeg"
  LABEL_INDEX = {"Seizure":0,"LRDA":1,"GRDA":2,"LPD":3,"GPD":4,"Other":5}
  INPUT_DIM = 20
  MODEL_DIM = 128
  NUM_HEADS = 4
  NUM_LAYERS = 2
  DIM_FEEDFORWARD = 256
  DROPOUT = 0.1
  NUM_CLASSES = len(LABEL_INDEX)
  BATCH_SIZE = 1
  LEARNING_RATE = 0.000001   # Lower LR to reduce NaN risk
  NUM_WORKERS = 16        # Increase for faster data loading
  NUM_EPOCHS = 1
  VAL_SPLIT = 0.2
  SEED = 42

  # Reproducibility
  torch.manual_seed(SEED)
  np.random.seed(SEED)

  # Device setup
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

  # Dataset and Dataloaders
  dataset = EEGDataset(data_folder=DATA_FOLDER, label_index=LABEL_INDEX)
  val_size = int(len(dataset) * VAL_SPLIT)
  train_size = len(dataset) - val_size
  train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

  train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
  val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

  # Build model
  transfm = EEGTransformer(
    input_dim=INPUT_DIM,
    model_dim=MODEL_DIM,
    num_heads=NUM_HEADS,
    num_layers=NUM_LAYERS,
    dim_feedforward=DIM_FEEDFORWARD,
    dropout=DROPOUT,
    num_classes=NUM_CLASSES
  )

  # If multiple GPUs, wrap in DataParallel
  if torch.cuda.device_count() > 1:
    transfm = nn.DataParallel(transfm)
    print("DataParallel in use.")

  transfm.to(device)

  # load checkpoint if exists, for incremental training
  checkpoint_path=MODELS_FOLDER+MODEL
  if os.path.exists(checkpoint_path):
    print(f"loading existing model from {checkpoint_path}...")
    state_dict = torch.load(checkpoint_path, map_location=device)
    # if using DataParallel, the state_dict keys might need prefix adjustment
    if isinstance(transfm, nn.DataParallel):
      transfm.module.load_state_dict(state_dict)
    else:
      transfm.load_state_dict(state_dict)
  else:
    print("no existing model to load.")

  # Define loss & optimizer
  criterion = nn.CrossEntropyLoss()
  optimizer = optim.Adam(transfm.parameters(), lr=LEARNING_RATE)

  # Training loop
  for epoch in range(NUM_EPOCHS):
    train_loss, train_acc = train(transfm, train_loader, criterion, optimizer, device)
    val_loss, val_acc = validate(transfm, val_loader, criterion, device)

    print(f"Epoch[{epoch+1}/{NUM_EPOCHS}]\n"
          f"TrainLoss:{train_loss:.4f}, TrainAcc:{train_acc:.4f} | "
          f"ValLoss:{val_loss:.4f}, ValAcc:{val_acc:.4f}")

  # save final model; if using DataParallel, save the underlying .module state_dict
  print("saving final model to transfm.pth")
  if isinstance(transfm, nn.DataParallel):
    torch.save(transfm.module.state_dict(), checkpoint_path)
  else:
    torch.save(transfm.state_dict(), checkpoint_path)

  print("training complete.")

if __name__ == "__main__":
  main()
