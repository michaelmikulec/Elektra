import os, csv, glob
from tqdm.auto import tqdm
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
  def __init__(self, c):
    super().__init__()
    self.conv1 = nn.Conv2d(c, c, 3, padding=1)
    self.bn1   = nn.BatchNorm2d(c)
    self.conv2 = nn.Conv2d(c, c, 3, padding=1)
    self.bn2   = nn.BatchNorm2d(c)
    self.relu  = nn.ReLU()

  def forward(self, x):
    out = self.conv1(x)
    out = self.bn1(out)
    out = self.relu(out)
    out = self.conv2(out)
    out = self.bn2(out)
    return self.relu(x + out)

class SpectrogramCNN(nn.Module):
  def __init__(self, in_channels=19, num_classes=6, dropout=0.1):
    super().__init__()
    self.features = nn.Sequential(
      nn.Conv2d(in_channels, 32, 3, padding=1),
      nn.BatchNorm2d(32),
      nn.ReLU(),
      nn.Dropout2d(dropout),
      nn.MaxPool2d(2),
      ResidualBlock(32),
      nn.Conv2d(32, 64, 3, padding=1),
      nn.BatchNorm2d(64),
      nn.ReLU(),
      nn.Dropout2d(dropout),
      nn.MaxPool2d(2),
      ResidualBlock(64),
      nn.Conv2d(64, 128, 3, padding=1),
      nn.BatchNorm2d(128),
      nn.ReLU(),
      nn.Dropout2d(dropout),
      nn.MaxPool2d(2),
      ResidualBlock(128)
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

# class SpectrogramCNN(nn.Module):
#   def __init__(self, in_channels=19, num_classes=6):
#     super().__init__()
#     self.features = nn.Sequential(
#       nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
#       nn.ReLU(),
#       nn.MaxPool2d(2),
#       nn.Conv2d(32, 64, kernel_size=3, padding=1),
#       nn.ReLU(),
#       nn.MaxPool2d(2),
#       nn.Conv2d(64, 128, kernel_size=3, padding=1),
#       nn.ReLU(),
#       nn.MaxPool2d(2)
#     )
#     self.classifier = nn.Sequential(
#       nn.AdaptiveAvgPool2d((1, 1)),
#       nn.Flatten(),
#       nn.Linear(128, num_classes)
#     )

#   def forward(self, x):
#     x = self.features(x)
#     x = self.classifier(x)
#     return x

class SpectrogramDataset(Dataset):
  def __init__(self, folder):
    self.files = sorted(glob.glob(os.path.join(folder, 'SPEC_*.pt')))

  def __len__(self):
    return len(self.files)

  def __getitem__(self, idx):
    data = torch.load(self.files[idx])
    return data['spectrogram'], data['label']

def train(
  model, 
  trainDL, 
  valDL, 
  optimizer, 
  criterion, 
  device, 
  epochs, 
  modelPath='checkpoint.pth', 
  trainingStats='training_stats.csv'
):
  model.to(device)
  if os.path.isfile(modelPath):
    model.load_state_dict(torch.load(modelPath, map_location=device))
    print(f"Resumed from checkpoint {modelPath}")

  print("Training...")
  best_val = float('inf')
  for epoch in range(1, epochs+1):
    print(f'Starting epoch {epoch}/{epochs}')
    model.train()
    tl = 0.0
    for xb, yb in tqdm(trainDL, desc=f'Train Epoch {epoch}/{epochs}', leave=False):
      xb, yb = xb.to(device), yb.to(device)
      optimizer.zero_grad(set_to_none=True)
      out  = model(xb)
      loss = criterion(out, yb)
      loss.backward()
      optimizer.step()
      tl += loss.item() * xb.size(0)
    tl /= len(trainDL.dataset)

    print("Evaluating on validation set...")
    model.eval()
    vl, correct = 0.0, 0
    for xb, yb in tqdm(valDL, desc=f'Val   Epoch {epoch}/{epochs}', leave=False):
      xb, yb  = xb.to(device), yb.to(device)
      out     = model(xb)
      loss    = criterion(out, yb)
      vl      += loss.item() * xb.size(0)
      correct += (out.argmax(1) == yb).sum().item()
    vl  /= len(valDL.dataset)
    acc  = correct / len(valDL.dataset)

    print(f'Epoch {epoch}/{epochs} train_loss={tl:.4f} val_loss={vl:.4f} val_acc={acc:.4f}')
    saved = 0
    if vl < best_val:
      saved = 1
      best_val = vl
      torch.save(model.state_dict(), modelPath)
      print(f"Model saved to {modelPath}")

    with open(trainingStats, 'a') as f:
      writer = csv.DictWriter(f, fieldnames=["epoch", "training_loss", "validation_loss", "validation_accuracy", "saved"])
      if f.tell() == 0:
        writer.writeheader()
      writer.writerow({"epoch": epoch, "training_loss": tl, "validation_loss": vl, "validation_accuracy": acc, "saved": saved})

  print("Training complete.")

if __name__ == '__main__':
  torch.manual_seed(42)

  dataDir       = "data/prep/specs"
  modelBaseName = "cnn1"
  modelPath     = f"models/{modelBaseName}.pth"
  trainingStats = f"logs/{modelBaseName}_training_stats.csv"
  numWorkers    = 16
  batchSize     = 150
  epochs        = 50
  dataset       = SpectrogramDataset(dataDir)
  lenDS         = len(dataset)
  lenTrainSplit = int(0.6 * lenDS)
  lenValSplit   = int(0.2 * lenDS)
  lenTestSplit  = lenDS - lenTrainSplit - lenValSplit

  trainDS, valDS, testDS = random_split(dataset, [lenTrainSplit, lenValSplit, lenTestSplit])
  trainDL = DataLoader(trainDS, batch_size=batchSize, shuffle=True,  num_workers=numWorkers, pin_memory=True)
  valDL   = DataLoader(valDS,   batch_size=batchSize, shuffle=False, num_workers=numWorkers, pin_memory=True)
  testDL  = DataLoader(testDS,  batch_size=batchSize, shuffle=False, num_workers=numWorkers, pin_memory=True)
  device  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  model   = SpectrogramCNN()
  opt     = torch.optim.AdamW(model.parameters(), lr=1e-4)
  crit    = torch.nn.CrossEntropyLoss()
  train(
    model         = model,
    trainDL       = trainDL,
    valDL         = valDL,
    optimizer     = opt,
    criterion     = crit,
    device        = device,
    epochs        = epochs,
    modelPath     = modelPath,
    trainingStats = trainingStats
  )

