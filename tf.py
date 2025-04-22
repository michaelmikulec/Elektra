import os, csv, math, glob
import pandas as pd
from tqdm.auto import tqdm
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

class EEGTransformer(nn.Module):
  def __init__(
    self,
    maxSeqLen      = 2000,
    numChannels    = 19,
    dimModel       = 128,
    dimFeedForward = 128,
    dropout        = 0.1,
    numHeads       = 8,
    numLayers      = 6,
    numClasses     = 6
  ):
    super().__init__()
    self.input_proj = nn.Linear(numChannels, dimModel)
    pe               = torch.zeros(maxSeqLen, dimModel)
    pos              = torch.arange(maxSeqLen).unsqueeze(1)
    div_term         = torch.exp(torch.arange(0, dimModel, 2) * -(math.log(10000.0) / dimModel))
    pe[:, 0::2]      = torch.sin(pos * div_term)
    pe[:, 1::2]      = torch.cos(pos * div_term)
    self.register_buffer('pe', pe.unsqueeze(0))

    encoder_layer = nn.TransformerEncoderLayer(
      d_model         = dimModel,
      nhead           = numHeads,
      dim_feedforward = dimFeedForward,
      dropout         = dropout,
      batch_first     = True
    )
    self.encoder    = nn.TransformerEncoder(encoder_layer, num_layers=numLayers)
    self.norm       = nn.LayerNorm(dimModel)
    self.classifier = nn.Sequential(
      nn.Linear(dimModel, dimModel//2),
      nn.ReLU(),
      nn.Dropout(dropout),
      nn.Linear(dimModel//2, numClasses)
    )

  def forward(self, x):
    x = self.input_proj(x)
    x = x + self.pe[:, :x.size(1)]
    x = self.encoder(x)
    x = x.mean(dim=1)
    x = self.norm(x)
    return self.classifier(x)

# class EEGDataset(Dataset):
#   def __init__(self, folder, classes):
#     self.files   = [f for f in os.listdir(folder) if f.endswith('.parquet')]
#     self.folder  = folder
#     self.cls2idx = {c:i for i,c in enumerate(classes)}

#   def __len__(self):
#     return len(self.files)

#   def __getitem__(self, idx):
#     fn        = self.files[idx]
#     label_str = fn.split('_')[-1].split('.')[0]
#     label_idx = self.cls2idx[label_str]
#     df        = pd.read_parquet(os.path.join(self.folder, fn))
#     data      = torch.from_numpy(df.values).float()
#     return data, torch.tensor(label_idx).long()

class EEGDataset(Dataset):
  def __init__(self, folder):
    self.files = sorted(glob.glob(os.path.join(folder, '*.pt')))

  def __len__(self):
    return len(self.files)

  def __getitem__(self, idx):
    sample = torch.load(self.files[idx])
    data   = sample['data']   
    label  = sample['label']
    return data, label

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
  print("Training...")
  model.to(device)
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

  data           = "data/prep/eegs_pt"
  modelBaseName  = "t5"
  modelPath      = f'models/{modelBaseName}.pth'
  trainingStats  = f'logs/{modelBaseName}_training_stats.csv'
  numWorkers     = 15
  batchSize      = 100
  numChannels    = 19 
  maxSeqLen      = 2000
  dimModel       = 256
  dimFeedForward = 512
  dropout        = 0.1
  numHeads       = 8
  numLayers      = 6
  numClasses     = 6
  epochs         = 50

  dataset        = EEGDataset(data)
  lenDS          = len(dataset)
  lenTrainSplit  = int(0.6 * lenDS)
  lenValSplit    = int(0.2 * lenDS)
  lenTestSplit   = lenDS - lenTrainSplit - lenValSplit

  trainDS, valDS, testDS = random_split(dataset, [lenTrainSplit, lenValSplit, lenTestSplit])
  trainDL = DataLoader(trainDS, batch_size=batchSize, shuffle=True,  num_workers=numWorkers, pin_memory=True)
  valDL   = DataLoader(valDS,   batch_size=batchSize, shuffle=False, num_workers=numWorkers, pin_memory=True)
  testDL  = DataLoader(testDS,  batch_size=batchSize, shuffle=False, num_workers=numWorkers, pin_memory=True)
  device  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  model   = EEGTransformer( numChannels=numChannels, dimModel=dimModel, numHeads=numHeads, numLayers=numLayers, dimFeedForward=dimFeedForward, dropout=dropout, numClasses=numClasses, maxSeqLen=maxSeqLen)
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
