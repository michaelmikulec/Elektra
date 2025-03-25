import os
import torch
import pandas as pd
from torch.utils.data import Dataset
import torch.nn as nn
import torch.nn.functional as F

class SpecDataset(Dataset):
  def __init__(self, data_folder, label_index):
    super().__init__()
    self.data_folder = data_folder
    self.data_files = sorted([
      os.path.join(data_folder, f)
      for f in os.listdir(data_folder)
      if f.endswith(".parquet")
    ])
    self.label_index = label_index
    self.str_labels = []
    for fp in self.data_files:
      str_label = self._extract_label_from_filename(fp)
      if str_label not in self.label_index:
        raise ValueError(
          f"File '{fp}' has label '{str_label}', "
          f"but it's not in {list(self.label_index.keys())}."
        )
      self.str_labels.append(str_label)

  def _extract_label_from_filename(self, filepath):
    base_name = os.path.basename(filepath)  
    parts = base_name.split("_")
    if len(parts) != 4:
      raise ValueError(
        f"Filename '{base_name}' does not have exactly three underscores "
        "in the expected format: '{int_label}_{str_label}_{id}_{sub_id}.parquet'"
      )
    str_label = parts[1]
    return str_label

  def __len__(self):
    return len(self.data_files)

  def __getitem__(self, idx):
    fp = self.data_files[idx]
    df = pd.read_parquet(fp)
    data_sample = torch.tensor(df.values, dtype=torch.float32)
    str_label = self.str_labels[idx]
    label_idx = self.label_index[str_label]
    label_tensor = torch.tensor(label_idx, dtype=torch.long)
    return data_sample, label_tensor

class SpectrogramCNN(nn.Module):
  def __init__(self, num_classes=6):
    super(SpectrogramCNN, self).__init__()
    self.conv_layers = nn.Sequential(
      nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1),
      nn.BatchNorm2d(16),
      nn.ReLU(),
      nn.MaxPool2d(kernel_size=(2, 2)),
      nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
      nn.BatchNorm2d(32),
      nn.ReLU(),
      nn.MaxPool2d(kernel_size=(2, 2)),
      nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
      nn.BatchNorm2d(64),
      nn.ReLU(),
      nn.MaxPool2d(kernel_size=(2, 2)),
    )
    self.fc_layers = nn.Sequential(
      nn.Linear(in_features=64 * 1 * 1, out_features=128),  
      nn.ReLU(),
      nn.Dropout(p=0.5),
      nn.Linear(in_features=128, out_features=num_classes)
    )

  def forward(self, x):
    x = x.unsqueeze(1) 
    x = self.conv_layers(x)
    bsz, c, h, w = x.shape 
    x = x.view(bsz, -1) 
    if self.fc_layers[0].in_features != c * h * w:
      self.fc_layers[0] = nn.Linear(c * h * w, 128)
    x = self.fc_layers(x) 
    return x
