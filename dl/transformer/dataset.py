import os
import pandas as pd
import torch
from torch.utils.data import Dataset

class EEGDataset(Dataset):
  def __init__(self, data_folder, label_index):
    super().__init__()
    self.data_folder = data_folder
    self.data_files = sorted([
      os.path.join(data_folder, f)
      for f in os.listdir(data_folder)
      if f.endswith(".parquet")
    ])
    self.label_index = label_index
    self.int_labels = []
    self.str_labels = []
    for fp in self.data_files:
      i, s = self.parse_label(fp)
      if s not in label_index:
        raise ValueError(str(s))
      self.int_labels.append(i)
      self.str_labels.append(s)

  def parse_label(self, fp):
    b = os.path.basename(fp)
    p = b.split("_")
    return int(p[0]), p[1]

  def __len__(self):
    return len(self.data_files)

  def __getitem__(self, idx):
    fp = self.data_files[idx]
    df = pd.read_parquet(fp)
    x = torch.tensor(df.values, dtype=torch.float32)
    s = self.str_labels[idx]
    y = self.label_index[s]
    y = torch.tensor(y, dtype=torch.long)
    return x, y
