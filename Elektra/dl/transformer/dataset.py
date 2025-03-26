import os
import torch
import pandas as pd
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
    for file_path in self.data_files:
      int_label, str_label = self.parse_label(file_path)
      if str_label not in label_index:
        raise ValueError(str_label)
      self.int_labels.append(int_label)
      self.str_labels.append(str_label)
  
  def parse_label(self, file_path):
    base_name = os.path.basename(file_path)
    parts = base_name.split("_")
    return int(parts[0]), parts[1]
  
  def __len__(self):
    return len(self.data_files)
  
  def __getitem__(self, index):
    file_path = self.data_files[index]
    df = pd.read_parquet(file_path)
    data_tensor = torch.tensor(df.values, dtype=torch.float32)
    label_str = self.str_labels[index]
    label_int = self.label_index[label_str]
    label_tensor = torch.tensor(label_int, dtype=torch.long)
    return data_tensor, label_tensor
