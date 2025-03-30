import os
import pandas as pd
import torch
from torch.utils.data import Dataset
import random
from typing import List

def normalize_sample(tensor: torch.Tensor) -> torch.Tensor:
  eps = 1e-8
  mean = tensor.mean()
  std = tensor.std()
  return (tensor - mean) / (std + eps)

def select_subset(dataset_directory: str, x: int) -> List[str]:
  label_to_files = {str(label): [] for label in range(6)}
  for fname in os.listdir(dataset_directory):
    if fname.endswith(".parquet"):
      label = fname.split("_")[0]
      if label in label_to_files:
        label_to_files[label].append(os.path.join(dataset_directory, fname))
  selected_files = []
  for label, files in label_to_files.items():
    if len(files) < x:
      raise ValueError(f"Not enough files for label {label}: requested {x}, found {len(files)}")
    selected_files.extend(random.sample(files, x))
  return selected_files

class EEGDataset(Dataset):
  def __init__(self, file_paths: List[str], transform=None):
    self.dataset_files = file_paths
    self.transform = transform

  def __len__(self):
    return len(self.dataset_files)

  def __getitem__(self, idx):
    file_path = self.dataset_files[idx]
    int_label = int(os.path.basename(file_path).split('_')[0])
    df = pd.read_parquet(file_path)
    data_tensor = torch.tensor(df.values, dtype=torch.float32)
    if self.transform:
      data_tensor = self.transform(data_tensor)
    return data_tensor, int_label


if __name__ == "__main__":
  subset = select_subset("./data/training_data/eegs/", 1000)
  print(subset)
