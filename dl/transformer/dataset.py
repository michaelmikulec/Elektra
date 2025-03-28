import os
import pandas as pd
import torch
from torch.utils.data import Dataset

class EEGDataset(Dataset):
  def __init__(self, dataset_directory: str, transform=None):
    self.dataset_directory: str = dataset_directory
    self.dataset_files: list[str] = sorted([os.path.join(dataset_directory, fname) for fname in os.listdir(dataset_directory) if fname.endswith('.parquet')])
    self.transform = transform

  def __len__(self) -> int: 
    return len(self.dataset_files)

  def __getitem__(self, idx) -> tuple[torch.Tensor, int]:
    file_path: str = self.dataset_files[idx]
    int_label: int = int(os.path.basename(file_path).split('_')[0])
    df: pd.DataFrame = pd.read_parquet(file_path)
    data_tensor: torch.Tensor = torch.tensor(
      data = df.values, 
      dtype = torch.float32
    )
    return data_tensor, int_label
