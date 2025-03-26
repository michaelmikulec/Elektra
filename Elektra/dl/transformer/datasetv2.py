import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from config import config

class EEGDataset(Dataset):
  def __init__(self, dir, transform=None):
    self.dir = dir
    self.files = [ os.path.join(dir, fname) for fname in os.listdir(dir) if fname.endswith('.parquet') ]
    self.transform = transform

  def __len__(self): 
    return len(self.files)

  def __getitem__(self, idx):
    file_path = self.files[idx]
    label = int(os.path.basename(file_path).split('_')[0])
    df = pd.read_parquet(file_path)
    data_tensor = torch.tensor(
      data = df.iloc[:, :-2].values, 
      dtype = torch.float32
    )
    if self.transform: 
      data_tensor = self.transform(data_tensor)
    return data_tensor, label

if __name__ == '__main__':
  training_data = config['training_data']
  eeg_dataset = EEGDataset(
    dir = training_data
  )
  eeg_dataloader = DataLoader(
    dataset = eeg_dataset,
    batch_size = config['batch_size'],
    shuffle = True,
    num_workers = config['num_workers']
  )
  for batch_data, batch_labels in eeg_dataloader:
    print(f"Data batch shape: {batch_data.shape}")
    print(f"Labels: {batch_labels}")