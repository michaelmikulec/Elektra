import os
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader

def getDataloader(
  index_csv,
  data_dir,
  num_samples = 60,
  batch_size  = 32,
  shuffle     = True,
  num_workers = 1,
  label_col   = "expert_consensus"
):
  class EEGDataset(Dataset):
    def __init__(self, df, data_dir):
      self.df       = df
      self.data_dir = data_dir

    def __len__(self):
      return len(self.df)

    def __getitem__(self, idx):
      row    = self.df.iloc[idx]
      sample = torch.load(os.path.join(self.data_dir, row["filename"]))
      return sample["eeg"], sample["spectrogram"], sample["label"]

  df = pd.read_csv(index_csv)
  df = df[df["status"] == "success"]

  if df.empty:
    raise ValueError("No rows with status == 'success'.")

  classes          = df[label_col].unique()
  num_classes      = len(classes)
  base             = num_samples // num_classes
  remainder        = num_samples % num_classes
  per_class_counts = {label: base + (1 if i < remainder else 0) for i, label in enumerate(classes)}
  samples          = []

  for label in classes:
    group = df[df[label_col] == label]
    count = per_class_counts[label]
    n     = min(len(group), count)

    if n < count:
      print(f"Not enough samples for label '{label}', using {n} instead of {count}")

    samples.append(group.sample(n=n, random_state=42))

  subset_df = pd.concat(samples).reset_index(drop=True)
  dataset   = EEGDataset(subset_df, data_dir)

  return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
