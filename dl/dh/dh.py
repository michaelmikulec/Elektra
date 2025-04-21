import os, torch, pandas as pd
from torch.utils.data import Dataset, DataLoader

class EEGDataset(Dataset):
  def __init__(self, folder, classes):
    self.files   = [f for f in os.listdir(folder) if f.endswith('.parquet')]
    self.folder  = folder
    self.cls2idx = {c:i for i,c in enumerate(classes)}

  def __len__(self):
    return len(self.files)

  def __getitem__(self, idx):
    fn        = self.files[idx]
    label_str = fn.split('_')[-1].split('.')[0]
    label_idx = self.cls2idx[label_str]
    df        = pd.read_parquet(os.path.join(self.folder, fn))
    data      = torch.from_numpy(df.values).float()
    return data, torch.tensor(label_idx).long()

ds = EEGDataset('data/prep/eegs', ['seizure', 'lpd', 'gpd', 'lrda', 'grda', 'other'])
dl = DataLoader(ds, batch_size=16, shuffle=True, num_workers=16, pin_memory=True)
print(len(dl))