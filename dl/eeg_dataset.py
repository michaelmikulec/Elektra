import torch
from torch.utils.data import Dataset
import os
import pandas as pd

class EEGDataset(Dataset):
  def __init__(self,data_folder,label_index):
    self.data_folder=data_folder
    self.data_files=sorted([os.path.join(data_folder,f) for f in os.listdir(data_folder) if f.endswith(".parquet")])
    self.label_index=label_index
    self.int_labels=[]
    self.str_labels=[]
    for fp in self.data_files:
      int_label,str_label=self.extract_labels(fp)
      if str_label not in label_index:
        raise ValueError(f"File '{fp}' has string label '{str_label}', but it's not in {list(label_index.keys())}")
      self.int_labels.append(int_label)
      self.str_labels.append(str_label)

  def extract_labels(self,filepath):
    base=os.path.basename(filepath)
    parts=base.split("_")
    int_label=int(parts[0])
    str_label=parts[1]
    return int_label,str_label

  def __len__(self):
    return len(self.data_files)

  def __getitem__(self,idx):
    fp=self.data_files[idx]
    df=pd.read_parquet(fp)
    data_sample=torch.tensor(df.values,dtype=torch.float32)
    mapped_label=self.label_index[self.str_labels[idx]]
    label_tensor=torch.tensor(mapped_label,dtype=torch.long)
    return data_sample,label_tensor
