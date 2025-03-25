import torch
from torch.utils.data import Dataset
import os
import pandas as pd

class EEGDataset(Dataset):
    def __init__(self, data_folder, labels_list):
        self.data_folder = data_folder
        self.data_files = sorted([
            os.path.join(data_folder, f)
            for f in os.listdir(data_folder)
            if f.endswith(".parquet")
        ])
        self.labels_list = labels_list
        self.label_to_index = {lbl: idx for idx, lbl in enumerate(labels_list)}
        self.num_classes = len(labels_list)
        self.label_strings = []
        for fp in self.data_files:
            label_str = self.extract_label(fp)
            if label_str not in self.label_to_index: raise ValueError(f"File '{fp}' has label '{label_str}', but it's not in {labels_list}")
            self.label_strings.append(label_str)

    def extract_label(self, filepath):
        base = os.path.basename(filepath)
        label_part = base.rsplit("_", 1)[-1].split(".")[0]
        return label_part

    def __len__(self): return len(self.data_files)

    def __getitem__(self, idx):
        fp = self.data_files[idx]
        df = pd.read_parquet(fp)
        data_sample = torch.tensor(df.values, dtype=torch.float32)
        label_str = self.label_strings[idx]
        label_idx = self.label_to_index[label_str]
        label_tensor = torch.tensor(label_idx, dtype=torch.long)
        return data_sample, label_tensor