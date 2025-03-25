import os
import torch
import pandas as pd
from torch.utils.data import Dataset

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


