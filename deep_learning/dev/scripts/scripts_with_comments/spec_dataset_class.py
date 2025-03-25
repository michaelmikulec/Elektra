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
        
        # We'll store two parallel lists:
        #  (1) the path to each .parquet file (data_files)
        #  (2) the string label extracted from that filename (str_labels).
        self.str_labels = []
        
        # Extract labels from each file and validate them
        for fp in self.data_files:
            str_label = self._extract_label_from_filename(fp)
            
            if str_label not in self.label_index:
                raise ValueError(
                    f"File '{fp}' has label '{str_label}', "
                    f"but it's not in {list(self.label_index.keys())}."
                )
            self.str_labels.append(str_label)

    def _extract_label_from_filename(self, filepath):
        base_name = os.path.basename(filepath)  # e.g., "123_Seizure_001_02.parquet"
        parts = base_name.split("_")
        
        if len(parts) != 4:
            raise ValueError(
                f"Filename '{base_name}' does not have exactly three underscores "
                "in the expected format: '{int_label}_{str_label}_{id}_{sub_id}.parquet'"
            )
        
        # The second element is the string label (e.g. "Seizure")
        str_label = parts[1]
        return str_label

    def __len__(self):
        return len(self.data_files)

    def __getitem__(self, idx):
        """
        1. Reads the parquet file at index `idx` (shape: [num_time_bins, 401]).
        2. Converts it to a torch.Tensor of dtype float32.
        3. Retrieves the corresponding string label, maps it to an integer index, 
           and returns both data + label.
        """
        fp = self.data_files[idx]
        
        # Load the Parquet file into a pandas DataFrame, shape: [num_time_bins, 401]
        df = pd.read_parquet(fp)
        
        # Convert to torch tensor; shape: [num_time_bins, 401]
        # You can optionally rearrange dimensions if needed (e.g., transpose).
        data_sample = torch.tensor(df.values, dtype=torch.float32)
        
        # Get the string label and map it to integer
        str_label = self.str_labels[idx]
        label_idx = self.label_index[str_label]
        label_tensor = torch.tensor(label_idx, dtype=torch.long)
        
        # Return the spectrogram and the label
        return data_sample, label_tensor


