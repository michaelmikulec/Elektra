import os
import json
import torch
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from tqdm import tqdm

# DIRS
EEG_DIR  = "data/raw/train_eegs"
SPEC_DIR = "data/raw/train_spectrograms"
PREP_DIR = "data/prep"

# FILES
RAW_META_PATH  = "data/raw/train.csv"
PREP_META_PATH = "data/prep.csv"
EEG_META_JSON  = "data/eeg.json"
SPEC_META_JSON = "data/spec.json"

def get_eeg_event_window(eeg_df, offset_sec):
  start = int((offset_sec + 20) * 200)
  end   = start + 2000
  data  = eeg_df.iloc[start:end].values.astype("float32")
  if data.shape != (2000, 20):
    raise ValueError()
  return torch.tensor(vectorized_fill_and_standardize(data))

def get_spec_event_window(spec_df, offset_sec):
  spec_data = spec_df.iloc[:, 1:].values.astype("float32")
  center    = int((offset_sec + 300 - 1) // 2)
  start     = center - 2
  end       = center + 3
  data      = spec_data[start:end]
  if data.shape != (5, 400): 
    raise ValueError()
  return torch.tensor(vectorized_fill_and_standardize(data))

raw_metadata = pd.read_csv(RAW_META_PATH)
eeg_json = (
  raw_metadata
    .groupby("eeg_id", group_keys=False)        
    .apply(lambda g: g[[
      "eeg_sub_id",
      "eeg_label_offset_seconds",
      "label_id",
      "expert_consensus"
    ]].to_dict("records"), include_groups=False)   
    .to_dict()
)
spec_json = (
  raw_metadata
    .groupby("spectrogram_id", group_keys=False)
    .apply(lambda g: g[[
      "spectrogram_sub_id",
      "spectrogram_label_offset_seconds",
      "label_id",
      "expert_consensus"
    ]].to_dict("records"), include_groups=False)
    .to_dict()
)
with open("eeg_metadata.json", "w") as f:
  f.write(json.dumps(eeg_json, indent=2))

with open("spec_metadata.json", "w") as f:
  f.write(json.dumps(spec_json, indent=2))

# for index, row in tqdm(raw_metadata.iterrows(), total=len(raw_metadata), desc="Preprocessing..."):
#   eeg_id:int                           = int(row["eeg_id"])
#   eeg_sub_id:int                       = int(row["eeg_sub_id"])
#   eeg_label_offset_seconds:int         = int(row["eeg_label_offset_seconds"])
#   spectrogram_id:int                   = int(row["spectrogram_id"])
#   spectrogram_sub_id:int               = int(row["spectrogram_sub_id"])
#   spectrogram_label_offset_seconds:int = int(row["spectrogram_label_offset_seconds"])
#   label_id:int                         = int(row["label_id"])
#   patient_id:int                       = int(row["patient_id"])
#   expert_consensus:str                 = str(row["expert_consensus"]).lower()
#   seizure_vote:int                     = int(row["seizure_vote"])
#   lpd_vote:int                         = int(row["lpd_vote"])
#   gpd_vote:int                         = int(row["gpd_vote"])
#   lrda_vote:int                        = int(row["lrda_vote"])
#   grda_vote:int                        = int(row["grda_vote"])
#   other_vote:int                       = int(row["other_vote"])

#   eeg_df  = pq.read_table(os.path.join(EEG_DIR, f"{eeg_id}.parquet")).to_pandas()
#   spec_df = pq.read_table(os.path.join(SPEC_DIR, f"{spectrogram_id}.parquet")).to_pandas()