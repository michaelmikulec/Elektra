import os
import json
import pandas as pd

with open("eeg_metadata.json") as f:
  eeg_map = json.load(f)

for eeg_id, windows in eeg_map.items():
  df = pd.read_parquet(f"data/raw/train_eegs/{eeg_id}.parquet")
  for w in windows:
    start = (int(w["eeg_label_offset_seconds"]) + 20) * 200
    end   = start + 10*200
    dfw   = df.iloc[start:end]
    dfw.to_parquet(f"data/prep2/eeg/{eeg_id}_{w['eeg_sub_id']}.parquet", index=False)