import os
import json
import pandas as pd

with open("spec_metadata.json") as f:
  spec_map = json.load(f)

for spec_id, windows in spec_map.items():
  df = pd.read_parquet(f"data/raw/train_spectrograms/{spec_id}.parquet").iloc[:, 1:]
  for w in windows:
    start = (int(w["spectrogram_label_offset_seconds"]) // 2) + 150
    end   = start 
    dfw   = df.iloc[start:end]
    dfw.to_parquet(f"data/prep2/spec/{w['expert_consensus'].tolower()}_{spec_id}_{w['spectrogram_sub_id']}.parquet", index=False)
