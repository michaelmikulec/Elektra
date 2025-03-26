import os
import pandas as pd
import numpy as np
import time
from .config import config

def find_parquet_file(folder, file_id):
  path = os.path.join(folder, f"{file_id}.parquet")
  return path if os.path.exists(path) else None

def extract_event_window(df, offset, rows_per_second, duration):
  start = int(offset * rows_per_second)
  stop = start + int(rows_per_second * duration)
  return df.iloc[start:stop]

def quality_check(file_id, df):
  if df.empty:
    print("Error with id", file_id, "empty file")
    return False
  if df.isna().values.any():
    print("Error with id", file_id, "contains nans")
    return False
  if np.isinf(df.values).any():
    print("Error with id", file_id, "contains infinite values")
    return False
  return True

def process_file(input_dir, output_dir, file_id, sub_id, offset, rps, dur, label_id, label_str):
  path = find_parquet_file(input_dir, file_id)
  if not path:
    print("Error handling", file_id)
    return
  try:
    df = pd.read_parquet(path)
    window = extract_event_window(df, offset, rps, dur)
  except:
    print("Error handling", file_id)
    return
  if not quality_check(file_id, window):
    return
  out_path = os.path.join(output_dir, f"{label_id}_{label_str}_{file_id}_{sub_id}.parquet")
  window.to_parquet(out_path, index=False)

def label_data():
  md = pd.read_csv(config["metadata_path"])
  for _, row in md.iterrows():
    lb = row["expert_consensus"]
    lb_id = config["label_index"][lb]
    process_file(
      config["unlabeled_eeg_dir"],
      config["labeled_eeg_dir"],
      row["eeg_id"],
      row["eeg_sub_id"],
      row["eeg_label_offset_seconds"],
      config["eeg_rows_per_second"],
      config["eeg_event_duration"],
      lb_id,
      lb
    )
    process_file(
      config["unlabeled_spec_dir"],
      config["labeled_spec_dir"],
      row["spectrogram_id"],
      row["spectrogram_sub_id"],
      row["spectrogram_label_offset_seconds"],
      config["spec_rows_per_second"],
      config["spec_event_duration"],
      lb_id,
      lb
    )

if __name__ == "__main__":
  start = time.time()
  label_data()
  end = time.time()
  print("Labeling took", end - start, "seconds")
