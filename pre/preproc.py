import os
from typing import List
import pandas as pd
import numpy as np

def find_parquet_file(folder, file_id):
  path = os.path.join(folder, f"{file_id}.parquet")
  return path if os.path.exists(path) else None

def extract_event_window(df, offset, rows_per_second, event_duration):
  start = int(offset * rows_per_second)
  stop = start + int(rows_per_second * event_duration)
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

def apply_notch_filter(input_dir, output_dir, fs=200, notch_freq=60.0, quality_factor=30.0):
  for f_name in os.listdir(input_dir):
    f_path = os.path.join(input_dir, f_name)
    print(f_path)
    df = pd.read_parquet(f_path)
    df.drop(columns=["EKG", "O2"], inplace=True)
    b, a = iirnotch(notch_freq, quality_factor, fs)
    filtered_df = df.copy()
    for col in df.columns:
      filtered_df[col] = filtfilt(b, a, df[col])
    filtered_df.to_parquet(os.path.join(output_dir, f_name), index=False)

def delete_corrupt_parquet_files(directory, log_path):
  deleted_files = []
  for root, _, files in os.walk(directory):
    for file in files:
      if file.endswith(".parquet"):
        full_path = os.path.join(root, file)
        try:
          pd.read_parquet(full_path)
        except pyarrow.lib.ArrowInvalid as e:
          if "Parquet magic bytes not found in footer" in str(e):
            print(f"Deleting corrupted file: {full_path}")
            os.remove(full_path)
            deleted_files.append(full_path)
          else:
            print(f"Skipped {full_path} due to different ArrowInvalid error:\n{e}")
        except Exception as e:
          print(f"Skipped {full_path} due to unexpected error:\n{e}")
  if deleted_files:
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    if not os.path.exists(log_path):
      open(log_path, "w").close()
    with open(log_path, "a", encoding="utf-8") as log_file:
      for path in deleted_files:
        log_file.write(f"[{datetime.now().isoformat()}] Deleted: {path}\n")
  print(f"\nDeleted {len(deleted_files)} corrupted parquet files.")
  return deleted_files

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

def preprocess_eeg_dataset():
  pass

if __name__ == "__main__":
  fnames = os.listdir("./data/unprocessed_data/eegs/")
  ids: List[int] = [int(os.path.basename(fname).split('.')[0]) for fname in fnames]

  metadata = pd.read_csv("./data/metadata.csv")
  e_ids: List[int] = []
  for index, row in metadata.iterrows():
    e_id: int = int(row['eeg_id'])
    e_sub_id: int = int(row['eeg_sub_id'])
    e_offset: int = int(row['eeg_label_offset_seconds'])
    s_id: int = int(row['spectrogram_id'])
    s_sub_id: int = int(row['spectrogram_sub_id'])
    s_offset: int = int(row['spectrogram_label_offset_seconds'])
    label_id: int = int(row['label_id'])
    label: str = str(row['expert_consensus'])

  for file in os.listdir(fnames):
    print(file)
   