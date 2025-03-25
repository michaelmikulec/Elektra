import os
import pandas as pd
import numpy as np
import time

def find_parquet_file(folder, file_id):
  file_path = os.path.join(folder, f"{file_id}.parquet")
  return file_path if os.path.exists(file_path) else None

def extract_event_window(df, seconds_offset, rows_per_second, event_duration):
  row_offset = int(seconds_offset * rows_per_second)
  event_rows = int(rows_per_second * event_duration)
  return df.iloc[row_offset : row_offset + event_rows]

def quality_check(id, df):
  quality = True
  if df.empty: 
    print(f"Error with id {id}: empty file")
    quality = False 
  if df.isna().values.any(): 
    print(f"Error with id {id}: contains nans")
    quality = False 
  if np.isinf(df.values).any(): 
    print(f"Error with id {id}: contains infinite values")
    quality = False
  return quality

def label_data():
  wd = os.getcwd()
  unlabeled_eeg_dir = os.path.join(wd, "data", "train_eegs")
  unlabeled_spec_dir = os.path.join(wd, "data", "train_spectrograms")
  labeled_eeg_dir = os.path.join(wd, "data", "labeled_train_eegs")
  labeled_spec_dir = os.path.join(wd, "data", "labeled_train_specs")
  metadata_path = os.path.join(wd, "data", "train.csv")
  metadata = pd.read_csv(metadata_path)

  for index, row in metadata.iterrows():
    eeg_id = row["eeg_id"]
    eeg_sub_id = row["eeg_sub_id"]
    eeg_offset = row["eeg_label_offset_seconds"]
    eeg_rows_per_second = 200
    eeg_event_duration = 50

    spec_id  = row["spectrogram_id"]
    spec_sub_id = row["spectrogram_sub_id"]
    spec_offset = row["spectrogram_label_offset_seconds"]
    spec_rows_per_second = 0.5
    spec_event_duration = 600

    label = row["expert_consensus"]
    label_index = { "Seizure": 0, "LRDA": 1, "GRDA": 2, "LPD": 3, "GPD": 4, "Other": 5 }    
    label_id = label_index[label]

    eeg_file_path = find_parquet_file(unlabeled_eeg_dir, eeg_id)
    try:
      eeg_df = pd.read_parquet(eeg_file_path)
      eeg_event_df = extract_event_window(eeg_df, eeg_offset, eeg_rows_per_second, eeg_event_duration)
      eeg_output_path = os.path.join(labeled_eeg_dir, f"{label_id}_{label}_{eeg_id}_{eeg_sub_id}.parquet")
    except:
      print(f"Error handling EEG {eeg_id}")
    if quality_check(eeg_id, eeg_event_df):
      eeg_event_df.to_parquet(eeg_output_path, index=False)

    spec_file_path = find_parquet_file(unlabeled_spec_dir, spec_id)
    try:
      spec_df = pd.read_parquet(spec_file_path)
      spec_event_df = extract_event_window(spec_df, spec_offset, spec_rows_per_second, spec_event_duration)
      spec_output_path = os.path.join(labeled_spec_dir, f"{label_id}_{label}_{spec_id}_{spec_sub_id}.parquet")
    except:
      print(f"Error handling Spectrogram {spec_id}")
    if quality_check(spec_id, spec_event_df):
      spec_event_df.to_parquet(spec_output_path, index=False)

if __name__ == "__main__":
  start = time.time()
  label_data()
  end = time.time()
  print(f"Labeling took {end - start} seconds")