import os
import json
import pandas as pd
import time

def find_parquet_file(folder, file_id):
  file_path = os.path.join(folder, f"{file_id}.parquet")
  return file_path if os.path.exists(file_path) else None

def extract_event_window(df, seconds_offset, rows_per_second, event_duration):
  row_offset = seconds_offset * rows_per_second
  event_rows = rows_per_second * event_duration
  return df.iloc[row_offset : row_offset + event_rows]

def label_data():
  metadata = pd.read_csv("./data/train.csv")
  unlabeled_eeg_dir = "./data/train_eegs/"
  unlabeled_spec_dir = "./data/train_spectrograms/"
  labeled_eeg_dir = "./data/labeled_train_eegs/"
  labeled_spec_dir = "./data/labeled_train_specs/"

  for index, row in metadata.iterrows():
    eeg_id = row["eeg_id"],
    eeg_sub_id = row["eeg_sub_id"],
    eeg_offset = int(row["eeg_label_offset_seconds"]),
    eeg_rows_per_second = 200,
    eeg_event_duration = 50

    spec_id  = row["spectrogram_id"],
    spec_sub_id = row["spectrogram_sub_id"], 
    spec_offset = int(row["spectrogram_label_offset_seconds"])
    spec_rows_per_second = 0.5,
    spec_event_duration = 600

    label = row["expert_consensus"]
    label_index = { "Seizure": 0, "LRDA": 1, "GRDA": 2, "LPD": 3, "GPD": 4, "Other": 5 }    
    label_id = label_index[label]

    eeg_file_path = find_parquet_file(unlabeled_eeg_dir, eeg_id)
    if eeg_file_path:
      try:
        eeg_df = pd.read_parquet(eeg_file_path)
        eeg_event_df = extract_event_window(eeg_df, eeg_offset, eeg_rows_per_second, eeg_event_duration)
        eeg_output_path = os.path.join(labeled_eeg_dir, f"{label_id}_{label}_{eeg_id}_{eeg_sub_id}.parquet")
        eeg_event_df.to_parquet(eeg_output_path, index=False)
      except:
        print(f"Error handling EEG {eeg_id}")

    spec_file_path = find_parquet_file(unlabeled_spec_dir, spec_id)
    if spec_file_path:
      try:
        spec_df = pd.read_parquet(spec_file_path)
        spec_event_df = extract_event_window(spec_df, spec_offset, spec_rows_per_second, spec_event_duration)
        spec_output_path = os.path.join(labeled_spec_dir, f"{label_id}_{label}_{spec_id}_{spec_sub_id}.parquet")
        spec_event_df.to_parquet(spec_output_path, index=False)
      except:
        print(f"Error handling Spectrogram {spec_id}")

if __name__ == "__main__":
  start = time.time()
  label_data()
  end = time.time()
  print(f"Labeling took {end - start} seconds")