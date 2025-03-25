import os
import pandas as pd
import time

def find_parquet_file(folder, file_id):
  file_path = os.path.join(folder, f"{file_id}.parquet")
  return file_path if os.path.exists(file_path) else None

def extract_event_window(df, offset, samples_per_second, event_duration=50):
  row_offset = offset * samples_per_second
  event_rows = samples_per_second * event_duration
  return df.iloc[row_offset : row_offset + event_rows]

def label_data(
  metadata_csv=None, 
  eeg_folder=None, 
  spec_folder=None, 
  processed_eeg_folder=None, 
  processed_spec_folder=None, 
  samples_per_second=200, 
  event_duration=50, 
  label_eegs=False, 
  label_specs=False
):
  metadata_df = pd.read_csv(metadata_csv)

  for index, row in metadata_df.iterrows():
    eeg_id, eeg_sub_id, eeg_offset = row["eeg_id"], row["eeg_sub_id"], int(row["eeg_label_offset_seconds"])
    spec_id, spec_sub_id, spec_offset = row["spectrogram_id"], row["spectrogram_sub_id"], int(row["spectrogram_label_offset_seconds"])
    label = row["expert_consensus"]
    label_index = {
      "Seizure": 0,
      "LRDA": 1, 
      "GRDA": 2, 
      "LPD": 3, 
      "GPD": 4, 
      "Other": 5
    }    
    label_id = label_index[label]

    eeg_file_path = find_parquet_file(eeg_folder, eeg_id)
    if eeg_file_path and label_eegs:
      try:
        eeg_df = pd.read_parquet(eeg_file_path)
        event_eeg = extract_event_window(eeg_df, eeg_offset, samples_per_second, event_duration)
        eeg_output_path = os.path.join(processed_eeg_folder, f"{label_id}_{label}_{eeg_id}_{eeg_sub_id}.parquet")
        event_eeg.to_parquet(eeg_output_path, index=False)
      except:
        print(f"Error handling EEG {eeg_id}")

    spec_file_path = find_parquet_file(spec_folder, spec_id)
    if spec_file_path and label_specs:
      try:
        spec_df = pd.read_parquet(spec_file_path)
        event_spec = extract_event_window(spec_df, spec_offset, samples_per_second, event_duration)
        spec_output_path = os.path.join(processed_spec_folder, f"{label_id}_{label}_{spec_id}_{spec_sub_id}.parquet")
        event_spec.to_parquet(spec_output_path, index=False)
      except:
        print(f"Error handling Spectrogram {spec_id}")

if __name__ == "__main__":
  label_data(
    metadata_csv="./data/train.csv", 
    eeg_folder="./data/train_eegs/", 
    spec_folder="./data/train_spectrograms/", 
    processed_eeg_folder="./data/labeled_eeg/", 
    processed_spec_folder="./data/labeled_spec/", 
    samples_per_second = 200,
    event_duration = 50,
    label_eegs=False,
    label_specs=False
  )