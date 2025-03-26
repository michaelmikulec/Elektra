import os

config = {
  "unlabeled_eeg_dir": os.path.join(os.getcwd(), "data", "train_eegs"),
  "unlabeled_spec_dir": os.path.join(os.getcwd(), "data", "train_spectrograms"),
  "labeled_eeg_dir": os.path.join(os.getcwd(), "data", "labeled_train_eegs"),
  "labeled_spec_dir": os.path.join(os.getcwd(), "data", "labeled_train_specs"),
  "metadata_path": os.path.join(os.getcwd(), "data", "train.csv"),
  "eeg_rows_per_second": 200,
  "eeg_event_duration": 50,
  "spec_rows_per_second": 0.5,
  "spec_event_duration": 600,
  "label_index": {
    "Seizure": 0,
    "LRDA": 1,
    "GRDA": 2,
    "LPD": 3,
    "GPD": 4,
    "Other": 5
  }
}