preproc_config = {
  "metadata_csv": "./data/",
  "unproc_data": "./data/unprocessed_data/",
  "unproc_eegs": "./data/unprocessed_data/eegs/",
  "unproc_spectrograms": "./data/unprocessed_data/spectrograms/",
  "proc_data": "./data/processed_data/",
  "proc_eegs": "./data/processed_data/eegs/",
  "proc_spectrograms": "./data/processed_data/spectrograms/",
  "training_data": "./data/training_data/",
  "training_eegs": "./data/training_data/eegs/",
  "training_spectrograms": "./data/training_data/spectrograms/",
  "eeg_rows_per_second": 200,
  "eeg_event_duration": 50,
  "spectrogram_rows_per_second": 0.5,
  "spectrogram_event_duration": 600,
  "label_map": {
    0: "Seizure",
    1: "LRDA",
    2: "GRDA",
    3: "LPD",
    4: "GPD",
    5: "Other"
  }
}
eeg_config = {
  
}
spec_config = {

}

notch_config = {
  "input_dir": "labeled_train_eegs",
  "output_dir": "notched_train_eegs",
  "fs": 200.0,
  "notch_freq": 60.0,
  "quality_factor": 30.0
}