import logging
import os
from typing import Any

PROJECT_ROOT:str = os.getcwd()
LOGGING_LVL = logging.DEBUG
LOGS:str = os.path.join(PROJECT_ROOT, "Elektra", "logs")
LOG_FILE:str = os.path.join(LOGS, "root.log")
DATA:str = os.path.join(PROJECT_ROOT, "Elektra", "data")
METADATA:str = os.path.join(DATA, "metadata.csv")
UNLABELED_EEGS:str = os.path.join(DATA, "train_eegs")
UNLABELED_SPECTROGRAMS:str = os.path.join(DATA, "train_spectrograms")
TRAINING_DATA:str = os.path.join(DATA, "training_data")
TRAINING_EEGS:str = os.path.join(TRAINING_DATA, "eegs")
TRAINING_SPECTROGRAMS:str = os.path.join(TRAINING_DATA, "spectrograms")

config:dict[str, Any] = {
  "project_root": PROJECT_ROOT, 
  "logs": LOGS,
  "log_file": LOG_FILE,
  "data": DATA,
  "unlabeled_eeg": UNLABELED_EEGS,
  "unlabeled_spec": UNLABELED_SPECTROGRAMS,
  "training_data": TRAINING_DATA,
  "training_eegs": TRAINING_EEGS,
  "training_spectrograms": TRAINING_SPECTROGRAMS,
  "metadata_path": METADATA,
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