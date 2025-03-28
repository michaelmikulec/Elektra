import logging
import os
from typing import Any


LOGGING_LVL = logging.DEBUG

# ./Elektra/Elektra/
ROOT = os.path.dirname(os.path.abspath(__file__))

# ./Elektra/Elektra/data/
DATA = os.path.join(ROOT, "data")

# ./Elektra/Elektra/data/training_data/
TRAINING_DATA = os.path.join(DATA, "training_data")

# ./Elektra/Elektra/data/training_data/eegs/
TRAINING_EEGS = os.path.join(TRAINING_DATA, "eegs")

# ./Elektra/Elektra/data/training_data/spectrograms/
TRAINING_SPECTROGRAMS = os.path.join(TRAINING_DATA, "spectrograms")

# ./Elektra/Elektra/data/unprocessed_data/
UNPROC_DATA = os.path.join(DATA, "unprocessed_data")

# ./Elektra/Elektra/data/unprocessed_data/metadata.csv
METADATA = os.path.join(UNPROC_DATA, "metadata.csv")

# ./Elektra/Elektra/data/unprocessed_data/train_eegs/
UNLABELED_EEGS = os.path.join(UNPROC_DATA, "train_eegs")

# ./Elektra/Elektra/data/unprocessed_data/train_spectrograms/
UNLABELED_SPECTROGRAMS = os.path.join(UNPROC_DATA, "train_spectrograms")

# ./Elektra/Elektra/user_data/
USER_DATA = os.path.join(DATA, "user_data")

# ./Elektra/Elektra/logs/
LOGS = os.path.join(ROOT, "logs")

# ./Elektra/Elektra/logs/elektra.log
LOG_FILE = os.path.join(LOGS, "elektra.log")

# ./Elektra/Elektra/dl/
DL_MOD = os.path.join(ROOT, "dl")

# ./Elektra/Elektra/dl/transformer/
TRANSFORMER_MOD = os.path.join(DL_MOD, "transformer")

# ./Elektra/Elektra/dl/transformer/transformer.pth
TRANSFORMER_MODEL = os.path.join(TRANSFORMER_MOD, "transformer.pth")

# ./Elektra/Elektra/dl/cnn/
CNN_MOD = os.path.join(DL_MOD, "cnn")

# ./Elektra/Elektra/dl/cnn/cnn.pth
CNN_MODEL = os.path.join(CNN_MOD, "cnn.pth")

# ./Elektra/Elektra/ml/
ML_MOD = os.path.join(ROOT, "ml")

# ./Elektra/Elektra/preproc/
PREPROC_MOD = os.path.join(ROOT, "preproc")

# ./Elektra/Elektra/ui/
UI_MOD = os.path.join(ROOT, "ui")


global_config:dict[str, Any] = {
  "project_root": ROOT, 
  "logging_lvl": LOGGING_LVL,
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

transformer_config:dict[str, Any] = {
  "training_data": TRAINING_EEGS,
  "batch_size": 1,
  "num_workers": 1,
  "learning_rate": 0.001,
  "num_epochs": 1
}

transformer_model_config:dict[str, Any] = {
  "model_path": TRANSFORMER_MODEL,
  "input_dim": 20,
  "model_dim": 128,
  "num_heads": 4,
  "num_layers": 2,
  "dim_feedforward": 256,
  "dropout": 0.1,
  "num_classes": 6,
  "max_len": 10000,
  "use_learnable_pos_emb": True,
  "use_cls_token": True,
  "pooling": "cls"
}

cnn_config:dict[str, Any] = { }