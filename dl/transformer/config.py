import os
from typing import Any

transformer_dir = os.path.dirname(os.path.abspath(__file__))

dataset_config = {
  "training_data": os.path.join(os.getcwd(), "data", "training_data", "eegs"),
  "eeg_rows_per_second": 200,
  "eeg_event_duration": 50,
  "label_map": {
    0: "Seizure",
    1: "LRDA",
    2: "GRDA",
    3: "LPD",
    4: "GPD",
    5: "Other"
  }
}

training_config = {
  "model_path": os.path.join(transformer_dir, "transformer.pth"),
  "batch_size": 1,
  "num_workers": 1,
  "learning_rate": 0.001,
  "num_epochs": 1
}

model_config = {
  "input_dim": 20,
  "model_dim": 128,
  "num_heads": 4,
  "num_layers": 2,
  "dim_feedforward": 256,
  "dropout": 0.1,
  "num_classes": 6,
  "max_len": 10001,
  "use_learnable_pos_emb": True,
  "use_cls_token": True,
  "pooling": "cls"
}
