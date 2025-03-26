import os

config = {
  "training_data": os.path.join(os.getcwd(), "data", "labeled_training_eegs"),
  "num_classes": 6,
  "label_index": {
    "Seizure": 0,
    "LRDA": 1,
    "GRDA": 2,
    "LPD": 3,
    "GPD": 4,
    "Other": 5
  },
  "max_len": 10000,
  "batch_size": 1,
  "num_workers": 1,

  "input_dim": 18,
  "model_dim": 128,
  "num_heads": 4,
  "num_layers": 2,
  "dim_feedforward": 256,
  "dropout": 0.1,
  "use_learnable_pos_emb": True,
  "use_cls_token": True,
  "pooling": "cls",
  "learning_rate": 0.001,
  "num_epochs": 1,
  "model_save_path": os.path.join(os.getcwd(), "dl", "transformer", "transformer.pt")
}
