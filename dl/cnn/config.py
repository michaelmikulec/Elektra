import os

config = {
  "training_data": os.path.join(os.getcwd(), "data", "labeled_training_spectrograms"),
  "num_classes": 6,
  "label_index": {
    "Seizure": 0,
    "LRDA": 1,
    "GRDA": 2,
    "LPD": 3,
    "GPD": 4,
    "Other": 5
  },
  "input_dim": 20,
  "num_filters": 32,
  "num_layers": 2,
  "kernel_size": 5,
  "dropout": 0.1,
  "batch_size": 1,
  "num_workers": 1,
  "learning_rate": 0.001,
  "num_epochs": 1,
  "model_save_path": os.path.join(os.getcwd(), "dl", "cnn", "cnn.pt")
}
