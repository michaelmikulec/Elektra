import os

dirs = [
  "data",
    #data/preprocessing_index.csv
    "data/preproc",
    "data/raw",
      #data/raw/metadata.csv
      "data/raw/train_eegs",
      "data/raw/train_spectrograms",

  "dl",
    "dl/cnn",
      #dl/cnn/__init__.py
      #dl/cnn/cnn.py
      #dl/cnn/infer.py
      #dl/cnn/train.py
    "dl/dataHandler",
      #dl/dataHandler/__init__.py
      #dl/dataHandler/dataHandler.py
    "dl/transformer",
      #dl/transformer/__init__.py
      #dl/transformer/infer.py
      #dl/transformer/train.py
      #dl/transformer/tranformer.py

  "logs",
    "logs/preproc",
      #logs/preprocessing.log
    "logs/dl",
      "logs/dl/transformer",
        #logs/dl/transformer/training.log
        #logs/dl/transformer/inference.log
      "logs/dl/cnn",
        #logs/dl/cnn/training.log
        #logs/dl/cnn/inference.log

  "ml",

  "models",
    "models/dl",
      "models/dl/transformer",
        #models/dl/transformer/tranformer.pt
      "models/dl/cnn",
        #models/dl/cnn/cnn.pt
    "models/ml",
      "models/ml/randomForest",
        #models/ml/randomForest/randomForest.pt

  "ui"
]

for dir in dirs:
  os.makedirs(dir, exist_ok=True)