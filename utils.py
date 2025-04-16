import os

dirs = [
#"main.py",
  "data/",
    #data/pidx.csv
    "data/prep/",
    "data/raw/",
      #data/raw/train.csv
      "data/raw/train_eegs/",
      "data/raw/train_spectrograms/",

  "dl/",
    "dl/cnn/",
      #dl/cnn/__init__.py
      #dl/cnn/cnn.py
    "dl/dh/",
      #dl/dh/__init__.py
      #dl/dh/dh.py
    "dl/prep/",
      #dl/prep/__init__.py
      #dl/prep/prep.py
      #dl/prep/prep_summary.txt
    "dl/tf/",
      #dl/tf/__init__.py
      #dl/tf/tf.py

  "ml/",
]

for dir in dirs:
  os.makedirs(dir, exist_ok=True)