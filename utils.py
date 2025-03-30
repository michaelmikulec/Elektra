import os
import logging
import glob
import pandas as pd
import matplotlib.pyplot as plt


def create_logger(logging_level=logging.DEBUG):
  logger = logging.getLogger();
  logger.propagate = False;
  logger.setLevel(logging_level);
  logFormatter = logging.Formatter(
    fmt=f"[%(asctime)s] [%(levelname)s]: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    style="%",
    validate=True
  );
  logHandler = logging.FileHandler(
    filename="./logs/logs.log",
    mode="a",
    encoding="utf-8",
    errors="strict",
    delay=False
  );
  logHandler.setFormatter(logFormatter);
  logger.addHandler(logHandler);
  return logger;

def build_workspace(logger=None) -> None:
  dirs = [
    "./logs/",
    "./data/",
    "./data/unprocessed_data/",
    "./data/unprocessed_data/eegs/",
    "./data/unprocessed_data/spectrograms/",
    "./data/processed_data/",
    "./data/processed_data/eegs/",
    "./data/processed_data/spectrograms/",
    "./data/training_data/",
    "./data/training_data/eegs/",
    "./data/training_data/spectrograms/",
    "./prep/",
    "./dl/",
    "./dl/transformer/",
    "./dl/cnn/",
    "./ml/",
    "./ui/"
  ];
  if logger: 
    logger.info("Building Workspace...")
  for dir in dirs:
    if not os.path.exists(dir):
      os.makedirs(dir);
      if logger: 
        logger.info(f"Created: {dir}");
    else:
      if logger: 
        logger.debug(f"Exists: {dir}");

def plot_eeg_from_parquet(parquet_file):
  df = pd.read_parquet(parquet_file)
  time_axis = df.index
  plt.figure(figsize=(14, 7))
  for column in df.columns:
    plt.plot(time_axis, df[column], label=column)
  plt.title(f"EEG Data from {os.path.basename(parquet_file)}")
  plt.xlabel("Time")
  plt.ylabel("EEG Signal")
  plt.legend(loc="best")
  plt.tight_layout()
  plt.show()

def main():
  data_dir = "./data/training_data/eegs"
  parquet_files = glob.glob(os.path.join(data_dir, "*.parquet"))
  if not parquet_files:
    print("No parquet files found in", data_dir)
    return
  for parquet_file in parquet_files:
    print(f"Plotting EEG data for {parquet_file}...")
    plot_eeg_from_parquet(parquet_file)

if __name__ == "__main__":
    main()
