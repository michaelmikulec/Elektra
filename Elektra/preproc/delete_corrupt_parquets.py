import os
import pandas as pd
import pyarrow
from datetime import datetime
from config import config

def delete_corrupt_parquet_files(directory, log_path):
  deleted_files = []

  for root, _, files in os.walk(directory):
    for file in files:
      if file.endswith(".parquet"):
        full_path = os.path.join(root, file)
        try:
          pd.read_parquet(full_path)
        except pyarrow.lib.ArrowInvalid as e:
          if "Parquet magic bytes not found in footer" in str(e):
            print(f"Deleting corrupted file: {full_path}")
            os.remove(full_path)
            deleted_files.append(full_path)
          else:
            print(f"Skipped {full_path} due to different ArrowInvalid error:\n{e}")
        except Exception as e:
          print(f"Skipped {full_path} due to unexpected error:\n{e}")

  if deleted_files:
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    if not os.path.exists(log_path):
      open(log_path, "w").close()
    with open(log_path, "a", encoding="utf-8") as log_file:
      for path in deleted_files:
        log_file.write(f"[{datetime.now().isoformat()}] Deleted: {path}\n")

  print(f"\nDeleted {len(deleted_files)} corrupted parquet files.")
  return deleted_files

if __name__ == "__main__":
  log_file_path = os.path.join(config["logs"], "deleted_corrupt_parquet_files.log")
  for dir in [config["training_eegs"], config["training_spectrograms"]]:
    delete_corrupt_parquet_files(dir, log_file_path)

