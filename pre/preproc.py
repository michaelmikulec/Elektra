import os
from typing import List, Tuple
import pandas as pd
import numpy as np


def rmBadSamples(file_id, df):
  if df.empty:
    print("Error with id", file_id, "empty file")
    return False
  if df.isna().values.any():
    print("Error with id", file_id, "contains nans")
    return False
  if np.isinf(df.values).any():
    print("Error with id", file_id, "contains infinite values")
    return False
  return True

def apply_notch_filter(input_dir, output_dir, fs=200, notch_freq=60.0, quality_factor=30.0):
  for f_name in os.listdir(input_dir):
    f_path = os.path.join(input_dir, f_name)
    print(f_path)
    df = pd.read_parquet(f_path)
    df.drop(columns=["EKG", "O2"], inplace=True)
    b, a = iirnotch(notch_freq, quality_factor, fs)
    filtered_df = df.copy()
    for col in df.columns:
      filtered_df[col] = filtfilt(b, a, df[col])
    filtered_df.to_parquet(os.path.join(output_dir, f_name), index=False)

def deleteCorruptFiles(directory, log_path):
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


def process_eeg_windows(eegs, unprocEEGDir: str, procEEGDir: str):
  if not os.path.exists(procEEGDir):
    os.makedirs(procEEGDir)
  fs = 200
  for e in eegs:
    id       = e[0]
    subID    = e[1]
    offset   = e[2]
    intLabel = e[3]
    labelID  = e[4]
    srcPath  = os.path.join(unprocEEGDir, f"{id}.parquet")
    if not os.path.isfile(srcPath):
      print(f"[WARNING] File not found: {srcPath}. Skipping.")
      continue

    try:
      df = pd.read_parquet(srcPath)
    except Exception as ex:
      print(f"[ERROR] Failed to read {srcPath}: {ex}. Skipping.")
      continue

    center: int = int(offset * fs) + (25 * fs)
    start: int  = center - (5 * fs)
    stop:  int  = center + (5 * fs)
    if start < 0 or stop > len(df):
      print(f"[WARNING] Window [{start}:{stop}] out of bounds for file {srcPath} (len={len(df)}). Skipping.")
      continue

    window_df = df.iloc[start:stop].copy()
    if window_df.shape != (2000, 20):
      print(f"[WARNING] Expected shape (2000, 20) but got {window_df.shape} in {srcPath}. Skipping.")
      continue

    mean = window_df.mean()
    std = window_df.std()
    if (std == 0).any():
      print(f"[WARNING] Zero standard deviation in columns for {srcPath}. Skipping.")
      continue

    window_df = (window_df - mean) / std
    if window_df.isnull().values.any():
      print(f"[WARNING] Found NaN values after normalization in {srcPath}. Skipping.")
      continue

    arr = window_df.to_numpy()
    if np.isinf(arr).any():
      print(f"[WARNING] Found Inf values after normalization in {srcPath}. Skipping.")
      continue
    if (arr > 1e6).any() or (arr < -1e6).any():
      print(f"[WARNING] Suspicious large magnitude values in {srcPath}. Skipping.")
      continue

    targetPath = os.path.join(procEEGDir, f"{intLabel}_{labelID}_EEG-{id}-{subID}.parquet")
    try:
      window_df.to_parquet(targetPath)
      # print(f"[INFO] Wrote processed window to {targetPath}")
    except Exception as ex:
      print(f"[ERROR] Failed to write {targetPath}: {ex}.")
      continue


def process_spectrogram_windows(eegs, unprocSpecDir: str, procSpecDir: str):
  if not os.path.exists(procSpecDir):
    os.makedirs(procSpecDir)
  for s in eegs:
    id       = s[0]
    subID    = s[1]
    offset   = s[2]
    intLabel = s[3]
    labelID  = s[4]
    srcPath  = os.path.join(unprocSpecDir, f"{id}.parquet")
    if not os.path.isfile(srcPath):
      print(f"[WARNING] File not found: {srcPath}. Skipping.")
      continue
    try:
      df = pd.read_parquet(srcPath)
    except Exception as ex:
      print(f"[ERROR] Failed to read {srcPath}: {ex}. Skipping.")
      continue

    center: int = int(offset // 2) + 150
    start: int  = center - 2
    stop:  int  = center + 3
    if start < 0 or stop > len(df):
      print(f"[WARNING] Window [{start}:{stop}] out of bounds for file {srcPath} (len={len(df)}). Skipping.")
      continue

    window_df = df.iloc[start:stop].copy()
    if len(window_df) != 5:
      print(f"[WARNING] Expected 5 rows but got {len(window_df)} in {srcPath}. Skipping.")
      continue
    if window_df.isnull().values.any():
      print(f"[WARNING] Found NaN values in window of file {srcPath}. Skipping.")
      continue

    window_df = window_df.iloc[:, 1:]
    mean = window_df.mean()
    std = window_df.std()
    if (std == 0).any():
      print(f"[WARNING] Zero standard deviation in {srcPath}. Skipping.")
      continue

    window_df = (window_df - mean) / std
    arr = window_df.to_numpy()
    if np.isinf(arr).any():
      print(f"[WARNING] Found Inf values in {srcPath}. Skipping.")
      continue
    if (arr > 1e6).any() or (arr < -1e6).any():
      print(f"[WARNING] Suspicious large magnitude values in {srcPath}. Skipping.")
      continue

    targetPath = os.path.join(procSpecDir, f"{intLabel}_{labelID}_SPEC-{id}-{subID}.parquet")
    try:
      window_df.to_parquet(targetPath)
      # print(f"[INFO] Wrote processed window to {targetPath}")
    except Exception as ex:
      print(f"[ERROR] Failed to write {targetPath}: {ex}.")
      continue


if __name__ == "__main__":
  labelMap = { "Seizure": 0, "LRDA": 1, "GRDA": 2, "LPD": 3, "GPD": 4, "Other": 5 };
  metadataPath:str = "./data/metadata.csv";
  eegs:List[Tuple]  = [];
  specs:List[Tuple] = [];
  metadata = pd.read_csv(metadataPath);
  for index, row in metadata.iterrows():
    try:
      eID:int      = int(row['eeg_id']);
      eSubID:int   = int(row['eeg_sub_id']);
      eOffset:int  = int(row['eeg_label_offset_seconds']);
      sID:int      = int(row['spectrogram_id']);
      sSubID:int   = int(row['spectrogram_sub_id']);
      sOffset:int  = int(row['spectrogram_label_offset_seconds']);
      labelID:int  = int(row['label_id']);
      strLabel:str = str(row['expert_consensus']);
      intLabel:int = labelMap[strLabel];
      eeg:Tuple    = (eID, eSubID, eOffset, intLabel, labelID);
      spec:Tuple   = (sID, sSubID, sOffset, intLabel, labelID);
      eegs.append(eeg); specs.append(spec);
    except Exception as e:
      print(f"Error: {e}\nContinuing...")

  unprocEEGDir:str = "./data/unprocessed_data/eegs/"
  procEEGDir:str = "./data/training_data/eegs/"
  process_eeg_windows(eegs, unprocEEGDir, procEEGDir)

  unprocSpecDir:str = "./data/unprocessed_data/spectrograms/"
  procSpecDir:str = "./data/training_data/spectrograms/"
  process_spectrogram_windows(eegs, unprocSpecDir, procSpecDir)