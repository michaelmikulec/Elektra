import os
import torch
import pandas as pd
import numpy as np
import pyarrow.parquet as pq
from tqdm import tqdm
from datetime import datetime

METADATA_PATH    = "data/raw/metadata.csv"
EEG_DIR          = "data/raw/train_eegs"
SPEC_DIR         = "data/raw/train_spectrograms"
OUTPUT_DIR       = "data/preproc"
INDEX_CSV        = "data/preprocessing_index.csv"
LOG_FILE         = "logs/preprocessing.log"
RESUME           = True

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)

log_entries   = []
index_rows    = []
processed_ids = set()

def log(message, quiet=False):
  timestamp    = datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")
  full_message = f"{timestamp} {message}"
  log_entries.append(full_message)
  if not quiet:
    tqdm.write(full_message)

  with open(LOG_FILE, "a") as f:
    f.write(full_message + "\n")

def get_eeg_event_window(df, offset_sec, fs=200):
  start = int((offset_sec + 20) * fs)
  end   = start + 2000
  data  = df.iloc[start:end].values.astype("float32")
  for ch in range(data.shape[1]):
    col = data[:, ch]
    if np.isnan(col).any():
      mean_val    = np.nanmean(col)
      data[:, ch] = np.where(np.isnan(col), mean_val, col)

  data = (data - data.mean(0)) / (data.std(0) + 1e-6)
  if data.shape != (2000, 20):
    raise ValueError(f"EEG shape {data.shape}, expected (2000, 20)")

  return torch.tensor(data)

def get_spec_event_window(df, offset_sec):
# time_col   = df.iloc[:, 0].values
  spec_data  = df.iloc[:, 1:].values.astype("float32")
  event_time = offset_sec + 300
  center_row = int((event_time - 1) // 2)
  start_row  = center_row - 2
  end_row    = center_row + 3
  sliced     = spec_data[start_row:end_row]
  
  for col in range(sliced.shape[1]):
    vec = sliced[:, col]
    if np.isnan(vec).any():
      mean_val       = np.nanmean(vec)
      sliced[:, col] = np.where(np.isnan(vec), mean_val, vec)

  sliced = (sliced - sliced.mean(0)) / (sliced.std(0) + 1e-6)
  if sliced.shape != (5, 400):
    raise ValueError(f"Spectrogram shape {sliced.shape}, expected (5, 400)")

  return torch.tensor(sliced)

def extract_soft_labels(row):
  labels = ["seizure", "lpd", "gpd", "lrda", "grda", "other"]
  votes  = [row[f"{l}_vote"] for l in labels]
  total  = sum(votes)
  return [v / total for v in votes]


if RESUME and os.path.exists(INDEX_CSV):
  try:
    existing_df   = pd.read_csv(INDEX_CSV)
    processed_ids = set(existing_df["label_id"].astype(str))
    index_rows    = existing_df.to_dict("records")
    log(f"Resuming: skipping {len(processed_ids)} already processed rows from index")

  except pd.errors.EmptyDataError:
    log("Index CSV is empty or corrupted. Starting from scratch.")
    processed_ids = set()
    index_rows    = []

meta  = pd.read_csv(METADATA_PATH)
total = len(meta)
saved, skipped, failed = 0, 0, 0

for _, row in tqdm(meta.iterrows(), total=total):
  label_id  = str(row["label_id"])
  if RESUME and label_id in processed_ids:
    continue

  eeg_id    = row["eeg_id"]
  spec_id   = row["spectrogram_id"]
  label_str = str(row["expert_consensus"]).strip().lower()
  eeg_path  = os.path.join(EEG_DIR, f"{eeg_id}.parquet")
  spec_path = os.path.join(SPEC_DIR, f"{spec_id}.parquet")
  out_file  = f"{label_str}_{label_id}.pt"
  out_path  = os.path.join(OUTPUT_DIR, out_file)
  index_row = {
    'filename'                         : out_file,
    'preprocessed'                     : 0,
    'status'                           : "",
    "eeg_mean"                         : None,
    "eeg_var"                          : None,
    "eeg_std"                          : None,
    "eeg_max_abs"                      : None,
    "spec_mean"                        : None,
    "spec_var"                         : None,
    "spec_std"                         : None,
    "spec_max_abs"                     : None,
    'label_id'                         : row['label_id'],
    'patient_id'                       : row['patient_id'],
    'expert_consensus'                 : row['expert_consensus'],
    'seizure_vote'                     : row['seizure_vote'],
    'lpd_vote'                         : row['lpd_vote'],
    'gpd_vote'                         : row['gpd_vote'],
    'lrda_vote'                        : row['lrda_vote'],
    'grda_vote'                        : row['grda_vote'],
    'other_vote'                       : row['other_vote'],
    'eeg_id'                           : row['eeg_id'],
    'eeg_sub_id'                       : row['eeg_sub_id'],
    'eeg_label_offset_seconds'         : row['eeg_label_offset_seconds'],
    'spectrogram_id'                   : row['spectrogram_id'],
    'spectrogram_sub_id'               : row['spectrogram_sub_id'],
    'spectrogram_label_offset_seconds' : row['spectrogram_label_offset_seconds'],
  }

  if not os.path.exists(eeg_path) or not os.path.exists(spec_path):
    if not os.path.exists(eeg_path): 
      msg = f"skipped: missing EEG"
    else:                            
      msg = f"skipped: missing spectrogram"

    log(f"Skipped {label_id}: {msg}")
    index_row["status"] = msg
    index_rows.append(index_row)
    pd.DataFrame(index_rows).to_csv(INDEX_CSV, index=False)
    skipped += 1
    continue

  try:
    eeg_df       = pq.read_table(eeg_path).to_pandas()
    spec_df      = pq.read_table(spec_path).to_pandas()
    eeg_tensor   = get_eeg_event_window(eeg_df, row["eeg_label_offset_seconds"])
    spec_tensor  = get_spec_event_window(spec_df, row["spectrogram_label_offset_seconds"])
    label_tensor = torch.tensor(extract_soft_labels(row), dtype=torch.float32)

    if ( torch.isnan(eeg_tensor).any() 
      or torch.isnan(spec_tensor).any() 
      or torch.isnan(label_tensor).any()
    ): raise ValueError("NaNs found in tensor(s)")

    torch.save(
      {
        "eeg"         : eeg_tensor,
        "spectrogram" : spec_tensor,
        "label"       : label_tensor
      }, 
      out_path
    )
    index_row["preprocessed"] = 1
    index_row["status"]       = "success"
    index_row["eeg_mean"]     = eeg_tensor.mean().item()
    index_row["eeg_var"]      = eeg_tensor.var().item()
    index_row["eeg_std"]      = eeg_tensor.std().item()
    index_row["eeg_max_abs"]  = torch.abs(eeg_tensor).max().item()
    index_row["spec_mean"]    = spec_tensor.mean().item()
    index_row["spec_var"]     = spec_tensor.var().item()
    index_row["spec_std"]     = spec_tensor.std().item()
    index_row["spec_max_abs"] = torch.abs(spec_tensor).max().item()

    index_rows.append(index_row)
    saved += 1

  except Exception as e:
    err_msg = f"failed: {str(e)}"
    log(f"Failed {label_id}: {err_msg}")
    index_row['status'] = err_msg
    index_rows.append(index_row)
    failed += 1

  pd.DataFrame(index_rows).to_csv(INDEX_CSV, index=False)

summary = [
  "",
  f"===== Preprocessing Summary ({datetime.now().strftime('%Y-%m-%d %H:%M:%S')}) =====",
  f"Total metadata rows: {total}",
  f"Saved successfully:  {saved}",
  f"Skipped (missing):   {skipped}",
  f"Failed (errors):     {failed}",
  f"Index saved:         {INDEX_CSV if index_rows else 'None'}",
  f"Resume mode:         {RESUME}",
  "============================================================",
  ""
]
log_entries.extend(summary)

with open(LOG_FILE, "a") as f:
  f.write("\n".join(log_entries) + "\n")

log(f"Index CSV written to {INDEX_CSV}")
