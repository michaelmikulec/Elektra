import os
import torch
import pandas as pd
import numpy as np
import pyarrow.parquet as pq
from tqdm import tqdm
from datetime import datetime

METADATA_PATH   = "data/raw/train.csv"
EEG_DIR         = "data/raw/train_eegs"
SPEC_DIR        = "data/raw/train_spectrograms"
OUTPUT_DIR      = "data/prep"
INDEX_CSV       = "data/pidx.csv"
SUMMARY_PATH    = "dl/prep/prep_summary.txt"
RESUME          = True
CSV_WRITE_BATCH = 1000

os.makedirs(OUTPUT_DIR, exist_ok=True)

index_rows    = []
processed_ids = set()
eeg_cache     = {}
spec_cache    = {}

def write_index_rows():
  if not index_rows: return
  pd.DataFrame(index_rows).to_csv(INDEX_CSV, index=False)

def vectorized_fill_and_standardize(data):
  col_means      = np.nanmean(data, axis = 0)
  nan_idxs       = np.where(np.isnan(data))
  data[nan_idxs] = np.take(col_means, nan_idxs[1])
  mean_vals      = data.mean(axis = 0, keepdims = True)
  std_vals       = data.std(axis = 0, keepdims = True) + 1e-6
  return (data - mean_vals) / std_vals

def get_eeg_df(eeg_id):
  if eeg_id not in eeg_cache:
    p = os.path.join(EEG_DIR, f"{eeg_id}.parquet")
    if not os.path.exists(p): return None
    eeg_cache[eeg_id] = pq.read_table(p).to_pandas()
  return eeg_cache[eeg_id]

def get_spec_df(spec_id):
  if spec_id not in spec_cache:
    p = os.path.join(SPEC_DIR, f"{spec_id}.parquet")
    if not os.path.exists(p): return None
    spec_cache[spec_id] = pq.read_table(p).to_pandas()
  return spec_cache[spec_id]

def get_eeg_event_window(eeg_df, offset_sec):
  start = int((offset_sec + 20) * 200)
  end   = start + 2000
  data  = eeg_df.iloc[start:end].values.astype("float32")

  if data.shape != (2000, 20):
    raise ValueError()

  return torch.tensor(vectorized_fill_and_standardize(data))

def get_spec_event_window(spec_df, offset_sec):
  spec_data = spec_df.iloc[:, 1:].values.astype("float32")
  center    = int((offset_sec + 300 - 1) // 2)
  start     = center - 2
  end       = center + 3
  data      = spec_data[start:end]

  if data.shape != (5, 400): 
    raise ValueError()

  return torch.tensor(vectorized_fill_and_standardize(data))

def extract_soft_labels(r):
  labels = ["seizure","lpd","gpd","lrda","grda","other"]
  votes  = [r[f"{x}_vote"] for x in labels]
  t      = sum(votes)
  if t <= 0: 
    return [0]*6
  return [v/t for v in votes]

if RESUME and os.path.exists(INDEX_CSV):
  try:
    df0           = pd.read_csv(INDEX_CSV)
    processed_ids = set(df0["label_id"].astype(str))
    index_rows    = df0.to_dict("records")
  except pd.errors.EmptyDataError:
    processed_ids = set()
    index_rows    = []

meta  = pd.read_csv(METADATA_PATH)
total = len(meta)
saved, skipped, failed = 0, 0, 0

with tqdm(total=total) as pbar:
  for _, row in meta.iterrows():
    pbar.update(1)
    lbl_id = str(row["label_id"])
    if RESUME and lbl_id in processed_ids:
      continue
    eeg_id    = row["eeg_id"]
    spec_id   = row["spectrogram_id"]
    lbl_str   = str(row["expert_consensus"]).strip().lower()
    eeg_path  = os.path.join(EEG_DIR, f"{eeg_id}.parquet")
    spec_path = os.path.join(SPEC_DIR, f"{spec_id}.parquet")
    out_file  = f"{lbl_str}_{lbl_id}.pt"
    out_path  = os.path.join(OUTPUT_DIR, out_file)
    rdict     = {
      "filename"                         : out_file,
      "preprocessed"                     : 0,
      "status"                           : "",
      "eeg_mean"                         : None,
      "eeg_var"                          : None,
      "eeg_std"                          : None,
      "eeg_max_abs"                      : None,
      "spec_mean"                        : None,
      "spec_var"                         : None,
      "spec_std"                         : None,
      "spec_max_abs"                     : None,
      "label_id"                         : row["label_id"],
      "patient_id"                       : row["patient_id"],
      "expert_consensus"                 : row["expert_consensus"],
      "seizure_vote"                     : row["seizure_vote"],
      "lpd_vote"                         : row["lpd_vote"],
      "gpd_vote"                         : row["gpd_vote"],
      "lrda_vote"                        : row["lrda_vote"],
      "grda_vote"                        : row["grda_vote"],
      "other_vote"                       : row["other_vote"],
      "eeg_id"                           : row["eeg_id"],
      "eeg_sub_id"                       : row["eeg_sub_id"],
      "eeg_label_offset_seconds"         : row["eeg_label_offset_seconds"],
      "spectrogram_id"                   : row["spectrogram_id"],
      "spectrogram_sub_id"               : row["spectrogram_sub_id"],
      "spectrogram_label_offset_seconds" : row["spectrogram_label_offset_seconds"],
    }
    if not os.path.exists(eeg_path) or not os.path.exists(spec_path):
      rdict["status"] = "skipped: missing file"
      index_rows.append(rdict)
      skipped += 1
      continue

    try:
      eeg_df  = get_eeg_df(eeg_id)
      spec_df = get_spec_df(spec_id)
      if eeg_df is None or spec_df is None:
        rdict["status"] = "skipped: missing file"
        index_rows.append(rdict)
        skipped += 1
        continue

      eegTensor   = get_eeg_event_window(eeg_df, row["eeg_label_offset_seconds"])
      specTensor  = get_spec_event_window(spec_df, row["spectrogram_label_offset_seconds"])
      labelTensor = torch.tensor(extract_soft_labels(row), dtype=torch.float32)

      if torch.isnan(eegTensor).any() or torch.isnan(specTensor).any() or torch.isnan(labelTensor).any():
        raise ValueError()

      torch.save({"eeg": eegTensor, "spectrogram": specTensor, "label": labelTensor}, out_path)
      rdict["preprocessed"] = 1
      rdict["status"]       = "success"
      rdict["eeg_mean"]     = eegTensor.mean().item()
      rdict["eeg_var"]      = eegTensor.var().item()
      rdict["eeg_std"]      = eegTensor.std().item()
      rdict["eeg_max_abs"]  = torch.abs(eegTensor).max().item()
      rdict["spec_mean"]    = specTensor.mean().item()
      rdict["spec_var"]     = specTensor.var().item()
      rdict["spec_std"]     = specTensor.std().item()
      rdict["spec_max_abs"] = torch.abs(specTensor).max().item()
      index_rows.append(rdict)
      saved += 1

    except:
      rdict["status"] = "failed"
      index_rows.append(rdict)
      failed += 1

    if (saved + skipped + failed) % CSV_WRITE_BATCH == 0:
      write_index_rows()

write_index_rows()

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

with open(SUMMARY_PATH, "w") as f:
  f.write("\n".join(summary))
