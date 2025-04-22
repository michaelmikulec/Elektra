import os
import glob
import torch
import numpy as np
import pandas as pd
from scipy.signal import iirnotch, filtfilt, stft

def apply_notch_filter(
  df    : pd.DataFrame,
  fs    : float = 200.0,
  w0    : float = 60.0,
  Q     : float = 30.0
) -> pd.DataFrame:
  b, a     = iirnotch(w0=w0, Q=Q, fs=fs)
  filtered = filtfilt(b, a, df.values, axis=0)
  return pd.DataFrame(filtered, index=df.index, columns=df.columns)

def normalize_data(df:pd.DataFrame) -> pd.DataFrame:
  if df.isnull().values.any() or df.isna().values.any() or df.isin([np.inf, -np.inf]).values.any():
    print("Data contains Null, NaN, or Inf values.")
  for col in df.columns:
    df[col] = (df[col] - df[col].mean()) / df[col].std()
  return df

def prep(metadataPath:str="data/train.csv", startRow:int=0, endRow:int=1086):
  metadata = pd.read_csv(metadataPath)
  metadata = metadata.iloc[startRow:endRow]
  for _, row in metadata.iterrows():
    label_id:int                 = int(row["label_id"])
    expert_consensus:str         = str(row["expert_consensus"]).lower()
    eeg_id:int                   = int(row["eeg_id"])
    eeg_label_offset_seconds:int = int(row["eeg_label_offset_seconds"])
    if not os.path.exists(f"data/train_eegs/{eeg_id}.parquet"):
      print(f"Missing EEG file for {eeg_id}")
      continue
    df = pd.read_parquet(f"data/train_eegs/{eeg_id}.parquet")
    df = df.drop(columns=["EKG"])
    df = apply_notch_filter(df, fs=200)
    beginning:int = eeg_label_offset_seconds * 200
    center:int    = beginning + 5000
    wStart:int    = center - 1000
    wEnd:int      = center + 1000
    try: 
      df = df.iloc[wStart:wEnd]
      df = normalize_data(df)
      df.to_parquet(f"data/prep/eegs/EEG_{label_id}_{expert_consensus.lower()}.parquet", index=False)
    except Exception as e:
      print(f"Error processing {eeg_id}: {e}")
      continue

def remove_bad_parquets(folder, log_path) -> None:
  bad = []
  for f in os.listdir(folder):
    if f.endswith('.parquet'):
      p   = os.path.join(folder, f)
      df  = pd.read_parquet(p)
      arr = df.values
      if not np.isfinite(arr).all():
        os.remove(p)
        bad.append(f)
  with open(log_path, 'w') as w:
    for fname in bad:
      w.write(fname + '\n')

def df_to_spectrograms(df, fs=200, win_len=256, hop_len=128, n_fft=None, window='hann', to_db=True, fmin=None, fmax=None):
  channel_names = list(df.columns)
  if n_fft is None:
    n_fft = win_len
  data = df.values.T
  frequency_bins, time_bins, Z = stft(
    data,
    fs       = fs,
    window   = window,
    nperseg  = win_len,
    noverlap = win_len-hop_len,
    nfft     = n_fft,
    padded   = False,
    boundary = None,
    axis     = -1
  )
  S = np.abs(Z)**2
  if to_db:
    S = 10 * np.log10(S + 1e-10)
  mask = np.ones_like(frequency_bins, dtype=bool)
  if fmin is not None:
    mask &= (frequency_bins >= fmin)
  if fmax is not None:
    mask &= (frequency_bins <= fmax)
  if not mask.all():
    S = S[:, mask, :]
    frequency_bins = frequency_bins[mask]

  mean         = S.mean(axis=2, keepdims=True)
  std          = S.std(axis=2, keepdims=True) + 1e-10
  spectrograms = (S - mean) / std
  return spectrograms, channel_names, frequency_bins, time_bins

def convert_to_spectrograms(eeg_dir="data/prep/eegs", spec_dir="data/prep/specs") -> None:
  os.makedirs(spec_dir, exist_ok=True)
  label_map = {"seizure":0, "lpd":1, "gpd":2, "lrda":3, "grda":4, "other":5}
  for eeg_path in glob.glob(f"{eeg_dir}/*.parquet"):
    base    = os.path.basename(eeg_path)
    name, _ = os.path.splitext(base)
    parts   = name.split("_")
    if len(parts) != 3:
      continue
    _, uid, label_str = parts
    if label_str not in label_map:
      continue
    df = pd.read_parquet(eeg_path)
    spectrograms, _, _, _ = df_to_spectrograms(df)
    spec_tensor = torch.from_numpy(spectrograms).float()
    label_idx   = label_map[label_str]
    out_name    = f"SPEC_{uid}_{label_str}.pt"
    out_path    = os.path.join(spec_dir, out_name)
    torch.save({"spectrogram": spec_tensor, "label": label_idx}, out_path)

if __name__ == "__main__":
  convert_to_spectrograms() 