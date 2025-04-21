import os
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

def df_to_spectrograms(df, fs=200, win_len=64, hop_len=32, n_fft=None, to_db=True):
  n_fft = n_fft or win_len
  spectrograms = []
  for ch in df.columns:
    f, t, Z = stft(
      df[ch].values,
      fs       = fs,
      window   = 'hann',
      nperseg  = win_len,
      noverlap = win_len - hop_len,
      nfft     = n_fft,
      padded   = False,
      boundary = None
    )
    S = np.abs(Z) ** 2
    if to_db:
      S = 10 * np.log10(S + 1e-10)
    spectrograms.append(S)
  spec = np.stack(spectrograms, axis=0)  
  return spec, f, t

def prep(metadataPath:str="data/train.csv", startRow:int=0, endRow:int=1086):
  metadata = pd.read_csv(metadataPath)
  metadata = metadata.iloc[startRow:endRow]
  for _, row in metadata.iterrows():
    label_id:int                         = int(row["label_id"])
    expert_consensus:str                 = str(row["expert_consensus"]).lower()
    eeg_id:int                           = int(row["eeg_id"])
    eeg_label_offset_seconds:int         = int(row["eeg_label_offset_seconds"])

    if not os.path.exists(f"data/train_eegs/{eeg_id}.parquet"):
      print(f"Missing EEG file for {eeg_id}")
      continue

    df = pd.read_parquet(f"data/train_eegs/{eeg_id}.parquet")
    df = df.drop(columns=["EKG"])
    df = apply_notch_filter(df, fs=200)

    beginning:int   = eeg_label_offset_seconds * 200
    center:int      = beginning + 5000
    wStart:int      = center - 1000
    wEnd:int        = center + 1000

    try: 
      df = df.iloc[wStart:wEnd]
      df = normalize_data(df)
      df.to_parquet(f"data/prep/eegs/EEG_{label_id}_{expert_consensus.lower()}.parquet", index=False)
    except Exception as e:
      print(f"Error processing {eeg_id}: {e}")
      continue

prep("data/train.csv", 0, -1)