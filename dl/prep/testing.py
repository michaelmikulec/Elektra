"""
train.csv 
  df = pd.read_csv("data/train.csv")
  15 columns:
    "label_id"                         : df["label_id"],
    "patient_id"                       : df["patient_id"],
    "expert_consensus"                 : df["expert_consensus"],
    "seizure_vote"                     : df["seizure_vote"],
    "lpd_vote"                         : df["lpd_vote"],
    "gpd_vote"                         : df["gpd_vote"],
    "lrda_vote"                        : df["lrda_vote"],
    "grda_vote"                        : df["grda_vote"],
    "other_vote"                       : df["other_vote"],
    "eeg_id"                           : df["eeg_id"],
    "eeg_sub_id"                       : df["eeg_sub_id"],
    "eeg_label_offset_seconds"         : df["eeg_label_offset_seconds"],
    "spectrogram_id"                   : df["spectrogram_id"],
    "spectrogram_sub_id"               : df["spectrogram_sub_id"],
    "spectrogram_label_offset_seconds" : df["spectrogram_label_offset_seconds"],

EEG Features:
  20 columns: [
    'Fp1', 'F3', 'C3', 'P3', 'F7', 'T3', 
    'T5', 'O1', 'Fz', 'Cz', 'Pz', 'Fp2', 
    'F4', 'C4', 'P4', 'F8', 'T4', 'T6', 
    'O2', 'EKG'
  ] 
  Sample Rate: 200 Hz
  Minimum Length: >= 10000 rows (50 seconds)
  

Spectrogram Features:
  401 columns: 
    * 1 time Column
    * 100 LL Columns
    * 100 RL Columns
    * 100 LP Columns
    * 100 RP Columns
  Minimum Length: >= 300 rows (600 seconds)
  * rows are 2 second windows centered at 1, 3, 5, ... seconds
  

"""  

# import pandas as pd
# import os

# eeg_files = os.listdir("data/train_eegs")
# print(f"Number of EEG files: {len(eeg_files)}")

# spectrogram_files = os.listdir("data/train_spectrograms")
# print(f"Number of Spectrogram files: {len(spectrogram_files)}")

# df = pd.read_csv("data/train.csv") 
# print(len(df["eeg_id"].unique()))


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import iirnotch, filtfilt

def apply_notch_filter(
  df    : pd.DataFrame,
  fs    : float = 200.0,
  w0    : float = 60.0,
  Q     : float = 30.0
) -> pd.DataFrame:
  b, a     = iirnotch(w0=w0, Q=Q, fs=fs)
  filtered = filtfilt(b, a, df.values, axis=0)
  return pd.DataFrame(filtered, index=df.index, columns=df.columns)

def graph_eeg(df:pd.DataFrame, start:int=0, end:int=-1, fs:int=200) -> None:
  df        = df.iloc[start:end]
  t         = df.index / fs
  dataRange = df.max().max() - df.min().min()
  offset    = dataRange
  plt.figure(figsize=(12, 8))
  for i, chan in enumerate(df.columns):
    plt.plot(t, df[chan] + i * offset)
  yticks = [i * offset for i in range(len(df.columns))]
  plt.yticks(yticks, df.columns)
  plt.xlabel("Time (s)")
  plt.tight_layout()
  plt.show()

def graph_eeg_spectrogram(df:pd.DataFrame, start:int=0, end:int=-1, fs:int=200, nfft:int=128, noverlap:int|None=None) -> None:
  df        = df.iloc[start:end]
  noverlap  = noverlap if noverlap is not None else nfft // 2
  n_ch      = df.shape[1]
  n_cols    = int(np.ceil(np.sqrt(n_ch)))
  n_rows    = int(np.ceil(n_ch / n_cols))
  fig, axes = plt.subplots(n_rows, n_cols, figsize = (12, 8))
  axes      = axes.flatten()  # so we can index them with a single i
  for i, chan in enumerate(df.columns):
    axes[i].specgram(
      df[chan].to_numpy(),
      NFFT=nfft,
      Fs=fs,
      noverlap=noverlap
    )
    axes[i].set_title(chan, fontsize=8)
    axes[i].set_xlabel("Time [s]", fontsize=7)
    axes[i].set_ylabel("Freq [Hz]", fontsize=7)
  for ax in axes[n_ch:]:
    ax.axis("off")
  plt.tight_layout()
  plt.show()

df = pd.read_parquet("data/train_eegs/1000913311.parquet")
df = apply_notch_filter(df, fs=200.0)
graph_eeg(df, start=4000, end=6000, fs=200)
graph_eeg_spectrogram(df, start=4000, end=6000, fs=200, nfft=128)