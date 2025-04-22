import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import stft

def df_to_spectrograms(df, fs=200, win_len=256, hop_len=128, n_fft=None, window='hann', to_db=True, fmin=None, fmax=None):
  channel_names = list(df.columns)
  if n_fft is None:
    n_fft = win_len
  data = df.values.T
  frequency_bins, time_bins, Z = stft(
    data,
    fs=fs,
    window=window,
    nperseg=win_len,
    noverlap=win_len-hop_len,
    nfft=n_fft,
    padded=False,
    boundary=None,
    axis=-1
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
  mean = S.mean(axis=2, keepdims=True)
  std = S.std(axis=2, keepdims=True) + 1e-10
  spectrograms = (S - mean) / std
  return spectrograms, channel_names, frequency_bins, time_bins

def plot_spectrogram_grid(S, channel_names, frequency_bins, time_bins, cols=5, figsize=(15, 10), shading='auto', cmap='viridis'):
  n_channels = S.shape[0]
  rows = math.ceil(n_channels / cols)
  fig, axes = plt.subplots(rows, cols, figsize=figsize, squeeze=False)
  for idx in range(rows * cols):
    r, c = divmod(idx, cols)
    ax = axes[r][c]
    if idx < n_channels:
      ax.pcolormesh(time_bins, frequency_bins, S[idx], shading=shading, cmap=cmap)
      ax.set_title(channel_names[idx])
      if r == rows - 1:
        ax.set_xlabel('Time (s)')
      if c == 0:
        ax.set_ylabel('Frequency (Hz)')
    else:
      ax.axis('off')
  fig.tight_layout()
  plt.show()
  return fig, axes

def plot_eeg(df:pd.DataFrame, start:int=0, end:int=-1, fs:int=200) -> None:
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
