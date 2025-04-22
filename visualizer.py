import math
import pandas as pd
import matplotlib.pyplot as plt

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
