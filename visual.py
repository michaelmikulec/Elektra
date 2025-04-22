import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import stft

def df_to_spectrograms(df, fs=200, win_len=128, hop_len=64, n_fft=None, to_db=True):
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

def plot_spectrogram_grid(spec, f, t, ch_names=None, rows=5, cols=4, figsize=(12, 10), cmap="viridis"):
  if ch_names is None:
    ch_names = [f"Ch{n}" for n in range(spec.shape[0])]
  fig, axes = plt.subplots(rows, cols, figsize=figsize, sharex=True, sharey=True, constrained_layout=True)
  flat = axes.ravel()
  for i in range(spec.shape[0]):
    pcm = flat[i].pcolormesh(t, f, spec[i], shading="gouraud", cmap=cmap)
    flat[i].set_title(ch_names[i], fontsize=8)
    flat[i].label_outer()
  for ax in flat[spec.shape[0]:]:
    ax.axis("off")
  fig.colorbar(pcm, ax=axes, orientation="vertical", label="Power (dB)")
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

# plot_eeg(pd.read_parquet("data/prep/eegs/EEG_1000898569_seizure.parquet"))
# plot_spectrogram_grid(
#   *df_to_spectrograms(
#     df:=pd.read_parquet("data/prep/eegs/EEG_1000898569_seizure.parquet"),
#     win_len=128, hop_len=64 
#   ),
#   ch_names=df.columns,
# )