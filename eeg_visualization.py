import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from scipy.signal import spectrogram

parquet_file = "data/demo/0_12031182_EEG-4266393632-3.parquet"

def create_eeg_spectrogram(df, sampling_rate=200, nperseg=400, noverlap=200):
    specs = {}
    for ch in df.columns:
        if ch != 'time':
            freqs, times, Sxx = spectrogram(df[ch], fs=sampling_rate, nperseg=nperseg, noverlap=noverlap)
            specs[ch] = {'frequencies': freqs, 'times': times, 'power': 10 * np.log10(Sxx)}
    return specs

def plot_spectrogram(spec_data, channel, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    
    spec = spec_data[channel]
    im = ax.pcolormesh(spec['times'], spec['frequencies'], spec['power'], shading='gouraud', cmap='viridis')
    
    ax.set_ylabel('Frequency [Hz]')
    ax.set_xlabel('Time [sec]')
    ax.set_title(f'Spectrogram - {channel}')
    plt.colorbar(im, ax=ax, label='Power [dB]')
    return ax

if not os.path.exists(parquet_file):
    print(f"Error: File {parquet_file} not found")
    demo_dir = "data/demo"
    if os.path.exists(demo_dir):
        for file in os.listdir(demo_dir): print(f"  - {file}")
    exit(1)

try:
    df = pd.read_parquet(parquet_file)
    if df.empty:
        print(f"Warning: The file {parquet_file} contains no data")
        exit(1)
        
    print(df.head())
    print(df.columns)
    
    sampling_rate = 200  
    duration = 10 
    if not df.index.name == 'time' and 'time' not in df.columns:
        time = np.linspace(0, duration, sampling_rate * duration)
        if len(df) == len(time): df['time'] = time
    
    ch_types = {
        'electrodes': [col for col in df.columns if col not in ['time', 'O2', 'EKG']],
        'O2': ['O2'] if 'O2' in df.columns else [],
        'EKG': ['EKG'] if 'EKG' in df.columns else []
    }
    
    n_channels = len([col for group in ch_types.values() for col in group if col != 'time'])
    if n_channels == 0:
        print("No channels found to plot")
        exit(1)
    
    spec_data = create_eeg_spectrogram(df.drop('time', errors='ignore'), sampling_rate=sampling_rate)
    
    if len(ch_types['electrodes']) > 0:
        demo_channels = ch_types['electrodes'][:min(3, len(ch_types['electrodes']))]
        
        fig, axs = plt.subplots(len(demo_channels), 2, figsize=(15, 5*len(demo_channels)))
        if len(demo_channels) == 1: axs = axs.reshape(1, 2)
            
        for i, ch in enumerate(demo_channels):
            if 'time' in df.columns: axs[i, 0].plot(df['time'], df[ch])
            else: axs[i, 0].plot(df[ch])
            
            axs[i, 0].set_title(f'Raw EEG - {ch}')
            axs[i, 0].set_xlabel('Time [sec]')
            axs[i, 0].set_ylabel('Amplitude')
            axs[i, 0].grid(True)
            
            plot_spectrogram(spec_data, ch, ax=axs[i, 1])
        
        plt.tight_layout()
        plt.show()
    
    colors = {'electrodes': 'blue', 'O2': 'red', 'EKG': 'green'}
    fig, axes = plt.subplots(n_channels, 1, figsize=(15, n_channels*1.2), sharex=True)
    if n_channels == 1: axes = [axes]
    
    def plot_channels(ch_group, color):
        global plot_idx
        for ch in ch_types[ch_group]:
            if ch != 'time':
                if 'time' in df.columns: axes[plot_idx].plot(df['time'], df[ch], color=color)
                else: axes[plot_idx].plot(df[ch], color=color)
                
                axes[plot_idx].set_ylabel(ch)
                axes[plot_idx].grid(True)
                axes[plot_idx].text(0.01, 0.85, ch_group, transform=axes[plot_idx].transAxes, fontsize=8, color=color)
                plot_idx += 1
    
    plot_idx = 0
    for group, color in colors.items():
        if ch_types[group]: plot_channels(group, color)
    
    axes[-1].set_xlabel('Time (seconds)')
    fig.suptitle('EEG Data Visualization (10 seconds)', fontsize=16)
    plt.tight_layout()
    plt.subplots_adjust(top=0.95)
    plt.show()
    
    spec_df = pd.DataFrame()
    for ch in df.columns:
        if ch != 'time' and ch in spec_data:
            freq_mask = spec_data[ch]['frequencies'] <= 50
            freqs = spec_data[ch]['frequencies'][freq_mask]
            
            for t_idx, t in enumerate(spec_data[ch]['times']):
                powers = spec_data[ch]['power'][freq_mask, t_idx]
                
                for f_idx, f in enumerate(freqs):
                    col = f"{ch}_{f:.1f}Hz"
                    if t_idx >= len(spec_df): spec_df.loc[t_idx, col] = powers[f_idx]
                    else: spec_df.at[t_idx, col] = powers[f_idx]
    
    output_file = parquet_file.replace('.parquet', '_spectrogram.parquet')
    spec_df.to_parquet(output_file)
    print(f"Saved spectrogram data to {output_file}")

except Exception as e:
    print(f"Error: {e}")
