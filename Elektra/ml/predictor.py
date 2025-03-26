# rf_model/predictor.py

import os
import pandas as pd
import numpy as np
import joblib
import mne
from scipy.signal import butter, filtfilt, iirnotch
from scipy.stats import skew, kurtosis
import warnings
from tqdm import tqdm

from .config import config

# Suppress logs
mne.set_log_level("WARNING")
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

FS = 200

BANDS = {
    "delta": (1, 3), "theta": (4, 7), "alpha1": (8, 9), "alpha2": (10, 12),
    "beta1": (13, 17), "beta2": (18, 30), "gamma1": (31, 40),
    "gamma2": (41, 50), "higher": (51, 100)
}

def apply_notch_filter(signal):
    b, a = iirnotch(w0=60.0, Q=30, fs=FS)
    return filtfilt(b, a, signal)

def apply_bandpass_filter(signal):
    nyq = 0.5 * FS
    low, high = 0.5 / nyq, 40.0 / nyq
    b, a = butter(5, [low, high], btype='band')
    return filtfilt(b, a, signal)

def normalize_signal(signal):
    return (signal - np.mean(signal)) / np.std(signal)

def apply_ica(df):
    try:
        n_components = min(10, len(df.columns))
        info = mne.create_info(ch_names=list(df.columns), sfreq=FS, ch_types=["eeg"] * len(df.columns))
        raw = mne.io.RawArray(df.values.T, info)
        raw.filter(l_freq=1.0, h_freq=None)
        ica = mne.preprocessing.ICA(n_components=n_components, random_state=42, max_iter="auto")
        ica.fit(raw)
        raw_clean = ica.apply(raw)
        return pd.DataFrame(raw_clean.get_data().T, columns=df.columns)
    except Exception:
        return df

def extract_time_features(signal):
    return {
        "mean": np.mean(signal),
        "variance": np.var(signal),
        "skewness": skew(signal),
        "kurtosis": kurtosis(signal),
        "rms": np.sqrt(np.mean(signal**2)),
        "zero_crossing_rate": np.sum(np.diff(np.sign(signal)) != 0) / len(signal),
        "mean_abs": np.mean(np.abs(signal)),
        "diff_rms1": np.sqrt(np.mean(np.diff(signal)**2)),
        "diff_rms2": np.sqrt(np.mean(np.diff(signal, n=2)**2)),
    }

def extract_frequency_features(signal):
    L = len(signal)
    Y = np.fft.fft(signal)
    P2 = np.abs(Y / L)
    P1 = P2[:L // 2 + 1]
    P1[1:-1] *= 2
    freqs = FS * np.arange(L // 2 + 1) / L
    band_power = {band: np.sum(P1[(freqs >= low) & (freqs <= high)]) for band, (low, high) in BANDS.items()}
    band_power["spectral_entropy"] = -np.sum(P1 * np.log(P1 + 1e-10))
    return band_power

def extract_flattened_features(df):
    all_feats = {}
    for ch in df.columns:
        signal = df[ch].values
        tf = extract_time_features(signal)
        ff = extract_frequency_features(signal)
        for k, v in {**tf, **ff}.items():
            all_feats[f"{ch}_{k}"] = v
    return pd.DataFrame([all_feats])

def predict_eeg_files(input_dir=None, output_csv=None):
    input_dir = input_dir or config["input_dir"]
    output_csv = output_csv or config["output_csv"]

    model = joblib.load(config["model_path"])
    encoder = joblib.load(config["encoder_path"])

    all_files = sorted([
        f for f in os.listdir(input_dir)
        if f.endswith(".parquet") and not f.startswith("~") and not f.startswith(".")
    ])
    print(f"📂 Found {len(all_files)} valid EEG files to process.\n")

    results = []
    for fname in tqdm(all_files, desc="Predicting"):
        fpath = os.path.join(input_dir, fname)
        try:
            df = pd.read_parquet(fpath)
            if "EKG" in df.columns:
                df = df.drop(columns=["EKG"])
            if len(df) < 30 * FS:
                print(f"⚠️ Skipping {fname}: too short")
                continue

            segment = df.iloc[20 * FS: 30 * FS].copy()
            for ch in segment.columns:
                segment[ch] = normalize_signal(apply_bandpass_filter(apply_notch_filter(segment[ch].values)))
            segment = apply_ica(segment)

            features = extract_flattened_features(segment)
            for col in model.feature_names_in_:
                if col not in features.columns:
                    features[col] = 0.0
            features = features[model.feature_names_in_]

            pred = model.predict(features)[0]
            label = encoder.inverse_transform([pred])[0]

            print(f"🧠 {fname} → Predicted: {label}")
            results.append({"eeg_file": fname, "prediction": label})
        except Exception as e:
            print(f"❌ Error processing {fname}: {e}")
            results.append({"eeg_file": fname, "prediction": "Error"})

    results_df = pd.DataFrame(results)
    results_df.to_csv(output_csv, index=False)
    print(f"\n✅ All predictions saved to: {output_csv}")
    return results_df
