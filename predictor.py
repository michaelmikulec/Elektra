# inference_random_forest.py
import pandas as pd
import numpy as np
from scipy.signal import butter, filtfilt, iirnotch
from scipy.stats import skew, kurtosis
import joblib
import mne

FS = 200

MODEL_PATH = "models/rf/random_forest_eeg_model_flat.pkl"
ENCODER_PATH = "models/rf/label_encoder_flat.pkl"

BANDS = {
    "delta": (1, 3), "theta": (4, 7), "alpha1": (8, 9), "alpha2": (10, 12),
    "beta1": (13, 17), "beta2": (18, 30), "gamma1": (31, 40), "gamma2": (41, 50)
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

# Load model + encoder once
model = joblib.load(MODEL_PATH)
encoder = joblib.load(ENCODER_PATH)

def random_forest_inference(file_path):
    df = pd.read_parquet(file_path)

    if "EKG" in df.columns:
        df = df.drop(columns=["EKG"])
    if len(df) < 30 * FS:
        raise ValueError("File too short")

    segment = df.iloc[20 * FS:30 * FS].copy()
    for ch in segment.columns:
        segment[ch] = normalize_signal(apply_bandpass_filter(apply_notch_filter(segment[ch].values)))
    segment = apply_ica(segment)

    features = extract_flattened_features(segment)
    for col in model.feature_names_in_:
        if col not in features.columns:
            features[col] = 0.0
    features = features[model.feature_names_in_]

    pred = model.predict(features)[0]
    prob = model.predict_proba(features)[0]
    confidence = max(prob)
    label = encoder.inverse_transform([pred])[0]

    return label, model.score(features, [pred]), confidence
