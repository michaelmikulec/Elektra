import os 
import pandas as pd
from scipy.signal import iirnotch, filtfilt
from notch_config import notch_config

def apply_notch_filter(input_dir, output_dir, fs=200, notch_freq=60.0, quality_factor=30.0):
  for f_name in os.listdir(input_dir):
    f_path = os.path.join(input_dir, f_name)
    print(f_path)
    df = pd.read_parquet(f_path)
    df.drop(columns=["EKG", "O2"], inplace=True)

    b, a = iirnotch(notch_freq, quality_factor, fs)
    filtered_df = df.copy()

    for col in df.columns:
      filtered_df[col] = filtfilt(b, a, df[col])

    filtered_df.to_parquet(os.path.join(output_dir, f_name), index=False)
  
if __name__ == "__main__":
  os.makedirs(input_dir := os.path.join(os.getcwd(), "data", notch_config["input_dir"]), exist_ok=True)
  os.makedirs(output_dir := os.path.join(os.getcwd(), "data", notch_config["output_dir"]), exist_ok=True)
  apply_notch_filter(
    input_dir,
    output_dir,
    notch_config["fs"], 
    notch_config["notch_freq"], 
    notch_config["quality_factor"]
  )