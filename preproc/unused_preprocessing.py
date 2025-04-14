def rmBadSamples(file_id, df):
  if df.empty:
    print("Error with id", file_id, "empty file")
    return False
  if df.isna().values.any():
    print("Error with id", file_id, "contains nans")
    return False
  if np.isinf(df.values).any():
    print("Error with id", file_id, "contains infinite values")
    return False
  return True

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

def deleteCorruptFiles(directory, log_path):
  deleted_files = []
  for root, _, files in os.walk(directory):
    for file in files:
      if file.endswith(".parquet"):
        full_path = os.path.join(root, file)
        try:
          pd.read_parquet(full_path)
        except pyarrow.lib.ArrowInvalid as e:
          if "Parquet magic bytes not found in footer" in str(e):
            print(f"Deleting corrupted file: {full_path}")
            os.remove(full_path)
            deleted_files.append(full_path)
          else:
            print(f"Skipped {full_path} due to different ArrowInvalid error:\n{e}")
        except Exception as e:
          print(f"Skipped {full_path} due to unexpected error:\n{e}")
  if deleted_files:
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    if not os.path.exists(log_path):
      open(log_path, "w").close()
    with open(log_path, "a", encoding="utf-8") as log_file:
      for path in deleted_files:
        log_file.write(f"[{datetime.now().isoformat()}] Deleted: {path}\n")
  print(f"\nDeleted {len(deleted_files)} corrupted parquet files.")
  return deleted_files