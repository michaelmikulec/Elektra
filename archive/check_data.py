import os, pandas as pd, numpy as np

def check_file(filepath, threshold=1e10):
  issues = []
  try: df = pd.read_parquet(filepath)
  except Exception as e: issues.append(f"error reading file: {e}"); return issues, None
  if df.empty: issues.append("empty file")
  if df.isna().values.any(): issues.append("contains nans")
  if np.isinf(df.values).any(): issues.append("contains infinite values")
  if (np.abs(df.values) > threshold).any(): issues.append(f"contains values exceeding {threshold}")
  return issues, df

def main():
  data_folder = "G:/my drive/fau/egn4952c_spring_2025/data/1200eeg"
  files = [f for f in os.listdir(data_folder) if f.endswith(".parquet")]
  print(f"checking {len(files)} files in {data_folder}".lower())
  deleted = 0
  for f in files:
    path = os.path.join(data_folder, f); issues, _ = check_file(path)
    if issues:
      print(f"deleting file: {f} -> issues: {', '.join(issues)}".lower())
      try: os.remove(path); deleted += 1
      except Exception as e: print(f"failed to delete {f}: {e}".lower())
    else: print(f"file: {f} -> ok".lower())
  print(f"total files deleted: {deleted} out of {len(files)}".lower())

if __name__ == "__main__": main()
