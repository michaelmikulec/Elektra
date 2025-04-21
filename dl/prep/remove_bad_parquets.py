import os, pandas as pd, numpy as np

def remove_bad_parquets(folder, log_path):
  bad = []
  for f in os.listdir(folder):
    if f.endswith('.parquet'):
      p = os.path.join(folder, f)
      df = pd.read_parquet(p)
      arr = df.values
      if not np.isfinite(arr).all():
        os.remove(p)
        bad.append(f)
  with open(log_path, 'w') as w:
    for fname in bad:
      w.write(fname + '\n')

remove_bad_parquets("data/prep/eegs", "log.txt")