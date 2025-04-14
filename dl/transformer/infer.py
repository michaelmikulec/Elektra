import os
import torch
import pyarrow.parquet as pq
from datetime import datetime

from dl.transformer.transformer import EEGTransformer
from preproc.preproc import get_eeg_event_window

def infer(
  parquet_path,
  offset_sec,
  model_dir = "models/dl/transformer",
  log_file  = "logs/dl/transformer/inference.log"
):
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

  df         = pq.read_table(parquet_path).to_pandas()
  eeg_tensor = get_eeg_event_window(df, offset_sec)
  eeg_tensor = eeg_tensor.unsqueeze(0).to(device)  # shape: (1, 2000, 20)

  model_files = [
    f for f in os.listdir(model_dir)
    if f.startswith("model_epoch") and f.endswith(".pt")
  ]
  if not model_files:
    raise FileNotFoundError("No model checkpoint found in directory.")

  latest_model = sorted(
    model_files,
    key=lambda x: int(x.split("epoch")[-1].split(".pt")[0])
  )[-1]
  model_path = os.path.join(model_dir, latest_model)

  model = EEGTransformer().to(device)
  model.load_state_dict(torch.load(model_path, map_location=device))
  model.eval()

  with torch.no_grad():
    output = model(eeg_tensor)
    probs  = torch.softmax(output, dim=1).cpu().squeeze().tolist()

  labels          = ["seizure", "lpd", "gpd", "lrda", "grda", "other"]
  predicted_index = int(torch.tensor(probs).argmax())
  predicted_label = labels[predicted_index]

  # Print results
  print("Predicted probabilities:")
  for label, prob in zip(labels, probs):
    print(f"{label:>8}: {prob:.4f}")
  print(f"\nPredicted class: {predicted_label}")

  # Log results
  timestamp = datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")
  log_lines = [f"{timestamp} Input: {parquet_path}", "Probabilities:"]
  log_lines += [f"  {label:>8}: {prob:.4f}" for label, prob in zip(labels, probs)]
  log_lines.append(f"Predicted: {predicted_label}\n")

  os.makedirs(os.path.dirname(log_file), exist_ok=True)
  with open(log_file, "a") as f:
    f.write("\n".join(log_lines) + "\n")
