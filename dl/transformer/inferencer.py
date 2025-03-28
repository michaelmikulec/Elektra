import torch
from torch.nn import DataParallel
from torch.cuda.amp import autocast
from model import EEGTransformer
from config import model_config

def infer(inputs, model_path):
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  model = EEGTransformer(**model_config)

  if torch.cuda.device_count() > 1:
    model = DataParallel(model)

  model.to(device)
  model.load_state_dict(torch.load(model_path, map_location=device))
  model.eval()

  if inputs.ndim == 2:
    inputs = inputs.unsqueeze(0)

  with torch.no_grad():
    inputs = inputs.to(device)
    with autocast():
      outputs = model(inputs)
    return outputs
