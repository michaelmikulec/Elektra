import torch
from torch.nn import DataParallel
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
  with torch.no_grad():
    inputs = inputs.to(device)
    return model(inputs)

