import torch
from torch.nn import DataParallel
from . import config, EEGCNN

def infer(inputs, model_path):
  
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  model = EEGCNN(
    input_dim=config["input_dim"],
    num_filters=config["num_filters"],
    num_layers=config["num_layers"],
    kernel_size=config["kernel_size"],
    dropout=config["dropout"],
    num_classes=config["num_classes"]
  )
  if torch.cuda.device_count() > 1:
    model = DataParallel(model)
  model.to(device)
  model.load_state_dict(torch.load(model_path, map_location=device))
  model.eval()
  with torch.no_grad():
    inputs = inputs.to(device)
    return model(inputs)
